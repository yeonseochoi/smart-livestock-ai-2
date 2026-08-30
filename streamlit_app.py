"""익산시 악취 민원 선제 대응 AI - Streamlit 배포 앱."""
from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime
import json
import math
import os
from pathlib import Path
import re

import folium
import streamlit as st
from streamlit_folium import st_folium

from administrative_agent.documents import create_response_guide
from administrative_agent.llm import llm_configured, provider_name, refine_with_llm
from administrative_agent.service import build_response_package, create_completed_followup
from generate_agent_documents import DEFAULT_METRICS, DEFAULT_PREDICTIONS, forecast_from_csv


ROOT = Path(__file__).resolve().parent
DEMO_DATA = ROOT / "demo" / "demo-data.js"
DOCUMENT_SCHEMA_VERSION = 2

st.set_page_config(page_title="익산 악취 대응 AI", page_icon="🌿", layout="wide", initial_sidebar_state="expanded")


def _load_secrets() -> None:
    """Streamlit Cloud secrets를 기존 Agent 환경변수 형식으로 연결한다."""
    for key in ("LLM_PROVIDER", "GEMINI_API_KEY", "GEMINI_MODEL", "OPENAI_API_KEY", "OPENAI_MODEL"):
        try:
            value = st.secrets.get(key)
        except Exception:
            value = None
        if value:
            os.environ.setdefault(key, str(value))


@st.cache_data
def load_demo_data() -> dict:
    text = DEMO_DATA.read_text(encoding="utf-8-sig").strip()
    match = re.fullmatch(r"window\.DEMO_DATA\s*=\s*(\{.*\})\s*;", text, flags=re.DOTALL)
    if not match:
        raise ValueError("demo/demo-data.js 형식을 읽을 수 없습니다.")
    return json.loads(match.group(1))


def risk_color(score: float, top3: bool) -> str:
    if top3:
        return "#dd3e36"
    return "#e9a11b" if score >= 0.52 else "#f2d56b"


def event_map(event: dict, show_actual: bool) -> folium.Map:
    reports = event.get("reports", [])
    grids = event.get("broad", [])
    points = [[r[0], r[1]] for r in reports] + [g["center"] for g in grids]
    center = points[0] if points else [35.95, 126.98]
    fmap = folium.Map(location=center, zoom_start=13, tiles="OpenStreetMap", control_scale=True)

    for idx, report in enumerate(reports, 1):
        folium.CircleMarker(
            [report[0], report[1]], radius=5, color="#ffffff", weight=2,
            fill=True, fill_color="#172e3d", fill_opacity=1,
            tooltip=f"초기 신고 {idx} · 악취강도 {report[2]}",
        ).add_to(fmap)

    for idx, grid in enumerate(grids, 1):
        lat, lon = grid["center"]
        dy = 1000 / 110540 / 2
        dx = 1000 / (111320 * max(0.1, math.cos(lat * math.pi / 180))) / 2
        top3 = idx <= 3
        base = risk_color(float(grid["score"]), top3)
        actual = show_actual and bool(grid.get("actual"))
        category = "현장 점검 대상" if top3 else ("우선 관찰" if grid["score"] >= 0.52 else "일반 관찰")
        folium.Rectangle(
            bounds=[[lat - dy, lon - dx], [lat + dy, lon + dx]],
            color="#15866f" if actual else base, weight=3 if actual else 1,
            fill=True, fill_color=base, fill_opacity=0.58 if top3 else 0.18 + 0.36 * grid["score"],
            tooltip=f"{idx}순위 1km 권역 · {category}" + (" · 실제 이후 신고" if actual else ""),
        ).add_to(fmap)

    if points:
        fmap.fit_bounds(points, padding=(35, 35), max_zoom=14)
    return fmap


def current_forecast(event: dict):
    forecast = forecast_from_csv(
        ROOT / DEFAULT_PREDICTIONS,
        ROOT / DEFAULT_METRICS,
        event_id=event["id"],
        event_time=event["hour"],
    )
    reports = event.get("reports", [])
    grid_centers = [tuple(grid["center"]) for grid in event.get("broad", [])]
    active_grids = {
        min(range(len(grid_centers)), key=lambda idx: (report[0] - grid_centers[idx][0]) ** 2 + (report[1] - grid_centers[idx][1]) ** 2)
        for report in reports
    } if reports and grid_centers else set()
    intensities = [float(report[2]) for report in reports if len(report) > 2]
    return replace(
        forecast,
        initial_complaint_count=int(event.get("initialCount", len(reports))),
        initial_grid_count=len(active_grids) or None,
        initial_intensity_average=round(sum(intensities) / len(intensities), 1) if intensities else None,
        initial_intensity_maximum=round(max(intensities), 1) if intensities else None,
        weather=event.get("weather", {}),
    )


def generate_documents(event: dict, update_progress=None) -> tuple[object, str, str | None]:
    update = update_progress or (lambda _value, _message: None)
    update(10, "Event 예측정보를 정리하고 있습니다.")
    forecast = current_forecast(event)
    update(35, "행정문서 기본 양식을 작성하고 있습니다.")
    safe = build_response_package(forecast)
    if not llm_configured():
        update(100, "안전 템플릿 문서 생성이 완료되었습니다.")
        return safe, "안전 템플릿", None
    try:
        update(60, f"{provider_name()}가 문안을 보정하고 있습니다.")
        refined = refine_with_llm(forecast, safe)
        update(100, "대응 문서 생성이 완료되었습니다.")
        return refined, provider_name(), None
    except Exception as exc:
        update(100, "LLM 대신 안전 템플릿으로 생성을 완료했습니다.")
        return safe, "안전 템플릿", f"LLM 호출 실패로 안전 템플릿을 사용했습니다: {exc}"


def store_generated_documents(event: dict) -> None:
    progress = st.progress(0, text="대응 문서 생성을 준비하고 있습니다.")
    package, mode, warning = generate_documents(
        event,
        lambda value, message: progress.progress(value, text=f"{value}% · {message}"),
    )
    st.session_state.documents = package
    st.session_state.document_event = event["id"]
    st.session_state.document_mode = mode
    st.session_state.document_warning = warning


def apply_styles() -> None:
    st.markdown("""
    <style>
    [data-testid="stHeader"]{background:#172e3d;height:3.6rem}
    [data-testid="stHeader"]:before{content:"익산  악취 대응 AI";color:white;font-weight:800;font-size:1.05rem;position:absolute;left:1.3rem;top:1rem}
    [data-testid="stSidebar"]{border-right:1px solid #dce2e5}
    [data-testid="stSidebar"]>div:first-child{padding-top:1.1rem}
    .block-container{padding-top:4.6rem;padding-bottom:1rem;max-width:none}
    .event-id{font-size:1.45rem;font-weight:800;color:#182126}.event-time{font-size:.78rem;color:#68757c;margin-bottom:.8rem}
    .eyebrow{font-size:.68rem;font-weight:800;color:#68757c;text-transform:uppercase;letter-spacing:.06em;margin:.35rem 0 .45rem}
    .risk-box{background:#fff8e8;border-left:4px solid #e9a11b;padding:.8rem .9rem;margin:.4rem 0 .8rem;font-size:.84rem}
    .risk-box small{color:#8a6827}.priority{background:#f7f8f8;border-left:3px solid #e9a11b;padding:.55rem .65rem;margin:.35rem 0;font-size:.8rem}.priority.first{border-color:#dd3e36}
    .notice{background:#fff6dc;border-left:3px solid #e9a11b;padding:.65rem .8rem;font-size:.8rem;margin-bottom:.7rem}
    .weather-cards{display:grid;grid-template-columns:1fr 1fr;gap:.55rem;margin:.25rem 0 .35rem}
    .weather-card{display:flex;align-items:center;gap:.55rem;min-width:0;background:#fff;border:1px solid #dce2e5;padding:.65rem .6rem}
    .weather-icon{font-size:1.15rem;line-height:1;flex:0 0 auto}.weather-copy{min-width:0}
    .weather-label{font-size:.68rem;color:#68757c;margin-bottom:.15rem}.weather-value{font-size:.92rem;font-weight:750;color:#182126;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
    div[data-testid="stMetric"]{background:white;border:1px solid #dce2e5;padding:.55rem}.stButton button{border-radius:4px;font-weight:700}
    iframe[title="streamlit_folium.st_folium"]{border:1px solid #dce2e5}
    .st-key-map_panel{position:sticky;top:4.2rem;align-self:flex-start}
    .st-key-agent_panel{height:calc(100vh - 5.1rem);overflow-y:auto;overflow-x:hidden;padding-right:.65rem;scrollbar-gutter:stable}
    .st-key-agent_panel::-webkit-scrollbar{width:8px}
    .st-key-agent_panel::-webkit-scrollbar-thumb{background:#bdc8cd;border-radius:8px}
    .st-key-agent_panel::-webkit-scrollbar-track{background:#f3f5f6}
    .st-key-agent_panel h1{font-size:1.7rem;line-height:1.25;margin:.8rem 0 .55rem}
    .st-key-agent_panel h2{font-size:1.25rem;line-height:1.3;margin:1rem 0 .45rem}
    .st-key-agent_panel h3{font-size:1.05rem;line-height:1.35;margin:.8rem 0 .4rem}
    .st-key-agent_panel p,.st-key-agent_panel li,.st-key-agent_panel table{font-size:.88rem;line-height:1.55}
    @media(max-width:760px){
      .block-container{padding-top:4.3rem;padding-left:.6rem;padding-right:.6rem}.weather-value{font-size:.84rem}
      .st-key-map_panel{position:static}.st-key-agent_panel{height:auto;overflow:visible;padding-right:0;scrollbar-gutter:auto}
    }
    </style>
    """, unsafe_allow_html=True)


_load_secrets()
apply_styles()
data = load_demo_data()
events = data.get("events", [])
if not events:
    st.error("표시할 Event 데이터가 없습니다.")
    st.stop()

if "event_index" not in st.session_state:
    st.session_state.event_index = len(events) - 1
if "documents" not in st.session_state:
    st.session_state.documents = None
if "document_event" not in st.session_state:
    st.session_state.document_event = None
if st.session_state.get("document_schema_version") != DOCUMENT_SCHEMA_VERSION:
    st.session_state.documents = None
    st.session_state.document_event = None
    st.session_state.document_schema_version = DOCUMENT_SCHEMA_VERSION

event = events[st.session_state.event_index]
grids = event.get("broad", [])
weather = event.get("weather", {})

with st.sidebar:
    st.markdown('<div class="eyebrow">Selected Event</div>', unsafe_allow_html=True)
    left, middle, right = st.columns([4, 1, 1])
    with left:
        st.markdown(f'<div class="event-id">{event["id"]}</div><div class="event-time">{event["hour"]} 기준</div>', unsafe_allow_html=True)
    with middle:
        if st.button("‹", disabled=st.session_state.event_index == 0, use_container_width=True, key="prev"):
            st.session_state.event_index -= 1
            st.session_state.documents = None
            st.session_state.show_actual = False
            st.rerun()
    with right:
        if st.button("›", disabled=st.session_state.event_index == len(events) - 1, use_container_width=True, key="next"):
            st.session_state.event_index += 1
            st.session_state.documents = None
            st.session_state.show_actual = False
            st.rerun()

    m1, m2, m3 = st.columns(3)
    m1.metric("초기 신고", event.get("initialCount", 0))
    m2.metric("점검 권역", min(3, len(grids)))
    m3.metric("이후 신고", event.get("futureCount", 0))

    st.markdown('<div class="eyebrow">1km Complaint Forecast</div>', unsafe_allow_html=True)
    st.markdown('<div class="risk-box"><b>최우선 점검 권역 안내</b><br>1순위 권역부터 현장 확인을 권고합니다.<br><small>예측 확률이 아닌 권역 간 상대 순위입니다.</small></div>', unsafe_allow_html=True)

    st.markdown('<div class="eyebrow">현장 참고 기상정보</div>', unsafe_allow_html=True)
    wind_speed = "-" if weather.get("windSpeed") is None else f'{weather["windSpeed"]} m/s'
    wind_direction = "-" if weather.get("windDirection") is None else f'{weather["windDirection"]}°'
    humidity = "-" if weather.get("humidity") is None else f'{weather["humidity"]} %'
    rainfall = "-" if weather.get("rainfall") is None else f'{weather["rainfall"]} mm'
    st.markdown(f'''<div class="weather-cards">
      <div class="weather-card"><span class="weather-icon">💨</span><div class="weather-copy"><div class="weather-label">풍속</div><div class="weather-value">{wind_speed}</div></div></div>
      <div class="weather-card"><span class="weather-icon">🧭</span><div class="weather-copy"><div class="weather-label">풍향</div><div class="weather-value">{wind_direction}</div></div></div>
      <div class="weather-card"><span class="weather-icon">💧</span><div class="weather-copy"><div class="weather-label">상대습도</div><div class="weather-value">{humidity}</div></div></div>
      <div class="weather-card"><span class="weather-icon">🌧️</span><div class="weather-copy"><div class="weather-label">최근 강수</div><div class="weather-value">{rainfall}</div></div></div>
    </div>''', unsafe_allow_html=True)
    st.caption("※ 기상정보는 민원 예측 모델 입력에 사용하지 않음")

    st.markdown('<div class="eyebrow">Dispatch Priority</div>', unsafe_allow_html=True)
    for idx, grid in enumerate(grids[:3], 1):
        cls = "priority first" if idx == 1 else "priority"
        action = "최우선 현장 확인" if idx == 1 else "순차 점검"
        st.markdown(f'<div class="{cls}"><b>{idx}순위 · 1km 권역</b><br><small>{action} · 상대점수 {grid["score"]:.0%}</small></div>', unsafe_allow_html=True)

    st.markdown('<div class="eyebrow">Administrative Agent</div>', unsafe_allow_html=True)
    if st.button("대응 문서 생성", type="primary", use_container_width=True):
        store_generated_documents(event)
    st.caption(f'{provider_name()} LLM 연결됨' if llm_configured() else "안전 템플릿 모드 · API 키 미설정")
    show_actual = st.toggle("검증용 실제 이후 신고 표시", value=False, key="show_actual")

map_column, agent_column = st.columns([1.45, 1], gap="large")

with map_column.container(key="map_panel"):
    st.markdown("#### 과거 Event 재현 모드")
    st.caption("1km 광역 경보 · 향후 30분 · 지도 위험도는 Event 내부 상대 순위입니다.")
    st_folium(event_map(event, show_actual), use_container_width=True, height=650, returned_objects=[])

with agent_column.container(key="agent_panel"):
    st.markdown("#### 행정 대응 Agent")
    if st.session_state.documents is None or st.session_state.document_event != event["id"]:
        st.markdown(
            '<div class="notice"><b>예측 결과를 현장 대응 문서로 변환합니다.</b><br>'
            '상황 브리핑, 현장점검 지시서와 입력 가능한 사후 결과보고서를 생성할 수 있습니다.</div>',
            unsafe_allow_html=True,
        )
        if st.button("이 Event의 대응 문서 생성", type="primary", use_container_width=True, key="generate_main"):
            store_generated_documents(event)
            st.rerun()
        st.caption("문서를 생성하면 이 영역에서 바로 확인하고 현장 결과를 입력할 수 있습니다.")
    else:
        package = st.session_state.documents
        warning = st.session_state.get("document_warning")
        message = warning or f'{st.session_state.get("document_mode", "안전 템플릿")} 문서 생성 완료 · 담당자 검토 필요'
        st.markdown(f'<div class="notice">{message}</div>', unsafe_allow_html=True)
        tab1, tab2, tab3, tab4 = st.tabs(["① 상황 브리핑", "② 점검 지시서", "③ 사후 결과 입력", "④ AI 대응 가이드"])
        with tab1:
            st.markdown(package.briefing)
            st.download_button("브리핑 다운로드", package.briefing, f"briefing_{event['id']}.md", "text/markdown")
        with tab2:
            st.markdown(package.dispatch_order)
            st.download_button("점검 지시서 다운로드", package.dispatch_order, f"inspection_{event['id']}.md", "text/markdown")
        with tab3:
            st.caption("문서에서 ‘미입력’으로 표시된 현장 확인값을 아래에서 직접 작성하세요.")
            with st.expander("빈 사후 결과보고서 양식 보기"):
                st.markdown(package.followup_report_template)
                st.download_button("빈 양식 다운로드", package.followup_report_template, f"followup_template_{event['id']}.md", "text/markdown")
            with st.form("followup_form"):
                author = st.text_input("작성자", placeholder="예: 홍길동 주무관")
                date_col, time_col = st.columns(2)
                inspected_date = date_col.date_input("점검 날짜", value=date.today())
                inspected_time = time_col.time_input("점검 시각", value=datetime.now().time().replace(second=0, microsecond=0))
                time1, time2, time3 = st.columns(3)
                dispatch_decided_at = time1.text_input("출동 결정시각", placeholder="예: 20:32")
                departed_at = time2.text_input("현장 출발시각", placeholder="예: 20:35")
                arrived_at = time3.text_input("현장 도착시각", placeholder="예: 20:48")
                distance_col, count_col = st.columns(2)
                total_distance_km = distance_col.text_input("총 출동거리(km)", placeholder="예: 5.2")
                actual_additional_area_count = count_col.text_input("실제 추가 민원 권역 수", placeholder="예: 2")
                areas = []
                for area in package.forecast.areas:
                    st.markdown(f"**{area.rank}순위 · {area.grid_id}**")
                    c1, c2 = st.columns(2)
                    additional = c1.selectbox("추가 민원", ["미확인", "발생", "미발생"], key=f"add_{event['id']}_{area.rank}")
                    detected = c2.selectbox("악취 감지", ["미확인", "감지", "미감지"], key=f"odor_{event['id']}_{area.rank}")
                    measurement = st.text_input("측정 결과", key=f"measurement_{event['id']}_{area.rank}")
                    action = st.text_input("조치 내용", key=f"action_{event['id']}_{area.rank}")
                    areas.append({"rank": area.rank, "additional_complaint": additional, "odor_detected": detected, "measurement": measurement, "action": action})
                checked_area = st.text_input("실제 우선 점검 권역", placeholder="예: 1순위 권역 또는 격자 ID")
                field_findings = st.text_area("현장 확인내용", placeholder="현장에서 확인한 악취 상태와 주변 상황을 입력하세요.")
                notes = st.text_area("담당자 의견 및 종합 결과", placeholder="실시 조치와 추가 확인 필요사항을 입력하세요.")
                followup_required = st.radio("추가 조치 필요 여부", ["미확인", "필요", "불필요"], horizontal=True)
                submitted = st.form_submit_button("입력값으로 결과보고서 완성", type="primary", use_container_width=True)
            if submitted:
                inspected_at = datetime.combine(inspected_date, inspected_time)
                report = create_completed_followup(package, {
                    "author": author, "inspected_at": inspected_at.isoformat(timespec="minutes"),
                    "dispatch_decided_at": dispatch_decided_at, "departed_at": departed_at, "arrived_at": arrived_at,
                    "total_distance_km": total_distance_km, "actual_additional_area_count": actual_additional_area_count,
                    "checked_area": checked_area, "field_findings": field_findings,
                    "followup_required": followup_required, "areas": areas, "notes": notes,
                })
                st.markdown(report)
                st.download_button("완성 보고서 다운로드", report, f"followup_{event['id']}.md", "text/markdown")
        with tab4:
            response_guide = getattr(package, "response_guide", None) or create_response_guide(package.forecast)
            st.markdown(response_guide)
            st.download_button("AI 대응 가이드 다운로드", response_guide, f"response_guide_{event['id']}.md", "text/markdown")
