"""익산시 악취 민원 선제 대응 AI - Streamlit 배포 앱."""
from __future__ import annotations

from datetime import date, datetime
import json
import math
import os
from pathlib import Path
import re

import folium
import streamlit as st
from streamlit_folium import st_folium

from administrative_agent.llm import llm_configured, provider_name, refine_with_llm
from administrative_agent.service import build_response_package, create_completed_followup
from generate_agent_documents import DEFAULT_METRICS, DEFAULT_PREDICTIONS, forecast_from_csv


ROOT = Path(__file__).resolve().parent
DEMO_DATA = ROOT / "demo" / "demo-data.js"

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
    return forecast_from_csv(
        ROOT / DEFAULT_PREDICTIONS,
        ROOT / DEFAULT_METRICS,
        event_id=event["id"],
        event_time=event["hour"],
    )


def generate_documents(event: dict) -> tuple[object, str, str | None]:
    forecast = current_forecast(event)
    safe = build_response_package(forecast)
    if not llm_configured():
        return safe, "안전 템플릿", None
    try:
        return refine_with_llm(forecast, safe), provider_name(), None
    except Exception as exc:
        return safe, "안전 템플릿", f"LLM 호출 실패로 안전 템플릿을 사용했습니다: {exc}"


def apply_styles() -> None:
    st.markdown("""
    <style>
    [data-testid="stHeader"]{background:#172e3d;height:3.6rem}
    [data-testid="stHeader"]:before{content:"익산  악취 대응 AI";color:white;font-weight:800;font-size:1.05rem;position:absolute;left:1.3rem;top:1rem}
    [data-testid="stSidebar"]{border-right:1px solid #dce2e5}
    [data-testid="stSidebar"]>div:first-child{padding-top:1.1rem}
    .block-container{padding-top:1.1rem;padding-bottom:1rem;max-width:none}
    .event-id{font-size:1.45rem;font-weight:800;color:#182126}.event-time{font-size:.78rem;color:#68757c;margin-bottom:.8rem}
    .eyebrow{font-size:.68rem;font-weight:800;color:#68757c;text-transform:uppercase;letter-spacing:.06em;margin:.35rem 0 .45rem}
    .risk-box{background:#fff8e8;border-left:4px solid #e9a11b;padding:.8rem .9rem;margin:.4rem 0 .8rem;font-size:.84rem}
    .risk-box small{color:#8a6827}.priority{background:#f7f8f8;border-left:3px solid #e9a11b;padding:.55rem .65rem;margin:.35rem 0;font-size:.8rem}.priority.first{border-color:#dd3e36}
    .notice{background:#fff6dc;border-left:3px solid #e9a11b;padding:.65rem .8rem;font-size:.8rem;margin-bottom:.7rem}
    div[data-testid="stMetric"]{background:white;border:1px solid #dce2e5;padding:.55rem}.stButton button{border-radius:4px;font-weight:700}
    iframe[title="streamlit_folium.st_folium"]{border:1px solid #dce2e5}
    @media(max-width:760px){.block-container{padding-left:.6rem;padding-right:.6rem}}
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
            st.rerun()
    with right:
        if st.button("›", disabled=st.session_state.event_index == len(events) - 1, use_container_width=True, key="next"):
            st.session_state.event_index += 1
            st.session_state.documents = None
            st.rerun()

    m1, m2, m3 = st.columns(3)
    m1.metric("초기 신고", event.get("initialCount", 0))
    m2.metric("점검 권역", min(3, len(grids)))
    m3.metric("이후 신고", event.get("futureCount", 0))

    st.markdown('<div class="eyebrow">1km Complaint Forecast</div>', unsafe_allow_html=True)
    st.markdown('<div class="risk-box"><b>최우선 점검 권역 안내</b><br>1순위 권역부터 현장 확인을 권고합니다.<br><small>예측 확률이 아닌 권역 간 상대 순위입니다.</small></div>', unsafe_allow_html=True)

    st.markdown('<div class="eyebrow">현장 참고 기상정보</div>', unsafe_allow_html=True)
    w1, w2 = st.columns(2)
    w1.metric("풍속", f'{weather.get("windSpeed", "-")} m/s')
    w2.metric("풍향", f'{weather.get("windDirection", "-")}°')
    w3, w4 = st.columns(2)
    w3.metric("상대습도", f'{weather.get("humidity", "-")} %')
    w4.metric("최근 강수", f'{weather.get("rainfall", "-")} mm')
    st.caption("※ 기상정보는 민원 예측 모델 입력에 사용하지 않음")

    st.markdown('<div class="eyebrow">Dispatch Priority</div>', unsafe_allow_html=True)
    for idx, grid in enumerate(grids[:3], 1):
        cls = "priority first" if idx == 1 else "priority"
        action = "최우선 현장 확인" if idx == 1 else "순차 점검"
        st.markdown(f'<div class="{cls}"><b>{idx}순위 · 1km 권역</b><br><small>{action} · 상대점수 {grid["score"]:.0%}</small></div>', unsafe_allow_html=True)

    st.markdown('<div class="eyebrow">Administrative Agent</div>', unsafe_allow_html=True)
    if st.button("대응 문서 생성", type="primary", use_container_width=True):
        with st.spinner("대응 문서를 생성하고 있습니다..."):
            package, mode, warning = generate_documents(event)
            st.session_state.documents = package
            st.session_state.document_event = event["id"]
            st.session_state.document_mode = mode
            st.session_state.document_warning = warning
    st.caption(f'{provider_name()} LLM 연결됨' if llm_configured() else "안전 템플릿 모드 · API 키 미설정")
    show_actual = st.toggle("검증용 실제 이후 신고 표시", value=False)

st.markdown("#### 과거 Event 재현 모드")
st.caption("1km 광역 경보 · 향후 30분 · 지도 위험도는 Event 내부 상대 순위입니다.")
st_folium(event_map(event, show_actual), use_container_width=True, height=650, returned_objects=[])

if st.session_state.documents is not None and st.session_state.document_event == event["id"]:
    package = st.session_state.documents
    st.divider()
    st.subheader("행정 대응 Agent")
    warning = st.session_state.get("document_warning")
    message = warning or f'{st.session_state.get("document_mode", "안전 템플릿")} 문서 생성 완료 · 담당자 검토 필요'
    st.markdown(f'<div class="notice">{message}</div>', unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs(["상황 브리핑", "점검 지시서", "사후보고서 예시", "현장 결과 입력"])
    with tab1:
        st.markdown(package.briefing)
        st.download_button("브리핑 다운로드", package.briefing, f"briefing_{event['id']}.md", "text/markdown")
    with tab2:
        st.markdown(package.dispatch_order)
        st.download_button("점검 지시서 다운로드", package.dispatch_order, f"inspection_{event['id']}.md", "text/markdown")
    with tab3:
        st.markdown(package.followup_report_template)
        st.download_button("사후보고서 양식 다운로드", package.followup_report_template, f"followup_template_{event['id']}.md", "text/markdown")
    with tab4:
        with st.form("followup_form"):
            author = st.text_input("작성자")
            date_col, time_col = st.columns(2)
            inspected_date = date_col.date_input("점검 날짜", value=date.today())
            inspected_time = time_col.time_input("점검 시각", value=datetime.now().time().replace(second=0, microsecond=0))
            areas = []
            for area in package.forecast.areas:
                st.markdown(f"**{area.rank}순위 · {area.grid_id}**")
                c1, c2 = st.columns(2)
                additional = c1.selectbox("추가 민원", ["미확인", "발생", "미발생"], key=f"add_{event['id']}_{area.rank}")
                detected = c2.selectbox("악취 감지", ["미확인", "감지", "미감지"], key=f"odor_{event['id']}_{area.rank}")
                measurement = st.text_input("측정 결과", key=f"measurement_{event['id']}_{area.rank}")
                action = st.text_input("조치 내용", key=f"action_{event['id']}_{area.rank}")
                areas.append({"rank": area.rank, "additional_complaint": additional, "odor_detected": detected, "measurement": measurement, "action": action})
            notes = st.text_area("담당자 의견")
            submitted = st.form_submit_button("사후 결과보고서 생성", type="primary")
        if submitted:
            inspected_at = datetime.combine(inspected_date, inspected_time)
            report = create_completed_followup(package, {"author": author, "inspected_at": inspected_at.isoformat(timespec="minutes"), "areas": areas, "notes": notes})
            st.markdown(report)
            st.download_button("완성 보고서 다운로드", report, f"followup_{event['id']}.md", "text/markdown")
