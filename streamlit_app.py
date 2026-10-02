"""익산시 악취 민원 선제 대응 AI - Streamlit 배포 앱."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from html import escape
import json
import math
import logging
import os
from pathlib import Path

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from administrative_agent.models import FieldCandidate, ForecastResult, RiskArea
from administrative_agent.time_utils import KST, now_kst
from administrative_agent.llm import llm_configured, provider_name, refine_with_llm
from administrative_agent.service import build_response_package, create_completed_followup
from generate_agent_documents import DEFAULT_METRICS, DEFAULT_PREDICTIONS, forecast_from_csv, _relative_scores
from odor_service.region_prediction.data import assign_grid


ROOT = Path(__file__).resolve().parent
REPLAY_DIR = ROOT / "outputs" / "odor_service" / "replay_legacy"
FARMS_FILE = ROOT / "outputs" / "odor_service" / "data" / "farms.parquet"
FONT_CSS_URL = "https://hangeul.pstatic.net/hangeul_static/css/nanum-square-neo.css"
DOCUMENT_SCHEMA_VERSION = 5
MAX_FIELD_CANDIDATES = 3
# 예측 권역은 순위보다 "추가 민원 예상 범위"라는 의미를 우선해 동일한 빨간색으로 표시한다.
RISK_COLOR = "#dd3e36"

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


def replay_revision() -> int:
    """Replay 파일이 다시 생성되면 Streamlit 캐시 키도 함께 바뀐다."""
    return max((path.stat().st_mtime_ns for path in REPLAY_DIR.glob("*.json")), default=0)


@st.cache_data
def load_field_replays(revision: int) -> list[dict]:
    """과거 Event 시각과 가장 가까운 현장 후보 replay를 찾기 위한 색인."""
    rows = []
    for path in sorted(REPLAY_DIR.glob("*.json")):
        if path.name == "index.json":
            continue
        replay = json.loads(path.read_text(encoding="utf-8"))
        for field_event in replay.get("events", []):
            rows.append({
                "event_time": datetime.fromisoformat(field_event["event_hour"]),
                "event": field_event,
                "standby": replay.get("standby", {}),
                "afternoon": replay.get("afternoon", {}),
            })
    return rows


@st.cache_data
def load_farm_locations() -> dict[str, tuple[float, float]]:
    if not FARMS_FILE.exists():
        return {}
    farms = pd.read_parquet(FARMS_FILE, columns=["farm_id", "lat", "lon"]).dropna(subset=["lat", "lon"])
    return {str(row.farm_id): (float(row.lat), float(row.lon)) for row in farms.itertuples(index=False)}


def field_context_for_event(event: dict) -> dict | None:
    """동일 ID의 사건만 연결한다. 시간 근접성을 사건 동일성으로 간주하지 않는다."""
    replays = load_field_replays(replay_revision())
    if not replays:
        return None
    exact = next((row for row in replays if row["event"].get("event_id") == event.get("id")), None)
    if exact:
        return {**exact, "delta_minutes": 0}
    return None


def unified_event_view(event: dict, field_context: dict | None) -> dict:
    """화면의 모든 단계가 같은 민원 집중 사건을 바라보도록 표시용 사건을 만든다."""
    if not field_context:
        return event
    source = field_context["event"]
    first = source.get("first30_complaints", [])
    broad = [
        {
            "center": [row["lat"], row["lon"]],
            "region": row.get("region_name"),
            "score": float(row.get("score", 0.0)),
            "actual": bool(row.get("hit", False)),
        }
        for row in source.get("top3", [])
    ]
    display = dict(event)
    # 화면 표시는 하나의 field event로 통일하되, 기존 예측 CSV 조회용 ID는 보존한다.
    # 문서 생성 등 레거시 기능이 기존 EVT-* 키를 계속 사용할 수 있도록 하기 위한 값이다.
    display["legacy_id"] = event.get("id")
    display["legacy_hour"] = event.get("hour")
    display.update({
        "id": source.get("event_id", event.get("id")),
        "hour": str(source.get("event_hour", event.get("hour"))).replace("T", " ")[:16],
        "initialCount": len(first),
        "futureCount": "—",
        "reports": [[row["lat"], row["lon"], row.get("intensity")] for row in first],
        "broad": broad,
        "weather": {
            "windDirection": source.get("direction_overlap", {}).get("wind_direction"),
            "windSpeed": source.get("direction_overlap", {}).get("wind_speed"),
            "humidity": source.get("weather", {}).get("humidity"),
            "rainfall": source.get("weather", {}).get("rainfall"),
        },
    })
    return display


def event_map(event: dict, show_actual: bool, field_context: dict | None = None) -> folium.Map:
    reports = event.get("reports", [])
    grids = event.get("broad", [])[:3]
    points = [[r[0], r[1]] for r in reports] + [g["center"] for g in grids]
    farm_locations = load_farm_locations() if field_context else {}
    if field_context:
        for card in field_context["event"].get("field", {}).get("cards", [])[:MAX_FIELD_CANDIDATES]:
            location = farm_locations.get(str(card.get("farm_id")))
            if not location and card.get("lat") is not None and card.get("lon") is not None:
                location = (float(card["lat"]), float(card["lon"]))
            if location:
                points.append(list(location))
    center = points[0] if points else [35.95, 126.98]
    fmap = folium.Map(location=center, zoom_start=13, tiles="OpenStreetMap", control_scale=True)
    fmap.get_root().header.add_child(folium.Element(f"""
    <style>
    @import url('{FONT_CSS_URL}');
    html, body, .leaflet-container, .leaflet-tooltip, .leaflet-popup-content {{
        font-family:'NanumSquareNeoVariable','NanumSquareNeo','Malgun Gothic',sans-serif !important;
    }}
    .leaflet-tooltip {{font-weight:650;letter-spacing:-0.02em;}}
    </style>
    """))

    for idx, report in enumerate(reports, 1):
        folium.CircleMarker(
            [report[0], report[1]], radius=5, color="#ffffff", weight=2,
            fill=True, fill_color="#172e3d", fill_opacity=1,
            tooltip=f"초기 신고 {idx}" + (f" · 악취강도 {report[2]}" if len(report) > 2 and report[2] is not None else " · 악취강도 미제공"),
        ).add_to(fmap)

    for idx, grid in enumerate(grids, 1):
        lat, lon = grid["center"]
        dy = 1000 / 110540 / 2
        dx = 1000 / (111320 * max(0.1, math.cos(lat * math.pi / 180))) / 2
        base = RISK_COLOR
        actual = show_actual and bool(grid.get("actual"))
        folium.Rectangle(
            bounds=[[lat - dy, lon - dx], [lat + dy, lon + dx]],
            color="#15866f" if actual else base, weight=3 if actual else 2,
            fill=True, fill_color=base, fill_opacity=0.58,
            tooltip=f"{idx}순위 1km 권역 · 현장 점검 대상" + (" · 실제 이후 신고" if actual else ""),
        ).add_to(fmap)

    if field_context:
        for card in field_context["event"].get("field", {}).get("cards", [])[:MAX_FIELD_CANDIDATES]:
            location = farm_locations.get(str(card.get("farm_id")))
            if not location and card.get("lat") is not None and card.get("lon") is not None:
                location = (float(card["lat"]), float(card["lon"]))
            if not location:
                continue
            merged_count = int(card.get("merged_count") or 1)
            merged_names = [item.strip() for item in str(card.get("merged_names") or card.get("name") or card.get("farm_id")).split("|") if item.strip()]
            names_text = ", ".join(merged_names)
            marker_label = f"공동 방문 지점 · {names_text} · 등록 {merged_count}건" if merged_count > 1 else names_text
            popup_names = "".join(f"<li>{escape(item)}</li>" for item in merged_names)
            folium.Marker(
                location,
                tooltip=f'{card["rank"]}순위 · {marker_label} · {card["score"]}점',
                popup=folium.Popup(
                    f'<b>{card["rank"]}순위 방문 지점</b><br><ul style="padding-left:18px;margin:8px 0">{popup_names}</ul>'
                    + (f'<small>같은 좌표 또는 300m 이내 동명 등록 {merged_count}건</small>' if merged_count > 1 else ""),
                    max_width=320,
                ),
                icon=folium.DivIcon(html=(
                    '<div style="background:#15866f;color:white;border:2px solid white;border-radius:50%;'
                    'width:28px;height:28px;line-height:24px;text-align:center;font-weight:800;'
                    'box-shadow:0 1px 4px #333">'
                    f'{card["rank"]}</div>'
                )),
            ).add_to(fmap)

    if points:
        fmap.fit_bounds(points, padding=(35, 35), max_zoom=14)
    return fmap


def render_operation_header(event: dict, field_context: dict | None) -> None:
    st.markdown(
        '<div class="operation-hero">'
        '<div><span class="status-badge">과거 상황 재현</span>'
        '<h2>이후 민원 발생 예측부터 우선 점검 농가 후보까지</h2>'
        '<p>이후 민원이 발생할 것으로 예측되는 권역과 우선적으로 확인할 농가 후보를 함께 보여줍니다.</p></div>'
        f'<div class="hero-event"><small>현재 Event</small><b>{escape(str(event["id"]))}</b>'
        f'<span>{escape(str(event["hour"]))}</span></div>'
        '</div>',
        unsafe_allow_html=True,
    )
def change_candidate(offset: int, candidate_count: int) -> None:
    current = int(st.session_state.get("candidate_index", 0))
    st.session_state.candidate_index = min(max(0, current + offset), candidate_count - 1)


def friendly_candidate_summary(card: dict) -> str:
    """후보 카드에 표시할 쉬운 설명을 만든다. 내부 계산 용어는 상세 근거에만 둔다."""
    tier = card.get("tier")
    support = card.get("support") or {}
    total = support.get("total", card.get("support_total", "-"))
    count = support.get("count", card.get("support_count", "-"))
    distance = card.get("complaint_km")
    if tier == "교차 확인":
        text = f"전체 민원 위치 {total}곳 중 {count}곳이 이 농가 방향을 가리켜 우선 확인 후보로 선정했습니다."
    elif tier == "단일 지점 참고":
        text = f"민원 위치 {count}곳이 이 농가 방향을 가리켜 참고 후보로 선정했습니다."
    else:
        text = "기본 조건을 만족하는 후보가 부족해 풍향과 민원 거리 등을 참고한 보충 후보입니다."
    if distance is not None:
        text += f" 가장 가까운 민원 위치와의 거리는 약 {float(distance):.1f}km입니다."
    return text


def render_field_candidates(field_context: dict | None) -> None:
    st.markdown("#### 우선 점검 농가 후보")
    st.caption("예측 Top 3 권역에서 농가를 다시 고르는 방식이 아닙니다. 이 사건의 초기 30분 민원 위치와 당시 풍향을 바탕으로 먼저 확인할 순서를 계산합니다.")
    if field_context is None:
        st.info("동일 사건 ID의 현장 후보 재현 자료가 없습니다.")
        return

    field_event = field_context["event"]
    field = field_event.get("field", {})
    cards = field.get("cards", [])[:MAX_FIELD_CANDIDATES]
    wind_speed = field_event.get("direction_overlap", {}).get("wind_speed")
    wind_text = "-" if wind_speed is None else f"{wind_speed} m/s"
    confidence = escape(str(field.get("confidence") or "정보 없음"))
    st.markdown(
        '<div class="field-summary">'
        f'<div><small>풍속</small><b>{escape(wind_text)}</b></div>'
        f'<div><small>풍향 신뢰도</small><b>{confidence}</b></div>'
        '</div>',
        unsafe_allow_html=True,
    )
    if field.get("selection") == "단일 지점 참고":
        st.warning("여러 민원 위치가 함께 가리키는 농가가 없어 단일 위치 기준 후보를 표시합니다.")
    if field.get("confidence") in ("낮음", "매우 낮음"):
        st.warning("바람이 약해 방향 신뢰도가 낮습니다. 현장에서 풍향과 냄새를 먼저 확인하세요.")
    if not cards:
        st.info("현재 조건에 맞는 현장 확인 후보가 없습니다.")

    if cards:
        current = min(max(0, st.session_state.get("candidate_index", 0)), len(cards) - 1)
        st.session_state.candidate_index = current
        previous, position, following = st.columns([1, 2, 1])
        with previous:
            st.button(
                "‹ 이전", disabled=current == 0, use_container_width=True, key="candidate_prev",
                on_click=change_candidate, args=(-1, len(cards)),
            )
        with following:
            st.button(
                "다음 ›", disabled=current == len(cards) - 1, use_container_width=True, key="candidate_next",
                on_click=change_candidate, args=(1, len(cards)),
            )
        with position:
            st.markdown(
                f'<div class="candidate-position"><b>{current + 1}</b> / {len(cards)}</div>',
                unsafe_allow_html=True,
            )

    current = st.session_state.get("candidate_index", 0)
    for card in cards[current:current + 1]:
        name = escape(str(card.get("name") or card.get("farm_id") or "이름 없음"))
        summary = escape(friendly_candidate_summary(card))
        merged_count = int(card.get("merged_count") or 1)
        merged_names = [escape(item.strip()) for item in str(card.get("merged_names") or card.get("name") or "").split("|") if item.strip()]
        addresses = [escape(item.strip()) for item in str(card.get("merged_addresses") or card.get("address") or "주소 정보 없음").split("|") if item.strip()]
        address_text = "<br>".join(dict.fromkeys(addresses)) or "주소 정보 없음"
        if merged_count > 1:
            names_html = "".join(f"<li>{item}</li>" for item in merged_names)
            repeated = f'<span>동일 명칭으로 등록된 시설 {merged_count}건</span>' if len(merged_names) == 1 else ""
            merged_note = (
                f'<div class="merged-note"><b>함께 확인할 등록 농장</b><ul>{names_html}</ul>{repeated}'
                f'</div>'
            )
            title = "공동 방문 지점"
        else:
            merged_note = ""
            title = name
        rank_class = " first" if card.get("rank") == 1 else ""
        support = card.get("support") or {}
        tier = card.get("tier")
        distance_html = ""
        else:
            distance_html = f'<span class="distance-badge">초기 30분 가축 민원 위치 중 가장 가까운 지점에서 {float(card["complaint_km"]):.1f} km</span>'
        score = float(card.get("score") or 0)
        st.markdown(
            f'<div class="farm-card{rank_class}">'
            f'<div class="farm-title"><b>{card.get("rank")}순위 · {title}</b>'
            f'<strong>{score:.1f}<small>/100</small></strong></div>'
            f'<div class="score-bar"><i style="width:{max(0, min(score, 100)):.1f}%"></i></div>'
            f'<div class="farm-address"><small>주소</small><span>{address_text}</span></div>'
            f'{merged_note}'
            f'<p class="farm-summary">{summary}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if card.get("coord_warning"):
            st.caption(f'{card.get("rank")}순위 · 대략적인 좌표입니다. 방문 전 위치를 확인하세요.')

        with st.popover(f'ⓘ {card.get("rank")}순위 점수·선정 근거'):
            st.caption('이 후보는 현재 민원 위치와 당시 바람 방향을 비교해 고른 우선 점검 대상입니다. 예측 권역 순위와는 별도로 계산합니다.')
            if tier == "교차 확인":
                st.markdown(f'**민원 위치 근거:** 전체 {support.get("total", "-")}곳 중 {support.get("count", "-")}곳이 이 농가 쪽을 가리킵니다.')
            elif tier == "단일 지점 참고":
                st.markdown('**한 민원 위치 근거:** 풍상·4km 조건을 만족한 초기 가축 민원 위치가 한 곳입니다.')
            elif tier == "보충 참고":
                st.markdown('**보충 후보:** 기본 조건 후보가 부족해 같은 점수 체계의 다음 후보를 표시합니다.')
            components = card.get("components") or {}
            st.markdown('**점수는 네 가지 근거를 합산합니다 (총 100점).**')
            st.markdown(f'**이 후보의 총점: {score:.1f}/100점**')
            score_parts = [
                ("풍향 일치", "후보 방향과 당시 바람 방향이 얼마나 맞는지", "풍향 일치", 20),
                ("민원 위치 거리", "가장 가까운 민원 위치와의 거리", "거리", 20),
                ("민원 중첩·공통 지목", "여러 민원 위치가 같은 후보를 가리키는 정도", "다중 측정 일치", 40),
                ("과거 반복 이력", "같은 시기·비슷한 풍향에서의 과거 민원", "과거 반복", 20),
            ]
            table = [
                "| 지표 | 쉬운 설명 | 이 후보 점수 |",
                "|---|---|---:|",
            ]
            table.extend(
                f'| {label} | {description} | **{float(components.get(component_key, 0)):.1f}/{maximum}점** |'
                for label, description, component_key, maximum in score_parts
            )
            st.markdown("\n".join(table))
            evidence = card.get("evidence") or []
            if evidence:
                st.markdown("**선정 근거**")
                for item in evidence:
                    st.markdown(f'- {escape(str(item))}')
    st.caption("※ 현장 확인 우선순위이며 발생원 확정을 위한 보조 도구로만 활용하세요.")


def current_forecast(event: dict, field_context: dict | None = None):
    field = field_context["event"].get("field", {}) if field_context else {}
    cards = field.get("cards", [])
    field_details = dict(field_confidence=field.get("confidence"),
                         field_stability=field.get("stability"),
                         field_stability_label=field.get("stability_label"),
                         field_notes=tuple(field.get("notes") or ()))
    candidates = tuple(FieldCandidate(
        rank=card["rank"],
        display_name=str(card.get("merged_names") or card.get("name") or card.get("farm_id")),
        address=str(card.get("merged_addresses") or card.get("address") or "주소 정보 없음"),
        score=float(card.get("score", 0)), components=card.get("components", {}),
        selection_summary=card.get("selection_summary") or card.get("reason") or "선정 요약 미제공",
        coord_warning=bool(card.get("coord_warning")),
        tier=card.get("tier"),
        support_count=(card.get("support") or {}).get("count"),
        support_total=(card.get("support") or {}).get("total"),
        complaint_km=card.get("complaint_km"),
        travel_km=card.get("travel_km"),
        visit_order=card.get("visit_order"),
    ) for card in cards[:MAX_FIELD_CANDIDATES])
    reports = event.get("reports", [])
    intensities = [float(report[2]) for report in reports if len(report) > 2 and report[2] is not None]
    if str(event.get("id", "")).startswith("EV2-"):
        initial_cells = assign_grid(pd.DataFrame(reports, columns=["latitude", "longitude", "intensity"])) if reports else None
        scores = _relative_scores(pd.Series([grid["score"] for grid in event.get("broad", [])[:3]]))
        # EV2 화면의 사건과 권역을 그대로 전달한다. 기존 모델의 지표는 섞지 않는다.
        return ForecastResult(
            event_id=event["id"], event_time=datetime.fromisoformat(event["hour"]),
            forecast_minutes=30, grid_size_m=1000,
            areas=tuple(RiskArea(
                rank=i, grid_id=f"{event['id']}-R{i}",
                relative_risk=scores[i - 1],
                center_latitude=grid["center"][0], center_longitude=grid["center"][1],
                region_name=grid.get("region"),
            ) for i, grid in enumerate(event.get("broad", [])[:3], 1)),
            model_metrics={}, generated_at=now_kst(),
            initial_complaint_count=event.get("initialCount"),
            initial_grid_count=len(initial_cells[["grid_x", "grid_y"]].drop_duplicates()) if initial_cells is not None else 0,
            initial_intensity_average=round(sum(intensities) / len(intensities), 1) if intensities else None,
            initial_intensity_maximum=max(intensities) if intensities else None,
            weather=event.get("weather", {}),
            event_time_is_boundary=True,
            field_candidates=candidates,
            **field_details,
        )
    legacy_id = event.get("legacy_id", event["id"])
    legacy_hour = event.get("legacy_hour", event["hour"])
    forecast = forecast_from_csv(
        ROOT / DEFAULT_PREDICTIONS,
        ROOT / DEFAULT_METRICS,
        event_id=legacy_id,
        event_time=legacy_hour,
    )
    reports = event.get("reports", [])
    grid_centers = [tuple(grid["center"]) for grid in event.get("broad", [])]
    active_grids = {
        min(range(len(grid_centers)), key=lambda idx: (report[0] - grid_centers[idx][0]) ** 2 + (report[1] - grid_centers[idx][1]) ** 2)
        for report in reports
    } if reports and grid_centers else set()
    intensities = [float(report[2]) for report in reports if len(report) > 2 and report[2] is not None]
    return replace(
        forecast,
        initial_complaint_count=int(event.get("initialCount", len(reports))),
        initial_grid_count=len(active_grids) or None,
        initial_intensity_average=round(sum(intensities) / len(intensities), 1) if intensities else None,
        initial_intensity_maximum=round(max(intensities), 1) if intensities else None,
        weather=event.get("weather", {}),
        field_candidates=candidates,
        **field_details,
    )


def generate_documents(event: dict, field_context: dict | None = None, update_progress=None) -> tuple[object, str, str | None]:
    update = update_progress or (lambda _value, _message: None)
    update(10, "Event 예측정보를 정리하고 있습니다.")
    forecast = current_forecast(event, field_context)
    update(35, "행정문서 기본 양식을 작성하고 있습니다.")
    safe = build_response_package(forecast)
    if not llm_configured():
        update(100, "기본 양식 문서 생성이 완료되었습니다.")
        return safe, "기본 양식", None
    try:
        update(60, f"{provider_name()}가 문안을 보정하고 있습니다.")
        refined = refine_with_llm(forecast, safe)
        if refined is safe:
            update(100, "검증된 기본 양식을 사용했습니다.")
            return safe, "기본 양식", "LLM 보정 결과를 검증하지 못해 기본 양식을 사용했습니다"
        update(100, "대응 문서 생성이 완료되었습니다.")
        return refined, provider_name(), None
    except Exception:
        logging.getLogger(__name__).exception("LLM 문서 보정 호출 실패")
        update(100, "LLM 대신 기본 양식으로 생성을 완료했습니다.")
        return safe, "기본 양식", "LLM 호출 실패로 기본 양식을 사용했습니다"


def store_generated_documents(event: dict, field_context: dict | None = None) -> None:
    progress = st.progress(0, text="대응 문서 생성을 준비하고 있습니다.")
    package, mode, warning = generate_documents(
        event, field_context,
        lambda value, message: progress.progress(value, text=f"{value}% · {message}"),
    )
    st.session_state.documents = package
    st.session_state.document_event = event["id"]
    st.session_state.document_mode = mode
    st.session_state.document_warning = warning


def apply_styles() -> None:
    st.markdown("""
    <style>
    @import url('https://hangeul.pstatic.net/hangeul_static/css/nanum-square-neo.css');
    :root{--app-font:'NanumSquareNeoVariable','NanumSquareNeo','Malgun Gothic',sans-serif;--app-font-bold:'NanumSquareNeoExtraBold','NanumSquareNeoVariable','Malgun Gothic',sans-serif}
    html,body,[data-testid="stAppViewContainer"],[data-testid="stSidebar"],button,input,textarea,select,label,p,span,div{font-family:var(--app-font)!important;letter-spacing:-.018em}
    .material-icons,.material-symbols-rounded,[class*="material-icons"],[class*="material-symbols"],[data-testid="stIconMaterial"],[data-testid="stIconMaterial"] *{font-family:'Material Symbols Rounded','Material Icons'!important;letter-spacing:normal!important}
    h1,h2,h3,h4,h5,h6,b,strong,.event-id,.weather-value,.farm-title,.step b{font-family:var(--app-font-bold)!important;letter-spacing:-.035em}
    button{font-family:'NanumSquareNeoBold','NanumSquareNeoVariable','Malgun Gothic',sans-serif!important}
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
    .agent-workflow{display:flex;align-items:center;gap:.55rem;flex-wrap:wrap;background:#edf5f3;border:1px solid #c7ded8;border-radius:9px;padding:.7rem .85rem;color:#28574f;margin:.2rem 0 .55rem;font-size:.84rem}.agent-workflow b{color:#173e49;margin-right:.25rem}.agent-workflow span{font-weight:700}.agent-workflow i{font-style:normal;color:#8aa59e}.agent-summary{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:.55rem;margin:0 0 1rem}.agent-summary span{border:1px solid #e0e7ec;background:#fafcfc;border-radius:8px;padding:.6rem .7rem;min-width:0}.agent-summary small,.agent-summary b{display:block}.agent-summary small{font-size:.69rem;color:#68757c;margin-bottom:.18rem}.agent-summary b{font-size:.85rem;color:#243a43;overflow-wrap:anywhere}
    .operation-hero{display:flex;justify-content:space-between;align-items:center;gap:1rem;background:linear-gradient(120deg,#173b47,#245f59);color:white;padding:1.1rem 1.25rem;border-radius:8px 8px 0 0}.operation-hero h2{font-size:1.35rem!important;margin:.35rem 0 .2rem!important;color:white}.operation-hero p{font-size:.82rem;margin:0;color:#d7e7e4}.status-badge{display:inline-block;background:#d9a441;color:#172e3d;font-size:.65rem;font-weight:850;padding:.22rem .48rem;border-radius:999px}.hero-event{min-width:165px;border-left:1px solid #ffffff44;padding-left:1rem}.hero-event small,.hero-event span{display:block;color:#c8dcd8;font-size:.67rem}.hero-event b{display:block;font-size:1.05rem;margin:.15rem 0}
    .operation-steps{display:grid;grid-template-columns:repeat(4,1fr);background:white;border:1px solid #dce2e5;border-top:0;margin-bottom:1rem}.step{position:relative;padding:.7rem .65rem .7rem 2.5rem;border-right:1px solid #e6ebed}.step:last-child{border-right:0}.step>span{position:absolute;left:.7rem;top:.75rem;width:1.35rem;height:1.35rem;border-radius:50%;background:#dce2e5;color:#68757c;text-align:center;line-height:1.35rem;font-size:.68rem;font-weight:800}.step b,.step small{display:block}.step b{font-size:.76rem;color:#304047}.step small{font-size:.63rem;color:#7c898f;margin-top:.12rem}.step.done>span{background:#15866f;color:white}.step.active{background:#fff8e8}.step.active>span{background:#e9a11b;color:#172e3d}
    .field-flow{display:flex;gap:.4rem;align-items:center;flex-wrap:wrap;background:#eef7f4;border:1px solid #c8e2da;padding:.65rem .75rem;margin:.4rem 0 .7rem;font-size:.78rem;color:#24584b}
    .field-flow span{color:#8ba69e}.farm-card{border:1px solid #dce2e5;border-left:4px solid #15866f;background:white;padding:.65rem .75rem;margin:.45rem 0}.farm-card.first{border-left-color:#dd3e36;background:#fffafa}
    .field-summary{display:grid;grid-template-columns:repeat(2,1fr);gap:.4rem;margin:.45rem 0}.field-summary div{border:1px solid #dce2e5;background:#f8faf9;padding:.45rem;text-align:center}.field-summary small{display:block;color:#68757c;font-size:.65rem}.field-summary b{display:block;color:#182126;font-size:.82rem;margin-top:.1rem}
    .candidate-position{text-align:center;line-height:2.35rem;color:#68757c;font-size:.82rem}.candidate-position b{color:#167b64;font-size:1rem}.farm-title{display:flex;justify-content:space-between;gap:.5rem;color:#182126}.farm-title strong{color:#15866f;font-size:1rem}.farm-title strong small{font-size:.6rem;color:#7c898f}.score-bar{height:5px;background:#e9eeee;margin:.4rem 0 .55rem;border-radius:5px;overflow:hidden}.score-bar i{display:block;height:100%;background:linear-gradient(90deg,#2c8f77,#e9a11b)}.farm-address{display:flex;gap:.45rem;align-items:flex-start;color:#46575e;font-size:.72rem;line-height:1.35;margin:.3rem 0 .55rem}.farm-address small{color:#68757c;font-weight:700;flex:0 0 auto}.farm-address span{overflow-wrap:anywhere}.farm-meta{margin:.2rem 0}.distance-badge{display:inline-block;background:#eef4f3;color:#315e55;border-radius:999px;padding:.18rem .5rem;font-size:.7rem;font-weight:700}.farm-summary{font-size:.8rem;line-height:1.5;color:#41555e;margin:.55rem 0 0}
    .weather-cards{display:grid;grid-template-columns:1fr 1fr;gap:.55rem;margin:.25rem 0 .35rem}
    .weather-card{display:flex;align-items:center;gap:.55rem;min-width:0;background:#fff;border:1px solid #dce2e5;padding:.65rem .6rem}
    .weather-icon{font-size:1.15rem;line-height:1;flex:0 0 auto}.weather-copy{min-width:0}
    .weather-label{font-size:.68rem;color:#68757c;margin-bottom:.15rem}.weather-value{font-size:.92rem;font-weight:750;color:#182126;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
    div[data-testid="stMetric"]{background:white;border:1px solid #dce2e5;padding:.55rem}.stButton button{border-radius:4px;font-weight:700}
    iframe[title="streamlit_folium.st_folium"]{border:1px solid #dce2e5}
    .st-key-agent_panel h1{font-size:1.7rem;line-height:1.25;margin:.8rem 0 .55rem}
    .st-key-agent_panel h2{font-size:1.25rem;line-height:1.3;margin:1rem 0 .45rem}
    .st-key-agent_panel h3{font-size:1.05rem;line-height:1.35;margin:.8rem 0 .4rem}
    .st-key-agent_panel p,.st-key-agent_panel li,.st-key-agent_panel table{font-size:.88rem;line-height:1.55}
    @media(max-width:760px){
      .block-container{padding-top:4.3rem;padding-left:.6rem;padding-right:.6rem}.weather-value{font-size:.84rem}
      .operation-hero{display:block}.hero-event{border-left:0;border-top:1px solid #ffffff44;padding:.6rem 0 0;margin-top:.7rem}.operation-steps{grid-template-columns:1fr 1fr}.step:nth-child(2){border-right:0}.step:nth-child(-n+2){border-bottom:1px solid #e6ebed}
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"]{background:#f3f6f8;color:#223440}
    [data-testid="stSidebar"]{background:#fff}
    .block-container{max-width:1800px;padding-left:1.6rem;padding-right:1.6rem}
    .operation-hero{border-radius:14px;padding:1.6rem 1.7rem;margin-bottom:1rem;background:#173e49}
    .operation-hero h2{font-size:1.7rem!important;letter-spacing:-.04em}
    .operation-hero p{font-size:.95rem;line-height:1.7;font-family:'NanumSquareNeoLight','NanumSquareNeoVariable','Malgun Gothic',sans-serif!important;letter-spacing:-.015em}
    .operation-steps{border-radius:12px;margin-top:1rem;overflow:hidden}
    .step{padding:1rem 1rem 1rem 2.8rem}.step b{font-size:.9rem}.step small{font-size:.78rem}
    .step>span{top:1.1rem}
    .st-key-map_panel,.st-key-field_panel,.st-key-agent_panel{position:static;height:auto;overflow:visible;background:white;border:1px solid #e0e7ec;border-radius:14px;padding:1.1rem;min-width:0}
    .farm-card{border-radius:10px;margin:.8rem 0;padding:1rem;border-left:4px solid #829aa3}
    .farm-card.first{background:#edf8f5;border-color:#b6dbd0;border-left-color:#167b64}
    .farm-title b{font-size:1.05rem}.farm-title strong{font-size:1.35rem}.farm-title strong small{font-size:.8rem}
    .merged-note{font-size:.8rem;color:#694f16;background:#fff8e8;border-radius:7px;padding:.5rem .65rem;margin:.5rem 0}.merged-note b,.merged-note span{display:block}.merged-note ul{margin:.25rem 0 0;padding-left:1.15rem}.merged-note span{font-weight:700}
    .field-summary small{font-size:.8rem}.field-summary b{font-size:1rem}.field-summary div{border-radius:8px;padding:.6rem}
    .legend-row{display:flex;gap:18px;flex-wrap:wrap;font-size:.85rem;color:#516874;padding:.6rem 0}
    .legend-row b{margin-right:5px}
    /* Streamlit의 지도·후보 2열은 노트북 폭부터 세로로 전환한다. */
    @media(max-width:1180px){
      div[data-testid="stHorizontalBlock"]:has(.st-key-map_panel){flex-wrap:wrap;gap:1rem!important}
      div[data-testid="stHorizontalBlock"]:has(.st-key-map_panel)>div[data-testid="stColumn"]{flex:1 1 100%!important;width:100%!important;min-width:0!important}
      iframe[title="streamlit_folium.st_folium"]{height:560px!important}
    }
    @media(max-width:900px){
      .block-container{padding-left:.8rem;padding-right:.8rem;padding-top:4.2rem}
      .operation-hero{padding:1.3rem;align-items:flex-start}.operation-hero h2{font-size:1.35rem!important}
      .operation-steps{grid-template-columns:repeat(2,minmax(0,1fr))}
      .step:nth-child(2){border-right:0}.step:nth-child(-n+2){border-bottom:1px solid #e6ebed}
      .st-key-map_panel,.st-key-field_panel,.st-key-agent_panel{padding:.9rem;border-radius:12px}
      iframe[title="streamlit_folium.st_folium"]{height:500px!important}
    }
    @media(max-width:600px){
      [data-testid="stHeader"]:before{left:.8rem;font-size:.9rem}
      .block-container{padding-left:.5rem;padding-right:.5rem}
      .operation-hero{display:block;padding:1.1rem;border-radius:11px}.operation-hero h2{font-size:1.2rem!important;line-height:1.35}
      .hero-event{min-width:0;border-left:0;border-top:1px solid #ffffff44;padding:.8rem 0 0;margin-top:.85rem}
      .operation-steps{grid-template-columns:1fr}.step{border-right:0!important;border-bottom:1px solid #e6ebed!important}.step:last-child{border-bottom:0!important}
      .weather-cards{grid-template-columns:1fr 1fr;gap:.4rem}.weather-card{padding:.55rem .45rem}
      .agent-summary{grid-template-columns:1fr}.agent-workflow{align-items:flex-start}
      .field-summary{grid-template-columns:1fr 1fr}.farm-title{align-items:flex-start}.farm-title b{font-size:.95rem}.farm-title strong{font-size:1.15rem;white-space:nowrap}
      .field-flow{align-items:flex-start;line-height:1.55}.legend-row{display:grid;grid-template-columns:1fr;gap:.4rem}
      .st-key-map_panel,.st-key-field_panel,.st-key-agent_panel{padding:.7rem;border-radius:10px}
      iframe[title="streamlit_folium.st_folium"]{height:420px!important}
      div[data-testid="stTabs"] button{font-size:.74rem!important;padding-left:.45rem!important;padding-right:.45rem!important}
    }
    @media(max-width:390px){
      .weather-cards,.field-summary{grid-template-columns:1fr}
      iframe[title="streamlit_folium.st_folium"]{height:360px!important}
    }
    </style>
    """, unsafe_allow_html=True)


_load_secrets()
apply_styles()
replays = load_field_replays(replay_revision())
# Event 선정·권역 예측·농장 후보가 모두 가축 관련 악취 민원으로 계산된 사건만 보여준다.
demo_replays = [
    row for row in replays
    if len(row["event"].get("field", {}).get("cards", [])) >= MAX_FIELD_CANDIDATES
] or replays
events = [
    {"id": row["event"]["event_id"], "hour": row["event"]["event_hour"]}
    for row in sorted(demo_replays, key=lambda row: row["event_time"])
] if demo_replays else []
if not events:
    st.error("표시할 Event 데이터가 없습니다.")
    st.stop()

if "event_index" not in st.session_state:
    st.session_state.event_index = len(events) - 1
st.session_state.event_index = min(max(0, st.session_state.event_index), len(events) - 1)
if "documents" not in st.session_state:
    st.session_state.documents = None
if "document_event" not in st.session_state:
    st.session_state.document_event = None
if "completed_followup_report" not in st.session_state:
    st.session_state.completed_followup_report = None
if "completed_followup_event" not in st.session_state:
    st.session_state.completed_followup_event = None
if "candidate_index" not in st.session_state:
    st.session_state.candidate_index = 0
if st.session_state.get("document_schema_version") != DOCUMENT_SCHEMA_VERSION:
    st.session_state.documents = None
    st.session_state.document_event = None
    st.session_state.document_schema_version = DOCUMENT_SCHEMA_VERSION

selected_event = events[st.session_state.event_index]
if st.session_state.get("completed_followup_event") != selected_event["id"]:
    st.session_state.completed_followup_report = None
    st.session_state.completed_followup_event = None
field_context = field_context_for_event(selected_event)
event = unified_event_view(selected_event, field_context)
grids = event.get("broad", [])
weather = event.get("weather", {})

with st.sidebar:
    st.markdown('<div class="eyebrow">선택한 과거 상황</div>', unsafe_allow_html=True)
    selected_index = st.selectbox(
        "사건 날짜 선택", range(len(events)), index=st.session_state.event_index,
        format_func=lambda i: f'{datetime.fromisoformat(events[i]["hour"]):%Y-%m-%d %H:%M} · {events[i]["id"]}',
    )
    if selected_index != st.session_state.event_index:
        st.session_state.event_index = selected_index
        st.session_state.documents = None
        st.session_state.candidate_index = 0
        st.session_state.show_actual = False
        st.rerun()
    left, middle, right = st.columns([4, 1, 1])
    with left:
        st.markdown(f'<div class="event-id">{event["id"]}</div><div class="event-time">{event["hour"]} 기준</div>', unsafe_allow_html=True)
    with middle:
        if st.button("‹", disabled=st.session_state.event_index == 0, use_container_width=True, key="prev"):
            st.session_state.event_index -= 1
            st.session_state.documents = None
            st.session_state.candidate_index = 0
            st.session_state.show_actual = False
            st.rerun()
    with right:
        if st.button("›", disabled=st.session_state.event_index == len(events) - 1, use_container_width=True, key="next"):
            st.session_state.event_index += 1
            st.session_state.documents = None
            st.session_state.candidate_index = 0
            st.session_state.show_actual = False
            st.rerun()

    show_actual = st.toggle("예측 결과와 실제 이후 신고 비교", value=False, key="show_actual")
    m1, m2 = st.columns(2)
    m1.metric("초기 신고", event.get("initialCount", 0))
    m2.metric("이후 민원 예측 권역", min(3, len(grids)))
    if show_actual:
        actual_hits = sum(bool(grid.get("actual")) for grid in grids[:3])
        st.metric(
            "실제 이후 신고가 포함된 예측 권역",
            f'{actual_hits}/3',
            help="예측 Top 3 중 이후 30분에 실제 신고가 접수된 1km 권역 수입니다.",
        )
        if actual_hits:
            st.success(f"Top 3 예측 권역 중 {actual_hits}곳에 실제 이후 신고가 포함되었습니다. 지도에서 초록색 테두리로 표시합니다.")

    st.markdown('<div class="eyebrow">현장 기상</div>', unsafe_allow_html=True)
    speed_value = weather.get("windSpeed")
    direction_value = weather.get("windDirection")
    wind_speed = "-" if speed_value is None else f'{speed_value} m/s'
    if speed_value is None or direction_value is None:
        wind_direction = "-"
    elif float(speed_value) < 0.5:
        wind_direction = "판단 어려움"
    elif float(direction_value) == 0:
        wind_direction = "0° (북풍)"
    else:
        wind_direction = f'{direction_value}°'
    optional_weather = ""
    for key, label, icon, unit in (("humidity", "상대습도", "💧", "%"), ("rainfall", "최근 1시간 강수", "🌧️", "mm")):
        if weather.get(key) is not None:
            optional_weather += (f'<div class="weather-card"><span class="weather-icon">{icon}</span>'
                                 f'<div class="weather-copy"><div class="weather-label">{label}</div>'
                                 f'<div class="weather-value">{escape(str(weather[key]))} {unit}</div></div></div>')
    st.markdown(f'''<div class="weather-cards">
      <div class="weather-card"><span class="weather-icon">💨</span><div class="weather-copy"><div class="weather-label">풍속</div><div class="weather-value">{wind_speed}</div></div></div>
      <div class="weather-card"><span class="weather-icon">🧭</span><div class="weather-copy"><div class="weather-label">풍향</div><div class="weather-value">{wind_direction}</div></div></div>
      {optional_weather}
    </div>''', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">행정 대응 Agent</div>', unsafe_allow_html=True)

    if st.button("대응 문서 생성", type="primary", use_container_width=True):
        store_generated_documents(event, field_context)

render_operation_header(event, field_context)

map_column, agent_column = st.columns([1.35, 1], gap="large")

with map_column.container(key="map_panel"):
    st.markdown("#### 상황 지도")
    st_folium(event_map(event, show_actual, field_context), use_container_width=True, height=650, returned_objects=[])
    legend = '<span><b style="color:#dd3e36">■</b>추가 민원 예상 권역</span>'
    actual_legend = '<span><b style="color:#15866f">▣</b>실제 이후 신고 포함</span>' if show_actual else ""
    st.markdown(f'<div class="legend-row">{legend}{actual_legend}<span><b style="color:#15866f">●</b>방문 후보 1–3</span></div>', unsafe_allow_html=True)

with agent_column.container(key="field_panel"):
    render_field_candidates(field_context)

with st.container(key="agent_panel"):
    st.markdown("#### 행정 대응 Agent")
    field = field_context["event"].get("field", {}) if field_context else {}
    lead_card = next(iter(field.get("cards", [])), {})
    lead_farm = escape(str(lead_card.get("name") or lead_card.get("farm_id") or "후보 정보 없음"))
    lead_region = escape(str((grids[0] if grids else {}).get("region") or "1순위 권역"))
    wind_confidence = escape(str(field.get("confidence") or "정보 없음"))
    st.markdown(
        '<div class="agent-workflow"><b>지금 할 일</b>'
        f'<span>1. 이후 민원 예측 권역: {lead_region}</span><i>→</i>'
        f'<span>2. 우선 점검 농가 후보: {lead_farm}</span><i>→</i>'
        '<span>3. 점검 지시서 생성 · 현장 결과 기록</span></div>'
        '<div class="agent-summary">'
        f'<span><small>이후 민원 예측 권역</small><b>{lead_region}</b></span>'
        f'<span><small>우선 점검 농가 후보</small><b>{lead_farm}</b></span>'
        f'<span><small>풍향 신뢰도</small><b>{wind_confidence}</b></span>'
        '</div>',
        unsafe_allow_html=True,
    )
    if st.session_state.documents is None or st.session_state.document_event != event["id"]:
        st.markdown(
            '<div class="notice"><b>예측 결과를 현장 대응 문서로 변환합니다.</b><br>'
            '상황 브리핑과 현장점검 지시서를 생성하고, 점검 후 결과를 기록할 수 있습니다.</div>',
            unsafe_allow_html=True,
        )
        if st.button("이 Event의 대응 문서 생성", type="primary", use_container_width=True, key="generate_main"):
            store_generated_documents(event, field_context)
            st.rerun()
        st.caption("문서를 생성하면 이 영역에서 바로 확인하고 현장 결과를 입력할 수 있습니다.")
    else:
        package = st.session_state.documents
        warning = st.session_state.get("document_warning")
        message = warning or f'{st.session_state.get("document_mode", "기본 양식")} 문서 생성 완료 · 담당자 검토 필요'
        st.markdown(f'<div class="notice">{escape(str(message))}</div>', unsafe_allow_html=True)
        tab1, tab2, tab3 = st.tabs(["① 상황 확인", "② 출동 지시", "③ 현장 결과 기록"])
        with tab1:
            st.markdown(package.briefing)
            st.download_button("브리핑 다운로드", package.briefing, f"briefing_{event['id']}.md", "text/markdown")
        with tab2:
            st.markdown(package.dispatch_order)
            st.download_button("점검 지시서 다운로드", package.dispatch_order, f"inspection_{event['id']}.md", "text/markdown")
        with tab3:
            st.caption("현장 점검을 마친 뒤에만 아래 입력 영역을 열어 결과를 기록하세요.")
            with st.expander("현장 점검 완료 후 결과 입력", expanded=False):
                with st.form("followup_form"):
                    author = st.text_input("작성자", placeholder="예: 홍길동 주무관")
                    date_col, time_col = st.columns(2)
                    current_time = now_kst()
                    inspected_date = date_col.date_input("점검 날짜", value=current_time.date())
                    inspected_time = time_col.time_input("점검 시각", value=current_time.time().replace(second=0, microsecond=0))
                    time1, time2, time3 = st.columns(3)
                    dispatch_decided_at = time1.text_input("출동 결정시각", placeholder="예: 20:32")
                    departed_at = time2.text_input("현장 출발시각", placeholder="예: 20:35")
                    arrived_at = time3.text_input("현장 도착시각", placeholder="예: 20:48")
                    distance_col, count_col = st.columns(2)
                    total_distance_km = distance_col.text_input("총 출동거리(km)", placeholder="예: 5.2")
                    actual_additional_area_count = count_col.text_input("실제 추가 민원 권역 수", placeholder="예: 2")
                    farms = []
                    for candidate in package.forecast.field_candidates:
                        st.markdown(f"**{candidate.rank}순위 · {candidate.display_name}**")
                        c1, c2 = st.columns(2)
                        detected = c1.selectbox("악취 감지", ["미확인", "감지", "미감지"], key=f"farm_odor_{event['id']}_{candidate.rank}")
                        wind_checked = c2.selectbox("현장 풍향 일치", ["미확인", "일치", "불일치"], key=f"farm_wind_{event['id']}_{candidate.rank}")
                        measurement = st.text_input("측정 결과", key=f"farm_measurement_{event['id']}_{candidate.rank}")
                        action = st.text_input("조치 내용", key=f"farm_action_{event['id']}_{candidate.rank}")
                        farms.append({"rank": candidate.rank, "name": candidate.display_name, "odor_detected": detected, "wind_checked": wind_checked, "measurement": measurement, "action": action})
                    field_findings = st.text_area("현장 확인내용", placeholder="현장에서 확인한 악취 상태와 주변 상황을 입력하세요.")
                    notes = st.text_area("담당자 의견 및 종합 결과", placeholder="실시 조치와 추가 확인 필요사항을 입력하세요.")
                    followup_required = st.radio("추가 조치 필요 여부", ["미확인", "필요", "불필요"], horizontal=True)
                    submitted = st.form_submit_button("입력값으로 결과보고서 완성", type="primary", use_container_width=True)
                if submitted:
                    inspected_at = datetime.combine(inspected_date, inspected_time, tzinfo=KST)
                    report = create_completed_followup(package, {
                        "author": author, "inspected_at": inspected_at.isoformat(timespec="minutes"),
                        "dispatch_decided_at": dispatch_decided_at, "departed_at": departed_at, "arrived_at": arrived_at,
                        "total_distance_km": total_distance_km, "actual_additional_area_count": actual_additional_area_count,
                        "field_findings": field_findings,
                        "followup_required": followup_required, "farms": farms, "notes": notes,
                    })
                    st.markdown(report)
                    st.session_state.completed_followup_report = report
                    st.session_state.completed_followup_event = event["id"]
                    st.download_button("완성 보고서 다운로드", report, f"followup_{event['id']}.md", "text/markdown")
                elif st.session_state.get("completed_followup_event") == event["id"]:
                    saved_report = st.session_state.get("completed_followup_report")
                    if saved_report:
                        st.success("현장 결과보고서를 저장했습니다. 아래에서 다시 확인하거나 다운로드할 수 있습니다.")
                        st.markdown(saved_report)
                        st.download_button("완성 보고서 다운로드", saved_report, f"followup_{event['id']}.md", "text/markdown", key="saved_followup_download")
