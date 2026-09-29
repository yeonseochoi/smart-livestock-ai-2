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
import re

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from administrative_agent.documents import create_response_guide
from administrative_agent.models import FieldCandidate, ForecastResult, RiskArea
from administrative_agent.time_utils import KST, now_kst
from administrative_agent.llm import llm_configured, provider_name, refine_with_llm
from administrative_agent.service import build_response_package, create_completed_followup
from generate_agent_documents import DEFAULT_METRICS, DEFAULT_PREDICTIONS, forecast_from_csv, _relative_scores
from odor_service.region_prediction.data import assign_grid


ROOT = Path(__file__).resolve().parent
DEMO_DATA = ROOT / "demo" / "demo-data.js"
REPLAY_DIR = ROOT / "outputs" / "odor_service" / "replay"
FARMS_FILE = ROOT / "outputs" / "odor_service" / "data" / "farms.parquet"
FONT_CSS_URL = "https://hangeul.pstatic.net/hangeul_static/css/nanum-square-neo.css"
DOCUMENT_SCHEMA_VERSION = 4

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
        for card in field_context["event"].get("field", {}).get("cards", []):
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
        base = "#dd3e36"
        actual = show_actual and bool(grid.get("actual"))
        folium.Rectangle(
            bounds=[[lat - dy, lon - dx], [lat + dy, lon + dx]],
            color="#15866f" if actual else base, weight=3 if actual else 1,
            fill=True, fill_color=base, fill_opacity=0.58,
            tooltip=f"{idx}순위 1km 권역 · 현장 점검 대상" + (" · 실제 이후 신고" if actual else ""),
        ).add_to(fmap)

    if field_context:
        for card in field_context["event"].get("field", {}).get("cards", []):
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
    field = field_context["event"].get("field", {}) if field_context else {}
    cards = field.get("cards", [])
    confidence = escape(str(field.get("confidence") or "연결 정보 없음"))
    st.markdown(
        '<div class="operation-hero">'
        '<div><span class="status-badge">과거 상황 재현</span>'
        '<h2>민원 확산 예측에서 현장 확인까지</h2>'
        '<p>예상 권역을 보여주는 데서 끝나지 않고, 담당자가 먼저 확인할 농가 순서까지 연결합니다.</p></div>'
        f'<div class="hero-event"><small>현재 Event</small><b>{escape(str(event["id"]))}</b>'
        f'<span>{escape(str(event["hour"]))}</span></div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.caption("과거 관측 날씨로 재현한 시연입니다. 실제 예보 연동·현장 배치·시료 채취 결과를 의미하지 않습니다.")
    st.markdown(
        '<div class="operation-steps">'
        '<div class="step done"><span>1</span><b>민원 집중 감지</b><small>최근 30분 신고</small></div>'
        '<div class="step done"><span>2</span><b>예상 권역 Top 3</b><small>추가 민원 가능 권역</small></div>'
        f'<div class="step active"><span>3</span><b>농가 후보 {len(cards)}곳</b><small>민원 위치·풍향 교차 근거</small></div>'
        f'<div class="step"><span>4</span><b>경계 시료 채취</b><small>현장 신뢰도 {confidence}</small></div>'
        '</div>',
        unsafe_allow_html=True,
    )


def render_field_candidates(field_context: dict | None) -> None:
    st.markdown("#### 먼저 확인할 농가")
    st.caption("가축 분뇨 냄새 민원 위치마다 바람이 불어오는 쪽 4 km 안의 농가를 찾고, 여러 민원 위치가 함께 가리키는 농가를 우선했습니다. "
               "대기 장소에서의 이동 거리는 순위에 넣지 않고 방문 동선 제안에만 씁니다.")
    if field_context is None:
        st.info("동일 사건 ID의 현장 후보 재현 자료가 없습니다.")
        return

    field_event = field_context["event"]
    standby = field_context.get("standby", {})
    chosen = next((p for p in standby.get("points", []) if p.get("point_id") == standby.get("chosen_point_id")), None)
    if chosen:
        st.caption(f'추천 대기 장소: {chosen.get("name", chosen["point_id"])} · 실제 담당자 GPS가 아닌 방문 동선 계산 출발점')
    field = field_event.get("field", {})
    cards = field.get("cards", [])
    wind_speed = field_event.get("direction_overlap", {}).get("wind_speed")
    wind_text = "-" if wind_speed is None else f"{wind_speed} m/s"
    confidence = escape(str(field.get("confidence") or "정보 없음"))
    stability = field.get("stability_label")
    stability_text = "-" if not stability else f'{stability} ({float(field.get("stability", 0)):.0%} 유지)'
    st.markdown(
        '<div class="field-summary">'
        f'<div><small>풍속</small><b>{escape(wind_text)}</b></div>'
        f'<div><small>풍향 신뢰도</small><b>{confidence}</b></div>'
        f'<div><small>풍향 ±20° 후보 안정성</small><b>{escape(stability_text)}</b></div>'
        '</div>',
        unsafe_allow_html=True,
    )
    if field.get("locations"):
        st.caption(f'가축 분뇨 냄새 민원의 고유 위치 {field["locations"]}곳(30m 이내 반복 신고는 1곳) · 민원 무리 {field.get("clusters", 1)}개 기준입니다. 권역 Top 3는 전체 신고로 계산합니다.')
    if field.get("selection") == "단일 지점 참고":
        st.warning("2곳 이상의 민원 위치가 함께 가리키는 농가가 없습니다. 아래 후보는 한 민원 위치 기준 참고 후보입니다.")
    st.caption(f'현재 민원 집중 사건 {field_event.get("event_id", "")}의 흐름을 이어서 표시합니다.')
    st.info("익산시 정상 영업·정확 좌표 농가를 우선합니다. 익산 후보가 없을 때만 김제 용지 방면의 근사 좌표 후보를 인접 지역 참고로 표시합니다.")
    if field.get("confidence") in ("낮음", "매우 낮음"):
        st.warning("바람이 약합니다. 방향 근거의 신뢰도가 낮으니 현장에서 바람과 냄새를 먼저 확인하세요.")
    if not cards:
        st.info("현재 조건에 맞는 방문 후보가 없습니다. 가축 분뇨 냄새 민원과 사건 시각 기상 자료를 확인하세요. 구체적인 사유는 아래 안내를 참고하세요.")

    for card in cards[:5]:
        components = card.get("components", {})
        name = escape(str(card.get("name") or card.get("farm_id") or "이름 없음"))
        summary = escape(str(card.get("selection_summary") or card.get("reason") or "선정 이유 정보 없음"))
        evidence = [
            item for item in (card.get("evidence") or [])
            if not re.search(r"중\s*0곳|0곳의\s*풍상", str(item))
        ]
        evidence_html = "".join(f"<li>{escape(str(item))}</li>" for item in evidence)
        merged_count = int(card.get("merged_count") or 1)
        merged_names = [escape(item.strip()) for item in str(card.get("merged_names") or card.get("name") or "").split("|") if item.strip()]
        addresses = [escape(item.strip()) for item in str(card.get("merged_addresses") or card.get("address") or "주소 정보 없음").split("|") if item.strip()]
        address_text = "<br>".join(dict.fromkeys(addresses)) or "주소 정보 없음"
        if merged_count > 1:
            names_html = "".join(f"<li>{item}</li>" for item in merged_names)
            repeated = f'<span>동일 명칭으로 등록된 시설 {merged_count}건</span>' if len(merged_names) == 1 else ""
            merged_note = (
                f'<div class="merged-note"><b>함께 확인할 등록 농장</b><ul>{names_html}</ul>{repeated}'
                f'<small>같은 좌표 또는 300m 이내 동명 등록 {merged_count}건을 한 방문 지점으로 묶었습니다.</small></div>'
            )
            title = "공동 방문 지점"
        else:
            merged_note = ""
            title = name
        rank_class = " first" if card.get("rank") == 1 else ""
        support = card.get("support") or {}
        tier_html = ""
        if card.get("tier"):
            tier_class = " single" if card["tier"] == "단일 지점 참고" else ""
            tier_html = (f'<span class="tier-badge{tier_class}">{escape(str(card["tier"]))} '
                         f'{support.get("count", "-")}/{support.get("total", "-")}</span>')
        distance_parts = []
        if card.get("complaint_km") is not None:
            distance_parts.append(f'가장 가까운 민원 위치 {float(card["complaint_km"]):.1f} km')
        if card.get("travel_km") is not None and card.get("visit_order") is not None:
            distance_parts.append(f'대기 장소에서 직선 {float(card["travel_km"]):.1f} km · 방문 동선 {int(card["visit_order"])}번째')
        distance_html = f'<div class="farm-distance">{escape(" · ".join(distance_parts))}</div>' if distance_parts else ""
        score = float(card.get("score") or 0)
        st.markdown(
            f'<div class="farm-card{rank_class}">'
            f'<div class="farm-title"><b>{card.get("rank")}순위 · {title} {tier_html}</b>'
            f'<strong>{score:.1f}<small>/100</small></strong></div>'
            f'<div class="score-bar"><i style="width:{max(0, min(score, 100)):.1f}%"></i></div>'
            f'<div class="farm-address"><small>주소</small><span>{address_text}</span></div>'
            f'<div class="score-grid">'
            f'<span><small>풍향</small><b>{components.get("풍향 일치", 0)}</b><em>/40</em></span>'
            f'<span><small>민원 근접</small><b>{components.get("거리", 0)}</b><em>/25</em></span>'
            f'<span><small>민원 중첩</small><b>{components.get("다중 측정 일치", 0)}</b><em>/20</em></span>'
            f'<span><small>과거 반복</small><b>{components.get("과거 반복", 0)}</b><em>/15</em></span>'
            f'</div>'
            f'{distance_html}'
            f'{merged_note}'
            f'<div class="farm-reason"><b>왜 {card.get("rank")}순위인가요?</b><p>{summary}</p>'
            f'<ul>{evidence_html}</ul></div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if card.get("coord_warning"):
            st.caption(f'{card.get("rank")}순위 · 대략적인 좌표입니다. 방문 전 위치를 확인하세요.')
    route = sorted((c for c in cards[:5] if c.get("visit_order") is not None), key=lambda c: int(c["visit_order"]))
    if route:
        names = " → ".join(escape(str(c.get("name") or c.get("farm_id"))) for c in route)
        total = field.get("route_km")
        st.markdown(
            f'<div class="next-action"><b>방문 동선 제안</b> (대기 장소 출발, 가까운 곳부터)<br>{names}'
            + (f'<br><small>직선 거리 합계 {float(total):.1f} km · 실제 도로 이동시간과 담당자 위치는 반영 전</small>' if total is not None else "")
            + '</div>',
            unsafe_allow_html=True,
        )
    for note in field.get("notes", []):
        st.warning(note)
    st.caption("※ 현장 확인 우선순위이며 발생원 확정이나 위반 판정이 아닙니다. 최종 판단은 농가 경계에서 채취한 시료로 합니다.")


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
    ) for card in cards[:5])
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
        update(100, "안전 템플릿 문서 생성이 완료되었습니다.")
        return safe, "안전 템플릿", None
    try:
        update(60, f"{provider_name()}가 문안을 보정하고 있습니다.")
        refined = refine_with_llm(forecast, safe)
        if refined is safe:
            update(100, "검증된 안전 템플릿을 사용했습니다.")
            return safe, "안전 템플릿", "LLM 보정 결과를 검증하지 못해 안전 템플릿을 사용했습니다"
        update(100, "대응 문서 생성이 완료되었습니다.")
        return refined, provider_name(), None
    except Exception:
        logging.getLogger(__name__).exception("LLM 문서 보정 호출 실패")
        update(100, "LLM 대신 안전 템플릿으로 생성을 완료했습니다.")
        return safe, "안전 템플릿", "LLM 호출 실패로 안전 템플릿을 사용했습니다"


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
    .operation-hero{display:flex;justify-content:space-between;align-items:center;gap:1rem;background:linear-gradient(120deg,#173b47,#245f59);color:white;padding:1.1rem 1.25rem;border-radius:8px 8px 0 0}.operation-hero h2{font-size:1.35rem!important;margin:.35rem 0 .2rem!important;color:white}.operation-hero p{font-size:.82rem;margin:0;color:#d7e7e4}.status-badge{display:inline-block;background:#d9a441;color:#172e3d;font-size:.65rem;font-weight:850;padding:.22rem .48rem;border-radius:999px}.hero-event{min-width:165px;border-left:1px solid #ffffff44;padding-left:1rem}.hero-event small,.hero-event span{display:block;color:#c8dcd8;font-size:.67rem}.hero-event b{display:block;font-size:1.05rem;margin:.15rem 0}
    .operation-steps{display:grid;grid-template-columns:repeat(4,1fr);background:white;border:1px solid #dce2e5;border-top:0;margin-bottom:1rem}.step{position:relative;padding:.7rem .65rem .7rem 2.5rem;border-right:1px solid #e6ebed}.step:last-child{border-right:0}.step>span{position:absolute;left:.7rem;top:.75rem;width:1.35rem;height:1.35rem;border-radius:50%;background:#dce2e5;color:#68757c;text-align:center;line-height:1.35rem;font-size:.68rem;font-weight:800}.step b,.step small{display:block}.step b{font-size:.76rem;color:#304047}.step small{font-size:.63rem;color:#7c898f;margin-top:.12rem}.step.done>span{background:#15866f;color:white}.step.active{background:#fff8e8}.step.active>span{background:#e9a11b;color:#172e3d}
    .field-flow{display:flex;gap:.4rem;align-items:center;flex-wrap:wrap;background:#eef7f4;border:1px solid #c8e2da;padding:.65rem .75rem;margin:.4rem 0 .7rem;font-size:.78rem;color:#24584b}
    .field-flow span{color:#8ba69e}.farm-card{border:1px solid #dce2e5;border-left:4px solid #15866f;background:white;padding:.65rem .75rem;margin:.45rem 0}.farm-card.first{border-left-color:#dd3e36;background:#fffafa}
    .field-summary{display:grid;grid-template-columns:repeat(3,1fr);gap:.4rem;margin:.45rem 0}.field-summary div{border:1px solid #dce2e5;background:#f8faf9;padding:.45rem;text-align:center}.field-summary small{display:block;color:#68757c;font-size:.65rem}.field-summary b{display:block;color:#182126;font-size:.82rem;margin-top:.1rem}
    .farm-title{display:flex;justify-content:space-between;gap:.5rem;color:#182126}.farm-title strong{color:#15866f;font-size:1rem}.farm-title strong small{font-size:.6rem;color:#7c898f}.score-bar{height:5px;background:#e9eeee;margin:.4rem 0 .55rem;border-radius:5px;overflow:hidden}.score-bar i{display:block;height:100%;background:linear-gradient(90deg,#2c8f77,#e9a11b)}.farm-address{display:flex;gap:.45rem;align-items:flex-start;color:#46575e;font-size:.72rem;line-height:1.35;margin:.3rem 0 .55rem}.farm-address small{color:#68757c;font-weight:700;flex:0 0 auto}.farm-address span{overflow-wrap:anywhere}.score-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:.25rem}.score-grid span{background:#f5f8f7;padding:.3rem;text-align:center}.score-grid small{display:block;font-size:.58rem;color:#68757c}.score-grid b{font-size:.75rem;color:#204e43}.score-grid em{font-style:normal;font-size:.55rem;color:#8a969b}.farm-distance{font-size:.69rem;color:#46575e;margin:.45rem 0 .2rem}.farm-reason{font-size:.69rem;color:#68757c;line-height:1.45;border-top:1px dashed #dce2e5;padding-top:.35rem}
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
    .operation-hero{border-radius:14px;padding:1.6rem 1.7rem;background:#173e49}
    .operation-hero h2{font-size:1.7rem!important;letter-spacing:-.04em}
    .operation-hero p{font-size:.95rem;line-height:1.7;font-family:'NanumSquareNeoLight','NanumSquareNeoVariable','Malgun Gothic',sans-serif!important;letter-spacing:-.015em}
    .operation-steps{border-radius:12px;margin-top:1rem;overflow:hidden}
    .step{padding:1rem 1rem 1rem 2.8rem}.step b{font-size:.9rem}.step small{font-size:.78rem}
    .step>span{top:1.1rem}
    .st-key-map_panel,.st-key-agent_panel{position:static;height:auto;overflow:visible;background:white;border:1px solid #e0e7ec;border-radius:14px;padding:1.1rem;min-width:0}
    [data-testid="stExpander"] summary{display:flex!important;align-items:center;gap:.55rem;min-height:2.2rem;padding:.45rem .65rem!important;line-height:1.35!important}
    [data-testid="stExpander"] summary p{margin:0!important;line-height:1.35!important;white-space:normal!important;overflow-wrap:anywhere}
    [data-testid="stExpander"] summary svg{flex:0 0 auto;width:1.1rem;height:1.1rem}
    .farm-card{border-radius:10px;margin:.8rem 0;padding:1rem;border-left:4px solid #829aa3}
    .farm-card.first{background:#edf8f5;border-color:#b6dbd0;border-left-color:#167b64}
    .farm-title b{font-size:1.05rem}.farm-title strong{font-size:1.35rem}.farm-title strong small{font-size:.8rem}
    .score-grid{gap:6px}.score-grid span{border-radius:6px;padding:.5rem .2rem;background:#f2f5f6}
    .score-grid small{font-size:.77rem}.score-grid b{font-size:1rem}.score-grid em{font-size:.72rem}
    .farm-distance,.farm-reason{font-size:.88rem;line-height:1.65}.merged-note{font-size:.8rem;color:#694f16;background:#fff8e8;border-radius:7px;padding:.55rem .65rem;margin:.5rem 0}.merged-note b,.merged-note span,.merged-note small{display:block}.merged-note ul{margin:.3rem 0;padding-left:1.15rem}.merged-note span{font-weight:700}.merged-note small{color:#806d42;margin-top:.25rem}.farm-reason{margin-top:.6rem;padding-top:.65rem}.farm-reason>p{margin:.35rem 0 .45rem;color:#253f48;font-weight:700}.farm-reason ul{margin:.25rem 0 .65rem;padding-left:1.15rem;color:#5b6f7b}.farm-reason li{margin:.14rem 0}.next-action{background:#eef7f4;border-radius:7px;padding:.55rem .65rem;color:#285b4f}
    .field-summary small{font-size:.8rem}.field-summary b{font-size:1rem}.field-summary div{border-radius:8px;padding:.6rem}
    .tier-badge{display:inline-block;margin-left:.35rem;padding:.1rem .45rem;border-radius:999px;background:#e3f3ee;color:#15664f;font-size:.72rem;font-weight:800;vertical-align:middle}
    .tier-badge.single{background:#fff1d6;color:#8a5a00}
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
      .st-key-map_panel,.st-key-agent_panel{padding:.9rem;border-radius:12px}
      iframe[title="streamlit_folium.st_folium"]{height:500px!important}
    }
    @media(max-width:600px){
      [data-testid="stHeader"]:before{left:.8rem;font-size:.9rem}
      .block-container{padding-left:.5rem;padding-right:.5rem}
      .operation-hero{display:block;padding:1.1rem;border-radius:11px}.operation-hero h2{font-size:1.2rem!important;line-height:1.35}
      .hero-event{min-width:0;border-left:0;border-top:1px solid #ffffff44;padding:.8rem 0 0;margin-top:.85rem}
      .operation-steps{grid-template-columns:1fr}.step{border-right:0!important;border-bottom:1px solid #e6ebed!important}.step:last-child{border-bottom:0!important}
      .weather-cards{grid-template-columns:1fr 1fr;gap:.4rem}.weather-card{padding:.55rem .45rem}
      .field-summary{grid-template-columns:1fr 1fr}.field-summary div:last-child{grid-column:1/-1}
      .score-grid{grid-template-columns:1fr 1fr}.farm-title{align-items:flex-start}.farm-title b{font-size:.95rem}.farm-title strong{font-size:1.15rem;white-space:nowrap}
      .field-flow{align-items:flex-start;line-height:1.55}.legend-row{display:grid;grid-template-columns:1fr;gap:.4rem}
      .st-key-map_panel,.st-key-agent_panel{padding:.7rem;border-radius:10px}
      [data-testid="stExpander"] summary{padding:.5rem .45rem!important;font-size:.82rem!important}
      iframe[title="streamlit_folium.st_folium"]{height:420px!important}
      div[data-testid="stTabs"] button{font-size:.74rem!important;padding-left:.45rem!important;padding-right:.45rem!important}
    }
    @media(max-width:390px){
      .weather-cards,.field-summary,.score-grid{grid-template-columns:1fr}
      .field-summary div:last-child{grid-column:auto}
      iframe[title="streamlit_folium.st_folium"]{height:360px!important}
    }
    </style>
    """, unsafe_allow_html=True)


_load_secrets()
apply_styles()
replays = load_field_replays(replay_revision())
# 재현 사건 자체를 선택한다. 가까운 레거시 사건으로 연결하면 일부 사건이
# 누락되거나 여러 선택지가 같은 EV2 사건을 가리킬 수 있다.
events = [
    {"id": row["event"]["event_id"], "hour": row["event"]["event_hour"]}
    for row in sorted(replays, key=lambda row: row["event_time"])
] if replays else load_demo_data().get("events", [])
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
if st.session_state.get("document_schema_version") != DOCUMENT_SCHEMA_VERSION:
    st.session_state.documents = None
    st.session_state.document_event = None
    st.session_state.document_schema_version = DOCUMENT_SCHEMA_VERSION

selected_event = events[st.session_state.event_index]
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
        st.session_state.show_actual = False
        st.rerun()
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

    show_actual = st.toggle("검증용 실제 이후 신고 표시", value=False, key="show_actual")
    m1, m2 = st.columns(2)
    m1.metric("초기 신고", event.get("initialCount", 0))
    m2.metric("점검 권역", min(3, len(grids)))
    if show_actual:
        st.metric("Top 3 중 실제 이후 신고 권역", f'{sum(bool(grid.get("actual")) for grid in grids[:3])}/3')

    st.markdown('<div class="eyebrow">1km Complaint Forecast</div>', unsafe_allow_html=True)
    st.markdown('<div class="risk-box"><b>최우선 점검 권역 안내</b><br>1순위 권역부터 현장 확인을 권고합니다.<br><small>예측 확률이 아닌 권역 간 상대 순위입니다.</small></div>', unsafe_allow_html=True)

    st.markdown('<div class="eyebrow">현장 참고 기상정보</div>', unsafe_allow_html=True)
    wind_speed = "-" if weather.get("windSpeed") is None else f'{weather["windSpeed"]} m/s'
    wind_direction = "-" if weather.get("windDirection") is None else f'{weather["windDirection"]}°'
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
    st.caption("※ 권역 예측 모델에는 미사용, 방문 농가 순위에는 풍향·풍속 사용")

    st.markdown('<div class="eyebrow">Dispatch Priority</div>', unsafe_allow_html=True)
    relative_scores = _relative_scores(pd.Series([grid["score"] for grid in grids[:3]])) if grids else []
    for idx, grid in enumerate(grids[:3], 1):
        cls = "priority first" if idx == 1 else "priority"
        action = "가장 먼저 현장 확인" if idx == 1 else "1순위 확인 후 순차 확인"
        st.markdown(f'<div class="{cls}"><b>{idx}순위 · 1km 권역</b><br><small>{action} · 상대점수 {relative_scores[idx - 1]}/100</small></div>', unsafe_allow_html=True)

    st.markdown('<div class="eyebrow">Administrative Agent</div>', unsafe_allow_html=True)
    if st.button("대응 문서 생성", type="primary", use_container_width=True):
        store_generated_documents(event, field_context)
    st.caption(f'{provider_name()} LLM 연결됨' if llm_configured() else "안전 템플릿 모드 · API 키 미설정")

render_operation_header(event, field_context)

map_column, agent_column = st.columns([1.35, 1], gap="large")

with map_column.container(key="map_panel"):
    st.markdown("#### 과거 Event 재현 모드")
    st.caption("1km 광역 경보 + 현장 확인 후보 · 초록 숫자는 방문 농가 순서입니다.")
    st_folium(event_map(event, show_actual, field_context), use_container_width=True, height=650, returned_objects=[])
    st.markdown('<div class="legend-row"><span><b style="color:#dd3e36">■</b>추가 민원 예상 권역</span><span><b style="color:#15866f">●</b>방문 후보 1–5</span></div>', unsafe_allow_html=True)
    with st.expander("자료와 계산 기준 확인"):
        st.write("이 화면은 하나의 민원 집중 사건을 기준으로, 민원 발생 예측 권역 Top 3 → 대기 장소 → 방문 농가 후보 순서로 연결해 보여줍니다.")
        st.write("농가 점수는 발생원일 확률이 아닙니다. 담당자가 어디부터 확인할지 정하는 참고 순위이며, 풍향 일치 40점·민원 근접 25점·여러 민원 위치 교차 20점·과거 반복 이력 15점으로 계산합니다.")
        st.write("후보는 가축 분뇨 냄새 신고를 30m로 묶은 고유 민원 위치마다 바람이 불어오는 쪽(±45°) 4 km 안의 농가입니다. 공장·하수구·소각·음식·기타 신고는 농가 후보 근거에서 제외하며, 권역 Top 3는 전체 신고로 계산합니다. 2곳 이상의 민원 위치가 함께 가리키는 농가를 우선하고, 그런 농가가 없을 때만 한 위치 기준 후보를 참고로 보여 줍니다. 후보 수를 5곳으로 억지로 채우지 않습니다.")
        st.write("민원이 2 km 이상 떨어진 여러 무리로 나뉘면, 후보가 남은 무리의 대표를 우선합니다. 교차 확인 조건·지점 병합·최대 5곳 제한 때문에 모든 무리의 후보가 포함되지는 않을 수 있습니다. 풍향 ±20° 후보 안정성은 풍향이 조금 달라져도 같은 후보가 유지되는 비율입니다.")
        st.write("대기 장소의 위치·등급은 전체 기간 자료, 대기 방면은 밤 전체 관측 평균 바람으로 계산했습니다. 사건 이후 정보가 포함된 과거 재현이며 실시간 운영 성능이 아닙니다. 사후 확인 등급은 현재 출동 판단에 사용하지 않습니다.")

with agent_column.container(key="agent_panel"):
    render_field_candidates(field_context)
    st.divider()
    st.markdown("#### 행정 대응 Agent")
    if st.session_state.documents is None or st.session_state.document_event != event["id"]:
        st.markdown(
            '<div class="notice"><b>예측 결과를 현장 대응 문서로 변환합니다.</b><br>'
            '상황 브리핑, 현장점검 지시서와 입력 가능한 사후 결과보고서를 생성할 수 있습니다.</div>',
            unsafe_allow_html=True,
        )
        if st.button("이 Event의 대응 문서 생성", type="primary", use_container_width=True, key="generate_main"):
            store_generated_documents(event, field_context)
            st.rerun()
        st.caption("문서를 생성하면 이 영역에서 바로 확인하고 현장 결과를 입력할 수 있습니다.")
    else:
        package = st.session_state.documents
        warning = st.session_state.get("document_warning")
        message = warning or f'{st.session_state.get("document_mode", "안전 템플릿")} 문서 생성 완료 · 담당자 검토 필요'
        st.markdown(f'<div class="notice">{escape(str(message))}</div>', unsafe_allow_html=True)
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
                inspected_at = datetime.combine(inspected_date, inspected_time, tzinfo=KST)
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
