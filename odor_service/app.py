"""과거 밤 상황을 다시 보는 발표용 화면."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
REPLAY_DIR = ROOT / "outputs/odor_service/replay"
SAMPLE = ROOT / "tests/fixtures/odor_service/replay_sample.json"
COMPASS = (
    "북", "북북동", "북동", "동북동", "동", "동남동", "남동", "남남동",
    "남", "남남서", "남서", "서남서", "서", "서북서", "북서", "북북서",
)


def compass_ko(degrees: float | None) -> str:
    if degrees is None:
        return "방향 정보 없음"
    return COMPASS[int((float(degrees) % 360 + 11.25) // 22.5) % 16]


def replay_files() -> list[Path]:
    files = [p for p in REPLAY_DIR.glob("*.json") if p.name != "index.json"]
    return sorted(files, key=lambda p: p.stem, reverse=True) if files else [SAMPLE]


def load_replay(path: Path | str | None = None) -> dict:
    selected = Path(path) if path is not None else replay_files()[0]
    with selected.open(encoding="utf-8") as handle:
        return json.load(handle)


def easy_text(value: object) -> str:
    """입력의 기술 표현을 발표 화면의 쉬운 말로 바꿈."""
    text = str(value)
    for old, new in (
        ("풍상 방향", "바람이 불어오는 방향"),
        ("풍상", "바람이 불어오는 쪽"),
        ("폭주", "민원 증가"),
        ("원인 농가", "확인 대상 농장"),
        ("확률", "위험 순위"),
        ("lift", "비교 수치"),
    ):
        text = text.replace(old, new)
    return text


def map_rows(replay: dict, event: dict | None = None) -> pd.DataFrame:
    rows = []
    chosen = replay.get("standby", {}).get("chosen_point_id")
    for point in replay.get("standby", {}).get("points", []):
        if point.get("grade") != "주":
            continue
        selected = point.get("point_id") == chosen
        rows.append({"lat": point["lat"], "lon": point["lon"],
                     "color": "#d62728" if selected else "#1f77b4",
                     "size": 110 if selected else 65})
    if event is not None:
        for complaint in event.get("first30_complaints", []):
            rows.append({"lat": complaint["lat"], "lon": complaint["lon"],
                         "color": "#ff9900", "size": 45})
        for region in event.get("top3", []):
            rows.append({"lat": region["lat"], "lon": region["lon"],
                         "color": "#2ca02c", "size": 85})
    return pd.DataFrame(rows, columns=["lat", "lon", "color", "size"])


def show_map(replay: dict, event: dict | None = None) -> None:
    rows = map_rows(replay, event)
    if rows.empty:
        st.info("표시할 위치 정보가 없음")
    else:
        st.map(rows, latitude="lat", longitude="lon", color="color", size="size")
    st.caption("빨강: 선택된 대기 장소 · 파랑: 다른 주 대기 장소" +
               (" · 주황: 첫 30분 민원 · 초록: 예측 동네" if event else ""))


def render_afternoon(replay: dict) -> None:
    afternoon = replay["afternoon"]
    st.header("오늘 밤 경보" if afternoon.get("alert") else "평상")
    st.write(f"5~10월 예측 순위: {afternoon.get('season_rank', '정보 없음')}위")
    st.write(f"다시 보기용 실제 값 · 밤 민원 수: {afternoon.get('observed_complaints', '정보 없음')}")
    wind = afternoon.get("night_wind", {})
    st.write(f"밤 평균 바람: {compass_ko(wind.get('direction'))}에서 불어옴 · 풍속 {wind.get('speed', '정보 없음')} m/s")
    st.subheader("대기 장소")
    show_map(replay)
    chosen = replay.get("standby", {}).get("chosen_point_id")
    for point in replay.get("standby", {}).get("points", []):
        if point.get("grade") == "주":
            label = "선택 · " if point.get("point_id") == chosen else ""
            st.write(f"{label}{point.get('name', point.get('point_id'))}")
    st.write(easy_text(replay.get("standby", {}).get("reason", "")))
    st.info("기존 24시간 상황실 인력이 저녁 6시~새벽 6시에 선택된 대기 장소에서 대기")


def render_direction(event: dict, replay: dict) -> None:
    overlap = event.get("direction_overlap", {})
    suggested = overlap.get("suggested_point_id")
    if suggested is None:
        st.write("바람이 약해 방향 판단 어려움")
        return
    point = next((p for p in replay.get("standby", {}).get("points", [])
                  if p.get("point_id") == suggested), None)
    count = next((p.get("n_pointing") for p in overlap.get("per_point", [])
                  if p.get("point_id") == suggested), None)
    name = point.get("name", suggested) if point else suggested
    st.write(f"민원 {overlap.get('n_complaints', '정보 없음')}곳 중 {count if count is not None else '정보 없음'}곳이 {name} 방면을 가리킴")


def render_card(card: dict) -> None:
    with st.container(border=True):
        st.subheader(f"{card.get('rank', '—')}순위 · {card.get('name', '이름 없음')}")
        st.write(f"확인 순서 점수: {card.get('score', '정보 없음')} · 거리: {card.get('distance_km', '정보 없음')} km")
        components = card.get("components", {})
        if components:
            st.bar_chart(pd.DataFrame({"항목": list(components), "점수": list(components.values())}).set_index("항목"))
        st.write(easy_text(card.get("reason", "")))
        if card.get("coord_warning"):
            st.caption("위치 대략")
        st.caption(easy_text(card.get("disclaimer") or "현장 확인 참고자료이며 발생원 확정이나 위반 판정이 아님"))


def render_night(replay: dict) -> None:
    events = replay.get("events", [])
    if not events:
        st.info("이 날짜에 표시할 사건이 없음")
        return
    event = st.selectbox("사건 시각", events, format_func=lambda item: item.get("event_hour", "시각 없음"))
    field = event.get("field", {})
    st.write(f"신뢰도: {field.get('confidence', '정보 없음')}")
    for note in field.get("notes", []):
        st.write(easy_text(note))
    st.subheader("첫 30분 민원 · 예측 동네 · 대기 장소")
    show_map(replay, event)
    for region in event.get("top3", []):
        result = "실제 민원 들어옴" if region.get("hit") else "실제 민원 없음"
        st.write(f"{region.get('rank')}순위 · {region.get('region_name')} · {result}")
    render_direction(event, replay)
    st.subheader("방문 순서")
    cards = field.get("cards", [])
    if not cards:
        st.info("방문 순서 정보가 없음")
    for card in cards[:5]:
        render_card(card)


def main() -> None:
    st.set_page_config(page_title="익산 악취상황실 다시 보기", layout="wide")
    st.title("익산 악취상황실 · 다시 보기")
    files = replay_files()
    selected = st.selectbox("날짜", files, format_func=lambda p: p.stem)
    replay = load_replay(selected)
    if selected == SAMPLE:
        st.warning("화면 개발용 샘플 · 실제 결과가 아님")
    if replay.get("_note"):
        st.caption(easy_text(replay["_note"]))
    afternoon_tab, night_tab = st.tabs(["오후 브리핑", "밤 현장"])
    with afternoon_tab:
        render_afternoon(replay)
    with night_tab:
        render_night(replay)


if __name__ == "__main__":
    main()
