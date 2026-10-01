from __future__ import annotations

from datetime import timedelta

from .models import ForecastResult
from .policy import DISCLAIMER
from .time_utils import as_kst


def _field_note_for_official(note: str) -> str | None:
    """내부 계산 메모를 담당자가 바로 읽을 수 있는 현장 안내로 바꾼다."""
    if "풍속 0.5 m/s 미만" in note:
        return "바람이 매우 약합니다. 후보 순서보다 현장에서 실제 풍향과 냄새를 먼저 확인하세요."
    if "풍속 1 m/s 미만" in note:
        return "바람이 약해 방향이 바뀔 수 있습니다. 현장 풍향을 확인한 뒤 후보 순서를 판단하세요."
    if "함께 가리키는 농가가 없어" in note:
        return "여러 민원 위치가 공통으로 가리키는 농가가 없어, 한 곳의 민원 위치를 기준으로 후보를 제시했습니다."
    if "교차 확인 후보가 부족해" in note or "기본 조건 후보가 부족해" in note:
        return "조건을 만족하는 후보가 부족해 보충 후보가 포함됐습니다. 앞 순위 농가를 먼저 확인하세요."
    if "풍향이 ±20° 달라지면" in note:
        return "풍향이 조금만 달라져도 후보가 바뀔 수 있습니다. 현장 풍향을 다시 확인하세요."
    return None


def field_candidate_body(forecast: ForecastResult) -> str:
    """LLM 보정 시에도 그대로 보존할 현장 후보의 확정 입력(제목 제외)."""
    def clean(value: object) -> str:
        return " ".join(str(value).replace("|", " / ").split())

    lines = []
    lines.append(f"- 풍향 신뢰도: {clean(forecast.field_confidence or '미제공')}")
    for note in forecast.field_notes:
        readable = _field_note_for_official(str(note))
        if readable:
            lines.append(f"- {readable}")
    lines.append("")
    if not forecast.field_candidates:
        lines += ["조건에 맞는 방문 후보 없음", ""]
    for candidate in forecast.field_candidates:
        lines += [
            f"### {candidate.rank}순위 · {clean(candidate.display_name)}",
            f"- 주소: {clean(candidate.address)}",
            f"- 점검 후보 점수: {candidate.score:.1f}/100",
            "- 항목별 점수: " + " · ".join(
                f"{name} {candidate.components.get(key, 0):.1f}/{maximum}"
                for key, name, maximum in (("풍향 일치", "풍향 일치", 20), ("거리", "민원 근접", 20),
                                           ("다중 측정 일치", "민원 중첩", 40), ("과거 반복", "과거 반복", 20))
            ),
        ]
        if candidate.tier and candidate.support_count is not None:
            lines.append(f"- 민원 위치 근거: 전체 {candidate.support_total}곳 중 {candidate.support_count}곳이 이 농가 방향을 가리킴")
        if candidate.complaint_km is not None:
            lines.append(f"- 가장 가까운 민원 위치와의 거리: {candidate.complaint_km:.1f} km")
        if candidate.coord_warning:
            lines.append("- 근사 좌표: 방문 전 주소 확인")
        lines.append("")
    lines += ["※ 현장 확인 우선순위이며 발생원 확정을 위한 보조 도구로만 활용하세요."]
    return "\n".join(lines)


def field_candidate_section(forecast: ForecastResult, heading: str = "## 우선 점검 농가 후보") -> str:
    return heading + "\n\n" + field_candidate_body(forecast)


def _location(area) -> str:
    """현장에서 찾아갈 수 있도록 읍면동 이름을 앞세우고 중심 좌표를 덧붙인다.

    격자 인덱스는 내부 식별자라 그대로 노출하면 담당자가 위치를 알 수 없다.
    읍면동이 없으면 격자 인덱스로 되돌린다.
    """
    label = area.region_name or area.grid_id
    if area.center_latitude is None or area.center_longitude is None:
        return label
    return f"{label} (중심 {area.center_latitude:.6f}, {area.center_longitude:.6f})"


def _value(value, suffix: str = "") -> str:
    return "미제공" if value is None else f"{value}{suffix}"


def _wind_direction(degrees: float | int | None) -> str:
    if degrees is None:
        return "미제공"
    names = ("북풍", "북동풍", "동풍", "남동풍", "남풍", "남서풍", "서풍", "북서풍")
    return f"{names[round(float(degrees) / 45) % 8]}({float(degrees):.0f}°)"


def _time_windows(forecast: ForecastResult) -> tuple[str, str]:
    if forecast.event_time_is_boundary:
        start = forecast.event_time - timedelta(minutes=forecast.forecast_minutes)
        end = forecast.event_time + timedelta(minutes=forecast.forecast_minutes)
        return f"{start:%H:%M}~{forecast.event_time:%H:%M}", f"{forecast.event_time:%H:%M}~{end:%H:%M}"
    boundary = forecast.event_time + timedelta(minutes=forecast.forecast_minutes)
    end = boundary + timedelta(minutes=forecast.forecast_minutes)
    return (
        f"{forecast.event_time:%H:%M}~{boundary:%H:%M}",
        f"{boundary:%H:%M}~{end:%H:%M}",
    )


def create_briefing(forecast: ForecastResult) -> str:
    analysis_window, forecast_window = _time_windows(forecast)
    regions = list(dict.fromkeys(area.region_name for area in forecast.areas if area.region_name))
    lines = [
        "# 악취 민원 확산 상황 브리핑", "",
        f"- **Event ID:** {forecast.event_id}",
        f"- **기준시각:** {forecast.event_time:%Y.%m.%d. %H:%M}",
        f"- **분석구간:** {analysis_window}",
        f"- **예측구간:** {forecast_window}", "",
        "## 1. 민원 발생 현황", "",
        f"- 초기 30분 신고 접수 건수: **{_value(forecast.initial_complaint_count, '건')}**",
        *([f"- 주요 우선확인 지역: {', '.join(regions)} 일대"] if regions else []),
        f"- 초기 신고 강도: 평균 {_value(forecast.initial_intensity_average)} / 최대 {_value(forecast.initial_intensity_maximum)}", "",
        "## 2. AI 예측 결과", "",
        "초기 민원 위치·강도·신고 분포와 과거 민원 패턴을 분석한 결과, 향후 30분 동안 추가 민원이 접수될 가능성이 상대적으로 높은 권역은 다음과 같습니다.", "",
        "|순위|추가 민원 접수 예상 지역|상대위험점수|", "|---:|---|---:|",
    ]
    for area in forecast.areas:
        lines.append(f"|{area.rank}|{_location(area)}|{area.relative_risk}/100|")
    weather = forecast.weather
    lines += [
        "", "※ 상대위험점수는 실제 악취 발생확률이 아닌 동일 Event 내 후보권역 간 우선순위 판단을 위한 상대적 점수입니다.", "",
        "## 3. 참고 기상정보", "",
        f"- 풍향: {_wind_direction(weather.get('windDirection'))}",
        f"- 풍속: {_value(weather.get('windSpeed'), 'm/s')}",
        f"- 상대습도: {_value(weather.get('humidity'), '%')}",
        f"- 최근 1시간 강수량: {_value(weather.get('rainfall'), 'mm')}",
        "",
        f"> **주의:** {DISCLAIMER}", "",
        field_candidate_section(forecast, "## 4. 우선 점검 농가 후보"),
    ]
    return "\n".join(lines)


def create_dispatch_order(forecast: ForecastResult) -> str:
    lines = [
        "# 악취 민원 현장점검 지시서", "",
        f"- **Event ID:** {forecast.event_id}",
        f"- **지시시각:** {as_kst(forecast.generated_at):%Y.%m.%d. %H:%M} (한국 표준시)",
        "- **점검목적:** 우선 점검 농가 후보의 현장 상태 확인",
        "- **승인상태:** 담당자 검토 필요", "",
        "## 1. 출동 전 확인", "",
        "아래 농가 후보를 순위순으로 검토하되, 출동 직전에 최신 민원 위치와 현장 풍향을 다시 확인합니다.", "",
        "## 2. 우선 점검 농가 후보", "",
        field_candidate_body(forecast),
        "", "## 3. 현장 확인 항목", "",
        "농가에 도착하면 다음 사항을 확인·기록합니다.", "",
        "- 실제 점검 농가와 도착시각",
        "- 현장 풍향·풍속과 민원 위치에서 본 방향의 일치 여부",
        "- 악취 감지 여부와 악취 강도 또는 측정값",
        "- 농가 주변의 추가 민원·주변 상황",
        "- 실시한 조치와 추가 점검 필요 여부", "",
        "## 4. 출동 기록", "",
        "- 출동 결정시각, 현장 출발시각, 현장 도착시각",
        "- 실제 점검 농가와 이동거리",
        "- 현장 확인내용, 측정값, 조치내용", "",
        "## 5. 유의사항", "", DISCLAIMER,
    ]
    return "\n".join(lines)


def create_followup_template(forecast: ForecastResult) -> str:
    _, forecast_window = _time_windows(forecast)
    lines = [
        "# 악취 민원 대응 사후 결과보고서", "",
        f"- **Event ID:** {forecast.event_id}",
        f"- **Event 발생시각:** {forecast.event_time:%Y.%m.%d. %H:%M}",
        "- **보고서 작성시각:** 미입력", "- **작성상태:** 현장 결과 입력 필요", "",
        "## 1. 발생 개요", "",
        f"- 초기 민원 접수: {_value(forecast.initial_complaint_count, '건')}",
        f"- 예측대상 기간: {forecast_window}", "",
        "## 2. 우선 점검 농가 현장 결과", "",
        "|순위|점검 농가|점검 후보 점수|악취 감지|현장 풍향|측정 결과|조치 내용|", "|---:|---|---:|---|---|---|---|",
        *[
            f"|{candidate.rank}|{candidate.display_name}|{candidate.score:.1f}/100|미입력|미입력|미입력|미입력|"
            for candidate in forecast.field_candidates
        ],
        "", "## 3. AI 우선권역 참고", "",
        "|순위|예측 권역|상대위험점수|이후 추가 민원|", "|---:|---|---:|---|",
    ]
    for area in forecast.areas:
        lines.append(f"|{area.rank}|{_location(area)}|{area.relative_risk}/100|미입력|")
    lines += [
        "", "## 4. 현장 대응 결과", "",
        "- 출동 결정시각: 미입력", "- 현장 출발시각: 미입력", "- 현장 도착시각: 미입력",
        "- 실제 점검 농가: 미입력", "- 현장 악취 감지 여부: 미입력", "- 현장 측정값: 미입력", "- 총 출동거리: 미입력", "",
        "## 5. 현장 확인 및 조치내용", "",
        "- 현장 확인내용: 미입력", "- 실시 조치: 미입력", "- 추가 조치 필요 여부: □ 필요 / □ 불필요", "",
        "## 6. AI 예측 권역과 실제 결과 비교", "",
        "- Top 3 내 실제 추가 민원 권역 포함 여부: 미입력", "- 실제 추가 민원 권역 수: 미입력",
        "- Top 3 내 포함 권역 수: 미입력", "- 현장 악취 확인 여부: 미입력", "- 최초 현장 도착 소요시간: 미입력", "",
        "## 7. 종합 결과", "",
        "현장 확인 결과와 조치 내용을 종합하여 작성하며, 동일 Event ID를 기준으로 저장해 향후 현장 실증 성능평가와 모델 개선 자료로 활용합니다.", "",
        "※ 확인되지 않은 현장 결과나 현재 예측 성능 평가값을 사후 사실처럼 자동 기입하지 않습니다.",
    ]
    return "\n".join(lines)


def create_response_guide(forecast: ForecastResult) -> str:
    """확인된 입력만 사용해 담당자 검토용 대응 참고사항을 만든다."""
    first = forecast.areas[0]
    weather = forecast.weather
    weather_items = [
        f"풍향 {_wind_direction(weather.get('windDirection'))}",
        f"풍속 {_value(weather.get('windSpeed'), 'm/s')}",
        f"상대습도 {_value(weather.get('humidity'), '%')}",
        f"최근 강수 {_value(weather.get('rainfall'), 'mm')}",
    ]
    return "\n".join([
        "# AI 현장 대응 참고 가이드", "",
        "## 우선 확인 방향", "",
        f"현재 Event에서는 **{_location(first)}**가 1순위 권역입니다. 해당 권역을 먼저 확인하고, 현장 확인 결과와 추가 민원 접수 상황을 함께 검토한 뒤 2·3순위 권역의 점검 필요성을 판단합니다.", "",
        "## 현장 확인 순서", "",
        "1. 출발 전 최신 추가 민원 위치와 접근 가능한 점검 경로를 확인합니다.",
        "2. 현장 도착 시 악취 감지 여부, 시각, 위치와 측정값을 기록합니다.",
        "3. 한 지점의 결과만으로 판단하지 않고 필요하면 권역 내 복수 지점을 확인합니다.",
        "4. 1순위에서 악취가 확인되지 않으면 추가 민원과 현장 여건을 근거로 2·3순위 이동을 검토합니다.", "",
        "## Event 참고정보", "",
        f"- 초기 접수 민원: {_value(forecast.initial_complaint_count, '건')}",
        f"- 현장 참고 기상: {', '.join(weather_items)}",
        "- 방문 후보는 현장 확인을 위한 참고 순위입니다.", "",
        "## 기록 및 후속 검토", "",
        "현장 확인값과 실시 조치를 동일 Event ID에 기록하고, 확인되지 않은 내용은 추정해 채우지 않습니다. 추가 민원이나 현장 측정 결과가 확보되면 담당자가 대응 우선순위를 다시 검토합니다.", "",
        "> **담당자 검토 필요:** 본 가이드는 AI가 생성한 현장 대응 참고사항이며 행정조치 지시가 아닙니다. 실제 대응은 현장 측정 결과, 안전수칙과 담당자의 판단을 따릅니다.",
    ])
