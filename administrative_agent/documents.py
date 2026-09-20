from __future__ import annotations

from datetime import timedelta

from .models import ForecastResult
from .policy import DISCLAIMER, response_level


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


def _bearing_name(degrees: float | None) -> str:
    if degrees is None:
        return "미제공"
    names = ("북", "북동", "동", "남동", "남", "남서", "서", "북서")
    return f"{names[round(float(degrees) / 45) % 8]}쪽({float(degrees):.0f}°)"


def _candidate_location(candidate) -> str:
    label = candidate.name
    if candidate.location_precision == "village":
        label += " [리 단위 추정]"
    return label


def _uncertainty_label(value: float | None) -> str:
    if value is None:
        return "미제공"
    level = "낮음" if value < 0.33 else ("중간" if value < 0.66 else "높음")
    return f"{level}({value:.2f})"


def source_candidate_lines(forecast: ForecastResult, heading: str) -> list[str]:
    """발생원 후보 절. 후보가 없으면 빈 목록을 돌려 문서에서 절 자체를 생략한다.

    표기 순서는 후보 → 근거(방향·거리·도달시간·풍향 일치) → 불확실성 → 주의 문구로 고정한다.
    시설을 원인으로 단정하는 표현은 쓰지 않는다.
    """
    candidates = forecast.source_candidates
    if not candidates:
        return []
    lines = [
        "", heading, "",
        "초기 30분 민원 위치와 Event 이전 바람으로 계산한 '현재 민원 분포를 설명하는 정도'가 높은 발생원 후보입니다. 원인 시설 판정이 아니라 현장 확인 시 상풍측 참고 정보입니다.", "",
        "|순위|발생원 후보|방향·거리|도달시간|풍향 일치|적합도|", "|---:|---|---|---:|---:|---:|",
    ]
    for c in candidates:
        distance = "미제공" if c.distance_km is None else f"{c.distance_km:.1f}km"
        travel = "미제공" if c.travel_time_min is None else f"약 {c.travel_time_min:.0f}분"
        alignment = "미제공" if c.wind_alignment is None else f"{c.wind_alignment:.2f}"
        fit = "미제공" if c.fit_score is None else f"{c.fit_score:.2f}"
        lines.append(f"|{c.rank}|{_candidate_location(c)}|{_bearing_name(c.bearing_deg)} {distance}|{travel}|{alignment}|{fit}|")
    lines += [
        "", f"- 역추적 불확실성: {_uncertainty_label(forecast.backtrack_uncertainty)}"
        + (f" (바람 출처: {forecast.backtrack_weather_source})" if forecast.backtrack_weather_source else ""),
        "- 방향은 후보에서 민원 지점을 향한 방위, 도달시간은 거리÷풍속의 점추정입니다. 관측소 바람이라 국지 풍향과 다를 수 있습니다.",
        "- 후보 확인은 민원 권역 점검을 대체하지 않으며, 시설 방문·지도는 별도 법적 절차와 담당자 판단을 따릅니다.",
    ]
    evidence = [c.evidence_text for c in candidates if c.evidence_text]
    if evidence:
        lines += ["", "근거 요약:"] + [f"- {text}" for text in evidence[:3]]
    return lines


def onset_alert_lines(forecast: ForecastResult, heading: str) -> list[str]:
    """사전 경보 절. 발생 위험 예보(1단) 격자가 없으면 빈 목록을 돌려 절을 생략한다.

    확산 예측(Top 3)과 다른 모델의 결과라는 점, 기상·발생원 노출·과거 빈도로 계산했다는 점을 밝힌다.
    """
    alerts = forecast.onset_alerts
    if not alerts:
        return []
    reference = forecast.onset_reference_time
    when = f"{reference:%H:%M}" if reference else "기준시각 1시간 전"
    lines = [
        "", heading, "",
        f"민원이 접수되기 전인 {when} 시점에, 기상(풍향·풍속·강수·기온)·상풍측 6km 축산 발생원 노출·격자별 과거 민원 빈도로 계산한 '다음 1시간 발생 위험' 상위 격자입니다. "
        "위 확산 예측(Top 3)과는 다른 모델(발생 위험 예보 원형)의 결과이며, 두 결과가 겹치는 격자는 우선 확인 근거가 하나 더 있는 것으로 볼 수 있습니다.", "",
        "|순위|격자|상대위험|상풍측 축산 노출 비율|", "|---:|---|---:|---:|",
    ]
    top3 = {area.grid_id for area in forecast.areas}
    for cell in alerts:
        share = "미제공" if cell.upwind_share is None else f"{cell.upwind_share:.0%}"
        mark = " (확산 예측 Top 3와 일치)" if cell.grid_id in top3 else ""
        lines.append(f"|{cell.rank}|{_location(cell)}{mark}|{cell.relative_risk}/100|{share}|")
    quiet = alerts[0].quiet_hour
    lines += [
        "", "- 상대위험은 같은 시각 격자 중 최고를 100으로 둔 상대값이며 발생 확률이 아닙니다.",
        "- 상풍측 축산 노출 비율은 격자 6km 안 축산 시설 배출 가중치(EMEP/EEA NH3 계수 × 사육두수) 중 그 시각 바람이 불어오는 쪽(±30°)의 비율입니다.",
    ]
    if quiet is not None:
        lines.append("- 이 시각은 직전 3시간 동안 시 전체 민원이 없던 '조용한 시각'" + ("입니다." if quiet else "이 아닙니다(이미 민원이 이어지던 상황).")
                     + " 조용한 시각의 상위 5 격자 적중률은 검증 구간에서 0.58 수준(참고)입니다.")
    if forecast.onset_alerts_by_type:
        # 유형별 모델(라벨을 그 냄새 종류 민원으로 제한)의 상위 3 격자. 담당 부서가 다르므로 따로 보인다.
        lines += ["", "냄새 유형별 위험 상위 격자(같은 시각, 유형별 모델):"]
        for odor_type, cells in forecast.onset_alerts_by_type.items():
            if not cells:
                continue
            parts = [f"{c.grid_id}({c.relative_risk})" for c in cells[:3]]
            lines.append(f"- {odor_type} 냄새: " + ", ".join(parts))
        lines.append("- 괄호는 그 유형 안에서의 상대위험(최고 100). 검증 구간 조용한 시각 Hit@5: 가축 0.57, 공장 0.75(참고).")
    return lines


def _time_windows(forecast: ForecastResult) -> tuple[str, str]:
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
        f"- 초기 30분 접수 민원: **{_value(forecast.initial_complaint_count, '건')}**",
        f"- 초기 민원 발생 격자: **{_value(forecast.initial_grid_count, '개')}**",
        f"- 주요 우선확인 지역: {', '.join(regions) + ' 일대' if regions else '미제공'}",
        f"- 초기 신고 강도: 평균 {_value(forecast.initial_intensity_average)} / 최대 {_value(forecast.initial_intensity_maximum)}", "",
        "## 2. AI 예측 결과", "",
        "초기 민원 위치·강도·신고 분포와 과거 민원 패턴을 분석한 결과, 향후 30분 동안 추가 민원이 접수될 가능성이 상대적으로 높은 권역은 다음과 같습니다.", "",
        "|순위|우선 확인 권역|상대위험점수|대응 수준|", "|---:|---|---:|---|",
    ]
    for area in forecast.areas:
        level, _ = response_level(area.relative_risk)
        lines.append(f"|{area.rank}|{_location(area)}|{area.relative_risk}/100|{level}|")
    weather = forecast.weather
    if forecast.narrowed_candidate_count is not None:
        lines += ["", f"- 후보 격자 {forecast.candidate_count}개 중 기준시각 1시간 전 발생 위험 예보 상위 {forecast.narrow_rank_limit}위 안 "
                      f"{forecast.narrowed_candidate_count}개에서 선정 (후보 축소 규칙, 적중률 변화 없음 확인)"]
    lines += [
        "", "※ 상대위험점수는 실제 악취 발생확률이 아닌 동일 Event 내 후보권역 간 우선순위 판단을 위한 상대적 점수입니다.", "",
        "## 3. 참고 기상정보", "",
        f"- 풍향: {_wind_direction(weather.get('windDirection'))}",
        f"- 풍속: {_value(weather.get('windSpeed'), 'm/s')}",
        f"- 상대습도: {_value(weather.get('humidity'), '%')}",
        f"- 최근 1시간 강수량: {_value(weather.get('rainfall'), 'mm')}",
        "- 기상정보는 확산 예측(Top 3) 모델 입력에는 쓰지 않음(검증 결과 개선 없음). 아래 참고 절이 있으면 그 계산에는 사용됨",
    ]
    section = 4
    if forecast.source_candidates:
        lines += source_candidate_lines(forecast, f"## {section}. 발생원 후보 (참고)")
        section += 1
    if forecast.onset_alerts:
        lines += onset_alert_lines(forecast, f"## {section}. 사전 경보 (참고)")
        section += 1
    lines += [
        "", f"## {section}. 상황 판단", "",
        "현재 민원 분포와 AI 예측 결과를 고려하여 1순위 권역을 우선 확인대상으로 검토하고, 현장 상황과 기상조건에 따라 2·3순위 권역을 순차적으로 확인할 필요가 있습니다.", "",
        f"> **주의:** {DISCLAIMER}",
    ]
    return "\n".join(lines)


def create_dispatch_order(forecast: ForecastResult) -> str:
    lines = [
        "# 악취 민원 현장점검 지시서", "",
        f"- **Event ID:** {forecast.event_id}",
        f"- **지시시각:** {forecast.generated_at:%Y.%m.%d. %H:%M}",
        "- **점검목적:** 추가 민원 가능성이 높은 권역의 우선 현장 확인",
        "- **승인상태:** 담당자 검토 필요", "",
        "## 1. 점검 대상", "",
        "|우선순위|점검 권역|상대위험점수|점검 기준|", "|---:|---|---:|---|",
    ]
    for area in forecast.areas:
        level, action = response_level(area.relative_risk)
        lines.append(f"|{area.rank}순위|{_location(area)}|{area.relative_risk}/100|{action}|")
    lines += source_candidate_lines(forecast, "### 점검 시 참고: 상풍측 발생원 후보")
    lines += [
        "", "## 2. 현장 확인 항목", "",
        "현장 도착 시 다음 사항을 확인·기록합니다.", "",
        "- 도착시각 및 실제 점검 위치",
        "- 현장 악취 감지 여부와 악취강도 또는 측정값",
        "- 현장 풍향·풍속 및 주변 악취 발생 상황",
        "- 추가 민원 발생 여부와 현장 조치 내용",
        "- 추가 점검 필요 여부", "",
        "## 3. 점검 결과 입력항목", "",
        "|항목|입력|", "|---|---|",
        "|출동 결정시각|미입력|", "|현장 출발시각|미입력|", "|현장 도착시각|미입력|",
        "|점검 권역|미입력|", "|악취 감지|□ 확인 / □ 미확인|", "|측정값|미입력|",
        "|조치 내용|미입력|", "|추가 점검|□ 필요 / □ 불필요|", "",
        "## 4. 담당자 확인", "",
        "- [ ] 우선순위와 가용 인력을 확인함", "- [ ] 현장 안전 및 점검 권한을 확인함", "",
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
        f"- 초기 민원 발생 격자: {_value(forecast.initial_grid_count, '개')}",
        f"- 예측대상 기간: {forecast_window}", "",
        "## 2. AI 우선권역", "",
        "|순위|예측 권역|상대위험점수|이후 추가 민원|", "|---:|---|---:|---|",
    ]
    for area in forecast.areas:
        lines.append(f"|{area.rank}|{_location(area)}|{area.relative_risk}/100|미입력|")
    lines += [
        "", "## 3. 현장 대응 결과", "",
        "- 출동 결정시각: 미입력", "- 현장 출발시각: 미입력", "- 현장 도착시각: 미입력",
        "- 실제 점검 권역: 미입력", "- 현장 악취 감지 여부: 미입력", "- 현장 측정값: 미입력", "- 총 출동거리: 미입력", "",
        "## 4. 현장 확인 및 조치내용", "",
        "- 현장 확인내용: 미입력", "- 실시 조치: 미입력", "- 추가 조치 필요 여부: □ 필요 / □ 불필요", "",
        "## 5. AI 예측 결과와 실제 결과 비교", "",
        "- Top 3 내 실제 추가 민원 권역 포함 여부: 미입력", "- 실제 추가 민원 권역 수: 미입력",
        "- Top 3 내 포함 권역 수: 미입력", "- 현장 악취 확인 여부: 미입력", "- 최초 현장 도착 소요시간: 미입력", "",
        "## 6. 종합 결과", "",
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
        ("- 기상정보는 현재 예측모델 입력이 아닙니다. 아래 발생원 후보는 바람·거리 기반 참고 정보이며 시설 특정이 아닙니다."
         if forecast.source_candidates else
         "- 기상정보는 현재 예측모델 입력이 아니며, 발생원 역추적이나 시설 특정에 사용하지 않습니다."),
        *source_candidate_lines(forecast, "## 발생원 후보 (참고)"), "",
        "## 기록 및 후속 검토", "",
        "현장 확인값과 실시 조치를 동일 Event ID에 기록하고, 확인되지 않은 내용은 추정해 채우지 않습니다. 추가 민원이나 현장 측정 결과가 확보되면 담당자가 대응 우선순위를 다시 검토합니다.", "",
        "> **담당자 검토 필요:** 본 가이드는 AI가 생성한 현장 대응 참고사항이며 행정조치 지시가 아닙니다. 실제 대응은 현장 측정 결과, 안전수칙과 담당자의 판단을 따릅니다.",
    ])
