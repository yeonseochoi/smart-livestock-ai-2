from __future__ import annotations

from .models import ForecastResult
from .policy import DISCLAIMER, response_level


def _location(area) -> str:
    if area.center_latitude is None or area.center_longitude is None:
        return area.grid_id
    return f"{area.grid_id} (중심 {area.center_latitude:.6f}, {area.center_longitude:.6f})"


def create_briefing(forecast: ForecastResult) -> str:
    lines = [
        "# 악취 민원 상황 브리핑", "",
        f"- 이벤트: {forecast.event_id}",
        f"- 기준 시각: {forecast.event_time:%Y-%m-%d %H:%M}",
        f"- 예측 구간: 기준 시각 이후 {forecast.forecast_minutes}분",
        "- 판단: 추가 민원이 접수될 가능성이 높은 1km 권역 3곳을 선별함", "",
        "## 우선 대응 권역", "",
        "|순위|1km 권역|상대 위험도|대응 수준|", "|---:|---|---:|---|",
    ]
    for area in forecast.areas:
        level, _ = response_level(area.relative_risk)
        lines.append(f"|{area.rank}|{_location(area)}|{area.relative_risk}/100|{level}|")
    lines += ["", "## 해석 시 주의사항", "", DISCLAIMER]
    return "\n".join(lines)


def create_dispatch_order(forecast: ForecastResult) -> str:
    lines = [
        "# 현장 점검 지시서", "",
        f"- 관련 이벤트: {forecast.event_id}",
        f"- 점검 권고 시간: {forecast.event_time:%Y-%m-%d %H:%M}부터 {forecast.forecast_minutes}분 이내",
        "- 승인 상태: 담당자 검토 필요", "",
        "## 점검 순서", "",
    ]
    for area in forecast.areas:
        level, action = response_level(area.relative_risk)
        lines += [
            f"### {area.rank}순위 — {_location(area)}", "",
            f"- 상대 위험도: {area.relative_risk}/100 ({level})",
            f"- 권고: {action}",
            "- 확인 항목: 현장 악취 감지 여부, 측정값, 풍향·풍속, 민원 추가 접수 여부",
            "- 기록 항목: 도착·종료 시각, 확인 위치, 측정 장비, 조치 내용", "",
        ]
    lines += ["## 담당자 확인", "", "- [ ] 우선순위와 가용 인력을 확인함", "- [ ] 현장 안전 및 점검 권한을 확인함", "", DISCLAIMER]
    return "\n".join(lines)


def create_followup_template(forecast: ForecastResult) -> str:
    lines = [
        "# 사후 결과보고서(확장 예시)", "",
        f"- 관련 이벤트: {forecast.event_id}",
        f"- 예측 기준 시각: {forecast.event_time:%Y-%m-%d %H:%M}",
        "- 작성 상태: 실제 서비스 도입 시 현장 결과 입력 필요", "",
        "## 권역별 결과", "",
        "|예측 순위|1km 권역|추가 민원 발생|현장 악취 감지|측정 결과|조치 내용|", "|---:|---|---|---|---|---|",
    ]
    for area in forecast.areas:
        lines.append(f"|{area.rank}|{_location(area)}|미입력|미입력|미입력|미입력|")
    lines += [
        "", "## 운영 성과 기록", "",
        "- Top 3 권역 내 추가 민원 적중 여부: 미입력",
        "- 최초 현장 도착 소요시간: 미입력",
        "- 불필요 출동 또는 누락 사유: 미입력",
        "- 담당자 의견 및 모델 개선 메모: 미입력", "",
        "※ 이 문서는 현재 예측 성능 평가값을 사후 사실처럼 채우지 않는 입력 템플릿입니다.",
    ]
    return "\n".join(lines)
