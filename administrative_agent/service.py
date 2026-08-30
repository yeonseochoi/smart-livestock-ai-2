from __future__ import annotations

import json
from pathlib import Path

from .documents import _location, create_briefing, create_dispatch_order, create_followup_template, create_response_guide
from .models import ForecastResult, ResponsePackage


def build_response_package(forecast: ForecastResult) -> ResponsePackage:
    """예측 엔진의 표준 출력만 받아 행정 대응 문서 3종을 생성한다."""
    return ResponsePackage(
        event_id=forecast.event_id,
        review_required=True,
        briefing=create_briefing(forecast),
        dispatch_order=create_dispatch_order(forecast),
        followup_report_template=create_followup_template(forecast),
        response_guide=create_response_guide(forecast),
        forecast=forecast,
    )


def write_response_package(package: ResponsePackage, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "complaint_briefing.md").write_text(package.briefing, encoding="utf-8")
    (output_dir / "field_inspection_order.md").write_text(package.dispatch_order, encoding="utf-8")
    (output_dir / "followup_report_template.md").write_text(package.followup_report_template, encoding="utf-8")
    (output_dir / "response_guide.md").write_text(package.response_guide, encoding="utf-8")
    (output_dir / "agent_output.json").write_text(
        json.dumps(package.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def create_completed_followup(package: ResponsePackage, outcome: dict[str, object]) -> str:
    """담당자가 입력한 현장 결과를 보존해 사후 결과보고서를 완성한다."""
    rows = outcome.get("areas", [])
    by_rank = {int(row["rank"]): row for row in rows if "rank" in row}
    clean = lambda value: str(value or "미입력").replace("|", "\\|").replace("\n", " ")
    top3_hits = sum(1 for row in by_rank.values() if row.get("additional_complaint") == "발생")
    fully_checked = by_rank and all(row.get("additional_complaint") in {"발생", "미발생"} for row in by_rank.values())
    hit_label = "포함" if top3_hits else ("미포함" if fully_checked else "미확인")
    forecast = package.forecast
    lines = [
        "# 악취 민원 대응 사후 결과보고서", "", f"- **Event ID:** {package.event_id}",
        f"- **Event 발생시각:** {forecast.event_time:%Y.%m.%d. %H:%M}",
        f"- **보고서 작성시각:** {clean(outcome.get('inspected_at'))}",
        f"- **작성자:** {clean(outcome.get('author'))}", "",
        "## 1. 발생 개요", "",
        f"- 초기 민원 접수: {f'{forecast.initial_complaint_count}건' if forecast.initial_complaint_count is not None else '미제공'}",
        f"- 초기 민원 발생 격자: {f'{forecast.initial_grid_count}개' if forecast.initial_grid_count is not None else '미제공'}",
        f"- 예측대상 기간: 기준시각 이후 {forecast.forecast_minutes}분", "",
        "## 2. AI 우선권역 및 실제 결과", "",
        "|순위|예측 권역|상대위험점수|추가 민원|악취 감지|측정 결과|조치 내용|", "|---:|---|---:|---|---|---|---|",
    ]
    for area in package.forecast.areas:
        row = by_rank.get(area.rank, {})
        values = [row.get("additional_complaint"), row.get("odor_detected"), row.get("measurement"), row.get("action")]
        safe = [clean(value) for value in values]
        lines.append(f"|{area.rank}|{_location(area)}|{area.relative_risk}/100|{safe[0]}|{safe[1]}|{safe[2]}|{safe[3]}|")
    lines += [
        "", "## 3. 현장 대응 결과", "",
        f"- 출동 결정시각: {clean(outcome.get('dispatch_decided_at'))}",
        f"- 현장 출발시각: {clean(outcome.get('departed_at'))}",
        f"- 현장 도착시각: {clean(outcome.get('arrived_at'))}",
        f"- 실제 우선 점검 권역: {clean(outcome.get('checked_area'))}",
        f"- 총 출동거리: {clean(outcome.get('total_distance_km'))}km", "",
        "## 4. 현장 확인 및 조치내용", "",
        f"- 현장 확인내용: {clean(outcome.get('field_findings'))}",
        f"- 추가 조치 필요 여부: {clean(outcome.get('followup_required'))}", "",
        "## 5. AI 예측 결과와 실제 결과 비교", "",
        f"- Top 3 내 실제 추가 민원 권역 포함 여부: **{hit_label}**",
        f"- 실제 추가 민원 권역 수: {clean(outcome.get('actual_additional_area_count'))}",
        f"- Top 3 내 포함 권역 수: {top3_hits if by_rank else '미확인'}", "",
        "## 6. 종합 결과", "", clean(outcome.get("notes")), "",
        "본 결과는 동일 Event ID를 기준으로 저장하여 향후 현장 실증 성능평가와 모델 개선 자료로 활용합니다.", "",
        "※ 현장 입력값을 기록한 문서이며 특정 농가·사업장 또는 원인 시설을 확정하는 문서가 아닙니다.",
    ]
    return "\n".join(lines)
