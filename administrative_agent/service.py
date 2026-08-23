from __future__ import annotations

import json
from pathlib import Path

from .documents import create_briefing, create_dispatch_order, create_followup_template
from .models import ForecastResult, ResponsePackage


def build_response_package(forecast: ForecastResult) -> ResponsePackage:
    """예측 엔진의 표준 출력만 받아 행정 대응 문서 3종을 생성한다."""
    return ResponsePackage(
        event_id=forecast.event_id,
        review_required=True,
        briefing=create_briefing(forecast),
        dispatch_order=create_dispatch_order(forecast),
        followup_report_template=create_followup_template(forecast),
        forecast=forecast,
    )


def write_response_package(package: ResponsePackage, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "complaint_briefing.md").write_text(package.briefing, encoding="utf-8")
    (output_dir / "field_inspection_order.md").write_text(package.dispatch_order, encoding="utf-8")
    (output_dir / "followup_report_template.md").write_text(package.followup_report_template, encoding="utf-8")
    (output_dir / "agent_output.json").write_text(
        json.dumps(package.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def create_completed_followup(package: ResponsePackage, outcome: dict[str, object]) -> str:
    """담당자가 입력한 현장 결과를 보존해 사후 결과보고서를 완성한다."""
    rows = outcome.get("areas", [])
    by_rank = {int(row["rank"]): row for row in rows if "rank" in row}
    lines = [
        "# 사후 결과보고서", "", f"- 관련 이벤트: {package.event_id}",
        f"- 작성자: {outcome.get('author') or '미입력'}", f"- 점검 일시: {outcome.get('inspected_at') or '미입력'}", "",
        "|예측 순위|1km 권역|추가 민원|악취 감지|측정 결과|조치 내용|", "|---:|---|---|---|---|---|",
    ]
    for area in package.forecast.areas:
        row = by_rank.get(area.rank, {})
        values = [row.get("additional_complaint"), row.get("odor_detected"), row.get("measurement"), row.get("action")]
        safe = [str(value or "미입력").replace("|", "\\|").replace("\n", " ") for value in values]
        lines.append(f"|{area.rank}|{area.grid_id}|{safe[0]}|{safe[1]}|{safe[2]}|{safe[3]}|")
    notes = str(outcome.get("notes") or "미입력").replace("\n", " ")
    lines += ["", "## 담당자 의견", "", notes, "", "※ 현장 입력값을 기록한 문서이며 원인 시설 확정 문서가 아닙니다."]
    return "\n".join(lines)
