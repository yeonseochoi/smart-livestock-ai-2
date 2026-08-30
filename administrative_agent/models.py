from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class RiskArea:
    rank: int
    grid_id: str
    relative_risk: int
    center_latitude: float | None = None
    center_longitude: float | None = None
    initial_complaints: int | None = None
    region_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ForecastResult:
    event_id: str
    event_time: datetime
    forecast_minutes: int
    grid_size_m: int
    areas: tuple[RiskArea, ...]
    model_metrics: dict[str, float]
    generated_at: datetime
    initial_complaint_count: int | None = None
    initial_grid_count: int | None = None
    initial_intensity_average: float | None = None
    initial_intensity_maximum: float | None = None
    weather: dict[str, float | int | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.grid_size_m != 1000:
            raise ValueError("행정 대응 Agent의 운영 입력은 1km 격자여야 합니다.")
        if len(self.areas) != 3:
            raise ValueError("행정 대응 Agent에는 정확히 Top 3 권역을 전달해야 합니다.")
        if [area.rank for area in self.areas] != [1, 2, 3]:
            raise ValueError("권역 순위는 1, 2, 3 순서여야 합니다.")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["event_time"] = self.event_time.isoformat()
        result["generated_at"] = self.generated_at.isoformat()
        return result


@dataclass(frozen=True)
class ResponsePackage:
    event_id: str
    review_required: bool
    briefing: str
    dispatch_order: str
    followup_report_template: str
    response_guide: str
    forecast: ForecastResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "review_required": self.review_required,
            "forecast": self.forecast.to_dict(),
            "documents": {
                "briefing": self.briefing,
                "dispatch_order": self.dispatch_order,
                "followup_report_template": self.followup_report_template,
                "response_guide": self.response_guide,
            },
        }
