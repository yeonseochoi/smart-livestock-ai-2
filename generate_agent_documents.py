"""1km 민원 예측 CSV에서 최신 Event의 행정 대응 문서 3종을 생성한다."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

import pandas as pd

from administrative_agent.models import ForecastResult, RiskArea, SourceCandidate
from administrative_agent.service import build_response_package, write_response_package

DEFAULT_PREDICTIONS = Path("outputs/operational_grid_comparison/test_predictions.csv")
DEFAULT_METRICS = Path("outputs/operational_grid_comparison/metrics.json")
DEFAULT_OUTPUT = Path("outputs/administrative_agent")
DEFAULT_SOURCE_CANDIDATES = Path("outputs/source_backtrack/source_candidates.csv")
DEFAULT_GRID_SCORES = Path("outputs/source_backtrack/grid_scores.csv")


def _relative_scores(scores: pd.Series) -> list[int]:
    """모델 원점수 Top 3를 이벤트 내부 상대 위험도 100~60으로 변환한다."""
    if float(scores.max()) == float(scores.min()):
        return [100 - 20 * i for i in range(len(scores))]
    scaled = 60 + 40 * (scores - scores.min()) / (scores.max() - scores.min())
    return scaled.round().astype(int).tolist()


def _load_metrics(path: Path) -> dict[str, float]:
    report = json.loads(path.read_text(encoding="utf-8"))
    metrics = report["grids"]["1000"]["test"]
    return {
        "top_k_recall": float(metrics["topk_recall"]),
        "roc_auc": float(metrics["roc_auc"]),
        "pr_auc": float(metrics["pr_auc"]),
    }


def _optional_float(value) -> float | None:
    return None if value is None or pd.isna(value) else float(value)


def load_source_candidates(path: Path | None, event_hour: pd.Timestamp, limit: int = 3) -> tuple[SourceCandidate, ...]:
    """Track A `source_candidates.csv`에서 해당 Event(event_hour 키)의 상위 후보를 읽는다. 없으면 빈 튜플."""
    if path is None or not path.exists():
        return ()
    table = pd.read_csv(path, encoding="utf-8-sig")
    table["event_hour"] = pd.to_datetime(table["event_hour"])
    rows = table[table["event_hour"] == pd.Timestamp(event_hour)].sort_values("rank").head(limit)
    text = lambda row, column: None if column not in rows.columns or pd.isna(row[column]) else str(row[column])
    return tuple(
        SourceCandidate(
            rank=int(row["rank"]), name=str(row["name"]), city=text(row, "city"), species=text(row, "species"),
            location_precision=text(row, "location_precision"),
            distance_km=_optional_float(row.get("distance_km")), bearing_deg=_optional_float(row.get("bearing_deg")),
            travel_time_min=_optional_float(row.get("travel_time_min")),
            wind_alignment=_optional_float(row.get("wind_alignment")), fit_score=_optional_float(row.get("fit_score")),
            evidence_text=text(row, "evidence_text"),
        )
        for _, row in rows.iterrows()
    )


def load_backtrack_context(path: Path | None, event_hour: pd.Timestamp) -> tuple[float | None, str | None]:
    """grid_scores.csv에서 Event의 불확실성(격자 평균)과 바람 출처를 읽는다. 없으면 (None, None)."""
    if path is None or not path.exists():
        return None, None
    table = pd.read_csv(path, encoding="utf-8-sig", usecols=["event_hour", "backtrack_uncertainty", "weather_source"])
    table["event_hour"] = pd.to_datetime(table["event_hour"])
    rows = table[table["event_hour"] == pd.Timestamp(event_hour)]
    if rows.empty:
        return None, None
    source = rows["weather_source"].mode()
    return _optional_float(rows["backtrack_uncertainty"].mean()), (None if source.empty else str(source.iloc[0]))


def forecast_from_csv(
    path: Path, metrics_path: Path, event_id: str | None = None, event_time: str | None = None,
    source_candidates_path: Path | None = DEFAULT_SOURCE_CANDIDATES, grid_scores_path: Path | None = DEFAULT_GRID_SCORES,
) -> ForecastResult:
    predictions = pd.read_csv(path)
    required = {"event_id", "event_hour", "grid_x", "grid_y", "score", "grid_m"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"예측 CSV 필수 컬럼 누락: {sorted(missing)}")
    predictions["event_hour"] = pd.to_datetime(predictions["event_hour"])
    one_km = predictions[predictions["grid_m"] == 1000].copy()
    if one_km.empty:
        raise ValueError("예측 CSV에 1km 격자 결과가 없습니다.")
    selected_id = event_id or one_km.sort_values("event_hour").iloc[-1]["event_id"]
    if event_id and not (one_km["event_id"] == event_id).any() and event_time:
        matching = one_km[one_km["event_hour"] == pd.Timestamp(event_time)]
        if not matching.empty:
            selected_id = matching.iloc[0]["event_id"]
    event = one_km[one_km["event_id"] == selected_id].nlargest(3, "score").copy()
    if len(event) < 3:
        raise ValueError(f"{selected_id}에 Top 3를 만들 충분한 후보 권역이 없습니다.")
    risks = _relative_scores(event["score"])

    def optional(row: pd.Series, column: str):
        """예측 CSV에 위치 컬럼이 없던 시기의 산출물과도 호환되도록 한다."""
        if column not in event.columns or pd.isna(row[column]):
            return None
        return row[column]

    areas = tuple(
        RiskArea(
            rank=rank,
            grid_id=f"G{int(row['grid_x']):+d}:{int(row['grid_y']):+d}",
            relative_risk=risks[rank - 1],
            center_latitude=(lambda v: None if v is None else float(v))(optional(row, "center_latitude")),
            center_longitude=(lambda v: None if v is None else float(v))(optional(row, "center_longitude")),
            region_name=(lambda v: None if v is None else str(v))(optional(row, "region_name")),
        )
        for rank, (_, row) in enumerate(event.iterrows(), 1)
    )
    event_hour = event.iloc[0]["event_hour"]
    uncertainty, weather_source = load_backtrack_context(grid_scores_path, event_hour)
    return ForecastResult(
        event_id=str(selected_id), event_time=event_hour.to_pydatetime(),
        forecast_minutes=30, grid_size_m=1000, areas=areas,
        model_metrics=_load_metrics(metrics_path),
        generated_at=datetime.now(),
        source_candidates=load_source_candidates(source_candidates_path, event_hour),
        backtrack_uncertainty=uncertainty, backtrack_weather_source=weather_source,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--event-id")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-candidates", type=Path, default=DEFAULT_SOURCE_CANDIDATES,
                        help="Track A source_candidates.csv. 없으면 발생원 후보 절을 생략한다")
    parser.add_argument("--grid-scores", type=Path, default=DEFAULT_GRID_SCORES)
    args = parser.parse_args()
    package = build_response_package(forecast_from_csv(
        args.predictions, args.metrics, args.event_id,
        source_candidates_path=args.source_candidates, grid_scores_path=args.grid_scores,
    ))
    write_response_package(package, args.output)
    print(f"행정 대응 문서 생성 완료: {args.output.resolve()}")


if __name__ == "__main__":
    main()
