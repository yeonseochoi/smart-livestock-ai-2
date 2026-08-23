"""1km 민원 예측 CSV에서 최신 Event의 행정 대응 문서 3종을 생성한다."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

import pandas as pd

from administrative_agent.models import ForecastResult, RiskArea
from administrative_agent.service import build_response_package, write_response_package

DEFAULT_PREDICTIONS = Path("outputs/operational_grid_comparison/test_predictions.csv")
DEFAULT_METRICS = Path("outputs/operational_grid_comparison/metrics.json")
DEFAULT_OUTPUT = Path("outputs/administrative_agent")


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


def forecast_from_csv(
    path: Path, metrics_path: Path, event_id: str | None = None, event_time: str | None = None,
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
    areas = tuple(
        RiskArea(
            rank=rank,
            grid_id=f"G{int(row.grid_x):+d}:{int(row.grid_y):+d}",
            relative_risk=risks[rank - 1],
            center_latitude=float(row.center_latitude) if "center_latitude" in event.columns and pd.notna(row.center_latitude) else None,
            center_longitude=float(row.center_longitude) if "center_longitude" in event.columns and pd.notna(row.center_longitude) else None,
        )
        for rank, (_, row) in enumerate(event.iterrows(), 1)
    )
    return ForecastResult(
        event_id=str(selected_id), event_time=event.iloc[0]["event_hour"].to_pydatetime(),
        forecast_minutes=30, grid_size_m=1000, areas=areas,
        model_metrics=_load_metrics(metrics_path),
        generated_at=datetime.now(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--event-id")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    package = build_response_package(forecast_from_csv(args.predictions, args.metrics, args.event_id))
    write_response_package(package, args.output)
    print(f"행정 대응 문서 생성 완료: {args.output.resolve()}")


if __name__ == "__main__":
    main()
