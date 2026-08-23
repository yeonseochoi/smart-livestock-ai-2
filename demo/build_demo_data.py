"""Build the browser demo dataset from validated model outputs."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import build_odor_ai_mvp as odor


OUTPUT = Path(__file__).with_name("demo-data.js")


def center(gx: int, gy: int, meta: dict, grid_m: int) -> list[float]:
    lat = meta["lat0"] + (gy + 0.5) * grid_m / 110_540
    lon = meta["lon0"] + (gx + 0.5) * grid_m / (
        111_320 * math.cos(math.radians(meta["lat0"]))
    )
    return [round(lat, 6), round(lon, 6)]


def normalized_score(values: pd.Series) -> np.ndarray:
    ranks = values.rank(method="average", pct=True).to_numpy(float)
    return np.clip((ranks - 0.45) / 0.55, 0, 1)


def top3_relative_score(values: pd.Series) -> np.ndarray:
    """Top 3 내부의 실제 모델 점수 간격을 0.35~1.0 상대 위험도로 변환한다."""
    lo, hi = float(values.min()), float(values.max())
    if math.isclose(lo, hi):
        return np.full(len(values), 0.65)
    return 0.35 + 0.65 * (values.to_numpy(float) - lo) / (hi - lo)


def weather_value(row: pd.Series, aws: str, asos: str) -> float | None:
    value = row.get(aws)
    if pd.isna(value):
        value = row.get(asos)
    return None if pd.isna(value) else round(float(value), 1)


def main() -> None:
    complaints, _, _, _, _ = odor.load_inputs()
    complaints, meta = odor.add_grid_columns(complaints)
    events, _ = odor.build_bounded_events(complaints)
    pred1000 = pd.read_csv(
        ROOT / "outputs/early_prediction_sensitivity/selected_test_predictions.csv",
        encoding="utf-8-sig", parse_dates=["event_hour"],
    )
    weather = pd.read_csv(
        ROOT / "outputs/weather_integration/event_weather_features.csv",
        encoding="utf-8-sig", parse_dates=["event_hour"],
    ).set_index("event_id")

    common = sorted(set(events["event_id"]) & set(pred1000["event_id"]))
    payload = {"meta": {"city": "익산시", "generated": "2026-08-18"}, "events": []}
    for event_id in common:
        p1000 = pred1000[pred1000["event_id"] == event_id].copy()
        event_hour = pd.Timestamp(p1000["event_hour"].iloc[0])
        raw = events[events["event_id"] == event_id].copy()
        initial = raw[raw["datetime"] < event_hour + pd.Timedelta(minutes=30)]
        future = raw[raw["datetime"] >= event_hour + pd.Timedelta(minutes=30)]
        p1000["display_score"] = normalized_score(p1000["model_score"])
        top1000 = p1000.nlargest(min(12, len(p1000)), "model_score")
        weather_row = weather.loc[event_id]
        rain = weather_value(weather_row, "aws_rainfall_60m", "asos_rainfall_hour")
        wind_speed = weather_value(weather_row, "aws_wind_speed", "asos_wind_speed")
        humidity = weather_value(weather_row, "aws_humidity", "asos_humidity")
        east = float(weather_row.get("aws_downwind_east", 0) or 0)
        north = float(weather_row.get("aws_downwind_north", 0) or 0)
        direction = (math.degrees(math.atan2(east, north)) + 360) % 360
        payload["events"].append({
            "id": event_id,
            "hour": event_hour.strftime("%Y-%m-%d %H:%M"),
            "initialCount": int(len(initial)),
            "futureCount": int(len(future)),
            "weather": {"windSpeed": wind_speed, "windDirection": round(direction),
                        "humidity": humidity, "rainfall": rain},
            "reports": [[round(float(r.latitude), 6), round(float(r.longitude), 6),
                         round(float(r.intensity), 1)] for r in initial.itertuples()],
            "broad": [{"center": center(int(r.grid_x), int(r.grid_y), meta, 1000),
                       "score": round(float(r.display_score), 3), "actual": bool(r.target)}
                      for r in top1000.itertuples()],
        })

    OUTPUT.write_text(
        "window.DEMO_DATA = " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + ";\n",
        encoding="utf-8",
    )
    print(f"events={len(payload['events'])} output={OUTPUT}")


if __name__ == "__main__":
    main()
