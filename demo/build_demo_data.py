"""운영 격자 비교 결과로 브라우저 데모 데이터를 만든다.

예측은 `outputs/operational_grid_comparison/test_predictions.csv`의 1km 결과만
사용한다. 이 파일은 제출 지표를 만드는 바로 그 산출물이므로 화면과 지표의 출처가
같아지고, 격자 크기와 입력·예측 시간창이 코드로 고정되어 화면 라벨과 어긋나지 않는다.

Event를 맞출 때 `event_id`를 쓰지 않는다. `event_id`는 시각순 일련번호라 민원
데이터가 바뀌면 같은 번호가 다른 사건을 가리킨다. 실행 간 값이 변하지 않는
`event_hour`를 기준으로 결합한다.
"""
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
PREDICTIONS = ROOT / "outputs/operational_grid_comparison/test_predictions.csv"
WEATHER = ROOT / "outputs/weather_integration/event_weather_features.csv"
GRID_M = 1000
INPUT_MINUTES = 30
MAX_CELLS = 12


def normalized_score(values: pd.Series) -> np.ndarray:
    ranks = values.rank(method="average", pct=True).to_numpy(float)
    return np.clip((ranks - 0.45) / 0.55, 0, 1)


def load_weather() -> pd.DataFrame:
    """기상은 현장 참고 정보이므로 없거나 일부만 있어도 진행한다."""
    if not WEATHER.exists():
        return pd.DataFrame()
    frame = pd.read_csv(WEATHER, encoding="utf-8-sig", parse_dates=["event_hour"])
    return frame.drop_duplicates("event_hour").set_index("event_hour")


def weather_value(row: pd.Series | None, aws: str, asos: str) -> float | None:
    if row is None:
        return None
    value = row.get(aws)
    if value is None or pd.isna(value):
        value = row.get(asos)
    return None if value is None or pd.isna(value) else round(float(value), 1)


def wind_direction(row: pd.Series | None) -> int | None:
    """풍하 방향 벡터를 방위각으로 되돌린다."""
    if row is None:
        return None
    east, north = row.get("aws_downwind_east"), row.get("aws_downwind_north")
    if east is None or north is None or pd.isna(east) or pd.isna(north):
        return None
    return round((math.degrees(math.atan2(float(east), float(north))) + 360) % 360)


def main() -> None:
    complaints, _, _, _, _ = odor.load_inputs()
    complaints, _ = odor.add_grid_columns(complaints)
    events, _ = odor.build_bounded_events(complaints)

    predictions = pd.read_csv(PREDICTIONS, encoding="utf-8-sig", parse_dates=["event_hour"])
    predictions = predictions[predictions["grid_m"] == GRID_M].copy()
    if predictions.empty:
        raise ValueError(f"예측 CSV에 {GRID_M}m 격자 결과가 없습니다.")
    missing = {"center_latitude", "center_longitude"} - set(predictions.columns)
    if missing:
        raise ValueError(
            f"예측 CSV에 위치 컬럼이 없습니다: {sorted(missing)}. "
            "compare_operational_grid_sizes.py를 다시 실행하세요."
        )
    weather = load_weather()

    by_hour = {pd.Timestamp(hour): group for hour, group in events.groupby("event_hour")}
    payload = {"meta": {"city": "익산시", "gridMeters": GRID_M, "inputMinutes": INPUT_MINUTES}, "events": []}
    skipped = 0
    for event_hour, cells in predictions.groupby("event_hour"):
        raw = by_hour.get(pd.Timestamp(event_hour))
        if raw is None:
            skipped += 1
            continue
        boundary = pd.Timestamp(event_hour) + pd.Timedelta(minutes=INPUT_MINUTES)
        initial = raw[raw["datetime"] < boundary]
        future = raw[raw["datetime"] >= boundary]

        cells = cells.copy()
        cells["display_score"] = normalized_score(cells["score"])
        top = cells.nlargest(min(MAX_CELLS, len(cells)), "score")
        row = weather.loc[event_hour] if event_hour in weather.index else None
        payload["events"].append({
            "id": str(cells["event_id"].iloc[0]),
            "hour": pd.Timestamp(event_hour).strftime("%Y-%m-%d %H:%M"),
            "initialCount": int(len(initial)),
            "futureCount": int(len(future)),
            "weather": {
                "windSpeed": weather_value(row, "aws_wind_speed", "asos_wind_speed"),
                "windDirection": wind_direction(row),
                "humidity": weather_value(row, "aws_humidity", "asos_humidity"),
                "rainfall": weather_value(row, "aws_rainfall_60m", "asos_rainfall_hour"),
            },
            "reports": [[round(float(r.latitude), 6), round(float(r.longitude), 6),
                         round(float(r.intensity), 1)] for r in initial.itertuples()],
            "broad": [{"center": [round(float(r.center_latitude), 6), round(float(r.center_longitude), 6)],
                       "region": None if pd.isna(getattr(r, "region_name", None)) else str(r.region_name),
                       "score": round(float(r.display_score), 3), "actual": bool(r.target)}
                      for r in top.itertuples()],
        })

    payload["events"].sort(key=lambda item: item["hour"])
    OUTPUT.write_text(
        "window.DEMO_DATA = " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + ";\n",
        encoding="utf-8",
    )
    with_weather = sum(1 for e in payload["events"] if e["weather"]["windSpeed"] is not None)
    with_future = sum(1 for e in payload["events"] if e["futureCount"] > 0)
    print(f"events={len(payload['events'])} 이후신고있음={with_future} 기상있음={with_weather} "
          f"민원매칭실패={skipped} output={OUTPUT}")


if __name__ == "__main__":
    main()
