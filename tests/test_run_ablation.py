"""run_ablation.py의 어댑터·기상 특징이 계약(키 결합, 누수 금지)을 지키는지 확인한다."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_ablation as ablation  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "grid_scores_sample.csv"


class AblationAdapterTest(unittest.TestCase):
    def test_backtrack_join_uses_event_hour_key_and_leaves_nan(self) -> None:
        scores = pd.read_csv(FIXTURE, parse_dates=["event_hour", "history_cutoff"], encoding="utf-8-sig")
        base = scores[["event_hour", "grid_x", "grid_y"]].copy()
        base["event_id"] = "OTHER-ID"  # event_id가 달라도 결합되어야 한다.
        extra = base.iloc[[0]].copy()
        extra["grid_x"] = 999  # 기준표에 없는 칸은 NaN으로 남아야 한다.
        base = pd.concat([base, extra], ignore_index=True)
        merged, info = ablation.add_backtrack_features(base, scores)
        self.assertEqual(info["matched_rows"], len(scores))
        self.assertTrue(np.isnan(merged.loc[len(merged) - 1, "source_fit_score"]))
        self.assertEqual(list(merged.columns[-len(ablation.BACKTRACK_FEATURES):]), ablation.BACKTRACK_FEATURES)

    def test_weather_window_never_looks_past_event_hour(self) -> None:
        hours = pd.date_range("2024-07-28 15:00", "2024-07-29 06:00", freq="h")
        asos = pd.DataFrame({
            "datetime": hours, "station_id": 146, "station_latitude": 35.84, "station_longitude": 127.12,
            "wind_direction": 270.0, "wind_speed": 2.0, "rainfall_hour": 0.0,
        })
        # Event 이후 시각에만 큰 강수를 넣는다. 특징에 잡히면 누수다.
        asos.loc[asos["datetime"] > "2024-07-28 23:00", "rainfall_hour"] = 50.0
        asos = _with_uv(asos)
        events = pd.DataFrame({
            "event_hour": [pd.Timestamp("2024-07-28 23:00")], "event_id": ["EVT-X"],
            "centroid_latitude": [35.95], "centroid_longitude": [126.98],
        })
        weather = ablation.event_weather(events, asos)
        self.assertEqual(weather.loc[0, "asos_rain_3h"], 0.0)
        # 서풍(270°에서 불어옴) → 동쪽(90°)으로 분다.
        self.assertAlmostEqual(weather.loc[0, "asos_blowing_to_deg"], 90.0, places=3)

    def test_downwind_alignment_sign(self) -> None:
        meta = {"lat0": 35.9566125, "lon0": 126.98382, "grid_m": 1000}
        data = pd.DataFrame({
            "event_hour": [pd.Timestamp("2024-07-28 23:00")] * 2, "event_id": ["EVT-X"] * 2,
            "grid_x": [3, -3], "grid_y": [0, 0],
        })
        events = pd.DataFrame({
            "event_hour": [pd.Timestamp("2024-07-28 23:00")], "event_id": ["EVT-X"],
            "centroid_latitude": [meta["lat0"]], "centroid_longitude": [meta["lon0"]],
        })
        weather = pd.DataFrame({
            "event_hour": [pd.Timestamp("2024-07-28 23:00")], "event_id": ["EVT-X"],
            "asos_blowing_to_deg": [90.0], "asos_wind_speed_3h": [2.0], "asos_rain_3h": [0.0],
            "asos_direction_sd": [5.0], "asos_stagnation_flag": [0], "asos_source": ["asos"],
        })
        result = ablation.add_weather_features(data, events, weather, meta)
        self.assertGreater(result.loc[0, "asos_downwind_alignment"], 0.9)   # 동쪽 칸: 바람 아래
        self.assertLess(result.loc[1, "asos_downwind_alignment"], -0.9)     # 서쪽 칸: 바람 위


def _with_uv(asos: pd.DataFrame) -> pd.DataFrame:
    rad = np.radians(asos["wind_direction"])
    asos = asos.copy()
    asos["u"] = -asos["wind_speed"] * np.sin(rad)
    asos["v"] = -asos["wind_speed"] * np.cos(rad)
    return asos


if __name__ == "__main__":
    unittest.main()
