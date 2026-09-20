from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import build_source_backtrack as backtrack


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "source_backtrack"

EXPECTED_GRID_COLUMNS = [
    "event_hour", "event_id", "grid_x", "grid_y", "center_latitude", "center_longitude",
    "source_fit_score", "forward_plume_score", "prior_downwind_score", "lagged_wind_alignment",
    "travel_time_min", "rain_1h", "rain_3h", "stagnation_flag", "backtrack_uncertainty",
    "weather_source", "history_cutoff",
]
EXPECTED_NON_LIVESTOCK_COLUMNS = [
    "event_hour", "rank", "source_id", "source_type", "name", "city", "location_precision",
    "distance_km", "bearing_deg", "travel_time_min", "wind_alignment", "fit_score", "evidence_text",
]


class SourceBacktrackTest(unittest.TestCase):
    def setUp(self) -> None:
        self.hour = pd.Timestamp("2024-07-01 12:00:00")
        self.profile = {
            "by_offset": {
                offset: {"wind_direction": 270.0, "wind_speed": 2.0, "rainfall_hour": 0.0}
                for offset in range(-3, 1)
            }
        }
        self.source_lat = np.array([35.95])
        self.source_lon = np.array([126.95])
        self.weight = np.array([100.0])

    def exposure(self, receptor_lat: float, receptor_lon: float) -> float:
        result = backtrack.lagged_exposure(
            self.source_lat, self.source_lon, self.weight, receptor_lat, receptor_lon,
            self.hour, self.profile,
        )
        return float(result["exposure"][0])

    def test_aligned_wind_maximizes_score(self) -> None:
        east = self.exposure(35.95, 126.96)
        north = self.exposure(35.96, 126.95)
        self.assertGreater(east, north)

    def test_opposite_wind_is_zero(self) -> None:
        west = self.exposure(35.95, 126.94)
        self.assertAlmostEqual(west, 0.0, places=10)

    def test_distance_decay_is_monotonic(self) -> None:
        near = self.exposure(35.95, 126.96)
        far = self.exposure(35.95, 126.98)
        self.assertGreater(near, far)

    def test_source_outside_radius_is_zero(self) -> None:
        result = backtrack.lagged_exposure(
            self.source_lat, self.source_lon, self.weight, 35.95, 127.05,
            self.hour, self.profile, max_source_km=6.0,
        )
        self.assertAlmostEqual(float(result["exposure"][0]), 0.0, places=10)

    def test_alignment_is_emission_weighted_mean(self) -> None:
        detail = {
            "within": np.array([True, True, False]),
            "alignment": np.array([1.0, -1.0, 1.0]),
        }
        value = backtrack.emission_weighted_alignment(detail, np.array([3.0, 1.0, 100.0]))
        self.assertAlmostEqual(value, 0.5)

    def test_wind_lag_modes_use_different_reference_winds(self) -> None:
        profile = {
            "by_offset": {
                -3: {"wind_direction": 90.0, "wind_speed": 1.0, "rainfall_hour": 0.0},
                -2: {"wind_direction": 90.0, "wind_speed": 1.0, "rainfall_hour": 0.0},
                -1: {"wind_direction": 90.0, "wind_speed": 1.0, "rainfall_hour": 0.0},
                0: {"wind_direction": 270.0, "wind_speed": 2.0, "rainfall_hour": 0.0},
            }
        }
        fixed = backtrack.lagged_exposure(
            self.source_lat, self.source_lon, self.weight, 35.95, 126.96,
            self.hour, profile, wind_lag="fixed0",
        )
        travel = backtrack.lagged_exposure(
            self.source_lat, self.source_lon, self.weight, 35.95, 126.96,
            self.hour, profile, wind_lag="travel",
        )
        self.assertGreater(float(fixed["exposure"][0]), 0.0)
        self.assertAlmostEqual(float(travel["exposure"][0]), 0.0, places=10)

    def test_future_complaints_do_not_change_initial_input(self) -> None:
        base = pd.DataFrame({
            "datetime": [self.hour + pd.Timedelta(minutes=5)],
            "latitude": [35.95], "longitude": [126.95],
        })
        with_future = pd.concat([base, pd.DataFrame({
            "datetime": [self.hour + pd.Timedelta(minutes=45)],
            "latitude": [36.10], "longitude": [127.10],
        })], ignore_index=True)
        left = backtrack.filter_initial_complaints(base, self.hour)
        right = backtrack.filter_initial_complaints(with_future, self.hour)
        pd.testing.assert_frame_equal(left.reset_index(drop=True), right.reset_index(drop=True))


class SourceBacktrackOutputSchemaTest(unittest.TestCase):
    def test_grid_scores_has_contract_columns_in_order(self) -> None:
        scores = pd.read_csv(OUTPUT_DIR / "grid_scores.csv", encoding="utf-8-sig")
        self.assertEqual(list(scores.columns), EXPECTED_GRID_COLUMNS)
        self.assertEqual(backtrack.GRID_COLUMNS, EXPECTED_GRID_COLUMNS)

    def test_non_livestock_candidates_schema(self) -> None:
        candidates = pd.read_csv(
            OUTPUT_DIR / "non_livestock_candidates.csv", encoding="utf-8-sig",
        )
        self.assertEqual(list(candidates.columns), EXPECTED_NON_LIVESTOCK_COLUMNS)
        self.assertEqual(backtrack.NON_LIVESTOCK_COLUMNS, EXPECTED_NON_LIVESTOCK_COLUMNS)
        self.assertFalse(pd.to_datetime(candidates["event_hour"], errors="coerce").isna().any())
        self.assertTrue(set(candidates["source_type"]).issubset({
            "factory", "wastewater", "industrial_zone", "manure_plant", "waste_facility",
        }))
        self.assertTrue(pd.api.types.is_integer_dtype(candidates["rank"]))
        self.assertTrue((candidates["rank"] >= 1).all())


if __name__ == "__main__":
    unittest.main()
