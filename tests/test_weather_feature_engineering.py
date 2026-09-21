import math
import unittest

import pandas as pd

from fetch_kma_weather import Station, aggregate_aws_station_windows, combine_aws_records
from weather_feature_engineering import add_candidate_weather_features, add_timing_weather_features


class WeatherFeatureEngineeringTests(unittest.TestCase):
    def test_time_windows_end_at_prediction_time(self):
        times = pd.date_range("2026-01-01 00:00", periods=150, freq="min")
        frame = pd.DataFrame({
            "datetime": times,
            "wind_direction": 270.0,
            "wind_speed": 2.0,
            "temperature": 10.0,
            "humidity": 80.0,
            "is_raining": 0.0,
            "rainfall_15m": 0.0,
            "rainfall_60m": 0.0,
        })
        station = Station("AWS", 1, 127.0, 36.0, "시험")
        result = aggregate_aws_station_windows(
            frame, station, 1.0, pd.Timestamp("2026-01-01 02:00")
        )
        self.assertEqual(result["w30_samples"], 30)
        self.assertEqual(result["w60_samples"], 60)
        self.assertEqual(result["w120_samples"], 120)
        self.assertEqual(result["pre30_samples"], 30)
        self.assertEqual(result["pre60_samples"], 60)
        self.assertEqual(result["initial30_samples"], 30)

    def test_downwind_candidate_scores_above_upwind(self):
        candidates = pd.DataFrame({
            "event_id": ["E1", "E1"],
            "grid_x": [1, -1],
            "grid_y": [0, 0],
            "initial_centroid_x": [0.0, 0.0],
            "initial_centroid_y": [0.0, 0.0],
        })
        weather = {"event_id": ["E1"], "aws_w30_humidity": [80.0],
                   "aws_w30_raining_fraction": [0.0]}
        for minutes in (30, 60, 120):
            weather[f"aws_w{minutes}_downwind_east"] = [1.0]
            weather[f"aws_w{minutes}_downwind_north"] = [0.0]
            weather[f"aws_w{minutes}_wind_speed"] = [1.0]
            weather[f"aws_w{minutes}_wind_consistency"] = [1.0]
        result = add_candidate_weather_features(candidates, pd.DataFrame(weather), 1000)
        self.assertGreater(result.loc[0, "wind_alignment_w30"], 0.99)
        self.assertLess(result.loc[1, "wind_alignment_w30"], -0.99)
        self.assertGreater(result.loc[0, "transport_match_w30"],
                           result.loc[1, "transport_match_w30"])

    def test_combined_wind_vector_is_normalized(self):
        record = {
            "station_id": 1, "station_name": "시험", "distance_km": 1.0,
        }
        for minutes in (30, 60, 120):
            record.update({
                f"w{minutes}_wind_from_sin": -0.5,
                f"w{minutes}_wind_from_cos": 0.0,
                f"w{minutes}_wind_resultant_length": 0.5,
                f"w{minutes}_wind_speed": 1.0,
                f"w{minutes}_wind_speed_max": 1.0,
                f"w{minutes}_rainfall_15m": 0.0,
                f"w{minutes}_rainfall_60m": 0.0,
                f"w{minutes}_raining_fraction": 0.0,
                f"w{minutes}_humidity": 80.0,
                f"w{minutes}_temperature": 20.0,
                f"w{minutes}_samples": float(minutes),
            })
        result = combine_aws_records([record])
        self.assertTrue(math.isclose(result["aws_w30_downwind_east"], 1.0))
        self.assertTrue(math.isclose(result["aws_w30_downwind_north"], 0.0))

    def test_pre_event_and_initial_windows_remain_separate(self):
        candidates = pd.DataFrame({
            "event_id": ["E1"], "grid_x": [1], "grid_y": [0],
            "initial_centroid_x": [0.0], "initial_centroid_y": [0.0],
        })
        weather = {"event_id": ["E1"]}
        directions = {"pre30": (-1.0, 0.0), "pre60": (-1.0, 0.0), "initial30": (1.0, 0.0)}
        for prefix, (east, north) in directions.items():
            weather[f"aws_{prefix}_downwind_east"] = [east]
            weather[f"aws_{prefix}_downwind_north"] = [north]
            weather[f"aws_{prefix}_wind_speed"] = [1.0]
            weather[f"aws_{prefix}_wind_consistency"] = [1.0]
            weather[f"aws_{prefix}_humidity"] = [80.0]
            weather[f"aws_{prefix}_raining_fraction"] = [0.0]
        result = add_timing_weather_features(candidates, pd.DataFrame(weather), 1000)
        self.assertLess(result.loc[0, "timing_alignment_pre30"], -0.99)
        self.assertGreater(result.loc[0, "timing_alignment_initial30"], 0.99)
        self.assertLess(result.loc[0, "timing_direction_match_pre30_initial30"], -0.99)


if __name__ == "__main__":
    unittest.main()
