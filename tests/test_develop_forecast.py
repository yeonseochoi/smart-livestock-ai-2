import unittest

import pandas as pd

from develop.forecast.run import SCHEMA, backtest


class ForecastContractTest(unittest.TestCase):
    def test_schema_and_training_cutoff(self):
        dates = pd.DatetimeIndex([d for y in range(2020, 2026) for d in pd.date_range(f"{y}-05-01", f"{y}-10-31")])
        frame = pd.DataFrame({"night_date": dates, "season": dates.year, "observed_complaints": (dates.day % 10 == 0).astype(int)})
        for col in ["night_temperature", "night_humidity", "night_wind_speed", "south_wind_ratio", "calm_ratio", "day_temperature", "day_rain", "previous_rain", "month", "dow", "doy"]:
            frame[col] = dates.dayofyear if col == "doy" else dates.month if col == "month" else dates.dayofweek if col == "dow" else 1.0
        result, _ = backtest(frame)
        self.assertEqual(result.columns.tolist(), SCHEMA)
        self.assertTrue((pd.to_datetime(result.train_end) < pd.to_datetime(result.night_date)).all())
        self.assertEqual(set(result.model_id), {"hgb_poisson", "calendar", "random"})


if __name__ == "__main__":
    unittest.main()
