import unittest

import pandas as pd

from odor_service import common
from odor_service.standby.run import clusters


class StandbyOutputsTest(unittest.TestCase):
    def test_farms_schema_and_missing_gimje_values(self):
        farms = pd.read_parquet(common.OUTPUT_DIR / "data" / "farms.parquet")
        required = {"farm_id", "city", "name", "address", "lat", "lon", "coord_precision", "species_raw", "species", "heads", "eq", "eq_assumed", "status"}
        self.assertEqual(required, set(farms.columns))
        self.assertFalse(farms.farm_id.duplicated().any())
        self.assertEqual(2778, len(farms))
        self.assertTrue(farms[farms.city == "김제"][["heads", "status", "eq"]].isna().all().all())
        self.assertTrue(set(farms.coord_precision) <= {"exact", "approx", "none"})
        self.assertEqual(farms.eq_assumed.dtype, bool)

    def test_weather_schema(self):
        weather = pd.read_parquet(common.OUTPUT_DIR / "data" / "weather_hourly.parquet")
        required = {"datetime", "station_id", "station_name", "source", "station_latitude", "station_longitude", "wind_direction", "wind_speed", "temperature", "humidity", "rainfall_hour"}
        self.assertEqual(required, set(weather.columns))
        self.assertEqual({"ASOS", "AWS"}, set(weather.source))
        self.assertFalse(weather[["source", "station_id", "datetime"]].duplicated().any())

    def test_standby_schema_and_grades(self):
        path = common.OUTPUT_DIR / "standby"
        points = pd.read_csv(path / "standby_points.csv")
        years = pd.read_csv(path / "lift_by_year.csv")
        required = {"point_id", "cluster_id", "grade", "city", "lat", "lon", "cluster_lat", "cluster_lon", "n_farms", "eq_sum", "lift_all", "lift_ci_low", "lift_ci_high", "lift_recent3", "wind_source"}
        self.assertEqual(required, set(points.columns))
        self.assertEqual({"cluster_id", "year", "lift", "n_complaints", "wind_source"}, set(years.columns))
        self.assertEqual({"ASOS", "AWS"}, set(points.wind_source))
        self.assertFalse(points.point_id.duplicated().any())
        self.assertFalse(years[["cluster_id", "year", "wind_source"]].duplicated().any())
        for row in points.itertuples(index=False):
            expected = "주" if row.lift_ci_low > 1 and row.lift_recent3 > 1 else "관찰" if row.lift_all > 1 else "제외"
            self.assertEqual(expected, row.grade)
        farms = pd.read_parquet(common.OUTPUT_DIR / "data" / "farms.parquet")
        self.assertEqual(set(clusters(farms).cluster_id), set(points.cluster_id))


if __name__ == "__main__":
    unittest.main()
