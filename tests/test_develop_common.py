import unittest

import numpy as np
import pandas as pd

from develop import common as c


class DevelopCommonTest(unittest.TestCase):
    def test_bearing_cardinal(self):
        self.assertAlmostEqual(float(c.bearing_deg(35.9, 127.0, 36.0, 127.0)), 0.0, places=6)
        self.assertAlmostEqual(float(c.bearing_deg(35.9, 127.0, 35.9, 127.1)), 90.0, places=6)
        self.assertAlmostEqual(float(c.bearing_deg(35.9, 127.0, 35.8, 127.0)), 180.0, places=6)

    def test_angle_diff_wraps(self):
        self.assertEqual(float(c.angle_diff(350, 10)), 20.0)
        self.assertEqual(float(c.angle_diff(0, 180)), 180.0)

    def test_north_wind_blows_south(self):
        dx, dy = c.downwind_vector(0)
        self.assertAlmostEqual(float(dx), 0.0, places=9)
        self.assertAlmostEqual(float(dy), -1.0, places=9)

    def test_night_date_and_window(self):
        ts = pd.Series(pd.to_datetime(["2025-07-01 23:00", "2025-07-02 03:00", "2025-07-02 05:00", "2025-07-02 12:00"]))
        nd = c.night_date(ts)
        self.assertTrue((nd.iloc[:2] == pd.Timestamp("2025-07-01")).all())
        self.assertEqual(c.in_night_window(ts).tolist(), [True, True, False, False])

    def test_distance_one_degree_lat(self):
        self.assertAlmostEqual(float(c.distance_km(35.0, 127.0, 36.0, 127.0)), c.KM_PER_DEG_LAT, places=6)


if __name__ == "__main__":
    unittest.main()
