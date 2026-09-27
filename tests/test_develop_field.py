import unittest

import numpy as np
import pandas as pd

from develop import common as c
from develop.field import engine as e


def _farms():
    # 담당자(35.90, 127.00) 기준: A 북쪽 1 km, B 북쪽 3 km, C 남쪽 1 km, D 동쪽 1 km, E 북쪽 6 km
    dlat = 1.0 / c.KM_PER_DEG_LAT
    dlon = 1.0 / c.KM_PER_DEG_LON
    return pd.DataFrame({
        "farm_id": ["A", "B", "C", "D", "E"],
        "name": ["A농장", "B농장", "C농장", "D농장", "E농장"],
        "lat": [35.90 + dlat, 35.90 + 3 * dlat, 35.90 - dlat, 35.90, 35.90 + 6 * dlat],
        "lon": [127.00, 127.00, 127.00, 127.00 + dlon, 127.00],
        "species": ["돼지", "돼지", "소", "가금", "돼지"],
        "eq": [5000.0, 8000.0, 100.0, 50.0, 9000.0],
        "coord_precision": ["exact", "approx", "exact", "exact", "exact"],
        "status": ["정상"] * 5,
    })


class FieldEngineTest(unittest.TestCase):
    def test_zero_scale_preserves_existing_score(self):
        r = e.score_candidates(_farms(), 35.90, 127.00, e.Wind(0.0, 1.5, 20.0), weights=e.Weights(scale=0.0))
        s = r.candidates
        expected = s[["s_wind", "s_dist", "s_multi", "s_hist"]].sum(axis=1)
        np.testing.assert_allclose(s["score"], expected)
        self.assertTrue((s["s_scale"] == 0.0).all())

    def test_missing_eq_uses_candidate_median(self):
        farms = _farms()
        farms.loc[farms["farm_id"] == "A", "eq"] = np.nan
        r = e.score_candidates(farms, 35.90, 127.00, e.Wind(0.0, 1.5, 20.0), weights=e.Weights(scale=15.0))
        s = r.candidates.set_index("farm_id")
        candidate_eq = s["eq"].dropna()
        expected = 15.0 * np.log1p(candidate_eq.median()) / np.log1p(candidate_eq.max())
        self.assertAlmostEqual(float(s.loc["A", "s_scale"]), expected)
        self.assertEqual(e.to_cards(r)[0]["components"]["규모"], round(float(r.candidates.iloc[0]["s_scale"]), 1))

    def test_all_missing_eq_has_zero_scale_score(self):
        farms = _farms()
        farms["eq"] = np.nan
        r = e.score_candidates(farms, 35.90, 127.00, e.Wind(0.0, 1.5, 20.0), weights=e.Weights(scale=15.0))
        self.assertTrue((r.candidates["s_scale"] == 0.0).all())

    def test_north_wind_ranks_north_farms(self):
        r = e.score_candidates(_farms(), 35.90, 127.00, e.Wind(0.0, 1.5, 20.0))
        ids = r.candidates["farm_id"].tolist()
        self.assertEqual(ids[0], "A")
        self.assertNotIn("C", ids)  # 풍하(남쪽)는 제외
        self.assertNotIn("E", ids)  # 4 km 밖 제외
        self.assertEqual(r.mode, "sector")

    def test_score_components_bounded(self):
        r = e.score_candidates(_farms(), 35.90, 127.00, e.Wind(0.0, 1.5, 20.0))
        s = r.candidates
        self.assertTrue((s["score"] <= 100.0 + 1e-9).all())
        self.assertAlmostEqual(float(s.loc[0, "s_wind"]), 40.0, places=6)
        self.assertAlmostEqual(float(s.loc[0, "s_dist"]), 25.0 * np.exp(-0.5), places=4)

    def test_multi_observation_counts(self):
        obs = [e.Observation(35.90, 127.00), e.Observation(35.90, 127.00 + 0.3 / c.KM_PER_DEG_LON), e.Observation(35.80, 127.00, smell=False)]
        r = e.score_candidates(_farms(), 35.90, 127.00, e.Wind(0.0, 1.5, 20.0), observations=obs)
        a = r.candidates.set_index("farm_id").loc["A"]
        self.assertEqual(int(a["multi_n"]), 2)
        self.assertEqual(int(a["multi_hit"]), 2)
        self.assertAlmostEqual(float(a["s_multi"]), 20.0)

    def test_weak_wind_flags_low_confidence(self):
        r = e.score_candidates(_farms(), 35.90, 127.00, e.Wind(0.0, 0.3, 20.0))
        self.assertEqual(r.mode, "gradient")
        self.assertEqual(r.confidence, "매우 낮음")
        self.assertTrue(r.notes)

    def test_cards_have_disclaimer_and_no_probability(self):
        r = e.score_candidates(_farms(), 35.90, 127.00, e.Wind(0.0, 1.5, 20.0))
        cards = e.to_cards(r)
        self.assertTrue(all(card["disclaimer"] == e.DISCLAIMER for card in cards))
        self.assertTrue(cards[[x["farm_id"] for x in cards].index("B")]["coord_warning"])
        self.assertNotIn("%", " ".join(card["reason"] for card in cards))

    def test_history_counts_downwind_only(self):
        f = _farms()
        # 북풍일 때 A농장 남쪽 5 km 민원 → A 풍하. 같은 월, 비슷한 풍향.
        comp = pd.DataFrame({
            "datetime": pd.to_datetime(["2024-07-01 23:00", "2024-07-02 23:00", "2024-08-01 23:00"]),
            "latitude": [35.90 - 4 / c.KM_PER_DEG_LAT] * 3,
            "longitude": [127.00] * 3,
            "wind_direction": [5.0, 350.0, 0.0],
            "wind_speed": [1.0, 1.0, 1.0],
        })
        n = e.history_counts(f, comp, e.Wind(0.0, 1.0), month=7)
        self.assertEqual(n[0], 2.0)  # 8월 민원은 제외
        self.assertEqual(n[2], 2.0)  # C도 같은 선상 풍상
        self.assertEqual(n[4], 2.0)  # E 10 km 떨어져도 15 km 이내


if __name__ == "__main__":
    unittest.main()
