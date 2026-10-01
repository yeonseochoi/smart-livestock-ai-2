import unittest

import numpy as np
import pandas as pd

from odor_service import common as c
from odor_service.field import engine as e


class ClusterNoteTest(unittest.TestCase):
    def test_cluster_note_counts_only_returned_representatives(self):
        farms = pd.DataFrame({"farm_id": ["west", "east"], "name": ["A농장", "B농장"],
                              "lat": [35.910, 35.910], "lon": [126.900, 127.000], "status": ["정상", "정상"]})
        locations = e.complaint_locations([(35.900, 126.900), (35.901, 126.900), (35.900, 127.000)])
        result = e.score_complaint_candidates(farms, locations, e.Wind(0.0, 1.5))
        self.assertEqual(result.clusters, 2)
        self.assertEqual(result.candidates["farm_id"].tolist(), ["west", "east"])
        note = next(n for n in result.notes if "무리" in n)
        self.assertIn("대표가 포함된 무리 2개", note)
        self.assertIn("보장하지 않음", note)
        fallback = e.score_complaint_candidates(farms, locations, e.Wind(180.0, 1.5))
        self.assertEqual(len(fallback.candidates), 2)
        self.assertTrue((fallback.candidates["tier"] == e.TIER_FALLBACK).all())


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
        self.assertAlmostEqual(float(s.loc[0, "s_wind"]), 20.0, places=6)
        self.assertAlmostEqual(float(s.loc[0, "s_dist"]), 20.0 * np.exp(-0.5), places=4)

    def test_multi_observation_counts(self):
        obs = [e.Observation(35.90, 127.00), e.Observation(35.90, 127.00 + 0.3 / c.KM_PER_DEG_LON), e.Observation(35.80, 127.00, smell=False)]
        r = e.score_candidates(_farms(), 35.90, 127.00, e.Wind(0.0, 1.5, 20.0), observations=obs)
        a = r.candidates.set_index("farm_id").loc["A"]
        self.assertEqual(int(a["multi_n"]), 2)
        self.assertEqual(int(a["multi_hit"]), 2)
        self.assertAlmostEqual(float(a["s_multi"]), 40.0)

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
        self.assertIn("먼저 확인", cards[0]["selection_summary"])
        self.assertGreaterEqual(len(cards[0]["evidence"]), 4)
        self.assertIn("농가 경계", cards[0]["next_action"])

    def test_explanation_states_when_cross_check_is_unavailable(self):
        r = e.score_candidates(_farms(), 35.90, 127.00, e.Wind(0.0, 1.5, 20.0), observations=[])
        card = e.to_cards(r)[0]
        self.assertTrue(any("2곳 미만" in item for item in card["evidence"]))
        self.assertFalse(any("민원 지점 0곳 중" in item for item in card["evidence"]))

    def test_same_coordinate_candidates_become_one_visit_point(self):
        farms = _farms()
        duplicate = farms.iloc[[0]].copy()
        duplicate["farm_id"] = "A-2"
        duplicate["name"] = "A농장 중복등록"
        farms = pd.concat([farms, duplicate], ignore_index=True)
        result = e.score_candidates(farms, 35.90, 127.00, e.Wind(0.0, 1.5, 20.0), top_k=5)
        cards = e.to_cards(result)
        self.assertEqual([1, 2, 3], [card["rank"] for card in cards])
        self.assertEqual(2, cards[0]["merged_count"])
        self.assertIn("A-2", cards[0]["merged_farm_ids"])
        self.assertIn("A농장 중복등록", cards[0]["merged_names"])

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

    def test_same_name_far_apart_preserved_nearby_registration_merged(self):
        farms = _farms().iloc[[0, 1]].copy()
        farms["name"] = "동명농장"  # 2 km 떨어져 있어 둘 다 보존한다.
        nearby = farms.iloc[[0]].copy()
        nearby["farm_id"] = "A-near"
        nearby["lat"] += 0.2 / c.KM_PER_DEG_LAT
        farms = pd.concat([farms, nearby], ignore_index=True)
        cards = e.to_cards(e.score_candidates(farms, 35.90, 127.00, e.Wind(0, 1.5)))
        self.assertEqual(len(cards), 2)
        self.assertEqual([c["rank"] for c in cards], [1, 2])
        self.assertEqual(cards[0]["merged_count"], 2)
        self.assertIn("A-near", cards[0]["merged_farm_ids"])
        self.assertEqual(cards[1]["farm_id"], "B")



DLAT = 1.0 / c.KM_PER_DEG_LAT
DLON = 1.0 / c.KM_PER_DEG_LON


def _farm_table(rows):
    """rows: (farm_id, 북쪽 km, 동쪽 km) — 기준점 (35.90, 127.00)."""
    return pd.DataFrame({
        "farm_id": [r[0] for r in rows], "name": [f"{r[0]}농장" for r in rows],
        "lat": [35.90 + r[1] * DLAT for r in rows], "lon": [127.00 + r[2] * DLON for r in rows],
        "status": ["정상"] * len(rows), "coord_precision": ["exact"] * len(rows),
    })


def _at(north_km, east_km):
    return (35.90 + north_km * DLAT, 127.00 + east_km * DLON)


class ComplaintAnchoredTest(unittest.TestCase):
    """민원 지점 기준 후보 선정(A 절충안)."""

    def test_repeated_same_place_has_one_vote(self):
        locations = e.complaint_locations([_at(0, 0)] * 10 + [_at(1, 0)])
        self.assertEqual(len(locations), 2)
        self.assertEqual(sorted(x.n_reports for x in locations), [1, 10])

    def test_split_reports_keep_candidates_for_each_group(self):
        # 남북으로 22 km 떨어진 두 민원 무리. 평균 중심 하나로는 두 농가 모두 4 km 밖이다.
        farms = _farm_table([("south", 1, 0), ("north", 23, 0)])
        locations = e.complaint_locations([_at(0, 0), _at(0, 0.1), _at(22, 0), _at(22, 0.1)])
        self.assertEqual(len({x.cluster for x in locations}), 2)
        r = e.score_complaint_candidates(farms, locations, e.Wind(0.0, 2.0), sensitivity=False)
        self.assertEqual(set(r.candidates["farm_id"]), {"south", "north"})
        self.assertEqual(r.selection, e.TIER_CROSS)
        self.assertEqual(r.clusters, 2)

    def test_cross_support_precedes_single_location_fillers(self):
        farms = _farm_table([("both", 2, 0.5), ("east_only", 2, 2.8)])
        locations = e.complaint_locations([_at(0, 0), _at(0, 1)])
        r = e.score_complaint_candidates(farms, locations, e.Wind(0.0, 2.0), sensitivity=False)
        self.assertEqual(r.candidates["farm_id"].tolist(), ["both", "east_only"])
        self.assertEqual(r.candidates.loc[0, "support_n"], 2)
        self.assertEqual(r.candidates.loc[1, "support_n"], 1)
        self.assertAlmostEqual(float(r.candidates.loc[0, "s_multi"]), 40.0)

    def test_top_three_is_filled_in_rank_order(self):
        farms = _farm_table([("cross", 2, 0.5), ("single", 2, 2.8), ("next", -2, 0), ("last", -3, 1)])
        locations = e.complaint_locations([_at(0, 0), _at(0, 1)])
        r = e.score_complaint_candidates(farms, locations, e.Wind(0.0, 2.0), top_k=3, sensitivity=False)
        self.assertEqual(len(r.candidates), 3)
        self.assertEqual(r.candidates["farm_id"].tolist()[:2], ["cross", "single"])
        self.assertEqual(r.candidates["tier"].tolist(), [e.TIER_CROSS, e.TIER_SINGLE, e.TIER_FALLBACK])

    def test_single_location_candidates_are_followed_by_ranked_fillers(self):
        farms = _farm_table([("east_only", 2, 2.8), ("downwind", -2, 0)])
        locations = e.complaint_locations([_at(0, 0), _at(0, 1)])
        r = e.score_complaint_candidates(farms, locations, e.Wind(0.0, 2.0), sensitivity=False)
        self.assertEqual(r.selection, e.TIER_SINGLE)
        self.assertEqual(r.candidates["farm_id"].tolist(), ["east_only", "downwind"])
        self.assertTrue(any("한 위치 기준" in note for note in r.notes))
        card = e.to_cards(r)[0]
        self.assertEqual(card["tier"], e.TIER_SINGLE)
        self.assertIn("근거가 한 민원 위치뿐", card["next_action"])
        self.assertEqual(e.to_cards(r)[1]["tier"], e.TIER_FALLBACK)

    def test_travel_distance_changes_route_not_ranking(self):
        farms = _farm_table([("near", 1, 0), ("far", 3.5, 0.3)])
        locations = e.complaint_locations([_at(0, 0), _at(0, 0.2)])
        a = e.score_complaint_candidates(farms, locations, e.Wind(0.0, 2.0), start=_at(-10, 0), sensitivity=False)
        b = e.score_complaint_candidates(farms, locations, e.Wind(0.0, 2.0), start=_at(15, 0), sensitivity=False)
        self.assertEqual(a.candidates["farm_id"].tolist(), b.candidates["farm_id"].tolist())
        np.testing.assert_allclose(a.candidates["score"], b.candidates["score"])
        order_a = dict(zip(a.candidates["farm_id"], a.candidates["visit_order"]))
        order_b = dict(zip(b.candidates["farm_id"], b.candidates["visit_order"]))
        self.assertEqual(order_a, {"near": 1, "far": 2})
        self.assertEqual(order_b, {"far": 1, "near": 2})
        self.assertGreater(b.route_km, 0)

    def test_small_complaint_group_keeps_a_representative(self):
        # 큰 무리 가까이에 강한 후보 6곳, 20 km 떨어진 작은 무리에는 약한 후보 1곳
        rows = [(f"big{i}", 1.0, -0.6 + 0.2 * i) for i in range(6)] + [("small", 23.5, 0.8)]
        farms = _farm_table(rows)
        locations = e.complaint_locations([_at(0, -0.3), _at(0, 0), _at(0, 0.3), _at(20, 0), _at(20, 0.1)])
        r = e.score_complaint_candidates(farms, locations, e.Wind(0.0, 2.0), top_k=5, sensitivity=False)
        self.assertEqual(len(r.candidates), 5)
        self.assertIn("small", r.candidates["farm_id"].tolist())

    def test_stability_and_weak_wind_notes(self):
        farms = _farm_table([("a", 2, 0), ("b", 2, 1.5)])
        locations = e.complaint_locations([_at(0, 0), _at(0, 0.3)])
        r = e.score_complaint_candidates(farms, locations, e.Wind(0.0, 0.3))
        self.assertIsNotNone(r.stability)
        self.assertTrue(0.0 <= r.stability <= 1.0)
        self.assertIn(r.stability_label, ("높음", "보통", "낮음"))
        self.assertEqual(r.confidence, "매우 낮음")
        self.assertTrue(any("0.5 m/s" in note for note in r.notes))

    def test_score_components_bounded_and_cards_have_support(self):
        farms = _farm_table([("a", 1, 0), ("b", 2.5, 0.4)])
        locations = e.complaint_locations([_at(0, 0), _at(0, 0.2), _at(0.2, 0)])
        r = e.score_complaint_candidates(farms, locations, e.Wind(0.0, 2.0), start=_at(-5, 0))
        s = r.candidates
        self.assertTrue((s["score"] <= 100.0 + 1e-9).all())
        self.assertTrue((s["s_wind"] <= 40.0 + 1e-9).all())
        for card in e.to_cards(r):
            self.assertEqual(card["support"]["total"], 3)
            self.assertIn("민원 위치", card["evidence"][0])
            self.assertNotIn("확률", card["reason"])


if __name__ == "__main__":
    unittest.main()
