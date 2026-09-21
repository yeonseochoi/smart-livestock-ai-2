"""2단계 평가 코드 검증.

  1. Hit@3/Recall@3 의 동점 기대값이 "모든 동점 순서를 다 해 본 평균"과 같다 (브루트포스).
  2. 모델 점수 == 기준선 점수이면 개선폭이 정확히 0이고 판정이 '구분 불가'다.
  3. 완벽한 점수는 무작위 기준선보다 '개선', 정반대 점수는 '악화'로 판정된다.
  4. 중복 제거 라벨은 원본 라벨의 부분집합이고, 제거 없는 라벨은 원본과 같다 (실제 데이터).
  5. 후보가 min_cands 이하인 Event는 Event 단위 지표에서 빠진다.

실행: python -m unittest discover -s tests -v   (v2 폴더에서)
"""
from __future__ import annotations

import itertools
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DATA_PATH, K_MIN_UNIQUE_LOCATIONS  # noqa: E402
from evaluate import event_topk  # noqa: E402
from evaluation import (HIT_ALL, HIT_C, PR, REC_ALL, REC_C, ROC, VERDICT_BETTER, VERDICT_UNSURE,  # noqa: E402
                        VERDICT_WORSE, compare, event_level)


def brute_force(y, s, k=3):
    n = len(y); hits = recs = 0.0; total = 0
    for perm in itertools.permutations(range(n)):
        order = sorted(perm, key=lambda i: -s[i])          # 안정 정렬: 동점은 perm 순서 = 무작위 동점 처리
        top = order[:k]
        found = sum(y[i] for i in top)
        hits += float(found > 0); recs += found / sum(y); total += 1
    return hits / total, recs / total


class TopKTies(unittest.TestCase):
    def test_expected_value_matches_bruteforce(self):
        rng = np.random.default_rng(0)
        for _ in range(40):
            n = int(rng.integers(4, 8))
            y = rng.integers(0, 2, n)
            if y.sum() == 0:
                y[0] = 1
            s = rng.integers(0, 3, n).astype(float)        # 동점이 많은 점수
            hit, rec = event_topk(y, s, 3)
            bh, br = brute_force(list(y), list(s), 3)
            self.assertAlmostEqual(hit, bh, places=9)
            self.assertAlmostEqual(rec, br, places=9)

    def test_no_positive_event_is_skipped(self):
        self.assertIsNone(event_topk(np.zeros(6, int), np.arange(6.0), 3))


def synthetic(n_events=40, n_cands=12, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for e in range(n_events):
        y = np.zeros(n_cands, int)
        y[rng.choice(n_cands, size=int(rng.integers(1, 4)), replace=False)] = 1
        for c in range(n_cands):
            rows.append({"event_id": f"E{e}", "cell": c, "target": int(y[c])})
    return pd.DataFrame(rows)


class CompareLogic(unittest.TestCase):
    def setUp(self):
        self.df = synthetic()
        rng = np.random.default_rng(1)
        self.noise = rng.random(len(self.df))

    def test_identical_scores_give_zero_difference(self):
        ref = self.df["target"].to_numpy() * 0.3 + self.noise
        out = compare(self.df, {"ref": ref, "same": ref.copy()}, "ref", n_boot=100)
        same = out[out["모델"] == "same"]
        self.assertTrue((same["차이"].abs() < 1e-12).all())
        self.assertTrue((same["95% 하한"].abs() < 1e-12).all() and (same["95% 상한"].abs() < 1e-12).all())
        self.assertTrue((same["판정"] == VERDICT_UNSURE).all())

    def test_perfect_beats_random_and_reversed_loses(self):
        y = self.df["target"].to_numpy(float)
        scores = {"ref": self.noise, "perfect": y + 0.01 * self.noise, "reversed": -y + 0.01 * self.noise}
        out = compare(self.df, scores, "ref", n_boot=200)
        verdict = lambda m, metric: out[(out["모델"] == m) & (out["지표"] == metric)]["판정"].iloc[0]
        for metric in (HIT_C, REC_C, HIT_ALL, REC_ALL, PR, ROC):
            self.assertEqual(verdict("perfect", metric), VERDICT_BETTER, metric)
        self.assertEqual(verdict("reversed", ROC), VERDICT_WORSE)

    def test_min_cands_drops_small_events(self):
        df = synthetic(n_events=6, n_cands=3)                  # 후보 3개뿐인 Event
        events = df["event_id"].unique()
        s = np.random.default_rng(0).random(len(df))
        arr = event_level(df.reset_index(drop=True), s, events, min_cands=3)
        self.assertTrue(np.isnan(arr).all())
        arr2 = event_level(df.reset_index(drop=True), s, events, min_cands=0)
        self.assertFalse(np.isnan(arr2[:, 0]).all())


@unittest.skipUnless(Path(DATA_PATH).exists(), "민원 xlsx 가 없어 건너뜀")
class DedupLabels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from data import ComplaintIndex, load_complaints
        from dedup import future_loc_sets
        from events import trigger_events
        from features import build_event_rows

        cls.ci = ComplaintIndex(load_complaints())
        cls.events = trigger_events(cls.ci, K_MIN_UNIQUE_LOCATIONS["trigger"]).head(60)
        cls.rows = pd.concat([build_event_rows(cls.ci, int(r.start_idx), r.event_id, int(r.t0_min)) for r in cls.events.itertuples()],
                             ignore_index=True)
        cls.fut = future_loc_sets(cls.ci, cls.events)

    def test_no_removal_reproduces_original_label(self):
        from dedup import relabel
        lab = relabel(self.rows, self.fut)
        self.assertTrue((lab == self.rows["target"].to_numpy()).all())

    def test_dedup_labels_are_subsets_of_original(self):
        from dedup import LABEL_VARIANTS, heavy_locations, relabel
        orig = self.rows["target"].to_numpy()
        prev = None
        for name, top_n, min_locs in LABEL_VARIANTS:
            lab = relabel(self.rows, self.fut, heavy_locations(self.ci, top_n), min_locs)
            self.assertTrue((lab <= orig).all(), name)
        # 제외하는 위치가 늘면 양성이 줄기만 한다
        n3 = relabel(self.rows, self.fut, heavy_locations(self.ci, 3)).sum()
        n10 = relabel(self.rows, self.fut, heavy_locations(self.ci, 10)).sum()
        self.assertGreaterEqual(orig.sum(), n3)
        self.assertGreaterEqual(n3, n10)


if __name__ == "__main__":
    unittest.main()
