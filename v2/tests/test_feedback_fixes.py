"""피드백 3건(P1 후보 밖 정답, P2 반기 폴드, P3 위치 병합의 미래 영향)에 대한 회귀 테스트.

P3 위치 병합
  1. '다리 좌표' 테스트: 예측 시점 뒤에 들어온 중간 좌표가 과거 두 위치를 하나로 이어 붙이지 못한다.
     (전체 기간 군집으로 세던 이전 방식은 여기서 1곳으로 센다 — 테스트가 그 차이를 실제로 잡는지도 함께 확인한다)
  2. 창 안 병합이 그 창의 점만으로 계산한 DBSCAN(30m, min_samples=1)과 같다 (실제 데이터 창 300개).
  3. 전체 기간 군집 ID(loc_diag)를 뒤섞어도 Event 선정·변수가 그대로다 (선정/변수가 그 열을 읽지 않는다는 증거).
  4. 격자 원점이 상수라 데이터를 잘라도 같은 민원은 같은 격자에 놓인다.
P1 후보 밖 정답
  5. 후보 밖에만 민원이 생긴 Event는 n_future_cells>0, has_positive=False, outside_only=True 로 남고,
     행의 target 은 전부 0, n_future_all 은 1이다.
  6. (전체) 지표는 그런 Event를 실패(0)로 세고 Recall 분모에 후보 밖 정답을 포함한다. (후보내) 지표는 그 Event를 뺀다.
P2 폴드
  7. 표본이 작은 기간도 합치지 않고 자기 폴드로 남으며 '표본 작음'으로 표시된다. 테스트 Event 0개 기간은 표에만 남는다.

실행: python -m unittest discover -s tests -v   (v2 폴더에서)
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (DATA_PATH, EARTH_RADIUS_M, GRID_ORIGIN_LAT, GRID_ORIGIN_LON, INPUT_MIN, K_MIN_UNIQUE_LOCATIONS,  # noqa: E402
                    LOC_EPS_M)
from data import M_PER_DEG, ComplaintIndex, assign_grid, load_complaints  # noqa: E402
from evaluate import event_topk_all, evaluate  # noqa: E402
from evaluation import HIT_ALL, HIT_C, REC_ALL, REC_C, event_level  # noqa: E402
from events import _event_stats, add_target_flags, session_table, trigger_events, trigger_scan  # noqa: E402
from features import build_event_rows  # noqa: E402
from folds import make_folds  # noqa: E402


def synth_df(reports) -> pd.DataFrame:
    """reports: (분, 격자x, 격자y, 격자 중심에서 동쪽으로 dx 미터, 북쪽으로 dy 미터)"""
    base = pd.Timestamp("2024-03-01 00:00")
    rows = []
    for minute, gx, gy, dx, dy in reports:
        x, y = gx * 1000 + 500 + dx, gy * 1000 + 500 + dy
        rows.append({
            "datetime": base + pd.Timedelta(minutes=minute),
            "latitude": GRID_ORIGIN_LAT + y / M_PER_DEG,
            "longitude": GRID_ORIGIN_LON + x / (M_PER_DEG * math.cos(math.radians(GRID_ORIGIN_LAT))),
            "intensity": 2.0, "grid_x": gx, "grid_y": gy,
        })
    return pd.DataFrame(rows)


class BridgePointDoesNotLeak(unittest.TestCase):
    """A(0m) - B(25m) - C(50m) 일렬. B가 예측 시점 뒤에 들어와도 입력 창의 위치 수는 A, C 두 곳이다."""

    def setUp(self):
        self.full = synth_df([(0, 0, 0, 0, 0), (5, 0, 0, 50, 0), (45, 0, 0, 25, 0), (100, 3, 3, 0, 0)])
        self.cut = self.full[self.full["datetime"] < pd.Timestamp("2024-03-01 00:30")]     # 예측 시점(T=30분) 이전만

    def test_window_location_count_ignores_later_bridge(self):
        ci_full, ci_cut = ComplaintIndex(self.full), ComplaintIndex(self.cut)
        n_full = ci_full.n_locs(0, int(np.searchsorted(ci_full.t, ci_full.t[0] + INPUT_MIN)))
        n_cut = ci_cut.n_locs(0, int(np.searchsorted(ci_cut.t, ci_cut.t[0] + INPUT_MIN)))
        self.assertEqual((n_full, n_cut), (2, 2))

    def test_event_selection_same_with_and_without_future_bridge(self):
        a = trigger_scan(ComplaintIndex(self.full), 2)
        b = trigger_scan(ComplaintIndex(self.cut), 2)
        self.assertEqual(int(a.iloc[0]["n_init_locs"]), 2)
        self.assertEqual(int(a.iloc[0]["start_idx"]), int(b.iloc[0]["start_idx"]))
        self.assertEqual(int(a.iloc[0]["n_init_locs"]), int(b.iloc[0]["n_init_locs"]))

    def test_old_global_clustering_would_have_merged_them(self):
        """테스트의 민감도 확인: 전체 좌표를 한꺼번에 군집화하면 A와 C가 B를 통해 1곳이 된다(이전 방식의 결함)."""
        coords = np.radians(self.full[["latitude", "longitude"]].to_numpy()[:3])
        lab = DBSCAN(eps=LOC_EPS_M / EARTH_RADIUS_M, min_samples=1, metric="haversine").fit_predict(coords)
        self.assertEqual(len(set(lab)), 1)
        lab_wo_bridge = DBSCAN(eps=LOC_EPS_M / EARTH_RADIUS_M, min_samples=1, metric="haversine").fit_predict(coords[[0, 1]])
        self.assertEqual(len(set(lab_wo_bridge)), 2)


@unittest.skipUnless(Path(DATA_PATH).exists(), "민원 xlsx 가 없어 건너뜀")
class LocationMergeOnRealData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = load_complaints()
        cls.ci = ComplaintIndex(cls.df)

    def test_window_merge_equals_dbscan_on_window_points(self):
        rng = np.random.default_rng(0)
        n = len(self.ci.t)
        coords = np.radians(self.ci.df[["latitude", "longitude"]].to_numpy())
        checked = mismatched = 0
        for j in rng.choice(n - 100, size=300, replace=False):
            i1 = int(np.searchsorted(self.ci.t, self.ci.t[j] + INPUT_MIN))
            if i1 - j < 2 or i1 - j > 150:
                continue
            ref = DBSCAN(eps=LOC_EPS_M / EARTH_RADIUS_M, min_samples=1, metric="haversine").fit_predict(coords[j:i1])
            mine = self.ci.loc_labels(j, i1)
            # 두 분할이 같은지: 같은 군집 쌍이 같아야 한다
            same = pd.crosstab(ref, mine)
            ok = (same.gt(0).sum(axis=0) == 1).all() and (same.gt(0).sum(axis=1) == 1).all()
            checked += 1
            mismatched += int(not ok)
        self.assertGreater(checked, 100)
        self.assertEqual(mismatched, 0, f"{checked}개 창 중 {mismatched}개에서 창 안 병합이 DBSCAN 과 다름")

    def test_selection_and_features_do_not_read_global_location_ids(self):
        scrambled = self.df.copy()
        scrambled["loc_diag"] = np.random.default_rng(1).permutation(scrambled["loc_diag"].to_numpy())
        ci2 = ComplaintIndex(scrambled)
        for mode in ("trigger",):
            k = K_MIN_UNIQUE_LOCATIONS[mode]
            a = trigger_events(self.ci, k); b = trigger_events(ci2, k)
            pd.testing.assert_frame_equal(a.drop(columns=["t0"]), b.drop(columns=["t0"]), obj=f"{mode} 선정")
        sa, sb = session_table(self.ci), session_table(ci2)
        pd.testing.assert_frame_equal(sa.drop(columns=["t0"]), sb.drop(columns=["t0"]), obj="session 표")
        ev = trigger_events(self.ci, K_MIN_UNIQUE_LOCATIONS["trigger"]).head(15)
        for r in ev.itertuples():
            pd.testing.assert_frame_equal(build_event_rows(self.ci, int(r.start_idx), r.event_id, int(r.t0_min)),
                                          build_event_rows(ci2, int(r.start_idx), r.event_id, int(r.t0_min)))

    def test_grid_origin_is_a_constant_not_refit_on_cut_data(self):
        full = assign_grid(self.df.drop(columns=["grid_x", "grid_y"]))
        cut_df = self.df[self.df["datetime"] < pd.Timestamp("2022-01-01")].drop(columns=["grid_x", "grid_y"])
        cut = assign_grid(cut_df)
        self.assertTrue((full.loc[cut.index, ["grid_x", "grid_y"]].to_numpy() == cut[["grid_x", "grid_y"]].to_numpy()).all())
        # 상수는 전체 중앙값에서 나온 값이다 (기존 파이프라인과 같은 격자)
        self.assertAlmostEqual(float(self.df["latitude"].median()), GRID_ORIGIN_LAT, places=9)
        self.assertAlmostEqual(float(self.df["longitude"].median()), GRID_ORIGIN_LON, places=9)


class OutsideCandidateLabels(unittest.TestCase):
    """입력 30분은 격자 (0,0) 한 곳. 예측 창의 민원은 (6,0)에 있어 후보(반경 2칸) 밖이다."""

    def setUp(self):
        reports = [(0, 0, 0, 0, 0), (5, 0, 0, 100, 0), (10, 0, 0, 200, 0),      # 입력 창: 격자 (0,0)
                   (40, 6, 0, 0, 0),                                              # 예측 창: 후보 밖
                   (100, 9, 9, 0, 0)]                                             # 예측 창이 끝났음을 알리는 이후 민원
        self.ci = ComplaintIndex(synth_df(reports))

    def test_event_stats_keep_outside_positive(self):
        row = _event_stats(self.ci, 0, 0.0)
        self.assertTrue(row["complete"])
        self.assertEqual((row["n_future_cells"], row["n_positive_cells"], row["n_outside_cells"]), (1, 0, 1))
        flags = add_target_flags(pd.DataFrame([row]))
        self.assertTrue(bool(flags["has_future"].iloc[0]))
        self.assertFalse(bool(flags["has_positive"].iloc[0]))
        self.assertTrue(bool(flags["outside_only"].iloc[0]))
        self.assertEqual(row["cov_cells_r2"], 0)
        self.assertEqual(row["cov_cells_r4"], 0)             # 거리 6칸이라 반경 4로 넓혀도 후보 밖

    def test_rows_have_zero_target_but_remember_future_cells(self):
        rows = build_event_rows(self.ci, 0, "E1")
        self.assertEqual(int(rows["target"].sum()), 0)
        self.assertTrue((rows["n_future_all"] == 1).all())
        self.assertFalse(((rows["grid_x"] == 6) & (rows["grid_y"] == 0)).any())      # (6,0)은 후보에 없다

    def test_full_metrics_count_it_as_failure_conditional_metrics_skip_it(self):
        rows = build_event_rows(self.ci, 0, "E1")
        score = np.arange(len(rows), dtype=float)
        # 이벤트 두 개: E1(후보 밖 정답만), E2(후보 안 정답 1 + 후보 밖 정답 1, 안쪽 정답을 1위로 점수)
        e2 = rows.copy(); e2["event_id"] = "E2"; e2["target"] = 0; e2["n_future_all"] = 2.0
        best = int(np.argmax(score)); e2.iloc[best, e2.columns.get_loc("target")] = 1
        both = pd.concat([rows, e2], ignore_index=True)
        sc = np.concatenate([score, score])
        arr = event_level(both, sc, np.array(["E1", "E2"]))
        cols = {"hit_c": 0, "rec_c": 1, "hit_a": 3, "rec_a": 4}
        self.assertTrue(np.isnan(arr[0, cols["hit_c"]]))                  # E1: 후보내 지표에서는 제외
        self.assertEqual((arr[0, cols["hit_a"]], arr[0, cols["rec_a"]]), (0.0, 0.0))     # E1: 전체 지표에서는 실패
        self.assertEqual((arr[1, cols["hit_c"]], arr[1, cols["rec_c"]]), (1.0, 1.0))
        self.assertEqual((arr[1, cols["hit_a"]], arr[1, cols["rec_a"]]), (1.0, 0.5))     # 분모는 후보 밖 포함 2개
        res = evaluate(both, sc)
        self.assertEqual((res["평가 Event 수(후보내)"], res["평가 Event 수(전체)"]), (1, 2))
        self.assertAlmostEqual(res[HIT_ALL], 0.5); self.assertAlmostEqual(res[HIT_C], 1.0)
        self.assertAlmostEqual(res[REC_ALL], 0.25); self.assertAlmostEqual(res[REC_C], 1.0)

    def test_event_without_any_future_complaint_is_excluded_from_full_metrics(self):
        self.assertIsNone(event_topk_all(np.zeros(5, int), np.arange(5.0), 0.0))
        self.assertEqual(event_topk_all(np.zeros(5, int), np.arange(5.0), 2.0), (0.0, 0.0))


class HalfYearFoldsAreNotMerged(unittest.TestCase):
    def test_small_periods_stay_separate_and_are_flagged(self):
        rng = np.random.default_rng(0)
        def stamps(start, end, n):
            lo, hi = pd.Timestamp(start).value // 60_000_000_000, pd.Timestamp(end).value // 60_000_000_000
            return pd.to_datetime(np.sort(rng.integers(lo, hi, n)), unit="m")
        t0 = stamps("2019-06-01", "2020-12-31", 100)
        for a, b, n in [("2021-01-01", "2021-12-31", 30), ("2022-01-01", "2022-12-31", 20), ("2023-07-01", "2023-12-31", 5),
                        ("2024-07-01", "2024-12-31", 3), ("2025-01-01", "2025-06-30", 1), ("2025-09-01", "2025-09-30", 2)]:
            t0 = t0.append(stamps(a, b, n))
        ev = pd.DataFrame({"t0": t0})
        ev["event_id"] = [f"E{i}" for i in range(len(ev))]
        for col in ("has_future", "has_positive"):
            ev[col] = True
        ev["outside_only"] = False
        folds, table = make_folds(ev)
        names = [f.name for f in folds]
        self.assertIn("2023-07~2023-12", names)            # 5개뿐이어도 자기 폴드
        self.assertIn("2024-07~2024-12", names)
        self.assertIn("2025-01~2025-06", names)
        self.assertEqual(len(folds[0].test_ids), 30)       # 2021 폴드는 그대로
        small = {f.name: f.reliable for f in folds}
        self.assertTrue(small["2021-01~2021-12"]); self.assertFalse(small["2023-07~2023-12"])
        row = table[table["폴드"] == "2023-01~2023-06"].iloc[0]
        self.assertTrue(row["상태"].startswith("제외(테스트 Event 0"))
        self.assertTrue(table[table["폴드"] == "2023-07~2023-12"].iloc[0]["표본"].startswith("작음"))
        # 폴드 수 = 테스트 Event가 있는 기간 수 (합쳐지지 않았다)
        self.assertEqual(sum(1 for _, r in table.iterrows() if r["테스트 Event"] > 0), len(folds))


if __name__ == "__main__":
    unittest.main()
