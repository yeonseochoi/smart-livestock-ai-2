"""시간 누수 방지 테스트.

핵심 아이디어: "예측 시점 T 이후의 민원을 전부 지워도 결과가 같아야 한다".
  1. 변수: T 이후(>= T) 민원을 지운 데이터로 만든 변수가 전체 데이터로 만든 변수와 완전히 같다 (라벨 제외).
  2. Event 선정: 임의의 시각 X 이후 데이터를 지워도, T <= X 인 Event는 똑같이 선정된다 (session/trigger 모두).
  3. prior: 학습 행의 prior 는 자기 Event의 라벨을 뺀 값이다 (브루트포스와 일치).
  4. 폴드: 학습 Event의 라벨 창 끝이 테스트 시작 이전이고, 학습/테스트가 겹치지 않는다.

실행: python -m unittest discover -s tests -v   (v2 폴더에서)
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DATA_PATH, HORIZON_MIN, INPUT_MIN, K_MIN_UNIQUE_LOCATIONS  # noqa: E402
from data import ComplaintIndex, load_complaints  # noqa: E402
from events import build_events, session_starts, trigger_scan, _event_stats  # noqa: E402
from features import BASE_FEATURES, HIST_FEATURES, add_prior, build_event_rows  # noqa: E402
from folds import make_folds  # noqa: E402

FEATURES_NO_PRIOR = [f for f in BASE_FEATURES if f != "prior"] + HIST_FEATURES


@unittest.skipUnless(Path(DATA_PATH).exists(), "민원 xlsx 가 없어 건너뜀")
class NoLeakage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = load_complaints()
        cls.ci = ComplaintIndex(cls.df)
        cls.events = {m: build_events(cls.ci, K_MIN_UNIQUE_LOCATIONS[m], m) for m in ("session", "trigger")}

    # ---- 1. 변수 ----
    def test_features_ignore_data_at_or_after_prediction_time(self):
        rng = np.random.default_rng(0)
        for mode, ev in self.events.items():
            for i in rng.choice(len(ev), size=min(30, len(ev)), replace=False):
                r = ev.iloc[int(i)]
                full = build_event_rows(self.ci, int(r.start_idx), r.event_id, int(r.t0_min))
                T = pd.to_datetime(int(r.t0_min) + INPUT_MIN, unit="m")
                cut = ComplaintIndex(self.df[self.df["datetime"] < T])
                trunc = build_event_rows(cut, int(r.start_idx), r.event_id, int(r.t0_min))
                self.assertEqual(len(full), len(trunc), f"{mode} {r.event_id}: 후보 격자 수가 달라짐")
                pd.testing.assert_frame_equal(
                    full[["grid_x", "grid_y"] + FEATURES_NO_PRIOR].reset_index(drop=True),
                    trunc[["grid_x", "grid_y"] + FEATURES_NO_PRIOR].reset_index(drop=True),
                    check_exact=False, rtol=1e-9, obj=f"{mode} {r.event_id}",
                )

    # ---- 2. Event 선정 ----
    def test_event_selection_is_causal(self):
        rng = np.random.default_rng(1)
        t_min, t_max = int(self.ci.t[0]), int(self.ci.t[-1])
        for x in rng.integers(t_min + 10_000, t_max, size=6):
            cut = ComplaintIndex(self.df[self.df["datetime"] < pd.to_datetime(int(x), unit="m")])
            # trigger
            k = K_MIN_UNIQUE_LOCATIONS["trigger"]
            a = trigger_scan(self.ci, k); b = trigger_scan(cut, k)
            a = a[a["t0_min"] + INPUT_MIN <= x][["start_idx", "n_init_locs"]].reset_index(drop=True)
            b = b[b["t0_min"] + INPUT_MIN <= x][["start_idx", "n_init_locs"]].reset_index(drop=True)
            pd.testing.assert_frame_equal(a, b, obj="trigger 선정")
            # session
            sa, _ = session_starts(self.ci); sb, _ = session_starts(cut)
            sa = [s for s in sa if self.ci.t[s] + INPUT_MIN <= x]
            sb = [s for s in sb if cut.t[s] + INPUT_MIN <= x]
            self.assertEqual(sa, sb, "session 시작점")
            for s in sa[-20:]:
                self.assertEqual(_event_stats(self.ci, int(s), 0, False)["n_init_locs"],
                                 _event_stats(cut, int(s), 0, False)["n_init_locs"])

    def test_events_include_zero_positive_events(self):
        """선정이 예측 창 결과에 의존하지 않는다는 간접 증거: 추가 민원이 없는 Event가 포함된다."""
        for mode, ev in self.events.items():
            self.assertGreater(int((~ev["has_positive"]).sum()), 0, mode)

    def test_trigger_events_do_not_overlap(self):
        ev = self.events["trigger"].sort_values("t0_min")
        span = ev["t0_min"].to_numpy()
        self.assertTrue(np.all(np.diff(span) >= INPUT_MIN + HORIZON_MIN), "앞 Event 라벨 창과 다음 입력창이 겹침")

    # ---- 3. prior ----
    def test_prior_is_leave_one_out_on_train_and_train_only(self):
        ev = self.events["session"]
        folds, _ = make_folds(ev)
        fold = folds[0]
        rows = pd.concat([build_event_rows(self.ci, int(r.start_idx), r.event_id, int(r.t0_min)) for r in ev.itertuples()])
        tr = rows[rows["event_id"].isin(fold.train_ids)]
        te = rows[rows["event_id"].isin(fold.test_ids)]
        tr2, te2 = add_prior(tr, te)
        n = tr["event_id"].nunique()
        pos = tr[tr["target"] == 1]
        # 양성 행 몇 개를 브루트포스로 확인: 자기 Event 를 제외하고 센 값
        for _, r in tr2[tr2["target"] == 1].head(15).iterrows():
            others = pos[(pos["grid_x"] == r.grid_x) & (pos["grid_y"] == r.grid_y) & (pos["event_id"] != r.event_id)]
            self.assertAlmostEqual(r.prior, others["event_id"].nunique() / (n - 1), places=12)
        # 테스트 prior 는 학습 Event만으로 계산
        r = te2.iloc[0]
        cnt = pos[(pos["grid_x"] == r.grid_x) & (pos["grid_y"] == r.grid_y)]["event_id"].nunique()
        self.assertAlmostEqual(r.prior, cnt / n, places=12)

    # ---- 4. 폴드 ----
    def test_folds_respect_time_order_and_purge(self):
        for mode, ev in self.events.items():
            folds, _ = make_folds(ev)
            self.assertGreaterEqual(len(folds), 2)
            t0 = ev.set_index("event_id")["t0"]
            for f in folds:
                self.assertFalse(set(f.train_ids) & set(f.test_ids))
                purge = pd.Timedelta(minutes=INPUT_MIN + HORIZON_MIN)
                self.assertTrue((t0[f.train_ids] + purge <= f.start).all(), f"{mode} {f.name}: 라벨 창이 테스트로 넘어감")
                self.assertTrue((t0[f.test_ids] >= f.start).all() and (t0[f.test_ids] < f.end).all())


if __name__ == "__main__":
    unittest.main()
