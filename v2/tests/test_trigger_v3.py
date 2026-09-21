"""새 trigger(최근 30분 안에 k번째 고유 위치 신고가 들어온 그 순간 발동)에 대한 테스트.

합성 데이터(손으로 발동 시점을 계산할 수 있는 작은 예)
  1. k번째 위치가 들어온 분의 끝(T=e+1)에 발동한다. 이전 방식(첫 신고+30분)은 더 늦게 발동한다(lead_gain_min).
  2. 같은 분에 들어온 신고도 모두 센다.
  3. 30분 경계: 창은 [e-29, e] 이다 (29분 간격이면 발동, 30분 간격이면 발동하지 않는다).
  4. 발동 뒤 60분은 쿨다운: 그 사이 신고로는 재발동하지 않고, 앞 Event의 라벨 창이 다음 입력 창에 들어가지 않는다.
  5. 신고 건수 기준(unit="reports")과 위치 기준(unit="locs")은 같은 위치 반복 신고를 다르게 센다.
실제 데이터
  6. 모든 trigger Event에서 T-1분에 신고가 있고(신고 때문에 발동), lead_gain_min >= 0, 앞 Event 라벨 창과 입력 창이 겹치지 않는다.
  7. 예측 시점 이후 데이터를 지워도 T 이하의 Event는 시각·위치 수까지 똑같이 선정된다.
후보 밖 비율
  8. ratio_ci 는 합계의 비와 구간을 돌려주고, 후보 밖 신고 수가 Event 통계에 들어 있다.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cand_coverage import ratio_ci  # noqa: E402
from config import DATA_PATH, HORIZON_MIN, INPUT_MIN, K_MIN_UNIQUE_LOCATIONS  # noqa: E402
from data import ComplaintIndex, load_complaints  # noqa: E402
from events import _event_stats, trigger_events, trigger_fixed_scan, trigger_scan  # noqa: E402
from test_feedback_fixes import synth_df  # noqa: E402

FAR = (100_000, 9, 9, 0, 0)          # 예측 창이 데이터 끝을 넘지 않게 하는 맨 뒤 민원 (Event 를 complete 로 만든다)


def scan(reports, k, unit="locs"):
    ci = ComplaintIndex(synth_df(list(reports) + [FAR]))
    return ci, trigger_scan(ci, k, unit)


def loc(minute, i):
    """i 번째 위치: 서로 200m 떨어진 같은 격자 안 점 (i=0..4)."""
    return (minute, 0, 0, -400 + 200 * i, 0)


class SyntheticTrigger(unittest.TestCase):
    def test_fires_when_kth_location_arrives(self):
        ci, sc = scan([loc(0, 0), loc(10, 1), loc(20, 2)], 3)
        t0 = int(ci.t[0])
        self.assertEqual(len(sc), 1)
        r = sc.iloc[0]
        self.assertEqual(int(r["t_pred_min"]), t0 + 21)                # 3번째 위치가 들어온 분(20)의 끝
        self.assertEqual(int(r["t0_min"]), t0 + 21 - INPUT_MIN)
        self.assertEqual(int(r["n_init_locs"]), 3)
        self.assertEqual(int(r["lead_gain_min"]), 30 - 21)             # 이전 방식(첫 신고+30분=30)보다 9분 빠름
        fixed = trigger_fixed_scan(ci, 3)
        self.assertEqual(int(fixed.iloc[0]["t0_min"]) + INPUT_MIN, t0 + 30)

    def test_second_location_is_not_enough_for_k3(self):
        _, sc = scan([loc(0, 0), loc(10, 1)], 3)
        self.assertEqual(len(sc), 0)

    def test_same_minute_reports_are_all_counted(self):
        ci, sc = scan([loc(0, 0), loc(0, 1), loc(0, 2)], 3)
        self.assertEqual(len(sc), 1)
        self.assertEqual(int(sc.iloc[0]["t_pred_min"]), int(ci.t[0]) + 1)

    def test_window_boundary_29_vs_30_minutes(self):
        ci, sc = scan([loc(0, 0), loc(10, 1), loc(29, 2)], 3)          # [0, 29] 30개 분 -> 발동
        self.assertEqual(len(sc), 1)
        self.assertEqual(int(sc.iloc[0]["t_pred_min"]), int(ci.t[0]) + 30)
        _, sc = scan([loc(0, 0), loc(10, 1), loc(30, 2)], 3)           # 30분째에는 첫 신고가 창 밖 -> 발동 안 함
        self.assertEqual(len(sc), 0)

    def test_repeat_reports_at_same_location_count_once(self):
        _, sc = scan([loc(0, 0), loc(5, 0), loc(10, 0), loc(15, 1)], 3)                  # 위치는 2곳뿐
        self.assertEqual(len(sc), 0)
        _, sc = scan([loc(0, 0), loc(5, 0), loc(10, 0), loc(15, 1)], 3, unit="reports")   # 신고는 4건
        self.assertEqual(len(sc), 1)

    def test_cooldown_blocks_refire_for_60_minutes(self):
        reports = [loc(0, 0), loc(1, 1),                    # e=1 발동 (T=2)
                   loc(30, 2), loc(31, 3),                  # 쿨다운 안(<61): 위치 2곳이 있어도 발동하지 않는다
                   loc(61, 4), loc(62, 0)]                  # e=61: 창 [32,61] 에 1곳 -> 대기, e=62: 2곳 -> 발동 (T=63)
        ci, sc = scan(reports, 2)
        t0 = int(ci.t[0])
        self.assertEqual([int(x) - t0 for x in sc["t_pred_min"]], [2, 63])
        gap = np.diff(sc["t0_min"].to_numpy())
        self.assertTrue((gap >= INPUT_MIN + HORIZON_MIN).all())          # 앞 라벨 창 끝 <= 다음 입력 창 시작

    def test_event_survives_deleting_everything_from_T_on(self):
        """발동 시점 T 이후 민원을 전부 지워도 같은 T에 발동한다 (미래를 엿보는 규칙이면 여기서 깨진다)."""
        reports = [loc(0, 0), loc(10, 1), loc(25, 2), loc(26, 3), loc(27, 4)]      # 3번째 위치는 25분, 뒤에도 신고가 이어짐
        full = ComplaintIndex(synth_df(reports))
        T = int(trigger_scan(full, 3).iloc[0]["t_pred_min"])
        self.assertEqual(T, int(full.t[0]) + 26)
        cut = ComplaintIndex(synth_df(reports[:3]))                                # 분 25 까지만
        self.assertEqual(int(trigger_scan(cut, 3).iloc[0]["t_pred_min"]), T)
        cut2 = ComplaintIndex(synth_df(reports[:2]))                               # 3번째 신고가 없으면 발동 안 함
        self.assertEqual(len(trigger_scan(cut2, 3)), 0)


@unittest.skipUnless(Path(DATA_PATH).exists(), "민원 xlsx 가 없어 건너뜀")
class RealDataTrigger(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = load_complaints()
        cls.ci = ComplaintIndex(cls.df)
        cls.k = K_MIN_UNIQUE_LOCATIONS["trigger"]
        cls.ev = trigger_events(cls.ci, cls.k)

    def test_every_event_is_fired_by_a_report_in_the_last_minute(self):
        T = (self.ev["t0_min"] + INPUT_MIN).to_numpy()
        self.assertTrue(np.isin(T - 1, self.ci.t).all())
        self.assertTrue((self.ev["n_init_locs"] >= self.k).all())

    def test_never_later_than_old_rule(self):
        self.assertTrue((self.ev["lead_gain_min"] >= 0).all())
        self.assertGreater(self.ev["lead_gain_min"].median(), 0)

    def test_labels_never_feed_next_input(self):
        t0 = self.ev.sort_values("t0_min")["t0_min"].to_numpy()
        self.assertTrue((np.diff(t0) >= INPUT_MIN + HORIZON_MIN).all())

    def test_each_event_survives_deleting_everything_from_T_on(self):
        """Event마다 자기 발동 시점 T 바로 앞까지만 남겨도 같은 T에 발동한다 (엿보기 변이를 잡는 가장 엄격한 검사)."""
        rng = np.random.default_rng(5)
        for i in rng.choice(len(self.ev), size=25, replace=False):
            r = self.ev.iloc[int(i)]
            T = int(r["t0_min"]) + INPUT_MIN
            cut_ci = ComplaintIndex(self.df[self.df["datetime"] < pd.to_datetime(T, unit="m")])
            got = trigger_scan(cut_ci, self.k)
            self.assertIn(T, set(got["t_pred_min"].astype(int)), f"{r['event_id']}: T 이후를 지우면 발동하지 않음")

    def test_selection_identical_when_future_is_deleted(self):
        rng = np.random.default_rng(3)
        t_min, t_max = int(self.ci.t[0]), int(self.ci.t[-1])
        full = trigger_scan(self.ci, self.k)
        for x in rng.integers(t_min + 10_000, t_max, size=8):
            cut = trigger_scan(ComplaintIndex(self.df[self.df["datetime"] < pd.to_datetime(int(x), unit="m")]), self.k)
            cols = ["t_pred_min", "t0_min", "n_init_locs", "lead_gain_min", "start_idx"]
            a = full[full["t_pred_min"] <= x][cols].reset_index(drop=True)
            b = cut[cut["t_pred_min"] <= x][cols].reset_index(drop=True)
            pd.testing.assert_frame_equal(a, b, obj=f"x={x}")


class OutsideRatio(unittest.TestCase):
    def test_ratio_of_sums_and_interval(self):
        p, lo, hi = ratio_ci(np.array([1, 0, 2, 0]), np.array([2, 2, 2, 2]), n_boot=500)
        self.assertAlmostEqual(p, 3 / 8)
        self.assertTrue(0 <= lo <= p <= hi <= 1)
        self.assertTrue(np.isnan(ratio_ci(np.zeros(3), np.zeros(3))[0]))

    def test_outside_report_counts(self):
        ci = ComplaintIndex(synth_df([(0, 0, 0, 0, 0), (5, 0, 0, 100, 0), (40, 6, 0, 0, 0), (41, 6, 0, 50, 0),
                                      (42, 0, 0, 0, 0), (100, 9, 9, 0, 0)]))
        row = _event_stats(ci, 0, 0.0)
        self.assertEqual(row["n_future_reports"], 3)
        self.assertEqual(row["n_future_reports_outside"], 2)          # (6,0) 격자의 2건이 후보 밖
        self.assertEqual((row["n_future_cells"], row["n_outside_cells"]), (2, 1))


if __name__ == "__main__":
    unittest.main()
