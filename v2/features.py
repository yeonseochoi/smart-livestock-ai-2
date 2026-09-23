"""격자 x Event 학습 행 생성.

시간 규칙(누수 방지):
  - 초기 변수(BASE)      : [t0, T) 의 민원만 사용 (T = t0 + 30분)
  - 이력 변수(HIST)      : t0 "이전" 민원만 사용 (t0 포함하지 않음)
  - prior               : 폴드 단위로 학습 Event만으로 계산 (add_prior). 여기서는 만들지 않는다.
  - target/future_cnt   : [T, T+30분) 민원 — 라벨 전용, 변수로 쓰지 않는다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import CANDIDATE_RADIUS, HORIZON_MIN, INPUT_MIN
from data import ComplaintIndex
from events import candidate_cells

# 기존 sensitivity_early_prediction.FEATURES 15개와 같은 정의 (prior는 폴드에서 채운다)
BASE_FEATURES = [
    "initial_count", "initial_intensity", "min_distance", "neighbor_count", "radius3_count",
    "prior", "hour_sin", "hour_cos", "centroid_distance", "nearest_intensity",
    "weighted_intensity", "observed_grid_count", "observed_report_count", "first15_count", "growth",
]
# 새로 추가하는 t0 이전 이력 변수
HIST_FEATURES = [
    "hist_total_3h",            # 직전 3시간 전체 신고 수 (세션 정의상 직전 60분은 0)
    "gap_before_t0_min",        # 직전 민원 이후 경과 분 (상한 1440)
    "hist_cell_24h",            # 직전 24시간 해당 격자 신고 수
    "hist_cell_7d",             # 직전 7일 해당 격자 신고 수
    "hist_same_hour_rate",      # 과거 하루 중 같은 시각(시)에 해당 격자가 활성이었던 날의 비율
    "hist_same_hour_30d",       # 직전 30일 중 같은 시각(시)에 활성이었던 날 수
    "prev_night_cell_active",   # 전날 같은 시각 60분 창에 해당 격자 신고 유무
    "prev_night_total",         # 전날 같은 시각 60분 창 전체 신고 수
]
META_COLUMNS = ["event_id", "t0", "grid_x", "grid_y", "target", "future_cnt"]


def _hist_features(ci: ComplaintIndex, s: int, cands: list[tuple[int, int]], t0: int | None = None) -> dict[str, np.ndarray]:
    t0 = int(ci.t[s]) if t0 is None else int(t0)
    day0 = t0 // 1440
    hour0 = (t0 % 1440) // 60
    gap = min(t0 - int(ci.t[s - 1]), 1440) if s > 0 else 1440           # t[s-1] 은 항상 t0 이전
    elapsed_days = max(day0 - ci.first_day, 1)
    prev_lo = t0 - 1440
    prev_hi = prev_lo + 60

    out = {k: np.zeros(len(cands)) for k in HIST_FEATURES if k not in ("hist_total_3h", "gap_before_t0_min", "prev_night_total")}
    for i, cell in enumerate(cands):
        out["hist_cell_24h"][i] = ci.cell_count_between(cell, t0 - 1440, t0)
        out["hist_cell_7d"][i] = ci.cell_count_between(cell, t0 - 7 * 1440, t0)
        days = ci.cell_hour_days.get((cell[0], cell[1], hour0))
        if days is not None:
            n_before = int(np.searchsorted(days, day0, "left"))                     # 오늘 이전 날짜만
            n_30d = n_before - int(np.searchsorted(days, day0 - 30, "left"))
            out["hist_same_hour_rate"][i] = n_before / elapsed_days
            out["hist_same_hour_30d"][i] = n_30d
        out["prev_night_cell_active"][i] = float(ci.cell_count_between(cell, prev_lo, prev_hi) > 0)
    n = len(cands)
    out["hist_total_3h"] = np.full(n, float(ci.count_between(t0 - 180, t0)))
    out["gap_before_t0_min"] = np.full(n, float(gap))
    out["prev_night_total"] = np.full(n, float(ci.count_between(prev_lo, prev_hi)))
    return out


def build_event_rows(ci: ComplaintIndex, s: int, event_id: str, t0_min: int | None = None) -> pd.DataFrame:
    """시작 인덱스 s 의 Event를 후보 격자별 행으로 만든다. t0_min 은 legacy_hour(정시 t0)에서만 지정한다."""
    t = ci.t
    t0 = int(t[s]) if t0_min is None else int(t0_min)
    T = t0 + INPUT_MIN; E = T + HORIZON_MIN
    i1 = int(np.searchsorted(t, T, "left")); i2 = int(np.searchsorted(t, E, "left"))

    init = pd.DataFrame({
        "gx": ci.gx[s:i1], "gy": ci.gy[s:i1], "inten": ci.inten[s:i1],
        "f15": (t[s:i1] < t0 + min(15, INPUT_MIN // 2)).astype(int),
    })
    obs = (init.groupby(["gx", "gy"])
           .agg(count=("inten", "size"), mean_intensity=("inten", "mean"), first15=("f15", "sum"))
           .reset_index())
    cands = candidate_cells(set(zip(obs["gx"], obs["gy"])), CANDIDATE_RADIUS)
    C = np.asarray(cands, dtype=float)

    ox, oy = obs["gx"].to_numpy(float), obs["gy"].to_numpy(float)
    cnt = obs["count"].to_numpy(float)
    mi = obs["mean_intensity"].to_numpy(float)
    f15 = obs["first15"].to_numpy(float)
    dist = np.sqrt((C[:, 0:1] - ox[None, :]) ** 2 + (C[:, 1:2] - oy[None, :]) ** 2)
    own = dist == 0
    initial_count = (cnt[None, :] * own).sum(1)
    first15 = (f15[None, :] * own).sum(1)
    cx, cy = np.average(ox, weights=cnt), np.average(oy, weights=cnt)
    w = 1.0 / (1.0 + dist)

    frac_hour = (t0 % 1440) / 60.0
    feats = {
        "initial_count": initial_count,
        "initial_intensity": (mi[None, :] * own).sum(1),
        "min_distance": dist.min(1),
        "neighbor_count": (cnt[None, :] * (dist <= 1.5)).sum(1),
        "radius3_count": (cnt[None, :] * (dist <= 3.0)).sum(1),
        "prior": np.zeros(len(cands)),                      # 폴드에서 채움
        "hour_sin": np.full(len(cands), np.sin(2 * np.pi * frac_hour / 24)),
        "hour_cos": np.full(len(cands), np.cos(2 * np.pi * frac_hour / 24)),
        "centroid_distance": np.hypot(C[:, 0] - cx, C[:, 1] - cy),
        "nearest_intensity": mi[dist.argmin(1)],
        "weighted_intensity": (w * mi[None, :]).sum(1) / w.sum(1),
        "observed_grid_count": np.full(len(cands), float(len(obs))),
        "observed_report_count": np.full(len(cands), float(cnt.sum())),
        "first15_count": first15,
        "growth": (initial_count - first15) - first15,
    }
    feats.update(_hist_features(ci, s, cands, t0))

    fut = pd.Series(list(zip(ci.gx[i1:i2], ci.gy[i1:i2]))).value_counts() if i2 > i1 else pd.Series(dtype=int)
    fut_map = fut.to_dict()
    future_cnt = np.array([fut_map.get((int(x), int(y)), 0) for x, y in cands], dtype=float)

    df = pd.DataFrame(feats)
    # n_future_all: 예측 창에 민원이 있었던 서로 다른 격자 수(후보 밖 포함). 평가 전용, 모델 변수가 아니다.
    df.insert(0, "n_future_all", float(len(fut_map)))
    df.insert(0, "future_cnt", future_cnt)
    df.insert(0, "target", (future_cnt > 0).astype(int))
    df.insert(0, "grid_y", C[:, 1].astype(int))
    df.insert(0, "grid_x", C[:, 0].astype(int))
    df.insert(0, "t0", pd.to_datetime(t0, unit="m"))
    df.insert(0, "event_id", event_id)
    return df


def add_prior(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """폴드별 prior. 학습 Event만 사용하고, 학습 행은 자기 Event의 라벨을 뺀 leave-one-out 값이다.

      test  행 : (학습 Event 중 그 격자가 활성이었던 수) / (학습 Event 수)
      train 행 : (활성 Event 수 - 자기 라벨) / (학습 Event 수 - 1)
    """
    n_train = train["event_id"].nunique()
    cnt = train[train["target"] == 1].groupby(["grid_x", "grid_y"])["event_id"].nunique()
    key = lambda d: list(zip(d["grid_x"], d["grid_y"]))
    tr, te = train.copy(), test.copy()
    c_tr = np.array([cnt.get(k, 0) for k in key(tr)], dtype=float)
    c_te = np.array([cnt.get(k, 0) for k in key(te)], dtype=float)
    tr["prior"] = (c_tr - tr["target"].to_numpy(float)) / max(n_train - 1, 1)
    te["prior"] = c_te / max(n_train, 1)
    return tr, te
