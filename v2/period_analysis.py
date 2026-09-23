"""시기별 성능 변화: 2022년까지 폴드 vs 2023년 이후(반기별 폴드를 합친 보고용 집계).

폴드는 합치지 않고 반기별로 그대로 평가하되, 반기 하나는 Event가 1~20개라 변화를 말하기 어렵다.
그래서 반기별 표와 함께 "2022년까지 vs 2023년 이후" 두 묶음을 비교해 시기 변화를 본다(학습·폴드 구성에는 영향 없음).

  Hit@3(전체) 의 두 묶음 차이  = (2023년 이후 평균) - (2022년까지 평균)      -> 기준선/모델 성능이 시기에 따라 달라졌는가
  기준선 대비 우위의 변화      = (2023년 이후의 모델-prior 평균차) - (2022년까지의 모델-prior 평균차)
                                 -> 모델이 prior보다 낫다는 결론이 시기에 따라 달라졌는가
서로 다른 Event 집합이므로 두 묶음을 각각 독립적으로 복원추출하는 부트스트랩 95% 구간을 쓴다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from evaluate import event_topk_all
from evaluation import MARK, VERDICT_BETTER, VERDICT_REF, VERDICT_UNSURE, VERDICT_WORSE


def hit_by_event(frame: pd.DataFrame, score_col: str) -> pd.Series:
    """Event별 Hit@3(전체). 예측 창에 민원이 없는 Event는 빠진다."""
    out = {}
    for eid, g in frame.groupby("event_id", sort=False):
        res = event_topk_all(g["target"].to_numpy(int), g[score_col].to_numpy(float), float(g["n_future_all"].iloc[0]))
        if res is not None:
            out[eid] = res[0]
    return pd.Series(out, dtype=float)


def _two_sample(a: np.ndarray, b: np.ndarray, n_boot: int, seed: int) -> tuple[float, float, float]:
    """mean(b) - mean(a) 와 독립 부트스트랩 95% 구간."""
    rng = np.random.default_rng(seed)
    ia = rng.integers(0, len(a), size=(n_boot, len(a)))
    ib = rng.integers(0, len(b), size=(n_boot, len(b)))
    d = b[ib].mean(1) - a[ia].mean(1)
    return float(b.mean() - a.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def _verdict(lo: float, hi: float) -> str:
    return VERDICT_BETTER if lo > 0 else (VERDICT_WORSE if hi < 0 else VERDICT_UNSURE)


def period_shift(pooled: pd.DataFrame, folds, models: list[str], reference: str, cutoff: str = "2023-01-01",
                 n_boot: int = 2000) -> pd.DataFrame:
    start = {f.name: f.start for f in folds}
    fold_of = pooled.drop_duplicates("event_id").set_index("event_id")["fold"]
    is_late = fold_of.map(lambda n: start[n] >= pd.Timestamp(cutoff))
    hits = {m: hit_by_event(pooled, m) for m in models}
    ref = hits[reference]
    rows = []
    for m in models:
        h = hits[m]
        late_ids = [e for e in h.index if is_late[e]]
        early_ids = [e for e in h.index if not is_late[e]]
        a, b = h[early_ids].to_numpy(), h[late_ids].to_numpy()
        d, lo, hi = _two_sample(a, b, n_boot, 0)
        row = {"모델": m, "2022년까지 Hit@3(전체)": float(a.mean()), "2023년 이후 Hit@3(전체)": float(b.mean()),
               "평가 Event 수(2022까지/2023이후)": f"{len(a)}/{len(b)}",
               "시기 변화(이후-이전)": f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}] " + ("" if _verdict(lo, hi) == VERDICT_UNSURE else ("▲" if lo > 0 else "▼"))}
        if m == reference:
            row["기준선 대비 우위(2022까지)"] = "(기준)"; row["기준선 대비 우위(2023이후)"] = "(기준)"; row["우위의 변화"] = ""
        else:
            g = (h - ref).dropna()
            ga, gb = g[[e for e in g.index if not is_late[e]]].to_numpy(), g[[e for e in g.index if is_late[e]]].to_numpy()
            gd, glo, ghi = _two_sample(ga, gb, n_boot, 1)
            row["기준선 대비 우위(2022까지)"] = f"{ga.mean():+.3f}"
            row["기준선 대비 우위(2023이후)"] = f"{gb.mean():+.3f}"
            row["우위의 변화"] = f"{gd:+.3f} [{glo:+.3f}, {ghi:+.3f}] " + ("" if _verdict(glo, ghi) == VERDICT_UNSURE else ("▲" if glo > 0 else "▼"))
        rows.append(row)
    return pd.DataFrame(rows)
