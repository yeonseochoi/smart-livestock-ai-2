"""평가 지표.

Hit@3 / Recall@3 는 Event 안에서 후보 격자를 점수순으로 3개 고를 때의 값이다.
동점(예: 초기 민원 수가 같은 격자들, prior가 모두 0인 격자들)은 동점 집단에서 무작위로 뽑는다고 보고
기대값으로 계산한다 (정렬 순서에 따른 우연한 유리/불리를 없앤다).

두 가지 지표 묶음을 항상 함께 낸다.
  (후보내)  조건부 지표: 후보 격자 안에 정답이 하나라도 있는 Event만 대상. 기존 방식과 같다.
            '후보에 이미 들어온 정답을 얼마나 위로 올리나'를 본다.
  (전체)    끝까지 지표: 예측 창에 민원이 하나라도 있는 모든 Event가 대상.
            정답이 전부 후보 밖이면 Hit=0, Recall=0으로 실패 처리하고, Recall의 분모는 후보 밖을 포함한 전체 정답 격자 수다.
            '확산 전체를 얼마나 맞히나'를 본다. (df 에 n_future_all 열이 없으면 후보 안 정답 수를 전체로 본다)
"""
from __future__ import annotations

from math import comb

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from config import N_BOOTSTRAP, TOPK


def event_topk(y: np.ndarray, score: np.ndarray, k: int = TOPK) -> tuple[float, float] | None:
    """(Hit 기대값, Recall 기대값). 양성이 없는 Event는 None."""
    pos = int(y.sum())
    if pos == 0:
        return None
    n = len(y)
    if n <= k:
        return 1.0, 1.0
    cutoff = np.sort(score)[-k]
    sure = score > cutoff
    tied = score == cutoff
    r = k - int(sure.sum())
    m = int(tied.sum())
    p = int(y[tied].sum())
    sure_pos = int(y[sure].sum())
    if r <= 0:
        return (1.0 if sure_pos > 0 else 0.0), sure_pos / pos
    p_none = comb(m - p, r) / comb(m, r) if m - p >= r else 0.0
    hit = 1.0 if sure_pos > 0 else 1.0 - p_none
    recall = (sure_pos + r * p / m) / pos
    return hit, recall


def event_topk_all(y: np.ndarray, score: np.ndarray, n_future_all: float, k: int = TOPK) -> tuple[float, float] | None:
    """전체 지표용 (Hit, Recall). 예측 창에 민원이 없는 Event는 None, 정답이 전부 후보 밖이면 (0, 0)."""
    if n_future_all <= 0:
        return None
    res = event_topk(y, score, k)
    if res is None:                       # 후보 안 정답 0개, 후보 밖 정답만 있음
        return 0.0, 0.0
    hit, rec = res
    return hit, rec * int(y.sum()) / n_future_all


def per_event_arrays(df: pd.DataFrame, score: np.ndarray, k: int = TOPK, universe: str = "cand") -> tuple[np.ndarray, np.ndarray]:
    """universe="cand": (후보내) 조건부, "all": (전체) 끝까지."""
    hits, recs = [], []
    cols = ["event_id", "target"] + (["n_future_all"] if "n_future_all" in df.columns else [])
    work = df[cols].copy()
    work["s"] = score
    for _, g in work.groupby("event_id", sort=False):
        y = g["target"].to_numpy(int); sc = g["s"].to_numpy(float)
        if universe == "all":
            n_all = float(g["n_future_all"].iloc[0]) if "n_future_all" in g else float(y.sum())
            res = event_topk_all(y, sc, n_all, k)
        else:
            res = event_topk(y, sc, k)
        if res is not None:
            hits.append(res[0]); recs.append(res[1])
    return np.array(hits), np.array(recs)


def evaluate(df: pd.DataFrame, score: np.ndarray) -> dict:
    y = df["target"].to_numpy(int)
    hits, recs = per_event_arrays(df, score)
    hits_a, recs_a = per_event_arrays(df, score, universe="all")
    both = len(np.unique(y)) == 2
    return {
        "ROC-AUC(후보내)": float(roc_auc_score(y, score)) if both else np.nan,
        "PR-AUC(후보내)": float(average_precision_score(y, score)) if both else np.nan,
        "Hit@3(후보내)": float(hits.mean()) if len(hits) else np.nan,
        "Recall@3(후보내)": float(recs.mean()) if len(recs) else np.nan,
        "Hit@3(전체)": float(hits_a.mean()) if len(hits_a) else np.nan,
        "Recall@3(전체)": float(recs_a.mean()) if len(recs_a) else np.nan,
        "평가 Event 수(후보내)": int(len(hits)),
        "평가 Event 수(전체)": int(len(hits_a)),
        "테스트 Event 수": int(df["event_id"].nunique()),
        "양성률": float(y.mean()),
    }


def bootstrap_ci(values: np.ndarray, n: int = N_BOOTSTRAP, seed: int = 0) -> tuple[float, float]:
    """Event 단위 부트스트랩 95% 구간."""
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n, len(values)), replace=True).mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def wilson_interval(successes: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """비율의 Wilson 95% 구간. 표본이 작은 폴드에서 구간을 넓게 보이려는 용도다(기대값 Hit는 소수일 수 있어 근사)."""
    if n <= 0:
        return np.nan, np.nan
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return float(max(center - half, 0.0)), float(min(center + half, 1.0))
