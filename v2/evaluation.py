"""prior 기준선 대비 개선폭과 Event 단위 부트스트랩 신뢰구간.

모든 지표를 "모델 값 - 기준선(prior만) 값"으로 계산하고, 같은 부트스트랩 표본(Event를 복원추출)으로
모델과 기준선을 함께 다시 계산해 차이의 95% 구간을 만든다(paired). 구간이 0을 포함하면 "구분 불가"로 적는다.

지표 (두 묶음을 구분해 이름에 표시한다)
  (전체)   Hit@3, Recall@3 : 예측 창에 민원이 있는 모든 Event 대상. 정답이 전부 후보 밖이면 실패(0)로 센다.
                            Recall 분모 = 후보 밖을 포함한 전체 정답 격자 수.  -> "확산 전체"를 맞히는 정도
  (후보내) Hit@3, Recall@3 : 후보 격자 안에 정답이 있는 Event만 대상 (조건부). -> "후보에 들어온 정답을 위로 올리는 정도"
  (후보내) PR-AUC, ROC-AUC : 후보 행을 모두 합쳐 계산.
  (후보내) Event내 AUC     : Event 안에서 양성 격자를 음성 격자보다 위에 놓는 정도 (Event별 AUC 평균)
동점은 기대값(무작위 동점 처리)으로 계산한다.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from config import TOPK
from evaluate import event_topk, event_topk_all

VERDICT_BETTER, VERDICT_WORSE, VERDICT_UNSURE, VERDICT_REF = "개선", "악화", "구분 불가(구간이 0 포함)", "기준"

HIT_ALL, REC_ALL = "Hit@3(전체)", "Recall@3(전체)"
HIT_C, REC_C = "Hit@3(후보내)", "Recall@3(후보내)"
AUC_EV = "Event내 AUC(후보내)"
PR, ROC = "PR-AUC(후보내)", "ROC-AUC(후보내)"
POOLED = (ROC, PR)
EVENT_METRICS = (HIT_C, REC_C, AUC_EV, HIT_ALL, REC_ALL)        # event_level 배열의 열 순서
ALL_METRICS = (*POOLED, *EVENT_METRICS)


def event_level(df: pd.DataFrame, score: np.ndarray, events: np.ndarray, min_cands: int = 0, k: int = TOPK) -> np.ndarray:
    """(len(events), 5) 배열, 열 순서 = EVENT_METRICS. 해당 없으면 NaN. df 는 reset_index 된 상태."""
    y_all = df["target"].to_numpy(int)
    n_all_col = df["n_future_all"].to_numpy(float) if "n_future_all" in df.columns else None
    out = np.full((len(events), len(EVENT_METRICS)), np.nan)
    groups = df.groupby("event_id", sort=False).indices
    for i, eid in enumerate(events):
        idx = groups.get(eid)
        if idx is None or len(idx) <= min_cands:
            continue
        y, s = y_all[idx], score[idx]
        res = event_topk(y, s, k)
        if res is not None:
            out[i, 0], out[i, 1] = res
        if 0 < y.sum() < len(y):
            out[i, 2] = roc_auc_score(y, s)
        n_all = float(n_all_col[idx[0]]) if n_all_col is not None else float(y.sum())
        res_all = event_topk_all(y, s, n_all, k)
        if res_all is not None:
            out[i, 3], out[i, 4] = res_all
    return out


def _pooled(y: np.ndarray, s: np.ndarray) -> tuple[float, float]:
    if y.min() == y.max():
        return np.nan, np.nan
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


def compare(df: pd.DataFrame, scores: dict[str, np.ndarray], reference: str, n_boot: int = 500,
            seed: int = 0, min_cands: int = 0) -> pd.DataFrame:
    """긴 형식 결과: (모델, 지표, 값, 기준선 값, 차이, 95% 하한, 상한, 판정, 평가 Event 수)."""
    df = df.reset_index(drop=True)
    events = df["event_id"].unique()
    groups = df.groupby("event_id", sort=False).indices
    idx_list = [groups[e] for e in events]
    y = df["target"].to_numpy(int)
    ev_tab = {m: event_level(df, sc, events, min_cands) for m, sc in scores.items()}

    def point(m: str) -> dict[str, float]:
        roc, ap = _pooled(y, scores[m])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            vals = np.nanmean(ev_tab[m], axis=0)
        return {ROC: roc, PR: ap, **dict(zip(EVENT_METRICS, vals))}

    pts = {m: point(m) for m in scores}
    rng = np.random.default_rng(seed)
    boot = {m: {k: [] for k in ALL_METRICS} for m in scores}
    n_ev = len(events)
    for _ in range(n_boot):
        pick = rng.integers(0, n_ev, n_ev)
        rows = np.concatenate([idx_list[i] for i in pick])
        yb = y[rows]
        for m, sc in scores.items():
            roc, ap = _pooled(yb, sc[rows])
            boot[m][ROC].append(roc); boot[m][PR].append(ap)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                v = np.nanmean(ev_tab[m][pick], axis=0)
            for name, val in zip(EVENT_METRICS, v):
                boot[m][name].append(val)

    out = []
    for m in scores:
        for metric in ALL_METRICS:
            val, ref = pts[m][metric], pts[reference][metric]
            d = np.asarray(boot[m][metric]) - np.asarray(boot[reference][metric])
            d = d[~np.isnan(d)]
            lo, hi = (np.percentile(d, 2.5), np.percentile(d, 97.5)) if len(d) else (np.nan, np.nan)
            if m == reference:
                verdict = VERDICT_REF
            elif np.isnan(lo):
                verdict = "계산 불가"
            elif lo > 0:
                verdict = VERDICT_BETTER
            elif hi < 0:
                verdict = VERDICT_WORSE
            else:
                verdict = VERDICT_UNSURE
            col = EVENT_METRICS.index(metric) if metric in EVENT_METRICS else None
            n_eval = int(np.sum(~np.isnan(ev_tab[m][:, col]))) if col is not None else len(events)
            out.append({"모델": m, "지표": metric, "값": val, "기준선 값": ref, "차이": val - ref if m != reference else 0.0,
                        "95% 하한": lo if m != reference else 0.0, "95% 상한": hi if m != reference else 0.0,
                        "판정": verdict, "평가 Event 수": n_eval})
    return pd.DataFrame(out)


MARK = {VERDICT_BETTER: "▲", VERDICT_WORSE: "▼", VERDICT_UNSURE: "○", VERDICT_REF: "", "계산 불가": "?"}


def wide(long: pd.DataFrame, metrics: list[str], order: list[str]) -> pd.DataFrame:
    """모델 x 지표 표. 각 칸: 값 (차이 [95% 구간] 표시). ▲ 개선 / ▼ 악화 / ○ 구분 불가."""
    rows = []
    for m in order:
        row = {"모델": m}
        for metric in metrics:
            r = long[(long["모델"] == m) & (long["지표"] == metric)].iloc[0]
            if r["판정"] == VERDICT_REF:
                row[metric] = f"{r['값']:.3f} (기준)"
            elif np.isnan(r["값"]):
                row[metric] = "-"
            else:
                row[metric] = f"{r['값']:.3f}  Δ{r['차이']:+.3f} [{r['95% 하한']:+.3f}, {r['95% 상한']:+.3f}] {MARK[r['판정']]}"
        rows.append(row)
    return pd.DataFrame(rows)
