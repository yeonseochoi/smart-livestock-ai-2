"""향후 30분 신고 Grid 예측에 사용하는 모델 라이브러리.

`compare_operational_grid_sizes.py`, `sensitivity_early_prediction.py`, `run_ablation.py`가
`fit_predict`와 `score_prediction`을 가져다 쓴다. 단독 실행 대상이 아니다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier, XGBRanker

import build_odor_ai_mvp as odor


def make_ranker(name: str, final: bool = False) -> XGBRanker:
    depth = int(name[-1])
    return XGBRanker(
        objective="rank:pairwise",
        eval_metric="ndcg",
        n_estimators=650 if final else 350,
        max_depth=depth,
        learning_rate={2: 0.035, 3: 0.03, 4: 0.025}[depth],
        min_child_weight={2: 8, 3: 7, 4: 10}[depth],
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.15,
        reg_lambda=2.0,
        random_state=odor.RANDOM_STATE,
        n_jobs=4,
    )


def grouped(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
    ordered = frame.sort_values(["event_hour", "event_id", "grid_x", "grid_y"]).copy()
    groups = ordered.groupby("event_id", sort=False).size().tolist()
    return ordered, groups


def fit_predict(
    model_name: str, fit: pd.DataFrame, validation: pd.DataFrame, features: list[str], final: bool = False,
) -> tuple[np.ndarray, object]:
    if model_name.startswith("rank_d"):
        fit_ordered, groups = grouped(fit)
        model = make_ranker(model_name, final=final)
        model.fit(fit_ordered[features], fit_ordered["target"], group=groups)
        # 랭커는 확률을 모형화하지 않는다. predict 출력이 곧 Event 내부 순위 점수다.
        return model.predict(validation[features]), model
    else:
        if model_name.startswith("xgb_"):
            positive = max(int(fit["target"].sum()), 1)
            configurations = {
                "xgb_d2": {"max_depth": 2, "min_child_weight": 8, "learning_rate": 0.035},
                "xgb_d3": {"max_depth": 3, "min_child_weight": 7, "learning_rate": 0.03},
                "xgb_d4": {"max_depth": 4, "min_child_weight": 10, "learning_rate": 0.025},
            }
            model = XGBClassifier(
                n_estimators=650 if final else 350,
                **configurations[model_name],
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.15,
                reg_lambda=2.0,
                scale_pos_weight=(len(fit) - positive) / positive,
                objective="binary:logistic",
                eval_metric="aucpr",
                random_state=odor.RANDOM_STATE,
                n_jobs=4,
            )
        else:
            model = odor.make_candidate_model(model_name, final=final)
        model.fit(fit[features], fit["target"])
    # 분류기의 predict는 0.5에서 자른 0/1 라벨이라 PR-AUC, Top-K Recall 같은
    # 순위 기반 지표에 쓸 수 없다. 양성 확률을 그대로 순위 점수로 사용한다.
    return model.predict_proba(validation[features])[:, 1], model


def score_prediction(frame: pd.DataFrame, score: np.ndarray) -> dict[str, float]:
    probe = frame[["event_id", "target"]].copy()
    probe["score"] = score
    return {
        "pr_auc": float(average_precision_score(probe["target"], score)),
        "topk_recall": odor._topk_recall(probe, "score"),
        "roc_auc": odor._safe_metrics(probe["target"].to_numpy(), score)["roc_auc"],
    }

