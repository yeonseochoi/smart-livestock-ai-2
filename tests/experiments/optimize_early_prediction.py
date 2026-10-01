"""향후 30분 신고 Grid 예측에 사용하는 모델 라이브러리.

`compare_operational_grid_sizes.py`와 `sensitivity_early_prediction.py`가
`fit_predict`와 `score_prediction`을 가져다 쓴다. 단독 실행 대상이 아니다.

기상 자료로 후보를 구성하던 최적화 실험은 현재 기획(민원 데이터만 사용)에서
제외되었고 필요한 모듈도 남아 있지 않아 제거했다.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier, XGBRanker

import build_odor_ai_mvp as odor


ADVANCED_FEATURES = [
    "spatial_smooth_prior", "hour_prior", "season_prior", "weekend_prior",
    "odor_prior", "recent_prior", "wind_sector_prior_advanced",
]


def count_maps(
    reference_ids: set[str], context: dict[str, dict[str, object]],
) -> tuple[Counter, dict[str, defaultdict[object, Counter]], dict[str, Counter]]:
    global_counts: Counter = Counter()
    conditional = {
        "hour_bin": defaultdict(Counter), "season": defaultdict(Counter),
        "weekend": defaultdict(Counter), "odor_type": defaultdict(Counter),
        "wind_sector": defaultdict(Counter),
    }
    denominators = {name: Counter() for name in conditional}
    for event_id in reference_ids:
        meta = context[event_id]
        cells = meta["future_cells"]
        global_counts.update(cells)
        for name in conditional:
            value = meta[name]
            conditional[name][value].update(cells)
            denominators[name][value] += 1
    return global_counts, conditional, denominators


def recent_prior_maps(
    target_ids: set[str], reference_ids: set[str], context: dict[str, dict[str, object]],
) -> dict[str, dict[tuple[int, int], float]]:
    result: dict[str, dict[tuple[int, int], float]] = {}
    for target_id in target_ids:
        target_time = context[target_id]["hour"]
        weighted: defaultdict[tuple[int, int], float] = defaultdict(float)
        total_weight = 0.0
        for reference_id in reference_ids:
            reference_time = context[reference_id]["hour"]
            if reference_id == target_id or reference_time >= target_time:
                continue
            age_days = max((target_time - reference_time).total_seconds() / 86400, 0.0)
            weight = math.exp(-age_days / 730.0)
            total_weight += weight
            for cell in context[reference_id]["future_cells"]:
                weighted[cell] += weight
        result[target_id] = {
            cell: value / total_weight for cell, value in weighted.items()
        } if total_weight else {}
    return result


def apply_advanced_priors(
    frame: pd.DataFrame, reference_ids: set[str], context: dict[str, dict[str, object]],
    leave_one_out: bool,
) -> pd.DataFrame:
    result = frame.copy()
    global_counts, conditional, denominators = count_maps(reference_ids, context)
    reference_n = len(reference_ids)
    recent = recent_prior_maps(set(result["event_id"]), reference_ids, context)
    rows: list[list[float]] = []
    smoothing = 8.0
    for row in result.itertuples(index=False):
        event_id = row.event_id
        cell = (int(row.grid_x), int(row.grid_y))
        meta = context[event_id]
        self_active = int(leave_one_out and event_id in reference_ids and cell in meta["future_cells"])
        denominator = reference_n - int(leave_one_out and event_id in reference_ids)
        global_prior = (global_counts[cell] - self_active) / max(denominator, 1)

        smooth_values = []
        smooth_weights = []
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                distance = math.hypot(dx, dy)
                if distance > 2:
                    continue
                neighbor = (cell[0] + dx, cell[1] + dy)
                neighbor_self = int(
                    leave_one_out and event_id in reference_ids and neighbor in meta["future_cells"]
                )
                smooth_values.append((global_counts[neighbor] - neighbor_self) / max(denominator, 1))
                smooth_weights.append(1 / (1 + distance))
        spatial_prior = float(np.average(smooth_values, weights=smooth_weights))

        conditional_values = []
        for name in ("hour_bin", "season", "weekend", "odor_type", "wind_sector"):
            value = meta[name]
            group_n = denominators[name][value] - int(leave_one_out and event_id in reference_ids)
            count = conditional[name][value][cell] - self_active
            conditional_values.append((count + smoothing * global_prior) / max(group_n + smoothing, 1))
        rows.append([
            global_prior, spatial_prior, *conditional_values[:4],
            recent[event_id].get(cell, global_prior), conditional_values[4],
        ])
    columns = ["prior", *ADVANCED_FEATURES]
    result.loc[:, columns] = np.asarray(rows, dtype=float)
    return result


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


def candidate_models() -> list[str]:
    return [
        "extra_leaf2", "extra_leaf4", "rf_leaf2", "rf_leaf4",
        "xgb_d2", "xgb_d3", "xgb_d4", "rank_d2", "rank_d3", "rank_d4",
    ]

