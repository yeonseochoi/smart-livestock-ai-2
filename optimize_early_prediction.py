"""향후 30분 신고 Grid 예측을 고급 prior와 Learning-to-Rank로 최적화한다."""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier, XGBRanker

import build_odor_ai_mvp as odor


OUTPUT_DIR = Path("outputs/early_prediction_optimization")
ADVANCED_FEATURES = [
    "spatial_smooth_prior", "hour_prior", "season_prior", "weekend_prior",
    "odor_prior", "recent_prior", "wind_sector_prior_advanced",
]


def event_context(events: pd.DataFrame, weather: dict[str, dict[str, float]]) -> dict[str, dict[str, object]]:
    context: dict[str, dict[str, object]] = {}
    for event_id, group in events.groupby("event_id"):
        hour = pd.Timestamp(group["event_hour"].iloc[0])
        initial = group[group["datetime"] < hour + pd.Timedelta(minutes=odor.INPUT_MINUTES)]
        future = group[group["datetime"] >= hour + pd.Timedelta(minutes=odor.INPUT_MINUTES)]
        odor_type = str(initial["odor_type"].mode().iloc[0]) if not initial.empty else "unknown"
        context[event_id] = {
            "hour": hour,
            "hour_bin": hour.hour // 4,
            "season": (hour.month % 12) // 3,
            "weekend": int(hour.dayofweek >= 5),
            "odor_type": odor_type,
            "wind_sector": weather_exp.wind_sector(weather[event_id], 12),
            "future_cells": set(zip(future["grid_x"].astype(int), future["grid_y"].astype(int))),
        }
    return context


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
    return model.predict(validation[features]), model


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


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    complaints, _, _, _, _ = odor.load_inputs()
    complaints, _ = odor.add_grid_columns(complaints)
    events, summary = odor.build_bounded_events(complaints)
    weather_frame = pd.read_csv(weather_exp.WEATHER_FILE, encoding="utf-8-sig")
    weather = weather_exp.weather_lookup(weather_frame)
    candidates, train_ids, test_ids = weather_exp.build_candidates(events, summary, weather)
    context = event_context(events, weather)
    train = candidates[candidates["is_train"]].copy()
    test = candidates[~candidates["is_train"]].copy()
    ordered_ids = (train[["event_id", "event_hour"]].drop_duplicates()
                   .sort_values("event_hour")["event_id"].tolist())
    cut = max(1, int(len(ordered_ids) * 0.8))
    fit_ids, validation_ids = set(ordered_ids[:cut]), set(ordered_ids[cut:])
    fit_raw = train[train["event_id"].isin(fit_ids)].copy()
    validation_raw = train[train["event_id"].isin(validation_ids)].copy()
    fit = apply_advanced_priors(fit_raw, fit_ids, context, leave_one_out=True)
    validation = apply_advanced_priors(validation_raw, fit_ids, context, leave_one_out=False)

    feature_sets = {
        "advanced": weather_exp.BASE_FEATURES + ADVANCED_FEATURES,
        "advanced_weather": weather_exp.BASE_FEATURES + ADVANCED_FEATURES
                            + weather_exp.DIRECTION_FEATURES + weather_exp.REGIME_FEATURES,
    }
    validation_results: dict[str, dict[str, dict[str, float]]] = {}
    for feature_name, features in feature_sets.items():
        validation_results[feature_name] = {}
        for model_name in candidate_models():
            score, _ = fit_predict(model_name, fit, validation, features)
            validation_results[feature_name][model_name] = score_prediction(validation, score)
            print(f"검증 {feature_name}/{model_name}: "
                  f"PR {validation_results[feature_name][model_name]['pr_auc']:.3f}, "
                  f"Top-K {validation_results[feature_name][model_name]['topk_recall']:.3f}")

    choices = [
        (metrics["pr_auc"], metrics["topk_recall"], feature_name, model_name)
        for feature_name, models in validation_results.items()
        for model_name, metrics in models.items()
    ]
    _, _, pr_features_name, pr_model_name = max(choices)
    _, _, topk_features_name, topk_model_name = max(
        (metrics["topk_recall"], metrics["pr_auc"], feature_name, model_name)
        for feature_name, models in validation_results.items()
        for model_name, metrics in models.items()
    )

    full_train = apply_advanced_priors(train, train_ids, context, leave_one_out=True)
    final_test = apply_advanced_priors(test, train_ids, context, leave_one_out=False)
    selected = {
        "pr_selected": (pr_features_name, pr_model_name),
        "topk_selected": (topk_features_name, topk_model_name),
    }
    test_results: dict[str, object] = {}
    prediction = final_test[["event_id", "event_hour", "grid_x", "grid_y", "target"]].copy()
    for objective, (feature_name, model_name) in selected.items():
        features = feature_sets[feature_name]
        score, model = fit_predict(model_name, full_train, final_test, features, final=True)
        prediction[f"{objective}_score"] = score
        test_results[objective] = {
            "feature_set": feature_name,
            "model": model_name,
            **score_prediction(final_test, score),
        }

    current = json.loads(Path("outputs/odor_ai_mvp/model_metrics.json").read_text(encoding="utf-8"))["early_prediction"]
    report = {
        "experiment": "advanced_prior_and_learning_to_rank",
        "selection_policy": "inner temporal validation only; final future test used once",
        "train_events": len(train_ids),
        "test_events": len(test_ids),
        "validation_results": validation_results,
        "selected": selected,
        "test_results": test_results,
        "current_model": {
            "pr_auc": current["selected_model_metrics"]["pr_auc"],
            "roc_auc": current["selected_model_metrics"]["roc_auc"],
            "topk_recall": current["model_topk_recall"],
        },
    }
    (OUTPUT_DIR / "optimization_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    prediction.to_csv(OUTPUT_DIR / "optimization_test_predictions.csv", index=False, encoding="utf-8-sig")

    print("===== 조기예측 최적화 결과 =====")
    print(f"현재 모델: PR {report['current_model']['pr_auc']:.3f}, "
          f"ROC {report['current_model']['roc_auc']:.3f}, Top-K {report['current_model']['topk_recall']:.3f}")
    for objective, result in test_results.items():
        print(f"{objective}: {result['feature_set']}/{result['model']} - "
              f"PR {result['pr_auc']:.3f}, ROC {result['roc_auc']:.3f}, Top-K {result['topk_recall']:.3f}")
    print(f"산출물: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
