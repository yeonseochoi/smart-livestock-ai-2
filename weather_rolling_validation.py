"""외부 학습기간 안에서 expanding-window rolling validation을 수행한다."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import build_odor_ai_mvp as odor
import optimize_early_prediction as optimize
import sensitivity_early_prediction as sensitivity
from weather_ablation_experiment import MODELS, evaluate
from weather_feature_engineering import (
    WEATHER_FEATURES,
    add_candidate_weather_features,
    add_timing_weather_features,
    load_timing_weather,
    load_weather,
    timing_feature_groups,
)


OUTPUT_DIR = Path("outputs/weather_rolling_validation")
GRID_M = 1000
INPUT_MINUTES = 30
FORECAST_MINUTES = 30
N_FOLDS = 5
VALIDATION_EVENTS = 15


def mean_metrics(folds: list[dict[str, float]]) -> dict[str, float]:
    keys = ["pr_auc", "topk_recall", "roc_auc", "recall_at_3", "event_hit_rate_at_3"]
    result: dict[str, float] = {}
    for key in keys:
        values = np.asarray([fold[key] for fold in folds], dtype=float)
        result[f"mean_{key}"] = float(np.nanmean(values))
        result[f"std_{key}"] = float(np.nanstd(values))
    return result


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    complaints, _, _, _, _ = odor.load_inputs()
    original_grid, _ = odor.add_grid_columns(complaints)
    _, selected_hours = odor.build_bounded_events(original_grid)
    data, outer_train_ids, outer_test_ids = sensitivity.build_data(
        complaints, selected_hours, GRID_M, INPUT_MINUTES, FORECAST_MINUTES
    )
    data = add_candidate_weather_features(data, load_weather(), GRID_M)
    data = add_timing_weather_features(data, load_timing_weather(), GRID_M)

    base = list(sensitivity.FEATURES)
    variants = {
        "complaints_only": base,
        "current_nested_windows": base + WEATHER_FEATURES,
        **{name: base + features for name, features in timing_feature_groups().items()},
    }
    outer_train = data[data["event_id"].isin(outer_train_ids)].copy()
    ordered_ids = (outer_train[["event_id", "event_hour"]].drop_duplicates()
                   .sort_values("event_hour")["event_id"].tolist())
    first_validation = len(ordered_ids) - N_FOLDS * VALIDATION_EVENTS
    if first_validation < 30:
        raise ValueError("rolling validation을 위한 초기 학습 이벤트가 부족합니다.")

    folds: list[dict[str, object]] = []
    raw_results: dict[str, dict[str, list[dict[str, float]]]] = {
        variant: {model: [] for model in MODELS} for variant in variants
    }
    for fold_index in range(N_FOLDS):
        validation_start = first_validation + fold_index * VALIDATION_EVENTS
        validation_end = validation_start + VALIDATION_EVENTS
        fit_ids = set(ordered_ids[:validation_start])
        validation_ids = set(ordered_ids[validation_start:validation_end])
        fit, validation = sensitivity.prepare_prior(
            outer_train[outer_train["event_id"].isin(fit_ids)],
            outer_train[outer_train["event_id"].isin(validation_ids)], fit_ids,
        )
        fold_info = {
            "fold": fold_index + 1,
            "fit_events": len(fit_ids),
            "validation_events": len(validation_ids),
            "validation_start": str(validation["event_hour"].min()),
            "validation_end": str(validation["event_hour"].max()),
        }
        folds.append(fold_info)
        print(f"fold {fold_index + 1}/{N_FOLDS}: {len(fit_ids)} -> {len(validation_ids)} events")
        for variant, features in variants.items():
            for model_name in MODELS:
                score, _ = optimize.fit_predict(model_name, fit, validation, features)
                raw_results[variant][model_name].append(evaluate(validation, score))

    aggregate: dict[str, dict[str, object]] = {}
    for variant, model_results in raw_results.items():
        aggregate[variant] = {}
        for model_name, fold_results in model_results.items():
            aggregate[variant][model_name] = {
                "aggregate": mean_metrics(fold_results),
                "folds": fold_results,
            }

    selected_models: dict[str, str] = {}
    for variant in variants:
        selected_models[variant] = max(MODELS, key=lambda model: (
            aggregate[variant][model]["aggregate"]["mean_pr_auc"],
            aggregate[variant][model]["aggregate"]["mean_event_hit_rate_at_3"],
            aggregate[variant][model]["aggregate"]["mean_recall_at_3"],
        ))
    selected_variant = max(variants, key=lambda variant: (
        aggregate[variant][selected_models[variant]]["aggregate"]["mean_pr_auc"],
        aggregate[variant][selected_models[variant]]["aggregate"]["mean_event_hit_rate_at_3"],
        aggregate[variant][selected_models[variant]]["aggregate"]["mean_recall_at_3"],
    ))

    baseline_model = selected_models["complaints_only"]
    baseline_folds = raw_results["complaints_only"][baseline_model]
    comparisons: dict[str, object] = {}
    for variant, model_name in selected_models.items():
        chosen_folds = raw_results[variant][model_name]
        comparisons[variant] = {
            "selected_model": model_name,
            "aggregate": aggregate[variant][model_name]["aggregate"],
            "pr_auc_fold_wins_vs_complaints": int(sum(
                candidate["pr_auc"] > baseline["pr_auc"]
                for candidate, baseline in zip(chosen_folds, baseline_folds)
            )),
            "recall_at_3_fold_wins_vs_complaints": int(sum(
                candidate["recall_at_3"] > baseline["recall_at_3"]
                for candidate, baseline in zip(chosen_folds, baseline_folds)
            )),
            "event_hit_at_3_fold_wins_vs_complaints": int(sum(
                candidate["event_hit_rate_at_3"] > baseline["event_hit_rate_at_3"]
                for candidate, baseline in zip(chosen_folds, baseline_folds)
            )),
        }

    report = {
        "protocol": "expanding-window validation using only outer training events",
        "outer_test_events_touched": False,
        "outer_train_events": len(outer_train_ids),
        "reserved_outer_test_events": len(outer_test_ids),
        "folds": folds,
        "selection": "highest mean rolling PR-AUC, then Event Hit@3 and Recall@3",
        "selected_variant": selected_variant,
        "selected_model": selected_models[selected_variant],
        "variant_features": variants,
        "comparisons": comparisons,
        "all_model_results": aggregate,
    }
    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "selected_variant": selected_variant,
        "selected_model": selected_models[selected_variant],
        "comparisons": comparisons,
    }, ensure_ascii=False, indent=2))
    print(f"산출물: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
