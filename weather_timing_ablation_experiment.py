"""기존 중첩 기상창과 이벤트 전·후 분리 기상창을 비교한다."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

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


OUTPUT_DIR = Path("outputs/weather_timing_ablation")
GRID_M = 1000
INPUT_MINUTES = 30
FORECAST_MINUTES = 30


def choose_for_variant(
    fit: pd.DataFrame, valid: pd.DataFrame, features: list[str],
) -> tuple[str, dict[str, dict[str, float]]]:
    results: dict[str, dict[str, float]] = {}
    for model_name in MODELS:
        score, _ = optimize.fit_predict(model_name, fit, valid, features)
        results[model_name] = evaluate(valid, score)
    selected = max(results, key=lambda name: (
        results[name]["pr_auc"], results[name]["event_hit_rate_at_3"],
        results[name]["recall_at_3"],
    ))
    return selected, results


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    complaints, _, _, _, _ = odor.load_inputs()
    original_grid, _ = odor.add_grid_columns(complaints)
    _, selected_hours = odor.build_bounded_events(original_grid)
    data, train_ids, test_ids = sensitivity.build_data(
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

    train = data[data["is_train"]].copy()
    test = data[~data["is_train"]].copy()
    ordered = (train[["event_id", "event_hour"]].drop_duplicates()
               .sort_values("event_hour")["event_id"].tolist())
    cut = max(1, int(len(ordered) * 0.8))
    fit_ids = set(ordered[:cut])
    fit, valid = sensitivity.prepare_prior(
        train[train["event_id"].isin(fit_ids)],
        train[~train["event_id"].isin(fit_ids)], fit_ids,
    )

    inner: dict[str, object] = {}
    selected_models: dict[str, str] = {}
    for variant, features in variants.items():
        model, model_results = choose_for_variant(fit, valid, features)
        selected_models[variant] = model
        inner[variant] = {"selected_model": model, "models": model_results}

    selected_variant = max(variants, key=lambda variant: (
        inner[variant]["models"][selected_models[variant]]["pr_auc"],
        inner[variant]["models"][selected_models[variant]]["event_hit_rate_at_3"],
        inner[variant]["models"][selected_models[variant]]["recall_at_3"],
    ))

    train, test = sensitivity.prepare_prior(train, test, train_ids)
    locked_test: dict[str, dict[str, object]] = {}
    selected_model_object = None
    for variant, features in variants.items():
        score, model = optimize.fit_predict(
            selected_models[variant], train, test, features, final=True
        )
        test[f"score_{variant}"] = score
        locked_test[variant] = {
            "selected_model": selected_models[variant],
            **evaluate(test, score),
        }
        if variant == selected_variant:
            selected_model_object = model

    selected_features = variants[selected_variant]
    report = {
        "protocol": "1km, 30-minute input/forecast, chronological outer split",
        "selection": "variant and model selected only on inner temporal validation PR-AUC",
        "window_definitions": {
            "current_nested_windows": "30/60/120 minutes ending at prediction time T",
            "pre30": "event_hour-30 minutes through event_hour-1 minute",
            "pre60": "event_hour-60 minutes through event_hour-1 minute",
            "initial30": "event_hour through prediction time T-1 minute",
        },
        "train_events": len(train_ids),
        "test_events": len(test_ids),
        "selected_variant": selected_variant,
        "selected_model": selected_models[selected_variant],
        "variant_features": variants,
        "inner_validation": inner,
        "locked_test": locked_test,
        "selected_feature_importance": dict(
            sorted(zip(selected_features, map(float, selected_model_object.feature_importances_)),
                   key=lambda item: item[1], reverse=True)
        ) if selected_model_object is not None and hasattr(selected_model_object, "feature_importances_") else {},
    }
    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    test.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False, encoding="utf-8-sig")
    print(json.dumps({
        "selected_variant": selected_variant,
        "selected_model": selected_models[selected_variant],
        "locked_test": locked_test,
    }, ensure_ascii=False, indent=2))
    print(f"산출물: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
