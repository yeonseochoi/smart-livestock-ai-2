"""민원 기준 모델과 풍향·이동 특징 모델을 동일 시간 분할로 비교한다."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import build_odor_ai_mvp as odor
import compare_operational_grid_sizes as operational
import optimize_early_prediction as optimize
import sensitivity_early_prediction as sensitivity
from weather_feature_engineering import WEATHER_FEATURES, add_candidate_weather_features, load_weather


OUTPUT_DIR = Path("outputs/weather_ablation")
GRID_M = 1000
INPUT_MINUTES = 30
FORECAST_MINUTES = 30
MODELS = ("extra_leaf4", "xgb_d2", "xgb_d3", "rank_d2", "rank_d3")
ALPHAS = tuple(float(x) for x in np.linspace(0.0, 0.5, 11))


def evaluate(frame: pd.DataFrame, score: np.ndarray) -> dict[str, float]:
    metrics = optimize.score_prediction(frame, score)
    metrics.update(operational.fixed_k_metrics(frame, score))
    return metrics


def normalize_within_event(frame: pd.DataFrame, score: np.ndarray) -> np.ndarray:
    """분류 확률과 랭커 점수를 이벤트 내부 백분위로 맞춘다."""
    probe = pd.DataFrame({"event_id": frame["event_id"].to_numpy(), "score": score})
    return probe.groupby("event_id", sort=False)["score"].rank(method="average", pct=True).to_numpy()


def choose_model(
    fit: pd.DataFrame, valid: pd.DataFrame, features: list[str],
) -> tuple[str, dict[str, dict[str, float]], dict[str, np.ndarray]]:
    metrics: dict[str, dict[str, float]] = {}
    scores: dict[str, np.ndarray] = {}
    for name in MODELS:
        score, _ = optimize.fit_predict(name, fit, valid, features)
        scores[name] = score
        metrics[name] = evaluate(valid, score)
    selected = max(metrics, key=lambda name: (
        metrics[name]["pr_auc"], metrics[name]["event_hit_rate_at_3"], metrics[name]["recall_at_3"]
    ))
    return selected, metrics, scores


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    complaints, _, _, _, _ = odor.load_inputs()
    original_grid, _ = odor.add_grid_columns(complaints)
    _, selected_hours = odor.build_bounded_events(original_grid)
    data, train_ids, test_ids = sensitivity.build_data(
        complaints, selected_hours, GRID_M, INPUT_MINUTES, FORECAST_MINUTES
    )
    data = add_candidate_weather_features(data, load_weather(), GRID_M)

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

    baseline_features = list(sensitivity.FEATURES)
    combined_features = baseline_features + WEATHER_FEATURES
    baseline_name, baseline_validation, baseline_scores = choose_model(fit, valid, baseline_features)
    weather_name, weather_validation, weather_scores = choose_model(fit, valid, combined_features)

    alpha_metrics: dict[str, dict[str, float]] = {}
    baseline_validation_rank = normalize_within_event(valid, baseline_scores[baseline_name])
    weather_validation_rank = normalize_within_event(valid, weather_scores[weather_name])
    for alpha in ALPHAS:
        blended = ((1.0 - alpha) * baseline_validation_rank
                   + alpha * weather_validation_rank)
        alpha_metrics[f"{alpha:.2f}"] = evaluate(valid, blended)
    selected_alpha = max(ALPHAS, key=lambda alpha: (
        alpha_metrics[f"{alpha:.2f}"]["pr_auc"],
        alpha_metrics[f"{alpha:.2f}"]["event_hit_rate_at_3"],
        alpha_metrics[f"{alpha:.2f}"]["recall_at_3"],
    ))

    train, test = sensitivity.prepare_prior(train, test, train_ids)
    baseline_test, _ = optimize.fit_predict(
        baseline_name, train, test, baseline_features, final=True
    )
    weather_test, weather_model = optimize.fit_predict(
        weather_name, train, test, combined_features, final=True
    )
    baseline_test_rank = normalize_within_event(test, baseline_test)
    weather_test_rank = normalize_within_event(test, weather_test)
    final_score = ((1.0 - selected_alpha) * baseline_test_rank
                   + selected_alpha * weather_test_rank)
    test["baseline_score"] = baseline_test
    test["weather_score"] = weather_test
    test["final_weather_blend_score"] = final_score

    report = {
        "protocol": "1km, 30-minute input/forecast, chronological 70/30 outer split",
        "selection": "inner temporal validation; PR-AUC, then Event Hit@3 and Recall@3",
        "train_events": len(train_ids),
        "test_events": len(test_ids),
        "baseline_model": baseline_name,
        "weather_model": weather_name,
        "selected_alpha": selected_alpha,
        "weather_features": WEATHER_FEATURES,
        "inner_validation": {
            "baseline_models": baseline_validation,
            "weather_models": weather_validation,
            "blend_alphas": alpha_metrics,
        },
        "locked_test": {
            "baseline": evaluate(test, baseline_test),
            "weather_only": evaluate(test, weather_test),
            "selected_blend": evaluate(test, final_score),
        },
        "weather_feature_importance": dict(
            sorted(zip(combined_features, map(float, weather_model.feature_importances_)),
                   key=lambda item: item[1], reverse=True)
        ) if hasattr(weather_model, "feature_importances_") else {},
    }
    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    test.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(report["locked_test"], ensure_ascii=False, indent=2))
    print(f"선택 alpha: {selected_alpha:.2f}")
    print(f"산출물: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
