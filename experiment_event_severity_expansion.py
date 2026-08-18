"""Test weak/medium Event expansion with episode-safe temporal splitting."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import build_odor_ai_mvp as odor
import compare_operational_grid_sizes as operational
import optimize_early_prediction as optimize
import sensitivity_early_prediction as sensitivity


OUTPUT_DIR = Path("outputs/event_severity_expansion")
GRID_M = 1000
MODELS = ("extra_leaf4", "xgb_d2", "xgb_d3", "rank_d2", "rank_d3")


def distance_km(a_lat: float, a_lon: float, b_lat: float, b_lon: float) -> float:
    lat = math.radians((a_lat + b_lat) / 2)
    dx = (b_lon - a_lon) * 111.320 * math.cos(lat)
    dy = (b_lat - a_lat) * 110.540
    return math.hypot(dx, dy)


def make_event_catalog(complaints: pd.DataFrame) -> pd.DataFrame:
    work = complaints.copy()
    work["event_hour"] = work["datetime"].dt.floor("h")
    hourly = work.groupby("event_hour").agg(
        reports=("datetime", "size"), unique_grids=("grid_id", "nunique"),
        center_latitude=("latitude", "mean"), center_longitude=("longitude", "mean"),
    ).reset_index()
    hourly = hourly[(hourly["reports"] >= 5) & (hourly["unique_grids"] >= 4)].copy()
    hourly["severity"] = "weak"
    hourly.loc[(hourly["reports"] >= 6) & (hourly["unique_grids"] >= 5), "severity"] = "medium"
    hourly.loc[(hourly["reports"] >= 10) & (hourly["unique_grids"] >= 5), "severity"] = "strong"
    hourly = hourly.sort_values("event_hour").reset_index(drop=True)
    hourly["event_id"] = [f"SEV-{index:04d}" for index in range(1, len(hourly) + 1)]

    episodes, episode = [], 0
    previous = None
    for row in hourly.itertuples(index=False):
        continuing = False
        if previous is not None:
            gap = (row.event_hour - previous.event_hour).total_seconds() / 3600
            spatial = distance_km(
                previous.center_latitude, previous.center_longitude,
                row.center_latitude, row.center_longitude,
            )
            continuing = gap <= 2 and spatial <= 3
        if not continuing:
            episode += 1
        episodes.append(f"EP-{episode:04d}")
        previous = row
    hourly["episode_id"] = episodes
    return hourly


def chronological_episode_split(catalog: pd.DataFrame, ratio: float) -> tuple[set[str], set[str]]:
    episodes = catalog.groupby("episode_id")["event_hour"].min().sort_values().index.tolist()
    cut = max(1, int(len(episodes) * ratio))
    return set(episodes[:cut]), set(episodes[cut:])


def filter_training(frame: pd.DataFrame, config: str) -> pd.DataFrame:
    if config == "strong_only":
        return frame[frame["severity"] == "strong"].copy()
    if config == "strong_medium":
        return frame[frame["severity"].isin(["strong", "medium"])].copy()
    if config == "all_severities":
        return frame.copy()
    raise ValueError(config)


def fit_with_prior(
    fit: pd.DataFrame, evaluation: pd.DataFrame, fit_ids: set[str], model_name: str,
    final: bool = False,
) -> tuple[np.ndarray, object]:
    fit, evaluation = sensitivity.prepare_prior(fit, evaluation, fit_ids)
    return optimize.fit_predict(model_name, fit, evaluation, sensitivity.FEATURES, final=final)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    complaints, _, _, _, _ = odor.load_inputs()
    complaints, _ = odor.add_grid_columns(complaints)
    catalog = make_event_catalog(complaints)
    data, _, _ = sensitivity.build_data(
        complaints, catalog[["event_hour", "event_id"]], GRID_M, 30, 30
    )
    data = data.drop(columns="is_train").merge(
        catalog[["event_id", "event_hour", "severity", "episode_id", "reports", "unique_grids"]],
        on=["event_id", "event_hour"], how="left",
    )
    valid_catalog = catalog[catalog["event_id"].isin(data["event_id"].unique())].copy()
    train_episodes, test_episodes = chronological_episode_split(valid_catalog, .7)
    outer_train = data[data["episode_id"].isin(train_episodes)].copy()
    strong_test = data[
        data["episode_id"].isin(test_episodes) & (data["severity"] == "strong")
    ].copy()
    if strong_test["event_id"].nunique() < 10:
        raise RuntimeError("Too few future strong Events for a stable comparison.")

    inner_catalog = valid_catalog[valid_catalog["episode_id"].isin(train_episodes)]
    fit_episodes, validation_episodes = chronological_episode_split(inner_catalog, .8)
    strong_validation = outer_train[
        outer_train["episode_id"].isin(validation_episodes)
        & (outer_train["severity"] == "strong")
    ].copy()

    configs = ("strong_only", "strong_medium", "all_severities")
    validation_results = {}
    choices = []
    for config in configs:
        fit = filter_training(outer_train[outer_train["episode_id"].isin(fit_episodes)], config)
        fit_ids = set(fit["event_id"])
        model_results = {}
        for model_name in MODELS:
            score, _ = fit_with_prior(fit, strong_validation.copy(), fit_ids, model_name)
            model_results[model_name] = optimize.score_prediction(strong_validation, score)
        selected_model = max(model_results, key=lambda name: (
            model_results[name]["pr_auc"], model_results[name]["topk_recall"]
        ))
        validation_results[config] = {
            "fit_events": int(fit["event_id"].nunique()),
            "selected_model": selected_model, "models": model_results,
        }
        selected_metrics = model_results[selected_model]
        choices.append((selected_metrics["pr_auc"], selected_metrics["topk_recall"], config, selected_model))

    _, _, selected_config, selected_model = max(choices)
    test_results = {}
    test_predictions = None
    for config in configs:
        train = filter_training(outer_train, config)
        config_model = validation_results[config]["selected_model"]
        score, _ = fit_with_prior(train, strong_test.copy(), set(train["event_id"]), config_model, final=True)
        metrics = optimize.score_prediction(strong_test, score)
        metrics.update(operational.fixed_k_metrics(strong_test, score))
        metrics["training_events"] = int(train["event_id"].nunique())
        metrics["training_episodes"] = int(train["episode_id"].nunique())
        test_results[config] = {"model": config_model, "metrics": metrics}
        if config == selected_config:
            test_predictions = strong_test[[
                "event_id", "episode_id", "event_hour", "grid_x", "grid_y", "target"
            ]].copy()
            test_predictions["score"] = score

    counts = (valid_catalog.groupby("severity").agg(
        events=("event_id", "nunique"), episodes=("episode_id", "nunique")
    ).to_dict("index"))
    report = {
        "protocol": {
            "grid_m": GRID_M, "input_minutes": 30, "forecast_minutes": 30,
            "eligible_event": "at least 5 reports and 4 grids in one hour",
            "episode_rule": "consecutive eligible hours within 2 hours and 3 km",
            "outer_split": "chronological 70/30 by episode",
            "selection": "training severity configuration and model selected on inner strong-Event PR-AUC",
            "final_evaluation": "future strong Events only",
        },
        "catalog": {
            "eligible_hourly_events": len(catalog), "valid_prediction_events": len(valid_catalog),
            "episodes": int(valid_catalog["episode_id"].nunique()), "severity_counts": counts,
            "outer_train_episodes": len(train_episodes), "outer_test_episodes": len(test_episodes),
            "strong_test_events": int(strong_test["event_id"].nunique()),
        },
        "selected_configuration": selected_config, "selected_model": selected_model,
        "inner_validation": validation_results,
        "future_strong_test_ablation": test_results,
    }
    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    assert test_predictions is not None
    test_predictions.to_csv(OUTPUT_DIR / "selected_test_predictions.csv", index=False, encoding="utf-8-sig")
    print(json.dumps({
        "catalog": report["catalog"], "selected_configuration": selected_config,
        "selected_model": selected_model, "test_ablation": test_results,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
