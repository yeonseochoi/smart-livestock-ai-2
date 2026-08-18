"""Compare 1 km, 1.5 km, and 2 km grids under one temporal test protocol."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import build_odor_ai_mvp as odor
import optimize_early_prediction as optimize
import sensitivity_early_prediction as sensitivity


OUTPUT_DIR = Path("outputs/operational_grid_comparison")
GRID_SIZES = (1000, 1500, 2000)
MODELS = ("extra_leaf4", "xgb_d2", "xgb_d3", "rank_d2", "rank_d3")


def fixed_k_metrics(frame: pd.DataFrame, score: np.ndarray, k: int = 3) -> dict[str, float]:
    probe = frame[["event_id", "target"]].copy()
    probe["score"] = score
    recalls, hits, inspected = [], [], []
    for _, group in probe.groupby("event_id"):
        positives = int(group["target"].sum())
        if positives == 0:
            continue
        chosen = group.nlargest(min(k, len(group)), "score")
        found = int(chosen["target"].sum())
        recalls.append(found / positives)
        hits.append(float(found > 0))
        inspected.append(len(chosen))
    return {
        f"recall_at_{k}": float(np.mean(recalls)),
        f"event_hit_rate_at_{k}": float(np.mean(hits)),
        "mean_inspected_cells": float(np.mean(inspected)),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    complaints, _, _, _, _ = odor.load_inputs()
    original, _ = odor.add_grid_columns(complaints)
    _, selected_hours = odor.build_bounded_events(original)

    datasets = {}
    for grid_m in GRID_SIZES:
        data, _, _ = sensitivity.build_data(complaints, selected_hours, grid_m, 30, 30)
        datasets[grid_m] = data
    common_ids = set.intersection(*(set(data["event_id"]) for data in datasets.values()))
    common_summary = (selected_hours[selected_hours["event_id"].isin(common_ids)]
                      .sort_values("event_hour").drop_duplicates("event_id"))
    train_ids, test_ids = odor.split_event_ids(common_summary)

    report = {
        "protocol": "common events, chronological 70/30 split, 30-minute input and forecast",
        "common_events": len(common_ids), "train_events": len(train_ids), "test_events": len(test_ids),
        "grids": {},
    }
    prediction_rows = []
    for grid_m, raw in datasets.items():
        data = raw[raw["event_id"].isin(common_ids)].copy()
        data["is_train"] = data["event_id"].isin(train_ids)
        train, test = data[data["is_train"]].copy(), data[~data["is_train"]].copy()
        ordered = (train[["event_id", "event_hour"]].drop_duplicates()
                   .sort_values("event_hour")["event_id"].tolist())
        cut = max(1, int(len(ordered) * .8))
        fit_ids = set(ordered[:cut])
        fit, valid = sensitivity.prepare_prior(
            train[train["event_id"].isin(fit_ids)],
            train[~train["event_id"].isin(fit_ids)], fit_ids,
        )
        validation = {}
        for model_name in MODELS:
            score, _ = optimize.fit_predict(model_name, fit, valid, sensitivity.FEATURES)
            validation[model_name] = optimize.score_prediction(valid, score)
        selected = max(validation, key=lambda name: (
            validation[name]["pr_auc"], validation[name]["topk_recall"]
        ))
        train, test = sensitivity.prepare_prior(train, test, train_ids)
        score, _ = optimize.fit_predict(selected, train, test, sensitivity.FEATURES, final=True)
        metrics = optimize.score_prediction(test, score)
        operations = fixed_k_metrics(test, score)
        positive_rate = float(test["target"].mean())
        metrics.update(operations)
        metrics.update({
            "positive_rate": positive_rate,
            "pr_auc_lift_over_prevalence": metrics["pr_auc"] / max(positive_rate, 1e-9),
            "mean_candidates_per_event": float(test.groupby("event_id").size().mean()),
            "mean_positive_cells_per_event": float(test.groupby("event_id")["target"].sum().mean()),
            "top3_inspection_area_km2": 3 * (grid_m / 1000) ** 2,
        })
        report["grids"][str(grid_m)] = {
            "selected_model": selected, "inner_validation": validation, "test": metrics,
        }
        output = test[["event_id", "event_hour", "grid_x", "grid_y", "target"]].copy()
        output["score"] = score
        output["grid_m"] = grid_m
        prediction_rows.append(output)
        print(grid_m, selected, json.dumps(metrics, ensure_ascii=False))

    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.concat(prediction_rows).to_csv(
        OUTPUT_DIR / "test_predictions.csv", index=False, encoding="utf-8-sig"
    )


if __name__ == "__main__":
    main()
