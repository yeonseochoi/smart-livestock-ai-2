"""Grid 크기와 입력·예측 시간을 내부 검증으로 선택하는 민감도 실험."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

import build_odor_ai_mvp as odor
import optimize_early_prediction as optimize


OUTPUT_DIR = Path("outputs/early_prediction_sensitivity")
FEATURES = [
    "initial_count", "initial_intensity", "min_distance", "neighbor_count", "radius3_count",
    "prior", "hour_sin", "hour_cos", "centroid_distance", "nearest_intensity",
    "weighted_intensity", "observed_grid_count", "observed_report_count", "first15_count", "growth",
]


def add_grid(df: pd.DataFrame, grid_m: int) -> pd.DataFrame:
    result = df.copy()
    lat0 = float(result["latitude"].median())
    lon0 = float(result["longitude"].median())
    x_m = (result["longitude"] - lon0) * 111_320 * math.cos(math.radians(lat0))
    y_m = (result["latitude"] - lat0) * 110_540
    result["grid_x"] = np.floor(x_m / grid_m).astype(int)
    result["grid_y"] = np.floor(y_m / grid_m).astype(int)
    result["grid_id"] = result["grid_x"].astype(str) + ":" + result["grid_y"].astype(str)
    return result


def prior(events: pd.DataFrame, ids: set[str], input_minutes: int) -> dict[tuple[int, int], float]:
    work = events[
        events["event_id"].isin(ids)
        & (events["datetime"] >= events["event_hour"] + pd.Timedelta(minutes=input_minutes))
    ]
    active = work.groupby(["grid_x", "grid_y"])["event_id"].nunique() / max(len(ids), 1)
    return {(int(x), int(y)): float(value) for (x, y), value in active.items()}


def build_data(
    complaints: pd.DataFrame, selected_hours: pd.DataFrame, grid_m: int,
    input_minutes: int, forecast_minutes: int,
) -> tuple[pd.DataFrame, set[str], set[str]]:
    work = add_grid(complaints, grid_m)
    mapping = selected_hours[["event_hour", "event_id"]]
    # Event 시작 이후 한 시간 안의 원시 민원을 동일한 Event 샘플로 사용한다.
    work["event_hour"] = work["datetime"].dt.floor("h")
    work = work.merge(mapping, on="event_hour", how="inner")
    valid_ids = []
    for event_id, group in work.groupby("event_id"):
        hour = pd.Timestamp(group["event_hour"].iloc[0])
        initial = group[(group["datetime"] >= hour) & (group["datetime"] < hour + pd.Timedelta(minutes=input_minutes))]
        future = group[(group["datetime"] >= hour + pd.Timedelta(minutes=input_minutes))
                       & (group["datetime"] < hour + pd.Timedelta(minutes=input_minutes + forecast_minutes))]
        if initial["grid_id"].nunique() >= 2 and future["grid_id"].nunique() >= 1:
            valid_ids.append(event_id)
    valid_summary = selected_hours[selected_hours["event_id"].isin(valid_ids)].copy()
    train_ids, test_ids = odor.split_event_ids(valid_summary)
    outer_prior = prior(work, train_ids, input_minutes)
    radius = max(1, int(round(1500 / grid_m)))
    rows = []
    for event_id, group in work[work["event_id"].isin(valid_ids)].groupby("event_id", sort=False):
        hour = pd.Timestamp(group["event_hour"].iloc[0])
        initial = group[(group["datetime"] >= hour) & (group["datetime"] < hour + pd.Timedelta(minutes=input_minutes))].copy()
        future = group[(group["datetime"] >= hour + pd.Timedelta(minutes=input_minutes))
                       & (group["datetime"] < hour + pd.Timedelta(minutes=input_minutes + forecast_minutes))]
        initial["is_first15"] = initial["datetime"] < hour + pd.Timedelta(minutes=min(15, input_minutes / 2))
        observed = set(zip(initial["grid_x"].astype(int), initial["grid_y"].astype(int)))
        target = set(zip(future["grid_x"].astype(int), future["grid_y"].astype(int)))
        agg = initial.groupby(["grid_x", "grid_y"]).agg(
            count=("datetime", "size"), mean_intensity=("intensity", "mean"),
            first15_count=("is_first15", "sum"),
        ).reset_index()
        for gx, gy in odor.candidate_cells(observed, radius=radius):
            values = odor.cell_features((gx, gy), agg, hour, outer_prior, include_trend=True)
            rows.append([event_id, event_id in train_ids, hour, gx, gy, int((gx, gy) in target), *values])
    columns = ["event_id", "is_train", "event_hour", "grid_x", "grid_y", "target", *FEATURES]
    return pd.DataFrame(rows, columns=columns), train_ids, test_ids


def prepare_prior(
    fit: pd.DataFrame, validation: pd.DataFrame, fit_ids: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    active = (fit[fit["target"] == 1]
              .groupby(["grid_x", "grid_y"])["event_id"].nunique() / max(len(fit_ids), 1))
    p = {(int(x), int(y)): float(value) for (x, y), value in active.items()}
    fit = fit.copy()
    validation = validation.copy()
    for frame in (fit, validation):
        frame["prior"] = [p.get((int(x), int(y)), 0.0) for x, y in zip(frame["grid_x"], frame["grid_y"])]
    return fit, validation


def validate_configuration(
    data: pd.DataFrame, events: pd.DataFrame, train_ids: set[str], input_minutes: int,
) -> dict[str, dict[str, float]]:
    train = data[data["is_train"]].copy()
    ordered = (train[["event_id", "event_hour"]].drop_duplicates().sort_values("event_hour")["event_id"].tolist())
    cut = max(1, int(len(ordered) * 0.8))
    fit_ids, validation_ids = set(ordered[:cut]), set(ordered[cut:])
    fit, validation = prepare_prior(
        train[train["event_id"].isin(fit_ids)], train[train["event_id"].isin(validation_ids)],
        fit_ids,
    )
    results = {}
    for model_name in ("extra_leaf4", "xgb_d2", "xgb_d3", "rank_d2", "rank_d3"):
        score, _ = optimize.fit_predict(model_name, fit, validation, FEATURES)
        results[model_name] = optimize.score_prediction(validation, score)
    return results


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    complaints, _, _, _, _ = odor.load_inputs()
    original_grid, _ = odor.add_grid_columns(complaints)
    _, selected_hours = odor.build_bounded_events(original_grid)
    configurations = [
        (grid_m, input_minutes, forecast_minutes)
        for grid_m in (250, 500, 750, 1000)
        for input_minutes, forecast_minutes in ((20, 20), (20, 30), (30, 20), (30, 30), (40, 20))
    ]
    validation_results = {}
    cached = {}
    choices = []
    for grid_m, input_minutes, forecast_minutes in configurations:
        name = f"g{grid_m}_i{input_minutes}_f{forecast_minutes}"
        data, train_ids, test_ids = build_data(
            complaints, selected_hours, grid_m, input_minutes, forecast_minutes
        )
        cached[name] = (data, train_ids, test_ids)
        results = validate_configuration(data, data, train_ids, input_minutes)
        validation_results[name] = {
            "grid_m": grid_m, "input_minutes": input_minutes,
            "forecast_minutes": forecast_minutes, "train_events": len(train_ids),
            "test_events": len(test_ids), "models": results,
        }
        for model_name, metrics in results.items():
            # PR-AUC를 우선하되 양성률 차이를 확인할 수 있도록 원값을 보존한다.
            choices.append((metrics["pr_auc"], metrics["topk_recall"], name, model_name))
        best = max(results, key=lambda model: (results[model]["pr_auc"], results[model]["topk_recall"]))
        print(f"{name}: {best} PR {results[best]['pr_auc']:.3f}, Top-K {results[best]['topk_recall']:.3f}")

    _, _, selected_name, selected_model = max(choices)
    data, train_ids, test_ids = cached[selected_name]
    config = validation_results[selected_name]
    train = data[data["is_train"]].copy()
    test = data[~data["is_train"]].copy()
    train, test = prepare_prior(train, test, train_ids)
    score, _ = optimize.fit_predict(selected_model, train, test, FEATURES, final=True)
    test_metrics = optimize.score_prediction(test, score)
    test["model_score"] = score
    current = json.loads(Path("outputs/odor_ai_mvp/model_metrics.json").read_text(encoding="utf-8"))["early_prediction"]
    report = {
        "selection_policy": "grid/input/forecast/model selected on inner temporal validation PR-AUC",
        "validation_results": validation_results,
        "selected_configuration": selected_name,
        "selected_model": selected_model,
        "test_events": len(test_ids),
        "test_metrics": test_metrics,
        "positive_rate_test": float(test["target"].mean()),
        "current_500m_30m_30m": {
            "pr_auc": current["selected_model_metrics"]["pr_auc"],
            "roc_auc": current["selected_model_metrics"]["roc_auc"],
            "topk_recall": current["model_topk_recall"],
        },
    }
    (OUTPUT_DIR / "sensitivity_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    test.to_csv(OUTPUT_DIR / "selected_test_predictions.csv", index=False, encoding="utf-8-sig")
    print("===== 민감도 최종 결과 =====")
    print(f"선택: {selected_name}/{selected_model}, 테스트 Event {len(test_ids)}개")
    print(f"PR {test_metrics['pr_auc']:.3f}, ROC {test_metrics['roc_auc']:.3f}, Top-K {test_metrics['topk_recall']:.3f}")


if __name__ == "__main__":
    main()
