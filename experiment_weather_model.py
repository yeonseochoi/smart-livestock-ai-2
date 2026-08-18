"""기존 조기예측 모델과 기상 특성 추가 모델을 동일 조건에서 비교한다."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

import build_odor_ai_mvp as odor


WEATHER_FILE = Path("outputs/weather_integration/event_weather_features.csv")
OUTPUT_DIR = Path("outputs/weather_experiment")

BASE_FEATURES = [
    "initial_count", "initial_intensity", "min_distance", "neighbor_count", "radius3_count",
    "prior", "hour_sin", "hour_cos", "centroid_distance", "nearest_intensity",
    "weighted_intensity", "observed_grid_count", "observed_report_count", "first15_count", "growth",
]
DIRECTION_FEATURES = [
    "weather_wind_speed", "weather_wind_reliability", "downwind_alignment",
    "downwind_distance", "crosswind_distance", "wind_advection", "is_downwind",
]
REGIME_FEATURES = [
    "weather_humidity", "weather_rainfall_15m", "weather_rainfall_60m",
    "weather_temperature", "weather_calm", "humid_calm", "rain_wind_interaction",
]
WIND_PRIOR_FEATURE = "wind_sector_prior"


def weather_lookup(weather: pd.DataFrame) -> dict[str, dict[str, float]]:
    """AWS 우선, 결측 시 ASOS로 보완한 Event별 기상값을 만든다."""
    result: dict[str, dict[str, float]] = {}
    for row in weather.itertuples(index=False):
        wind_speed = row.aws_wind_speed if pd.notna(row.aws_wind_speed) else row.asos_wind_speed
        humidity = row.aws_humidity if pd.notna(row.aws_humidity) else row.asos_humidity
        temperature = row.aws_temperature if pd.notna(row.aws_temperature) else row.asos_temperature
        rainfall_60m = row.aws_rainfall_60m if pd.notna(row.aws_rainfall_60m) else row.asos_rainfall_hour
        rainfall_15m = row.aws_rainfall_15m if pd.notna(row.aws_rainfall_15m) else rainfall_60m / 4
        east = float(row.aws_downwind_east)
        north = float(row.aws_downwind_north)
        reliability = math.hypot(east, north)
        if reliability > 1e-9:
            east /= reliability
            north /= reliability
        result[row.event_id] = {
            "wind_speed": float(wind_speed),
            "humidity": float(humidity),
            "temperature": float(temperature),
            "rainfall_15m": float(rainfall_15m),
            "rainfall_60m": float(rainfall_60m),
            "downwind_east": east,
            "downwind_north": north,
            "wind_reliability": reliability,
        }
    return result


def candidate_weather_features(
    gx: int, gy: int, initial_agg: pd.DataFrame, weather: dict[str, float],
) -> list[float]:
    weights = initial_agg["count"].to_numpy(float)
    centroid_x = float(np.average(initial_agg["grid_x"], weights=weights))
    centroid_y = float(np.average(initial_agg["grid_y"], weights=weights))
    dx, dy = gx - centroid_x, gy - centroid_y
    distance = math.hypot(dx, dy)
    east, north = weather["downwind_east"], weather["downwind_north"]
    projection = dx * east + dy * north
    crosswind = abs(dx * north - dy * east)
    alignment = projection / distance if distance > 1e-9 else 0.0
    speed = weather["wind_speed"]
    humidity = weather["humidity"]
    rain15 = weather["rainfall_15m"]
    rain60 = weather["rainfall_60m"]
    calm = float(speed < 0.5)
    return [
        speed,
        weather["wind_reliability"],
        alignment,
        projection,
        crosswind,
        max(0.0, projection) * speed,
        float(projection > 0),
        humidity,
        rain15,
        rain60,
        weather["temperature"],
        calm,
        humidity * calm,
        rain60 * speed,
    ]


def build_candidates(
    events: pd.DataFrame, summary: pd.DataFrame, weather: dict[str, dict[str, float]],
) -> tuple[pd.DataFrame, set[str], set[str]]:
    valid_ids = []
    for event_id, group in events.groupby("event_id"):
        hour = pd.Timestamp(group["event_hour"].iloc[0])
        initial = group[group["datetime"] < hour + pd.Timedelta(minutes=odor.INPUT_MINUTES)]
        future = group[group["datetime"] >= hour + pd.Timedelta(minutes=odor.INPUT_MINUTES)]
        if initial["grid_id"].nunique() >= 2 and future["grid_id"].nunique() >= 1 and event_id in weather:
            valid_ids.append(event_id)
    valid_summary = summary[summary["event_id"].isin(valid_ids)].copy()
    train_ids, test_ids = odor.split_event_ids(valid_summary)
    prior = odor.training_prior(events, train_ids, target_window="future")
    rows = []
    for event_id, group in events[events["event_id"].isin(valid_ids)].groupby("event_id", sort=False):
        hour = pd.Timestamp(group["event_hour"].iloc[0])
        initial = group[group["datetime"] < hour + pd.Timedelta(minutes=odor.INPUT_MINUTES)].copy()
        future = group[group["datetime"] >= hour + pd.Timedelta(minutes=odor.INPUT_MINUTES)]
        observed = set(zip(initial["grid_x"].astype(int), initial["grid_y"].astype(int)))
        target = set(zip(future["grid_x"].astype(int), future["grid_y"].astype(int)))
        initial["is_first15"] = initial["datetime"] < hour + pd.Timedelta(minutes=15)
        initial_agg = initial.groupby(["grid_x", "grid_y"]).agg(
            count=("datetime", "size"), mean_intensity=("intensity", "mean"),
            first15_count=("is_first15", "sum"),
        ).reset_index()
        for gx, gy in odor.candidate_cells(observed, radius=3):
            base_features = odor.cell_features((gx, gy), initial_agg, hour, prior, include_trend=True)
            weather_features = candidate_weather_features(gx, gy, initial_agg, weather[event_id])
            rows.append([
                event_id, event_id in train_ids, hour, gx, gy, int((gx, gy) in target),
                *base_features, *weather_features,
            ])
    columns = ["event_id", "is_train", "event_hour", "grid_x", "grid_y", "target",
               *BASE_FEATURES, *DIRECTION_FEATURES, *REGIME_FEATURES]
    return pd.DataFrame(rows, columns=columns), train_ids, test_ids


def wind_sector(event_weather: dict[str, float], sectors: int) -> int:
    angle = (math.degrees(math.atan2(
        event_weather["downwind_east"], event_weather["downwind_north"]
    )) + 360) % 360
    return int(angle / (360 / sectors)) % sectors


def wind_conditioned_prior(
    events: pd.DataFrame, event_ids: set[str], weather: dict[str, dict[str, float]],
    sectors: int, smoothing: float,
) -> tuple[dict[tuple[int, int, int], float], dict[tuple[int, int], float]]:
    """유사 풍하 방향의 과거 Event에서 미래 Grid가 나타난 빈도를 계산한다."""
    future = events[
        events["event_id"].isin(event_ids)
        & (events["datetime"] >= events["event_hour"] + pd.Timedelta(minutes=odor.INPUT_MINUTES))
    ].copy()
    active = future[["event_id", "grid_x", "grid_y"]].drop_duplicates()
    active["sector"] = active["event_id"].map(lambda event_id: wind_sector(weather[event_id], sectors))
    sector_by_event = {event_id: wind_sector(weather[event_id], sectors) for event_id in event_ids}
    sector_counts = pd.Series(sector_by_event).value_counts().to_dict()
    global_prior = odor.training_prior(events, event_ids, target_window="future")
    conditional_counts = active.groupby(["sector", "grid_x", "grid_y"]).size().to_dict()
    result: dict[tuple[int, int, int], float] = {}
    cells = set(global_prior) | {(int(x), int(y)) for _, x, y in conditional_counts}
    for sector in range(sectors):
        denominator = float(sector_counts.get(sector, 0)) + smoothing
        for gx, gy in cells:
            count = float(conditional_counts.get((sector, gx, gy), 0))
            result[(sector, gx, gy)] = (count + smoothing * global_prior.get((gx, gy), 0.0)) / max(denominator, 1.0)
    return result, global_prior


def apply_priors(
    frame: pd.DataFrame, prior_ids: set[str], events: pd.DataFrame,
    weather: dict[str, dict[str, float]], sectors: int, smoothing: float,
) -> pd.DataFrame:
    result = frame.copy()
    wind_prior, global_prior = wind_conditioned_prior(events, prior_ids, weather, sectors, smoothing)
    result["prior"] = [global_prior.get((int(x), int(y)), 0.0) for x, y in zip(result["grid_x"], result["grid_y"])]
    result[WIND_PRIOR_FEATURE] = [
        wind_prior.get((wind_sector(weather[event_id], sectors), int(x), int(y)), 0.0)
        for event_id, x, y in zip(result["event_id"], result["grid_x"], result["grid_y"])
    ]
    return result


def make_experiment_model(name: str, y: pd.Series, final: bool = False):
    if not name.startswith("xgb_"):
        return odor.make_candidate_model(name, final=final)
    positive = max(int(y.sum()), 1)
    scale = (len(y) - positive) / positive
    configurations = {
        "xgb_d2": {"max_depth": 2, "min_child_weight": 8, "learning_rate": 0.035},
        "xgb_d3": {"max_depth": 3, "min_child_weight": 7, "learning_rate": 0.03},
        "xgb_d4": {"max_depth": 4, "min_child_weight": 10, "learning_rate": 0.025},
    }
    return XGBClassifier(
        n_estimators=650 if final else 350,
        **configurations[name],
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.15,
        reg_lambda=2.0,
        scale_pos_weight=scale,
        objective="binary:logistic",
        eval_metric="aucpr",
        random_state=odor.RANDOM_STATE,
        n_jobs=4,
    )


def add_boosting_scores(
    fit: pd.DataFrame, validation: pd.DataFrame, features: list[str],
    scores: dict[str, dict[str, float]],
) -> None:
    for model_name in ("xgb_d2", "xgb_d3", "xgb_d4"):
        model = make_experiment_model(model_name, fit["target"])
        model.fit(fit[features], fit["target"])
        validation[model_name] = model.predict_proba(validation[features])[:, 1]
        scores[model_name] = {
            "pr_auc": float(average_precision_score(validation["target"], validation[model_name])),
            "topk_recall": odor._topk_recall(validation, model_name),
        }


def select_wind_prior_model(
    train: pd.DataFrame, ordered_ids: list[str], events: pd.DataFrame,
    weather: dict[str, dict[str, float]], sectors: int, smoothing: float,
) -> tuple[str, dict[str, dict[str, float]]]:
    cut = max(1, int(len(ordered_ids) * .8))
    fit_ids, validation_ids = set(ordered_ids[:cut]), set(ordered_ids[cut:])
    fit = apply_priors(train[train["event_id"].isin(fit_ids)], fit_ids, events, weather, sectors, smoothing)
    validation = apply_priors(
        train[train["event_id"].isin(validation_ids)], fit_ids, events, weather, sectors, smoothing
    )
    features = BASE_FEATURES + [WIND_PRIOR_FEATURE]
    scores: dict[str, dict[str, float]] = {}
    for model_name in ["extra_leaf1", "extra_leaf2", "extra_leaf4", "rf_leaf2", "rf_leaf4"]:
        model = odor.make_candidate_model(model_name)
        model.fit(fit[features], fit["target"])
        validation[model_name] = model.predict_proba(validation[features])[:, 1]
        scores[model_name] = {
            "pr_auc": float(average_precision_score(validation["target"], validation[model_name])),
            "topk_recall": odor._topk_recall(validation, model_name),
        }
    add_boosting_scores(fit, validation, features, scores)
    best = max(scores, key=lambda name: (scores[name]["pr_auc"], scores[name]["topk_recall"]))
    return best, scores


def select_configuration(
    train: pd.DataFrame, train_ids: set[str], events: pd.DataFrame,
    weather: dict[str, dict[str, float]],
) -> tuple[str, str, list[str], dict[str, object], dict[str, float]]:
    ordered_ids = (train[["event_id", "event_hour"]].drop_duplicates()
                   .sort_values("event_hour")["event_id"].tolist())
    configurations = {
        "base": BASE_FEATURES,
        "weather_direction": BASE_FEATURES + DIRECTION_FEATURES,
        "weather_full": BASE_FEATURES + DIRECTION_FEATURES + REGIME_FEATURES,
    }
    results: dict[str, object] = {}
    chosen: tuple[str, str, list[str]] | None = None
    chosen_params: dict[str, float] = {}
    chosen_score = (-np.inf, -np.inf)
    for config_name, features in configurations.items():
        model_name, scores, fit_frame, validation_frame = odor.select_model_on_inner_validation(
            train, ordered_ids, features, events, target_window="future"
        )
        add_boosting_scores(fit_frame, validation_frame, features, scores)
        model_name = max(scores, key=lambda name: (scores[name]["pr_auc"], scores[name]["topk_recall"]))
        result = scores[model_name]
        results[config_name] = {"selected_model": model_name, "models": scores, "selected": result}
        score = (result["pr_auc"], result["topk_recall"])
        if score > chosen_score:
            chosen = (config_name, model_name, features)
            chosen_score = score
            chosen_params = {}
    for sectors in (4, 8, 12):
        for smoothing in (3.0, 8.0, 15.0):
            config_name = f"weather_prior_s{sectors}_k{smoothing:g}"
            model_name, scores = select_wind_prior_model(
                train, ordered_ids, events, weather, sectors, smoothing
            )
            result = scores[model_name]
            results[config_name] = {"selected_model": model_name, "models": scores, "selected": result}
            score = (result["pr_auc"], result["topk_recall"])
            if score > chosen_score:
                chosen = (config_name, model_name, BASE_FEATURES + [WIND_PRIOR_FEATURE])
                chosen_score = score
                chosen_params = {"sectors": sectors, "smoothing": smoothing}
    assert chosen is not None
    return *chosen, results, chosen_params


def evaluate(
    train: pd.DataFrame, test: pd.DataFrame, features: list[str], model_name: str,
) -> tuple[pd.DataFrame, dict[str, object], object]:
    model = make_experiment_model(model_name, train["target"], final=True)
    model.fit(train[features], train["target"])
    result = test.copy()
    result["model_score"] = model.predict_proba(result[features])[:, 1]
    raw_baseline = result["initial_count"] + 1 / (1 + result["min_distance"])
    result["persistence_baseline"] = raw_baseline / max(raw_baseline.max(), 1e-9)
    metrics = {
        "selected_model_metrics": odor._safe_metrics(result["target"].to_numpy(), result["model_score"].to_numpy()),
        "model_topk_recall": odor._topk_recall(result, "model_score"),
        "baseline_metrics": odor._safe_metrics(
            result["target"].to_numpy(), result["persistence_baseline"].to_numpy()),
        "baseline_topk_recall": odor._topk_recall(result, "persistence_baseline"),
        "feature_importance": dict(zip(features, map(float, model.feature_importances_))),
    }
    return result, metrics, model


def evaluate_validation_selected_ensemble(
    train: pd.DataFrame, test: pd.DataFrame, train_ids: set[str], events: pd.DataFrame,
    weather: dict[str, dict[str, float]], weather_model_name: str,
    selected_params: dict[str, float], weather_features: list[str],
) -> tuple[pd.DataFrame, dict[str, object]]:
    ordered_ids = (train[["event_id", "event_hour"]].drop_duplicates()
                   .sort_values("event_hour")["event_id"].tolist())
    cut = max(1, int(len(ordered_ids) * .8))
    fit_ids, validation_ids = set(ordered_ids[:cut]), set(ordered_ids[cut:])
    fit_base = apply_priors(train[train["event_id"].isin(fit_ids)], fit_ids, events, weather, 4, 8.0)
    val_base = apply_priors(train[train["event_id"].isin(validation_ids)], fit_ids, events, weather, 4, 8.0)
    if WIND_PRIOR_FEATURE in weather_features:
        sectors = int(selected_params["sectors"])
        smoothing = float(selected_params["smoothing"])
        fit_weather = apply_priors(train[train["event_id"].isin(fit_ids)], fit_ids, events, weather, sectors, smoothing)
        val_weather = apply_priors(train[train["event_id"].isin(validation_ids)], fit_ids, events, weather, sectors, smoothing)
    else:
        fit_weather, val_weather = fit_base, val_base

    base_inner = make_experiment_model("extra_leaf4", fit_base["target"])
    weather_inner = make_experiment_model(weather_model_name, fit_weather["target"])
    base_inner.fit(fit_base[BASE_FEATURES], fit_base["target"])
    weather_inner.fit(fit_weather[weather_features], fit_weather["target"])
    base_score = base_inner.predict_proba(val_base[BASE_FEATURES])[:, 1]
    weather_score = weather_inner.predict_proba(val_weather[weather_features])[:, 1]
    alpha_results: dict[float, dict[str, float]] = {}
    for alpha in np.linspace(0, 1, 21):
        score = alpha * base_score + (1 - alpha) * weather_score
        probe = val_base[["event_id", "target"]].copy()
        probe["score"] = score
        alpha_results[float(alpha)] = {
            "pr_auc": float(average_precision_score(probe["target"], score)),
            "topk_recall": odor._topk_recall(probe, "score"),
        }
    alpha = max(alpha_results, key=lambda value: (
        alpha_results[value]["pr_auc"], alpha_results[value]["topk_recall"]
    ))

    full_base = apply_priors(train, train_ids, events, weather, 4, 8.0)
    test_base = apply_priors(test, train_ids, events, weather, 4, 8.0)
    if WIND_PRIOR_FEATURE in weather_features:
        full_weather = apply_priors(
            train, train_ids, events, weather, int(selected_params["sectors"]), float(selected_params["smoothing"])
        )
        test_weather = apply_priors(
            test, train_ids, events, weather, int(selected_params["sectors"]), float(selected_params["smoothing"])
        )
    else:
        full_weather, test_weather = full_base, test_base
    base_final = make_experiment_model("extra_leaf4", full_base["target"], final=True)
    weather_final = make_experiment_model(weather_model_name, full_weather["target"], final=True)
    base_final.fit(full_base[BASE_FEATURES], full_base["target"])
    weather_final.fit(full_weather[weather_features], full_weather["target"])
    result = test.copy()
    result["ensemble_base_score"] = base_final.predict_proba(test_base[BASE_FEATURES])[:, 1]
    result["ensemble_weather_score"] = weather_final.predict_proba(test_weather[weather_features])[:, 1]
    result["ensemble_score"] = alpha * result["ensemble_base_score"] + (1 - alpha) * result["ensemble_weather_score"]
    metrics = {
        "base_weight_selected_on_validation": float(alpha),
        "validation_weights": alpha_results,
        "test_metrics": odor._safe_metrics(result["target"].to_numpy(), result["ensemble_score"].to_numpy()),
        "test_topk_recall": odor._topk_recall(result, "ensemble_score"),
    }
    return result, metrics


def select_topk_configuration(validation: dict[str, object]) -> tuple[str, str, list[str], dict[str, float]]:
    choices = []
    for config_name, config in validation.items():
        for model_name, metrics in config["models"].items():
            choices.append((metrics["topk_recall"], metrics["pr_auc"], config_name, model_name))
    _, _, config_name, model_name = max(choices)
    params: dict[str, float] = {}
    if config_name == "base":
        features = BASE_FEATURES
    elif config_name == "weather_direction":
        features = BASE_FEATURES + DIRECTION_FEATURES
    elif config_name == "weather_full":
        features = BASE_FEATURES + DIRECTION_FEATURES + REGIME_FEATURES
    else:
        match = re.fullmatch(r"weather_prior_s(\d+)_k([0-9.]+)", config_name)
        if not match:
            raise ValueError(f"알 수 없는 설정: {config_name}")
        params = {"sectors": int(match.group(1)), "smoothing": float(match.group(2))}
        features = BASE_FEATURES + [WIND_PRIOR_FEATURE]
    return config_name, model_name, features, params


def main() -> None:
    if not WEATHER_FILE.exists():
        raise FileNotFoundError(f"먼저 fetch_kma_weather.py를 실행하세요: {WEATHER_FILE}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    complaints, _, _, _, _ = odor.load_inputs()
    complaints, _ = odor.add_grid_columns(complaints)
    events, summary = odor.build_bounded_events(complaints)
    weather_frame = pd.read_csv(WEATHER_FILE, encoding="utf-8-sig", parse_dates=["event_hour"])
    candidates, train_ids, test_ids = build_candidates(events, summary, weather_lookup(weather_frame))
    train = candidates[candidates["is_train"]].copy()
    test = candidates[~candidates["is_train"]].copy()

    weather = weather_lookup(weather_frame)
    config_name, model_name, features, validation, selected_params = select_configuration(
        train, train_ids, events, weather
    )
    if WIND_PRIOR_FEATURE in features:
        train_for_weather = apply_priors(
            train, train_ids, events, weather,
            int(selected_params["sectors"]), float(selected_params["smoothing"]),
        )
        test_for_weather = apply_priors(
            test, train_ids, events, weather,
            int(selected_params["sectors"]), float(selected_params["smoothing"]),
        )
    else:
        train_for_weather, test_for_weather = train, test
    weather_result, weather_metrics, _ = evaluate(
        train_for_weather, test_for_weather, features, model_name
    )

    base_model_name = validation["base"]["selected_model"]
    base_result, base_metrics, _ = evaluate(train, test, BASE_FEATURES, base_model_name)
    ensemble_result, ensemble_metrics = evaluate_validation_selected_ensemble(
        train, test, train_ids, events, weather, model_name, selected_params, features
    )
    topk_config, topk_model, topk_features, topk_params = select_topk_configuration(validation)
    if WIND_PRIOR_FEATURE in topk_features:
        topk_train = apply_priors(
            train, train_ids, events, weather, int(topk_params["sectors"]), float(topk_params["smoothing"])
        )
        topk_test = apply_priors(
            test, train_ids, events, weather, int(topk_params["sectors"]), float(topk_params["smoothing"])
        )
    else:
        topk_train, topk_test = train, test
    topk_result, topk_metrics, _ = evaluate(topk_train, topk_test, topk_features, topk_model)
    comparison = {
        "experiment": "weather_feature_ablation_for_early_prediction",
        "selection_rule": "feature configuration and model selected on inner validation PR-AUC, then Top-K recall",
        "train_events": len(train_ids),
        "test_events": len(test_ids),
        "train_rows": len(train),
        "test_rows": len(test),
        "selected_configuration": config_name,
        "selected_model": model_name,
        "selected_parameters": selected_params,
        "inner_validation": validation,
        "base_test": base_metrics,
        "weather_test": weather_metrics,
        "ensemble": ensemble_metrics,
        "topk_selected": {
            "configuration": topk_config,
            "model": topk_model,
            "parameters": topk_params,
            **topk_metrics,
        },
        "delta": {
            "pr_auc": weather_metrics["selected_model_metrics"]["pr_auc"] - base_metrics["selected_model_metrics"]["pr_auc"],
            "roc_auc": weather_metrics["selected_model_metrics"]["roc_auc"] - base_metrics["selected_model_metrics"]["roc_auc"],
            "topk_recall": weather_metrics["model_topk_recall"] - base_metrics["model_topk_recall"],
        },
    }
    weather_result.to_csv(OUTPUT_DIR / "weather_model_test_predictions.csv", index=False, encoding="utf-8-sig")
    ensemble_result.to_csv(OUTPUT_DIR / "ensemble_test_predictions.csv", index=False, encoding="utf-8-sig")
    topk_result.to_csv(OUTPUT_DIR / "topk_model_test_predictions.csv", index=False, encoding="utf-8-sig")
    base_result[["event_id", "grid_x", "grid_y", "target", "model_score"]].rename(
        columns={"model_score": "base_model_score"}
    ).to_csv(OUTPUT_DIR / "base_model_test_predictions.csv", index=False, encoding="utf-8-sig")
    (OUTPUT_DIR / "weather_model_metrics.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    base_pr = base_metrics["selected_model_metrics"]["pr_auc"]
    weather_pr = weather_metrics["selected_model_metrics"]["pr_auc"]
    print("===== 기상 특성 추가 실험 =====")
    print(f"내부 검증 선택: {config_name} / {model_name}")
    print(f"PR-AUC: {base_pr:.3f} -> {weather_pr:.3f} ({weather_pr-base_pr:+.3f})")
    print(f"ROC-AUC: {base_metrics['selected_model_metrics']['roc_auc']:.3f} -> "
          f"{weather_metrics['selected_model_metrics']['roc_auc']:.3f}")
    print(f"Top-K Recall: {base_metrics['model_topk_recall']:.3f} -> "
          f"{weather_metrics['model_topk_recall']:.3f} "
          f"({weather_metrics['model_topk_recall']-base_metrics['model_topk_recall']:+.3f})")
    print(f"Ensemble (기존 가중치 {ensemble_metrics['base_weight_selected_on_validation']:.2f}) "
          f"PR-AUC: {ensemble_metrics['test_metrics']['pr_auc']:.3f}, "
          f"Top-K: {ensemble_metrics['test_topk_recall']:.3f}")
    print(f"Top-K 목적 선택: {topk_config} / {topk_model}, "
          f"PR-AUC {topk_metrics['selected_model_metrics']['pr_auc']:.3f}, "
          f"Top-K {topk_metrics['model_topk_recall']:.3f}")
    print(f"산출물: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
