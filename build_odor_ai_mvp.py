"""민원 기반 악취 Event 복원·단기 확산예측·대응 리포트 MVP.

실행:
    python build_odor_ai_mvp.py

주의:
    이 코드는 신고 발생 가능성을 예측한다. 실제 악취 농도나 원인 농가를
    확정하지 않으며, 농가 좌표가 확보되기 전에는 시설 공간 순위를 생성하지 않는다.
"""
from __future__ import annotations

import json
import math
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import BallTree

import analyze_spatiotemporal_complaints as base


OUTPUT_DIR = Path("outputs/odor_ai_mvp")
GRID_M = 500
EVENT_MIN_REPORTS = 10
EVENT_MIN_GRIDS = 5
INPUT_MINUTES = 30
FORECAST_MINUTES = 30
RANDOM_STATE = 42
EARTH_RADIUS_KM = 6371.0088


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """민원을 읽고, 존재하는 경우에만 과거 보조 실험 데이터를 함께 읽는다."""
    raw, mapping = base.load_data()
    complaints = base.filter_target_region(base.preprocess_data(raw, mapping), include_adjacent=False)
    tables: list[pd.DataFrame] = []
    for path in Path("data").glob("*.csv"):
        for encoding in ("utf-8-sig", "cp949"):
            try:
                table = pd.read_csv(path, encoding=encoding)
                table.attrs["source_path"] = str(path)
                tables.append(table)
                break
            except UnicodeDecodeError:
                continue
    def optional_table(required: set[str]) -> pd.DataFrame:
        return next((x for x in tables if required.issubset(x.columns)), pd.DataFrame())

    sensor = optional_table({"센서명", "복합악취", "황화수소", "암모니아", "TVOC"})
    farms = optional_table({"사육업종", "시설면적(제곱미터)", "사육두수"})
    pig_waste = optional_table({"가축분뇨폐수량", "처리방법"})
    wanju = optional_table({"시군구명", "주사육업종"})
    return complaints, sensor, farms, pig_waste, wanju


def add_grid_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """익산시 중심 위·경도를 기준으로 약 500m 격자 좌표를 만든다."""
    out = df.copy()
    lat0 = float(out["latitude"].median())
    lon0 = float(out["longitude"].median())
    x_m = (out["longitude"] - lon0) * 111_320 * math.cos(math.radians(lat0))
    y_m = (out["latitude"] - lat0) * 110_540
    out["grid_x"] = np.floor(x_m / GRID_M).astype(int)
    out["grid_y"] = np.floor(y_m / GRID_M).astype(int)
    out["grid_id"] = out["grid_x"].astype(str) + ":" + out["grid_y"].astype(str)
    return out, {"lat0": lat0, "lon0": lon0, "grid_m": GRID_M}


def grid_centroid(gx: int, gy: int, meta: dict[str, float]) -> tuple[float, float]:
    """격자 인덱스를 WGS84 중심 좌표로 되돌린다."""
    x = (gx + 0.5) * meta["grid_m"]
    y = (gy + 0.5) * meta["grid_m"]
    lat = meta["lat0"] + y / 110_540
    lon = meta["lon0"] + x / (111_320 * math.cos(math.radians(meta["lat0"])))
    return lat, lon


def st_dbscan(
    df: pd.DataFrame, eps_km: float = 1.0, eps_minutes: int = 60,
    min_samples: int = 5, max_event_minutes: int = 180,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """BallTree 이웃을 이용한 bounded ST-DBSCAN을 수행한다.

    공간과 시간 조건을 동시에 만족하는 이웃이 min_samples개 이상인 점을
    core point로 정의한다. 표준 DBSCAN은 장기간 반복 신고가 연쇄 연결되는
    문제가 있으므로 최초 seed 기준 최대 3시간까지만 하나의 Event로 확장한다.
    """
    work = df.sort_values("datetime").reset_index(drop=True).copy()
    coords = np.radians(work[["latitude", "longitude"]].to_numpy(float))
    # pandas 버전에 따라 datetime64 해상도가 달라질 수 있으므로 Timedelta로 분을 계산한다.
    minutes = ((work["datetime"] - pd.Timestamp("1970-01-01")) / pd.Timedelta(minutes=1)).to_numpy(float)
    tree = BallTree(coords, metric="haversine")
    spatial = tree.query_radius(coords, r=eps_km / EARTH_RADIUS_KM)
    neighbors = [idx[np.abs(minutes[idx] - minutes[i]) <= eps_minutes] for i, idx in enumerate(spatial)]
    core = np.array([len(x) >= min_samples for x in neighbors])
    labels = np.full(len(work), -1, dtype=int)
    visited = np.zeros(len(work), dtype=bool)
    cluster_id = 0
    for i in range(len(work)):
        if visited[i]:
            continue
        visited[i] = True
        if not core[i]:
            continue
        labels[i] = cluster_id
        # 정렬된 최초 core point를 Event 시작점으로 삼아 과거 방향 연쇄를 막는다.
        queue = deque(int(x) for x in neighbors[i] if x != i and minutes[x] >= minutes[i])
        queued = set(queue)
        while queue:
            j = queue.popleft()
            if not visited[j]:
                visited[j] = True
                if core[j]:
                    for k in neighbors[j]:
                        k = int(k)
                        if minutes[i] <= minutes[k] <= minutes[i] + max_event_minutes and k not in queued:
                            queue.append(k)
                            queued.add(k)
            if labels[j] == -1:
                labels[j] = cluster_id
        cluster_id += 1
    work["st_cluster"] = labels
    summaries = []
    for cid, group in work[work["st_cluster"] >= 0].groupby("st_cluster"):
        summaries.append({
            "cluster_id": cid,
            "start": group["datetime"].min(),
            "end": group["datetime"].max(),
            "duration_minutes": (group["datetime"].max() - group["datetime"].min()).total_seconds() / 60,
            "reports": len(group),
            "unique_grids": group["grid_id"].nunique(),
            "mean_intensity": group["intensity"].mean(),
            "major_region": group["region"].value_counts().index[0],
            "major_odor_type": group["odor_type"].value_counts().index[0],
        })
    return work, pd.DataFrame(summaries).sort_values("reports", ascending=False)


def build_bounded_events(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """예측 학습에는 시간 연쇄를 막기 위해 독립적인 1시간 Event를 사용한다."""
    work = df.copy()
    work["event_hour"] = work["datetime"].dt.floor("h")
    summary = work.groupby("event_hour").agg(
        reports=("datetime", "size"), unique_grids=("grid_id", "nunique"),
        mean_intensity=("intensity", "mean"),
    ).reset_index()
    summary = summary[(summary["reports"] >= EVENT_MIN_REPORTS) & (summary["unique_grids"] >= EVENT_MIN_GRIDS)]
    selected = work[work["event_hour"].isin(summary["event_hour"])].copy()
    summary = summary.sort_values("event_hour").reset_index(drop=True)
    summary["event_id"] = [f"EVT-{i:04d}" for i in range(1, len(summary) + 1)]
    selected = selected.merge(summary[["event_hour", "event_id"]], on="event_hour", how="inner")
    return selected, summary


def candidate_cells(observed: set[tuple[int, int]], radius: int = 3) -> list[tuple[int, int]]:
    """관측 Grid 주변만 예측 후보로 구성하여 익산시 전체 0 편향을 줄인다."""
    cells: set[tuple[int, int]] = set()
    for gx, gy in observed:
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius:
                    cells.add((gx + dx, gy + dy))
    return sorted(cells)


def cell_features(
    cell: tuple[int, int], observed: pd.DataFrame, hour: pd.Timestamp,
    prior: dict[tuple[int, int], float], include_trend: bool = False,
) -> list[float]:
    """후보 Grid와 관측 Grid 관계를 거리·주변 활성도·강도로 표현한다."""
    gx, gy = cell
    if {"count", "mean_intensity"}.issubset(observed.columns):
        agg = observed
    else:
        agg = observed.groupby(["grid_x", "grid_y"]).agg(
            count=("datetime", "size"), mean_intensity=("intensity", "mean")
        ).reset_index()
    if agg.empty:
        return [0.0] * (15 if include_trend else 13)
    dist = np.sqrt((agg["grid_x"] - gx) ** 2 + (agg["grid_y"] - gy) ** 2)
    own = agg[(agg["grid_x"] == gx) & (agg["grid_y"] == gy)]
    near = agg[dist <= 1.5]
    near3 = agg[dist <= 3.0]
    weights = 1 / (1 + dist)
    centroid_x = np.average(agg["grid_x"], weights=agg["count"])
    centroid_y = np.average(agg["grid_y"], weights=agg["count"])
    nearest_intensity = float(agg.loc[dist.idxmin(), "mean_intensity"])
    features = [
        float(own["count"].sum()),
        float(own["mean_intensity"].mean()) if not own.empty else 0.0,
        float(dist.min()),
        float(near["count"].sum()),
        float(near3["count"].sum()),
        float(prior.get(cell, 0.0)),
        math.sin(2 * math.pi * hour.hour / 24),
        math.cos(2 * math.pi * hour.hour / 24),
        float(math.hypot(gx - centroid_x, gy - centroid_y)),
        nearest_intensity,
        float(np.average(agg["mean_intensity"], weights=weights)),
        float(len(agg)),
        float(agg["count"].sum()),
    ]
    if include_trend:
        if "first15_count" in agg.columns:
            own_trend = agg[(agg["grid_x"] == gx) & (agg["grid_y"] == gy)]
            f_count = int(own_trend["first15_count"].sum())
            s_count = int(own_trend["count"].sum()) - f_count
        else:
            midpoint = hour + pd.Timedelta(minutes=15)
            first = observed[observed["datetime"] < midpoint]
            second = observed[observed["datetime"] >= midpoint]
            f_count = int(((first["grid_x"] == gx) & (first["grid_y"] == gy)).sum())
            s_count = int(((second["grid_x"] == gx) & (second["grid_y"] == gy)).sum())
        features.extend([float(f_count), float(s_count - f_count)])
    return features


def training_prior(events: pd.DataFrame, event_ids: set[str], target_window: str = "all") -> dict[tuple[int, int], float]:
    """학습 Event에만 기반한 Grid 발생 사전확률을 계산해 시간 누수를 방지한다."""
    work = events[events["event_id"].isin(event_ids)].copy()
    if target_window == "future":
        work = work[work["datetime"] >= work["event_hour"] + pd.Timedelta(minutes=INPUT_MINUTES)]
    n = max(1, len(event_ids))
    active = work.groupby(["grid_x", "grid_y"])["event_id"].nunique() / n
    return {(int(x), int(y)): float(v) for (x, y), v in active.items()}


def split_event_ids(summary: pd.DataFrame) -> tuple[set[str], set[str]]:
    """과거 Event로 학습하고 미래 Event로 평가한다."""
    cut = max(1, int(len(summary) * 0.7))
    return set(summary.iloc[:cut]["event_id"]), set(summary.iloc[cut:]["event_id"])


def _safe_metrics(y: np.ndarray, score: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    pred = score >= threshold
    result = {
        "pr_auc": float(average_precision_score(y, score)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
    }
    result["roc_auc"] = float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else float("nan")
    return result


def _topk_recall(frame: pd.DataFrame, score_col: str) -> float:
    """동점 경계에서는 무작위 선택의 기대 적중수를 사용해 정렬 편향을 제거한다."""
    recalls = []
    for _, group in frame.groupby("event_id"):
        positives = int(group["target"].sum())
        if positives == 0:
            continue
        scores = group[score_col].to_numpy(float)
        targets = group["target"].to_numpy(int)
        kth = np.partition(scores, -positives)[-positives]
        above = scores > kth
        tied = scores == kth
        remaining = positives - int(above.sum())
        expected_hits = float(targets[above].sum())
        if remaining > 0 and tied.sum() > 0:
            expected_hits += remaining * float(targets[tied].mean())
        recalls.append(expected_hits / positives)
    return float(np.mean(recalls)) if recalls else float("nan")


def make_candidate_model(name: str, final: bool = False):
    """학습기간 내부 검증에서 비교할 보수적인 트리 모델 후보를 만든다."""
    trees = 550 if final else 250
    if name == "extra_leaf1":
        return ExtraTreesClassifier(n_estimators=trees, min_samples_leaf=1, max_features=.8,
                                    class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1)
    if name == "extra_leaf2":
        return ExtraTreesClassifier(n_estimators=trees, min_samples_leaf=2, max_features=.8,
                                    class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1)
    if name == "extra_leaf4":
        return ExtraTreesClassifier(n_estimators=trees, min_samples_leaf=4, max_features=.65,
                                    class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1)
    if name == "rf_leaf2":
        return RandomForestClassifier(n_estimators=trees, min_samples_leaf=2, max_features=.8,
                                      class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=1)
    return RandomForestClassifier(n_estimators=trees, min_samples_leaf=4, max_features="sqrt",
                                  class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=1)


def select_model_on_inner_validation(
    train: pd.DataFrame, ordered_ids: list[str], features: list[str], events: pd.DataFrame,
    target_window: str,
) -> tuple[str, dict[str, dict[str, float]], pd.DataFrame, pd.DataFrame]:
    """최종 미래 테스트를 보지 않고 학습기간 내부의 마지막 20%로 모델을 선택한다."""
    cut = max(1, int(len(ordered_ids) * .8))
    fit_ids, validation_ids = set(ordered_ids[:cut]), set(ordered_ids[cut:])
    fit_frame = train[train["event_id"].isin(fit_ids)].copy()
    validation = train[train["event_id"].isin(validation_ids)].copy()
    fit_prior = training_prior(events, fit_ids, target_window=target_window)
    for frame in (fit_frame, validation):
        frame.loc[:, "prior"] = [fit_prior.get((int(x), int(y)), 0.0) for x, y in zip(frame["grid_x"], frame["grid_y"])]
    scores: dict[str, dict[str, float]] = {}
    for name in ["extra_leaf1", "extra_leaf2", "extra_leaf4", "rf_leaf2", "rf_leaf4"]:
        candidate = make_candidate_model(name)
        candidate.fit(fit_frame[features], fit_frame["target"])
        validation[name] = candidate.predict_proba(validation[features])[:, 1]
        scores[name] = {
            "pr_auc": float(average_precision_score(validation["target"], validation[name])),
            "topk_recall": _topk_recall(validation, name),
        }
    best_name = max(scores, key=lambda n: (scores[n]["pr_auc"], scores[n]["topk_recall"]))
    validation["model_score"] = validation[best_name]
    return best_name, scores, fit_frame, validation


def run_reconstruction_experiment(events: pd.DataFrame, summary: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """Event의 활성 Grid 30%를 숨기고 주변 신고만으로 복원한다."""
    train_ids, test_ids = split_event_ids(summary)
    prior = training_prior(events, train_ids)
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    coverage = {"train": [0, 0], "test": [0, 0]}  # [후보에 포함된 숨김 Grid, 전체 숨김 Grid]
    for event_id, group in events.groupby("event_id", sort=False):
        grids = sorted(set(zip(group["grid_x"].astype(int), group["grid_y"].astype(int))))
        if len(grids) < EVENT_MIN_GRIDS:
            continue
        hide_n = max(1, int(round(len(grids) * 0.3)))
        hidden = set(grids[i] for i in rng.choice(len(grids), hide_n, replace=False))
        observed_grids = set(grids) - hidden
        observed = group[[tuple(x) in observed_grids for x in zip(group["grid_x"], group["grid_y"])]]
        observed_agg = observed.groupby(["grid_x", "grid_y"]).agg(
            count=("datetime", "size"), mean_intensity=("intensity", "mean")
        ).reset_index()
        # 이미 관측된 Grid는 복원 대상이 아니므로 후보에서 제외한다.
        candidates = [cell for cell in candidate_cells(observed_grids, radius=3) if cell not in observed_grids]
        split = "train" if event_id in train_ids else "test"
        coverage[split][0] += len(hidden.intersection(candidates))
        coverage[split][1] += len(hidden)
        for gx, gy in candidates:
            feats = cell_features((gx, gy), observed_agg, pd.Timestamp(group["event_hour"].iloc[0]), prior)
            rows.append([event_id, event_id in train_ids, gx, gy, int((gx, gy) in hidden), *feats])
    columns = ["event_id", "is_train", "grid_x", "grid_y", "target",
               "own_count", "own_intensity", "min_distance", "neighbor_count", "radius3_count",
               "prior", "hour_sin", "hour_cos", "centroid_distance", "nearest_intensity",
               "weighted_intensity", "observed_grid_count", "observed_report_count"]
    data = pd.DataFrame(rows, columns=columns)
    features = columns[5:]
    train, test = data[data["is_train"]], data[~data["is_train"]].copy()
    ordered_train_ids = sorted(train_ids, key=lambda x: int(x.split("-")[-1]))
    best_name, validation_scores, _, _ = select_model_on_inner_validation(
        train, ordered_train_ids, features, events, target_window="all")
    model = make_candidate_model(best_name, final=True)
    model.fit(train[features], train["target"])
    test["model_score"] = model.predict_proba(test[features])[:, 1]
    test["distance_baseline"] = 1 / (1 + test["min_distance"])
    metrics = {
        "experiment": "virtual_sensor_grid_masking",
        "train_events": len(train_ids), "test_events": len(test_ids),
        "train_rows": len(train), "test_rows": len(test),
        "selected_model": best_name, "inner_validation_models": validation_scores,
        "hidden_grid_candidate_coverage_train": coverage["train"][0] / max(coverage["train"][1], 1),
        "hidden_grid_candidate_coverage_test": coverage["test"][0] / max(coverage["test"][1], 1),
        "positive_rate_test": float(test["target"].mean()),
        "selected_model_metrics": _safe_metrics(test["target"].to_numpy(), test["model_score"].to_numpy()),
        "distance_baseline": _safe_metrics(test["target"].to_numpy(), test["distance_baseline"].to_numpy()),
        "model_topk_recall": _topk_recall(test, "model_score"),
        "baseline_topk_recall": _topk_recall(test, "distance_baseline"),
        "feature_importance": dict(zip(features, map(float, model.feature_importances_))),
    }
    return test, metrics


def run_early_prediction_experiment(events: pd.DataFrame, summary: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """Event 초기 30분으로 이후 30분의 신고 발생 Grid를 예측한다."""
    valid_ids = []
    for event_id, group in events.groupby("event_id"):
        hour = pd.Timestamp(group["event_hour"].iloc[0])
        initial = group[group["datetime"] < hour + pd.Timedelta(minutes=INPUT_MINUTES)]
        future = group[group["datetime"] >= hour + pd.Timedelta(minutes=INPUT_MINUTES)]
        if initial["grid_id"].nunique() >= 2 and future["grid_id"].nunique() >= 1:
            valid_ids.append(event_id)
    valid_summary = summary[summary["event_id"].isin(valid_ids)].copy()
    train_ids, test_ids = split_event_ids(valid_summary)
    prior = training_prior(events, train_ids, target_window="future")
    rows = []
    for event_id, group in events[events["event_id"].isin(valid_ids)].groupby("event_id", sort=False):
        hour = pd.Timestamp(group["event_hour"].iloc[0])
        initial = group[group["datetime"] < hour + pd.Timedelta(minutes=INPUT_MINUTES)]
        future = group[group["datetime"] >= hour + pd.Timedelta(minutes=INPUT_MINUTES)]
        observed = set(zip(initial["grid_x"].astype(int), initial["grid_y"].astype(int)))
        target = set(zip(future["grid_x"].astype(int), future["grid_y"].astype(int)))
        initial = initial.copy()
        initial["is_first15"] = initial["datetime"] < hour + pd.Timedelta(minutes=15)
        initial_agg = initial.groupby(["grid_x", "grid_y"]).agg(
            count=("datetime", "size"), mean_intensity=("intensity", "mean"),
            first15_count=("is_first15", "sum"),
        ).reset_index()
        for gx, gy in candidate_cells(observed, radius=3):
            feats = cell_features((gx, gy), initial_agg, hour, prior, include_trend=True)
            rows.append([event_id, event_id in train_ids, hour, gx, gy, int((gx, gy) in target), *feats])
    columns = ["event_id", "is_train", "event_hour", "grid_x", "grid_y", "target",
               "initial_count", "initial_intensity", "min_distance", "neighbor_count", "radius3_count",
               "prior", "hour_sin", "hour_cos", "centroid_distance", "nearest_intensity",
               "weighted_intensity", "observed_grid_count", "observed_report_count",
               "first15_count", "growth"]
    data = pd.DataFrame(rows, columns=columns)
    features = columns[6:]
    train, test = data[data["is_train"]], data[~data["is_train"]].copy()
    # 하이브리드 가중치는 최종 테스트가 아니라 학습기간 내부의 마지막 20% Event로 선택한다.
    ordered_train_ids = (train[["event_id", "event_hour"]].drop_duplicates()
                         .sort_values("event_hour")["event_id"].tolist())
    best_name, validation_scores, _, validation = select_model_on_inner_validation(
        train, ordered_train_ids, features, events, target_window="future")
    validation_raw = validation["initial_count"] + 1 / (1 + validation["min_distance"])
    validation["persistence_baseline"] = validation_raw / max(validation_raw.max(), 1e-9)
    alpha_candidates = np.linspace(0, 1, 21)
    alpha_scores = {}
    for alpha in alpha_candidates:
        validation["candidate_hybrid"] = alpha * validation["model_score"] + (1-alpha) * validation["persistence_baseline"]
        alpha_scores[float(alpha)] = _topk_recall(validation, "candidate_hybrid")
    hybrid_alpha = max(alpha_scores, key=alpha_scores.get)
    model = make_candidate_model(best_name, final=True)
    model.fit(train[features], train["target"])
    test["model_score"] = model.predict_proba(test[features])[:, 1]
    # 단순 기준선: 현재 활성 Grid와 가깝고 현재 신고가 많은 Grid가 계속 활성화된다고 가정한다.
    raw_baseline = test["initial_count"] + 1 / (1 + test["min_distance"])
    test["persistence_baseline"] = raw_baseline / max(raw_baseline.max(), 1e-9)
    # 지속성은 상위 Grid 순위가 강하고 ML은 전체 분류가 강하므로 보수적으로 결합한다.
    test["hybrid_score"] = hybrid_alpha * test["model_score"] + (1-hybrid_alpha) * test["persistence_baseline"]
    metrics = {
        "experiment": "first_30_to_next_30_grid_prediction",
        "eligible_events": len(valid_ids), "train_events": len(train_ids), "test_events": len(test_ids),
        "train_rows": len(train), "test_rows": len(test), "positive_rate_test": float(test["target"].mean()),
        "selected_model": best_name, "inner_validation_models": validation_scores,
        "hybrid_model_weight_selected_on_validation": float(hybrid_alpha),
        "validation_hybrid_topk_recall": float(alpha_scores[hybrid_alpha]),
        "selected_model_metrics": _safe_metrics(test["target"].to_numpy(), test["model_score"].to_numpy()),
        "persistence_baseline": _safe_metrics(test["target"].to_numpy(), test["persistence_baseline"].to_numpy()),
        "hybrid": _safe_metrics(test["target"].to_numpy(), test["hybrid_score"].to_numpy()),
        "model_topk_recall": _topk_recall(test, "model_score"),
        "baseline_topk_recall": _topk_recall(test, "persistence_baseline"),
        "hybrid_topk_recall": _topk_recall(test, "hybrid_score"),
        "feature_importance": dict(zip(features, map(float, model.feature_importances_))),
    }
    return test, metrics


def validate_sensor_case(complaints: pd.DataFrame, sensor: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """공개 센서와 겹치는 하루를 시간 기준 사례 검증한다.

    센서 좌표가 없으므로 현재는 공간 검증이 아니라 시간 일치도만 평가한다.
    """
    s = sensor.copy()
    s["날짜"] = pd.to_datetime(s["날짜"], errors="coerce")
    metrics_cols = ["복합악취", "황화수소", "암모니아", "TVOC"]
    for col in metrics_cols:
        s[col] = pd.to_numeric(s[col], errors="coerce")
        median = s.groupby("센서명")[col].transform("median")
        mad = s.groupby("센서명")[col].transform(lambda x: np.median(np.abs(x - np.median(x))))
        s[f"{col}_robust_z"] = 0.6745 * (s[col] - median) / mad.replace(0, np.nan)
    s["max_abs_robust_z"] = s[[f"{x}_robust_z" for x in metrics_cols]].abs().max(axis=1)
    sensor_time = s.groupby("날짜").agg(
        sensor_anomaly=("max_abs_robust_z", "max"),
        compound_odor=("복합악취", "mean"),
        ammonia=("암모니아", "mean"),
        hydrogen_sulfide=("황화수소", "mean"),
    ).reset_index()
    lo, hi = s["날짜"].min(), s["날짜"].max()
    c = complaints[complaints["datetime"].between(lo, hi)].copy()
    c["날짜"] = c["datetime"].dt.floor("5min")
    counts = c.groupby("날짜").size().rename("complaints")
    sensor_time = sensor_time.merge(counts, on="날짜", how="left").fillna({"complaints": 0})
    correlations = {}
    for lag in range(-12, 13):
        correlations[lag * 5] = sensor_time["complaints"].corr(sensor_time["sensor_anomaly"].shift(lag))
    valid_corr = {k: v for k, v in correlations.items() if pd.notna(v)}
    best_lag = max(valid_corr, key=lambda k: abs(valid_corr[k])) if valid_corr else None
    case = {
        "sensor_start": str(lo), "sensor_end": str(hi), "sensor_count": int(s["센서명"].nunique()),
        "overlap_complaints": len(c), "best_absolute_correlation_lag_minutes": best_lag,
        "correlation_at_best_lag": float(valid_corr[best_lag]) if best_lag is not None else None,
        "limitation": "센서 자료가 하루뿐이고 센서 좌표가 없어 시간 일치도 사례검증만 수행함",
    }
    return s, sensor_time, case


def enrich_prediction_coordinates(prediction: pd.DataFrame, meta: dict[str, float]) -> pd.DataFrame:
    """예측 Grid에 지도 표시용 중심 위·경도를 추가한다."""
    out = prediction.copy()
    centers = [grid_centroid(int(x), int(y), meta) for x, y in zip(out["grid_x"], out["grid_y"])]
    out["center_latitude"] = [x[0] for x in centers]
    out["center_longitude"] = [x[1] for x in centers]
    return out


def build_agent_report(
    prediction: pd.DataFrame, event_summary: pd.DataFrame, farms: pd.DataFrame,
    pig_waste: pd.DataFrame, sensor_case: dict[str, object], metrics: dict[str, object],
) -> str:
    """가장 최근 테스트 Event에 대해 근거와 한계를 포함한 대응 리포트를 생성한다."""
    latest_id = prediction.sort_values("event_hour").iloc[-1]["event_id"]
    event_pred = prediction[prediction["event_id"] == latest_id].nlargest(5, "hybrid_score")
    event = event_summary[event_summary["event_id"] == latest_id].iloc[0]
    farm_types = farms["사육업종"].value_counts().head(5).to_dict()
    waste = pd.to_numeric(pig_waste["가축분뇨폐수량"], errors="coerce")
    lines = [
        "# 악취 선제대응 AI Agent 리포트(MVP)", "",
        f"- Event: {latest_id}", f"- 발생 시간대: {event['event_hour']}",
        f"- 전체 신고: {event['reports']}건 / 독립 Grid: {event['unique_grids']}개", "",
        "## 향후 30분 신고 위험 Grid Top 5", "",
        "|순위|중심 위도|중심 경도|신고 발생확률|초기 신고수|", "|---:|---:|---:|---:|---:|",
    ]
    for rank, (_, row) in enumerate(event_pred.iterrows(), 1):
        lines.append(f"|{rank}|{row['center_latitude']:.6f}|{row['center_longitude']:.6f}|{row['hybrid_score']:.1%}|{int(row['initial_count'])}|")
    lines += [
        "", "## 시설 데이터 현황", "",
        f"- 익산 축산농가 {len(farms):,}곳: {json.dumps(farm_types, ensure_ascii=False)}",
        f"- 돼지농장 분뇨자료 {len(pig_waste):,}곳, 폐수량 합계 {waste.sum():,.2f}",
        "- 농가 파일에는 주소만 있고 좌표가 없으므로 현재 리포트에서는 거리 기반 후보 순위를 제시하지 않음.",
        "- 주소 좌표화 이후 풍상 방향·거리·축종·사육두수·분뇨량으로 점검 후보를 산정해야 함.",
        "", "## 권고 대응", "",
        "1. 위험확률 상위 Grid에 이동형 악취 측정기를 우선 배치한다.",
        "2. Event 초기 30분 이후 새로 활성화될 가능성이 높은 Grid를 순찰한다.",
        "3. 농가 좌표와 기상자료가 연결되면 풍상 방향 시설부터 운영·분뇨처리 상태를 확인한다.",
        "4. 모델 결과는 신고 발생 가능성이며 실제 악취 농도 또는 원인 시설 확정 결과가 아니다.",
        "", "## 모델 근거와 한계", "",
        f"- Early Prediction 테스트 Event: {metrics['test_events']}개",
        f"- 선택 모델({metrics['selected_model']}) PR-AUC: {metrics['selected_model_metrics']['pr_auc']:.3f}",
        f"- Baseline PR-AUC: {metrics['persistence_baseline']['pr_auc']:.3f}",
        f"- 센서 중첩 신고: {sensor_case['overlap_complaints']}건",
        f"- 센서 검증 한계: {sensor_case['limitation']}",
    ]
    return "\n".join(lines)


def create_plots(recon: pd.DataFrame, pred: pd.DataFrame, sensor_time: pd.DataFrame) -> None:
    """복원·예측 성능과 센서 사례를 시각화한다."""
    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.figure(figsize=(8, 5))
    plt.hist(recon.loc[recon["target"] == 0, "model_score"], bins=20, alpha=.65, label="비활성 Grid")
    plt.hist(recon.loc[recon["target"] == 1, "model_score"], bins=20, alpha=.65, label="숨긴 활성 Grid")
    plt.xlabel("복원 확률"); plt.ylabel("Grid 수"); plt.title("Virtual Sensor 공간 마스킹 복원 결과"); plt.legend(); plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "virtual_sensor_reconstruction.png", dpi=160); plt.close()

    last = pred.sort_values("event_hour").iloc[-1]["event_id"]
    p = pred[pred["event_id"] == last]
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(p["center_longitude"], p["center_latitude"], c=p["hybrid_score"], s=55, cmap="YlOrRd", vmin=0, vmax=max(.5, p["hybrid_score"].max()))
    actual = p[p["target"] == 1]
    plt.scatter(actual["center_longitude"], actual["center_latitude"], facecolors="none", edgecolors="blue", s=110, label="실제 다음 30분 신고 Grid")
    plt.colorbar(sc, label="예측확률"); plt.legend(); plt.xlabel("경도"); plt.ylabel("위도"); plt.title(f"{last} 다음 30분 Grid 예측"); plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "early_prediction_grid.png", dpi=160); plt.close()

    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax1.plot(sensor_time["날짜"], sensor_time["sensor_anomaly"], color="#d62728", label="센서 최대 이상도")
    ax1.set_ylabel("센서 이상도"); ax2 = ax1.twinx()
    ax2.bar(sensor_time["날짜"], sensor_time["complaints"], width=0.0025, alpha=.45, color="#1f77b4", label="민원")
    ax2.set_ylabel("5분 민원 수"); ax1.set_title("2024-09-02 무인악취 센서와 민원 시간 비교")
    fig.autofmt_xdate(); fig.tight_layout(); plt.savefig(OUTPUT_DIR / "sensor_complaint_case.png", dpi=160); plt.close()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    complaints, sensor, farms, pig_waste, wanju = load_inputs()
    complaints, grid_meta = add_grid_columns(complaints)

    clustered, cluster_summary = st_dbscan(complaints)
    events, event_summary = build_bounded_events(complaints)
    reconstruction, reconstruction_metrics = run_reconstruction_experiment(events, event_summary)
    prediction, prediction_metrics = run_early_prediction_experiment(events, event_summary)
    prediction = enrich_prediction_coordinates(prediction, grid_meta)
    _, sensor_time, sensor_case = validate_sensor_case(complaints, sensor)

    cluster_summary.to_csv(OUTPUT_DIR / "st_dbscan_events.csv", index=False, encoding="utf-8-sig")
    clustered[["datetime", "latitude", "longitude", "region", "intensity", "odor_type", "grid_id", "st_cluster"]].to_csv(
        OUTPUT_DIR / "st_dbscan_labels.csv", index=False, encoding="utf-8-sig")
    event_summary.to_csv(OUTPUT_DIR / "bounded_event_summary.csv", index=False, encoding="utf-8-sig")
    reconstruction.to_csv(OUTPUT_DIR / "virtual_sensor_test_predictions.csv", index=False, encoding="utf-8-sig")
    prediction.to_csv(OUTPUT_DIR / "early_prediction_test_predictions.csv", index=False, encoding="utf-8-sig")
    sensor_time.to_csv(OUTPUT_DIR / "sensor_complaint_timeline.csv", index=False, encoding="utf-8-sig")
    with (OUTPUT_DIR / "model_metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"grid": grid_meta, "reconstruction": reconstruction_metrics, "early_prediction": prediction_metrics,
                   "sensor_case": sensor_case, "wanju_farms": len(wanju)}, f, ensure_ascii=False, indent=2)
    report = build_agent_report(prediction, event_summary, farms, pig_waste, sensor_case, prediction_metrics)
    (OUTPUT_DIR / "agent_response.md").write_text(report, encoding="utf-8")
    create_plots(reconstruction, prediction, sensor_time)

    print("===== 악취 AI MVP 실행 결과 =====")
    print(f"ST-DBSCAN 군집: {len(cluster_summary)}개")
    print(f"학습 대상 독립 1시간 Event: {len(event_summary)}개")
    print(f"Virtual Sensor PR-AUC: {reconstruction_metrics['selected_model_metrics']['pr_auc']:.3f} "
          f"(거리 Baseline {reconstruction_metrics['distance_baseline']['pr_auc']:.3f})")
    print(f"Early Prediction PR-AUC: {prediction_metrics['selected_model_metrics']['pr_auc']:.3f} "
          f"(지속성 Baseline {prediction_metrics['persistence_baseline']['pr_auc']:.3f})")
    print(f"Early Prediction Top-K Recall: {prediction_metrics['model_topk_recall']:.3f} "
          f"(Baseline {prediction_metrics['baseline_topk_recall']:.3f})")
    print(f"Hybrid Top-K Recall: {prediction_metrics['hybrid_topk_recall']:.3f} "
          f"(검증셋 선택 ML 가중치 {prediction_metrics['hybrid_model_weight_selected_on_validation']:.2f})")
    print(f"센서 사례일 중첩 민원: {sensor_case['overlap_complaints']}건")
    print(f"산출물: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
