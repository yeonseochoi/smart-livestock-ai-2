"""기상·발생원 역추적 특징의 소거 실험(ablation study).

`compare_operational_grid_sizes.py`와 같은 공통 Event(1km·1.5km·2km 교집합), 시간순 70/30 분할,
동일 지표(PR-AUC, Top-K Recall, Recall@3, Event Hit@3)로 아래 실험군을 비교한다.

  M0  기존 특징만 (기준선)
  M1  M0 + ASOS 시간 자료로 만든 Event 이전 3시간 바람·강수 특징
  M2  M0 + Track A 역추적 점수 (outputs/source_backtrack/grid_scores.csv)
  M3  M0 점수와 source_fit_score의 후결합(late fusion, 가중합) — 가중치는 내부검증으로 선택

모델·가중치 선택은 학습 구간 마지막 20% Event 내부검증으로만 하고 최종 테스트는 실험군당 1회다.
Track A 산출물이 없으면 M2·M3는 건너뛰고 M0·M1만 돌린다(컬럼 부재는 오류가 아니라 분기).

실행: python run_ablation.py [--backtrack PATH] [--allow-fixture] [--rebuild-cache]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import build_odor_ai_mvp as odor
import compare_operational_grid_sizes as compare
import optimize_early_prediction as optimize
import sensitivity_early_prediction as sensitivity

OUTPUT_DIR = Path("outputs/ablation")
CACHE_PATH = OUTPUT_DIR / "base_dataset_1000m.csv"
ASOS_PATH = Path("outputs/weather_integration/asos_hourly_2020_2026.csv")
BACKTRACK_PATH = Path("outputs/source_backtrack/grid_scores.csv")
FIXTURE_PATH = Path("tests/fixtures/grid_scores_sample.csv")
GRID_M = 1000
LAG_HOURS = 3
MODELS = compare.MODELS
LIVESTOCK_LABEL = "가축"
FUSION_WEIGHTS = (0.0, 0.25, 0.5, 1.0, 2.0)
SEEDS = (42, 7, 123, 2024, 31)
DECISION_MARGIN = 0.01

# M1: Event 이전 3시간 ASOS. 격자별로 달라지는 값은 downwind 정렬 하나뿐이고 나머지는 Event 상수다.
WEATHER_FEATURES = ["asos_downwind_alignment", "asos_wind_speed_3h", "asos_rain_3h", "asos_stagnation_flag"]
# M2: Track A 계약 컬럼 중 물리적으로 설명 가능한 5개만 쓴다.
BACKTRACK_FEATURES = ["source_fit_score", "forward_plume_score", "lagged_wind_alignment", "rain_3h", "stagnation_flag"]
KEY = ["event_hour", "grid_x", "grid_y"]


# ---------------------------------------------------------------- 기준 데이터

def build_base_dataset(rebuild: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """compare_operational_grid_sizes와 동일한 공통 Event·분할의 1km 데이터를 만든다.

    반환: (후보 격자 표, Event 요약(initial 30분 기준 중심·축산 비율), 격자 메타)
    """
    complaints, _, _, _, _ = odor.load_inputs()
    meta = {
        "lat0": float(complaints["latitude"].median()),
        "lon0": float(complaints["longitude"].median()),
        "grid_m": GRID_M,
    }
    original, _ = odor.add_grid_columns(complaints)
    _, selected_hours = odor.build_bounded_events(original)

    if CACHE_PATH.exists() and not rebuild:
        data = pd.read_csv(CACHE_PATH, parse_dates=["event_hour"], encoding="utf-8-sig")
    else:
        datasets = {}
        for grid_m in compare.GRID_SIZES:
            frame, _, _ = sensitivity.build_data(complaints, selected_hours, grid_m, 30, 30)
            datasets[grid_m] = frame
        common_ids = set.intersection(*(set(frame["event_id"]) for frame in datasets.values()))
        common_summary = (selected_hours[selected_hours["event_id"].isin(common_ids)]
                          .sort_values("event_hour").drop_duplicates("event_id"))
        train_ids, _ = odor.split_event_ids(common_summary)
        data = datasets[GRID_M][datasets[GRID_M]["event_id"].isin(common_ids)].copy()
        data["is_train"] = data["event_id"].isin(train_ids)
        data["event_hour"] = pd.to_datetime(data["event_hour"])
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        data.to_csv(CACHE_PATH, index=False, encoding="utf-8-sig")

    events = summarize_events(complaints, data)
    return data, events, meta


def summarize_events(complaints: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
    """Event별로 처음 30분 민원만 써서 중심 좌표와 축산 비율을 만든다(누수 없음)."""
    work = complaints.copy()
    work["event_hour"] = work["datetime"].dt.floor("h")
    hours = data[["event_hour", "event_id"]].drop_duplicates()
    work = work.merge(hours, on="event_hour", how="inner")
    initial = work[work["datetime"] < work["event_hour"] + pd.Timedelta(minutes=odor.INPUT_MINUTES)]
    initial = initial.assign(is_livestock=initial["odor_type"].astype(str).str.startswith(LIVESTOCK_LABEL))
    summary = initial.groupby(["event_hour", "event_id"]).agg(
        centroid_latitude=("latitude", "mean"), centroid_longitude=("longitude", "mean"),
        initial_reports=("datetime", "size"), livestock_share=("is_livestock", "mean"),
    ).reset_index()
    return summary


# ---------------------------------------------------------------- M1: ASOS 3시간

def load_asos() -> pd.DataFrame | None:
    if not ASOS_PATH.exists():
        return None
    asos = pd.read_csv(ASOS_PATH, parse_dates=["datetime"], encoding="utf-8-sig")
    # 기상 풍향은 "불어오는" 방향이다. 벡터 평균을 위해 u/v로 바꾼다.
    rad = np.radians(asos["wind_direction"])
    asos["u"] = -asos["wind_speed"] * np.sin(rad)
    asos["v"] = -asos["wind_speed"] * np.cos(rad)
    return asos


def event_weather(events: pd.DataFrame, asos: pd.DataFrame) -> pd.DataFrame:
    """Event 시각 이전 3시간(정시 값 3개)의 관측소 거리 가중 평균 바람과 강수 합."""
    stations = asos.groupby("station_id")[["station_latitude", "station_longitude"]].first()
    rows = []
    for row in events.itertuples(index=False):
        hour = pd.Timestamp(row.event_hour)
        window = asos[(asos["datetime"] > hour - pd.Timedelta(hours=LAG_HOURS)) & (asos["datetime"] <= hour)]
        if window.empty:
            rows.append([row.event_hour, row.event_id, np.nan, np.nan, np.nan, np.nan, np.nan, "none"])
            continue
        weights = {}
        for station_id, position in stations.iterrows():
            d = haversine_km(row.centroid_latitude, row.centroid_longitude,
                             position["station_latitude"], position["station_longitude"])
            weights[station_id] = 1.0 / max(d, 1.0) ** 2
        per_station = window.groupby("station_id").agg(
            u=("u", "mean"), v=("v", "mean"), speed=("wind_speed", "mean"),
            rain=("rainfall_hour", "sum"), direction_sd=("wind_direction", circular_sd),
        )
        w = np.array([weights[s] for s in per_station.index], dtype=float)
        w = w / w.sum()
        u = float(np.nansum(per_station["u"] * w))
        v = float(np.nansum(per_station["v"] * w))
        speed = float(np.nansum(per_station["speed"] * w))
        rain = float(np.nansum(per_station["rain"] * w))
        direction_sd = float(np.nansum(per_station["direction_sd"] * w))
        blowing_to = (math.degrees(math.atan2(u, v)) + 360.0) % 360.0
        stagnation = int(speed < 1.0 or direction_sd > 45.0)
        rows.append([row.event_hour, row.event_id, blowing_to, speed, rain, direction_sd, stagnation, "asos"])
    columns = ["event_hour", "event_id", "asos_blowing_to_deg", "asos_wind_speed_3h", "asos_rain_3h",
               "asos_direction_sd", "asos_stagnation_flag", "asos_source"]
    return pd.DataFrame(rows, columns=columns)


def circular_sd(degrees: pd.Series) -> float:
    values = degrees.dropna().to_numpy(float)
    if len(values) == 0:
        return float("nan")
    rad = np.radians(values)
    r = math.hypot(np.sin(rad).mean(), np.cos(rad).mean())
    return float(math.degrees(math.sqrt(-2.0 * math.log(max(r, 1e-9)))))


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi, dlmb = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * odor.EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlmb = math.radians(lon2 - lon1)
    x = math.sin(dlmb) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dlmb)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def add_weather_features(data: pd.DataFrame, events: pd.DataFrame, weather: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """격자별 downwind 정렬(현재 민원 중심 → 격자 방위와 바람이 부는 방향의 cos) + Event 상수 3개."""
    result = data.merge(events[["event_hour", "event_id", "centroid_latitude", "centroid_longitude"]],
                        on=["event_hour", "event_id"], how="left")
    result = result.merge(weather, on=["event_hour", "event_id"], how="left")
    alignment = []
    for row in result.itertuples(index=False):
        if pd.isna(row.asos_blowing_to_deg):
            alignment.append(np.nan)
            continue
        lat, lon = odor.grid_centroid(int(row.grid_x), int(row.grid_y), meta)
        d = haversine_km(row.centroid_latitude, row.centroid_longitude, lat, lon)
        if d < 0.05:
            alignment.append(0.0)  # 중심 칸 자체는 방향이 정의되지 않는다.
            continue
        theta = bearing_deg(row.centroid_latitude, row.centroid_longitude, lat, lon)
        alignment.append(math.cos(math.radians(theta - row.asos_blowing_to_deg)))
    result["asos_downwind_alignment"] = alignment
    return result


# ---------------------------------------------------------------- M2: Track A 어댑터

def load_backtrack(path: Path, allow_fixture: bool) -> tuple[pd.DataFrame | None, dict]:
    """Track A 산출물을 검증기로 확인한 뒤 읽는다. 없으면 (None, 사유)."""
    info: dict = {"path": str(path), "status": "ok"}
    used = path
    if not path.exists():
        if not (allow_fixture and FIXTURE_PATH.exists()):
            info["status"] = "missing"
            return None, info
        used, info["path"], info["status"] = FIXTURE_PATH, str(FIXTURE_PATH), "fixture"
    scores = pd.read_csv(used, parse_dates=["event_hour", "history_cutoff"], encoding="utf-8-sig")
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from tools import validate_backtrack_output as validator
        reference, ref_meta = validator.load_reference()
        failures, summary = validator.validate(
            scores, reference, ref_meta, min_coverage=0 if used == FIXTURE_PATH else 0.5)
        info["validation_failures"] = failures
        info["validation_summary"] = summary
        if failures:
            info["status"] = "invalid"
            return None, info
    except Exception as error:  # 검증기 부재는 실험 중단 사유가 아니다. 기록만 남긴다.
        info["validator_error"] = repr(error)
    info["events"] = int(scores["event_hour"].nunique())
    info["rows"] = int(len(scores))
    return scores, info


def add_backtrack_features(data: pd.DataFrame, scores: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """(event_hour, grid_x, grid_y)로 결합해 특징 5개를 추가한다. 없는 칸은 NaN."""
    picked = scores[KEY + BACKTRACK_FEATURES].copy()
    picked["event_hour"] = pd.to_datetime(picked["event_hour"])
    merged = data.merge(picked, on=KEY, how="left")
    matched_events = merged.loc[merged["source_fit_score"].notna(), "event_id"].nunique()
    info = {
        "matched_rows": int(merged["source_fit_score"].notna().sum()),
        "total_rows": int(len(merged)),
        "matched_events": int(matched_events),
        "total_events": int(merged["event_id"].nunique()),
    }
    return merged, info


# ---------------------------------------------------------------- 실험

def inner_split(train: pd.DataFrame) -> tuple[set[str], set[str]]:
    ordered = (train[["event_id", "event_hour"]].drop_duplicates()
               .sort_values("event_hour")["event_id"].tolist())
    cut = max(1, int(len(ordered) * .8))
    return set(ordered[:cut]), set(ordered[cut:])


def fuse(frame: pd.DataFrame, base_score: np.ndarray, weight: float) -> np.ndarray:
    """M3 후결합: 기존 모델 확률 × (1 + weight × source_fit_score). 하드 필터 없음, 결측은 0 기여.

    Event 안 순위(백분위)로 바꾼 뒤 더하면 Event 간 확률 척도가 사라져 PR-AUC를 M0과 비교할 수 없다.
    곱셈 보정은 척도를 유지하므로 같은 지표로 비교 가능하다.
    """
    aux = frame["source_fit_score"].fillna(0.0).to_numpy(float)
    return base_score * (1.0 + weight * aux)


def seeded_fit_predict(model_name: str, fit: pd.DataFrame, valid: pd.DataFrame, features: list[str],
                       seed: int, final: bool = False) -> tuple[np.ndarray, object]:
    """optimize.fit_predict는 odor.RANDOM_STATE를 호출 시점에 읽는다. 공용 파일을 고치지 않고 seed만 바꾼다."""
    previous = odor.RANDOM_STATE
    odor.RANDOM_STATE = seed
    try:
        return optimize.fit_predict(model_name, fit, valid, features, final=final)
    finally:
        odor.RANDOM_STATE = previous


def mean_metrics(rows: list[dict]) -> dict:
    keys = rows[0].keys()
    out = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    out.update({f"{k}_sd": float(np.std([r[k] for r in rows])) for k in ("pr_auc", "event_hit_rate_at_3") if k in keys})
    return out


def run_arm(name: str, data: pd.DataFrame, features: list[str], train_ids: set[str]) -> dict:
    """내부검증(seed 평균)으로 모델을 고르고 최종 테스트는 seed별 1회씩, 평균과 표준편차를 보고한다.

    같은 코드·데이터라도 XGBoost 수치가 환경에 따라 PR-AUC 0.01 안팎 흔들리는 것을 확인했다
    (커밋된 metrics.json과 이 기기의 재실행 차이). 단일 seed 비교는 그 잡음을 개선으로 오인할 수 있다.
    """
    train, test = data[data["is_train"]].copy(), data[~data["is_train"]].copy()
    fit_ids, valid_ids = inner_split(train)
    fit, valid = sensitivity.prepare_prior(
        train[train["event_id"].isin(fit_ids)], train[train["event_id"].isin(valid_ids)], fit_ids)
    validation = {}
    for model_name in MODELS:
        per_seed = []
        for seed in SEEDS:
            score, _ = seeded_fit_predict(model_name, fit, valid, features, seed)
            per_seed.append(optimize.score_prediction(valid, score))
        validation[model_name] = mean_metrics(per_seed)
    selected = max(validation, key=lambda m: (validation[m]["pr_auc"], validation[m]["topk_recall"]))
    train, test = sensitivity.prepare_prior(train, test, train_ids)
    scores, per_seed, importance = [], [], []
    for seed in SEEDS:
        score, model = seeded_fit_predict(selected, train, test, features, seed, final=True)
        scores.append(score)
        per_seed.append(evaluate(test, score))
        if hasattr(model, "feature_importances_"):
            importance.append(model.feature_importances_)
    ensemble = np.mean(scores, axis=0)
    return {
        "arm": name, "features": features, "selected_model": selected, "seeds": list(SEEDS),
        "inner_validation": validation, "inner_selected": validation[selected],
        "test": mean_metrics(per_seed), "test_seed_ensemble": evaluate(test, ensemble),
        "feature_importance": ({f: float(v) for f, v in zip(features, np.mean(importance, axis=0))}
                               if importance else {}),
        "_test_frame": test, "_test_score": ensemble, "_valid_frame": valid,
        "_fit": fit, "_valid_ids": valid_ids,
    }


def evaluate(frame: pd.DataFrame, score: np.ndarray) -> dict:
    metrics = optimize.score_prediction(frame, score)
    metrics.update(compare.fixed_k_metrics(frame, score))
    metrics["events"] = int(frame["event_id"].nunique())
    metrics["positive_rate"] = float(frame["target"].mean())
    return metrics


def subset_tables(arms: dict[str, dict], test_meta: pd.DataFrame) -> dict:
    """전체 / 건조·강수 / 축산 비율 / 기상 자료 유무별 테스트 지표. 분할 기준은 모두 Event 상수다."""
    conditions = {
        "all": lambda m: pd.Series(True, index=m.index),
        "dry_rain3h_eq_0": lambda m: m["asos_rain_3h"].fillna(-1) == 0,
        "wet_rain3h_gt_0": lambda m: m["asos_rain_3h"].fillna(-1) > 0,
        "livestock_share_ge_50": lambda m: m["livestock_share"] >= 0.5,
        "livestock_share_lt_50": lambda m: m["livestock_share"] < 0.5,
        "weather_available": lambda m: m["asos_source"] == "asos",
        "weather_none": lambda m: m["asos_source"] != "asos",
    }
    tables = {}
    for condition, rule in conditions.items():
        keep_events = set(test_meta.loc[rule(test_meta), "event_id"])
        tables[condition] = {"events": len(keep_events), "arms": {}}
        if not keep_events:
            continue
        for name, arm in arms.items():
            frame = arm["_test_frame"]
            mask = frame["event_id"].isin(keep_events).to_numpy()
            if frame.loc[mask, "target"].sum() == 0:
                continue
            tables[condition]["arms"][name] = evaluate(frame[mask], arm["_test_score"][mask])
    return tables


def markdown_table(report: dict) -> str:
    lines = ["# 소거 실험 결과 (1km, 공통 Event, 시간순 70/30)", ""]
    lines.append(f"- 공통 Event {report['common_events']}개, 학습 {report['train_events']} / 테스트 {report['test_events']}")
    lines.append(f"- Track A 산출물: {report['backtrack']['status']} ({report['backtrack'].get('path')})")
    lines.append(f"- 판정: {report['decision']['verdict']} — {report['decision']['reason']}")
    gate = report["decision"].get("backtrack_gate", {})
    if gate.get("status") == "ok":
        lines.append(f"- Track A 무작위 풍향 대조 p값 {gate['p_value']:.3f} → "
                     + ("참고 정보로만 사용(계약 기준 0.05 초과)" if gate["reference_only"] else "모델 입력 후보 가능"))
    lines.append("")
    lines.append(f"- seed {report['seeds']} 평균 ± 표준편차. 판정 여유(margin) {report['decision']['margin']:.3f}")
    lines.append("")
    lines.append("| 실험군 | 특징 수 | 선택 모델 | 내부검증 PR-AUC | 테스트 PR-AUC | Top-K Recall | Recall@3 | Event Hit@3 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for name, arm in report["arms"].items():
        t, v = arm["test"], arm["inner_selected"]
        sd = lambda key: f" ±{t[key + '_sd']:.3f}" if key + "_sd" in t else ""
        lines.append(f"| {name} | {len(arm['features'])} | {arm['selected_model']} | {v['pr_auc']:.3f}{' ±%.3f' % v['pr_auc_sd'] if 'pr_auc_sd' in v else ''} | "
                     f"{t['pr_auc']:.3f}{sd('pr_auc')} | {t['topk_recall']:.3f} | {t['recall_at_3']:.3f} | {t['event_hit_rate_at_3']:.3f}{sd('event_hit_rate_at_3')} |")
    lines.append("")
    lines.append("## 조건별 테스트 (Event Hit@3 / PR-AUC)")
    lines.append("")
    arm_names = list(report["arms"])
    lines.append("| 조건 | Event 수 | " + " | ".join(arm_names) + " |")
    lines.append("|---|---|" + "---|" * len(arm_names))
    for condition, table in report["subsets"].items():
        cells = []
        for name in arm_names:
            m = table["arms"].get(name)
            cells.append(f"{m['event_hit_rate_at_3']:.3f} / {m['pr_auc']:.3f}" if m else "—")
        lines.append(f"| {condition} | {table['events']} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def backtrack_gate(path: Path) -> dict:
    """Track A validation.json의 무작위 풍향 대조 p값. 0.05 초과면 계약상 역추적 점수는 참고 정보로만 쓴다."""
    candidate = path.parent / "validation.json"
    if not candidate.exists():
        return {"status": "missing"}
    try:
        report = json.loads(candidate.read_text(encoding="utf-8"))
        p_value = float(report["random_wind_contrast"]["all"]["p_value"])
    except (KeyError, ValueError, TypeError) as error:
        return {"status": "unreadable", "error": repr(error)}
    return {"status": "ok", "p_value": p_value, "reference_only": p_value > 0.05}


def decide(arms: dict[str, dict]) -> dict:
    """내부검증 seed 평균 PR-AUC가 M0보다 여유(margin) 이상 높은 실험군만 채택 후보.

    여유 = max(0.01, M0의 seed 표준편차). 최종 테스트 수치는 판정에 쓰지 않는다.
    """
    base = arms["M0"]["inner_selected"]["pr_auc"]
    margin = max(DECISION_MARGIN, arms["M0"]["inner_selected"].get("pr_auc_sd", 0.0))
    deltas = {name: arm["inner_selected"]["pr_auc"] - base for name, arm in arms.items() if name != "M0"}
    better = {name: d for name, d in deltas.items() if d > margin}
    if not better:
        return {"verdict": "M0 유지", "margin": margin, "deltas": deltas,
                "reason": f"내부검증 PR-AUC에서 M0({base:.3f})을 여유 {margin:.3f} 이상 넘는 실험군 없음"}
    best = max(better, key=better.get)
    return {"verdict": f"{best} 조건부 채택", "margin": margin, "deltas": deltas,
            "reason": f"내부검증 PR-AUC +{better[best]:.3f} (M0 {base:.3f}, 여유 {margin:.3f})"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backtrack", type=Path, default=BACKTRACK_PATH)
    parser.add_argument("--allow-fixture", action="store_true", help="실제 산출물이 없으면 fixture로 M2·M3 코드 경로만 확인")
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data, events, meta = build_base_dataset(rebuild=args.rebuild_cache)
    train_ids = set(data.loc[data["is_train"], "event_id"])
    data = data.merge(events[["event_hour", "event_id", "livestock_share", "initial_reports"]],
                      on=["event_hour", "event_id"], how="left")

    asos = load_asos()
    weather = event_weather(events, asos) if asos is not None else None
    if weather is not None:
        data = add_weather_features(data, events, weather, meta)
    else:
        data["asos_source"] = "none"

    scores, backtrack_info = load_backtrack(args.backtrack, args.allow_fixture)
    if scores is not None:
        data, join_info = add_backtrack_features(data, scores)
        backtrack_info["join"] = join_info

    arms: dict[str, dict] = {}
    arms["M0"] = run_arm("M0", data, list(sensitivity.FEATURES), train_ids)
    if weather is not None:
        arms["M1"] = run_arm("M1", data, list(sensitivity.FEATURES) + WEATHER_FEATURES, train_ids)
    if scores is not None:
        arms["M2"] = run_arm("M2", data, list(sensitivity.FEATURES) + BACKTRACK_FEATURES, train_ids)
        # M3: M0 모델은 그대로 두고 점수만 후결합. 가중치는 M0의 내부검증 구간에서 고른다.
        m0 = arms["M0"]
        valid = m0["_valid_frame"]
        valid_score = np.mean([seeded_fit_predict(m0["selected_model"], m0["_fit"], valid, m0["features"], seed)[0]
                               for seed in SEEDS], axis=0)
        fusion_validation = {}
        for w in FUSION_WEIGHTS:
            fusion_validation[str(w)] = optimize.score_prediction(valid, fuse(valid, valid_score, w))
        best_w = max(FUSION_WEIGHTS, key=lambda w: (fusion_validation[str(w)]["pr_auc"],
                                                     fusion_validation[str(w)]["topk_recall"]))
        test = m0["_test_frame"]
        fused = fuse(test, m0["_test_score"], best_w)
        arms["M3"] = {
            "arm": "M3", "features": m0["features"] + ["source_fit_score(late fusion)"],
            "selected_model": f"{m0['selected_model']}+w{best_w}", "inner_validation": fusion_validation,
            "inner_selected": fusion_validation[str(best_w)], "test": evaluate(test, fused),
            "test_seed_ensemble": evaluate(test, fused), "seeds": list(SEEDS),
            "feature_importance": {}, "_test_frame": test, "_test_score": fused,
        }

    test_meta = (data[~data["is_train"]][["event_id", "event_hour", "livestock_share", "asos_source"]
                 + (["asos_rain_3h"] if "asos_rain_3h" in data.columns else [])]
                 .drop_duplicates("event_id"))
    if "asos_rain_3h" not in test_meta.columns:
        test_meta["asos_rain_3h"] = np.nan
    subsets = subset_tables(arms, test_meta)
    decision = decide(arms)
    gate = backtrack_gate(Path(backtrack_info["path"])) if scores is not None else {"status": "skipped"}
    decision["backtrack_gate"] = gate
    if gate.get("reference_only") and decision["verdict"].startswith(("M2", "M3")):
        decision["verdict"] = "M0 유지"
        decision["reason"] += f"; Track A 무작위 풍향 대조 p={gate['p_value']:.2f}>0.05 → 역추적 점수는 참고 정보로만"

    report = {
        "protocol": "common events (1/1.5/2 km), chronological 70/30 split, 30-min input/forecast, 1 km grid",
        "common_events": int(data["event_id"].nunique()), "train_events": len(train_ids),
        "test_events": int(data.loc[~data["is_train"], "event_id"].nunique()),
        "weather": {
            "source": str(ASOS_PATH) if asos is not None else None, "lag_hours": LAG_HOURS,
            "events_with_weather": int((weather["asos_source"] == "asos").sum()) if weather is not None else 0,
            "features": WEATHER_FEATURES,
        },
        "backtrack": backtrack_info,
        "arms": {name: {k: v for k, v in arm.items() if not k.startswith("_")} for name, arm in arms.items()},
        "subsets": subsets, "decision": decision, "seeds": list(SEEDS),
    }
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "ablation_table.md").write_text(markdown_table(report), encoding="utf-8")

    predictions = arms["M0"]["_test_frame"][["event_id", "event_hour", "grid_x", "grid_y", "target"]].copy()
    for name, arm in arms.items():
        predictions[f"score_{name}"] = arm["_test_score"]
    complaints, _, _, _, _ = odor.load_inputs()
    predictions = compare.add_location_columns(predictions, complaints, GRID_M)
    predictions.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False, encoding="utf-8-sig")
    if weather is not None:
        weather.to_csv(OUTPUT_DIR / "event_weather_3h.csv", index=False, encoding="utf-8-sig")

    print(markdown_table(report))


if __name__ == "__main__":
    main()
