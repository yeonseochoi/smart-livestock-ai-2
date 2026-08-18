"""Evaluate farm and manure-source features for 30-minute odor-area prediction.

VWorld geocoding results are kept only in process memory. Raw geocoded coordinates
are not written to disk because the API usage notice prohibits separate storage.
"""
from __future__ import annotations

import json
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import average_precision_score

import build_odor_ai_mvp as odor
import experiment_weather_model as weather_exp


API_URL = "https://api.vworld.kr/req/address"
WEATHER_FILE = Path("outputs/weather_integration/event_weather_features.csv")
OUTPUT_DIR = Path("outputs/vworld_source_experiment")

SOURCE_FEATURES = [
    "farm_nearest_km", "farm_count_1km", "farm_count_3km", "farm_count_5km",
    "farm_size_decay", "animal_count_decay", "pig_count_decay", "cattle_count_decay",
    "poultry_count_decay", "manure_nearest_km", "manure_count_3km",
    "manure_volume_decay", "source_downwind_score", "pig_downwind_score",
    "manure_downwind_score",
]


def load_key() -> str:
    key = os.environ.get("VWORLD_API_KEY", "").strip()
    if not key and Path(".env").exists():
        for raw in Path(".env").read_text(encoding="utf-8-sig").splitlines():
            if raw.strip().startswith("VWORLD_API_KEY="):
                key = raw.split("=", 1)[1].strip().strip('"').strip("'")
                break
    if not key:
        raise RuntimeError("VWORLD_API_KEY is missing from .env.")
    return key


def normalize_address(value: object) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    text = re.sub(r"\([^)]*\)", "", text).strip()
    text = re.sub(r"(\d+)번지\s*(\d+)호", r"\1-\2", text)
    text = re.sub(r"(\d+)번지", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def address_variants(value: object) -> list[str]:
    address = normalize_address(value)
    variants = [address]
    if address.startswith("전북특별자치도"):
        variants.extend([
            address.replace("전북특별자치도", "전라북도", 1),
            address.replace("전북특별자치도 ", "", 1),
        ])
    return list(dict.fromkeys(item for item in variants if item))


def geocode_one(address: str, key: str, session: requests.Session) -> dict[str, object]:
    # The current province name is accepted by VWorld. Additional variants are
    # attempted only after the direct query fails, limiting API traffic.
    for query_index, query in enumerate(address_variants(address)):
        for address_type in ("parcel", "road"):
            for attempt in range(3):
                try:
                    response = session.get(API_URL, params={
                        "service": "address", "request": "GetCoord", "version": "2.0",
                        "crs": "EPSG:4326", "address": query, "refine": "true",
                        "simple": "false", "format": "json", "type": address_type, "key": key,
                    }, timeout=20)
                except requests.RequestException:
                    time.sleep(0.5 + attempt)
                    continue
                if response.status_code == 429:
                    time.sleep(1.0 + attempt)
                    continue
                if response.status_code >= 500:
                    time.sleep(0.5 + attempt)
                    continue
                if response.status_code >= 400:
                    raise RuntimeError(f"VWorld HTTP error: {response.status_code}")
                payload = response.json().get("response", {})
                if payload.get("status") == "OK" and (payload.get("result") or {}).get("point"):
                    point = payload["result"]["point"]
                    return {"latitude": float(point["y"]), "longitude": float(point["x"])}
                if payload.get("status") == "ERROR":
                    error = payload.get("error") or {}
                    code = str(error.get("code", ""))
                    if code and code not in {"NOT_FOUND", "INVALID_REQUEST"}:
                        raise RuntimeError(f"VWorld API error {code}: {error.get('text', '')}")
                break
        if query_index == 0 and not query.startswith("전북특별자치도"):
            break
    return {"latitude": np.nan, "longitude": np.nan}


def geocode_addresses(addresses: pd.Series, key: str, workers: int) -> dict[str, dict[str, object]]:
    unique = sorted(set(addresses.dropna().astype(str).str.strip()) - {""})
    results: dict[str, dict[str, object]] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for address in unique:
            session = requests.Session()
            futures[pool.submit(geocode_one, address, key, session)] = address
        for index, future in enumerate(as_completed(futures), 1):
            address = futures[future]
            try:
                results[address] = future.result()
            except Exception as exc:
                results[address] = {"latitude": np.nan, "longitude": np.nan}
                print(f"  skipped one address: {type(exc).__name__}", flush=True)
            if index % 25 == 0 or index == len(unique):
                matched = sum(pd.notna(item["latitude"]) for item in results.values())
                print(f"  geocoding {index}/{len(unique)}, matched={matched}", flush=True)
    return results


def numeric(series: pd.Series) -> np.ndarray:
    cleaned = series.astype(str).str.replace(",", "", regex=False).str.extract(r"([-+]?\d*\.?\d+)")[0]
    return pd.to_numeric(cleaned, errors="coerce").fillna(0).to_numpy(float)


def project_sources(frame: pd.DataFrame, address_col: str, coords: dict, meta: dict) -> pd.DataFrame:
    result = frame.copy()
    lookup_address = result[address_col].astype(str).str.strip()
    result["latitude"] = lookup_address.map(lambda x: coords.get(x, {}).get("latitude", np.nan))
    result["longitude"] = lookup_address.map(lambda x: coords.get(x, {}).get("longitude", np.nan))
    result = result.dropna(subset=["latitude", "longitude"]).copy()
    result["x_km"] = (
        (result["longitude"] - meta["lon0"]) * 111.320 * math.cos(math.radians(meta["lat0"]))
    )
    result["y_km"] = (result["latitude"] - meta["lat0"]) * 110.540
    return result


def prepare_source_arrays(farms: pd.DataFrame, waste: pd.DataFrame) -> dict[str, np.ndarray]:
    animal = farms["사육업종"].fillna("").astype(str)
    heads = numeric(farms["사육두수"])
    area = numeric(farms["시설면적(제곱미터)"])
    return {
        "farm_x": farms["x_km"].to_numpy(float), "farm_y": farms["y_km"].to_numpy(float),
        "farm_area": np.log1p(area), "farm_heads": np.log1p(heads),
        "pig": np.log1p(heads) * animal.str.contains("돼지").to_numpy(float),
        "cattle": np.log1p(heads) * animal.str.contains("한우|육우|젖소").to_numpy(float),
        "poultry": np.log1p(heads) * animal.str.contains("계|닭|오리|부화").to_numpy(float),
        "waste_x": waste["x_km"].to_numpy(float), "waste_y": waste["y_km"].to_numpy(float),
        "waste_volume": np.log1p(numeric(waste["가축분뇨폐수량"])),
    }


def decay_features(dx: np.ndarray, dy: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    distance = np.hypot(dx, dy)
    decay = np.exp(-distance / 2.0)
    return distance, decay * values


def wind_score(dx: np.ndarray, dy: np.ndarray, values: np.ndarray, east: float, north: float) -> float:
    distance = np.hypot(dx, dy)
    alignment = np.divide(dx * east + dy * north, distance, out=np.zeros_like(distance), where=distance > 1e-9)
    return float(np.sum(values * np.exp(-distance / 3.0) * np.maximum(alignment, 0.0)))


def add_source_features(
    candidates: pd.DataFrame, arrays: dict[str, np.ndarray], meta: dict,
    event_weather: dict[str, dict[str, float]],
) -> pd.DataFrame:
    rows = []
    scale = meta["grid_m"] / 1000.0
    for row in candidates.itertuples(index=False):
        x = (float(row.grid_x) + 0.5) * scale
        y = (float(row.grid_y) + 0.5) * scale
        fdx, fdy = x - arrays["farm_x"], y - arrays["farm_y"]
        farm_distance, area_decay = decay_features(fdx, fdy, arrays["farm_area"])
        _, head_decay = decay_features(fdx, fdy, arrays["farm_heads"])
        wdx, wdy = x - arrays["waste_x"], y - arrays["waste_y"]
        waste_distance, volume_decay = decay_features(wdx, wdy, arrays["waste_volume"])
        wind = event_weather[row.event_id]
        east, north = wind["downwind_east"], wind["downwind_north"]
        rows.append([
            float(farm_distance.min(initial=99.0)), float(np.sum(farm_distance <= 1)),
            float(np.sum(farm_distance <= 3)), float(np.sum(farm_distance <= 5)),
            float(area_decay.sum()), float(head_decay.sum()),
            float((np.exp(-farm_distance / 2.0) * arrays["pig"]).sum()),
            float((np.exp(-farm_distance / 2.0) * arrays["cattle"]).sum()),
            float((np.exp(-farm_distance / 2.0) * arrays["poultry"]).sum()),
            float(waste_distance.min(initial=99.0)), float(np.sum(waste_distance <= 3)),
            float(volume_decay.sum()),
            wind_score(fdx, fdy, arrays["farm_heads"], east, north),
            wind_score(fdx, fdy, arrays["pig"], east, north),
            wind_score(wdx, wdy, arrays["waste_volume"], east, north),
        ])
    result = candidates.copy()
    result[SOURCE_FEATURES] = pd.DataFrame(rows, index=result.index)
    return result


def select_configuration(train: pd.DataFrame, events: pd.DataFrame) -> tuple[str, str, list[str], dict]:
    ordered = (train[["event_id", "event_hour"]].drop_duplicates()
               .sort_values("event_hour")["event_id"].tolist())
    configs = {
        "base": weather_exp.BASE_FEATURES,
        "source": weather_exp.BASE_FEATURES + SOURCE_FEATURES,
        "weather": weather_exp.BASE_FEATURES + weather_exp.DIRECTION_FEATURES + weather_exp.REGIME_FEATURES,
        "source_weather": weather_exp.BASE_FEATURES + weather_exp.DIRECTION_FEATURES
                          + weather_exp.REGIME_FEATURES + SOURCE_FEATURES,
    }
    validation = {}
    best = None
    best_key = (-np.inf, -np.inf)
    for config_name, features in configs.items():
        _, scores, fit, valid = odor.select_model_on_inner_validation(
            train, ordered, features, events, target_window="future"
        )
        weather_exp.add_boosting_scores(fit, valid, features, scores)
        model_name = max(scores, key=lambda name: (scores[name]["pr_auc"], scores[name]["topk_recall"]))
        validation[config_name] = {"selected_model": model_name, "models": scores}
        score = scores[model_name]
        key = (score["pr_auc"], score["topk_recall"])
        if key > best_key:
            best_key = key
            best = (config_name, model_name, features)
    assert best is not None
    return *best, validation


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not WEATHER_FILE.exists():
        raise FileNotFoundError(f"Run fetch_kma_weather.py first: {WEATHER_FILE}")

    key = load_key()
    complaints, _, farms, waste, _ = odor.load_inputs()
    complaints, meta = odor.add_grid_columns(complaints)
    events, summary = odor.build_bounded_events(complaints)
    weather_frame = pd.read_csv(WEATHER_FILE, encoding="utf-8-sig", parse_dates=["event_hour"])
    event_weather = weather_exp.weather_lookup(weather_frame)
    candidates, train_ids, test_ids = weather_exp.build_candidates(events, summary, event_weather)

    print("[1/3] VWorld live geocoding (coordinates are not persisted)", flush=True)
    all_addresses = pd.concat([farms["소재지"], waste["주소"]], ignore_index=True)
    coords = geocode_addresses(all_addresses, key, args.workers)
    farms_geo = project_sources(farms, "소재지", coords, meta)
    waste_geo = project_sources(waste, "주소", coords, meta)
    print(f"  farms={len(farms_geo)}/{len(farms)}, manure={len(waste_geo)}/{len(waste)}", flush=True)
    if farms_geo.empty or waste_geo.empty:
        raise RuntimeError("Too few VWorld matches to build source features.")

    print("[2/3] Building candidate source and wind-alignment features", flush=True)
    candidates = add_source_features(candidates, prepare_source_arrays(farms_geo, waste_geo), meta, event_weather)
    train = candidates[candidates["is_train"]].copy()
    test = candidates[~candidates["is_train"]].copy()

    print("[3/3] Selecting on inner validation and evaluating future test events", flush=True)
    config_name, model_name, features, validation = select_configuration(train, events)
    result, metrics, _ = weather_exp.evaluate(train, test, features, model_name)
    base_result, base_metrics, _ = weather_exp.evaluate(train, test, weather_exp.BASE_FEATURES, "extra_leaf4")
    comparison = {
        "experiment": "vworld_live_farm_and_manure_source_features",
        "coordinate_storage": "none; VWorld responses used in process memory only",
        "source_snapshot_warning": "2024 source inventory applied to 2020-2025 events",
        "geocoding": {
            "unique_addresses": len(coords),
            "matched_addresses": sum(pd.notna(item["latitude"]) for item in coords.values()),
            "farms_matched": len(farms_geo), "farms_total": len(farms),
            "manure_matched": len(waste_geo), "manure_total": len(waste),
        },
        "train_events": len(train_ids), "test_events": len(test_ids),
        "selected_configuration": config_name, "selected_model": model_name,
        "selected_features": features, "inner_validation": validation,
        "base_reference": base_metrics, "source_test": metrics,
        "delta_vs_base_reference": {
            "pr_auc": metrics["selected_model_metrics"]["pr_auc"] - base_metrics["selected_model_metrics"]["pr_auc"],
            "roc_auc": metrics["selected_model_metrics"]["roc_auc"] - base_metrics["selected_model_metrics"]["roc_auc"],
            "topk_recall": metrics["model_topk_recall"] - base_metrics["model_topk_recall"],
        },
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    output_columns = ["event_id", "event_hour", "grid_x", "grid_y", "target", "model_score", "persistence_baseline"]
    result[output_columns].to_csv(OUTPUT_DIR / "test_predictions.csv", index=False, encoding="utf-8-sig")
    print(json.dumps({
        "selected_configuration": config_name, "selected_model": model_name,
        "base_pr_auc": base_metrics["selected_model_metrics"]["pr_auc"],
        "source_pr_auc": metrics["selected_model_metrics"]["pr_auc"],
        "base_topk_recall": base_metrics["model_topk_recall"],
        "source_topk_recall": metrics["model_topk_recall"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
