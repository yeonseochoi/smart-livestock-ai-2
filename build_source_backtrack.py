"""축산시설과 Event 이전 ASOS 기상으로 1 km 격자 역추적 적합도를 만든다.

1차 실행은 저장된 ASOS 시간자료만 사용한다. VWorld 응답과 좌표 포함 발생원
테이블은 저장소에 커밋하지 않으며, 후보 격자는 계약 기준표를 그대로 사용한다.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

import build_odor_ai_mvp as odor


ROOT = Path(__file__).resolve().parent
IKSAN_FILE = ROOT / "data" / "전북특별자치도 익산시_축산농가 현황_20241231 (1).csv"
GIMJE_FILE = ROOT / "data" / "전북특별자치도 김제시_축산현황_20250515_geocoded.csv"
REFERENCE_FILE = ROOT / "outputs" / "backtrack_contract" / "reference_candidates.csv"
REFERENCE_META = ROOT / "outputs" / "backtrack_contract" / "reference_meta.json"
ASOS_FILE = ROOT / "outputs" / "weather_integration" / "asos_hourly_2020_2026.csv"
OUTPUT_DIR = ROOT / "outputs" / "source_backtrack"
CACHE_FILE = ROOT / "data" / "vworld_geocode_cache" / "geocode.json"
VWORLD_URL = "https://api.vworld.kr/req/address"

SPECIES_WEIGHT = {
    "돼지": 1.0,
    "한우": 0.3,
    "육우": 0.3,
    "젖소": 0.4,
    "염소": 0.1,
    "산양": 0.1,
    "사슴": 0.1,
    "육계": 0.01,
    "종계/산란계": 0.01,
    "오리": 0.01,
    "메추리": 0.01,
    "부화용알생산": 0.01,
}
MIN_WIND_SPEED = 0.5
MAX_TRAVEL_HOURS = 3.0
DECAY_LENGTH_KM = {"fast": 1.5, "moderate": 2.5, "slow_or_night": 4.0}
TOP_SOURCE_COUNT = 5
RANDOM_SEED = 42
DEFAULT_WIND_LAG = "fixed0"
DEFAULT_MAX_SOURCE_KM = 6.0
COMPARISON_CONFIGS = {
    "travel_10km": {"wind_lag": "travel", "max_source_km": 10.0},
    "fixed0_6km": {"wind_lag": "fixed0", "max_source_km": 6.0},
}

GRID_COLUMNS = [
    "event_hour", "event_id", "grid_x", "grid_y", "center_latitude", "center_longitude",
    "source_fit_score", "forward_plume_score", "prior_downwind_score", "lagged_wind_alignment",
    "travel_time_min", "rain_1h", "rain_3h", "stagnation_flag", "backtrack_uncertainty",
    "weather_source", "history_cutoff",
]
SOURCE_COLUMNS = [
    "source_id", "source_type", "city", "name", "species", "head_count", "area_m2", "status",
    "address", "latitude", "longitude", "location_precision", "geocode_method", "emission_weight",
    "weight_imputed",
]
CANDIDATE_COLUMNS = [
    "event_hour", "rank", "source_id", "name", "city", "species", "location_precision",
    "distance_km", "bearing_deg", "travel_time_min", "wind_alignment", "fit_score", "evidence_text",
]


def load_env_key(name: str, path: Path = ROOT / ".env") -> str:
    """환경변수 또는 .env에서 키를 읽되 값을 출력하지 않는다."""
    key = os.environ.get(name, "").strip()
    if not key and path.exists():
        for raw in path.read_text(encoding="utf-8-sig").splitlines():
            line = raw.strip()
            if line.startswith(f"{name}="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
                break
    if not key:
        raise RuntimeError(f"{name}가 없습니다. .env를 확인하세요.")
    return key


def normalize_address(value: object) -> str:
    """VWorld 조회용 지번 주소를 정규화한다."""
    text = re.sub(r"\s+", " ", str(value or "").strip())
    text = re.sub(r"\([^)]*\)", "", text).strip()
    text = re.sub(r"\s+외\s*\d*(?:필지)?.*$", "", text)
    text = re.sub(r",.*$", "", text)
    text = re.sub(r"(\d+)번지\s*(\d+)호", r"\1-\2", text)
    text = re.sub(r"(\d+)번지", r"\1", text)
    # 한 셀에 공백으로 여러 필지가 이어진 경우 첫 지번만 조회한다.
    text = re.sub(
        r"((?:산\s*)?\d+(?:-\d+)?)(?:\s+(?:산\s*)?\d+(?:-\d+)?)+$", r"\1", text,
    )
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


def gimje_village_address(value: object) -> str | None:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    match = re.search(r"(?:전북특별자치도|전라북도)\s+김제시\s+(\S+(?:읍|면|동))\s+(\S+리)", text)
    if not match:
        return None
    return f"전북특별자치도 김제시 {match.group(1)} {match.group(2)}"


def geocode_one(address: str, key: str) -> dict[str, object]:
    """주소 변형과 parcel/road 조회를 순서대로 시도한다."""
    session = requests.Session()
    for query in address_variants(address):
        for address_type in ("parcel", "road"):
            for attempt in range(3):
                try:
                    response = session.get(VWORLD_URL, params={
                        "service": "address", "request": "GetCoord", "version": "2.0",
                        "crs": "EPSG:4326", "address": query, "refine": "true",
                        "simple": "false", "format": "json", "type": address_type, "key": key,
                    }, timeout=25)
                    if response.status_code == 429 or response.status_code >= 500:
                        time.sleep(0.8 * (attempt + 1))
                        continue
                    response.raise_for_status()
                    payload = response.json().get("response", {})
                    if payload.get("status") == "OK" and (payload.get("result") or {}).get("point"):
                        point = payload["result"]["point"]
                        return {
                            "latitude": float(point["y"]), "longitude": float(point["x"]),
                            "query": query, "address_type": address_type, "status": "OK",
                        }
                    error = payload.get("error") or {}
                    code = str(error.get("code", ""))
                    if payload.get("status") == "ERROR" and code not in {"", "NOT_FOUND", "INVALID_REQUEST"}:
                        raise RuntimeError(f"VWorld API 오류: {code}")
                    break
                except (requests.RequestException, ValueError):
                    if attempt == 2:
                        break
                    time.sleep(0.8 * (attempt + 1))
    return {"latitude": None, "longitude": None, "query": address, "address_type": None, "status": "NOT_FOUND"}


def load_geocode_cache() -> dict[str, dict[str, object]]:
    if not CACHE_FILE.exists():
        return {}
    return json.loads(CACHE_FILE.read_text(encoding="utf-8"))


def save_geocode_cache(cache: dict[str, dict[str, object]]) -> None:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def geocode_addresses(addresses: Iterable[str], key: str, workers: int = 6) -> dict[str, dict[str, object]]:
    cache = load_geocode_cache()
    unique = sorted({str(x).strip() for x in addresses if x and str(x).strip()})
    missing = [x for x in unique if x not in cache]
    if missing:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(geocode_one, address, key): address for address in missing}
            for index, future in enumerate(as_completed(futures), 1):
                address = futures[future]
                try:
                    cache[address] = future.result()
                except Exception as exc:  # noqa: BLE001
                    cache[address] = {
                        "latitude": None, "longitude": None, "query": address,
                        "address_type": None, "status": f"ERROR:{type(exc).__name__}",
                    }
                if index % 25 == 0 or index == len(missing):
                    save_geocode_cache(cache)
                    matched = sum(cache[x].get("latitude") is not None for x in unique if x in cache)
                    print(f"  VWorld {index}/{len(missing)} 신규 조회, 전체 성공 {matched}/{len(unique)}", flush=True)
    return cache


def numeric(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.replace(",", "", regex=False).str.extract(r"([-+]?\d*\.?\d+)")[0]
    return pd.to_numeric(cleaned, errors="coerce")


def prepare_sources(workers: int = 6) -> tuple[pd.DataFrame, dict[str, object]]:
    """익산 point와 김제 point/village 후보를 계약 스키마로 결합한다."""
    if not IKSAN_FILE.exists() or not GIMJE_FILE.exists():
        raise FileNotFoundError("익산 또는 김제 축산시설 원본이 없습니다.")
    key = load_env_key("VWORLD_API_KEY")
    iksan = pd.read_csv(IKSAN_FILE, encoding="utf-8-sig")
    gimje = pd.read_csv(GIMJE_FILE, encoding="utf-8-sig")

    iksan_lookup = iksan["소재지"].map(normalize_address)
    gimje_village = gimje["소재지"].map(gimje_village_address)
    lookup = pd.concat([iksan_lookup, gimje_village[gimje["위도"].isna()]], ignore_index=True)
    cache = geocode_addresses(lookup.dropna().tolist(), key, workers=workers)

    iksan_lat = iksan_lookup.map(lambda x: cache.get(x, {}).get("latitude"))
    iksan_lon = iksan_lookup.map(lambda x: cache.get(x, {}).get("longitude"))
    iksan_method = iksan_lookup.map(
        lambda x: f"vworld_{cache.get(x, {}).get('address_type')}"
        if cache.get(x, {}).get("latitude") is not None else "not_found"
    )
    iksan_heads = numeric(iksan["사육두수"])
    iksan_area = numeric(iksan["시설면적(제곱미터)"])
    iksan_coeff = iksan["사육업종"].map(SPECIES_WEIGHT)
    if iksan_coeff.isna().any():
        unknown = sorted(iksan.loc[iksan_coeff.isna(), "사육업종"].dropna().astype(str).unique())
        raise RuntimeError(f"축종 계수가 없는 익산 행이 있습니다: {unknown}")
    impute_median = iksan.assign(_heads=iksan_heads).groupby("사육업종")["_heads"].median().to_dict()
    iksan_imputed = iksan_heads.isna()
    iksan_heads_filled = iksan_heads.fillna(iksan["사육업종"].map(impute_median))
    iksan_weight = iksan_heads_filled * iksan_coeff
    iksan_weight = iksan_weight.where(iksan["영업상태"].ne("휴업"), 0.0)
    iksan_out = pd.DataFrame({
        "source_id": [f"IKSAN-{i:04d}" for i in range(1, len(iksan) + 1)],
        "source_type": "livestock", "city": "익산시", "name": iksan["업체명"],
        "species": iksan["사육업종"], "head_count": iksan_heads,
        "area_m2": iksan_area, "status": iksan["영업상태"], "address": iksan["소재지"],
        "latitude": iksan_lat, "longitude": iksan_lon,
        "location_precision": np.where(iksan_lat.notna() & iksan_lon.notna(), "point", None),
        "geocode_method": iksan_method,
        "emission_weight": iksan_weight, "weight_imputed": iksan_imputed.astype(int),
    })

    existing_point = gimje["위도"].notna() & gimje["경도"].notna()
    village_lat = gimje_village.map(lambda x: cache.get(x, {}).get("latitude") if x else None)
    village_lon = gimje_village.map(lambda x: cache.get(x, {}).get("longitude") if x else None)
    gimje_lat = pd.to_numeric(gimje["위도"], errors="coerce").where(existing_point, village_lat)
    gimje_lon = pd.to_numeric(gimje["경도"], errors="coerce").where(existing_point, village_lon)
    gimje_coeff = gimje["사육업종"].map(SPECIES_WEIGHT)
    if gimje_coeff.isna().any():
        unknown = sorted(gimje.loc[gimje_coeff.isna(), "사육업종"].dropna().astype(str).unique())
        raise RuntimeError(f"축종 계수가 없는 김제 행이 있습니다: {unknown}")
    # 익산 원본에 메추리가 없어 사용자 승인에 따라 같은 가금류인 종계/산란계 중앙값을 쓴다.
    impute_median["메추리"] = impute_median.get("종계/산란계", np.nan)
    gimje_heads = gimje["사육업종"].map(impute_median)
    if gimje_heads.isna().any():
        unknown = sorted(gimje.loc[gimje_heads.isna(), "사육업종"].dropna().astype(str).unique())
        raise RuntimeError(f"익산 중앙값으로 대체할 수 없는 김제 축종이 있습니다: {unknown}")
    gimje_method = np.select(
        [existing_point, gimje_lat.notna() & gimje_lon.notna()],
        ["provided_point", "vworld_village"], default="not_found",
    )
    gimje_precision = np.select(
        [existing_point, gimje_lat.notna() & gimje_lon.notna()], ["point", "village"], default=None,
    )
    gimje_out = pd.DataFrame({
        "source_id": [f"GIMJE-{i:04d}" for i in range(1, len(gimje) + 1)],
        "source_type": "livestock", "city": "김제시", "name": gimje["업체명"],
        "species": gimje["사육업종"], "head_count": gimje_heads,
        "area_m2": np.nan, "status": np.nan, "address": gimje["소재지"],
        "latitude": gimje_lat, "longitude": gimje_lon,
        "location_precision": gimje_precision, "geocode_method": gimje_method,
        "emission_weight": gimje_heads * gimje_coeff, "weight_imputed": 1,
    })
    sources = pd.concat([iksan_out, gimje_out], ignore_index=True)[SOURCE_COLUMNS]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sources.to_csv(OUTPUT_DIR / "sources.csv", index=False, encoding="utf-8-sig")

    iksan_rate = float(iksan_out["location_precision"].eq("point").mean())
    gimje_village_rows = int((~existing_point).sum())
    gimje_village_matched = int((~existing_point & gimje_lat.notna() & gimje_lon.notna()).sum())
    metrics = {
        "iksan_point_rate": iksan_rate,
        "iksan_point": int(iksan_out["location_precision"].eq("point").sum()),
        "iksan_total": len(iksan_out),
        "gimje_point": int(existing_point.sum()),
        "gimje_village_rate": gimje_village_matched / max(gimje_village_rows, 1),
        "gimje_village": gimje_village_matched,
        "gimje_village_total": gimje_village_rows,
    }
    if iksan_rate < 0.85:
        failed = iksan_out.loc[iksan_out["latitude"].isna(), "address"].head(10).tolist()
        raise RuntimeError(
            f"익산 point 지오코딩 성공률이 85% 미만입니다: {iksan_rate:.1%}; 실패 예시={failed}"
        )
    return sources, metrics


def haversine_and_bearing(
    source_lat: np.ndarray, source_lon: np.ndarray, receptor_lat: float, receptor_lon: float,
) -> tuple[np.ndarray, np.ndarray]:
    """각 발생원에서 한 수용점까지의 거리와 초기 방위각을 반환한다."""
    p1 = np.radians(source_lat)
    p2 = math.radians(receptor_lat)
    dlat = p2 - p1
    dlon = np.radians(receptor_lon - source_lon)
    a = np.sin(dlat / 2) ** 2 + np.cos(p1) * math.cos(p2) * np.sin(dlon / 2) ** 2
    distance = 2 * odor.EARTH_RADIUS_KM * np.arcsin(np.minimum(1.0, np.sqrt(a)))
    y = np.sin(dlon) * math.cos(p2)
    x = np.cos(p1) * math.sin(p2) - np.sin(p1) * math.cos(p2) * np.cos(dlon)
    bearing = (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0
    return distance, bearing


def filter_initial_complaints(complaints: pd.DataFrame, event_hour: pd.Timestamp) -> pd.DataFrame:
    """현재 Event 시작 이상, 30분 미만인 민원만 반환한다."""
    cutoff = event_hour + pd.Timedelta(minutes=odor.INPUT_MINUTES)
    return complaints[(complaints["datetime"] >= event_hour) & (complaints["datetime"] < cutoff)].copy()


def circular_std_deg(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2:
        return 0.0
    radians = np.radians(arr)
    resultant = math.hypot(float(np.mean(np.sin(radians))), float(np.mean(np.cos(radians))))
    if resultant <= 1e-12:
        return 180.0
    return min(180.0, math.degrees(math.sqrt(max(0.0, -2.0 * math.log(resultant)))))


def event_weather_profile(asos: pd.DataFrame, event_hour: pd.Timestamp, event_lat: float, event_lon: float) -> dict:
    """Event 중심에서 가까운 ASOS 한 지점의 -3h~0h 시간자료를 반환한다."""
    stations = asos[["station_id", "station_latitude", "station_longitude"]].drop_duplicates("station_id")
    d = np.hypot(
        (stations["station_latitude"] - event_lat) * 110.540,
        (stations["station_longitude"] - event_lon) * 111.320 * math.cos(math.radians(event_lat)),
    )
    station_id = int(stations.loc[d.idxmin(), "station_id"])
    frame = asos[
        (asos["station_id"] == station_id)
        & (asos["datetime"] >= event_hour - pd.Timedelta(hours=3))
        & (asos["datetime"] <= event_hour)
    ].copy()
    by_offset: dict[int, dict[str, float]] = {}
    for offset in range(-3, 1):
        row = frame[frame["datetime"] == event_hour + pd.Timedelta(hours=offset)]
        if row.empty:
            continue
        item = row.iloc[0]
        by_offset[offset] = {
            "wind_direction": float(item["wind_direction"]),
            "wind_speed": float(item["wind_speed"]),
            "rainfall_hour": float(item["rainfall_hour"]) if pd.notna(item["rainfall_hour"]) else np.nan,
        }
    return {"station_id": station_id, "by_offset": by_offset}


def fixed0_weather(profile: dict) -> tuple[float, float]:
    """Event 정시와 직전 정시의 풍향을 풍속 가중 원형평균한다."""
    records = [profile["by_offset"].get(offset) for offset in (-1, 0)]
    valid = [
        item for item in records if item is not None
        and np.isfinite(item.get("wind_direction", np.nan))
        and np.isfinite(item.get("wind_speed", np.nan))
    ]
    if not valid:
        return np.nan, np.nan
    directions = np.radians([float(item["wind_direction"]) for item in valid])
    speeds = np.asarray([float(item["wind_speed"]) for item in valid], dtype=float)
    weights = np.maximum(speeds, 0.1)
    from_sin = float(np.average(np.sin(directions), weights=weights))
    from_cos = float(np.average(np.cos(directions), weights=weights))
    direction = float((np.degrees(np.arctan2(from_sin, from_cos)) + 360.0) % 360.0)
    return direction, float(np.mean(speeds))


def lagged_exposure(
    source_lat: np.ndarray, source_lon: np.ndarray, emission_weight: np.ndarray,
    receptor_lat: float, receptor_lon: float, event_hour: pd.Timestamp, profile: dict,
    direction_delta: float = 0.0, direction_override: dict[int, float] | None = None,
    wind_lag: str = DEFAULT_WIND_LAG, max_source_km: float = DEFAULT_MAX_SOURCE_KM,
) -> dict[str, np.ndarray]:
    """선택한 풍향 시차와 반경 안에서 발생원별 노출 적합도를 계산한다."""
    if wind_lag not in {"travel", "fixed0"}:
        raise ValueError(f"지원하지 않는 wind_lag: {wind_lag}")
    if max_source_km <= 0:
        raise ValueError("max_source_km는 0보다 커야 합니다.")
    distance, bearing = haversine_and_bearing(source_lat, source_lon, receptor_lat, receptor_lon)
    within = distance <= max_source_km
    current = profile["by_offset"].get(0)
    if current is None or not np.isfinite(current["wind_speed"]):
        nan = np.full(len(source_lat), np.nan)
        return {"distance": distance, "bearing": bearing, "travel_min": nan, "alignment": nan,
                "exposure": nan, "wind_from": nan, "wind_speed": nan,
                "decay_base": np.zeros(len(source_lat)), "offset": np.zeros(len(source_lat), dtype=int),
                "within": within}

    if wind_lag == "fixed0":
        fixed_direction, fixed_speed = fixed0_weather(profile)
        if direction_override:
            override_profile = {
                "by_offset": {
                    offset: {
                        **profile["by_offset"].get(offset, {}),
                        "wind_direction": direction_override.get(
                            offset, profile["by_offset"].get(offset, {}).get("wind_direction", np.nan),
                        ),
                    }
                    for offset in (-1, 0)
                }
            }
            fixed_direction, _ = fixed0_weather(override_profile)
        effective_scalar = max(fixed_speed, MIN_WIND_SPEED) if np.isfinite(fixed_speed) else np.nan
        effective_speed = np.full(len(source_lat), effective_scalar, dtype=float)
        wind_from = np.full(len(source_lat), fixed_direction, dtype=float)
        offsets = np.zeros(len(source_lat), dtype=int)
        travel = np.minimum(distance / effective_speed * 60.0 / 3.6, MAX_TRAVEL_HOURS * 60.0)
        used_speed = np.full(len(source_lat), fixed_speed, dtype=float)
    else:
        initial_speed = max(float(current["wind_speed"]), MIN_WIND_SPEED)
        travel = np.minimum(distance / initial_speed * 60.0 / 3.6, MAX_TRAVEL_HOURS * 60.0)
        offsets = np.clip(-np.ceil(travel / 60.0 - 1e-12).astype(int), -3, 0)
        lag_speed = np.array([
            profile["by_offset"].get(int(offset), current).get("wind_speed", np.nan) for offset in offsets
        ], dtype=float)
        effective_speed = np.maximum(np.where(np.isfinite(lag_speed), lag_speed, initial_speed), MIN_WIND_SPEED)
        travel = np.minimum(distance / effective_speed * 60.0 / 3.6, MAX_TRAVEL_HOURS * 60.0)
        offsets = np.clip(-np.ceil(travel / 60.0 - 1e-12).astype(int), -3, 0)
        wind_from = np.array([
            (direction_override or {}).get(
                int(offset), profile["by_offset"].get(int(offset), current).get("wind_direction", np.nan)
            )
            for offset in offsets
        ], dtype=float)
        used_speed = np.array([
            profile["by_offset"].get(int(offset), current).get("wind_speed", np.nan) for offset in offsets
        ], dtype=float)
        effective_speed = np.maximum(np.where(np.isfinite(used_speed), used_speed, initial_speed), MIN_WIND_SPEED)

    downwind = (wind_from + 180.0 + direction_delta) % 360.0
    signed_alignment = np.cos(np.radians(downwind - bearing))
    alignment = np.maximum(signed_alignment, 0.0)
    night = event_hour.hour >= 21 or event_hour.hour < 6
    decay_length = np.where(
        (effective_speed < 1.0) | night, DECAY_LENGTH_KM["slow_or_night"],
        np.where(effective_speed >= 2.0, DECAY_LENGTH_KM["fast"], DECAY_LENGTH_KM["moderate"]),
    )
    exposure = emission_weight * alignment * np.exp(-distance / decay_length)
    decay_base = emission_weight * np.exp(-distance / decay_length)
    invalid = ~np.isfinite(wind_from) | ~np.isfinite(effective_speed)
    exposure[invalid] = np.nan
    decay_base[invalid] = 0.0
    signed_alignment[invalid] = np.nan
    exposure[~within] = 0.0
    decay_base[~within] = 0.0
    return {
        "distance": distance, "bearing": bearing, "travel_min": travel,
        "alignment": signed_alignment, "exposure": exposure,
        "wind_from": wind_from, "wind_speed": used_speed,
        "decay_base": decay_base, "offset": offsets, "within": within,
    }


def emission_weighted_alignment(detail: dict[str, np.ndarray], emission_weight: np.ndarray) -> float:
    """반경 안 발생원의 배출 가중 평균 cosine 일치도를 계산한다."""
    valid = detail["within"] & np.isfinite(detail["alignment"]) & np.isfinite(emission_weight)
    denominator = float(emission_weight[valid].sum())
    if denominator <= 0:
        return np.nan
    return float(np.sum(emission_weight[valid] * detail["alignment"][valid]) / denominator)


def normalize_event_score(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).any():
        return np.full_like(values, np.nan)
    maximum = float(np.nanmax(values))
    if maximum <= 0:
        return np.zeros_like(values)
    return values / maximum


def wind_name(direction: float) -> str:
    names = ["북풍", "북동풍", "동풍", "남동풍", "남풍", "남서풍", "서풍", "북서풍"]
    return names[int(((direction + 22.5) % 360) // 45)]


def source_group_key(source: pd.Series) -> str:
    """point는 시설별, village는 읍면·리별 후보로 묶는다."""
    if source["location_precision"] == "village":
        return gimje_village_address(source["address"]) or str(source["source_id"])
    return str(source["source_id"])


def rank_source_groups(explanation: np.ndarray, sources: pd.DataFrame) -> list[dict[str, object]]:
    groups: dict[str, dict[str, object]] = {}
    for index, score in enumerate(explanation):
        source = sources.iloc[index]
        key = source_group_key(source)
        group = groups.setdefault(key, {"key": key, "score": 0.0, "indices": []})
        group["score"] = float(group["score"]) + float(score)
        group["indices"].append(index)
    ranked = sorted(groups.values(), key=lambda item: float(item["score"]), reverse=True)
    for group in ranked:
        indices = np.asarray(group["indices"], dtype=int)
        group["representative"] = int(indices[np.argmax(explanation[indices])])
        group["count"] = int(len(indices))
    return ranked


def build_outputs(
    sources: pd.DataFrame, wind_lag: str = DEFAULT_WIND_LAG,
    max_source_km: float = DEFAULT_MAX_SOURCE_KM,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    reference = pd.read_csv(REFERENCE_FILE, encoding="utf-8-sig", parse_dates=["event_hour"])
    meta = json.loads(REFERENCE_META.read_text(encoding="utf-8"))
    asos = pd.read_csv(ASOS_FILE, encoding="utf-8-sig", parse_dates=["datetime"])
    complaints, _, _, _, _ = odor.load_inputs()

    valid_sources = sources.dropna(subset=["latitude", "longitude", "emission_weight"]).copy().reset_index(drop=True)
    source_lat = valid_sources["latitude"].to_numpy(float)
    source_lon = valid_sources["longitude"].to_numpy(float)
    emission = valid_sources["emission_weight"].to_numpy(float)
    village = valid_sources["location_precision"].eq("village").to_numpy(float)

    score_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    validation_events: list[dict[str, object]] = []
    for event_index, (event_hour, candidates) in enumerate(reference.groupby("event_hour", sort=True), 1):
        event_hour = pd.Timestamp(event_hour)
        initial = filter_initial_complaints(complaints, event_hour)
        if initial.empty:
            raise RuntimeError(f"기준 Event의 초기 민원을 찾지 못했습니다: {event_hour}")
        event_lat = float(initial["latitude"].mean())
        event_lon = float(initial["longitude"].mean())
        profile = event_weather_profile(asos, event_hour, event_lat, event_lon)
        has_weather = 0 in profile["by_offset"]
        rain_1h = profile["by_offset"].get(0, {}).get("rainfall_hour", np.nan)
        rain_values = [profile["by_offset"].get(offset, {}).get("rainfall_hour", np.nan) for offset in (-2, -1, 0)]
        rain_3h = float(np.nansum(rain_values)) if np.isfinite(rain_values).any() else np.nan
        directions = [item["wind_direction"] for item in profile["by_offset"].values()]
        wind_sd = circular_std_deg(directions)
        current_speed = profile["by_offset"].get(0, {}).get("wind_speed", np.nan)
        stagnation = int(current_speed < 1.0) if np.isfinite(current_speed) else np.nan

        if not has_weather:
            for row in candidates.itertuples(index=False):
                score_rows.append({
                    "event_hour": event_hour, "event_id": row.event_id, "grid_x": int(row.grid_x),
                    "grid_y": int(row.grid_y), "center_latitude": row.center_latitude,
                    "center_longitude": row.center_longitude, "source_fit_score": np.nan,
                    "forward_plume_score": np.nan, "prior_downwind_score": np.nan,
                    "lagged_wind_alignment": np.nan, "travel_time_min": np.nan,
                    "rain_1h": np.nan, "rain_3h": np.nan, "stagnation_flag": np.nan,
                    "backtrack_uncertainty": np.nan, "weather_source": "none",
                    "history_cutoff": event_hour,
                })
            continue

        observed = initial.groupby(["latitude", "longitude"]).size().reset_index(name="count")
        explanation = np.zeros(len(valid_sources), dtype=float)
        best_observed = np.zeros(len(valid_sources), dtype=float)
        best_observed_details: dict[str, np.ndarray] | None = None
        observed_details: list[tuple[float, float, int]] = []
        for receptor in observed.itertuples(index=False):
            detail = lagged_exposure(
                source_lat, source_lon, emission, float(receptor.latitude), float(receptor.longitude),
                event_hour, profile, wind_lag=wind_lag, max_source_km=max_source_km,
            )
            contribution = np.nan_to_num(detail["exposure"], nan=0.0) * int(receptor.count)
            explanation += contribution
            improve = contribution > best_observed
            if best_observed_details is None:
                best_observed_details = {key: value.copy() for key, value in detail.items()}
            else:
                for key in best_observed_details:
                    best_observed_details[key][improve] = detail[key][improve]
            best_observed[improve] = contribution[improve]
            observed_details.append((float(receptor.latitude), float(receptor.longitude), int(receptor.count)))
        explanation_weight = explanation / explanation.sum() if explanation.sum() > 0 else explanation

        forward_raw: list[float] = []
        fit_raw: list[float] = []
        grid_details: list[dict[str, float]] = []
        candidate_score_distributions: list[float] = []
        validation_base: list[np.ndarray] = []
        validation_bearing: list[np.ndarray] = []
        validation_offset: list[np.ndarray] = []
        for row in candidates.itertuples(index=False):
            detail = lagged_exposure(
                source_lat, source_lon, emission, float(row.center_latitude), float(row.center_longitude),
                event_hour, profile, wind_lag=wind_lag, max_source_km=max_source_km,
            )
            exposure = np.nan_to_num(detail["exposure"], nan=0.0)
            forward = float(exposure.sum())
            reweighted = exposure * explanation_weight
            fit = float(reweighted.sum())
            top_basis = reweighted if reweighted.max(initial=0.0) > 0 else exposure
            top_index = int(np.argmax(top_basis))
            eligible = np.where(detail["within"])[0]
            top_n = min(5, len(eligible))
            top5 = eligible[np.argsort(exposure[eligible])[-top_n:]] if top_n else np.array([], dtype=int)
            village_share = float(village[top5].mean()) if len(top5) else 0.0
            uncertainty = float(np.mean([min(wind_sd / 180.0, 1.0), float(stagnation), village_share]))
            forward_raw.append(forward)
            fit_raw.append(fit)
            candidate_score_distributions.append(forward)
            validation_base.append(detail["decay_base"].astype(np.float32))
            validation_bearing.append(detail["bearing"].astype(np.float32))
            validation_offset.append(detail["offset"].astype(np.int8))
            grid_details.append({
                "alignment": emission_weighted_alignment(detail, emission),
                "travel": float(detail["travel_min"][top_index]), "uncertainty": uncertainty,
            })
        forward_norm = normalize_event_score(np.asarray(forward_raw))
        fit_norm = normalize_event_score(np.asarray(fit_raw))
        for position, row in enumerate(candidates.itertuples(index=False)):
            score_rows.append({
                "event_hour": event_hour, "event_id": row.event_id, "grid_x": int(row.grid_x),
                "grid_y": int(row.grid_y), "center_latitude": row.center_latitude,
                "center_longitude": row.center_longitude, "source_fit_score": fit_norm[position],
                "forward_plume_score": forward_norm[position], "prior_downwind_score": np.nan,
                "lagged_wind_alignment": grid_details[position]["alignment"],
                "travel_time_min": grid_details[position]["travel"], "rain_1h": rain_1h,
                "rain_3h": rain_3h, "stagnation_flag": stagnation,
                "backtrack_uncertainty": grid_details[position]["uncertainty"],
                "weather_source": "asos", "history_cutoff": event_hour,
            })

        assert best_observed_details is not None
        ranked_groups = rank_source_groups(explanation, valid_sources)
        top_groups = ranked_groups[:TOP_SOURCE_COUNT]
        top_max = float(top_groups[0]["score"]) if top_groups else 0.0
        for rank, group in enumerate(top_groups, 1):
            source_index = int(group["representative"])
            source = valid_sources.iloc[source_index]
            wind_from = float(best_observed_details["wind_from"][source_index])
            distance = float(best_observed_details["distance"][source_index])
            travel = float(best_observed_details["travel_min"][source_index])
            alignment = float(best_observed_details["alignment"][source_index])
            precision = str(source["location_precision"])
            if precision == "village":
                village_address = gimje_village_address(source["address"]) or "김제시 읍면·리"
                short = village_address.replace("전북특별자치도 ", "")
                name = f"{short} 축산 밀집 구역({group['count']}곳)"
            else:
                name = source["name"]
            candidate_rows.append({
                "event_hour": event_hour, "rank": rank, "source_id": source["source_id"], "name": name,
                "city": source["city"], "species": source["species"], "location_precision": precision,
                "distance_km": distance, "bearing_deg": float(best_observed_details["bearing"][source_index]),
                "travel_time_min": travel, "wind_alignment": alignment,
                "fit_score": float(group["score"] / top_max) if top_max > 0 else 0.0,
                "evidence_text": (
                    f"{wind_name(wind_from)} {best_observed_details['wind_speed'][source_index]:.1f} m/s "
                    + (
                        f"Event 정시·직전 정시 평균 기준, {distance:.1f} km, 예상 도달 {travel:.0f}분, "
                        if wind_lag == "fixed0" else
                        f"기준, {distance:.1f} km, 약 {travel:.0f}분 전 풍향, "
                    )
                    + f"방위 일치 {alignment:.2f}"
                ),
            })
        complaint_gx, complaint_gy = latlon_to_grid(
            initial["latitude"].to_numpy(float), initial["longitude"].to_numpy(float), meta,
        )
        observed_grid_keys = set(zip(complaint_gx.tolist(), complaint_gy.tolist()))
        candidate_keys = list(zip(candidates["grid_x"].astype(int), candidates["grid_y"].astype(int)))
        observed_positions = np.array(
            [index for index, key in enumerate(candidate_keys) if key in observed_grid_keys], dtype=int,
        )
        if not len(observed_positions):
            raise RuntimeError(f"초기 민원 격자가 후보 기준표에 없습니다: {event_hour}")
        candidate_distribution = np.asarray(candidate_score_distributions, dtype=float)
        actual_percentiles = [
            float(np.mean(candidate_distribution <= candidate_distribution[position]))
            for position in observed_positions
        ]
        validation_events.append({
            "event_hour": event_hour, "rain_3h": rain_3h, "profile": profile,
            "observed": observed_details, "actual_percentile": float(np.mean(actual_percentiles)),
            "observed_positions": observed_positions,
            "candidate_base": np.stack(validation_base),
            "candidate_bearing": np.stack(validation_bearing),
            "candidate_offset": np.stack(validation_offset),
            "baseline_top3": [str(group["key"]) for group in top_groups[:3]],
            "wind_lag": wind_lag,
        })
        if event_index % 20 == 0 or event_index == reference["event_hour"].nunique():
            print(
                f"  {wind_lag}/{max_source_km:g}km {event_index}/{reference['event_hour'].nunique()} Event",
                flush=True,
            )

    scores = pd.DataFrame(score_rows, columns=GRID_COLUMNS)
    source_candidates = pd.DataFrame(candidate_rows, columns=CANDIDATE_COLUMNS)
    return scores, source_candidates, {
        "events": validation_events, "valid_sources": valid_sources,
        "source_lat": source_lat, "source_lon": source_lon, "emission": emission,
        "complaints": complaints, "asos": asos, "meta": meta,
        "wind_lag": wind_lag, "max_source_km": max_source_km,
    }


def direction_sensitivity(context: dict, degrees: Iterable[int] = (10, 20, 30)) -> tuple[dict, dict]:
    valid = context["valid_sources"]
    source_lat, source_lon, emission = context["source_lat"], context["source_lon"], context["emission"]
    by_condition: dict[str, dict[str, float]] = {"all": {}, "dry": {}, "wet": {}}
    for degree in degrees:
        values: dict[str, list[float]] = {"all": [], "dry": [], "wet": []}
        for event in context["events"]:
            base = set(event["baseline_top3"])
            for delta in (-degree, degree):
                explanation = np.zeros(len(valid), dtype=float)
                for lat, lon, count in event["observed"]:
                    detail = lagged_exposure(
                        source_lat, source_lon, emission, lat, lon, event["event_hour"], event["profile"],
                        direction_delta=float(delta),
                        wind_lag=context["wind_lag"], max_source_km=context["max_source_km"],
                    )
                    explanation += np.nan_to_num(detail["exposure"], nan=0.0) * count
                perturbed = set(str(group["key"]) for group in rank_source_groups(explanation, valid)[:3])
                retention = len(base & perturbed) / 3.0
                values["all"].append(retention)
                condition = "wet" if event["rain_3h"] > 0 else "dry"
                values[condition].append(retention)
        for condition in values:
            by_condition[condition][str(degree)] = float(np.mean(values[condition])) if values[condition] else np.nan
    return by_condition["all"], by_condition


def random_wind_contrast(context: dict, repetitions: int = 100) -> dict[str, dict[str, float]]:
    """Event 간 풍향 profile을 섞고 매회 후보 격자 전체 순위를 다시 계산한다."""
    events = context["events"]
    rng = np.random.default_rng(RANDOM_SEED)
    conditions = {
        "all": np.arange(len(events)),
        "dry": np.array([i for i, event in enumerate(events) if event["rain_3h"] == 0], dtype=int),
        "wet": np.array([i for i, event in enumerate(events) if event["rain_3h"] > 0], dtype=int),
    }
    real = {name: float(np.mean([events[i]["actual_percentile"] for i in idx])) if len(idx) else np.nan
            for name, idx in conditions.items()}
    shuffled: dict[str, list[float]] = {name: [] for name in conditions}
    profiles = [event["profile"] for event in events]
    for _ in range(repetitions):
        order = rng.permutation(len(events))
        per_event: list[float] = []
        for i, event in enumerate(events):
            donor = profiles[int(order[i])]
            if event["wind_lag"] == "fixed0":
                donor_direction, _ = fixed0_weather(donor)
                if not np.isfinite(donor_direction):
                    donor_direction, _ = fixed0_weather(event["profile"])
                downwind = (donor_direction + 180.0) % 360.0
            else:
                recipient = event["profile"]
                directions = []
                for offset in range(-3, 1):
                    donor_item = donor["by_offset"].get(offset)
                    recipient_item = recipient["by_offset"].get(offset, recipient["by_offset"].get(0, {}))
                    value = donor_item.get("wind_direction", np.nan) if donor_item else np.nan
                    if not np.isfinite(value):
                        value = recipient_item.get("wind_direction", np.nan)
                    directions.append(float(value))
                direction_array = np.asarray(directions, dtype=float)
                offsets = event["candidate_offset"].astype(int) + 3
                downwind = (direction_array[offsets] + 180.0) % 360.0
            alignment = np.maximum(
                np.cos(np.radians(downwind - event["candidate_bearing"])), 0.0,
            )
            distribution = np.sum(event["candidate_base"] * alignment, axis=1)
            percentiles = [
                float(np.mean(distribution <= distribution[position]))
                for position in event["observed_positions"]
            ]
            per_event.append(float(np.mean(percentiles)))
        for name, idx in conditions.items():
            shuffled[name].append(float(np.mean(np.asarray(per_event)[idx])) if len(idx) else np.nan)
    result = {}
    for name in conditions:
        values = np.asarray(shuffled[name], dtype=float)
        p_value = (1 + int(np.sum(values >= real[name]))) / (1 + int(np.isfinite(values).sum())) if np.isfinite(real[name]) else np.nan
        result[name] = {
            "real_percentile": real[name], "shuffled_mean": float(np.nanmean(values)),
            "mean_difference": real[name] - float(np.nanmean(values)), "p_value": p_value,
        }
    return result


def destination_point(lat: float, lon: float, bearing_deg: float, distance_km: float) -> tuple[float, float]:
    angular = distance_km / odor.EARTH_RADIUS_KM
    bearing = math.radians(bearing_deg)
    p1, l1 = math.radians(lat), math.radians(lon)
    p2 = math.asin(math.sin(p1) * math.cos(angular) + math.cos(p1) * math.sin(angular) * math.cos(bearing))
    l2 = l1 + math.atan2(math.sin(bearing) * math.sin(angular) * math.cos(p1),
                         math.cos(angular) - math.sin(p1) * math.sin(p2))
    return math.degrees(p2), math.degrees(l2)


def latlon_to_grid(lat: np.ndarray, lon: np.ndarray, meta: dict) -> tuple[np.ndarray, np.ndarray]:
    gx = np.floor((lon - meta["lon0"]) * 111_320 * math.cos(math.radians(meta["lat0"])) / meta["grid_m"]).astype(int)
    gy = np.floor((lat - meta["lat0"]) * 110_540 / meta["grid_m"]).astype(int)
    return gx, gy


def facility_lift(context: dict) -> tuple[float, dict[str, float]]:
    """ASOS 기간 전체 민원의 5 km 역방향 선분과 시설 격자 겹침 lift를 계산한다."""
    meta, sources = context["meta"], context["valid_sources"]
    complaints, asos = context["complaints"], context["asos"]
    source_gx, source_gy = latlon_to_grid(
        sources["latitude"].to_numpy(float), sources["longitude"].to_numpy(float), meta,
    )
    source_weights: dict[tuple[int, int], float] = {}
    for gx, gy, weight in zip(source_gx, source_gy, sources["emission_weight"].to_numpy(float)):
        source_weights[(int(gx), int(gy))] = source_weights.get((int(gx), int(gy)), 0.0) + float(weight)

    station_rows = asos[["station_id", "station_latitude", "station_longitude"]].drop_duplicates("station_id")
    weather = asos.set_index(["station_id", "datetime"])[["wind_direction", "rainfall_hour"]]
    counts: dict[str, dict[tuple[int, int], int]] = {"all": {}, "dry": {}, "wet": {}}
    for complaint in complaints.itertuples(index=False):
        hour = pd.Timestamp(complaint.datetime).floor("h")
        distances = np.hypot(
            (station_rows["station_latitude"] - float(complaint.latitude)) * 110.540,
            (station_rows["station_longitude"] - float(complaint.longitude))
            * 111.320 * math.cos(math.radians(float(complaint.latitude))),
        )
        station_id = int(station_rows.loc[distances.idxmin(), "station_id"])
        key = (station_id, hour)
        if key not in weather.index:
            continue
        current = weather.loc[key]
        current_direction = float(current["wind_direction"])
        if not np.isfinite(current_direction):
            continue
        rain_values = []
        for offset in (-2, -1, 0):
            rain_key = (station_id, hour + pd.Timedelta(hours=offset))
            rain_values.append(
                float(weather.loc[rain_key, "rainfall_hour"]) if rain_key in weather.index else np.nan
            )
        rain_3h = float(np.nansum(rain_values)) if np.isfinite(rain_values).any() else np.nan
        condition = "wet" if rain_3h > 0 else "dry"
        line_cells: set[tuple[int, int]] = set()
        for distance in np.linspace(0.0, 5.0, 21):
            point_lat, point_lon = destination_point(
                float(complaint.latitude), float(complaint.longitude), current_direction, float(distance),
            )
            gx, gy = latlon_to_grid(np.array([point_lat]), np.array([point_lon]), meta)
            line_cells.add((int(gx[0]), int(gy[0])))
        for cell in line_cells:
            counts["all"][cell] = counts["all"].get(cell, 0) + 1
            counts[condition][cell] = counts[condition].get(cell, 0) + 1

    lifts: dict[str, float] = {}
    for name, cell_counts in counts.items():
        if not cell_counts:
            lifts[name] = np.nan
            continue
        ordered = sorted(cell_counts, key=cell_counts.get, reverse=True)
        top_n = max(1, math.ceil(len(ordered) * 0.1))
        observed_weight = sum(source_weights.get(cell, 0.0) for cell in ordered[:top_n])
        total_weight = sum(source_weights.get(cell, 0.0) for cell in ordered)
        expected = total_weight * top_n / len(ordered)
        lifts[name] = observed_weight / expected if expected > 0 else np.nan
    return lifts["all"], lifts


def validate_and_write(
    scores: pd.DataFrame, candidates: pd.DataFrame, context: dict,
    comparison_contexts: dict[str, dict], geocoding: dict[str, object], repetitions: int,
) -> dict[str, object]:
    comparison: dict[str, dict[str, dict[str, float]]] = {}
    for name, comparison_context in comparison_contexts.items():
        print(f"[검증] 무작위 풍향 대조: {name}", flush=True)
        comparison[name] = random_wind_contrast(comparison_context, repetitions=repetitions)
    selected_name = next(
        (
            name for name, config in COMPARISON_CONFIGS.items()
            if config["wind_lag"] == context["wind_lag"]
            and config["max_source_km"] == context["max_source_km"]
        ),
        None,
    )
    if selected_name in comparison:
        random_contrast = comparison[selected_name]
    else:
        print("[검증] 무작위 풍향 대조: 선택 설정", flush=True)
        random_contrast = random_wind_contrast(context, repetitions=repetitions)
    print("[검증] 풍향 민감도", flush=True)
    sensitivity_all, sensitivity_by_condition = direction_sensitivity(context)
    print("[검증] 시설 겹침 lift", flush=True)
    lift_all, lift_by_condition = facility_lift(context)
    validation = {
        "random_wind_contrast": random_contrast,
        "random_wind_contrast_comparison": comparison,
        "direction_sensitivity": sensitivity_all,
        "direction_sensitivity_by_condition": sensitivity_by_condition,
        "facility_lift_top10pct": lift_all,
        "facility_lift_top10pct_by_condition": lift_by_condition,
        "geocoding": geocoding,
        "constants": {
            "SPECIES_WEIGHT": SPECIES_WEIGHT, "L_km": DECAY_LENGTH_KM,
            "min_wind_speed": MIN_WIND_SPEED, "max_travel_hours": MAX_TRAVEL_HOURS,
            "wind_lag": context["wind_lag"], "max_source_km": context["max_source_km"],
        },
        "method_notes": {
            "wind_direction": "KMA wind-from direction converted to downwind by adding 180 degrees",
            "fixed0": "wind-speed-weighted circular mean of event_hour and event_hour-1h wind, shared by every source-grid pair",
            "lagged_wind_alignment": "emission_weight-weighted mean cosine alignment among sources within max_source_km",
            "stagnation": "ASOS hourly phase: wind speed <1 m/s only; 30-minute direction SD unavailable",
            "random_contrast": f"candidate-grid ranks recomputed for each of {repetitions} event-profile wind shuffles",
            "facility_lift_top10pct": (
                "numerator=sum(emission_weight in the top 10% of traversed cells ranked by 5km upwind-line count); "
                "denominator=sum(emission_weight in all traversed cells) * number_of_top_cells / number_of_traversed_cells"
            ),
        },
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scores.to_csv(OUTPUT_DIR / "grid_scores.csv", index=False, encoding="utf-8-sig")
    candidates.to_csv(OUTPUT_DIR / "source_candidates.csv", index=False, encoding="utf-8-sig")
    (OUTPUT_DIR / "validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8",
    )
    return validation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--sources-only", action="store_true")
    parser.add_argument("--skip-geocoding", action="store_true", help="기존 로컬 sources.csv 재사용")
    parser.add_argument("--shuffle-repetitions", type=int, default=100)
    parser.add_argument("--wind-lag", choices=("travel", "fixed0"), default=DEFAULT_WIND_LAG)
    parser.add_argument("--max-source-km", type=float, default=DEFAULT_MAX_SOURCE_KM)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.skip_geocoding:
        sources = pd.read_csv(OUTPUT_DIR / "sources.csv", encoding="utf-8-sig")
        iksan = sources[sources["city"] == "익산시"]
        gimje = sources[sources["city"] == "김제시"]
        geocoding = {
            "iksan_point_rate": float(iksan["location_precision"].eq("point").mean()),
            "iksan_point": int(iksan["location_precision"].eq("point").sum()), "iksan_total": len(iksan),
            "gimje_point": int(gimje["location_precision"].eq("point").sum()),
            "gimje_village_rate": float(gimje["location_precision"].eq("village").sum() / max(gimje["location_precision"].ne("point").sum(), 1)),
            "gimje_village": int(gimje["location_precision"].eq("village").sum()),
            "gimje_village_total": int(gimje["location_precision"].ne("point").sum()),
        }
    else:
        print("[A-1] 발생원 주소 정규화 및 VWorld 지오코딩", flush=True)
        sources, geocoding = prepare_sources(workers=args.workers)
    print(json.dumps(geocoding, ensure_ascii=False, indent=2), flush=True)
    if args.sources_only:
        return

    if args.max_source_km <= 0:
        parser.error("--max-source-km는 0보다 커야 합니다.")
    print(
        f"[A-2/A-3] ASOS 역추적 점수: wind_lag={args.wind_lag}, max_source_km={args.max_source_km:g}",
        flush=True,
    )
    scores, candidates, context = build_outputs(
        sources, wind_lag=args.wind_lag, max_source_km=args.max_source_km,
    )
    comparison_contexts: dict[str, dict] = {}
    for name, config in COMPARISON_CONFIGS.items():
        if config["wind_lag"] == args.wind_lag and config["max_source_km"] == args.max_source_km:
            comparison_contexts[name] = context
            continue
        print(
            f"[비교 설정] wind_lag={config['wind_lag']}, max_source_km={config['max_source_km']:g}",
            flush=True,
        )
        _, _, comparison_contexts[name] = build_outputs(sources, **config)
    validation = validate_and_write(
        scores, candidates, context, comparison_contexts, geocoding, args.shuffle_repetitions,
    )
    print(json.dumps({
        "grid_rows": len(scores), "events": int(scores["event_hour"].nunique()),
        "candidate_rows": len(candidates), "validation": validation,
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
