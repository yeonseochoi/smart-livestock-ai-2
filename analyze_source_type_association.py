"""민원 유형과 재수집 발생원의 상풍측 연관을 층화 셔플로 검정한다.

시설을 원인으로 판정하지 않고, 민원 시각 풍향과 후보 발생원 위치의 연관만 계산한다.
산업단지는 중심점이 아니라 민원 좌표에서 폴리곤 경계까지의 최단점으로 거리·방위를 계산한다.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import build_source_backtrack as backtrack


ROOT = Path(__file__).resolve().parent
SOURCES_PATH = ROOT / "outputs" / "source_backtrack" / "sources.csv"
ZONE_PATH = ROOT / "data" / "public_source_cache" / "industrial_zones.geojson"
OUTPUT_PATH = ROOT / "outputs" / "source_backtrack" / "source_type_association.json"
AIR_FILES = {
    "익산시": ROOT / "data" / "public_source_cache" / "air_emission_iksan.csv",
    "김제시": ROOT / "data" / "public_source_cache" / "air_emission_gimje.csv",
}

RADII_KM = (6.0, 10.0)
SECTOR_DEG = 30.0
BIN_DEG = 5.0
N_BINS = int(360 / BIN_DEG)
PERMUTATIONS = 300
MIN_WIND = 1.0
SEED = 42


def text(value: object) -> str:
    return "" if pd.isna(value) else str(value).strip()


def complete_address(city: str, value: object) -> str:
    address = text(value)
    if not address:
        return ""
    if city not in address:
        address = f"전북특별자치도 {city} {address}"
    return backtrack.normalize_address(address)


def old_air_type(name: str, industry: str, product: str) -> str:
    combined = " ".join((name, industry, product))
    if any(key in combined for key in ("하수처리", "폐수처리", "분뇨처리", "오수처리")):
        return "wastewater"
    if any(key in combined for key in (
        "폐기물 처리", "폐기물처리", "폐기물 재활용", "폐기물재활용",
        "소각", "매립", "퇴비", "액비", "자원화",
    )):
        return "other"
    return "factory"


def load_prefilter_factories() -> tuple[pd.DataFrame, int]:
    """3판의 대기배출시설 분류를 재현하고 기존 VWorld 캐시 좌표를 붙인다."""
    cache = json.loads(backtrack.CACHE_FILE.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    for city, path in AIR_FILES.items():
        frame = pd.read_csv(path, encoding="cp949")
        frame = frame[frame["영업상태명"].eq("영업/정상")]
        for item in frame.to_dict("records"):
            address = complete_address(
                city,
                item.get("도로명주소") if text(item.get("도로명주소")) else item.get("지번주소"),
            )
            name = text(item.get("사업장명"))
            industry = text(item.get("업태구분명"))
            product = text(item.get("주생산품명"))
            if old_air_type(name, industry, product) != "factory" or not address:
                continue
            hit = cache.get(address, {})
            rows.append({
                "source_id": text(item.get("관리번호")), "city": city, "name": name,
                "address": address, "latitude": hit.get("latitude"), "longitude": hit.get("longitude"),
            })
    all_rows = pd.DataFrame(rows).drop_duplicates(["city", "name", "address"], keep="last")
    points = all_rows.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
    return points, int(len(all_rows))


def load_complaints_and_wind() -> pd.DataFrame:
    complaints, _, _, _, _ = backtrack.odor.load_inputs()
    complaints = complaints.dropna(subset=["latitude", "longitude"]).copy()
    complaints = complaints[complaints["datetime"] >= "2020-01-01"]
    complaints["hour"] = complaints["datetime"].dt.floor("h")

    asos = pd.read_csv(backtrack.ASOS_FILE, encoding="utf-8-sig", parse_dates=["datetime"])
    center_lat = float(complaints["latitude"].median())
    center_lon = float(complaints["longitude"].median())
    stations = asos.groupby("station_id")[["station_latitude", "station_longitude"]].first()
    station_weight = {
        station_id: 1.0 / max(
            float(backtrack.haversine_and_bearing(
                np.array([float(row.station_latitude)]),
                np.array([float(row.station_longitude)]),
                center_lat,
                center_lon,
            )[0][0]),
            1.0,
        ) ** 2
        for station_id, row in stations.iterrows()
    }
    theta = np.radians(asos["wind_direction"].to_numpy(float))
    speed = asos["wind_speed"].to_numpy(float)
    weighted = asos[["datetime", "station_id"]].copy()
    weighted["weight"] = weighted["station_id"].map(station_weight)
    weighted["u_weighted"] = (-speed * np.sin(theta)) * weighted["weight"]
    weighted["v_weighted"] = (-speed * np.cos(theta)) * weighted["weight"]
    weighted["speed_weighted"] = speed * weighted["weight"]
    grouped = weighted.groupby("datetime")[["weight", "u_weighted", "v_weighted", "speed_weighted"]].sum()
    u = grouped["u_weighted"] / grouped["weight"]
    v = grouped["v_weighted"] / grouped["weight"]
    grouped["speed"] = grouped["speed_weighted"] / grouped["weight"]
    grouped["from_deg"] = (np.degrees(np.arctan2(-u, -v)) + 360.0) % 360.0

    joined = complaints.join(grouped[["speed", "from_deg"]], on="hour", how="inner")
    joined = joined[joined["speed"] >= MIN_WIND].reset_index(drop=True)
    joined["complaint_group"] = np.select(
        [
            joined["odor_type"].astype(str).str.startswith("공장"),
            joined["odor_type"].astype(str).str.startswith("하수"),
            joined["odor_type"].astype(str).str.startswith("가축"),
        ],
        ["factory", "sewer", "livestock"],
        default="other",
    )
    return joined


def bearing_bins_points(complaints: pd.DataFrame, sources: pd.DataFrame, radius_km: float) -> np.ndarray:
    lat_c = np.radians(complaints["latitude"].to_numpy(float))
    lon_c = np.radians(complaints["longitude"].to_numpy(float))
    lat_s = np.radians(sources["latitude"].to_numpy(float))
    lon_s = np.radians(sources["longitude"].to_numpy(float))
    bins = np.zeros((len(complaints), N_BINS), dtype=float)
    for start in range(0, len(complaints), 500):
        stop = min(start + 500, len(complaints))
        dlat = lat_s[None, :] - lat_c[start:stop, None]
        dlon = lon_s[None, :] - lon_c[start:stop, None]
        a = np.sin(dlat / 2) ** 2 + np.cos(lat_c[start:stop, None]) * np.cos(lat_s[None, :]) * np.sin(dlon / 2) ** 2
        distance = 2 * backtrack.odor.EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
        x = np.sin(dlon) * np.cos(lat_s[None, :])
        y = (
            np.cos(lat_c[start:stop, None]) * np.sin(lat_s[None, :])
            - np.sin(lat_c[start:stop, None]) * np.cos(lat_s[None, :]) * np.cos(dlon)
        )
        bearing = (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0
        bearing_bin = (bearing // BIN_DEG).astype(int) % N_BINS
        weight = np.where((distance <= radius_km) & (distance > 0.05), 1.0, 0.0)
        local_rows = np.repeat(np.arange(stop - start), bearing_bin.shape[1])
        np.add.at(bins[start:stop], (local_rows, bearing_bin.ravel()), weight.ravel())
    return bins


def nearest_boundary(lat: float, lon: float, ring: list[list[float]]) -> tuple[float, float]:
    """민원 좌표 기준 국소 평면에서 폴리곤 경계 최단거리와 방위를 반환한다."""
    cos_lat = math.cos(math.radians(lat))
    best_distance = math.inf
    best_x = 0.0
    best_y = 0.0
    for left, right in zip(ring, ring[1:]):
        ax = (left[0] - lon) * 111.32 * cos_lat
        ay = (left[1] - lat) * 110.574
        bx = (right[0] - lon) * 111.32 * cos_lat
        by = (right[1] - lat) * 110.574
        dx = bx - ax
        dy = by - ay
        denom = dx * dx + dy * dy
        fraction = 0.0 if denom == 0 else max(0.0, min(1.0, -(ax * dx + ay * dy) / denom))
        x = ax + fraction * dx
        y = ay + fraction * dy
        distance = math.hypot(x, y)
        if distance < best_distance:
            best_distance, best_x, best_y = distance, x, y
    bearing = (math.degrees(math.atan2(best_x, best_y)) + 360.0) % 360.0
    return best_distance, bearing


def bearing_bins_zones(complaints: pd.DataFrame, radius_km: float) -> tuple[np.ndarray, int]:
    geojson = json.loads(ZONE_PATH.read_text(encoding="utf-8"))
    rings = [feature["geometry"]["coordinates"][0] for feature in geojson["features"]]
    bins = np.zeros((len(complaints), N_BINS), dtype=float)
    for row_index, complaint in enumerate(complaints.itertuples(index=False)):
        for ring in rings:
            distance, bearing = nearest_boundary(float(complaint.latitude), float(complaint.longitude), ring)
            if 0.05 < distance <= radius_km:
                bins[row_index, int(bearing // BIN_DEG) % N_BINS] += 1.0
    return bins, len(rings)


def upwind_weight(bins: np.ndarray, from_deg: np.ndarray) -> np.ndarray:
    half = int(round(SECTOR_DEG / BIN_DEG))
    center = (from_deg // BIN_DEG).astype(int) % N_BINS
    offsets = np.arange(-half, half + 1)
    index = (center[:, None] + offsets[None, :]) % N_BINS
    return np.take_along_axis(bins, index, axis=1).sum(axis=1)


def stratified_shuffle(from_deg: np.ndarray, strata: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    shuffled = from_deg.copy()
    for key in np.unique(strata):
        index = np.where(strata == key)[0]
        shuffled[index] = from_deg[rng.permutation(index)]
    return shuffled


def test_association(complaints: pd.DataFrame, bins: np.ndarray, seed_label: str) -> dict[str, float | int]:
    from_deg = complaints["from_deg"].to_numpy(float)
    strata = (
        complaints["hour"].dt.month.astype(str)
        + "-"
        + (complaints["hour"].dt.hour // 6).astype(str)
    ).to_numpy()
    real_mean = float(upwind_weight(bins, from_deg).mean())
    seed = SEED + int(hashlib.sha1(seed_label.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    shuffled_means = np.asarray([
        float(upwind_weight(bins, stratified_shuffle(from_deg, strata, rng)).mean())
        for _ in range(PERMUTATIONS)
    ])
    shuffled_mean = float(shuffled_means.mean())
    shuffled_sd = float(shuffled_means.std())
    return {
        "n_complaints": int(len(complaints)),
        "real_mean": real_mean,
        "shuffled_mean": shuffled_mean,
        "ratio": real_mean / max(shuffled_mean, 1e-12),
        "z": (real_mean - shuffled_mean) / max(shuffled_sd, 1e-12),
        "p_one_sided": float((np.sum(shuffled_means >= real_mean) + 1) / (PERMUTATIONS + 1)),
    }


def main() -> None:
    complaints = load_complaints_and_wind()
    sources = pd.read_csv(SOURCES_PATH, encoding="utf-8-sig")
    sources = sources.dropna(subset=["latitude", "longitude"]).copy()
    prefilter, prefilter_total = load_prefilter_factories()
    source_sets = {
        "factory_filtered": sources[sources["source_type"].eq("factory")],
        "wastewater_plus_manure": sources[sources["source_type"].isin(["wastewater", "manure_plant"])],
        "livestock": sources[sources["source_type"].eq("livestock")],
    }
    rows: list[dict[str, object]] = []
    for radius_km in RADII_KM:
        for group_name, set_name, point_sources in (
            ("factory", "factory_prefilter", prefilter),
            ("factory", "factory_filtered", source_sets["factory_filtered"]),
            ("sewer", "wastewater_plus_manure", source_sets["wastewater_plus_manure"]),
            ("livestock", "livestock", source_sets["livestock"]),
        ):
            group = complaints[complaints["complaint_group"].eq(group_name)].reset_index(drop=True)
            bins = bearing_bins_points(group, point_sources, radius_km)
            result = test_association(group, bins, f"{radius_km}|{group_name}|{set_name}")
            rows.append({
                "radius_km": radius_km, "complaint_group": group_name,
                "source_set": set_name, "n_sources": int(len(point_sources)), **result,
            })

        factory_group = complaints[complaints["complaint_group"].eq("factory")].reset_index(drop=True)
        zone_bins, zone_count = bearing_bins_zones(factory_group, radius_km)
        zone_result = test_association(factory_group, zone_bins, f"{radius_km}|factory|industrial_zone_boundary")
        rows.append({
            "radius_km": radius_km, "complaint_group": "factory",
            "source_set": "industrial_zone_boundary", "n_sources": zone_count, **zone_result,
        })
        filtered_bins = bearing_bins_points(factory_group, source_sets["factory_filtered"], radius_km)
        combined_result = test_association(
            factory_group, filtered_bins + zone_bins, f"{radius_km}|factory|filtered_plus_zone",
        )
        rows.append({
            "radius_km": radius_km, "complaint_group": "factory",
            "source_set": "factory_filtered_plus_zone_boundary",
            "n_sources": int(len(source_sets["factory_filtered"]) + zone_count), **combined_result,
        })

    payload = {
        "checked_at": "2026-09-20",
        "method": {
            "sector_deg": SECTOR_DEG, "radii_km": RADII_KM, "permutations": PERMUTATIONS,
            "min_wind_ms": MIN_WIND, "wind_window": "0,1,speed",
            "shuffle_strata": "month x 6-hour block", "source_weight": "one per facility/zone",
            "industrial_zone_distance": "shortest distance to polygon boundary",
        },
        "prefilter_sources_total": prefilter_total,
        "rows": rows,
    }
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"prefilter={prefilter_total}, geocoded={len(prefilter)}")
    for row in rows:
        print(
            f"R={row['radius_km']:.0f} {row['complaint_group']:9s} x {row['source_set']:38s} "
            f"n={row['n_complaints']:4d} sources={row['n_sources']:4d} "
            f"ratio={row['ratio']:.3f} z={row['z']:+.2f} p={row['p_one_sided']:.3f}"
        )


if __name__ == "__main__":
    main()
