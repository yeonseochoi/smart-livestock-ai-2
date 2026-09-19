"""민원 전체(2020~)에서 "민원 시각의 바람이 발생원 쪽에서 불어오는가"를 무작위 대조로 검정한다.

Eckmann et al. (2017)처럼 확산 모델 없이 반복 관측 × 풍향 통계만 쓴다. Event 160개가 아니라
민원 수천 건을 쓰므로, 관측소 시간 단위 바람에 발생원 신호가 조금이라도 있는지 판별하는 용도다.

통계량: 민원마다 상풍측(바람이 불어오는 방향 ±sector°) 반경 R km 안의 발생원 배출 가중치 합.
        실제 풍향 vs 풍향을 (월·시간대 층 안에서) 뒤섞은 것 N회의 평균을 비교해 경험적 p값을 낸다.
층화 셔플을 쓰는 이유: 계절·시간대별 주풍이 있어 단순 셔플은 기후 신호를 발생원 신호로 오인할 수 있다.

실행: python test_wind_source_association.py  →  outputs/wind_source_association/result.json, result.md
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import build_odor_ai_mvp as odor
import run_ablation as ab

OUTPUT_DIR = Path("outputs/wind_source_association")
SOURCES_PATH = Path("outputs/source_backtrack/sources.csv")  # git 미커밋(VWorld 좌표). 로컬 전용
RADIUS_KM = 10.0
SECTOR_DEG = 30.0
LAGS_H = (0, 1, 2, 3)
PERMUTATIONS = 500
MIN_WIND = 1.0
SEED = 42


def load_complaints() -> pd.DataFrame:
    complaints, _, _, _, _ = odor.load_inputs()
    c = complaints.dropna(subset=["latitude", "longitude"]).copy()
    c = c[c["datetime"] >= "2020-01-01"]
    c["hour"] = c["datetime"].dt.floor("h")
    c["is_livestock"] = c["odor_type"].astype(str).str.startswith("가축")
    c["is_factory"] = c["odor_type"].astype(str).str.startswith("공장")
    return c.reset_index(drop=True)


def hourly_wind(asos: pd.DataFrame, center: tuple[float, float]) -> pd.DataFrame:
    """시각별 관측소 거리 가중 평균 바람(민원 중심 기준 고정 가중치). 불어오는 방향(from)으로 반환."""
    stations = asos.groupby("station_id")[["station_latitude", "station_longitude"]].first()
    weights = {sid: 1.0 / max(ab.haversine_km(center[0], center[1], r.station_latitude, r.station_longitude), 1.0) ** 2
               for sid, r in stations.iterrows()}
    asos = asos.assign(w=asos["station_id"].map(weights))
    grouped = asos.groupby("datetime")[["u", "v", "wind_speed", "rainfall_hour", "w"]].apply(lambda g: pd.Series({
        "u": np.average(g["u"], weights=g["w"]), "v": np.average(g["v"], weights=g["w"]),
        "speed": np.average(g["wind_speed"], weights=g["w"]), "rain": np.average(g["rainfall_hour"], weights=g["w"]),
    }))
    grouped["from_deg"] = (np.degrees(np.arctan2(-grouped["u"], -grouped["v"])) + 360.0) % 360.0
    return grouped


BIN_DEG = 5.0
N_BINS = int(360 / BIN_DEG)


def bearing_bins(complaints: pd.DataFrame, sources: pd.DataFrame) -> np.ndarray:
    """민원별 × 5° 방위 구간별, 반경 R 안 발생원 배출 가중치 합. 한 번만 계산해 셔플마다 재사용한다."""
    lat_c = np.radians(complaints["latitude"].to_numpy()); lon_c = np.radians(complaints["longitude"].to_numpy())
    lat_s = np.radians(sources["latitude"].to_numpy()); lon_s = np.radians(sources["longitude"].to_numpy())
    w_s = sources["emission_weight"].to_numpy(float)
    bins = np.zeros((len(complaints), N_BINS))
    step = 1000
    for start in range(0, len(complaints), step):
        sl = slice(start, start + step)
        dlat = lat_s[None, :] - lat_c[sl, None]
        dlon = lon_s[None, :] - lon_c[sl, None]
        a = np.sin(dlat / 2) ** 2 + np.cos(lat_c[sl, None]) * np.cos(lat_s[None, :]) * np.sin(dlon / 2) ** 2
        dist = 2 * odor.EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))
        x = np.sin(dlon) * np.cos(lat_s[None, :])
        y = np.cos(lat_c[sl, None]) * np.sin(lat_s[None, :]) - np.sin(lat_c[sl, None]) * np.cos(lat_s[None, :]) * np.cos(dlon)
        bearing = (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0  # 민원 → 발생원 방위
        b = (bearing // BIN_DEG).astype(int) % N_BINS
        weight = np.where((dist <= RADIUS_KM) & (dist > 0.05), w_s[None, :], 0.0)
        rows = np.repeat(np.arange(b.shape[0]), b.shape[1])
        np.add.at(bins[sl], (rows, b.ravel()), weight.ravel())
    return bins


def upwind_weight(bins: np.ndarray, from_deg: np.ndarray) -> np.ndarray:
    """상풍측 from_deg ± SECTOR 안의 구간 합."""
    half = int(round(SECTOR_DEG / BIN_DEG))
    center = (from_deg // BIN_DEG).astype(int) % N_BINS
    offsets = np.arange(-half, half + 1)
    idx = (center[:, None] + offsets[None, :]) % N_BINS
    return np.take_along_axis(bins, idx, axis=1).sum(axis=1)


def stratified_shuffle(from_deg: np.ndarray, strata: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    shuffled = from_deg.copy()
    for key in np.unique(strata):
        idx = np.where(strata == key)[0]
        shuffled[idx] = from_deg[rng.permutation(idx)]
    return shuffled


def run_test(sub: pd.DataFrame, bins: np.ndarray, rng: np.random.Generator) -> dict:
    from_deg = sub["from_deg"].to_numpy(float)
    strata = (sub["hour"].dt.month.astype(str) + "-" + (sub["hour"].dt.hour // 6).astype(str)).to_numpy()
    real_mean = float(upwind_weight(bins, from_deg).mean())
    shuffled_means = np.asarray([
        float(upwind_weight(bins, stratified_shuffle(from_deg, strata, rng)).mean()) for _ in range(PERMUTATIONS)
    ])
    p_value = float((np.sum(shuffled_means >= real_mean) + 1) / (PERMUTATIONS + 1))
    return {
        "n": int(len(sub)), "real_mean": real_mean, "shuffled_mean": float(shuffled_means.mean()),
        "shuffled_sd": float(shuffled_means.std()), "ratio_real_over_shuffled": real_mean / max(shuffled_means.mean(), 1e-9),
        "z": (real_mean - shuffled_means.mean()) / max(shuffled_means.std(), 1e-9), "p_value_one_sided": p_value,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    complaints = load_complaints()
    sources = pd.read_csv(SOURCES_PATH, encoding="utf-8-sig").dropna(subset=["latitude", "longitude"])
    sources = sources[sources["emission_weight"] > 0]
    asos = ab.load_asos()
    center = (float(complaints["latitude"].median()), float(complaints["longitude"].median()))
    wind = hourly_wind(asos, center)
    bins_all = bearing_bins(complaints, sources)
    bins_pigs = bearing_bins(complaints, sources[sources["species"] == "돼지"])

    results: dict = {"radius_km": RADIUS_KM, "sector_deg": SECTOR_DEG, "permutations": PERMUTATIONS,
                     "min_wind_ms": MIN_WIND, "sources_used": int(len(sources)), "tests": {}}
    for lag in LAGS_H:
        joined = complaints.copy()
        joined["ref_hour"] = joined["hour"] - pd.Timedelta(hours=lag)
        joined = joined.join(wind, on="ref_hour", how="inner")
        joined = joined[joined["speed"] >= MIN_WIND]
        groups = {
            "all": joined, "livestock": joined[joined["is_livestock"]], "factory": joined[joined["is_factory"]],
            "other": joined[~joined["is_livestock"] & ~joined["is_factory"]],
            "dry": joined[joined["rain"] == 0], "wet": joined[joined["rain"] > 0],
            "night_21_06": joined[(joined["hour"].dt.hour >= 21) | (joined["hour"].dt.hour < 6)],
        }
        for name, sub in groups.items():
            if len(sub) < 100:
                continue
            key = f"lag{lag}h/{name}"
            results["tests"][key] = run_test(sub, bins_all[sub.index.to_numpy()], rng)
            r = results["tests"][key]
            print(f"{key:22s} n={r['n']:5d} real/shuffled={r['ratio_real_over_shuffled']:.3f} z={r['z']:+.2f} p={r['p_value_one_sided']:.3f}")

    # 돼지만(배출 가중치 1.0)으로도 한 번: 축종 계수 가정에 결과가 좌우되는지 본다.
    joined = complaints.copy(); joined["ref_hour"] = joined["hour"] - pd.Timedelta(hours=1)
    joined = joined.join(wind, on="ref_hour", how="inner"); joined = joined[joined["speed"] >= MIN_WIND]
    for name, sub in (("all", joined), ("livestock", joined[joined["is_livestock"]])):
        key = f"lag1h/pigs_only/{name}"
        results["tests"][key] = run_test(sub, bins_pigs[sub.index.to_numpy()], rng)
        r = results["tests"][key]
        print(f"{key:22s} n={r['n']:5d} real/shuffled={r['ratio_real_over_shuffled']:.3f} z={r['z']:+.2f} p={r['p_value_one_sided']:.3f}")

    (OUTPUT_DIR / "result.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 민원 전체 풍향–발생원 연관 검정 (Eckmann 2017형, 확산 모델 없음)", "",
             f"- 민원 2020~ 좌표 있음, 풍속 ≥{MIN_WIND} m/s, 상풍측 ±{SECTOR_DEG:.0f}° 반경 {RADIUS_KM:.0f} km, 셔플 {PERMUTATIONS}회(월·6시간대 층화)",
             f"- 발생원 {len(sources)}곳(배출 가중치>0, VWorld 좌표는 로컬 전용)", "",
             "| 조건 | n | 실제/셔플 비 | z | p(단측) |", "|---|---:|---:|---:|---:|"]
    for key, r in results["tests"].items():
        lines.append(f"| {key} | {r['n']} | {r['ratio_real_over_shuffled']:.3f} | {r['z']:+.2f} | {r['p_value_one_sided']:.3f} |")
    (OUTPUT_DIR / "result.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
