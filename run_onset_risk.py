"""1단 원형: 발생 위험 예보 — "다음 1시간 안에 이 1km 격자에서 민원이 생길까"를 시간 단위로 예측한다.

기존 모델(2단, run_ablation.py의 M0)은 민원이 30분 쌓인 뒤 확산을 예측하므로 기상·발생원이 끼어들 여지가 작다.
이 원형은 민원이 아직 없는 시점에서 기상(ASOS)·발생원 노출(6km 상풍측 시설 가중치)·시간·과거 빈도로
격자별 발생 위험을 낸다. 학습 단위는 (시각 t, 격자 g), 라벨은 [t, t+1h) 민원 유무.

실험군(소거):
  R0  시간(시각·월·주말) + 격자 과거 빈도(365일·30일 rolling) + 최근 24시간 민원 유무(관성 통제)
  R1  R0 + 기상(풍속·풍향 sin/cos·강수 1h/3h·기온·습도·정체·야간)
  R2  R1 + 발생원 노출(상풍측 6km 가중치 합, 전방위 6km 가중치 합, 그 비율) — 계수 세트 2벌(EEA NH3, 시설 수)
  R3  R2 + 건물 밀도(outputs/grid_buildings/grid_buildings_1km.csv, 있을 때만)
  R2s R2 + 축산 외 발생원(공장·하수처리·분뇨/폐기물 시설 수, 산업단지 경계) 상풍측/전방위 6km
  R2z R2 + 산업단지 경계만(폴리곤 경계를 100m 간격 점으로 바꿔 상풍측 비율, 최단거리)
  R2w R2 + 대기안정도(Pasquill-Gifford 등급, ASOS 산정; outputs/weather_integration/asos_hourly_stability_2020_2026.csv 있을 때만)
  R4  R3 + 축산 외 발생원 + 대기안정도
  R5  R2 + 풍향 조건부 과거 민원 지도(CPF형: 격자 g에서 "지금 풍향 구간"일 때 과거 1년 민원율, 전체·가축·공장 유형별)
  R5o R0 + 풍향 조건부 과거 민원 지도만 (외부 발생원 자료 없이 되는지)

분할: 시간순. 학습 2020-01~2024-12, 테스트 2025-01~2026-07. 음성(민원 없음) 행은 학습에서 3% 표본 추출.
평가: (1) 테스트(민원 있던 시각 전 격자) PR-AUC, (2) 민원이 1건 이상 있던 시각마다 위험 상위 K 격자 적중률(Hit@K, K=5·10),
      (3) 선행 시간: 공통 Event 160개 중 테스트 구간 Event의 첫 민원 격자가 event_hour-1h 시점 상위 10위 안에 있던 비율.
누수 금지: 모든 특징은 t 이전 정보만. 과거 빈도는 t 미만 민원, 기상은 t 정시 관측(t 이후 값 없음).

실행: python run_onset_risk.py [--neg-rate 0.03] [--radius-km 6] [--wind-window lag,window,simple|speed] [--tag 이름]
  --wind-window 은 발생원 노출 계산에 쓰는 참조 풍향만 바꾼다(기본 0,1,speed = t 정시 바람). 기상 특징 자체는 t 정시 그대로.
  --label-type all|livestock|factory|sewage : 양성 라벨을 그 악취종류 민원으로 제한(음성·과거 빈도 특징은 전체 민원 그대로).
  --seeds N : seed 수(기본 3, 순서 42,7,123,0,1,2,...).
  --wind-source asos|aws|both : 바람 관측 자료원(wind_sources.py). 기본 both(전주·군산 ASOS + 익산·함라·여산·김제·진봉 AWS).
  --wind-mode center|local : center = 민원 중심점 한 곳의 거리 가중(구 방식), local(기본) = 격자 중심마다 거리 가중(IDW). 기온·습도·강수는 항상 center.
  --stations 146,140,702,... : 쓸 관측 지점만 남김(품질 의심 지점 제외 실험용).
  --calm-fill none|persist : 정체(풍속<1 m/s) 시각의 발생원 노출 참조 풍향. none(기본) = 방향 없음(노출 NaN),
                              persist = 그 격자에서 직전 3시간 안 마지막 비정체 풍향을 씀(냄새가 머문다는 가정).
  --train-end YYYY-MM-DD : 학습/테스트 분할 시점(기본 2025-01-01). 2단 테스트 Event 시각의 1단 점수를 연도별 확장 창
                           (expanding window)으로 만들 때 2021-01-01, 2022-01-01, … 로 바꿔 돌린다(fuse_onset_spread.py 입력).
  산출물: metrics{tag}.json, onset_risk_table{tag}.md, onset_alerts{tag}.csv(테스트 시각별 상위 10),
          onset_event_scores{tag}.csv(2단 테스트 Event의 event_hour-1h 시점 전 격자 점수·순위 — 1단·2단 결합 평가용).
  바람 자료원 비교(2026-09-20, 조용한 시각 Hit@5 R5): asos/center 0.581, asos/local 0.578, aws/local 0.598, both/local 0.599, both/center 0.593.
풍향 조건부 지도(CPF형)는 수용체 모델의 조건부 확률 함수(conditional probability function)를 격자 단위로 옮긴 것이다.
  cpf_<type>_365d = [t-365d, t) 동안 격자 g에 생긴 <type> 민원 중 그 시각 풍향 구간(30°, 정체는 별도 구간)이 지금과 같은 것의 수
                    ÷ 같은 기간 그 풍향 구간이었던 시간 수. 모두 t 미만 정보만 쓴다.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

import build_odor_ai_mvp as odor
import run_ablation as ab
import sensitivity_early_prediction as sensitivity
import species_weight_sets as sw
import test_wind_source_association as assoc

OUTPUT_DIR = Path("outputs/onset_risk")
BUILDINGS = Path("outputs/grid_buildings/grid_buildings_1km.csv")
GRID_M = 1000
TRAIN_END = pd.Timestamp("2025-01-01")
PERIOD_START = pd.Timestamp("2020-01-01")
SEEDS = (42, 7, 123)
SEED_POOL = (42, 7, 123, 0, 1, 2, 3, 4, 5, 6, 8, 9)

TIME_FEATURES = ["hour_sin", "hour_cos", "month_sin", "month_cos", "weekend", "night"]
PRIOR_FEATURES = ["cell_rate_365d", "cell_rate_30d", "cell_recent_24h", "city_rate_30d"]
WEATHER_FEATURES = ["wind_speed", "wind_from_sin", "wind_from_cos", "rain_1h", "rain_3h", "temperature", "humidity", "stagnation"]
SOURCE_FEATURES = ["upwind_eea_6km", "total_eea_6km", "upwind_share_eea", "upwind_count_6km", "total_count_6km", "upwind_share_count"]
BUILDING_FEATURES = ["building_count", "residential_count", "commercial_count"]
NON_LIVESTOCK_FEATURES = ["upwind_factory_6km", "total_factory_6km", "upwind_wastewater_6km", "total_wastewater_6km",
                          "upwind_manure_6km", "total_manure_6km"]
INDUSTRIAL_FEATURES = ["upwind_izone_6km", "total_izone_6km", "dist_izone_km"]
INDUSTRIAL_ZONES = Path("data/public_source_cache/industrial_zones.geojson")
STABILITY_FEATURES = ["stability_pg", "stable_flag"]
CPF_FEATURES = ["cpf_all_365d", "cpf_all_ratio", "cpf_livestock_365d", "cpf_factory_365d"]
CPF_SECTOR_DEG = 30
STABILITY = Path("outputs/weather_integration/asos_hourly_stability_2020_2026.csv")
STABILITY_CODE = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6}
ARMS = {
    "R0": TIME_FEATURES + PRIOR_FEATURES,
    "R1": TIME_FEATURES + PRIOR_FEATURES + WEATHER_FEATURES,
    "R2": TIME_FEATURES + PRIOR_FEATURES + WEATHER_FEATURES + SOURCE_FEATURES,
    "R3": TIME_FEATURES + PRIOR_FEATURES + WEATHER_FEATURES + SOURCE_FEATURES + BUILDING_FEATURES,
    "R2s": TIME_FEATURES + PRIOR_FEATURES + WEATHER_FEATURES + SOURCE_FEATURES + NON_LIVESTOCK_FEATURES + INDUSTRIAL_FEATURES,
    "R2z": TIME_FEATURES + PRIOR_FEATURES + WEATHER_FEATURES + SOURCE_FEATURES + INDUSTRIAL_FEATURES,
    "R2w": TIME_FEATURES + PRIOR_FEATURES + WEATHER_FEATURES + SOURCE_FEATURES + STABILITY_FEATURES,
    "R4": TIME_FEATURES + PRIOR_FEATURES + WEATHER_FEATURES + SOURCE_FEATURES + BUILDING_FEATURES + NON_LIVESTOCK_FEATURES + INDUSTRIAL_FEATURES + STABILITY_FEATURES,
    "R5": TIME_FEATURES + PRIOR_FEATURES + WEATHER_FEATURES + SOURCE_FEATURES + CPF_FEATURES,
    "R5o": TIME_FEATURES + PRIOR_FEATURES + WEATHER_FEATURES + CPF_FEATURES,
}


def reference_wind(wind: pd.DataFrame, spec: dict | None) -> pd.DataFrame:
    """발생원 노출용 참조 풍향. spec 없으면 t 정시 바람. 있으면 wind_window_sweep.window_reference(lag, window, weighting)."""
    if not spec or (spec["lag"] == 0 and spec["window"] == 1):
        return wind[["from_deg", "speed"]]
    import wind_window_sweep as wws
    base = wind.rename(columns={"rainfall_hour": "rain"})[["u", "v", "speed", "from_deg", "rain"]]
    return wws.window_reference(base, spec["lag"], spec["window"], spec["weighting"])[["from_deg", "speed"]]


def load_stability_series() -> pd.Series:
    if not STABILITY.exists():
        return pd.Series(dtype=object)
    frame = pd.read_csv(STABILITY, parse_dates=["datetime"]).dropna(subset=["stability_class"])
    frame["priority"] = np.where(frame["station_id"] == 146, 0, 1)  # 전주 우선, 없으면 군산
    frame = frame.sort_values(["datetime", "priority"]).drop_duplicates("datetime")
    return frame.set_index("datetime")["stability_class"]


LABEL_PREFIX = {"all": None, "livestock": "가축", "factory": "공장", "sewage": "하수"}


def build_panel(neg_rate: float, radius_km: float, rng: np.random.Generator, wind_spec: dict | None = None,
                label_type: str = "all", wind_source: str = "asos", wind_mode: str = "center",
                stations: tuple[int, ...] = (), calm_fill: str = "none") -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    complaints, _, _, _, _ = odor.load_inputs()
    meta = {"lat0": float(complaints["latitude"].median()), "lon0": float(complaints["longitude"].median()), "grid_m": GRID_M}
    gridded = sensitivity.add_grid(complaints, GRID_M)
    gridded["hour"] = gridded["datetime"].dt.floor("h")

    # 격자 집합: 2019~2026 민원이 1건 이상 있던 칸. 후보를 넓히면 음성만 늘어난다.
    cells = gridded.groupby(["grid_x", "grid_y"]).size().rename("n").reset_index()
    cells = cells[cells["n"] >= 1].reset_index(drop=True)
    cells["cell_id"] = np.arange(len(cells))
    centers = np.array([odor.grid_centroid(int(x), int(y), meta) for x, y in zip(cells["grid_x"], cells["grid_y"])])
    cells["center_latitude"], cells["center_longitude"] = centers[:, 0], centers[:, 1]

    # 시각 축: 관측이 있는 2020-01 ~ 마지막 관측 시각. 바람 자료원·보간 방식은 wind_sources.WindField
    import wind_sources as ws
    station_table = ws.load_wind_stations(wind_source)
    if stations:
        station_table = station_table[station_table["station_id"].isin(stations)]
    field = ws.WindField(station_table)
    center_table = field.center_table((meta["lat0"], meta["lon0"]))
    wind = center_table[["u", "v", "speed", "from_deg"]].copy()
    asos = ab.load_asos()
    extra = asos.groupby("datetime")[["rainfall_hour", "temperature", "humidity"]].mean()
    wind = wind.join(extra)
    wind = wind[wind["speed"].notna()]
    wind["rain_3h"] = wind["rainfall_hour"].rolling(3, min_periods=1).sum()
    wind = wind[wind.index >= PERIOD_START]
    hours = wind.index

    # 라벨: (hour, cell) 민원 유무
    in_period = gridded[(gridded["hour"] >= hours.min()) & (gridded["hour"] <= hours.max())]
    in_period = in_period.merge(cells[["grid_x", "grid_y", "cell_id"]], on=["grid_x", "grid_y"])
    prefix = LABEL_PREFIX[label_type]
    labelled = in_period if prefix is None else in_period[in_period["odor_type"].astype(str).str.startswith(prefix)]
    positives = labelled.groupby(["hour", "cell_id"]).size().rename("n_complaints").reset_index()
    positives["target"] = 1

    # 음성 표본: 학습 구간은 시각×격자에서 neg_rate 표본. 테스트 구간은 "민원이 있던 시각"과
    # "공통 Event의 event_hour-1h"에 대해 전 격자(시각별 순위 평가용). 그 외 테스트 시각은 쓰지 않는다.
    n_cells = len(cells)
    train_hours = hours[hours < TRAIN_END]
    mask = rng.random((len(train_hours), n_cells)) < neg_rate
    h_idx, c_idx = np.where(mask)
    train_neg = pd.DataFrame({"hour": train_hours[h_idx], "cell_id": c_idx})
    event_lead_hours = pd.DatetimeIndex(first_cells_of_events(cells)["event_hour"]) - pd.Timedelta(hours=1)
    test_hours = pd.DatetimeIndex(sorted(set(positives.loc[positives["hour"] >= TRAIN_END, "hour"]) | set(event_lead_hours)))
    test_hours = test_hours[test_hours.isin(hours)]
    test_neg = pd.DataFrame({"hour": np.repeat(test_hours.to_numpy(), n_cells), "cell_id": np.tile(np.arange(n_cells), len(test_hours))})
    negatives = pd.concat([train_neg, test_neg], ignore_index=True)
    negatives = negatives.merge(positives[["hour", "cell_id"]], on=["hour", "cell_id"], how="left", indicator=True)
    negatives = negatives[negatives["_merge"] == "left_only"].drop(columns="_merge")
    negatives["n_complaints"] = 0
    negatives["target"] = 0
    positives = positives[(positives["hour"] < TRAIN_END) | positives["hour"].isin(test_hours)]
    panel = pd.concat([positives, negatives], ignore_index=True)
    panel = panel.merge(cells, on="cell_id", how="left")
    panel["is_train"] = panel["hour"] < TRAIN_END
    panel["sample_weight"] = np.where(panel["is_train"] & (panel["target"] == 0), 1.0 / neg_rate, 1.0)

    # 시간 특징
    panel["hour_sin"] = np.sin(2 * np.pi * panel["hour"].dt.hour / 24)
    panel["hour_cos"] = np.cos(2 * np.pi * panel["hour"].dt.hour / 24)
    panel["month_sin"] = np.sin(2 * np.pi * panel["hour"].dt.month / 12)
    panel["month_cos"] = np.cos(2 * np.pi * panel["hour"].dt.month / 12)
    panel["weekend"] = (panel["hour"].dt.dayofweek >= 5).astype(int)
    panel["night"] = ((panel["hour"].dt.hour >= 21) | (panel["hour"].dt.hour < 6)).astype(int)

    # 과거 빈도(누수 없음): 격자별 민원 시각 정렬 → t 미만 개수 - (t-윈도) 미만 개수
    panel = add_rolling_rates(panel, in_period, gridded, cells)
    # 풍향 조건부 과거 민원 지도(CPF형, t 미만 정보만)
    panel = add_wind_conditioned_rates(panel, gridded, cells, wind)

    # 기상(t 정시). local 이면 풍향·풍속은 격자 중심별 IDW 값으로 덮어쓴다(기온·습도·강수는 center 그대로).
    w = wind.reindex(panel["hour"].to_numpy())
    local_from, local_speed = w["from_deg"].to_numpy().copy(), w["speed"].to_numpy().copy()
    filled_from = local_from.copy()  # calm_fill=persist 용: 정체 시각에 직전 비정체 풍향

    def persisted(table: pd.DataFrame) -> np.ndarray:
        direction = table["from_deg"].where(table["speed"] >= 1.0)
        return direction.ffill(limit=3).to_numpy()

    if wind_mode == "center" and calm_fill == "persist":
        filled_from = pd.Series(persisted(wind), index=wind.index).reindex(panel["hour"].to_numpy()).to_numpy()
    if wind_mode == "local":
        hour_pos = field.hours.get_indexer(panel["hour"])
        cell_ids_arr = panel["cell_id"].to_numpy()
        for cell in cells.itertuples(index=False):
            table = field.at(float(cell.center_latitude), float(cell.center_longitude))
            idx = np.where(cell_ids_arr == cell.cell_id)[0]
            pos = hour_pos[idx]
            ok = pos >= 0
            local_from[idx[ok]] = table["from_deg"].to_numpy()[pos[ok]]
            local_speed[idx[ok]] = table["speed"].to_numpy()[pos[ok]]
            if calm_fill == "persist":
                filled_from[idx[ok]] = persisted(table)[pos[ok]]
    w = w.assign(from_deg=local_from, speed=local_speed)
    panel["wind_speed"] = w["speed"].to_numpy()
    panel["wind_from_sin"] = np.sin(np.radians(w["from_deg"].to_numpy()))
    panel["wind_from_cos"] = np.cos(np.radians(w["from_deg"].to_numpy()))
    panel["rain_1h"] = w["rainfall_hour"].to_numpy()
    panel["rain_3h"] = w["rain_3h"].to_numpy()
    panel["temperature"] = w["temperature"].to_numpy()
    panel["humidity"] = w["humidity"].to_numpy()
    panel["stagnation"] = (w["speed"].to_numpy() < 1.0).astype(int)
    panel["wind_from_deg"] = w["from_deg"].to_numpy()
    if wind_mode == "local" or not wind_spec or (wind_spec["lag"] == 0 and wind_spec["window"] == 1):
        panel["ref_from_deg"] = w["from_deg"].to_numpy()  # local 이면 격자별 바람, 아니면 t 정시 center 바람
        panel["ref_speed"] = w["speed"].to_numpy()
    else:
        ref = reference_wind(wind, wind_spec).reindex(panel["hour"].to_numpy())
        panel["ref_from_deg"] = ref["from_deg"].to_numpy()
        panel["ref_speed"] = ref["speed"].to_numpy()

    # 대기안정도(Pasquill-Gifford A~F → 1~6, 안정 E·F 플래그). 파일 없으면 NaN.
    stability = load_stability_series()
    panel["stability_pg"] = panel["hour"].map(stability).map(STABILITY_CODE) if len(stability) else np.nan
    panel["stable_flag"] = np.where(panel["stability_pg"].isna(), np.nan, (panel["stability_pg"] >= 5).astype(float))

    # 발생원 노출: 격자 중심 기준 방위 구간별 가중치(계수 2벌) → t의 풍향으로 상풍측 합
    raw_sources = pd.read_csv(assoc.SOURCES_PATH, encoding="utf-8-sig").dropna(subset=["latitude", "longitude"])
    if "source_type" not in raw_sources.columns:
        raw_sources["source_type"] = "livestock"
    livestock = raw_sources[raw_sources["source_type"] == "livestock"]
    assoc.RADIUS_KM = radius_km
    cell_frame = cells.rename(columns={"center_latitude": "latitude", "center_longitude": "longitude"})
    if calm_fill == "persist":
        ref_dir = np.where(panel["ref_speed"].to_numpy() >= 1.0, panel["ref_from_deg"].to_numpy(), filled_from)
        from_deg = np.nan_to_num(ref_dir, nan=0.0)
        has_direction = ~np.isnan(ref_dir)  # 정체여도 직전 3시간 비정체 풍향이 있으면 방향 있음
    else:
        from_deg = np.nan_to_num(panel["ref_from_deg"].to_numpy(), nan=0.0)
        has_direction = panel["ref_speed"].to_numpy() >= 1.0  # 정체 시 방향 없음

    def exposure(src: pd.DataFrame, tag: str, share: bool) -> None:
        bins = assoc.bearing_bins(cell_frame, src)
        cell_bins = bins[panel["cell_id"].to_numpy()]
        upwind = np.where(has_direction, assoc.upwind_weight(cell_bins, from_deg), np.nan)
        total = bins.sum(axis=1)[panel["cell_id"].to_numpy()]
        panel[f"upwind_{tag}_6km"] = upwind
        panel[f"total_{tag}_6km"] = total
        if share:
            with np.errstate(divide="ignore", invalid="ignore"):
                panel[f"upwind_share_{tag}"] = np.where(total > 0, upwind / total, np.nan)

    for tag, set_name in (("eea", "eea_nh3"), ("count", "count")):
        src = sw.apply_weight_set(livestock, set_name)
        exposure(src[src["emission_weight"] > 0], tag, share=True)
    # 축산 외 발생원: 배출 대리량 없음 → 시설 1곳 = 1. manure = 가축분뇨 처리시설 + 폐기물 처리시설
    kind_types = {"factory": ("factory",), "wastewater": ("wastewater",), "manure": ("manure_plant", "waste_facility")}
    for kind, types in kind_types.items():
        src = raw_sources[raw_sources["source_type"].isin(types)].copy()
        src["emission_weight"] = 1.0
        if len(src):
            exposure(src, kind, share=False)
        else:
            panel[f"upwind_{kind}_6km"] = np.nan
            panel[f"total_{kind}_6km"] = np.nan
    # 산업단지: 폴리곤 경계를 100 m 간격 점으로 바꿔 단지당 총 가중치 1 → 상풍측 값 = 단지 경계 중 상풍측 비율의 합
    izone_points, izone_polys = load_industrial_zones()
    if len(izone_points):
        exposure(izone_points, "izone", share=False)
        centers = cell_frame[["latitude", "longitude"]].to_numpy()
        dist = np.array([min(odor_geo_distance_km(lat, lon, poly) for poly in izone_polys) for lat, lon in centers])
        panel["dist_izone_km"] = dist[panel["cell_id"].to_numpy()]
    else:
        for col in INDUSTRIAL_FEATURES:
            panel[col] = np.nan

    # 건물 밀도(있을 때만)
    if BUILDINGS.exists():
        buildings = pd.read_csv(BUILDINGS, encoding="utf-8-sig")
        present = [c for c in BUILDING_FEATURES if c in buildings.columns]
        panel = panel.merge(buildings[["grid_x", "grid_y"] + present], on=["grid_x", "grid_y"], how="left")
        for col in set(BUILDING_FEATURES) - set(present):
            panel[col] = np.nan
    else:
        for col in BUILDING_FEATURES:
            panel[col] = np.nan
    return panel, cells, meta


def load_industrial_zones() -> tuple[pd.DataFrame, list]:
    """산업단지 폴리곤(GeoJSON) → 경계 100 m 간격 점(단지당 가중치 합 1)과 shapely 폴리곤 목록. 파일 없으면 빈 값."""
    if not INDUSTRIAL_ZONES.exists():
        return pd.DataFrame(columns=["latitude", "longitude", "emission_weight"]), []
    from shapely.geometry import shape
    import json as _json
    features = _json.load(INDUSTRIAL_ZONES.open(encoding="utf-8"))["features"]
    rows, polys = [], []
    for feat in features:
        geom = shape(feat["geometry"])
        polys.append(geom)
        boundary = geom.boundary
        # 위경도 단위 경계 길이 → 대략 m 로 환산해 100 m 간격 표본
        n = max(int(boundary.length * 111_000 / 100), 8)
        pts = [boundary.interpolate(i / n, normalized=True) for i in range(n)]
        for pt in pts:
            rows.append({"latitude": pt.y, "longitude": pt.x, "emission_weight": 1.0 / n, "name": feat["properties"].get("name")})
    return pd.DataFrame(rows), polys


def odor_geo_distance_km(lat: float, lon: float, poly) -> float:
    """격자 중심에서 폴리곤까지 최단거리(km). 안에 있으면 0. 위경도 차를 km 로 근사."""
    from shapely.geometry import Point
    pt = Point(lon, lat)
    if poly.contains(pt):
        return 0.0
    nearest = poly.exterior.interpolate(poly.exterior.project(pt))
    return float(ab.haversine_km(lat, lon, nearest.y, nearest.x))


def wind_sector(from_deg: np.ndarray, speed: np.ndarray) -> np.ndarray:
    """30° 풍향 구간 0~11, 정체(풍속<1)·결측은 12."""
    sec = (np.nan_to_num(from_deg, nan=0.0) // CPF_SECTOR_DEG).astype(int) % (360 // CPF_SECTOR_DEG)
    calm = np.isnan(from_deg) | np.isnan(speed) | (speed < 1.0)
    return np.where(calm, 360 // CPF_SECTOR_DEG, sec)


def add_wind_conditioned_rates(panel: pd.DataFrame, gridded: pd.DataFrame, cells: pd.DataFrame, wind: pd.DataFrame) -> pd.DataFrame:
    """격자 g × 풍향 구간 s 별로 [t-365d, t) 민원 수 ÷ 같은 기간 s였던 시간 수. 유형별(전체·가축·공장)."""
    n_sec = 360 // CPF_SECTOR_DEG + 1
    hours_sector = pd.Series(wind_sector(wind["from_deg"].to_numpy(float), wind["speed"].to_numpy(float)), index=wind.index)
    hist = gridded.merge(cells[["grid_x", "grid_y", "cell_id"]], on=["grid_x", "grid_y"])
    hist = hist[hist["hour"].isin(hours_sector.index)].copy()
    hist["sector"] = hours_sector.reindex(hist["hour"]).to_numpy()
    hist["is_livestock"] = hist["odor_type"].astype(str).str.startswith("가축")
    hist["is_factory"] = hist["odor_type"].astype(str).str.startswith("공장")
    day = np.timedelta64(1, "D")
    t = panel["hour"].to_numpy().astype("datetime64[ns]")
    cell_ids = panel["cell_id"].to_numpy()
    sectors = hours_sector.reindex(panel["hour"]).to_numpy()

    def count_between(times: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
        return np.searchsorted(times, end, side="left") - np.searchsorted(times, start, side="left")

    # 분모: 구간별 시간 수 (t 미만 365일)
    sector_hours = {s: np.sort(hours_sector.index[hours_sector.to_numpy() == s].to_numpy().astype("datetime64[ns]")) for s in range(n_sec)}
    denom = np.zeros(len(panel))
    for s in range(n_sec):
        idx = np.where(sectors == s)[0]
        denom[idx] = count_between(sector_hours[s], t[idx] - 365 * day, t[idx])
    out = {}
    for tag, mask in (("all", np.ones(len(hist), bool)), ("livestock", hist["is_livestock"].to_numpy()), ("factory", hist["is_factory"].to_numpy())):
        sub = hist[mask]
        times_by = {k: np.sort(g["datetime"].to_numpy().astype("datetime64[ns]")) for k, g in sub.groupby(["cell_id", "sector"])}
        num = np.zeros(len(panel))
        for (cell_id, s), times in times_by.items():
            idx = np.where((cell_ids == cell_id) & (sectors == s))[0]
            if len(idx):
                num[idx] = count_between(times, t[idx] - 365 * day, t[idx])
        with np.errstate(divide="ignore", invalid="ignore"):
            out[f"cpf_{tag}_365d"] = np.where(denom > 0, num / denom, np.nan)
    for k, v in out.items():
        panel[k] = v
    with np.errstate(divide="ignore", invalid="ignore"):
        hourly_rate = panel["cell_rate_365d"].to_numpy() / 24.0  # 일 단위 → 시간 단위
        panel["cpf_all_ratio"] = np.where(hourly_rate > 0, panel["cpf_all_365d"].to_numpy() / hourly_rate, np.nan)
    return panel


def add_rolling_rates(panel: pd.DataFrame, in_period: pd.DataFrame, gridded: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    """격자별·전체 과거 민원 수를 t 미만 기준으로 센다(2019년 민원 포함)."""
    all_times = {}
    hist = gridded.merge(cells[["grid_x", "grid_y", "cell_id"]], on=["grid_x", "grid_y"])
    for cell_id, group in hist.groupby("cell_id"):
        all_times[cell_id] = np.sort(group["datetime"].to_numpy().astype("datetime64[ns]"))
    city_times = np.sort(gridded["datetime"].to_numpy().astype("datetime64[ns]"))
    t = panel["hour"].to_numpy().astype("datetime64[ns]")
    cell_ids = panel["cell_id"].to_numpy()

    def count_between(times: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
        return np.searchsorted(times, end, side="left") - np.searchsorted(times, start, side="left")

    out = {k: np.zeros(len(panel)) for k in ("cell_rate_365d", "cell_rate_30d", "cell_recent_24h")}
    day = np.timedelta64(1, "D")
    for cell_id in np.unique(cell_ids):
        idx = np.where(cell_ids == cell_id)[0]
        times = all_times.get(cell_id, np.array([], dtype="datetime64[ns]"))
        tt = t[idx]
        out["cell_rate_365d"][idx] = count_between(times, tt - 365 * day, tt) / 365.0
        out["cell_rate_30d"][idx] = count_between(times, tt - 30 * day, tt) / 30.0
        out["cell_recent_24h"][idx] = (count_between(times, tt - day, tt) > 0).astype(float)
    for k, v in out.items():
        panel[k] = v
    panel["city_rate_30d"] = count_between(city_times, t - 30 * day, t) / 30.0
    hour3 = np.timedelta64(3, "h")
    panel["city_quiet_3h"] = (count_between(city_times, t - hour3, t) == 0).astype(int)  # 평가 분할용(특징 아님)
    return panel


def fit(train: pd.DataFrame, features: list[str], seed: int) -> XGBClassifier:
    model = XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05, min_child_weight=20, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=2.0, objective="binary:logistic", eval_metric="aucpr", tree_method="hist", random_state=seed, n_jobs=4,
    )
    model.fit(train[features], train["target"], sample_weight=train["sample_weight"])
    return model


def hit_at_k(test: pd.DataFrame, score: np.ndarray, k: int, quiet_only: bool = False) -> float:
    """민원이 있던 시각마다 위험 상위 k 격자에 실제 민원 격자가 하나라도 있는 비율.

    quiet_only=True면 직전 3시간 시 전체에 민원이 없던 시각만 — 관성 없이 기상·발생원·과거 빈도로만 맞혀야 하는 '선제' 상황.
    """
    probe = test[["hour", "cell_id", "target", "city_quiet_3h"]].copy()
    probe["score"] = score
    hours = probe.loc[probe["target"] == 1, "hour"].unique()
    if quiet_only:
        hours = probe.loc[(probe["target"] == 1) & (probe["city_quiet_3h"] == 1), "hour"].unique()
    hits = []
    for _, group in probe[probe["hour"].isin(hours)].groupby("hour"):
        top = group.nlargest(k, "score")
        hits.append(float(top["target"].sum() > 0))
    return float(np.mean(hits)) if hits else float("nan")


def onset_lead(test: pd.DataFrame, score: np.ndarray, events: pd.DataFrame, cells: pd.DataFrame, k: int = 10) -> dict:
    """공통 Event(테스트 구간)의 첫 민원 격자가 event_hour-1h 시점 상위 k에 있었는지."""
    probe = test[["hour", "cell_id", "grid_x", "grid_y"]].copy()
    probe["score"] = score
    ranks = []
    for row in events.itertuples(index=False):
        before = pd.Timestamp(row.event_hour) - pd.Timedelta(hours=1)
        frame = probe[probe["hour"] == before]
        if frame.empty:
            continue
        frame = frame.assign(rank=frame["score"].rank(ascending=False, method="min"))
        hit = frame[(frame["grid_x"] == row.first_grid_x) & (frame["grid_y"] == row.first_grid_y)]
        ranks.append(float(hit["rank"].iloc[0]) if not hit.empty else float("nan"))
    ranks = np.asarray(ranks)
    valid = ranks[~np.isnan(ranks)]
    return {"events": int(len(valid)), f"first_cell_in_top{k}_1h_before": float(np.mean(valid <= k)) if len(valid) else float("nan"),
            "median_rank_1h_before": float(np.median(valid)) if len(valid) else float("nan")}


def first_cells_of_events(cells: pd.DataFrame) -> pd.DataFrame:
    """run_ablation의 공통 Event 표에서 테스트 구간 Event의 첫 민원 격자를 뽑는다."""
    complaints, _, _, _, _ = odor.load_inputs()
    data, _, _ = ab.build_base_dataset()
    gridded = sensitivity.add_grid(complaints, GRID_M)
    gridded["event_hour"] = gridded["datetime"].dt.floor("h")
    hours = data[["event_hour"]].drop_duplicates()
    merged = gridded.merge(hours, on="event_hour").sort_values("datetime")
    first = merged.groupby("event_hour").first().reset_index()
    test_ids = set(data.loc[~data["is_train"], "event_id"])  # 2단 테스트 Event만(학습 Event는 결합 평가 대상이 아님)
    test_hours = set(data.loc[data["event_id"].isin(test_ids), "event_hour"])
    first = first[(first["event_hour"] >= TRAIN_END) & first["event_hour"].isin(test_hours)]
    return first.rename(columns={"grid_x": "first_grid_x", "grid_y": "first_grid_y"})[["event_hour", "first_grid_x", "first_grid_y"]]


def main() -> None:
    global TRAIN_END
    parser = argparse.ArgumentParser()
    parser.add_argument("--neg-rate", type=float, default=0.03)
    parser.add_argument("--radius-km", type=float, default=6.0)
    parser.add_argument("--wind-window", default="0,1,speed", help="발생원 노출용 참조 풍향 창 'lag,window,simple|speed'")
    parser.add_argument("--tag", default="", help="산출물 파일 이름 접미사(실험 구분용)")
    parser.add_argument("--arms", default="", help="실행할 실험군 쉼표 목록(기본 전체)")
    parser.add_argument("--label-type", default="all", choices=list(LABEL_PREFIX), help="양성 라벨로 쓸 악취종류")
    parser.add_argument("--seeds", type=int, default=len(SEEDS), help="seed 수(최대 12)")
    parser.add_argument("--wind-source", default="both", choices=["asos", "aws", "both", "both_hm"])
    parser.add_argument("--wind-mode", default="local", choices=["center", "local"])
    parser.add_argument("--stations", default="", help="쓸 지점 번호 쉼표 목록(비우면 자료원 전체)")
    parser.add_argument("--calm-fill", default="none", choices=["none", "persist"])
    parser.add_argument("--train-end", default=str(TRAIN_END.date()), help="학습/테스트 분할 시점 YYYY-MM-DD")
    args = parser.parse_args()
    TRAIN_END = pd.Timestamp(args.train_end)
    seeds = SEED_POOL[: args.seeds]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    lag, window, weighting = args.wind_window.split(",")
    wind_spec = {"lag": int(lag), "window": int(window), "weighting": weighting}
    station_ids = tuple(int(x) for x in args.stations.split(",")) if args.stations else ()
    panel, cells, meta = build_panel(args.neg_rate, args.radius_km, rng, wind_spec, args.label_type, args.wind_source, args.wind_mode,
                                     station_ids, args.calm_fill)
    train, test = panel[panel["is_train"]], panel[~panel["is_train"]]
    events = first_cells_of_events(cells)
    summary = {
        "cells": int(len(cells)), "train_rows": int(len(train)), "train_positives": int(train["target"].sum()),
        "test_rows": int(len(test)), "test_positives": int(test["target"].sum()),
        "test_hours_with_complaints": int(test.loc[test["target"] == 1, "hour"].nunique()),
        "test_quiet_hours_with_complaints": int(test.loc[(test["target"] == 1) & (test["city_quiet_3h"] == 1), "hour"].nunique()),
        "neg_rate": args.neg_rate, "radius_km": args.radius_km, "buildings_available": bool(BUILDINGS.exists()),
        "stability_available": bool(STABILITY.exists()), "wind_window": wind_spec, "label_type": args.label_type,
        "wind_source": args.wind_source, "wind_mode": args.wind_mode, "stations": list(station_ids), "calm_fill": args.calm_fill,
        "train_end": str(TRAIN_END.date()),
        "seeds": list(seeds), "arms": {},
    }
    arms = dict(ARMS)
    if not BUILDINGS.exists():
        arms = {k: v for k, v in arms.items() if k not in ("R3", "R4")}
    if not STABILITY.exists():
        arms = {k: v for k, v in arms.items() if k not in ("R2w", "R4")}
    if args.arms:
        wanted = [a.strip() for a in args.arms.split(",")]
        arms = {k: v for k, v in arms.items() if k in wanted}
    for name, features in arms.items():
        per_seed, importances, scores = [], [], []
        for seed in seeds:
            model = fit(train, features, seed)
            score = model.predict_proba(test[features])[:, 1]
            scores.append(score)
            per_seed.append({
                "pr_auc": float(average_precision_score(test["target"], score)),
                "hit_at_5": hit_at_k(test, score, 5), "hit_at_10": hit_at_k(test, score, 10),
                "quiet_hit_at_5": hit_at_k(test, score, 5, quiet_only=True), "quiet_hit_at_10": hit_at_k(test, score, 10, quiet_only=True),
            })
            importances.append(model.feature_importances_)
        mean_score = np.mean(scores, axis=0)
        result = {k: float(np.mean([r[k] for r in per_seed])) for k in per_seed[0]}
        result.update({f"{k}_sd": float(np.std([r[k] for r in per_seed])) for k in per_seed[0]})
        result["onset_lead"] = onset_lead(test, mean_score, events, cells)
        result["feature_importance"] = {f: float(v) for f, v in sorted(zip(features, np.mean(importances, axis=0)), key=lambda kv: -kv[1])[:10]}
        result["features"] = features
        summary["arms"][name] = result
        print(f"{name}: PR-AUC {result['pr_auc']:.4f} | Hit@5 {result['hit_at_5']:.3f} Hit@10 {result['hit_at_10']:.3f} | quiet Hit@5 {result['quiet_hit_at_5']:.3f} Hit@10 {result['quiet_hit_at_10']:.3f}")
    baseline = summary["test_positives"] / max(summary["test_rows"], 1)
    summary["test_positive_rate"] = baseline
    # 서비스 연결용: 채택 실험군(R5 = R2 + 풍향 조건부 민원 지도, 없으면 R2)의 테스트 시각별 위험 상위 10 격자. 격자 중심 좌표는 민원 격자 기준(발생원 좌표 아님).
    export_arm = "R5" if "R5" in summary["arms"] else ("R2" if "R2" in summary["arms"] else list(summary["arms"])[-1])
    export_features = summary["arms"][export_arm]["features"]
    export_scores = np.mean([fit(train, export_features, seed).predict_proba(test[export_features])[:, 1] for seed in seeds], axis=0)
    alerts = test[["hour", "grid_x", "grid_y", "center_latitude", "center_longitude", "city_quiet_3h",
                   "wind_from_deg", "wind_speed", "upwind_share_eea", "upwind_eea_6km", "target"]].copy()
    alerts["risk_score"] = export_scores
    alerts["rank"] = alerts.groupby("hour")["risk_score"].rank(ascending=False, method="first").astype(int)
    # 1단·2단 결합 평가용: 2단 테스트 Event의 event_hour-1h 시점 전 격자 점수(fuse_onset_spread.py가 읽음)
    lead = alerts[alerts["hour"].isin(pd.DatetimeIndex(events["event_hour"]) - pd.Timedelta(hours=1))].copy()
    lead["event_hour"] = lead["hour"] + pd.Timedelta(hours=1)
    lead["train_end"] = str(TRAIN_END.date())
    lead = lead[["event_hour", "hour", "grid_x", "grid_y", "risk_score", "rank", "train_end"]].sort_values(["event_hour", "rank"])
    lead.to_csv(OUTPUT_DIR / f"onset_event_scores{'_' + args.tag if args.tag else ''}.csv", index=False, encoding="utf-8-sig")
    alerts = alerts[alerts["rank"] <= 10].sort_values(["hour", "rank"])
    hour_max = alerts.groupby("hour")["risk_score"].transform("max")
    alerts["relative_risk"] = (100 * alerts["risk_score"] / hour_max).round().astype(int)
    alerts["arm"] = export_arm
    alerts.to_csv(OUTPUT_DIR / f"onset_alerts{'_' + args.tag if args.tag else ''}.csv", index=False, encoding="utf-8-sig")
    summary["alerts_export"] = {"arm": export_arm, "hours": int(alerts["hour"].nunique()), "rows": int(len(alerts))}
    suffix = f"_{args.tag}" if args.tag else ""
    (OUTPUT_DIR / f"metrics{suffix}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 발생 위험 예보 원형 (1단) 소거 실험", "",
             f"- 발생원 노출 참조 풍향 창: lag {wind_spec['lag']}h · window {wind_spec['window']}h · {wind_spec['weighting']}"
             + (f" · 대기안정도 {'있음' if STABILITY.exists() else '없음'}"),
             f"- 격자 {summary['cells']}개, 학습 {summary['train_rows']:,}행(양성 {summary['train_positives']:,}), 테스트 {summary['test_rows']:,}행(양성 {summary['test_positives']:,}, 양성률 {baseline:.4f})",
             f"- 테스트 구간 민원 있던 시각 {summary['test_hours_with_complaints']:,}개, 그중 직전 3시간 시 전체 민원 없던 '조용한 시각' {summary['test_quiet_hours_with_complaints']:,}개. seed {list(seeds)} 평균. 라벨: {args.label_type}. 바람: {args.wind_source}/{args.wind_mode}. 분할 {summary['train_end']}.", "",
             "| 실험군 | 특징 수 | PR-AUC | Hit@5 | Hit@10 | 조용한 시각 Hit@5 | 조용한 시각 Hit@10 | Event 첫 격자 1h 전 top10 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, r in summary["arms"].items():
        lead = r["onset_lead"]
        lines.append(f"| {name} | {len(r['features'])} | {r['pr_auc']:.4f} ±{r['pr_auc_sd']:.4f} | {r['hit_at_5']:.3f} | {r['hit_at_10']:.3f} | "
                     f"{r['quiet_hit_at_5']:.3f} | {r['quiet_hit_at_10']:.3f} | {lead.get('first_cell_in_top10_1h_before', float('nan')):.2f} (n={lead['events']}) |")
    lines += ["", "## 특징 중요도 상위(마지막 실험군)", ""]
    last = list(summary["arms"].values())[-1]
    lines += [f"- {k}: {v:.3f}" for k, v in last["feature_importance"].items()]
    (OUTPUT_DIR / f"onset_risk_table{suffix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
