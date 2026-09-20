"""1단 발생 위험 예보 — "다음 1시간 안에 이 1km 격자에서 민원이 생길까"를 시간 단위로 예측한다.

기존 모델(2단, run_ablation.py의 M0)은 민원이 30분 쌓인 뒤 확산을 예측하므로 기상·발생원이 끼어들 여지가 작다.
이 모델은 민원이 아직 없는 시점에서 기상(ASOS+AWS 격자별 IDW)·발생원 노출(6km 상풍측 축산 배출)·시간·과거 빈도로
격자별 발생 위험을 낸다. 학습 단위는 (시각 t, 격자 g), 라벨은 [t, t+1h) 민원 유무.

실험군(소거):
  R0  시간(시각·월·주말·야간) + 격자 과거 빈도(365일·30일 rolling) + 최근 24시간 민원 유무 + 시 전체 30일 민원율
  R1  R0 + 기상(풍속·풍향 sin/cos·강수 1h/3h·기온·습도·정체)
  R2  R1 + 발생원 노출(상풍측 6km 가중치 합, 전방위 6km 가중치 합, 그 비율) — 계수 세트 2벌(EEA NH3, 시설 수)
  R5  R2 + 풍향 조건부 과거 민원 지도(CPF형: 격자 g에서 "지금 풍향 구간"일 때 과거 1년 민원율, 전체·가축·공장) — 채택
  R5o R0 + 기상 + 풍향 조건부 과거 민원 지도만 (외부 발생원 자료 없이 되는지)
  건물 밀도·축산 외 시설·산업단지 경계·대기안정도 실험군(R3·R2s·R2z·R2w·R4)은 기여가 없어 제거했다.
  결과는 outputs/onset_risk/experiment_summary.md, 코드는 git 태그 pre-refactor-2026-09-20.

분할: 시간순. 학습 2020-01~2024-12, 테스트 2025-01~2026-07(--train-end 로 변경). 음성(민원 없음) 행은 학습에서 3% 표본.
평가: (1) 테스트(민원 있던 시각 전 격자) PR-AUC, (2) 민원이 1건 이상 있던 시각마다 위험 상위 K 격자 적중률(Hit@K, K=5·10),
      조용한 시각(직전 3시간 시 전체 민원 없음)만 따로, (3) 2단 테스트 Event의 첫 민원 격자가 event_hour-1h 시점 상위 10위 안인 비율.
누수 금지: 모든 특징은 t 이전 정보만. 과거 빈도·CPF는 t 미만 민원, 기상은 t 정시 관측.

실행: python run_onset_risk.py [--neg-rate 0.03] [--radius-km 6] [--label-type all|livestock|factory|sewage]
                               [--wind-source asos|aws|both] [--seeds 3] [--arms R0,R5] [--train-end 2025-01-01] [--tag 이름]
  --label-type : 양성 라벨을 그 악취종류 민원으로 제한(음성·과거 빈도 특징은 전체 민원 그대로).
  --wind-source: 바람 관측 자료원(wind_sources.py). 기본 both(전주·군산 ASOS + 익산·함라·여산·김제·진봉 AWS). 풍향·풍속은
                 격자 중심마다 IDW, 기온·습도·강수는 민원 중심점 값.
  --train-end  : 2단 테스트 Event 시각의 1단 점수를 연도별 확장 창(expanding window)으로 만들 때
                 2021-01-01, 2022-01-01, … 로 바꿔 돌린다(fuse_onset_spread.py 입력).
  산출물: metrics{tag}.json, onset_risk_table{tag}.md, onset_alerts{tag}.csv(테스트 시각별 상위 30, 서비스 입력),
          onset_cells.csv(1단이 다루는 격자 186개 목록. 후보 축소 규칙에서 "1단 격자 밖" 판정용),
          onset_event_scores{tag}.csv(전체 라벨일 때만. 2단 테스트 Event의 event_hour-1h 시점 전 격자 점수·순위 — 결합 평가용).

풍향 조건부 지도(CPF형)는 수용체 모델의 조건부 확률 함수(conditional probability function)를 격자 단위로 옮긴 것이다.
  cpf_<type>_365d = [t-365d, t) 동안 격자 g에 생긴 <type> 민원 중 그 시각 풍향 구간(30°, 정체는 별도 구간)이 지금과 같은 것의 수
                    ÷ 같은 기간 그 풍향 구간이었던 시간 수. 모두 t 미만 정보만 쓴다.
"""
from __future__ import annotations

import argparse
import json
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
import wind_sources as ws

OUTPUT_DIR = Path("outputs/onset_risk")
GRID_M = 1000
DEFAULT_TRAIN_END = pd.Timestamp("2025-01-01")
PERIOD_START = pd.Timestamp("2020-01-01")
SEED_POOL = (42, 7, 123, 0, 1, 2, 3, 4, 5, 6, 8, 9)
DEFAULT_SEEDS = 3
CALM_MS = 1.0  # 정체 기준 풍속. 이 밑이면 풍향이 정의되지 않는다.
CPF_SECTOR_DEG = 30
LABEL_PREFIX = {"all": None, "livestock": "가축", "factory": "공장", "sewage": "하수"}

TIME_FEATURES = ["hour_sin", "hour_cos", "month_sin", "month_cos", "weekend", "night"]
PRIOR_FEATURES = ["cell_rate_365d", "cell_rate_30d", "cell_recent_24h", "city_rate_30d"]
WEATHER_FEATURES = ["wind_speed", "wind_from_sin", "wind_from_cos", "rain_1h", "rain_3h", "temperature", "humidity", "stagnation"]
SOURCE_FEATURES = ["upwind_eea_6km", "total_eea_6km", "upwind_share_eea", "upwind_count_6km", "total_count_6km", "upwind_share_count"]
CPF_FEATURES = ["cpf_all_365d", "cpf_all_ratio", "cpf_livestock_365d", "cpf_factory_365d"]
ARMS = {
    "R0": TIME_FEATURES + PRIOR_FEATURES,
    "R1": TIME_FEATURES + PRIOR_FEATURES + WEATHER_FEATURES,
    "R2": TIME_FEATURES + PRIOR_FEATURES + WEATHER_FEATURES + SOURCE_FEATURES,
    "R5": TIME_FEATURES + PRIOR_FEATURES + WEATHER_FEATURES + SOURCE_FEATURES + CPF_FEATURES,
    "R5o": TIME_FEATURES + PRIOR_FEATURES + WEATHER_FEATURES + CPF_FEATURES,
}
EXPORT_ARM_PRIORITY = ("R5", "R2")
ALERT_TOP_N = 30  # onset_alerts*.csv 에 남기는 시각별 상위 격자 수(문서의 사전 경보 5개 + 후보 축소 규칙 30위)


def count_between(times: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """정렬된 시각 배열에서 [start, end) 안의 개수(행별)."""
    return np.searchsorted(times, end, side="left") - np.searchsorted(times, start, side="left")


def first_cells_of_events(data: pd.DataFrame, complaints: pd.DataFrame, train_end: pd.Timestamp) -> pd.DataFrame:
    """2단(run_ablation) 테스트 Event의 첫 민원 격자. 학습 Event는 결합 평가 대상이 아니다."""
    gridded = sensitivity.add_grid(complaints, GRID_M)
    gridded["event_hour"] = gridded["datetime"].dt.floor("h")
    test_hours = data.loc[~data["is_train"], ["event_hour"]].drop_duplicates()
    merged = gridded.merge(test_hours, on="event_hour").sort_values("datetime")
    first = merged.groupby("event_hour").first().reset_index()
    first = first[first["event_hour"] >= train_end]
    return first.rename(columns={"grid_x": "first_grid_x", "grid_y": "first_grid_y"})[["event_hour", "first_grid_x", "first_grid_y"]]


def build_panel(complaints: pd.DataFrame, events: pd.DataFrame, neg_rate: float, radius_km: float,
                rng: np.random.Generator, label_type: str, wind_source: str, train_end: pd.Timestamp,
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(시각, 격자) 패널과 격자 표. 특징은 모두 t 이전 정보만 쓴다."""
    meta = {"lat0": float(complaints["latitude"].median()), "lon0": float(complaints["longitude"].median()), "grid_m": GRID_M}
    gridded = sensitivity.add_grid(complaints, GRID_M)
    gridded["hour"] = gridded["datetime"].dt.floor("h")

    # 격자 집합: 2019~2026 민원이 1건 이상 있던 칸. 후보를 넓히면 음성만 늘어난다.
    cells = gridded.groupby(["grid_x", "grid_y"]).size().rename("n").reset_index()
    cells["cell_id"] = np.arange(len(cells))
    centers = np.array([odor.grid_centroid(int(x), int(y), meta) for x, y in zip(cells["grid_x"], cells["grid_y"])])
    cells["center_latitude"], cells["center_longitude"] = centers[:, 0], centers[:, 1]

    # 시각 축: 관측이 있는 2020-01 ~ 마지막 관측 시각. 중심점 바람은 CPF 구간과 정체 판정에, 격자별 바람은 특징에 쓴다.
    field = ws.WindField(ws.load_wind_stations(wind_source))
    wind = field.center_table((meta["lat0"], meta["lon0"]))[["u", "v", "speed", "from_deg"]].copy()
    extra = ab.load_asos().groupby("datetime")[["rainfall_hour", "temperature", "humidity"]].mean()
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
    # "2단 테스트 Event의 event_hour-1h"에 대해 전 격자(시각별 순위 평가용). 그 외 테스트 시각은 쓰지 않는다.
    n_cells = len(cells)
    train_hours = hours[hours < train_end]
    mask = rng.random((len(train_hours), n_cells)) < neg_rate
    h_idx, c_idx = np.where(mask)
    train_neg = pd.DataFrame({"hour": train_hours[h_idx], "cell_id": c_idx})
    event_lead_hours = pd.DatetimeIndex(events["event_hour"]) - pd.Timedelta(hours=1)
    test_hours = pd.DatetimeIndex(sorted(set(positives.loc[positives["hour"] >= train_end, "hour"]) | set(event_lead_hours)))
    test_hours = test_hours[test_hours.isin(hours)]
    test_neg = pd.DataFrame({"hour": np.repeat(test_hours.to_numpy(), n_cells), "cell_id": np.tile(np.arange(n_cells), len(test_hours))})
    negatives = pd.concat([train_neg, test_neg], ignore_index=True)
    negatives = negatives.merge(positives[["hour", "cell_id"]], on=["hour", "cell_id"], how="left", indicator=True)
    negatives = negatives[negatives["_merge"] == "left_only"].drop(columns="_merge")
    negatives["n_complaints"] = 0
    negatives["target"] = 0
    positives = positives[(positives["hour"] < train_end) | positives["hour"].isin(test_hours)]
    panel = pd.concat([positives, negatives], ignore_index=True)
    panel = panel.merge(cells, on="cell_id", how="left")
    panel["is_train"] = panel["hour"] < train_end
    panel["sample_weight"] = np.where(panel["is_train"] & (panel["target"] == 0), 1.0 / neg_rate, 1.0)

    # 시간 특징
    panel["hour_sin"] = np.sin(2 * np.pi * panel["hour"].dt.hour / 24)
    panel["hour_cos"] = np.cos(2 * np.pi * panel["hour"].dt.hour / 24)
    panel["month_sin"] = np.sin(2 * np.pi * panel["hour"].dt.month / 12)
    panel["month_cos"] = np.cos(2 * np.pi * panel["hour"].dt.month / 12)
    panel["weekend"] = (panel["hour"].dt.dayofweek >= 5).astype(int)
    panel["night"] = ((panel["hour"].dt.hour >= 21) | (panel["hour"].dt.hour < 6)).astype(int)

    panel = add_rolling_rates(panel, gridded, cells)
    panel = add_wind_conditioned_rates(panel, gridded, cells, wind)

    # 기상(t 정시). 풍향·풍속은 격자 중심별 IDW 값, 기온·습도·강수는 중심점 값.
    w = wind.reindex(panel["hour"].to_numpy())
    local_from, local_speed = w["from_deg"].to_numpy().copy(), w["speed"].to_numpy().copy()
    hour_pos = field.hours.get_indexer(panel["hour"])
    cell_ids_arr = panel["cell_id"].to_numpy()
    for cell in cells.itertuples(index=False):
        table = field.at(float(cell.center_latitude), float(cell.center_longitude))
        idx = np.where(cell_ids_arr == cell.cell_id)[0]
        pos = hour_pos[idx]
        ok = pos >= 0
        local_from[idx[ok]] = table["from_deg"].to_numpy()[pos[ok]]
        local_speed[idx[ok]] = table["speed"].to_numpy()[pos[ok]]
    panel["wind_speed"] = local_speed
    panel["wind_from_deg"] = local_from
    panel["wind_from_sin"] = np.sin(np.radians(local_from))
    panel["wind_from_cos"] = np.cos(np.radians(local_from))
    panel["rain_1h"] = w["rainfall_hour"].to_numpy()
    panel["rain_3h"] = w["rain_3h"].to_numpy()
    panel["temperature"] = w["temperature"].to_numpy()
    panel["humidity"] = w["humidity"].to_numpy()
    panel["stagnation"] = (local_speed < CALM_MS).astype(int)

    # 발생원 노출: 격자 중심 기준 방위 구간별 축산 배출 가중치(계수 2벌) → t의 격자별 풍향으로 상풍측 합. 정체 시 방향 없음(NaN).
    raw_sources = pd.read_csv(assoc.SOURCES_PATH, encoding="utf-8-sig").dropna(subset=["latitude", "longitude"])
    if "source_type" in raw_sources.columns:
        raw_sources = raw_sources[raw_sources["source_type"] == "livestock"]
    cell_frame = cells.rename(columns={"center_latitude": "latitude", "center_longitude": "longitude"})
    from_deg = np.nan_to_num(local_from, nan=0.0)
    has_direction = local_speed >= CALM_MS
    for tag, set_name in (("eea", "eea_nh3"), ("count", "count")):
        src = sw.apply_weight_set(raw_sources, set_name)
        bins = assoc.bearing_bins(cell_frame, src[src["emission_weight"] > 0], radius_km)
        cell_bins = bins[cell_ids_arr]
        upwind = np.where(has_direction, assoc.upwind_weight(cell_bins, from_deg), np.nan)
        total = bins.sum(axis=1)[cell_ids_arr]
        panel[f"upwind_{tag}_6km"] = upwind
        panel[f"total_{tag}_6km"] = total
        with np.errstate(divide="ignore", invalid="ignore"):
            panel[f"upwind_share_{tag}"] = np.where(total > 0, upwind / total, np.nan)
    return panel, cells


def wind_sector(from_deg: np.ndarray, speed: np.ndarray) -> np.ndarray:
    """30° 풍향 구간 0~11, 정체(풍속<1)·결측은 12."""
    n_sec = 360 // CPF_SECTOR_DEG
    sec = (np.nan_to_num(from_deg, nan=0.0) // CPF_SECTOR_DEG).astype(int) % n_sec
    calm = np.isnan(from_deg) | np.isnan(speed) | (speed < CALM_MS)
    return np.where(calm, n_sec, sec)


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

    # 분모: 구간별 시간 수 (t 미만 365일)
    sector_hours = {s: np.sort(hours_sector.index[hours_sector.to_numpy() == s].to_numpy().astype("datetime64[ns]")) for s in range(n_sec)}
    denom = np.zeros(len(panel))
    for s in range(n_sec):
        idx = np.where(sectors == s)[0]
        denom[idx] = count_between(sector_hours[s], t[idx] - 365 * day, t[idx])
    for tag, mask in (("all", np.ones(len(hist), bool)), ("livestock", hist["is_livestock"].to_numpy()), ("factory", hist["is_factory"].to_numpy())):
        sub = hist[mask]
        times_by = {k: np.sort(g["datetime"].to_numpy().astype("datetime64[ns]")) for k, g in sub.groupby(["cell_id", "sector"])}
        num = np.zeros(len(panel))
        for (cell_id, s), times in times_by.items():
            idx = np.where((cell_ids == cell_id) & (sectors == s))[0]
            if len(idx):
                num[idx] = count_between(times, t[idx] - 365 * day, t[idx])
        with np.errstate(divide="ignore", invalid="ignore"):
            panel[f"cpf_{tag}_365d"] = np.where(denom > 0, num / denom, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        hourly_rate = panel["cell_rate_365d"].to_numpy() / 24.0  # 일 단위 → 시간 단위
        panel["cpf_all_ratio"] = np.where(hourly_rate > 0, panel["cpf_all_365d"].to_numpy() / hourly_rate, np.nan)
    return panel


def add_rolling_rates(panel: pd.DataFrame, gridded: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    """격자별·전체 과거 민원 수를 t 미만 기준으로 센다(2019년 민원 포함)."""
    hist = gridded.merge(cells[["grid_x", "grid_y", "cell_id"]], on=["grid_x", "grid_y"])
    all_times = {cell_id: np.sort(group["datetime"].to_numpy().astype("datetime64[ns]")) for cell_id, group in hist.groupby("cell_id")}
    city_times = np.sort(gridded["datetime"].to_numpy().astype("datetime64[ns]"))
    t = panel["hour"].to_numpy().astype("datetime64[ns]")
    cell_ids = panel["cell_id"].to_numpy()
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
    positive = probe["target"] == 1
    if quiet_only:
        positive &= probe["city_quiet_3h"] == 1
    hours = probe.loc[positive, "hour"].unique()
    hits = [float(group.nlargest(k, "score")["target"].sum() > 0) for _, group in probe[probe["hour"].isin(hours)].groupby("hour")]
    return float(np.mean(hits)) if hits else float("nan")


def onset_lead(test: pd.DataFrame, score: np.ndarray, events: pd.DataFrame, k: int = 10) -> dict:
    """2단 테스트 Event의 첫 민원 격자가 event_hour-1h 시점 상위 k에 있었는지."""
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


def evaluate_arm(train: pd.DataFrame, test: pd.DataFrame, features: list[str], seeds: tuple[int, ...],
                 events: pd.DataFrame) -> tuple[dict, np.ndarray]:
    """seed별 학습·평가 후 평균과 표준편차. 두 번째 반환값은 seed 평균 점수(서비스 산출물·결합 평가에 재사용)."""
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
    result["onset_lead"] = onset_lead(test, mean_score, events)
    result["feature_importance"] = {f: float(v) for f, v in sorted(zip(features, np.mean(importances, axis=0)), key=lambda kv: -kv[1])[:10]}
    result["features"] = features
    return result, mean_score


def export_alerts(test: pd.DataFrame, score: np.ndarray, arm: str, events: pd.DataFrame, train_end: pd.Timestamp, suffix: str,
                  event_scores: bool) -> dict:
    """서비스 연결용 상위 10 격자(onset_alerts)와, 전체 라벨일 때만, 1단·2단 결합 평가용 Event 전 격자 점수(onset_event_scores)를 쓴다."""
    alerts = test[["hour", "grid_x", "grid_y", "center_latitude", "center_longitude", "city_quiet_3h",
                   "wind_from_deg", "wind_speed", "upwind_share_eea", "upwind_eea_6km", "target"]].copy()
    alerts["risk_score"] = score
    alerts["rank"] = alerts.groupby("hour")["risk_score"].rank(ascending=False, method="first").astype(int)
    if event_scores:
        lead = alerts[alerts["hour"].isin(pd.DatetimeIndex(events["event_hour"]) - pd.Timedelta(hours=1))].copy()
        lead["event_hour"] = lead["hour"] + pd.Timedelta(hours=1)
        lead["train_end"] = str(train_end.date())
        lead = lead[["event_hour", "hour", "grid_x", "grid_y", "risk_score", "rank", "train_end"]].sort_values(["event_hour", "rank"])
        lead.to_csv(OUTPUT_DIR / f"onset_event_scores{suffix}.csv", index=False, encoding="utf-8-sig")
    alerts = alerts[alerts["rank"] <= ALERT_TOP_N].sort_values(["hour", "rank"])
    hour_max = alerts.groupby("hour")["risk_score"].transform("max")
    alerts["relative_risk"] = (100 * alerts["risk_score"] / hour_max).round().astype(int)
    alerts["arm"] = arm
    alerts.to_csv(OUTPUT_DIR / f"onset_alerts{suffix}.csv", index=False, encoding="utf-8-sig")
    return {"arm": arm, "hours": int(alerts["hour"].nunique()), "rows": int(len(alerts))}


def markdown_table(summary: dict, seeds: tuple[int, ...]) -> str:
    lines = ["# 발생 위험 예보 (1단) 소거 실험", "",
             f"- 격자 {summary['cells']}개, 학습 {summary['train_rows']:,}행(양성 {summary['train_positives']:,}), "
             f"테스트 {summary['test_rows']:,}행(양성 {summary['test_positives']:,}, 양성률 {summary['test_positive_rate']:.4f})",
             f"- 테스트 구간 민원 있던 시각 {summary['test_hours_with_complaints']:,}개, 그중 직전 3시간 시 전체 민원 없던 '조용한 시각' "
             f"{summary['test_quiet_hours_with_complaints']:,}개. seed {list(seeds)} 평균. 라벨: {summary['label_type']}. "
             f"바람: {summary['wind_source']}(격자별 IDW). 분할 {summary['train_end']}.", "",
             "| 실험군 | 특징 수 | PR-AUC | Hit@5 | Hit@10 | 조용한 시각 Hit@5 | 조용한 시각 Hit@10 | Event 첫 격자 1h 전 top10 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, r in summary["arms"].items():
        lead = r["onset_lead"]
        lines.append(f"| {name} | {len(r['features'])} | {r['pr_auc']:.4f} ±{r['pr_auc_sd']:.4f} | {r['hit_at_5']:.3f} | {r['hit_at_10']:.3f} | "
                     f"{r['quiet_hit_at_5']:.3f} | {r['quiet_hit_at_10']:.3f} | {lead.get('first_cell_in_top10_1h_before', float('nan')):.2f} (n={lead['events']}) |")
    lines += ["", "## 특징 중요도 상위(마지막 실험군)", ""]
    last = list(summary["arms"].values())[-1]
    lines += [f"- {k}: {v:.3f}" for k, v in last["feature_importance"].items()]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--neg-rate", type=float, default=0.03)
    parser.add_argument("--radius-km", type=float, default=6.0)
    parser.add_argument("--tag", default="", help="산출물 파일 이름 접미사(실험 구분용)")
    parser.add_argument("--arms", default="", help="실행할 실험군 쉼표 목록(기본 전체)")
    parser.add_argument("--label-type", default="all", choices=list(LABEL_PREFIX), help="양성 라벨로 쓸 악취종류")
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS, help=f"seed 수(최대 {len(SEED_POOL)})")
    parser.add_argument("--wind-source", default="both", choices=["asos", "aws", "both"])
    parser.add_argument("--train-end", default=str(DEFAULT_TRAIN_END.date()), help="학습/테스트 분할 시점 YYYY-MM-DD")
    args = parser.parse_args()
    train_end = pd.Timestamp(args.train_end)
    seeds = SEED_POOL[: args.seeds]
    suffix = f"_{args.tag}" if args.tag else ""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    complaints, _, _, _, _ = odor.load_inputs()
    spread_data, _, _ = ab.build_base_dataset()
    events = first_cells_of_events(spread_data, complaints, train_end)
    panel, cells = build_panel(complaints, events, args.neg_rate, args.radius_km, np.random.default_rng(0),
                               args.label_type, args.wind_source, train_end)
    cells[["grid_x", "grid_y", "center_latitude", "center_longitude"]].to_csv(OUTPUT_DIR / "onset_cells.csv", index=False, encoding="utf-8-sig")
    train, test = panel[panel["is_train"]], panel[~panel["is_train"]]
    summary = {
        "cells": int(len(cells)), "train_rows": int(len(train)), "train_positives": int(train["target"].sum()),
        "test_rows": int(len(test)), "test_positives": int(test["target"].sum()),
        "test_hours_with_complaints": int(test.loc[test["target"] == 1, "hour"].nunique()),
        "test_quiet_hours_with_complaints": int(test.loc[(test["target"] == 1) & (test["city_quiet_3h"] == 1), "hour"].nunique()),
        "test_positive_rate": int(test["target"].sum()) / max(len(test), 1),
        "neg_rate": args.neg_rate, "radius_km": args.radius_km, "label_type": args.label_type,
        "wind_source": args.wind_source, "train_end": str(train_end.date()), "seeds": list(seeds), "arms": {},
    }
    wanted = [a.strip() for a in args.arms.split(",")] if args.arms else list(ARMS)
    unknown = [a for a in wanted if a not in ARMS]
    if unknown:
        raise SystemExit(f"모르는 실험군 {unknown}. 가능: {list(ARMS)}")
    arm_scores = {}
    for name in wanted:
        summary["arms"][name], arm_scores[name] = evaluate_arm(train, test, ARMS[name], seeds, events)
        r = summary["arms"][name]
        print(f"{name}: PR-AUC {r['pr_auc']:.4f} | Hit@5 {r['hit_at_5']:.3f} Hit@10 {r['hit_at_10']:.3f} | "
              f"quiet Hit@5 {r['quiet_hit_at_5']:.3f} Hit@10 {r['quiet_hit_at_10']:.3f}")

    # 서비스 연결: 채택 실험군(R5, 없으면 R2, 없으면 마지막)의 seed 평균 점수. 격자 중심 좌표는 민원 격자 기준(발생원 좌표 아님).
    export_arm = next((a for a in EXPORT_ARM_PRIORITY if a in arm_scores), wanted[-1])
    summary["alerts_export"] = export_alerts(test, arm_scores[export_arm], export_arm, events, train_end, suffix,
                                             event_scores=args.label_type == "all")
    (OUTPUT_DIR / f"metrics{suffix}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    table = markdown_table(summary, seeds)
    (OUTPUT_DIR / f"onset_risk_table{suffix}.md").write_text(table, encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
