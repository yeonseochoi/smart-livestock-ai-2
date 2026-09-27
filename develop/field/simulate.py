"""C 현장 추적 시뮬레이션(1차): 점수 카드 vs 기준선 3종, 가중치 소거 실험.

가상 발생원 농가를 정하고 담당자를 그 풍하 플룸 안에 둔 뒤, 추천 순위에서 발생원이 몇 번째인지 본다.
추적 로직과 같은 가정으로 만들지 않도록 관측 풍향 흔들림, 플룸 사행, 근사 좌표 오차, 동시 배출을 넣는다.
과거 반복(hist) 항목은 가상 발생원과 무관한 실제 민원 이력이라 여기서 검증할 수 없으므로 끈다.

실행: python -m develop.field.simulate
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from dataclasses import asdict

import numpy as np
import pandas as pd

from develop import common as c
from develop.field import engine as e

FARMS_FILE = Path(os.environ.get("DEVELOP_FARMS", c.OUTPUT_DIR / "data" / "farms.parquet"))
OUT_DIR = c.OUTPUT_DIR / "field"
SIGMAS = (10.0, 20.0, 45.0)
PRIORS = ("uniform", "eq")
N_TRIALS = 400
APPROX_ERR_KM = 0.4
P_SECOND_SOURCE = 0.3


def load_farms() -> pd.DataFrame:
    f = pd.read_parquet(FARMS_FILE)
    f = f.dropna(subset=["lat", "lon"])
    f = f[f["status"].isna() | (f["status"] == "정상")].reset_index(drop=True)
    return f


def _offset(lat, lon, bearing, dist_km):
    t = np.radians(bearing)
    return lat + dist_km * np.cos(t) / c.KM_PER_DEG_LAT, lon + dist_km * np.sin(t) / c.KM_PER_DEG_LON


def _rank_of(ids: list, sources: set) -> float:
    for i, fid in enumerate(ids):
        if fid in sources:
            return float(i + 1)
    return np.nan


def _baselines(seen: pd.DataFrame, slat, slon, obs_dir) -> dict[str, list]:
    d = c.distance_km(slat, slon, seen["lat"], seen["lon"])
    dtheta = c.angle_diff(c.bearing_deg(slat, slon, seen["lat"], seen["lon"]), obs_dir)
    near = seen.assign(d=d, dt=dtheta)
    near = near[near["d"] <= e.MAX_RADIUS_KM]
    return {
        "가까운 순": near.sort_values("d")["farm_id"].tolist(),
        "풍상 반원 가까운 순": near[near["dt"] <= 90].sort_values("d")["farm_id"].tolist(),
        "규모 큰 순": near.assign(q=near["eq"].fillna(-1)).sort_values(["q", "d"], ascending=[False, True])["farm_id"].tolist(),
    }


def run_trial(farms, rng, sigma, prior, weight_sets):
    if prior == "eq":
        w = farms["eq"].fillna(farms["eq"].median()).clip(lower=1).to_numpy()
        p = w / w.sum()
    else:
        p = None
    src_idx = [int(rng.choice(len(farms), p=p))]
    theta = rng.uniform(0, 360)  # 실제 풍향(불어오는 방향)
    s = farms.iloc[src_idx[0]]
    down = (theta + 180.0) % 360.0
    slat, slon = _offset(s["lat"], s["lon"], down + rng.normal(0, sigma), rng.uniform(0.5, 3.0))

    if rng.random() < P_SECOND_SOURCE:  # 같은 풍향에서 담당자 풍상 4 km 안 다른 농가가 동시 배출
        d = c.distance_km(slat, slon, farms["lat"], farms["lon"])
        dt = c.angle_diff(c.bearing_deg(slat, slon, farms["lat"], farms["lon"]), theta)
        pool = np.where((d <= 4) & (dt <= 60) & (np.arange(len(farms)) != src_idx[0]))[0]
        if len(pool):
            src_idx.append(int(rng.choice(pool)))
    sources = set(farms.iloc[src_idx]["farm_id"])

    obs_dir = (theta + rng.normal(0, sigma)) % 360.0  # 관측 풍향 흔들림
    obs = [e.Observation(slat, slon)]
    for _ in range(2):  # 플룸 안 추가 측정 2곳
        la, lo = _offset(s["lat"], s["lon"], down + rng.normal(0, sigma), rng.uniform(0.5, 3.0))
        obs.append(e.Observation(la, lo))

    seen = farms.copy()  # 근사 좌표는 오차를 준 위치로 보인다
    approx = (seen["coord_precision"] == "approx").to_numpy()
    n = approx.sum()
    if n:
        seen.loc[approx, "lat"] += rng.normal(0, APPROX_ERR_KM, n) / c.KM_PER_DEG_LAT
        seen.loc[approx, "lon"] += rng.normal(0, APPROX_ERR_KM, n) / c.KM_PER_DEG_LON

    wind = e.Wind(obs_dir, 1.5, sigma)
    out = {}
    for name, wts in weight_sets.items():
        r = e.score_candidates(seen, slat, slon, wind, observations=obs, weights=wts, top_k=10_000)
        out[name] = _rank_of(r.candidates["farm_id"].tolist(), sources)
    for name, ids in _baselines(seen, slat, slon, obs_dir).items():
        out[name] = _rank_of(ids, sources)
    return out


def summarize(ranks: pd.Series) -> dict:
    found = ranks.dropna()
    return {
        "top1": round(float((ranks <= 1).mean()), 3),
        "top3": round(float((ranks <= 3).mean()), 3),
        "top5": round(float((ranks <= 5).mean()), 3),
        "miss": round(float(ranks.isna().mean()), 3),
        "wasted_visits_median_if_found": None if found.empty else float(found.median() - 1),
    }


def main(n_trials: int = N_TRIALS, seed: int = 0) -> dict:
    farms = load_farms()
    no_hist = dict(hist=0.0)
    weight_sets = {
        "규모 추가 15": e.Weights(scale=15.0, **no_hist),
        "규모 추가 30": e.Weights(scale=30.0, **no_hist),
        "점수 카드(풍향40·거리25·다중20)": e.Weights(**no_hist),
        "소거: 풍향 제외": e.Weights(wind=0.0, **no_hist),
        "소거: 거리 제외": e.Weights(dist=0.0, **no_hist),
        "소거: 다중 측정 제외": e.Weights(multi=0.0, **no_hist),
    }
    rng = np.random.default_rng(seed)
    results = {}
    for prior in PRIORS:
        for sigma in SIGMAS:
            rows = [run_trial(farms, rng, sigma, prior, weight_sets) for _ in range(n_trials)]
            df = pd.DataFrame(rows)
            results[f"prior={prior}, sigma={int(sigma)}"] = {m: summarize(df[m]) for m in df.columns}
    meta = {
        "n_farms": int(len(farms)),
        "n_trials_per_condition": n_trials,
        "seed": seed,
        "conditions": "관측 풍향 N(0,σ) 흔들림, 플룸 사행 N(0,σ), 담당자 0.5~3 km 풍하, 근사 좌표 N(0,0.4 km), 30% 동시 배출, 냄새 확인 지점 3곳",
        "not_tested": "과거 반복(hist) 항목, 약풍 농도 경사 모드, 반복 추적(multi-hop)",
        "weights": {k: asdict(v) for k, v in weight_sets.items()},
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "simulation_v2_scale.json").write_text(json.dumps({"meta": meta, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"meta": meta, "results": results}


if __name__ == "__main__":
    res = main()
    for cond, r in res["results"].items():
        print(cond)
        for m, s in r.items():
            print(f"  {m:28s} top3={s['top3']:.3f} top1={s['top1']:.3f} miss={s['miss']:.3f} 헛방문중앙값={s['wasted_visits_median_if_found']}")
