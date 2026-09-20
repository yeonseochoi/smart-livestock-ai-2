"""바람 자료원 비교(B6): ASOS 중심 고정 가중(현행) vs 격자별 IDW vs AWS 추가, 풍향–발생원 연관 검정으로 비교.

같은 통계량(민원 시각 상풍측 ±30°, 반경 6 km 축산 배출 가중치 합, 층화 셔플)에서 바람만 바꾼다.
  asos_center : 전주·군산 ASOS, 민원 중심점 한 곳의 거리 가중(현행 test_wind_source_association)
  asos_local  : 같은 ASOS 두 지점, 민원 격자 중심마다 거리 가중(위치 고려)
  aws_local   : AWS 5지점(익산·함라·여산·김제·진봉)만, 격자별 거리 가중
  aws_iksan   : 익산 702 한 지점만
  both_local  : ASOS + AWS 7지점, 격자별 거리 가중(결측은 나머지로 재정규화)
민원 유형(전체·가축·공장·하수)별로 비율(실제 ÷ 셔플)과 z 를 낸다.

실행: python compare_wind_sources.py [--permutations 300] → outputs/wind_lag_sweep/wind_source_compare.{json,md}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import sensitivity_early_prediction as sensitivity
import species_weight_sets as sw
import test_wind_source_association as t
import wind_sources as ws

OUTPUT_DIR = Path("outputs/wind_lag_sweep")
RADIUS_KM = 6.0
GRID_M = 1000


def per_cell_wind(complaints: pd.DataFrame, field: ws.WindField, station_filter=None) -> pd.DataFrame:
    """민원 격자(1 km) 중심마다 field.at(...) 을 구해 민원 행에 from_deg·speed 를 붙인다."""
    gridded = sensitivity.add_grid(complaints, GRID_M)
    meta = {"lat0": float(complaints["latitude"].median()), "lon0": float(complaints["longitude"].median()), "grid_m": GRID_M}
    import build_odor_ai_mvp as odor
    cells = gridded[["grid_x", "grid_y"]].drop_duplicates()
    out = np.full((len(gridded), 2), np.nan)
    hour_pos = field.hours.get_indexer(gridded["hour"])
    for gx, gy in cells.itertuples(index=False):
        lat, lon = odor.grid_centroid(int(gx), int(gy), meta)
        table = field.at(lat, lon)
        idx = np.where((gridded["grid_x"] == gx) & (gridded["grid_y"] == gy))[0]
        pos = hour_pos[idx]
        ok = pos >= 0
        out[idx[ok], 0] = table["from_deg"].to_numpy()[pos[ok]]
        out[idx[ok], 1] = table["speed"].to_numpy()[pos[ok]]
    joined = gridded.copy()
    joined["from_deg"], joined["speed"] = out[:, 0], out[:, 1]
    return joined


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--permutations", type=int, default=300)
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t.PERMUTATIONS = args.permutations
    t.RADIUS_KM = RADIUS_KM
    rng = np.random.default_rng(t.SEED)
    complaints = t.load_complaints()
    complaints["is_sewage"] = complaints["odor_type"].astype(str).str.startswith("하수")
    sources = pd.read_csv(t.SOURCES_PATH, encoding="utf-8-sig").dropna(subset=["latitude", "longitude"])
    livestock = sw.apply_weight_set(sources[sources["source_type"].eq("livestock")] if "source_type" in sources else sources, "eea_nh3")
    livestock = livestock[livestock["emission_weight"] > 0]
    center = (float(complaints["latitude"].median()), float(complaints["longitude"].median()))

    asos = ws.load_wind_stations("asos")
    aws = ws.load_wind_stations("aws")
    both = ws.load_wind_stations("both")
    variants = {
        "asos_center": ("center", ws.WindField(asos)),
        "asos_local": ("local", ws.WindField(asos)),
        "aws_local": ("local", ws.WindField(aws)),
        "aws_iksan": ("local", ws.WindField(aws[aws["station_id"] == 702])),
        "both_local": ("local", ws.WindField(both)),
    }
    groups = {"전체": None, "가축냄새": "is_livestock", "공장냄새": "is_factory", "하수구냄새": "is_sewage"}
    rows = []
    for name, (mode, field) in variants.items():
        if mode == "center":
            table = field.center_table(center)
            joined = complaints.join(table[["from_deg", "speed"]], on="hour", how="inner")
        else:
            joined = per_cell_wind(complaints, field)
        joined = joined.dropna(subset=["from_deg", "speed"])
        joined = joined[joined["speed"] >= t.MIN_WIND].reset_index(drop=True)
        for gname, col in groups.items():
            sub = joined if col is None else joined[joined[col]]
            sub = sub.reset_index(drop=True)
            bins = t.bearing_bins(sub, livestock)
            r = t.run_test(sub, bins, rng)
            rows.append({"wind": name, "group": gname, "n": r["n"], "ratio": r["ratio_real_over_shuffled"], "z": r["z"], "p": r["p_value_one_sided"]})
            print(f"{name:12s} × {gname:5s} n={r['n']:5d} ratio {r['ratio_real_over_shuffled']:.3f} z {r['z']:+.1f} p {r['p_value_one_sided']:.3f}", flush=True)
    (OUTPUT_DIR / "wind_source_compare.json").write_text(json.dumps({"radius_km": RADIUS_KM, "permutations": args.permutations, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    table = pd.DataFrame(rows)
    lines = ["# 바람 자료원 비교: 풍향–발생원(축산, EEA NH3 가중) 연관 검정", "",
             f"통계량 = 민원 시각 상풍측 ±{t.SECTOR_DEG:.0f}° 반경 {RADIUS_KM:.0f} km 배출 가중치 합, 실제 ÷ 층화 셔플 평균({args.permutations}회). 풍속 ≥1 m/s 민원만. n 은 자료원마다 다름(결측·정체 차이).", "",
             "| 바람 자료원 | " + " | ".join(f"{g} (n)" for g in groups) + " |", "|---|" + "---|" * len(groups)]
    for name in variants:
        part = table[table["wind"] == name]
        lines.append(f"| {name} | " + " | ".join(f"{r.ratio:.3f} (z {r.z:+.0f}, n {r.n:,})" for r in part.itertuples()) + " |")
    (OUTPUT_DIR / "wind_source_compare.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
