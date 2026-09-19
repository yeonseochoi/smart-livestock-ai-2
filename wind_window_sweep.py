"""참조 바람 창(시차 × 평균 창 × 가중) 24조합의 풍향–발생원 연관 검정 sweep.

질문: 민원 시각의 바람을 그대로 쓰는 게 맞나, 1~3시간 전 바람이나 2~3시간 평균이 더 맞나?
방법: test_wind_source_association.py의 통계량(민원 시각 상풍측 ±30°, 반경 R 안 발생원 배출 가중치 합,
      월×6시간대 층화 셔플 대비 비율)을 그대로 두고, 참조 풍향만 바꿔 가며 비교한다.
  - 시차 lag ∈ {0,1,2,3}h: 민원 정시 h 기준 h-lag 시각부터
  - 창 window ∈ {1,2,3}: h-lag 부터 과거로 window개 정시를 평균 (1 = 단일 시각)
  - 가중 ∈ {simple, speed}: 단순 원형평균(단위벡터 합) / 풍속 가중(u,v 합)
  기본 Track A 2판 설정 = (lag 0, window 2, speed).
추가로 대기안정도(Pasquill-Gifford, ASOS 산정) 안정(E·F)/중립(D)/불안정(A~C) 층별로 같은 표를 낸다.

실행: python wind_window_sweep.py [--permutations 300] [--radius 6]
출력: outputs/wind_lag_sweep/association_sweep.{json,md}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_ablation as ab
import species_weight_sets as sw
import test_wind_source_association as t

OUTPUT_DIR = Path("outputs/wind_lag_sweep")
STABILITY_CANDIDATES = (
    Path("outputs/weather_integration/asos_hourly_stability_2020_2026.csv"),
    Path("../2026 스마트 축산 AI 공모전/outputs/weather_integration/asos_hourly_stability_2020_2026.csv"),
)
LAGS = (0, 1, 2, 3)
WINDOWS = (1, 2, 3)
WEIGHTINGS = ("simple", "speed")
GROUPS = ("all", "livestock", "factory", "dry", "wet", "night_21_06")
STABILITY_GROUPS = {"stable_EF": ("E", "F"), "neutral_D": ("D",), "unstable_ABC": ("A", "B", "C")}


def window_reference(wind: pd.DataFrame, lag: int, window: int, weighting: str) -> pd.DataFrame:
    """시각별 참조 풍향(from_deg)·평균 풍속. 창 안에 결측 시각이 있으면 있는 것만 평균."""
    hourly = wind.reindex(pd.date_range(wind.index.min(), wind.index.max(), freq="h"))
    if weighting == "speed":
        comp_u, comp_v = hourly["u"], hourly["v"]
    else:
        rad = np.radians(hourly["from_deg"])
        comp_u, comp_v = -np.sin(rad), -np.cos(rad)  # 단위 벡터(부는 방향)
    frame = pd.DataFrame({"cu": comp_u, "cv": comp_v, "speed": hourly["speed"], "rain": hourly["rain"]}, index=hourly.index)
    rolled = frame.rolling(window, min_periods=1).mean()  # h-window+1 .. h
    rolled = rolled.shift(lag)  # h-lag-window+1 .. h-lag
    out = pd.DataFrame({
        "from_deg": (np.degrees(np.arctan2(-rolled["cu"], -rolled["cv"])) + 360.0) % 360.0,
        "speed": rolled["speed"], "rain": hourly["rain"],  # 강수는 민원 시각 기준(조건 분리용)
    })
    return out.dropna(subset=["from_deg", "speed"])


def load_stability() -> pd.Series:
    for path in STABILITY_CANDIDATES:
        if path.exists():
            frame = pd.read_csv(path, parse_dates=["datetime"])
            frame = frame.dropna(subset=["stability_class"])
            # 전주(146) 우선, 없으면 군산(140)
            frame["priority"] = np.where(frame["station_id"] == 146, 0, 1)
            frame = frame.sort_values(["datetime", "priority"]).drop_duplicates("datetime")
            return frame.set_index("datetime")["stability_class"]
    return pd.Series(dtype=object)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--permutations", type=int, default=300)
    parser.add_argument("--radius", type=float, default=6.0)
    parser.add_argument("--weight-set", default="eea_nh3", choices=list(sw.SETS))
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t.PERMUTATIONS = args.permutations
    t.RADIUS_KM = args.radius
    rng = np.random.default_rng(t.SEED)

    complaints = t.load_complaints()
    raw = pd.read_csv(t.SOURCES_PATH, encoding="utf-8-sig").dropna(subset=["latitude", "longitude"])
    raw = raw[raw.get("source_type", "livestock").eq("livestock")] if "source_type" in raw else raw
    sources = sw.apply_weight_set(raw, args.weight_set)
    sources = sources[sources["emission_weight"] > 0]
    bins = t.bearing_bins(complaints, sources)
    asos = ab.load_asos()
    wind = t.hourly_wind(asos, (float(complaints["latitude"].median()), float(complaints["longitude"].median())))
    stability = load_stability()
    complaints["stability_class"] = complaints["hour"].map(stability)

    rows = []
    for lag in LAGS:
        for window in WINDOWS:
            for weighting in WEIGHTINGS:
                ref = window_reference(wind, lag, window, weighting)
                joined = complaints.join(ref, on="hour", how="inner")
                joined = joined[joined["speed"] >= t.MIN_WIND]
                groups = {
                    "all": joined, "livestock": joined[joined["is_livestock"]], "factory": joined[joined["is_factory"]],
                    "dry": joined[joined["rain"] == 0], "wet": joined[joined["rain"] > 0],
                    "night_21_06": joined[(joined["hour"].dt.hour >= 21) | (joined["hour"].dt.hour < 6)],
                }
                for name, classes in STABILITY_GROUPS.items():
                    groups[name] = joined[joined["stability_class"].isin(classes)]
                for group, sub in groups.items():
                    if len(sub) < 100:
                        continue
                    r = t.run_test(sub, bins[sub.index.to_numpy()], rng)
                    rows.append({"lag_h": lag, "window_h": window, "weighting": weighting, "group": group,
                                 "n": r["n"], "ratio": r["ratio_real_over_shuffled"], "z": r["z"], "p": r["p_value_one_sided"]})
                head = next(x for x in rows if x["lag_h"] == lag and x["window_h"] == window and x["weighting"] == weighting and x["group"] == "all")
                print(f"lag{lag} win{window} {weighting:6s} all n={head['n']} ratio={head['ratio']:.3f} z={head['z']:+.1f}", flush=True)

    table = pd.DataFrame(rows)
    ranking = table[table["group"] == "all"].sort_values("ratio", ascending=False).reset_index(drop=True)
    top3 = ranking.head(3)[["lag_h", "window_h", "weighting", "ratio", "z"]].to_dict("records")
    result = {"radius_km": args.radius, "sector_deg": t.SECTOR_DEG, "permutations": args.permutations,
              "min_wind_ms": t.MIN_WIND, "weight_set": args.weight_set, "sources_used": int(len(sources)),
              "stability_available": bool(len(stability)), "rows": rows, "top3_all": top3}
    (OUTPUT_DIR / "association_sweep.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# 참조 바람 창 sweep: 풍향–발생원 연관 검정", "",
             f"통계량 = 민원 시각 상풍측 ±{t.SECTOR_DEG:.0f}° 반경 {args.radius:.0f} km 발생원 배출 가중치({args.weight_set}) 합, "
             f"실제 ÷ 층화 셔플 평균({args.permutations}회). 1.00 = 우연과 같음. 풍속 ≥{t.MIN_WIND} m/s(창 평균).", "",
             "창 표기: lag = 민원 정시에서 몇 시간 전부터, window = 몇 개 정시를 평균(1=단일 시각), 가중 = simple(단순 원형평균)/speed(풍속 가중). "
             "Track A 2판 기본 = lag0·window2·speed.", ""]
    pivot_groups = ["all", "livestock", "factory", "dry", "wet", "night_21_06"]
    lines += ["| lag | window | 가중 | " + " | ".join(pivot_groups) + " |", "|---:|---:|---|" + "---|" * len(pivot_groups)]
    for (lag, window, weighting), sub in table.groupby(["lag_h", "window_h", "weighting"], sort=True):
        cells = []
        for group in pivot_groups:
            hit = sub[sub["group"] == group]
            cells.append(f"{hit['ratio'].iloc[0]:.3f} (z {hit['z'].iloc[0]:+.0f})" if len(hit) else "-")
        lines.append(f"| {lag} | {window} | {weighting} | " + " | ".join(cells) + " |")
    if len(stability):
        lines += ["", "## 대기안정도(Pasquill-Gifford) 층별, 전체 민원", "",
                  "| lag | window | 가중 | 안정(E·F) | 중립(D) | 불안정(A~C) |", "|---:|---:|---|---|---|---|"]
        for (lag, window, weighting), sub in table.groupby(["lag_h", "window_h", "weighting"], sort=True):
            cells = []
            for group in STABILITY_GROUPS:
                hit = sub[sub["group"] == group]
                cells.append(f"{hit['ratio'].iloc[0]:.3f} (n {hit['n'].iloc[0]}, z {hit['z'].iloc[0]:+.0f})" if len(hit) else "-")
            lines.append(f"| {lag} | {window} | {weighting} | " + " | ".join(cells) + " |")
    lines += ["", "## 전체 민원 기준 상위 3 조합", ""]
    for i, row in enumerate(top3, 1):
        lines.append(f"{i}. lag {row['lag_h']}h · window {row['window_h']}h · {row['weighting']} — 비 {row['ratio']:.3f} (z {row['z']:+.1f})")
    (OUTPUT_DIR / "association_sweep.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(top3, ensure_ascii=False))


if __name__ == "__main__":
    main()
