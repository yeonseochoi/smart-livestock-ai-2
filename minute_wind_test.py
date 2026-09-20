"""분 단위 참조 바람 창 검정(W3): 익산 AWS 702 분 자료 × 민원 신고 분(minute).

질문: 시간 자료에서는 "정시 바람"이 최선이었다. 분 자료로 보면 민원 직전 몇 분의 바람이 발생원과 가장 잘 맞나?
방법: test_wind_source_association 의 통계량(민원 상풍측 ±30° 반경 6 km 축산 배출 가중치 합, 월×6시간대 층화 셔플)을
      그대로 두고, 참조 바람만 "민원 시각 t 기준 (t-lag-width, t-lag] 분의 벡터 평균"으로 바꿔 비교한다.
      지점은 익산 702 한 곳(분 자료가 이 지점만 있음). 풍속(창 평균) ≥ 1 m/s 민원만.
자료: data/kma_aws_minute/aws702_minute_*.zip (기상자료개방포털 파일셋, 2020-01~2026-09, 3,505,792행)

실행: python minute_wind_test.py [--permutations 300] → outputs/wind_lag_sweep/minute_window_test.{json,md}
"""
from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import species_weight_sets as sw
import test_wind_source_association as t

MINUTE_DIR = Path("data/kma_aws_minute")
MINUTE_CSV = Path("outputs/weather_integration/aws702_minute_2020_2026.csv")
OUTPUT_DIR = Path("outputs/wind_lag_sweep")
RADIUS_KM = 6.0
# (lag_min, width_min): 민원 시각 t 기준 (t-lag-width, t-lag] 평균. width=1 은 그 분의 값.
WINDOWS = [(0, 1), (0, 10), (0, 20), (0, 30), (0, 45), (0, 60), (0, 90), (0, 120),
           (10, 10), (20, 10), (30, 10), (45, 15), (60, 10), (90, 30), (120, 30),
           (10, 20), (20, 20), (30, 30), (60, 60)]
HOURLY_LABEL = "정시 10분 평균(시간 자료와 동일)"


def build_minute_series() -> pd.DataFrame:
    if MINUTE_CSV.exists():
        return pd.read_csv(MINUTE_CSV, parse_dates=["datetime"]).set_index("datetime")
    parts = []
    for path in sorted(MINUTE_DIR.glob("aws702_minute_*.zip")):
        with zipfile.ZipFile(path) as outer:
            for name in outer.namelist():
                with zipfile.ZipFile(io.BytesIO(outer.read(name))) as inner:
                    for member in inner.namelist():
                        frame = pd.read_csv(io.BytesIO(inner.read(member)), encoding="cp949")
                        frame = frame.rename(columns={"일시": "datetime", "풍향(deg)": "wind_direction", "풍속(m/s)": "wind_speed"})
                        parts.append(frame[["datetime", "wind_direction", "wind_speed"]])
    minute = pd.concat(parts, ignore_index=True)
    minute["datetime"] = pd.to_datetime(minute["datetime"])
    minute = minute.drop_duplicates("datetime").sort_values("datetime").set_index("datetime")
    for col in ("wind_direction", "wind_speed"):
        minute[col] = pd.to_numeric(minute[col], errors="coerce")
    minute = minute.reindex(pd.date_range(minute.index.min(), minute.index.max(), freq="min"))
    minute.index.name = "datetime"
    MINUTE_CSV.parent.mkdir(parents=True, exist_ok=True)
    minute.to_csv(MINUTE_CSV, encoding="utf-8-sig")
    return minute


def window_wind(minute: pd.DataFrame, lag: int, width: int) -> pd.DataFrame:
    rad = np.radians(minute["wind_direction"])
    u = -minute["wind_speed"] * np.sin(rad)
    v = -minute["wind_speed"] * np.cos(rad)
    frame = pd.DataFrame({"u": u, "v": v, "speed": minute["wind_speed"]})
    rolled = frame.rolling(width, min_periods=max(1, width // 2)).mean().shift(lag)
    out = pd.DataFrame({"from_deg": (np.degrees(np.arctan2(-rolled["u"], -rolled["v"])) + 360.0) % 360.0,
                        "speed": rolled["speed"]})
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--permutations", type=int, default=300)
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t.PERMUTATIONS = args.permutations
    t.RADIUS_KM = RADIUS_KM
    rng = np.random.default_rng(t.SEED)

    complaints = t.load_complaints()
    complaints["minute"] = complaints["datetime"].dt.floor("min")
    complaints["is_sewage"] = complaints["odor_type"].astype(str).str.startswith("하수")
    sources = pd.read_csv(t.SOURCES_PATH, encoding="utf-8-sig").dropna(subset=["latitude", "longitude"])
    livestock = sw.apply_weight_set(sources[sources["source_type"].eq("livestock")] if "source_type" in sources else sources, "eea_nh3")
    livestock = livestock[livestock["emission_weight"] > 0]
    bins_all = t.bearing_bins(complaints, livestock)
    minute = build_minute_series()
    print(f"분 자료 {len(minute):,}행 {minute.index.min()}~{minute.index.max()}, 풍향 결측 {minute['wind_direction'].isna().mean():.3f}", flush=True)

    specs = [("hourly", None)] + [(f"lag{lag}_w{width}", (lag, width)) for lag, width in WINDOWS]
    rows = []
    for label, spec in specs:
        if spec is None:
            # 시간 자료와 같은 정의: 민원 정시(h)의 10분 평균 = (h-10, h] 분
            ref = window_wind(minute, 0, 10)
            joined = complaints.join(ref, on="hour", how="inner")
        else:
            ref = window_wind(minute, *spec)
            joined = complaints.join(ref, on="minute", how="inner")
        joined = joined.dropna(subset=["from_deg", "speed"])
        joined = joined[joined["speed"] >= t.MIN_WIND]
        for gname, mask in (("all", None), ("livestock", "is_livestock"), ("factory", "is_factory")):
            sub = joined if mask is None else joined[joined[mask]]
            bins = bins_all[sub.index.to_numpy()]
            r = t.run_test(sub.reset_index(drop=True), bins, rng)
            rows.append({"window": label, "group": gname, "n": r["n"], "ratio": r["ratio_real_over_shuffled"], "z": r["z"], "p": r["p_value_one_sided"]})
        a = [x for x in rows if x["window"] == label]
        print(f"{label:12s} all {a[0]['ratio']:.3f} (z {a[0]['z']:+.1f}, n {a[0]['n']}) | livestock {a[1]['ratio']:.3f} (z {a[1]['z']:+.1f}) | factory {a[2]['ratio']:.3f} (z {a[2]['z']:+.1f})", flush=True)

    (OUTPUT_DIR / "minute_window_test.json").write_text(json.dumps({"radius_km": RADIUS_KM, "permutations": args.permutations, "station": 702, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    table = pd.DataFrame(rows)
    lines = ["# 분 단위 참조 바람 창 검정 (익산 AWS 702 분 자료 × 민원 신고 분)", "",
             f"통계량 = 민원 상풍측 ±{t.SECTOR_DEG:.0f}° 반경 {RADIUS_KM:.0f} km 축산 배출 가중치(EEA NH3) 합, 실제 ÷ 층화 셔플 평균({args.permutations}회). 창 평균 풍속 ≥1 m/s 민원만.",
             "창 표기: lagL_wW = 민원 시각 t 기준 (t−L−W, t−L] 분 평균. hourly = 민원 정시의 10분 평균(시간 자료 정의와 같음).", "",
             "| 창 | 전체 비율 (z, n) | 가축 냄새 | 공장 냄새 |", "|---|---|---|---|"]
    for label, _ in specs:
        part = table[table["window"] == label].set_index("group")
        lines.append(f"| {label} | {part.loc['all','ratio']:.3f} (z {part.loc['all','z']:+.0f}, n {int(part.loc['all','n']):,}) | "
                     f"{part.loc['livestock','ratio']:.3f} (z {part.loc['livestock','z']:+.0f}) | {part.loc['factory','ratio']:.3f} (z {part.loc['factory','z']:+.0f}) |")
    (OUTPUT_DIR / "minute_window_test.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
