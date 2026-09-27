"""Build the shared farm and hourly weather tables."""
from pathlib import Path

import numpy as np
import pandas as pd

from develop import common
from species_weight_sets import EEA_ASSUMED, EEA_NH3

OUT = common.OUTPUT_DIR / "data"
WEATHER_COLUMNS = ["datetime", "station_id", "station_name", "source", "station_latitude", "station_longitude", "wind_direction", "wind_speed", "temperature", "humidity", "rainfall_hour"]


def species(raw):
    value = str(raw)
    if "돼지" in value:
        return "돼지"
    if any(x in value for x in ("계", "오리", "메추리", "가금", "부화")):
        return "가금"
    if any(x in value for x in ("한우", "젖소", "육우", "소")):
        return "소"
    return "기타"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ik = pd.read_csv(common.IKSAN_FARMS_FILE, encoding="utf-8-sig")
    gj = pd.read_csv(common.GIMJE_FARMS_FILE, encoding="utf-8-sig")
    a = pd.DataFrame({"farm_id": "IK-" + ik["순번"].astype(str), "city": "익산", "name": ik["업체명"], "lat": ik["lat"], "lon": ik["lon"], "coord_precision": ik["geocode_method"].map({"지번정확": "exact", "리중심": "approx", "첫필지": "approx", "실패": "none"}), "species_raw": ik["사육업종"], "heads": pd.to_numeric(ik["사육두수"], errors="coerce"), "status": ik["영업상태"]})
    b = pd.DataFrame({"farm_id": "GJ-" + gj["연번"].astype(str), "city": "김제", "name": gj["업체명"], "lat": pd.to_numeric(gj["위도"], errors="coerce"), "lon": pd.to_numeric(gj["경도"], errors="coerce"), "coord_precision": np.where(gj["위도"].notna() & gj["경도"].notna(), "approx", "none"), "species_raw": gj["사육업종"], "heads": np.nan, "status": np.nan})
    farms = pd.concat([a, b], ignore_index=True)
    farms.loc[farms[["lat", "lon"]].isna().any(axis=1), "coord_precision"] = "none"
    farms["species"] = farms["species_raw"].map(species)
    farms["eq"] = farms["heads"] * farms["species_raw"].map(EEA_NH3)
    farms["eq_assumed"] = farms["species_raw"].isin(EEA_ASSUMED)
    farms = farms[["farm_id", "city", "name", "lat", "lon", "coord_precision", "species_raw", "species", "heads", "eq", "eq_assumed", "status"]]
    farms.to_parquet(OUT / "farms.parquet", index=False)
    counts = farms.groupby(["city", "coord_precision"]).size().to_string()
    (OUT / "farms_report.md").write_text(f"# 농가 통합\n\n원본: `{common.IKSAN_FARMS_FILE.relative_to(common.ROOT)}`, `{common.GIMJE_FARMS_FILE.relative_to(common.ROOT)}`. 확인일 2026-09-27.\n\n행 수: 익산 {len(ik)}, 김제 {len(gj)}, 합계 {len(farms)}. 김제 좌표는 매칭 결과이므로 approx, heads/status는 결측으로 둠.\n\n```\n{counts}\n```\n", encoding="utf-8")
    parts = []
    for source, path in [("ASOS", common.ASOS_FILE), ("AWS", common.AWS_FILE)]:
        w = pd.read_csv(path, encoding="utf-8-sig")
        w["source"] = source
        w["datetime"] = pd.to_datetime(w["datetime"])
        parts.append(w[WEATHER_COLUMNS])
    weather = pd.concat(parts, ignore_index=True)
    weather.to_parquet(OUT / "weather_hourly.parquet", index=False)
    lines = ["# 시간별 기상 통합", "", "원본: `outputs/weather_integration/asos_hourly_2020_2026.csv`, `outputs/weather_integration/aws_hourly_2020_2026.csv`. 확인일 2026-09-27.", "", f"총 {len(weather)}행. 결측률은 관측 행 내 비율이며 누락된 시각은 별도 계산하지 않음.", "", "| source | station_id | station_name | 행 수 | 시작 | 끝 | 풍향 결측률 | 풍속 결측률 | 기온 결측률 | 습도 결측률 | 강수 결측률 |", "|---|---:|---|---:|---|---|---:|---:|---:|---:|---:|"]
    for (source, station), group in weather.groupby(["source", "station_id"]):
        miss = group[["wind_direction", "wind_speed", "temperature", "humidity", "rainfall_hour"]].isna().mean()
        lines.append(f"| {source} | {station} | {group.station_name.iloc[0]} | {len(group)} | {group.datetime.min()} | {group.datetime.max()} | " + " | ".join(f"{v:.1%}" for v in miss) + " |")
    (OUT / "weather_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"farms={len(farms)}, weather={len(weather)}")


if __name__ == "__main__":
    main()
