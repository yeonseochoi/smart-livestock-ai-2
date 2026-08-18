"""기상청 ASOS/AWS 자료를 수집해 악취 Event와 결합한다.

ASOS는 지정 기간 전체를 월 단위로 캐시하고, AWS 매분자료는 모델에 실제로
사용되는 Event의 초기 30분만 수집한다. API 호출 결과는 재실행 시 재사용한다.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

import analyze_spatiotemporal_complaints as base
import build_odor_ai_mvp as odor


API_ROOT = "https://apihub.kma.go.kr/api/typ01"
CACHE_DIR = Path("data/kma_weather_cache")
OUTPUT_DIR = Path("outputs/weather_integration")
REQUEST_TIMEOUT = 45
INVALID_THRESHOLD = -90.0


@dataclass(frozen=True)
class Station:
    station_type: str
    station_id: int
    longitude: float
    latitude: float
    name: str


def load_env_key(path: Path = Path(".env")) -> str:
    """환경변수 또는 .env에서 키를 읽되 값은 출력하지 않는다."""
    key = os.environ.get("KMA_API_KEY", "").strip()
    if not key and path.exists():
        for raw in path.read_text(encoding="utf-8-sig").splitlines():
            line = raw.strip()
            if line.startswith("KMA_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
                break
    if not key:
        raise RuntimeError("KMA_API_KEY가 없습니다. .env에 KMA_API_KEY=... 형식으로 입력하세요.")
    return key


def request_text(path: str, params: dict[str, object], api_key: str, retries: int = 4) -> str:
    safe_params = {**params, "authKey": api_key}
    url = f"{API_ROOT}/{path}"
    for attempt in range(retries):
        try:
            response = requests.get(url, params=safe_params, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            response.encoding = response.apparent_encoding or "utf-8"
            text = response.text
            if "ERROR" in text.upper() or "인증키" in text and "오류" in text:
                raise RuntimeError("기상청 API가 오류 응답을 반환했습니다.")
            return text
        except (requests.RequestException, RuntimeError):
            if attempt + 1 == retries:
                raise
            time.sleep(1.5 * (attempt + 1))
    raise AssertionError("unreachable")


def parse_stations(text: str, station_type: str) -> list[Station]:
    stations: list[Station] = []
    name_index = 10 if station_type == "ASOS" else 8
    for line in text.splitlines():
        if not re.match(r"^\s*\d+\s+\d", line):
            continue
        fields = line.split()
        try:
            stations.append(Station(
                station_type=station_type,
                station_id=int(fields[0]),
                longitude=float(fields[1]),
                latitude=float(fields[2]),
                name=fields[name_index],
            ))
        except (ValueError, IndexError):
            continue
    if not stations:
        raise RuntimeError(f"{station_type} 관측소 목록을 해석하지 못했습니다.")
    return stations


def fetch_station_list(api_key: str, station_type: str, when: pd.Timestamp) -> list[Station]:
    inf = "SFC" if station_type == "ASOS" else "AWS"
    cache = CACHE_DIR / "stations" / f"{station_type.lower()}_{when:%Y}.txt"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.exists():
        text = request_text(
            "url/stn_inf.php",
            {"inf": inf, "stn": "", "tm": when.strftime("%Y%m%d0000"), "help": 0},
            api_key,
        )
        cache.write_text(text, encoding="utf-8")
    return parse_stations(cache.read_text(encoding="utf-8"), station_type)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * odor.EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def nearest_stations(stations: Iterable[Station], lat: float, lon: float, count: int) -> list[tuple[Station, float]]:
    ranked = [(s, haversine_km(lat, lon, s.latitude, s.longitude)) for s in stations]
    return sorted(ranked, key=lambda item: item[1])[:count]


def month_ranges(start: pd.Timestamp, end: pd.Timestamp) -> Iterable[tuple[pd.Timestamp, pd.Timestamp]]:
    cursor = start.normalize().replace(day=1)
    while cursor <= end:
        month_end = cursor + pd.offsets.MonthEnd(0) + pd.Timedelta(hours=23)
        yield max(start, cursor), min(end, month_end)
        cursor = cursor + pd.offsets.MonthBegin(1)


def parse_number(value: object, *, wind: bool = False, rain: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    if number <= INVALID_THRESHOLD or (wind and number < 0):
        return np.nan
    if rain and number < 0:
        return 0.0
    return number


def parse_asos(text: str) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for line in text.splitlines():
        if not re.match(r"^\d{12}\s+\d+", line):
            continue
        f = line.split()
        if len(f) < 19:
            continue
        records.append({
            "datetime": pd.to_datetime(f[0], format="%Y%m%d%H%M"),
            "station_id": int(f[1]),
            "wind_direction": parse_number(f[2], wind=True) * 10.0,
            "wind_speed": parse_number(f[3], wind=True),
            "temperature": parse_number(f[11]),
            "humidity": parse_number(f[13]),
            "rainfall_hour": parse_number(f[15], rain=True),
            "rainfall_intensity": parse_number(f[18], rain=True),
        })
    return pd.DataFrame(records)


def fetch_asos_month(api_key: str, station: Station, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    cache = CACHE_DIR / "asos" / str(station.station_id) / f"{start:%Y%m}.txt"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.exists():
        text = request_text(
            "url/kma_sfctm3.php",
            {
                "tm1": start.strftime("%Y%m%d%H%M"),
                "tm2": end.strftime("%Y%m%d%H%M"),
                "stn": station.station_id,
                "help": 0,
            },
            api_key,
        )
        cache.write_text(text, encoding="utf-8")
    frame = parse_asos(cache.read_text(encoding="utf-8"))
    if not frame.empty:
        frame["station_name"] = station.name
        frame["station_latitude"] = station.latitude
        frame["station_longitude"] = station.longitude
    return frame


def parse_aws(text: str) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for line in text.splitlines():
        if not re.match(r"^\d{12},\d+", line):
            continue
        f = [x.strip() for x in line.split(",")]
        if len(f) < 18:
            continue
        records.append({
            "datetime": pd.to_datetime(f[0], format="%Y%m%d%H%M"),
            "station_id": int(f[1]),
            "wind_direction": parse_number(f[6], wind=True),
            "wind_speed": parse_number(f[7], wind=True),
            "temperature": parse_number(f[8]),
            "is_raining": parse_number(f[9], rain=True),
            "rainfall_15m": parse_number(f[10], rain=True),
            "rainfall_60m": parse_number(f[11], rain=True),
            "humidity": parse_number(f[14]),
        })
    return pd.DataFrame(records)


def fetch_aws_event_window(
    api_key: str, station: Station, event_hour: pd.Timestamp, input_minutes: int,
) -> pd.DataFrame:
    end = event_hour + pd.Timedelta(minutes=input_minutes - 1)
    cache = CACHE_DIR / "aws" / str(station.station_id) / f"{event_hour:%Y%m%d_%H%M}_{input_minutes}.txt"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.exists():
        text = request_text(
            "cgi-bin/url/nph-aws2_min",
            {
                "tm1": event_hour.strftime("%Y%m%d%H%M"),
                "tm2": end.strftime("%Y%m%d%H%M"),
                "stn": station.station_id,
                "disp": 1,
                "help": 0,
            },
            api_key,
        )
        cache.write_text(text, encoding="utf-8")
    frame = parse_aws(cache.read_text(encoding="utf-8"))
    if not frame.empty:
        frame["station_name"] = station.name
        frame["station_latitude"] = station.latitude
        frame["station_longitude"] = station.longitude
    return frame


def circular_wind(frame: pd.DataFrame) -> tuple[float, float, float]:
    valid = frame.dropna(subset=["wind_direction", "wind_speed"])
    if valid.empty:
        return np.nan, np.nan, np.nan
    radians = np.radians(valid["wind_direction"].to_numpy(float))
    weights = np.maximum(valid["wind_speed"].to_numpy(float), 0.1)
    from_sin = float(np.average(np.sin(radians), weights=weights))
    from_cos = float(np.average(np.cos(radians), weights=weights))
    direction = float((np.degrees(np.arctan2(from_sin, from_cos)) + 360) % 360)
    return direction, from_sin, from_cos


def aggregate_aws_station(frame: pd.DataFrame, station: Station, distance_km: float) -> dict[str, object]:
    direction, from_sin, from_cos = circular_wind(frame)
    return {
        "station_id": station.station_id,
        "station_name": station.name,
        "distance_km": distance_km,
        "samples": len(frame),
        "wind_direction": direction,
        "wind_from_sin": from_sin,
        "wind_from_cos": from_cos,
        "wind_speed": frame["wind_speed"].mean(),
        "wind_speed_max": frame["wind_speed"].max(),
        "rainfall_15m": frame["rainfall_15m"].max(),
        "rainfall_60m": frame["rainfall_60m"].max(),
        "raining_fraction": frame["is_raining"].mean(),
        "humidity": frame["humidity"].mean(),
        "temperature": frame["temperature"].mean(),
    }


def weighted_mean(records: list[dict[str, object]], key: str) -> float:
    values, weights = [], []
    for record in records:
        value = record.get(key)
        if value is None or pd.isna(value):
            continue
        values.append(float(value))
        weights.append(1.0 / max(float(record["distance_km"]), 1.0) ** 2)
    return float(np.average(values, weights=weights)) if values else np.nan


def combine_aws_records(records: list[dict[str, object]]) -> dict[str, object]:
    keys = [
        "wind_from_sin", "wind_from_cos", "wind_speed", "wind_speed_max",
        "rainfall_15m", "rainfall_60m", "raining_fraction", "humidity", "temperature",
    ]
    result = {f"aws_{key}": weighted_mean(records, key) for key in keys}
    sin_value, cos_value = result["aws_wind_from_sin"], result["aws_wind_from_cos"]
    result["aws_wind_direction"] = (
        float((np.degrees(np.arctan2(sin_value, cos_value)) + 360) % 360)
        if not pd.isna(sin_value) and not pd.isna(cos_value) else np.nan
    )
    result["aws_downwind_east"] = -sin_value if not pd.isna(sin_value) else np.nan
    result["aws_downwind_north"] = -cos_value if not pd.isna(cos_value) else np.nan
    result["aws_station_count"] = len(records)
    result["aws_station_ids"] = ":".join(str(x["station_id"]) for x in records)
    result["aws_station_names"] = ":".join(str(x["station_name"]) for x in records)
    return result


def event_table(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    raw, mapping = base.load_data()
    complaints = base.filter_target_region(base.preprocess_data(raw, mapping), include_adjacent=False)
    complaints, _ = odor.add_grid_columns(complaints)
    events, summary = odor.build_bounded_events(complaints)
    summary = summary[(summary["event_hour"] >= start) & (summary["event_hour"] <= end)].copy()
    centers = events.groupby("event_id").agg(
        event_latitude=("latitude", "mean"), event_longitude=("longitude", "mean")
    ).reset_index()
    return summary.merge(centers, on="event_id", how="left")


def attach_asos(events: pd.DataFrame, asos: pd.DataFrame, stations: list[Station]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    by_station = {sid: group.sort_values("datetime") for sid, group in asos.groupby("station_id")}
    for event in events.itertuples(index=False):
        records: list[dict[str, object]] = []
        for station, distance in nearest_stations(stations, event.event_latitude, event.event_longitude, 2):
            frame = by_station.get(station.station_id)
            if frame is None or frame.empty:
                continue
            idx = (frame["datetime"] - event.event_hour).abs().idxmin()
            item = frame.loc[idx]
            if abs(item["datetime"] - event.event_hour) > pd.Timedelta(hours=1):
                continue
            records.append({
                "station_id": station.station_id,
                "station_name": station.name,
                "distance_km": distance,
                "wind_direction": item["wind_direction"],
                "wind_speed": item["wind_speed"],
                "humidity": item["humidity"],
                "temperature": item["temperature"],
                "rainfall_hour": item["rainfall_hour"],
            })
        direction_records = []
        for record in records:
            if not pd.isna(record["wind_direction"]):
                radians = math.radians(float(record["wind_direction"]))
                direction_records.append({**record, "sin": math.sin(radians), "cos": math.cos(radians)})
        sin_value = weighted_mean(direction_records, "sin")
        cos_value = weighted_mean(direction_records, "cos")
        rows.append({
            "event_id": event.event_id,
            "asos_wind_direction": (
                float((np.degrees(np.arctan2(sin_value, cos_value)) + 360) % 360)
                if not pd.isna(sin_value) and not pd.isna(cos_value) else np.nan
            ),
            "asos_wind_speed": weighted_mean(records, "wind_speed"),
            "asos_humidity": weighted_mean(records, "humidity"),
            "asos_temperature": weighted_mean(records, "temperature"),
            "asos_rainfall_hour": weighted_mean(records, "rainfall_hour"),
            "asos_station_ids": ":".join(str(x["station_id"]) for x in records),
            "asos_station_names": ":".join(str(x["station_name"]) for x in records),
        })
    return events.merge(pd.DataFrame(rows), on="event_id", how="left")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2026-07-30 23:59")
    parser.add_argument("--aws-stations", type=int, default=3)
    parser.add_argument("--asos-stations", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    api_key = load_env_key()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/5] 악취 Event 구성")
    events = event_table(start, end)
    print(f"      대상 Event: {len(events)}개")

    print("[2/5] 연도별 관측소 자동 탐색")
    years = range(start.year, end.year + 1)
    aws_by_year = {year: fetch_station_list(api_key, "AWS", pd.Timestamp(year=year, month=7, day=1)) for year in years}
    asos_stations = fetch_station_list(api_key, "ASOS", end)
    center_lat, center_lon = events["event_latitude"].mean(), events["event_longitude"].mean()
    selected_asos = [x[0] for x in nearest_stations(asos_stations, center_lat, center_lon, args.asos_stations)]
    print("      ASOS: " + ", ".join(f"{s.name}({s.station_id})" for s in selected_asos))

    print("[3/5] ASOS 월별 시간자료 수집")
    asos_frames: list[pd.DataFrame] = []
    jobs = [(s, left, right) for s in selected_asos for left, right in month_ranges(start, end)]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_asos_month, api_key, *job): job for job in jobs}
        for index, future in enumerate(as_completed(futures), 1):
            frame = future.result()
            if not frame.empty:
                asos_frames.append(frame)
            if index % 25 == 0 or index == len(jobs):
                print(f"      ASOS {index}/{len(jobs)} 구간 완료")
    asos = pd.concat(asos_frames, ignore_index=True).drop_duplicates(["datetime", "station_id"])
    asos.to_csv(OUTPUT_DIR / "asos_hourly_2020_2026.csv", index=False, encoding="utf-8-sig")
    combined = attach_asos(events, asos, selected_asos)

    print("[4/5] Event 초기 30분 AWS 매분자료 수집")
    job_meta: list[tuple[str, pd.Timestamp, Station, float]] = []
    for event in events.itertuples(index=False):
        candidates = nearest_stations(
            aws_by_year[event.event_hour.year], event.event_latitude, event.event_longitude, args.aws_stations
        )
        job_meta.extend((event.event_id, event.event_hour, station, distance) for station, distance in candidates)
    station_records: dict[str, list[dict[str, object]]] = {event_id: [] for event_id in events["event_id"]}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(fetch_aws_event_window, api_key, station, hour, odor.INPUT_MINUTES):
            (event_id, station, distance)
            for event_id, hour, station, distance in job_meta
        }
        for index, future in enumerate(as_completed(futures), 1):
            event_id, station, distance = futures[future]
            frame = future.result()
            if not frame.empty:
                station_records[event_id].append(aggregate_aws_station(frame, station, distance))
            if index % 50 == 0 or index == len(job_meta):
                print(f"      AWS {index}/{len(job_meta)} Event-지점 구간 완료")

    aws_rows = [{"event_id": event_id, **combine_aws_records(records)} for event_id, records in station_records.items()]
    aws_features = pd.DataFrame(aws_rows)
    combined = combined.merge(aws_features, on="event_id", how="left")

    print("[5/5] Event 결합 결과 저장")
    combined.to_csv(OUTPUT_DIR / "event_weather_features.csv", index=False, encoding="utf-8-sig")
    station_details = {
        "period": {"start": str(start), "end": str(end)},
        "asos": [asdict(s) for s in selected_asos],
        "aws_by_event": {
            event_id: records for event_id, records in station_records.items()
        },
    }
    (OUTPUT_DIR / "station_selection.json").write_text(
        json.dumps(station_details, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    coverage = {
        "events": len(combined),
        "asos_wind_coverage": float(combined["asos_wind_speed"].notna().mean()),
        "asos_humidity_coverage": float(combined["asos_humidity"].notna().mean()),
        "aws_wind_coverage": float(combined["aws_wind_speed"].notna().mean()),
        "aws_humidity_coverage": float(combined["aws_humidity"].notna().mean()),
        "aws_average_station_count": float(combined["aws_station_count"].mean()),
    }
    (OUTPUT_DIR / "collection_summary.json").write_text(
        json.dumps(coverage, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(coverage, ensure_ascii=False, indent=2))
    print(f"산출물: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
