"""기존 산출물을 밤 단위 다시 보기 JSON으로 묶는다.

실행: python -m develop.event_link.replay
"""
from __future__ import annotations

import json
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from develop import common
from develop.field.engine import Wind, score_candidates, to_cards


ROOT = common.ROOT
OUT = ROOT / "outputs/develop/replay"
TRIGGER_RULE = "기존 사건 기준(1시간 안 10건·5칸 이상)"
NAMES = {"YONGJI": "용지 방면", "IK-0": "왕궁 방면", "IK-7": "춘포 방면"}


def scalar(value):
    """JSON에 NaN을 남기지 않는다."""
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def wind_at(weather: pd.DataFrame, stamp: pd.Timestamp) -> tuple[float | None, float | None]:
    # 시간별 관측이라 사건 시각을 정시로 내려 그 시각 관측을 쓴다(사건 시각 이전 자료).
    rows = weather.loc[weather["datetime"] == pd.Timestamp(stamp).floor("h")]
    if rows.empty:
        return None, None
    row = rows.iloc[0]
    return scalar(row["wind_direction"]), scalar(row["wind_speed"])


def night_wind(weather: pd.DataFrame, date: pd.Timestamp) -> dict:
    rows = weather.loc[
        (common.night_date(weather["datetime"]) == date)
        & common.in_night_window(weather["datetime"])
        & (weather["wind_speed"] >= 0.5)
    ].dropna(subset=["wind_direction", "wind_speed"])
    if rows.empty:
        return {"direction": None, "speed": None, "station_id": 702}
    radians = np.deg2rad(rows["wind_direction"].to_numpy(dtype=float))
    direction = float(np.rad2deg(np.arctan2(np.sin(radians).mean(), np.cos(radians).mean())) % 360)
    return {"direction": direction, "speed": float(rows["wind_speed"].mean()), "station_id": 702}


def choose_point(points: list[dict], direction: float | None) -> tuple[dict, str]:
    if direction is None:
        point = max(points, key=lambda p: (p["lift_all"], p["point_id"]))
        return point, "밤 평균 풍향 없음; lift_all이 가장 큰 주 대기 장소 선택"
    point = min(points, key=lambda p: (float(common.angle_diff(common.bearing_deg(35.948, 126.957, p["lat"], p["lon"]), direction)), p["point_id"]))
    delta = float(common.angle_diff(common.bearing_deg(35.948, 126.957, point["lat"], point["lon"]), direction))
    return point, f"밤 평균 풍향 {direction:.1f}°, 도시 중심에서 대기 장소 방위와 차이 {delta:.1f}°"


def overlap(complaints: pd.DataFrame, points: list[dict], direction: float | None, speed: float | None) -> dict:
    counts = []
    valid = direction is not None and speed is not None and speed >= 0.5
    for point in points:
        n = 0
        if valid and not complaints.empty:
            distance = common.distance_km(complaints["latitude"], complaints["longitude"], point["lat"], point["lon"])
            bearing = common.bearing_deg(complaints["latitude"], complaints["longitude"], point["lat"], point["lon"])
            n = int(((distance <= 15) & (common.angle_diff(bearing, direction) <= 45)).sum())
        counts.append({"point_id": point["point_id"], "n_pointing": n})
    suggested = None
    if valid:
        lift = {p["point_id"]: p["lift_all"] for p in points}
        suggested = max(counts, key=lambda x: (x["n_pointing"], lift[x["point_id"]], x["point_id"]))["point_id"]
    return {"wind_direction": direction, "wind_speed": speed, "n_complaints": len(complaints), "per_point": counts, "suggested_point_id": suggested}


def make_replays(events_source: str = "v2") -> tuple[list[dict], str]:
    if events_source not in ("v2", "legacy"):
        raise ValueError(events_source)
    base = ROOT / "outputs"
    predictions = pd.read_csv(base / ("develop/replay/v2_events.csv" if events_source == "v2" else "operational_grid_comparison/test_predictions.csv"))
    forecast = pd.read_csv(base / "develop/forecast/night_risk.csv")
    standby = pd.read_csv(base / "develop/standby/standby_points.csv")
    weather = pd.read_parquet(base / "develop/data/weather_hourly.parquet")
    farms = pd.read_parquet(base / "develop/data/farms.parquet")
    complaints = common.load_livestock_complaints()

    if events_source == "v2":
        predictions = predictions.loc[predictions["k"] == 5].copy()
        predictions["event_hour"] = pd.to_datetime(predictions["t0"]) + pd.Timedelta(minutes=30)
        predictions["night_date"] = pd.to_datetime(predictions["night_date"])
        departure = pd.to_datetime(pd.read_csv(base / "develop/replay/v2_events.csv").query("k == 7")["t0"])
    else:
        predictions["event_hour"] = pd.to_datetime(predictions["event_hour"])
        predictions["night_date"] = common.night_date(predictions["event_hour"])
        departure = pd.DatetimeIndex([])
    forecast = forecast.loc[forecast["model_id"] == "hgb_poisson"].copy()
    forecast["night_date"] = pd.to_datetime(forecast["night_date"])
    if forecast["night_date"].duplicated().any():
        raise ValueError("hgb_poisson night_date 중복")
    weather = weather.loc[(weather["source"] == "AWS") & (pd.to_numeric(weather["station_id"]) == 702)].copy()
    weather["datetime"] = pd.to_datetime(weather["datetime"])
    if weather["datetime"].duplicated().any():
        raise ValueError("AWS 702 datetime 중복")
    point_rows = standby.loc[(standby["wind_source"] == "AWS") & (standby["grade"] == "주")]
    points = [
        {"point_id": r.point_id, "cluster_id": r.cluster_id, "name": NAMES.get(r.cluster_id, r.cluster_id),
         "grade": r.grade, "lat": float(r.lat), "lon": float(r.lon), "lift_all": float(r.lift_all)}
        for r in point_rows.itertuples()
    ]
    if not points:
        raise ValueError("AWS 주 대기 장소 없음")
    forecast = forecast.set_index("night_date")
    predictions = predictions.loc[predictions["night_date"].isin(forecast.index)]
    index = []
    chosen_counts = Counter()
    suggested_total = suggested_same = alert_events = total_events = 0
    level_total, level_alert = Counter(), Counter()
    OUT.mkdir(parents=True, exist_ok=True)
    for date, night_rows in predictions.groupby("night_date", sort=True):
        risk = forecast.loc[date]
        nw = night_wind(weather, date)
        chosen, reason = choose_point(points, nw["direction"])
        chosen_counts[chosen["point_id"]] += 1
        events = []
        for stamp, rows in night_rows.groupby("event_hour", sort=True):
            input_start = stamp - pd.Timedelta(minutes=30) if events_source == "v2" else stamp
            first = complaints.loc[(complaints["datetime"] >= input_start) & (complaints["datetime"] < input_start + pd.Timedelta(minutes=30))]
            direction, speed = wind_at(weather, stamp)
            related = overlap(first, points, direction, speed)
            if related["suggested_point_id"] is not None:
                suggested_total += 1
                suggested_same += related["suggested_point_id"] == chosen["point_id"]
            if events_source == "v2":
                top3 = json.loads(rows.iloc[0]["top3"])
            else:
                top_rows = rows.sort_values("score", ascending=False).head(3)
                if len(top_rows) != 3:
                    raise ValueError(f"{stamp}: Top 3 행 부족")
                top3 = [
                    {"rank": rank, "region_name": r.region_name, "lat": float(r.center_latitude),
                     "lon": float(r.center_longitude), "score": float(r.score), "hit": bool(r.target == 1)}
                    for rank, r in enumerate(top_rows.itertuples(), 1)
                ]
            field_direction = direction if direction is not None else nw["direction"]
            field_speed = speed if speed is not None else nw["speed"]
            if field_direction is None or field_speed is None:
                field = {"staff_point_id": chosen["point_id"], "mode": None, "confidence": None,
                         "notes": ["사건 시각과 밤 평균 풍향·풍속이 없어 방문 순서 카드를 생성할 수 없음"], "cards": []}
            else:
                result = score_candidates(farms, chosen["lat"], chosen["lon"], Wind(field_direction, field_speed, 20.0), top_k=5)
                field = {"staff_point_id": chosen["point_id"], "mode": result.mode,
                         "confidence": result.confidence, "notes": result.notes, "cards": to_cards(result)}
            event_ids = rows["event_id"].dropna().unique()
            if len(event_ids) != 1:
                raise ValueError(f"{stamp}: event_id 유일하지 않음")
            trigger_start = pd.Timestamp(rows.iloc[0]["t0"]) if events_source == "v2" else None
            level = ("출동" if ((departure >= trigger_start) & (departure < trigger_start + pd.Timedelta(minutes=30))).any() else "확인") if events_source == "v2" else None
            events.append({"event_id": event_ids[0], "event_hour": stamp.isoformat(), "trigger_rule": "30분 안 서로 다른 장소 5곳(확인) / 7곳(출동)" if events_source == "v2" else TRIGGER_RULE,
                           "level": level,
                           "first30_complaints": [{"datetime": r.datetime.isoformat(), "lat": float(r.latitude), "lon": float(r.longitude)} for r in first.itertuples()],
                           "top3": top3, "direction_overlap": related, "field": field})
        alert = str(risk["alert"]).lower() in ("true", "1")
        replay = {"night_date": date.strftime("%Y-%m-%d"),
                  "afternoon": {"alert": alert, "season_rank": int(risk["season_rank"]),
                                "risk_score": scalar(risk["risk_score"]), "observed_complaints": scalar(risk["observed_complaints"]),
                                "model_id": risk["model_id"], "night_wind": nw},
                  "standby": {"chosen_point_id": chosen["point_id"], "reason": reason, "points": points}, "events": events}
        if events_source == "v2":
            (OUT / f"{replay['night_date']}.json").write_text(json.dumps(replay, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        index.append({"night_date": replay["night_date"], "alert": alert, "season_rank": int(risk["season_rank"]),
                      "n_events": len(events), "hit_any_top3": any(x["hit"] for event in events for x in event["top3"])})
        total_events += len(events)
        if alert:
            alert_events += len(events)
        for event in events:
            level_total[event["level"]] += 1
            if alert:
                level_alert[event["level"]] += 1
    if events_source == "v2":
        (OUT / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    chosen_text = ", ".join(f"{point_id} {count}회" for point_id, count in sorted(chosen_counts.items())) or "없음"
    ratio = f"{suggested_same}/{suggested_total} ({suggested_same / suggested_total:.1%})" if suggested_total else "산출 불가 (유효 풍향 사건 0건)"
    summary = (f"# 과거 하루 다시 보기 요약\n\n대상 밤 수: {len(index)}\n\n경보 난 밤 수: {sum(x['alert'] for x in index)}\n\n"
               f"경보 난 밤의 사건 수/전체 사건 수: {alert_events}/{total_events}\n\n선택된 대기 장소별 횟수: {chosen_text}\n\n"
               f"suggested와 chosen 일치 비율: {ratio}\n")
    if events_source == "v2":
        _, legacy = make_replays("legacy")
        level_text = "\n".join(f"{level}: {level_alert[level]}/{level_total[level]} ({level_alert[level] / level_total[level]:.1%})"
                               for level in ("확인", "출동") if level_total[level])
        summary = ("# 과거 하루 다시 보기 요약\n\n## v2 기준\n\n" + summary.split("\n\n", 1)[1]
                   + "\n경보 난 밤의 사건 비율(등급별):\n\n" + level_text + "\n\n"
                   + "## 옛 기준 (--events legacy)\n\n" + legacy.split("\n\n", 1)[1])
        (OUT / "summary.md").write_text(summary, encoding="utf-8")
    return index, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", choices=("v2", "legacy"), default="v2")
    args = parser.parse_args()
    _, summary_text = make_replays(args.events)
    print(summary_text)
