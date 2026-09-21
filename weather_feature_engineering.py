"""기상 관측을 후보 격자별 이동·확산 특징으로 변환한다."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


WEATHER_PATH = Path("outputs/weather_integration/event_weather_features.csv")
WINDOWS = (30, 60, 120)
ADVECTION_FACTOR = 0.25


def weather_feature_names() -> list[str]:
    names: list[str] = []
    for minutes in WINDOWS:
        suffix = f"w{minutes}"
        names.extend([
            f"wind_alignment_{suffix}",
            f"alongwind_grids_{suffix}",
            f"crosswind_grids_{suffix}",
            f"wind_reliability_{suffix}",
            f"alignment_speed_{suffix}",
            f"downwind_decay_{suffix}",
            f"transport_match_{suffix}",
        ])
    names.extend([
        "wind_turn_30_60", "wind_turn_60_120", "wind_speed_change_30_120",
        "calm_w30", "humid_calm_w30", "raining_w30",
    ])
    return names


WEATHER_FEATURES = weather_feature_names()
TIMING_WINDOWS = ("pre30", "pre60", "initial30")


def _timing_spatial_names(prefix: str) -> list[str]:
    return [
        f"timing_alignment_{prefix}", f"timing_alongwind_{prefix}",
        f"timing_crosswind_{prefix}", f"timing_reliability_{prefix}",
        f"timing_alignment_speed_{prefix}", f"timing_downwind_decay_{prefix}",
        f"timing_transport_match_{prefix}",
    ]


def timing_feature_groups() -> dict[str, list[str]]:
    initial = _timing_spatial_names("initial30") + [
        "timing_calm_initial30", "timing_humid_calm_initial30", "timing_raining_initial30",
    ]
    pre30 = _timing_spatial_names("pre30")
    pre60 = _timing_spatial_names("pre60")
    change30 = ["timing_direction_match_pre30_initial30", "timing_speed_change_pre30_initial30"]
    change60 = ["timing_direction_match_pre60_initial30", "timing_speed_change_pre60_initial30"]
    return {
        "initial30_only": initial,
        "pre30_only": pre30,
        "pre60_only": pre60,
        "pre30_plus_initial30": pre30 + initial + change30,
        "pre60_plus_initial30": pre60 + initial + change60,
        "pre30_pre60_initial30": pre30 + pre60 + initial + change30 + change60,
    }


def _required_weather_columns() -> list[str]:
    columns = ["event_id"]
    for minutes in WINDOWS:
        prefix = f"aws_w{minutes}"
        columns.extend([
            f"{prefix}_downwind_east", f"{prefix}_downwind_north",
            f"{prefix}_wind_speed", f"{prefix}_wind_consistency",
        ])
    columns.extend([
        "aws_w30_humidity", "aws_w30_raining_fraction",
    ])
    return columns


def load_weather(path: Path = WEATHER_PATH) -> pd.DataFrame:
    weather = pd.read_csv(path, encoding="utf-8-sig")
    missing = sorted(set(_required_weather_columns()) - set(weather.columns))
    if missing:
        raise ValueError(
            "30·60·120분 기상 특징이 없습니다. fetch_kma_weather.py를 다시 실행하세요. "
            f"누락 열: {', '.join(missing)}"
        )
    return weather[_required_weather_columns()].drop_duplicates("event_id")


def load_timing_weather(path: Path = WEATHER_PATH) -> pd.DataFrame:
    weather = pd.read_csv(path, encoding="utf-8-sig")
    columns = ["event_id"]
    for prefix in TIMING_WINDOWS:
        raw = f"aws_{prefix}"
        columns.extend([
            f"{raw}_downwind_east", f"{raw}_downwind_north",
            f"{raw}_wind_speed", f"{raw}_wind_consistency",
            f"{raw}_humidity", f"{raw}_raining_fraction",
        ])
    missing = sorted(set(columns) - set(weather.columns))
    if missing:
        raise ValueError(
            "이벤트 전·후 분리 기상 특징이 없습니다. fetch_kma_weather.py를 다시 실행하세요. "
            f"누락 열: {', '.join(missing)}"
        )
    return weather[columns].drop_duplicates("event_id")


def _safe_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default).astype(float)


def add_candidate_weather_features(
    candidates: pd.DataFrame, weather: pd.DataFrame, grid_m: int,
) -> pd.DataFrame:
    """이벤트 공통 기상을 후보 격자 방향·거리와 결합해 순위 특징을 만든다."""
    required = {"event_id", "grid_x", "grid_y", "initial_centroid_x", "initial_centroid_y"}
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise ValueError(f"후보 데이터 누락 열: {', '.join(missing)}")

    frame = candidates.merge(weather, on="event_id", how="left", validate="many_to_one")
    dx_m = (frame["grid_x"] - frame["initial_centroid_x"]) * grid_m
    dy_m = (frame["grid_y"] - frame["initial_centroid_y"]) * grid_m
    distance_m = np.hypot(dx_m, dy_m)
    safe_distance = np.maximum(distance_m, 1.0)

    for minutes in WINDOWS:
        raw = f"aws_w{minutes}"
        suffix = f"w{minutes}"
        east = _safe_numeric(frame[f"{raw}_downwind_east"])
        north = _safe_numeric(frame[f"{raw}_downwind_north"])
        speed = _safe_numeric(frame[f"{raw}_wind_speed"])
        consistency = _safe_numeric(frame[f"{raw}_wind_consistency"]).clip(0.0, 1.0)

        along = dx_m * east + dy_m * north
        cross = np.abs(dx_m * north - dy_m * east)
        alignment = np.divide(along, safe_distance)
        reliability = consistency * np.minimum(speed / 1.5, 1.0)
        expected = ADVECTION_FACTOR * speed * minutes * 60.0
        sigma_along = np.maximum(float(grid_m), expected * 0.75)
        sigma_cross = max(float(grid_m), 1.0)
        transport = np.exp(
            -0.5 * ((along - expected) / sigma_along) ** 2
            -0.5 * (cross / sigma_cross) ** 2
        )
        transport = np.where(along >= 0, transport, transport * 0.1)

        frame[f"wind_alignment_{suffix}"] = alignment
        frame[f"alongwind_grids_{suffix}"] = along / grid_m
        frame[f"crosswind_grids_{suffix}"] = cross / grid_m
        frame[f"wind_reliability_{suffix}"] = reliability
        frame[f"alignment_speed_{suffix}"] = alignment * speed * consistency
        frame[f"downwind_decay_{suffix}"] = (
            np.maximum(alignment, 0.0) * reliability / (1.0 + distance_m / grid_m)
        )
        frame[f"transport_match_{suffix}"] = transport * reliability

    def direction_similarity(left: int, right: int) -> pd.Series:
        return (
            _safe_numeric(frame[f"aws_w{left}_downwind_east"])
            * _safe_numeric(frame[f"aws_w{right}_downwind_east"])
            + _safe_numeric(frame[f"aws_w{left}_downwind_north"])
            * _safe_numeric(frame[f"aws_w{right}_downwind_north"])
        ).clip(-1.0, 1.0)

    frame["wind_turn_30_60"] = direction_similarity(30, 60)
    frame["wind_turn_60_120"] = direction_similarity(60, 120)
    frame["wind_speed_change_30_120"] = (
        _safe_numeric(frame["aws_w30_wind_speed"]) - _safe_numeric(frame["aws_w120_wind_speed"])
    )
    speed30 = _safe_numeric(frame["aws_w30_wind_speed"])
    humidity30 = _safe_numeric(frame["aws_w30_humidity"])
    frame["calm_w30"] = (speed30 < 0.5).astype(float)
    frame["humid_calm_w30"] = frame["calm_w30"] * np.maximum(humidity30 - 70.0, 0.0) / 30.0
    frame["raining_w30"] = (_safe_numeric(frame["aws_w30_raining_fraction"]) > 0).astype(float)

    frame[WEATHER_FEATURES] = frame[WEATHER_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return frame


def add_timing_weather_features(
    candidates: pd.DataFrame, weather: pd.DataFrame, grid_m: int,
) -> pd.DataFrame:
    """event 이전 창과 초기 관측 창을 분리한 후보 격자 특징을 만든다."""
    frame = candidates.merge(weather, on="event_id", how="left", validate="many_to_one")
    dx_m = (frame["grid_x"] - frame["initial_centroid_x"]) * grid_m
    dy_m = (frame["grid_y"] - frame["initial_centroid_y"]) * grid_m
    distance_m = np.hypot(dx_m, dy_m)
    safe_distance = np.maximum(distance_m, 1.0)
    durations = {"pre30": 30, "pre60": 60, "initial30": 30}

    for prefix in TIMING_WINDOWS:
        raw = f"aws_{prefix}"
        east = _safe_numeric(frame[f"{raw}_downwind_east"])
        north = _safe_numeric(frame[f"{raw}_downwind_north"])
        speed = _safe_numeric(frame[f"{raw}_wind_speed"])
        consistency = _safe_numeric(frame[f"{raw}_wind_consistency"]).clip(0.0, 1.0)
        along = dx_m * east + dy_m * north
        cross = np.abs(dx_m * north - dy_m * east)
        alignment = np.divide(along, safe_distance)
        reliability = consistency * np.minimum(speed / 1.5, 1.0)
        expected = ADVECTION_FACTOR * speed * durations[prefix] * 60.0
        sigma_along = np.maximum(float(grid_m), expected * 0.75)
        transport = np.exp(
            -0.5 * ((along - expected) / sigma_along) ** 2
            -0.5 * (cross / max(float(grid_m), 1.0)) ** 2
        )
        transport = np.where(along >= 0, transport, transport * 0.1)

        values = [
            alignment, along / grid_m, cross / grid_m, reliability,
            alignment * speed * consistency,
            np.maximum(alignment, 0.0) * reliability / (1.0 + distance_m / grid_m),
            transport * reliability,
        ]
        frame[_timing_spatial_names(prefix)] = np.column_stack(values)

    def direction_match(left: str, right: str) -> pd.Series:
        return (
            _safe_numeric(frame[f"aws_{left}_downwind_east"])
            * _safe_numeric(frame[f"aws_{right}_downwind_east"])
            + _safe_numeric(frame[f"aws_{left}_downwind_north"])
            * _safe_numeric(frame[f"aws_{right}_downwind_north"])
        ).clip(-1.0, 1.0)

    frame["timing_direction_match_pre30_initial30"] = direction_match("pre30", "initial30")
    frame["timing_direction_match_pre60_initial30"] = direction_match("pre60", "initial30")
    initial_speed = _safe_numeric(frame["aws_initial30_wind_speed"])
    frame["timing_speed_change_pre30_initial30"] = (
        initial_speed - _safe_numeric(frame["aws_pre30_wind_speed"])
    )
    frame["timing_speed_change_pre60_initial30"] = (
        initial_speed - _safe_numeric(frame["aws_pre60_wind_speed"])
    )
    frame["timing_calm_initial30"] = (initial_speed < 0.5).astype(float)
    humidity = _safe_numeric(frame["aws_initial30_humidity"])
    frame["timing_humid_calm_initial30"] = (
        frame["timing_calm_initial30"] * np.maximum(humidity - 70.0, 0.0) / 30.0
    )
    frame["timing_raining_initial30"] = (
        _safe_numeric(frame["aws_initial30_raining_fraction"]) > 0
    ).astype(float)

    all_features = sorted({name for names in timing_feature_groups().values() for name in names})
    frame[all_features] = frame[all_features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return frame
