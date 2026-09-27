"""DEVELOP 공통 규약: 좌표·풍향·밤 날짜·가축 민원 로딩.

규약은 docs/contracts/develop_contract.md "공통 규약" 절과 같다. 모든 트랙이 import해서 쓰고,
변경은 양쪽 합의 후에만 한다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COMPLAINT_FILE = ROOT / "data" / "익산시 악취 민원 데이터_20190528-20260818.xlsx"
IKSAN_FARMS_FILE = ROOT / "v2" / "data_ext" / "farms_geocoded.csv"
GIMJE_FARMS_FILE = ROOT / "data" / "전북특별자치도 김제시_축산현황_20250515_geocoded.csv"
ASOS_FILE = ROOT / "outputs" / "weather_integration" / "asos_hourly_2020_2026.csv"
AWS_FILE = ROOT / "outputs" / "weather_integration" / "aws_hourly_2020_2026.csv"
OUTPUT_DIR = ROOT / "outputs" / "develop"

KM_PER_DEG_LAT = 110.57
KM_PER_DEG_LON = 111.32 * np.cos(np.radians(35.95))

NIGHT_HOURS = (21, 22, 23, 0, 1, 2, 3, 4)  # 21:00~04:59, 밤 날짜는 datetime - 6h
SEASON_MONTHS = (5, 6, 7, 8, 9, 10)


def to_xy(lat, lon):
    """위경도 → 평면 km 좌표(x=동, y=북)."""
    return np.asarray(lon, dtype=float) * KM_PER_DEG_LON, np.asarray(lat, dtype=float) * KM_PER_DEG_LAT


def distance_km(lat1, lon1, lat2, lon2):
    x1, y1 = to_xy(lat1, lon1)
    x2, y2 = to_xy(lat2, lon2)
    return np.hypot(x2 - x1, y2 - y1)


def bearing_deg(lat1, lon1, lat2, lon2):
    """지점1에서 지점2를 바라본 방위각(북=0°, 시계방향)."""
    x1, y1 = to_xy(lat1, lon1)
    x2, y2 = to_xy(lat2, lon2)
    return np.degrees(np.arctan2(x2 - x1, y2 - y1)) % 360.0


def angle_diff(a, b):
    """두 방위각의 최소 차이(0~180°)."""
    d = np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) % 360.0
    return np.minimum(d, 360.0 - d)


def downwind_vector(wind_from_deg):
    """풍향(불어오는 방향) → 바람이 가는 방향 단위벡터 (x=동, y=북)."""
    t = np.radians(np.asarray(wind_from_deg, dtype=float))
    return -np.sin(t), -np.cos(t)


def night_date(ts: pd.Series) -> pd.Series:
    """밤 날짜: datetime - 6h 의 날짜."""
    return (pd.to_datetime(ts) - pd.Timedelta(hours=6)).dt.normalize()


def in_night_window(ts: pd.Series) -> pd.Series:
    return pd.to_datetime(ts).dt.hour.isin(NIGHT_HOURS)


def load_livestock_complaints() -> pd.DataFrame:
    """익산시 가축 민원: datetime, latitude, longitude, night_date, in_night.

    기존 전처리(analyze_spatiotemporal_complaints)를 그대로 쓴다. 좌표 결측 행은 제외한다.
    """
    import analyze_spatiotemporal_complaints as asc

    raw, mapping = asc.load_data(COMPLAINT_FILE)
    df = asc.filter_target_region(asc.preprocess_data(raw, mapping), include_adjacent=False)
    df = df[df["odor_type"].astype(str).str.contains("가축")]
    df = df.dropna(subset=["datetime", "latitude", "longitude"])
    out = df[["datetime", "latitude", "longitude"]].copy()
    out["datetime"] = pd.to_datetime(out["datetime"])
    out["night_date"] = night_date(out["datetime"])
    out["in_night"] = in_night_window(out["datetime"])
    return out.sort_values("datetime").reset_index(drop=True)
