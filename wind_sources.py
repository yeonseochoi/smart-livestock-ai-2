"""바람 관측 자료 통합: ASOS(종관, 전주 146·군산 140) + AWS(방재, 익산 702·함라 733·여산 763·김제 737·진봉 736).

왜 둘을 같이 쓰나
  ASOS 두 지점은 익산에서 18~19 km 떨어져 있지만 2020~2026 전 기간이 빠짐없이 있고 운량·일사(안정도 재료)가 있다.
  AWS는 익산 시내(702, 시청 3.3 km)와 북부 읍면(함라·여산)에 붙어 있어 국지 바람을 보지만 결측·오류 플래그가 있다.
  그래서 "가까운 지점을 우선하되 비면 먼 지점으로 메운다"는 거리 가중(IDW, 역거리 제곱)으로 위치별 바람을 만든다.

자료 출처
  ASOS: outputs/weather_integration/asos_hourly_2020_2026.csv (기상청 API 허브 kma_sfctm2)
  AWS : data/kma_aws_hourly/aws_fileset_2020_2025.zip (기상자료개방포털 파일셋, 지점·연도별 zip 30개, 2020~2025)
        data/kma_aws_hourly/aws_hourly_2026_01_08.csv   (같은 포털 '자료' 탭 조회, 2026-01-01~08-31)
  AWS 지점 좌표: API 허브 stn_inf.php?inf=AWS (2026-07 기준)

사용
  python wind_sources.py            → outputs/weather_integration/aws_hourly_2020_2026.csv 생성 + 결측 요약
  from wind_sources import load_wind_stations, WindField
  field = WindField(load_wind_stations("both"))      # "asos" | "aws" | "both"
  table = field.at(lat, lon)                          # 시각별 from_deg·speed (해당 위치 IDW)
  table = field.center_table((lat0, lon0))            # 기존 방식(중심점 고정 가중)과 동일 결과
"""
from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
ASOS_PATH = ROOT / "outputs" / "weather_integration" / "asos_hourly_2020_2026.csv"
AWS_PATH = ROOT / "outputs" / "weather_integration" / "aws_hourly_2020_2026.csv"
AWS_RAW_DIR = ROOT / "data" / "kma_aws_hourly"
AWS_FILESET = AWS_RAW_DIR / "aws_fileset_2020_2025.zip"
AWS_2026 = AWS_RAW_DIR / "aws_hourly_2026_01_08.csv"

# API 허브 stn_inf(inf=AWS, 2026-07) 기준. 지점 이설 이력은 미확인.
AWS_STATIONS = {
    702: ("익산", 35.93807, 126.99252),
    733: ("함라", 36.04610, 126.89170),
    763: ("여산", 36.05920, 127.06200),
    737: ("김제", 35.80905, 126.87765),
    736: ("진봉", 35.84679, 126.78401),
}
COLUMN_MAP = {"지점": "station_id", "일시": "datetime", "기온(°C)": "temperature", "풍향(deg)": "wind_direction",
              "풍속(m/s)": "wind_speed", "강수량(mm)": "rainfall_hour", "습도(%)": "humidity"}
IDW_POWER = 2.0
MIN_DISTANCE_KM = 1.0  # 관측소 바로 위에서 가중치가 발산하지 않게


def _read_aws_csv(raw: bytes) -> pd.DataFrame:
    frame = pd.read_csv(io.BytesIO(raw), encoding="cp949")
    frame = frame.rename(columns={c: COLUMN_MAP[c] for c in frame.columns if c in COLUMN_MAP})
    keep = [c for c in ("station_id", "datetime", "temperature", "wind_direction", "wind_speed", "rainfall_hour", "humidity") if c in frame.columns]
    return frame[keep]


def build_aws_hourly() -> pd.DataFrame:
    """포털 원본(zip 안의 zip, cp949)을 ASOS 파일과 같은 스키마로 합친다."""
    parts = []
    with zipfile.ZipFile(AWS_FILESET) as outer:
        for name in outer.namelist():
            with zipfile.ZipFile(io.BytesIO(outer.read(name))) as inner:
                for member in inner.namelist():
                    parts.append(_read_aws_csv(inner.read(member)))
    if AWS_2026.exists():
        parts.append(_read_aws_csv(AWS_2026.read_bytes()))
    aws = pd.concat(parts, ignore_index=True)
    aws["datetime"] = pd.to_datetime(aws["datetime"])
    aws["station_id"] = aws["station_id"].astype(int)
    for col in ("temperature", "wind_direction", "wind_speed", "rainfall_hour", "humidity"):
        aws[col] = pd.to_numeric(aws.get(col), errors="coerce")
    aws = aws.drop_duplicates(["datetime", "station_id"]).sort_values(["datetime", "station_id"])
    aws["rainfall_intensity"] = np.nan
    aws["station_name"] = aws["station_id"].map(lambda s: AWS_STATIONS[s][0])
    aws["station_latitude"] = aws["station_id"].map(lambda s: AWS_STATIONS[s][1])
    aws["station_longitude"] = aws["station_id"].map(lambda s: AWS_STATIONS[s][2])
    columns = ["datetime", "station_id", "wind_direction", "wind_speed", "temperature", "humidity", "rainfall_hour",
               "rainfall_intensity", "station_name", "station_latitude", "station_longitude"]
    aws = aws[columns]
    AWS_PATH.parent.mkdir(parents=True, exist_ok=True)
    aws.to_csv(AWS_PATH, index=False, encoding="utf-8-sig")
    return aws


def load_wind_stations(source: str = "both") -> pd.DataFrame:
    """source = asos | aws | both. u/v(불어가는 방향 벡터)와 station_type을 붙여 돌려준다."""
    frames = []
    if source in ("asos", "both"):
        asos = pd.read_csv(ASOS_PATH, parse_dates=["datetime"], encoding="utf-8-sig")
        asos["station_type"] = "ASOS"
        frames.append(asos)
    if source in ("aws", "both"):
        if not AWS_PATH.exists():
            build_aws_hourly()
        aws = pd.read_csv(AWS_PATH, parse_dates=["datetime"], encoding="utf-8-sig")
        aws["station_type"] = "AWS"
        frames.append(aws)
    if not frames:
        raise ValueError("source 는 asos | aws | both")
    wind = pd.concat(frames, ignore_index=True)
    rad = np.radians(wind["wind_direction"])
    wind["u"] = -wind["wind_speed"] * np.sin(rad)
    wind["v"] = -wind["wind_speed"] * np.cos(rad)
    return wind


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


class WindField:
    """시각 × 관측소 행렬. 위치가 주어지면 역거리 제곱 가중으로 그 자리의 바람(벡터 평균)을 만든다.

    결측 처리: 그 시각에 값이 없는 관측소는 가중치에서 빼고 나머지로 정규화한다(가까운 AWS가 비면 자동으로 ASOS 비중이 커짐).
    모든 관측소가 비면 NaN.
    """

    def __init__(self, wind: pd.DataFrame):
        stations = wind.groupby("station_id")[["station_latitude", "station_longitude", "station_name"]].first()
        self.station_ids = stations.index.to_numpy()
        self.lat = stations["station_latitude"].to_numpy(float)
        self.lon = stations["station_longitude"].to_numpy(float)
        self.names = stations["station_name"].to_dict()
        self.hours = pd.DatetimeIndex(sorted(wind["datetime"].unique()))
        pivot = lambda col: (wind.pivot_table(index="datetime", columns="station_id", values=col, aggfunc="first")
                             .reindex(index=self.hours, columns=self.station_ids))
        self.U = pivot("u"); self.V = pivot("v"); self.S = pivot("wind_speed")
        self.extra = {col: pivot(col) for col in ("temperature", "humidity", "rainfall_hour") if col in wind.columns}

    def weights(self, lat: float, lon: float) -> np.ndarray:
        d = np.maximum(haversine_km(lat, lon, self.lat, self.lon), MIN_DISTANCE_KM)
        return 1.0 / d ** IDW_POWER

    def at(self, lat: float, lon: float) -> pd.DataFrame:
        """해당 위치의 시각별 from_deg·speed(+기온·습도·강수). 관측소별 결측은 가중 재정규화."""
        w = self.weights(lat, lon)
        out = {}
        for key, mat in (("u", self.U), ("v", self.V), ("speed", self.S)):
            values = mat.to_numpy(float)
            mask = ~np.isnan(values)
            wsum = (mask * w).sum(axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                out[key] = np.where(wsum > 0, np.nansum(values * w, axis=1) / wsum, np.nan)
        for key, mat in self.extra.items():
            values = mat.to_numpy(float)
            mask = ~np.isnan(values)
            wsum = (mask * w).sum(axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                out[key] = np.where(wsum > 0, np.nansum(values * w, axis=1) / wsum, np.nan)
        table = pd.DataFrame(out, index=self.hours)
        table["from_deg"] = (np.degrees(np.arctan2(-table["u"], -table["v"])) + 360.0) % 360.0
        table.loc[table["speed"].isna(), "from_deg"] = np.nan
        return table

    def center_table(self, center: tuple[float, float]) -> pd.DataFrame:
        return self.at(center[0], center[1])

    def nearest_station(self, lat: float, lon: float) -> tuple[int, float]:
        d = haversine_km(lat, lon, self.lat, self.lon)
        i = int(np.argmin(d))
        return int(self.station_ids[i]), float(d[i])


def summarize(aws: pd.DataFrame) -> str:
    lines = ["| 지점 | 행 수 | 기간 | 풍향 결측 | 풍속 결측 | 풍속<1 m/s 비율 |", "|---|---:|---|---:|---:|---:|"]
    for sid, g in aws.groupby("station_id"):
        lines.append(f"| {AWS_STATIONS[int(sid)][0]}({sid}) | {len(g):,} | {g['datetime'].min():%Y-%m-%d}~{g['datetime'].max():%Y-%m-%d} | "
                     f"{g['wind_direction'].isna().mean():.3f} | {g['wind_speed'].isna().mean():.3f} | {(g['wind_speed'] < 1).mean():.2f} |")
    return "\n".join(lines)


if __name__ == "__main__":
    aws = build_aws_hourly()
    print(f"AWS 시간자료 {len(aws):,}행 → {AWS_PATH}")
    print(summarize(aws))
