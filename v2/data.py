"""민원 로드 + 격자 + 시각 인덱스.

ComplaintIndex는 "어떤 시각 이전 데이터만 보는" 질의를 빠르게 하기 위한 정렬 배열 묶음이다.
누수 테스트(tests/test_no_leakage.py)는 이 객체를 예측 시점(t0+30분)에서 잘라 만들어
변수가 달라지지 않는지 확인한다.

위치(30m) 처리 원칙
  - Event 선정·변수에 쓰는 '고유 위치 수'는 창 안의 신고끼리만 30m 연결로 병합해 센다(ComplaintIndex.n_locs).
    전체 기간을 한 번에 군집화한 ID는 쓰지 않는다: 미래의 '중간 좌표'가 과거의 두 위치를 이어 붙일 수 있다.
  - loc_diag(전체 기간 DBSCAN)는 진단용이다. 상습 신고 위치(상위 N곳)를 가려내는 라벨 쪽 강건성 점검(dedup.py)에만 쓴다.
    Event 선정·변수·모델 점수에는 쓰지 않는다(테스트가 이 열을 뒤섞어도 결과가 같은지 확인한다).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import DBSCAN

from config import CITY, DATA_PATH, EARTH_RADIUS_M, GRID_M, GRID_ORIGIN_LAT, GRID_ORIGIN_LON, LOC_EPS_M

M_PER_DEG = math.pi / 180.0 * EARTH_RADIUS_M            # 위도 1도 ≈ 111,195m


def assign_grid(df: pd.DataFrame) -> pd.DataFrame:
    """1km 격자 열(grid_x, grid_y). 원점이 고정 상수라 데이터를 시점별로 잘라도 같은 민원은 같은 격자에 놓인다."""
    x_m = (df["longitude"] - GRID_ORIGIN_LON) * 111_320 * math.cos(math.radians(GRID_ORIGIN_LAT))
    y_m = (df["latitude"] - GRID_ORIGIN_LAT) * 110_540
    out = df.copy()
    out["grid_x"] = np.floor(x_m / GRID_M).astype(int)
    out["grid_y"] = np.floor(y_m / GRID_M).astype(int)
    return out


def load_complaints(path=DATA_PATH, city: str = CITY) -> pd.DataFrame:
    """익산시 민원만 읽는다(완주군 등 제외). 기존 base.filter_target_region과 같은 14,994건."""
    raw = pd.read_excel(path)
    raw = raw[raw["시군구"] == city]
    df = pd.DataFrame({
        "datetime": pd.to_datetime(raw["악취발생일시"]),
        "latitude": raw["위도"].astype(float),
        "longitude": raw["경도"].astype(float),
        "intensity": raw["악취강도코드"].astype(float),
    }).dropna()
    df = df.sort_values("datetime", kind="stable").reset_index(drop=True)

    # 격자: 기존 add_grid_columns와 같은 방식(1km), 원점은 config 의 고정 상수(전체 중앙값을 한 번 계산해 둔 값)
    df = assign_grid(df)

    # 진단용 전체 기간 위치 ID (Event 선정·변수·모델에는 쓰지 않는다. 모듈 설명 참고)
    coords = np.radians(df[["latitude", "longitude"]].to_numpy())
    df["loc_diag"] = DBSCAN(
        eps=LOC_EPS_M / EARTH_RADIUS_M, min_samples=1, metric="haversine", algorithm="ball_tree"
    ).fit_predict(coords)
    return df


class ComplaintIndex:
    """분 단위 정수 시각 배열과 격자별 정렬 배열."""

    def __init__(self, df: pd.DataFrame):
        d = df.sort_values("datetime", kind="stable").reset_index(drop=True)
        self.df = d
        self.t = d["datetime"].values.astype("datetime64[m]").astype("int64")   # epoch 이후 분
        self.gx = d["grid_x"].to_numpy(int)
        self.gy = d["grid_y"].to_numpy(int)
        self.inten = d["intensity"].to_numpy(float)
        # 창 안 위치 병합용 평면 좌표(m). 원점은 고정 상수라 데이터를 잘라도 값이 변하지 않는다.
        self.x_m = (d["longitude"].to_numpy(float) - GRID_ORIGIN_LON) * M_PER_DEG * math.cos(math.radians(GRID_ORIGIN_LAT))
        self.y_m = (d["latitude"].to_numpy(float) - GRID_ORIGIN_LAT) * M_PER_DEG
        # 진단 전용(있을 때만). 선정·변수 코드는 이 값을 읽지 않는다.
        self.loc_diag = d["loc_diag"].to_numpy(int) if "loc_diag" in d.columns else np.full(len(d), -1)
        self.first_day = int(self.t[0] // 1440) if len(self.t) else 0

        self.cell_minutes: dict[tuple[int, int], np.ndarray] = {}
        for (x, y), idx in d.groupby(["grid_x", "grid_y"]).indices.items():
            self.cell_minutes[(int(x), int(y))] = np.sort(self.t[idx])

        # (격자, 시각(시)) -> 민원이 있었던 날짜(epoch 이후 일) 정렬 배열
        tmp = pd.DataFrame({
            "x": self.gx, "y": self.gy,
            "h": (self.t % 1440) // 60, "d": self.t // 1440,
        }).drop_duplicates()
        self.cell_hour_days: dict[tuple[int, int, int], np.ndarray] = {}
        for (x, y, h), g in tmp.groupby(["x", "y", "h"]):
            self.cell_hour_days[(int(x), int(y), int(h))] = np.sort(g["d"].to_numpy())

    def loc_labels(self, i0: int, i1: int) -> np.ndarray:
        """배열 구간 [i0, i1)의 신고를 30m 이내끼리 연결(단일 연결)해 위치 번호를 붙인다. 그 구간의 신고만 본다.

        DBSCAN(eps=30m, min_samples=1)을 구간 안의 점에만 적용한 것과 같은 결과다.
        """
        n = i1 - i0
        if n <= 1:
            return np.zeros(max(n, 0), dtype=int)
        xy = np.column_stack([self.x_m[i0:i1], self.y_m[i0:i1]])
        diff = xy[:, None, :] - xy[None, :, :]
        adj = (diff ** 2).sum(-1) <= LOC_EPS_M ** 2
        return connected_components(csr_matrix(adj), directed=False)[1]

    def n_locs(self, i0: int, i1: int) -> int:
        """구간 [i0, i1) 안의 서로 다른 30m 위치 수 (그 구간의 신고만으로 계산)."""
        n = i1 - i0
        return 0 if n <= 0 else (1 if n == 1 else int(self.loc_labels(i0, i1).max()) + 1)

    def count_between(self, lo: int, hi: int) -> int:
        """[lo, hi) 분 구간 전체 민원 수."""
        return int(np.searchsorted(self.t, hi, "left") - np.searchsorted(self.t, lo, "left"))

    def cell_count_between(self, cell: tuple[int, int], lo: int, hi: int) -> int:
        arr = self.cell_minutes.get(cell)
        if arr is None:
            return 0
        return int(np.searchsorted(arr, hi, "left") - np.searchsorted(arr, lo, "left"))
