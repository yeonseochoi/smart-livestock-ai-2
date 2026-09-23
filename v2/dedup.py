"""신고 위치 중복 제거 라벨.

원본 라벨: 예측 창에 그 격자에서 민원이 1건이라도 접수됨.
문제: 같은 위치(30m)에서 반복 신고하는 소수 위치가 라벨을 채울 수 있다
      (전체 신고의 상위 1곳이 7%, 상위 5곳이 29% — 앞선 진단).
변형 라벨은 이 반복 신고를 걷어낸 뒤에도 모델이 맞히는지를 본다.

  drop_top_n      : 전체 기간 신고 건수 상위 N개 위치의 신고를 예측 창에서 뺀 뒤 라벨을 만든다
  min_locs        : 예측 창에 서로 다른 위치 M곳 이상에서 신고가 있어야 양성

위치 처리
  - '서로 다른 위치' 수(min_locs)는 예측 창 [T, T+30분) 안의 신고끼리만 30m 연결로 병합해 센다.
  - '상위 N개 위치'는 전체 기간 군집(loc_diag)으로 정한다. 이것은 미래를 포함한 진단용 강건성 점검이며
    운영 규칙이 아니다. 라벨만 바꾸고 변수·Event 선정·모델 점수는 바꾸지 않으므로 누수 경로가 아니다.
라벨 변형은 전체 지표용 정답 수(n_future_all)도 같은 규칙으로 다시 센다 (후보 밖 격자 포함).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from config import HORIZON_MIN, INPUT_MIN
from data import ComplaintIndex

# 격자 -> [(창 안 위치 번호, 전체기간 진단 위치 ID), ...]  (신고 1건당 한 쌍)
FutureCells = dict[tuple[int, int], list[tuple[int, int]]]


def future_loc_sets(ci: ComplaintIndex, events: pd.DataFrame) -> dict[str, FutureCells]:
    """Event별 예측 창 [T, T+30분) 의 격자 -> 신고들의 (창 안 위치 번호, 진단 위치 ID). 후보 밖 격자도 모두 포함."""
    out: dict[str, FutureCells] = {}
    t = ci.t
    for r in events.itertuples():
        T = int(r.t0_min) + INPUT_MIN
        i1 = int(np.searchsorted(t, T, "left"))
        i2 = int(np.searchsorted(t, T + HORIZON_MIN, "left"))
        lab = ci.loc_labels(i1, i2)
        cells: FutureCells = defaultdict(list)
        for gx, gy, ll, dg in zip(ci.gx[i1:i2], ci.gy[i1:i2], lab, ci.loc_diag[i1:i2]):
            cells[(int(gx), int(gy))].append((int(ll), int(dg)))
        out[r.event_id] = cells
    return out


def heavy_locations(ci: ComplaintIndex, n: int) -> frozenset[int]:
    """전체 기간 신고 건수 상위 n개 위치(진단 위치 ID). 진단용이며 미래를 포함한다."""
    if n <= 0:
        return frozenset()
    counts = pd.Series(ci.loc_diag).value_counts()
    return frozenset(int(x) for x in counts.index[:n])


def _cell_is_positive(reports: list[tuple[int, int]], drop: frozenset[int], min_locs: int) -> bool:
    return len({ll for ll, dg in reports if dg not in drop}) >= min_locs


def relabel(rows: pd.DataFrame, fut: dict[str, FutureCells], drop: frozenset[int] = frozenset(),
            min_locs: int = 1) -> np.ndarray:
    """rows(event_id, grid_x, grid_y) 각 행의 새 라벨(0/1)."""
    out = np.zeros(len(rows), dtype=int)
    for i, (eid, gx, gy) in enumerate(zip(rows["event_id"], rows["grid_x"], rows["grid_y"])):
        reports = fut[eid].get((int(gx), int(gy)))
        if reports:
            out[i] = int(_cell_is_positive(reports, drop, min_locs))
    return out


def relabel_totals(fut: dict[str, FutureCells], drop: frozenset[int] = frozenset(), min_locs: int = 1) -> dict[str, int]:
    """Event별 새 규칙의 정답 격자 수(후보 밖 포함). 전체 지표의 분모."""
    return {eid: sum(_cell_is_positive(rep, drop, min_locs) for rep in cells.values()) for eid, cells in fut.items()}


# (표시 이름, drop_top_n, min_locs)
LABEL_VARIANTS = [
    ("원본 라벨", 0, 1),
    ("상위 3개 위치 신고 제외", 3, 1),
    ("상위 10개 위치 신고 제외", 10, 1),
    ("예측 창에 서로 다른 위치 2곳 이상", 0, 2),
]
