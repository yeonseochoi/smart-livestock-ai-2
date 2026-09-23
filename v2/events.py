"""Event 정의: 주 운영 = trigger, 보조 검증 = session (보수적).

  trigger (주)  : 최근 30분 안에 k번째 고유 위치의 신고가 들어온 순간 예측 (trigger_scan)
  session (보조): 60분 이상 조용하다가 시작된 세션의 첫 신고 후 30분 (아래 설명) — 발동 조건이 엄격해 Event가 적다
  trigger_fixed : 이전 trigger(첫 신고 후 30분을 채운 뒤 발동). 개선 효과를 재는 비교용
  legacy_hour   : 기존 정시 Event 재현(참고용)

세션 기반 정의(session).

  t0  = 직전 민원 이후 SESSION_GAP_MIN(60분) 이상 조용하다가 들어온 첫 민원의 시각
        (t0 시점에 "직전 60분 무민원"은 이미 알 수 있으므로 미래 정보가 아니다)
  T   = t0 + 30분  = 예측 시점. 입력은 [t0, T) 의 민원만 사용.
  라벨 = [T, T+30분) 에 민원이 접수된 격자.

Event 선정 조건은 예측 시점 T에 알 수 있는 값(초기 30분 내 서로 다른 30m 위치 수)만 쓴다.
서로 다른 위치 수는 그 창 안의 신고끼리만 30m로 연결해 센다(ComplaintIndex.n_locs) — 전체 기간 군집 ID를 쓰지 않는다.
예측 구간의 민원 수는 선정에 쓰지 않으므로, 추가 민원이 없는 Event(라벨 전부 0)도 포함된다.

후보 격자와 정답의 관계(보고용 열, 선정에는 쓰지 않는다)
  n_future_cells     : 예측 창에 민원이 있는 서로 다른 1km 격자 수 (후보 안팎 모두)
  n_positive_cells   : 그 중 후보 격자 안에 있는 수  -> has_positive (후보 안 정답이 있음)
  n_outside_cells    : 후보 밖 정답 격자 수
  n_future_reports_outside : 후보 밖 격자에 접수된 예측 창 신고 건수 (n_future_reports 중)
  has_future         : 예측 창에 민원이 하나라도 있음(후보와 무관)
  cov_cells_r{1..4}  : 후보 반경을 1~4칸으로 잡았을 때 포함되는 정답 격자 수 (반경 2가 현재 후보)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import CANDIDATE_RADIUS, HORIZON_MIN, INPUT_MIN, SESSION_GAP_MIN
from data import ComplaintIndex

CAP_GAP_MIN = 1440
COVERAGE_RADII = (1, 2, 3, 4)


def candidate_cells(observed: set[tuple[int, int]], radius: int = CANDIDATE_RADIUS) -> list[tuple[int, int]]:
    """관측 격자 주변 반경 radius 칸을 후보로 한다 (기존 odor.candidate_cells와 동일)."""
    cells: set[tuple[int, int]] = set()
    for gx, gy in observed:
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius:
                    cells.add((gx + dx, gy + dy))
    return sorted(cells)


def session_starts(ci: ComplaintIndex, gap_min: int = SESSION_GAP_MIN) -> tuple[np.ndarray, np.ndarray]:
    """세션 시작 위치(배열 인덱스)와 직전 무민원 시간(분, 상한 CAP_GAP_MIN)."""
    t = ci.t
    gaps = np.diff(t, prepend=t[0] - 10 * CAP_GAP_MIN)
    starts = np.flatnonzero(gaps >= gap_min)
    return starts, np.minimum(gaps[starts], CAP_GAP_MIN)


def _event_stats(ci: ComplaintIndex, s: int, gap: float, with_candidates: bool = True, t0: int | None = None) -> dict:
    """시작 인덱스 s 의 입력창 [t0, T) 통계 + (보고용) 예측창 통계.

    n_future_*, n_positive_cells 는 보고용이며 선정 조건에 쓰지 않는다.
    """
    t = ci.t
    t0 = int(t[s]) if t0 is None else int(t0)
    T = t0 + INPUT_MIN; E = T + HORIZON_MIN
    i1 = int(np.searchsorted(t, T, "left")); i2 = int(np.searchsorted(t, E, "left"))
    row = {
        "start_idx": int(s), "t0_min": t0, "gap_before_min": float(gap),
        "n_init_reports": i1 - s, "n_init_locs": ci.n_locs(s, i1),
        "n_init_grids": len(set(zip(ci.gx[s:i1], ci.gy[s:i1]))),
        "n_future_reports": i2 - i1,
        "complete": E <= int(t[-1]),                 # 예측 창이 데이터 끝을 넘지 않는가
        "n_candidates": np.nan, "n_positive_cells": np.nan,
        "n_future_cells": np.nan, "n_outside_cells": np.nan, "n_future_reports_outside": np.nan,
        **{f"cov_cells_r{r}": np.nan for r in COVERAGE_RADII},
    }
    if with_candidates and row["complete"]:
        observed = set(zip(ci.gx[s:i1], ci.gy[s:i1]))
        cand = set(candidate_cells(observed))
        fut = set(zip(ci.gx[i1:i2], ci.gy[i1:i2]))
        row["n_candidates"] = len(cand)
        row["n_positive_cells"] = len(fut & cand)
        row["n_future_cells"] = len(fut)
        row["n_outside_cells"] = len(fut - cand)
        row["n_future_reports_outside"] = int(sum(c not in cand for c in zip(ci.gx[i1:i2], ci.gy[i1:i2])))
        for r in COVERAGE_RADII:
            row[f"cov_cells_r{r}"] = len(fut & set(candidate_cells(observed, r)))
    return row


_EMPTY_COLS = ["start_idx", "t0_min", "gap_before_min", "n_init_reports", "n_init_locs", "n_init_grids", "n_future_reports",
               "complete", "n_candidates", "n_positive_cells", "n_future_cells", "n_outside_cells", "n_future_reports_outside",
               *[f"cov_cells_r{r}" for r in COVERAGE_RADII], "t0_override", "t_pred_min", "lead_gain_min"]


def _finish(rows: list[dict]) -> pd.DataFrame:
    out = pd.DataFrame(rows) if rows else pd.DataFrame(columns=_EMPTY_COLS)      # 발동이 한 번도 없어도 열은 유지
    out["t0"] = pd.to_datetime(out["t0_min"], unit="m")
    return out


def add_target_flags(ev: pd.DataFrame) -> pd.DataFrame:
    """has_positive: 후보 안에 정답이 있음(조건부 지표의 대상). has_future: 후보와 무관하게 예측 창에 민원이 있음."""
    ev["has_positive"] = ev["n_positive_cells"] > 0
    ev["has_future"] = ev["n_future_cells"] > 0
    ev["outside_only"] = ev["has_future"] & ~ev["has_positive"]      # 정답이 전부 후보 밖 -> 전체 지표에서는 실패
    return ev


def session_table(ci: ComplaintIndex) -> pd.DataFrame:
    """[mode=session] 모든 세션의 통계. k와 무관하게 한 번 계산해 두고 k만 바꿔 재사용한다."""
    starts, gaps = session_starts(ci)
    return _finish([_event_stats(ci, int(s), g) for s, g in zip(starts, gaps)])


def select_events(sessions: pd.DataFrame, k: int) -> pd.DataFrame:
    """예측 시점에 알 수 있는 값(초기 30분 고유 위치 수 >= k)만으로 고른다."""
    ev = sessions[(sessions["n_init_locs"] >= k) & sessions["complete"]].copy()
    ev = ev.sort_values("t0").reset_index(drop=True)
    ev["event_id"] = [f"EV2-{i:04d}" for i in range(1, len(ev) + 1)]
    return add_target_flags(ev)


def trigger_fixed_scan(ci: ComplaintIndex, k: int, unit: str = "locs") -> pd.DataFrame:
    """[이전 방식, 비교용] 민원 시각 t_j 를 창 시작으로 보고 [t_j, t_j+30분) 의 고유 위치 수가 k 이상이면
    창이 끝나는 t_j+30분에 발동한다. k번째 위치가 일찍 들어와도 창이 끝날 때까지 기다리므로 발동이 늦다.
    """
    t = ci.t
    rows: list[dict] = []
    next_ok = -10**12
    for j in range(len(t)):
        if t[j] < next_ok:
            continue
        i1 = int(np.searchsorted(t, t[j] + INPUT_MIN, "left"))
        if i1 - j < k:                                   # 신고 건수가 k 미만이면 위치 수도 k 미만 (군집화 생략)
            continue
        amount = ci.n_locs(j, i1) if unit == "locs" else i1 - j
        if amount < k:
            continue
        gap = min(int(t[j] - t[j - 1]), CAP_GAP_MIN) if j > 0 else CAP_GAP_MIN
        rows.append(_event_stats(ci, j, gap))
        next_ok = t[j] + INPUT_MIN + HORIZON_MIN
    return _finish(rows)


def trigger_scan(ci: ComplaintIndex, k: int, unit: str = "locs") -> pd.DataFrame:
    """[주 운영 Event] 최근 30분 안에 k번째 고유 위치의 신고가 들어온 그 순간 발동한다.

    분 단위로 본다. 신고가 들어온 분 e 가 끝났을 때, 그 분을 포함한 최근 30분 [e-29, e] 의 서로 다른 30m 위치 수가
    처음으로 k 이상이 되면 발동한다(창 안 신고끼리만 병합해 센다). 예측 시점 T = e+1, 입력 = [T-30, T),
    라벨 = [T, T+30). 신고가 늘어나야 위치 수가 늘어나므로 신고가 들어온 분만 검사하면 충분하다.

    이전 방식(trigger_fixed_scan)은 첫 신고부터 30분을 채운 뒤에 발동해서, k번째 위치가 일찍 들어와도 그만큼 기다렸다.
    같은 입력 창이라면 이 방식이 항상 같거나 더 일찍 발동한다(lead_gain_min).

    인과성: 발동 판단은 분 e 이하의 신고만, 변수는 t < T 의 신고만 쓴다. 발동 뒤에는 라벨 창 [T, T+30) 이 끝나 다음 입력 창이
    시작할 수 있을 때까지(e+60분) 새 발동을 막는다 (앞 Event의 라벨이 다음 Event의 입력에 들어가지 않게).
    인과성 테스트가 이 함수를 직접 쓴다.

    unit="locs"   : 창 안의 서로 다른 30m 위치 수 >= k  (기본)
    unit="reports": 창 안의 신고 건수 >= k               (비교용)
    """
    t = ci.t
    n = len(t)
    rows: list[dict] = []
    next_ok_e = -10**12                                   # 이 분 이후에만 다시 발동할 수 있다
    i = 0
    while i < n:
        e = int(t[i])
        hi = int(np.searchsorted(t, e, "right"))          # 분 e 의 신고를 모두 포함
        i = hi
        if e < next_ok_e:
            continue
        lo = int(np.searchsorted(t, e - INPUT_MIN + 1, "left"))
        if hi - lo < k:                                   # 신고 건수가 k 미만이면 위치 수도 k 미만 (군집화 생략)
            continue
        amount = ci.n_locs(lo, hi) if unit == "locs" else hi - lo
        if amount < k:
            continue
        T = e + 1
        t0 = T - INPUT_MIN
        gap = min(t0 - int(t[lo - 1]), CAP_GAP_MIN) if lo > 0 else CAP_GAP_MIN
        row = _event_stats(ci, lo, gap, True, t0=t0)
        row["t0_override"] = t0
        row["t_pred_min"] = T
        row["lead_gain_min"] = int(t[lo]) + INPUT_MIN - T   # 이전 방식(첫 신고+30분)이었다면 더 기다렸을 분
        rows.append(row)
        next_ok_e = e + INPUT_MIN + HORIZON_MIN
    return _finish(rows)


def _trigger_table(scan: pd.DataFrame) -> pd.DataFrame:
    ev = scan[scan["complete"]].sort_values("t0").reset_index(drop=True)
    ev["event_id"] = [f"EV2-{i:04d}" for i in range(1, len(ev) + 1)]
    return add_target_flags(ev)


def trigger_events(ci: ComplaintIndex, k: int, unit: str = "locs") -> pd.DataFrame:
    """[mode=trigger, 주 운영 Event] trigger_scan 에서 예측 창이 데이터 끝을 넘지 않는 Event만 남긴다."""
    return _trigger_table(trigger_scan(ci, k, unit))


def trigger_fixed_events(ci: ComplaintIndex, k: int, unit: str = "locs") -> pd.DataFrame:
    """[mode=trigger_fixed, 이전 방식·비교용] 첫 신고 후 30분을 채운 뒤 발동."""
    return _trigger_table(trigger_fixed_scan(ci, k, unit))


def legacy_hour_events(ci: ComplaintIndex) -> pd.DataFrame:
    """[mode=legacy_hour] 기존 방식 재현(참고용). 시계 정시를 t0로 하고, 선정에 그 시간대 전체(예측 구간 포함)를 쓴다.

    선정: 1시간 내 10건 이상 & 서로 다른 1km 격자 5개 이상, 그리고 (입력 격자 >= 2, 예측 창 격자 >= 1).
    이 함수는 v2 평가 틀 안에서 "Event 정의만 바꿨을 때" 성능이 어떻게 달라지는지 보기 위한 비교 대상이다.
    """
    t = ci.t
    hour_start = (t // 60) * 60
    frame = pd.DataFrame({"h": hour_start, "g": ci.gx * 100_000 + ci.gy})
    agg = frame.groupby("h").agg(n=("g", "size"), u=("g", "nunique"))
    hours = agg[(agg["n"] >= 10) & (agg["u"] >= 5)].index.to_numpy()
    rows = []
    for h in hours:
        s = int(np.searchsorted(t, h, "left"))
        i1 = int(np.searchsorted(t, h + INPUT_MIN, "left")); i2 = int(np.searchsorted(t, h + INPUT_MIN + HORIZON_MIN, "left"))
        if len(set(zip(ci.gx[s:i1], ci.gy[s:i1]))) < 2 or len(set(zip(ci.gx[i1:i2], ci.gy[i1:i2]))) < 1:
            continue
        gap = min(int(h - t[s - 1]), CAP_GAP_MIN) if s > 0 else CAP_GAP_MIN
        rows.append({**_event_stats(ci, s, gap, True, t0=int(h)), "t0_override": int(h)})
    ev = _finish(rows)
    ev = ev[ev["complete"]].sort_values("t0").reset_index(drop=True)
    ev["event_id"] = [f"EV2-{i:04d}" for i in range(1, len(ev) + 1)]
    return add_target_flags(ev)


def build_events(ci: ComplaintIndex, k: int, mode: str = "session", sessions: pd.DataFrame | None = None) -> pd.DataFrame:
    if mode == "legacy_hour":
        return legacy_hour_events(ci)
    if mode == "session":
        return select_events(session_table(ci) if sessions is None else sessions, k)
    if mode == "trigger":
        return trigger_events(ci, k)
    if mode == "trigger_fixed":
        return trigger_fixed_events(ci, k)
    raise ValueError(f"unknown mode: {mode}")
