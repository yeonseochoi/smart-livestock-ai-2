"""rolling-origin(expanding window) 폴드.

각 폴드: 테스트 기간 시작 이전의 모든 Event로 학습, 해당 기간 Event로 테스트.
purge: 학습 Event의 라벨 창(t0+60분)이 테스트 시작을 넘지 않도록 t0+60분 <= 테스트 시작 인 Event만 학습에 쓴다.

테스트 기간은 FOLD_BOUNDS 그대로다(2021, 2022는 연 단위, 2023년 이후는 반기 단위).
Event가 적은 기간도 다른 기간과 합치지 않는다 — 합치면 기간별 성능 변화가 가려진다.
대신 테스트 Event가 RELIABLE_MIN_TEST_EVENTS 미만인 폴드는 '표본 작음'으로 표시하고,
폴드별 표에서는 구간(Wilson, 부트스트랩)을 함께 보고한다. 테스트 Event가 0개인 기간은 사유와 함께 표에만 남긴다.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from config import FOLD_BOUNDS, HORIZON_MIN, INPUT_MIN, MIN_TRAIN_EVENTS, RELIABLE_MIN_TEST_EVENTS


@dataclass
class Fold:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp
    train_ids: list[str]
    test_ids: list[str]
    reliable: bool = True            # 테스트 Event >= RELIABLE_MIN_TEST_EVENTS


def make_folds(events: pd.DataFrame, min_train: int = MIN_TRAIN_EVENTS,
               reliable_min: int = RELIABLE_MIN_TEST_EVENTS) -> tuple[list[Fold], pd.DataFrame]:
    """(폴드 목록, 폴드표) 반환. 폴드표에는 제외된 폴드도 사유와 함께 남긴다."""
    ev = events.sort_values("t0")
    data_end = ev["t0"].max() + pd.Timedelta(minutes=1)
    bounds = [pd.Timestamp(b) for b in FOLD_BOUNDS] + [data_end]
    bounds = [b for b in bounds if b <= data_end]
    periods = list(zip(bounds[:-1], bounds[1:]))
    if bounds[-1] < data_end:
        periods.append((bounds[-1], data_end))

    purge = pd.Timedelta(minutes=INPUT_MIN + HORIZON_MIN)
    folds, rows = [], []
    for s, e in periods:
        test = ev[(ev["t0"] >= s) & (ev["t0"] < e)]
        train = ev[ev["t0"] + purge <= s]
        last = e - pd.Timedelta(days=1)
        name = f"{s:%Y-%m}~{last:%Y-%m}"
        if len(test) == 0:
            state = "제외(테스트 Event 0개)"
        elif len(train) < min_train:
            state = f"제외(학습 Event<{min_train})"
        else:
            state = "사용"
        reliable = len(test) >= reliable_min
        rows.append({
            "폴드": name, "학습 기간": f"{train['t0'].min():%Y-%m}~{train['t0'].max():%Y-%m}" if len(train) else "-",
            "학습 Event": len(train), "테스트 Event": len(test),
            "예측창에 민원 있음": int(test["has_future"].sum()),
            "후보 안 정답 있음": int(test["has_positive"].sum()),
            "후보 밖에만 있음": int(test["outside_only"].sum()),
            "상태": state,
            "표본": "" if state != "사용" else ("충분" if reliable else f"작음(<{reliable_min}, 참고용)"),
        })
        if state == "사용":
            folds.append(Fold(name, s, e, train["event_id"].tolist(), test["event_id"].tolist(), reliable))
    return folds, pd.DataFrame(rows)
