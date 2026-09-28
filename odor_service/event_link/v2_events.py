"""v2 trigger 사건의 시험 폴드 Top 3 예측을 재생용으로 생성한다."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from odor_service import common

from odor_service.region_prediction.config import GRID_M, GRID_ORIGIN_LAT, GRID_ORIGIN_LON
from odor_service.region_prediction.data import ComplaintIndex, load_complaints
from odor_service.region_prediction.events import trigger_events
from odor_service.region_prediction.features import BASE_FEATURES, HIST_FEATURES, add_prior, build_event_rows
from odor_service.region_prediction.folds import make_folds
from odor_service.region_prediction.run_v2 import fit_xgb

OUT = common.OUTPUT_DIR / "replay" / "v2_events.csv"


def center(gx: int, gy: int) -> tuple[float, float]:
    """v2/data.py assign_grid의 고정 원점과 축척을 역산한 격자 중심."""
    lat = GRID_ORIGIN_LAT + (gy + 0.5) * GRID_M / 110_540
    lon = GRID_ORIGIN_LON + (gx + 0.5) * GRID_M / (111_320 * math.cos(math.radians(GRID_ORIGIN_LAT)))
    return lat, lon


def main() -> pd.DataFrame:
    ci = ComplaintIndex(load_complaints())
    parts = []
    for k in (5, 7):
        events = trigger_events(ci, k)
        folds, _ = make_folds(events)
        rows = pd.concat(
            [build_event_rows(ci, int(r.start_idx), r.event_id, int(r.t0_override)) for r in events.itertuples()],
            ignore_index=True,
        )
        event_times = events.set_index("event_id")["t0"]
        for fold in folds:
            train = rows.loc[rows.event_id.isin(fold.train_ids)]
            test = rows.loc[rows.event_id.isin(fold.test_ids)]
            if train.t0.max() >= fold.start or test.t0.min() < fold.start:
                raise ValueError(f"{fold.name}: 시간 분할 위반")
            train, test = add_prior(train, test)
            test = test.copy()
            test["score"] = fit_xgb(train, test, BASE_FEATURES + HIST_FEATURES)
            for event_id, group in test.groupby("event_id", sort=False):
                top = group.sort_values("score", ascending=False).head(3)
                if len(top) != 3:
                    raise ValueError(f"{event_id}: Top 3 부족")
                top3 = []
                for rank, r in enumerate(top.itertuples(), 1):
                    lat, lon = center(r.grid_x, r.grid_y)
                    top3.append({"rank": rank, "region_name": None, "lat": lat, "lon": lon,
                                 "score": float(r.score), "hit": bool(r.target)})
                t0 = event_times.loc[event_id]
                parts.append({"event_id": event_id, "k": k, "t0": t0.isoformat(),
                              "night_date": common.night_date(pd.Series([t0])).iloc[0].strftime("%Y-%m-%d"),
                              "top3": json.dumps(top3, ensure_ascii=False), "hit": any(r["hit"] for r in top3)})
        print(f"k={k}: 전체 {len(events)}건, 시험 폴드 {sum(len(f.test_ids) for f in folds)}건")
    result = pd.DataFrame(parts)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT, index=False, encoding="utf-8-sig")
    return result


if __name__ == "__main__":
    main()
