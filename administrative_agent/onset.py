from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd


EVENT_SCORE_COLUMNS = ("event_hour", "hour", "grid_x", "grid_y", "risk_score", "rank", "train_end")


def latest_event_scores(paths: Path | Iterable[Path] | None,
                        event_hour: pd.Timestamp | None = None) -> pd.DataFrame:
    """사건 시점 이후를 학습하지 않은 가장 최신 1단 점수를 선택한다."""
    if paths is None:
        return pd.DataFrame(columns=EVENT_SCORE_COLUMNS)
    candidates = [paths] if isinstance(paths, Path) else list(paths)
    tables = []
    for path in candidates:
        path = Path(path)
        if not path.exists():
            continue
        table = pd.read_csv(path, encoding="utf-8-sig")
        if not set(EVENT_SCORE_COLUMNS).issubset(table.columns):
            continue
        tables.append(table.loc[:, EVENT_SCORE_COLUMNS])
    if not tables:
        return pd.DataFrame(columns=EVENT_SCORE_COLUMNS)

    scores = pd.concat(tables, ignore_index=True)
    for column in ("event_hour", "hour", "train_end"):
        scores[column] = pd.to_datetime(scores[column])
    if event_hour is not None:
        scores = scores[scores["event_hour"] == pd.Timestamp(event_hour)]
    scores = scores[scores["train_end"] <= scores["event_hour"]]
    if scores.empty:
        return scores
    latest = scores.groupby("event_hour")["train_end"].transform("max")
    scores = scores[scores["train_end"] == latest]
    key = ["event_hour", "grid_x", "grid_y"]
    if scores.duplicated(key).any():
        raise ValueError("같은 Event·격자에 선택 가능한 1단 점수가 둘 이상입니다.")
    return scores.sort_values(["event_hour", "rank"]).reset_index(drop=True)
