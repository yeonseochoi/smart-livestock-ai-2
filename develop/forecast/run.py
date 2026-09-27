"""Rolling, observed-weather backtest for nightly complaint counts."""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from develop import common

OUT = common.OUTPUT_DIR / "forecast"
FEATURES = ["night_temperature", "night_humidity", "night_wind_speed", "south_wind_ratio", "calm_ratio", "day_temperature", "day_rain", "previous_rain", "month", "dow", "doy"]
CALENDAR = ["month", "dow", "doy"]
SCHEMA = ["night_date", "season", "risk_score", "season_rank", "alert", "model_id", "weather_input", "train_end", "observed_complaints"]


def nights() -> pd.DataFrame:
    dates = pd.DatetimeIndex(np.concatenate([pd.date_range(f"{y}-05-01", f"{y}-10-31").values for y in range(2020, 2026)]), name="night_date")
    complaints = common.load_livestock_complaints()
    counts = complaints.loc[complaints.in_night].groupby("night_date").size()
    frame = pd.DataFrame(index=dates)
    frame["observed_complaints"] = counts.reindex(dates, fill_value=0).astype(int)
    frame["season"] = dates.year
    frame["month"] = dates.month
    frame["dow"] = dates.dayofweek
    frame["doy"] = dates.dayofyear

    weather = pd.read_csv(common.ASOS_FILE, parse_dates=["datetime"])
    weather = weather.loc[weather.station_id.eq(146)].copy()
    if weather.empty or weather.datetime.duplicated().any():
        raise ValueError("ASOS station 146 missing or duplicate hourly timestamps")
    for col in ["wind_direction", "wind_speed", "temperature", "humidity", "rainfall_hour"]:
        weather[col] = pd.to_numeric(weather[col], errors="coerce")
    weather["date"] = weather.datetime.dt.normalize()
    weather["night_date"] = common.night_date(weather.datetime)
    day = weather.groupby("date").agg(day_temperature=("temperature", "mean"), day_rain=("rainfall_hour", lambda x: x.sum(min_count=1)))
    frame = frame.join(day.rename_axis("night_date"))
    frame["previous_rain"] = day.day_rain.reindex(dates - pd.Timedelta(days=1)).to_numpy()
    night = weather.loc[common.in_night_window(weather.datetime)].copy()
    night["south"] = np.where(night.wind_speed.notna() & night.wind_direction.notna(), ((night.wind_direction.between(135, 225)) & (night.wind_speed >= .5)).astype(float), np.nan)
    night["calm"] = np.where(night.wind_speed.notna(), (night.wind_speed < .5).astype(float), np.nan)
    agg = night.groupby("night_date").agg(night_temperature=("temperature", "mean"), night_humidity=("humidity", "mean"), night_wind_speed=("wind_speed", "mean"), south_wind_ratio=("south", "mean"), calm_ratio=("calm", "mean"))
    return frame.join(agg).reset_index()


def model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(loss="poisson", max_iter=300, learning_rate=.05, max_depth=3, random_state=0)


def rates(y: np.ndarray, selected: np.ndarray) -> tuple[float, float]:
    burst = y >= np.quantile(y, .9)
    return float(np.mean(y[selected] <= 1)), float(np.mean(selected[burst]))


def ci(y: np.ndarray, selected: np.ndarray, seed: int) -> list[list[float]]:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(1000):
        ix = rng.integers(0, len(y), len(y))
        s = selected[ix]
        if s.any():
            values.append(rates(y[ix], s))
    return np.quantile(values, [.025, .975], axis=0).T.tolist()


def backtest(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows, metrics = [], {}
    for season in range(2022, 2026):
        train = frame.loc[frame.season < season]
        test = frame.loc[frame.season == season].reset_index(drop=True)
        if train.empty or len(test) != 184:
            raise ValueError(f"season {season}: incomplete train/test calendar")
        train_end = train.night_date.max()
        if not (train_end < test.night_date.min()):
            raise ValueError("training overlaps evaluation")
        y = test.observed_complaints.to_numpy()
        season_metrics = {}
        for model_id, columns in [("hgb_poisson", FEATURES), ("calendar", CALENDAR)]:
            estimator = model().fit(train[columns], train.observed_complaints)
            score = estimator.predict(test[columns])
            order = np.argsort(-score, kind="stable")
            rank = np.empty(len(test), dtype=int)
            rank[order] = np.arange(1, len(test) + 1)
            selected = rank <= 30
            empty, capture = rates(y, selected)
            intervals = ci(y, selected, season)
            season_metrics[model_id] = {"empty_night_rate": empty, "burst_capture_rate": capture, "empty_night_ci95": intervals[0], "burst_capture_ci95": intervals[1]}
            for date, s, r, a, observed in zip(test.night_date, score, rank, selected, y):
                rows.append([date.date().isoformat(), season, float(s), int(r), bool(a), model_id, "observed", train_end.date().isoformat(), int(observed)])
        rng = np.random.default_rng(season)
        random_rates, random_bootstrap = [], []
        for draw in range(1000):
            selected = np.zeros(len(y), dtype=bool)
            selected[rng.choice(len(y), 30, replace=False)] = True
            random_rates.append(rates(y, selected))
            ix = rng.integers(0, len(y), len(y))
            random_bootstrap.append(rates(y[ix], selected[ix]))
            if draw == 0:
                rank = np.empty(len(y), dtype=int)
                rank[np.r_[np.flatnonzero(selected), np.flatnonzero(~selected)]] = np.arange(1, len(y) + 1)
                for date, r, a, observed in zip(test.night_date, rank, selected, y):
                    rows.append([date.date().isoformat(), season, float(len(y) + 1 - r), int(r), bool(a), "random", "observed", train_end.date().isoformat(), int(observed)])
        random_rates = np.asarray(random_rates)
        random_bootstrap = np.asarray(random_bootstrap)
        season_metrics["random"] = {"empty_night_rate": float(random_rates[:, 0].mean()), "burst_capture_rate": float(random_rates[:, 1].mean()), "empty_night_ci95": np.quantile(random_bootstrap[:, 0], [.025, .975]).tolist(), "burst_capture_ci95": np.quantile(random_bootstrap[:, 1], [.025, .975]).tolist()}
        metrics[str(season)] = season_metrics
    return pd.DataFrame(rows, columns=SCHEMA), metrics


def main() -> None:
    frame = nights()
    results, metrics = backtest(frame)
    OUT.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT / "night_risk.csv", index=False)
    (OUT / "backtest_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# A 폭주 밤 예보 백테스트", "", "관측 기상 기준, 운영 시 저하 가능", "", "| 시즌 | 기준 | 공친 밤 비율 (95% CI) | 폭주 밤 포착률 (95% CI) |", "|---|---|---:|---:|"]
    for season, group in metrics.items():
        for name, value in group.items():
            fmt = lambda x: f"{x:.1%}"
            lines.append(f"| {season} | {name} | {fmt(value['empty_night_rate'])} ({fmt(value['empty_night_ci95'][0])}~{fmt(value['empty_night_ci95'][1])}) | {fmt(value['burst_capture_rate'])} ({fmt(value['burst_capture_ci95'][0])}~{fmt(value['burst_capture_ci95'][1])}) |")
    lines += ["", "## DEVELOP.md 4장 수치와 차이", "", "차이 단위는 %p임. 재현 기준은 반올림한 정수 %임.", "", "| 시즌 | 공친 밤: 재현 / 문서 / 차이 | 포착률: 재현 / 문서 / 차이 |", "|---|---:|---:|"]
    for season, target_empty, target_capture in zip(range(2022, 2026), [17, 33, 7, 20], [39, 32, 45, 42]):
        value = metrics[str(season)]["hgb_poisson"]
        a, b = value["empty_night_rate"] * 100, value["burst_capture_rate"] * 100
        lines.append(f"| {season} | {a:.1f}% / {target_empty}% / {a-target_empty:+.1f} | {b:.1f}% / {target_capture}% / {b-target_capture:+.1f} |")
    lines += ["", "밤 날짜는 21:00~04:59를 시작일로 묶음. 5~10월 모든 밤을 포함하고 0건 밤을 유지함. 폭주 밤은 해당 시즌 관측 건수의 90백분위수 이상(동률 포함)임. 무작위 수치는 30밤 추출 1,000회 평균이며 CSV의 random 행은 첫 추출 사례임. 95% CI는 평가 밤의 paired bootstrap 1,000회임(무작위는 각 반복마다 경보도 재추출). 야간 관측값을 사용한 후향 평가이므로 예보 입력으로 운영한 성능은 확인되지 않음."]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
