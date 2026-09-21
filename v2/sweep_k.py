"""k(초기 30분 내 서로 다른 30m 위치 수 최소값) 후보별 Event 구성 비교.

성능(정확도)은 보지 않는다 — k를 테스트 성능으로 고르면 그 자체가 누수다.
기준은 (1) 롤링 평가에 쓸 만큼 Event가 남는가(학습 Event가 충분한 폴드들의 테스트 Event 합),
(2) 여러 위치에서 신고한 Event인가(고유위치/신고 비), (3) 추가 민원이 없는 Event 비율,
(4) 기존 정시 Event와의 겹침이다.

실행: python sweep_k.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import MIN_POOLED_TEST_EVENTS, OUT_DIR
from data import ComplaintIndex, load_complaints
from events import build_events, session_table
from folds import make_folds

KS = {"trigger": (4, 5, 6, 7, 8, 10, 12), "session": (2, 3, 4, 5, 6, 8, 10), "trigger_fixed": (4, 5, 6, 8, 10)}


def legacy_hour_events(df: pd.DataFrame) -> pd.Series:
    """기존 정시 Event(1시간 내 10건 이상 & 5개 격자 이상)의 시각. 겹침 비교용."""
    d = df.assign(h=df["datetime"].dt.floor("h"), g=df["grid_x"].astype(str) + ":" + df["grid_y"].astype(str))
    s = d.groupby("h").agg(n=("g", "size"), u=("g", "nunique")).reset_index()
    return s[(s["n"] >= 10) & (s["u"] >= 5)]["h"].reset_index(drop=True)


def overlap(ev: pd.DataFrame, legacy: pd.Series) -> int:
    t0 = ev["t0"].to_numpy("datetime64[m]")
    n = 0
    for h in legacy:
        h = np.datetime64(h, "m")
        n += int(((t0 > h - np.timedelta64(60, "m")) & (t0 < h + np.timedelta64(60, "m"))).any())
    return n


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    df = load_complaints()
    ci = ComplaintIndex(df)
    sessions = session_table(ci)
    legacy = legacy_hour_events(df)
    print(f"민원 {len(df):,}건 / 세션(60분 무신호 뒤 첫 민원) {len(sessions):,}개 / 기존 정시 Event {len(legacy)}개\n")

    rows = []
    for mode, ks in KS.items():
        for k in ks:
            ev = build_events(ci, k, mode, sessions)
            folds, _ = make_folds(ev) if len(ev) >= 20 else ([], None)
            n_test = int(sum(len(f.test_ids) for f in folds))
            by_year = ev.groupby(ev["t0"].dt.year).size()
            rows.append({
                "mode": mode, "k": k, "Event 수": len(ev),
                "예측창에 민원 있음 비율": ev["has_future"].mean(),
                "후보 안 정답 있음 비율": ev["has_positive"].mean(),
                "정답 전부 후보 밖 Event": int(ev["outside_only"].sum()),
                "0-양성 Event": int((~ev["has_positive"]).sum()),
                "발동 앞당김(분, 중앙값)": float(ev["lead_gain_min"].median()) if "lead_gain_min" in ev.columns else np.nan,
                "후보격자/Event": ev["n_candidates"].mean(),
                "양성격자/Event": ev["n_positive_cells"].mean(),
                "고유위치/신고(중앙값)": (ev["n_init_locs"] / ev["n_init_reports"]).median(),
                "2019-21": int(by_year.reindex([2019, 2020, 2021]).fillna(0).sum()),
                "2022": int(by_year.get(2022, 0)),
                "2023+": int(by_year[by_year.index >= 2023].sum()),
                "사용 폴드": len(folds), "표본 작은 폴드(<15)": int(sum(not f.reliable for f in folds)),
                "폴드 테스트 Event 합": n_test, f"규칙(테스트 합>={MIN_POOLED_TEST_EVENTS})": n_test >= MIN_POOLED_TEST_EVENTS,
                "기존 정시 겹침": f"{overlap(ev, legacy)}/{len(legacy)}",
            })
    out = pd.DataFrame(rows)
    rule_col = f"규칙(테스트 합>={MIN_POOLED_TEST_EVENTS})"
    chosen = out[out[rule_col]].groupby("mode")["k"].max()
    out["규칙이 고른 k"] = out.apply(lambda r: bool(r[rule_col] and r["k"] == chosen.get(r["mode"])), axis=1)
    out.to_csv(OUT_DIR / "k_sweep.csv", index=False, encoding="utf-8-sig")
    with pd.option_context("display.width", 260, "display.max_columns", 30, "display.float_format", "{:.2f}".format):
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
