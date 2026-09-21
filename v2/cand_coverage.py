"""후보 밖 실제 민원 비율(candidate coverage의 반대): 예측 창의 실제 민원 중 후보 격자 밖에서 생긴 비율.

후보 = 입력 30분에 관측된 격자 주변 반경 2칸. 예측 창의 민원이 후보 밖에서 생기면 모델이 순위에 올릴 방법이 없다.
그래서 모든 성능표와 함께 이 비율을 보고한다.
  - 격자 기준 : 예측 창의 정답 격자 중 후보 밖 격자의 비율 (Recall@3(전체)의 상한을 (1 - 이 값)으로 제한한다)
  - 신고 건수 기준 : 예측 창의 신고 중 후보 밖 격자에 접수된 비율 (한 격자에 신고가 몰리면 격자 기준과 달라진다)
  - Event 기준 : 예측 창에 민원이 있는 Event 중 정답이 전부 후보 밖인 Event의 비율 ((전체) 지표에서 실패로 세는 Event)
비율은 Event를 복원추출하는 부트스트랩 95% 구간과 함께 낸다(합계의 비). 포함률은 모델과 무관한 Event 구조의 성질이다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from evaluate import wilson_interval
from events import COVERAGE_RADII


def ratio_ci(num: np.ndarray, den: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """합계의 비 sum(num)/sum(den) 와 Event 단위 부트스트랩 95% 구간."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    if den.sum() <= 0:
        return np.nan, np.nan, np.nan
    point = num.sum() / den.sum()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), size=(n_boot, len(num)))
    d = den[idx].sum(1)
    ok = d > 0
    r = num[idx].sum(1)[ok] / d[ok]
    return float(point), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def _fmt(p: float, lo: float, hi: float) -> str:
    return "-" if np.isnan(p) else f"{p:.1%} [{lo:.1%}, {hi:.1%}]"


def coverage_row(ev: pd.DataFrame, label: str, n_boot: int = 2000) -> dict:
    ev = ev[ev["complete"]]
    has = ev[ev["has_future"]]
    n_fut = float(ev["n_future_cells"].sum())
    row = {
        "구분": label, "Event 수": len(ev),
        "예측창에 민원 있음": int(ev["has_future"].sum()),
        "후보 안 정답 있음": int(ev["has_positive"].sum()),
        "정답이 전부 후보 밖": int(ev["outside_only"].sum()),
        "미래 정답 격자 수(합)": int(n_fut),
        "후보 밖 정답 격자 수(합)": int(ev["n_outside_cells"].sum()),
        "예측창 신고 수(합)": int(ev["n_future_reports"].sum()),
        "후보 밖 신고 수(합)": int(ev["n_future_reports_outside"].sum()),
    }
    g = ratio_ci(ev["n_outside_cells"], ev["n_future_cells"], n_boot)
    r = ratio_ci(ev["n_future_reports_outside"], ev["n_future_reports"], n_boot, seed=1)
    row["후보 밖 비율(격자) 95%CI"] = _fmt(*g)
    row["후보 밖 비율(신고 건수) 95%CI"] = _fmt(*r)
    lo, hi = wilson_interval(float(has["outside_only"].sum()), len(has))
    row["정답 전부 후보 밖 Event 비율 95%CI"] = _fmt(float(has["outside_only"].mean()) if len(has) else np.nan, lo, hi)
    row["후보 밖 비율(격자)"] = g[0]
    row["후보 밖 비율(신고 건수)"] = r[0]
    row["후보 포함률(격자 기준)"] = 1 - g[0] if not np.isnan(g[0]) else np.nan
    for rad in COVERAGE_RADII:
        row[f"반경{rad} 포함률"] = float(ev[f"cov_cells_r{rad}"].sum() / n_fut) if n_fut else np.nan
    return row


def coverage_table(events: pd.DataFrame, groups: dict[str, list[str]] | None = None, n_boot: int = 2000) -> pd.DataFrame:
    """전체 + (선택) 그룹별(예: 폴드 테스트 Event) 후보 밖 비율 표."""
    rows = [coverage_row(events, "전체 Event", n_boot)]
    for name, ids in (groups or {}).items():
        rows.append(coverage_row(events[events["event_id"].isin(ids)], name, n_boot))
    return pd.DataFrame(rows)


HEADLINE_COLS = ["구분", "Event 수", "예측창에 민원 있음", "정답이 전부 후보 밖", "후보 밖 비율(격자) 95%CI",
                 "후보 밖 비율(신고 건수) 95%CI", "정답 전부 후보 밖 Event 비율 95%CI"]
