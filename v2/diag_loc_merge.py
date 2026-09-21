"""P3 진단: 전체 기간 군집 ID(loc_diag)로 세던 방식과, 창 안 신고끼리만 병합하는 방식(현재)의 차이 크기.

  (1) 모든 30분 창에서 두 방식의 고유 위치 수가 다른 창의 비율
  (2) 그 차이가 k 선정을 바꾼 창 수 (trigger k=6, session k=4)
  (3) 전체 기간 군집이 '사슬'로 크게 이어진 정도 (한 군집 안의 최대 지름/신고 수)

실행: python diag_loc_merge.py       결과: outputs/diag_loc_merge.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import INPUT_MIN, K_MIN_UNIQUE_LOCATIONS, LOC_EPS_M, OUT_DIR
from data import ComplaintIndex, load_complaints
from events import session_starts


def main() -> None:
    df = load_complaints()
    ci = ComplaintIndex(df)
    t = ci.t
    rows = []
    n = len(t)
    i1 = np.searchsorted(t, t + INPUT_MIN, "left")
    old = np.empty(n, int); new = np.empty(n, int)
    for j in range(n):
        old[j] = np.unique(ci.loc_diag[j:i1[j]]).size
        new[j] = ci.n_locs(j, int(i1[j]))
    diff = old != new
    rows.append({"항목": "모든 시작 창(민원 1건당 1개)", "값": n})
    rows.append({"항목": "고유 위치 수가 달라진 창", "값": int(diff.sum())})
    rows.append({"항목": "  그 중 전체기간 방식이 더 작게 셈(사슬 병합)", "값": int((old < new).sum())})
    rows.append({"항목": "  그 중 전체기간 방식이 더 크게 셈", "값": int((old > new).sum())})

    for mode, k in K_MIN_UNIQUE_LOCATIONS.items():
        if mode == "trigger":
            flip = (old >= k) != (new >= k)
            rows.append({"항목": f"trigger k={k}: 시작 창 자격이 바뀐 창", "값": int(flip.sum())})
        else:
            starts, _ = session_starts(ci)
            flip = (old[starts] >= k) != (new[starts] >= k)
            rows.append({"항목": f"session k={k}: 세션 {len(starts)}개 중 자격이 바뀐 세션", "값": int(flip.sum())})

    # 전체기간 군집의 사슬 정도: 한 군집 안에서 가장 먼 두 점의 거리
    d = pd.DataFrame({"g": ci.loc_diag, "x": ci.x_m, "y": ci.y_m})
    spans = d.groupby("g").agg(n=("x", "size"), w=("x", lambda s: s.max() - s.min()), h=("y", lambda s: s.max() - s.min()))
    span = np.hypot(spans["w"], spans["h"])
    rows.append({"항목": f"전체기간 군집 수 / 신고 수", "값": f"{len(spans)} / {n}"})
    rows.append({"항목": f"군집 최대 폭이 {LOC_EPS_M}m 초과인 군집 수(사슬로 이어진 군집)", "값": int((span > LOC_EPS_M).sum())})
    rows.append({"항목": "군집 최대 폭 상위 3개(m)", "값": ", ".join(f"{v:.0f}" for v in span.sort_values(ascending=False).head(3))})
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "diag_loc_merge.csv", index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
