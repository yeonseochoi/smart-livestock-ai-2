"""시나리오 비교: 주 운영 trigger vs 보수적 보조 검증 session, 그리고 발동 시점 개선(trigger vs trigger_fixed).

먼저 각 mode 에 대해 run_v2.py 와 evaluate_step2.py 를 실행해 두어야 한다.
결과: outputs/scenario_compare.md

  1) 결론 일치표: 같은 비교(모델 vs prior 기준선)가 주 시나리오와 보조 시나리오에서 같은 판정(▲/○/▼)인가
  2) 발동 시점 개선 효과: trigger vs trigger_fixed 의 Event 수, 발동이 앞당겨진 분, 성능
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import K_MIN_UNIQUE_LOCATIONS, MODE_ROLE, OUT_DIR
from evaluation import (HIT_ALL, MARK, PR, REC_ALL, ROC, VERDICT_BETTER, VERDICT_REF, VERDICT_UNSURE, VERDICT_WORSE)
from experiments import REFERENCE
from run_v2 import md

BLOCK = "A. 전체 후보 · 원본 라벨"
MODELS = ["XGB 기존15변수(LOO prior)", "XGB 15변수+이력", "기준선 초기민원수 상위3"]
METRICS = [HIT_ALL, REC_ALL, PR, ROC]
PRIMARY, AUX, OLD = "trigger", "session", "trigger_fixed"


def load(mode: str) -> pd.DataFrame | None:
    p = OUT_DIR / f"step2_{mode}_k{K_MIN_UNIQUE_LOCATIONS[mode]}" / "eval_long.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    return d[d["블록"] == BLOCK]


def cell(r: pd.Series) -> str:
    return f"{r['값']:.3f}  Δ{r['차이']:+.3f} [{r['95% 하한']:+.3f}, {r['95% 상한']:+.3f}] {MARK.get(r['판정'], '')}"


def main() -> None:
    data = {m: load(m) for m in (PRIMARY, AUX, OLD)}
    text = "# 시나리오 비교\n\n"
    text += (f"- **{PRIMARY} (k={K_MIN_UNIQUE_LOCATIONS[PRIMARY]})**: {MODE_ROLE[PRIMARY]} — 최근 30분 안에 k번째 고유 위치의 신고가 들어온 순간 예측\n"
             f"- **{AUX} (k={K_MIN_UNIQUE_LOCATIONS[AUX]})**: {MODE_ROLE[AUX]} — 60분 무신호 뒤 시작한 세션의 첫 민원 후 30분. Event가 적고 조건이 엄격하다\n"
             f"- **{OLD} (k={K_MIN_UNIQUE_LOCATIONS[OLD]})**: {MODE_ROLE[OLD]} — 첫 신고부터 30분을 채운 뒤 발동\n\n"
             "칸 형식: 값  Δ(prior만 대비)  [95% 구간]  ▲개선 ○구분 불가 ▼악화\n\n")

    # ---- 1. 결론 일치 ----
    rows = []
    for model in MODELS:
        for metric in METRICS:
            row = {"모델": model, "지표": metric}
            verdicts = {}
            for mode in (PRIMARY, AUX):
                d = data[mode]
                if d is None:
                    row[f"{mode} ({MODE_ROLE[mode]})"] = "-"; continue
                r = d[(d["모델"] == model) & (d["지표"] == metric)].iloc[0]
                row[f"{mode} ({MODE_ROLE[mode]})"] = cell(r)
                verdicts[mode] = r["판정"]
            if len(verdicts) == 2:
                a, b = verdicts[PRIMARY], verdicts[AUX]
                row["결론 일치"] = "같음" if a == b else f"다름(주 {MARK.get(a, '')} / 보조 {MARK.get(b, '')})"
            rows.append(row)
    cmp1 = pd.DataFrame(rows)
    text += f"## 1. 주 시나리오와 보조 시나리오의 결론이 같은가\n\n{md(cmp1)}\n\n"
    same = int((cmp1["결론 일치"] == "같음").sum()) if "결론 일치" in cmp1 else 0
    text += f"판정이 같은 비교: {same} / {len(cmp1)}. 다른 칸은 두 시나리오의 Event 집합·표본 크기가 달라 생긴 차이일 수 있으므로 구간이 0에 얼마나 가까운지도 함께 본다.\n\n"

    # ---- 2. 발동 시점 개선 ----
    ev_new = pd.read_csv(OUT_DIR / f"{PRIMARY}_k{K_MIN_UNIQUE_LOCATIONS[PRIMARY]}" / "events.csv")
    ev_old_p = OUT_DIR / f"{OLD}_k{K_MIN_UNIQUE_LOCATIONS[OLD]}" / "events.csv"
    ev_old = pd.read_csv(ev_old_p) if ev_old_p.exists() else None
    gain = ev_new["lead_gain_min"] if "lead_gain_min" in ev_new else pd.Series(dtype=float)
    tim = pd.DataFrame([{
        "구분": f"{PRIMARY} (k={K_MIN_UNIQUE_LOCATIONS[PRIMARY]})", "Event 수": len(ev_new),
        "이전 방식이었다면 더 기다렸을 시간(분) 평균": float(gain.mean()) if len(gain) else np.nan,
        "중앙값": float(gain.median()) if len(gain) else np.nan, "75%": float(gain.quantile(.75)) if len(gain) else np.nan,
        "최대": float(gain.max()) if len(gain) else np.nan,
        "즉시 발동(앞당김 0분) 비율": float((gain == 0).mean()) if len(gain) else np.nan}])
    text += ("## 2. 발동 시점 개선의 효과\n\n"
             "새 방식은 k번째 고유 위치가 들어온 그 순간에 발동하고, 이전 방식은 첫 신고부터 30분을 채운 뒤에 발동한다. "
             "같은 입력 창이라면 새 방식이 항상 같거나 일찍 발동하며, 그 차이(경고를 앞당긴 분)는 아래와 같다.\n\n"
             f"{md(tim, '{:.1f}')}\n\n")
    perf = []
    for mode in (PRIMARY, OLD):
        d = data[mode]
        if d is None:
            continue
        for model in [REFERENCE, *MODELS[:2]]:
            r = d[(d["모델"] == model) & (d["지표"] == HIT_ALL)].iloc[0]
            perf.append({"발동 방식": f"{mode} (k={K_MIN_UNIQUE_LOCATIONS[mode]})", "모델": model,
                         "Hit@3(전체)": r["값"], "평가 Event 수": int(r["평가 Event 수"]),
                         "prior 대비 Δ [95%]": "(기준)" if r["판정"] == VERDICT_REF else cell(r).split("  ", 1)[1]})
    text += f"성능(같은 평가 틀, Hit@3(전체)):\n\n{md(pd.DataFrame(perf))}\n\n"
    text += ("주의: 두 방식은 Event 집합과 예측 시점이 달라서 Hit@3 값을 직접 비교할 수는 없다. "
             "이 표는 발동을 앞당겨도 결론(prior 대비 개선 여부)이 유지되는지를 보는 용도다.\n")
    (OUT_DIR / "scenario_compare.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
