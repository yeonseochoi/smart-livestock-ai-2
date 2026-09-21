"""세 mode(session / trigger / legacy_hour)의 전체 결과를 한 표로 모은다. run_v2.py 를 mode별로 먼저 실행해야 한다.

  trigger       : 주 운영 Event (최근 30분 안에 k번째 고유 위치의 신고가 들어온 순간 발동)
  session       : 보수적 보조 검증 (무신호 60분 뒤 시작한 세션의 첫 민원 후 30분)
  trigger_fixed : 이전 trigger (첫 신고부터 30분을 채운 뒤 발동) — 발동 시점 개선의 효과를 재는 비교용
  legacy_hour   : 기존 정시 Event를 같은 v2 평가 틀(rolling-origin, LOO prior)에 넣은 참고값
"""
from __future__ import annotations

import pandas as pd

from cand_coverage import HEADLINE_COLS
from config import K_MIN_UNIQUE_LOCATIONS, LEGACY_README, MODE_ROLE, OUT_DIR
from run_v2 import md

# 주 운영 trigger 를 맨 위에 둔다. session 은 보수적 보조 검증, trigger_fixed 는 이전 정의(비교용), legacy_hour 는 참고용.
MODES = [("trigger", K_MIN_UNIQUE_LOCATIONS["trigger"]), ("session", K_MIN_UNIQUE_LOCATIONS["session"]),
         ("trigger_fixed", K_MIN_UNIQUE_LOCATIONS["trigger_fixed"]), ("legacy_hour", 0)]


def main() -> None:
    parts, folds, covs = [], [], []
    for mode, k in MODES:
        d = OUT_DIR / f"{mode}_k{k}"
        if not (d / "results_pooled.csv").exists():
            print(f"건너뜀: {d} (run_v2.py --mode {mode} 먼저 실행)")
            continue
        tab = pd.read_csv(d / "results_pooled.csv")
        tab.insert(0, "mode", f"{mode}" + (f" (k={k})" if k else "") + f" [{MODE_ROLE[mode]}]")
        parts.append(tab)
        ft = pd.read_csv(d / "fold_table.csv")
        ft.insert(0, "mode", f"{mode}" + (f" (k={k})" if k else "") + f" [{MODE_ROLE[mode]}]")
        folds.append(ft)
        cv = pd.read_csv(d / "coverage.csv").head(1)
        cv.insert(0, "mode", f"{mode}" + (f" (k={k})" if k else "") + f" [{MODE_ROLE[mode]}]")
        covs.append(cv)
    allt = pd.concat(parts, ignore_index=True)
    cols = ["mode", "모델", "Hit@3(전체)", "Hit@3(전체) 95%CI", "Hit@3(전체) - prior기준선", "전체 차이 95%CI", "Recall@3(전체)",
            "Hit@3(후보내)", "Hit@3(후보내) 95%CI", "Hit@3(후보내) - prior기준선", "후보내 차이 95%CI", "Recall@3(후보내)",
            "PR-AUC(후보내)", "ROC-AUC(후보내)", "평가 Event 수(전체)", "평가 Event 수(후보내)", "테스트 Event 수"]
    legacy = pd.DataFrame([{"방식": "기존 README (단일 70/30, 정시 Event, 1km)", **LEGACY_README}])
    sweep = pd.read_csv(OUT_DIR / "k_sweep.csv")
    text = f"""# mode 비교 요약

## 전체(모든 테스트 폴드 합산)
(전체) = 예측 창에 민원이 있는 모든 Event, 정답이 전부 후보 밖이면 실패로 센다 / (후보내) = 후보 안에 정답이 있는 Event만 (조건부)
{md(allt[cols])}

## 후보 밖 실제 민원 비율 (예측 창의 실제 민원 중 후보 격자 밖에서 생긴 비율, Event 단위 부트스트랩 95%)
{md(pd.concat(covs, ignore_index=True)[["mode"] + [c for c in HEADLINE_COLS if c != "구분"]], "{:.3f}")}

## 기존 README 수치
{md(legacy)}

## 폴드표
{md(pd.concat(folds, ignore_index=True))}

## k 후보 비교 (성능 없이 개수·구성만)
{md(sweep, "{:.2f}")}
"""
    (OUT_DIR / "summary.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
