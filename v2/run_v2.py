"""v2 파이프라인: 세션 Event -> rolling-origin 평가 -> 기준선 3개 + 모델 비교 -> report.md

실행 예:
    python sweep_k.py                       # k 후보 비교표 (한 번)
    python run_v2.py                        # 주 운영 Event(trigger, k=7)
    python run_v2.py --mode session         # 보수적 보조 검증(session, k=4)
    python run_v2.py --mode trigger_fixed   # 이전 trigger(비교용)
    python run_v2.py --robust               # k 를 ±1 바꿔도 결론이 유지되는지

순서: (1) Event/폴드표 출력  (2) 폴드마다 prior 재계산(LOO)  (3) 기준선 + 모델 평가  (4) 결과 저장.
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb

from config import (DEFAULT_MODE, K_MIN_UNIQUE_LOCATIONS, LEGACY_README, MODE_ROLE, N_BOOTSTRAP, OUT_DIR, SEEDS, XGB_PARAMS)
from data import ComplaintIndex, load_complaints
from cand_coverage import HEADLINE_COLS, coverage_table
from evaluate import bootstrap_ci, evaluate, per_event_arrays
from events import build_events
from features import add_prior, build_event_rows
from folds import make_folds

warnings.filterwarnings("ignore")

from experiments import EXPERIMENTS as MODELS, REFERENCE as PRIOR_NAME  # noqa: E402  (실험 목록은 experiments.py 에서 관리)


def fit_xgb(train: pd.DataFrame, test: pd.DataFrame, feats: list[str]) -> np.ndarray:
    pos = int(train["target"].sum())
    spw = (len(train) - pos) / max(pos, 1)
    preds = []
    for seed in SEEDS:
        model = xgb.XGBClassifier(**XGB_PARAMS, scale_pos_weight=spw, random_state=seed, tree_method="hist")
        model.fit(train[feats], train["target"])
        preds.append(model.predict_proba(test[feats])[:, 1])
    return np.mean(preds, axis=0)


def score_models(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name, spec in MODELS:
        if spec is None:
            out[name] = np.zeros(len(test))                       # 전부 동점 -> 무작위 기대값
        elif isinstance(spec, str):
            out[name] = test[spec].to_numpy(float)
        else:
            out[name] = fit_xgb(train, test, spec)
    return out


def run(ci: ComplaintIndex, k: int, mode: str, verbose: bool = True):
    events = build_events(ci, k, mode)
    folds, fold_table = make_folds(events)
    if verbose:
        print(f"[Event] mode={mode}, k={k}: {len(events)}개 (예측창에 민원 있음 {events['has_future'].mean():.0%}, "
              f"후보 안 정답 있음 {events['has_positive'].mean():.0%}, 정답이 전부 후보 밖 {events['outside_only'].mean():.0%}, "
              f"후보격자 평균 {events['n_candidates'].mean():.1f}, 후보 안 정답 격자 평균 {events['n_positive_cells'].mean():.2f})\n")
        print("[폴드표] (평가 전에 먼저 확인)")
        print(fold_table.to_string(index=False), "\n")
    if not folds:
        raise SystemExit("사용 가능한 폴드가 없습니다. k 를 낮추거나 MIN_TRAIN_EVENTS 를 조정하세요.")

    rows = pd.concat(
        [build_event_rows(ci, int(r.start_idx), r.event_id, getattr(r, "t0_override", None)) for r in events.itertuples()],
        ignore_index=True
    )
    by_fold, pooled_parts = [], []
    for fold in folds:
        tr = rows[rows["event_id"].isin(fold.train_ids)]
        te = rows[rows["event_id"].isin(fold.test_ids)]
        assert tr["t0"].max() < fold.start <= te["t0"].min(), "시간 순서 위반"
        tr, te = add_prior(tr, te)
        scores = score_models(tr, te)
        for name, sc in scores.items():
            by_fold.append({"폴드": fold.name, "표본": "충분" if fold.reliable else "작음(참고용)", "모델": name, **evaluate(te, sc)})
        part = te[["event_id", "grid_x", "grid_y", "target", "future_cnt", "n_future_all", "prior"]].copy()
        part.insert(0, "fold", fold.name)
        for name, sc in scores.items():
            part[name] = sc
        pooled_parts.append(part)
        if verbose:
            print(f"  fold {fold.name}: 학습 {tr['event_id'].nunique()} / 테스트 {te['event_id'].nunique()} Event 완료")

    pooled = pd.concat(pooled_parts, ignore_index=True)
    return events, folds, fold_table, pd.DataFrame(by_fold), pooled


def pooled_table(pooled: pd.DataFrame) -> pd.DataFrame:
    """전체(모든 테스트 폴드 합산) 지표 + Event 단위 부트스트랩 구간 + prior 기준선 대비 차이.

    (전체): 예측 창에 민원이 있는 모든 Event, 정답이 전부 후보 밖이면 실패로 센다.
    (후보내): 후보 격자 안에 정답이 있는 Event만 (조건부).
    """
    base = {u: per_event_arrays(pooled, pooled[PRIOR_NAME].to_numpy(float), universe=u) for u in ("all", "cand")}
    out = []
    for name, _ in MODELS:
        sc = pooled[name].to_numpy(float)
        m = evaluate(pooled, sc)
        h_all, r_all = per_event_arrays(pooled, sc, universe="all")
        h_c, r_c = per_event_arrays(pooled, sc, universe="cand")
        ci_ha = bootstrap_ci(h_all); ci_ra = bootstrap_ci(r_all)
        ci_hc = bootstrap_ci(h_c); ci_rc = bootstrap_ci(r_c)
        d_a = bootstrap_ci(h_all - base["all"][0], seed=1); d_c = bootstrap_ci(h_c - base["cand"][0], seed=1)
        out.append({
            "모델": name,
            "Hit@3(전체)": m["Hit@3(전체)"], "Hit@3(전체) 95%CI": f"{ci_ha[0]:.2f}~{ci_ha[1]:.2f}",
            "Hit@3(전체) - prior기준선": float(np.mean(h_all - base["all"][0])), "전체 차이 95%CI": f"{d_a[0]:+.2f}~{d_a[1]:+.2f}",
            "Recall@3(전체)": m["Recall@3(전체)"], "Recall@3(전체) 95%CI": f"{ci_ra[0]:.2f}~{ci_ra[1]:.2f}",
            "Hit@3(후보내)": m["Hit@3(후보내)"], "Hit@3(후보내) 95%CI": f"{ci_hc[0]:.2f}~{ci_hc[1]:.2f}",
            "Hit@3(후보내) - prior기준선": float(np.mean(h_c - base["cand"][0])), "후보내 차이 95%CI": f"{d_c[0]:+.2f}~{d_c[1]:+.2f}",
            "Recall@3(후보내)": m["Recall@3(후보내)"], "Recall@3(후보내) 95%CI": f"{ci_rc[0]:.2f}~{ci_rc[1]:.2f}",
            "PR-AUC(후보내)": m["PR-AUC(후보내)"], "ROC-AUC(후보내)": m["ROC-AUC(후보내)"],
            "평가 Event 수(전체)": m["평가 Event 수(전체)"], "평가 Event 수(후보내)": m["평가 Event 수(후보내)"],
            "테스트 Event 수": m["테스트 Event 수"],
        })
    return pd.DataFrame(out)


def md(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    def cell(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else floatfmt.format(v)
        return str(v)
    head = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "|".join("---" for _ in df.columns) + "|\n"
    return head + "\n".join("| " + " | ".join(cell(v) for v in r) + " |" for r in df.itertuples(index=False))


def write_report(path, mode, k, events, fold_table, by_fold, pooled_tab, k_sweep, cov) -> None:
    legacy = pd.DataFrame([{"평가 방식": "기존 README (단일 70/30, 정시 Event, 1km)", **{c: v for c, v in LEGACY_README.items()}}])
    rows = []
    for label, model in (("기존15변수", "XGB 기존15변수(LOO prior)"), ("15변수+이력", "XGB 15변수+이력")):
        r = pooled_tab[pooled_tab["모델"] == model].iloc[0]
        rows.append({"평가 방식": f"v2 rolling-origin ({mode}, k={k}), {label}", "ROC-AUC": r["ROC-AUC(후보내)"],
                     "PR-AUC": r["PR-AUC(후보내)"], "Recall@3": r["Recall@3(후보내)"], "Hit@3": r["Hit@3(후보내)"],
                     "평가 Event 수": r["평가 Event 수(후보내)"]})
    v2 = pd.DataFrame(rows)
    role = MODE_ROLE.get(mode, "")
    text = f"""# v2 Event/평가 재설계 결과 (mode={mode} [{role}], k={k})

이 파일은 `run_v2.py` 가 자동 생성한다. 수치는 실행할 때마다 코드/데이터에서 다시 계산된다.

## 1. Event 정의
- 익산시 민원만, 1km 격자, 입력 30분 / 예측 30분 (기존 유지)
- mode=trigger(주 운영): 최근 30분 안에 k번째 고유 위치의 신고가 들어온 순간(그 분이 끝날 때) 예측. 입력 = 최근 30분, 라벨 = 다음 30분.
  발동 뒤 60분간은 재발동하지 않는다(앞 Event의 라벨이 다음 입력에 들어가지 않게).
- mode=session(보수적 보조 검증): 직전 민원 이후 60분 이상 조용하다가 들어온 첫 민원 = t0, 예측 시점 = t0+30분
- mode=trigger_fixed(이전 방식, 비교용): 첫 신고부터 30분을 채운 뒤 발동
- 선정 조건은 예측 시점에 알 수 있는 값만 사용: 초기 30분 내 서로 다른 30m 위치 수 >= {k}
  (위치 수는 그 창 안의 신고끼리만 30m 연결로 병합해 센다 — 전체 기간 군집 ID를 쓰지 않는다)
- 예측 창의 민원 수는 선정에 쓰지 않는다 -> 추가 민원이 없는 Event도 포함 (선정 편향 제거)

## 2. 후보 밖 실제 민원 비율 (모든 성능표와 함께 보아야 하는 값)
후보 = 입력 30분 관측 격자 주변 반경 2칸. 정답이 후보 밖에 있으면 모델이 올릴 방법이 없다. 구간은 Event 단위 부트스트랩 95%.
{md(cov[HEADLINE_COLS], "{:.3f}")}

후보 반경을 넓히면 정답이 얼마나 더 후보에 들어오는지(포함률):

{md(cov[["구분", "반경1 포함률", "반경2 포함률", "반경3 포함률", "반경4 포함률"]], "{:.3f}")}

## 3. k 후보 비교 (성능이 아닌 개수·구성 기준)
{md(k_sweep, "{:.2f}")}

## 4. 폴드표 (rolling-origin, expanding window, 기간을 합치지 않음)
{md(fold_table)}

## 5. 폴드별 결과
{md(by_fold)}

## 6. 전체(모든 테스트 폴드 합산) 결과
지표 이름의 (전체)는 예측 창에 민원이 있는 모든 Event 기준(정답이 전부 후보 밖이면 실패), (후보내)는 후보 안에 정답이 있는 Event만 기준(조건부)이다.

{md(pooled_tab)}

## 7. 기존 README 수치와 비교
기존 수치는 (후보내) 조건부 지표와 같은 방식이다.

{md(legacy)}

{md(v2)}

주의: 기존 수치는 Event 정의(정시 슬라이스 + 예측 구간 포함 선정)와 분할(단일 70/30)이 달라 이 표와 직접 비교할 수 없다.
Event 정의와 평가 방식 중 어느 쪽이 차이를 만드는지는 outputs/summary.md 의 mode 비교(legacy_hour 포함)를 본다.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default=DEFAULT_MODE, choices=["trigger", "session", "trigger_fixed", "legacy_hour"])
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--robust", action="store_true", help="k-1, k, k+1 로 결론이 유지되는지 확인")
    args = ap.parse_args()
    k = args.k or K_MIN_UNIQUE_LOCATIONS.get(args.mode, 0)

    df = load_complaints()
    ci = ComplaintIndex(df)
    out_dir = OUT_DIR / f"{args.mode}_k{k}"
    out_dir.mkdir(parents=True, exist_ok=True)

    events, folds, fold_table, by_fold, pooled = run(ci, k, args.mode)
    fold_table.to_csv(out_dir / "fold_table.csv", index=False, encoding="utf-8-sig")
    events.drop(columns=["start_idx", "t0_min"]).to_csv(out_dir / "events.csv", index=False, encoding="utf-8-sig")
    by_fold.to_csv(out_dir / "results_by_fold.csv", index=False, encoding="utf-8-sig")
    tab = pooled_table(pooled)
    tab.to_csv(out_dir / "results_pooled.csv", index=False, encoding="utf-8-sig")

    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.3f}".format):
        print("\n[폴드별]"); print(by_fold.to_string(index=False))
        print("\n[전체]"); print(tab.to_string(index=False))

    sweep_path = OUT_DIR / "k_sweep.csv"
    k_sweep = pd.read_csv(sweep_path) if sweep_path.exists() else pd.DataFrame()
    cov = coverage_table(events, {f"테스트 {f.name}": f.test_ids for f in folds})
    cov.to_csv(out_dir / "coverage.csv", index=False, encoding="utf-8-sig")
    write_report(out_dir / "report.md", args.mode, k, events, fold_table, by_fold, tab, k_sweep, cov)
    print(f"\n저장: {out_dir}")

    if args.robust:
        rows = []
        for kk in (k - 1, k, k + 1):
            try:
                _, _, _, _, pol = run(ci, kk, args.mode, verbose=False)
            except SystemExit:
                continue
            t = pooled_table(pol)
            for name in (PRIOR_NAME, "XGB 기존15변수(LOO prior)", "XGB 15변수+이력"):
                r = t[t["모델"] == name].iloc[0]
                rows.append({"k": kk, "모델": name, "Hit@3(전체)": r["Hit@3(전체)"], "Recall@3(전체)": r["Recall@3(전체)"],
                             "Hit@3(후보내)": r["Hit@3(후보내)"], "PR-AUC(후보내)": r["PR-AUC(후보내)"],
                             "평가 Event 수(전체)": r["평가 Event 수(전체)"]})
        rob = pd.DataFrame(rows)
        rob.to_csv(out_dir / "robustness_k.csv", index=False, encoding="utf-8-sig")
        print("\n[k 민감도 — k 선택에 쓰지 않은 참고용]")
        print(rob.to_string(index=False))


if __name__ == "__main__":
    main()
