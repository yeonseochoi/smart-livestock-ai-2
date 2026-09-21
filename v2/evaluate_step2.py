"""2단계: 평가 기준 교체.

  0) 후보 밖 실제 민원 비율(격자·신고 건수·Event 기준, 구간 포함): 모든 성능표와 함께 본다
  1) 모든 지표를 prior 기준선 대비 개선폭 + Event 단위 부트스트랩 95% 구간으로 보고
       (전체)  = 예측 창에 민원이 있는 모든 Event, 정답이 전부 후보 밖이면 실패로 센다
       (후보내) = 후보 안에 정답이 있는 Event만 (조건부)
  2) 신고 위치 중복 제거 라벨(반복 신고 위치를 걷어낸 뒤에도 맞히는가)
  3) 핫스팟 안 순위: prior 가 높은 격자들끼리만 놓고, 어느 격자가 활성화되는지 맞히는가
  4) Event 선정: 고유 위치 수 기준 vs 신고 건수 기준 비교
  5) 폴드(반기)별 결과: 합치지 않고 그대로, 표본이 작은 폴드는 Wilson 구간으로 넓게 보고
     + 2022년까지 vs 2023년 이후 시기 변화(성능 변화와 '모델이 prior보다 낫다'는 우위의 변화)

새 실험(4단계 변수 그룹)은 experiments.py 에 한 줄만 추가하면 아래 모든 표에 한 줄로 들어간다.

실행: python evaluate_step2.py            (주 운영 trigger, k는 config 의 규칙 값)
      python evaluate_step2.py --mode session   (보수적 보조 검증)
      python evaluate_step2.py --mode trigger_fixed   (이전 trigger, 비교용)
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

from cand_coverage import HEADLINE_COLS, coverage_row, coverage_table
from config import K_MIN_UNIQUE_LOCATIONS, MODE_ROLE, OUT_DIR, RELIABLE_MIN_TEST_EVENTS
from data import ComplaintIndex, load_complaints
from dedup import LABEL_VARIANTS, future_loc_sets, heavy_locations, relabel, relabel_totals
from evaluate import bootstrap_ci, per_event_arrays, wilson_interval
from evaluation import AUC_EV, HIT_ALL, HIT_C, PR, REC_ALL, REC_C, ROC, compare, wide
from events import trigger_events
from experiments import EXPERIMENTS, REFERENCE
from period_analysis import period_shift
from run_v2 import md, run

warnings.filterwarnings("ignore")
ORDER = [name for name, _ in EXPERIMENTS]
MAIN_METRICS = [HIT_ALL, REC_ALL, HIT_C, REC_C, PR, ROC]
HOTSPOT_METRICS = [HIT_C, REC_C, AUC_EV, PR]
HOTSPOT_THETAS = (0.10, 0.15, 0.25)
LEGEND = ("칸 형식: 값  Δ차이 [95% 구간]  ▲개선(구간이 0 초과) ▼악화(구간이 0 미만) ○구분 불가(구간이 0을 포함)\n"
          "지표: (전체)=예측 창에 민원이 있는 모든 Event, 정답이 전부 후보 밖이면 실패 / (후보내)=후보 안에 정답이 있는 Event만(조건부)")


def selection_compare(ci: ComplaintIndex, base_events: pd.DataFrame, k: int) -> pd.DataFrame:
    """Event 선정 기준을 바꿨을 때: 고유 위치 수(k) vs 신고 건수(r)."""
    base_t0 = np.sort(base_events["t0_min"].to_numpy())
    rows = []
    for label, unit, thr in [(f"고유 위치 수 >= {k}", "locs", k)] + [(f"신고 건수 >= {r}", "reports", r) for r in (8, 10, 12, 15)]:
        ev = base_events if unit == "locs" else trigger_events(ci, thr, unit)
        t0 = ev["t0_min"].to_numpy()
        pos = np.searchsorted(base_t0, t0)
        near = np.minimum(np.abs(base_t0[np.clip(pos, 0, len(base_t0) - 1)] - t0),
                          np.abs(base_t0[np.clip(pos - 1, 0, len(base_t0) - 1)] - t0))
        rows.append({
            "선정 기준": label, "Event 수": len(ev),
            "예측창에 민원 있음 비율": ev["has_future"].mean(),
            "후보 안 정답 있음 비율": ev["has_positive"].mean(),
            f"고유 위치 {k}곳 미만인 Event 비율": float((ev["n_init_locs"] < k).mean()),
            "고유 위치/신고 건수(중앙값)": float((ev["n_init_locs"] / ev["n_init_reports"]).median()),
            "고유 위치 수(중앙값)": float(ev["n_init_locs"].median()),
            "위치 기준 Event와 30분 내 겹침 비율": float((near <= 30).mean()),
        })
    return pd.DataFrame(rows)


def fold_table_hit(pooled: pd.DataFrame, folds, reference: str, boot: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """폴드(반기)별 Hit@3(전체): 값, Wilson 95% 구간, 기준선 대비 차이. 합치지 않은 폴드를 그대로 보고한다.

    표본이 RELIABLE_MIN_TEST_EVENTS 미만인 폴드는 '참고용'으로 표시하고 차이의 구간은 내지 않는다.
    마지막 줄 '2023년 이후 반기 합산'은 보고용 합산일 뿐 학습·폴드 구성에는 영향이 없다.
    """
    groups = [(f.name, f.start, pooled[pooled["fold"] == f.name]) for f in folds]
    late = pooled[pooled["fold"].isin([f.name for f in folds if f.start >= pd.Timestamp("2023-01-01")])]
    groups.append(("(합산) 2023년 이후 반기들", None, late))
    long_rows, table_rows = [], []
    for name, _, frame in groups:
        frame = frame.reset_index(drop=True)
        base_h, _ = per_event_arrays(frame, frame[reference].to_numpy(float), universe="all")
        n = len(base_h)
        row = {"폴드": name, "평가 Event(전체)": n, "표본": ("충분" if n >= RELIABLE_MIN_TEST_EVENTS else f"작음(<{RELIABLE_MIN_TEST_EVENTS}, 참고용)")}
        for m in ORDER:
            h, _ = per_event_arrays(frame, frame[m].to_numpy(float), universe="all")
            val = float(h.mean()) if n else np.nan
            lo, hi = wilson_interval(float(h.sum()), n)
            d = h - base_h
            d_lo, d_hi = bootstrap_ci(d, n=boot, seed=2) if n >= RELIABLE_MIN_TEST_EVENTS else (np.nan, np.nan)
            if m == reference:
                cell = f"{val:.2f} [{lo:.2f}, {hi:.2f}] (기준)" if n else "-"
            elif not n:
                cell = "-"
            elif np.isnan(d_lo):
                cell = f"{val:.2f} [{lo:.2f}, {hi:.2f}] Δ{d.mean():+.2f}"
            else:
                mark = "▲" if d_lo > 0 else ("▼" if d_hi < 0 else "○")
                cell = f"{val:.2f} [{lo:.2f}, {hi:.2f}] Δ{d.mean():+.2f} [{d_lo:+.2f}, {d_hi:+.2f}] {mark}"
            row[m] = cell
            long_rows.append({"폴드": name, "모델": m, "평가 Event(전체)": n, "Hit@3(전체)": val, "Wilson 하한": lo, "Wilson 상한": hi,
                              "기준선 대비 Δ": float(d.mean()) if n else np.nan, "Δ 하한": d_lo, "Δ 상한": d_hi})
        table_rows.append(row)
    return pd.DataFrame(table_rows), pd.DataFrame(long_rows)


def add_outside_column(fold_tab: pd.DataFrame, events: pd.DataFrame, pooled: pd.DataFrame, folds) -> pd.DataFrame:
    """반기별 표에 그 기간 테스트 Event의 후보 밖 비율(격자, 신고 건수)을 붙인다."""
    out = fold_tab.copy()
    cells, reps = [], []
    for name in out["폴드"]:
        if name.startswith("(합산)"):
            ids = pooled[pooled["fold"].isin([f.name for f in folds if f.start >= pd.Timestamp("2023-01-01")])]["event_id"].unique()
        else:
            ids = pooled[pooled["fold"] == name]["event_id"].unique()
        r = coverage_row(events[events["event_id"].isin(ids)], name, n_boot=1000)
        cells.append(r["후보 밖 비율(격자) 95%CI"]); reps.append(r["후보 밖 비율(신고 건수) 95%CI"])
    out.insert(3, "후보 밖 비율(격자) [95%CI]", cells)
    out.insert(4, "후보 밖 비율(신고 건수) [95%CI]", reps)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="trigger", choices=["trigger", "session", "trigger_fixed"])
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--boot", type=int, default=500, help="부트스트랩 반복 수")
    args = ap.parse_args()
    k = args.k or K_MIN_UNIQUE_LOCATIONS[args.mode]

    df = load_complaints()
    ci = ComplaintIndex(df)
    out_dir = OUT_DIR / f"step2_{args.mode}_k{k}"
    out_dir.mkdir(parents=True, exist_ok=True)

    events, folds, fold_table, by_fold, pooled = run(ci, k, args.mode, verbose=False)
    pooled = pooled.reset_index(drop=True)
    print(f"[Event] {args.mode} k={k}: {len(events)}개, 폴드 {len(folds)}개, 테스트 Event {pooled['event_id'].nunique()}개, 후보 행 {len(pooled):,}")
    print(fold_table.to_string(index=False), "\n")
    scores = {name: pooled[name].to_numpy(float) for name in ORDER}
    parts: list[str] = [f"### 폴드표 (기간을 합치지 않음)\n\n{md(fold_table)}\n"]
    long_all: list[pd.DataFrame] = []

    def block(title: str, frame: pd.DataFrame, metrics: list[str], min_cands: int = 0, note: str = "") -> pd.DataFrame:
        sc = {m: scores[m][frame.index.to_numpy()] for m in ORDER}
        long = compare(frame, sc, REFERENCE, n_boot=args.boot, min_cands=min_cands)
        long.insert(0, "블록", title)
        long_all.append(long)
        tab = wide(long, metrics, ORDER)
        print(f"=== {title} ===\n{note}\n{tab.to_string(index=False)}\n")
        parts.append(f"### {title}\n{note}\n\n{md(tab)}\n")
        return long

    # ---- 0. 후보 포함률 ----
    cov = coverage_table(events, {f"테스트 {f.name}": f.test_ids for f in folds}, n_boot=1000)
    cov_all = pd.concat([cov.head(1), coverage_table(events[events["event_id"].isin(pooled["event_id"].unique())], n_boot=1000).assign(구분="모든 테스트 Event")],
                        ignore_index=True)
    parts.append(f"### 0. 후보 밖 실제 민원 비율 (모든 성능표와 함께 보아야 하는 값)\n"
                 f"후보 = 입력 30분 관측 격자 주변 반경 2칸. 예측 창의 실제 민원 중 후보 밖에서 생긴 비율이며, 이만큼은 어떤 모델도 순위에 올릴 수 없다. "
                 f"구간은 Event 단위 부트스트랩 95%.\n\n{md(cov_all[HEADLINE_COLS], '{:.3f}')}\n\n"
                 f"후보 반경을 넓히면 정답이 얼마나 더 들어오는지(포함률):\n\n"
                 f"{md(cov_all[['구분', '반경1 포함률', '반경2 포함률', '반경3 포함률', '반경4 포함률']], '{:.3f}')}\n\n"
                 f"폴드(반기)별:\n\n{md(cov[HEADLINE_COLS], '{:.3f}')}\n")
    print("=== 0. 후보 밖 실제 민원 비율 ===\n", cov_all[HEADLINE_COLS].T.to_string(header=False), "\n")

    # ---- A. 전체 (원본 라벨) ----
    block("A. 전체 후보 · 원본 라벨", pooled, MAIN_METRICS,
          note=f"기준선 = {REFERENCE}. (전체) 지표는 후보 밖에만 정답이 있는 Event도 실패로 센다.")

    # ---- A2. 폴드별 ----
    fold_tab, fold_long = fold_table_hit(pooled, folds, REFERENCE, args.boot)
    fold_tab = add_outside_column(fold_tab, events, pooled, folds)
    parts.append(f"### A2. 폴드(반기)별 {HIT_ALL}과 후보 밖 비율\n칸 형식: 값 [Wilson 95% 구간] Δ기준선대비 [부트스트랩 95% 구간] ▲▼○. "
                 f"표본이 작은 폴드는 구간을 넓게 보고 Δ의 구간은 생략한다(참고용). 마지막 줄은 2023년 이후 반기들의 보고용 합산이다.\n\n{md(fold_tab)}\n")
    print(f"=== A2. 폴드별 {HIT_ALL} ===\n{fold_tab.to_string(index=False)}\n")

    # ---- A3. 시기 변화 ----
    shift = period_shift(pooled, folds, ORDER, REFERENCE, n_boot=max(args.boot, 1000))
    parts.append(f"### A3. 시기 변화: 2022년까지 vs 2023년 이후 ({HIT_ALL})\n"
                 f"'시기 변화'는 (2023년 이후 평균) - (2022년까지 평균). '우위의 변화'는 모델이 prior 대비 얻는 이득이 시기에 따라 달라졌는지. "
                 f"서로 다른 Event 집합이라 두 묶음을 독립적으로 복원추출한 95% 구간이다. ▲▼는 구간이 0을 벗어난 경우.\n\n{md(shift)}\n")
    print(f"=== A3. 시기 변화 ===\n{shift.to_string(index=False)}\n")
    shift.to_csv(out_dir / "period_shift.csv", index=False, encoding="utf-8-sig")

    # ---- B. 신고 위치 중복 제거 라벨 ----
    fut = future_loc_sets(ci, events)
    variant_counts = []
    for name, top_n, min_locs in LABEL_VARIANTS:
        drop = heavy_locations(ci, top_n)
        lab = relabel(pooled, fut, drop, min_locs)
        totals = relabel_totals(fut, drop, min_locs)
        variant_counts.append({"라벨": name, "양성 격자-Event 수": int(lab.sum()),
                               "원본 대비": lab.sum() / max(int(pooled["target"].sum()), 1),
                               "후보 안 정답 있는 Event 수": int(pooled.assign(t=lab).groupby("event_id")["t"].max().sum()),
                               "정답이 있는 Event 수(후보 밖 포함)": int(sum(totals[e] > 0 for e in pooled["event_id"].unique()))})
        if top_n == 0 and min_locs == 1:
            assert (lab == pooled["target"].to_numpy()).all(), "원본 라벨 재현 실패"
            assert all(int(totals[e]) == int(g["n_future_all"].iloc[0]) for e, g in pooled.groupby("event_id")), "전체 정답 수 재현 실패"
            continue
        v = pooled.copy()
        v["target"] = lab
        v["n_future_all"] = v["event_id"].map(totals).astype(float)
        block(f"B. 중복 제거 · {name}", v, [HIT_ALL, REC_ALL, HIT_C, REC_C, PR, ROC],
              note="라벨만 바꿨다(변수·모델 점수는 그대로). 기준선도 같은 라벨로 다시 채점. 상위 N개 위치는 전체 기간 기준의 진단용 규칙.")
    vc = pd.DataFrame(variant_counts)
    parts.append(f"### 라벨 변형별 양성 수\n\n{md(vc)}\n")
    print("=== 라벨 변형별 양성 수 ===\n", vc.to_string(index=False), "\n")

    # ---- C. 핫스팟 안 순위 ----
    hot_rows = []
    for theta in HOTSPOT_THETAS:
        sub = pooled[pooled["prior"] >= theta]
        n_c = sub.groupby("event_id").size()
        hot_rows.append({"θ(prior 하한)": theta, "핫스팟 후보 행": len(sub),
                         "Event당 핫스팟 후보(평균)": n_c.mean(), "핫스팟 후보>3인 Event 수": int((n_c > 3).sum()),
                         "핫스팟 안 양성률": sub["target"].mean()})
        block(f"C. 핫스팟 안 순위 · prior >= {theta}", sub.drop(columns="n_future_all"), HOTSPOT_METRICS, min_cands=3,
              note="prior 가 θ 이상인 후보 격자만 놓고 평가한다(이 블록의 (후보내)는 '핫스팟 후보 안'을 뜻한다). 후보가 3개 이하인 Event는 제외.")
    hc = pd.DataFrame(hot_rows)
    parts.append(f"### 핫스팟 정의별 규모\n\n{md(hc)}\n")
    print("=== 핫스팟 정의별 규모 ===\n", hc.to_string(index=False), "\n")

    # ---- D. Event 선정 기준 비교 ----
    sel = selection_compare(ci, events, k)
    parts.append(f"### D. Event 선정: 고유 위치 수 vs 신고 건수\n\n{md(sel, '{:.2f}')}\n")
    print("=== D. Event 선정 비교 ===\n", sel.to_string(index=False), "\n")

    long = pd.concat(long_all, ignore_index=True)
    long.to_csv(out_dir / "eval_long.csv", index=False, encoding="utf-8-sig")
    fold_long.to_csv(out_dir / "fold_hit_long.csv", index=False, encoding="utf-8-sig")
    cov.to_csv(out_dir / "coverage.csv", index=False, encoding="utf-8-sig")
    sel.to_csv(out_dir / "event_selection_compare.csv", index=False, encoding="utf-8-sig")
    vc.to_csv(out_dir / "label_variants.csv", index=False, encoding="utf-8-sig")
    hc.to_csv(out_dir / "hotspot_definitions.csv", index=False, encoding="utf-8-sig")
    text = (f"# 2단계 평가 결과 (mode={args.mode} [{MODE_ROLE.get(args.mode, '')}], k={k}, 부트스트랩 {args.boot}회)\n\n"
            f"기준선: **{REFERENCE}**. {LEGEND}\n\n" + "\n".join(parts))
    (out_dir / "eval_report.md").write_text(text, encoding="utf-8")
    print(f"저장: {out_dir}")


if __name__ == "__main__":
    main()
