"""1단(발생 위험 예보)·2단(확산 예측) 결합 평가 — "2단 Top 3 중 1개를 최우선으로 고를 수 있나".

규칙(후처리, 2단 모델·Top 3 구성은 그대로): 2단 Top 3 격자 중 민원 접수 1시간 전(event_hour-1h) 1단 위험 순위가
가장 높은 격자를 1순위로 올린다. 나머지는 2단 점수 순서. Top 3 집합이 같으므로 Hit@3는 정의상 보존된다.

입력
  outputs/operational_grid_comparison/test_predictions.csv (grid_m=1000, 2단 테스트 Event의 후보 격자 점수)
  outputs/onset_risk/onset_event_scores.csv + onset_event_scores_fold*.csv (run_onset_risk.py [--train-end …] 산출, 전체 라벨.
      Event마다 event_hour 이전 자료로만 학습한 분할 중 가장 늦은 것을 쓴다: 2021년 Event는 2021-01-01 분할, 2022년은 2022-01-01, … 누수 없음)
평가: 2단 단독 Hit@1/2/3 vs 결합 Hit@1. 결합이 고친 Event 수·망친 Event 수(부호 검정).
      2021년 Event는 1단 학습 자료가 2020년 1년뿐이라 2022년 이후 Event만 따로도 본다.
실행: python fuse_onset_spread.py [--pool 3] → outputs/onset_spread_fusion/{metrics.json, fusion_table.md, event_table.csv}
      python fuse_onset_spread.py --narrow 30 → narrow_table.md, narrow_metrics.json (아래 후보 축소 규칙)

후보 축소 규칙(--narrow K): 기존 2단 Top 3를 보호하고, 나머지 후보 중 event_hour-1h 1단 위험 순위가 K 이내인 격자를 남긴다.
      예외: 1단 격자 밖(민원 이력 없는 칸) 후보는 2단 점수가 K 안 1위보다 높을 때만 남긴다(새 지역 탈출구).
      2단 모델·점수와 기존 Top 3는 그대로다. 목적은 후보 수 축소이지 성능 향상이 아니다.
"""
from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

from administrative_agent.policy import narrow_candidates
from administrative_agent.onset import latest_event_scores

SPREAD_PATH = Path("outputs/operational_grid_comparison/test_predictions.csv")
ONSET_DIR = Path("outputs/onset_risk")
ONSET_PATTERNS = ("onset_event_scores.csv", "onset_event_scores_fold*.csv")  # 유형별(_livestock 등) 파일은 제외
OUTPUT_DIR = Path("outputs/onset_spread_fusion")
GRID_M = 1000
LATE_START = pd.Timestamp("2022-01-01")


def sign_test_p(wins: int, losses: int) -> float:
    """양측 부호 검정(이항, p=0.5). 고친 수 vs 망친 수."""
    n = wins + losses
    if n == 0:
        return float("nan")
    k = min(wins, losses)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return float(min(1.0, 2 * tail))


def load_onset_files() -> list[str]:
    return sorted(p.as_posix() for pattern in ONSET_PATTERNS for p in ONSET_DIR.glob(pattern))


def load_onset_scores() -> pd.DataFrame:
    files = load_onset_files()
    if not files:
        raise SystemExit(f"1단 Event 점수 파일 없음: {ONSET_DIR}/{ONSET_PATTERNS} (run_onset_risk.py 를 먼저 실행)")
    try:
        onset = latest_event_scores([Path(file) for file in files])
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if onset.empty:
        raise SystemExit("Event 이전 자료로 학습한 1단 점수가 없습니다.")
    return onset.rename(columns={"risk_score": "onset_score", "rank": "onset_rank"})[
        ["event_hour", "grid_x", "grid_y", "onset_score", "onset_rank", "train_end"]]


def fuse_event(group: pd.DataFrame, pool_size: int) -> dict:
    """한 Event의 2단 후보 표(onset_rank 포함)에서 2단 단독·결합 Top-1 적중을 계산한다."""
    top = group.nlargest(min(3, len(group)), "score").copy()  # 2단 Top 3 (점수 순)
    pool = top.head(pool_size)
    # 1단 순위가 없는 격자(1단 격자 밖)는 가장 뒤로. 동률이면 2단 점수 순서 유지(stable sort).
    fused_first = pool.assign(_r=pool["onset_rank"].fillna(np.inf)).sort_values("_r", kind="stable").iloc[0]
    true_ranks = top.loc[top["target"] == 1, "onset_rank"]
    false_ranks = top.loc[top["target"] == 0, "onset_rank"]
    return {
        "spread_top1_hit": int(top.iloc[0]["target"]), "spread_top2_hit": int(top.head(2)["target"].max()),
        "spread_top3_hit": int(top["target"].max()),
        "fused_top1_hit": int(fused_first["target"]), "fused_choice_spread_rank": int(pool.index.get_loc(fused_first.name)) + 1,
        "onset_rank_true_min": float(true_ranks.min()) if len(true_ranks) else float("nan"),
        "onset_rank_false_min": float(false_ranks.min()) if len(false_ranks) else float("nan"),
        "candidates": int(len(group)), "onset_covered": int(group["onset_rank"].notna().sum()),
    }


def summarize(table: pd.DataFrame, label: str) -> dict:
    wins = int(((table["fused_top1_hit"] == 1) & (table["spread_top1_hit"] == 0)).sum())
    losses = int(((table["fused_top1_hit"] == 0) & (table["spread_top1_hit"] == 1)).sum())
    return {
        "label": label, "events": int(len(table)),
        "spread_hit_at_1": float(table["spread_top1_hit"].mean()), "spread_hit_at_2": float(table["spread_top2_hit"].mean()),
        "spread_hit_at_3": float(table["spread_top3_hit"].mean()),
        "fused_hit_at_1": float(table["fused_top1_hit"].mean()),
        "fixed_events": wins, "broken_events": losses, "sign_test_p": sign_test_p(wins, losses),
    }


def narrow_event(group: pd.DataFrame, k: int) -> dict:
    """한 Event의 2단 후보를 1단 순위 K 이내로 좁힌 뒤 Top 3 적중. 규칙은 administrative_agent.policy.narrow_candidates."""
    ordered = group.sort_values("score", ascending=False)
    kept, info = narrow_candidates(group, k)
    top = kept.head(3)
    base = ordered.head(3)
    return {"candidates": info["candidates"], "kept": info["kept"], "filled": info["filled"], "escaped_outside": info["escaped_outside"],
            "base_hit1": int(base.iloc[0]["target"]), "base_hit2": int(base.head(2)["target"].max()), "base_hit3": int(base["target"].max()),
            "narrow_hit1": int(top.iloc[0]["target"]), "narrow_hit2": int(top.head(2)["target"].max()), "narrow_hit3": int(top["target"].max()),
            "truth_min_onset_rank": float(group.loc[group["target"] == 1, "onset_rank"].min())}


def run_narrow(merged: pd.DataFrame, k: int) -> None:
    rows = []
    for event_id, group in merged.groupby("event_id"):
        if group["target"].sum() == 0:
            continue
        r = narrow_event(group, k)
        r.update({"event_id": event_id, "event_hour": group["event_hour"].iloc[0]})
        rows.append(r)
    table = pd.DataFrame(rows).sort_values("event_hour")
    truth = merged[merged["target"] == 1]
    summary = {
        "k": k, "events": int(len(table)),
        "mean_candidates_before": float(table["candidates"].mean()), "mean_candidates_after": float(table["kept"].mean()),
        "events_filled": int((table["filled"] > 0).sum()), "events_with_outside_escape": int((table["escaped_outside"] > 0).sum()),
        "base": {f"hit_at_{i}": float(table[f"base_hit{i}"].mean()) for i in (1, 2, 3)},
        "narrow": {f"hit_at_{i}": float(table[f"narrow_hit{i}"].mean()) for i in (1, 2, 3)},
        "truth_onset_rank": {"outside_share": float(truth["onset_rank"].isna().mean()), "median": float(truth["onset_rank"].median()),
                             f"within_{k}": float((truth["onset_rank"] <= k).mean())},
    }
    for i in (1, 2, 3):
        summary[f"hit{i}_fixed"] = int(((table[f"narrow_hit{i}"] == 1) & (table[f"base_hit{i}"] == 0)).sum())
        summary[f"hit{i}_broken"] = int(((table[f"narrow_hit{i}"] == 0) & (table[f"base_hit{i}"] == 1)).sum())
    (OUTPUT_DIR / "narrow_metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    table.to_csv(OUTPUT_DIR / "narrow_event_table.csv", index=False, encoding="utf-8-sig")
    b, n = summary["base"], summary["narrow"]
    lines = [f"# 후보 축소 규칙: 기존 2단 Top 3 보호 + 1단 위험 순위 {k}위 이내", "",
             f"- 테스트 Event {summary['events']}개. 2단 모델·점수·기존 Top 3를 보존하고 나머지 후보만 좁힘. 1단 점수는 Event 이전 자료로만 학습한 연도별 분할.",
             f"- 후보 수 평균 {summary['mean_candidates_before']:.1f} → {summary['mean_candidates_after']:.1f}. "
             f"채움 발생 Event {summary['events_filled']}개, 1단 격자 밖 예외 허용 Event {summary['events_with_outside_escape']}개.",
             f"- 정답 격자의 1단 순위: 중앙값 {summary['truth_onset_rank']['median']:.0f}위, {k}위 이내 {summary['truth_onset_rank'][f'within_{k}']:.3f}, "
             f"1단 격자 밖 {summary['truth_onset_rank']['outside_share']:.3f}.", "",
             "| | Hit@1 | Hit@2 | Hit@3 |", "|---|---:|---:|---:|",
             f"| 2단 단독(현행) | {b['hit_at_1']:.3f} | {b['hit_at_2']:.3f} | {b['hit_at_3']:.3f} |",
             f"| 1단 상위 {k}로 축소 | {n['hit_at_1']:.3f} | {n['hit_at_2']:.3f} | {n['hit_at_3']:.3f} |",
             f"| 고침 / 망침 Event | {summary['hit1_fixed']}/{summary['hit1_broken']} | {summary['hit2_fixed']}/{summary['hit2_broken']} | {summary['hit3_fixed']}/{summary['hit3_broken']} |"]
    (OUTPUT_DIR / "narrow_table.md").write_text(chr(10).join(lines) + chr(10), encoding="utf-8")
    print(chr(10).join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=int, default=3, help="1단 순위로 재정렬할 2단 상위 후보 수(기본 3)")
    parser.add_argument("--narrow", type=int, default=0, help="후보 축소 규칙만 실행: 기존 Top 3 보호 + 1단 위험 순위 K 이내(0이면 안 함)")
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    spread = pd.read_csv(SPREAD_PATH, parse_dates=["event_hour"])
    spread = spread[spread["grid_m"] == GRID_M]
    onset = load_onset_scores()
    merged = spread.merge(onset, on=["event_hour", "grid_x", "grid_y"], how="left")
    if args.narrow:
        run_narrow(merged, args.narrow)
        return

    rows = []
    for event_id, group in merged.groupby("event_id"):
        if group["target"].sum() == 0:
            continue
        r = fuse_event(group, args.pool)
        r.update({"event_id": event_id, "event_hour": group["event_hour"].iloc[0],
                  "train_end": group["train_end"].dropna().iloc[0] if group["train_end"].notna().any() else pd.NaT})
        rows.append(r)
    table = pd.DataFrame(rows).sort_values("event_hour")
    covered = table[table["onset_covered"] > 0]
    uncovered_events = int((table["onset_covered"] == 0).sum())

    summaries = [summarize(covered, "1단 점수가 있는 Event 전체"),
                 summarize(covered[covered["event_hour"] >= LATE_START], f"{LATE_START.year}년 이후 Event(1단 학습 2년 이상)")]
    by_year = [summarize(g, str(y)) for y, g in covered.groupby(covered["event_hour"].dt.year)]
    # 1단 신호 진단: Top 3 안에 정답·오답이 함께 있는 Event에서 정답의 1단 순위가 오답 최소 순위보다 앞선 비율
    mixed = covered.dropna(subset=["onset_rank_true_min", "onset_rank_false_min"])
    diag = {"mixed_events": int(len(mixed)),
            "true_cell_ranked_above_false": float((mixed["onset_rank_true_min"] < mixed["onset_rank_false_min"]).mean()) if len(mixed) else float("nan"),
            "median_onset_rank_true": float(covered["onset_rank_true_min"].median()),
            "median_onset_rank_false": float(covered["onset_rank_false_min"].median())}
    result = {"protocol": {"spread_source": str(SPREAD_PATH), "grid_m": GRID_M, "pool": args.pool,
                           "onset_files": load_onset_files(), "rule": "2단 Top pool 중 1단(event_hour-1h) 순위 최상을 1순위로",
                           "events_total": int(len(table)), "events_without_onset_scores": uncovered_events,
                           "candidate_coverage": float(covered["onset_covered"].sum() / covered["candidates"].sum())},
              "summaries": summaries, "by_year": by_year, "onset_signal": diag}
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    table.to_csv(OUTPUT_DIR / "event_table.csv", index=False, encoding="utf-8-sig")

    lines = ["# 1단·2단 결합 평가: 2단 Top 3 중 1단 순위로 1순위 고르기", "",
             f"- 2단: {SPREAD_PATH} (1 km). 1단: {len(result['protocol']['onset_files'])}개 분할 파일, Event마다 event_hour 이전 분할 중 가장 늦은 것.",
             f"- Event {len(table)}개 중 1단 점수 없는 Event {uncovered_events}개. 후보 격자 중 1단 격자와 겹친 비율 {result['protocol']['candidate_coverage']:.3f}.",
             f"- 규칙: 2단 상위 {args.pool}개 중 1단 순위 최상 격자를 1순위로. Top 3 집합은 그대로(Hit@3 보존).", "",
             "| 구간 | Event | 2단 Hit@1 | 2단 Hit@2 | 2단 Hit@3 | 결합 Hit@1 | 고침 | 망침 | 부호검정 p |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in summaries + by_year:
        lines.append(f"| {s['label']} | {s['events']} | {s['spread_hit_at_1']:.3f} | {s['spread_hit_at_2']:.3f} | {s['spread_hit_at_3']:.3f} | "
                     f"{s['fused_hit_at_1']:.3f} | {s['fixed_events']} | {s['broken_events']} | {s['sign_test_p']:.2f} |")
    lines += ["", "## 1단 신호 진단", "",
              f"- Top 3에 정답·오답이 함께 있는 Event {diag['mixed_events']}개 중 정답 격자의 1단 순위가 오답보다 앞선 비율: {diag['true_cell_ranked_above_false']:.3f}",
              f"- 정답 격자 1단 순위 중앙값 {diag['median_onset_rank_true']:.0f}, 오답(Top 3 내) 격자 {diag['median_onset_rank_false']:.0f}"]
    (OUTPUT_DIR / "fusion_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
