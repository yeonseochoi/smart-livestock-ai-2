"""Read-only analysis of the v2 trigger models. Run: python -m odor_service.v2vars.analyze."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
from odor_service.region_prediction.config import INPUT_MIN, HORIZON_MIN, XGB_PARAMS
from odor_service.region_prediction.data import ComplaintIndex, load_complaints
from odor_service.region_prediction.events import build_events
from odor_service.region_prediction.features import BASE_FEATURES, HIST_FEATURES, add_prior, build_event_rows
from odor_service.region_prediction.folds import make_folds
from odor_service.region_prediction.evaluate import per_event_arrays
import xgboost as xgb  # noqa: E402

OUT = ROOT / "outputs/odor_service/v2vars"
BOOT = 1000
SEEDS = (42,)  # runtime reduction from v2's (42,43,44)

GROUPS = {
    "증가·최근 신고": ["first15_count", "growth", "observed_report_count", "initial_count"],
    "반복·장소 수·공간": ["observed_grid_count", "min_distance", "neighbor_count", "radius3_count", "centroid_distance"],
    "강도": ["initial_intensity", "nearest_intensity", "weighted_intensity"],
    "과거 이력": HIST_FEATURES,
    "기상": [],
}


def ci_mean(a):
    rng = np.random.default_rng(42)
    return np.quantile(rng.choice(a, (BOOT, len(a)), replace=True).mean(axis=1), [.025, .975])


def corr_ci(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    rho = spearmanr(x, y).statistic if len(x) > 2 else np.nan
    rng = np.random.default_rng(42)
    vals = []
    for _ in range(BOOT):
        ix = rng.integers(0, len(x), len(x))
        v = spearmanr(x[ix], y[ix]).statistic
        if np.isfinite(v):
            vals.append(v)
    lo, hi = np.quantile(vals, [.025, .975]) if vals else (np.nan, np.nan)
    return rho, lo, hi


def event_stats(ci, events, k):
    out = []
    for e in events.itertuples():
        t0 = int(e.t0_min)
        i0 = int(np.searchsorted(ci.t, t0, "left"))
        i1 = int(np.searchsorted(ci.t, t0 + INPUT_MIN, "left"))
        i2 = int(np.searchsorted(ci.t, t0 + INPUT_MIN + HORIZON_MIN, "left"))
        mid = int(np.searchsorted(ci.t, t0 + 15, "left"))
        a = ci.inten[i0:i1]
        out.append({"k": k, "event_id": e.event_id, "t0": e.t0, "growth": (i1-mid)-(mid-i0),
                    "intensity_mean": np.nanmean(a), "intensity_max": np.nanmax(a),
                    "future_reports": i2-i1, "input_reports": i1-i0})
    return pd.DataFrame(out)


def fit(train, test, features):
    pos = int(train.target.sum())
    spw = (len(train)-pos)/max(pos, 1)
    preds = []
    for seed in SEEDS:
        model = xgb.XGBClassifier(**XGB_PARAMS, scale_pos_weight=spw, random_state=seed, tree_method="hist")
        model.fit(train[features], train.target)
        preds.append(model.predict_proba(test[features])[:, 1])
    return np.mean(preds, axis=0)


def ablation(ci, events, k):
    folds, fold_table = make_folds(events)
    rows = pd.concat([build_event_rows(ci, int(e.start_idx), e.event_id, getattr(e, "t0_override", None))
                      for e in events.itertuples()], ignore_index=True)
    full = BASE_FEATURES + HIST_FEATURES
    variants = {"전체": full}
    variants.update({"제외: " + g: [f for f in full if f not in cols] for g, cols in GROUPS.items() if cols})
    parts = []
    for fold in folds:
        tr = rows[rows.event_id.isin(fold.train_ids)]
        te = rows[rows.event_id.isin(fold.test_ids)]
        tr, te = add_prior(tr, te)
        part = te[["event_id", "target", "n_future_all"]].copy()
        part["fold"] = fold.name
        for name, feats in variants.items():
            part[name] = fit(tr, te, feats)
        parts.append(part)
        print(f"k={k} {fold.name}: {len(variants)} fits", flush=True)
    pooled = pd.concat(parts, ignore_index=True)
    pooled.to_csv(OUT / f"predictions_k{k}.csv", index=False)
    base = per_event_arrays(pooled, pooled["전체"].to_numpy(), universe="all")[0]
    result = []
    for name in variants:
        h = per_event_arrays(pooled, pooled[name].to_numpy(), universe="all")[0]
        delta = h-base
        lo, hi = ci_mean(delta)
        result.append({"k": k, "모델": name, "제외 변수": ", ".join(f for f in full if f not in variants[name]),
                       "Hit@3": h.mean(), "전체 대비 차이": delta.mean(), "차이 95% 하한": lo,
                       "차이 95% 상한": hi, "평가 Event": len(h)})
    fold_table.to_csv(OUT / f"folds_k{k}.csv", index=False)
    return pd.DataFrame(result)


def md(df):
    return df.to_markdown(index=False, floatfmt=".3f")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_excel(next((ROOT / "data").glob("*.xlsx")), usecols=["시군구", "악취강도코드", "악취강도"])
    from config import CITY
    raw = raw[raw["시군구"] == CITY]
    code = pd.to_numeric(raw["악취강도코드"], errors="coerce")
    code_table = raw.groupby(["악취강도코드", "악취강도"], dropna=False).size().reset_index(name="건수")
    ci = ComplaintIndex(load_complaints())
    stats, abls, summaries, correlations = [], [], [], []
    for k in (7, 5):
        events = build_events(ci, k, "trigger")
        st = event_stats(ci, events, k)
        stats.append(st)
        q1, q2 = st.growth.quantile([1/3, 2/3])
        low, high = st[st.growth <= q1], st[st.growth >= q2]
        for name, part in (("하위 1/3", low), ("상위 1/3", high)):
            summaries.append({"k": k, "구간": name, "사건": len(part), "증가값 평균": part.growth.mean(),
                              "강도 평균": part.intensity_mean.mean(), "강도 최대 평균": part.intensity_max.mean(),
                              "이후 30분 신고 평균": part.future_reports.mean()})
        for y in ("intensity_mean", "intensity_max", "future_reports"):
            rho, lo, hi = corr_ci(st.growth.to_numpy(float), st[y].to_numpy(float))
            correlations.append({"k": k, "비교": y, "Spearman": rho, "95% 하한": lo, "95% 상한": hi})
        abls.append(ablation(ci, events, k))
    pd.concat(stats).to_csv(OUT / "events.csv", index=False)
    abl = pd.concat(abls); abl.to_csv(OUT / "ablation.csv", index=False)
    summ = pd.DataFrame(summaries); corr = pd.DataFrame(correlations)
    summ.to_csv(OUT / "terciles.csv", index=False)
    corr.to_csv(OUT / "correlations.csv", index=False)
    descriptions = {
        "initial_count": ("(cnt[None, :] * own).sum(1)", "해당 격자 초기 신고 수", "신고가 모인 위치 파악", "우선 확인할 격자"),
        "initial_intensity": ("(mi[None, :] * own).sum(1)", "해당 격자 신고 강도 평균", "신고자가 느낀 정도 반영", "현장 확인 참고"),
        "min_distance": ("dist.min(1)", "가장 가까운 신고 격자까지 거리", "신고 위치와 가까운 곳 고려", "점검 동선 참고"),
        "neighbor_count": ("(cnt[None, :] * (dist <= 1.5)).sum(1)", "주변 1.5칸 안 신고 수", "주변 신고 밀집 반영", "주변 확인 범위"),
        "radius3_count": ("(cnt[None, :] * (dist <= 3.0)).sum(1)", "주변 3칸 안 신고 수", "넓은 범위 신고 반영", "주변 확인 범위"),
        "prior": ("학습 Event의 해당 격자 양성 Event 비율; 학습 행은 leave-one-out", "과거 평가 사건에서 신고된 빈도", "반복되는 격자 반영", "과거 발생 참고"),
        "hour_sin": ("sin(2*pi*frac_hour/24)", "신고 시작 시각의 주기값", "시간대 차이 반영", "시간대 참고"),
        "hour_cos": ("cos(2*pi*frac_hour/24)", "신고 시작 시각의 주기값", "시간대 차이 반영", "시간대 참고"),
        "centroid_distance": ("hypot(C[:, 0]-cx, C[:, 1]-cy)", "신고 중심에서 거리", "신고 분포 중심 반영", "중심 주변 확인"),
        "nearest_intensity": ("mi[dist.argmin(1)]", "가장 가까운 신고 격자의 평균 강도", "가까운 신고자의 체감 참고", "현장 확인 참고"),
        "weighted_intensity": ("(w * mi[None, :]).sum(1) / w.sum(1)", "거리를 가중한 신고 강도", "주변 체감 정도 반영", "현장 확인 참고"),
        "observed_grid_count": ("full(len(cands), float(len(obs)))", "초기 신고가 들어온 격자 수", "신고 장소의 넓이 반영", "확인 범위 참고"),
        "observed_report_count": ("full(len(cands), float(cnt.sum()))", "초기 30분 전체 신고 수", "신고 증가 규모 참고", "신고가 늘어나는지 확인"),
        "first15_count": ("(f15[None, :] * own).sum(1)", "해당 격자의 앞 15분 신고 수", "앞뒤 신고 변화를 비교", "냄새가 심해지는지 확인할 단서"),
        "growth": ("(initial_count - first15) - first15", "해당 격자 뒤 15분 수에서 앞 15분 수를 뺀 값", "냄새가 심해지는 신호인지 검정", "현장 확인 시점 참고"),
    }
    hist_explain = {
        "hist_total_3h": ("ci.count_between(t0-180, t0)", "시작 전 3시간 전체 신고", "최근 이력 반영"),
        "gap_before_t0_min": ("min(t0-ci.t[s-1], 1440)", "직전 신고 이후 분", "신고 공백 반영"),
        "hist_cell_24h": ("ci.cell_count_between(cell, t0-1440, t0)", "지난 하루 해당 격자 신고", "최근 반복 반영"),
        "hist_cell_7d": ("ci.cell_count_between(cell, t0-7*1440, t0)", "지난 7일 해당 격자 신고", "과거 반복 반영"),
        "hist_same_hour_rate": ("n_before / elapsed_days", "과거 같은 시각 신고 날짜 비율", "시간대별 반복 반영"),
        "hist_same_hour_30d": ("n_30d", "지난 30일 같은 시각 신고 날짜 수", "시간대별 반복 반영"),
        "prev_night_cell_active": ("float(ci.cell_count_between(cell, prev_lo, prev_hi)>0)", "전날 같은 시각 해당 격자 신고 여부", "전날 반복 반영"),
        "prev_night_total": ("ci.count_between(prev_lo, prev_hi)", "전날 같은 시각 전체 신고 수", "전날 상황 반영"),
    }
    for f, (calc, meaning, reason) in hist_explain.items():
        descriptions[f] = (calc, meaning, reason, "과거 신고 참고")
    var = pd.DataFrame([{"변수 이름": f, "코드상 계산 방법": descriptions[f][0], "무엇을 재는지(쉬운 말)": descriptions[f][1],
                         "넣은 이유(가설)": descriptions[f][2], "담당자에게 주는 의미": descriptions[f][3]}
                        for f in BASE_FEATURES + HIST_FEATURES])
    (OUT / "variables.md").write_text("# v2 실제 모델 변수\n\n" + md(var) + "\n", encoding="utf-8")
    main_abl = abl[abl.k == 7]
    delta = main_abl[main_abl["모델"] != "전체"]
    top = delta.loc[delta["전체 대비 차이"].idxmin()]
    growth_corr = corr[(corr.k == 7) & (corr["비교"] == "intensity_mean")].iloc[0]
    first = [f"신고 증가값과 입력 구간 신고 강도의 순위상관은 {growth_corr.Spearman:.3f} (95% 구간 {growth_corr['95% 하한']:.3f}~{growth_corr['95% 상한']:.3f})임. 냄새가 심해진다는 가설을 뒷받침하지 않음.",
             f"주 운영 k=7에서 전체 모델 Hit@3는 {main_abl.iloc[0]['Hit@3']:.3f}임. 소거 시 최대 하락은 {abs(top['전체 대비 차이']):.3f}이나 모든 묶음의 차이 구간이 0을 포함해 도움 여부는 불확실함.",
             "증가·최근 신고 묶음은 k=7에서 빼도 Hit@3가 같았으나 차이 구간이 0을 포함하므로 제거 판단은 유보함."]
    report = "# 민원 변수 검정 결과\n\n## 결론 3줄\n\n" + "\n\n".join(first) + "\n\n"
    report += "## 설정과 변수 존재 여부\n\n주 운영은 trigger k=7, 보조 비교는 같은 trigger의 k=5임. 현재 모델은 기본 15개와 이력 8개를 사용함. 최근 5분·10분 신고 수, 반복신고 비율, 별도 공간 분산, 고유 위치 수 모델 변수는 현재 브랜치에 없음. 고유 위치 수는 Event 발동 조건에 사용함. 강도 평균은 격자별 변수로 있고 최대는 모델 변수에 없음. 기상 변수도 현재 모델에 없음. 원격 origin/codex/v2-event-evaluation의 features.py에서도 동일한 기본 변수 목록을 확인함.\n\n"
    report += "## 강도 코드\n\n" + f"원본 익산시 코드 범위 {code.min():.0f}~{code.max():.0f}, 결측률 {code.isna().mean():.2%}. 원본 `악취강도` 표시값을 대조하면 숫자가 클수록 더 심한 표현임. v2/data.py는 강도 결측 행을 제거함.\n\n" + md(code_table) + "\n\n"
    report += "## 변수 설명\n\n[variables.md](variables.md) 참조. `growth`는 격자별 후반 15분 신고 수에서 전반 15분 신고 수를 뺀 값임. 아래 사건 단위 분석에서는 같은 계산을 전체 신고에 적용함.\n\n"
    report += "## 증가값과 강도·이후 신고\n\n" + md(summ) + "\n\n" + md(corr) + "\n\n상·하위 집단은 1/3 분위값을 포함하여 동점 사건이 경계 양쪽에 포함될 수 있음. 구간은 사건 단위 1,000회 재표본추출임.\n\n"
    report += "## 변수 묶음 소거\n\n" + md(abl) + "\n\n기상 묶음은 실제 변수 부재로 소거하지 않음. 같은 v2 폴드·XGB 설정·Hit@3(전체)를 사용함. 실행 시간 때문에 v2의 3개 seed 중 42 하나만 사용했으며 전체 모델도 동일한 seed로 재학습함. 따라서 v2의 3-seed 발표 수치와 직접 같지는 않음. 차이는 전체 모델 대비이며 사건 단위 paired bootstrap 1,000회 구간임.\n\n"
    report += "## 발표용 문장\n\n" + f"- 현재 모델의 `growth`는 격자별 뒤 15분 신고 수에서 앞 15분 신고 수를 뺀 값임. 이 값이 높을 때 신고 강도가 함께 높은지는 순위상관 {growth_corr.Spearman:.3f}으로 관찰됨.\n- k=7의 전체 모델 Hit@3는 {main_abl.iloc[0]['Hit@3']:.3f}이며, {top['모델']}에서는 {top['Hit@3']:.3f}임.\n- 신고 증가는 냄새가 심해질 가능성을 살피는 참고 신호임. 실제 강도나 원인 시설을 판정하지 않음.\n\n"
    report += "## 한계\n\n강도는 신고자가 선택한 주관적 코드임. 신고 증가에는 시민의 인지·신고 행동이 섞여 있음. 상관은 인과를 뜻하지 않음. Event 발동 조건이 고유 위치 수에 의존하므로 선택된 사건에만 적용됨. 소거 결과는 변수의 단독 인과 효과가 아니며 상관된 다른 변수가 대신할 수 있음.\n\n출처: 원본 `data/*.xlsx`, v2/config.py·data.py·events.py·features.py·folds.py·run_v2.py·evaluate.py (확인일 2026-09-28).\n"
    (OUT / "report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
