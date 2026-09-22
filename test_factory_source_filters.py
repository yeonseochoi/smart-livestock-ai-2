"""공장 냄새 민원 ↔ 대기배출시설 데이터 상풍측 연관: 업종·종별 필터 전후 비교 (Track B, B1 실험).

질문: 3판 "공장" 발생원(대기오염물질 배출시설 신고 사업장)이 공장 냄새 민원과 연관이 없는 이유가
      비악취 업종(자동차 수리·목욕탕 등)이 섞여서인지, 아니면 데이터 자체가 원점을 못 잡는지.
방법: test_wind_source_association.py 의 통계량(민원 시각 상풍측 ±30°, 반경 6 km 시설 수, 월×6시간대 층화 셔플)을
      그대로 두고, 발생원 집합만 바꿔 가며 민원 유형별로 비교한다. 시설 수 기준(emission_weight=1).
필터:
  all         3판 factory 전체
  odor_ind    업종명에 악취 관련 키워드(폐기물·사료·비료·도축·육류·가금·식품·화학·플라스틱·고무·레미콘·아스콘·피혁·염색·도금·도장·주정·소주·전분·유지)
  big_class   facility_class 1~3종
  odor_big    odor_ind ∧ big_class
  unknown_ind 업종 결측(203/519가 결측이라 이 집단이 어디 붙는지 확인)

실행: python test_factory_source_filters.py → outputs/wind_lag_sweep/factory_filter_test.{json,md}
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_ablation as ab
import test_wind_source_association as t

OUTPUT_DIR = Path("outputs/wind_lag_sweep")
PERMUTATIONS = 300
RADIUS_KM = 6.0  # 다른 검정(association_sweep, run_onset_risk)과 같은 반경
ODOR_KEYWORDS = ("폐기물", "사료", "비료", "도축", "육류", "가금", "식품", "화학", "플라스틱", "고무", "레미콘", "아스콘",
                 "피혁", "염색", "도금", "도장", "주정", "소주", "전분", "유지", "조미료", "하수", "분뇨", "재생")
NON_ODOR_KEYWORDS = ("수리", "욕탕", "세탁", "병원", "학교", "대학", "교육", "행정", "종교", "부동산", "소매", "보험",
                     "영화관", "요양", "숙박", "골프", "세차", "훈련")


def factory_table(sources: pd.DataFrame) -> pd.DataFrame:
    f = sources[sources["source_type"] == "factory"].dropna(subset=["latitude", "longitude"]).copy()
    extra = f["extra_json"].apply(lambda s: json.loads(s) if isinstance(s, str) else {})
    f["industry"] = extra.apply(lambda d: d.get("industry_name") or "")
    f["facility_class"] = extra.apply(lambda d: d.get("facility_class") or "")
    f["class_num"] = pd.to_numeric(f["facility_class"].str.extract(r"(\d)")[0], errors="coerce")
    f["odor_ind"] = f["industry"].apply(lambda s: any(k in s for k in ODOR_KEYWORDS) and not any(k in s for k in NON_ODOR_KEYWORDS))
    f["big_class"] = f["class_num"].le(3)
    f["unknown_ind"] = f["industry"].eq("")
    f["emission_weight"] = 1.0
    return f


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t.PERMUTATIONS = PERMUTATIONS
    t.RADIUS_KM = RADIUS_KM
    rng = np.random.default_rng(t.SEED)
    complaints = t.load_complaints()
    complaints["is_sewage"] = complaints["odor_type"].astype(str).str.startswith("하수")
    sources = pd.read_csv(t.SOURCES_PATH, encoding="utf-8-sig")
    factories = factory_table(sources)
    asos = ab.load_asos()
    center = (float(complaints["latitude"].median()), float(complaints["longitude"].median()))
    wind = t.hourly_wind(asos, center)
    joined = complaints.join(wind, on="hour", how="inner")
    joined = joined[joined["speed"] >= t.MIN_WIND].reset_index(drop=True)

    subsets = {
        "all": factories,
        "odor_ind": factories[factories["odor_ind"]],
        "big_class": factories[factories["big_class"]],
        "odor_big": factories[factories["odor_ind"] & factories["big_class"]],
        "unknown_ind": factories[factories["unknown_ind"]],
        "non_odor_ind": factories[~factories["odor_ind"] & ~factories["unknown_ind"]],
    }
    groups = {
        "전체": np.ones(len(joined), bool), "가축냄새": joined["is_livestock"].to_numpy(),
        "공장냄새": joined["is_factory"].to_numpy(), "하수구냄새": joined["is_sewage"].to_numpy(),
    }
    rows = []
    for sname, sub_sources in subsets.items():
        for gname, mask in groups.items():
            sub = joined[mask].reset_index(drop=True)
            bins = t.bearing_bins(sub, sub_sources)
            r = t.run_test(sub, bins, rng)
            rows.append({"source_set": sname, "n_sources": int(len(sub_sources)), "complaint_group": gname,
                         "n_complaints": r["n"], "ratio": r["ratio_real_over_shuffled"], "z": r["z"], "p": r["p_value_one_sided"]})
            print(f"{sname:12s} ({len(sub_sources):4d}) × {gname:5s} n={r['n']:5d}  ratio {r['ratio_real_over_shuffled']:.3f}  z {r['z']:+.1f}  p {r['p_value_one_sided']:.3f}", flush=True)
    result = {"permutations": PERMUTATIONS, "radius_km": t.RADIUS_KM, "sector_deg": t.SECTOR_DEG,
              "odor_keywords": ODOR_KEYWORDS, "non_odor_keywords": NON_ODOR_KEYWORDS,
              "factory_counts": {k: int(len(v)) for k, v in subsets.items()}, "rows": rows}
    (OUTPUT_DIR / "factory_filter_test.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    table = pd.DataFrame(rows)
    lines = ["# 공장 냄새 민원 ↔ 대기배출시설 필터 전후 상풍측 연관", "",
             f"통계량 = 민원 시각 상풍측 ±{t.SECTOR_DEG:.0f}° 반경 {t.RADIUS_KM:.0f} km 시설 수, 실제 ÷ 층화 셔플 평균({PERMUTATIONS}회). 1.00 = 우연과 같음.", "",
             "| 발생원 집합 | 시설 수 | " + " | ".join(groups) + " |", "|---|---:|" + "---|" * len(groups)]
    for sname in subsets:
        part = table[table["source_set"] == sname]
        cells = [f"{r.ratio:.3f} (z {r.z:+.0f})" for r in part.itertuples()]
        lines.append(f"| {sname} | {int(part['n_sources'].iloc[0])} | " + " | ".join(cells) + " |")
    (OUTPUT_DIR / "factory_filter_test.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
