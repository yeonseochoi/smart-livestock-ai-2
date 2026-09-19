"""축종 계수 세트 4벌 × 반경 2종 × 민원 집합 3종의 풍향–발생원 연관 검정 민감도.

test_wind_source_association.py의 통계량(상풍측 ±30° 반경 R 발생원 가중치 합, 층화 셔플 대비)을 그대로 쓴다.
실행: python sensitivity_species_weights.py → outputs/wind_source_association/species_weight_sensitivity.{json,md}
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import run_ablation as ab
import species_weight_sets as sw
import test_wind_source_association as t

PERMUTATIONS = 300
RADII = (6.0, 10.0)


def main() -> None:
    rng = np.random.default_rng(7)
    complaints = t.load_complaints()
    raw = pd.read_csv(t.SOURCES_PATH, encoding="utf-8-sig").dropna(subset=["latitude", "longitude"])
    asos = ab.load_asos()
    wind = t.hourly_wind(asos, (float(complaints["latitude"].median()), float(complaints["longitude"].median())))
    joined = complaints.copy()
    joined["ref_hour"] = joined["hour"]
    joined = joined.join(wind, on="ref_hour", how="inner")
    joined = joined[joined["speed"] >= t.MIN_WIND]
    groups = {"all": joined, "livestock": joined[joined["is_livestock"]], "factory": joined[joined["is_factory"]]}
    t.PERMUTATIONS = PERMUTATIONS

    results = {"permutations": PERMUTATIONS, "min_wind_ms": t.MIN_WIND, "sector_deg": t.SECTOR_DEG, "rows": []}
    for name, spec in sw.SETS.items():
        sources = sw.apply_weight_set(raw, name)
        sources = sources[sources["emission_weight"] > 0]
        share = (sources.groupby("species")["emission_weight"].sum() / sources["emission_weight"].sum()).round(3)
        for radius in RADII:
            t.RADIUS_KM = radius
            bins = t.bearing_bins(complaints, sources)
            for group, sub in groups.items():
                r = t.run_test(sub, bins[sub.index.to_numpy()], rng)
                row = {"weight_set": name, "radius_km": radius, "complaints": group, "n": r["n"],
                       "ratio": r["ratio_real_over_shuffled"], "z": r["z"], "p": r["p_value_one_sided"],
                       "sources": int(len(sources)), "top_species_share": share.sort_values(ascending=False).head(3).to_dict()}
                results["rows"].append(row)
                print(f"{name:10s} R={radius:4.0f} {group:9s} n={r['n']:5d} ratio={r['ratio_real_over_shuffled']:.3f} z={r['z']:+.1f} p={r['p_value_one_sided']:.3f}")
    results["sets"] = {k: {"source": v["source"], "per": v["per"], "assumed": sorted(v["assumed"]),
                           "factor": v["factor"]} for k, v in sw.SETS.items()}

    out = t.OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    (out / "species_weight_sensitivity.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 축종 계수 민감도: 풍향–발생원 연관 검정", "",
             f"통계량 = 민원 시각 상풍측 ±{t.SECTOR_DEG:.0f}° 반경 R 안 발생원 가중치 합, 실제 ÷ 층화 셔플 평균({PERMUTATIONS}회). 1.00이면 우연과 같음.", "",
             "| 계수 세트 | 근거 | 단위 | R | 전체 민원 | 축산 민원 | 공장 민원 |", "|---|---|---|---:|---|---|---|"]
    for name, spec in sw.SETS.items():
        for radius in RADII:
            cells = []
            for group in ("all", "livestock", "factory"):
                r = next(x for x in results["rows"] if x["weight_set"] == name and x["radius_km"] == radius and x["complaints"] == group)
                cells.append(f"{r['ratio']:.3f} (z {r['z']:+.1f})")
            lines.append(f"| {name} | {spec['source']} | {spec['per']} | {radius:.0f} km | " + " | ".join(cells) + " |")
    lines += ["", "## 계수 세트와 가정 표시", ""]
    for name, spec in sw.SETS.items():
        if spec["factor"] is None:
            lines.append(f"- **{name}**: 시설 1곳 = 1")
            continue
        parts = [f"{k} {v}" + ("*" if k in spec["assumed"] else "") for k, v in spec["factor"].items()]
        lines.append(f"- **{name}** ({spec['per']}): " + ", ".join(parts))
    lines += ["", "`*` 문헌에 없어 가정한 값(ASSUMED)."]
    (out / "species_weight_sensitivity.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
