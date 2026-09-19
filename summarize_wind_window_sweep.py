"""역추적 엔진의 참조 바람 창 비교 요약 (Track A).

build_source_backtrack.py 를 `--wind-window lag,window,weighting --output-dir outputs/wind_lag_sweep/engine_<tag>` 로 돌린 결과의
validation.json 만 모아 표로 만든다(grid_scores 등 큰 산출물은 커밋하지 않음).

실행: python summarize_wind_window_sweep.py → outputs/wind_lag_sweep/engine_sweep.{json,md}
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SWEEP_DIR = ROOT / "outputs" / "wind_lag_sweep"
def load(path: Path, label: str) -> dict:
    v = json.loads(path.read_text(encoding="utf-8"))
    spec = v["constants"].get("wind_window", {"lag": 0, "window": 2, "weighting": "speed"})
    contrast = v["random_wind_contrast"]
    return {
        "label": label, "lag_h": spec["lag"], "window_h": spec["window"], "weighting": spec["weighting"],
        "max_source_km": v["constants"]["max_source_km"],
        **{f"{cond}_{key}": contrast[cond][key] for cond in ("all", "dry", "wet")
           for key in ("real_percentile", "shuffled_mean", "mean_difference", "p_value")},
        "sensitivity_10_20_30": [v["direction_sensitivity"][d] for d in ("10", "20", "30")],
    }


def main() -> None:
    rows = []
    for folder in sorted(SWEEP_DIR.glob("engine_*")):
        validation = folder / "validation.json"
        if validation.exists():
            rows.append(load(validation, folder.name.replace("engine_", "")))
    rows.sort(key=lambda r: (r["lag_h"], r["window_h"], r["weighting"], r["label"]))
    (SWEEP_DIR / "engine_sweep.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 역추적 엔진 참조 바람 창 비교", "",
             "각 설정으로 build_source_backtrack.py 를 다시 돌려 무작위 풍향 대조(Event 풍향 profile 100회 셔플)를 비교했다. "
             "'차이' = 실제 수용점 격자의 forward_plume_score 순위 백분위 − 셔플 평균. 클수록 바람이 민원 위치를 더 잘 설명한다. p 는 경험적 단측.", "",
             "| 설정(lag,window,가중) | 반경 | 전체 실제 | 전체 차이 | 전체 p | 건조 실제 | 건조 차이 | 건조 p | 강수 실제 | 강수 차이 | 강수 p | 풍향 ±10/20/30° 상위3 유지 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        sens = "/".join(f"{x:.2f}" for x in r["sensitivity_10_20_30"])
        lines.append(f"| {r['lag_h']},{r['window_h']},{r['weighting']} ({r['label']}) | {r['max_source_km']:g} km | "
                     f"{r['all_real_percentile']:.4f} | {r['all_mean_difference']:+.4f} | {r['all_p_value']:.3f} | "
                     f"{r['dry_real_percentile']:.4f} | {r['dry_mean_difference']:+.4f} | {r['dry_p_value']:.3f} | "
                     f"{r['wet_real_percentile']:.4f} | {r['wet_mean_difference']:+.4f} | {r['wet_p_value']:.3f} | {sens} |")
    lines += ["", "0-1시간 창은 전체 조건 p=0.030이지만 건조 p=0.069, 강수 p=0.158이다. 5개 창 비교에는 다중비교 보정을 적용하지 않았다."]
    (SWEEP_DIR / "engine_sweep.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
