"""Track A 산출물 `grid_scores.csv`가 계약(docs/contracts/source_backtrack_contract.md)을 지키는지 검사한다.

Track A는 납품 전에, Track B는 수신 직후에 실행한다. 검사가 실패하면 산출물을 고친다.
검증기 자체를 고치지 않는다. 검증기가 틀렸다고 판단되면 근거와 함께 보고한다.

사용:
    python tools/validate_backtrack_output.py outputs/source_backtrack/grid_scores.csv
    python tools/validate_backtrack_output.py --build-reference   # 기준 후보 격자표 재생성

기준 후보 격자표(`outputs/backtrack_contract/reference_candidates.csv`)는
`compare_operational_grid_sizes.py`와 동일한 절차(공통 Event 160개, 1km, 후보 반경 2칸)로 만든다.
생성에 약 1~2분 걸리므로 파일로 저장해 두고 재사용한다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import build_odor_ai_mvp as odor  # noqa: E402
import sensitivity_early_prediction as sensitivity  # noqa: E402

REFERENCE_DIR = ROOT / "outputs" / "backtrack_contract"
REFERENCE_FILE = REFERENCE_DIR / "reference_candidates.csv"
REFERENCE_META = REFERENCE_DIR / "reference_meta.json"
GRID_M = 1000
GRID_SIZES = (1000, 1500, 2000)

COLUMNS = [
    "event_hour", "event_id", "grid_x", "grid_y", "center_latitude", "center_longitude",
    "source_fit_score", "forward_plume_score", "prior_downwind_score", "lagged_wind_alignment",
    "travel_time_min", "rain_1h", "rain_3h", "stagnation_flag", "backtrack_uncertainty",
    "weather_source", "history_cutoff",
]
UNIT_SCORES = ["source_fit_score", "forward_plume_score", "prior_downwind_score", "backtrack_uncertainty"]
WEATHER_SOURCES = {"aws", "asos", "none"}
CENTER_TOLERANCE_M = 1.0


def _complaints_signature() -> str:
    path = ROOT / "data" / "익산시 악취 민원 데이터_20190528-20260818.xlsx"
    stat = path.stat()
    return hashlib.md5(f"{path.name}:{stat.st_size}:{int(stat.st_mtime)}".encode()).hexdigest()


def build_reference() -> tuple[pd.DataFrame, dict[str, float]]:
    """compare_operational_grid_sizes.main 앞부분과 같은 절차로 1km 후보 격자표를 만든다."""
    complaints, _, _, _, _ = odor.load_inputs()
    original, _ = odor.add_grid_columns(complaints)
    _, selected_hours = odor.build_bounded_events(original)
    datasets = {}
    for grid_m in GRID_SIZES:
        data, _, _ = sensitivity.build_data(complaints, selected_hours, grid_m, odor.INPUT_MINUTES, odor.FORECAST_MINUTES)
        datasets[grid_m] = data
    common_ids = set.intersection(*(set(d["event_id"]) for d in datasets.values()))
    frame = datasets[GRID_M]
    frame = frame[frame["event_id"].isin(common_ids)].copy()
    meta = {
        "lat0": float(complaints["latitude"].median()),
        "lon0": float(complaints["longitude"].median()),
        "grid_m": GRID_M,
    }
    centers = np.array([odor.grid_centroid(int(x), int(y), meta) for x, y in zip(frame["grid_x"], frame["grid_y"])])
    frame["center_latitude"] = centers[:, 0]
    frame["center_longitude"] = centers[:, 1]
    reference = frame[["event_id", "is_train", "event_hour", "grid_x", "grid_y", "target",
                       "center_latitude", "center_longitude"]].reset_index(drop=True)
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    reference.to_csv(REFERENCE_FILE, index=False, encoding="utf-8-sig")
    meta_out = {
        **meta,
        "candidate_radius": max(1, int(round(1500 / GRID_M))),
        "events": int(reference["event_id"].nunique()),
        "rows": int(len(reference)),
        "complaints_signature": _complaints_signature(),
    }
    REFERENCE_META.write_text(json.dumps(meta_out, ensure_ascii=False, indent=2), encoding="utf-8")
    return reference, meta


def load_reference(refresh: bool = False) -> tuple[pd.DataFrame, dict[str, float]]:
    if refresh or not REFERENCE_FILE.exists() or not REFERENCE_META.exists():
        return build_reference()
    meta = json.loads(REFERENCE_META.read_text(encoding="utf-8"))
    if meta.get("complaints_signature") != _complaints_signature():
        print("[참고] 민원 원본이 바뀌어 기준 후보 격자표를 다시 만듭니다.")
        return build_reference()
    reference = pd.read_csv(REFERENCE_FILE, encoding="utf-8-sig", parse_dates=["event_hour"])
    return reference, {"lat0": meta["lat0"], "lon0": meta["lon0"], "grid_m": meta["grid_m"]}


def validate(
    scores: pd.DataFrame, reference: pd.DataFrame, meta: dict[str, float], min_coverage: float = 0.5,
) -> tuple[list[str], dict[str, object]]:
    """실패 메시지 목록과 요약 통계를 돌려준다. 목록이 비어 있으면 통과.

    `min_coverage`는 기준 Event 중 포함해야 하는 최소 비율. fixture처럼 일부 Event만 담은 파일은 0으로 검사한다.
    """
    failures: list[str] = []
    summary: dict[str, object] = {}

    # 1. 컬럼과 순서
    if list(scores.columns) != COLUMNS:
        missing = [c for c in COLUMNS if c not in scores.columns]
        extra = [c for c in scores.columns if c not in COLUMNS]
        failures.append(f"컬럼 불일치. 누락={missing} 추가={extra} 순서={list(scores.columns) == sorted(COLUMNS, key=COLUMNS.index)}")
        return failures, summary

    frame = scores.copy()
    # 2. 자료형
    try:
        frame["event_hour"] = pd.to_datetime(frame["event_hour"])
        frame["history_cutoff"] = pd.to_datetime(frame["history_cutoff"])
    except Exception as exc:  # noqa: BLE001
        failures.append(f"event_hour/history_cutoff 시각 변환 실패: {exc}")
        return failures, summary
    for col in ("grid_x", "grid_y"):
        if not np.issubdtype(frame[col].dtype, np.integer):
            failures.append(f"{col}은 정수여야 함 (현재 {frame[col].dtype})")
    numeric = ["center_latitude", "center_longitude", *UNIT_SCORES, "lagged_wind_alignment",
               "travel_time_min", "rain_1h", "rain_3h"]
    for col in numeric:
        if not np.issubdtype(frame[col].dtype, np.number):
            failures.append(f"{col}은 숫자여야 함 (현재 {frame[col].dtype})")
    if failures:
        return failures, summary

    # 3. 키 유일성
    dup = frame.duplicated(["event_hour", "grid_x", "grid_y"]).sum()
    if dup:
        failures.append(f"(event_hour, grid_x, grid_y) 중복 {dup}행")

    # 4. 값 범위
    for col in UNIT_SCORES:
        bad = frame[col].dropna()
        bad = bad[(bad < -1e-9) | (bad > 1 + 1e-9)]
        if len(bad):
            failures.append(f"{col} 범위 밖(0~1) {len(bad)}행")
    align = frame["lagged_wind_alignment"].dropna()
    if ((align < -1 - 1e-9) | (align > 1 + 1e-9)).any():
        failures.append("lagged_wind_alignment 범위 밖(-1~1)")
    for col in ("travel_time_min", "rain_1h", "rain_3h"):
        if (frame[col].dropna() < 0).any():
            failures.append(f"{col} 음수 존재")
    flags = set(frame["stagnation_flag"].dropna().unique().tolist())
    if not flags.issubset({0, 1, 0.0, 1.0, True, False}):
        failures.append(f"stagnation_flag는 0/1이어야 함: {sorted(map(str, flags))[:5]}")
    sources = set(frame["weather_source"].dropna().astype(str).unique())
    if not sources.issubset(WEATHER_SOURCES):
        failures.append(f"weather_source 허용값 {sorted(WEATHER_SOURCES)} 밖: {sorted(sources - WEATHER_SOURCES)}")

    # 5. 누수: history_cutoff <= event_hour
    leak = frame.dropna(subset=["history_cutoff"])
    leak = leak[leak["history_cutoff"] > leak["event_hour"]]
    if len(leak):
        failures.append(f"history_cutoff가 event_hour보다 늦은 행 {len(leak)}개 (누수 의심)")

    # 6. 격자 중심 좌표가 기준 원점과 일치
    expected = np.array([odor.grid_centroid(int(x), int(y), meta) for x, y in zip(frame["grid_x"], frame["grid_y"])])
    dlat_m = (frame["center_latitude"].to_numpy() - expected[:, 0]) * 110_540
    dlon_m = (frame["center_longitude"].to_numpy() - expected[:, 1]) * 111_320 * np.cos(np.radians(meta["lat0"]))
    off = np.sqrt(dlat_m ** 2 + dlon_m ** 2)
    if np.nanmax(off) > CENTER_TOLERANCE_M:
        failures.append(f"격자 중심 좌표가 기준과 최대 {np.nanmax(off):.1f} m 어긋남 (허용 {CENTER_TOLERANCE_M} m). 원점(lat0, lon0)이나 grid_m 확인")

    # 7. Event 집합과 후보 격자 포함 여부
    ref_hours = set(reference["event_hour"].unique())
    got_hours = set(frame["event_hour"].unique())
    unknown = got_hours - ref_hours
    if unknown:
        failures.append(f"기준 Event에 없는 event_hour {len(unknown)}개 (예: {sorted(unknown)[:3]})")
    ref_keys = set(zip(reference["event_hour"], reference["grid_x"].astype(int), reference["grid_y"].astype(int)))
    got_keys = set(zip(frame["event_hour"], frame["grid_x"].astype(int), frame["grid_y"].astype(int)))
    covered = [h for h in got_hours if h in ref_hours]
    missing_cells = [k for k in ref_keys if k[0] in set(covered) and k not in got_keys]
    if missing_cells:
        failures.append(f"포함된 Event의 기준 후보 격자 중 빠진 칸 {len(missing_cells)}개 (예: {missing_cells[:3]}). candidate_cells radius=2 확인")

    # 8. 정규화와 결측 비율 (경고성 검사)
    per_event_max = frame.groupby("event_hour")["source_fit_score"].max()
    normalized = per_event_max.dropna()
    off_norm = normalized[(normalized - 1).abs() > 1e-6]
    if len(off_norm):
        failures.append(f"source_fit_score가 Event 내 최댓값 1로 정규화되지 않은 Event {len(off_norm)}개")
    summary = {
        "rows": int(len(frame)),
        "events": int(frame["event_hour"].nunique()),
        "reference_events": int(len(ref_hours)),
        "event_coverage": round(len(covered) / max(len(ref_hours), 1), 3),
        "nan_rate": {c: round(float(frame[c].isna().mean()), 3) for c in UNIT_SCORES + ["lagged_wind_alignment"]},
        "weather_source": frame["weather_source"].value_counts(dropna=False).to_dict(),
        "max_center_offset_m": round(float(np.nanmax(off)), 3) if len(off) else None,
    }
    if summary["event_coverage"] < min_coverage:
        failures.append(f"기준 Event의 {summary['event_coverage']:.0%}만 포함. 최소 {min_coverage:.0%} 필요")
    return failures, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", nargs="?", default="outputs/source_backtrack/grid_scores.csv")
    parser.add_argument("--build-reference", action="store_true", help="기준 후보 격자표만 다시 만들고 종료")
    args = parser.parse_args()

    if args.build_reference:
        reference, meta = build_reference()
        print(f"기준 후보 격자표 생성: {REFERENCE_FILE} ({reference['event_id'].nunique()} Event, {len(reference)}행)")
        return 0

    path = Path(args.path)
    if not path.exists():
        print(f"파일 없음: {path}")
        return 2
    reference, meta = load_reference()
    scores = pd.read_csv(path, encoding="utf-8-sig")
    failures, summary = validate(scores, reference, meta)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failures:
        print("\n검증 실패:")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("\n검증 통과")
    return 0


if __name__ == "__main__":
    sys.exit(main())
