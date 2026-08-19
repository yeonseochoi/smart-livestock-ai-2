"""익산시 악취 민원의 시공간 군집성과 행정경계 확장 가능성을 분석한다.

실행: python analyze_spatiotemporal_complaints.py
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Iterable

import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from folium.plugins import TimestampedGeoJson
from sklearn.neighbors import BallTree


# ========================= 사용자 설정 =========================
TARGET_CITY = "익산시"
ADJACENT_COUNTY = "완주군"
ADJACENT_TOWN = "삼례읍"
INCLUDE_ADJACENT = False  # 기본 분석 대상 선택용. 메인 결과는 이 값과 무관하게 항상 익산시만 사용한다.
INPUT_FILE = Path("data/익산시 악취 민원 데이터_20190528-20260818.xlsx")
OUTPUT_DIR = Path("outputs")
EARTH_RADIUS_KM = 6371.0088
SAME_LOCATION_KM = 0.05
CROSS_BOUNDARY_MAX_HOURS = 2.0
CROSS_BOUNDARY_MAX_KM = 3.0

PROXIMITY_CONDITIONS = [
    ("30분 + 500m", 30, 0.5),
    ("30분 + 1km", 30, 1.0),
    ("1시간 + 1km", 60, 1.0),
    ("1시간 + 2km", 60, 2.0),
    ("2시간 + 3km", 120, 3.0),
]
REPEAT_EXCLUDED_CONDITIONS = [
    ("30분 + 1km", 30, 1.0),
    ("1시간 + 2km", 60, 2.0),
    ("2시간 + 3km", 120, 3.0),
]
EVENT_LEVELS = {"Level 1": (10, 5), "Level 2": (20, 10), "Level 3": (30, 15)}

# 원본 파일의 표현이 달라도 정규화된 내부 열 이름으로 매핑한다.
COLUMN_ALIASES = {
    "datetime": ["악취발생일시", "악취 발생 일시", "발생일시", "신고일시"],
    "latitude": ["위도", "lat", "latitude"],
    "longitude": ["경도", "lon", "lng", "longitude"],
    "province": ["시도", "광역시도"],
    "city": ["시군구", "시군구명"],
    "region": ["지역", "읍면동", "주소"],
    "intensity": ["악취강도코드", "악취강도", "강도"],
    "intensity_label": ["악취강도", "악취강도명"],
    "odor_type": ["악취종류", "악취유형", "냄새종류"],
}


def _norm(text: object) -> str:
    return re.sub(r"[\s_\-()]", "", str(text)).lower()


def _find_column(columns: Iterable[object], aliases: list[str], required: bool = True) -> str | None:
    normalized = {_norm(c): str(c) for c in columns}
    for alias in aliases:
        if _norm(alias) in normalized:
            return normalized[_norm(alias)]
    if required:
        raise KeyError(f"필수 컬럼을 찾지 못했습니다. 후보={aliases}, 실제={list(columns)}")
    return None


def load_data(path: Path = INPUT_FILE) -> tuple[pd.DataFrame, dict[str, str]]:
    """엑셀을 읽고 실제 컬럼명을 검사하여 내부 표준 이름에 매핑한다."""
    if not path.exists():
        candidates = sorted(Path("data").glob("*.xlsx"))
        if len(candidates) != 1:
            raise FileNotFoundError(f"입력 파일을 찾지 못했습니다: {path}")
        path = candidates[0]
    raw = pd.read_excel(path)
    mapping: dict[str, str] = {}
    for key, aliases in COLUMN_ALIASES.items():
        mapping[key] = _find_column(raw.columns, aliases, required=key != "intensity_label")  # type: ignore[assignment]
    # 악취강도코드를 우선 숫자 강도로 쓰고, 없으면 표시 문자열에서 숫자를 추출한다.
    if mapping.get("intensity") == mapping.get("intensity_label"):
        mapping["intensity_label"] = mapping["intensity"]
    print("[컬럼 매핑]", json.dumps(mapping, ensure_ascii=False))
    return raw, mapping


def preprocess_data(raw: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """날짜·좌표·강도를 분석 가능한 형식으로 변환하고 원본 행 번호를 보존한다."""
    df = pd.DataFrame(index=raw.index)
    for target in ["datetime", "latitude", "longitude", "province", "city", "region", "odor_type"]:
        df[target] = raw[mapping[target]]
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    intensity_raw = raw[mapping["intensity"]]
    df["intensity"] = pd.to_numeric(intensity_raw, errors="coerce")
    if df["intensity"].isna().any():
        df["intensity"] = df["intensity"].fillna(
            pd.to_numeric(intensity_raw.astype(str).str.extract(r"(\d+(?:\.\d+)?)")[0], errors="coerce")
        )
    df["intensity_label"] = raw[mapping.get("intensity_label", mapping["intensity"])].astype("string")
    for c in ["province", "city", "region", "odor_type"]:
        df[c] = df[c].astype("string").str.strip()
    df["source_row"] = raw.index + 2
    df["valid_for_spacetime"] = df[["datetime", "latitude", "longitude"]].notna().all(axis=1)
    return df


def inspect_regions(raw: pd.DataFrame, df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """전체/지역별 건수, 결측치, 시간 및 좌표 범위를 요약한다."""
    city = df["city"].eq(TARGET_CITY).fillna(False)
    county = df["city"].eq(ADJACENT_COUNTY).fillna(False)
    town = county & df["region"].str.contains(ADJACENT_TOWN, na=False).fillna(False)
    key_missing = {k: int(raw[v].isna().sum()) for k, v in mapping.items() if v is not None}
    rows = [
        ("전체 행 수", len(df)), (f"{TARGET_CITY} 데이터 수", int(city.sum())),
        (f"{ADJACENT_COUNTY} 데이터 수", int(county.sum())),
        (f"{ADJACENT_COUNTY} 중 {ADJACENT_TOWN} 데이터 수", int(town.sum())),
        ("그 외 지역 데이터 수", int((~city & ~town).sum())),
        ("시공간 분석 유효 행 수", int(df["valid_for_spacetime"].sum())),
        ("시간 최솟값", df["datetime"].min()), ("시간 최댓값", df["datetime"].max()),
        ("위도 최솟값", df["latitude"].min()), ("위도 최댓값", df["latitude"].max()),
        ("경도 최솟값", df["longitude"].min()), ("경도 최댓값", df["longitude"].max()),
        ("주요 컬럼 결측치(JSON)", json.dumps(key_missing, ensure_ascii=False)),
    ]
    return pd.DataFrame(rows, columns=["항목", "값"])


def filter_target_region(df: pd.DataFrame, include_adjacent: bool = INCLUDE_ADJACENT) -> pd.DataFrame:
    """익산시만, 또는 익산시와 완주군 삼례읍만 선택한다."""
    iksan = df["city"].eq(TARGET_CITY).fillna(False)
    samrye = df["city"].eq(ADJACENT_COUNTY).fillna(False) & df["region"].str.contains(ADJACENT_TOWN, na=False).fillna(False)
    mask = iksan | samrye if include_adjacent else iksan
    return df.loc[mask & df["valid_for_spacetime"]].copy().sort_values("datetime").reset_index(drop=True)


def calculate_proximity_statistics(
    df: pd.DataFrame,
    conditions: list[tuple[str, int, float]] = PROXIMITY_CONDITIONS,
    min_distance_km: float = 0.0,
    analysis_name: str = "익산시만",
) -> pd.DataFrame:
    """BallTree로 공간 후보를 먼저 찾고 시간 차를 검사해 O(N²) 전수 비교를 피한다."""
    n = len(df)
    columns = ["분석대상", "조건", "시간한계_분", "거리하한_km", "거리상한_km", "인접민원존재_신고수", "전체신고수", "비율_pct"]
    if n == 0:
        return pd.DataFrame(columns=columns)
    coords = np.radians(df[["latitude", "longitude"]].to_numpy(float))
    # pandas 내부 datetime 해상도(ns/us)에 의존하지 않고 실제 경과 분으로 변환한다.
    times = ((df["datetime"] - pd.Timestamp("1970-01-01")) / pd.Timedelta(minutes=1)).to_numpy(float)
    tree = BallTree(coords, metric="haversine")
    rows = []
    for label, minutes, max_km in conditions:
        indices, distances = tree.query_radius(coords, r=max_km / EARTH_RADIUS_KM, return_distance=True)
        has_neighbor = np.zeros(n, dtype=bool)
        for i, (neighbors, angular) in enumerate(zip(indices, distances)):
            km = angular * EARTH_RADIUS_KM
            dt = np.abs(times[neighbors] - times[i])
            valid = (neighbors != i) & (dt <= minutes) & (km > min_distance_km + 1e-12)
            has_neighbor[i] = bool(np.any(valid))
        count = int(has_neighbor.sum())
        rows.append([analysis_name, label, minutes, min_distance_km, max_km, count, n, count / n * 100])
    return pd.DataFrame(rows, columns=columns)


def _counts_json(series: pd.Series) -> str:
    return json.dumps(series.fillna("미상").astype(str).value_counts().to_dict(), ensure_ascii=False)


def aggregate_hourly_events(df: pd.DataFrame, label: str = "익산시만") -> pd.DataFrame:
    """민원을 1시간 단위로 묶고 위치·지역·강도·종류 통계를 만든다."""
    work = df.copy()
    work["event_hour"] = work["datetime"].dt.floor("h")
    work["coordinate"] = list(zip(work["latitude"].round(6), work["longitude"].round(6)))
    records = []
    for hour, g in work.groupby("event_hour", sort=True):
        type_counts = g["odor_type"].fillna("미상").astype(str).value_counts()
        records.append({
            "분석대상": label, "발생시각": hour, "신고수": len(g),
            "서로다른좌표수": g["coordinate"].nunique(),
            "지역별신고수": _counts_json(g["region"]),
            "평균강도": g["intensity"].mean(), "최대강도": g["intensity"].max(),
            "악취종류별신고수": json.dumps(type_counts.to_dict(), ensure_ascii=False),
            "주요악취종류": type_counts.index[0] if len(type_counts) else "미상",
        })
    return pd.DataFrame(records)


def find_major_events(hourly: pd.DataFrame) -> tuple[dict[str, int], pd.DataFrame]:
    """세 단계 Event 기준을 판정하고 Level 1 이상을 Top 20으로 반환한다."""
    out = hourly.copy()
    counts = {}
    for level, (min_reports, min_locations) in EVENT_LEVELS.items():
        mask = (out["신고수"] >= min_reports) & (out["서로다른좌표수"] >= min_locations)
        out[level] = mask
        counts[level] = int(mask.sum())
    top20 = out.loc[out["Level 1"]].sort_values(["신고수", "서로다른좌표수"], ascending=False).head(20)
    return counts, top20


def analyze_top_events(df: pd.DataFrame, hourly: pd.DataFrame, top_n: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """신고가 많은 상위 Event를 요약하고 Event 내부를 10분 단위로 분석한다."""
    top_hours = hourly.sort_values(["신고수", "서로다른좌표수"], ascending=False).head(top_n)["발생시각"]
    event_rows, bins = [], []
    for rank, hour in enumerate(top_hours, 1):
        g = df[(df["datetime"] >= hour) & (df["datetime"] < hour + pd.Timedelta(hours=1))].copy()
        coords = pd.Series(list(zip(g["latitude"].round(6), g["longitude"].round(6))))
        event_rows.append({
            "순위": rank, "발생시각": hour, "신고수": len(g), "서로다른좌표수": coords.nunique(),
            "지역별신고수": _counts_json(g["region"]), "악취강도분포": _counts_json(g["intensity_label"]),
            "악취종류분포": _counts_json(g["odor_type"]), "최초신고시각": g["datetime"].min(),
            "마지막신고시각": g["datetime"].max(), "평균강도": g["intensity"].mean(),
        })
        g["10분구간시작"] = g["datetime"].dt.floor("10min")
        for bin_start, b in g.groupby("10분구간시작"):
            bcoords = pd.Series(list(zip(b["latitude"].round(6), b["longitude"].round(6))))
            bins.append({
                "Event순위": rank, "Event발생시각": hour, "10분구간시작": bin_start,
                "10분구간종료": bin_start + pd.Timedelta(minutes=10), "신고수": len(b),
                "서로다른위치수": bcoords.nunique(), "평균위도": b["latitude"].mean(),
                "평균경도": b["longitude"].mean(), "평균악취강도": b["intensity"].mean(),
            })
    return pd.DataFrame(event_rows), pd.DataFrame(bins)


def visualize_event_map(df: pd.DataFrame, event_hour: pd.Timestamp, output: Path) -> None:
    """최대 Event를 시간 재생 가능한 HTML 지도와 popup으로 저장한다."""
    g = df[(df["datetime"] >= event_hour) & (df["datetime"] < event_hour + pd.Timedelta(hours=1))].copy()
    if g.empty:
        return
    center = [g["latitude"].mean(), g["longitude"].mean()]
    fmap = folium.Map(location=center, zoom_start=13, control_scale=True)
    features = []
    for _, r in g.sort_values("datetime").iterrows():
        popup = (f"신고 시각: {r['datetime']}<br>지역: {r['region']}<br>"
                 f"악취강도: {r['intensity_label']}<br>악취종류: {r['odor_type']}")
        features.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [r["longitude"], r["latitude"]]},
                         "properties": {"time": r["datetime"].isoformat(), "popup": popup,
                                        "icon": "circle", "iconstyle": {"fillColor": "#d62728", "fillOpacity": 0.75, "stroke": True, "radius": 6}}})
    TimestampedGeoJson({"type": "FeatureCollection", "features": features}, period="PT1M", duration="PT10M",
                       add_last_point=True, auto_play=False, loop=False, max_speed=10).add_to(fmap)
    fmap.save(str(output))


def analyze_adjacent_region(iksan: pd.DataFrame, combined: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """익산시 단독과 익산시+삼례읍의 인접률 및 Event 수를 별도로 비교한다."""
    prox = pd.concat([
        calculate_proximity_statistics(iksan, analysis_name="익산시만"),
        calculate_proximity_statistics(combined, analysis_name="익산시+삼례읍"),
    ], ignore_index=True)
    combined_hourly = aggregate_hourly_events(combined, "익산시+삼례읍")
    combined_counts, _ = find_major_events(combined_hourly)
    iksan_counts, _ = find_major_events(aggregate_hourly_events(iksan, "익산시만"))
    comparison = pd.DataFrame([
        {"분석대상": name, **counts} for name, counts in [("익산시만", iksan_counts), ("익산시+삼례읍", combined_counts)]
    ])
    return prox, comparison, combined_hourly, combined_counts


def find_cross_boundary_events(iksan: pd.DataFrame, samrye: pd.DataFrame) -> pd.DataFrame:
    """2시간 이내이면서 양 지역 신고 사이 최소거리가 3km 이내인 시간대 쌍을 찾는다."""
    columns = ["익산시시간대", "삼례읍시간대", "시간차_분", "익산시신고수", "삼례읍신고수", "합계신고수", "최소거리_km", "근접교차쌍수"]
    if iksan.empty or samrye.empty:
        return pd.DataFrame(columns=columns)
    a, b = iksan.copy(), samrye.copy()
    a["hour"] = a["datetime"].dt.floor("h"); b["hour"] = b["datetime"].dt.floor("h")
    records = []
    for ah, ag in a.groupby("hour"):
        nearby_b = b[(b["hour"] >= ah - pd.Timedelta(hours=CROSS_BOUNDARY_MAX_HOURS)) &
                     (b["hour"] <= ah + pd.Timedelta(hours=CROSS_BOUNDARY_MAX_HOURS))]
        for bh, bg in nearby_b.groupby("hour"):
            tree = BallTree(np.radians(bg[["latitude", "longitude"]].to_numpy()), metric="haversine")
            dist, _ = tree.query(np.radians(ag[["latitude", "longitude"]].to_numpy()), k=1)
            min_km = float(dist.min() * EARTH_RADIUS_KM)
            if min_km <= CROSS_BOUNDARY_MAX_KM:
                pairs = sum(len(x) for x in tree.query_radius(np.radians(ag[["latitude", "longitude"]].to_numpy()), r=CROSS_BOUNDARY_MAX_KM / EARTH_RADIUS_KM))
                records.append([ah, bh, abs((bh-ah).total_seconds())/60, len(ag), len(bg), len(ag)+len(bg), min_km, pairs])
    return pd.DataFrame(records, columns=columns).sort_values(["합계신고수", "근접교차쌍수"], ascending=False)


def _set_korean_font() -> None:
    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def create_visualizations(hourly: pd.DataFrame, detail_bins: pd.DataFrame, event_comparison: pd.DataFrame) -> None:
    """요청된 네 가지 PNG 그래프를 생성한다."""
    _set_korean_font()
    plt.figure(figsize=(9, 5)); plt.hist(hourly["신고수"], bins=min(40, max(10, int(math.sqrt(len(hourly))))), color="#4c78a8")
    plt.xlabel("1시간 신고 건수"); plt.ylabel("시간대 수"); plt.title("익산시 1시간 단위 신고 건수 분포"); plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "iksan_hourly_complaint_distribution.png", dpi=160); plt.close()

    plt.figure(figsize=(8, 6)); plt.scatter(hourly["신고수"], hourly["서로다른좌표수"], alpha=.55, s=22, color="#f58518")
    plt.xlabel("신고 건수"); plt.ylabel("서로 다른 좌표 수"); plt.title("익산시 Event별 신고 수와 위치 수"); plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "iksan_event_scatter.png", dpi=160); plt.close()

    top = detail_bins[detail_bins["Event순위"] == 1]
    plt.figure(figsize=(9, 5)); plt.plot(top["10분구간시작"], top["신고수"], marker="o", color="#e45756")
    plt.xlabel("10분 구간"); plt.ylabel("신고 건수"); plt.title("익산시 최대 Event의 10분 단위 신고 건수"); plt.xticks(rotation=35); plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "iksan_top_event_timeline.png", dpi=160); plt.close()

    comp = event_comparison.set_index("분석대상")[["Level 1", "Level 2", "Level 3"]]
    comp.plot(kind="bar", figsize=(9, 5), color=["#72b7b2", "#f2cf5b", "#e45756"])
    plt.ylabel("Event 수"); plt.title("익산시 단독 vs 삼례읍 포함 Event 수"); plt.xticks(rotation=0); plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "iksan_vs_adjacent_event_comparison.png", dpi=160); plt.close()


def save_results(results: dict[str, pd.DataFrame]) -> None:
    """모든 표를 UTF-8 BOM CSV로 저장하여 Excel에서도 한글이 깨지지 않게 한다."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filenames = {
        "region": "data_region_summary.csv", "proximity": "iksan_proximity_summary.csv",
        "hourly": "iksan_hourly_event_summary.csv", "top20": "iksan_top20_events.csv",
        "detail": "iksan_top_event_detail.csv", "adj_proximity": "adjacent_proximity_comparison.csv",
        "adj_event": "adjacent_event_comparison.csv", "cross": "cross_boundary_events.csv",
    }
    for key, filename in filenames.items():
        results[key].to_csv(OUTPUT_DIR / filename, index=False, encoding="utf-8-sig")


def print_summary(region: pd.DataFrame, proximity: pd.DataFrame, event_counts: dict[str, int],
                  top_detail: pd.DataFrame, adjacent_comparison: pd.DataFrame, cross: pd.DataFrame) -> None:
    """분석의 핵심 지표를 한눈에 보이도록 콘솔에 출력한다."""
    r = dict(zip(region["항목"], region["값"]))
    print("\n===== 데이터 범위 =====\n")
    for key in ["전체 행 수", f"{TARGET_CITY} 데이터 수", f"{ADJACENT_COUNTY} 중 {ADJACENT_TOWN} 데이터 수", "그 외 지역 데이터 수"]:
        print(f"{key:<22}: {r[key]}건")
    print("\n===== 익산시 메인 분석 =====\n\n[시간·공간 인접 민원]")
    for _, x in proximity[proximity["거리하한_km"] == 0].iterrows():
        print(f"{x['조건']:<14}: {x['인접민원존재_신고수']}건 ({x['비율_pct']:.1f}%)")
    print("\n[동일 위치(50m 이하) 제외 검증]")
    for _, x in proximity[proximity["거리하한_km"] > 0].iterrows():
        print(f"{x['조건']:<14}: {x['인접민원존재_신고수']}건 ({x['비율_pct']:.1f}%)")
    print("\n[대규모 Event]")
    for level, count in event_counts.items(): print(f"{level:<8}: {count}개")
    if not top_detail.empty:
        t = top_detail.iloc[0]
        print("\n[최대 Event]")
        print(f"발생 시각: {t['발생시각']}\n신고 건수: {t['신고수']}\n서로 다른 위치: {t['서로다른좌표수']}")
        print(f"주요 지역: {t['지역별신고수']}\n평균 악취강도: {t['평균강도']:.2f}\n악취종류: {t['악취종류분포']}")
    print("\n===== 행정경계 확장 실험 =====\n")
    for _, x in adjacent_comparison.iterrows(): print(f"{x['분석대상']} Event 수(Level 1/2/3): {x['Level 1']}/{x['Level 2']}/{x['Level 3']}")
    print(f"Cross-boundary Event 후보: {len(cross)}개")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw, mapping = load_data()
    df = preprocess_data(raw, mapping)
    region = inspect_regions(raw, df, mapping)

    # 메인 분석은 전역 설정과 무관하게 반드시 익산시만 사용한다.
    iksan = filter_target_region(df, include_adjacent=False)
    proximity_base = calculate_proximity_statistics(iksan, analysis_name="익산시만_기본")
    proximity_no_repeat = calculate_proximity_statistics(
        iksan, REPEAT_EXCLUDED_CONDITIONS, min_distance_km=SAME_LOCATION_KM,
        analysis_name="익산시만_50m이하제외",
    )
    proximity = pd.concat([proximity_base, proximity_no_repeat], ignore_index=True)
    hourly = aggregate_hourly_events(iksan)
    event_counts, top20 = find_major_events(hourly)
    top_detail, detail_bins = analyze_top_events(iksan, hourly)
    # 상세 요약과 10분 자료를 한 파일에 구분 가능한 형태로 함께 저장한다.
    detail_export = pd.concat([
        top_detail.assign(자료구분="Event요약"), detail_bins.assign(자료구분="10분상세")
    ], ignore_index=True, sort=False)

    # 보조 실험은 익산시+완주군 삼례읍만 사용하며 메인 결과와 파일을 분리한다.
    combined = filter_target_region(df, include_adjacent=True)
    samrye = combined[combined["city"].eq(ADJACENT_COUNTY)].copy()
    adj_prox, adj_event, _, _ = analyze_adjacent_region(iksan, combined)
    cross = find_cross_boundary_events(iksan, samrye)

    results = {"region": region, "proximity": proximity, "hourly": hourly, "top20": top20,
               "detail": detail_export, "adj_proximity": adj_prox, "adj_event": adj_event, "cross": cross}
    save_results(results)
    if not top_detail.empty:
        visualize_event_map(iksan, pd.Timestamp(top_detail.iloc[0]["발생시각"]), OUTPUT_DIR / "iksan_top_odor_event_map.html")
    create_visualizations(hourly, detail_bins, adj_event)
    print_summary(region, proximity, event_counts, top_detail, adj_event, cross)


if __name__ == "__main__":
    main()
