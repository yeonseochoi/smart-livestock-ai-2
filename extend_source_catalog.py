"""공개 원문에서 축산 외 후보 발생원을 표준화해 로컬 sources.csv에 추가한다.

좌표가 포함된 결과와 VWorld 응답 캐시는 저장소에 커밋하지 않는다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import build_source_backtrack as backtrack


CACHE_DIR = Path("data/public_source_cache")
SOURCES_PATH = Path("outputs/source_backtrack/sources.csv")
SUMMARY_PATH = Path("outputs/source_backtrack/source_expansion_summary.json")
AIR_FILES = {
    "익산시": CACHE_DIR / "air_emission_iksan.csv",
    "김제시": CACHE_DIR / "air_emission_gimje.csv",
}
SEWAGE_FILE = CACHE_DIR / "public_sewage_20241231.csv"
AIR_SOURCE_URL = "https://file.localdata.go.kr/file/air_pollution_facility_installation/info"
SEWAGE_SOURCE_URL = "https://www.data.go.kr/data/3073222/fileData.do"


def stable_id(prefix: str, value: object) -> str:
    token = hashlib.sha1(f"{prefix}|{value}".encode("utf-8")).hexdigest()[:12].upper()
    return f"{prefix}-{token}"


def text(value: object) -> str:
    return "" if pd.isna(value) else str(value).strip()


def first_text(*values: object) -> str:
    for value in values:
        cleaned = text(value)
        if cleaned:
            return cleaned
    return ""


def complete_address(city: str, value: object) -> str:
    address = text(value)
    if not address:
        return ""
    if city not in address:
        address = f"전북특별자치도 {city} {address}"
    return backtrack.normalize_address(address)


def classify_air_source(name: str, industry: str, product: str) -> str:
    combined = " ".join((name, industry, product))
    if any(key in combined for key in ("하수처리", "폐수처리", "분뇨처리", "오수처리")):
        return "wastewater"
    if any(key in combined for key in (
        "폐기물 처리", "폐기물처리", "폐기물 재활용", "폐기물재활용",
        "소각", "매립", "퇴비", "액비", "자원화",
    )):
        return "other"
    return "factory"


def geocode_frame(frame: pd.DataFrame, key: str, workers: int) -> pd.DataFrame:
    cache = backtrack.geocode_addresses(frame["address"].tolist(), key, workers=workers)
    result = frame.copy()
    result["latitude"] = result["address"].map(lambda x: cache.get(x, {}).get("latitude"))
    result["longitude"] = result["address"].map(lambda x: cache.get(x, {}).get("longitude"))
    result["location_precision"] = np.where(
        result["latitude"].notna() & result["longitude"].notna(), "point", None,
    )
    result["geocode_method"] = result["address"].map(
        lambda x: f"vworld_{cache.get(x, {}).get('address_type')}"
        if cache.get(x, {}).get("latitude") is not None else "not_found"
    )
    return result


def load_air_sources() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for city, path in AIR_FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"대기배출시설 원문이 없습니다: {path}")
        frame = pd.read_csv(path, encoding="cp949")
        frame = frame[frame["영업상태명"].eq("영업/정상")].copy()
        for item in frame.to_dict("records"):
            address = complete_address(city, first_text(item.get("도로명주소"), item.get("지번주소")))
            name = text(item.get("사업장명"))
            industry = text(item.get("업태구분명"))
            product = text(item.get("주생산품명"))
            extra = {
                "dataset": "행정안전부 지방행정 인허가 대기오염물질배출시설설치사업장",
                "source_url": AIR_SOURCE_URL,
                "original_id": text(item.get("관리번호")),
                "industry_name": industry or None,
                "industry_code": text(item.get("업종구분코드")) or None,
                "facility_class": text(item.get("종별명")) or None,
                "main_product": product or None,
                "last_modified": text(item.get("최종수정시점")) or None,
            }
            rows.append({
                "source_id": stable_id("AIR", item.get("관리번호")),
                "source_type": classify_air_source(name, industry, product),
                "city": city, "name": name, "species": np.nan, "head_count": np.nan,
                "area_m2": np.nan, "status": text(item.get("영업상태명")), "address": address,
                "emission_weight": np.nan, "weight_imputed": 0,
                "extra_json": json.dumps(extra, ensure_ascii=False, separators=(",", ":")),
            })
    return pd.DataFrame(rows)


def load_sewage_sources() -> pd.DataFrame:
    if not SEWAGE_FILE.exists():
        raise FileNotFoundError(f"공공하수처리시설 원문이 없습니다: {SEWAGE_FILE}")
    frame = pd.read_csv(SEWAGE_FILE, encoding="utf-8-sig")
    frame = frame[frame["구군"].isin(["익산시", "김제시"])].copy()
    rows: list[dict[str, object]] = []
    for item in frame.to_dict("records"):
        city = text(item.get("구군"))
        name = f"{text(item.get('시설명'))} 공공하수처리시설"
        address = complete_address(city, item.get("소재지"))
        identity = f"{city}|{item.get('시설명')}|{address}"
        extra = {
            "dataset": "한국환경공단 공공하수처리시설 현황 2024",
            "source_url": SEWAGE_SOURCE_URL,
            "facility_capacity_m3_day": item.get("시설용량"),
            "treatment_method": text(item.get("처리방법")) or None,
            "startup_date": text(item.get("가동개시일")) or None,
            "linked_night_soil": item.get("연계처리량_분뇨"),
            "linked_livestock_wastewater": item.get("연계처리량_축산"),
            "data_date": "2024-12-31",
        }
        rows.append({
            "source_id": stable_id("SEWAGE", identity), "source_type": "wastewater",
            "city": city, "name": name, "species": np.nan, "head_count": np.nan,
            "area_m2": np.nan, "status": "운영 현황 수록", "address": address,
            "emission_weight": np.nan, "weight_imputed": 0,
            "extra_json": json.dumps(extra, ensure_ascii=False, separators=(",", ":"), default=str),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if not SOURCES_PATH.exists():
        raise FileNotFoundError(f"기존 축산 발생원 표가 없습니다: {SOURCES_PATH}")

    original = pd.read_csv(SOURCES_PATH, encoding="utf-8-sig")
    livestock = original[original["source_type"].eq("livestock")].copy()
    if "extra_json" not in livestock:
        livestock["extra_json"] = "{}"
    else:
        livestock["extra_json"] = livestock["extra_json"].fillna("{}")

    candidates = pd.concat([load_air_sources(), load_sewage_sources()], ignore_index=True)
    candidates = candidates[candidates["address"].ne("")].drop_duplicates(
        ["source_type", "city", "name", "address"], keep="last",
    )
    key = backtrack.load_env_key("VWORLD_API_KEY")
    candidates = geocode_frame(candidates, key, args.workers)
    combined = pd.concat([livestock, candidates], ignore_index=True)[backtrack.SOURCE_COLUMNS]
    combined.to_csv(SOURCES_PATH, index=False, encoding="utf-8-sig")

    by_type: dict[str, dict[str, object]] = {}
    for source_type, group in candidates.groupby("source_type"):
        by_type[source_type] = {
            "rows": int(len(group)),
            "geocoded": int(group["latitude"].notna().sum()),
            "geocode_rate": float(group["latitude"].notna().mean()),
            "by_city": {str(k): int(v) for k, v in group["city"].value_counts().items()},
        }
    summary = {
        "livestock_rows_preserved": int(len(livestock)),
        "non_livestock_rows": int(len(candidates)),
        "sources_total": int(len(combined)),
        "by_type": by_type,
        "source_urls": [AIR_SOURCE_URL, SEWAGE_SOURCE_URL],
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
