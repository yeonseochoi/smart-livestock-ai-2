"""공개 원문에서 축산 외 후보 발생원을 표준화해 로컬 sources.csv에 추가한다.

좌표가 포함된 결과와 VWorld 응답 캐시는 저장소에 커밋하지 않는다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import build_source_backtrack as backtrack


CACHE_DIR = Path("data/public_source_cache")
SOURCES_PATH = Path("outputs/source_backtrack/sources.csv")
SUMMARY_PATH = Path("outputs/source_backtrack/source_expansion_summary.json")
ODOR_FACTORY_FILE = CACHE_DIR / "odor_relevant_factories.csv"
AIR_FILES = {
    "익산시": CACHE_DIR / "air_emission_iksan.csv",
    "김제시": CACHE_DIR / "air_emission_gimje.csv",
}
SEWAGE_FILE = CACHE_DIR / "public_sewage_20241231.csv"
INDUSTRIAL_ZONE_FILE = CACHE_DIR / "industrial_zones.geojson"
MANURE_FILE = CACHE_DIR / "manure_facilities.csv"
WASTE_FILE = CACHE_DIR / "waste_facilities.csv"
AIR_SOURCE_URL = "https://file.localdata.go.kr/file/air_pollution_facility_installation/info"
SEWAGE_SOURCE_URL = "https://www.data.go.kr/data/3073222/fileData.do"
INDUSTRIAL_ZONE_SOURCE_URL = "https://www.vworld.kr/dtmk/dtmk_ntads_s002.do?dsId=30137"
MANURE_SOURCE_URL = "https://www.data.go.kr/data/15055647/fileData.do"


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


def geocode_frame(frame: pd.DataFrame, key: str, workers: int) -> pd.DataFrame:
    result = frame.copy()
    missing = result["latitude"].isna() & result["longitude"].isna() & result["address"].ne("")
    addresses = result.loc[missing, "address"].tolist()
    cache = backtrack.geocode_addresses(addresses, key, workers=workers) if addresses else {}
    result.loc[missing, "latitude"] = result.loc[missing, "address"].map(
        lambda x: cache.get(x, {}).get("latitude")
    )
    result.loc[missing, "longitude"] = result.loc[missing, "address"].map(
        lambda x: cache.get(x, {}).get("longitude")
    )
    matched = missing & result["latitude"].notna() & result["longitude"].notna()
    point_precision = matched & result["location_precision"].isna()
    result.loc[point_precision, "location_precision"] = "point"
    result.loc[matched, "geocode_method"] = result.loc[matched, "address"].map(
        lambda x: f"vworld_{cache.get(x, {}).get('address_type')}"
    )
    result.loc[missing & ~matched, "geocode_method"] = "not_found"
    return result


def empty_coordinates(row: dict[str, object]) -> dict[str, object]:
    row.update({
        "latitude": np.nan,
        "longitude": np.nan,
        "location_precision": None,
        "geocode_method": None,
    })
    return row


def load_factory_sources() -> pd.DataFrame:
    if not ODOR_FACTORY_FILE.exists():
        raise FileNotFoundError(f"필터 후 공장 목록이 없습니다: {ODOR_FACTORY_FILE}")
    frame = pd.read_csv(ODOR_FACTORY_FILE, encoding="utf-8-sig")
    rows: list[dict[str, object]] = []
    for item in frame.to_dict("records"):
        city = text(item.get("city"))
        address = complete_address(city, item.get("address"))
        extra = {
            "dataset": "행정안전부 지방행정 인허가 대기오염물질배출시설설치사업장",
            "source_url": AIR_SOURCE_URL,
            "original_id": text(item.get("management_id")),
            "business_type": text(item.get("business_type")) or None,
            "industry_name": text(item.get("industry_name")) or None,
            "facility_class": text(item.get("facility_class")) or None,
            "main_product": text(item.get("main_product")) or None,
            "filter_reason": text(item.get("filter_reason")),
            "last_modified": text(item.get("last_modified")) or None,
        }
        rows.append(empty_coordinates({
            "source_id": stable_id("AIR", item.get("management_id")),
            "source_type": "factory", "odor_relevant": True,
            "city": city, "name": text(item.get("name")), "species": np.nan,
            "head_count": np.nan, "area_m2": np.nan, "status": text(item.get("status")),
            "address": address, "emission_weight": np.nan, "weight_imputed": 0,
            "extra_json": json.dumps(extra, ensure_ascii=False, separators=(",", ":")),
        }))
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
            "odor_relevant": True,
            "city": city, "name": name, "species": np.nan, "head_count": np.nan,
            "area_m2": np.nan, "status": "운영 현황 수록", "address": address,
            "latitude": np.nan, "longitude": np.nan,
            "location_precision": None, "geocode_method": None,
            "emission_weight": np.nan, "weight_imputed": 0,
            "extra_json": json.dumps(extra, ensure_ascii=False, separators=(",", ":"), default=str),
        })
    return pd.DataFrame(rows)


def load_air_wastewater_sources() -> pd.DataFrame:
    """대기배출시설 중 명칭·업종에 하수·폐수·오수·분뇨처리가 명시된 시설을 보완한다."""
    rows: list[dict[str, object]] = []
    for city, path in AIR_FILES.items():
        frame = pd.read_csv(path, encoding="cp949")
        frame = frame[frame["영업상태명"].eq("영업/정상")].copy()
        combined = frame[["사업장명", "업태구분명", "업종구분명", "주생산품명"]].fillna("").agg(" ".join, axis=1)
        frame = frame[combined.str.contains(r"하수처리|폐수처리|분뇨처리|오수처리", regex=True)]
        for item in frame.to_dict("records"):
            address = complete_address(city, first_text(item.get("도로명주소"), item.get("지번주소")))
            extra = {
                "dataset": "행정안전부 지방행정 인허가 대기오염물질배출시설설치사업장",
                "source_url": AIR_SOURCE_URL,
                "original_id": text(item.get("관리번호")),
                "facility_class": text(item.get("종별명")) or None,
            }
            rows.append(empty_coordinates({
                "source_id": stable_id("AIR-WW", item.get("관리번호")),
                "source_type": "wastewater", "odor_relevant": True, "city": city,
                "name": text(item.get("사업장명")), "species": np.nan, "head_count": np.nan,
                "area_m2": np.nan, "status": text(item.get("영업상태명")), "address": address,
                "emission_weight": np.nan, "weight_imputed": 0,
                "extra_json": json.dumps(extra, ensure_ascii=False, separators=(",", ":")),
            }))
    return pd.DataFrame(rows)


def polygon_centroid(ring: list[list[float]]) -> tuple[float, float]:
    """GeoJSON 외곽선의 면적가중 중심점을 경위도로 반환한다."""
    twice_area = 0.0
    x_sum = 0.0
    y_sum = 0.0
    for left, right in zip(ring, ring[1:]):
        cross = left[0] * right[1] - right[0] * left[1]
        twice_area += cross
        x_sum += (left[0] + right[0]) * cross
        y_sum += (left[1] + right[1]) * cross
    if math.isclose(twice_area, 0.0):
        return float(np.mean([p[0] for p in ring])), float(np.mean([p[1] for p in ring]))
    return x_sum / (3.0 * twice_area), y_sum / (3.0 * twice_area)


def load_industrial_zones() -> pd.DataFrame:
    if not INDUSTRIAL_ZONE_FILE.exists():
        raise FileNotFoundError(f"산업단지 경계가 없습니다: {INDUSTRIAL_ZONE_FILE}")
    payload = json.loads(INDUSTRIAL_ZONE_FILE.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    for feature in payload["features"]:
        prop = feature["properties"]
        geometry = feature["geometry"]
        if geometry["type"] != "Polygon":
            raise ValueError(f"지원하지 않는 산업단지 geometry: {geometry['type']}")
        lon, lat = polygon_centroid(geometry["coordinates"][0])
        identity = f"{prop['dan_id']}|{prop['name']}"
        extra = {
            "dataset": "VWorld 산업단지 경계도면",
            "source_url": INDUSTRIAL_ZONE_SOURCE_URL,
            "dan_id": prop["dan_id"],
            "zone_type": prop["zone_type"],
            "boundary_file": str(INDUSTRIAL_ZONE_FILE).replace("\\", "/"),
            "source_updated": prop["source_updated"],
        }
        rows.append({
            "source_id": stable_id("ZONE", identity), "source_type": "industrial_zone",
            "odor_relevant": True, "city": prop["city"], "name": prop["name"],
            "species": np.nan, "head_count": np.nan, "area_m2": np.nan,
            "status": "경계 수집", "address": f"전북특별자치도 {prop['city']} {prop['name']}",
            "latitude": lat, "longitude": lon, "location_precision": "polygon_centroid",
            "geocode_method": "vworld_boundary_centroid", "emission_weight": np.nan,
            "weight_imputed": 0,
            "extra_json": json.dumps(extra, ensure_ascii=False, separators=(",", ":")),
        })
    return pd.DataFrame(rows)


def load_facility_file(path: Path, source_type: str, prefix: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"시설 목록이 없습니다: {path}")
    frame = pd.read_csv(path, encoding="utf-8-sig")
    rows: list[dict[str, object]] = []
    basic = {"city", "name", "address", "address_precision", "status"}
    for item in frame.to_dict("records"):
        city = text(item.get("city"))
        address = complete_address(city, item.get("address"))
        identity = f"{city}|{item.get('name')}|{address}"
        extra = {
            key: (None if pd.isna(value) else value)
            for key, value in item.items() if key not in basic
        }
        precision = "village" if text(item.get("address_precision")) == "리" else None
        rows.append({
            "source_id": stable_id(prefix, identity), "source_type": source_type,
            "odor_relevant": True, "city": city, "name": text(item.get("name")),
            "species": np.nan, "head_count": np.nan, "area_m2": np.nan,
            "status": first_text(item.get("status"), item.get("operation")), "address": address,
            "latitude": np.nan, "longitude": np.nan, "location_precision": precision,
            "geocode_method": None, "emission_weight": np.nan, "weight_imputed": 0,
            "extra_json": json.dumps(extra, ensure_ascii=False, separators=(",", ":"), default=str),
        })
    return pd.DataFrame(rows)


def normalized_key(value: object) -> str:
    return "".join(text(value).split()).replace("(주)", "주식회사")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if not SOURCES_PATH.exists():
        raise FileNotFoundError(f"기존 축산 발생원 표가 없습니다: {SOURCES_PATH}")

    original = pd.read_csv(SOURCES_PATH, encoding="utf-8-sig")
    livestock = original[original["source_type"].eq("livestock")].copy()
    livestock["odor_relevant"] = True
    if "extra_json" not in livestock:
        livestock["extra_json"] = "{}"
    else:
        livestock["extra_json"] = livestock["extra_json"].fillna("{}")

    factories = load_factory_sources()
    sewage = pd.concat([load_sewage_sources(), load_air_wastewater_sources()], ignore_index=True)
    zones = load_industrial_zones()
    manure = load_facility_file(MANURE_FILE, "manure_plant", "MANURE")
    waste = load_facility_file(WASTE_FILE, "waste_facility", "WASTE")

    # 같은 사업장이 공장 원문에도 있으면 더 구체적인 시설 유형을 우선함.
    specialized = pd.concat([sewage, manure, waste], ignore_index=True)
    specialized_addresses = set(specialized["address"].map(normalized_key)) - {""}
    specialized_names = set(specialized["name"].map(normalized_key)) - {""}
    duplicate_factory = (
        factories["address"].map(normalized_key).isin(specialized_addresses)
        | factories["name"].map(normalized_key).isin(specialized_names)
    )
    factories = factories[~duplicate_factory].copy()

    candidates = pd.concat([factories, sewage, zones, manure, waste], ignore_index=True)
    candidates = candidates[candidates["address"].ne("")].drop_duplicates("source_id", keep="last")
    key = backtrack.load_env_key("VWORLD_API_KEY")
    candidates = geocode_frame(candidates, key, args.workers)
    combined = pd.concat([livestock, candidates], ignore_index=True)[backtrack.SOURCE_COLUMNS]
    combined.to_csv(SOURCES_PATH, index=False, encoding="utf-8-sig")

    by_type: dict[str, dict[str, object]] = {}
    for source_type, group in combined.groupby("source_type"):
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
        "source_urls_by_type": {
            "livestock": [],
            "factory": [AIR_SOURCE_URL],
            "wastewater": [SEWAGE_SOURCE_URL, AIR_SOURCE_URL],
            "industrial_zone": [INDUSTRIAL_ZONE_SOURCE_URL],
            "manure_plant": [
                MANURE_SOURCE_URL,
                AIR_SOURCE_URL,
                "https://repository.krei.re.kr/bitstream/2018.oak/20967/1/D384.pdf",
                "https://www.mafra.go.kr/bbs/home/792/573454/artclView.do",
                "https://www.gimje.go.kr/town/board/view.gimje?boardId=BBS_0000027&dataSid=164109",
            ],
            "waste_facility": [
                "https://council.iksan.go.kr/old/board/view.iksan?boardId=BBS_0000014&dataSid=20546",
                "https://www.iksan.go.kr/01kr/images/iksannews/11.pdf",
                "https://www.jeonbuk.go.kr/upload_data/board_data/BBS_0000017/147515311456176.pdf",
            ],
        },
        "checked_at": "2026-09-20",
        "source_urls": [
            AIR_SOURCE_URL, SEWAGE_SOURCE_URL, INDUSTRIAL_ZONE_SOURCE_URL, MANURE_SOURCE_URL,
            "https://council.iksan.go.kr/old/board/view.iksan?boardId=BBS_0000014&dataSid=20546",
        ],
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
