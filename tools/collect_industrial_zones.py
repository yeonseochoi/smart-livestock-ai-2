"""VWorld 산업단지 경계도면에서 익산시·김제시 경계를 추출한다.

원본: 국토교통부 산업단지 경계도면(DAM_DAN.zip, VWorld dsId=30137)
원본 좌표계: EPSG:5186, 출력 좌표계: EPSG:4326
"""
from __future__ import annotations

import argparse
import json
import tempfile
import zipfile
from pathlib import Path

try:
    import shapefile
    from pyproj import CRS, Transformer
except ImportError as exc:  # pragma: no cover - 실행 환경 안내
    raise SystemExit("pyshp와 pyproj가 필요함: python -m pip install pyshp pyproj") from exc


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "public_source_cache" / "DAM_DAN.zip"
DEFAULT_OUTPUT = ROOT / "data" / "public_source_cache" / "industrial_zones.geojson"
SOURCE_URL = "https://www.vworld.kr/dtmk/dtmk_ntads_s002.do?svcCde=MK&dsId=30137"
SOURCE_UPDATED = "2026-04-07"
CHECKED_AT = "2026-09-20"

# 익산시·김제시 공식 산업단지 현황과 SHP의 단지명·중심 위치를 대조한 코드임.
ZONE_CITY = {
    # 익산시: 국가 2, 일반 3, 농공 5
    "145030": "익산시",  # 익산 국가산업단지
    "145040": "익산시",  # 국가식품클러스터
    "245050": "익산시",  # 익산2
    "245190": "익산시",  # 익산4
    "245200": "익산시",  # 익산3(산업)
    "445130": "익산시",  # 낭산
    "445140": "익산시",  # 삼기
    "445150": "익산시",  # 황등
    "445380": "익산시",  # 왕궁
    "445580": "익산시",  # 함열
    # 김제시: 일반 4, 농공 7
    "245020": "김제시",  # 김제순동
    "245210": "김제시",  # 지평선
    "245290": "김제시",  # 백구 일반산업단지
    "245300": "김제시",  # 지평선2
    "445040": "김제시",  # 만경
    "445050": "김제시",  # 김제봉황
    "445060": "김제시",  # 서흥
    "445070": "김제시",  # 월촌
    "445080": "김제시",  # 황산
    "445350": "김제시",  # 대동
    "445540": "김제시",  # 백구 농공단지
}
ZONE_TYPE = {
    "1": "국가산업단지",
    "2": "일반산업단지",
    "3": "도시첨단산업단지",
    "4": "농공단지",
}


def transform_coordinates(value: object, transformer: Transformer) -> object:
    """GeoJSON 좌표 배열의 모든 x, y를 EPSG:4326으로 변환한다."""
    if isinstance(value, (tuple, list)) and len(value) >= 2 and all(
        isinstance(item, (int, float)) for item in value[:2]
    ):
        lon, lat = transformer.transform(float(value[0]), float(value[1]))
        return [round(lon, 8), round(lat, 8)]
    if isinstance(value, (tuple, list)):
        return [transform_coordinates(item, transformer) for item in value]
    return value


def build_geojson(source_zip: Path) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="track_a_industrial_zones_") as temp_dir:
        with zipfile.ZipFile(source_zip) as archive:
            archive.extractall(temp_dir)
        base = Path(temp_dir) / "DAM_DAN"
        source_crs = CRS.from_wkt(base.with_suffix(".prj").read_text(encoding="utf-8"))
        transformer = Transformer.from_crs(source_crs, "EPSG:4326", always_xy=True)
        reader = shapefile.Reader(str(base), encoding="cp949")

        features: list[dict[str, object]] = []
        found: set[str] = set()
        for item in reader.iterShapeRecords():
            record = item.record.as_dict()
            dan_id = str(record["DAN_ID"]).strip()
            if dan_id not in ZONE_CITY:
                continue
            found.add(dan_id)
            geometry = dict(item.shape.__geo_interface__)
            geometry["coordinates"] = transform_coordinates(geometry["coordinates"], transformer)
            features.append({
                "type": "Feature",
                "id": dan_id,
                "properties": {
                    "dan_id": dan_id,
                    "name": str(record["DAN_NAME"]).strip(),
                    "city": ZONE_CITY[dan_id],
                    "zone_type_code": str(record["DANJI_TYPE"]).strip(),
                    "zone_type": ZONE_TYPE[str(record["DANJI_TYPE"]).strip()],
                    "source_updated": SOURCE_UPDATED,
                },
                "geometry": geometry,
            })
        reader.close()

    missing = sorted(set(ZONE_CITY) - found)
    if missing:
        raise RuntimeError(f"원본 SHP에서 대상 산업단지 코드를 찾지 못함: {missing}")
    features.sort(key=lambda feature: (
        feature["properties"]["city"],
        feature["properties"]["zone_type_code"],
        feature["properties"]["name"],
    ))
    return {
        "type": "FeatureCollection",
        "name": "익산시_김제시_산업단지_경계",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "source": "국토교통부 산업단지 경계도면",
        "source_url": SOURCE_URL,
        "source_updated": SOURCE_UPDATED,
        "checked_at": CHECKED_AT,
        "license": "CC BY-NC-ND (VWorld 표시 기준)",
        "features": features,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build_geojson(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    by_city: dict[str, int] = {}
    for feature in result["features"]:
        city = feature["properties"]["city"]
        by_city[city] = by_city.get(city, 0) + 1
    print(json.dumps({"rows": len(result["features"]), "by_city": by_city}, ensure_ascii=False))


if __name__ == "__main__":
    main()
