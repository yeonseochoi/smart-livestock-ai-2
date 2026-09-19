"""VWorld 건물통합정보(LT_C_BLDGINFO)를 1km 격자별 집계값으로 수집한다.

목적: "바람 아래라서"와 "사람이 많아서"를 분리하기 위한 인구·건물 밀도 대리 변수.
SGIS 격자 인구는 로그인 다운로드가 필요해 우선 VWorld로 건물 밀도를 만든다.

저장하는 것은 격자별 집계(건물 수, 주거·근생 건물 수, 연면적 합, 평균 지상층수)뿐이다. 건물 좌표·개별 속성은 저장하지 않는다.
격자 원점·크기는 build_odor_ai_mvp.add_grid_columns와 같고(민원 위·경도 중앙값, 1000m), 중심 좌표를 함께 기록한다.
캐시: outputs/grid_buildings/cells.jsonl (격자별 한 줄, 재실행 시 이어서).

실행: python fetch_grid_buildings.py [--margin-cells 2] [--max-cells N]
"""
from __future__ import annotations

import argparse
import json
import math
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

import build_odor_ai_mvp as odor

OUTPUT_DIR = Path("outputs/grid_buildings")
CACHE = OUTPUT_DIR / "cells.jsonl"
GRID_M = 1000
PAGE = 1000
# usability는 건축물대장 주용도 코드(5자리). 01000 단독주택, 02000 공동주택, 03000/04000 근린생활시설.
RESIDENTIAL_PREFIXES = ("01", "02")
COMMERCIAL_PREFIXES = ("03", "04")


def api_key() -> str:
    env = Path(".env")
    if not env.exists():
        raise SystemExit(".env에 VWORLD_API_KEY가 필요합니다.")
    values = dict(line.strip().split("=", 1) for line in env.read_text(encoding="utf-8").splitlines() if "=" in line)
    return values["VWORLD_API_KEY"]


def cell_bounds(gx: int, gy: int, meta: dict) -> tuple[float, float, float, float]:
    lat_c, lon_c = odor.grid_centroid(gx, gy, meta)
    half_lat = (GRID_M / 2) / 110_540
    half_lon = (GRID_M / 2) / (111_320 * math.cos(math.radians(meta["lat0"])))
    return lon_c - half_lon, lat_c - half_lat, lon_c + half_lon, lat_c + half_lat


def fetch_box(bounds: tuple[float, float, float, float], key: str, page_size: int = PAGE, depth: int = 0) -> dict:
    """한 상자의 건물 집계. 밀집 구역에서 서버가 연결을 끊으면 상자를 4등분해 재귀 수집한다."""
    lon0, lat0, lon1, lat1 = bounds
    box = f"BOX({lon0:.6f},{lat0:.6f},{lon1:.6f},{lat1:.6f})"
    total = None
    count = residential = commercial = 0
    floor_area = 0.0
    floors = []
    page = 1
    while True:
        query = urllib.parse.urlencode({
            "service": "data", "request": "GetFeature", "data": "LT_C_BLDGINFO", "key": key, "domain": "localhost",
            "geomFilter": box, "size": page_size, "page": page, "format": "json", "crs": "EPSG:4326", "geometry": "false",
        })
        for attempt in range(4):
            try:
                with urllib.request.urlopen("https://api.vworld.kr/req/data?" + query, timeout=60) as response:
                    payload = json.loads(response.read().decode("utf-8"))["response"]
                break
            except Exception:
                time.sleep(2 * (attempt + 1))
        else:
            if depth >= 3:
                raise RuntimeError(f"VWorld 응답 실패: {box} page {page}")
            mid_lon, mid_lat = (lon0 + lon1) / 2, (lat0 + lat1) / 2
            parts = [fetch_box(b, key, max(100, page_size // 2), depth + 1) for b in (
                (lon0, lat0, mid_lon, mid_lat), (mid_lon, lat0, lon1, mid_lat),
                (lon0, mid_lat, mid_lon, lat1), (mid_lon, mid_lat, lon1, lat1))]
            floors_all = [f for part in parts for f in part["_floors"]]
            return {"count": sum(p["count"] for p in parts), "residential": sum(p["residential"] for p in parts),
                    "commercial": sum(p["commercial"] for p in parts), "floor_area": sum(p["floor_area"] for p in parts),
                    "_floors": floors_all, "pages": sum(p["pages"] for p in parts)}
        if payload.get("status") == "NOT_FOUND":
            total = 0
            break
        if payload.get("status") != "OK":
            raise RuntimeError(f"VWorld 오류: {payload.get('error')}")
        total = int(payload["record"]["total"])
        features = payload["result"]["featureCollection"]["features"]
        for feature in features:
            props = feature["properties"]
            count += 1
            use = str(props.get("usability") or "").strip().zfill(5)
            if use.startswith(RESIDENTIAL_PREFIXES):
                residential += 1
            elif use.startswith(COMMERCIAL_PREFIXES):
                commercial += 1
            try:
                floor_area += float(props.get("totalarea") or 0)
            except ValueError:
                pass
            try:
                floors.append(int(float(props.get("grnd_flr") or 0)))
            except ValueError:
                pass
        if count >= total or not features:
            break
        page += 1
    return {"count": total if total is not None else count, "residential": residential, "commercial": commercial,
            "floor_area": floor_area, "_floors": floors, "pages": page}


def fetch_cell(gx: int, gy: int, meta: dict, key: str) -> dict:
    agg = fetch_box(cell_bounds(gx, gy, meta), key)
    lat_c, lon_c = odor.grid_centroid(gx, gy, meta)
    floors = agg["_floors"]
    return {
        "grid_x": gx, "grid_y": gy, "center_latitude": lat_c, "center_longitude": lon_c,
        "building_count": agg["count"], "residential_count": agg["residential"], "commercial_count": agg["commercial"],
        "total_floor_area_m2": round(agg["floor_area"], 1), "mean_ground_floors": (sum(floors) / len(floors)) if floors else 0.0,
        "pages": agg["pages"],
    }


def target_cells(margin: int) -> tuple[list[tuple[int, int]], dict]:
    complaints, _, _, _, _ = odor.load_inputs()
    meta = {"lat0": float(complaints["latitude"].median()), "lon0": float(complaints["longitude"].median()), "grid_m": GRID_M}
    x_m = (complaints["longitude"] - meta["lon0"]) * 111_320 * math.cos(math.radians(meta["lat0"]))
    y_m = (complaints["latitude"] - meta["lat0"]) * 110_540
    gx = (x_m // GRID_M).astype(int)
    gy = (y_m // GRID_M).astype(int)
    cells = set()
    for x, y in set(zip(gx, gy)):
        for dx in range(-margin, margin + 1):
            for dy in range(-margin, margin + 1):
                cells.add((int(x + dx), int(y + dy)))
    return sorted(cells), meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--margin-cells", type=int, default=2, help="민원이 있는 격자 주변 몇 칸까지 수집할지")
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.15)
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    key = api_key()
    cells, meta = target_cells(args.margin_cells)
    done = {}
    if CACHE.exists():
        for line in CACHE.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            done[(row["grid_x"], row["grid_y"])] = row
    todo = [c for c in cells if c not in done]
    if args.max_cells:
        todo = todo[: args.max_cells]
    print(f"대상 격자 {len(cells)}개, 완료 {len(done)}개, 이번 실행 {len(todo)}개")
    with CACHE.open("a", encoding="utf-8") as sink:
        for index, (gx, gy) in enumerate(todo, 1):
            row = fetch_cell(gx, gy, meta, key)
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            sink.flush()
            done[(gx, gy)] = row
            if index % 25 == 0:
                print(f"  {index}/{len(todo)} (최근 격자 건물 {row['building_count']}동)")
            time.sleep(args.sleep)
    table = pd.DataFrame(list(done.values())).sort_values(["grid_x", "grid_y"])
    table["grid_m"] = GRID_M
    table["lat0"] = meta["lat0"]
    table["lon0"] = meta["lon0"]
    table.to_csv(OUTPUT_DIR / "grid_buildings_1km.csv", index=False, encoding="utf-8-sig")
    print(f"저장: {OUTPUT_DIR / 'grid_buildings_1km.csv'} ({len(table)}격자, 건물 합계 {int(table['building_count'].sum()):,}동)")


if __name__ == "__main__":
    main()
