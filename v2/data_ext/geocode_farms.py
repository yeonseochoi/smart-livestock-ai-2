"""Iksan livestock-farm address cleaning and resumable Kakao address geocoding."""
from __future__ import annotations

import argparse, hashlib, json, os, random, re, time
from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "전북특별자치도 익산시_축산농가 현황_20241231.csv"
CACHE = HERE / "cache" / "kakao_address_cache.json"
OUT = HERE / "farms_geocoded.csv"
FAILED = HERE / "failed_addresses.csv"
REPORT = HERE / "farms_geocode_report.md"
KAKAO_URL = "https://dapi.kakao.com/v2/local/search/address.json"
OUT_COLS = ["순번", "업체명", "소재지(원문)", "정제주소", "lat", "lon", "geocode_method", "n_parcels", "사육업종", "사육두수", "시설면적", "영업상태", "신고일자"]


def load_v2_env() -> None:
    """Load v2/.env without logging its secrets; existing OS variables win."""
    env_file = HERE.parent / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name, value = name.strip(), value.strip().strip('"').strip("'")
        if name in {"KAKAO_REST_KEY", "KMA_API_KEY"} and value:
            os.environ.setdefault(name, value)


def normalize_address(raw: object) -> tuple[str, int, str]:
    """Return (first-parcel address, parcel count, ri-level fallback query)."""
    text = re.sub(r"\s+", " ", str(raw or "")).strip(" ,")
    # Exact duplicated local prefix, e.g. '성당면 두동리 성당면 두동리 99-1'.
    text = re.sub(r"\b((?:\S+(?:읍|면|동))\s+\S+리)\s+\1\b", r"\1", text)
    text = re.sub(r"(\d+)\s*번지\s*(\d+)\s*호", r"\1-\2", text)
    text = re.sub(r"(\d+)\s*번지\b", r"\1", text)
    text = re.sub(r"(\d+)\s*호\b", r"\1", text)
    parts = [p.strip() for p in text.split(",")]
    main = parts[0] if parts else text
    # A bare negative fragment is a residual annotation, not an additional parcel.
    extra = [p for p in parts[1:] if re.fullmatch(r"\d+(?:-\d+)?", p or "")]
    main_match = re.search(r"\b\d+(?:-\d+)?\b\s*$", main)
    n_parcels = int(bool(main_match)) + len(extra)
    clean = re.sub(r"\s+", " ", main).strip(" ,")
    m = re.search(r"^(.*?\b\S+(?:읍|면|동)\s+\S+리)\b", clean)
    ri_query = m.group(1) if m else re.sub(r"\s+\d+(?:-\d+)?\s*$", "", clean)
    return clean, max(n_parcels, 1), ri_query.strip()


def load_cache() -> dict:
    if not CACHE.exists(): return {}
    try: return json.loads(CACHE.read_text(encoding="utf-8"))
    except json.JSONDecodeError: return {}


def save_cache(cache: dict) -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    # Windows file indexers can transiently reject an otherwise valid path.
    # A failed checkpoint must never discard already-received API responses or
    # abort the full job; the following request will retry the checkpoint.
    payload = json.dumps(cache, ensure_ascii=False)
    for attempt in range(3):
        try:
            CACHE.write_text(payload, encoding="utf-8")
            return
        except OSError:
            time.sleep(0.25 * (attempt + 1))


def cache_key(query: str) -> str: return hashlib.sha256(query.encode("utf-8")).hexdigest()


def kakao_search(session: requests.Session, key: str, query: str, cache: dict) -> dict:
    ck = cache_key(query)
    if ck in cache and not str(cache[ck].get("error") or "").startswith("HTTP 40"):
        return cache[ck]
    cache.pop(ck, None)
    headers = {"Authorization": f"KakaoAK {key}"}
    result: dict = {"query": query, "documents": [], "error": None}
    for attempt in range(4):
        try:
            res = session.get(KAKAO_URL, headers=headers, params={"query": query}, timeout=20)
            if res.status_code in (401, 403):
                # Authentication failures are not cached: after the user fixes
                # v2/.env the same address must be fetched again.
                return {"query": query, "documents": [], "error": f"HTTP {res.status_code}: Kakao Local API 활용키/권한 확인 필요"}
            if res.status_code == 429 or res.status_code >= 500:
                time.sleep(1.0 * (2 ** attempt)); continue
            res.raise_for_status(); result["documents"] = res.json().get("documents", []); break
        except requests.RequestException as exc:
            result["error"] = type(exc).__name__
            time.sleep(1.0 * (2 ** attempt))
    cache[ck] = result; save_cache(cache); time.sleep(0.12)
    return result


def geocode(df: pd.DataFrame, key: str) -> pd.DataFrame:
    cache, session, rows = load_cache(), requests.Session(), []
    for _, src in df.iterrows():
        original = src["소재지"]
        clean, n_parcels, ri = normalize_address(original)
        first = kakao_search(session, key, clean, cache)
        docs, method = first.get("documents", []), "지번정확" if not n_parcels > 1 else "첫필지"
        if not docs:
            second = kakao_search(session, key, ri, cache)
            docs, method = second.get("documents", []), "리중심"
        lat = lon = None
        if docs:
            doc = docs[0]; lon, lat = doc.get("x"), doc.get("y")
        else: method = "실패"
        rows.append({"순번": src["순번"], "업체명": src["업체명"], "소재지(원문)": original, "정제주소": clean,
                     "lat": lat, "lon": lon, "geocode_method": method, "n_parcels": n_parcels,
                     "사육업종": src.get("사육업종"), "사육두수": src.get("사육두수"),
                     "시설면적": src.get("시설면적(제곱미터)"), "영업상태": src.get("영업상태"), "신고일자": src.get("신고일자")})
    return pd.DataFrame(rows, columns=OUT_COLS)


def admin(address: str) -> str:
    m = re.search(r"(\S+(?:읍|면|동))", address or "")
    return m.group(1) if m else "미상"


def markdown_table(frame: pd.DataFrame) -> str:
    """Dependency-free GitHub-flavoured Markdown table."""
    if frame.empty:
        return "없음"
    def cell(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")
    cols = [str(c) for c in frame.columns]
    rows = [[cell(v) for v in row] for row in frame.itertuples(index=False, name=None)]
    return "| " + " | ".join(cols) + " |\n| " + " | ".join(["---"] * len(cols)) + " |\n" + "\n".join("| " + " | ".join(row) + " |" for row in rows)


def report(result: pd.DataFrame) -> None:
    ok = result.lat.notna() & result.lon.notna()
    outside = result[ok & ~result.lat.astype(float).between(35.8, 36.2) | ok & ~result.lon.astype(float).between(126.8, 127.2)]
    work = result[ok].copy(); work["admin"] = work["정제주소"].map(admin)
    work["lat"] = work.lat.astype(float); work["lon"] = work.lon.astype(float)
    med = work.groupby("admin")[["lat", "lon"]].median().rename(columns={"lat":"mlat","lon":"mlon"})
    work = work.join(med, on="admin")
    work["distance_km"] = ((work.lat-work.mlat)*110.54).pow(2).add(((work.lon-work.mlon)*111.32*0.81).pow(2)).pow(.5)
    suspicious = work[work.distance_km > 5]
    sample = result.sample(min(20, len(result)), random_state=20260921)[["순번","업체명","정제주소","lat","lon","geocode_method"]]
    counts = result.groupby("geocode_method").size().rename("건수").to_frame(); counts["성공률"] = (counts.index != "실패").astype(float)
    REPORT.write_text("# 축산농가 지오코딩 검증\n\n## 방법별 건수·성공률\n\n" + markdown_table(counts.reset_index()) +
        f"\n\n전체 성공률: **{ok.mean():.1%}**\n\n## 익산시 범위 밖 ({len(outside)}건)\n\n" + markdown_table(outside[OUT_COLS]) +
        f"\n\n## 읍면동 중앙값에서 5km 초과 ({len(suspicious)}건)\n\n" + markdown_table(suspicious[["순번","업체명","정제주소","lat","lon","admin","distance_km"]]) +
        "\n\n## 무작위 20건\n\n" + markdown_table(sample) + "\n", encoding="utf-8")
    result[result.geocode_method.eq("실패")].to_csv(FAILED, index=False, encoding="utf-8-sig")


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--input", type=Path, default=SOURCE); args = ap.parse_args()
    load_v2_env()
    key = os.getenv("KAKAO_REST_KEY")
    if not key: raise SystemExit("KAKAO_REST_KEY가 없습니다. v2/.env 또는 현재 환경변수에 설정하세요.")
    source = pd.read_csv(args.input, encoding="utf-8-sig")
    result = geocode(source, key); result.to_csv(OUT, index=False, encoding="utf-8-sig"); report(result)
    print(f"완료: {len(result)}개 농가. 결과={OUT.name}, 실패목록={FAILED.name}")

if __name__ == "__main__": main()
