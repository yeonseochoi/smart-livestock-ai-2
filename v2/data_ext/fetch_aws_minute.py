"""Resumable KMA AWS minute-data collector for leakage-safe v2 weather features.

Official endpoints are checked with help=1 before collection:
  station: https://apihub.kma.go.kr/api/typ01/url/stn_inf.php (inf=AWS)
  minute : https://apihub.kma.go.kr/api/typ01/cgi-bin/url/nph-aws2_min
The collector makes one cached request per (nearby station, calendar date), then
keeps only [T-60min, T) records for requested Event prediction times.
"""
from __future__ import annotations

import argparse, csv, io, json, math, os, re, threading, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CACHE = HERE / "cache" / "kma"
STATIONS = HERE / "stations.csv"
OUT = HERE / "aws_minute_events.csv"
REPORT = HERE / "coverage_report.md"
PREFLIGHT = CACHE / "preflight.json"
STATION_URL = "https://apihub.kma.go.kr/api/typ01/url/stn_inf.php"
AWS_URL = "https://apihub.kma.go.kr/api/typ01/cgi-bin/url/nph-aws2_min"
SAMPLE_DAYS = ("20190715", "20210615", "20230615", "20250615")
MISSING = {-9, -9.0, -99, -99.0, -999, -999.0}
REQUEST_LOCK = threading.Lock()
NEXT_REQUEST_AT = 0.0


def load_v2_env() -> None:
    """Load v2/.env without logging its secrets; existing OS variables win."""
    env_file = ROOT / ".env"
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


def haversine_km(lat1, lon1, lat2, lon2):
    p = math.pi / 180; a = math.sin((lat2-lat1)*p/2)**2 + math.cos(lat1*p)*math.cos(lat2*p)*math.sin((lon2-lon1)*p/2)**2
    return 6371.0088 * 2 * math.asin(math.sqrt(a))


def get_key() -> str:
    key = os.getenv("KMA_API_KEY")
    if not key: raise SystemExit("KMA_API_KEY 환경변수가 없습니다. 키는 코드·로그·출력에 기록하지 않습니다.")
    return key


def request_cached(session, url: str, params: dict, cache_path: Path, key: str) -> str:
    if cache_path.exists(): return cache_path.read_text(encoding="utf-8")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    safe = dict(params); safe["authKey"] = key
    for attempt in range(4):
        try:
            global NEXT_REQUEST_AT
            with REQUEST_LOCK:
                wait = max(0.0, NEXT_REQUEST_AT - time.monotonic())
                if wait:
                    time.sleep(wait)
                NEXT_REQUEST_AT = time.monotonic() + 0.25  # max four new requests/sec
            r = session.get(url, params=safe, timeout=45)
            if r.status_code in (401, 403):
                api = "지상관측 지점정보 조회(AWS)" if url == STATION_URL else "지상관측 AWS 매분자료 조회"
                raise PermissionError(f"HTTP {r.status_code}: {api} API 활용신청 또는 인증키 권한이 필요합니다 (재시도하지 않음).")
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(1.2 * (2 ** attempt)); continue
            r.raise_for_status(); cache_path.write_text(r.text, encoding="utf-8"); return r.text
        except PermissionError: raise
        except requests.RequestException:
            if attempt == 3: raise
            time.sleep(1.2 * (2 ** attempt))
    raise RuntimeError("unreachable")


def csv_records(text: str) -> list[dict[str, str]]:
    """Parse KMA help/disp CSV while ignoring prose/comment lines."""
    lines = [x.strip() for x in text.splitlines() if x.strip() and not x.lstrip().startswith(("#", "<"))]
    header_i = next((i for i, line in enumerate(lines)
                     if "," in line and ("STN_ID" in line or (re.search(r"(^|,)TM(,|$)", line) and re.search(r"(^|,)STN(,|$)", line)))), None)
    if header_i is None: return []
    return list(csv.DictReader(io.StringIO("\n".join(lines[header_i:]))))


def station_records(text: str) -> list[dict[str, str]]:
    """Parse the API Hub's documented fixed-width AWS station listing."""
    rows = []
    pattern = re.compile(
        r"^\s*(?P<STN_ID>\d+)\s+(?P<LON>-?\d+(?:\.\d+)?)\s+(?P<LAT>-?\d+(?:\.\d+)?)"
        r"\s+\S+\s+\S+\s+\S+\s+\d+\s+\d+\s+(?P<STN_KO>.*?)\s{2,}(?:----|\S+)\s+"
    )
    for line in text.splitlines():
        m = pattern.match(line)
        if m:
            rows.append({k: v.strip() for k, v in m.groupdict().items()})
    return rows


def minute_records(text: str) -> list[dict[str, str]]:
    """Parse comma-separated observations following AWS's fixed-width help."""
    names = ["TM", "STN", "WD1", "WS1", "WDS", "WSS", "WD10", "WS10", "TA", "RE",
             "RN-15m", "RN-60m", "RN-12H", "RN-DAY", "HM", "PA", "PS", "TD"]
    rows = []
    for line in text.splitlines():
        if re.match(r"^\d{12},\d+", line.strip()):
            values = next(csv.reader([line]))
            rows.append(dict(zip(names, values)))
    return rows


def numeric(v):
    try:
        x = float(str(v).strip()); return np.nan if x in MISSING or x <= -50 else x
    except (ValueError, TypeError): return np.nan


def markdown_table(frame: pd.DataFrame) -> str:
    """Dependency-free GitHub-flavoured Markdown table."""
    if frame.empty:
        return "없음"
    def cell(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")
    cols = [str(c) for c in frame.columns]
    rows = [[cell(v) for v in row] for row in frame.itertuples(index=False, name=None)]
    return "| " + " | ".join(cols) + " |\n| " + " | ".join(["---"] * len(cols)) + " |\n" + "\n".join("| " + " | ".join(row) + " |" for row in rows)


def pick(row: dict, *names):
    upper = {str(k).upper(): v for k, v in row.items()}
    for name in names:
        if name.upper() in upper: return upper[name.upper()]
    return None


def complaint_center() -> tuple[float, float]:
    raw = pd.read_excel(ROOT.parent / "data" / "익산시 악취 민원 데이터_20190528-20260818.xlsx")
    x = raw[raw["시군구"].eq("익산시")]
    return float(x["위도"].astype(float).median()), float(x["경도"].astype(float).median())


def fetch_stations(key: str) -> pd.DataFrame:
    center_lat, center_lon = complaint_center(); session = requests.Session()
    # help=1 first; cache raw help exactly, without putting authKey in any output.
    help_text = request_cached(session, STATION_URL, {"inf":"AWS", "stn":"", "tm":"202506150000", "help":1}, CACHE / "schema" / "stations_help.txt", key)
    rows = station_records(help_text)
    if not rows: raise RuntimeError("AWS 지점정보 응답에서 고정폭 지점목록을 찾지 못했습니다. cache/schema/stations_help.txt를 확인하세요.")
    out = []
    for r in rows:
        lat, lon = numeric(pick(r, "LAT")), numeric(pick(r, "LON"))
        if np.isnan(lat) or np.isnan(lon): continue
        dist = haversine_km(center_lat, center_lon, lat, lon)
        if dist <= 20:
            out.append({"station_id": str(pick(r,"STN_ID")), "station_name": pick(r,"STN_KO") or "",
                        "lat":lat, "lon":lon, "distance_from_iksan_center_km":dist, "station_type":"AWS"})
    stations = pd.DataFrame(out).sort_values("distance_from_iksan_center_km")
    stations.to_csv(STATIONS, index=False, encoding="utf-8-sig"); return stations


def fetch_day(session, key: str, station: str, day: str, help_mode: int | None = None) -> str:
    path = CACHE / "minute" / f"aws_{station}_{day}.txt"
    params = {"tm1":day+"0000", "tm2":day+"2359", "stn":station, "disp":1}
    if help_mode is not None:
        params["help"] = help_mode
    return request_cached(session, AWS_URL, params, path, key)


def normalize_minute_rows(text: str, station: str) -> pd.DataFrame:
    rows = minute_records(text); output = []
    for r in rows:
        tm = pick(r, "TM", "TM1", "TIME")
        ts = pd.to_datetime(str(tm), format="%Y%m%d%H%M", errors="coerce")
        if pd.isna(ts): continue
        output.append({"station_id":station, "observed_at":ts,
                       "wind_direction_raw":pick(r,"WD1"), "wind_speed_raw":pick(r,"WS1"), "temperature_raw":pick(r,"TA"),
                       "humidity_raw":pick(r,"HM"), "precipitation_raw":pick(r,"RN-60m"),
                       "wind_direction":numeric(pick(r,"WD1")), "wind_speed":numeric(pick(r,"WS1")),
                       "temperature":numeric(pick(r,"TA")), "humidity":numeric(pick(r,"HM")), "precipitation":numeric(pick(r,"RN-60m"))})
    return pd.DataFrame(output)


def event_windows(path: Path) -> pd.DataFrame:
    ev = pd.read_csv(path); id_col = "event_id" if "event_id" in ev else "Event ID"
    if "t0" not in ev: raise ValueError(f"{path}에 t0 열이 없습니다.")
    out = pd.DataFrame({"event_id":ev[id_col].astype(str), "T":pd.to_datetime(ev["t0"]) + pd.Timedelta(minutes=30)})
    out["window_start"] = out["T"] - pd.Timedelta(minutes=60); out["window_end"] = out["T"] - pd.Timedelta(minutes=1)
    return out


def availability(stations: pd.DataFrame, key: str) -> pd.DataFrame:
    session = requests.Session(); rows = []
    # Document the minute schema before sampling real records.
    if len(stations): fetch_day(session, key, str(stations.iloc[0].station_id), SAMPLE_DAYS[0], help_mode=1)
    for day in SAMPLE_DAYS:
        for stn in stations.station_id.astype(str):
            df = normalize_minute_rows(fetch_day(session, key, stn, day), stn)
            fields = ["wind_direction","wind_speed","temperature","humidity","precipitation"]
            rows.append({"day":day, "station_id":stn, "records":len(df), **{f"missing_{f}":float(df[f].isna().mean()) if len(df) else 1.0 for f in fields}})
    return pd.DataFrame(rows)


def collect(events: pd.DataFrame, stations: pd.DataFrame, key: str) -> pd.DataFrame:
    session = requests.Session(); wanted_days = sorted(set(events.window_start.dt.strftime("%Y%m%d")) | set(events.window_end.dt.strftime("%Y%m%d")))
    frames = {}
    jobs = [(stn, day) for stn in stations.station_id.astype(str) for day in wanted_days]
    def fetch_one(stn: str, day: str):
        with requests.Session() as worker_session:
            return stn, day, normalize_minute_rows(fetch_day(worker_session, key, stn, day), stn)
    # Eight concurrent in-flight calls, but request starts are globally limited.
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(fetch_one, stn, day) for stn, day in jobs]
        for future in as_completed(futures):
            stn, day, frame = future.result()
            frames[(stn, day)] = frame
    station_frames = {
        stn: pd.concat([frames[(stn, day)] for day in wanted_days], ignore_index=True)
                .sort_values("observed_at").set_index("observed_at", drop=False)
        for stn in stations.station_id.astype(str)
    }
    output = []
    for e in events.itertuples(index=False):
        for stn in stations.station_id.astype(str):
            d = station_frames[stn].loc[e.window_start:e.window_end].copy()
            if len(d): d.insert(0, "event_id", e.event_id); d.insert(2, "station_name", stations.loc[stations.station_id.astype(str).eq(stn), "station_name"].iloc[0]); output.append(d)
    result = pd.concat(output, ignore_index=True) if output else pd.DataFrame(columns=["event_id","station_id","station_name","observed_at"])
    if len(result):
        joined = result.merge(events[["event_id", "T"]], on="event_id", how="left")
        assert (joined["observed_at"] < joined["T"]).all(), "누수: T 이후 관측시각 발견"
    return result


def write_report(events, stations, weather, avail, events_path: Path) -> None:
    fcols = ["wind_direction","wind_speed","temperature","humidity","precipitation"]
    expected = len(events) * len(stations) * 60
    by_station = weather.groupby("station_id")[fcols].apply(lambda x: x.isna().mean()).reset_index() if len(weather) else pd.DataFrame()
    by_event = weather.groupby("event_id")[fcols].apply(lambda x: x.isna().mean()).reset_index() if len(weather) else pd.DataFrame()
    found = weather.groupby(["event_id","station_id"]).size() if len(weather) else pd.Series(dtype=int)
    missing = [(e.event_id,s) for e in events.itertuples(index=False) for s in stations.station_id.astype(str) if found.get((e.event_id,s),0)==0]
    REPORT.write_text("# AWS 분자료 수집·결측 리포트\n\n" +
        f"- Event 파일: `{events_path}`\n- 관측소: {len(stations)}개, Event: {len(events)}개\n- 요청·저장 대상: 각 T 직전 60분만 (`관측시각 < T` assert 통과)\n" +
        "- 공식 기본 한도: 일 2,000회·5GB. 이 수집은 관측소×고유 날짜 요청이며 실제 잔여 한도는 API 허브 마이페이지에서 확인 필요.\n" +
        f"- 전체 수집 예상 호출: {len(stations) * len(set(events.window_start.dt.date) | set(events.window_end.dt.date))}회 (일 2,000회 기준 1일 이내 여부는 계정 잔여 한도에 따름)\n\n" +
        "## 표본일 가용성\n\n" + (markdown_table(avail) if len(avail) else "미실행") +
        "\n\n## 관측소별 변수 결측률\n\n" + (markdown_table(by_station) if len(by_station) else "수집 자료 없음") +
        "\n\n## Event별 변수 결측률\n\n" + (markdown_table(by_event) if len(by_event) else "수집 자료 없음") +
        f"\n\n## 자료가 전혀 없는 Event·관측소 조합 ({len(missing)}개)\n\n" + (markdown_table(pd.DataFrame(missing, columns=["event_id","station_id"])) if missing else "없음") + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--events", type=Path, default=ROOT / "outputs" / "trigger_k7" / "events.csv"); ap.add_argument("--preflight", action="store_true"); args = ap.parse_args()
    load_v2_env()
    key = get_key(); stations = fetch_stations(key); avail = availability(stations, key)
    events = event_windows(args.events)
    if args.preflight:
        PREFLIGHT.parent.mkdir(parents=True, exist_ok=True)
        PREFLIGHT.write_text(json.dumps({"events": str(args.events), "stations": len(stations), "sample_days": SAMPLE_DAYS}, ensure_ascii=False), encoding="utf-8")
        write_report(events, stations, pd.DataFrame(), avail, args.events); print("사전 점검 완료: stations.csv, coverage_report.md"); return
    if not PREFLIGHT.exists():
        raise SystemExit("대량 수집 전 --preflight를 먼저 실행하세요. 표본일 가용성과 관측소 목록을 확인해야 합니다.")
    weather = collect(events, stations, key)
    public = weather.rename(columns={"station_name":"관측소", "observed_at":"관측시각",
                                    "wind_direction":"풍향", "wind_speed":"풍속", "temperature":"기온",
                                    "humidity":"습도", "precipitation":"강수",
                                    "wind_direction_raw":"풍향_원자료", "wind_speed_raw":"풍속_원자료",
                                    "temperature_raw":"기온_원자료", "humidity_raw":"습도_원자료",
                                    "precipitation_raw":"강수_원자료"})
    public.to_csv(OUT, index=False, encoding="utf-8-sig"); write_report(events, stations, weather, avail, args.events)
    print(f"완료: {OUT.name} ({len(weather)}행)")

if __name__ == "__main__": main()
