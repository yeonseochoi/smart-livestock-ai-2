"""Observed wind association for candidate livestock clusters."""
import os

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from develop import common

OUT = common.OUTPUT_DIR / "standby"
SEED = 20260927
N_BASELINE = 30
N_BOOTSTRAP = int(os.environ.get("STANDBY_BOOTSTRAP", "200"))
# 기존 돼지 1두=1 규모 3000을 EEA 돼지 4.6 kg NH3/년으로 환산.
MIN_CLUSTER_NH3_KG_YEAR = 3000 * 4.6  # 돼지 3,000두 상당 = 13,800 kg NH3/년


def clusters(farms):
    ik = farms[(farms.city == "익산") & (farms.status == "정상") & farms.lat.notna() & farms.lon.notna()].copy()
    x, y = common.to_xy(ik.lat, ik.lon)
    ik["label"] = DBSCAN(eps=.8, min_samples=5).fit_predict(np.column_stack([x, y]))
    result = []
    for label, group in ik[ik.label >= 0].groupby("label"):
        eq = group["eq"].sum()
        if eq < MIN_CLUSTER_NH3_KG_YEAR:
            continue
        lat, lon = group.lat.mean(), group.lon.mean()
        radius = np.percentile(common.distance_km(lat, lon, group.lat, group.lon), 90)
        result.append(dict(cluster_id=f"IK-{label}", city="익산", cluster_lat=lat, cluster_lon=lon, n_farms=len(group), eq_sum=eq, radius=radius))
    gj = pd.read_csv(common.GIMJE_FARMS_FILE, encoding="utf-8-sig")
    yongji = gj[gj["소재지"].astype(str).str.contains("용지면")].copy()
    yongji["위도"] = pd.to_numeric(yongji["위도"], errors="coerce")
    yongji["경도"] = pd.to_numeric(yongji["경도"], errors="coerce")
    yongji = yongji.dropna(subset=["위도", "경도"])
    if yongji.empty:
        raise ValueError("용지면 좌표 보유 농가 없음")
    result.append(dict(cluster_id="YONGJI", city="김제", cluster_lat=yongji["위도"].median(), cluster_lon=yongji["경도"].median(), n_farms=len(yongji), eq_sum=np.nan, radius=0.0))
    return pd.DataFrame(result)


def evaluate(cluster, complaints, weather, rng):
    weather = weather[(weather.wind_speed >= .5) & weather.wind_direction.notna()].copy()
    weather["hour"] = weather.datetime.dt.floor("h")
    weather = weather.drop_duplicates("hour")
    obs = complaints.merge(weather[["hour", "wind_direction"]], on="hour", how="inner")
    distance = common.distance_km(obs.latitude.to_numpy(), obs.longitude.to_numpy(), cluster.cluster_lat, cluster.cluster_lon)
    obs = obs.loc[distance <= 15].copy()
    if obs.empty:
        return None
    obs["bearing"] = common.bearing_deg(obs.latitude.to_numpy(), obs.longitude.to_numpy(), cluster.cluster_lat, cluster.cluster_lon)
    obs["actual"] = (common.angle_diff(obs.bearing, obs.wind_direction) <= 30).astype(float)
    weather["month"] = weather.datetime.dt.month
    weather["clock_hour"] = weather.datetime.dt.hour
    pools = {(m, h): g[["hour", "wind_direction"]].to_numpy() for (m, h), g in weather.groupby(["month", "clock_hour"])}
    baseline = np.full(len(obs), np.nan)
    for i, (hour, bearing) in enumerate(zip(obs.hour, obs.bearing)):
        pool = pools.get((hour.month, hour.hour))
        if pool is None:
            continue
        eligible = pool[pool[:, 0] != hour]
        if not len(eligible):
            continue
        selected = eligible[rng.integers(len(eligible), size=N_BASELINE), 1].astype(float)
        baseline[i] = np.mean(common.angle_diff(bearing, selected) <= 30)
    obs["baseline"] = baseline
    obs = obs.dropna(subset=["baseline"])
    return obs


def lift(frame):
    den = frame.baseline.mean()
    return frame.actual.mean() / den if den > 0 else np.nan


def interval(frame, rng):
    nights = [g for _, g in frame.groupby("night_date")]
    if not nights:
        return np.nan, np.nan
    a = np.array([g.actual.sum() for g in nights])
    b = np.array([g.baseline.sum() for g in nights])
    n = np.array([len(g) for g in nights])
    picks = rng.integers(len(nights), size=(N_BOOTSTRAP, len(nights)))
    ratios = (a[picks].sum(axis=1) / n[picks].sum(axis=1)) / (b[picks].sum(axis=1) / n[picks].sum(axis=1))
    ratios = ratios[np.isfinite(ratios)]
    return tuple(np.percentile(ratios, [2.5, 97.5])) if len(ratios) else (np.nan, np.nan)


def standby_location(cluster, obs):
    aligned = obs[obs.actual == 1]
    if aligned.empty:
        return cluster.cluster_lat, cluster.cluster_lon
    dx = np.mean(common.to_xy(aligned.latitude, aligned.longitude)[0] - common.to_xy(cluster.cluster_lat, cluster.cluster_lon)[0])
    dy = np.mean(common.to_xy(aligned.latitude, aligned.longitude)[1] - common.to_xy(cluster.cluster_lat, cluster.cluster_lon)[1])
    norm = np.hypot(dx, dy)
    if norm == 0:
        return cluster.cluster_lat, cluster.cluster_lon
    distance = cluster.radius + 1.5
    return cluster.cluster_lat + distance * dy / norm / common.KM_PER_DEG_LAT, cluster.cluster_lon + distance * dx / norm / common.KM_PER_DEG_LON


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    farms = pd.read_parquet(common.OUTPUT_DIR / "data" / "farms.parquet")
    groups = clusters(farms)
    complaints = common.load_livestock_complaints()
    complaints["hour"] = complaints.datetime.dt.floor("h")
    weather = pd.read_parquet(common.OUTPUT_DIR / "data" / "weather_hourly.parquet")
    points, years = [], []
    for source, station in [("ASOS", 146), ("AWS", 702)]:
        station_weather = weather[(weather.source == source) & (weather.station_id == station)].copy()
        for cluster in groups.itertuples(index=False):
            rng = np.random.default_rng(SEED)
            obs = evaluate(cluster, complaints, station_weather, rng)
            if obs is None or obs.empty:
                continue
            all_lift = lift(obs)
            low, high = interval(obs, rng)
            max_year = int(obs.datetime.dt.year.max())
            recent = obs[obs.datetime.dt.year >= max_year - 2]
            recent_lift = lift(recent) if not recent.empty else np.nan
            grade = "주" if low > 1 and recent_lift > 1 else "관찰" if all_lift > 1 else "제외"
            lat, lon = standby_location(cluster, obs)
            points.append(dict(point_id=f"{source}-{cluster.cluster_id}", cluster_id=cluster.cluster_id, grade=grade, city=cluster.city, lat=lat, lon=lon, cluster_lat=cluster.cluster_lat, cluster_lon=cluster.cluster_lon, n_farms=cluster.n_farms, eq_sum=cluster.eq_sum, lift_all=all_lift, lift_ci_low=low, lift_ci_high=high, lift_recent3=recent_lift, wind_source=source))
            for year, group in obs.groupby(obs.datetime.dt.year):
                years.append(dict(cluster_id=cluster.cluster_id, year=year, lift=lift(group), n_complaints=len(group), wind_source=source))
    p = pd.DataFrame(points)
    y = pd.DataFrame(years)
    p.to_csv(OUT / "standby_points.csv", index=False, encoding="utf-8-sig")
    y.to_csv(OUT / "lift_by_year.csv", index=False, encoding="utf-8-sig")
    lines = ["# 대기 지점의 관측 풍향 연관성", "", "확인일 2026-09-27. 관측 기상으로 계산한 과거 연관성이며 운영 중 예보 성능이 아님. 군집은 후보이고 원인 시설 판정이 아님.", "", f"민원 {len(complaints)}건. 월×시각의 다른 날짜 풍향 {N_BASELINE}개 복원추출, seed {SEED}; 밤 단위 부트스트랩 {N_BOOTSTRAP}회. 최근 3년은 해당 관측원의 마지막 연도 포함 3개 연도. 도로 스냅 생략.", "", "| 풍향 | 군집 | 농가 | 등급 | 전체 lift [95% CI] | 최근 3년 | DEVELOP 4장 |", "|---|---|---:|---|---|---:|---|"]
    for row in p.itertuples(index=False):
        reference = {"YONGJI": "용지 2.02 [1.79, 2.26], 최근 2.21", "IK-0": "왕궁 1.19, 최근 0.99", "IK-7": "춘포 1.11, 최근 0.87"}.get(row.cluster_id, "기준 없음")
        lines.append(f"| {row.wind_source} | {row.cluster_id} | {row.n_farms} | {row.grade} | {row.lift_all:.2f} [{row.lift_ci_low:.2f}, {row.lift_ci_high:.2f}] | {row.lift_recent3:.2f} | {reference} |")
    lines += ["", "## DEVELOP 4장 대비", "", "기존 산출 코드가 없어 동일 분석 정의인지는 미확인. 군집 좌표와 대기 좌표가 일치하는 IK-0을 왕궁, IK-7을 춘포로 대응함. 이 대응은 행정 경계 검증은 아님."]
    for source in ("ASOS", "AWS"):
        row = p[(p.wind_source == source) & (p.cluster_id == "YONGJI")].iloc[0]
        annual = y[(y.wind_source == source) & (y.cluster_id == "YONGJI")]
        lines.append(f"- {source} 용지: 전체 {row.lift_all:.3f} (기준 2.02, 차이 {row.lift_all-2.02:+.3f}); CI [{row.lift_ci_low:.3f}, {row.lift_ci_high:.3f}] (기준 [1.79, 2.26]); 연도별 {annual.lift.min():.3f}~{annual.lift.max():.3f} (기준 1.31~2.61); 최근 3년 {row.lift_recent3:.3f} (기준 2.21, 차이 {row.lift_recent3-2.21:+.3f}).")
        for cid, name in [("IK-0", "왕궁"), ("IK-7", "춘포")]:
            local = p[(p.wind_source == source) & (p.cluster_id == cid)].iloc[0]
            lines.append(f"- {source} {name}: 최근 3년 {local.lift_recent3:.3f}; 1 미만이면 최근 연관성이 약화한 것으로 해석함. 전체 등급은 계약 규칙상 {local.grade}.")
    lines += ["", "## 연도별 lift", "", y.to_markdown(index=False), "", "한계: 2019년 기상 없음. ASOS는 2026-07-30 종료. 풍향 관측소의 지역 대표성과 민원 신고 행동의 편향을 반영하지 못함. 김제 용지 좌표는 주소 매칭 35개 중앙값이며 정확한 필지 좌표가 아님."]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(p.to_string(index=False))


if __name__ == "__main__":
    main()
