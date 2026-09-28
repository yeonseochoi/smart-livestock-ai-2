"""C 현장 추적: 방문 후보 점수 카드.

운영 기본은 `score_complaint_candidates`(민원 지점 기준)다.
- 후보: 30m로 묶은 고유 민원 위치마다 풍상 ±45°·4 km 안의 농가를 찾아 합친다.
- 2곳 이상의 민원 위치가 함께 가리키는 농가(교차 확인)를 우선하고, 그런 농가가 없을 때만
  한 위치만 가리키는 농가를 '단일 지점 참고'로 제시한다. 후보 수를 5개로 억지로 채우지 않는다.
- 점수(100점) = 풍향 일치 40 + 민원 근접 25 + 민원 중첩 20 + 과거 반복 15.
  담당자 이동 거리는 관련성 점수에서 분리해 방문 동선 제안에만 쓴다.

`score_candidates`(담당자 위치 기준)는 이전 방식 비교·테스트용으로 남긴다.
점수는 현장 확인 순서를 정하기 위한 참고 순위이며 발생원 확률이 아니다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

import numpy as np
import pandas as pd

from odor_service import common as c

W_WIND, W_DIST, W_MULTI, W_HIST = 40.0, 25.0, 20.0, 15.0
MAX_RADIUS_KM = 4.0
DIST_SCALE_KM = 2.0
MULTI_TOL_DEG = 45.0
HIST_RADIUS_KM = 15.0
HIST_WIND_TOL_DEG = 30.0
DISCLAIMER = "발생원 확정 또는 위반 판정이 아닌 현장 확인 참고자료"

# 민원 지점 기준 후보 선정(A 절충안) 설정
LOCATION_MERGE_KM = 0.03  # v2 사건 선정과 같은 30m 고유 위치
SUPPORT_TOL_DEG = 45.0  # 민원 위치에서 본 농가 방위가 풍향(불어오는 방향)과 이 각도 안이면 지지
CROSS_SUPPORT_MIN = 2  # 교차 확인 후보가 되기 위한 최소 지지 위치 수
CLUSTER_LINK_KM = 2.0  # 이 거리 안으로 이어지는 민원 위치는 같은 민원 무리
SENSITIVITY_DEG = 20.0  # 풍향 오차 가정(민감도 점검용, 관측 오차 추정값 아님)
TIER_CROSS = "교차 확인"
TIER_SINGLE = "단일 지점 참고"


@dataclass
class Wind:
    direction: float  # 불어오는 방향(°)
    speed: float  # m/s
    sigma: float = 20.0  # 풍향 표준편차(°)

    @property
    def mode(self) -> str:
        if self.speed >= 1.0:
            return "sector"
        if self.speed >= 0.5:
            return "mixed"
        return "gradient"


@dataclass
class Observation:
    """냄새 확인 측정 지점. smell=True인 지점만 다중 측정 일치에 쓴다."""
    lat: float
    lon: float
    smell: bool = True
    value: float | None = None  # (선택) H2S·NH3 등 측정값


@dataclass
class Weights:
    wind: float = W_WIND
    dist: float = W_DIST
    multi: float = W_MULTI
    hist: float = W_HIST
    scale: float = 0.0


@dataclass
class TrackResult:
    mode: str
    confidence: str
    candidates: pd.DataFrame
    notes: list[str] = field(default_factory=list)
    # 민원 지점 기준 선정에서만 채운다.
    selection: str | None = None  # "교차 확인" / "단일 지점 참고" / None
    locations: int = 0
    clusters: int = 0
    stability: float | None = None
    stability_label: str | None = None
    route_km: float | None = None


@dataclass
class ComplaintLocation:
    """30m 안의 신고를 묶은 고유 민원 위치(실제 신고 좌표인 medoid)."""
    lat: float
    lon: float
    n_reports: int = 1
    cluster: int = 0


def window_deg(wind: Wind) -> float:
    return max(90.0, 1.5 * wind.sigma)


def history_counts(farms: pd.DataFrame, complaints: pd.DataFrame, wind: Wind, month: int) -> np.ndarray:
    """비슷한 풍향(±30°)·같은 월에 각 농가 풍하 15 km 안에서 난 과거 가축 민원 수.

    complaints는 datetime, latitude, longitude, wind_direction, wind_speed를 가져야 한다.
    """
    h = complaints[
        (complaints["datetime"].dt.month == month)
        & (complaints["wind_speed"] >= 0.5)
        & (c.angle_diff(complaints["wind_direction"], wind.direction) <= HIST_WIND_TOL_DEG)
    ]
    if h.empty or farms.empty:
        return np.zeros(len(farms))
    flat, flon = farms["lat"].to_numpy()[:, None], farms["lon"].to_numpy()[:, None]
    clat, clon = h["latitude"].to_numpy()[None, :], h["longitude"].to_numpy()[None, :]
    d = c.distance_km(flat, flon, clat, clon)
    toward = c.bearing_deg(flat, flon, clat, clon)  # 농가→민원 방위
    downwind = (h["wind_direction"].to_numpy()[None, :] + 180.0) % 360.0
    hit = (d <= HIST_RADIUS_KM) & (c.angle_diff(toward, downwind) <= HIST_WIND_TOL_DEG)
    return hit.sum(axis=1).astype(float)


def select_visit_points(scored: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    """점수 순으로 좌표 중복 또는 같은 이름의 300m 이내 등록만 합친다.

    대표점과의 거리를 비교하여 연쇄 병합으로 멀리 떨어진 동명 시설이 사라지지 않게 한다.
    원본 등록 ID·이름·주소는 합쳐진 카드의 메타데이터에 보존한다.
    """
    if scored.empty:
        return scored.copy().reset_index(drop=True)
    tie = scored["eq"].fillna(-1) if "eq" in scored else 0
    ordered = scored.assign(_tie=tie).sort_values(["score", "_tie"], ascending=False, kind="stable").drop(columns="_tie")
    groups: list[list[pd.Series]] = []
    for _, row in ordered.iterrows():
        name = str(row["name"]).strip() if pd.notna(row.get("name")) else ""
        match = None
        for group in groups:
            representative = group[0]
            same_coord = (round(row["lat"], 7), round(row["lon"], 7)) == (round(representative["lat"], 7), round(representative["lon"], 7))
            same_name = name and any(pd.notna(r.get("name")) and str(r["name"]).strip() == name for r in group)
            near = float(c.distance_km(row["lat"], row["lon"], representative["lat"], representative["lon"])) <= 0.3
            if same_coord or (same_name and near):
                match = group
                break
        if match is None:
            groups.append([row])
        else:
            match.append(row)
    representatives = []
    for group in groups[:top_k]:
        row = group[0].copy()
        row["merged_count"] = len(group)
        for source, target in (("farm_id", "merged_farm_ids"), ("name", "merged_names"), ("address", "merged_addresses")):
            row[target] = " | ".join(dict.fromkeys(str(r[source]).strip() for r in group
                                                 if source in r and pd.notna(r[source]) and str(r[source]).strip()))
        representatives.append(row)
    result = pd.DataFrame(representatives).reset_index(drop=True)
    result["rank"] = np.arange(1, len(result) + 1)
    return result


def score_candidates(
    farms: pd.DataFrame,
    staff_lat: float,
    staff_lon: float,
    wind: Wind,
    observations: list[Observation] | None = None,
    hist: np.ndarray | None = None,
    weights: Weights = Weights(),
    top_k: int = 5,
) -> TrackResult:
    """담당자 위치에서 풍상 방향 농가를 점수화해 Top K를 돌려준다."""
    notes: list[str] = []
    f = farms.dropna(subset=["lat", "lon"]).copy()
    if "status" in f:
        f = f[f["status"].isna() | (f["status"] == "정상")]
    if hist is not None:
        f["hist_n"] = pd.Series(hist, index=farms.index).reindex(f.index).to_numpy()
    else:
        f["hist_n"] = 0.0

    f["dist_km"] = c.distance_km(staff_lat, staff_lon, f["lat"], f["lon"])
    f["bearing"] = c.bearing_deg(staff_lat, staff_lon, f["lat"], f["lon"])
    f["dtheta"] = c.angle_diff(f["bearing"], wind.direction)
    f = f[(f["dist_km"] <= MAX_RADIUS_KM) & (f["dtheta"] <= window_deg(wind))].copy()

    confidence = {"sector": "보통", "mixed": "낮음", "gradient": "매우 낮음"}[wind.mode]
    if wind.mode == "gradient":
        notes.append("풍속 0.5 m/s 미만: 풍향 신뢰도가 낮아 측정값 경사로 방향을 보정해야 함")

    f["s_wind"] = weights.wind * np.clip(1.0 - f["dtheta"] / 90.0, 0.0, 1.0)
    f["s_dist"] = weights.dist * np.exp(-f["dist_km"] / DIST_SCALE_KM)

    smell_pts = [o for o in (observations or []) if o.smell]
    if len(smell_pts) >= 2:
        k = np.zeros(len(f))
        for o in smell_pts:
            b = c.bearing_deg(o.lat, o.lon, f["lat"], f["lon"])
            d = c.distance_km(o.lat, o.lon, f["lat"], f["lon"])
            k += np.asarray((c.angle_diff(b, wind.direction) <= MULTI_TOL_DEG) & (d <= MAX_RADIUS_KM))
        f["multi_hit"] = k
        f["multi_n"] = len(smell_pts)
        f["s_multi"] = weights.multi * k / len(smell_pts)
    else:
        f["multi_hit"] = np.nan
        f["multi_n"] = len(smell_pts)
        f["s_multi"] = weights.multi * np.clip(1.0 - f["dtheta"] / 90.0, 0.0, 1.0)
        notes.append("냄새 확인 측정 지점이 2곳 미만: 다중 측정 일치는 풍향 일치 비율로 대체")

    lh = np.log1p(f["hist_n"].astype(float))
    f["s_hist"] = weights.hist * (lh / lh.max() if len(lh) and lh.max() > 0 else 0.0)

    eq = pd.to_numeric(f["eq"], errors="coerce") if "eq" in f else pd.Series(np.nan, index=f.index)
    eq = eq.fillna(eq.dropna().median() if eq.notna().any() else 0.0)
    log_eq = np.log1p(eq)
    f["s_scale"] = weights.scale * (log_eq / log_eq.max() if len(log_eq) and log_eq.max() > 0 else 0.0)

    f["score"] = f[["s_wind", "s_dist", "s_multi", "s_hist", "s_scale"]].sum(axis=1)
    shortlisted = select_visit_points(f, top_k)
    return TrackResult(wind.mode, confidence, shortlisted.reset_index(drop=True), notes)


def complaint_locations(points) -> list[ComplaintLocation]:
    """신고 좌표를 30m 고유 위치로 묶고, 2 km 안으로 이어지는 위치를 같은 민원 무리로 표시한다.

    같은 장소의 반복 신고가 여러 표를 갖지 않도록 위치당 1표만 준다.
    대표점은 평균이 아니라 실제 신고 좌표인 medoid다.
    """
    pts = np.asarray([[float(p[0]), float(p[1])] for p in points], dtype=float).reshape(-1, 2)
    if not len(pts):
        return []
    d = c.distance_km(pts[:, 0, None], pts[:, 1, None], pts[None, :, 0], pts[None, :, 1])
    _, labels = connected_components(csr_matrix(d <= LOCATION_MERGE_KM), directed=False)
    reps, sizes = [], []
    for label in np.unique(labels):
        idx = np.flatnonzero(labels == label)
        reps.append(pts[idx[np.argmin(d[np.ix_(idx, idx)].sum(axis=1))]])
        sizes.append(len(idx))
    reps = np.asarray(reps)
    dd = c.distance_km(reps[:, 0, None], reps[:, 1, None], reps[None, :, 0], reps[None, :, 1])
    _, clusters = connected_components(csr_matrix(dd <= CLUSTER_LINK_KM), directed=False)
    return [ComplaintLocation(float(lat), float(lon), int(n), int(k))
            for (lat, lon), n, k in zip(reps, sizes, clusters)]


def _support(farms: pd.DataFrame, locations: list[ComplaintLocation], direction: float):
    """(위치 × 농가) 거리·방위 차이·지지 여부. 지지 = 4 km 안 + 민원 위치에서 본 농가가 풍상 ±45°."""
    loc = np.asarray([[x.lat, x.lon] for x in locations], dtype=float)
    lat, lon = farms["lat"].to_numpy(dtype=float)[None, :], farms["lon"].to_numpy(dtype=float)[None, :]
    dist = c.distance_km(loc[:, 0, None], loc[:, 1, None], lat, lon)
    ang = c.angle_diff(c.bearing_deg(loc[:, 0, None], loc[:, 1, None], lat, lon), direction)
    return dist, ang, (dist <= MAX_RADIUS_KM) & (ang <= SUPPORT_TOL_DEG)


def _complaint_scored(farms: pd.DataFrame, locations: list[ComplaintLocation], wind: Wind,
                      past: pd.DataFrame | None, month: int | None) -> tuple[pd.DataFrame, str | None]:
    """교차 확인 후보가 있으면 그것만, 없으면 단일 지점 참고 후보를 점수화한다."""
    f = farms.dropna(subset=["lat", "lon"]).copy()
    if "status" in f:
        f = f[f["status"].isna() | (f["status"] == "정상")]
    if f.empty or not locations:
        return f.iloc[0:0], None
    dist, ang, hits = _support(f, locations, wind.direction)
    counts = hits.sum(axis=0)
    cross = counts >= CROSS_SUPPORT_MIN
    keep = cross if cross.any() else counts >= 1
    if not keep.any():
        return f.iloc[0:0], None
    selection = TIER_CROSS if cross.any() else TIER_SINGLE
    f = f.loc[keep].copy()
    h, a, dk, n = hits[:, keep], ang[:, keep], dist[:, keep], counts[keep]
    f["support_n"] = n
    f["support_total"] = len(locations)
    f["tier"] = np.where(n >= CROSS_SUPPORT_MIN, TIER_CROSS, TIER_SINGLE)
    f["dtheta"] = (a * h).sum(axis=0) / n  # 지지하는 민원 위치들의 평균 방위 차이
    f["s_wind"] = W_WIND * (np.clip(1.0 - a / 90.0, 0.0, 1.0) * h).sum(axis=0) / n
    nearest = np.where(h, dk, np.inf)
    f["complaint_km"] = nearest.min(axis=0)
    f["s_dist"] = W_DIST * np.exp(-f["complaint_km"] / DIST_SCALE_KM)
    f["multi_hit"] = n
    f["multi_n"] = len(locations)
    f["s_multi"] = W_MULTI * n / len(locations)
    cluster_of = np.asarray([x.cluster for x in locations])
    f["cluster"] = cluster_of[nearest.argmin(axis=0)]
    if past is not None and month is not None and not past.empty:
        f["hist_n"] = history_counts(f, past, wind, month)
    else:
        f["hist_n"] = 0.0
    lh = np.log1p(f["hist_n"].astype(float))
    f["s_hist"] = W_HIST * (lh / lh.max() if len(lh) and lh.max() > 0 else 0.0)
    f["s_scale"] = 0.0
    f["score"] = f[["s_wind", "s_dist", "s_multi", "s_hist"]].sum(axis=1)
    return f, selection


def _pick_with_cluster_coverage(scored: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """동일 지점을 합친 뒤, 민원 무리마다 최고 후보 1곳을 먼저 확보하고 나머지를 점수순으로 채운다.

    민원이 멀리 떨어진 두 무리로 나뉘면 큰 무리의 후보만 5곳을 차지해 작은 무리가 빠지는 것을 막는다.
    """
    merged = select_visit_points(scored, top_k=len(scored))
    if merged.empty:
        return merged
    chosen: list[int] = []
    for _, group in merged.groupby("cluster", sort=False):
        chosen.append(int(group.index[0]))  # merged는 이미 점수순
    chosen = sorted(chosen, key=lambda i: -float(merged.loc[i, "score"]))[:top_k]
    for i in merged.index:
        if len(chosen) >= top_k:
            break
        if i not in chosen:
            chosen.append(int(i))
    result = merged.loc[chosen].sort_values("score", ascending=False, kind="stable").reset_index(drop=True)
    result["rank"] = np.arange(1, len(result) + 1)
    return result


def _visit_route(candidates: pd.DataFrame, start: tuple[float, float] | None) -> float | None:
    """출발점에서 가까운 후보부터 도는 직선 거리 기준 방문 동선(최근접 이웃). 관련성 순위와 별개다."""
    if start is None or candidates.empty:
        return None
    lat, lon = candidates["lat"].to_numpy(dtype=float), candidates["lon"].to_numpy(dtype=float)
    candidates["travel_km"] = c.distance_km(start[0], start[1], lat, lon)
    order, legs, here, left = [], [], start, list(range(len(candidates)))
    while left:
        d = [float(c.distance_km(here[0], here[1], lat[i], lon[i])) for i in left]
        j = left.pop(int(np.argmin(d)))
        order.append(j)
        legs.append(min(d))
        here = (lat[j], lon[j])
    visit = np.empty(len(candidates), dtype=int)
    leg = np.empty(len(candidates))
    for step, (j, km) in enumerate(zip(order, legs), 1):
        visit[j], leg[j] = step, km
    candidates["visit_order"] = visit
    candidates["leg_km"] = leg
    return float(sum(legs))


def _stability_label(value: float | None) -> str | None:
    if value is None:
        return None
    return "높음" if value >= 0.8 else "보통" if value >= 0.5 else "낮음"


def score_complaint_candidates(
    farms: pd.DataFrame,
    locations: list[ComplaintLocation],
    wind: Wind,
    past_complaints: pd.DataFrame | None = None,
    month: int | None = None,
    start: tuple[float, float] | None = None,
    top_k: int = 5,
    sensitivity: bool = True,
) -> TrackResult:
    """민원 지점 기준 방문 후보(A 절충안).

    - 후보 자격: 고유 민원 위치에서 4 km 안·풍상 ±45°. 2곳 이상 지지 후보가 있으면 그것만 쓴다.
    - 관련성 점수와 이동 거리를 분리한다. `start`(대기 장소 등)는 방문 동선 제안에만 쓴다.
    - 풍향 ±20° 민감도로 후보 안정성을 함께 돌려준다.
    """
    notes: list[str] = []
    confidence = {"sector": "보통", "mixed": "낮음", "gradient": "매우 낮음"}[wind.mode]
    if wind.mode == "gradient":
        notes.append("풍속 0.5 m/s 미만: 관측 풍향을 신뢰하기 어려워 후보 순서는 참고용이며 현장 풍향 확인이 우선")
    elif wind.mode == "mixed":
        notes.append("풍속 1 m/s 미만: 풍향이 바뀌기 쉬워 현장 풍향 확인 후 후보를 좁힐 것")
    scored, selection = _complaint_scored(farms, locations, wind, past_complaints, month)
    candidates = _pick_with_cluster_coverage(scored, top_k) if selection else scored.iloc[0:0].copy()
    route_km = _visit_route(candidates, start)
    n_clusters = len({x.cluster for x in locations})
    if selection == TIER_SINGLE:
        notes.append("2곳 이상의 민원 위치가 함께 가리키는 농가가 없어 한 위치 기준 참고 후보만 제시")
    if n_clusters > 1:
        notes.append(f"민원이 {n_clusters}개 무리로 떨어져 있어 무리마다 대표 후보를 먼저 포함")
    stability = None
    if sensitivity and not candidates.empty:
        base = set(candidates["farm_id"])
        kept = []
        for delta in (-SENSITIVITY_DEG, SENSITIVITY_DEG):
            shifted = Wind((wind.direction + delta) % 360.0, wind.speed, wind.sigma)
            other, sel = _complaint_scored(farms, locations, shifted, past_complaints, month)
            ids = set(_pick_with_cluster_coverage(other, top_k)["farm_id"]) if sel else set()
            kept.append(len(base & ids) / len(base))
        stability = float(np.mean(kept))
        if stability < 0.5:
            notes.append("풍향이 ±20° 달라지면 후보 절반 이상이 바뀜: 현장 풍향으로 후보를 다시 확인")
    return TrackResult(wind.mode, confidence, candidates.reset_index(drop=True), notes,
                       selection=selection, locations=len(locations), clusters=n_clusters,
                       stability=stability, stability_label=_stability_label(stability), route_km=route_km)


def complaint_explanation(row: pd.Series) -> tuple[str, list[str], str]:
    """민원 지점 기준 후보 설명."""
    n, total = int(row["support_n"]), int(row["support_total"])
    if row.get("tier") == TIER_SINGLE:
        summary = "한 민원 위치에서만 바람이 불어오는 쪽에 있어 참고로 확인할 후보입니다."
    elif n == total:
        summary = "모든 민원 위치에서 바람이 불어오는 쪽이 이 농가로 겹쳐 우선 확인할 후보입니다."
    else:
        summary = f"민원 위치 {n}곳이 함께 이 농가 쪽을 바람 상류로 가리켜 우선 확인할 후보입니다."
    evidence = [
        f"민원 위치 {total}곳 중 {n}곳에서 이 농가가 바람이 불어오는 쪽(±45°, 4 km 안)에 있습니다.",
        f"그 위치들에서 본 농가 방향과 풍향의 차이는 평균 {float(row['dtheta']):.0f}°입니다.",
        f"가장 가까운 민원 위치에서 {float(row['complaint_km']):.1f} km 떨어져 있습니다.",
    ]
    history = int(row.get("hist_n", 0))
    if history > 0:
        evidence.append(f"같은 달·비슷한 풍향에서 이 농가의 바람 아랫방향 민원이 과거 {history}건 있었습니다. 인과관계가 아닌 참고 이력입니다.")
    else:
        evidence.append("같은 달·비슷한 풍향의 과거 반복 이력은 확인되지 않았습니다.")
    action = "농가 경계에서 실제 풍향과 냄새를 먼저 확인하고, 두 조건이 일치할 때 시료를 채취하세요."
    if row.get("tier") == TIER_SINGLE:
        action = "근거가 한 민원 위치뿐이므로 " + action
    if row.get("coord_precision") not in (None, "exact"):
        action = "좌표가 근사값이므로 출발 전에 정확한 주소를 확인한 뒤, " + action
    return summary, evidence, action


def explanation(row: pd.Series) -> tuple[str, list[str], str]:
    """담당자가 순위의 의미와 다음 행동을 바로 이해할 수 있는 설명을 만든다."""
    angle = float(row["dtheta"])
    distance = float(row["dist_km"])
    if angle <= 20 and distance <= 2:
        summary = "현재 바람이 불어오는 방향에 정확히 놓인 가까운 농가라 먼저 확인합니다."
    elif angle <= 45:
        summary = "현재 바람을 거슬러 올라가는 경로에 있어 우선 확인할 필요가 있습니다."
    else:
        summary = "바람 방향과 완전히 일치하지는 않지만, 거리와 과거 조건을 함께 보면 확인 가치가 있습니다."

    evidence = [
        f"바람이 불어오는 중심 방향에서 {angle:.0f}° 차이입니다.",
        f"현재 담당자 위치에서 {distance:.1f} km 떨어져 있습니다.",
    ]
    if not pd.isna(row.get("multi_hit")):
        hit, total = int(row["multi_hit"]), int(row["multi_n"])
        if hit > 0:
            evidence.append(f"민원 지점 {total}곳 중 {hit}곳에서 이 농가 쪽을 바람의 상류 방향으로 가리킵니다.")
    else:
        evidence.append("확인된 민원 지점이 2곳 미만이라 여러 지점의 방향 교차 확인은 아직 하지 못했습니다.")
    history = int(row.get("hist_n", 0))
    if history > 0:
        evidence.append(f"같은 달·비슷한 풍향에서 이 농가의 바람 아랫방향 민원이 과거 {history}건 있었습니다. 인과관계가 아닌 참고 이력입니다.")
    else:
        evidence.append("같은 달·비슷한 풍향의 과거 반복 이력은 확인되지 않았습니다.")

    action = "농가 경계에서 실제 풍향과 냄새를 먼저 확인하고, 두 조건이 일치할 때 시료를 채취하세요."
    if row.get("coord_precision") not in (None, "exact"):
        action = "좌표가 근사값이므로 출발 전에 정확한 주소를 확인한 뒤, " + action
    return summary, evidence, action


def _explain(row: pd.Series) -> tuple[str, list[str], str]:
    return complaint_explanation(row) if "support_n" in row and not pd.isna(row.get("support_n")) else explanation(row)


def reason(row: pd.Series) -> str:
    """기존 JSON 소비자를 위한 한 줄 설명."""
    summary, evidence, _ = _explain(row)
    return f"{summary} {' '.join(evidence)}"


def _staff_km(row: pd.Series) -> float:
    for key in ("dist_km", "travel_km"):
        if key in row and not pd.isna(row.get(key)):
            return float(row[key])
    return float("nan")


def to_cards(result: TrackResult) -> list[dict]:
    cards = []
    for i, r in result.candidates.iterrows():
        summary, evidence, next_action = _explain(r)
        card = {
            "rank": i + 1,
            "farm_id": r.get("farm_id"),
            "name": r.get("name"),
            "score": round(float(r["score"]), 1),
            "components": {
                "풍향 일치": round(float(r["s_wind"]), 1),
                "거리": round(float(r["s_dist"]), 1),
                "다중 측정 일치": round(float(r["s_multi"]), 1),
                "과거 반복": round(float(r["s_hist"]), 1),
                "규모": round(float(r["s_scale"]), 1),
            },
            # 담당자(대기 장소) 기준 직선 거리. 민원 지점 기준 선정에서는 방문 동선용 travel_km와 같다.
            "distance_km": None if pd.isna(_staff_km(r)) else round(float(_staff_km(r)), 2),
            "lat": float(r["lat"]),
            "lon": float(r["lon"]),
            "reference": {"species": r.get("species"), "eq": None if pd.isna(r.get("eq")) else float(r.get("eq"))},
            "coord_warning": r.get("coord_precision") not in (None, "exact"),
            "merged_count": int(r.get("merged_count", 1)),
            "merged_farm_ids": r.get("merged_farm_ids", r.get("farm_id")),
            "merged_names": r.get("merged_names", r.get("name")),
            "address": r.get("address"),
            "merged_addresses": r.get("merged_addresses", r.get("address")),
            "reason": reason(r),
            "selection_summary": summary,
            "evidence": evidence,
            "next_action": next_action,
            "mode": result.mode,
            "confidence": result.confidence,
            "disclaimer": DISCLAIMER,
        }
        if "support_n" in r and not pd.isna(r.get("support_n")):
            card.update({
                "tier": r["tier"],
                "support": {"count": int(r["support_n"]), "total": int(r["support_total"])},
                "complaint_km": round(float(r["complaint_km"]), 2),
                "cluster": int(r["cluster"]),
            })
            if "travel_km" in r and not pd.isna(r.get("travel_km")):
                card.update({"travel_km": round(float(r["travel_km"]), 2),
                             "visit_order": int(r["visit_order"]), "leg_km": round(float(r["leg_km"]), 2)})
        cards.append(card)
    return cards
