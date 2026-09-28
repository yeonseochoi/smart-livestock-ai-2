"""C 현장 추적: 담당자 위치 기준 방문 후보 점수 카드.

점수(100점) = 풍향 일치 40 + 거리 25 + 다중 측정 일치 20 + 과거 반복 15 (DEVELOP.md 6장 C).
점수는 현장 확인 순서를 정하기 위한 참고 순위이며 발생원 확률이 아니다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

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
    tie = f["eq"].fillna(-1) if "eq" in f else 0
    f = f.assign(_tie=tie).sort_values(["score", "_tie"], ascending=False).drop(columns="_tie")
    # 순위 후보를 먼저 고른 뒤 같은 좌표는 하나의 실제 방문 지점으로 합친다.
    # 원본 허가 행은 보존하고 화면·방문 순서에서만 중복을 제거한다.
    # 중복 제거 전에 Top K를 자르면 같은 지점이 순위를 차지해 다른 후보가 누락된다.
    shortlisted = f.copy()
    if not shortlisted.empty:
        shortlisted["_location_lat"] = shortlisted["lat"].round(7)
        shortlisted["_location_lon"] = shortlisted["lon"].round(7)
        location_key = [shortlisted["_location_lat"], shortlisted["_location_lon"]]
        groups = shortlisted.groupby(location_key, sort=False, dropna=False)
        shortlisted["merged_count"] = groups["farm_id"].transform("size")
        shortlisted["merged_farm_ids"] = groups["farm_id"].transform(lambda s: " | ".join(map(str, s)))
        shortlisted["merged_names"] = groups["name"].transform(lambda s: " | ".join(dict.fromkeys(map(str, s))))
        if "address" in shortlisted:
            shortlisted["merged_addresses"] = groups["address"].transform(
                lambda s: " | ".join(dict.fromkeys(str(value).strip() for value in s.dropna() if str(value).strip()))
            )
        # 동일 좌표의 여러 농가는 한 지점으로 묶고, 이름만 반복되는 등록 행도 한 번만 제시한다.
        shortlisted = shortlisted.drop_duplicates(["_location_lat", "_location_lon"], keep="first")
        shortlisted = shortlisted.drop_duplicates(["name"], keep="first")
        shortlisted = shortlisted.head(top_k).drop(columns=["_location_lat", "_location_lon"]).reset_index(drop=True)
        shortlisted["rank"] = np.arange(1, len(shortlisted) + 1)
    return TrackResult(wind.mode, confidence, shortlisted.reset_index(drop=True), notes)


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


def reason(row: pd.Series) -> str:
    """기존 JSON 소비자를 위한 한 줄 설명."""
    summary, evidence, _ = explanation(row)
    return f"{summary} {' '.join(evidence)}"


def to_cards(result: TrackResult) -> list[dict]:
    cards = []
    for i, r in result.candidates.iterrows():
        summary, evidence, next_action = explanation(r)
        cards.append({
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
            "distance_km": round(float(r["dist_km"]), 2),
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
        })
    return cards
