from __future__ import annotations

import pandas as pd

NARROW_RANK_LIMIT = 30  # 확산 예측 후보를 발생 위험 예보 순위 몇 위까지로 좁히나
MIN_CANDIDATES = 3      # Top 3를 만들 최소 후보 수


def narrow_candidates(frame: pd.DataFrame, k: int = NARROW_RANK_LIMIT) -> tuple[pd.DataFrame, dict]:
    """확산 예측(2단) 후보를 발생 위험 예보(1단) 순위 k 이내 격자로 좁힌다.

    frame 열: score(2단 점수), onset_rank(기준시각 1시간 전 1단 순위. 1단 격자 밖이면 NaN).
    규칙: 기존 2단 Top 3는 항상 보존한다. 그 밖에는 순위 k 이내를 남기고,
          1단 격자 밖 후보(민원 이력 없는 칸)는 2단 점수가 남은 후보 1위보다 높을 때만 남긴다.
    검증(테스트 Event 47개): 후보 평균 31.1 → 15.1, Hit@1/2/3 변화 없음. outputs/onset_spread_fusion/narrow_table.md
    """
    ordered = frame.sort_values("score", ascending=False)
    spread_top = ordered.head(MIN_CANDIDATES)
    inside = ordered[ordered["onset_rank"] <= k]
    outside = ordered[ordered["onset_rank"].isna()]
    if len(inside) and len(outside):
        outside = outside[outside["score"] > inside["score"].iloc[0]]
    filtered = pd.concat([inside, outside])
    protected = spread_top.drop(filtered.index, errors="ignore")
    kept = pd.concat([filtered, protected]).sort_values("score", ascending=False)
    info = {"rank_limit": k, "candidates": int(len(frame)), "kept": int(len(kept)),
            "filled": 0, "protected_top3": int(len(protected)), "escaped_outside": int(len(outside))}
    return kept, info


def response_level(relative_risk: int) -> tuple[str, str]:
    """상대 위험도에 따른 설명 가능한 대응 문구를 반환한다."""
    if relative_risk >= 80:
        return "우선 점검", "가용 인력 범위에서 가장 먼저 현장 확인"
    if relative_risk >= 60:
        return "순차 점검", "1순위 확인 후 순차적으로 현장 확인"
    return "상황 관찰", "추가 민원 유입을 관찰하고 필요 시 점검"


DISCLAIMER = (
    "본 결과는 민원 데이터에서 학습한 향후 추가 민원 가능 권역의 상대적 우선순위입니다. "
    "악취 농도, 실제 악취 발생 여부 또는 원인 시설을 확정하지 않으며 담당자 검토 후 활용해야 합니다."
)
