from __future__ import annotations


def response_level(rank: int) -> tuple[str, str]:
    """점수 크기와 무관하게 권역 순위에 따라 점검 순서를 안내한다."""
    if rank == 1:
        return "우선 점검", "가장 먼저 현장 확인"
    return "순차 점검", "1순위 확인 후 순차 확인"


DISCLAIMER = (
    "본 결과는 민원 데이터에서 학습한 향후 추가 민원 가능 권역의 상대적 우선순위입니다. "
    "악취 농도, 실제 악취 발생 여부 또는 원인 시설을 확정하지 않으며 담당자 검토 후 활용해야 합니다."
)
