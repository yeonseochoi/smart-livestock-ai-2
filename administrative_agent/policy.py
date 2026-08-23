from __future__ import annotations


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
