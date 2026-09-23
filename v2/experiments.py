"""평가할 실험(모델) 목록. 새 변수 실험은 여기에 한 줄만 추가하면 2단계 평가표에 한 줄로 들어간다.

각 항목: (이름, 점수 생성 방식)
  None           : 전부 동점 -> 무작위 선택의 기대값
  "컬럼명"        : 그 컬럼 값을 그대로 점수로 사용 (기준선)
  [변수 목록]     : 해당 변수로 XGBoost 학습 (폴드마다 학습 구간만으로 재학습)

4단계에서 변수 그룹을 추가할 때는 features.py 에 컬럼을 만든 뒤 아래에 예를 들어
    ("XGB 15변수+이력+기상", BASE_FEATURES + HIST_FEATURES + WEATHER_FEATURES),
처럼 한 줄을 넣는다. 나머지(폴드, prior 재계산, 기준선 대비 개선폭·신뢰구간, 중복 제거, 핫스팟 내 순위)는 그대로 적용된다.
"""
from __future__ import annotations

from features import BASE_FEATURES, HIST_FEATURES

REFERENCE = "기준선 prior만"          # 모든 개선폭은 이 기준선 대비로 계산한다

EXPERIMENTS = [
    ("기준선 무작위", None),
    ("기준선 초기민원수 상위3", "initial_count"),
    (REFERENCE, "prior"),
    ("XGB 기존15변수(LOO prior)", BASE_FEATURES),
    ("XGB 15변수+이력", BASE_FEATURES + HIST_FEATURES),
]
