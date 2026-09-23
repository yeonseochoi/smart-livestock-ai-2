"""v2 설정. 기존 파이프라인(build_odor_ai_mvp.py 등)은 수정하지 않고 이 폴더에서 독립 실행한다.

유지하는 조건(기존과 동일): 익산시만, 1km 격자, 30분 입력 / 30분 예측.
바뀌는 조건: 시계 정시 Event -> 무신호 60분 뒤 첫 민원을 시작점으로 하는 세션 Event,
            단일 70/30 분할 -> rolling-origin(expanding window) 다중 폴드.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT.parent / "data" / "익산시 악취 민원 데이터_20190528-20260818.xlsx"
OUT_DIR = ROOT / "outputs"

CITY = "익산시"
GRID_M = 1000                 # 1km 격자 (기존 유지)
# 격자 원점: 전체 민원 위경도 중앙값을 "한 번 계산해 상수로 고정"했다(기존 add_grid_columns와 같은 값).
# 데이터를 시점별로 잘라도 격자가 바뀌지 않게 하려는 것이다(원점을 매번 중앙값으로 다시 구하면 미래 데이터가 과거 격자에 영향을 준다).
GRID_ORIGIN_LAT = 35.9566125
GRID_ORIGIN_LON = 126.98382
EARTH_RADIUS_M = 6_371_008.8
INPUT_MIN = 30                # 입력 창: t0 ~ t0+30분
HORIZON_MIN = 30              # 예측 창: t0+30 ~ t0+60분
CANDIDATE_RADIUS = 2          # 기존: max(1, round(1500 / grid_m)) = 2 (관측 격자 주변 반경 2칸)

# ---- Event 정의 ----
SESSION_GAP_MIN = 60          # 민원이 이 시간(분) 이상 없다가 들어온 첫 민원 = 새 세션(t0)
LOC_EPS_M = 30                # "서로 다른 위치" 판정 반경 (30m 이내 좌표는 같은 위치)
# 위치 병합은 "그 창(예: 입력 30분) 안의 신고끼리만" 30m 연결(단일 연결 군집)로 계산한다.
# 전체 기간 좌표로 한 번에 군집화하면 미래에 들어온 '중간 좌표'가 과거 두 위치를 하나로 이어 붙여
# 과거 Event의 고유 위치 수·선정 여부를 바꿀 수 있어서다.
# 예측 시점(t0+30분)에 이미 알 수 있는 정보만으로 Event를 고른다:
#   초기 30분 내 서로 다른 30m 위치 수 >= k
# 모드:
#   "session"(요청 사양) : 무신호 60분 뒤 첫 민원이 t0
#   "trigger"(주 운영)   : 최근 30분 안에 k번째 고유 위치의 신고가 들어온 순간 발동 (분 단위, 발동 후 60분간 재발동 없음)
#   "trigger_fixed"      : 이전 trigger. 첫 신고부터 30분을 채운 뒤 발동 (비교용)
PRIMARY_MODE = "trigger"       # 주 운영 Event: 최근 30분 안에 k번째 고유 위치가 들어온 순간 예측
AUXILIARY_MODE = "session"     # 보수적인 보조 검증 시나리오: 60분 무신호 뒤 시작한 세션만
DEFAULT_MODE = PRIMARY_MODE
# k 선정 규칙(성능을 보지 않는 개수 기준): "학습 Event >= MIN_TRAIN_EVENTS 인 폴드의 테스트 Event 합이
# MIN_POOLED_TEST_EVENTS 이상 남는 가장 큰 k". 근거표: outputs/k_sweep.csv (sweep_k.py)
# (이전 규칙은 '테스트 15개 이상 폴드 3개'였고 작은 반기를 합쳐야 성립했다. 폴드를 합치지 않기로 해서 규칙도 바꿨다.)
K_MIN_UNIQUE_LOCATIONS = {"session": 4, "trigger": 7, "trigger_fixed": 6}
# trigger=7: 새 trigger 정의(최근 30분 내 k번째 위치)로 sweep_k.py 를 다시 돌려 위 규칙이 고른 값(테스트 합 102). trigger_fixed=6 은 이전 정의의 규칙 결과.
MODE_ROLE = {"trigger": "주 운영 Event", "session": "보수적 보조 검증", "trigger_fixed": "이전 trigger(비교용)", "legacy_hour": "참고용(기존 정시 Event)"}
MIN_POOLED_TEST_EVENTS = 100

# ---- rolling-origin ----
RELIABLE_MIN_TEST_EVENTS = 15 # 폴드 테스트 Event가 이보다 적으면 '표본 작음' 표시(합치지 않고 그대로 보고, 구간을 넓게 본다)
MIN_TRAIN_EVENTS = 40         # 학습 최소 Event 수 (미달 폴드는 제외)
FOLD_BOUNDS = [               # 테스트 기간 경계: 2021, 2022는 연 단위, 2023년 이후는 반기 단위 (기간을 합치지 않는다)
    "2021-01-01", "2022-01-01", "2023-01-01", "2023-07-01", "2024-01-01",
    "2024-07-01", "2025-01-01", "2025-07-01", "2026-01-01", "2026-07-01",
]

# ---- 모델/평가 ----
TOPK = 3
SEEDS = (42, 43, 44)
XGB_PARAMS = dict(            # 기존 xgb_d3 최종 설정과 동일
    n_estimators=650, max_depth=3, min_child_weight=7, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.15, reg_lambda=2.0,
    objective="binary:logistic", eval_metric="aucpr", n_jobs=4,
)
N_BOOTSTRAP = 2000

# 기존 README 수치 (단일 70/30 분할, 테스트 48 Event 중 평가 가능 47개, 1km)
LEGACY_README = {
    "ROC-AUC": 0.908, "PR-AUC": 0.486, "Recall@3": 0.443,
    "Hit@3": 0.851, "평가 Event 수": 47,
}
