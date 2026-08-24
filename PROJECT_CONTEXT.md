# 익산시 악취 민원 선제 대응 프로젝트

## 프로젝트 목적

익산악취24가 시민 신고와 현재 관측 상황을 수집·조회하는 시스템이라면, 이 프로젝트는 현재까지 접수된 민원의 시공간 패턴을 바탕으로 향후 30분 안에 추가 민원이 발생할 가능성이 높은 1km 권역 3곳을 우선순위로 제시한다.

예측 결과는 악취의 물리적 발생 확률이나 배출원 판정이 아니다. 현장 담당자가 악취측정 차량, 순찰, 현장 측정 및 점검 순서를 정할 때 사용하는 상대적 민원 위험 순위다.

## 제품 구성

### 1. 민원 위험 예측 엔진

- 입력: 현재 Event의 초기 30분 민원 위치·시각·강도와 과거 민원 패턴
- 출력: 이후 30분 추가 민원 위험이 높은 1km 권역 Top 3
- 운영 단위: 1km 격자
- 표현: 확률이 아닌 상대 위험 점수와 순위
- 신규 데이터 검증 결과: PR-AUC 0.486, Top-K Recall 0.507, Recall@3 0.443, Event Hit@3 0.851
- 선택 모델: `xgb_d3`. 학습 구간 내부의 마지막 20% Event로만 선택했고 최종 테스트는 1회 사용했다.

기상정보는 현재 최종 예측 점수에 사용하지 않는다. 동일한 시간 분할 실험에서 개선이 작고 일관되지 않았기 때문이다. 다만 풍향·풍속·강수·습도는 담당자의 현장 판단을 위한 상황 정보로 제공할 수 있다.

### 2. 행정 대응 Agent

예측 결과와 확인 가능한 현재 상황을 이용해 다음 문서의 초안을 생성한다.

- 악취 민원 상황 브리핑
- 현장 점검 지시서
- 사후 결과보고서

Agent는 시설을 원인으로 단정하거나 자동으로 행정조치를 내리지 않는다. 모든 문서는 근거와 불확실성을 표시하고 담당자 검토 후 사용한다.

예측 엔진과 Agent는 분리되어 있다. 예측 엔진은 1km 권역 Top 3와 상대 순위를 출력하고,
`administrative_agent/`는 이 표준 출력만 입력받아 문서를 만든다. Agent는 민원 원본, 농가,
센서 데이터를 직접 조회하거나 원인 시설을 추론하지 않는다.

- `generate_agent_documents.py`: 운영 격자 비교 결과에서 최신 Event 문서 3종 생성
- `outputs/administrative_agent/complaint_briefing.md`: 악취 민원 상황 브리핑
- `outputs/administrative_agent/field_inspection_order.md`: 현장 점검 지시서
- `outputs/administrative_agent/followup_report_template.md`: 사후 결과보고서 확장 예시
- `outputs/administrative_agent/agent_output.json`: 서비스 연동용 구조화 결과

## 데이터와 검증

- 민원 원본: `data/익산시 악취 민원 데이터_20190528-20260818.xlsx`
- 실제 민원 시각 범위: 2019-05-28 23:23~2026-08-17 22:54
- 전체 행: 16,113건
- 익산시 민원: 14,994건
- 공통 Event: 160개
- 시간순 학습/테스트: 112개/48개
- 최종 지표: `outputs/operational_grid_comparison/metrics.json`
- 테스트 예측: `outputs/operational_grid_comparison/test_predictions.csv`

성능은 Event를 시간순으로 나누고 미래 Event만 테스트하는 방식으로 측정했다. Event Hit@3 85.1%는 실제 추가 민원이 있는 평가 가능 테스트 Event 47개에서 상위 3개 권역 중 하나 이상에 추가 민원이 포함된 비율이다. 표본이 47개이므로 이 비율의 95% 신뢰구간은 대략 75~95%다.

예측 CSV에는 격자 인덱스와 함께 중심 위·경도와 대표 읍면동 이름을 저장한다. 격자 원점은 민원 위·경도 중앙값이라 데이터가 바뀌면 이동하므로, 인덱스만으로는 다른 실행의 산출물과 위치를 맞출 수 없다.

## 주요 파일

- `compare_operational_grid_sizes.py`: 1km·1.5km·2km 공통 검증 및 최종 성능 재현
- `sensitivity_early_prediction.py`: 격자·시간창 후보를 내부검증으로 비교한 설계 근거. 제품 파이프라인의 입력은 아니다
- `optimize_early_prediction.py`: 후보 모델 학습·평가
- `build_odor_ai_mvp.py`: 민원 전처리와 Event 구성 공용 함수
- `fetch_kma_weather.py`: 담당자 상황 정보용 기상자료 수집
- `demo/index.html`: 운영 화면 데모
- `demo/build_demo_data.py`: `compare_operational_grid_sizes.py`의 1km 결과로 데모 데이터 생성

산출물끼리 결합할 때는 `event_id`를 키로 쓰지 않는다. `event_id`는 Event를 시각순으로
정렬한 뒤 붙이는 일련번호여서, 민원 데이터가 바뀌면 같은 번호가 다른 사건을 가리킨다.
실행 간 값이 변하지 않는 `event_hour`를 기준으로 결합한다.

## 현재 한계

- 테스트 Event가 48개로 늘었지만 여전히 연도별 추가 검증이 필요하다.
- 민원은 시민 인지와 신고 행동의 영향을 받으므로 실제 악취 영향권과 동일하지 않다.
- 장기간 센서 시계열과 현장 출동 결과가 없어 악취 발생 자체나 대응 효과를 학습하지 못했다.
- 센서·기상·배출원 정보가 모델 성능을 개선한다는 근거는 아직 확보되지 않았다.

## 확장 조건

장기간 센서 측정값, 측정소 좌표, 출동·도착 시각, 현장 감지 여부, 측정 결과와 조치 이력을 확보하면 센서 융합 모델과 대응 효과 평가로 확장한다. RAG는 수치 예측이 아니라 대응 매뉴얼, 법령, 과거 점검 이력 및 문서 양식을 검색하는 데 사용한다.

## 실행

Windows에서는 `run_demo.bat`을 더블클릭하면 Gemini 모드로 실행된다. API 호출 없이 발표하려면
`run_demo_template.bat`을 사용한다. 두 파일 모두 필요한 경우 `.venv`와 패키지를 자동으로 준비하고
브라우저에서 `http://127.0.0.1:8765`를 연다.

```powershell
python -m pip install -r requirements.txt
python compare_operational_grid_sizes.py
python generate_agent_documents.py
python demo/server.py
```

브라우저에서 `http://127.0.0.1:8765`를 연다. `.env`에 `GEMINI_API_KEY`와
`GEMINI_MODEL=gemini-3.6-flash`를 설정하면 Gemini로 문안을 생성한다. OpenAI를 사용하려면
`LLM_PROVIDER=openai`, `OPENAI_API_KEY`, `OPENAI_MODEL`을 설정한다. 키를 설정하지 않으면 검증 가능한 안전 템플릿을 사용한다.
API 키는 브라우저나 `demo/index.html`에 입력하지 않는다. 현장 결과는 화면에서 입력할 수 있으며
완성된 사후 결과보고서는 `outputs/administrative_agent/`에 저장된다.

현재 로컬 `.venv`는 존재하지 않는 Python 3.12 설치 경로를 참조하므로 Python을 설치하거나 가상환경을 다시 만들어야 한다.
