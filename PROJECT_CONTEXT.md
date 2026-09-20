# 익산시 악취 민원 선제 대응 프로젝트

> 최종 정리일: 2026-09-20

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

기상·발생원 정보는 이 확산 예측 점수에 넣지 않는다. 2026-09 소거 실험(ablation study)에서 기상, 발생원 노출, 역추적 점수, 풍향 조건부 과거 민원 지도 4방식 모두 개선이 없었다. 이유는 이후 30분 민원의 51%가 같은 격자, 89%가 2km 안에서 발생해 "어디로 번지나"가 현재 위치로 거의 정해지기 때문이다(`outputs/ablation/ablation_table.md`).

### 1-1. 발생 위험 예보 (1단 원형, 2026-09 추가)

- 입력: 시각 t의 바람(ASOS+AWS 격자별 보간)·강수·기온·습도, 격자 6km 상풍측 축산 배출 노출(EMEP/EEA NH3 계수 × 사육두수), 격자별 과거 민원 빈도, 격자×풍향 구간별 과거 민원율(조건부 확률 함수 CPF형, t 이전 365일)
- 출력: "다음 1시간 안에 이 1km 격자에서 민원이 생길 위험" 상위 격자. 확률이 아닌 상대값(같은 시각 최고 = 100)
- 스크립트: `run_onset_risk.py`. 채택 실험군 R5. 라벨을 냄새 유형(가축·공장·하수)으로 제한한 유형별 모델도 같은 스크립트(`--label-type`)
- 검증(시간순 분할, 학습 2020~2024, 테스트 2025-01~2026-07, seed 3 평균, seed 10 확인은 experiment_summary.md의 b4_seed10): 직전 3시간 시 전체 민원이 없던 '조용한 시각'에 상위 5 격자 적중률(Hit@5)
  - 과거 빈도만(R0) 0.546 → +기상(R1) 0.548 → +축산 발생원(R2) 0.590 → +풍향 조건부 민원 지도(R5) 0.599. 바람을 ASOS만 쓰면 R5 0.578~0.581 (`outputs/onset_risk/metrics.json`, 변형 실행은 `outputs/onset_risk/experiment_summary.md`)
  - 유형별(`metrics_{livestock,factory,sewage}.json`): 가축 0.492 → 0.588, 공장 0.644 → 0.747, 하수 0.648 → 0.685(n 작음)
- 이 예보는 확산 예측(Top 3)과 별개의 참고 정보로, 행정 문서와 화면의 '사전 경보 (참고)' 절에 민원 접수 1시간 전 시각 기준으로 붙는다(`outputs/onset_risk/onset_alerts*.csv`)

이 구조는 "발생원과 기상을 대기과학 역추적으로 반영하되 기존 성능은 지킨다"는 1차 멘토링 요구를 두 단계로 나눠 충족한다. 확산 예측(2단)은 손대지 않아 성능이 보존되고, 발생 예보(1단)에서 발생원·바람이 수용체 모델(receptor model) 방식으로 들어간다.

#### 1단·2단 결합으로 권역 1개만 추천할 수 있는지 (2026-09-20 검토, 무효)

- 질문: 멘토 지적(권역 3곳은 넓다)에 따라 2단 Top 3 중 1곳을 1단 위험 순위로 골라 1개만 추천할 수 있는가.
- 방법(`fuse_onset_spread.py`): 2단 모델·Top 3 집합은 그대로 두고, 민원 접수 1시간 전 1단 순위가 가장 높은 격자를 1순위로 올리는 후처리. 1단 점수는 Event마다 그 이전 자료로만 학습한 연도별 확장 창(expanding window) 분할(`run_onset_risk.py --train-end 2021-01-01 … 2024-01-01`)에서 얻어 누수가 없다.
- 결과(테스트 Event 47개, `outputs/onset_spread_fusion/metrics.json`): 2단 단독 Hit@1 0.553 · Hit@2 0.745 · Hit@3 0.851. 1단으로 Top 3 중 1순위를 고르면 Hit@1 0.553(고친 Event 5, 망친 Event 5), Top 2 중 고르면 0.574(5/4, 부호 검정 p=1.0). 개선 근거 없음.
- 원인: 2단 후보는 첫 민원 인근 격자라 1단에서도 모두 상위에 있다(Top 3 안 정답 격자의 1단 순위 중앙값 3위, 오답 2위). 두 모델은 "어느 동네"에는 동의하지만 그 안의 1 km 격자 하나를 가르는 정보는 1단에 없다.
- 후보 축소 규칙(채택, `administrative_agent/policy.py` `narrow_candidates`; 검증 `fuse_onset_spread.py --narrow 30`, `outputs/onset_spread_fusion/narrow_table.md`): 2단 후보를 Event 1시간 전 1단 위험 순위 30위 이내 격자로 좁히면 후보 평균 31.1 → 15.1개, Hit@1/2/3은 0.553/0.745/0.851 그대로(바뀐 Event 0). 정답 격자의 1단 순위 중앙값 6위, 30위 이내 98.7%. `generate_agent_documents.py`와 Streamlit이 기본으로 적용하며, 1단 산출물(`onset_alerts.csv` 상위 30, `onset_cells.csv`)이 없으면 후보 전체에서 고른다. 1단은 민원 이력 있는 186격자만 다루므로 1단 격자 밖 후보는 2단 점수가 더 높을 때만 예외 허용한다. 성능 향상은 아니고 후보 수 축소다.
- 대안(2단 격자 크기): `outputs/operational_grid_comparison/metrics.json` 기준 Hit@3는 1 km 0.851 · 1.5 km 0.957 · 2 km 0.894. 저장된 테스트 예측에서 재계산한 Hit@1은 1 km 0.553 · 1.5 km 0.609 · 2 km 0.617, Hit@2는 0.745 · 0.870 · 0.830. 권역 수를 줄이려면 모델 결합보다 격자 크기 조정이 근거가 있다(미채택, 발표용 판단 필요).

### 2. 행정 대응 Agent

예측 결과와 확인 가능한 현재 상황을 이용해 다음 문서의 초안을 생성한다.

- 악취 민원 상황 브리핑
- 현장 점검 지시서
- 사후 결과보고서
- Event별 AI 현장 대응 참고 가이드

Agent는 시설을 원인으로 단정하거나 자동으로 행정조치를 내리지 않는다. 모든 문서는 근거와 불확실성을 표시하고 담당자 검토 후 사용한다.

예측 엔진과 Agent는 분리되어 있다. 예측 엔진은 1km 권역 Top 3와 상대 순위를 출력하고,
`administrative_agent/`는 이 표준 출력만 입력받아 문서를 만든다. Agent는 민원 원본, 농가,
센서 데이터를 직접 조회하거나 원인 시설을 추론하지 않는다.

- `generate_agent_documents.py`: 운영 격자 비교 결과에서 지정 Event(미지정 시 데이터상 최신 Event)의 문서 3종 생성
- `outputs/administrative_agent/complaint_briefing.md`: 악취 민원 상황 브리핑
- `outputs/administrative_agent/field_inspection_order.md`: 현장 점검 지시서
- `outputs/administrative_agent/followup_report_template.md`: 사후 결과보고서 확장 예시
- `outputs/administrative_agent/agent_output.json`: 서비스 연동용 구조화 결과

현재 저장된 문서 예시는 `EVT-0175`(2024-09-05 01:00)를 기준으로 생성되어 있다. 이는 사용자가 선택한 과거 Event의 대응 문서 예시이며, 예측 CSV에서 시간상 가장 최신인 `EVT-0182`와는 다르다. `generate_agent_documents.py`를 Event 지정 없이 다시 실행하면 최신 Event 기준 산출물로 교체된다.

## 데이터와 검증

- 민원 원본: `data/익산시 악취 민원 데이터_20190528-20260818.xlsx` (악취종류: 가축 분뇨 8,142 / 기타 3,523 / 공장 2,665 / 하수구 1,369 / 소각 210 / 음식조리 184)
- 기상: ASOS 전주 146·군산 140(2020-01~2026-07, `outputs/weather_integration/asos_hourly_2020_2026.csv`), AWS 익산 702·함라 733·여산 763·김제 737·진봉 736(2020-01~2026-08, `outputs/weather_integration/aws_hourly_2020_2026.csv`, 기상자료개방포털 파일셋). `wind_sources.py`가 두 자료를 역거리 가중(IDW)으로 격자별 바람으로 합친다. 익산 시내 기준 ASOS와 AWS 풍향 차이 중앙값 26°
- 발생원: 축산 2,778(익산·김제 축산현황, VWorld 지오코딩) + 공장 225·하수처리 93·산업단지 경계 21·분뇨처리 7·폐기물 3 (Track A 4판, `outputs/source_backtrack/sources.csv` 로컬 전용). 축산 외 발생원은 검정과 모델 모두에서 기여가 없어 채택하지 않았다(`docs/track_a_report.md`, `outputs/wind_lag_sweep/factory_filter_test.md`)
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
- `fetch_kma_weather.py`: ASOS 시간자료 수집(대기안정도 재료 옵션은 Track A 추가분)
- `wind_sources.py`: ASOS+AWS 통합, 격자별 IDW 바람(`WindField`)
- `species_weight_sets.py`: 축종 배출계수 세트(채택 EMEP/EEA NH3, 대조 시설 수). 선택 근거는 `sensitivity_species_weights.py` → `outputs/wind_source_association/species_weight_sensitivity.md`
- `run_onset_risk.py`: 1단 발생 위험 예보 학습·평가(R0·R1·R2·R5·R5o)와 `onset_alerts*.csv` 생성(`--train-end`로 분할 시점 변경, `onset_event_scores*.csv`는 결합 평가 입력). 기여 없던 실험군(건물 밀도 R3·외부 시설 R2s·산단 경계 R2z·대기안정도 R2w·R4)과 실험 옵션(참조 풍향 창·중심점 모드·지점 선택·정체 처리)은 2026-09-20 리팩토링에서 제거했다. 결과는 `experiment_summary.md`, 코드는 git 태그 `pre-refactor-2026-09-20`
- `fuse_onset_spread.py`: 1단·2단 결합(2단 Top 3 중 1단 순위로 1순위 선택) 평가 → `outputs/onset_spread_fusion/`
- `작업 최종 아키텍쳐.md`: 최종 구조 설명(입력·2단·1단·결합·Agent·실행·피드백 대응). `outputs/onset_risk/experiment_summary.md`: 1단 변형 실행 25건 요약(원본 json은 정리 시 삭제, git 이력에 있음)
- `run_ablation.py`: 2단 확산 예측에 기상·발생원을 넣는 소거 실험(M0~M4)
- `test_wind_source_association.py`, `wind_window_sweep.py`, `minute_wind_test.py`, `compare_wind_sources.py`, `test_factory_source_filters.py`: 풍향–발생원 연관 검정(층화 셔플, 참조 바람 창 24조합, 분 자료 창 19종, 바람 자료원 5종, 공장 필터). 결론 표는 `outputs/wind_lag_sweep/*.md`
- `build_source_backtrack.py`(Track A): Event별 발생원 역추적 후보와 격자 점수
- `demo/index.html`: 운영 화면 데모
- `demo/build_demo_data.py`: `compare_operational_grid_sizes.py`의 1km 결과로 데모 데이터 생성

산출물끼리 결합할 때는 `event_id`를 키로 쓰지 않는다. `event_id`는 Event를 시각순으로
정렬한 뒤 붙이는 일련번호여서, 민원 데이터가 바뀌면 같은 번호가 다른 사건을 가리킨다.
실행 간 값이 변하지 않는 `event_hour`를 기준으로 결합한다.

## 현재 한계

- 테스트 Event가 48개로 늘었지만 여전히 연도별 추가 검증이 필요하다.
- 민원은 시민 인지와 신고 행동의 영향을 받으므로 실제 악취 영향권과 동일하지 않다.
- 장기간 센서 시계열과 현장 출동 결과가 없어 악취 발생 자체나 대응 효과를 학습하지 못했다.
- 기상·배출원 정보는 확산 예측(2단)에서는 개선 근거가 없고, 발생 예보(1단)에서만 근거가 있다. 1단의 절대 수준은 186개 격자 중 5개를 찍어 조용한 시각 59~60%이며 과거 빈도만으로도 54%가 나온다. 발생원·바람의 기여는 그 위 +4~6%p(전체), 공장 냄새 +10%p다.
- 역추적 엔진(Track A)의 무작위 풍향 대조는 전체 조건 p=0.030(0~1시간 창)이지만 건조·강수 층별로는 유의하지 않다. 발생원 후보는 참고 정보다.
- AWS 함라·여산 지점은 풍속 1 m/s 미만 비율이 63~65%로 익산(30%)·김제(10%)와 달라 관측 환경 차이가 의심된다.
- 1단 예보 수치는 테스트 구간 사후 계산값이다. 실시간 운영에는 기상 실시간 호출과 매시간 추론 경로가 따로 필요하다.
- 1단·2단을 결합해 권역 1곳만 추천하는 방식은 개선 근거가 없다(Hit@1 0.553 그대로). 1개 추천을 하려면 2단 Hit@1 0.553을 그대로 감수하거나 격자 크기를 키워야 한다.

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

## 최종 확인 상태

- Git: 2026-09-20 기준 Track A(`feat/source-backtrack`)와 Track B(`feat/integration-ablation`)를 `main`에 병합했다(`fb6d597`). 이후 main에서 결합 평가·정리·리팩토링을 커밋했고 원격 push는 하지 않았다. 워크트리 `_trackB`는 삭제했다.
- 테스트: `python -m unittest discover -s tests` 35건 전부 통과(Python 3.10, `tests/test_streamlit_app.py`는 `tomllib` 없으면 `toml`로 대체).
- 1단 산출물: `outputs/onset_risk/onset_alerts.csv`(전체, 시각별 상위 30)와 `onset_alerts_{livestock,factory,sewage}.csv`(유형별)는 ASOS+AWS 격자별 바람·R5 기준으로 생성한다. `onset_cells.csv`는 1단 격자 186개 목록(후보 축소 규칙용).

```powershell
python -m unittest discover -s tests -v
```
