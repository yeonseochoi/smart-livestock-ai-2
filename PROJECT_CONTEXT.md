# 익산시 축산악취 AI 프로젝트 인수인계 문서

## 1. 문서 목적

이 문서는 새로운 컴퓨터 또는 새로운 Codex 대화에서 현재 프로젝트를 그대로 이어가기 위한 인수인계 자료임.

새 환경에서는 이 문서를 먼저 읽고 다음 파일을 확인해야 함.

```text
build_odor_ai_mvp.py
analyze_spatiotemporal_complaints.py
outputs/odor_ai_mvp/model_metrics.json
outputs/odor_ai_mvp/agent_response.md
```

이 문서에 기록된 수치는 시간 단위 오류와 실험 구조 문제를 수정한 뒤 다시 실행하여 검증한 최종 결과임.

---

## 2. 현재 주제

### 현재 구현 수준에 맞는 주제명

> 시민 가상센서 기반 축산악취 Event 탐지 및 단기 민원 확산지역 예측 시스템

### 최종 확장 목표

> 시민 가상센서 기반 축산악취 Event 복원·단기 확산 예측 및 선제대응 AI Agent

### 핵심 설명

익산시 악취 민원을 단순한 행정 신고가 아니라 위치·시각·체감강도·악취종류를 제공하는 시민 가상센서로 해석함.

비슷한 시간과 인접한 장소에서 발생한 여러 신고를 하나의 악취 Event 후보로 묶고, Event 초기 신고를 이용하여 이후 신고가 발생할 가능성이 높은 500m Grid를 예측함.

현재 모델이 직접 예측하는 대상은 실제 악취 농도가 아니라 향후 신고 발생 가능성이 높은 지역임.

```text
악취 농도 예측            : 아직 구현하지 않음
실제 물리적 확산 예측      : 아직 구현하지 않음
향후 민원 발생 Grid 예측   : 구현함
미관측 신고 Grid 복원      : 구현함
```

---

## 3. 확보 데이터

### 3.1 익산시 악취 민원

```text
data/익산시 악취 민원 데이터.xlsx
```

- 전체 행: 13,039건
- 익산시: 11,955건
- 완주군 삼례읍: 1,017건
- 기타 또는 지역 결측: 67건
- 기간: 2020-01-01 21:38~2026-07-30 00:17
- 주요 컬럼:
  - 악취발생일시
  - 위도
  - 경도
  - 시도
  - 시군구
  - 지역
  - 악취강도코드
  - 악취강도
  - 악취종류

### 3.2 익산시 무인악취 센서

```text
data/전북특별자치도 익산시_무인악취 수집데이터_20250911.csv
```

- 3,744행
- 13개 센서
- 5분 간격
- 실제 수록 기간: 2024-09-02 하루
- 측정값:
  - 복합악취
  - 황화수소
  - 암모니아
  - TVOC
- 같은 날짜 익산시 민원: 25건
- 센서 좌표가 없고 하루 자료뿐이므로 시간 일치도 사례검증에만 사용함
- 복원·예측 모델 학습에는 사용하지 않음

### 3.3 익산시 축산농가

```text
data/전북특별자치도 익산시_축산농가 현황_20241231.csv
```

- 1,267곳
- 정상 영업 1,213곳
- 주요 축종:
  - 한우 906곳
  - 돼지 169곳
  - 육계 114곳
- 포함 정보:
  - 주소
  - 축종
  - 사육두수
  - 시설면적
  - 영업상태
- 위·경도가 없어 아직 공간 모델에는 직접 사용하지 않음

### 3.4 익산시 돼지농장 가축분뇨

```text
data/전북특별자치도 익산시_돼지농장 가축분뇨 발생량 및 처리공법 정보_20241211.csv
```

- 241곳
- 폐수량 0 초과 시설 225곳
- 폐수량 합계 736.53
- 주소, 가축분뇨폐수량, 처리방법 포함
- 위·경도가 없어 아직 Event 주변 농가 순위에는 사용하지 않음

### 3.5 완주군 축산업

```text
data/전북특별자치도 완주군_축산업 현황_20250123.csv
```

- 전체 884곳
- 삼례읍 22곳
- 소재지가 읍·면 수준이므로 개별 농장 정밀 좌표화에는 제한이 있음

---

## 4. 주요 코드

### 4.1 기본 시공간 분석

```text
analyze_spatiotemporal_complaints.py
```

기능:

- 데이터 및 지역 분리
- BallTree 기반 시간·공간 인접률
- 동일 위치 50m 이하 반복 신고 제거 검증
- 1시간 Event 집계
- Event Level 1~3 탐색
- Top Event 분석
- 익산시와 삼례읍 비교
- Cross-boundary Event 후보 탐색
- HTML 지도 및 PNG 그래프 생성

실행:

```bash
python analyze_spatiotemporal_complaints.py
```

### 4.2 악취 AI MVP

```text
build_odor_ai_mvp.py
```

기능:

- 500m Grid 생성
- 최대 3시간 bounded ST-DBSCAN
- 복원·예측용 1시간 Event 샘플 생성
- Virtual Odor Sensor 공간 마스킹 복원 실험
- 초기 30분 → 이후 30분 신고 Grid 예측
- 내부 시간 검증 기반 모델 선택
- 센서와 민원 하루 사례 분석
- 선제대응 Agent 형식 Markdown 리포트 생성
- CSV·JSON·PNG 산출물 생성

실행:

```bash
python build_odor_ai_mvp.py
```

### 4.3 의존성

```text
requirements.txt
```

설치:

```bash
python -m pip install -r requirements.txt
```

---

## 5. 중요 버그 수정 이력

### pandas datetime 시간 단위 오류

초기 코드가 pandas datetime 값을 나노초로 가정했으나 현재 환경에서는 마이크로초 해상도로 저장되어 시간 차가 1/1000로 축소되는 문제가 있었음.

이로 인해 초기 시간·공간 인접률 93~99%는 잘못된 과대평가 결과였음.

현재는 다음과 같이 Timedelta를 이용해 실제 분 단위로 계산함.

```python
times = (
    (df["datetime"] - pd.Timestamp("1970-01-01"))
    / pd.Timedelta(minutes=1)
).to_numpy(float)
```

수정 전 93~99% 인접률은 사용하면 안 됨.

---

## 6. 수정 후 기본 시공간 분석 결과

### 익산시 시간·공간 인접 민원

| 조건 | 인접 신고 수 | 비율 |
|---|---:|---:|
| 30분 + 500m | 4,704 | 39.3% |
| 30분 + 1km | 5,834 | 48.8% |
| 1시간 + 1km | 6,497 | 54.3% |
| 1시간 + 2km | 7,489 | 62.6% |
| 2시간 + 3km | 8,547 | 71.5% |

### 동일 위치 50m 이하 제외

| 조건 | 인접 신고 수 | 비율 |
|---|---:|---:|
| 30분 + 1km | 5,449 | 45.6% |
| 1시간 + 2km | 7,188 | 60.1% |
| 2시간 + 3km | 8,290 | 69.3% |

### 규칙 기반 Event

- Level 1: 136개
- Level 2: 30개
- Level 3: 10개

최대 Event:

- 발생시각: 2020-08-02 21:00
- 신고: 68건
- 서로 다른 좌표: 64곳
- 주요 지역: 영등동, 어양동, 부송동, 동산동
- 평균 악취강도: 4.09

---

## 7. Event 정의 구분

### bounded ST-DBSCAN

- 공간 반경: 1km
- 시간 반경: 60분
- 최소 이웃: 5건
- Event 최대 지속시간: 180분
- 최종 군집: 323개
- 군집 탐색 및 시각화용으로 사용함

표준 DBSCAN은 신고가 연쇄 연결되어 수개월 또는 수년짜리 군집이 생길 수 있으므로 최초 core point 기준 최대 3시간으로 제한함.

### 머신러닝용 1시간 Event 샘플

민원을 정시 기준 1시간 단위로 집계함.

다음 조건을 모두 만족하는 시간 구간을 학습 샘플로 사용함.

- 신고 10건 이상
- 서로 다른 500m Grid 5곳 이상

총 129개의 1시간 구간을 사용함.

이 129개는 실제 악취 현상의 완전한 경계를 의미하지 않음. 동일한 길이의 머신러닝 학습·평가 샘플을 만들기 위한 분석 단위임.

기존 Level 1 Event가 136개인 이유는 서로 다른 원시 좌표 5곳을 기준으로 하기 때문임. Virtual Sensor 실험은 서로 다른 500m Grid 5곳을 사용해 129개가 됨.

---

## 8. Virtual Odor Sensor 복원 실험

### 실험 질문

> 하나의 Event에서 일부 시민 신고가 아직 관측되지 않았다고 가정할 때, 주변 신고만으로 숨겨진 신고 Grid를 복원할 수 있는가?

### 실험 구조

1. 익산시를 500m Grid로 분할함
2. 각 Event의 실제 활성 Grid 중 30%를 숨김
3. 나머지 70%만 모델 입력으로 사용함
4. 관측 Grid 반경 3칸, 약 1.5km 이내의 미관측 Grid를 후보로 생성함
5. 이미 관측된 Grid는 복원 후보에서 제외함
6. 숨긴 Grid인지 아닌지를 분류함

### 데이터 분리

- 과거 Event 70%: 학습
- 미래 Event 30%: 최종 테스트
- 학습 Event: 90개
- 테스트 Event: 39개
- 학습 후보 Grid: 8,574개
- 테스트 후보 Grid: 3,155개
- 테스트 양성 비율: 약 3.07%
- 테스트 숨김 Grid 후보 포함률: 90.7%

### 입력 특성

- 최근접 관측 Grid 거리
- 반경 1.5 Grid 신고 수
- 반경 3 Grid 신고 수
- Event 중심과 후보 Grid 거리
- 최근접 Grid 악취강도
- 거리 가중 악취강도
- 관측 Grid 수
- 관측 신고 수
- 시간대 sin/cos
- 과거 학습 Event의 Grid 발생빈도

### 모델 선택

최종 미래 테스트는 모델 선택에 사용하지 않음.

학습기간 내부의 마지막 20%를 검증셋으로 두고 다음 모델을 비교함.

- Extra Trees leaf 1
- Extra Trees leaf 2
- Extra Trees leaf 4
- Random Forest leaf 2
- Random Forest leaf 4

선택 모델:

```text
Extra Trees, min_samples_leaf=4
```

### 최종 결과

| 지표 | 선택 모델 | 거리 Baseline |
|---|---:|---:|
| PR-AUC | 0.226 | 0.083 |
| ROC-AUC | 0.886 | 0.790 |
| Precision | 23.6% | 9.5% |
| Recall | 43.3% | 67.0% |
| Event별 Top-K Recall | 23.6% | 10.7% |

테스트 양성 비율 약 3.07%와 비교하면 모델 PR-AUC 0.226은 무작위 기대 수준보다 약 7.4배 높음.

거리 Baseline보다 PR-AUC가 약 2.7배 높음.

### 정확한 해석

주변 민원과 과거 공간 패턴으로 숨긴 신고 Grid를 단순 거리보다 잘 구분할 가능성을 확인함.

실제 악취 농도 또는 신고가 없었던 공간의 실제 악취 존재를 복원한 것은 아님.

정확한 명칭은 다음과 같음.

> 시민 민원 기반 미관측 신고 Grid 복원 실험

---

## 9. 향후 30분 신고 Grid 예측 실험

### 실험 구조

```text
Event 최초 30분 신고 → 모델 입력
Event 이후 30분 신고 Grid → 정답
```

- 사용 가능 Event: 117개
- 학습 Event: 81개
- 미래 테스트 Event: 36개
- 테스트 후보 Grid: 2,682개
- 테스트 양성 비율: 약 5.03%

### 추가 특성

- 초기 15분 신고 수
- 다음 15분 증가량
- 초기 신고 Grid별 신고 수
- 초기 신고강도
- 공간 밀도 및 Event 중심거리
- 과거 Grid 발생빈도

### 모델 선택

학습기간 내부 검증에서 선택된 모델:

```text
Extra Trees, min_samples_leaf=4
```

### 최종 결과

| 지표 | 선택 모델 | 지속성 Baseline |
|---|---:|---:|
| PR-AUC | 0.258 | 0.138 |
| ROC-AUC | 0.888 | 0.749 |
| Precision | 23.3% | 23.8% |
| Recall | 58.5% | 3.7% |
| Top-K Recall | 28.3% | 24.5% |

ML과 지속성 점수를 결합한 하이브리드 가중치는 학습 내부 검증에서 선택함.

- 검증셋 선택 ML 가중치: 0.85
- 하이브리드 테스트 Top-K Recall: 28.2%

미래 테스트에서는 ML 단독 Top-K 28.3%가 하이브리드보다 소폭 높으므로 성능 보고에는 ML 단독을 주지표로 사용함.

### 정확한 해석

초기 민원만으로 이후 신고 Grid를 예측할 수 있는 신호가 있으며 전체 분류와 Top-K 모두 지속성 Baseline을 넘어섬.

아직 실제 악취의 물리적 확산 또는 농도를 예측한 것은 아님.

---

## 10. 센서 사례 분석

- 날짜: 2024-09-02
- 센서: 13개
- 같은 날 민원: 25건
- 시간 시차 상관 최대 절댓값: 약 0.299
- 탐색 범위 끝인 60분 시차에서 최대치가 발생함

하루 자료이며 센서 좌표도 없으므로 장기 성능 검증이나 일반화 근거로 사용하면 안 됨.

다음 표현만 사용하는 것이 안전함.

> 공개 센서와 기간이 겹치는 하루를 대상으로 민원과 센서 반응의 시간 일치도 사례 분석을 수행함.

---

## 11. 현재 Agent 수준

현재 Agent는 완전한 자율 Agent 또는 RAG 시스템이 아님.

구현된 기능:

- Event 요약
- 향후 30분 위험 Grid Top 5
- Grid 중심 위·경도
- 모델 성능과 한계 표시
- 이동형 센서 배치 및 순찰 권고
- Markdown 대응 리포트 생성

산출물:

```text
outputs/odor_ai_mvp/agent_response.md
```

아직 구현하지 않은 기능:

- 법령·악취관리 매뉴얼 RAG
- 농가 주소 좌표화
- Event 주변 농가 자동 검색
- 기상정보 연결
- 풍상 방향 잠재 배출원 순위
- 과거 유사 Event 검색
- 실시간 수집 및 알림

---

## 12. 현재 주장 가능한 내용

- 익산시 악취 민원에는 시간·공간적 군집 경향이 존재함
- 동일 위치 50m 이하 신고를 제외해도 인접성이 유지됨
- 민원을 Event 단위로 구성할 수 있음
- 주변 신고로 숨긴 신고 Grid를 거리 기준보다 잘 복원할 수 있음
- 초기 30분 신고로 이후 30분 신고 Grid를 Baseline보다 잘 예측할 수 있음
- 예측 Grid를 이동형 센서·현장 순찰 우선순위로 활용할 가능성이 있음

---

## 13. 현재 주장하면 안 되는 내용

- 실제 악취 농도를 복원함
- 실제 악취 확산을 정확히 예측함
- 특정 농장이 악취 원인임
- 센서 데이터로 전체 모델 성능이 검증됨
- 실시간 조기경보 성능이 검증됨
- 완성된 법령 RAG Agent임

현재 예측 대상은 실제 악취 농도가 아니라 향후 신고 발생 Grid임.

---

## 14. 주요 산출물

### 기본 분석

```text
outputs/data_region_summary.csv
outputs/iksan_proximity_summary.csv
outputs/iksan_hourly_event_summary.csv
outputs/iksan_top20_events.csv
outputs/iksan_top_event_detail.csv
outputs/iksan_top_odor_event_map.html
outputs/adjacent_proximity_comparison.csv
outputs/adjacent_event_comparison.csv
outputs/cross_boundary_events.csv
```

### AI MVP

```text
outputs/odor_ai_mvp/model_metrics.json
outputs/odor_ai_mvp/st_dbscan_events.csv
outputs/odor_ai_mvp/st_dbscan_labels.csv
outputs/odor_ai_mvp/bounded_event_summary.csv
outputs/odor_ai_mvp/virtual_sensor_test_predictions.csv
outputs/odor_ai_mvp/early_prediction_test_predictions.csv
outputs/odor_ai_mvp/sensor_complaint_timeline.csv
outputs/odor_ai_mvp/agent_response.md
outputs/odor_ai_mvp/virtual_sensor_reconstruction.png
outputs/odor_ai_mvp/early_prediction_grid.png
outputs/odor_ai_mvp/sensor_complaint_case.png
```

---

## 15. 다음 작업 우선순위

### 1순위: 기상자료 결합

- Event 시각별 풍향
- 풍속
- 강수
- 습도
- 가장 가까운 AWS 또는 ASOS 지점 연결

목표:

- 과거 민원 다발지역 의존도 감소
- Event별 동적 확산 방향 특성 추가
- 초기 신고 이동 방향과 풍향 일치도 계산

### 2순위: 농가 주소 좌표화

- 익산시 농가 주소를 위·경도로 변환
- 돼지농장 분뇨자료와 농가 자료 병합
- Event 및 위험 Grid 주변 농가 탐색

후보 점수 예시:

```text
거리
+ 풍상 방향 일치도
+ 축종 일치
+ 사육두수
+ 시설면적
+ 분뇨량
+ 처리방식 위험도
```

반드시 원인 농가 확정이 아니라 현장점검 우선 후보라고 표현해야 함.

### 3순위: Grid 및 Event 민감도 분석

- 250m, 500m, 1km Grid 비교
- 후보 반경 1km, 1.5km, 2km 비교
- 입력 20분, 30분, 60분 비교
- 예측 30분, 60분 비교
- 정시 1시간 Event 경계 민감도 검증

### 4순위: 모델 개선

- 기상·농가 동적 특성 추가
- Event별 좌표 중심 이동벡터
- 방향성 이웃 특성
- XGBoost/LightGBM 사용 가능 여부 확인
- Event 단위 시계열 교차검증
- 확률 calibration
- 비용 기반 Precision@K 최적화

### 5순위: Agent 확장

- 과거 유사 Event 검색
- 관련 법령·매뉴얼 문서 확보
- RAG 구축
- 예측 근거와 불확실성 표시
- 행정 점검 체크리스트 생성

---

## 16. 새 환경에서의 시작 프롬프트

새 Codex 대화에서 다음과 같이 요청하면 됨.

```text
이 저장소는 이전 Codex 대화에서 진행한 익산시 축산악취 AI 프로젝트야.

먼저 PROJECT_CONTEXT.md를 처음부터 끝까지 읽고,
build_odor_ai_mvp.py,
analyze_spatiotemporal_complaints.py,
outputs/odor_ai_mvp/model_metrics.json을 확인해줘.

PROJECT_CONTEXT.md에 기록된 완료 작업을 다시 하지 말고,
현재 검증된 결과와 한계를 유지하면서 다음 작업부터 이어서 진행해줘.
```

---

## 17. GitHub 업로드 주의사항

원본 민원 및 농가 데이터에 개인정보 또는 공개 제한 정보가 있는지 확인해야 함.

공개 저장소라면 다음 항목을 그대로 업로드하지 않는 것을 권장함.

```text
data/
*.xlsx
민감한 원본 CSV
개별 신고 위치가 포함된 상세 산출물
```

공개 가능한 코드·설명·요약 지표만 GitHub에 올리고 원본 데이터는 별도 보관하는 방식이 안전함.

