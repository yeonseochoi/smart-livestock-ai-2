# DEVELOP 계약: 폭주 밤 예보(Codex-1) · 데이터·대기 지점(Codex-2) · 현장 추적·보고서·UI(Claude)

작성일 2026-09-27. 서비스 설계는 [DEVELOP.md](../../DEVELOP.md)를 따르고, 이 문서는 세 트랙 사이의 스키마·파일 소유권·검증 규칙만 정한다.
기존 Event Top 3 모델과 역추적 계약([source_backtrack_contract.md](source_backtrack_contract.md))은 그대로 유지한다. DEVELOP의 E 모듈이 기존 Event·Top 3를 읽기 전용으로 재사용한다.

## 트랙과 브랜치

| 트랙 | 담당 | 브랜치 | 범위 |
|---|---|---|---|
| Forecast | Codex-1 | `feat/odor-service-forecast` | M1 A 폭주 밤 예보 |
| Standby | Codex-2 | `feat/odor-service-standby` | M0 데이터 정비, M2 B 대기 지점 |
| Field | Claude | `feat/odor-service-field` | M3 C 현장 추적, M4 E 연동, M5 보고서·재보정, M6 UI |

- 세 트랙은 동시에 시작한다. M1은 기상 원본 CSV(`common.ASOS_FILE`, `common.AWS_FILE`)를 직접 읽으므로 M0를 기다리지 않는다.
- 세 브랜치 모두 main에서 분기하고 main에 직접 커밋하지 않는다. main 반영은 PR로 한다.
- Field 트랙은 M0 산출물(`farms.parquet`)이 나오기 전에는 `tests/fixtures/odor_service/`의 가짜 농가 데이터로 개발한다.

## 파일 소유권

| 트랙 | 소유 파일 |
|---|---|
| Forecast (Codex-1) | `odor_service/forecast/**`, `outputs/odor_service/forecast/**`, `tests/test_odor_service_forecast*.py` |
| Standby (Codex-2) | `odor_service/data/**`, `odor_service/standby/**`, `outputs/odor_service/data/**`, `outputs/odor_service/standby/**`, `tests/test_odor_service_standby*.py` |
| Link (Codex-3) | `odor_service/event_link/**`, `outputs/odor_service/replay/**`, `tests/test_odor_service_link*.py` |
| UI (Codex-4) | `odor_service/app.py`, `tests/test_odor_service_app*.py` |
| Field (Claude) | `odor_service/field/**`, `administrative_agent/**`, `streamlit_app.py`, `demo/**`, `outputs/odor_service/field/**`, `outputs/odor_service/report/**`, `tests/test_odor_service_field*.py` |
| 공용 (수정 시 합의) | `DEVELOP.md`, 이 계약 문서, `odor_service/__init__.py`, `odor_service/common.py`, `tests/test_odor_service_common.py`, `tests/fixtures/odor_service/**` |

- 상대 트랙 소유 파일은 읽기만 한다. `odor_service/common.py`(2026-09-27 작성, 테스트 5건)는 모든 트랙이 import만 하고, 변경이 필요하면 멈추고 묻는다.
- 원본 데이터(`data/**`)는 모든 트랙이 수정하지 않는다.

## 공통 규약 (`odor_service/common.py`, 공용)

- 풍향: 불어오는 방향, 북=0°, 시계방향. 풍하 벡터 = (−sinθ, −cosθ).
- 거리: x = lon·111.32·cos(35.95°), y = lat·110.57 (km).
- 밤 날짜: `datetime − 6h`의 날짜(`night_date`). 야간 집계 창은 21:00~04:59(`in_night_window`). 창 밖 시각은 밤 집계에 넣지 않는다.
- 가축 민원 로딩은 `load_livestock_complaints()`만 쓴다(7,545건, 2019-05-28~2026-08-17).
- 시각은 KST, tz 정보 없는 `datetime64`.
- 결측은 NaN으로 남기고 0으로 채우지 않는다.

## 산출물 스키마

### `outputs/odor_service/data/farms.parquet` (M0, Codex-2)

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `farm_id` | str | `IK-<순번>` / `GJ-<연번>` |
| `city` | str | `익산` / `김제` |
| `name` | str | 업체명 |
| `lat`, `lon` | float | 좌표 없으면 NaN |
| `coord_precision` | str | `exact`(지번정확) / `approx`(리중심·첫필지) / `none` |
| `species_raw` | str | 원본 사육업종 |
| `species` | str | `돼지` / `가금` / `소` / `기타` |
| `heads` | float | 사육두수. 김제 원본에는 없으므로 NaN |
| `eq` | float | `heads × species_weight_sets.EEA_NH3[species_raw]` (kg NH3/년). 계수표에 없는 업종 또는 `heads` 결측은 NaN; 가정 계수 적용 여부는 `eq_assumed`로 표시 |
| `status` | str | 영업상태. 김제 원본에는 없으므로 NaN |

- 출처: 익산 `data/farms/iksan/farms_geocoded.csv`(1,267행), 김제 `data/farms/gimje/전북특별자치도 김제시_축산현황_20250515_geocoded.csv`(1,511행, 좌표 184곳).
- 행 수와 `coord_precision`별 건수를 `outputs/odor_service/data/farms_report.md`에 기록한다.

### `outputs/odor_service/data/weather_hourly.parquet` (M0, Codex-2)

`datetime, station_id, station_name, source(ASOS|AWS), station_latitude, station_longitude, wind_direction, wind_speed, temperature, humidity, rainfall_hour`

- 출처: `outputs/weather_integration/asos_hourly_2020_2026.csv`, `outputs/weather_integration/aws_hourly_2020_2026.csv`. 컬럼명은 원본과 같다.
- 관측소별 기간과 결측률을 `outputs/odor_service/data/weather_report.md`에 기록한다.

### `outputs/odor_service/forecast/night_risk.csv` (M1, Codex-1) — A의 교체 가능 인터페이스

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `night_date` | date | 밤 날짜 |
| `season` | int | 연도 |
| `risk_score` | float | 클수록 위험. 모델 A는 예상 가축 민원 수 |
| `season_rank` | int | 시즌 내 `risk_score` 순위(1=최고) |
| `alert` | bool | 시즌 상위 N밤 경보 여부(기본 N=30) |
| `model_id` | str | 예: `hgb_poisson`, `calendar`, `random`, `public_forecast` |
| `weather_input` | str | `observed` / `forecast` |
| `train_end` | date | 학습에 쓴 마지막 밤. `train_end < night_date`여야 함 |
| `observed_complaints` | float | 백테스트용 실제 가축 민원 수. 운영 시 NaN |

- 다른 예측원(공공 확산 예보 등)도 이 스키마를 채우면 Field 트랙과 UI가 수정 없이 읽는다.
- 백테스트 지표는 `outputs/odor_service/forecast/backtest_metrics.json`에 시즌별 공친 밤 비율, 폭주 밤 포착률, 무작위·달력 기준선, 부트스트랩 95% CI로 기록한다.

### `outputs/odor_service/standby/standby_points.csv` (M2, Codex-2)

`point_id, cluster_id, grade(주|관찰|제외), city, lat, lon, cluster_lat, cluster_lon, n_farms, eq_sum, lift_all, lift_ci_low, lift_ci_high, lift_recent3, wind_source(ASOS|AWS)`

- 연도별 lift는 `outputs/odor_service/standby/lift_by_year.csv`(`cluster_id, year, lift, n_complaints`)에 둔다.

### `outputs/odor_service/replay/<night_date>.json` (연결, Codex-3) — 과거 하루 다시 보기

오후 경보 → 대기 장소 → 기존 사건·동네 Top 3 → 방문 순서를 한 파일로 묶는다. 형식은 `tests/fixtures/odor_service/replay_sample.json`과 같다.

- 대상: 기존 시험 사건(`outputs/operational_grid_comparison/test_predictions.csv`, 48개) 중 `night_risk.csv`(hgb_poisson) 기간 안에 있는 사건의 밤. 경보 여부와 관계없이 만든다.
- `afternoon`: `night_risk.csv` 값 + 그날 밤(21~04시) 익산 AWS 702 평균 풍향·풍속(벡터 평균).
- `standby.points`: `standby_points.csv`의 AWS·주 등급. `chosen_point_id`는 밤 평균 풍향의 불어오는 쪽(도시 중심 35.948, 126.957 기준 방위와 풍향 차이 최소)인 주 대기 장소.
- `events[].top3`: 기존 모델 예측(`test_predictions.csv`)의 score 상위 3칸. `hit` = target.
- `events[].direction_overlap`: 사건 첫 30분 가축 민원 지점마다, 사건 시각 AWS 702 풍향 기준으로 각 대기 장소가 불어오는 쪽 ±45°·15 km 안인지 센다. 표시용이며 예측·순위에 쓰지 않는다.
- `events[].field`: 선택된 대기 장소를 담당자 위치로 두고 `odor_service.field.engine.score_candidates` 기본 가중치(팀원 안)로 만든 카드 5장.
- 색인: `outputs/odor_service/replay/index.json` = [{night_date, alert, n_events, hit_any_top3}].

### 화면 (Codex-4): `odor_service/app.py`

- `streamlit run odor_service/app.py`. 기존 `streamlit_app.py`는 수정하지 않는다.
- 입력은 `outputs/odor_service/replay/*.json`만 읽는다. 없으면 `tests/fixtures/odor_service/replay_sample.json`으로 보여 준다.

### Field 트랙 산출물 (Claude)

- `outputs/odor_service/field/simulation_v2.json`: DEVELOP 8장 조건별 헛방문 수·Top 3 적중, 기준선 3종.
- `outputs/odor_service/report/`: 보고서 스키마와 가짜 보고서 50건 재보정 결과.

## 누수·검증 규칙

- A: 시간 분할만 허용한다. 시즌 s 평가 모델은 s 이전 시즌으로만 학습한다(`train_end < night_date` 검사).
- A: 특징은 그 밤의 기상과 달력만 쓴다. 같은 밤이나 이후의 민원은 특징에 넣지 않는다.
- B: lift는 서술 통계라 전 기간을 써도 되지만, 등급을 정할 때 쓴 기간을 `standby_points.csv` 옆 메모에 적는다.
- 관측 기상으로 만든 결과에는 "운영 시 저하 가능"을 명시한다.
- 4장 수치를 재현하지 못하면 DEVELOP.md 수치를 재현값으로 고치고 차이를 기록한다. 수치를 맞추려고 설정을 바꾸지 않는다.

## 막힘 처리

키 없음, 스키마 불일치, 원본 결측 등으로 막히면 추측으로 채우지 않고 멈춘 뒤 사용자에게 묻는다.
