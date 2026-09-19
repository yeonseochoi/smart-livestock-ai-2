# Track A ↔ Track B 계약: 발생원·기상 역추적 산출물

> 작성 2026-09-20. 이 문서가 두 트랙 사이의 유일한 인터페이스 정의다. 설계 페이지나 프롬프트와 다르면 이 문서를 따른다.
> 변경이 필요하면 두 트랙이 모두 동의한 뒤 이 파일을 고치고, `tools/validate_backtrack_output.py`를 같이 고친다.

## 목적

Track A(역추적 엔진)는 축산시설 위치와 Event 이전 기상으로 "각 1km 격자에 발생원 냄새가 도달했을 가능성" 점수를 만든다.
Track B(성능보존형 통합)는 그 점수를 기존 XGBoost 특징에 **추가**해 기존 성능이 유지·개선되는지 동일 분할·동일 지표로 판정한다.
두 트랙은 아래 CSV 하나로만 만난다.

## 이미 시도했다가 실패한 것 (반복 금지)

2026-08 커밋 `ab93076`의 `experiment_vworld_source_features.py`에서 익산 농가 1,094곳(VWorld 지오코딩) + Event **당시** 풍향으로 만든
정적 downwind 특징 15개와 기상 특징 14개를 한꺼번에 넣었다. 내부검증은 소폭 개선, 최종 테스트는 PR-AUC −0.023으로 **하락**해 제거됐다.
(`git show ab93076:outputs/vworld_source_experiment/metrics.json`)

이번 설계가 다른 점 — 이 넷이 지켜지지 않으면 같은 결과가 나온다:
1. Event 당시 풍향이 아니라 **도달시간만큼 전**의 풍향을 쓴다 (거리 ÷ 풍속으로 역산).
2. 정적 "격자 주변 농가 수"가 아니라 **지금 민원 위치를 설명하는 발생원을 먼저 고르고**, 그 발생원들이 다음에 어디로 보낼지를 점수화한다 (`source_fit_score`).
3. 추가 특징은 **3~5개**로 제한한다. 학습 Event가 112개라 29개는 과적합이었다.
4. **김제 농가를 포함**한다. 이전 실험은 익산만 썼다. 익산 남부 민원의 원인 가설이 김제다.

`git show ab93076:experiment_vworld_source_features.py`의 `normalize_address`, `address_variants`는 주소 정규화에 재사용할 수 있다.

## 결합 키와 격자 규칙

- **결합 키는 `event_hour`** (+ `grid_x`, `grid_y`). `event_id`는 Event를 시각순 정렬해 붙인 일련번호라 데이터가 바뀌면 같은 번호가 다른 사건을 가리킨다. 참고 컬럼으로만 싣고 키로 쓰지 않는다.
- 격자 원점은 민원 위·경도 중앙값 `(lat0, lon0)`, `grid_m = 1000`. `build_odor_ai_mvp.add_grid_columns`와 같은 식이며 `sensitivity_early_prediction.add_grid(complaints, 1000)`으로 계산한다. 원점은 데이터가 바뀌면 이동하므로 **모든 행에 격자 중심 좌표를 함께 기록**한다(`odor.grid_centroid`).
- Event 목록과 후보 격자는 `compare_operational_grid_sizes.py`와 동일하다: 1km·1.5km·2km 공통 Event(현재 160개), 후보 격자는 초기 30분 관측 격자 주변 **반경 2칸** (`odor.candidate_cells(observed, radius=2)`; `sensitivity.build_data`의 `round(1500 / grid_m)`). 반경 3이 아니다.
- 기준표 `outputs/backtrack_contract/reference_candidates.csv`에 정확한 (event_hour, grid_x, grid_y, 중심 좌표)가 있다. Track A는 이 표의 칸을 **모두 포함**해야 하며(더 넓게 내도 됨), 표에 없는 event_hour를 내면 안 된다.

## 누수 금지

- Event별 점수는 `event_hour + 30분`(`odor.INPUT_MINUTES`)까지의 정보만 쓴다. 해당 Event의 이후 30분 민원, 미래 Event의 민원은 어떤 형태로도 참조하지 않는다.
- 과거 민원으로 만드는 발생원 지도(2차)는 해당 Event의 `event_hour` **이전** 민원만으로 만든다(rolling). 실제로 쓴 마지막 민원 시각을 `history_cutoff`에 기록한다. 과거 지도를 안 썼으면 `event_hour`를 그대로 넣는다.
- 검증기는 `history_cutoff <= event_hour`를 확인한다. 이것으로 누수가 전부 잡히진 않으므로, Track A는 단위 테스트에 "미래 민원을 입력에 섞어도 결과 불변"을 포함한다.

## 산출물

### `outputs/source_backtrack/grid_scores.csv` (Track B가 읽는 파일, git 커밋)

컬럼 순서 고정. 인코딩 utf-8-sig.

| 컬럼 | 형 | 뜻 | 규칙 |
|---|---|---|---|
| `event_hour` | datetime | Event 시작 시각 | **키** |
| `event_id` | str | 참고용 일련번호 | 키 아님 |
| `grid_x`, `grid_y` | int | 1km 격자 인덱스 | **키**, Event 내 유일 |
| `center_latitude`, `center_longitude` | float | 격자 중심 | 기준표와 1m 이내 |
| `source_fit_score` | float | 현재 민원을 잘 설명하는 발생원들이 이 격자로 보낼 가능성 | 0~1, **Event 내 최댓값=1로 정규화**, 결측 NaN |
| `forward_plume_score` | float | 전 발생원 정방향 노출 합 | 0~1, Event 내 정규화, 결측 NaN |
| `prior_downwind_score` | float | 과거 민원 발생원 지도의 정방향 투영 (2차) | 0~1 또는 NaN. **1차는 전부 NaN** |
| `lagged_wind_alignment` | float | 최대 기여 발생원 기준, 도달시간 전 풍향과 방위의 cos | −1~1 |
| `travel_time_min` | float | 최대 기여 발생원의 도달시간 | 분, ≥0 |
| `rain_1h`, `rain_3h` | float | 강수량 | mm, ≥0. 판단 없이 값만 |
| `stagnation_flag` | int | 풍속<1 m/s 또는 30분 풍향 SD>45° | 0/1 |
| `backtrack_uncertainty` | float | 풍향 변동·정체·village 정밀도 비율 평균 | 0~1 |
| `weather_source` | str | 바람 출처 | `aws` / `asos` / `none` |
| `history_cutoff` | datetime | 과거 지도에 쓴 마지막 민원 시각 | ≤ `event_hour` |

- 결측은 0이 아니라 **NaN**으로 남긴다. "값이 없음"과 "가능성 0"을 구분해야 한다. XGBoost는 NaN을 그대로 다룬다.
- 2019년 Event는 ASOS 자료(2020~)가 없다. 그 경우 `weather_source="none"`, 점수는 NaN.
- 기준 Event의 50% 미만만 포함하면 검증 실패.

### 부수 산출물

| 파일 | 내용 | git |
|---|---|---|
| `outputs/source_backtrack/sources.csv` | 발생원 테이블: `source_id, source_type(livestock/factory/wastewater/other), city, name, species, head_count, area_m2, status, address, latitude, longitude, location_precision(point/village), geocode_method, emission_weight, weight_imputed`. 1차는 `livestock`만. 공장·하수처리시설은 2차에 같은 표에 추가하며 `grid_scores.csv` 스키마는 바꾸지 않는다 | **커밋 금지**(지오코딩 API 결과 저장 제한). `.gitignore`에 등록 |
| `outputs/source_backtrack/source_candidates.csv` | Event별 "현재 민원 설명력" 상위 5 발생원: `event_hour, rank, source_id, name, city, species, location_precision, distance_km, bearing_deg, travel_time_min, wind_alignment, fit_score, evidence_text` | 커밋 |
| `outputs/source_backtrack/validation.json` | 자체 검증 수치 (아래) | 커밋 |
| `outputs/source_backtrack/source_map_{season}_{daypart}.csv` | 2차. PSCF형 발생원 지도 4벌 | 커밋 |
| `docs/track_a_report.md` | 방법·상수·검증 결과·한계 한 페이지 | 커밋 |

`source_candidates.csv`의 `village` 행은 `name`을 "김제시 용지면 ○○리 축산 밀집 구역(n곳)"처럼 쓴다. 시설을 원인으로 단정하는 표현은 어디에도 쓰지 않는다.

### `validation.json` 최소 항목 (1차)

```json
{
  "random_wind_contrast": {"all": {"real_percentile": ..., "shuffled_mean": ..., "p_value": ...}, "dry": {...}, "wet": {...}},
  "direction_sensitivity": {"10": ..., "20": ..., "30": ...},
  "facility_lift_top10pct": ...,
  "geocoding": {"iksan_point_rate": ..., "gimje_point": 184, "gimje_village_rate": ...},
  "constants": {"SPECIES_WEIGHT": {...}, "L_km": {...}, "min_wind_speed": 0.5, "max_travel_hours": 3}
}
```

`random_wind_contrast.all.p_value`가 0.05를 넘으면 Track B는 역추적 점수를 모델 입력이 아닌 **참고 정보**로만 쓴다.

## 파일 소유권

| Track A (Codex) | Track B (Claude) | 공용(수정 시 양쪽 합의) |
|---|---|---|
| 신규 `build_source_backtrack.py`, `tests/test_source_backtrack.py`, `docs/track_a_report.md`, `outputs/source_backtrack/**`, `data/kakao_geocode_cache/**` | `optimize_early_prediction.py`, `compare_operational_grid_sizes.py`, `sensitivity_early_prediction.py`, 신규 `run_ablation.py`, `administrative_agent/**`, `streamlit_app.py`, `demo/**`, `PROJECT_CONTEXT.md`, `README.md` | 이 문서, `tools/validate_backtrack_output.py`, `tests/test_backtrack_contract.py`, `tests/fixtures/**`, `build_odor_ai_mvp.py`, `.gitignore` |
| `fetch_kma_weather.py`: 옵션 추가만, 기존 동작·출력 파일 불변 | | |

- 브랜치: A는 `feat/source-backtrack`, B는 `feat/integration-ablation`. main 직접 커밋 금지.
- 검증기 실패 시 **산출물을 고친다**. 검증기가 틀렸다고 판단되면 고치지 말고 근거와 함께 보고한다.

## 검증 절차

```powershell
python tools/validate_backtrack_output.py outputs/source_backtrack/grid_scores.csv
python -m unittest tests.test_backtrack_contract -v
```

기준표가 없거나 민원 원본이 바뀌면 검증기가 자동으로 다시 만든다(1~2분). 강제로 다시 만들려면 `--build-reference`.
Track B는 진짜 파일이 오기 전까지 `tests/fixtures/grid_scores_sample.csv`(실제 Event 3개, 임의 점수)로 어댑터를 개발한다.
