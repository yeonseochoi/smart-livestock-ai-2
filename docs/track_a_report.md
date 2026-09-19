# Track A 2판 결과: fixed0·6 km 기본, 모델 입력은 보류

> 확인일: 2026-09-20. 계약 CSV 스키마는 1판과 같음. 1차 ASOS 시간자료만 사용했으며 AWS 분자료와 PSCF형 prior 지도는 수행하지 않음.

## 1판에서 2판으로 바꾼 근거

Track B의 민원 전체 검정(`c602934` 및 후속 `travel_lag_result.json`)에서 민원 시각 풍향은 실제/셔플 비 1.065(+6.5%), 도달시간 `τ=d/u` 이전 풍향은 1.047(+4.7%)이었음. 거리별 lag0 신호는 0~2 km 1.161(+16.1%), 4~6 km 1.142(+14.2%)였고, travel의 8~10 km는 1.004(+0.4%, p=0.323)였음. 따라서 2판 기본값은 `--wind-lag fixed0 --max-source-km 6`으로 정함. `fixed0`은 모든 발생원–격자 쌍에 `event_hour`와 `event_hour-1h` ASOS 풍향의 풍속 가중 원형평균을 적용함. `travel`은 1판의 거리/풍속 기반 시차 선택을 유지하며 `--wind-lag travel --max-source-km 10`으로 비교함. 거리 감쇠 `L`과 도달시간 출력 방식은 유지함.

1판 `lagged_wind_alignment`는 최대 기여 발생원 값을 써 평균 약 0.84로 거의 고정됐음. 2판은 반경 안 모든 발생원의 cosine 일치도를 `emission_weight`로 가중 평균함. 기본 산출물의 평균은 0.186, 표준편차 0.373, 범위는 −0.875~0.890임. 양성/음성 격자 평균은 각각 0.200/0.184이며, 이는 분산이 생겼다는 확인이지 예측력 입증은 아님.

## 계산과 비교 검증

시설별 `사육두수 × 축종 계수`, 풍향 양의 cosine, `exp(-d/L)`을 곱해 `forward_plume_score`를 만들고, 초기 30분 민원을 설명하는 시설 적합도로 재가중해 `source_fit_score`를 만듦. 점수는 시설을 원인으로 판정하는 값이 아니라 후보 적합도임. ASOS가 있는 Event는 113개, 2019년 47개는 `weather_source=none`과 NaN을 유지함.

| 설정 | 실제 수용점 순위 | 셔플 평균 | 차이 | 경험적 p |
|---|---:|---:|---:|---:|
| travel / 10 km | 0.4639 | 0.4632 | +0.0007 | 0.406 |
| fixed0 / 6 km | 0.5374 | 0.5325 | +0.0048 | 0.109 |

각 설정에서 Event 풍향 profile을 100회 섞고 매회 후보 격자 전체 순위를 다시 계산함. 기본 설정의 건조 p=0.297, 강수 p=0.040이나 강수 Event는 11개뿐이므로 전체 결과를 뒤집는 근거로 쓰지 않음. 풍향 ±10°/±20°/±30°에서 상위 3 후보 유지율은 0.907/0.841/0.758임.

시설 겹침 lift는 0.526임. 분자는 ASOS 기간 전체 민원의 역방향 5 km 선분 통과 횟수 상위 10% 격자 안 `emission_weight` 합임. 분모는 전체 통과 격자의 시설 가중치 합에 `상위 격자 수/전체 통과 격자 수`를 곱한 균등 무작위 기대값임. 1보다 낮아 이 간접 검증도 후보 시설 적합성을 지지하지 않음.

## 발생원·상수·문헌

익산 point 지오코딩은 1,180/1,267건(93.13%), 김제는 기존 point 184건과 village 1,174/1,327건(88.47%)임. 좌표 미확보 240행은 점수에서 제외함. 축종 계수는 초기 가정이며, 메추리 1행은 사용자 승인에 따라 익산 종계/산란계 두수 중앙값을 사용함. 최소 유효 풍속 0.5 m/s, 도달시간 상한 3시간, `L`은 풍속 2 m/s 이상 1.5 km, 1~2 m/s 2.5 km, 1 m/s 미만 또는 21~06시 4 km임.

- [Jia et al., Identification of origins and influencing factors of environmental odor episodes using trajectory and proximity analyses](https://doi.org/10.1016/j.jenvman.2021.113084), 2021: 민원 위치 지오코딩, 인접도와 역궤적을 결합한 행정 악취 분석 사례임.
- [Eckmann et al., Combining Ordinary Kriging with wind directions to identify sources of industrial odors in Portland, Oregon](https://doi.org/10.1371/journal.pone.0189175), 2018: 19개 지점의 반복 악취 관측 19,665건과 5분 풍향을 결합해 방향 연관성을 확인함. 요청문에는 2017로 표기됐으나 DOI와 PubMed의 출판연도는 2018임.
- [Yu et al., Development of a Livestock Odor Dispersion Model: Part I](https://doi.org/10.3155/1047-3289.61.3.269), 2011: 시간별 기상과 복수 축산 발생원을 함께 다룬 근거임.

## 한계와 Track B 전달사항

2024~2025 시설 목록을 과거 Event에 소급 적용했고 실제 배출량·가동 이력·지형·대기 안정도·방출 높이가 없음. ASOS 시간자료로 직전 30분 풍향 표준편차를 계산할 수 없어 `stagnation_flag`는 풍속 1 m/s 미만만 반영함. 김제 village 좌표는 개별 시설 위치가 아니며 민원은 농도 관측이 아니라 시민 인지와 신고 행동의 결과임.

기본 설정이 travel/10 km보다 나았지만 전체 p=0.109로 0.05를 넘음. Track B에는 참고 정보로 전달하고, 모델 입력 채택은 보류함. 성능 실험을 하더라도 M0 고정, 학습 구간 내부검증, 특징 3~5개 제한과 소거 실험을 적용하며 개선이 없으면 M0을 유지해야 함. `prior_downwind_score`는 전부 NaN이고 `history_cutoff=event_hour`임.
