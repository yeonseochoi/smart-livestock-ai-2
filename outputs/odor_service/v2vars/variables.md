# v2 실제 모델 변수

| 변수 이름              | 코드상 계산 방법                                                | 무엇을 재는지(쉬운 말)                      | 넣은 이유(가설)               | 담당자에게 주는 의미          |
|:-----------------------|:----------------------------------------------------------------|:--------------------------------------------|:------------------------------|:------------------------------|
| initial_count          | (cnt[None, :] * own).sum(1)                                     | 해당 격자 초기 신고 수                      | 신고가 모인 위치 파악         | 우선 확인할 격자              |
| initial_intensity      | (mi[None, :] * own).sum(1)                                      | 해당 격자 신고 강도 평균                    | 신고자가 느낀 정도 반영       | 현장 확인 참고                |
| min_distance           | dist.min(1)                                                     | 가장 가까운 신고 격자까지 거리              | 신고 위치와 가까운 곳 고려    | 점검 동선 참고                |
| neighbor_count         | (cnt[None, :] * (dist <= 1.5)).sum(1)                           | 주변 1.5칸 안 신고 수                       | 주변 신고 밀집 반영           | 주변 확인 범위                |
| radius3_count          | (cnt[None, :] * (dist <= 3.0)).sum(1)                           | 주변 3칸 안 신고 수                         | 넓은 범위 신고 반영           | 주변 확인 범위                |
| prior                  | 학습 Event의 해당 격자 양성 Event 비율; 학습 행은 leave-one-out | 과거 평가 사건에서 신고된 빈도              | 반복되는 격자 반영            | 과거 발생 참고                |
| hour_sin               | sin(2*pi*frac_hour/24)                                          | 신고 시작 시각의 주기값                     | 시간대 차이 반영              | 시간대 참고                   |
| hour_cos               | cos(2*pi*frac_hour/24)                                          | 신고 시작 시각의 주기값                     | 시간대 차이 반영              | 시간대 참고                   |
| centroid_distance      | hypot(C[:, 0]-cx, C[:, 1]-cy)                                   | 신고 중심에서 거리                          | 신고 분포 중심 반영           | 중심 주변 확인                |
| nearest_intensity      | mi[dist.argmin(1)]                                              | 가장 가까운 신고 격자의 평균 강도           | 가까운 신고자의 체감 참고     | 현장 확인 참고                |
| weighted_intensity     | (w * mi[None, :]).sum(1) / w.sum(1)                             | 거리를 가중한 신고 강도                     | 주변 체감 정도 반영           | 현장 확인 참고                |
| observed_grid_count    | full(len(cands), float(len(obs)))                               | 초기 신고가 들어온 격자 수                  | 신고 장소의 넓이 반영         | 확인 범위 참고                |
| observed_report_count  | full(len(cands), float(cnt.sum()))                              | 초기 30분 전체 신고 수                      | 신고 증가 규모 참고           | 신고가 늘어나는지 확인        |
| first15_count          | (f15[None, :] * own).sum(1)                                     | 해당 격자의 앞 15분 신고 수                 | 앞뒤 신고 변화를 비교         | 냄새가 심해지는지 확인할 단서 |
| growth                 | (initial_count - first15) - first15                             | 해당 격자 뒤 15분 수에서 앞 15분 수를 뺀 값 | 냄새가 심해지는 신호인지 검정 | 현장 확인 시점 참고           |
| hist_total_3h          | ci.count_between(t0-180, t0)                                    | 시작 전 3시간 전체 신고                     | 최근 이력 반영                | 과거 신고 참고                |
| gap_before_t0_min      | min(t0-ci.t[s-1], 1440)                                         | 직전 신고 이후 분                           | 신고 공백 반영                | 과거 신고 참고                |
| hist_cell_24h          | ci.cell_count_between(cell, t0-1440, t0)                        | 지난 하루 해당 격자 신고                    | 최근 반복 반영                | 과거 신고 참고                |
| hist_cell_7d           | ci.cell_count_between(cell, t0-7*1440, t0)                      | 지난 7일 해당 격자 신고                     | 과거 반복 반영                | 과거 신고 참고                |
| hist_same_hour_rate    | n_before / elapsed_days                                         | 과거 같은 시각 신고 날짜 비율               | 시간대별 반복 반영            | 과거 신고 참고                |
| hist_same_hour_30d     | n_30d                                                           | 지난 30일 같은 시각 신고 날짜 수            | 시간대별 반복 반영            | 과거 신고 참고                |
| prev_night_cell_active | float(ci.cell_count_between(cell, prev_lo, prev_hi)>0)          | 전날 같은 시각 해당 격자 신고 여부          | 전날 반복 반영                | 과거 신고 참고                |
| prev_night_total       | ci.count_between(prev_lo, prev_hi)                              | 전날 같은 시각 전체 신고 수                 | 전날 상황 반영                | 과거 신고 참고                |
