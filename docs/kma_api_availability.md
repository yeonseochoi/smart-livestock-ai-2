# 기상청 API 허브 가용성 확인 (대기안정도·현장 바람 재료)

확인일 2026-09-20. 인증키는 `.env`의 `KMA_API_KEY`만 사용하며 이 문서와 코드에 값을 적지 않는다. 각 항목은 1회 호출로 확인했다.

## 결론

| 재료 | 상태 | 비고 |
|---|---|---|
| ASOS 시간자료(전주 146·군산 140) | 사용 중 | `outputs/weather_integration/asos_hourly_2020_2026.csv`, 2020-01~2026-08 |
| ASOS 안정도 산정 재료(운량·일사·지면온도) | 수집 완료 | `fetch_kma_weather.py --stability-only` → `asos_hourly_stability_2020_2026.csv` 117,607행, Pasquill-Gifford A~F 산정(결측 441행) |
| AWS 매분자료(익산 702 등) | 호출 가능 | `typ01/cgi-bin/url/nph-aws2_min`, 30분 요청 시 30행(풍향·풍속·기온·강수·습도) 확인 |
| LDAPS(국지예보모델 1.5 km) 격자 바람 | **활용신청 필요** | `typ06/url/nwp_vars_down.php`(nwp=l015) 호출 시 HTTP 403 "활용신청이 필요한 API". API 허브에서 신청 후 재시도 |

## AWS 지점 (익산시청 기준 거리)

| 지점번호 | 이름 | 거리 |
|---:|---|---:|
| 702 | 익산 | 3.3 km |
| 733 | 함라 | 12.4 km |
| 763 | 여산 | 15.5 km |
| 737 | 김제 | 17.1 km |
| 146 | 전주(ASOS) | 18.6 km |
| 140 | 군산(ASOS) | 18.8 km |
| 736 | 진봉 | 19.3 km |

현재 역추적·발생 위험 모델은 ASOS 두 지점(18~19 km) 바람을 쓴다. 익산 702(3.3 km)·함라·여산을 쓰면 시내와 북부 읍면의 국지 바람을 더 가깝게 반영할 수 있다. 다만 AWS 매분자료는 기간 일괄 내려받기 형태가 아니라 구간 요청이므로, 2020~2026 전 기간 시간자료를 만들려면 지점당 약 5.8만 시간 구간을 나누어 호출해야 한다(2차 작업).

## 대기안정도

ASOS 시간자료의 총운량·일조·일사·지면온도로 Pasquill-Gifford 안정도 등급(A 매우 불안정 ~ F 안정)을 산정했다(`fetch_kma_weather.parse_asos_stability`, `pasquill_gifford_class`). 등급 분포(2020~2026, 두 지점 합): E 35,588 / B 29,305 / F 22,837 / C 17,457 / D 8,297 / A 3,682.
민원 풍향–발생원 연관 검정을 안정도 층별로 나눈 결과와 발생 위험 모델에 안정도를 넣은 결과는 Track B `outputs/wind_lag_sweep/association_sweep.md`, `outputs/onset_risk/onset_risk_table.md`에 있다.

## 확인에 쓴 호출

- 지점 목록: `typ01/url/stn_inf.php?inf=AWS&tm=202507010000`
- AWS 매분: `typ01/cgi-bin/url/nph-aws2_min?tm1=202608151400&tm2=202608151429&stn=702&disp=1`
- LDAPS: `typ06/url/nwp_vars_down.php?nwp=l015&sub=unis&vars=ugrd&tmfc=2026081500&ef=3&dataType=GRIB` → 403

참고: 공식 문서상 LDAPS 산출물은 GRIB2 형식이며 지점 추출 API는 확인되지 않았다(격자 파일을 받아 직접 잘라야 함). 출처: 기상기후데이터위키 통합모델(UM) 항목, API 허브 수치모델 경량화 다운로드.
