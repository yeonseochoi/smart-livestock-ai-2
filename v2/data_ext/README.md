# 외부 데이터 수집 도구

이 폴더는 v2의 기존 파일을 바꾸지 않고 축산농가 지오코딩과 AWS 분자료 수집을 수행한다. 비밀키는 환경변수에서만 읽으며 캐시·로그·출력에는 기록하지 않는다.

## 축산농가

```powershell
$env:KAKAO_REST_KEY = "..." # 현재 PowerShell 세션에서만 설정
..\.venv\Scripts\python.exe geocode_farms.py
```

- 입력: `전북특별자치도 익산시_축산농가 현황_20241231.csv`
- 출력: `farms_geocoded.csv`, `farms_geocode_report.md`, `failed_addresses.csv`
- 재개: `cache/kakao_address_cache.json`을 보존한 채 같은 명령을 다시 실행한다.

## AWS 분자료

```powershell
$env:KMA_API_KEY = "..." # 현재 PowerShell 세션에서만 설정
..\.venv\Scripts\python.exe fetch_aws_minute.py --preflight
..\.venv\Scripts\python.exe fetch_aws_minute.py --events ..\outputs\trigger_k7\events.csv
```

`--preflight`는 먼저 공식 API의 `help=1` 응답을 캐시하고, AWS 관측소·표본일 가용성을 검사한다. 이후 실행은 Event 예측시점 `T` 직전 60분만 출력하며 `관측시각 < T`를 assert한다.

공식 명세: [AWS 지점정보](https://apihub.kma.go.kr/apiList.do?apiMov=%EC%A7%80%EC%83%81%EA%B4%80%EC%B8%A1+%EC%A7%80%EC%A0%90%EC%A0%95%EB%B3%B4+%EC%A1%B0%ED%9A%8C&seqApi=2&seqApiSub=317), [AWS 매분자료](https://apihub.kma.go.kr/apiList.do?seqApi=2&seqApiSub=239). API 허브 기본 한도는 일 2,000회·5GB이나 계정별 잔여량은 마이페이지에서 확인한다.
