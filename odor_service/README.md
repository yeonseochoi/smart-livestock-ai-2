# 익산 악취 대응 서비스 코드

운영 흐름에 필요한 코드를 한 패키지에 모았다.

```text
odor_service/
├─ forecast/             오늘 밤 민원 위험 예측
├─ standby/              현장 인력 대기 장소 추천
├─ region_prediction/    민원 집중 감지와 다음 30분 예상 권역 Top 3
├─ field/                방문 농가 후보 점수와 설명
├─ event_link/           각 단계의 결과를 하루 단위 재생 자료로 연결
├─ data/                 공통 농가·기상 테이블 생성
├─ common.py             시간·거리·풍향 및 원본 경로
└─ app.py                독립형 과거 재생 화면
```

메인 화면은 프로젝트 루트의 `streamlit_app.py`를 실행한다. 원본 농가 자료는
코드와 분리해 `data/farms/iksan`, `data/farms/gimje`에 둔다. 계산 결과는
`outputs/odor_service`에 저장한다.

`region_prediction`은 과거 `v2` 폴더였던 권역 예측 엔진이다. `v2`는 모델의
버전 설명에만 사용하고 더 이상 프로젝트 최상위 폴더명으로 사용하지 않는다.

