# 익산시 악취 민원 선제 대응 AI

과거 악취 민원의 시공간 패턴을 학습해 현재 Event 이후 30분 안에 추가 민원이 접수될 가능성이 높은 1km 권역 Top 3를 제시하고, 담당자 검토용 행정 대응 문서를 생성하는 프로젝트입니다.

이 결과는 악취 농도나 실제 배출 확률이 아니라 **민원 접수 위험의 상대적 우선순위**입니다. 원인 시설을 판정하거나 행정조치를 자동 결정하지 않습니다.

## 주요 기능

- 초기 30분 민원을 바탕으로 이후 30분의 추가 민원 위험 권역 예측
- 1km·1.5km·2km 격자 모델 비교와 시간순 검증
- 운영 단위인 1km 격자의 Top 3 권역 및 상대 위험도 제공
- 악취 민원 상황 브리핑, 현장 점검 지시서, 사후 결과보고서 생성
- 브라우저 기반 운영 데모와 현장 결과 입력
- API 키가 없는 경우 검증 가능한 안전 템플릿 사용
- Gemini 또는 OpenAI 연동 시 문안 보정 지원

## 검증 결과

민원 Event를 시간순으로 70% 학습, 30% 테스트로 분리했습니다. 운영 기준인 1km 격자의 테스트 결과는 다음과 같습니다.

|지표|결과|
|---|---:|
|PR-AUC|0.472|
|ROC-AUC|0.908|
|Top-K Recall|0.494|
|Recall@3|0.444|
|Event Hit@3|0.830|

Event Hit@3는 추가 민원이 실제로 존재하는 평가 가능 Event 가운데, 예측 상위 3개 권역 중 하나 이상에 추가 민원이 포함된 비율입니다. 테스트 Event 수는 48개이므로 추가적인 연도별·현장 검증이 필요합니다.

상세 결과는 `outputs/operational_grid_comparison/metrics.json`에서 확인할 수 있습니다.

## 데모 화면

> 아래 파일명으로 화면을 캡처해 `docs/screenshots/` 폴더에 추가한 뒤, 아래 주석 안의 마크업을 README 본문으로 옮기면 됩니다.
>
> - `demo-overview.png`: 전체 화면과 지도
> - `demo-weather.png`: 현장 참고 기상정보
> - `demo-documents.png`: 상황 브리핑·현장점검 지시서
> - `demo-followup.png`: 사후 결과보고서

<!-- 스크린샷을 추가한 뒤 이 줄과 아래의 주석 종료 줄을 삭제하세요.

### 민원 확산 예측 대시보드

<p align="center">
  <img src="docs/screenshots/demo-overview.png" alt="익산시 악취 민원 선제 대응 AI 데모 전체 화면" width="920">
</p>

초기 30분 민원과 AI가 예측한 향후 30분의 우선 확인 권역 Top 3를 지도에서 한눈에 확인할 수 있습니다.

### 주요 운영 화면

<table>
  <tr>
    <td width="50%" align="center">
      <img src="docs/screenshots/demo-weather.png" alt="현장 참고 기상정보" width="100%"><br>
      <b>현장 참고 기상정보</b><br>
      풍속·풍향·상대습도·최근 강수 확인
    </td>
    <td width="50%" align="center">
      <img src="docs/screenshots/demo-documents.png" alt="행정 대응 문서 생성 화면" width="100%"><br>
      <b>행정 대응 문서</b><br>
      상황 브리핑과 현장점검 지시서 생성
    </td>
  </tr>
  <tr>
    <td colspan="2" align="center">
      <img src="docs/screenshots/demo-followup.png" alt="사후 결과보고서 작성 화면" width="80%"><br>
      <b>사후 결과보고서</b><br>
      AI 예측과 실제 출동·측정·조치 결과를 Event ID 기준으로 기록
    </td>
  </tr>
</table>

-->

## 빠른 실행

### Streamlit 데모

```powershell
.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

브라우저에서 표시되는 로컬 주소를 열면 기존 데모와 동일한 Event 이동, 초기 민원·위험 격자 지도, 실제 이후 신고 표시, 기상정보, 대응 문서 생성과 현장 결과보고서 다운로드 기능을 사용할 수 있습니다.

Streamlit Community Cloud에서는 GitHub 저장소를 연결하고 실행 파일을 `streamlit_app.py`로 지정합니다. LLM을 연동하려면 App settings의 Secrets에 [.streamlit/secrets.toml.example](.streamlit/secrets.toml.example)과 같은 형식으로 키를 등록합니다. 키가 없으면 안전 템플릿 모드로 실행됩니다.

### Windows 데모

Python 3.12가 설치된 환경에서 다음 중 하나를 실행합니다.

- `run_demo_template.bat`: 외부 API 호출 없는 안전 템플릿 모드
- `run_demo.bat`: `.env` 설정에 따른 Gemini/OpenAI 연동 모드

실행 후 브라우저에서 <http://127.0.0.1:8765>를 엽니다.

기존 `.venv`가 다른 위치의 Python을 참조해 깨진 경우에는 먼저 해당 가상환경을 제거한 뒤 배치 파일을 실행하거나 아래 수동 설치 절차로 다시 생성해야 합니다.

### 수동 설치 및 실행

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe demo\server.py --port 8765
```

## LLM 설정

`.env.example`을 `.env`로 복사한 뒤 사용할 공급자의 값을 설정합니다. 실제 API 키는 커밋하지 마세요.

Gemini 예시:

```dotenv
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.6-flash
```

OpenAI 예시:

```dotenv
LLM_PROVIDER=openai
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=your_supported_model_id
```

키가 없거나 LLM 호출이 실패하면 안전 템플릿으로 자동 대체됩니다. API 키는 브라우저 화면이나 `demo/index.html`에 입력하지 않습니다.

## 예측 및 문서 재생성

모델 비교 결과를 다시 생성하려면:

```powershell
.venv\Scripts\python.exe compare_operational_grid_sizes.py
```

예측 CSV에서 시간상 최신 Event의 행정 문서를 생성하려면:

```powershell
.venv\Scripts\python.exe generate_agent_documents.py
```

특정 Event를 지정하려면:

```powershell
.venv\Scripts\python.exe generate_agent_documents.py --event-id EVT-0175
```

생성 결과는 `outputs/administrative_agent/`에 저장됩니다.

- `complaint_briefing.md`: 악취 민원 상황 브리핑
- `field_inspection_order.md`: 현장 점검 지시서
- `followup_report_template.md`: 사후 결과 입력 템플릿
- `agent_output.json`: 서비스 연동용 구조화 결과

## 프로젝트 구조

```text
administrative_agent/              행정 대응 문서 모델·템플릿·LLM 연동
data/                              원본 및 가공 데이터
demo/                              브라우저 데모와 로컬 API 서버
outputs/                           모델 평가 및 생성 문서
tests/                             행정 대응 Agent 단위 테스트
build_odor_ai_mvp.py               민원 전처리와 Event 구성 공용 함수
compare_operational_grid_sizes.py  격자 크기 비교와 최종 성능 재현
generate_agent_documents.py        예측 결과 기반 문서 생성
optimize_early_prediction.py       후보 모델 학습·평가
sensitivity_early_prediction.py    격자별 학습 데이터 구성
fetch_kma_weather.py               현장 참고용 기상자료 수집
```

예측 엔진과 행정 대응 Agent는 분리되어 있습니다. Agent는 예측 엔진의 표준 출력만 사용하며 민원 원본, 농가, 센서 데이터를 직접 조회하거나 원인 시설을 추론하지 않습니다.

## 테스트

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## 데이터와 한계

- 원본 데이터: `data/익산시 악취 민원 데이터_20190528-20260818.xlsx`
- 실제 민원 시각 범위: 2019-05-28 23:23 ~ 2026-08-17 22:54
- 전체 16,113건 중 익산시 민원 14,994건
- 공통 Event 160개, 시간순 학습 112개 / 테스트 48개
- 시민의 인지와 신고 행동이 반영된 데이터이므로 실제 악취 영향권과 같지 않습니다.
- 장기간 센서 시계열, 현장 출동 결과, 배출원 정보가 없어 악취 발생 자체나 대응 효과를 학습하지 못했습니다.
- 기상정보는 현재 최종 예측 점수에 사용하지 않으며 담당자의 현장 판단을 위한 참고 정보로만 제공합니다.

프로젝트의 상세 설계는 [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md)를 참고하세요.
