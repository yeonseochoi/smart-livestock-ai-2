<div align="center">

# 🐄 익산시 악취 민원 선제 대응 AI

**민원 확산을 예측하고 현장 대응 문서를 생성하는 AI 기반 의사결정 지원 서비스**

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.62-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.x-EB5B25?style=flat-square)](https://xgboost.ai/)

[🚀 데모 실행하기](https://smart-livestock-ai-2.streamlit.app/) · [▶️ 데모 영상 보기](https://www.youtube.com/watch?v=ZC-iM7M16fY) · [📘 프로젝트 상세 문서](PROJECT_CONTEXT.md)

</div>

---

## 📋 목차

1. [프로젝트에 대한 정보](#1-프로젝트에-대한-정보)
2. [시작 가이드](#2-시작-가이드)
3. [기술 스택](#3-기술-스택)
4. [주요 기능](#4-주요-기능)
5. [모델 성능](#5-모델-성능)
6. [프로젝트 구조](#6-프로젝트-구조)
7. [데이터와 활용 한계](#7-데이터와-활용-한계)

---

<a id="1-프로젝트에-대한-정보"></a>
## 1. 📌 프로젝트에 대한 정보

### 프로젝트 소개

과거 악취 민원의 시공간 패턴을 학습해 현재 Event 이후 30분 안에 **추가 민원이 접수될 가능성이 높은 1km 권역 Top 3**를 제시합니다. 예측 결과를 바탕으로 담당자 검토용 상황 브리핑, 현장점검 지시서 및 사후 결과보고서를 생성합니다.

> 예측 결과는 악취 농도나 실제 배출 확률이 아닌 **민원 접수 위험의 상대적 우선순위**입니다. 특정 시설을 원인으로 판정하거나 행정조치를 자동으로 결정하지 않습니다.

| 구분 | 내용 |
|---|---|
| 프로젝트명 | 익산시 악취 민원 선제 대응 AI |
| 소속 | 인공지능 및 빅데이터 연합동아리 BITAmin |
| 대상 지역 | 전북특별자치도 익산시 |
| 예측 목표 | 초기 30분 이후, 향후 30분의 추가 민원 위험 권역 Top 3 |
| 운영 단위 | 1km 격자 |
| 배포 주소 | [Streamlit 데모](https://smart-livestock-ai-2.streamlit.app/) |
| 저장소 | [GitHub Repository](https://github.com/yeonseochoi/smart-livestock-ai-2) |

### 🖥️ 데모 메인 화면

<img width="1635" height="907" alt="image" src="https://github.com/user-attachments/assets/8cdf192e-f3fd-4a29-bf2c-09e1b032237c" />

### ▶️ 데모 영상

아래 이미지를 클릭하면 YouTube 데모 영상으로 이동합니다.

<p align="center">
  <a href="https://www.youtube.com/watch?v=ZC-iM7M16fY">
    <img src="https://img.youtube.com/vi/ZC-iM7M16fY/hqdefault.jpg" alt="익산시 악취 민원 선제 대응 AI 데모 영상" width="720">
  </a>
</p>

<p align="center"><a href="https://www.youtube.com/watch?v=ZC-iM7M16fY"><b>유튜브에서 데모 영상 시청하기</b></a></p>

---

<a id="2-시작-가이드"></a>
## 2. 🚀 시작 가이드

### 요구 사항

- Python 3.12
- Windows PowerShell 또는 호환 터미널
- 선택 사항: Gemini 또는 OpenAI API 키

### Streamlit 데모 실행

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

터미널에 표시되는 로컬 주소를 브라우저에서 열면 됩니다. API 키가 없어도 안전 템플릿 모드로 모든 핵심 기능을 확인할 수 있습니다.

### Windows 데모 실행

- `run_demo_template.bat`: 외부 API를 호출하지 않는 안전 템플릿 모드
- `run_demo.bat`: `.env` 설정에 따른 Gemini/OpenAI 연동 모드

실행 후 브라우저에서 <http://127.0.0.1:8765>를 엽니다.

### LLM 연동

`.env.example`을 `.env`로 복사한 뒤 사용할 공급자의 값을 설정합니다. **실제 API 키는 GitHub에 커밋하지 마세요.**

```dotenv
# Gemini
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.6-flash

# OpenAI를 사용할 경우 위 설정 대신 사용
# LLM_PROVIDER=openai
# OPENAI_API_KEY=your_openai_api_key_here
# OPENAI_MODEL=your_supported_model_id
```

Streamlit Community Cloud에서는 실행 파일을 `streamlit_app.py`로 지정하고, App settings의 Secrets에 [.streamlit/secrets.toml.example](.streamlit/secrets.toml.example)과 같은 형식으로 키를 등록합니다.

---

<a id="3-기술-스택"></a>
## 3. ✨ 기술 스택

### AI · Data

<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas">
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/XGBoost-EB5B25?style=for-the-badge" alt="XGBoost">
</p>

### Demo · Map · LLM

<p>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Folium-77B829?style=for-the-badge&logo=leaflet&logoColor=white" alt="Folium">
  <img src="https://img.shields.io/badge/Gemini-8E75B2?style=for-the-badge&logo=googlegemini&logoColor=white" alt="Gemini">
  <img src="https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white" alt="OpenAI">
</p>

---

<a id="4-주요-기능"></a>
## 4. 📍 주요 기능

- **민원 확산 예측** — 초기 30분 민원을 바탕으로 향후 30분의 추가 민원 위험 권역을 예측합니다.
- **운영 격자 선정** — 1km·1.5km·2km 격자 모델을 비교하고 시간순으로 검증합니다.
- **Top 3 우선순위** — 1km 격자의 상위 3개 권역과 상대 위험도를 지도에 표시합니다.
- **현장 참고정보** — 풍속·풍향·습도·강수 정보를 담당자의 현장 판단 자료로 제공합니다.
- **행정 대응 Agent** — 상황 브리핑과 현장점검 지시서 초안, Event별 AI 대응 참고 가이드를 생성합니다.
- **사후 결과 기록** — 출동·측정·조치 결과를 직접 입력하고 완성된 결과보고서를 내려받습니다.
- **안전한 문서 생성** — API 키가 없거나 LLM 호출이 실패하면 검증 가능한 안전 템플릿을 사용합니다.

---

<a id="5-모델-성능"></a>
## 5. 📊 모델 성능

민원 Event를 시간순으로 70% 학습, 30% 테스트로 분리했습니다. 운영 기준인 1km 격자의 테스트 결과입니다.

| 평가 지표 | 결과 |
|---|---:|
| ROC-AUC | 0.908 |
| PR-AUC | 0.486 |
| Top-K Recall | 0.507 |
| Recall@3 | 0.443 |
| Event Hit@3 | 85.1% |
| 실제 추가 민원 격자 비율 | 약 10.7% |
| Event당 평균 후보 권역 | 약 31개 |
| 선택 모델 | XGBoost 분류모델 `xgb_d3` |
| 운영 출력 | 상위 3개 권역 |

`Event Hit@3`는 추가 민원이 존재하는 평가 Event 중 예측 상위 3개 권역 하나 이상에 실제 추가 민원이 포함된 비율입니다. 테스트 Event 48개 가운데 추가 민원이 실제 발생해 평가가 가능한 Event는 47개이며, 그중 40개에서 상위 3개 권역이 적중했습니다. 표본이 47개이므로 이 비율의 95% 신뢰구간은 대략 75~95%입니다. 연도별·현장 검증이 추가로 필요합니다.

후보 모델은 학습 구간 안에서만 선택했습니다. 학습 112개 Event의 앞 80%로 학습하고 뒤 20%를 내부 검증에 사용해 5개 후보 중 하나를 고른 뒤, 최종 테스트 48개 Event는 한 번만 사용했습니다. 모델 출력은 절대확률이 아니라 하나의 Event 안에서 후보 권역 간 상대적 우선순위로 활용합니다.

> 재현 참고자료: [실험 코드](compare_operational_grid_sizes.py) · [결과 산출물](outputs/operational_grid_comparison/metrics.json)

### 발생 위험 예보 (1단, 참고 정보)

위 확산 예측은 민원이 30분 쌓인 뒤 "어디로 번지나"를 맞힙니다. 그 앞단으로, 민원이 아직 없는 시각에 "다음 1시간 안에 어느 1km 격자에서 민원이 생길까"를 내는 발생 위험 예보를 두었습니다. 발생원(축산 시설 배출 노출)과 바람(ASOS+AWS 격자별 보간)은 여기서 쓰입니다. 확산 예측에 넣었을 때는 개선이 없었기 때문입니다(이후 30분 민원의 89%가 2km 안에서 발생).

평가는 직전 3시간 시 전체에 민원이 없던 '조용한 시각'에 위험 상위 5 격자를 찍어 실제 민원 격자를 맞힌 비율(Hit@5, 2025-01~2026-07 테스트, seed 평균)입니다.

| 특징 구성 (바람 ASOS+AWS 격자별 보간) | 조용한 시각 Hit@5 |
|---|---:|
| R0 시간 + 격자 과거 빈도 | 0.546 |
| R1 + 기상 | 0.548 |
| R2 + 축산 발생원 노출 | 0.590 |
| R5 + 격자×풍향별 과거 민원율(채택) | **0.599** |

바람을 ASOS만 쓰면 R5는 0.578(격자별)~0.581(중심점)이라 AWS 추가가 +0.02입니다(`outputs/onset_risk/experiment_summary.md`). 냄새 유형별로 라벨을 나누면 가축 0.492 → 0.588, 공장 0.644 → 0.747, 하수 0.648 → 0.685(n 작음)입니다. 공장·하수처리·산업단지 등 외부 시설 데이터는 검토했으나 개선이 없어 채택하지 않았습니다. 186개 격자 중 5개를 찍는 문제이며 과거 빈도만으로도 54%가 나오므로, 발생원·바람의 기여는 그 위 +4~6%p(공장 냄새 +10%p)입니다.

행정 문서와 화면에는 민원 접수 1시간 전 시점의 위험 상위 격자가 '사전 경보 (참고)' 절로 붙습니다. 확률이 아닌 상대값이며 원인 시설을 단정하지 않습니다.

두 모델을 합쳐 권역을 1곳으로 줄이는 방식도 검토했습니다. 확산 예측 Top 3 중 발생 위험 순위가 가장 높은 격자를 1순위로 올리는 후처리인데, 테스트 Event 47개에서 Hit@1은 확산 예측 단독 0.553과 같았습니다(고친 Event 5, 망친 Event 5). 확산 예측 후보는 첫 민원 인근이라 발생 위험 예보에서도 모두 상위에 있어 그중 하나를 가를 정보가 없기 때문입니다. 권역 1곳 추천은 채택하지 않았습니다(`fuse_onset_spread.py`, `outputs/onset_spread_fusion/fusion_table.md`). 반대로 발생 위험 순위 30위 이내 격자로 확산 예측 후보를 미리 좁히는 규칙은 후보를 평균 31개에서 15개로 줄이면서 Hit@1/2/3이 그대로였습니다(`fuse_onset_spread.py --narrow 30`, `narrow_table.md`). 이 규칙은 문서 생성과 화면에 기본으로 적용됩니다(`administrative_agent/policy.py` `narrow_candidates`). 성능 향상이 아니라 후보 수 축소 규칙이며, 1단 산출물이 없으면 후보 전체에서 고릅니다.

> 재현 참고자료: [run_onset_risk.py](run_onset_risk.py) · [결과 표](outputs/onset_risk/onset_risk_table.md) · [바람 자료원 비교](outputs/wind_lag_sweep/wind_source_compare.md) · [발생원·바람 검정 요약](PROJECT_CONTEXT.md)

---

<a id="6-프로젝트-구조"></a>
## 6. 🗂️ 프로젝트 구조

```text
[서비스]
streamlit_app.py                   Streamlit 화면 (Top 3 + 사전 경보 카드)
generate_agent_documents.py        2단 후보를 1단 상위 30으로 좁혀 Top 3 → 행정 대응 문서 4종 + agent_output.json
administrative_agent/              문서 모델(models)·rule-base 대응 단계(policy)·문서 생성(documents)·LLM 연동(llm)
demo/                              브라우저 데모(index.html, demo-data.js)와 로컬 API 서버(server.py)

[2단 확산 예측: 민원 30분 뒤 어디로 번지나]
analyze_spatiotemporal_complaints.py  민원 엑셀 읽기·열 매핑·익산 필터 (모든 스크립트의 입력 단계)
build_odor_ai_mvp.py               Event 구성·격자·후보 격자 특징 공용 함수
compare_operational_grid_sizes.py  1km·1.5km·2km 공통 검증과 최종 성능 재현 (운영 모델 xgb_d3)
optimize_early_prediction.py       후보 모델 학습·평가 라이브러리
sensitivity_early_prediction.py    격자·시간창 후보 비교(설계 근거, 제품 입력 아님)
run_ablation.py                    기상·발생원 역추적을 넣는 소거 실험 M0~M4 (개선 없음 → 미채택)

[1단 발생 위험 예보: 민원 없는 지금, 다음 1시간 어디서 나나]
run_onset_risk.py                  1단 학습·평가(R0·R1·R2·R5·R5o)와 onset_alerts*.csv(시각별 상위 30)·onset_cells.csv 생성
wind_sources.py                    ASOS+AWS 시간자료 통합, 격자별 역거리 가중(IDW) 바람
species_weight_sets.py             축종 배출계수 세트(EMEP/EEA NH3 등)
fetch_kma_weather.py               기상청 API 허브 ASOS 수집(+대기안정도 재료, Track A 옵션)
fuse_onset_spread.py               1단·2단 결합 평가: 권역 1곳 재정렬(효과 없음), 후보 축소 규칙 검증(--narrow 30, 채택)

[근거 검정: 바람·발생원 연관 (결론은 outputs/wind_lag_sweep, outputs/wind_source_association)]
test_wind_source_association.py    민원 시각 풍향 ↔ 축산 발생원 층화 셔플 검정, --travel-lag 로 거리별 시차 가설
wind_window_sweep.py               참조 바람 창 24조합(시차×평균 창×가중) 비교
minute_wind_test.py                익산 AWS 분 자료로 신고 직전 창 19종 비교
compare_wind_sources.py            바람 자료원 5종(ASOS 중심/격자별, AWS, 결합) 비교
sensitivity_species_weights.py     배출계수 세트 4벌 × 반경 2종 민감도
test_factory_source_filters.py     대기배출시설 업종 필터 전후 연관 (무효)

[Track A 발생원 역추적 (Codex 담당, 참고 정보)]
build_source_backtrack.py          Event별 발생원 후보·격자 점수 (outputs/source_backtrack)
extend_source_catalog.py           축산 외 시설 목록 확장,  analyze_source_type_association.py  유형별 연관 분석
tools/                             역추적 산출물 검증기, 산단 경계·공장 목록 수집
docs/contracts/                    Track A/B 인터페이스 계약

[문서·데이터·테스트]
작업 최종 아키텍쳐.md              전체 구조 설명(입력→2단·1단→결합→Agent), 멘토 피드백 대응표
PROJECT_CONTEXT.md                 현재 상태·성능·한계 (기획 기준 문서)
docs/kma_api_availability.md       기상청 API 실시간 조건 확인
data/                              민원 원본, 축산농가 현황, AWS 파일셋, 외부 시설 캐시
outputs/                           실험 산출물(metrics.json 등)과 생성 문서
tests/                             행정 Agent·Streamlit·소거 실험·계약 테스트 (35건)
```

### 예측 및 문서 재생성

```powershell
# 모델 비교 결과 재생성
.venv\Scripts\python.exe compare_operational_grid_sizes.py

# 최신 Event 행정문서 생성
.venv\Scripts\python.exe generate_agent_documents.py

# 특정 Event 행정문서 생성
.venv\Scripts\python.exe generate_agent_documents.py --event-id EVT-0175

# 발생 위험 예보(1단) 재실행: 전체(R0·R1·R2·R5·R5o) + 유형별
python run_onset_risk.py
python run_onset_risk.py --label-type factory --tag factory --arms R0,R2,R5

# 1단·2단 결합(권역 1곳) 평가: 연도별 분할로 1단 점수를 만든 뒤 결합
python run_onset_risk.py --train-end 2021-01-01 --arms R5 --tag fold2021   # 2022, 2023, 2024도 같은 방식
python fuse_onset_spread.py

# 전체 테스트
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

생성된 문서는 `outputs/administrative_agent/`에 저장됩니다.

---

<a id="7-데이터와-활용-한계"></a>
## 7. ⚠️ 데이터와 활용 한계

- 데이터: `data/익산시 악취 민원 데이터_20190528-20260818.xlsx`
- 실제 민원 시각 범위: 2019-05-28 23:23 ~ 2026-08-17 22:54
- 전체 16,113건 중 익산시 민원 14,994건
- 공통 Event 160개, 시간순 학습 112개 / 테스트 48개
- 신고 위치는 신고 행동이 반영된 데이터이며 실제 악취 영향권과 같지 않습니다.
- 장기간의 현장 출동 결과와 배출원 정보가 없어 악취 발생 자체나 원인 시설을 학습하지 않았습니다.
- 기상정보는 확산 예측(Top 3) 점수에는 사용하지 않습니다. 발생 위험 예보(사전 경보)와 발생원 후보 계산에만 사용합니다.
- 기상은 ASOS 전주·군산(18~19km)과 AWS 익산·함라·여산·김제·진봉(3~19km) 시간자료를 거리 가중으로 합친 값입니다. 발생 위험 예보는 테스트 구간 사후 계산이며, 실시간 운영에는 기상 실시간 호출과 매시간 추론 경로가 추가로 필요합니다.

예측 엔진과 행정 대응 Agent는 분리되어 있습니다. Agent는 예측 엔진이 전달한 Top 3와 확인 가능한 상황정보만 사용하며, 민원 원본이나 시설 데이터를 직접 조회하거나 원인 시설을 추론하지 않습니다.

더 자세한 설계와 의사결정 근거는 [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md)를 확인하세요.
