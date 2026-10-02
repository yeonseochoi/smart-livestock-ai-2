<div align="center">

# 🐄 익산시 악취 민원 선제 대응 AI

**축산 악취 민원·기상·농가 정보를 연계한 AI 기반 현장대응 의사결정 지원 시스템**

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.62-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.x-EB5B25?style=flat-square)](https://xgboost.ai/)

[🚀 데모 실행하기](https://smart-livestock-ai-2.streamlit.app/) · [📘 프로젝트 상세 문서](PROJECT_CONTEXT.md)

</div>

---

## 📋 목차

1. [프로젝트에 대한 정보](#1-프로젝트에-대한-정보)
2. [기술 스택](#3-기술-스택)
3. [주요 프로세스](#4-주요-기능)
4. [모델 성능](#5-모델-성능)
5. [프로젝트 구조](#6-프로젝트-구조)
6. [데이터와 활용 한계](#7-데이터와-활용-한계)

---

<a id="1-프로젝트에-대한-정보"></a>
## 1. 📌 프로젝트에 대한 정보

### 프로젝트 소개

이 서비스는 하나의 사건을 두 단계로 나누어 지원합니다. 먼저 가축 관련 악취 민원의 시공간 패턴으로 이후 민원이 발생할 가능성이 높은 권역을 예측하고, 이어서 초기 민원 위치와 풍향을 해석해 현장에서 확인할 농가 후보를 제시합니다. 결과를 바탕으로 담당자 검토용 상황 브리핑·출동 지시서·사후 결과보고서도 생성합니다.

### 2단계 예측 흐름

| 단계 | 입력 | 결과 | 역할 |
|---|---|---|---|
| 1단계 · 이후 민원 발생 예측 | 초기 30분의 가축 악취 민원과 과거 시공간 패턴 | 이후 30분 추가 민원 위험 1km 권역 Top 3 | 추가 민원이 발생할 가능성이 높은 지역을 넓게 살펴보기 위한 정보 |
| 2단계 · 점검 후보 우선순위 | 같은 사건 초기 30분의 실제 민원 위치와 사건 시각 풍향 | 현장 확인 농가 후보 최대 3개 | 그 지역 안팎에서 실제로 먼저 방문해 확인할 시설을 정하기 위한 정보 |

> 2단계 농가 후보는 1단계 예측 권역 Top 3에서 다시 고르는 방식이 아닙니다. 해당 사건의 실제 초기 민원 위치와 풍향을 별도로 사용합니다.

> 권역 예측은 악취 농도나 실제 배출 확률이 아닌 **민원 접수 위험의 상대적 우선순위**입니다. 농가 후보 역시 현장 확인 참고 결과이며, 특정 시설을 발생원으로 확정하거나 행정조치를 자동으로 결정하지 않습니다.

| 구분 | 내용 |
|---|---|
| 프로젝트명 | 익산시 악취 민원 선제 대응 AI |
| 소속 | 인공지능 및 빅데이터 연합동아리 BITAmin |
| 대상 지역 | 전북특별자치도 익산시 |
| 악취 민원 예측 | 초기 30분 이후, 향후 30분의 추가 가축 악취 민원 위험 권역 Top 3 |
| 점검 후보 제안 | 초기 30분 민원 위치·풍향·과거 반복 이력 기반 후보 최대 3개 |

### 🖥️ 데모 메인 화면

<img width="1879" height="893" alt="image" src="https://github.com/user-attachments/assets/9c749776-5781-4e51-be4a-1ca98454e80e" />


### ▶️ 데모 영상

아래 이미지를 클릭하면 YouTube 데모 영상으로 이동합니다.

<p align="center">
  <a href="https://www.youtube.com/watch?v=msrWObjSZks">
    <img src="https://img.youtube.com/vi/msrWObjSZks/hqdefault.jpg" alt="익산시 악취 민원 선제 대응 AI 데모 영상" width="720">
  </a>
</p>

<p align="center"><a href="[https://www.youtube.com/watch?v=ZC-iM7M16fY](https://www.youtube.com/watch?v=msrWObjSZks)"><b>유튜브에서 데모 영상 시청하기</b></a></p>

### 🧮 농가 점검 후보 점수 (100점)

| 항목 | 배점 | 의미 |
|---|---:|---|
| 풍향 일치 | 20점 | 후보 방향과 당시 바람 방향의 일치 정도 |
| 민원 지점 거리 | 20점 | 가장 가까운 민원 위치와의 거리 |
| 민원 중첩·공통 지목 | **40점** | 여러 민원 위치가 같은 후보를 가리키는 정도 |
| 과거 반복 이력 | 20점 | 같은 월·유사 풍향에서의 과거 반복 민원 |
| **합계** | **100점** | 현장 점검 우선순위 참고 점수 |

3번 공통 지목은 사전에 정해진 수치가 아니라 해당 사건의 여러 민원 위치와 풍향을 해석해 얻는 핵심 근거이므로 가장 큰 비중을 적용했습니다.

---

<a id="2-기술-스택"></a>
## 2. ✨ 기술 스택

### AI · Data

<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas">
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/XGBoost-EB5B25?style=for-the-badge" alt="XGBoost">
</p>


---

<a id="3-주요-기능"></a>
## 3. 📍 주요 프로세스 
<img width="594" height="293" alt="image" src="https://github.com/user-attachments/assets/ec08a1b8-9c3b-492b-ae28-fd98983da46f" />



---

<a id="5-모델-성능"></a>
## 4. 📊 모델 성능

가축 관련 악취 민원만으로 Event·입력·정답을 구성하고 시간순으로 70% 학습, 30% 테스트로 분리했습니다. 운영 기준인 1km 격자의 자체평가 결과입니다.

| 평가 지표 | 결과 |
|---|---:|
| ROC-AUC | 0.855 |
| PR-AUC | 0.351 |
| Top-K Recall | 0.393 |
| Recall@3 | 0.292 |
| Event Hit@3 | 81.3% |
| 실제 추가 민원 격자 비율 | 약 10.3% |
| Event당 평균 후보 권역 | 약 35.5개 |
| 선택 모델 | XGBoost 랭킹모델 `rank_d3` |
| 운영 출력 | 상위 3개 권역 |

위 수치는 과거 관측 자료와 시간순 70/30 분할을 사용한 자체평가 결과로, 실제 현장 성능을 의미하지 않습니다. 재현 참고자료는 `outputs/operational_grid_comparison/metrics.json`과 `test_predictions.csv`입니다.

---

<a id="6-프로젝트-구조"></a>
## 5. 🗂️ 프로젝트 구조

```text
administrative_agent/              행정 대응 문서 모델·템플릿·LLM 연동
data/                              원본 및 가공 데이터
odor_service/                      이벤트·예측·대기 지점·현장 후보 운영 코드
outputs/                           모델 평가 및 데모 재현 산출물
tests/                             운영 코드 테스트
tests/experiments/                 직접 재실행하지 않는 실험·민감도 코드 보관
streamlit_before/                  이전 데모 보관(최종 실행 대상 아님)
streamlit_app.py                   최종 Streamlit 실행 파일
generate_agent_documents.py        예측 결과 기반 행정 문서 생성
PROJECT_CONTEXT.md                 프로젝트 상세 설계·운영 문서
```


<a id="6-데이터와-활용-한계"></a>
## 7. ⚠️ 참고 정보 =

- 데이터: `data/익산시 악취 민원 데이터_20190528-20260818.xlsx`
- 실제 민원 시각 범위: 2019-05-28 ~ 2026-08-17
- 공통 Event 55개, 시간순 학습 38개 / 테스트 17개
- 신고 위치는 신고 행동이 반영된 데이터이며 실제 악취 영향권과 같지 않습니다.
- 장기간의 현장 출동 결과와 배출원 정보가 없어 악취 발생 자체나 원인 시설을 학습하지 않았습니다.
- 풍속 1.0m/s 미만이면 풍향 신뢰도를 낮음, 0.5m/s 미만이면 매우 낮음으로 표시합니다.
- 점검 후보 점수는 현장 확인 우선순위 참고값이며 특정 농가의 원인·위반을 확정하지 않습니다.

권역 예측과 농가 후보 선정은 분리되어 있습니다. 권역 예측은 가축 악취 민원의 시간·공간 패턴을, 농가 후보는 해당 Event 초기 30분 민원 위치와 풍향을 사용합니다. 행정 대응 Agent는 전달받은 권역·농가 후보·상황정보만으로 문서를 만들며, 원인 시설을 확정하지 않습니다.

더 자세한 설계와 의사결정 근거는 [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md)를 확인하세요.
