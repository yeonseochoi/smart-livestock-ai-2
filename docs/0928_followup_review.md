# 0928 후속 코드 수정 및 P0 비교

브랜치: `feat/develop-standby-service`. 기존 미커밋 변경을 보존하고 작업 트리에서만 수정했다. 커밋·푸시·stash·reset·브랜치 생성/전환은 하지 않았다.

## P0: 미적용 비교 실험

실험 코드: `scratch/p0_compare.py` (`python -m scratch.p0_compare`). 결과: `scratch/p0_comparison.md`, `scratch/p0_results.json`, `scratch/p0_events.csv`, `scratch/p0_cards.csv`.

현재 작업 트리의 136개 사건을 같은 기상·농가·과거 이력으로 비교했다. 사용자 제시 수치인 131개 사건·645개 카드와는 데이터 버전/범위가 다르므로 이번 비교의 분모를 따로 표시한다. P2의 300m 병합은 세 안에 동일 적용했다. 기본 후보 반경·기준점·20점 계산은 변경하지 않았다.

|안|20점 0점 카드 비율|김제 대체 사건 비율|카드–민원 중심 거리 중간값|익산 후보 1곳 이상 사건 비율|후보 0곳 사건|
|---|---:|---:|---:|---:|---:|
|현재|587/670 (87.6%)|73/136 (53.7%)|9.93km|61/136 (44.9%)|2|
|A: 민원 중심 후보|102/637 (16.0%)|0/136 (0.0%)|2.98km|134/136 (98.5%)|2|
|B: 대기 장소 후보·방향 항목 실험|255/670 (38.1%)|73/136 (53.7%)|9.93km|61/136 (44.9%)|2|

A는 초기 민원 좌표의 산술평균 중심에서 풍상 4km 후보를 찾고, 거리 25점만 대기 장소와 농가 간 거리로 계산했다. 민원 중첩 20점의 지점별 4km·±45° 조건은 유지했다.

B는 대기 장소 중심의 후보 범위를 유지하되, 20점을 거리 제한 없는 민원 지점별 ±45° 방향 일치 비율로 시험했다. 방향만 일치하는 먼 농가에도 점수를 줄 수 있으므로 수송 범위·현장 적합성이 검증된 설계는 아니다. 이 실험적 재설계의 적용 여부도 사용자 결정 대상이다.

김제·익산 비율은 후보 없는 사건을 포함한 전체 사건 수가 분모다. 거리와 0점 비율은 카드가 분모다. `current` 실험의 카드 ID·순서·반올림 총점이 기본 replay와 일치함을 검사했다. 실험 실행 전후 replay 디렉터리 전체 파일의 SHA-256이 같음을 확인했다. 실험 중 기본 replay에 쓰지 않는다.

현장 민원과의 공간적 연계를 우선한다면 A가 현재 목적에 더 잘 맞는다. 다만 이것은 후보 분포 비교이며 실제 발생원 적중률이나 현장 성과 검증이 아니다. **A/B 모두 기본 동작에 적용하지 않았으며 사용자 선택을 기다린다.** 용지 대기 장소 근처 같은 군집이 반복 추천되는 구조 역시 이번에는 변경하지 않았다. 김제 대체 동작과 반대였던 주석만 바로잡았다.

## P1: 적용 사항

- `FieldCandidate`와 문서 입력 경로를 추가했다. 브리핑·지시서에 순위, 묶인 농가명, 주소, 총점·항목별 점수, 선정 요약, 근사 좌표 경고와 경계 시료 판단 문구를 넣는다. 후보가 없다는 안내도 제공한다.
- 두 LLM 경로 모두 후보 섹션·권역 표 보존 및 금지 표현을 검증한다. 위반 응답은 안전 템플릿으로 대체한다. 외부 호출을 사용하지 않는 회귀 테스트로 확인했다.
- EV2에도 `_relative_scores`의 100~60 변환을 적용했다. 권역 점수의 `%` 표시를 없앴고 대응 문구는 순위 기준이다. 습도의 물리 단위 `%`와 CSS 백분율은 유지한다.
- AWS 702 사건 시각 정시 관측의 습도·1시간 강수를 replay의 `weather`에 관측시각과 함께 저장한다. 컬럼 없음·결측은 화면 카드를 숨기고 문서에는 미제공으로 남긴다.
- 초기 민원 강도를 저장·표시하고 문서 평균·최대를 계산한다. 초기 격자 수는 v2 `assign_grid`와 같은 1km 격자의 고유 개수다.
- 저장소에 확인 가능한 읍·면·동 경계 파일이 없어 이름을 추정하지 않았다. 지역명이 없는 경우 문서의 주요 우선확인 지역 줄을 숨긴다.
- 검증 토글을 켰을 때만 Top 3의 실제 이후 신고 권역 수를 `n/3`으로 표시한다.
- 시간 근접 매칭을 삭제했다. 동일 ID가 없으면 현장 재현 자료 없음으로 안내한다.
- 생성·입력 기본 시각은 `ZoneInfo("Asia/Seoul")`를 사용한다. 지시시각도 한국 표준시로 변환한다.
- LLM 예외 상세는 로그에만 기록하고 화면에는 일반 안내를 표시한다. 안내 문자열은 HTML escape 처리한다.
- 사건 선택 시각 형식을 정리하고, IK-8 및 미매핑 군집의 읽을 수 있는 표시명을 제공한다.

## P2: 적용 사항 및 남는 한계

- 같은 이름만으로 다른 위치를 삭제하던 처리를 없앴다. 동일 좌표 또는 같은 이름이면서 대표점 300m 이내인 등록만 묶고, 묶인 ID·주소를 보존한다. 연쇄 병합으로 먼 시설이 사라지지 않게 대표점 기준으로 비교한다.
- 수정 전후 136개 사건 모두 후보 수·ID·순서가 동일했다. 총 670 → 670개, 후보 수 변화 사건 0개다. 실제 이번 데이터에는 결과를 바꾸는 해당 조건이 없었으므로 200m 동명 병합·2km 동명 보존 사례를 별도 테스트했다. 수정 전 스냅샷은 `scratch/review_before.json`이다.
- sticky/내부 스크롤의 덮어써지는 CSS를 제거하고 실제 static/auto/visible 스타일을 검사한다. 지도는 Top 3만 다루며 그 밖의 관찰 등급 분기는 제거했다.
- 과거 반복 15점은 후보 간 상대 정규화 값이라 양의 반복 이력이 있으면 최대 후보가 15점이 됨을 기준 문서에 명시했다.
- 대기 장소 lift·등급·위치는 2020~2026 전체 자료로 만들어져 과거 사건 이후 정보가 섞인다. 밤 전체 관측 평균 풍향 역시 사건 당시 이용 가능한 정보가 아니다. 연도 이전 민원뿐 아니라 당시 농가 스냅샷과 배치 기준을 함께 맞춰야 하므로 이번에는 불완전한 연도 필터를 추가하지 않고 한계를 문서·화면에 명시했다.
- `level`은 사후 확인 등급으로만 해석하며 화면과 행정 문서의 출동 판단에는 쓰지 않는다. 요약 파일의 등급 제목도 사후 확인 등급으로 명확히 했다.

기본 replay의 이번 갱신은 P1의 관측값·강도 저장 및 P2의 병합 규칙 반영을 위한 것이다. P0의 A/B 실험 결과로 기본 replay를 교체하지 않았다.

## 검증 환경

프로젝트 `.venv`의 Python 실행 파일이 삭제된 설치 경로를 참조하므로, 앱에 포함된 Python 3.12에 `.venv/Lib/site-packages`를 연결해 테스트했다. Python이나 패키지를 설치·변경하지 않았다. 실제 외부 LLM 호출과 브라우저의 외부 지도 타일 로딩은 검증 범위 밖이다.

전체 테스트 결과와 최종 Git 출력은 아래에 기록한다. Git 출력에는 이전 작업의 미커밋 변경도 포함되며, `git diff --stat`에는 untracked 파일 내용이 포함되지 않는다.

검증 결과: 전체 43개 테스트 통과 (`scratch/review_tests.log`), `git diff --check` 통과.

## 최종 git status --short

```text
 M administrative_agent/documents.py
 M administrative_agent/llm.py
 M administrative_agent/models.py
 M administrative_agent/policy.py
 M docs/0928_연서.md
 M generate_agent_documents.py
 M odor_service/event_link/replay.py
 M odor_service/field/engine.py
 M outputs/odor_service/replay/2022-05-05.json
 M outputs/odor_service/replay/2022-06-25.json
 M outputs/odor_service/replay/2022-06-26.json
 M outputs/odor_service/replay/2022-06-27.json
 M outputs/odor_service/replay/2022-06-29.json
 M outputs/odor_service/replay/2022-06-30.json
 M outputs/odor_service/replay/2022-07-02.json
 M outputs/odor_service/replay/2022-07-03.json
 M outputs/odor_service/replay/2022-07-04.json
 M outputs/odor_service/replay/2022-07-05.json
 M outputs/odor_service/replay/2022-07-06.json
 M outputs/odor_service/replay/2022-07-07.json
 M outputs/odor_service/replay/2022-07-10.json
 M outputs/odor_service/replay/2022-07-13.json
 M outputs/odor_service/replay/2022-07-15.json
 M outputs/odor_service/replay/2022-07-16.json
 M outputs/odor_service/replay/2022-07-17.json
 M outputs/odor_service/replay/2022-07-19.json
 M outputs/odor_service/replay/2022-07-22.json
 M outputs/odor_service/replay/2022-07-25.json
 M outputs/odor_service/replay/2022-07-26.json
 M outputs/odor_service/replay/2022-07-27.json
 M outputs/odor_service/replay/2022-07-29.json
 M outputs/odor_service/replay/2022-08-01.json
 M outputs/odor_service/replay/2022-08-02.json
 M outputs/odor_service/replay/2022-08-04.json
 M outputs/odor_service/replay/2022-08-14.json
 M outputs/odor_service/replay/2022-08-18.json
 M outputs/odor_service/replay/2022-08-20.json
 M outputs/odor_service/replay/2022-08-21.json
 M outputs/odor_service/replay/2022-08-22.json
 M outputs/odor_service/replay/2022-08-26.json
 M outputs/odor_service/replay/2022-09-02.json
 M outputs/odor_service/replay/2022-09-06.json
 M outputs/odor_service/replay/2022-09-14.json
 M outputs/odor_service/replay/2022-09-15.json
 M outputs/odor_service/replay/2022-09-16.json
 M outputs/odor_service/replay/2022-09-18.json
 M outputs/odor_service/replay/2022-09-22.json
 M outputs/odor_service/replay/2022-10-02.json
 M outputs/odor_service/replay/2023-06-26.json
 M outputs/odor_service/replay/2023-07-05.json
 M outputs/odor_service/replay/2023-07-13.json
 M outputs/odor_service/replay/2023-08-21.json
 M outputs/odor_service/replay/2023-09-04.json
 M outputs/odor_service/replay/2023-09-08.json
 M outputs/odor_service/replay/2023-09-10.json
 M outputs/odor_service/replay/2023-09-11.json
 M outputs/odor_service/replay/2023-09-12.json
 M outputs/odor_service/replay/2023-09-17.json
 M outputs/odor_service/replay/2023-09-18.json
 M outputs/odor_service/replay/2023-09-19.json
 M outputs/odor_service/replay/2023-09-30.json
 M outputs/odor_service/replay/2023-10-05.json
 M outputs/odor_service/replay/2023-10-13.json
 M outputs/odor_service/replay/2024-05-25.json
 M outputs/odor_service/replay/2024-06-01.json
 M outputs/odor_service/replay/2024-06-18.json
 M outputs/odor_service/replay/2024-06-23.json
 M outputs/odor_service/replay/2024-06-27.json
 M outputs/odor_service/replay/2024-07-02.json
 M outputs/odor_service/replay/2024-07-04.json
 M outputs/odor_service/replay/2024-07-05.json
 M outputs/odor_service/replay/2024-07-14.json
 M outputs/odor_service/replay/2024-07-17.json
 M outputs/odor_service/replay/2024-07-22.json
 M outputs/odor_service/replay/2024-07-23.json
 M outputs/odor_service/replay/2024-07-28.json
 M outputs/odor_service/replay/2024-07-30.json
 M outputs/odor_service/replay/2024-08-15.json
 M outputs/odor_service/replay/2024-08-20.json
 M outputs/odor_service/replay/2024-08-31.json
 M outputs/odor_service/replay/2024-09-01.json
 M outputs/odor_service/replay/2024-09-03.json
 M outputs/odor_service/replay/2024-09-04.json
 M outputs/odor_service/replay/2024-09-07.json
 M outputs/odor_service/replay/2024-09-08.json
 M outputs/odor_service/replay/2024-09-12.json
 M outputs/odor_service/replay/2024-09-13.json
 M outputs/odor_service/replay/2024-09-25.json
 M outputs/odor_service/replay/2024-09-29.json
 M outputs/odor_service/replay/2024-10-01.json
 M outputs/odor_service/replay/2024-10-03.json
 M outputs/odor_service/replay/2024-10-12.json
 M outputs/odor_service/replay/2024-10-27.json
 M outputs/odor_service/replay/2025-06-06.json
 M outputs/odor_service/replay/2025-06-19.json
 M outputs/odor_service/replay/2025-07-30.json
 M outputs/odor_service/replay/2025-08-08.json
 M outputs/odor_service/replay/2025-08-13.json
 M outputs/odor_service/replay/2025-08-16.json
 M outputs/odor_service/replay/2025-08-19.json
 M outputs/odor_service/replay/2025-08-20.json
 M outputs/odor_service/replay/2025-08-28.json
 M outputs/odor_service/replay/2025-08-31.json
 M outputs/odor_service/replay/2025-09-05.json
 M outputs/odor_service/replay/2025-09-13.json
 M outputs/odor_service/replay/2025-09-14.json
 M outputs/odor_service/replay/2025-09-16.json
 M outputs/odor_service/replay/2025-09-20.json
 M outputs/odor_service/replay/2025-09-23.json
 M outputs/odor_service/replay/2025-09-30.json
 M outputs/odor_service/replay/2025-10-10.json
 M outputs/odor_service/replay/2025-10-11.json
 M outputs/odor_service/replay/2025-10-12.json
 M outputs/odor_service/replay/summary.md
 M streamlit_app.py
 M tests/test_administrative_agent.py
 M tests/test_odor_service_field.py
 M tests/test_odor_service_link.py
 M tests/test_streamlit_app.py
?? administrative_agent/time_utils.py
?? check_data.py
?? docs/0928_followup_review.md
?? "docs/0928_유진(코드 점검).md"
?? docs/IMPLEMENTATION_PLAN.md
?? iksan_odor_ai_yujin/
?? scratch/
?? tests/test_streamlit_logic.py
?? v2/
```

## 최종 git diff --stat

```text
 administrative_agent/documents.py           |  48 ++-
 administrative_agent/llm.py                 |  51 ++-
 administrative_agent/models.py              |  13 +
 administrative_agent/policy.py              |  12 +-
 "docs/0928_\354\227\260\354\204\234.md"     |  20 +-
 generate_agent_documents.py                 |   3 +-
 odor_service/event_link/replay.py           |  34 +-
 odor_service/field/engine.py                |  63 ++--
 outputs/odor_service/replay/2022-05-05.json | 114 ++++---
 outputs/odor_service/replay/2022-06-25.json |  38 ++-
 outputs/odor_service/replay/2022-06-26.json | 209 ++++++------
 outputs/odor_service/replay/2022-06-27.json |  33 +-
 outputs/odor_service/replay/2022-06-29.json |  78 ++++-
 outputs/odor_service/replay/2022-06-30.json | 100 +++++-
 outputs/odor_service/replay/2022-07-02.json |  93 ++++--
 outputs/odor_service/replay/2022-07-03.json | 494 ++++++++++++++++------------
 outputs/odor_service/replay/2022-07-04.json |  46 ++-
 outputs/odor_service/replay/2022-07-05.json |  31 +-
 outputs/odor_service/replay/2022-07-06.json |  76 ++++-
 outputs/odor_service/replay/2022-07-07.json |  23 +-
 outputs/odor_service/replay/2022-07-10.json |  34 +-
 outputs/odor_service/replay/2022-07-13.json | 332 +++++++++++++------
 outputs/odor_service/replay/2022-07-15.json |  38 ++-
 outputs/odor_service/replay/2022-07-16.json |  38 ++-
 outputs/odor_service/replay/2022-07-17.json | 219 ++++++------
 outputs/odor_service/replay/2022-07-19.json |  66 +++-
 outputs/odor_service/replay/2022-07-22.json | 165 ++++++----
 outputs/odor_service/replay/2022-07-25.json | 113 ++++++-
 outputs/odor_service/replay/2022-07-26.json |  36 +-
 outputs/odor_service/replay/2022-07-27.json |  62 ++--
 outputs/odor_service/replay/2022-07-29.json | 162 +++++----
 outputs/odor_service/replay/2022-08-01.json |  74 ++++-
 outputs/odor_service/replay/2022-08-02.json |  31 +-
 outputs/odor_service/replay/2022-08-04.json |  31 +-
 outputs/odor_service/replay/2022-08-14.json | 114 ++++++-
 outputs/odor_service/replay/2022-08-18.json | 205 +++++++-----
 outputs/odor_service/replay/2022-08-20.json |  81 +++--
 outputs/odor_service/replay/2022-08-21.json |  28 +-
 outputs/odor_service/replay/2022-08-22.json |  69 +++-
 outputs/odor_service/replay/2022-08-26.json |  38 ++-
 outputs/odor_service/replay/2022-09-02.json | 233 +++++++------
 outputs/odor_service/replay/2022-09-06.json | 230 ++++++++-----
 outputs/odor_service/replay/2022-09-14.json | 115 ++++---
 outputs/odor_service/replay/2022-09-15.json | 162 +++++----
 outputs/odor_service/replay/2022-09-16.json |  77 +++--
 outputs/odor_service/replay/2022-09-18.json | 191 ++++++-----
 outputs/odor_service/replay/2022-09-22.json |  80 +++--
 outputs/odor_service/replay/2022-10-02.json |  36 +-
 outputs/odor_service/replay/2023-06-26.json |  23 +-
 outputs/odor_service/replay/2023-07-05.json |  47 ++-
 outputs/odor_service/replay/2023-07-13.json |  30 +-
 outputs/odor_service/replay/2023-08-21.json |  27 +-
 outputs/odor_service/replay/2023-09-04.json |  23 +-
 outputs/odor_service/replay/2023-09-08.json |  23 +-
 outputs/odor_service/replay/2023-09-10.json | 204 +++++++-----
 outputs/odor_service/replay/2023-09-11.json | 227 +++++++------
 outputs/odor_service/replay/2023-09-12.json |  23 +-
 outputs/odor_service/replay/2023-09-17.json |  29 +-
 outputs/odor_service/replay/2023-09-18.json |  28 +-
 outputs/odor_service/replay/2023-09-19.json |  75 +++--
 outputs/odor_service/replay/2023-09-30.json |  23 +-
 outputs/odor_service/replay/2023-10-05.json |  23 +-
 outputs/odor_service/replay/2023-10-13.json |  23 +-
 outputs/odor_service/replay/2024-05-25.json |  23 +-
 outputs/odor_service/replay/2024-06-01.json |  23 +-
 outputs/odor_service/replay/2024-06-18.json |  23 +-
 outputs/odor_service/replay/2024-06-23.json |  51 ++-
 outputs/odor_service/replay/2024-06-27.json |  23 +-
 outputs/odor_service/replay/2024-07-02.json |  51 ++-
 outputs/odor_service/replay/2024-07-04.json |  44 ++-
 outputs/odor_service/replay/2024-07-05.json |  28 +-
 outputs/odor_service/replay/2024-07-14.json |  28 +-
 outputs/odor_service/replay/2024-07-17.json | 124 +++++--
 outputs/odor_service/replay/2024-07-22.json |  47 ++-
 outputs/odor_service/replay/2024-07-23.json |  44 ++-
 outputs/odor_service/replay/2024-07-28.json |  77 +++--
 outputs/odor_service/replay/2024-07-30.json |  28 +-
 outputs/odor_service/replay/2024-08-15.json |  75 +++--
 outputs/odor_service/replay/2024-08-20.json |  57 +++-
 outputs/odor_service/replay/2024-08-31.json |  93 ++++--
 outputs/odor_service/replay/2024-09-01.json |  30 +-
 outputs/odor_service/replay/2024-09-03.json |  29 +-
 outputs/odor_service/replay/2024-09-04.json |  23 +-
 outputs/odor_service/replay/2024-09-07.json |  29 +-
 outputs/odor_service/replay/2024-09-08.json |  30 +-
 outputs/odor_service/replay/2024-09-12.json |  46 ++-
 outputs/odor_service/replay/2024-09-13.json |  72 ++--
 outputs/odor_service/replay/2024-09-25.json |  23 +-
 outputs/odor_service/replay/2024-09-29.json |  34 +-
 outputs/odor_service/replay/2024-10-01.json |  40 ++-
 outputs/odor_service/replay/2024-10-03.json |  26 +-
 outputs/odor_service/replay/2024-10-12.json |  28 +-
 outputs/odor_service/replay/2024-10-27.json |  23 +-
 outputs/odor_service/replay/2025-06-06.json |  40 ++-
 outputs/odor_service/replay/2025-06-19.json |  23 +-
 outputs/odor_service/replay/2025-07-30.json |  30 +-
 outputs/odor_service/replay/2025-08-08.json |  63 +++-
 outputs/odor_service/replay/2025-08-13.json |  23 +-
 outputs/odor_service/replay/2025-08-16.json |  26 +-
 outputs/odor_service/replay/2025-08-19.json |  23 +-
 outputs/odor_service/replay/2025-08-20.json |  23 +-
 outputs/odor_service/replay/2025-08-28.json |  23 +-
 outputs/odor_service/replay/2025-08-31.json |  23 +-
 outputs/odor_service/replay/2025-09-05.json |  23 +-
 outputs/odor_service/replay/2025-09-13.json |  23 +-
 outputs/odor_service/replay/2025-09-14.json |  23 +-
 outputs/odor_service/replay/2025-09-16.json |  44 ++-
 outputs/odor_service/replay/2025-09-20.json |  33 +-
 outputs/odor_service/replay/2025-09-23.json |  42 ++-
 outputs/odor_service/replay/2025-09-30.json | 249 ++++++++------
 outputs/odor_service/replay/2025-10-10.json |  29 +-
 outputs/odor_service/replay/2025-10-11.json |  23 +-
 outputs/odor_service/replay/2025-10-12.json |  23 +-
 outputs/odor_service/replay/summary.md      |   4 +-
 streamlit_app.py                            | 177 ++++++----
 tests/test_administrative_agent.py          |  45 ++-
 tests/test_odor_service_field.py            |  14 +
 tests/test_odor_service_link.py             |  19 +-
 tests/test_streamlit_app.py                 |  84 ++++-
 119 files changed, 5459 insertions(+), 2432 deletions(-)
```
