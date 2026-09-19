# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 사고방식

**영어로 사고**
- 토큰 최적화 및 논리적 사고를 위해, 사용자와 한글로 대화하되, 사고는 영어로 진행한다.

## 답변방식
**불필요한 미사여구, 강조표현은 사용하지않는다**
**친절하고 간결하고 사실기반 담백하게 답변한다**
**복잡한 현상을 설명할땐, 이해하기 쉽게 두괄식으로 작성한다**
**항상 근거기반 주장을 제시한다**
**실제 통용되는 용어기반으로 작성하고, 맘대로 명칭을 제작하지 아니한다**
- ex. "규칙 엔진" : X / rule-base : O

## 공통 프로젝트 지침

이 프로젝트는 Codex와 Claude가 문서를 공유한다.

- 공통 작업 규칙(응답 원칙, 문서 참조, 작업 기준, 확장 검정, 데이터 작업, 성능 비교, 병렬 작업 규칙, 구현 착수 조건)은 [AGENTS.md](AGENTS.md)를 기준으로 한다.
- 이 파일에는 Claude 전용 지침만 유지하고 `AGENTS.md`의 내용을 중복 작성하지 않는다.
- 프로젝트 문서를 처음부터 전부 읽지 않는다.
- 현재 요청에 필요한 문서만 선택해서 읽는다.

### 작업별 문서 선택

- 프로젝트 목적·현재 성능·한계·다음 확장 조건이 필요한 경우: [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md)
- Track A/B 인터페이스·스키마·파일 소유권·누수 규칙이 필요한 경우: [docs/contracts/source_backtrack_contract.md](docs/contracts/source_backtrack_contract.md)
- 실행 방법·공개 수치가 필요한 경우: [README.md](README.md)
- 멘토 지적 사항의 원문이 필요한 경우: `1차 멘토링 피드백.txt`
- 특정 실험의 수치가 필요한 경우에만: `outputs/<실험명>/metrics.json`
- 행정 Agent 문서 양식이 필요한 경우에만: `outputs/administrative_agent/`의 해당 문서

### Claude 담당 범위

- Claude는 Track B(성능보존형 통합 + 서비스 출력)를 맡는다. 브랜치는 `feat/integration-ablation`.
- 수정 가능: `optimize_early_prediction.py`, `compare_operational_grid_sizes.py`, `sensitivity_early_prediction.py`, 신규 `run_ablation.py`, `administrative_agent/**`, `streamlit_app.py`, `demo/**`, `PROJECT_CONTEXT.md`, `README.md`.
- 수정 금지: Track A 소유 파일(`build_source_backtrack.py`, `outputs/source_backtrack/**`, `docs/track_a_report.md`). 계약 문서·검증기·fixture·`build_odor_ai_mvp.py`는 양쪽 합의 후에만 수정한다.
- Track A 산출물이 없을 때도 M0·M1 실험은 돌아가야 한다. 컬럼 부재는 오류가 아니라 분기로 처리한다.

### 문서 사용 원칙

- 관련 없는 실험 산출물·문서를 일괄 로드하지 않는다.
- `AGENTS.md`와 `PROJECT_CONTEXT.md`가 충돌하면 실제 원본 데이터와 최신 확인일이 있는 문서를 우선한다.
- 새로운 사실을 확인하면 같은 내용을 여러 문서에 복제하지 않고 담당 문서 하나만 갱신한다.
- 작업 규칙은 `AGENTS.md`, 현재 상태·성능·한계는 `PROJECT_CONTEXT.md`, 인터페이스는 계약 문서, 사용자용 안내는 `README.md`에 기록한다.
- 일시적인 오류, 도구 선택 과정, 시행착오 등 이후 판단에 필요 없는 작업 로그는 상태 문서에 남기지 않는다.
