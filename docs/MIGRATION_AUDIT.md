# 코드와 분석의 결합 구조

이 저장소는 `thesis_repo`의 Stage0–9 생산 코드와 V10의 post-freeze·확장 분석을 한 실행 흐름으로 연결합니다. 논문 DOCX는 표·그림·본문 수치 목록을 읽는 기준 문서입니다.

## 코드 출처

- Oracle·RL·Simulator·LLM Stage0–9: `thesis_repo/src/credit_recourse`
- E2/E3/E4·N5·반복성·ablation 분석: V10 core의 `credit_recourse/analysis`와 관련 설정·실행 스크립트
- 논문 항목 목록: `docs/thesis/canonical_thesis.docx`
- V11: 공개 runner 2개와 실행 산출물 기반 논문 Excel 생성기

`tools` 최상위의 공개 runner는 다음 두 개뿐입니다.

- `tools\RUN_REPRODUCTION.ps1`
- `tools\BUILD_THESIS_OUTPUTS.ps1`

## 실행 순서

공개 runner는 Oracle·RL·Simulator·LLM의 Stage0–9 생산 코드와 V10의 E2/E3/E4·N5·반복성·ablation 분석을 실제 생성 순서대로 호출합니다.

OracleRLClean의 연결 순서는 다음과 같습니다.

1. 원본 Stage2A raw action source
2. 원본 input split 및 candidate projection
3. 원본 counterfactual mixed-transition handoff
4. V10 LoopA/B2 확장 분석
5. 원본 final-paper Stage2–6 preset

원본 final-paper runner는 통합 Stage2–6 과정에서 같은 설정의 Stage2를 한 번 더 생성합니다. V11도 이 순서를 따릅니다.

## 저장소 이동을 위한 조정

- 프로젝트 경로를 현재 저장소 루트에서 찾도록 정리했습니다.
- Windows에서 pandas·pyarrow와 PyTorch를 함께 여는 import 순서를 적용했습니다.
- 이동된 내부 runner를 `tools\internal\original` 경로로 연결했습니다.
- clean Stage2 생산 순서를 원본 producer의 입력 의존성에 맞췄습니다.
- V10 LoopA/B2는 10% 진단 허용값을 사용하고, 반사실 transition은 원본의 5% `warn` 진단과 계산식을 사용합니다.

Candidate-IQL 학습식, P50 후보행동, C4/C4R/C6 생산, clipping, Simulator·Oracle 점수화와 논문 분석식은 각 원본 producer를 따릅니다.

## 결과 경로와 보존 원칙

- `frozen_outputs`: 연구자가 사용한 역사적 snapshot과 FrozenReplay 입력
- `data\final_freeze`: 현재 clean 실행의 Oracle·RL·LLM 작업 산출물
- `data\analysis`: 현재 실행의 downstream 논문 분석
- `data\runs`: 실행 기록과 교체 전 작업본
- `data\thesis_outputs`: 선택 실행에서 계산한 reviewer용 Excel

FrozenReplay는 `frozen_outputs`에서 시작하고, OracleClean과 OracleRLClean은 `data\raw`에서 시작합니다. 논문 Excel의 자료 표기는 현재 존재하는 `frozen_outputs` 경로 또는 workbook 내부 포함 자료(`embedded://`)를 사용합니다.

## Excel 계산기의 역할

Excel 생성기는 선택 run의 CSV·Parquet·JSON을 읽고 표본 필터·동일기업 대응·평균·차이·집계를 다시 수행합니다. workbook에는 source 행, 계산 과정과 논문 표시값을 함께 두어 사람이 Excel에서 검토하고 편집할 수 있습니다.

예를 들어 Candidate-IQL과 비교정책의 차이는 Stage6의 동일기업을 `row_id`로 대응시켜 계산합니다. 요약 근거로 보존된 역사적 LLM 수치는 `preserved evidence`로 구분합니다.
