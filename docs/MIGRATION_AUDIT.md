# 원본 결합 기록

이 저장소는 원본 `thesis_repo`의 Stage0–9와 V10의 post-freeze·확장 분석을 함께 실행합니다. 두 원본 폴더와 논문 DOCX는 읽기 전용 입력으로 사용했으며 수정하지 않았습니다.

## 실제 코드 출처

- Stage0–9 scientific producer: `thesis_repo/src/credit_recourse`
- V10 확장 분석: V10 core의 `credit_recourse/analysis`와 관련 설정·실행 스크립트
- 논문 inventory: `docs/thesis/canonical_thesis.docx`
- V11 신규 코드: 공개 runner 2개, 실행 산출물 기반 논문 Excel 계산기, reviewer 문서

`tools` 최상위의 공개 runner는 다음 두 개뿐입니다.

- `tools\RUN_REPRODUCTION.ps1`
- `tools\BUILD_THESIS_OUTPUTS.ps1`

## 원본과 V10을 함께 보존한 방식

원본 Stage0–9를 V11 해석으로 다시 쓰지 않았습니다. Oracle·RL·Simulator·LLM의 scientific producer와 V10의 E2/E3/E4·N5·반복성·ablation consumer를 모두 보존하고, 공개 runner가 실제 생성 순서대로 호출합니다.

OracleRLClean의 연결 순서는 다음과 같습니다.

1. 원본 Stage2A raw action source
2. 원본 input split 및 candidate projection
3. 원본 counterfactual mixed-transition handoff
4. V10 LoopA/B2 확장 분석
5. 원본 final-paper Stage2–6 preset

통합 Stage2–6이 Stage2를 같은 설정으로 다시 생성하는 것은 원본 final-paper runner의 동작입니다. V11이 별도 후보행동이나 보상식을 삽입하지 않습니다.

## 과학적 계산을 바꾸지 않은 호환성 조정

- V10 분석이 현재 저장소의 `data\final_freeze`와 `data\raw`를 읽도록 역사적 개발 PC 절대경로를 현재 경로보다 후순위로 두었습니다.
- Windows에서 pandas·pyarrow와 PyTorch가 함께 열리도록 검증 코드의 import 순서만 조정했습니다.
- 이동된 내부 runner 경로를 `tools\internal\original`로 연결했습니다.
- clean Stage2 생산 순서를 원본 producer의 실제 입력 의존성에 맞췄습니다.
- V10 LoopA/B2의 10% 진단 허용값은 현재 원본 확장 분석이 요구하는 실행 조건입니다. 반사실 transition 자체의 원본 5% `warn` 진단과 계산식은 변경하지 않았습니다.

Candidate-IQL 학습식, P50 후보행동, C4/C4R/C6 생산, clipping, Simulator와 Oracle 점수화, 논문 분석식은 새 규칙으로 교체하지 않았습니다.

## 결과 경로와 보존 원칙

- `frozen_outputs`: 연구자가 사용한 역사적 snapshot. FrozenReplay만 읽습니다.
- `data\final_freeze`: 현재 clean 실행의 Oracle·RL·LLM 작업 산출물
- `data\analysis`: 현재 실행의 downstream 논문 분석
- `data\runs`: 실행 기록과 교체 전 작업본
- `data\thesis_outputs`: 선택 실행에서 계산한 reviewer용 Excel

OracleClean과 OracleRLClean은 `frozen_outputs`를 계산 입력으로 사용하지 않았습니다. 검증 중에는 FrozenReplay 결과를 clean 실행 직전에 `data\runs\V11_ORACLE_CLEAN_ACTUAL\saved_previous_working_set`으로 보존했으며, 최종 배포본에서는 검증 실행 흔적을 제거하고 논문 Excel의 자료 표기를 존재하는 `frozen_outputs` 경로 또는 workbook 내부 포함 자료(`embedded://`)로 정리했습니다.

## Excel 계산기의 역할

Excel 생성기는 논문이나 기존 인쇄표에서 숫자를 복사하는 도구가 아닙니다. 선택 run의 CSV·Parquet·JSON을 읽고 표본 필터·동일기업 대응·평균·차이·집계를 다시 수행합니다. workbook에는 source 행, 계산 과정, 논문 표시값을 함께 두어 사람이 Excel에서 검토하고 편집할 수 있습니다.

예를 들어 Candidate-IQL과 비교정책의 차이는 Stage6의 동일기업을 `row_id`로 대응시켜 다시 계산합니다. 계산할 producer가 남지 않은 역사적 LLM 수치는 보존 증거로만 표시합니다.