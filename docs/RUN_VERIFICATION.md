# 실제 실행 확인

이 문서는 교수·심사위원이 확인할 수 있는 공개 runner의 실제 실행 결과만 기록합니다. `exit_status=0`과 핵심 산출물 개방을 함께 확인한 경우에만 완료로 표시합니다.

## 최종 확인 결과

| 실행 | 최종 run ID | 결과 | 실제 산출물 |
|---|---|---:|---|
| FrozenReplay | `V11_FINAL_FROZEN_REPLAY_RESUME` | exit 0 | 보존 산출물에서 V10 논문 분석 및 E2/E3/E4 재계산 |
| 논문 Excel | 위 FrozenReplay 선택 | exit 0 | Excel 73개: 표 57, 그림 14, index·수치 ledger 2 |
| OracleClean | `V11_ORACLE_CLEAN_ACTUAL` | exit 0 | raw에서 Stage0–1 및 Oracle 3개 backend 재생성 |
| OracleRLClean | `V11_ORACLE_RL_CLEAN_ACTUAL` | exit 0 | 재사용 가능한 Oracle 뒤 Stage2–6 재생성 |
| OracleRLLLMClean | PlanOnly만 실행 | exit 0 | live API는 실행하지 않음 |
| FullClean | PlanOnly만 실행 | exit 0 | live API와 그 뒤 전체 분석은 실행하지 않음 |

## FrozenReplay

```powershell
.\tools\RUN_REPRODUCTION.ps1 -Mode FrozenReplay -RunId V11_FINAL_FROZEN_REPLAY_RESUME
```

- 검증 당시 최종 상태: `exit_status=0` (`data\runs` 실행 흔적은 최종 배포 정리에서 제거)
- 실행 기간: 2026-08-30 04:38–2026-08-31 10:46 KST
- API 호출 및 재학습: 없음
- 재계산 내용: V10 post-freeze 본분석, 9개 shuffle 분석, E2/E3/E4 확장 분석
초기 실행과 재개 과정의 임시·실패 로그는 최종 배포본에서 제거했습니다. 위 결과는 완료된 공개 실행의 최종 exit 0과 산출물 개방을 확인한 기록입니다.

## 논문 Excel 생성

```powershell
.\tools\BUILD_THESIS_OUTPUTS.ps1 -RunId V11_FINAL_FROZEN_REPLAY_RESUME
```

- 검증 당시 최종 상태: exit 0 (supervisor 실행 흔적은 최종 배포 정리에서 제거)
- 논문 DOCX에서 동적으로 읽은 inventory: 표 57개, 그림 14개
- 생성 결과: `data\thesis_outputs`의 XLSX 73개
- 핵심 파일:
  - `data\thesis_outputs\THESIS_OUTPUT_INDEX.xlsx`
  - `data\thesis_outputs\NUMERIC_CLAIMS.xlsx`
  - `data\thesis_outputs\tables\TABLE_*.xlsx`
  - `data\thesis_outputs\figures\FIGURE_*.xlsx`
- 검증 당시 FrozenReplay 작업본에서 산출했고, 최종 배포 정리에서 73개 workbook의 자료 경로를 현재 존재하는 `frozen_outputs` 경로 또는 workbook 안에 포함된 계산자료(`embedded://`)로 정규화했습니다.
- 최초 검증에서 73개 workbook을 모두 다시 열고 핵심 ledger·index·대표 표·대표 그림의 18개 sheet를 렌더링했습니다.
- 2026-09-08 최종 정리에서 요구 시트명으로 교정한 뒤 73개 파일, 수식 43,737개, 편집 가능한 chart 9개와 `CHART_SPEC` 5개를 다시 열어 확인했습니다. 손상 파일·깨진 참조·수식 오류 표시·삭제된 `data/runs` 경로·로컬 절대경로는 모두 0건이었습니다.

## OracleClean

```powershell
.\tools\RUN_REPRODUCTION.ps1 -Mode OracleClean -RunId V11_ORACLE_CLEAN_ACTUAL
```

- 검증 당시 최종 상태: `exit_status=0` (`data\runs` 실행 흔적은 최종 배포 정리에서 제거)
- 실행 기간: 2026-08-31 13:19–14:02 KST
- 입력: `data\raw`
- 결과: Stage0 재무·등급 입력 처리, Stage1 결합, Oracle alpha·beta·gamma 산출물
- 최종 Oracle 분석 표본: 4,924 기업-연도 행
- clean 계산 중 `frozen_outputs`는 읽지 않았습니다.

## OracleRLClean

```powershell
.\tools\RUN_REPRODUCTION.ps1 -Mode OracleRLClean -RunId V11_ORACLE_RL_CLEAN_ACTUAL
```

- 검증 당시 최종 상태: `exit_status=0` (`data\runs` 실행 흔적은 최종 배포 정리에서 제거)
- 실행 기간: 2026-08-31 15:03–20:11 KST
- OracleClean 산출물을 재사용한 뒤 원본 Stage2A → 입력 분할 → candidate projection → 반사실 transition → V10 LoopA/B2 → 원본 Stage3–6 순으로 실행했습니다.
- Stage2 혼합 transition: 34,749행(관측 3,159 + 반사실 31,590)
- 실제 개방 확인:
  - Stage2 평가 후보 575행, IQL 학습행 3,159행
  - Stage6 정책행동 7,475행
  - Stage6 multi-oracle 평가 10,350행
  - Stage3·Stage4·Stage5 PyTorch checkpoint 3개 모두 CPU에서 정상 개방

## 다섯 모드 PlanOnly 재확인

최종 배포 정리 후 빈 `data/final_freeze`, `data/analysis`, `data/runs` 상태에서 다섯 모드를 다시 PlanOnly로 실행했으며 모두 exit 0이었습니다.

- FrozenReplay: 보존본을 읽어 분석을 재계산한다고 표시
- OracleClean: 현재 Oracle 산출물을 재사용 가능하다고 표시
- OracleRLClean: Oracle·RL을 재사용 가능하다고 표시
- OracleRLLLMClean: Oracle·RL 재사용 후 live LLM을 실행한다고 표시
- FullClean: Oracle·RL 재사용 후 live LLM과 전체 분석을 실행한다고 표시

## 실제 실행하지 않은 live LLM

확인 시점에 `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `.env.local`이 모두 없었습니다. 따라서 비용과 외부 호출이 필요한 `OracleRLLLMClean` 및 `FullClean`의 실제 live LLM 구간은 실행하지 않았습니다. upstream Oracle·RL 성공은 유지되며, Stage7–9를 새로 생성했다고 주장하지 않습니다.

## 배포 정리

최종 배포본은 `frozen_outputs`, `data\raw`, `data\thesis_outputs`만 보존합니다. 검증 중 생성된 `data\final_freeze`, `data\analysis`, `data\runs`, supervisor·임시·실패 로그와 과거 `thesis_outputs` 백업은 제거했습니다. 새 실행은 빈 작업 폴더에 다시 기록됩니다. 공개 runner와 논문 산출물 producer에는 이 컴퓨터의 사용자 경로나 `thesis_repo` 가상환경 경로가 남아 있지 않으며, `requirements.lock.txt`는 pip dry-run과 `pip check`를 통과했습니다.