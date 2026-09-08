# 실행 결과

공개 runner로 직접 수행한 결과를 정리했습니다.

## 최종 확인 결과

| 실행 | 최종 run ID | 결과 | 실제 산출물 |
|---|---|---:|---|
| FrozenReplay | `V11_FINAL_FROZEN_REPLAY_RESUME` | exit 0 | 보존 산출물에서 V10 논문 분석 및 E2/E3/E4 재계산 |
| 논문 Excel | 위 FrozenReplay 선택 | exit 0 | Excel 73개: 표 57, 그림 14, index·수치 ledger 2 |
| OracleClean | `V11_ORACLE_CLEAN_ACTUAL` | exit 0 | raw에서 Stage0–1 및 Oracle 3개 backend 재생성 |
| OracleRLClean | `V11_ORACLE_RL_CLEAN_ACTUAL` | exit 0 | 재사용 가능한 Oracle 뒤 Stage2–6 재생성 |
| OracleRLLLMClean | PlanOnly 확인 | exit 0 | Stage7 live API 진입 순서 확인 |
| FullClean | PlanOnly 확인 | exit 0 | live LLM과 전체 분석 실행 순서 확인 |

## FrozenReplay

```powershell
.\tools\RUN_REPRODUCTION.ps1 -Mode FrozenReplay -RunId V11_FINAL_FROZEN_REPLAY_RESUME
```

- 실행 상태: `exit_status=0`
- 실행 기간: 2026-08-30 04:38–2026-08-31 10:46 KST
- API 호출 및 재학습: 없음
- 재계산 내용: V10 post-freeze 본분석, 9개 shuffle 분석, E2/E3/E4 확장 분석

## 논문 Excel 생성

```powershell
.\tools\BUILD_THESIS_OUTPUTS.ps1 -RunId V11_FINAL_FROZEN_REPLAY_RESUME
```

- 실행 상태: exit 0
- 논문 DOCX에서 동적으로 읽은 inventory: 표 57개, 그림 14개
- 생성 결과: `data\thesis_outputs`의 XLSX 73개
- 핵심 파일:
  - `data\thesis_outputs\THESIS_OUTPUT_INDEX.xlsx`
  - `data\thesis_outputs\NUMERIC_CLAIMS.xlsx`
  - `data\thesis_outputs\tables\TABLE_*.xlsx`
  - `data\thesis_outputs\figures\FIGURE_*.xlsx`
- 73개 workbook의 자료 경로는 현재 존재하는 `frozen_outputs` 경로 또는 workbook 안에 포함된 계산자료(`embedded://`)를 가리킵니다.
- 최초 검증에서 73개 workbook을 모두 다시 열고 핵심 ledger·index·대표 표·대표 그림의 18개 sheet를 렌더링했습니다.
- 2026-09-08 최종 정리에서 요구 시트명으로 교정한 뒤 73개 파일, 수식 43,737개, 편집 가능한 chart 9개와 `CHART_SPEC` 5개를 다시 열어 확인했습니다.

## OracleClean

```powershell
.\tools\RUN_REPRODUCTION.ps1 -Mode OracleClean -RunId V11_ORACLE_CLEAN_ACTUAL
```

- 실행 상태: `exit_status=0`
- 실행 기간: 2026-08-31 13:19–14:02 KST
- 입력: `data\raw`
- 결과: Stage0 재무·등급 입력 처리, Stage1 결합, Oracle alpha·beta·gamma 산출물
- 최종 Oracle 분석 표본: 4,924 기업-연도 행

## OracleRLClean

```powershell
.\tools\RUN_REPRODUCTION.ps1 -Mode OracleRLClean -RunId V11_ORACLE_RL_CLEAN_ACTUAL
```

- 실행 상태: `exit_status=0`
- 실행 기간: 2026-08-31 15:03–20:11 KST
- OracleClean 산출물을 재사용한 뒤 원본 Stage2A → 입력 분할 → candidate projection → 반사실 transition → V10 LoopA/B2 → 원본 Stage3–6 순으로 실행했습니다.
- Stage2 혼합 transition: 34,749행(관측 3,159 + 반사실 31,590)
- 주요 산출물:
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

## LLM API 실행 조건

`OracleRLLLMClean`과 `FullClean`의 Stage7–9는 이용자가 환경변수 또는 `.env.local`에 API key를 넣은 뒤 시작됩니다. 배포 확인에서는 key 없이 live 호출 직전까지 실행 순서를 확인했습니다. OracleClean과 OracleRLClean은 각각 exit 0으로 완료했습니다.

## 배포 정리

최종 배포본에는 `frozen_outputs`, `data\raw`, `data\thesis_outputs`가 들어 있습니다. `data\final_freeze`, `data\analysis`, `data\runs`는 새 실행에서 채워집니다. 공개 runner는 저장소의 `.venv`를 우선 사용하며, `requirements.lock.txt`는 pip dry-run과 `pip check`를 통과했습니다.
