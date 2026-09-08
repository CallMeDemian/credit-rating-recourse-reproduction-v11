# 논문 재현 패키지

이 저장소는 원본 `thesis_repo`의 Oracle·RL·LLM Stage0–9 코드와 재현성 kit V10의 논문 확장 분석을 한 실행 구조로 묶은 최종 배포본입니다. 논문의 표·그림·본문 수치를 선택한 실행의 실제 산출물에서 다시 계산합니다.

## GitHub에서 받은 뒤 준비

저장소를 받은 뒤 [V11 데이터 Release](https://github.com/CallMeDemian/credit-rating-recourse-reproduction-v11/releases/tag/v11.0.0)에서 다음 ZIP 4개를 다운로드합니다.

- `credit-rating-recourse-v11_raw-data.zip`
- `credit-rating-recourse-v11_frozen-core.zip`
- `credit-rating-recourse-v11_frozen-stage2-projection.zip`
- `credit-rating-recourse-v11_frozen-llm-analysis.zip`

다운로드한 파일을 저장소 루트의 `release_assets` 폴더에 넣고, 저장소 루트에서 다음 명령을 실행합니다. 각 ZIP에는 `data\raw` 또는 `frozen_outputs` 경로가 포함되어 있습니다.

```powershell
Get-ChildItem .\release_assets\*.zip | ForEach-Object {
    Expand-Archive -LiteralPath $_.FullName -DestinationPath . -Force
}
```

압축을 푼 뒤 아래 두 경로가 채워졌는지 확인합니다.

```text
data\raw
frozen_outputs
```

그다음 Windows 64-bit Python 3.12로 저장소 자체의 가상환경을 만듭니다.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r .\requirements.lock.txt
```

## 가장 빠른 실행법

PowerShell에서 저장소 루트로 이동한 뒤 다음 두 runner만 사용합니다.

```powershell
.\tools\RUN_REPRODUCTION.ps1
.\tools\BUILD_THESIS_OUTPUTS.ps1
```

첫 번째 명령은 실행 방식을 고르는 메뉴를 띄웁니다. 두 번째 명령은 가장 최근에 정상 완료된 실행을 골라 `data\thesis_outputs`에 논문 수치·표·그림용 Excel을 만듭니다.

## 5개 실행 방식

| 방식 | 실제 수행 내용 | API 호출 |
|---|---|---|
| `FrozenReplay` | 보존된 Oracle·RL·LLM 산출물을 작업 폴더에 복사하고 V10 post-freeze/확장 분석 재실행 | 없음 |
| `OracleClean` | `data\raw`에서 원본 Oracle Stage0–1 실행 | 없음 |
| `OracleRLClean` | 열 수 있는 Oracle 결과는 재사용하고, 이어서 원본 RL Stage2–6 실행 | 없음 |
| `OracleRLLLMClean` | Oracle·RL 결과를 재사용한 뒤 원본 LLM Stage7–9 실행 | 있음 |
| `FullClean` | Oracle→RL→LLM→V10 논문 분석 전체 실행 | 있음 |

인자로 직접 고를 수도 있습니다.

```powershell
.\tools\RUN_REPRODUCTION.ps1 -Mode FrozenReplay -RunId REVIEWER_FROZEN
.\tools\RUN_REPRODUCTION.ps1 -Mode OracleClean -RunId REVIEWER_ORACLE
.\tools\RUN_REPRODUCTION.ps1 -Mode OracleRLClean -RunId REVIEWER_RL
```

실제 파일을 만들지 않고 실행 순서만 확인하려면 `-PlanOnly`를 붙입니다.

```powershell
.\tools\RUN_REPRODUCTION.ps1 -Mode FullClean -PlanOnly
```

## 논문 Excel 만들기

```powershell
.\tools\BUILD_THESIS_OUTPUTS.ps1 -RunId REVIEWER_FROZEN
```

생성 위치는 `data\thesis_outputs`입니다.

- `THESIS_OUTPUT_INDEX.xlsx`: 논문 DOCX에서 읽은 표·그림 목록과 계산 가능한 Excel 연결
- `NUMERIC_CLAIMS.xlsx`: 교수·심사위원이 확인할 핵심 본문 수치와 표본 흐름
- `tables\TABLE_*.xlsx`: `README`, `SOURCE`, `CALCULATION`, `PAPER_TABLE`
- `figures\FIGURE_*.xlsx`: `README`, `SOURCE`, `CHART_DATA`, 편집 가능한 `CHART` 또는 `CHART_SPEC`

과거 LLM 실험 중 요약 근거로 보존된 항목은 `THESIS_OUTPUT_INDEX.xlsx`에서 `preserved evidence`로 구분합니다.

## 폴더 역할

- `src`: 원본 Stage0–9 코드와 V10 확장 분석 코드
- `tools`: 공개 runner 2개
- `tools\internal`: 공개 runner가 호출하는 원본/V10 실행 스크립트
- `data\raw`: 읽기 전용 원자료 입력
- `data\final_freeze`: 현재 실행의 Oracle·RL·LLM 작업 산출물
- `data\analysis`: 현재 실행의 논문 분석 산출물
- `data\thesis_outputs`: 사람이 검토하고 편집할 Excel
- `data\runs`: 실행별 시작·종료와 보관 작업본
- `frozen_outputs`: 연구자가 실제 사용한 역사적 산출물의 불변 보존본

FrozenReplay는 `frozen_outputs`를 `data\final_freeze` 작업본으로 옮겨 분석합니다. Clean run은 `data\raw`에서 시작합니다.

## 실행 환경과 API key

Python 환경에는 `requirements.lock.txt`의 패키지가 필요합니다. 저장소의 `.venv\Scripts\python.exe`가 있으면 runner가 자동으로 사용하며, 다른 Python은 `-PythonExe`로 지정할 수 있습니다.

LLM 실행용 key는 환경변수 또는 저장소 루트의 `.env.local`에만 둡니다.

```text
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
```

runner는 key를 환경변수나 `.env.local`에서 읽습니다. Stage7 첫 호출 때 key를 확인하며, 그 전까지 계산된 Oracle·RL 결과는 유지됩니다.

실제 실행 결과는 [실행 확인](docs/RUN_VERIFICATION.md)에, 실행별 참고사항은 [실행 및 해석 참고사항](docs/KNOWN_LIMITATIONS.md)에 정리했습니다. 원본 `thesis_repo`와 V10의 결합 구조는 [코드와 분석의 결합 구조](docs/MIGRATION_AUDIT.md)에 설명합니다.

전체 Raw→Oracle→RL→LLM→논문 Excel 흐름은 [TRACEABILITY_ATLAS.pptx](docs/TRACEABILITY_ATLAS.pptx)에서 확인할 수 있습니다.
