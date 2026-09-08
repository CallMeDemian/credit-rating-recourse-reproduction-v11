# 실행 및 해석 참고사항

## 역사적 LLM 산출물

FrozenReplay는 `frozen_outputs`에 보존된 Stage7–9 결과를 입력으로 사용합니다. 일부 과거 LLM 실험은 최종 요약과 응답 증거 중심으로 보존되어 있으며, 해당 항목은 `preserved evidence`로 표시됩니다.

## Clean LLM 실행

`OracleRLLLMClean`과 `FullClean`은 OpenAI 및 Anthropic live API를 사용합니다. 최종 배포 검증은 API key 없이 live 호출 직전까지 진행했으며, Stage7–9 호출은 이용자 key 입력 후 시작됩니다. API 서비스 상태, 모델 버전과 확률적 응답 특성이 역사적 응답과의 차이에 영향을 줍니다.

## 금융업 표본 처리

Clean 실행은 결과 계보를 유지하기 위해 원본 `thesis_repo` Stage0–1 producer를 사용합니다. 원본 Stage0 등급 입력은 전업종 표본으로 구성됩니다. Clean Oracle의 최종 등급 결합 표본은 4,924 기업-연도이고, 논문의 70,777 비금융 표본과 구성 기준에 차이가 있습니다. 표본 흐름은 논문 Excel에 구분해 표시했습니다.

## RL clean run의 수치 특성

Stage3–5는 고정 seed를 사용합니다. CUDA kernel 설정에 따라 checkpoint와 최종 소수점에 미세한 차이가 있습니다.

원본 final-paper Stage2 counterfactual producer는 재무항등 진단을 `warn` 모드로 실행합니다. Clean run에서 Merton과 FCFF 진단의 최대 상대오차가 원본 5% 경계를 넘은 항목은 warning으로 기록되었고, 원본 실행 흐름에 따라 Stage2–6이 완료됐습니다.

## 논문 표시값과 재계산값의 차이

- GPT 제안행동 L1 평균: 논문 표시 `0.432 / 2.298`, 선택 run 재계산 `0.431 / 2.297`로 각각 0.001 차이입니다.
- 반복성 범위: 논문 표시 `0.11–0.18`, literal C6−C4R 재계산은 `0.16–0.17`입니다. V10의 4-panel producer 전체 범위는 약 `0.113–0.174`입니다.
- Oracle-alpha best-of-11 `+1.131`은 보존된 요약 근거에 연결됩니다. Candidate-IQL `+0.630`은 선택 run에서 다시 계산됩니다.

관련 값은 `NUMERIC_CLAIMS.xlsx`에서 source·계산식·run ID와 함께 확인할 수 있습니다.

## FrozenReplay와 clean run

FrozenReplay는 역사적 산출물에서 downstream 분석을 재계산합니다. Clean run은 raw data에서 Oracle·RL·LLM 단계로 이어집니다. 두 실행은 서로 다른 provenance와 질문을 가지며 결과에는 run ID가 함께 기록됩니다.
