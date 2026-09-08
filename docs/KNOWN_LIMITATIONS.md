# 알려진 한계

## 역사적 LLM 산출물

FrozenReplay는 `frozen_outputs`에 실제로 보존된 Stage7–9 결과를 입력으로 사용합니다. 일부 과거 LLM 실험은 최종 요약 또는 응답 증거만 남고 최초 producer snapshot이 완전하게 보존되지 않았습니다. 이런 항목은 보존 증거로만 다루며 fresh API regeneration이라고 주장하지 않습니다.

## Clean LLM 실행

`OracleRLLLMClean`과 `FullClean`은 OpenAI 및 Anthropic live API를 사용합니다. 최종 확인 시 환경변수와 `.env.local`에 key가 없어 실제 live 실행은 하지 않았습니다. key가 없으면 첫 live 호출 직전에 중단되고 이미 완료된 Oracle·RL은 유지됩니다. API 서비스 상태, 모델 가용성, 비용과 비결정성 때문에 이후 실행도 역사적 응답과 byte 단위로 같다고 보장할 수 없습니다.

## 금융업 제외에 관한 원본 producer의 범위

결과를 바꾸지 않기 위해 원본 `thesis_repo` Stage0–1을 그대로 사용했습니다. 원본 Stage0 등급 입력은 전업종 표본이며, 별도의 금융업 업종코드 제외 로직은 scientific producer에서 확인되지 않았습니다. 따라서 이 kit가 새 필터를 삽입하거나 논문의 70,777 비금융 표본을 원본 코드에서 재현했다고 주장하지 않습니다. clean Oracle의 최종 등급 결합 표본은 4,924 기업-연도이며 Excel 표본 흐름에 차이를 표시합니다.

## RL clean run의 수치적 재현성

Stage3–5 학습은 seed를 고정하지만 원본 설정상 CUDA kernel을 완전 결정론적으로 강제하지 않습니다. 같은 코드·데이터·설정의 과학적 재현은 가능하지만 checkpoint와 최종 소수점이 byte 단위로 같다는 보장은 없습니다.

원본 final-paper Stage2 counterfactual producer는 재무항등 진단을 `warn` 모드로 사용합니다. 이번 clean run에서도 Merton과 FCFF 진단의 최대 상대오차가 원본 5% 경계를 넘었지만 producer는 설계대로 계속 실행해 전체 Stage2–6을 완료했습니다. 이를 strict fidelity 통과로 표현하지 않습니다.

## 논문 표시값과 재계산값의 확인된 차이

- GPT 제안행동 L1 평균: 논문 표시 `0.432 / 2.298`, 선택 run 재계산 `0.431 / 2.297`로 각각 0.001 차이입니다.
- 반복성 범위: 논문 표시 `0.11–0.18`, literal C6−C4R 재계산은 `0.16–0.17`입니다. V10의 4-panel producer 전체 범위는 약 `0.113–0.174`입니다.
- Oracle-alpha best-of-11 `+1.131`을 최초 생성한 완전한 producer snapshot은 보존본에서 확인되지 않아 preserved evidence로만 둡니다. Candidate-IQL `+0.630`은 선택 run에서 다시 계산됩니다.

이 차이는 숨기거나 논문 값에 맞춰 덮어쓰지 않고 `NUMERIC_CLAIMS.xlsx`에 source·계산식·run ID와 함께 남깁니다.

## FrozenReplay와 clean run

FrozenReplay는 역사적 산출물에서 downstream 분석을 재계산하며 Oracle·RL 재학습이나 LLM API 호출을 하지 않습니다. Clean run은 `frozen_outputs`를 계산 입력으로 사용하지 않습니다. 두 실행은 provenance와 질문이 다르므로 run ID를 함께 확인해야 합니다.