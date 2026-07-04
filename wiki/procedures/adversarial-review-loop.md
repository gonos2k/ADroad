---
title: "적대적 검토 반복 루프"
instance_of: Procedure
page_kind: procedure-page
epistemic_status: validated
---

# 적대적 검토 반복 루프

## Purpose

외부 정적 검토를 라운드로 받아 우선순위(P0/P1/P2)대로 코드에 반영하고 회귀 테스트로 고정하는 워크플로우.

## Preconditions

검토 의견이 파일·함수·라인 단위로 구체적일 것. 반영 후 검증 배치가 준비돼 있을 것.

## Steps

1. 검토 항목을 P0(정확성/안전) → P1 → P2 순으로 태스크화
2. 각 항목을 실제 코드에 반영하고, 그 항목을 잠그는 회귀 테스트를 추가
3. 코어/JAX 배치로 재검증 + raw-primitive 감사([[run-tests-in-batches]])
4. 실측과 충돌하는 제안은 근거를 남기고 조정(예: [[logaddexp-already-stable]])
5. README에 라운드별 반영 요약을 정직하게 기록
6. 커밋·푸시(마운트 .git 락 이슈는 [[git-index-lock-mount]] 참조)

## Postconditions / Verification

각 라운드마다 테스트 수 증가 + 전부 통과 + 감사 clean. 소스 [[adversarial-review-series]].
