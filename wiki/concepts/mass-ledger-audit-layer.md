---
title: "mass ledger 감사층"
instance_of: Concept
page_kind: concept-page
epistemic_status: validated
---

# mass ledger 감사층

## Definition

저장항(물+눈+얼음+서리) 질량 흐름을 primary/auxiliary/external source·sink/internal transfer로 분리 기록하는 감사 계약. 단일 'total water' 숫자가 아니라, 누락 전이를 숨길 수 없도록 key set을 고정.

## Why It Matters

상변화 디버깅과 smooth surrogate deviation budget에는 전이량 자체가 핵심. residual이 실제 코드 누출을 탐지하는 도구가 되어야 신뢰할 수 있다.

## Current Understanding

external을 net delta로 후처리하면 residual이 tautology가 된다(→ [[residual-tautology-discovery]]). 해결은 [[branch-local-external-accounting]]. 역할 분리는 [[residual-vs-diagnostics-separation]]. 구현은 [[droad-ledger]].
