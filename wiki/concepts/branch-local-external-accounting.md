---
title: "branch-local external 회계"
instance_of: Concept
page_kind: concept-page
epistemic_status: validated
---

# branch-local external 회계

## Definition

external source/sink를 (after-before) 순 변화량으로 사후 계산하지 않고, 각 분기 실행 지점에서 condensation(source)·wear loss·clamp export(sink)를 직접 누적하는 방식. residual = actual - (before + source - sink).

## Why It Matters

이렇게 해야 residual이 '설명되지 않은 mass leak' 탐지기가 된다. 정상 코드는 residual≈0, 어떤 분기가 mass를 바꾸고 기록하지 않으면 residual≠0.

## Current Understanding

internal transfer는 primary-neutral이므로 external과 독립. reference quirk(available 초과 융해)는 clamp import/export로 충실히 기록되어 residual 0을 유지하고, 물리 타당성은 diagnostics로 분리 ([[residual-vs-diagnostics-separation]]). 상위 개념 [[mass-ledger-audit-layer]].
