---
title: "residual이 tautology였던 문제 발견"
instance_of: Experience
page_kind: experience-page
date: 2026-07-04
---

# residual이 tautology였던 문제 발견

## Context

ledger가 internal transfer를 분기별로 기록하도록 개선된 뒤에도, 적대적 검토가 residual이 '항상 맞는 장부'라고 지적.

## Attempted

_phase_ledger가 external_source=max(delta,0), external_sink=max(-delta,0)로 net delta를 외부 흐름으로 자동 분류하고 있었다.

## Outcome

이 구조에서는 어떤 분기가 mass를 버그로 만들거나 없애도 external로 흡수되어 residual이 항상 0 → 누출 탐지 불가.

## Root Cause

external을 사후 delta로 유도하면, 회계가 정의상 닫혀버려 감사 의미가 사라진다.

## Resolution

external source/sink를 각 분기(condensation/wear/clamp)에서 직접 누적하도록 변경 ([[branch-local-external-accounting]]). residual = actual-(before+src-sink)가 실제 누출을 드러냄.

## Lesson

감사 잔차는 독립적으로 측정된 흐름에서 유도되어야 한다 → [[residual-is-code-leak-not-physics]].
