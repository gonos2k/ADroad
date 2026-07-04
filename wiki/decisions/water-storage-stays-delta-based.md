---
title: "water_storage는 delta 기반 external 회계 유지"
instance_of: Decision
page_kind: decision-page
date: 2026-07-04
---

# water_storage는 delta 기반 external 회계 유지

## Context

모든 저장 스텝을 branch-local external 회계로 전환하는 중, water_storage 처리 방침 결정.

## Decision

water_storage는 내부 phase transfer가 없는 all-external 단계이므로 net-delta external 회계를 유지한다(단, hard-projection hit는 diagnostics로 반환).

## Rationale

evaporation/condensation/wear/clamp가 모두 external이라 net delta = 외부 흐름 합이 정확하다. 숨길 internal transfer가 없어 delta 기반이 tautology가 아니다.

## Alternatives Considered

snow/ice/deposit처럼 완전 branch-local 회계 — 여기선 이득 없이 복잡도만 증가.

## Consequences

3-tuple (w, ledger, diagnostics) 반환으로 water clamp 진단만 추가. 개념 [[branch-local-external-accounting]], 구현 [[droad-ledger]].
