---
title: "merge_ledgers는 자식 residual을 fail-fast로 거부"
instance_of: Decision
page_kind: decision-page
date: 2026-07-04
---

# merge_ledgers는 자식 residual을 fail-fast로 거부

## Context

road_cond가 water/snow/ice/deposit/reclamp 자식 ledger를 merge_ledgers로 집계.

## Decision

merge 전에 각 자식의 |residual|<=atol을 검사해 자식 누출을 즉시 LedgerError로 거부한다.

## Rationale

연속성만 확인하면 +0.5/-0.5 자식 누출이 상쇄돼 aggregate가 깨끗해 보일 수 있다. 자식 무결성을 강제해야 조립 단계에서도 누출이 드러난다.

## Alternatives Considered

aggregate residual만 재계산(기존) — 상쇄 누출을 숨김. 기각.

## Consequences

road_cond 실제 자식은 residual~0이라 정상 통과(full_step/rollout 확인). 관련 [[mass-ledger-audit-layer]], 구현 [[droad-ledger]].
