---
title: "smooth_compat에서 질량 bounds는 hard projection 유지"
instance_of: Decision
page_kind: decision-page
date: 2026-07-04
---

# smooth_compat에서 질량 bounds는 hard projection 유지

## Context

미분가능 최적화를 위해 저장 rollout을 smooth화하되, 질량 비음수/용량 불변식을 지켜야 함.

## Decision

phase 게이트·강수 램프는 smooth화하되, 물/눈/얼음 질량 bounds는 hard jnp.clip(subgradient)로 유지한다.

## Rationale

0 하한 soft clamp는 dry-reduction 불변식을 깬다([[soft-clip-dry-reduction-break]]). 감사 관점에서도 clamp export/import를 external로 기록하면 residual 0이 유지된다.

## Alternatives Considered

전면 soft clamp — 허수 질량·불변식 파괴. 결과적으로 기각.

## Consequences

smooth_compat = 'smooth 게이트 + hard projection'로 문서화. differentiable a.e., subgradient at bounds. 개념 [[smooth-compat-mode]], 노하우 [[hard-clip-lower-bound-preserves-dry-reduction]].
