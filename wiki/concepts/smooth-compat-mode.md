---
title: "smooth_compat (미분가능 대체 모드)"
instance_of: Concept
page_kind: concept-page
epistemic_status: validated
---

# smooth_compat (미분가능 대체 모드)

## Definition

임계 분기·클램프·상변화를 미분가능 대체(σ 게이트, softplus soft_min/max, apparent-heat-capacity 엔탈피)로 바꾼 모델 모드. τ→0에서 hard 연산으로 수렴하되 임계점에서 gradient가 유한.

## Why It Matters

gradient 기반 최적화(DA/보정)를 위해 상변화 구간을 통과하는 유한 gradient가 필요하다.

## Current Understanding

정확히는 'smooth 게이트 + hard projection bounds'다. 질량 하한/상한은 여전히 jnp.clip(subgradient) — 0 하한에 soft clamp를 쓰면 dry-reduction 불변식이 깨진다([[soft-clip-dry-reduction-break]]). differentiable almost-everywhere, subgradient at bounds. 소속 [[droad]].
