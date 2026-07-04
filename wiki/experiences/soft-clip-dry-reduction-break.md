---
title: "soft_clip이 dry-reduction을 깬 사건"
instance_of: Experience
page_kind: experience-page
date: 2026-07-04
---

# soft_clip이 dry-reduction을 깬 사건

## Context

검토 제안으로 jax_storage의 질량 하한/상한 jnp.clip을 미분가능 soft_clip으로 교체.

## Attempted

Wat/Ice/Snow에 soft_clip(x, 0, cap, tau) 적용.

## Outcome

강수 없는 dry 케이스에서 soft_clip(0,0,cap,tau)=soft_max(0,0,tau)=tau·log2 만큼 허수 질량이 생겨 dry-reduction 불변식(dry 환원 <1e-5)이 0.1K 규모로 깨짐.

## Root Cause

soft clamp는 x=0 하한에서 tau·log2의 오프셋을 만든다. 0 하한 projection에는 부적합.

## Resolution

질량 하한/상한은 hard jnp.clip 유지(subgradient OK), 강수 fraction 램프만 soft화. 이유를 코드 주석에 명시.

## Lesson

[[hard-clip-lower-bound-preserves-dry-reduction]]. smooth_compat은 'smooth 게이트 + hard projection'이다 ([[smooth-compat-mode]]).
