---
title: "0 하한은 soft clamp 말고 hard clip"
instance_of: Heuristic
page_kind: heuristic-page
epistemic_status: validated
confidence: high
---

# 0 하한은 soft clamp 말고 hard clip

## Rule

질량 비음수(0 하한) projection은 미분가능 soft clamp가 아니라 hard clip으로 둔다. 상한/램프만 필요 시 soft화한다.

## Why

soft clamp는 x=0에서 tau·log2의 허수 질량 오프셋을 만들어 dry-reduction 불변식을 깬다. 0 근처에서 subgradient clip이 물리적으로 옳다.

## Applies When

비음수 상태량의 하한 projection(물/눈/얼음 질량 등).

## Does Not Apply When

임계 게이트·상변화 전이처럼 통과 gradient가 필요한 곳 — 거긴 smooth 게이트.

## Evidence

[[soft-clip-dry-reduction-break]]. 개념 [[smooth-compat-mode]].
