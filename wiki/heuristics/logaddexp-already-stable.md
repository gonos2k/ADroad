---
title: "logaddexp/softplus는 이미 안정 — 인자를 클립하지 마라"
instance_of: Heuristic
page_kind: heuristic-page
epistemic_status: validated
confidence: high
---

# logaddexp/softplus는 이미 안정 — 인자를 클립하지 마라

## Rule

soft_min/soft_max의 logaddexp 인자는 클립하지 않는다(tau 하한만). 오버플로 방어는 다른 곳에서.

## Why

logaddexp는 max+log1p(exp(-|d|))로 구현돼 오버플로가 없다. 인자를 클립하면 τ→0 hard 수렴에 필요한 선형 영역이 파괴된다.

## Applies When

softplus/logsumexp 계열의 안정적 primitive를 다룰 때.

## Does Not Apply When

sigmoid의 exp 인자처럼 실제로 오버플로가 나는 raw exp — 거긴 클립이 옳다.

## Evidence

[[logaddexp-clip-broke-convergence]]. smoothing.soft_min/max 구현.
