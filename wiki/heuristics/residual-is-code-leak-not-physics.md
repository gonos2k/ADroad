---
title: "잔차는 코드 누출 신호, 물리 판단이 아니다"
instance_of: Heuristic
page_kind: heuristic-page
epistemic_status: validated
confidence: high
---

# 잔차는 코드 누출 신호, 물리 판단이 아니다

## Rule

mass 잔차(residual)는 '우리 코드가 변화를 장부에 기록했는가'만 판정한다. 물리적 타당성(over-melt 등)은 별도 diagnostics로 남긴다.

## Why

둘을 섞으면 reference quirk를 잔차 실패로 오판하거나(exact-compat 파괴), 정상 흐름을 감사 실패로 오탐한다.

## Applies When

감사층이 reference 동작을 충실히 따라가면서도 자기 코드 버그를 잡아야 할 때.

## Does Not Apply When

deviation budget처럼 '물리적으로 바람직한가'를 판단해야 할 때 — 그건 diagnostics/deviation의 몫.

## Evidence

[[residual-tautology-discovery]]에서 branch-local 회계로 전환 후 확립. 개념 [[residual-vs-diagnostics-separation]].
