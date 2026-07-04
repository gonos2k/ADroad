---
title: "logaddexp 인자 클립이 τ→0 수렴을 깬 사건"
instance_of: Experience
page_kind: experience-page
date: 2026-07-04
---

# logaddexp 인자 클립이 τ→0 수렴을 깬 사건

## Context

오버플로 방어로 soft_min/soft_max의 logaddexp 인자를 [-60,60]으로 클립.

## Attempted

soft_min = cap - tau*logaddexp(0, clip((cap-x)/tau, -60, 60)).

## Outcome

x≪cap일 때 인자가 +60에 걸려 soft_min이 x가 아니라 cap-60τ로 수렴 → τ→0 hard min 수렴 테스트 실패.

## Root Cause

logaddexp는 이미 수치적으로 안정(max+log1p(exp(-|d|)))이라 오버플로가 없다. 인자 클립은 선형 영역을 파괴.

## Resolution

soft_min/max의 인자 클립 제거, safe_tau(하한)만 유지. gate의 sigmoid 인자 클립은 유지(그건 안전).

## Lesson

[[logaddexp-already-stable]]. '방어'가 정확성을 깰 수 있으니 대상의 수치 특성을 먼저 확인한다.
