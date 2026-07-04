---
title: "numeric 필드에서 bool·str을 거부한다"
instance_of: Heuristic
page_kind: heuristic-page
epistemic_status: validated
confidence: high
---

# numeric 필드에서 bool·str을 거부한다

## Rule

질량/흐름 같은 numeric 감사 필드는 str('1.0')과 bool(True)을 명시적으로 거부한다. mass amount는 numeric, event flag만 bool.

## Why

Python bool은 int subclass라 float(True)=1.0으로 조용히 coerce되고, 문자열은 truthy로 오염될 수 있다. 감사 record에서는 침묵 coercion이 최악.

## Applies When

감사/검증 계층의 스칼라 필드 정규화.

## Does Not Apply When

사용자 편의를 위해 관대한 파싱이 필요한 UI 경계.

## Evidence

_as_finite_float가 _is_boolish+str를 거부. 경험 [[numpy-bool-event-flags-false-positive]].
