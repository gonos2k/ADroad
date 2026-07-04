---
title: "event_flags bool 검증이 numpy bool 오탐"
instance_of: Experience
page_kind: experience-page
date: 2026-07-04
---

# event_flags bool 검증이 numpy bool 오탐

## Context

event_flags 값이 실제 bool인지 검증(문자열 'False' truthy 오염 방지)을 추가.

## Attempted

`if not isinstance(v, bool): raise`.

## Outcome

teacher-forced 경로에서 numpy 비교 결과 np.True_(numpy 2.x에서 type name 'bool', module 'numpy')가 isinstance(bool) 실패 → full_step/full_rollout 오탐 실패.

## Root Cause

numpy bool은 Python bool의 subclass가 아니다. 순수 isinstance(bool)로는 정상 bool을 거부.

## Resolution

_is_boolish(Python bool 또는 numpy bool 허용, str/int 거부) + 저장 시 bool()로 Python bool 정규화.

## Lesson

backend-neutral 검증은 numpy scalar를 duck-typing으로 허용하되, 저장은 표준 타입으로 정규화한다.
