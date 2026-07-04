---
title: "정규화 → 검증 → freeze 순서로 감사 record를 만든다"
instance_of: Heuristic
page_kind: heuristic-page
epistemic_status: validated
confidence: high
---

# 정규화 → 검증 → freeze 순서로 감사 record를 만든다

## Rule

감사 record는 numeric을 float로 정규화하고, event flag를 bool로 정규화한 뒤, 검증하고, 마지막에 mapping을 불변화(freeze)한다.

## Why

먼저 정규화하면 이후 비교·산술이 numpy/jax/int/str 혼입에도 타입 오류 없이 일관되게 동작하고, freeze로 사후 변조를 막아 audit 신뢰성을 확보한다.

## Applies When

불변 감사 record(ledger)나 계약형 dataclass를 설계할 때.

## Does Not Apply When

성능이 극도로 중요한 내부 hot loop — 그땐 검증을 경계에서 한 번만.

## Evidence

StorageLedger.__post_init__ 구현. 관련 [[droad-ledger]], [[reject-bool-str-in-numeric-fields]].
