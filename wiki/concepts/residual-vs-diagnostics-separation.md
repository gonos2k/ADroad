---
title: "residual↔diagnostics 역할 분리"
instance_of: Concept
page_kind: concept-page
epistemic_status: validated
---

# residual↔diagnostics 역할 분리

## Definition

두 감사 신호를 명확히 분리: primary_mass_residual은 '코드가 mass 변화를 장부에 기록했는가'(회계 누출), diagnostics는 '기록은 됐지만 물리적으로 주목할 일이 있었는가'(over-melt·negative-pre-clamp·overflow).

## Why It Matters

over-melt 같은 reference quirk를 residual 실패로 만들지 않으면서, exact-compat과 deviation-budget을 동시에 만족. 잘못 섞으면 '항상 맞는 장부'가 되거나 정상 동작을 오탐한다.

## Current Understanding

diagnostics는 DIAGNOSTIC_CODES registry로 오타를 거부하고, residual과 독립적으로 StorageResult에 실린다. 핵심 노하우는 [[residual-is-code-leak-not-physics]]. 구현은 [[droad-ledger]].
