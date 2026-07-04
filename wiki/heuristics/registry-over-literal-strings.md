---
title: "키·코드는 literal이 아니라 registry로 관리한다"
instance_of: Heuristic
page_kind: heuristic-page
epistemic_status: validated
confidence: high
---

# 키·코드는 literal이 아니라 registry로 관리한다

## Rule

transfer key와 diagnostics code는 자유 문자열 대신 고정 registry(상수+frozenset)로 정의하고, 미등록 값은 즉시 거부한다. 오류 메시지 정렬은 sorted(key=str)로 혼합 타입에 안전하게.

## Why

오타가 조용히 드롭되면 누락을 residual/진단이 못 잡는다. registry는 author-time·runtime 양쪽에서 오타를 차단한다.

## Applies When

고정된 어휘(enum성)를 갖는 감사/진단 필드.

## Does Not Apply When

어휘가 열려 있고 확장이 잦아 검증 비용이 이득을 넘을 때.

## Evidence

INTERNAL_TRANSFER_KEYS/DIAGNOSTIC_CODES + _normalize_diagnostics. 관련 [[droad-ledger]].
