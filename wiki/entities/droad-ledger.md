---
title: "droad/ledger.py — mass 감사층"
instance_of: Artifact
page_kind: entity-page
epistemic_status: validated
---

# droad/ledger.py — mass 감사층

## Role

저장항 질량 회계의 계약 모듈. 모든 상태변경 저장 스텝이 StorageResult(state_next, ledger, diagnostics)를 반환하고, 스칼라 헬퍼는 (value, ledger)를 반환한다. residual은 코드 누출 탐지, diagnostics는 물리 신호.

## Key Facts

- StorageLedger: key 검증 → finite/정규화 → event bool → 비음수(flows+primary) → expected/residual 일관성 → mapping/diagnostics 불변(MappingProxyType/tuple)
- merge_ledgers: 자식 residual fail-fast + 연속성(after==next.before) + atol 검증
- 직렬화: ledger_to_dict / storage_result_to_dict / rollout_audit_to_dict
- 모든 bad input(str/bool/array/NaN/Inf/미등록 키·코드)은 LedgerError로 통일

## Connections

개념은 [[mass-ledger-audit-layer]], [[branch-local-external-accounting]], [[residual-vs-diagnostics-separation]]. 소속 패키지는 [[droad]].
