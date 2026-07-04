---
title: "저장 함수에 ledger·diagnostics 배선하기"
instance_of: Procedure
page_kind: procedure-page
epistemic_status: validated
---

# 저장 함수에 ledger·diagnostics 배선하기

## Purpose

새 상태변경 저장 스텝을 mass 감사 계약에 맞게 추가한다(residual=0 유지, 진단 분리).

## Preconditions

[[mass-ledger-audit-layer]] 이해. INTERNAL_TRANSFER_KEYS·DIAGNOSTIC_CODES registry 숙지.

## Steps

1. before = _primary(...) 스냅샷, tr={}, ext_src=ext_sink=0.0, diag=[] 초기화
2. 각 분기에서 internal transfer는 tr[key]에, 외부 유출입(condensation/wear/clamp)은 ext_src/ext_sink에 직접 누적(순 delta 사후계산 금지)
3. 물리 주목 이벤트(over-melt·negative-pre-clamp·overflow)는 diag에 상수(DIAG_*)로 append
4. `_phase_ledger(before, after, ice2_in, ice2, tr, ext_src, ext_sink, ice2_reset, events)`
5. `StorageResult(state_next, ledger, tuple(diag))` 반환; 상위 road_cond는 merge_ledgers로 집계

## Postconditions / Verification

정상 케이스 residual<1e-9, 강제 누출 시 residual≠0, transfer 오타는 LedgerError. 관련 [[branch-local-external-accounting]], 절차대상 [[droad-ledger]].
