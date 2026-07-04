---
title: "dROAD 설계계획서 v0.7 + P0 실행계약"
instance_of: Source
page_kind: source-page
epistemic_status: validated
date_ingested: 2026-07-04
---

# dROAD 설계계획서 v0.7 + P0 실행계약

## Summary

미분가능 RoadSurf(dROAD)의 청사진. 5라운드 적대적 검토로 위험을 소진한 뒤 확정된 설계서(v0.7)와, 충돌 시 우선하는 실행계약(구현계획_P0_derisked). 핵심 가드레일: compatibility_target×model_mode enum 분리, branch registry(raw-primitive 감사), mass ledger, forecast baseline B0–B3, report-only (단일 사례). 참조 우선(bit-exact) → 미분가능(smooth) 2모드, backend 단계화(NumPy→JAX). 관련: [[droad]], [[differentiable-data-assimilation]], [[no-coupling-exclusion]].
