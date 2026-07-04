---
title: "dROAD 최종 보고서"
instance_of: Source
page_kind: source-page
epistemic_status: validated
date_ingested: 2026-07-04
---

# dROAD 최종 보고서

## Summary

RoadSurf를 JAX로 재구현해 자료동화(DA)와 물리모수 보정을 하나의 손실로 동시에 수행하는 미분가능 시스템의 결과 보고. 참조 대비 rollout Tsurf RMSE 1.4e-16(기계정밀도), 쌍둥이·순환 dual estimation 파라미터·상태 추적 성공, 100+ 테스트 통과. 정직한 발견: 실관측 단일 창은 좋은 prior를 못 이김 → forecast skill gate·report-only 정당. 관련: [[droad]], [[cycled-dual-estimation]], [[equifinality-single-site]].
