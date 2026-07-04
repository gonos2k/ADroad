---
title: "dROAD (미분가능 RoadSurf, JAX)"
instance_of: Artifact
page_kind: entity-page
epistemic_status: validated
---

# dROAD (미분가능 RoadSurf, JAX)

## Role

RoadSurf를 JAX로 재구현한 미분가능 패키지. 자료동화(초기상태 보정)와 물리모수 보정을 gradient로 동시에 수행하는 하이브리드 모델링 시스템.

## Key Facts

- 16 모듈: config, branches, ledger, radiation, thermal, boundary, storage, roadcond, model, driver, jax_model, jax_storage, smoothing, assimilate, dual
- 참조 대비 bit-exact(rollout Tsurf RMSE 1.4e-16), 140개 테스트 통과, raw-primitive 감사 clean
- 1차 adjoint(VJP)·2차 Gauss-Newton/HVP·Laplace/Hutchinson UQ·순환 dual estimation
- GitHub: gonos2k/ADroad

## Connections

원본은 [[roadsurf]]. 핵심 감사층은 [[droad-ledger]]. 개념은 [[differentiable-data-assimilation]], [[smooth-compat-mode]], [[cycled-dual-estimation]]. 프레임워크 선택은 [[jax-over-pytorch]].
