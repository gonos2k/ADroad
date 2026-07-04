---
title: "프레임워크로 PyTorch 대신 JAX 채택"
instance_of: Decision
page_kind: decision-page
date: 2026-07-04
---

# 프레임워크로 PyTorch 대신 JAX 채택

## Context

작은 상태·긴 시간루프 모델의 미분가능화. 동적 연산그래프 필요성도 검토.

## Decision

JAX를 채택한다(jit+scan, vmap, 암시적 미분 생태계).

## Rationale

작은 상태·긴 rollout에는 jit+lax.scan이 최적이고, VJP/JVP/HVP·custom_vjp·linearize/linear_transpose로 matrix-free 2차 최적화가 자연스럽다. 동적 그래프 이점은 이 모델 규모에서 결정적이지 않음.

## Alternatives Considered

PyTorch — 동적 그래프·생태계 크지만 긴 시간루프 scan/jit 최적화·함수형 변환이 상대적으로 번거로움.

## Consequences

jax_enable_x64로 기계정밀도 parity 확보, matrix-free GN/HVP·Hutchinson UQ 구현. 대상 [[droad]], 개념 [[differentiable-data-assimilation]].
