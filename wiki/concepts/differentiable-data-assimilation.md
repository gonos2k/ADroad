---
title: "미분가능 하이브리드 자료동화"
instance_of: Concept
page_kind: concept-page
epistemic_status: validated
---

# 미분가능 하이브리드 자료동화

## Definition

물리 모델을 미분가능화(JAX)해 초기상태 보정(변분 DA)과 물리모수 보정(calibration)을 하나의 스칼라 손실로 동시에 gradient 최적화하는 방식. adjoint(VJP)로 gradient 비용이 제어변수 차원과 무관.

## Why It Matters

RoadSurf의 휴리스틱 커플링을 gradient 기반 일반화로 대체. state와 parameter를 결합 추정해 관측 정합과 물리 신뢰성을 함께 확보한다.

## Current Understanding

1차(adjoint/TLM)·2차(HVP·matrix-free Gauss-Newton)·UQ(Laplace/Hutchinson)까지 실증. 단일 지점·단일 사례에서는 state-parameter [[equifinality-single-site]] 한계가 있어 report-only. 구현체는 [[droad]]의 assimilate/dual 모듈. 상태·모수 동시추적은 [[cycled-dual-estimation]].
