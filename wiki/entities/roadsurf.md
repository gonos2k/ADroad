---
title: "RoadSurf (FMI Fortran 모델)"
instance_of: Artifact
page_kind: entity-page
epistemic_status: observed
---

# RoadSurf (FMI Fortran 모델)

## Role

FMI의 도로 노면 기상 모델(Fortran). 노면온도와 저장항(물·눈·얼음·서리)을 예측하고, 관측 정합을 복사보정계수 반복탐색('커플링') 휴리스틱으로 처리한다. dROAD의 재현 기준.

## Key Facts

- 다층 지중 열전도(명시적 Euler), 경계층 conductance(반복 BLC 해), 순복사, 저장/상변화
- 포팅 기준 구현: RoadSurf-Python(commit 61b5ee1)
- dROAD는 '방안 A'로 커플링을 제외한 no-coupling 전체 모델을 대상으로 함

## Connections

물리 정의는 [[karsisto-2024]]. 미분가능 재구현은 [[droad]]. 커플링 제외 결정은 [[no-coupling-exclusion]].
