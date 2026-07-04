---
title: "커플링 제외 (방안 A)로 no-coupling 전체 모델을 대상화"
instance_of: Decision
page_kind: decision-page
date: 2026-07-04
---

# 커플링 제외 (방안 A)로 no-coupling 전체 모델을 대상화

## Context

RoadSurf의 관측 정합은 복사보정계수 반복탐색('커플링') 휴리스틱. 미분가능화·재현 대상 범위를 정해야 함.

## Decision

커플링을 제외한 no-coupling 전체 모델을 dROAD의 재현·미분 대상으로 삼는다(방안 A).

## Rationale

커플링은 비미분 휴리스틱이라 gradient DA의 목적(초기상태·모수 gradient 일반화)과 상충. 제외하면 bit-exact 재현 기준이 명확해지고 DA가 커플링을 원리적으로 대체한다.

## Alternatives Considered

(a) 커플링까지 미분가능 근사 — 복잡도·비물리 위험 큼. (b) 커플링 유지한 부분 재현 — 미분 목적과 충돌.

## Consequences

no-coupling에서 12,959스텝 bit-exact(Tsurf RMSE 1.4e-16) 확보. 원본 [[roadsurf]], 대상 [[droad]].
