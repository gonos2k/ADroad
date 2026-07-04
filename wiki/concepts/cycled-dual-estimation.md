---
title: "순환 dual estimation"
instance_of: Concept
page_kind: concept-page
epistemic_status: validated
---

# 순환 dual estimation

## Definition

매 동화 주기에서 상태(빠름, per-cycle 변분 분석)와 모수(느림, 주기 간 gradient 갱신)를 교대 추정하고, 예보 end-state를 다음 주기의 배경으로 잇는 순환(cycling) 하이브리드 DA(§7.8).

## Why It Matters

일괄 결합최적화를 넘어 명실상부한 cycling 자료동화 체계로 승격. 저차원 상태 제어 + 배경 정칙화로 모수 식별성을 유지한다.

## Current Understanding

쌍둥이에서 모수 0.82→0.97 정착, 윈도우 misfit 11×↓, 상태보정 13×↓. 잔여 편향(~0.04)은 [[equifinality-single-site]] 한계. 상위 개념 [[differentiable-data-assimilation]], 구현 [[droad]].
