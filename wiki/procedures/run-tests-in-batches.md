---
title: "dROAD 테스트를 배치로 실행하기"
instance_of: Procedure
page_kind: procedure-page
epistemic_status: validated
---

# dROAD 테스트를 배치로 실행하기

## Purpose

JAX Hessian 테스트가 무거워 전체 pytest가 45s 샌드박스 타임아웃을 넘기므로, 마커로 코어/JAX를 나눠 배치 실행하고 감사 clean을 확인한다.

## Preconditions

pyproject에 pytest 마커(jax, realdata)와 optional-deps(jax[cpu], optax)가 있어야 하고, JAX 테스트 파일 상단에 `pytestmark = pytest.mark.jax`가 선언돼 있어야 한다.

## Steps

1. 감사: `python -c "from tools.check_raw_primitives import find_raw_primitives; print(find_raw_primitives('droad'))"` → `[]` 기대
2. 코어: `pytest -q -m "not jax" -p no:cacheprovider`
3. JAX 경량: smoothing/jax_dry/jax_storage/assimilate 파일 묶음
4. JAX 무거움은 파일별로(second_order 단독 ~40s, dual/real_obs, gauss/uq/multiwindow)
5. `-p no:cacheprovider`로 pycache 오염 방지

## Postconditions / Verification

모든 배치 통과 + 감사 `[]`. 관련 노하우 [[registry-over-literal-strings]], 대상 [[droad]].
