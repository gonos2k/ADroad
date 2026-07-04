# dROAD 코드 스캐폴딩 (P0)

실행 계약: `구현계획_P0_derisked.md` · 청사진: `dROAD_설계계획서.md` (v0.7)
원칙: **간단·명료·직관적**, backend-neutral NumPy 먼저(JAX는 parity 이후).

## 현재 구현 (착수 1차)
확정된 착수 순서 중 1–3을 구현·테스트 완료:

| # | 모듈 | 내용 | 테스트 |
|---|---|---|---|
| 1 | `droad/config.py` | `compatibility_target × model_mode` 2축 enum + 조합 검증. `paper_physics`는 validation_suite(runtime target 금지) | `tests/test_config.py` |
| 2 | `droad/branches.py` + `tools/check_raw_primitives.py` | site-aware `safe_where/guarded_where/guarded_sqrt` + BRANCH_REGISTRY + AST 기반 raw primitive 금지 감사 | `tests/test_branches.py` |
| 3 | `droad/ledger.py` | `StorageLedger/StorageResult` typed 계약, key set 고정, `merge_ledgers`(residual 재계산) | `tests/test_ledger.py` |

## 실행
```bash
pip install -e ".[dev]"     # 또는: pip install numpy pytest
pytest -q                   # 20 passed
python -c "from tools.check_raw_primitives import find_raw_primitives; print(find_raw_primitives('droad'))"
```

| 4 | `tools/run_no_coupling.py` + `fixtures/no_coupling/` | main.py와 **동일** 결과 재현 드라이버(coupling OFF, matplotlib 불필요) + snapshot hook(purity 확인) + pinned manifest(sha256) | `tests/test_fixture.py` |
| 5(부분) | `droad/radiation.py` | `calc_rnet` — RoadSurf-Python `CalcRNet`과 bit-parity(G0a) | `tests/test_python_compat_radiation.py` |

| 5 | `droad/thermal.py` | `calc_hcap_hcond` + `calc_cap_cond` + `calc_profile` — 지중 열전도 1-step. reference와 TmpNw/VSH/HS **1e-12 parity(G0)** | `tests/test_python_compat_thermal.py` |

| 5b | `droad/boundary.py` (BLC-v0/v1 + LE) + `droad/model.py` (`set_day_dependent`, `balance_one_step_dry`) | dry one-step 전체 조립. reference `BalanceModelOneStep`와 TmpNw abs<1e-9, BLCond/LE **bit-exact** | `tests/test_python_compat_onestep.py` |
| 5c | `droad/driver.py` (`dry_rollout`) | free-running dry rollout(400스텝) vs storage-disabled reference. **Tsurf RMSE 5e-17** (기계정밀도, G1b) | `tests/test_python_compat_dry_rollout.py` |

## 검증 상태
```
pytest -q  →  32 passed
재현 드라이버 vs main.py       →  IDENTICAL
snapshot hook on/off          →  CSV IDENTICAL (purity OK)
calc_rnet vs reference        →  abs < 1e-12 (G0a)
thermal kernel vs reference   →  TmpNw abs < 1e-12 (G0)
BLC / LE vs reference         →  bit-exact (diff 0)
dry one-step vs reference     →  TmpNw abs < 1e-9 (G1a)
dry rollout(400스텝) vs ref    →  Tsurf RMSE 5e-17 (G1b, 기계정밀도)
강수 상변화 vs ref(전 스텝)     →  Rain/Snow abs < 1e-12 (M2a)
wear/water storage vs ref      →  abs < 1e-12 (M2b, 전 레짐)
snow/ice/dep/melt-heat vs ref  →  abs < 1e-12 (M2c, 전 분기)
full step vs ref (전 스텝)      →  TmpNw/저장 abs < 1e-9 (M2d, teacher-forced)
full rollout vs ref (12959스텝)→  Tsurf RMSE 1.8e-16, 저장 5종 diff=0 (bit-exact)
JAX dry rollout vs numpy(v0)   →  abs < 1e-8 (M3)
jax.grad(Emiss) vs 유한차분     →  rel < 1e-4 ; JVP↔VJP dot-product 일치 (M4dry)
쌍둥이 실험(twin) 복원          →  Emiss 0.92000, offset 2.0000 (동시), loss→2e-11
smooth 원시연산 τ→0            →  hard 수렴 + 임계점 grad 유한 (M4)
HVP 대칭성·Newton·Laplace       →  uᵀHv=vᵀHu, Newton 6스텝, cov SPD (§7.4/7.6)
smooth storage rollout         →  dry 환원 1.9e-6, 상변화 통과 grad 유한, 습설 twin 복원 (M4 full MVP)
다중윈도우 결합추정             →  전역 Emiss 0.930 + 윈도우별 offset 동시 복원 (§7.2)
실관측 DA(troad)              →  무제약 과적합 재현 / 제약+정칙화로 물리복원·persistence 개선 (§7.3/§8/§11)
enhanced_enthalpy 모드         →  0°C 잠열 흡수(N10 완화), enth_L=0=base·grad 유한 (§5a)
Gauss-Newton(§7.4)            →  matrix-free JVP∘VJP+CG, 4D 초기프로파일 1e-4 복원, 3 outer 수렴
Hutchinson UQ(§7.6)          →  matrix-free Hessian 대각 추정 = dense 대각
순환 dual estimation(§7.8)    →  모수 0.82→0.97 정착, RMSE 11×↓, 상태보정 13×↓ (equifinality 잔여편향 정직)
NumPy exact-core raw-primitive audit → violations: [] (JAX/smoothing은 allowlist,
                                       별도 안정성 테스트로 감사 — audit 대상 아님)
pytest -q                     →  전부 통과 (배치 권장: pytest -m "not jax" core, -m jax JAX/DA)
```

주의(정직성): `guarded_exp`가 인자를 [-60,60]으로 clip하므로 bit-exact parity는
**fixture 도메인(no-coupling rollout) 내부**에서만 보장된다. 극단 비물리 입력에서
reference(clip 없음)와 갈라질 수 있다 — exact parity는 정상 기상 범위 주장이다.

## 코드 리뷰 반영 (11차: public input 방어 래핑)
비정상 public input을 raw Python 예외 대신 `LedgerError`로 일관 처리: `_normalize_diagnostics`가
None/비이터러블 거부, `_check_keys`가 non-mapping 거부(collections.abc.Mapping), `rollout_audit_to_dict`가
out non-mapping 거부, `merge_ledgers`가 자식 non-StorageLedger 거부. 회귀 테스트 4건 추가.

## 코드 리뷰 반영 (10차: merge fail-fast + 검증 helper 통합)
`merge_ledgers`가 자식 ledger의 residual을 **fail-fast로 검사**(각 |residual|<=atol) — 이전엔 +0.5/-0.5
자식 누출이 상쇄돼 aggregate가 깨끗해 보일 수 있었으나, 이제 개별 자식 누출이 즉시 `LedgerError`.
diagnostics 검증을 `_normalize_diagnostics` 공유 helper로 통합(StorageResult·rollout_audit_to_dict 중복 제거),
`rollout_audit_to_dict`가 step별 diagnostics code까지 registry 검증. `make_ledger`가 산술 전에 scalar를
float로 정규화(모든 bad input → LedgerError).

## 코드 리뷰 반영 (9차: 타입 방어 마무리)
`_as_finite_float`가 **bool**(numpy bool 포함)을 numeric field에서 거부 — `True`가 조용히 1.0으로
coerce되는 것을 차단. `merge_ledgers`가 `atol`을 finite·non-negative 검증(NaN atol이 continuity check를
우회하는 것 방지). `rollout_audit_to_dict`가 `ledger`/`ledger_detail` 항목이 실제 `StorageLedger`인지 검증.
`StorageResult`가 diagnostic code가 str인지 검증.

## 코드 리뷰 반영 (8차: public helper 방어성 완성)
`_require_finite`를 `_as_finite_float`로 바꿔 모든 numeric field를 **plain float로 정규화**(str/array/NaN/Inf
거부) — 이후 `<0`·산술이 어떤 scalar 타입(numpy/jax/int)에서도 TypeError 없이 동작. `make_ledger`의
선행 산술을 `LedgerError`로 래핑. `StorageResult`가 bare string diagnostic을 문자 분해 대신 `(str,)`로 래핑.
`rollout_audit_to_dict`가 `ledger_detail` 각 항목이 2-tuple인지 검증. 모든 bad input이 일관되게 `LedgerError`.

## 코드 리뷰 반영 (7차: 감사 record 방어성 마무리)
`StorageLedger`가 `event_flags` 값이 실제 bool인지 검증(numpy bool 허용·Python bool로 정규화 —
문자열 "False"의 truthy 오염 차단)하고, `primary_before`/`primary_after_actual`의 **비음수**를 검증
(mass state는 음수 불가). `_require_finite`는 비스칼라 입력의 `float()` 예외를 `LedgerError`로 래핑.
unknown key/diagnostic 오류의 `sorted`는 `key=str`로 혼합 타입 안전. `rollout_audit_to_dict`는
누락 키·길이 불일치를 명시적으로 거부.

## 코드 리뷰 반영 (6차: finite 검증 + 불변화 마무리)
`StorageLedger.__post_init__`가 모든 numeric scalar(primary/external/transfer/aux)에 **finite 검증**
(`math.isfinite`)을 추가 — NaN/Inf는 `<0`·`|·|>tol` 비교를 통과하므로 감사 레이어에서 명시적으로 거부.
`StorageResult`가 `diagnostics`를 tuple로 정규화·freeze(list 전달도 불변). `rollout_audit_to_dict()`로
full_rollout 감사 trail 전체를 JSON화. 테스트는 `DIAG_*` 상수 사용으로 전환(문자열 값 계약은 직렬화
테스트에서 고정).

## 코드 리뷰 반영 (5차: 진단 polish)
`diagnostics`를 registry화 — `DIAGNOSTIC_CODES` + 상수(`DIAG_*`)로 정의하고 `StorageResult.__post_init__`가
미등록 코드를 `LedgerError`로 거부(transfer key와 동일한 엄격도). storage 분기는 literal 대신 상수 사용.
`water_storage`도 hard-projection hit(`water_overflow`/`water_negative_pre_clamp`)을 진단으로 반환(3-tuple),
`road_cond`가 water reclamp 진단까지 집계. `storage_result_to_dict()` 직렬화(ledger+diagnostics) 추가.
`full_rollout(return_ledger=True)`의 shape/key/length/step residual을 직접 고정하는 테스트 추가.

## 코드 리뷰 반영 (4차: diagnostic richness)
residual(=코드 누출 탐지)과 **분리된** 물리 feasibility 진단을 `StorageResult.diagnostics`에
추가: over-melt(available 초과 융해), negative-pre-clamp, overflow(hard-projection hit).
mass 회계에는 영향 없음(clamp import/export가 external로 기록되어 residual은 여전히 0).
`road_cond`가 자식 진단을 집계하고 `step_full`이 `diagnostics`로 노출. `full_rollout(return_ledger=True)`가
이제 step마다 merged `step_ledger`(+`ledger_detail`, `diagnostics`)를 저장 — step_full과 감사 API 정합.

## 코드 리뷰 반영 (3차: ledger 감사화)
`_phase_ledger`가 external source/sink를 net delta로 후처리하지 않고 **branch 지점에서
직접 누적** — residual이 이제 "설명 안 된 mass leak" 탐지기(정상=0, 누락 시 ≠0). transfer
typo는 `LedgerError`로 거부, `StorageLedger`는 불변(MappingProxy)+expected/residual
일관성·비음수 검증, `merge_ledgers`는 child 연속성 검증. `ice2_reset` 실제 사용(forceIceMelting).
`step_full`에 `step_ledger=merge_ledgers(prec, cond)`. `laplace_cov` 고유값 shift로 실제 SPD 보장.
`_blc_v0` Kelvin denominator 가드. 계약 문구 정정(StorageResult/tuple 구분, smooth_compat는
smooth gate+hard projection, `__all__` 정리). adversarial 회귀 테스트(누출 탐지·typo·불변·연속성·
enth_dT=0·PLim 역전) 추가. **114 passed**.

## 코드 리뷰 반영 (총평 하드닝)
`pyproject`에 `jax`/`dev` extras(jax[cpu]·optax)와 pytest 마커(`jax`,`realdata`) 추가 —
`pytest -m "not jax"`로 순수 NumPy 코어만 실행 가능. `smoothing.gate` τ 하한·인자 클립·`jax.nn.sigmoid`,
`soft_clip` 추가. `jax_model` 포화수증기 `exp` 인자/분모 가드, `branches.guarded_exp` 오버플로 클립.
`assimilate.fit` best-control 추적, `newton`/`laplace_cov` Hessian 대칭화+damping.
`dual_estimation` `theta_before/after` 분리·선택적 sigmoid 범위제약. `thermal` 입력 shape 검증.
저장 계약 관통: `snow/ice/deposit_storage`→`StorageResult`, `road_cond`가 `merge_ledgers`로 집계,
`step_full`이 `precipitation_to_storage` 사용.
reference/RoadSurf-Python은 commit 61b5ee1 사본. fixture 재생성: `python tools/run_no_coupling.py`.

| 6(M2a) | `droad/storage.py` (`calc_prec_type`, `precipitation_to_storage`) | 강수 상변화(eq 42 시그모이드)+저장 투입. 전 스텝(강수 포함) reference와 1e-12 일치, ledger 질량보존 | `tests/test_python_compat_storage.py` |
| 6(M2b) | `droad/storage.py` (`wear_factors`, `water_storage`) | 교통 마모계수 + 물 저장(증발·응결·마모·클램프). 전 레짐 합성케이스 reference와 1e-12 일치 | `tests/test_python_compat_water_storage.py` |
| 6(M2c) | `droad/storage.py` (`Surf`, `snow_storage`, `ice_storage`, `deposit_storage`, `new_melt_freeze_heat`) | 눈/얼음/서리 저장 + 상변화(동결·융해·마모·전환·클램프) + 융해열. 전 분기 케이스 reference와 1e-12 | `tests/test_python_compat_snow_ice_dep.py` |
| 6(M2d) | `droad/storage.py` (`melting`, `calc_albedo`), `droad/thermal.py` (`calc_hstor`), `droad/roadcond.py` (`road_cond`), `droad/model.py` (`step_full`), `droad/driver.py` (`full_rollout`) | 융해(지층 결합)·알베도·RoadCond 조립·전체 스텝·전체 롤아웃 | `tests/test_python_compat_full_step.py`, `tests/test_python_compat_full_rollout.py` |
| 7(M3+M4dry) | `droad/jax_model.py` (`dry_rollout`, `loss`, `make_dry_step`) | **미분가능 JAX dry rollout**(jnp+`lax.scan`+`fori_loop`, 가드 분기). numpy(BLC-v0)와 parity, `jax.grad`(Emiss·초기프로파일), JVP/VJP dot-product | `tests/test_jax_dry.py` |
| 8(DA) | `droad/assimilate.py` (`fit`, optax+jit) | **쌍둥이 실험**: 파라미터 보정(Emiss)·초기상태 동화(offset)·**동시 추정** 모두 gradient로 복원 | `tests/test_assimilate.py` |
| 9(M4 primitives) | `droad/smoothing.py` (`gate`, `select`, `soft_min/max`, `transfer`, `ceff`) | smooth_compat 원시연산: τ→0 hard 수렴 + 임계점 유한 grad + 질량보존 + 엔탈피 에너지적분 | `tests/test_smoothing.py` |
| 10(2차/UQ) | `droad/assimilate.py` (`hvp`, `newton`, `laplace_cov`) | HVP(forward-over-reverse)·Newton(6스텝 복원)·Laplace 공분산. HVP 대칭성·HVP=dense Hessian 검증 | `tests/test_second_order.py` |
| 11(M4 full MVP) | `droad/jax_storage.py` (`rollout`, smooth 강수·물·눈·얼음·상변화·알베도) | **미분가능 smooth_compat storage rollout**. dry 환원(<1e-5), 한파서 상변화 활성, phase 통과 grad 유한, 습설 모델로 Emiss 복원 | `tests/test_jax_storage.py` |
| 12(§7.2 결합추정) | `droad/assimilate.py`(`fit`) + `vmap` | **다중윈도우 결합추정**: 윈도우별 초기상태 + 전역 물리모수 동시 복원(Emiss 0.930, offs 오차<3e-2). 전역 grad가 모든 윈도우 집계 | `tests/test_multiwindow.py` |
| 13(실관측 DA) | 실제 `troad` 관측 + baseline | **힌드캐스트 변분 DA**: 무제약은 과적합(Emiss>1, 예측 악화); **범위제약+정칙화**로 물리 복원(0.997≤1)+persistence 개선. 단일창은 기본prior 못 이김→§11 report-only | `tests/test_real_obs_da.py` |
| 14(enhanced_enthalpy) | `droad/jax_storage.py` (지층 열용량 + `smoothing.ceff` 잠열항) | **N10 노면온도 진동 완화 모드**(§5a): 0°C 부근 유효 열용량에 잠열 흡수. enth_L=0=base, >0=physics-changing, grad 유한 | `tests/test_jax_storage.py` |
| 15(§7.4 GN) | `droad/assimilate.py` (`gauss_newton`) | **matrix-free Gauss-Newton / 증분 4D-Var**: JVP∘VJP + CG(J 미형성). 4차원 초기프로파일 1e-4 복원, 3 outer 수렴 | `tests/test_gauss_newton.py` |
| 16(§7.6 UQ) | `droad/assimilate.py` (`hutchinson_diag`) | **matrix-free 확장형 UQ**: HVP+Hutchinson Hessian 대각 추정(H 미형성). dense 대각과 일치 | `tests/test_uq_hutchinson.py` |
| 17(§7.8 dual) | `droad/dual.py` (`dual_estimation`) + `jax_model.dry_rollout_carry` | **순환 dual estimation**: 주기마다 상태(빠름) 분석 + 모수(느림) 갱신, 예보로 다음 배경 연결. 모수 추적·정착, 윈도우 misfit·상태보정 감소 | `tests/test_dual.py` |

## 상태
- **M1 dry thermal core 완료** — dry 물리 궤적 전체 bit-exact.
- **M2a 강수 상변화 완료** — PrecPhase 결측→시그모이드 경로, 전 스텝 parity.

## 상태: M1+M2 exact-mode 완료 ✅
RoadSurf **no-coupling 전체 모델**(dry thermal + storage/상변화)이 droad에서 **bit-exact 재현**(12959스텝 rollout: Tsurf RMSE 1.8e-16, 저장 5종 diff=0).

## 상태: M3 + M4(dry) 착수 완료
미분가능 JAX **dry rollout** 작동 — numpy parity, `jax.grad`(Emiss·초기프로파일), JVP/VJP dot-product, `jit`.

## 상태: M4 full smooth_compat MVP 완료
강수·상변화 구간까지 미분가능 rollout 작동(dry 환원 1.9e-6, 상변화 통과 grad, 습설 twin 복원). *MVP: 물/눈/얼음 + 상변화·알베도. deposit/ice2·엔탈피 정밀화·G3 deviation budget 정량화는 후속.*

## 다음
- **다중 사례·다중 지점 데이터**로 forecast skill gate promotion + dual estimation의 equifinality 편향 완화(관측 다양성).
- **정밀화**: smooth storage deposit/ice2·에너지-제한 융해; hybrid-4DEnVar·B^½ 전처리(§7.7); dual estimation을 storage/smooth 모델로 확장.
