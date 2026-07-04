---
status: execution_contract
authoritative_for: [implementation_order, gates, ci, config_schema]
blueprint: dROAD_설계계획서.md   # v0.7
backend_path: numpy_to_jax
supersedes_on_conflict: dROAD_설계계획서.md
---

# dROAD 구현계획 — De-risked 첫 구현 (P0)

> 목적: 설계서(`dROAD_설계계획서.md`, **v0.7**)는 **최종 청사진**이다. 이 문서는 외부 적대적 검토를 반영해 **"실패하지 않는 첫 구현 계획"**을 청사진에서 분리한다. **구현 순서·gate·CI·config schema는 본 문서가 청사진보다 우선한다.**
> 원칙: **forward parity를 먼저 고정**하고, JAX 고급기능(jit/scan/custom_vjp)·2차 최적화·EnVar·hybrid는 parity가 잠긴 뒤 단계적으로 켠다.
> **우선순위 계약**: 구현 순서·gate가 `dROAD_설계계획서.md`(청사진)와 충돌하면 **이 문서가 우선한다.**
> 작성일: 2026-07-02

## 검토 수용 요약
외부 검토의 핵심 4대 실패지점에 **전면 동의**하고 아래로 해소한다:
1. JAX-first + jit/scan/custom_vjp를 너무 빨리 넣음 → **백엔드 단계화**(§8) + BLC 3단계(§5).
2. Python 기준과 Fortran 기준이 `exact` 안에서 충돌 → **compatibility_target 분리**(§1).
3. `smooth_compat`와 `enhanced_enthalpy`가 한 모드에 섞임 → **모드 3분할**(§2).
4. M7(GN/EnVar/UQ/FSOI)이 한 마일스톤에 과밀 → **M7a–f 분해**(§9), MVP 축소(§7).

**전문가 보강(검토를 더 다듬은 지점):**
- **G1 허용오차 근거 정정**: `jit`/`scan`은 결과를 거의 안 바꾸고 `remat`은 **비트 동일**(같은 연산 재계산)이다. 오차 완화는 이들이 아니라 **(a) reduction 연산순서, (b) custom_vjp/implicit로 알고리즘이 바뀌는 노드**에서 온다. 따라서 허용오차는 "백엔드"가 아니라 **"알고리즘 변경 여부"** 기준으로 매긴다(§6).
- **τ→0 수렴은 primitive 단위로만 검증**: end-to-end 롤아웃은 상변화 캐스케이드·히스테리시스 때문에 τ→0에서 exact로 수렴 안 할 수 있다. 각 **매끄러운 원시연산 단위**로 수렴을 보이고, 롤아웃은 deviation/skill로 평가(§2, §6).
- **식별성은 휴리스틱 규칙 + 정량 진단 병행**: day/night active-set 규칙(검토 §12)에 더해, **Gauss–Newton Hessian $J^\top J$의 스펙트럼·상관행렬**로 near-singular 방향(equifinality)을 정량 검출(§10).
- **B^{1/2}(정적 대각) 전처리는 GN과 함께 유지**: 저비용이므로 멀리 미루지 않되, 앙상블 B만 후순위(§9 M7c).

---

## 1. compatibility_target 분리 (CI gate 명확화)
`exact` 한 단어가 Python-exact인지 Fortran-exact인지 흔들리는 문제를 없앤다.
```yaml
reference_policy:
  default: python_primary_fortran_audited

compatibility_targets:          # ← "무엇과 일치하나" (mode 아님)
  python_compat:
    must_match: [roadsurf_python_no_coupling_fixture]
    may_deviate_from: [fortran_if_documented_python_port_gap]
  fortran_compat:
    must_match: [fortran_golden_fixture]
    may_deviate_from: [python_if_documented_port_gap]
  paper_physics:
    not_a_bitwise_target: true

model_modes:                    # ← "어떤 물리/근사로 도나" (target 아님)
  roadsurf_exact:
    purpose: hard-branch baseline compatibility
  smooth_compat:
    purpose: differentiable surrogate of current compatibility target
    must_match: [deviation_budget, gradient_contract]
    must_not_break: [physics_diagnostics]
  enhanced_enthalpy_v1:
    purpose: physics-changing candidate
    evaluated_by: [physics_diagnostics, unseen_forecast_skill]
```
> `smooth_compat`·`enhanced_enthalpy_v1`은 **target이 아니라 mode**다. config schema에서 target 필드에 넣으면 `test_target_mode_combo_valid` 실패.
**테스트 네이밍 규약**: `test_python_compat_*`, `test_fortran_compat_*`, `test_paper_physics_*`, `test_smooth_deviation_*`, `test_forecast_skill_*`. `exact`라는 모호한 접두사 금지.

**enum 통일 (no-go #9) — 두 축 분리.** "타깃"과 "모드"를 섞지 않는다:
```yaml
compatibility_target: [python_compat, fortran_compat]           # runtime target은 둘뿐
validation_suite:     [paper_physics, smooth_deviation, forecast_skill]
model_mode:           [roadsurf_exact, smooth_compat, enhanced_enthalpy_v1]
```
**paper_physics는 runtime target이 아니라 validation suite (no-go #2).** "paper는 bitwise 대상 아님"과 "exact 모델을 paper case에서 부호·flux·보존 검증한다"가 동시에 성립해야 한다:
```yaml
compatibility_target: [python_compat, fortran_compat]   # runtime target은 둘뿐
validation_suite:     [paper_physics, smooth_deviation, forecast_skill]
target_mode_combo:
  paper_physics:
    allowed_as_runtime_target: false
    allowed_as_validation_suite: true
    compatible_model_modes: [roadsurf_exact, smooth_compat, enhanced_enthalpy_v1]
```
허용 runtime 조합: `python_compat×roadsurf_exact`, `fortran_compat×roadsurf_exact`, `python_compat×smooth_compat`, `python_compat×enhanced_enthalpy_v1`.
금지: `smooth_compat×enhanced_enthalpy_v1`(물리변경 혼합), paper_physics를 **runtime target으로** 지정. CI: `test_target_mode_combo_valid`, `test_paper_physics_not_runtime_target`, `test_roadsurf_exact_runs_in_paper_suite`.

## 2. 모드 3분할 (검증이 꼬이지 않게)
```text
roadsurf_exact      : 하드 분기 + 원본 잠열 한계. python/fortran compat 대상.
smooth_compat       : exact와 물리의도 동일, 미분가능 근사만. deviation budget + (의미있는 곳)τ→0 수렴.
enhanced_enthalpy   : 물리 변경 모드(지층 잠열 반영). τ→0 exact 수렴 요구 안 함. 물리진단+미관측 예보스킬로 평가.
```
- **엔탈피법은 `smooth_compat`가 아니라 `enhanced_enthalpy_v1`**이다. exact로 수렴하면 물리개선도 사라지므로 수렴을 요구하지 않는다.
- τ→0 수렴은 **매끄러운 원시연산 단위**로만 검증(캐스케이드/히스테리시스 개입 시 롤아웃 수렴 비보장).

## 3. mass ledger 정의 (보존 테스트가 틀리지 않게)
```text
primary_mass    = water + snow + ice + deposit
auxiliary       = ice2                     # hazard proxy, 기본 primary에 미포함
external_source = rain + snow_precip + condensation
external_sink   = evaporation + wear_export + overflow_export
diagnostic_total= primary_mass + w_ice2 * ice2   # w_ice2 기본 0
```
process별 원장(예):
```text
precipitation      : primary += precip_input
evaporation        : primary -= evap_export
freezing           : water→ice,  primary 불변
snow→ice (wear)    : snow→ice,   primary 불변 / ice2는 auxiliary 갱신(중복합산 금지)
clipping/overflow  : primary 변화 = external_export (수치오차 아님)
deposit_overflow   : deposit→water 또는 external (exact 타깃에 맞춤)
```
→ 검증은 **`primary_mass_residual`과 `auxiliary(ice2) residual`을 분리**해서 본다. "총수분 보존 잔차" 단일 지표 금지.

**target-dependent 모호성 제거 (no-go #7).** "exact 타깃에 맞춤" 같은 표현 금지 — target별 원장 테이블을 코드 상수로 박는다:
```yaml
ledger_policy:
  python_compat:       {deposit_overflow: deposit_to_water,            snow_overflow: external_export, ice2_primary_weight: 0}
  fortran_compat:      {deposit_overflow: match_fortran,               snow_overflow: match_fortran,   ice2_primary_weight: 0}
  enhanced_enthalpy_v1:{deposit_overflow: physically_constrained_xfer, snow_overflow: external_export, ice2_primary_weight: 0}
```
**모든 storage step은 상태 + 원장을 함께 반환하되, dict가 아니라 typed dataclass 계약**(no-go #5 — ledger를 진단 출력이 아니라 버그 차단 타입으로):
```python
@dataclass(frozen=True)
class StorageLedger:
    primary_before: Array; external_source: Array; external_sink: Array
    internal_transfer: Mapping[str, Array]; auxiliary_update: Mapping[str, Array]
    primary_after_expected: Array; primary_after_actual: Array
    primary_mass_residual: Array; event_flags: Mapping[str, Array]

@dataclass(frozen=True)
class StorageResult:
    state_next: State; ledger: StorageLedger
```
반환 타입 고정(전 storage 함수): `precipitation_to_storage/water_storage/snow_storage/ice_storage/deposit_storage/road_cond_storage → StorageResult`.

**내부 dict key set 고정 (no-go #5)** — 느슨한 Mapping은 핵심 transfer 누락을 residual=0으로 숨긴다. key를 상수로 고정하고 해당 없는 process는 0을 채운다:
```python
INTERNAL_TRANSFER_KEYS = ("water_to_ice","ice_to_water","snow_to_water","snow_to_ice","deposit_to_water")
AUXILIARY_UPDATE_KEYS  = ("ice2_increase","ice2_decrease","ice2_reset")
EVENT_FLAG_KEYS        = ("freeze_event","melt_event","snow_event","deposit_melt_event")
```
**aggregation 계약 (no-go #4)** — `road_cond_storage`가 child ledger를 합칠 때 규칙을 API로 고정. **핵심: child residual을 단순 합산하지 말고, 마지막 actual state 기준으로 residual을 재계산**:
```python
def merge_ledgers(*ls: StorageLedger) -> StorageLedger:
    primary_before = ls[0].primary_before
    external_source = sum(l.external_source for l in ls)
    external_sink   = sum(l.external_sink   for l in ls)
    internal_transfer = sum_by_required_keys(INTERNAL_TRANSFER_KEYS, ls)
    auxiliary_update  = sum_by_required_keys(AUXILIARY_UPDATE_KEYS,  ls)
    primary_after_actual   = ls[-1].primary_after_actual
    primary_after_expected = primary_before + external_source - external_sink
    primary_mass_residual  = primary_after_actual - primary_after_expected   # 재계산
    event_flags = or_by_required_keys(EVENT_FLAG_KEYS, ls)
    return StorageLedger(...)
```
CI: `test_storage_step_returns_typed_ledger`, `test_storage_ledger_has_all_required_keys`, `test_storage_ledger_rejects_unknown_keys`, `test_merge_ledgers_preserves_required_keys`, `test_merge_ledgers_recomputes_primary_expected_not_sum_residuals`, `test_merge_ledgers_event_flags_are_boolean_or_thresholded`, `test_primary_mass_residual_within_budget`, `test_ledger_identity(primary_after_expected==actual ±budget)`.

**ledger_policy 항목별 oracle (no-go #6)** — policy가 "모델을 합리화하는 원장"이 되지 않게 실제 RoadSurf-Python 동작으로 검증:
```text
test_python_compat_deposit_overflow_policy      # deposit→water 확인
test_python_compat_snow_overflow_policy         # external_export vs heuristic 확인
test_python_compat_ice2_not_primary_mass        # ice2_primary_weight=0
test_fortran_compat_policy_marked_unavailable_without_fortran_fixture
```

## 4. JAX safe branch policy (jit 후 NaN 추적 가능하게)
`where`는 selection일 뿐, 선택 안 된 branch도 계산되어 `sqrt(-)`,`log(≤0)`,0-나눗셈,`exp` overflow를 낼 수 있다(예: 경계층 불안정식 $\sqrt{1-16\zeta}$).
```text
safe_where     : 두 branch가 모든 입력에서 수학적으로 유효 (선형혼합, bounded clip)
guarded_where  : branch 입력을 sanitize 후 계산 (sqrt(max(arg,eps)), log(max(arg,eps)))
lax_cond       : 활성영역 밖에서 계산 불가한 branch (초기엔 비벡터화 허용)
custom_smooth  : exact branch를 전영역 유효한 매끄러운 식으로 대체
```
모든 분기는 위 4등급 중 하나로 **명시 태깅**한다.

**registry + CI gate (no-go #5) — 문서 문구가 아니라 강제 테스트로.**
```python
BRANCH_REGISTRY = {
    "boundary_layer.psi_unstable":   "guarded_where",   # sqrt(1-16ζ)
    "boundary_layer.raero_cap":      "safe_where",
    "heat_capacity.water_ice_props": "safe_where",
    "storage.freeze_gate":           "custom_smooth",
    "storage.melt_gate":             "custom_smooth",
    "albedo.snow_ice_switch":        "custom_smooth",
    "calc_le.no_water_gate":         "custom_smooth",
    "wearfactors.snow_lt_0p2":       "custom_smooth",
}
```
**wrapper 강제 + raw primitive 금지 (no-go #3).** registry는 문서가 아니라 **코드 우회 불가**여야 한다. core에서 `jnp.where`/`lax.cond` 직접 호출 금지, 반드시 site를 받는 wrapper 경유:
```python
def guarded_where(site, cond, x, y):
    assert_branch_registered(site, "guarded_where"); return jnp.where(cond, x, y)
def safe_where(site, cond, x, y):
    assert_branch_registered(site, "safe_where");    return jnp.where(cond, x, y)
```
**audit 범위는 where/cond를 넘어 domain 위험 primitive까지 (no-go #3).** storage·boundary core에서 `jnp.clip/maximum/minimum/sqrt/log/exp`도 domain·gradient 문제를 낸다. 전부 금지가 아니라 `sqrt/log/exp`는 `guarded_*`/`safe_math` wrapper 경유:
```python
def guarded_sqrt(site, x, eps):
    assert_branch_registered(site, "guarded_math"); return jnp.sqrt(jnp.maximum(x, eps))
```
필수 테스트: `test_all_branch_sites_registered`, `test_no_raw_jnp_where_in_core`(AST/grep), `test_no_raw_lax_cond_in_core`, `test_no_raw_jnp_clip_in_storage_core`, `test_no_raw_jnp_sqrt_log_exp_in_boundary_core`, `test_safe_math_wrapper_requires_registered_site`, `test_branch_wrapper_requires_registered_site`, `test_branch_registry_has_no_dead_entries`, `test_guarded_where_no_nan_on_inactive_branch`, `test_boundary_layer_stable_unstable_boundary`, `test_softclip_extreme_sharpness_finite`, `test_storage_near_zero_finite_grad`. registry 누락·raw 사용 시 CI 실패.

## 5. BLC 반복솔버 3단계 (초반 최대 함정 회피)
custom_vjp를 처음부터 넣지 않는다(고정점 수렴·유일성·조건수·분기안정·tol 분리가 모두 필요).
```text
BLC-v0 : 고정 40회 unroll, 조기종료 없음      → gradient smoke test 전용(느리지만 투명)
BLC-v1 : exact forward + 조기종료, backward는 stop_gradient 또는 유한차분 감사 → parity 타깃
BLC-v2 : custom_vjp IFT                        → v0·v1 일치 확인 후 dot-product test 통과 시 승격, 실패 시 unroll fallback
```
→ custom_vjp는 M3가 아니라 **M4/M5 이후 성능·안정화 작업**.

**BLC-v1의 `stop_gradient`를 DA 경로에 넣지 말 것 (no-go #8).** v1은 parity 전용이며 미분 그래프에 들어가면 경계층 민감도가 사라져 DA가 왜곡된다.
```yaml
blc_exact_early_stop:            # BLC-v1
  differentiable: false
  allowed_in:  [python_compat_forward, parity_report]
  forbidden_in: [smooth_da, graph_jvp, loss_vjp, gnvp]
```
CI: `test_benchmark_parity_uses_blc_v1`, `test_da_smooth_forbids_blc_v1`, `test_jvp_vjp_uses_blc_v0_or_v2_only`.

**마일스톤 M1c/M1d BLC 계약 (no-go #7 — 구현자가 mode를 다시 판단하지 않게):**
```text
M1c (BLC no-LE):
  implementation: BLC-v0 fixed 40 unroll
  reference:      BLC-v1 exact early-stop (RoadSurf-Python 방식)
  required: [local BLCond diff <= tol, sensible-heat sign tests, smooth path jvp finite]
  forbidden:[custom_vjp, BLC-v1 in DA path]
M1d (BLC + LE):
  implementation: BLC-v0 fixed 40 unroll + LE
  reference:      RoadSurf-Python local cases
  required: [water-availability gate tests, RH 0/100 finite, warm-air/cold-road & cold-air/warm-road flux sign tests]
```

## 6. gate 등급 (레짐×알고리즘 단계별 허용오차)
```text
G0a scalar formula, 동일 수식 재사용   : abs err <= 1e-12
G0b one-step kernel, 독립 재구현        : abs err <= 1e-9  또는 rel err <= 1e-10  # 연산순서 차이 흡수
G1a dry microfixture, 고정 BLCond       : Tsurf RMSE <= 1e-7
G1b full dry (BLC/LE 포함)              : Tsurf RMSE <= 1e-6
    custom_vjp/implicit 도입 후          : RMSE <= 1e-5   # 알고리즘 변경 기준(백엔드 아님)
G2  exact storage event
    event 시퀀스 정확 일치 + storage MAE <= 1e-3 mm + Tsurf RMSE <= 1e-2 °C
G3  smooth surrogate (deviation budget)
    dry RMSE vs exact <= 0.03 °C
    wet no-phase water MAE <= 0.05 mm
    freeze/melt event shift <= 1 output step   # event 정의 아래
    primary_mass residual <= 아래 수치
```
**G3 event 정의 (no-go #6) — smooth에서 event는 연속 gate이므로 명시한다:**
```text
freeze_event_exact : water→ice transfer > 0  (또는 exact 동결 branch true)
freeze_event_smooth: freeze_transfer_mm > max(0.01 mm, 0.01*water_before)  또는 (freeze_gate>0.5 & water_before>min_water)
melt_event_exact   : snow/ice→water transfer > 0
melt_event_smooth  : melt_transfer_mm  > max(0.01 mm, 0.01*snow_ice_before) 또는 melt_gate>0.5
```
**primary_mass residual 예산 (수치 확정):**
```text
dry             : <= 1e-8 mm/h
wet_no_phase    : <= 0.01 mm/h
freeze_melt     : <= 0.05 mm/h
snow_event_ph2  : report-only
snow_event_ph3  : <= 0.10 mm/h
```

## 7. MVP 범위 (첫 구현은 작게)
```text
droad/
  state.py  params.py  forcing.py
  radiation.py  ground.py
  boundary_layer_unrolled.py        # BLC-v0
  storage_exact_minimal.py
  model.py  loss.py  sensitivities.py
tests/
  test_rnet.py  test_dry_profile_one_step.py
  test_dry_rollout.py  test_jvp_vjp_dot.py
examples/
  run_dry_window.py  calibrate_x0_swcoef.py
```
**`storage_exact_minimal.py` 범위 명시 (no-go #4)** — "minimal"이 무엇을 빼는지 못박는다:
```text
포함: precipitation_to_storage(고정 위상), 눈↔얼음 전환 없는 water storage, 비음수 clamp
제외: wet snow 전환, ice2 갱신, deposit overflow, traffic wear, albedo 분기, melt/freeze 에너지 결합
```
이들은 M2c/M2d에서 파일 단위로 편입: `M2a precipitation_to_storage → M2b water_storage → M2c ice/snow/deposit → M2d albedo → M2e storage event sequence parity`.

**MVP 이후로 미룸**: `graph/fixed_point.py`, `implicit_step.py`, `estimation/second_order.py`, `estimation/uq.py`, EnVar, `hybrid/`. hybrid는 **기본 import에서도 제외**(아래).
```yaml
hybrid:
  enabled: false
  importable: false_until_M8
  components:                       # no-go #7: true/false 하나로 부족 → 컴포넌트별 3플래그
    forcing_bias_model:  {registered: false, importable: false, allowed_after: state_forcing_multicase_pass}
    parameter_map:       {registered: false, importable: false, allowed_after: selected_phy_twin_pass}
    radiation_residual:  {registered: false, importable: false, allowed_after: unseen_forecast_scorecard_pass}
```
> `enabled=false`인데 컴포넌트가 그래프 registry에 올라가는 상황을 막는다. `registered`가 false면 어떤 NN 성분도 연산그래프에 등록 불가. CI: `test_no_hybrid_component_registered_before_M8`.

## 8. 백엔드 단계화 (JAX는 parity 뒤에 "켠다")
```text
M1 pure NumPy reference   : jit/scan/custom_vjp 없음. RoadSurf-Python과 함수단위 parity만.
M2 JAX eager              : disable_jit=True, lax.scan 금지(Python loop), NaN-safe branch 감사.
M3 lax.scan + jit         : dry thermal core만 먼저 jit. storage/phase는 단계 편입.
M4 custom_vjp/implicit/remat : forward parity 고정 후 도입.
```

## 9. 세분 마일스톤 (gate 중심, 일정 아님)
```text
M0_must: fixture manifest + no-coupling dry fixture + layer/index convention manifest + branch registry(실동작) + BLC-v0 isolated unit test + 아래 Phase 0 executable tests(전부 통과)
M0c_optional: Fortran compile + 1-step golden   # required_if target==fortran_compat; non_blocking_if target==python_compat (audit sidecar)
M1a radiation + dry profile 1-step   M1b dry rollout                  M1c BLC no-LE       M1d BLC + LE
M2a precipitation only               M2b water storage                M2c snow/ice/deposit  M2d albedo
M3a JAX eager   M3b scan(no jit)     M3c jit dry only                 M3d jit storage
M4  enhanced_enthalpy + mass-ledger gate + τ per-primitive + grad tests
M5  DA: x0 + SW/LW coef (or forcing-bias), VJP adjoint
M6  물리모수 보정 + 식별성 게이트(§10)
M7a JVP/VJP operator contract(dot test)   M7b GN vector product(dry/smooth, CG smoke)
M7c B^{1/2} control transform(정적 대각 먼저)  M7d EnVar(다지점 후)
M7e UQ Lanczos/Hutchinson(사후최적 후)         M7f FSOI(관측연산자·스코어카드 후)
M8  (선택) hybrid ML / 커플링 재현
```
**M7c 기본값 제한 (no-go #10)** — 구현이 가벼워도 MVP/Alpha 기본 실행 경로에 끼우지 않는다:
```yaml
control_transform:
  B_sqrt_diagonal:
    available_from: M7c
    default_before_M7c: false
    required_for: [GNVP conditioning experiments]
    forbidden_for: [MVP, Phase1 parity, Phase2 smooth deviation]
```
> 현실 경고: **M0~M4만으로도 10~12주**가 될 수 있다. 일정이 아니라 **gate 통과**로 진도를 관리한다.

**Phase 0 executable tests (no-go #4 — "skeleton" 금지, 각 테스트는 고의 실패 fixture 보유).**
```text
test_branch_registry_has_required_entries     # 미등록 branch명 → 반드시 실패
test_guarded_where_no_nan_on_inactive_branch  # sqrt(음수) 비활성 branch NaN → 실패
test_storage_ledger_identity_synthetic        # 합성 저장 스텝 primary_mass 항등 위반 → 실패
test_event_matcher_freeze_melt_synthetic      # freeze/melt event 오분류 → 실패
test_dtype_shape_validator_rejects_float32    # float32 입력 → 반드시 실패
test_no_inplace_audit_detects_known_bad_example # 알려진 in-place 예제 → 반드시 검출
test_target_mode_combo_valid                  # paper_physics를 runtime target 지정 → 실패
test_roadsurf_exact_runs_in_paper_suite       # paper suite에서 exact 실행 가능(허용)
```
Phase 0 완료 기준은 위 테스트가 **실제 통과**(고의 실패 fixture가 실제로 실패)해야 하며, 빈 skeleton 통과는 금지.

## 10. 식별성 active-set 승격 규칙 (+정량 진단)
단일 Tsurf 관측에서 x0·SW·LW·Tair bias·traffic heat는 서로 오차를 대체한다. 아래 조건 충족 시에만 제어변수를 연다.
```text
SW/LW coef  : 주간 잔차가 SW 레짐과 상관 & JVP |dTsurf/dsw_coef| > 관측잡음 & 사후계수 미포화
Tair bias   : 야간/저복사 잔차 지속 & dTsurf/dTair가 dTsurf/dx0와 구별 가능
traffic heat: 주야 잔차 대비 존재 & 지점/도로등급 반복 패턴
phys param  : 다중창 gradient 부호 일관 & 미관측 예보 개선(분석창만 아님)
```
**정량 병행(전문가 보강)**: GN Hessian $J^\top J$의 **고유스펙트럼·상관행렬**로 near-singular(equifinality) 방향을 검출 → 상관 |ρ|>0.95 쌍은 동시 개방 금지.

## 11. forecast baseline hierarchy (커플링 정의 명확화)
Python `Coupling.py`는 DA reference 부적합(감사 확정). "운영 커플링 대비"의 대상을 분리한다.
```yaml
forecast_baselines:
  B0_persistence:                 required: true
  B1_no_coupling_roadsurf_python: required: true
  B2_fortran_coupling:            required_if: fortran_available
  B3_python_coupling_smoke:       required: false   # DA reference 아님
```
→ "운영 커플링 대비 RMSE 동등 이상"은 **B2가 있을 때만**. 없으면 **B1** 대비로 판정.

**효과 크기·표본 수 (no-go #8) — 작은 노이즈 개선을 통과로 착각하지 않게:**
```yaml
forecast_skill_gate:
  minimum_windows: 30
  minimum_stations: 3
  minimum_freeze_or_precip_events: 5
  primary_metric: [Tsurf_RMSE_+4h, Tsurf_RMSE_+6h]
  pass_if: [relative_rmse_improvement >= 3%, OR, absolute_rmse_improvement >= 0.05 degC]
  paired_uncertainty: station-window paired bootstrap, 95% CI excludes 0
  no_regression: [+12h/+24h degradation <= 1%, around_zero degradation <= 0.05 degC]
```
데이터 부족 초기에는 report-only. 이를 **config 계약**으로 못박아 "데이터 부족으로 gate를 못 봤는데 모델을 승격"하는 실패를 막는다(no-go #9):
```yaml
forecast_skill_gate:
  mode: hard_if_data_sufficient
  sufficient_data: {windows: 30, stations: 3, freeze_or_precip_events: 5}
  fallback_if_insufficient:
    mode: report_only
    allowed_for:   [MVP, Alpha-local]
    forbidden_for: [model_promotion, hybrid_enable, physics_default_change]
```

## 12. 설계서 문구 교체 매핑 (v0.6 반영)
| 위치 | 이전 | 교체 |
|---|---|---|
| §4.1 | "프레임워크 = JAX 확정" | "Phase 1 백엔드는 JAX이나 **M1은 pure NumPy parity kernel 먼저 고정**; jit/scan은 parity fixture 통과 후 단계 활성" |
| §6 | "BLC 암시적 고정점(custom_vjp) 구현" | "BLC는 **fixed-unroll → exact early-stop → custom_vjp IFT 3단계**; custom_vjp는 parity+dot-product 통과 후 승격" |
| §5 | "τ→0에서 smooth→exact 수렴" | "단순 threshold surrogate만 τ→0 수렴 요구; **hysteresis/enthalpy/ice2 개입 모드는 deviation/skill/물리진단으로 평가**" |
| §5/§4.1 | "총수분·에너지 보존 잔차" | "**mass ledger**(primary/auxiliary ice2/external source·sink/overflow) 분리; energy residual은 compat/enhanced 모드 별 해석" |
| §8/DoD | "운영 커플링 대비 RMSE 동등 이상" | "**baseline B0–B3 분리**; Python coupling은 primary baseline 아님" |

## 13. no-coupling fixture 생성 recipe + snapshot oracle (no-go #10)
"어떤 파일을 어떻게 patch해 어떤 output을 만드는가"를 못박는다(실제 커밋 사용):
```yaml
no_coupling_fixture_generation:
  repo: RoadSurf-Python
  commit: 61b5ee1864e1d5b6e2cdafce67a2dc9544520335   # HEAD 2026-05-28 (실측)
  source_file: src/main.py
  modifications: [use_coupling=False, disable_plotting=True, MPLBACKEND=Agg, ensure output/ exists]
  forbidden_modifications: [physics modules, example_data, initialization physics]
  command: "python src/main.py"
  expected_output: output/testi_output.csv
  manifest: {input_sha256: ..., config_sha256: ..., output_sha256: ...}
fortran_golden (audit sidecar):
  repo: RoadSurf
  commit: 8845d85e1ddc7775afe9d2142fe5ca8fd039bfbf   # HEAD 2026-06-01 (실측)
```
**state snapshot oracle (CSV만으론 부족, no-go #8).** balance 이후 snapshot 하나로는 BLC/RNet/profile/melting 중 어디가 틀렸는지 분리 불가 → **함수 경계까지 확장**:
```text
after_initialization
after_set_current_values
after_precipitation_to_storage
after_boundary_layer         # BLC + LE
after_calc_rnet
after_calc_hcap_hcond
after_calc_profile
after_melting
after_roadcond_storage
after_calc_albedo
after_save_output
```
MVP에서 전부 만들 필요는 없으나 **Phase 1e(full no-coupling parity) 전에 최소 6개**(set_current_values·boundary_layer·calc_rnet·calc_profile·melting·roadcond_storage)는 필수. dROAD 각 노드는 대응 snapshot과 대조.

**계측 전략 (no-go #4 — "physics module 수정 금지"와 "중간 snapshot"의 충돌 해소).** 원본을 건드리지 않고 wrapper monkeypatch로만 상태를 deep copy:
```yaml
snapshot_strategy:
  mode: wrapper_instrumentation
  allowed_modifications: [src/main.py, "new: src/snapshot_hooks.py"]
  forbidden_modifications: [physics equations, branch conditions]
  allowed_monkeypatch:      # 입출력 deep copy만, 계산식 불변
    - BalanceModel.BalanceModelOneStep
    - BoundaryLayer.CalcBLCondAndLE
    - BalanceModel.CalcRNet
    - BalanceModel.calcProfile
    - Storage.PrecipitationToStorage
    - Storage.melting
    - Cond.RoadCond
    - Cond.CalcAlbedo
```
별도 runner가 import 후 monkeypatch → snapshot이 fixture 자체를 오염시키지 않음.

**hook purity 계약 (no-go #5)** — output이 같아도 중간 object aliasing이 바뀌면 이후 snapshot이 거짓이 된다. `deepcopy`가 pointer/numpy view/aliasing을 바꾸지 않도록:
```text
snapshot hooks:
  - no mutation, no reassignment of model objects
  - no dtype conversion, no copy written back into model state
  - capture only after function return (pre-call은 명시 표시)
  - output은 detached serializable clone
```
CI: `test_snapshot_hooks_do_not_alter_output`(hook on/off CSV 동일), `test_snapshot_hooks_preserve_object_identity_for_runtime_state`, `test_snapshot_hooks_preserve_array_memory_aliasing_or_explicitly_document_breaks`.
