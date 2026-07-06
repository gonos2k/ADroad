# Full-model forecast DA — 확장 설계 (Step 3)

> **Status**: design (구현 전) · **Date**: 2026-07-06 · **Basis**: commit `fddee2b`
> **선행 결과**: `docs/report/dROAD_report.md` §5.4–5.6, `reports/forecast_da*.md`, `reports/forecast_da_grid.md`

이 문서는 현재 **dry-only** 상태추정 forecast DA를 **full model(storage/phase-change 포함)**로 확장하는 설계다. 구현 전 리뷰용이며, 확정 시 `tools/`에 프로토타입을 추가한다.

---

## 1. 동기와 핵심 신규 능력

현재 forecast DA는 dry thermal 모델 안에서만 동화·예보를 수행한다(§4.4). 장점은 cross-model 아티팩트가 없다는 것이고, 한계는 **storage/phase state를 교정하지 못하고 deviation/physics-burden 감사가 적용되지 않는다**는 것이다(dry 모델은 저장소를 진행하지 않음).

full model로 가면 두 가지가 새로 가능해진다:

1. **deviation_budget 감사가 다시 핵심이 된다.** full_rollout은 눈·물·얼음·퇴적 저장소를 진행하므로, DA가 예보 skill을 올리면서 **물리 부담(over-melt/overflow/negative-pre-clamp)을 악화시키지 않는지**를 `deviation_budget` + `skill_gate`로 판정할 수 있다. 이것이 dry 모델에서 불가능했던 부분이다.
2. **operational RoadSurf에 근접**한다. 실제 노면예보는 열+수분+상전이 결합이므로, full-model DA가 최종 목표에 가깝다.

핵심 질문(설계가 답하려는 것):

> dry 모델에서 추정한 상태보정 `dx`(또는 그 확장)를 full 모델에 주입했을 때,
> (a) 예보 skill 개선이 **살아남는가**, (b) 물리 부담(deviation)이 **악화되지 않는가**?

---

## 2. 문제 정식화

현재 dry DA의 제어변수는 near-surface 온도 offset $dx \in \mathbb{R}^4$ (layers 1:5)이며, 동화창에서 다음을 최소화한다(§4.4):

$$J(dx) = \frac{1}{\sum w_t}\sum_t w_t\big(H M^{\text{dry}}_t(x_b \oplus dx) - y_t\big)^2 + bg_w\lVert dx\rVert^2$$

full model 확장의 본질은 **"$dx$(또는 storage 보정)를 full 모델의 어떤 상태에 어떻게 주입하고, 무엇을 제어변수로 열 것인가"**이다. full 모델 상태는 열상태 $(T_{\text{mp}}, T_{\text{mpNw}}, h_{\text{BLC}})$ + 저장소 $(\text{Snow}, \text{Water}, \text{Ice}, \text{Ice2}, \text{Dep})$ + albedo/노면상태로 구성된다.

---

## 3. 설계 대안 A / B / C

### 설계 A — thermal-only 주입 (최소·즉시 구현 가능)

```
1. full NumPy 모델을 분석시각 k0+window 까지 spin → 배경 분석상태(열+저장소 전부)
2. dry JAX 모델에서 dx 추정 (현재 방식 그대로, 동화창 [k0, k0+window))
3. 배경 분석상태의 Tmp/TmpNw layers 1:5 에 dx 가산 (저장소는 그대로)
4. full_rollout(return_ledger=True)로 lead 자유예보 실행 → pred_da_full
5. 배경 분석상태(무보정)로도 full_rollout → pred_bg_full
6. 평가: skill_gate(예보 RMSE) + deviation_budget(물리 부담) 둘 다
```

- **장점**: 현재 dry DA와 즉시 연결, 구현 최소, **deviation 감사가 처음으로 forecast DA에 적용**된다. 정직한 첫 질문(열 보정이 full 예보에서 살아남고 물리 부담을 해치지 않는가)에 바로 답한다.
- **단점**: storage/phase state 오차는 교정하지 못한다. dx가 dry 모델 기준이라 full 모델 열역학과 미세 불일치 가능.
- **제어변수**: $dx$ (dry에서 추정, full에 주입). full 모델은 미분 불필요(NumPy 예보만).

### 설계 B — smooth_compat storage까지 제어 (야심)

```
1. jax_storage smooth 경로 포함, thermal + storage state를 control로 개방
2. differentiable full-ish forecast로 loss 계산 (미분가능 대체모델)
3. full NumPy exact 모델로 audited forecast 재실행
4. smooth surrogate ↔ exact 모델 차이를 deviation_budget으로 감시
```

- **장점**: full physics에 가장 근접, storage/phase 오차까지 교정.
- **단점**: smooth surrogate와 exact 모델의 parity를 항상 감사해야 함(대체모델 오차가 DA 이득을 오염시킬 위험). 구현·검증 비용 큼.
- **제어변수**: thermal offset + storage 보정(비음수 제약, soft_clip 필요).

### 설계 C — two-stage DA (현실적 중간안)

```
Stage 1: dry thermal dx 추정 (설계 A)
Stage 2: storage diagnostics가 많은 case에서만 storage 보정 후보를 제한적으로 개방
```

- **장점**: 설계 A의 안전성 + 필요한 case에만 storage 자유도 추가. deviation 진단이 storage 보정 필요성을 스스로 신호.
- **단점**: 2단계 로직·게이트 설계 필요.

**권장 순서: A → C → B.** A로 "열 보정이 full 예보에서 살아남는가"를 먼저 확인하고, C로 storage 자유도를 진단 기반으로 제한 개방하며, B는 parity 감사가 충분히 성숙한 뒤.

---

## 4. 반드시 유지할 게이트 (정직성 계약)

full model로 가면 dry에서 미적용이던 감사가 **필수**로 복귀한다. 모든 설계는 다음을 보존한다:

| 게이트 | 조건 | 현재 도구 |
|---|---|---|
| forecast skill | DA 예보 RMSE ≤ no-DA (hard) | `skill_gate` |
| accounting | `max_primary_residual` < 1e-9 (P0) | `deviation_budget` / `accounting_gate` |
| diagnostics burden | over_melt/overflow/rate가 baseline 대비 비악화 | `skill_gate(deviation=…, baseline_deviation=…)` |
| state correction 크기 | `dx_l2`, `dx_max_abs` report (+ storage 보정 크기) | forecast DA meta |
| promotion | 단일 fixture는 REPORT_ONLY, 독립 case 필요 | `promotion_gate(n_cases=1)` |

즉 full-model DA는 **"skill 개선 + 물리 부담 비악화 + 질량 잔차 clean"** 세 조건을 동시에 만족할 때만 의미가 있다. dry DA가 skill-only였던 것과 달리, 여기서는 `physics_worse=True`면 skill이 좋아도 flag된다.

---

## 5. 설계 A 상세 스펙 (첫 프로토타입)

**도구(예정)**: `tools/report_forecast_da_fullmodel.py` — 기존 `report_forecast_da`(dx 추정)와 `report_da_evaluation`(full spin + full_rollout)의 재사용.

**의사코드**
```
build_A(k0, window, lead, bg_w):
    # dx 추정 (dry, 현재 방식)
    dx = estimate_dry_dx(k0, window, bg_w)          # report_forecast_da 재사용

    # full 모델 분석상태 (k0+window 까지 spin)
    objs = build_model(); advance_full(objs, 0, k0+window)
    state_bg = capture_full_state(objs)             # Surf, Tmp, TmpNw, Albedo, BLCond

    # 두 예보 (lead)
    out_bg = full_rollout(state_bg,               ..., n_steps=lead, return_ledger=True)
    state_da = apply_dx(state_bg, dx)               # Tmp/TmpNw layers 1:5 += dx
    out_da = full_rollout(state_da,               ..., n_steps=lead, return_ledger=True)

    # 평가
    obs_lead = TSurfObs[k0+window : k0+window+lead];  valid = obs>-100
    m_bg = forecast_metrics(out_bg.Tsurf[valid], obs[valid])
    m_da = forecast_metrics(out_da.Tsurf[valid], obs[valid])
    dev_bg = deviation_budget(out_bg, steps=lead_valid_interval)
    dev_da = deviation_budget(out_da, steps=lead_valid_interval)
    gate = skill_gate(m_da, m_bg, deviation=dev_da, baseline_deviation=dev_bg)  # skill + 물리부담
    return {m_bg, m_da, dev_bg, dev_da, gate, dx_l2, ...}
```

**평가 축**
- `gate_vs_bg`: DA 예보가 no-DA를 이기는가 (RMSE hard) **AND** 물리 부담 비악화.
- `physics_worse`: DA가 over-melt/overflow를 늘렸는가 — dry에서 볼 수 없던 신규 신호.
- 다중 window/grid: dry와 동일하게 `build_multi`류 재현 검증 → promotion REPORT_ONLY 유지.

**정직한 예상 시나리오**
1. **열 개선이 살아남고 물리 부담 clean** → 설계 A가 dry 결과를 full로 이관. 가장 좋은 경우.
2. **열은 개선되나 물리 부담 악화(physics_worse=True)** → dx가 열을 맞추려다 융해/상전이를 왜곡. `skill_gate`가 flag → 설계 C(storage 자유도)로 이동 신호.
3. **개선이 사라짐** → dry 열 보정이 full 결합(수분·알베도 피드백)에서 희석. dry↔full 이관의 한계 규명.

세 경우 모두 **정직한 결과**이며, dry-only에서 알 수 없던 것을 드러낸다.

**리스크·주의**
- dx는 dry 모델 loss로 추정되므로 full 모델 최적 보정과 다를 수 있음(설계 B가 이를 해결하나 비용 큼).
- full spin은 k0+window까지 필요 → 계산량 증가(grid는 슬라이스 누적 실행 유지).
- deviation `steps=`는 lead의 유효 구간(연속 interval)로 집계 — dry의 holdout-aligned 정책 재사용.

---

## 6. Open questions

1. dx 주입을 layers 1:5로 유지할지, full 모델의 층 구조(NLayers)에 맞춰 재정의할지.
2. storage 보정(설계 C)의 제어변수 파라미터화 — 비음수 제약을 soft_clip으로 열지, 로그 재파라미터화로 열지.
3. surrogate parity(설계 B) 허용 오차와 감사 방법(smooth vs exact rollout의 Tsurf/저장소 잔차 게이트).
4. full-model 평가의 baseline — no-DA(background) 외에 constant_initial·default Emiss도 병기할지.

---

## 7. 다음 액션

- [ ] 설계 A 프로토타입 `tools/report_forecast_da_fullmodel.py` 구현(단일 window) → skill + deviation 동시 게이트.
- [ ] 다중 window 재현 + grid(짧은 동화창·긴 lead prior 재사용) → promotion REPORT_ONLY 확인.
- [ ] 결과를 `docs/report/dROAD_report.md` §5에 편입, 정직성 톤 유지(단일 fixture, physics_worse 여부 명시).
- [ ] 설계 C는 A 결과에서 physics_worse가 관측될 때 착수.
