# Full-model forecast DA — A0 STRESS (dx_scale=15)

> **주의: 이 artifact는 실제 DA 성능 평가가 아니라 fitted dx를 15배 확대한 unphysical STRESS test다.** 목적은 RMSE가 개선되어도 lead physics burden이 악화되면 skill_gate가 FAIL하는지(정직성 계약)를 검증하는 것. 아래 큰 dx_l2/max|dx|는 이 인위적 확대의 결과다.

dry에서 추정한 near-surface 상태보정 dx를 **full 모델** k0 상태에 주입하고(TsurfAve 동기화), [k0, k0+window+lead)를 obs 미삽입 free-run으로 예보한다. dry DA와 달리 storage가 진행되므로 **deviation 감사(물리 부담)가 forecast DA에 처음 적용**된다. gate: RMSE hard + 물리 부담 비악화.

k0=3800 · 동화창 120 · 예보 lead 480 valid obs 480개 · dx_scale=15. raw dx at k0 (A0). analysis-window diagnostics는 report-only, lead-aligned budget이 primary gate.

| model | rmse | mae | freeze_thaw_accuracy | max_primary_residual | over_melt_count | overflow_count | gate_vs_bg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| constant_initial | 1.5361 | 1.0802 | 0.8229 |  |  |  | baseline |
| no_DA(background) | 1.3441 | 1.2707 | 0.8771 | 0.0000 | 0 | 0 | baseline |
| DA(state, full) | 1.2073 | 1.0052 | 0.8229 | 0.0000 | 0 | 0 | FAIL — diagnostic_steps_rate worse than baseline |

## DA vs no-DA (핵심)
- Δrmse (DA − background): -0.1368  (개선)
- physics_worse (over_melt/overflow/rate 악화 여부): **True**
- diag_steps_rate — **lead(primary gate)**: bg 0.0000 / da 0.3250  ·  **window(report-only)**: bg 0.0417 / da 0.1333
- state_correction_large (dx_l2>3 또는 max|dx|>2): True (dx_l2=40.970, max|dx|=35.724) — report-only 진단(gate 아님); multi-window/grid에서 반복되면 bg_w↑ 또는 overfit signal로 해석
- Δover_melt: 0.0 · Δoverflow: 0.0 · Δdiag_rate: +0.3250
- state correction dx (layers 1:5): [+1.167, +5.006, +35.724, +19.388] (l2=40.970, max|dx|=35.724)

## Analysis-window diagnostics (report-only)
- background: over_melt=0 overflow=0 rate=0.0417
- DA:         over_melt=0 overflow=0 rate=0.1333

**stress 해석**: lead(primary gate)에서 DA diagnostic burden이 background보다 증가했다 (da 0.3250 > bg 0.0000). 따라서 RMSE가 개선되어도 skill_gate가 FAIL한다 — 이것이 의도한 end-to-end gate 검증(정직성 계약)이다.

해석: 이 artifact는 RMSE가 개선되어도(DA<bg) lead physics burden이 악화되면 gate가 FAIL한다는 정직성 계약을 검증한다 — 실제 DA 성능 주장에는 사용하지 않는다(→ 설계 C: 열 보정이 융해/상전이를 왜곡).
