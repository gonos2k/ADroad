# Full-model forecast DA — 설계 A0 (480 step lead)

dry에서 추정한 near-surface 상태보정 dx를 **full 모델** k0 상태에 주입하고(TsurfAve 동기화), [k0, k0+window+lead)를 obs 미삽입 free-run으로 예보한다. dry DA와 달리 storage가 진행되므로 **deviation 감사(물리 부담)가 forecast DA에 처음 적용**된다. gate: RMSE hard + 물리 부담 비악화.

k0=2000 · 동화창 120 · 예보 lead 480 valid obs 480개 · dx_scale=1. raw dx at k0 (A0). analysis-window diagnostics는 report-only, lead-aligned budget이 primary gate.

| model | rmse | mae | freeze_thaw_accuracy | max_primary_residual | over_melt_count | overflow_count | gate_vs_bg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| constant_initial | 0.9180 | 0.8044 | 1.0000 |  |  |  | baseline |
| no_DA(background) | 0.2210 | 0.1796 | 1.0000 | 0.0000 | 0 | 0 | baseline |
| DA(state, full) | 0.2082 | 0.1870 | 1.0000 | 0.0000 | 0 | 0 | PASS |

## DA vs no-DA (핵심)
- Δrmse (DA − background): -0.0128  (개선)
- physics_worse (over_melt/overflow/rate 악화 여부): **False**
- diag_steps_rate — **lead(primary gate)**: bg 0.0000 / da 0.0000  ·  **window(report-only)**: bg 0.0000 / da 0.0000
- state_correction_large (dx_l2>3 또는 max|dx|>2): False (dx_l2=0.736, max|dx|=0.586) — report-only 진단(gate 아님); multi-window/grid에서 반복되면 bg_w↑ 또는 overfit signal로 해석
- Δover_melt: 0.0 · Δoverflow: 0.0 · Δdiag_rate: +0.0000
- state correction dx (layers 1:5): [-0.008, -0.055, -0.586, -0.441] (l2=0.736, max|dx|=0.586)

## Analysis-window diagnostics (report-only)
- background: over_melt=0 overflow=0 rate=0.0000
- DA:         over_melt=0 overflow=0 rate=0.0000

**주의**: diagnostic 활동이 window(report-only)에서만 발생하고 lead(primary gate)에서 0이면, 이 case는 'storage-active signal은 있으나 lead deviation gate는 clean'이다 — lead gate가 실제 burden 증가를 처리했다는 증거는 아니다(그건 lead 구간에 diagnostics가 발생하는 window/stress 필요).

해석: DA가 lead 예보 RMSE를 낮추면서(gate PASS) physics_worse=False면 열 보정이 full 예보에서 살아남고 물리 부담도 clean. physics_worse=True면 열을 맞추려다 융해/상전이를 왜곡한 것 → 설계 C 신호.
