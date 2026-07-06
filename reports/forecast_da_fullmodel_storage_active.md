# Full-model forecast DA — 설계 A0 (480 step lead)

dry에서 추정한 near-surface 상태보정 dx를 **full 모델** k0 상태에 주입하고(TsurfAve 동기화), [k0, k0+window+lead)를 obs 미삽입 free-run으로 예보한다. dry DA와 달리 storage가 진행되므로 **deviation 감사(물리 부담)가 forecast DA에 처음 적용**된다. gate: RMSE hard + 물리 부담 비악화.

k0=3800 · 동화창 120 · 예보 lead 480 valid obs 480개. raw dx at k0 (A0). analysis-window diagnostics는 report-only, lead-aligned budget이 primary gate.

| model | rmse | mae | freeze_thaw_accuracy | max_primary_residual | over_melt_count | overflow_count | gate_vs_bg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| constant_initial | 1.5361 | 1.0802 | 0.8229 |  |  |  | baseline |
| no_DA(background) | 1.3441 | 1.2707 | 0.8771 | 0.0000 | 0 | 0 | baseline |
| DA(state, full) | 1.1866 | 1.1126 | 0.8979 | 0.0000 | 0 | 0 | PASS |

## DA vs no-DA (핵심)
- Δrmse (DA − background): -0.1575  (개선)
- physics_worse (over_melt/overflow/rate 악화 여부): **False**
- Δover_melt: 0.0 · Δoverflow: 0.0 · Δdiag_rate: +0.0000
- state correction dx (layers 1:5): [+0.078, +0.334, +2.382, +1.293] (l2=2.731, max|dx|=2.382)

## Analysis-window diagnostics (report-only)
- background: over_melt=0 overflow=0 rate=0.0417
- DA:         over_melt=0 overflow=0 rate=0.0417

해석: DA가 lead 예보 RMSE를 낮추면서(gate PASS) physics_worse=False면 열 보정이 full 예보에서 살아남고 물리 부담도 clean. physics_worse=True면 열을 맞추려다 융해/상전이를 왜곡한 것 → 설계 C 신호.
