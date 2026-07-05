# Multi-window forecast DA (4 windows)

단일 fixture에서 관측된 'state-DA가 no-DA를 이긴다'가 여러 연속 analysis window에서 재현되는지 검증한다. 매 window에서 DA가 background를 이겨야 신뢰(한 창의 운 아님). gate: RMSE만 hard, deviation 감사 미적용(dry 모델). promotion_gate: 단일/소수 case는 REPORT_ONLY, 충분한 case에서 모두 이기면 PROMOTE 후보.

| k0 | da_rmse | bg_rmse | delta_da_minus_bg | train_da | train_bg | da_degradation | dx_l2 | beats_bg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1500 | 0.6431 | 0.5248 | +0.1183 | 0.3913 | 0.7145 | 1.6435 | 1.5265 | False |
| 2100 | 0.2313 | 0.2113 | +0.0200 | 0.1723 | 0.3210 | 1.3426 | 0.7032 | False |
| 2700 | 0.4580 | 0.4626 | -0.0046 | 0.1309 | 0.1740 | 3.4987 | 0.3257 | True |
| 3300 | 1.1007 | 1.1392 | -0.0385 | 0.3316 | 0.5888 | 3.3190 | 1.2708 | True |

## Window reproducibility (같은 fixture, 연속 window)
- windows: 4
- DA beats background in: 2/4 windows
- mean Δrmse (DA − background): +0.0238
- beats background in ALL windows: False

## Promotion gate (design §11)
**window ≠ 독립 case**: 한 fixture의 연속 window는 서로 독립이 아니므로 promotion은 `n_cases=1`(단일 fixture)로 판정한다. 따라서 window를 모두 이겨도 단일 fixture로는 PROMOTE되지 않는다(독립 station/day를 늘려야 n_cases 증가).
- promotion_cases: 1  ·  window_reproducibility: 2/4
- verdict: **REPORT_ONLY**
- insufficient cases: 1 < 3 (report-only)
- does not beat baseline in every window
