# Multi-window forecast skill (6 periods)

기간별 default vs constant_initial baseline(분석시각 obs 고정; 1-step persistence는 30s에서 자명해 미사용). 여러 기간에서 일관되게 이겨야 신뢰(한 창의 운 아님). gate: RMSE만 hard.

| window | n | default_rmse | const_initial_rmse | freeze_thaw_accuracy | beats_const_initial |
| --- | --- | --- | --- | --- | --- |
| 0 | 960 | 0.3440 | 3.1959 | 0.9771 | True |
| 1 | 959 | 0.2232 | 3.6377 | 0.9927 | True |
| 2 | 959 | 0.0590 | 1.9784 | 1.0000 | True |
| 3 | 959 | 0.1123 | 3.7561 | 0.9958 | True |
| 4 | 959 | 0.2629 | 3.1586 | 0.9823 | True |
| 5 | 959 | 0.1209 | 2.1186 | 1.0000 | True |

## Aggregate (default across periods)
- n_windows: 6
- rmse_mean: 0.1870
- rmse_max (worst window): 0.3440
- rmse_min (best window): 0.0590
- freeze_thaw_accuracy_mean: 0.9913
- beats constant_initial in ALL windows: True

## Promotion gate (design §11)
- verdict: **REPORT_ONLY**
- insufficient cases: 1 < 3 (report-only)
