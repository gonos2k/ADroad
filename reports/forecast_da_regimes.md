# Forecast DA regime analysis (4 windows)

state-DA가 어떤 window에서 no-DA를 이기고 어떤 window에서 지는지, forcing/obs/DA 내부 feature로 설명한다. **표본이 작아 통계검정이 아니라 case-study**다 — win/lose 그룹 평균 차이가 큰 feature를 후보 신호로 제시할 뿐, 인과는 독자가 판단한다.

## Summary
- windows: 4  ·  DA wins: 2/4
- win windows (k0): [2700, 3300]
- lose windows (k0): [1500, 2100]
- mean Δrmse (DA−bg): +0.0238  ·  median: +0.0077

## Per-window snapshot
| k0 | beats_bg | delta_rmse | bg_init_error | train_delta | degradation_da | dx_l2 | obs_std | tair_trend_abs | freeze_crossing_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1500 | False | +0.1183 | 0.7145 | -0.3232 | 1.6435 | 1.5265 | 0.4126 | 1.7458 | 0 |
| 2100 | False | +0.0200 | 0.3210 | -0.1487 | 1.3426 | 0.7032 | 0.3941 | 1.4958 | 0 |
| 2700 | True | -0.0046 | 0.1740 | -0.0431 | 3.4987 | 0.3257 | 0.6047 | 0.1017 | 0 |
| 3300 | True | -0.0385 | 0.5888 | -0.2572 | 3.3190 | 1.2708 | 1.3203 | 2.4967 | 1 |

## Candidate regime signals (win vs lose group means)
가장 잘 분리하는 feature 상위 12개(상대 gap 기준):

| feature | win_mean | lose_mean | direction | separation |
| --- | ---: | ---: | --- | ---: |
| dx_layer1 | 0.0501 | -0.0493 | higher in wins | 1.983 |
| dx_layer2 | 0.1714 | -0.2184 | higher in wins | 1.785 |
| dx_layer3 | 0.6806 | -0.9396 | higher in wins | 1.724 |
| dx_layer4 | 0.3757 | -0.5543 | higher in wins | 1.678 |
| tair_mean | -0.0231 | 0.8439 | higher in losses | 1.027 |
| freeze_crossing_count | 0.5000 | 0.0000 | higher in wins | 1.000 |
| sw_mean | 28.1499 | 0.0556 | higher in wins | 0.998 |
| degradation_bg | 2.2963 | 0.6965 | higher in wins | 0.697 |
| obs_step_change_max | 0.0167 | 0.0063 | higher in wins | 0.625 |
| obs_std | 0.9625 | 0.4034 | higher in wins | 0.581 |
| const_rmse | 1.8827 | 0.8122 | higher in wins | 0.569 |
| degradation_da | 3.4089 | 1.4930 | higher in wins | 0.562 |

주의: `dx_layer*`/`dx_l2`/`dx_max_abs`는 DA가 **실제로 가한 보정(내생적)**이라 win/lose의 원인이 아니라 결과에 가깝다. 독립적인 regime 원인은 forcing/obs feature (tair_*, sw_mean, freeze_crossing_count, obs_* 등)에서 찾아야 한다.

## 가설 점검(케이스 스터디)
- **A (배경오차 큼 + model error 작음 → 이김)**: bg_init_error/train_delta가 win에서 더 크게 유리한가?
- **B (lead 난이도 상승 → 짐)**: degradation_da·obs_std·tair_trend_abs가 lose에서 큰가?
- **C (dx 과도 → overfit으로 짐)**: dx_l2·dx_max_abs가 lose에서 큰가?
