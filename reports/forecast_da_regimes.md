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
| 1500 | False | +0.1183 | 0.7145 | -0.3232 | 1.6434 | 1.5265 | 0.4126 | 1.7458 | 0 |
| 2100 | False | +0.0200 | 0.3210 | -0.1487 | 1.3424 | 0.7032 | 0.3941 | 1.4958 | 0 |
| 2700 | True | -0.0046 | 0.1740 | -0.0431 | 3.4989 | 0.3256 | 0.6047 | 0.1017 | 0 |
| 3300 | True | -0.0385 | 0.5888 | -0.2572 | 3.3190 | 1.2708 | 1.3203 | 2.4967 | 1 |

## Candidate regime signals (win vs lose group means)
**n_win=2, n_lose=2 — 표본이 작아 separator ranking은 매우 불안정하다 (outlier 하나에 흔들림). 인과·일반규칙이 아니라 hypothesis generator로만 사용.**

### A. Ex-ante forcing (예보시각에 알 수 있는 후보 regime 신호) — **1차 해석 기준**
이 diagnostic은 forcing을 '예보로 주어진 forcing'으로 취급한다. 실측 forcing만 있으면 post-hoc 설명 feature로 해석할 것.

| feature | win_mean | lose_mean | direction | separation |
| --- | ---: | ---: | --- | ---: |
| tair_mean | -0.0231 | 0.8439 | higher in losses | 1.027 |
| sw_mean | 28.1499 | 0.0556 | higher in wins | 0.998 |
| is_night_fraction | 0.3125 | 0.6875 | higher in losses | 0.545 |
| tair_trend_abs | 1.2992 | 1.6208 | higher in losses | 0.198 |
| rhz_mean | 81.5243 | 66.2596 | higher in wins | 0.187 |
| tair_std | 0.3626 | 0.4225 | higher in losses | 0.142 |
| tair_range | 1.3983 | 1.6208 | higher in losses | 0.137 |
| lw_mean | 245.9426 | 223.7561 | higher in wins | 0.090 |

### B. Post-hoc obs difficulty (사후 난이도 — 예보 전엔 모름)
lead의 실측 관측에서 계산 — '왜 어려웠나'는 설명하나 ex-ante 신호는 아님.

| feature | win_mean | lose_mean | direction | separation |
| --- | ---: | ---: | --- | ---: |
| freeze_crossing_count | 0.5000 | 0.0000 | higher in wins | 1.000 |
| obs_step_change_max | 0.0167 | 0.0063 | higher in wins | 0.625 |
| obs_std | 0.9625 | 0.4034 | higher in wins | 0.581 |
| const_rmse | 1.8827 | 0.8122 | higher in wins | 0.569 |
| obs_range | 3.0950 | 1.4700 | higher in wins | 0.525 |
| obs_trend_abs | 3.0700 | 1.4700 | higher in wins | 0.521 |
| obs_step_change_mean | 0.0069 | 0.0035 | higher in wins | 0.497 |
| cold_fraction | 0.6167 | 1.0000 | higher in losses | 0.383 |

### C. DA response / diagnostics (DA가 실제로 한 보정 — 원인 아닌 결과)
dx_*·train·degradation은 DA의 반응이라 win/lose를 잘 나눠도 사전 원인으로 읽지 말 것.

| feature | win_mean | lose_mean | direction | separation |
| --- | ---: | ---: | --- | ---: |
| dx_layer1 | 0.0501 | -0.0493 | higher in wins | 1.984 |
| dx_layer2 | 0.1714 | -0.2184 | higher in wins | 1.785 |
| dx_layer3 | 0.6805 | -0.9396 | higher in wins | 1.724 |
| dx_layer4 | 0.3757 | -0.5543 | higher in wins | 1.678 |
| degradation_bg | 2.2964 | 0.6964 | higher in wins | 0.697 |
| degradation_da | 3.4090 | 1.4929 | higher in wins | 0.562 |
| train_delta | -0.1502 | -0.2359 | higher in wins | 0.364 |
| dx_l2 | 0.7982 | 1.1149 | higher in losses | 0.284 |

## 가설 점검(케이스 스터디)
- **A (배경오차 큼 + model error 작음 → 이김)**: bg_init_error/train_delta가 win에서 유리?
- **B (lead 난이도 상승 → 짐)**: degradation_da·obs_std·tair_trend_abs가 lose에서 큰가?
- **C (dx 과도 → overfit으로 짐)**: dx_l2·dx_max_abs가 lose에서 큰가?

관찰(이번 run): win group은 평균적으로 더 낮은 tair_mean·더 높은 sw_mean·freeze crossing 쪽으로 치우치나 **window별 예외가 있어**(예: win k0=2700은 is_night_fraction≈0.63으로 야간 비중이 큼) causal rule이 아니라 후보 regime signal로만 유지한다.
