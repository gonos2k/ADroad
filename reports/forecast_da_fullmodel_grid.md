# A0 full-model forecast DA — grid bg_w×window×lead (12/12 combos)

A0가 **어떤 hyperparameter 영역에서 안정적으로 유효한지** 탐색한다(우승자 선정 아님). 단일 fixture라 어떤 조합도 promotion 대상이 아니다(promotion_eligible=False). 평균 RMSE가 아니라 rate를 본다: physics_worse_rate>0이거나 worst_delta_rmse가 크면 평균이 좋아도 operational로 위험. 랭킹: gate_pass_rate↑ → physics_worse_rate↓ → worst_delta↓ → mean_delta↓ → state_large_rate↓.

windows: k0=1500+i×600 (i<4). combos: bg_w[0.01, 0.05, 0.2] × window[60, 120] × lead[240, 480]. Δrmse=DA−BG(클수록 나쁨). residual은 code-leak 게이트(clean 유지 기대).

| bg_w | window | lead | n_valid | gate_pass_rate | skill_improved_rate | physics_worse_rate | state_large_rate | mean_delta_rmse | worst_delta_rmse | max_lead_diag_delta | max_residual | residual_clean | window_precondition_met |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.2000 | 60 | 240 | 4 | 0.5000 | 0.5000 | 0.0000 | 0.0000 | 0.0010 | 0.0193 | 0.0000 | 0.0000 | True | False |
| 0.0500 | 120 | 480 | 4 | 0.5000 | 0.5000 | 0.0000 | 0.0000 | 0.0244 | 0.1182 | 0.0000 | 0.0000 | True | False |
| 0.0100 | 60 | 480 | 4 | 0.5000 | 0.5000 | 0.0000 | 0.5000 | 0.0282 | 0.1547 | 0.0000 | 0.0000 | True | False |
| 0.0100 | 120 | 480 | 4 | 0.5000 | 0.5000 | 0.0000 | 0.2500 | 0.0445 | 0.1998 | 0.0000 | 0.0000 | True | False |
| 0.2000 | 60 | 480 | 4 | 0.5000 | 0.7500 | 0.2500 | 0.0000 | 0.0018 | 0.0303 | 0.0021 | 0.0000 | True | False |
| 0.0500 | 60 | 240 | 4 | 0.2500 | 0.2500 | 0.0000 | 0.0000 | 0.0198 | 0.0801 | 0.0000 | 0.0000 | True | False |
| 0.2000 | 120 | 240 | 4 | 0.2500 | 0.2500 | 0.0000 | 0.0000 | 0.0256 | 0.0802 | 0.0000 | 0.0000 | True | False |
| 0.0500 | 120 | 240 | 4 | 0.2500 | 0.2500 | 0.0000 | 0.0000 | 0.0661 | 0.1923 | 0.0000 | 0.0000 | True | False |
| 0.2000 | 120 | 480 | 4 | 0.2500 | 0.5000 | 0.2500 | 0.0000 | 0.0088 | 0.0462 | 0.0021 | 0.0000 | True | False |
| 0.0500 | 60 | 480 | 4 | 0.2500 | 0.5000 | 0.2500 | 0.0000 | 0.0125 | 0.0909 | 0.0021 | 0.0000 | True | False |
| 0.0100 | 60 | 240 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.0553 | 0.1496 | 0.0000 | 0.0000 | True | False |
| 0.0100 | 120 | 240 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.2500 | 0.1176 | 0.3109 | 0.0000 | 0.0000 | True | False |

## 관찰 (single fixture → 모두 REPORT_ONLY)
- physics_worse 없이 과반 window PASS인 조합: 0/12
- 최상위: bg_w=0.2 window=60 lead=240 (gate_pass_rate=0.50, physics_worse_rate=0.00, worst_delta=+0.0193)
- 해석축: 짧은 window vs 긴 window, lead↑에 따른 state memory 소실, bg_w↓의 overfit/state_large, bg_w↑의 DA 효과 소실. grid 최적값은 결론이 아니라 독립-case(Step 4) 실험의 탐색 범위.
