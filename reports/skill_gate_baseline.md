# Forecast Skill Gate — default vs constant_initial baseline

skill = 예측 정확도(낮을수록 좋음). gate = skill + 회계 residual + diagnostics 부담 종합.

| model | n | rmse | mae | freeze_thaw_accuracy | cold_n | cold_rmse | gate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| constant_initial | 1999 | 5.043245367372966 | 4.244531432382859 | 0.2766383191595798 | 1446 | 5.91088345510227 | baseline |
| default | 1999 | 0.2049395080143071 | 0.16798185405918278 | 0.991495747873937 | 1446 | 0.1275412551566676 | PASS |

baseline = constant_initial(분석시각 obs를 lead 전체에 고정). 1-step persistence는 30s 해상도에서 RMSE~0.006으로 자명해 gate baseline으로 부적합. gate: RMSE만 hard.

## Accounting / deviation (default)
- max_primary_residual: 4.441e-16 (PASS)
- diagnostic_steps_rate: 0.0043
- over_melt_count / overflow_count: 0 / 0
