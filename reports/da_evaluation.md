# Diagnostics-aware DA evaluation (3000 steps)

skill(RMSE↓) + accounting(residual~0) + physics(diagnostics 부담). default Emiss=0.950, calibrated Emiss=0.995.

| model | rmse | mae | freeze_thaw_accuracy | max_primary_residual | diagnostic_steps_rate | over_melt_count | overflow_count | gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| persistence | 6.379721914286672 | 5.710068367517092 | 0.2726363181590795 |  |  |  |  | baseline |
| default | 0.1508434652330266 | 0.10827422402434175 | 0.9964982491245623 | 0.0 | 0.003 | 0 | 0 | PASS |
| DA(Emiss=0.995) | 0.147252460262171 | 0.1126559747181825 | 0.9969984992496248 | 0.0 | 0.003 | 0 | 0 | PASS |

## Diagnostics delta (DA − default)
- Δover_melt_count: 0
- Δoverflow_count: 0
- Δdiagnostic_steps_rate: 0.0000
- physics_worse: False
