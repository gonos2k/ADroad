# Diagnostics-aware calibrated-parameter evaluation (4000 steps)

**범위 주의**: 이것은 완전한 forecast DA가 아니라, JAX dry model에서 보정한 Emiss를 full model에 넣었을 때의 single-fixture 파라미터 민감도 + 진단 리포트다. 평가는 t=0부터 전체 trajectory를 다시 도는 free-run(analysis-state forecast 아님)이고, 보정한 Tair bias는 full model에 적용하지 않는다(Emiss만).

calibration window = [2000, 2200) · valid obs 200개(missing masked) · evaluation = 그 이후 valid obs 1800개(train 누수 없음). default Emiss=0.950, calibrated Emiss=0.995. baseline = constant_initial(분석시각 obs 고정). 참조: 1-step persistence RMSE = 0.0078(30s에서 자명 → gate baseline 부적합, gate에 미사용). gate: RMSE만 hard, MAE/freeze-thaw는 report-only.

표의 residual/over_melt/overflow는 **holdout interval [2200, 3999] 집계**(skill window와 정렬; forecast 오차는 obs 시각에서, 물리 부담은 그 사이 모든 model step에서 누적 — analysis-time 첫 step 포함). 전체 rollout 감사값은 아래 'Full-run audit' 참조.

| model | rmse | mae | freeze_thaw_accuracy | max_primary_residual | over_melt_count | overflow_count | gate_vs_const | gate_vs_default |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| constant_initial | 2.5839228454067205 | 1.9617176209005003 | 0.7398554752640356 |  |  |  | baseline | baseline |
| default | 0.14937350130788926 | 0.10573691837129533 | 0.9977765425236242 | 1.734723475976807e-18 | 0 | 0 | PASS | baseline |
| DA(Emiss=0.995) | 0.15548661686804696 | 0.11642946488326573 | 0.9983324068927182 | 1.734723475976807e-18 | 0 | 0 | PASS | FAIL — forecast RMSE 0.1555 worse than baseline 0.1494 |

## DA vs default (직접 비교)
- Δrmse (DA − default): +0.0061  (악화)
- Δmae  (DA − default): +0.0107

## Diagnostics delta (DA − default)
- Δover_melt_count: 0.0
- Δoverflow_count: 0.0
- Δdiagnostic_steps_rate: 0.0000
- physics_worse: False

## Full-run audit (전체 rollout, residual = 코드 누출 게이트 P0)
- default: residual=1.735e-18 (PASS) · over_melt=0 · overflow=0 · diag_rate=0.0065
- DA: residual=1.735e-18 (PASS) · over_melt=0 · overflow=0 · diag_rate=0.0065
