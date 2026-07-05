# True forecast DA — state estimation (480 step lead)

초기 near-surface 온도 상태를 변분동화한 뒤 **자유 예보**(obs 미삽입)가 no-DA(background) 대비 개선되는지 정직하게 판별한다. 파라미터는 default 고정 → 순수 STATE 추정. dry thermal 모델 내부에서 동화·예보가 같은 모델을 쓰므로 cross-model 아티팩트가 없다.

analysis start k0=2000 · 동화창 [k0, k0+120) valid obs 120개 · 예보 lead 480스텝 valid obs 480개. gate: RMSE만 hard. **deviation/physics-burden 감사는 미적용**(dry 모델은 storage를 진행하지 않음 — full-model 개념). degradation_ratio = 예보RMSE / 동화창RMSE (>1이면 overfit 또는 lead 구간이 동화창보다 본질적으로 더 어려움 — 둘을 분리해 볼 것).

| model | rmse | mae | freeze_thaw_accuracy | train_rmse | degradation_ratio | gate_vs_bg | gate_vs_const |
| --- | --- | --- | --- | --- | --- | --- | --- |
| constant_initial | 0.9180 | 0.8044 | 1.0000 |  |  | baseline | baseline |
| no_DA(background) | 0.2210 | 0.1796 | 1.0000 | 0.3530 | 0.6262 | baseline | PASS |
| DA(state) | 0.2082 | 0.1871 | 1.0000 | 0.2183 | 0.9541 | PASS | PASS |

## DA vs no-DA (초기상태 보정 효과)
- Δrmse (DA − background): -0.0128  (개선)
- DA degradation (예보/동화창): 0.954
- background degradation: 0.626
- state correction dx (layers 1:5, degC): [-0.008, -0.055, -0.586, -0.441]

해석: Δrmse<0이면 초기상태 동화가 자유예보를 개선(초기조건 오차가 lead에서 지배적). Δrmse≥0이면 model error가 지배적이거나 동화가 창에 overfit, 또는 lead가 동화창보다 본질적으로 어려운 구간일 수 있음(degradation_ratio를 window 난이도와 분리해 판단).
