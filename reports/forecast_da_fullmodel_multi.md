# A0 full-model forecast DA — multi-window (4/4 windows)

단일 fixture A0가 보여줄 수 없는 것을 본다: 여러 연속 window에서 DA가 skill을 개선하는 빈도, lead physics burden이 악화되는(physics_worse) 빈도, state correction이 큰 빈도, 그리고 매 window에서 DA가 no-DA를 **skill과 physics 모두** 이기는지. per-window gate는 full-model gate(RMSE hard + 물리부담 비악화)라 gate_pass가 둘을 이미 포함한다. skill↑ yet physics↓가 핵심 관심 실패 양상이라 둘을 따로 표기.

windows: k0=1500+i×600 (i<4) · 동화창 120 · lead 480 · bg_w 0.05. promotion: 단일 fixture라 n_cases=1 고정 → REPORT_ONLY(정직성). 다수 독립 case는 Step 4(cases.yaml).

| k0 | bg_rmse | da_rmse | rmse_delta | skill_improved | gate_pass | physics_worse | state_large | lead_diag_da | win_diag_da | resid_da |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1500 | 0.5248 | 0.6431 | 0.1183 | False | False | False | False | 0.0000 | 0.0000 | 0.0000 |
| 2100 | 0.2113 | 0.2313 | 0.0200 | False | False | False | False | 0.0000 | 0.0000 | 0.0000 |
| 2700 | 0.4639 | 0.4593 | -0.0046 | True | True | False | False | 0.0000 | 0.0333 | 0.0000 |
| 3300 | 1.2196 | 1.1822 | -0.0374 | True | True | False | False | 0.0333 | 0.0000 | 0.0000 |

## 집계 (single fixture → REPORT_ONLY)
- gate PASS: 2/4 (rate 0.50) · beats-all=False
- skill 개선: 2/4 (rate 0.50) · physics_worse: 0/4 (rate 0.00)
- state_correction_large: 0/4 · mean Δrmse +0.0240 · worst Δrmse +0.1183
- max lead diag_rate(DA): 0.0333 · max residual 1.73e-18 (clean=True)
- **promotion: REPORT_ONLY** — insufficient cases: 1 < 3 (report-only); does not beat baseline in every window

해석: gate_pass_rate<1이면 DA가 매 window를 이기지 못한 것(regime-dependent). physics_worse_rate>0이면 일부 window에서 열 보정이 물리 부담을 키운 것 — skill이 좋아도 그 window는 FAIL해야 정직. residual이 clean하면 실패 원인은 accounting leak이 아니라 deviation burden이다.
