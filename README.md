# dROAD — Differentiable RoadSurf

FMI **RoadSurf** 도로 노면 기상 모델의 **미분가능(differentiable) 버전**. JAX로 재구현하여
**자료동화(DA)와 물리모수 보정(calibration)을 하나의 손실로 동시에** 수행하는 하이브리드
미분모델링 시스템이다. 물리: Karsisto (2024, *Geosci. Model Dev.*, 17, 4837–4853).

## 무엇을 하는가
- RoadSurf(no-coupling)를 **비트 수준으로 재현**한 참조 커널(NumPy)
- 그 위의 **미분가능 JAX 롤아웃** — `jax.grad`(VJP/adjoint), JVP(TLM), HVP(2차)
- gradient 기반 **자료동화 + 물리모수 동시 추정** (쌍둥이 실험·다중윈도우·실관측)

## 문서 (설계 → 실행)
| 문서 | 역할 |
|---|---|
| `dROAD_설계계획서.md` (v0.7) | **청사진** — 물리·아키텍처·§7 미분DA 시스템·검증전략·부록 지배방정식 |
| `구현계획_P0_derisked.md` | **실행 계약**(우선) — compatibility_target/mode, gate 등급, branch 정책, fixture recipe |
| `README_code.md` | 코드 진척·검증 상태 표 |
| 검토/분석 문서 | `dROAD_설계계획서_적대적검토.md`, `적대적검토2_미분연산_하이브리드DA.md`, `프레임워크_재검토_JAX_vs_PyTorch.md`, `격차분석_Fortran_vs_Python.md`, `AD가능성_감사_Python.md` |

## 코드 구조
```
droad/
  config.py      # compatibility_target × model_mode enum + 검증
  branches.py    # site-aware where/guard 래퍼 + registry (raw primitive 금지)
  ledger.py      # mass ledger 계약(StorageLedger/merge_ledgers)
  radiation.py thermal.py boundary.py storage.py roadcond.py   # 물리 (NumPy, exact)
  model.py driver.py    # step_full / dry·full rollout
  jax_model.py jax_storage.py   # 미분가능 JAX 롤아웃 (dry / smooth_compat)
  smoothing.py   # σ게이트·soft_min/max·엔탈피 (smooth_compat 원시연산)
  assimilate.py  # optax fit + hvp/newton/laplace_cov
tools/           # raw-primitive 감사, no-coupling fixture 생성·snapshot
reference/RoadSurf-Python/   # 원본 사본(commit 61b5ee1), 포팅·검증 기준
fixtures/no_coupling/        # pinned fixture(sha256) + snapshot
tests/           # python_compat·JAX·DA (20 파일)
```

## 검증 요약
```
M1 dry thermal        참조와 bit-exact (rollout Tsurf RMSE 5e-17)
M2 storage/상변화      참조와 bit-exact (full rollout Tsurf 1.8e-16, 저장 diff 0)
미분 정합             VJP=유한차분, JVP↔VJP dot-product, HVP 대칭성
DA 쌍둥이             param·state·joint 복원, 다중윈도우 결합추정(전역+국소)
실관측 DA            무제약 과적합 재현 → 제약+정칙화로 물리복원·persistence 개선
전체                 pytest 93 passed (20 파일), core raw-primitive violations 0
```

## 실행
```bash
pip install -e ".[dev]"            # 또는: pip install numpy pytest "jax[cpu]" optax
pytest -q                          # 전체 테스트
python examples/demo_da.py         # 쌍둥이 보정 + 실관측 DA 데모
python tools/run_no_coupling.py    # no-coupling fixture 재생성
```

## 상태 · 다음
연구 프로토타입으로 **설계→구현→수치검증→실관측 적용** 전 파이프라인 완성.
실사용 승격: 다중 사례·다중 지점 데이터로 forecast skill gate promotion(§11),
smooth storage 정밀화(deposit/ice2·에너지제한 융해·G3), hybrid-4DEnVar(§7.7).

라이선스: 원본 RoadSurf는 MIT(FMI). 본 저장소는 그 재구현·확장.
