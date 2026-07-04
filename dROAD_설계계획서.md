---
status: active_blueprint
backend_path: numpy_to_jax
execution_contract: 구현계획_P0_derisked.md   # 충돌 시 P0 우선
note: PyTorch-first 대체 설계서는 이 저장소에 없음 (프레임워크=JAX, 재검토 근거는 프레임워크_재검토_JAX_vs_PyTorch.md)
---

# dROAD — 미분가능 RoadSurf 설계·계획서

> FMI RoadSurf 도로 노면 기상 모델의 **미분가능(differentiable) 버전**을 JAX로 구현하기 위한 설계 및 계획 문서
> 목적: **자료동화(DA) + 물리모수 최적화(calibration)를 동시에 수행하는 하이브리드 미분모델링 시스템**
> 작성일: 2026-07-02 · 버전 0.7 (외부 적대적 검토 2차 반영 — 문서 간 계약 정합)
>
> **우선순위 계약**: 구현 순서·gate가 이 청사진과 `구현계획_P0_derisked.md`(실행 계약) 사이에서 충돌하면 **P0 문서가 우선한다.**
>
> **v0.7 변경점**: 외부 검토 2차의 no-go 반영 — (1) §9 로드맵 M3에서 **custom_vjp 문구 제거**(BLC-v0만; v2는 M4/M5 promotion gate), (2) §4.2에 **M1 pure NumPy(backend-neutral) vs M2+ `jax.Array` 예시 구분** 명시, (3) §8.4 **compatibility_target/model_mode 2축 enum 통일**·허용/금지 조합. 나머지 실행 계약(branch registry+CI, BLC-v1 DA 금지, G3 event·mass 수치, storage_minimal 범위, M0c optional, forecast 효과크기, fixture recipe·snapshot oracle, ledger_policy)은 **P0 문서에 반영**.
>
> **v0.6 변경점**: 외부 적대적 검토 수용 — 본 문서는 **최종 청사진**으로 두고, **실패하지 않는 첫 구현 계획은 `구현계획_P0_derisked.md`로 분리**. 반영: (a) §4.1 백엔드 **단계화**(M1 pure NumPy parity 먼저, JAX는 이후), (b) §5 **모드 3분할**(exact / smooth_compat / enhanced_enthalpy)·τ→0 수렴은 원시연산 단위로만, (c) §5 **mass ledger**(primary/auxiliary ice2/external) 분리, (d) §6 **BLC 3단계**(unroll→early-stop→custom_vjp), (e) §8 **compatibility_target 분리**·gate 등급 G0–G3·**forecast baseline B0–B3**·테스트 네이밍, (f) §8 `where` **safe/guarded/cond 정책**, (g) §12 **MVP 축소·hybrid 기본 import 제외**. 세부 P0 체크리스트·세분 마일스톤은 구현계획 문서 참조.
>
> **v0.5 변경점**: Python AD 가능성 감사(`AD가능성_감사_Python.md`) 반영 — (a) §4.4에 **VeryCold 히스테리시스**(경로 의존 이산상태) 명시, (b) §6에 **forcing-bias 제어**를 방안 A의 대체 제어변수로 추가, (c) §5 목록에 **CalcLE 무수분 증발 게이트·WearFactors 3× 점프·CalcAlbedo None** 하드포인트 명시, (d) §8에 **4계층 검증 픽스처/게이트**(pinned·no-coupling core·paper-physics·forecast skill) 편입. 또한 BLC 40회 반복 솔버·Coupling.py 결함(속성명 불일치·primary ice 미저장·복원 필드 오류·복사 복원 주석)을 코드로 확정.
>
> **v0.4 변경점**: (1) 적대적 검토②(`적대적검토2_미분연산_하이브리드DA.md`)의 Critical/Major 반영 — **모드 선택 규칙 + 소수모수 JVP 민감도**(§7.4, C-J1), **하이브리드-4DEnVar**(앙상블 B·α-제어변수·국소화, §7.7, C-D1), **B^{1/2} 제어변수 변환 전처리**(§7.4, C-H2), GN 우선·**HVP+Lanczos/Hutchinson UQ**(§7.6, M-H1/H3), **NN=약제약 모델오차** 통합·**순환(cycling) 추정**(§7.8, M-D3/D4), adjoint **관측영향(FSOI)**·**linearize-once**(§7.9/§7.4), **TLM test·HVP 대칭성·조건수 모니터링**(§8, M-Ver1/M-C1). (2) **격차분석**(`격차분석_Fortran_vs_Python.md`) 반영 — 추가 변환 실필요분은 소량(§2.6), 로드맵 M0에 격차 이식 포함(§9).
>
> **이전 이력**: v0.3 = 적대적 검토① 반영(DoD 정량화, CFL, 상변화 분리, BLC 암시, 레짐별 검증) + §7 하이브리드 시스템 확장 + JAX 확정. v0.2 = Karsisto (2024) 지배방정식(식 1–50) 이론 보강.

---

## 1. 개요와 목표

### 1.1 무엇을 만드는가
RoadSurf는 핀란드 기상청(FMI)의 도로 노면 기상 모델로, 노면 온도와 저장항(물·눈·얼음·서리)을 예측한다. 원본은 Fortran 라이브러리(약 4,000줄, C++ 인터페이스)이며, 물리는 Karsisto (2024, *Geosci. Model Dev.*, 17, 4837–4853)에 문서화되어 있다. FMI는 연구용 **Python 버전**(RoadSurf-Python, numpy/pandas 기반, 약 2,500줄)도 공개했다.

dROAD는 이 모델을 **입력·초기상태·물리 파라미터에 대해 자동미분(AD)이 가능한** 형태로 재구현하는 프로젝트다. 이를 통해 gradient 기반 최적화로 (a) 관측에 맞춘 초기상태 보정(자료동화), (b) 물리 파라미터 보정(calibration)을 수행한다.

### 1.2 왜 미분가능해야 하는가
현재 RoadSurf의 관측 정합 방식은 **커플링(coupling)** — 복사 보정계수를 반복 탐색(secant/bisection)해 초기화 구간 끝에서 모델 노면온도를 관측에 맞추는 방식이다. 이는 단일 스칼라(복사계수) 조정에 국한된 휴리스틱 역문제 풀이다.

미분가능 버전은 이를 일반화한다. **핵심 지향점은 자료동화와 물리모수 최적화를 하나의 미분가능 연산그래프 위에서 동시에 푸는 것**이다(§7):
- **자료동화**: 손실 $J$(모델 vs 관측)를 초기 온도 프로파일·저장항에 대해 미분(VJP=adjoint) → 4D-Var류 변분 동화.
- **파라미터 개선**: $J$를 물리 파라미터(방사율, 공극률, 열전도·열용량, 임계온도 등)에 대해 미분 → gradient 기반 보정.
- **동시 추정**: 제어변수 $z=[$윈도우별 초기상태 $x_0^{(w)}$; 전역 물리모수 $\theta$; (선택) 신경망 가중치 $\theta_{NN}]$을 **하나의 목적함수**로 결합. VJP로 1차 gradient, JVP∘VJP로 Gauss–Newton/Hessian-vector 2차 정보까지(§7).
- **하이브리드 ML**: NN 성분(잔차 보정·불확실 폐합식 대체)을 그래프에 삽입하면 $\theta_{NN}$도 같은 VJP로 학습 → 물리모수와 **공동 학습**(§7.5).
- **부가 효과**: 민감도 분석(JVP), 사후 불확실성(Hessian 기반 Laplace, §7.6).

### 1.3 성공 기준 (Definition of Done) — 정량화 (검토 C4·C5 반영)
1. **정확도(레짐별)**: `exact` 모드가 RoadSurf-Python과 (a) **무강수·무상변화 구간** 노면온도 상대오차 < 1e-9(float64), (b) **상변화 활성 구간**은 비트 일치가 아니라 **결정경계(분기) 시퀀스 일치 + 저장항 < 1e-3 mm, Tsurf < 1e-2 °C** 로 완화. 핵심 물리(식 27–32, 46–50)의 **Fortran 삼자 대조는 `compatibility_target=fortran_compat`일 때만 hard gate**; `python_compat`(P0 기본)에서는 optional audit sidecar이고 hard gate는 RoadSurf-Python no-coupling fixture + paper-physics validation suite다(§8.3).
2. **미분가능성**: 초기상태·파라미터에 대한 VJP gradient가 중심차분과 상대오차 < 1e-4로 일치(`check_grads`). 또한 JVP(tangent)와 VJP(adjoint)의 **내적 일치**($\langle Jv,u\rangle=\langle v,J^\top u\rangle$, 상대오차 < 1e-8).
3. **자료동화**: 쌍둥이 실험에서 알려진 초기상태를 관측으로부터 복원(상태 RMSE↓), **독립 검증구간에서 운영 커플링 대비 노면온도 RMSE 동등 이상**.
4. **파라미터 최적화**: 쌍둥이 실험에서 **알려진 물리모수 3–5개를 목표 범위(예: ±10%) 내 복원**, 다중 사례에서 보정값이 물리적으로 타당·안정.
5. **동시 추정**: 초기상태와 물리모수를 **하나의 목적함수**로 공동 최적화하는 파이프라인이 수렴하고, Gauss–Newton(HVP 기반) 경로가 1차법 대비 반복수를 단축.
6. **수치 강건성**: 보정 전 파라미터 탐색 범위에서 forward가 **CFL 안정**(NaN/발산 없음, §3 N11).

---

## 2. 원본 모델 분석

### 2.1 시뮬레이션 흐름 (Python `main.py` 기준)
```
초기화 → for i in range(SimLen):
    CheckValues            # 입력 이상치 처리
    CouplingOperations1    # 커플링 위상 판정 + 복사계수 적용 (i를 되감을 수 있음!)
    SetCurrentValues       # 현재 시각 대기 강제력 로드
    roadModelOneStep(i):
        PrecipitationToStorage   # 강수 상변화 판정 → 저장항 추가
        ModRadiationBySurroundings  # sky view / 지평선 보정 (선택)
        BalanceModelOneStep      # ★ 열수지 1스텝 (지중 온도 프로파일 전진)
        WearFactors + RoadCond   # 교통 마모 + 노면상태 → 저장항 갱신
        CalcAlbedo               # 눈/얼음 상태로 알베도
    SaveOutput
    CheckEndCoupling       # 커플링 구간 끝이면 복사계수 재조정 트리거
```

### 2.2 상태벡터 (시간에 따라 전진되는 값)
| 그룹 | 변수 | 크기 | 비고 |
|---|---|---|---|
| 지중 온도 | `ground.Tmp` | NLayers+2 = **17** | index 0=기온, 1..15=지층, 16=바닥 기후값. 실질 예측 상태의 핵심 |
| 표면 저장항 | `SrfWatmms, SrfSnowmms, SrfIcemms, SrfIce2mms, SrfDepmms` | 5 | 물·눈·얼음·2차얼음·서리 (수당량 mm) |
| 융해 상태 | `Q2Melt, T4Melt` | 2 | 융해에 필요/사용 열, 융해 임계온도 |
| 표면 보조 | `TsurfAve, Albedo, VeryCold(bool), WearSurf(bool)` | — | 첫 두 층 평균온도 등 |
| 커플링 상태 | `RadCoeff, SwRadCof, LWRadCof, Tsurf_end_coup1, TsurfNearest{Above,Below}, Coupling_iterations, ...` | 다수 | 반복 탐색용 (§5 참조) |

**설계 관점의 상태벡터**: `x = (Tmp[17], SrfWat, SrfSnow, SrfIce, SrfIce2, SrfDep, Q2Melt, T4Melt, Albedo)` — 연속값 위주. bool·정수 플래그는 별도 처리(§4.4).

### 2.3 강제력(입력) 시계열
`Tair, Tdew(→RH), VZ(풍속), SW(단파), LW(장파), SW_dir, Prec/PrecPhase, TSurfObs`. 시각(hour)은 낮/밤 교통마찰 전환에 쓰인다. 모두 시간축 배열로 사전 준비 가능.

### 2.4 파라미터 (calibration 후보)
- **`PhysicalParameters`** (~30): `Emiss`(방사율 0.95), `Poro1/Poro2`(공극률), `vsh1/vsh2`(건조 열용량), `LVap/LFus`(잠열), `TClimG/AZ/DampDpth`(바닥 경계), 조도(`ZMom/ZHeat`), 흙 물성(`Silt,RhoB→Afc..Efc`) 등. → 지중 열전도·열용량·경계층에 영향.
- **`RoadCondParameters`** (~40): 임계온도(`TLimFreeze=-0.25, TLimMeltSnow/Ice=0.25, TLimMeltDep=1.25, TLimDew=0.25`), 알베도(`AlbDry=0.1, AlbSnow=0.6`), 밀도·잠열(`WatDens, WatMHeat`), 저장항 min/max 클램프, 마모계수 등.
- **`ModelSettings`**: `NLayers=15`, `DTSecs`(시간스텝), 낮/밤 마찰(`TrfFricDay=10, TrfFricNgt=5`), 최소풍속(`CalmLim`) 등.
- **`ground` 초기값**: `WCont`(층별 함수량), `ZDpth`(층 깊이) — 초기 온도 프로파일과 함께 자료동화 대상 가능.

### 2.5 핵심 물리 (미분 대상 방정식)
> 방정식 번호는 Karsisto (2024)를 따른다. 전체 목록·미분가능화 대응은 **부록 B** 참조.

**표면 에너지수지 (식 1)**: $G = R_n - \text{LE} + H + T_r$ — 지중 열류 = 순복사 − 잠열 + 현열 + 교통열. 이 식이 지표 경계조건으로 지중 프로파일을 구동한다.

1. **지중 열전도 (식 27–28, 명시적 전진차분)** — `BalanceModel.calcProfile`. 연속식 $\rho_g c_g\,\partial_t T = \partial_z(K\,\partial_z T)$ (식 27, Patankar 1980)를 층 부피·시간에 대해 적분해 **forward-difference explicit**로 이산화(식 28):
   $$T_i^{j+1} = T_i^{j} + \frac{1}{\rho_g c_g \frac{z_{i+1}-z_{i-1}}{2\Delta t}}\left(K_i\frac{T_{i+1}^j-T_i^j}{z_{i+1}-z_i} - K_{i-1}\frac{T_i^j-T_{i-1}^j}{z_i-z_{i-1}}\right)$$
   **implicit solver가 없어** 미분가능화가 크게 유리하다(선형계 역전파 불필요, `lax.scan`으로 그대로 전개). 단, 명시적 스킴이므로 CFL 안정 조건이 $\Delta t$·층두께에 걸린다(계산 안정성 주의). 출력 노면온도 = **첫 두 층 평균**(최상층보다 관측 정합 우수, 논문 3.2).
2. **순복사 (식 2, 9)** — `CalcRNet`: $R_n = \text{SW}_{down}(1-\alpha_s)\,c_{SW} + \varepsilon\,\text{LW}_{down}\,c_{LW} - \varepsilon\sigma_{SB} T_{sK}^4$. 흑체방출 $\text{LW}_{up}=\varepsilon\sigma_{SB}T_{sK}^4$(식 9). 매끄러움(단 $T^4$). SVF/지평선 보정은 식 4–8(선택).
3. **알베도 (식 10)** — `CalcAlbedo`: 눈이 있으면 $\alpha_{snow}$, 얼음 우세 시 $\alpha_s=\alpha_{asp}+\frac{\text{St}_{sum,ice}/1.5}{\alpha_{snow}-\alpha_{asp}}$, 총빙 $>1.5$mm이면 $\alpha_{snow}$. 저장항 문턱에 의한 **분기**.
4. **열용량/전도 (식 29–36)** — `CalcHCapHCond`: 습윤 지반 부피열용량 $\rho_g c_g=(1-\phi)\rho_s c_s+\phi\rho_w c_w$(식 29). 물/얼음 **분기**: $T_w\ge0$이면 물 밀도·비열이 온도 의존(식 30–31), $<0$이면 얼음 상수(920 kg m⁻³, 2100 J kg⁻¹K⁻¹). 열전도 $\lambda=A+B\theta-(A-D)e^{-(C\theta)^E}$(식 32–36).
5. **경계층 전도·현열·잠열 (식 11–26)** — `BoundaryLayer.CalcBLCondAndLE`. 현열 $H=\text{BLC}(T_s-T_a)$(식 11), BLC(식 12)는 안정도 보정 $\Psi_h,\Psi_m$(식 16–19, 안정/불안정 분기)에 의존해 **반복 수렴**(§3 N9). 잠열 LE(식 20), 포화수증기압은 물(식 22)/얼음(식 23) 분기, 응결↔증발은 LE 부호로 결정.
6. **강수 상변화 (식 42)** — `CalcPrecType`: 입력 위상이 없으면 $P_{rain}=1/(1+e^{P_{exp}})$, $P_{exp}=22-2.7T_a-0.2\text{RH}$ — **이미 시그모이드**(미분가능 친화적). $<0.3$ 눈, $>0.7$ 비, 사이는 진눈깨비(반반).
7. **저장항 (식 43–50)** — `Storage.py`: 마모(식 43, 저장항별 계수), 증발/응결(식 44–45), **융해**(식 46–49)와 **동결**(즉시, 잠열 미반영 → 노면온도 진동, §3 N10), 눈↔물↔얼음 전환(식 50). 저장항 min/max 클램프. **하드 임계값이 가장 밀집**(§5의 주 대상). Table 1에 각 사건의 저장항 증감 정리.
8. **커플링 (식 40)** — `Coupling.py`: 초기화 구간 복사보정계수 반복 탐색, 이후 $C_f(t)=1+C_R e^{-t/t_c}$($t_c=4$h)로 1에 점근(§6).
9. **완화 (식 41, 선택)** — 관측→예보 전환 점프 완화 $X(t)=X_F(t)-(X_{FO}-X_O)e^{-t/t_c}$. Python 버전은 미구현.

### 2.6 포팅 범위와 추가 변환 분량 (격차분석 요약)
두 repo 대조 결과(`격차분석_Fortran_vs_Python.md`): Python은 **미분화에 필요한 물리 코어를 사실상 커버**한다(열수지·지중전도·저장항·상변화·경계층·복사·태양위치·커플링 반복). 순수 미포팅 Fortran **라이브러리** 기능은 ~300–450줄에 불과하고, 방안 A(커플링 OFF) 기준 **실필요분은 ~150–200줄**이다.
- **추가 변환 후보(소량)**: Relaxation(~50, 저우선), getTempAtDepth 출력보간(~30, 저우선), 입력 QC·지반물성 초기화 세부(B4/B5, ~150–200, 중간). 입력 복사계수 경로는 방안 A면 불필요.
- **불필요**: ConnectFortran2Carrays(C바인딩 86줄), RoadSurf.f90(인터페이스 272줄), C++ 예제(main·I/O·SkyView 계산).
- **선택 결정**: SVF/지평선 **지형 보정을 쓸 경우에만** SkyView 계산을 예제 C++에서 별도 포팅(수백 줄). 적용부(ModRadiation)는 이미 존재.
- **함의**: 이 프로젝트의 작업량 핵심은 "빠진 Fortran 번역"이 아니라 **기존 Python 물리(~2,100줄)의 JAX 전환·미분화·검증**(M1–M4)이며, §8 Fortran 삼자 대조 과정에서 B4/B5 격차와 §8.1 불일치를 함께 흡수한다.

---

## 3. 미분가능화의 핵심 난점

| # | 난점 | 원본 형태 | 영향 |
|---|---|---|---|
| N1 | **가변 OO 상태** | 클래스 속성 제자리 변경, `arr[i]=..`, `.copy()` | JAX는 순수함수·불변배열 필요 → 함수형 상태로 전면 재설계 |
| N2 | **데이터 의존 분기** | `if Tsurf > TLimMelt: ...` 등 상태값 기반 if/else | `jit`/`grad`에서 트레이싱 불가 또는 gradient 0 |
| N3 | **하드 임계값** | 동결 −0.25°C, 융해 0.25°C 등 계단 전환 | 임계점에서 gradient=0/불연속 → 최적화 신호 소실 |
| N4 | **클램프/포화** | `if x<Min: x=0`, `if x>Max: x=Max` | subgradient 존재(→`clip`으로 처리 가능) but 죽은 영역 |
| N5 | **시간 루프 되감기** | 커플링이 `i`를 구간 시작으로 리셋, 최대 25회 재실행 | 내부 최적화 루프 → AD로 감쌀지/대체할지 결정 필요(§6) |
| N6 | **정수/불리언 플래그** | `SnowType`, `WearSurf`, `VeryCold`, `simulation_failed` | 이산 상태 → gradient 없음, 마스킹/사전계산으로 우회 |
| N7 | **결측 센티넬** | `-9999`, `<-100` 관측 결측 표시 | 마스크로 분리, 손실에서 제외 |
| N8 | **부작용** | `print`, `datetime.now()`(율리우스일), CSV I/O | 순수함수에서 제거·사전계산 |
| N9 | **경계층 반복 솔버** | BLC를 안정도 보정과 함께 최대 40회 반복 수렴(식 12–19). 커플링과 별개의 **두 번째 내부 반복** | 고정 반복 unroll + 역전파, 또는 수렴값에 음함수정리(IFT) 적용(§6과 동일 전략). 안정/불안정 분기는 `where` |
| N10 | **동결 잠열 불연속** | 동결이 **즉시**·잠열 미반영(논문 3.6): 물→얼음 순간 전환, 노면온도가 동결한계 위아래로 **진동** 가능 | gradient에 치명적. §5의 **엔탈피(유효 열용량)법**으로 (a)지층 상변화를 연속·에너지보존 형태로 재정식화; (b)표면저장항은 질량보존 soft 전달 |
| N11 | **CFL × 보정 발산** | 명시적 스킴은 $\Delta t<\Delta z^2/(2\alpha)$ 필요. **보정이 $K,\rho c$를 바꿔 $\alpha$ 변경** → 최적화 중 발산·NaN → gradient NaN | 상단층 서브스텝 또는 상단 몇 층 **부분 암시(θ-method)** + 미분가능 선형해(lineax), 재매개화 시 CFL 여유 제약, 발산 감지 페널티 |

---

## 4. 아키텍처 설계 (JAX)

### 4.1 설계 원칙
- **함수형·불변**: 모든 스텝을 `x_{t+1}, y_t = step(x_t, forcing_t, params)` 순수함수로. 상태·파라미터는 **pytree**(`flax.struct.dataclass` 또는 `NamedTuple`).
- **파라미터/상태/강제력 분리**: `params`(미분 대상 물리 파라미터) · `state`(전진 상태) · `forcing`(시간축 입력, 미분 대상 아님 기본) · `static`(NLayers 등 정적 설정, `static_argnums`).
- **시간 루프 = `lax.scan`**: 역전파 메모리 효율 + `jit` 호환. (긴 시계열은 필요시 `jax.checkpoint`로 재계산-메모리 절충.)
- **float64 필수**: `jax.config.update("jax_enable_x64", True)` — 열확산 수치 재현·gradient 검증에 필수. (성능: CPU float64 + `vmap` 우선; GPU float64는 처리량 1/32~1/64라 배치 성능전략은 §11.)
- **이중 모드**: `exact`(원본 하드 임계값, 정확도 검증용) / `smooth`(매끄러운 대체, 최적화용). 동일 코드에서 스위치.
- **질량·에너지 보존 하드제약 (검토 M3)**: `smooth_compat`에서도 저수지 간 전달은 **전달량 = soft게이트 × 가용량**으로 정식화해 비음수·보존을 구조적으로 보장. 보존은 단일 "총수분"이 아니라 **mass ledger**(primary=water+snow+ice+deposit / auxiliary=ice2 / external source·sink·overflow)로 분리 진단(§5, 구현계획 §3). energy residual은 compat/enhanced 모드 별로 해석. NN 출력도 동일 제약 하 삽입(§7.5).
- **정적 vs 동적 이산 분리 (검토 M4)**: 시각 기반(낮/밤)처럼 상태 무관한 이산은 **사전계산**; `VeryCold`·`WetSnowFrozen`·`SnowType`처럼 **진화 상태 의존** 이산은 연속상태로 승격(예: sigmoid)하거나 straight-through로 gradient 경로 명시.
- **프레임워크 = JAX (단, 백엔드는 단계적으로 활성)**: 최종 백엔드는 JAX(작은 상태·긴 루프에 jit+scan 최적, `vmap`·암시적미분 생태계). **그러나 M1은 pure NumPy parity kernel을 먼저 고정**하고, jit/scan/custom_vjp는 parity fixture 통과 후 단계적으로 켠다(NumPy → JAX eager(`disable_jit`) → scan → jit → custom_vjp; `구현계획_P0_derisked.md` §8). event-heavy 모델을 처음부터 jit/scan으로 포팅하면 실패 원인 분리가 불가능하다. 개발 편의는 Equinox + `jax.disable_jit()`. 전환 트리거는 §11.

### 4.2 상태·파라미터 표현
> **주의(검토 no-go #2)**: 아래 `jax.Array`/`flax.struct` 예시는 **M2 이후 구현용**이다. **M1 reference kernel은 backend-neutral NumPy**(`@dataclass(frozen=True)` + `np.ndarray`/`float`)로 작성하고 이 JAX 객체를 쓰지 않는다(P0 §8 백엔드 단계화). M1은 RoadSurf-Python과 함수단위 parity만 목표.
```python
# M1: backend-neutral NumPy reference (JAX 미사용)
@dataclass(frozen=True)
class StateNP:
    tmp: np.ndarray; water: float; snow: float; ice: float; ice2: float; deposit: float

# M2+: JAX 구현 (아래는 M2 이후 예시)
@flax.struct.dataclass
class State:
    Tmp: jax.Array          # (NLayers+2,) 지중 온도 프로파일
    SrfWat: jax.Array       # scalar 저장항들
    SrfSnow: jax.Array
    SrfIce: jax.Array
    SrfIce2: jax.Array
    SrfDep: jax.Array
    Q2Melt: jax.Array
    T4Melt: jax.Array
    Albedo: jax.Array
    # 커플링/보조 연속상태 필요시 추가

@flax.struct.dataclass
class Params:               # 미분·보정 대상
    Emiss: jax.Array; Poro1: jax.Array; Poro2: jax.Array
    vsh1: jax.Array; vsh2: jax.Array
    TLimFreeze: jax.Array; TLimMeltSnow: jax.Array; TLimMeltIce: jax.Array
    AlbDry: jax.Array; AlbSnow: jax.Array
    TrfFricDay: jax.Array; TrfFricNgt: jax.Array
    # ... (calibration 후보 전체; 미분 안 할 값은 static으로)
```

### 4.3 스텝 함수 구조
```python
def step(state, forcing_t, params, static):
    state = precipitation_to_storage(state, forcing_t, params)
    rnet  = calc_rnet(state, forcing_t, params)
    ground = calc_hcap_hcond(state, params, static)   # 물/얼음 분기 → where
    state = calc_profile(state, ground, forcing_t, params, static)  # 명시적 오일러
    state = melting(state, params)                     # 상변화 (smooth/exact)
    state = road_cond(state, forcing_t, params)        # 마모·저장항 클램프
    state = calc_albedo(state, params)
    y = observe(state)                                 # 출력(노면온도 등)
    return state, y

def rollout(state0, forcings, params, static):
    step_fn = lambda s, f: step(s, f, params, static)
    state_T, ys = jax.lax.scan(step_fn, state0, forcings)
    return ys
```

### 4.4 이산 요소 처리
- **낮/밤 마찰**: `hour`는 입력 → 시간축 배열로 **사전계산**(`is_night[t]`). gradient는 마찰 파라미터로만 흐름(`where(is_night, TrfFricNgt, TrfFricDay)`).
- **물/얼음 열물성 분기(N2)**: `where(Tmp>=0, water_props, ice_props)` — 두 branch 모두 계산 후 선택. 경계에서 연속(값은 연속, 도함수 꺾임 → 필요시 §5 smooth).
- **SnowType/WearSurf 등 bool**: 연속 완화 or 마스크. 대부분 저장항 부호(>0)에 연동 → `smooth_gate`(§5)로 표현.
- **VeryCold 히스테리시스(경로 의존, 감사 범주5)**: `Cond.RoadCond`의 `VeryCold`는 이중 문턱(`TLimColdH`/`TLimColdL`)과 **이전 상태**에 의존하는 히스테리시스 → 단순 임계보다 어렵다. 값을 **연속 상태변수로 승격**(예: 지중온도의 sigmoid를 상태로 carry)하거나 이력을 State에 담아 `lax.scan` 캐리로 전개. 사전계산 불가(동적 이산).
- **`where` 안전 정책 (외부 검토 반영)**: JAX `where`는 선택 안 된 branch도 계산해 `sqrt(-)`,`log(≤0)`,0-나눗셈,overflow를 낼 수 있다(예: 경계층 $\sqrt{1-16\zeta}$). 모든 분기를 4등급으로 태깅: `safe_where`(두 branch 전영역 유효) / `guarded_where`(입력 sanitize: `sqrt(max(·,ε))`) / `lax_cond`(활성영역 밖 계산 불가) / `custom_smooth`(전영역 유효 매끄러운 식으로 대체). 상세는 `구현계획_P0_derisked.md` §4.

---

## 5. 하드 임계값 → 매끄러운 대체 전략

최적화 신호가 살아있으려면 계단·클램프를 미분가능 근사로 바꾼다. **`smooth` 모드에서만 적용**하고, `exact` 모드는 원본과 비트 수준 근접을 유지해 검증에 쓴다.

| 원본 패턴 | 매끄러운 대체 | 비고 |
|---|---|---|
| `if x > thr: A else B` | `σ((x-thr)/τ)·A + (1-σ(..))·B`, σ=sigmoid | τ=전환 폭(온도는 0.1~0.5°C 권장) |
| `max(x, 0)` / 하한 클램프 | `softplus` 또는 `clip`(subgrad OK) | 저장항 음수 방지 |
| `min(x, Max)` / 상한 | `-softplus(-(x-Max))+Max` 또는 `clip` | 오버플로 |
| 융해량 계단 배분(식 46–49) | 가중 배분(σ 게이트) | `Storage.melting`, `SnowStorage` |
| 강수 상변화(식 42) | **불필요** — 원식 $1/(1+e^{P_{exp}})$가 이미 시그모이드 | 위상 미제공 시. 위상 입력 시엔 이산값→램프 |
| 눈↔물↔얼음 비율 전환(식 50, `WSrat`>0.6/0.1) | σ 게이트 | 문턱 전환 |
| **무수분 증발 차단**(`CalcLE`: `LE>0 & SrfWat≤0 → LE=0`) | σ게이트로 물 있을 때만 증발 | 불연속 하드 게이트 |
| **눈 마모 3× 점프**(`WearFactors`: `SrfSnow<0.2 → ×3`) | `SrfSnow`에 대한 매끄러운 배수 램프 | 계단 점프(불연속) |
| **알베도 None 반환**(`CalcAlbedo`: `WearSurf=False`/무분기 시 None) | 기본치(`AlbDry`) 보장 + 전환 σ게이트 | 버그성: 하류 곱셈 폭발 방지 |

**상변화는 서로 다른 두 위치를 분리해 처리한다 (검토 C1 — 이전 판은 이를 혼동).**
RoadSurf의 상변화는 성격이 다른 두 곳에 있으며 **하나의 기법으로 뭉뚱그리면 안 된다**:

**(a) 지층 온도 방정식의 잠열 — 엔탈피(유효 열용량)법.**
논문 3.6이 말하는 "노면온도가 동결한계 근처에서 진동"은 **지중 온도**에 잠열이 반영되지 않아 생긴다(N10). 여기에 잠열 $L_{fus}$를 유효 열용량에 흡수: 0°C 부근 폭 $\Delta T$에서 $c_{eff}(T)=c+\frac{L_{fus}}{\Delta T}\,\varphi'(T)$ ($\varphi$=녹은분율의 매끄러운 함수). 잠열이 에너지보존적으로 반영되어 진동이 사라지고 온도가 연속·미분가능해진다.

**(b) 표면 저장항의 질량 이전 — 질량보존형 soft 전달.**
물(mm)↔얼음(mm)↔눈(mm) 전환(식 46–50)은 단일 매질의 엔탈피가 아니라 **별개 저수지 간 질량 이동**이라 (a)로 해결되지 않는다. 각 전환을 **전달량 = σ게이트 × 가용저장량** 으로 정식화해 $\sum$저장항 보존·비음수를 구조적으로 보장(§4.1 M3). 예: 동결 $\Delta m = \sigma\!\big(\tfrac{T_{melt}-T_s}{\tau}\big)\cdot \text{SrfWat}$ 를 물→얼음 이동.

두 처리의 **상호작용**(융해수가 지층 온도에 주는 피드백)도 검증 대상에 포함한다. `smooth` 모드 핵심 구성요소로 M4에서 도입하되 `exact` 모드와 병존시켜 물리 영향(§8, 민감도)을 평가한다.

**모드 3분할 (검토 반영 — 검증이 꼬이지 않게).** 이전 판이 "smooth"에 뭉뚱그린 것을 셋으로 나눈다:
- **`roadsurf_exact`**: 하드 분기 + 원본 잠열 한계. python/fortran compat 대상(§8).
- **`smooth_compat`**: exact와 **물리 의도 동일**, 미분가능 근사만. deviation budget(§8 G3) + **의미 있는 곳에서만** τ→0 수렴.
- **`enhanced_enthalpy`**: 위 (a)엔탈피법처럼 **물리를 바꾸는 모드**. exact로 수렴하면 물리개선도 사라지므로 **τ→0 수렴을 요구하지 않고** 물리진단·미관측 예보스킬로 평가.

**τ 정책**: τ(엔탈피 $\Delta T$ 포함)는 하이퍼파라미터. **τ→0 수렴은 매끄러운 원시연산 단위로만 검증**한다 — 상변화 캐스케이드·히스테리시스가 개입하는 롤아웃은 임계점 근처 branch flip이 장기 증폭되어 end-to-end 수렴이 보장되지 않는다. 최적화 시 큰 τ→점진 축소(annealing).

**mass ledger (보존 테스트 정의).** "총수분 보존"을 단일 지표로 쓰지 않는다. `primary_mass = water+snow+ice+deposit`, `auxiliary = ice2`(hazard proxy, 기본 미포함), `external = rain+snow_precip+condensation − evap − wear_export − overflow_export` 로 분리하고 process별 원장을 명시(`구현계획_P0_derisked.md` §3). 검증은 **primary_mass residual**과 **ice2 residual**을 따로 본다. clipping/overflow는 수치오차가 아니라 external_export.

---

## 6. 커플링(coupling) 처리 전략

커플링은 초기화 구간에서 복사계수 `RadCoeff`를 반복 조정(최대 25회, `i`를 구간 시작으로 되감아 재적분)해 노면온도를 관측에 맞추는 **내부 역문제 풀이**다. 미분가능 버전에서 이를 다루는 세 가지 방안:

**방안 A — 커플링 OFF + gradient 자료동화로 대체 (권장 출발점).**
커플링을 끄고, 대신 dROAD의 gradient로 초기상태·복사계수를 직접 최적화한다. 이는 프로젝트의 본래 목적(변분 동화)과 정확히 일치하며 미분가능화가 가장 단순하다. 제어변수는 **두 가지 형태** 중 선택·병용한다:
- **복사(SW/LW) 계수 제어**: 커플링의 "복사계수 조정"을 손실의 제어변수로 흡수(원 커플링과 직접 대응).
- **forcing-bias 제어(감사 권장)**: 강제력(예: $T_{air}$·복사)의 편차/바이어스 항을 제어변수로 두어 관측에 정합. 복사만으로 설명 안 되는 계통 오차를 흡수하며, §7.8의 **NN=약제약 모델오차**와 자연스럽게 연결된다.

> **주의(감사 확정)**: Python `Coupling.py`는 **저장/복원 속성명 불일치, primary ice 미저장, 복원이 잘못된 소문자 필드에 기록, 복사 복원 주석처리**로 **DA 레퍼런스로 부적합**. 커플링을 비트 재현 대상으로 삼지 말 것. 재현이 꼭 필요하면 Fortran `Coupling.f90`을 진실로(방안 B/C).

**방안 B — 커플링을 미분가능 고정점으로 재정식화.**
반복 탐색을 매끄러운 잔차 $r(c)=T_s^{end}(c)-T_{obs}$의 근 찾기로 보고, 수렴값 $c^*$에 대해 **음함수 정리(implicit function theorem)**로 gradient를 구한다(전체 반복을 unroll하지 않아 메모리 효율적). **주의(검토 M2)**: 원본 커플링은 bisection/secant + 25회 상한 + 실패 플래그 + 되감기가 섞인 **이산 알고리즘**이라 IFT를 그대로 적용할 수 없다. IFT는 위의 **매끄러운 잔차 $r(c)$로 재정의한 경우에만** 유효하며, 원본 동작을 비트 수준으로 재현하려면 방안 C에 가깝다.

**방안 C — 반복 unroll.**
고정 횟수 반복을 그대로 `lax.scan`/`fori_loop`으로 전개하고 AD. 구현은 쉽지만 되감기 재적분 때문에 비용·메모리 큼. 검증·소규모에서만.

→ **1단계는 방안 A**로 진행(커플링 비활성 경로를 먼저 완성). 원본 커플링 재현이 필요하면 방안 B를 추가한다.

**경계층 반복 솔버(N9)는 3단계로 (외부 검토 반영 — custom_vjp를 처음부터 넣지 않는다).** BLC 수렴(식 12–19, 최대 40회)은 시간 루프 안에서 매 스텝 돈다. custom_vjp 고정점 미분은 수렴·유일성·조건수·분기안정·tol 분리가 모두 필요해, 먼저 넣으면 forward parity 오류와 backward rule 오류를 분리할 수 없다. 따라서:
- **BLC-v0**: 고정 40회 unroll, 조기종료 없음 — gradient smoke test 전용(투명).
- **BLC-v1**: exact forward + 조기종료, backward는 `stop_gradient`/유한차분 감사 — **parity 타깃**.
- **BLC-v2**: `custom_vjp` IFT(수렴점 $(I-\partial g/\partial x)^{-\top}$) — v0·v1 일치 + dot-product test 통과 후에만 기본 경로로 승격, 실패 시 unroll fallback.

custom_vjp는 M3가 아니라 **M4/M5 이후 성능·안정화 작업**. 안정/불안정 분기(식 17 vs 18)는 §8의 `guarded_where`($\sqrt{\max(\cdot,\epsilon)}$)로 처리(선택 안 된 branch의 NaN 방지).

---

## 7. 하이브리드 미분모델링 시스템 (동시 자료동화 + 물리모수 최적화)

dROAD의 최종 형태는 단순한 "미분가능 forward"가 아니라, **연산그래프 위에서 JVP(전방)·VJP(후방)를 조합해 상태·물리모수·(선택)신경망 가중치를 하나의 목적함수로 동시에 추정**하는 시스템이다. 이 절이 그 아키텍처를 정의한다.

### 7.1 연산그래프와 미분 연산자 (JVP / VJP)
전체 모델을 미분가능 원시연산(primitive)의 유향그래프로 본다. forward 연산자를 $\mathcal{M}:z\mapsto \{x_t\}$ (제어변수 → 상태궤적), 관측연산자를 $H$라 하자.

- **JVP (전방모드, pushforward)** $\;\dot y = \partial\mathcal{M}(z)\,\dot z$ — 입력 섭동 $\dot z$의 전파. **접선선형모델(TLM)** 그 자체. 입력 수가 적고 출력이 많을 때, 방향미분·민감도·Hessian-vector에 사용.
- **VJP (후방모드, pullback)** $\;\bar z = \partial\mathcal{M}(z)^\top\,\bar y$ — 출력 코탄젠트의 역전파. **애조인트(adjoint) 모델** 그 자체. 스칼라 손실의 gradient를 **입력 수와 무관하게 한 번의 역전파**로 얻음 → 4D-Var gradient의 핵심.
- JAX에서 두 연산은 각각 `jax.jvp` / `jax.vjp`(또는 `jax.linearize`)로 얻고, 대부분 원시연산의 규칙이 자동 합성된다. **불투명 노드는 커스텀 규칙을 직접 정의**(§7.3).

| | JVP (tangent/TLM) | VJP (adjoint) |
|---|---|---|
| 계산 방향 | 입력→출력 | 출력→입력 |
| 효율적 상황 | 소수 입력, 다수 출력 | 다수 입력, 스칼라 출력 |
| dROAD 용도 | 민감도, Gauss–Newton $Jv$, HVP | **손실 gradient**, GN $J^\top v$ |
| 검증 | $\langle Jv,u\rangle=\langle v,J^\top u\rangle$ (dot-product test) | 좌동 |

### 7.2 제어변수와 결합 목적함수 (동시 추정)
관측 윈도우 $w=1..W$(여러 사례·지점)를 동시에 다룬다. 제어변수를 **국소(윈도우별)·전역(공유)** 로 분리한다:
$$z=\big[\;\underbrace{x_0^{(1)},\dots,x_0^{(W)}}_{\text{윈도우별 초기상태(국소)}},\;\underbrace{\theta}_{\text{물리모수(전역)}},\;\underbrace{\theta_{NN}}_{\text{(선택) NN 가중치(전역)}}\;\big]$$
$$J(z)=\sum_{w=1}^{W}\Big[\tfrac12\lVert H(\mathcal{M}_w(x_0^{(w)},\theta,\theta_{NN}))-y_w\rVert^2_{R^{-1}} + \tfrac12\lVert x_0^{(w)}-x_b^{(w)}\rVert^2_{B^{-1}}\Big] + \lambda_\theta\rho(\theta) + \lambda_{NN}\lVert\theta_{NN}\rVert^2$$
- **동시성의 의미**: 초기상태(윈도우마다 다름, 빠른 국소 변수)와 물리모수(모든 윈도우 공유, 느린 전역 변수)를 **하나의 $J$** 로 결합해 최적화. 이것이 "자료동화 + 물리모수 최적화 동시 수행".
- **gradient 구조**: $\nabla_{x_0^{(w)}}J$ 는 윈도우 $w$의 adjoint(VJP)로 국소 계산; $\nabla_\theta J=\sum_w(\cdot)$ 는 전 윈도우 기여의 합. 윈도우는 서로 독립이라 `vmap`으로 병렬.
- **관측**: 노면온도는 **첫 두 층 평균**에 걸음(논문 3.2, 검토 m3). 저장항 관측이 있으면 다변량으로 확장. 결측(−9999)은 $R^{-1}$ 마스크로 제외.
- **약제약(weak-constraint) 확장**: 모델오차 항을 넣어 $x_0$ 대신 전 궤적을 제어변수로 두는 4D-Var로 확장 가능(모델오차 공분산 $Q$).

### 7.3 커스텀 JVP/VJP 노드 (불투명·비매끄러움 처리)
자동미분이 그대로 통하지 않거나 비효율적인 노드는 **미분 규칙을 손으로 등록**한다. 이들이 시스템의 수치 안정성을 좌우한다.
- **고정점 노드(BLC, 선택적 커플링)**: $x^\*=g(x^\*,\theta)$ 에 대해 `custom_vjp`로 IFT 적용, 역전파 $=(I-\partial_x g)^{-\top}\bar x$ (§6, M1). 순전파 반복은 미분에서 분리.
- **부분 암시 시간스텝(CFL 대응, N11)**: 상단층 θ-method 선형해 $A(\theta)x^{n+1}=b$ 는 `lineax`로 풀고 애조인트는 $A^\top$ 해로 정의(반복 unroll 불필요).
- **매끄러운 상변화 게이트**: 엔탈피 $c_{eff}$ 비선형(§5a)과 질량전달 게이트(§5b)에 해석적 JVP를 부여해 0°C 부근 수치 안정 확보.
- **straight-through(필요 시)**: 동작 재현이 필수인 하드 분기는 forward=exact, backward=smooth 대체 도함수로 연결.

### 7.4 최적화기 — 모드 선택 · 1차 · 2차(JVP∘VJP)
**모드 선택 규칙 (검토 C-J1, 비용 차원이 결정).** "항상 VJP"가 아니다:
- **스칼라 손실 gradient**: 입력 다수·출력 1 → **VJP 1회**(adjoint). 초기상태 $x_0$(고차원) 방향에 최적.
- **소수 모수 민감도 야코비안** $\partial(\text{obs})/\partial\theta$: 관측 시점 수천(출력 다수)·모수 3–5(입력 소수) → **JVP $\times$ 모수 수(3–5회)** 로 전체 TLM 열을 직접 구성(VJP보다 100~1000배 저렴). 이 경우 GN 정규방정식을 **$\theta$-공간(소차원)에서 직접 형성**하는 편이 CG보다 빠를 수 있다.

**2차(변분 동화 핵심): Gauss–Newton / 증분 4D-Var.** 정규방정식을 형성하지 않고 행렬-자유 CG:
- $Jv$=**JVP(TLM)**, $J^\top v$=**VJP(adjoint)** → GN Hessian $\approx J^\top R^{-1}J$ 작용을 matrix-free로. **GN은 항상 PSD라 기본 채택.**
- 완전 HVP=**forward-over-reverse** `jax.jvp(jax.grad(J))(z)(v)` 는 **UQ·곡률진단 용도로 한정**(상변화 근처 부정치 → Newton-CG는 신뢰영역/damping 필수, 검토 M-H1).
- **B^{1/2} 제어변수 변환 전처리 (검토 C-H2, 필수).** 배경 $B^{-1}$·관측 $R^{-1}$ 혼합 헤시안은 조건수가 나빠 순진한 CG가 수렴 안 함. 제어변수를 $\chi=B^{-1/2}(x-x_b)$ 로 변환해 배경항을 항등화(운영 4D-Var 표준). $B^{1/2}$ 는 앙상블(§7.7)로 저랭크 근사.
- **linearize-once, apply-many (검토 M-E1).** GN 내부 CG는 **고정 선형화점**에서 $J,J^\top$ 를 반복 적용 → `jax.linearize`로 한 번 선형화한 연산자(+전치)를 CG 내내 재사용(비선형 롤아웃 재계산 방지).

**1차·준뉴턴**: `optax`(Adam/AdaBelief, NN 포함 대규모), `optimistix`/`jaxopt` L-BFGS(중규모). 모두 gradient는 VJP.
**제약·재매개화**: 물리 범위 무제약화(방사율 `sigmoid`, 양수량 `softplus`/`exp`), CFL 여유 제약(N11), τ annealing(§5).
**다중스케일 대안(조건수 나쁠 때)**: 국소 상태(내부)·전역 모수(외부) **교대/이중수준(bilevel)**. 완전 동시 최적화가 ill-conditioned면 fallback.

### 7.5 하이브리드 ML 통합
NN 성분을 물리 그래프에 **삽입**하되 물리를 골격으로 유지한다:
- **삽입 지점**: (i) 경향 잔차 보정 $\dot x \mathrel{+}= f_{NN}(\text{features})$, (ii) 불확실 폐합식 대체(경계층 안정도함수 $\Psi$, 알베도, 마모계수), (iii) 특성 의존 파라미터장 $\theta=\theta_0+g_{NN}(\text{features})$.
- **공동 학습**: $\theta_{NN}$ 은 제어변수의 일부(§7.2)라 **같은 VJP로 물리모수와 동시에** 학습된다 — 별도 학습 루프 불필요.
- **가드레일**: NN 출력에도 질량·에너지 보존·비음수 제약(§4.1 M3), 출력 범위는 활성함수로 제한, $\lambda_{NN}$ 정칙화로 NN→0에서 순수 물리로 환원(해석성·과적합 방지). "physics-first, NN as correction".

### 7.6 불확실성 정량화 (2차 정보 재사용, 검토 M-H3)
결합 제어차원(윈도우×상태 + 모수 + NN)에서 $H$ 를 형성·역산하는 것은 불가능하므로 **matrix-free HVP를 재사용**한다:
- **Laplace + Lanczos**: 최적점 사후공분산 $\approx H^{-1}$ 의 **선도 고유모드**(불확실성 주방향)를 HVP 기반 Lanczos로 저랭크 추출.
- **Hutchinson**: 확률적 대각추정으로 모수·상태 **분산(오차막대)** 산출.
- **하이브리드 EnVar**과 일관(§7.7): 앙상블 공분산으로 비가우스성 보완.

### 7.7 하이브리드-4DEnVar — 변분 × 앙상블 (검토 C-D1)
"하이브리드 자료동화"의 본뜻: 정적 배경오차와 **앙상블 기반 유동(flow-dependent)** 배경오차를 결합한다.
$$B=\beta_1 B_{static}+\beta_2 B_{ens},\qquad \delta x=\sum_k \alpha_k\circ x'_k\ (\text{α-제어변수}),\ \text{국소화 적용}$$
- 단일 정적 $B$ 는 **결빙 등 레짐 의존 오차구조**를 못 담는다. 앙상블 $B$ 가 이를 담고, **adjoint/TLM이 정확한 하강방향**을 준다 — 앙상블(공분산)+미분(gradient)의 시너지가 미분가능 모델의 핵심 이점.
- 앙상블은 `vmap`으로 저비용 생성. α-제어변수·국소화로 표본부족·원거리 허위상관 억제. §7.4의 $B^{1/2}$ 전처리에 앙상블 $B$ 를 저랭크로 투입.
- **프런티어(옵션, 검토 m-D2)**: 미분가능성으로 $B/R$(또는 그 NN 파라미터화)까지 손실에 넣어 **학습**.

### 7.8 순환(cycling)·온라인 추정 + NN=모델오차 (검토 M-D4/M-D3)
- **순환 아키텍처**: 새 관측이 오면 동화창을 밀며 상태를 갱신하고, **전역 모수·NN은 창을 넘겨 지속(warm-start)**. 창별 상태(빠름)·전역 모수(느림, 창간 누적 gradient/증분 갱신)의 시간 분리.
- **NN = 약제약 모델오차 (통합)**: §7.5의 NN 잔차보정은 곧 **학습된 계통 모델오차항**이며, 약제약 4D-Var(§7.2)의 모델오차 제어변수와 동일 대상. 결합하면 NN이 계통오차를 학습하면서 DA가 상태를 잡는다(하이브리드 DA ⊗ 하이브리드 ML의 정점).
- (선택) 증강상태 EnKF와의 접속으로 모수 온라인 추정.

### 7.9 진단·검증 지표
- **관측영향(FSOI, 검토 M-V1)**: adjoint(VJP) 재사용으로 **각 관측·시점이 예측 개선에 기여한 정도**를 정량화 → 관측 품질관리·타깃 관측. 추가 비용 거의 0.
- **민감도**: JVP로 입력·모수→진단(최대 빙량, 첫 결빙시각 등) 방향미분.
- **정합성 검증**: JVP↔VJP dot-product test, TLM test, HVP 대칭성(§8).
- **성능 검증**: 쌍둥이 실험으로 식별성·복원 정확도; 독립구간 **운영 커플링 대비 RMSE**; 다중 사례 전역 모수 안정성.

---

## 8. 검증 전략

### 8.1 논문 ↔ Python 코드 불일치 (검증 기준 명확화)
논문 정독 결과, Karsisto (2024) 방정식과 RoadSurf-Python 구현 사이에 차이가 확인됐다(Python README도 "Fortran 버전과 일부 차이"를 명시). **dROAD의 1차 대조 기준은 Python 코드**(포팅 원본)로 하되, 아래 항목은 논문 원식과도 교차 확인한다.

| 항목 | 논문 | Python 코드 | 판단 |
|---|---|---|---|
| 지층 깊이 증가식 | 식 37: $Z_{i+1}=Z_i+0.01442(i-1)+Z_{Add}$ (**선형** 증가) | `initDepth`: `ZDpth[I+1]=ZDpth[I]+0.0103*1.4**I+ZAdd` (**기하** 증가) | 서로 다른 층 구조 → 열용량·전도·프로파일 전반에 영향. **1차 기준=Python**, 논문식은 옵션 |
| 지층 수 | 3.2.2: "16 layers" | `NLayers=15` (`Tmp` 크기 17 = 공기+15층+바닥) | 계수 방식 차이 가능성. 검증 시 실제 배열 크기 기준 |
| 물 밀도·비열(식 30–31) | 계수 명시 | `CalcHCapHCond` 계수 **일치** 확인 | ✅ 일치 |
| 융해 온도 변화 | 식 49로 $T_1$ 갱신 | `melting`에서 `CanMeltingChangeTemperature` 기본 False → 온도 미변경 | 옵션 플래그로 동작 상이 |
| 완화(식 41) | 정식 기술 | 미구현(`use_relaxation=False`) | Python 기준 OFF |

> 시사점: "논문과 비트 일치"가 아니라 **"Python 코드와 수치 일치"**가 1차 목표. 논문식은 물리 해석·엔탈피법 재정식화(§5)·향후 Fortran 정합의 근거로 활용.

### 8.2 검증 절차
1. **모듈 단위 대조**: 각 물리 함수(`calc_rnet`, `calc_profile`, `melting`, …)를 `exact` 모드로 RoadSurf-Python 대응 함수와 동일 입력에서 비교(float64).
2. **전체 순전파 대조(레짐별 허용오차, 검토 C5)**: `example_data/test_input.csv`로 dROAD(exact) vs Python 롤아웃 비교. **무강수·무상변화 구간** 노면온도 상대오차 < 1e-9; **상변화 활성 구간**은 비트 일치 대신 **분기 결정 시퀀스 일치 + 저장항 < 1e-3 mm, Tsurf < 1e-2 °C**. 커플링 포함/제외 모두.
3. **Fortran 삼자 대조(검토 C6)**: Python이 LLM 변환본이므로, 핵심 물리(식 27–32, 46–50)는 Fortran 원본을 컴파일해 **Fortran vs Python vs dROAD** 삼자 대조. 불일치 시 Fortran을 진실로. (§8.1 지층 깊이식 불일치가 대표 사례.)
4. **미분 연산 정합성 (검토 M-Ver1)**: (a) VJP gradient vs 중심차분(<1e-4, `check_grads`); (b) **TLM test** $\lVert\mathcal{M}(x+\alpha\delta x)-\mathcal{M}(x)-\alpha\mathcal{M}'\delta x\rVert/\alpha\to0$(로그-로그 기울기 1); (c) **JVP↔VJP dot-product test** $\langle Jv,u\rangle=\langle v,J^\top u\rangle$(<1e-8); (d) **HVP 대칭성** $u^\top Hv=v^\top Hu$ 및 $Hv$ vs grad 유한차분.
5. **smooth 수렴 & 보존 (P0 기준)**: τ→0 수렴은 **end-to-end 롤아웃이 아니라 매끄러운 원시연산 단위**로만 요구; 롤아웃은 deviation budget(G3)으로 평가. 보존은 단일 "총수분"이 아니라 **mass ledger**(primary / auxiliary ice2 / external source·sink)로 분리 진단(§5, P0 §3).
6. **수치 강건성(검토 C2/N11)**: 보정 파라미터 탐색 범위 전반에서 forward가 CFL 안정(NaN/발산 없음). **조건수 모니터링(검토 M-C1)**: 상변화 임계 근처에서 스텝별 TLM 노름·(GN)Hessian 스펙트럼을 추적해 gradient 스파이크·부정치 조기 감지(τ로 안정화).
7. **식별성 게이트(검토 C3)**: 다중 사례에서 쌍둥이 실험으로 대상 모수 복원 가능성을 **보정 착수 전** 확인. 복원 불가 시 대상 축소.
8. **회귀 테스트**: pytest 스위트로 고정(원본엔 테스트 없음 → dROAD가 기준선 확보).
9. **[고위험] 서브에이전트 교차검토**: 상변화·잠열·보존 처리의 물리 타당성을 독립 재검토.

### 8.3 검증 픽스처 4계층 (AD 감사 채택)
위 절차를 **4계층 픽스처/게이트**로 조직한다(`AD가능성_감사_Python.md`):
1. **RoadSurf-Python pinned fixture**: 현 Python 출력을 동결한 회귀 기준. 단 **Coupling·Relaxation 결함부는 기준에서 격리**(비트 재현 대상 아님).
2. **no-coupling core fixture**: 커플링 OFF 순전파를 dROAD `exact`와 대조(§8.2-2). **미분 코어의 1차 진실.**
3. **paper-physics case**: 핵심 물리(식 27–32·46–50·경계층 12–19)를 이상화 조건에서 검증. **Fortran 삼자 대조는 `compatibility_target=fortran_compat`일 때만 hard gate**이며, `python_compat`(P0 기본)에서는 **optional audit sidecar**다. python_compat의 hard gate는 RoadSurf-Python no-coupling fixture + paper-physics 해석/부호 진단이다(P0 §11, M0c_optional).
4. **forecast skill gate**: 독립 검증구간에서 **운영 커플링 대비 노면온도 RMSE 동등 이상**(DoD-3).

여기에 미분 정합성(§8.2-4)·보존/CFL(§8.2-5,6)·식별성(§8.2-7) 게이트를 결합해 마일스톤 통과 조건으로 삼는다.

### 8.4 compatibility_target·gate 등급·baseline (외부 검토 반영)
`exact`가 Python-exact인지 Fortran-exact인지 흔들리던 문제를 **명시 분리**한다(상세·YAML은 `구현계획_P0_derisked.md` §1,6,11):
- **2축 enum 통일 (no-go #9)**: "타깃"과 "모드"를 섞지 않는다. **runtime `compatibility_target = {python_compat, fortran_compat}`**, `model_mode = {roadsurf_exact, smooth_compat, enhanced_enthalpy_v1}`, **`paper_physics`는 runtime target이 아니라 `validation_suite`**(smooth_deviation·forecast_skill과 함께). 허용: `python_compat×roadsurf_exact`, `fortran_compat×roadsurf_exact`, `python_compat×smooth_compat`, `python_compat×enhanced_enthalpy_v1`, 그리고 **모든 model_mode를 `paper_physics` validation suite에서 평가**. 금지: **`paper_physics`를 runtime target으로 사용**, `smooth_compat×enhanced_enthalpy_v1`(surrogate와 physics-changing 혼합). `fortran_compat` 불일치 시 Fortran 진실. 테스트 `test_python_compat_*`/`test_fortran_compat_*`/`test_paper_physics_*`/`test_smooth_deviation_*`/`test_forecast_skill_*`(모호한 `exact_*` 금지). YAML·조합표는 P0 §1.
- **gate 등급**: G0 스칼라/단위(abs<1e-10) → G1 dry 롤아웃(동일 알고리즘 RMSE<1e-6, custom_vjp/implicit **알고리즘 변경 후** <1e-5) → G2 exact 저장이벤트(시퀀스 일치+저장 MAE<1e-3mm, Tsurf RMSE<1e-2°C) → G3 smooth deviation budget. *허용오차는 백엔드가 아니라 **알고리즘 변경 여부** 기준*(jit/scan은 결과 거의 불변, remat은 비트 동일).
- **forecast baseline 계층**: B0 persistence(필수) / B1 no-coupling RoadSurf-Python(필수) / B2 Fortran operational coupling(있으면) / B3 Python coupling smoke(**DA reference 아님**). "운영 커플링 대비"는 **B2가 있을 때만**, 없으면 **B1** 대비로 판정.

---

## 9. 단계별 구현 로드맵

**M0 — 스캐폴딩 & 데이터 & 참조 실행** (1주)
레포 구조·의존성(JAX, Equinox, optax, optimistix, lineax) 고정. RoadSurf-Python 기준선 출력 생성. **다중 사례·(가능하면)다중 지점 데이터 확보**(검토 C3: 식별성의 전제) + Fortran 컴파일로 삼자 대조 환경(검토 C6). **격차분석(§2.6) 확정** — 이식할 소량 기능(B4/B5) 목록화, SVF 지형보정 채택 여부 결정.

**M1 — 함수형 상태 + 열수지 코어(numpy)** (1주)
`State`/`Params` pytree, `calc_profile`+`calc_hcap_hcond`+`calc_rnet`+boundary layer 포팅, **커플링 OFF**로 Python·Fortran 대조(§8-1,2,3).

**M2 — 저장항·상변화 포팅(exact 모드)** (1주)
`Storage`/`Cond` 전체 포팅, 하드 임계값 그대로. 레짐별 허용오차로 순전파 일치(§8-2).

**M3 — JAX eager/scan/jit 단계 활성** (1주)
numpy→jnp, 분기→`where`/`cond`(§4.4 safe/guarded 정책), 루프→`scan`, float64. **BLC는 BLC-v0(fixed-unroll)만 사용**; BLC-v1(exact early-stop)은 parity 확인용으로만. **BLC-v2 custom_vjp는 여기 넣지 않는다 — M4/M5 이후 별도 promotion gate**(P0 §5). CFL 점검·필요 시 상단층 서브스텝/부분암시(N11).

**M4 — 미분가능성 + smooth 모드 (연구 난제, 버퍼 포함)** (2~3주)
(a)지층 엔탈피법·(b)저장항 질량보존 게이트(§5), τ annealing, 소실 gradient 대응. VJP 유한차분·**JVP↔VJP dot-product test**·보존 잔차 검증(§8-4,5). *가장 불확실 → 디버깅 버퍼 명시.*

**M5 — 자료동화 데모(VJP/adjoint)** (1주)
초기상태 최적화로 노면온도 정합(방안 A). optax/optimistix 파이프라인, 운영 커플링 대비 평가(DoD-3).

**M6 — 물리모수 보정 + 식별성** (1.5주)
민감도 스크리닝으로 대상 3–5개 선정, 재매개화·정칙화, **쌍둥이 실험 식별성 게이트**(검토 C3, §8-7).

**M7 — 동시 추정·2차·UQ (단일 마일스톤 아님 → M7a–f로 분해)**
검토 반영: GN·EnVar·UQ·FSOI는 각각 논문급이라 한 덩어리로 착수 금지. forward exact·smooth gradient가 잠긴 뒤 순차 게이트로만 진행(세부는 P0 §9).
- **M7a — JVP/VJP operator contract**: dot-product test·블록 gradient norm만. optimizer 비의존.
- **M7b — Gauss–Newton vector product**: dry/smooth 코어, CG 5–10회 smoke.
- **M7c — B^{1/2} 제어변수 변환**: 정적 대각 먼저(앙상블 B는 이후).
- **M7d — EnVar** *(not Alpha default)*: 다지점 데이터 확보 후.
- **M7e — UQ (Lanczos/Hutchinson)** *(not Alpha default)*: 사후 최적점 확보 후.
- **M7f — FSOI** *(not Alpha default)*: 관측연산자·스코어카드 안정 후.
> 다중 윈도우 결합 목적함수(§7.2)·모드 선택(§7.4)·하이브리드-4DEnVar(§7.7)·UQ(§7.6)·FSOI(§7.9)의 이론은 청사진 §7에 있으나, **착수 단위는 위 M7a–f**다. **diff-op 게이트 단계화**: Phase 0의 미분 게이트는 dot-product test까지만; GNVP/HVP/vmap-over-site는 M7b 이후.

**M8 — (선택) 순환 추정 + 하이브리드 ML / 커플링 재현** (여유)
순환(cycling)·warm-start(§7.8), NN=약제약 모델오차 공동학습(§7.5/7.8), 또는 원본 커플링 재현(방안 B).

> 예상 총 **10~12주**(1인 기준, M8 제외) — 단 **M0~M4만으로도 10~12주가 될 수 있다**(외부 검토). 일정이 아니라 **gate 통과로 진도 관리**. 위 M0~M8은 청사진 수준이며, **실행용 세분 마일스톤(M0a–c, M1a–d, M2a–d, M3a–d, M7a–f)과 P0 통과 조건은 `구현계획_P0_derisked.md` §9**에 있다. 백엔드는 M1 pure NumPy parity → JAX 단계 활성(§4.1).

---

## 10. 리스크 & 완화

| 리스크 | 영향 | 완화 |
|---|---|---|
| **CFL×보정 발산 (C2/N11)** | 최적화 중 forward NaN → gradient NaN | 상단층 서브스텝/부분암시(lineax), 재매개화 CFL 제약, 발산 페널티 |
| **식별성·데이터 부족 (C3)** | 단일 사례 보정 시 과적합·비유일(equifinality) | **다중 사례 데이터(M0)**, 민감도로 대상 3–5개 축소, 정칙화, **쌍둥이 실험 게이트(M6)** |
| **성공 기준 부재 (C4)** | 마일스톤 완료해도 무입증 위험 | DoD 정량화(§1.3): 커플링 대비·모수 복원율 |
| 상변화 (a)/(b) 혼동 (C1) | 엔탈피법이 저장항 질량이전 못 고침 | §5 분리 처리 + 질량보존 게이트 |
| Python이 LLM 변환본 (C6) | 검증 기준 자체 버그 | Fortran 삼자 대조(§8-3) |
| smooth 근사가 물리 왜곡 | 보정 결과 편향 | exact로 최종 평가, τ 민감도, 보존 잔차 점검 |
| 긴 시계열 역전파 메모리 | OOM | `jax.checkpoint`(remat)+scan, 구간 분할 |
| boundary layer 안정도 함수 미확인 | 포팅 정확도 | M1에서 `BoundaryLayer.py`/`ModRadiation.py` 상세 확인 |

## 11. 열린 질문 (다음 결정 필요)

1. **커플링 재현 필요성**: 방안 A(대체)만으로 충분한가, 아니면 원본 커플링 동작도 재현(B/C)해야 하는가? → **PyTorch 전환 트리거 ①**(가변길이 되감기 충실 재현이 필수면 재평가, `프레임워크_재검토` §4).
2. **잠열 처리**: (a)지층 엔탈피법 채택 여부, (b)저장항 질량전달 게이트 폭 τ.
3. **관측 종류**: 노면온도 외에 저장항/노면상태 관측도 손실에 넣을지(다변량 $R$).
4. **보정 대상 모수 우선순위**: 민감도 스크리닝으로 3~5개 선정(추천 가능).
5. **성능 목표**: CPU float64+`vmap`(권장) vs GPU. float64+GPU 처리량 한계 고려(§4.1).
6. **하이브리드 ML 범위**: NN 삽입 지점(잔차/폐합식/파라미터장)과 도입 시점. 대형 NN 결합이 주목적이 되면 → **PyTorch 전환 트리거 ③**.
7. **동시 vs 교대 최적화**: 상태·모수 완전 동시(§7.4) vs 이중수준(bilevel) — 조건수 보고 결정.

---

## 12. 제안 디렉토리 구조
```
dROAD/
├── dROAD_설계계획서.md        # 본 문서
├── pyproject.toml             # JAX, optax, jaxopt, flax 등
├── reference/
│   ├── RoadSurf-Python/       # 포팅 1차 기준
│   └── RoadSurf/              # Fortran 원본(삼자 대조·격차 이식 소스, §2.6)
├── droad/
│   ├── state.py               # State/Params/Control pytree (Equinox)
│   ├── physics/
│   │   ├── balance.py         # 열수지·프로파일 (+부분암시 스텝, N11)
│   │   ├── radiation.py       # 순복사·ModRadiation
│   │   ├── boundary.py        # 경계층·잠열 (BLC 고정점)
│   │   └── storage.py         # 저장항·상변화 (exact/smooth, 질량보존)
│   ├── smoothing.py           # smooth_gate, soft_min/max, enthalpy c_eff, τ
│   ├── graph/                 # 미분 규칙 노드 (검토 M1·C2)
│   │   ├── fixed_point.py     # custom_vjp IFT (BLC/커플링)
│   │   └── implicit_step.py   # lineax 선형해 + adjoint
│   ├── model.py               # step(), rollout() = lax.scan (+remat)
│   ├── operators.py           # forward M, obs H, TLM(jvp)·adjoint(vjp) 래퍼
│   ├── init.py                # 초기 상태·파라미터
│   ├── coupling.py            # (선택) 방안 B/C
│   ├── hybrid/                # NN 성분 (§7.5)
│   │   └── nn_closures.py     # 잔차·폐합식·파라미터장 + 보존 가드레일
│   └── estimation/            # 동시 추정 시스템 (§7)
│       ├── cost.py            # 다중윈도우 J(z)
│       ├── grad.py            # VJP gradient
│       ├── second_order.py    # GN/Newton-CG, HVP=jvp(grad) (§7.4)
│       ├── uq.py              # Laplace/EnVar (§7.6)
│       └── optimize.py        # optax/optimistix 드라이버
├── tests/                     # 원본·Fortran 삼자 대조, grad·dot-product test, 보존
└── examples/                  # 동화·보정·동시추정·하이브리드 데모
```
> **MVP는 이 트리의 부분집합만 (외부 검토 반영).** 1차 구현은 `state/params/forcing/radiation/ground/boundary_layer_unrolled(BLC-v0)/storage_exact_minimal/model/loss/sensitivities` + dry 테스트(`test_rnet`,`test_dry_profile_one_step`,`test_dry_rollout`,`test_jvp_vjp_dot`)로 축소. `graph/`(fixed_point·implicit_step), `estimation/`(second_order·uq), EnVar, **`hybrid/`는 MVP 이후**. 특히 **`hybrid/`는 기본 import에서도 제외**(`enabled:false`, M8 게이트 통과 전 비활성) — 식별성 보호. 세부 MVP·세분 마일스톤은 `구현계획_P0_derisked.md` §7,9.

---

## 부록 B — 지배방정식 ↔ 미분가능화 대응표

Karsisto (2024)의 식 번호 기준. "미분가능화"는 dROAD `smooth` 모드에서의 처리. C=연속·매끄러움(그대로), W=`where`/게이트, IT=반복솔버(unroll/IFT), EN=엔탈피법 권장, S=이미 시그모이드.

### B.1 표면 에너지수지 · 복사
| 식 | 내용 | 처리 |
|---|---|---|
| 1 | $G=R_n-\text{LE}+H+T_r$ (표면 열수지) | C |
| 2,9 | $R_n=\text{SW}_{down}(1-\alpha_s)+\text{LW}_{down}-\text{LW}_{up}$, $\text{LW}_{up}=\varepsilon\sigma_{SB}T_{sK}^4$ | C ($T^4$ 매끄러움) |
| 3–8 | SVF·지평선 확산/반사 보정, 태양위치 | C, 단 "직달일사=0" 지평선 판정은 W |
| 10 | 알베도(눈/얼음/총빙 1.5mm 문턱) | W (게이트) |

### B.2 경계층 (현열·잠열) — 반복 솔버 N9
| 식 | 내용 | 처리 |
|---|---|---|
| 11,12 | $H=\text{BLC}(T_s-T_a)$, $\text{BLC}=c_a\rho_a k u^*/(\ln(\cdot)+\Psi_h)$ | IT |
| 13,14 | $c_a=1005+(T_{aK}-250)^2/3364$, $\rho_a=p/(R_d T_{aK})$ | C |
| 15,16 | $u^*$, 안정도 $\zeta=-kz_T gH/(c_a\rho_a T_{aK}u^{*3})$ | C (IT 내부) |
| 17 | 안정: $\Psi_h=\Psi_m=4.7\zeta$ | W (분기) |
| 18,19 | 불안정: $\Psi_h=-2\ln(\frac{1+\sqrt{1-16\zeta}}{2})$, $\Psi_m=0.6\Psi_h$ | W (분기) |
| 20,21 | $\text{LE}=\frac{\rho_m c_a}{\gamma}\frac{e_s-e_a}{r_o}$, $\gamma$ | C |
| 22,23 | 포화수증기압 물/얼음(0°C 분기) | W (or EN 정합) |
| 24,26 | $e_a=\frac{\text{RH}}{100}e_s$, 공기역학저항 $r_o$(상한 30) | C, 상한은 clip |

### B.3 지중 열전도 — 미분가능 핵심(implicit solver 없음)
| 식 | 내용 | 처리 |
|---|---|---|
| 27 | $\rho_g c_g\,\partial_t T=\partial_z(K\partial_z T)$ (Patankar 1980) | — |
| 28 | 명시적 전진차분 이산화(층별 $T_i^{j+1}$) | C, `lax.scan` (CFL 주의) |
| 29 | $\rho_g c_g=(1-\phi)\rho_s c_s+\phi\rho_w c_w$ | C |
| 30,31 | 물 밀도·비열 온도의존(코드와 계수 일치) | C |
| 32–36 | 열전도 $\lambda=A+B\theta-(A-D)e^{-(C\theta)^E}$ | C |
| 37,38 | 층 깊이(§8.1 **논문↔코드 불일치**) | 코드 기준 |
| 39 | 바닥층 계절 온도 $T_{m+1}=T_c+A_y\sin(\Omega J+\cdots)$ | C (초기화) |

### B.4 저장항 · 상변화 (하드 임계값 밀집)
| 식 | 내용 | 처리 |
|---|---|---|
| 42 | $P_{rain}=1/(1+e^{P_{exp}})$, $P_{exp}=22-2.7T_a-0.2\text{RH}$ | **S** (그대로) |
| 43 | 마모 $\text{Wear}_x=\text{Wf}_x\text{St}_x\Delta t/3600$ (Wf: snow .45/ice .319/ice2 2.552/dep 1.16/water .145) | C + 하한 clip |
| 44,45 | 증발 $\text{EV}_w=\text{LE}\,\Delta t/E_{m^3}$, $E_{m^3}=L_{wat}\rho_w$ | C |
| 46 | 융해량 $\text{Me}_{snow}=1000\,Q_{melt}\Delta t/(L_{melt}\rho_w)$ | C (게이트 W) |
| 47 | 가용열 $Q_{melt}=\rho_g c_g\frac{z_2-z_1}{2\Delta t}(T_1-T_{melt})$ | C |
| 48,49 | 전량융해열 $Q_{all}$, 잔여열로 $T_1$ 갱신 | W, **EN 권장** |
| 동결 | 즉시·잠열 미반영 → 노면온도 진동(N10) | **EN 권장** |
| 50 | 물-눈비 $\text{WS}_{rat}$(>0.6 물화, >0.1&동결 얼음화) | W (게이트) |
| — | 저장항 min/max 클램프(snow 100·ice 50·dep 2·water 1 mm) | clip |

### B.5 관측 정합 (커플링·완화)
| 식 | 내용 | 처리 |
|---|---|---|
| 40 | 커플링 후 $C_f(t)=1+C_R e^{-t/t_c}$ ($t_c$=4h) | C. 반복탐색부는 §6(방안 A 우선) |
| 41 | 완화 $X(t)=X_F-(X_{FO}-X_O)e^{-t/t_c}$ | C (Python 미구현) |

---

### 부록 A. 참고문헌
- Karsisto, V. E. (2024). *RoadSurf 1.1: open-source road weather model library.* Geosci. Model Dev., 17, 4837–4853. https://doi.org/10.5194/gmd-17-4837-2024
- 원본: https://github.com/fmidev/RoadSurf · Python: https://github.com/fmidev/RoadSurf-Python
- 복사 보정(커플링) 근거: Karsisto et al. (2016), *Meteorol. Appl.*, 23, 503–513; Crevier & Delage (2001), *J. Appl. Meteorol.*, 40, 2026–2037.
