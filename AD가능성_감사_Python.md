# RoadSurf-Python 자동미분(AD) 가능성 감사

> 목적: Python 코드 전 파일을 재분석해 **직접 autograd를 막는 코드**를 파일·행 단위로 식별하고, JAX 전환 시 처방과 난이도를 정리한다.
> 방법: 전 파일 패턴 스캔 + 그래프-내부 파일 정독(BalanceModel/BoundaryLayer/Storage/Cond/ModRadiation/InputOutput/Coupling).
> 검토일: 2026-07-02

## 결론 요약
- **직접 autograd 불가는 사실이다.** NumPy/math 스칼라, 가변 객체 필드, in-place 갱신, 하드 분기, **반복 솔버(BLC 40회, 데이터 의존 break)**, 정수·불리언 이산 상태, datetime/round 등이 전반에 있다.
- **그러나 대부분은 "기계적 전환" 또는 "사전계산"으로 해소**된다. 진짜 연구적 난제는 소수(상변화 하드 분기, VeryCold 히스테리시스, BLC 고정점 미분)뿐.
- **핵심 통찰**: AD-비친화 코드의 상당수가 **제어변수(상태·물리모수)가 아니라 시간·기하·입력에만 의존** → 그 부분은 **AD 그래프 밖에서 사전계산**하면 되고 미분 자체가 불필요.
- **Coupling.py는 DA 레퍼런스로 부적합**(아래 §6, 코드 증거로 확인). 커플링은 **복사(SW/LW)계수 또는 forcing-bias를 제어변수로** 대체한다(계획서 방안 A와 일치).

---

## AD-blocker 분류 (범주별)

### 범주 1 — In-place 변경 / 가변 객체 필드 (전역, 기계적 해소)
- 클래스 속성 제자리 갱신·배열 원소 대입·`+=`·`.copy()`가 거의 모든 파일에.
  - `BalanceModel.calcProfile`: `ground.TmpNw[j] = ...`(13개 idx 대입), `ground.Tmp = ground.TmpNw.copy()`.
  - `Storage.*`: `surf.SrfWatmms += ...` 등 **29개 복합대입**, 저수지 간 이동.
  - `Coupling`: `.copy()`, 속성 대입 다수.
  - `InputOutput.SetCurrentValues/SaveOutput`: `ground.Tmp[0]=`, `modelOutput.*[i]=`.
- **처방**: `State`/`Params`를 불변 pytree(Equinox/NamedTuple)로, 모든 갱신을 함수형 반환·`.at[].set()`으로. **난이도 하(분량 많음).**

### 범주 2 — 데이터 의존 분기 (연속 상태 기반)
- **연속값 임계 분기 → `where`/`lax.cond`; 하드 임계는 §5 smoothing 필요.**
  - `Storage.py`(48 if): 융해/동결/증발/마모의 상태 임계(`TsurfAve>T4Melt`, `SrfWat>MaxPor`, `WSrat>0.6/0.1`, min/max 클램프). **AD 신호 소실의 주 원인.**
  - `BalanceModel.CalcHCapHCond`: `TmpNw>=0` 물/얼음 분기(밀도·비열).
  - `BoundaryLayer.CalcLE`: 포화수증기압 **물/얼음 분기**(`Tsurf<0`, `TAmb<0`), 증발 분기(`Tsurf>=0`).
  - `Cond.CalcAlbedo`: 눈/얼음 임계 알베도 전환.
- **처방**: `where` + §5 매끄러운 게이트/엔탈피. **난이도 중~상(연구적).**

### 범주 3 — 반복 솔버 (데이터 의존 종료)
- **`BoundaryLayer.CalcBLCondAndLE`: `for j in range(1,41)` 안정도 반복 + `if abs(ΔBLC)<ConvLim and j>=5: break`** — 즉 **최대 40회 고정점 반복 + 조기종료**. (논문 식 12–19 확인.) 안정/불안정 분기(`Stab>0`)와 `Stab>1` 클램프 포함.
- **처방**: `custom_vjp` 기반 **암시적 미분 고정점**(수렴점 IFT; 순전파 조기종료 허용) — 계획서 §6·§7.3와 일치. **난이도 중.**

### 범주 4 — 비미분 이산 연산 (대부분 사전계산 가능 → 실질 blocker 아님)
- `ModRadiation`: `azim_idx = round(sun_azim)` 로 `local_horizons[azim_idx]` **정수 인덱스 gather**, `shadow_fac ∈ {0,1}` 이진. → **태양기하·지형(시간·입력)에만 의존 → 사전계산**. sky_view·albedo_surroundings 혼합만 미분(선형, 매끄러움).
- `SunPosition`(23 if, 14 int/모듈로): 천문 계산 → **시간·위경도 의존 → 전량 사전계산**(제어변수 아님).
- `Cond.CalcPrecType`: 입력 위상 정수코드 분기(`PrecPhase in [...]`)·`PrecType` 정수 → **입력 의존, 사전계산**. 해석식 경로의 `p_rain=1/(1+e^{p_exp})`는 **이미 시그모이드**(미분가능), 단 `PLimSnow/PLimRain` 하드 컷은 §5 램프 필요.
- **처방**: 시간/기하/입력 의존 항은 **AD 그래프 밖 배열로 precompute**. **난이도 하(개념 정리 필요).**

### 범주 5 — 경로 의존(히스테리시스)·이산 상태
- `Cond.RoadCond`: **`VeryCold` 불리언 히스테리시스**(`TLimColdH`/`TLimColdL` 이중 문턱, 이전 상태 의존) — 단순 임계보다 어려움(경로 의존).
- `atm.SnowType`(정수), `surf.WearSurf`, `CP.WetSnowFrozen` 등 진화 이산 상태.
- **처방**: 연속 완화(예: VeryCold→sigmoid 상태변수로 승격) 또는 이력을 연속 상태로 carry. **난이도 중~상.**

### 범주 6 — 부작용·버그성 코드
- `print(...)`(BoundaryLayer, InputOutput, Coupling), `settings.simulation_failed=True`(조기탈출 플래그).
- **`Cond.CalcAlbedo`가 `WearSurf=False`거나 어떤 분기도 안 맞으면 `Albedo=None` 반환** → 하류 곱셈서 폭발. **정상값 기본치 필요.**
- `CalcLE`: `if LE_Flux>0 and SrfWatmms<=0: LE_Flux=0`(물 없으면 증발 차단) — **하드 게이트(불연속)**.
- `WearFactors`: `if SrfSnowmms<0.2: SnowTran*=3` — **3배 계단 점프(불연속)**.
- **처방**: 부작용 제거, None→기본치, 하드 게이트는 §5 게이트로. **난이도 하~중.**

### 범주 7 — 결측 센티넬
- `-9999`, `TSurfObs>-100`(InputOutput/GroundVariables/main). → 손실·초기화 마스크로 분리(초기상태를 제어변수로 두면 초기화 로직과 충돌 주의).

### 범주 8 — 시간 루프·되감기
- `main.py`: `while i<SimLen-1 and not simulation_failed` + 커플링 되감기(`i` 리셋). → `lax.scan`으로 전환(방안 A로 되감기 제거). `.hour`(낮/밤)는 사전계산.

---

## 사용자 코드수준 결론 — 코드 증거로 확인
| 결론 | 확인 | 코드 증거 |
|---|---|---|
| BalanceModel/BoundaryLayer/Storage/Cond 구조상 포팅 가능 | ✅ | 루틴 대응 일치(격차분석 §A) |
| NumPy/math·가변필드·in-place·하드분기로 **직접 autograd 불가** | ✅ | 범주 1·2·6 |
| **Coupling.py는 DA 레퍼런스 부적합** | ✅ **확정** | 아래 상세 |
| Relaxation.py 미완성 | ✅ | `main.py:29` "not fully implemented", 호출부 주석; `Relaxation.py`는 스텁 |
| 커플링 → SW/LW 복사 제어 또는 forcing-bias 제어로 대체 | ✅ 타당 | 계획서 §6 방안 A(복사계수=제어변수)와 일치. forcing-bias 제어를 추가 옵션으로 채택 권장 |

### Coupling.py 결함 (행 인용)
1. **저장/복원 속성명 불일치**: 저장은 소문자 `coupling.srfWatmmsSave`(63행)인데 복원은 대문자 `coupling.SrfWatmmsSave`(82행) 참조 → **AttributeError/무효**.
2. **primary ice 미저장**: 64–65행이 `srfIce2mmsSave`를 **중복** 대입, `SrfIcemms`(1차 얼음) 저장 누락.
3. **복원이 잘못된 소문자 필드에 기록**: `surf.srfWatmms=`(82행) 등 — 실제 속성은 `surf.SrfWatmms`. **실제 저장항이 복원되지 않고 유령 속성 생성**.
4. **복사 저장 루프 무의미**: 73–77행이 스칼라 하나를 매 반복 덮어씀(스텝별 배열 아님).
5. **복사(radiation) 복원 주석처리**: 92–95행 비활성 → 되감기 시 복사 강제력 미복원.
6. **되감기 loop**: `datai=coupling.saveDatai; return datai`(80·97행)로 시간 인덱스 되돌림.
→ 종합: 커플링을 **비트 재현 대상으로 삼지 말 것**. dROAD는 **복사계수/forcing-bias 제어변수**로 대체(방안 A). 원본 재현이 꼭 필요하면 Fortran `Coupling.f90`(565줄)을 진실로.

---

## 검증 전략 (사용자 제안 채택 — 4계층 픽스처/게이트)
1. **RoadSurf-Python pinned fixture**: 현 Python 출력을 **동결(pinned)** 회귀 기준. 단, Coupling·Relaxation 결함부는 기준에서 제외/격리.
2. **no-coupling core fixture**: **커플링 OFF** 경로의 순전파를 dROAD `exact`와 대조(레짐별 허용오차, 계획서 §8-2). 이것이 미분 코어의 1차 진실.
3. **paper-physics case**: 핵심 물리(식 27–32·46–50·경계층 12–19)를 **Fortran 원본과 삼자 대조**(Python이 LLM 변환본이라, §8-3). 논문식과의 정합도 확인.
4. **forecast skill gate**: 독립 검증구간에서 **운영 커플링 대비 노면온도 RMSE 동등 이상**(DoD-3). 실사용 가치 게이트.
- 여기에 미분 정합성(TLM/dot-product/HVP, §8-4)과 보존·CFL 게이트(§8-5,6)를 결합.

---

## 처방·난이도 종합
| 범주 | 분량 | 난이도 | 계획서 반영 |
|---|---|---|---|
| 1 in-place→함수형 | 많음 | 하 | §4.1 |
| 2 연속분기→where/§5 | 중 | 중~상 | §5, N2/N10 |
| 3 BLC 고정점 | 국소 | 중 | §6, §7.3 |
| 4 이산연산 사전계산 | 중 | 하 | §4.4(정적/동적 분리) |
| 5 히스테리시스/이산상태 | 국소 | 중~상 | §4.1(동적 이산) — *VeryCold 명시 추가 권장* |
| 6 부작용·None·하드게이트 | 소 | 하~중 | §5 |
| 7 센티넬 마스크 | 소 | 하 | §7.2 |
| 8 시간루프/되감기 | 코어 | 중 | §4.1, §6 |

**추가 반영 권장(계획서 v0.5)**: (a) §4.4에 **VeryCold 히스테리시스**를 동적 이산 예시로 명시, (b) §6에 **forcing-bias 제어**를 방안 A의 대체 제어변수로 추가, (c) §8에 **4계층 픽스처/게이트**를 검증 절차로 편입, (d) §5 목록에 **CalcLE 무수분 증발 게이트·WearFactors 3× 점프·CalcAlbedo None** 명시.
