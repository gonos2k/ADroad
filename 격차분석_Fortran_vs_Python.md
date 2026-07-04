# 격차 분석 — RoadSurf(Fortran) vs RoadSurf-Python: 추가 변환 필요 분량

> 목적: Python 변환본이 "빠른 연구 실험용·부분 기능·LLM 보조"임을 감안해, **미분가능 버전(dROAD)까지 가는 데 추가로 변환해야 할 분량**을 정량화한다.
> 방법: 두 repo clone 후 파일·루틴 단위 대조.
> 검토일: 2026-07-02

## 요약 (숫자로)
| 구분 | Fortran | Python |
|---|---|---|
| src 총 줄수 | 4,034 (.f90 + .inc 타입정의) | 2,499 (.py) |
| 물리/로직 실질 줄수 | ~3,250 (인터페이스·C바인딩·타입정의 제외) | ~2,100 (I/O 헬퍼 제외) |
| 구현 루틴 수(대략) | ~60 (파일별 end 기준) | ~55 (def, 클래스 __init__ 포함) |

**핵심 결론**: Python은 **미분화에 필요한 물리 코어를 사실상 이미 커버**한다(열수지·지중전도·저장항·상변화·경계층·복사·태양위치·커플링 반복). 원본 Fortran 대비 **정말로 빠진 라이브러리 기능은 소량(~300–450줄)** 이고, 그나마 대부분 **dROAD의 목적(방안 A: 커플링 OFF, 변분 DA)에서는 저우선/불필요**다.
→ 즉 이 프로젝트의 실제 작업량은 **"빠진 Fortran 기능 번역"이 아니라 "기존 Python 물리(~2,100줄)의 JAX 전환 + 미분가능화 + 검증"**(계획서 M1–M4)에 있다.

---

## A. Python이 이미 커버 — 추가 변환 불필요 (미분화 대상 코어)
| Fortran 파일 | 루틴 수 | Python 대응 | 상태 |
|---|---|---|---|
| BalanceModel.f90 | 11 | BalanceModel.py (8 def) | 코어 일치(열수지·프로파일·열용량/전도·현열). 누락은 §B 참조 |
| BoundaryLayer.f90 | 3 | BoundaryLayer.py (3) | ✅ 현열·잠열·공기역학저항 |
| Cond.f90 | 4 | Cond.py (4) | ✅ 강수형·마모·노면상태·알베도 |
| Storage.f90 | 7 | Storage.py (7) | ✅ 강수→저장·융해·물/눈/얼음/서리·융해열 |
| ModRadiation.f90 | 1 | ModRadiation.py (1) | ✅ SVF/지평선 복사보정 적용부 (규모 유사) |
| SunPosition.f90 | 3 | SunPosition.py (3) | ✅ 태양고도/방위·율리우스일 |
| Coupling.f90 | 10 | Coupling.py (7) | 반복탐색(Coupling_control) 등 코어 포함. 누락 §B |
| (타입정의 .inc ×15) | — | 각 클래스 __init__ | ✅ Params/State 상당분 |

## B. Python이 누락/축소 — 추가 변환 후보 (분량·우선순위)
| # | 빠진 것 | Fortran 위치 | 대략 분량 | dROAD 우선순위 |
|---|---|---|---|---|
| B1 | **Relaxation**(관측→예보 전환 완화) | Relaxation.f90 (48줄) | ~50줄 | **낮음** — main.py에서 주석처리, use_relaxation=False. 필요 시 이식 |
| B2 | **getTempAtDepth**(사용자 지정 높이로 출력온도 보간) | BalanceModel.f90 | ~30줄 | **낮음** — 출력 편의. dROAD는 첫 두층 평균 사용 |
| B3 | **couplingCofWithInputRadCof / initCoupling / InputRadiationCoefficient** (입력 복사계수 경로) | Coupling.f90 + .inc | ~80줄 | **불필요(방안 A)** — 커플링 OFF면 제외. 방안 B/C 시에만 |
| B4 | **입력 파라미터/설정 QC 세부**(setInputParam, initSettings 일부, CheckValues 확장) | Initialization/InputOutput.f90 | ~100–150줄 | **중간** — 결측·이상치 처리 일부. DA 마스킹으로 대체 가능하나 확인 필요 |
| B5 | **condInit / ground_prop_init 세부**(지반 물성 초기화 분기) | Initialization.f90 | ~50줄 | **중간** — Python 클래스 init이 상당 커버, 다층 물성 분기 대조 필요 |

**B 소계: 약 300–450줄.** 이 중 dROAD 방안 A 기준 **실제 필요분은 B4·B5 일부(~150–200줄)** 정도.

## C. 두 버전 모두 제외 — 라이브러리 범위 밖 (사용자/응용에서 구현)
README·논문(3절)이 명시: **마찰(friction)·노면상태 분류(wet/icy/snowy 등)는 라이브러리 미포함.** 아래는 Fortran "예제(C++)"에만 있고 라이브러리·Python 모두에 없다.
| 기능 | 위치 | dROAD 필요성 |
|---|---|---|
| main program·입출력 소스(JSON/QueryData) | examples/example1 (2,619줄), example2 (6,362줄) | Python main.py + readInputcsv/writecsv가 최소 대체. dROAD는 자체 데이터 로더로 대체 |
| **SkyView 계산**(SVF·지평선각 산출 자체) | example1/2의 SkyView.cpp | SVF/지평선을 **입력으로 쓰는** 경로(ModRadiation)는 있음. 지형에서 **계산**은 없음 → SVF 보정 쓸 때만 별도 포팅(수백 줄) |
| MeteorologyTools(습도·복사 파생량) | 예제 C++ | 일부는 Python InputArrays(CalcRH 등)에 존재 |
| 친화(friction)·노면상태 산출 | 라이브러리 밖 | dROAD 범위 밖(향후 후처리) |

## D. 결론 — 작업량 재정의
1. **"추가 변환 필요 분량"의 실체는 작다.** 순수 미포팅 Fortran 라이브러리 기능은 ~300–450줄, dROAD 방안 A 기준 실필요 ~150–200줄(주로 입력 QC·지반 초기화 대조). C 바인딩(86줄)·인터페이스(272줄)·C++ 예제는 불필요.
2. **진짜 작업은 번역이 아니라 전환·미분화·검증**: 기존 Python 물리 ~2,100줄을 JAX로(계획서 M1–M4). 여기에 §8의 **Fortran 삼자 대조**가 결합돼, 대조 과정에서 B4/B5 격차와 §8.1 불일치(지층 깊이식 등)를 흡수한다.
3. **선택 기능 결정 필요**: (a) SVF/지평선 지형 보정을 쓸 것인가 → 쓰면 SkyView 계산을 예제 C++에서 포팅(수백 줄). (b) Relaxation·getTempAtDepth는 후순위 옵션.
4. **삼자 대조 대상 우선순위**(정확도 신뢰 확보): 식 27–32(지중전도·열물성), 46–50(상변화), 경계층 반복(12–19), 커플링(방안 B 시). Python이 LLM 변환본이라 이 구간은 Fortran을 진실로 삼는다.

## 부록 — 파일별 루틴 수 대조(구현부 end 기준)
```
파일                 Fortran루틴  Python def   비고
BalanceModel            11           8         getTempAtDepth 등 누락
BoundaryLayer            3           3         ✅
Cond                     4           4         ✅
Storage                  7           7         ✅
Coupling                10           7         입력복사계수 경로 누락
Initialization          14        1+클래스init  분산 구현, 세부 대조 필요
InputOutput              7           4         입력 QC 일부 누락
ModRadiation             1           1         ✅
Relaxation               1        2(스텁)       미구현(주석)
SunPosition              3           3         ✅
ConnectFortran2Carrays   3           0         C바인딩 — 불필요
RoadSurf(interface)     14           0         인터페이스 — 불필요
```
