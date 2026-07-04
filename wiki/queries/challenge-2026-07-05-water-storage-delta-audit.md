---
title: "Challenge — water_storage는 delta 기반이어도 감사상 안전하다"
type: query
date: 2026-07-05
---
# Challenge: water_storage delta 회계의 감사 안전성

## Claim under challenge
"water_storage는 delta 기반이어도 감사상 안전하다" (근거: [[decisions/water-storage-stays-delta-based]]).

## Strongest contradiction
water_storage의 residual은 **구조적으로 항상 0**이다. 코드에서 external_source=max(delta,0),
external_sink=max(-delta,0), delta=w-before이므로 expected=before+delta=actual → residual≡0.
따라서 water_storage 자체 코드 버그(예: wear 이중 차감)로 물이 생성/소멸돼도 그 변화가 external로
재라벨링되어 residual은 0을 유지한다. merge_ledgers의 자식 residual fail-fast도 0은 항상 통과하므로
road_cond 집계 감사에서도 불가시. 즉 water_storage의 ledger는 자기 코드에 대한 **감사력이 0**이다.
이는 [[experiences/residual-tautology-discovery]]의 Lesson("감사 잔차는 독립적으로 측정된 흐름에서
유도되어야 한다")과 정면으로 충돌한다 — water의 external 흐름은 독립 측정이 아니라 delta에서 유도된다.

## Weaker signals
- [[concepts/branch-local-external-accounting]]는 delta-후처리를 명시적으로 거부; water_storage는 그 예외.
- "hard-projection hit는 diagnostics로 반환"하지만 diagnostics는 mass 정확성을 보증하지 않음.

## Residual support that survives
- 좁은 의미의 "안전"은 성립: water_storage에는 internal transfer가 없어 delta 회계가 *내부 전이를 숨기지*
  않는다(tautology가 아닌 이유는 이 좁은 정의 한정).
- water 질량 정확성은 **parity 테스트**로 독립 보증: test_python_compat_water_storage가 w==reference를
  1e-12로 검사(ledger가 아니라 parity가 안전망).

## Verdict
**Claim weakened / should be revised.** 시스템 전체는 안전하나(parity가 잡음), "감사상 안전"은 과장.
정확한 진술: "water_storage의 ledger residual은 감사력이 없다(구조적 0). 물 정확성은 parity 테스트가
보증하며, delta 회계는 '내부 전이를 숨기지 않는다'는 좁은 의미에서만 안전하다. 이 안전성은 water가
all-external로 유지된다는 가정에 조건부다 — 향후 water에 내부 전이가 생기면 조용히 무력화된다."

## Operative question
이 진술로 [[decisions/water-storage-stays-delta-based]]를 수정할까, 아니면 원 주장을 유지할까?
