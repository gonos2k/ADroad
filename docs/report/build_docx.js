const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun,
  Header, Footer, AlignmentType, LevelFormat, TabStopType, TabStopPosition,
  TableOfContents, HeadingLevel, BorderStyle, WidthType, ShadingType, PageNumber, PageBreak,
} = require("docx");

const CW = 9360;            // content width (US Letter, 1" margins)
const MONO = "Consolas";
const BLUE = "1F4E79", LBLUE = "D5E8F0", GREY = "F2F2F2", LINE = "BBBBBB";

// ---------- helpers ----------
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const H3 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(t)] });
const P = (t, opts = {}) => new Paragraph({ spacing: { after: 120, line: 276 }, children:
  Array.isArray(t) ? t : [new TextRun(t)], ...opts });
const runs = (...r) => r;
const B = (t) => new TextRun({ text: t, bold: true });
const T = (t) => new TextRun(t);
const CODEF = (t) => new TextRun({ text: t, font: MONO, size: 18 });

function BUL(items) {
  return items.map((it) => new Paragraph({
    numbering: { reference: "bul", level: 0 }, spacing: { after: 60, line: 264 },
    children: Array.isArray(it) ? it : [new TextRun(it)],
  }));
}
function NUM(items) {
  return items.map((it) => new Paragraph({
    numbering: { reference: "num", level: 0 }, spacing: { after: 60, line: 264 },
    children: Array.isArray(it) ? it : [new TextRun(it)],
  }));
}
const border = { style: BorderStyle.SINGLE, size: 1, color: LINE };
const borders = { top: border, bottom: border, left: border, right: border };

function cell(text, w, { head = false, bold = false, mono = false, align } = {}) {
  const rns = (Array.isArray(text) ? text : [text]).map((s) =>
    typeof s === "string"
      ? new TextRun({ text: s, bold: head || bold, color: head ? "FFFFFF" : "000000",
          font: mono ? MONO : "Arial", size: mono ? 18 : 20 })
      : s);
  return new TableCell({
    borders, width: { size: w, type: WidthType.DXA },
    shading: { fill: head ? BLUE : "FFFFFF", type: ShadingType.CLEAR },
    margins: { top: 60, bottom: 60, left: 110, right: 110 },
    children: [new Paragraph({ alignment: align, children: rns })],
  });
}
function TABLE(headers, rows, widths, opts = {}) {
  const mono = opts.mono || false;
  const head = new TableRow({ tableHeader: true, children: headers.map((h, i) => cell(h, widths[i], { head: true })) });
  const body = rows.map((r, ri) => new TableRow({
    children: r.map((c, i) => {
      const zebra = ri % 2 === 1;
      const cc = cell(c, widths[i], { mono: mono && i > 0 });
      if (zebra) cc.options.shading = { fill: GREY, type: ShadingType.CLEAR };
      return cc;
    }),
  }));
  return new Table({ width: { size: CW, type: WidthType.DXA }, columnWidths: widths, rows: [head, ...body] });
}
function CODE(lines) {
  // single-cell shaded box (has content -> allowed)
  const paras = lines.map((l) => new Paragraph({ spacing: { after: 0, line: 240 },
    children: [new TextRun({ text: l || " ", font: MONO, size: 18 })] }));
  return new Table({
    width: { size: CW, type: WidthType.DXA }, columnWidths: [CW],
    rows: [new TableRow({ children: [new TableCell({
      borders, width: { size: CW, type: WidthType.DXA },
      shading: { fill: GREY, type: ShadingType.CLEAR },
      margins: { top: 100, bottom: 100, left: 140, right: 140 }, children: paras,
    })] })],
  });
}
const SP = () => new Paragraph({ spacing: { after: 80 }, children: [new TextRun("")] });
const tH1 = (t) => new Paragraph({ spacing: { before: 60, after: 20 },
  children: [new TextRun({ text: t, bold: true, size: 22, color: BLUE })] });
const tH2 = (t) => new Paragraph({ spacing: { after: 8 }, indent: { left: 480 },
  children: [new TextRun({ text: t, size: 20, color: "333333" })] });

// ---- math run helpers (real sub/superscripts) ----
const MF = "Cambria Math";
const mi = (t) => new TextRun({ text: t, italics: true, font: MF, size: 22 });   // variable
const mo = (t) => new TextRun({ text: t, font: MF, size: 22 });                  // operator/number
const sub = (t) => new TextRun({ text: t, subScript: true, font: MF, size: 22 });
const sup = (t) => new TextRun({ text: t, superScript: true, font: MF, size: 22 });
function EQ(children) {
  return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 90, after: 120 },
    children });
}
function FIG(file, w, h, caption) {
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 140, after: 40 },
      children: [new ImageRun({ type: "png", data: fs.readFileSync(file),
        transformation: { width: w, height: h },
        altText: { title: caption, description: caption, name: caption } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 180 },
      children: [new TextRun({ text: caption, italics: true, size: 18, color: "666666" })] }),
  ];
}

// ---------- content ----------
const body = [];
const push = (...x) => x.forEach((e) => body.push(e));

// Title block
push(
  new Paragraph({ spacing: { before: 1400, after: 120 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "dROAD", bold: true, size: 72, color: BLUE })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 },
    children: [new TextRun({ text: "미분가능 RoadSurf 노면기상 모델 · 자료동화 시스템", bold: true, size: 32 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 },
    children: [new TextRun({ text: "기술보고서 및 사용자 매뉴얼", size: 26 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 80, after: 60 },
    children: [new TextRun({ text: "작성일: 2026-07-06 (v1.0)", bold: true, size: 22, color: "333333" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 40 },
    children: [new TextRun({ text: "대상 리포지토리 상태: 커밋 a9f635f (origin/main)", size: 20, color: "666666" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "코어 187 + JAX 34 테스트 통과 · raw-primitive 감사 clean", size: 20, color: "666666" })] }),
  new Paragraph({ children: [new PageBreak()] }),
);

// TOC (static outline — always visible; complete section list)
push(new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("목차")] }));
push(
  tH1("1. 개요"), tH2("1.1 배경과 목적"), tH2("1.2 설계 원칙"),
  tH1("2. 시스템 아키텍처"), tH2("2.1 모듈 구성"), tH2("2.2 도구 및 리포트"),
  tH1("3. 핵심 컴포넌트 및 수학적·공학적 정식화"),
  tH2("3.1 질량 원장 · 3.2 Deviation budget · 3.3 Skill gate · 3.4 Forecast DA(파라미터 vs 상태)"),
  tH2("3.5 열전달 지배방정식 · 3.6 질량보존 불변식 · 3.7 변분 DA · 3.8 2차/UQ · 3.9 평가 메트릭 · 3.10 공학적 특성"),
  tH1("4. 실험 결과"),
  tH2("4.1 질량 감사 기준선 · 4.2 예보 skill 기준선 · 4.3 파라미터 민감도 DA"),
  tH2("4.4 상태추정 forecast DA · 4.5 다중 window 재현 검증 · 4.6 win/lose regime 분석"),
  tH1("5. 사용자 매뉴얼"),
  tH2("5.1 환경 준비 · 5.2 테스트 실행 · 5.3 리포트 도구 실행법 · 5.4 리포트 해석 가이드 · 5.5 forecast DA 파라미터"),
  tH1("6. 품질 보증 및 개발 프로세스"),
  tH1("7. 결론 및 향후 과제"), tH2("7.1 다음 단계 (우선순위)"),
  tH1("부록 A. 기호 정의 (Nomenclature)"),
  tH1("부록 B. 재현 절차 및 파일 구조"), tH2("B.1 재현 절차 · B.2 파일 구조 · B.3 커밋 이정표"),
);
push(SP(), new Paragraph({ spacing: { before: 40, after: 20 },
  children: [new TextRun({ text: "그림 목차", bold: true, size: 22, color: BLUE })] }));
push(
  tH2("그림 1. 시스템 레이어 파이프라인"),
  tH2("그림 2. 상태추정 forecast DA 사이클"),
  tH2("그림 3. 단일 window 예보 RMSE"),
  tH2("그림 4. 다중 window 재현(Δrmse)"),
);
push(new Paragraph({ children: [new PageBreak()] }));

// 1. 개요
push(H1("1. 개요"));
push(H2("1.1 배경과 목적"));
push(P([B("dROAD"), T("는 핀란드 기상청(FMI)의 노면기상 예측 모델 "), B("RoadSurf"),
  T("를 파이썬으로 재구현하되, 핵심 열·질량 과정을 "), B("미분가능(differentiable)"),
  T("하게 만들어 그래디언트 기반 자료동화(DA)와 물리 파라미터 보정을 가능하게 하는 연구용 시스템이다. "),
  T("원본 Fortran 모델은 예측만 수행하지만, dROAD는 관측을 이용해 초기상태·파라미터를 역으로 추정할 수 있다.")]));
push(P([T("본 시스템의 두 번째 축은 "), B("질량 감사(mass-audit) 레이어"),
  T("다. 자료동화가 물리적으로 타당한지 검증하기 위해, 매 스텝의 질량 수지를 불변식으로 기록하고 "),
  T("잔차(residual)와 물리 진단(diagnostics)을 분리 추적한다. 즉 “수치적으로 맞는가”와 "),
  T("“물리적으로 신뢰할 수 있는가”를 별도 게이트로 판정한다.")]));
push(H2("1.2 설계 원칙"));
push(...BUL([
  [B("정직성 우선: "), T("긍정 결과를 과장하지 않는다. 단일 성공은 다중 검증 전까지 report-only로 유지한다.")],
  [B("잔차 ↔ 진단 분리: "), T("질량 잔차(코드 누출)는 hard 게이트, 물리 진단은 카운트(실패 아님)로 취급한다.")],
  [B("방어적 공개 API: "), T("모든 공개 함수는 입력 타입·유한성·의미 범위를 검증하고 명시적 오류를 던진다.")],
  [B("간단·명료·직관: "), T("각 모듈은 단일 책임을 갖고, 감사 가능한 계약(contract)으로 결합된다.")],
]));

// 2. 아키텍처
push(new Paragraph({ children: [new PageBreak()] }));
push(H1("2. 시스템 아키텍처"));
push(P([T("dROAD는 아래 4개 레이어가 계약으로 연결된 파이프라인이다. 하위 레이어의 출력이 상위 레이어의 "),
  T("검증된 입력이 되며, 각 경계에서 방어적 검증이 수행된다.")]));
push(CODE([
  "ledger (질량 원장·불변 감사 기록)",
  "   └─> deviation budget (잔차 게이트 + 진단 부담 + storage-jump provenance)",
  "          └─> skill gate (RMSE hard 게이트 + 진단 부담 guardrail)",
  "                 └─> forecast DA report (state/parameter DA, report-only + machine-readable sidecar)",
]));
push(...FIG("fig_arch.png", 600, 300, "그림 1. 시스템 레이어 파이프라인 — 하위 감사 출력이 상위 게이트의 검증된 입력이 된다."));
push(H2("2.1 모듈 구성 (droad/ 총 17개 코어 모듈, 약 2,836 LOC)"));
push(TABLE(
  ["모듈", "역할"],
  [
    ["config.py", "설정 스키마 검증(validator)"],
    ["branches.py", "분기 래퍼 + raw-primitive 감사 경계"],
    ["ledger.py", "StorageLedger(불변 질량 원장), merge_ledgers, 진단 registry, 검증 헬퍼"],
    ["thermal / boundary / radiation.py", "M1 건조 열전달 커널·경계층(BLC)·복사"],
    ["storage.py", "강수 상변화·수분/눈/얼음/퇴적 저장·융해열(NumPy 정밀 경로)"],
    ["roadcond.py", "노면상태·마모·알베도"],
    ["driver.py", "full_rollout — 건조+저장 결합 전체 적분(감사 옵션)"],
    ["model.py", "step_full — 1스텝 결합 + 원장/진단 노출"],
    ["jax_model.py", "미분가능 건조 열 rollout(JAX, lax.scan)"],
    ["jax_storage.py / smoothing.py", "미분가능 저장 원시연산·soft_clip 평활화"],
    ["assimilate.py", "변분 DA·2차/UQ(fit, gauss_newton, newton, laplace, hutchinson)"],
    ["dual.py", "순환 이중추정(상태 fast + 파라미터 slow)"],
    ["deviation.py", "deviation budget 집계·잔차 게이트·storage-jump provenance"],
    ["skill_gate.py", "forecast 메트릭·RMSE hard 게이트·promotion 게이트"],
  ],
  [3200, 6160]));
push(H2("2.2 도구 및 리포트 (tools/ 9개)"));
push(TABLE(
  ["도구", "산출 리포트", "역할"],
  [
    ["report_deviation_budget.py", "deviation_budget_baseline", "기준 rollout의 질량 감사·진단 부담"],
    ["report_skill_gate.py", "skill_gate_baseline", "default vs constant_initial 예보 skill 게이트"],
    ["report_multiwindow_skill.py", "multiwindow_skill", "6개 기간 skill 안정성 + promotion 게이트"],
    ["report_da_evaluation.py", "da_evaluation", "파라미터 민감도 DA(Emiss) 평가"],
    ["report_forecast_da.py", "forecast_da", "상태추정 forecast DA(단일 window)"],
    ["report_forecast_da_multi.py", "forecast_da_multi", "다중 window 재현 검증 + promotion"],
    ["analyze_forecast_da_regimes.py", "forecast_da_regimes", "win/lose regime 분석(feature family)"],
    ["check_raw_primitives.py", "(감사)", "NumPy 코어의 원시연산 금지 감사"],
  ],
  [3100, 2700, 3560]));

// 3. 핵심 컴포넌트
push(new Paragraph({ children: [new PageBreak()] }));
push(H1("3. 핵심 컴포넌트 및 수학적·공학적 정식화"));

push(H2("3.1 질량 원장 (ledger.py)"));
push(P([B("StorageLedger"), T("는 매 스텝 질량 수지를 담는 frozen dataclass다. 외부 유입/유출(external "),
  T("source/sink), 내부 전이(internal_transfer), 보조 갱신(auxiliary_update), 이벤트 플래그(event_flags)를 "),
  T("불변(MappingProxyType) 매핑으로 보관하고, "), B("primary_mass_residual"),
  T("(= 코드 누출 탐지용 잔차)을 계산한다. "), B("merge_ledgers"),
  T("는 자식 원장의 잔차 fail-fast + 연속성(질량 보존)을 강제한다.")]));
push(P([B("검증 스택: "), CODEF("as_finite_float"), T("(str/bool/np.bool_/NaN/Inf 거부), "),
  CODEF("normalize_diagnostics"), T("(진단 코드 정규화), event_flag는 순수 bool만 허용. "),
  T("모든 잘못된 입력은 "), CODEF("LedgerError"), T("로 일관 처리된다.")]));
push(P([B("메모리 최적화: "), CODEF("full_rollout"), T("는 기본적으로 스텝당 원장 1개+진단만 저장하고, "),
  T("세부 원장(prec/cond)은 "), CODEF("return_ledger_detail=True"),
  T(" 옵트인일 때만 저장한다. 감사 경로가 읽지 않는 세부 원장(감사 메모리의 약 2/3, n=4000에서 ~21MB)을 "),
  T("공통 경로에서 제거했다.")]));

push(H2("3.2 Deviation budget (deviation.py)"));
push(P([CODEF("deviation_budget(out, steps=None)"),
  T("는 rollout 감사 트레일을 품질 지표로 집계한다. 잔차(P0 게이트), 진단 코드별 카운트, "),
  B("max_storage_jump"), T("(+ 원본 rollout step provenance)를 산출한다. "),
  CODEF("steps="), T(" 인자로 holdout 구간만 슬라이스해 skill window와 물리 부담 window를 정렬할 수 있다.")]));
push(P([B("입력 방어: "), T("step index 정수성·strictly-increasing·컨테이너 타입(딕셔너리 우회 차단)까지 검증하여 "),
  T("공개 API가 ledger 하드닝 수준과 동일하다.")]));

push(H2("3.3 Skill gate (skill_gate.py)"));
push(P([T("게이트가 실제로 강제하는 hard 실패는 "), B("Tsurf RMSE 하나"),
  T("다. MAE·freeze-thaw 정확도·cold RMSE는 report-only(해석용)이다. 여기에 회계 잔차(~0)와 물리 부담 "),
  T("(diagnostic_steps_rate·over_melt_count·overflow_count)가 baseline 대비 악화되지 않을 것을 guardrail로 둔다. "),
  T("이 세 부담은 "), CODEF("diagnostics_delta().physics_worse"), T("가 보는 항목과 정확히 일치한다.")]));
push(P([B("promotion_gate: "), T("충분한 독립 case + 모든 window에서 baseline 우위 + 잔차 clean + 물리 비악화가 "),
  T("모두 성립할 때만 PROMOTE. 단일 fixture는 구조적으로 REPORT_ONLY(설계 §11의 report-only 정책을 실행 게이트화).")]));

push(H2("3.4 Forecast DA — 파라미터 민감도 vs 상태추정"));
push(P([T("dROAD는 두 종류의 DA를 "), B("명확히 분리"), T("해 평가한다.")]));
push(...BUL([
  [B("파라미터 민감도 DA "), CODEF("(report_da_evaluation.py)"),
    T(": Emiss(+Tair bias)를 dry 모델에서 보정한 뒤 full 모델을 t=0부터 free-run. cal/eval window를 "),
    T("분리(train 누수 제거)하고 DA를 default와 직접 게이트.")],
  [B("상태추정 forecast DA "), CODEF("(report_forecast_da.py)"),
    T(": dry 모델 내부에서 초기 near-surface 온도 상태(layers 1:5 offset dx)를 변분동화한 뒤 "),
    B("자유 예보"), T("(관측 미삽입)를 수행. 파라미터는 default 고정 → 순수 상태추정. 동화·예보가 같은 모델을 "),
    T("써서 cross-model 아티팩트가 없다.")],
]));
push(...FIG("fig_cycle.png", 600, 267,
  "그림 2. 상태추정 forecast DA 사이클 — 배경 spin → 동화창에서 dx 최적화 → lead 자유예보(관측 미삽입). DA(녹색)가 no-DA(적색)보다 관측을 잘 추종."));

// ---- 3.5 열전달 지배방정식 ----
push(new Paragraph({ children: [new PageBreak()] }));
push(H2("3.5 열전달 지배 방정식과 이산화"));
push(P([T("노면 하부를 "), mi("N"), T("개 층으로 이산화한 1차원 비정상 열확산을 푼다. 층 온도 벡터를 "),
  mi("T"), T(" = ("), mi("T"), sub("0"), T(", …, "), mi("T"), sub("N+1"), T(")라 하면, 노면온도는 상위 두 층의 평균이다.")]));
push(EQ([mi("T"), sub("surf"), mo(" = ( "), mi("T"), sub("1"), mo(" + "), mi("T"), sub("2"), mo(" ) / 2")]));
push(P([T("표면 순복사속(net radiation)은 단파 흡수·장파 흡수·흑체 방출의 합이다("), mi("α"),
  T(" 알베도, "), mi("ε"), T(" 방사율, "), mi("σ"), T(" Stefan–Boltzmann 상수):")]));
push(EQ([mi("R"), sub("net"), mo(" = (1 − "), mi("α"), mo(")"), mi("SW"), mo(" + "), mi("ε"), mi("·LW"),
  mo(" − "), mi("ε σ"), mo(" ("), mi("T"), sub("surf"), mo(" + 273.15)"), sup("4")]));
push(P([T("지중 열유속은 표면 경계(전도·잠열·교통마찰·경계층 현열)와 층간 전도로 구성된다:")]));
push(EQ([mi("G"), sub("0"), mo(" = "), mi("R"), sub("net"), mo(" − "), mi("LE"), mo(" + "), mi("Q"), sub("tr"),
  mo(" + "), mi("h"), sub("BLC"), mo(" ("), mi("T"), sub("0"), mo(" − "), mi("T"), sub("1"), mo("),   "),
  mi("G"), sub("j"), mo(" = −("), mi("CC"), mo("/"), mi("DyK"), sub("j"), mo(")("), mi("T"), sub("j+1"),
  mo(" − "), mi("T"), sub("j"), mo(")")]));
push(P([T("시간 전진은 명시적(explicit) 유한체적 갱신이며, "), mi("C"), sub("vsh"),
  T("는 온도 의존 체적 열용량이다:")]));
push(EQ([mi("T"), sub("j"), sup("n+1"), mo(" = "), mi("T"), sub("j"), sup("n"), mo(" + "), mi("Δt · capDZ"),
  sub("j"), mo(" ("), mi("G"), sub("j"), mo(" − "), mi("G"), sub("j−1"), mo("),   capDZ"), sub("j"),
  mo(" = −1 / ("), mi("DyC"), sub("j"), mi(" C"), sub("vsh"), mo(")")]));
push(EQ([mi("C"), sub("vsh"), mo(" = "), mi("C"), sub("dry"), mo(" + "), mi("W ρ"), sub("w"), mi(" c"), sub("w"),
  mo("("), mi("T"), mo("),   "), mi("ρ"), sub("w"), mo(", "), mi("c"), sub("w"), mo(" : 온도 다항식")]));
push(P([T("경계층 전도도 "), mi("h"), sub("BLC"), T("는 Monin–Obukhov류 안정도 보정을 40회 고정점 반복으로 푼다"),
  T("(마찰속도 "), mi("U"), sub("*"), T(", 안정도 "), mi("S"), T(", 안정도 함수 "), mi("ψ"), sub("M"), T("·"),
  mi("ψ"), sub("H"), T("). 잠열속 "), mi("LE"), T("는 포화수증기압 차 기반 Penman류로 계산한다.")]));
push(P([B("공학적 안정화: "), T("Δt = 30 s·다층 구조로 explicit scheme의 확산 안정 조건을 만족시키고, "),
  T("Kelvin 하한("), mi("T"), sub("aK"), T(" ≥ 1), 분모 보호("), CODEF("_safe_den"), T("), 지수 클립("),
  CODEF("_safe_exp"), T(", ±60)으로 미선택 분기에서도 그래디언트가 유한하도록 보장한다.")]));

// ---- 3.6 질량 보존 불변식 ----
push(H2("3.6 질량 보존 불변식 (감사 이론)"));
push(P([T("각 저장소(눈·물·얼음·퇴적)의 스텝별 질량 수지를 다음 항등식으로 기록한다. "),
  mi("S"), sub("ext"), T(", "), mi("K"), sub("ext"),
  T("는 외부 유입·유출이며, 내부 전이는 질량 이동일 뿐이라 잔차에 포함되지 않는다.")]));
push(EQ([mi("M"), sub("after"), mo(" = "), mi("M"), sub("before"), mo(" + "), mi("S"), sub("ext"), mo(" − "),
  mi("K"), sub("ext"), mo(",     "), mi("r"), mo(" = "), mi("M"), sub("after"), mo(" − ("), mi("M"), sub("before"),
  mo(" + "), mi("S"), sub("ext"), mo(" − "), mi("K"), sub("ext"), mo(")")]));
push(P([T("잔차 "), mi("r"), T(" ≈ 0 은 물리가 아니라 "), B("코드 정합성"),
  T("을 뜻한다(누출 탐지). 내부 전이는 실측 누적으로 별도 기록되어 잔차에 섞이지 않으므로, 서로 상쇄되는 "),
  T("두 오류가 0 잔차로 위장되지 않는다. 병합(merge) 시에는 "), B("자식 잔차를 합산하지 않고"),
  T(" 각 자식이 개별적으로 "), mi("|r|"), T(" ≤ atol 이며 연속성("), mi("M"), sub("after"), sup("(i)"),
  T(" = "), mi("M"), sub("before"), sup("(i+1)"), T(")을 만족해야 한다. P0 게이트: max"), sub("t"),
  T(" |"), mi("r"), sub("t"), T("| ≤ 1e-9.")]));

// ---- 3.7 변분 DA ----
push(H2("3.7 변분 자료동화 정식화"));
push(P([T("일반 4D-Var 비용함수는 배경항(사전, "), mi("B"), T(" 오차공분산)과 관측항("), mi("R"),
  T(" 관측공분산, "), mi("H"), T(" 관측연산자, "), mi("M"), sub("t"), T(" 모델 전파)의 합이다:")]));
push(EQ([mi("J"), mo("("), mi("x"), sub("0"), mo(") = ½("), mi("x"), sub("0"), mo("−"), mi("x"), sub("b"),
  mo(")"), sup("T"), mi("B"), sup("−1"), mo("("), mi("x"), sub("0"), mo("−"), mi("x"), sub("b"), mo(") + ½ Σ"),
  sub("t"), mo(" ("), mi("H"), mi("M"), sub("t"), mo("("), mi("x"), sub("0"), mo(") − "), mi("y"), sub("t"),
  mo(")"), sup("T"), mi("R"), sup("−1"), mo("(·)")]));
push(P([T("dROAD의 forecast DA 구현형은 제어변수를 near-surface 상태보정 "), mi("dx"),
  T("(layers 1:5 offset)로 두고, 배경항을 "), mi("bg"), sub("w"), mo("·"), mi("‖dx‖"), sup("2"),
  T("로, 관측항을 동화창의 가중 MSE로 둔다("), mi("H"), T("("), mi("T"), T(") = ("), mi("T"), sub("1"),
  T("+"), mi("T"), sub("2"), T(")/2, ⊕ = layers 1:5 가산):")]));
push(EQ([mi("J"), mo("("), mi("dx"), mo(") = (1/Σ"), mi("w"), sub("t"), mo(") Σ"), sub("t"), mi(" w"), sub("t"),
  mo(" ( "), mi("H M"), sub("t"), mo("("), mi("x"), sub("b"), mo(" ⊕ "), mi("dx"), mo(") − "), mi("y"), sub("t"),
  mo(" )"), sup("2"), mo(" + "), mi("bg"), sub("w"), mo(" ‖"), mi("dx"), mo("‖"), sup("2")]));
push(P([B("그래디언트: "), mo("∇"), sub("dx"), mi("J"), T(" 는 역방향 자동미분(VJP·adjoint) "),
  B("단일 패스"), T("로 계산되며, 비용이 제어차원과 무관하다(O(1) in dim). 최적화는 Adam(기본, best-iterate 추적) "),
  T("또는 Gauss–Newton/incremental 4D-Var를 사용한다:")]));
push(EQ([mo("("), mi("J"), sub("r"), sup("T"), mi("J"), sub("r"), mo(" + "), mi("λI"), mo(") "), mi("δz"),
  mo(" = − "), mi("J"), sub("r"), sup("T"), mi("r"), mo(",   ("), mi("J"), sub("r"), mi("v"), mo(" = JVP,  "),
  mi("J"), sub("r"), sup("T"), mi("v"), mo(" = VJP,  matrix-free CG)")]));
push(P([T("분석(analysis) 상태는 동화창 끝에서 carry된 뒤, lead 구간을 "), B("관측 미삽입 자유예보"),
  T("로 적분해 no-DA(background) 예보와 비교한다.")]));

// ---- 3.8 2차/UQ ----
push(H2("3.8 2차 정보와 불확실성 정량화 (UQ)"));
push(P([T("Hessian–vector 곱은 forward-over-reverse로 matrix-free 계산된다:")]));
push(EQ([mi("H v"), mo(" = ∇( ∇"), mi("J"), mo(" · "), mi("v"), mo(" ),    Newton: "), mi("x"), mo(" ← "),
  mi("x"), mo(" + ("), mi("H"), mo(" + "), mi("λI"), mo(")"), sup("−1"), mo("(−"), mi("g"), mo(")")]));
push(P([T("Laplace 근사 사후공분산은 정칙화된 역헤시안이며, 음의 곡률을 clip해 SPD를 보장한다:")]));
push(EQ([mi("Σ"), mo(" ≈ ("), mi("H"), mo(" + "), mi("s I"), mo(")"), sup("−1"), mo(",   "), mi("s"),
  mo(" = "), mi("λ"), mo(" + max(0, −"), mi("λ"), sub("min"), mo(")")]));
push(P([T("고차원 제어에서는 Hutchinson 추정으로 헤시안 대각을 형성 없이 근사한다("), mi("v"),
  T(" ~ Rademacher):")]));
push(EQ([mo("diag("), mi("H"), mo(") ≈ 𝔼[ "), mi("v"), mo(" ⊙ "), mi("H v"), mo(" ]")]));

// ---- 3.9 평가 메트릭 ----
push(H2("3.9 평가 메트릭의 수학적 정의"));
push(EQ([mi("RMSE"), mo(" = √( (1/"), mi("n"), mo(") Σ ("), mi("p"), sub("i"), mo(" − "), mi("o"), sub("i"),
  mo(")"), sup("2"), mo(" ),    "), mi("MAE"), mo(" = (1/"), mi("n"), mo(") Σ |"), mi("p"), sub("i"), mo(" − "),
  mi("o"), sub("i"), mo("|")]));
push(EQ([mi("A"), sub("ft"), mo(" = (1/"), mi("n"), mo(") Σ 𝟙[ ("), mi("p"), sub("i"), mo(" ≥ "), mi("τ"),
  mo(") = ("), mi("o"), sub("i"), mo(" ≥ "), mi("τ"), mo(") ],    degradation = "), mi("RMSE"), sub("holdout"),
  mo(" / "), mi("RMSE"), sub("train")]));
push(P([T("skill 게이트의 hard 조건은 다음 부등식들이다("), mi("f"), T(" 허용 비율, atol 잔차 허용, "),
  T("Δ 부담 slack):")]));
push(EQ([mi("c"), sub("rmse"), mo(" ≤ "), mi("b"), sub("rmse"), mo("(1+"), mi("f"), mo("),   "), mi("r"),
  mo(" ≤ atol,   "), mi("c"), sub("rate/om/of"), mo(" ≤ "), mi("b"), sub("rate/om/of"), mo(" + Δ")]));
push(P([T("regime 분석의 분리도(separation)는 그룹 평균의 상대 gap이며, 부호가 반대이면 1을 넘을 수 있고 "),
  T("통계적 유의성이 아니다:")]));
push(EQ([mi("sep"), mo(" = |"), mi("μ"), sub("win"), mo(" − "), mi("μ"), sub("lose"), mo("| / max(|"),
  mi("μ"), sub("win"), mo("|, |"), mi("μ"), sub("lose"), mo("|, "), mi("ε"), mo(")")]));

// ---- 3.10 공학적 설계 특성 ----
push(H2("3.10 공학적 설계 특성"));
push(...BUL([
  [B("수치 정밀도·parity: "), T("전 경로 float64(JAX x64 enable). NumPy 정밀 경로와 JAX 미분 경로가 "),
    T("동일 결과(parity)를 내도록 회귀로 고정 — 미분가능화가 물리를 바꾸지 않음을 보장.")],
  [B("미분가능성 보존: "), CODEF("soft_clip"), T("(평활 clip), "), CODEF("_safe_den"), T("/"), CODEF("_safe_exp"),
    T("/Kelvin floor가 미선택 분기의 NaN 그래디언트를 차단 — 무가드 원시연산은 감사로 금지.")],
  [B("계산 복잡도: "), T("adjoint 그래디언트는 rollout 1회 비용이며 제어차원과 무관(O(1) in dim); "),
    T("BLC 고정점 O(n"), T("_iter"), T("=40); Gauss–Newton O(outer×cg_maxiter).")],
  [B("메모리: "), T("감사 rollout은 스텝당 원장 1개(+진단), 세부 원장은 opt-in; trajectory는 6×"), mi("n"),
    T(" float64로 선형.")],
]));

// 4. 실험 결과
push(new Paragraph({ children: [new PageBreak()] }));
push(H1("4. 실험 결과"));

push(H2("4.1 질량 감사 기준선 (deviation budget)"));
push(P([T("기준 rollout에서 최대 primary 잔차는 사실상 0(≈4.4e-16, PASS)으로 질량 보존이 코드 수준에서 지켜진다. "),
  T("물리 진단은 negative-pre-clamp 56회(물·퇴적·눈에 분산)가 관측되며 이는 실패가 아니라 참조 물리 특성으로 "),
  T("카운트된다. 최대 storage jump는 Ice, step 4312, +0.1006 이다.")]));

push(H2("4.2 예보 skill 기준선"));
push(TABLE(
  ["모델", "RMSE(°C)", "MAE", "freeze-thaw", "게이트"],
  [
    ["constant_initial (기준)", "5.0432", "4.2445", "0.2766", "baseline"],
    ["default", "0.2049", "0.1680", "0.9915", "PASS"],
  ], [3360, 1600, 1400, 1600, 1400]));
push(P([T("다중창(6기간) skill에서도 default가 "), B("6/6 window 전부"),
  T("에서 constant_initial을 이긴다(RMSE 평균 0.187, 최악 0.344). 다만 단일 fixture이므로 promotion은 "),
  B("REPORT_ONLY"), T("로 유지된다.")]));

push(H2("4.3 파라미터 민감도 DA (홀드아웃)"));
push(TABLE(
  ["모델", "RMSE(°C)", "vs constant", "vs default"],
  [
    ["constant_initial", "2.5839", "baseline", "baseline"],
    ["default (Emiss 0.950)", "0.1494", "PASS", "baseline"],
    ["DA (Emiss 0.995)", "0.1555", "PASS", "FAIL"],
  ], [3560, 1800, 2000, 2000]));
push(P([B("결과: "), T("cal/eval window를 분리하니 DA-보정 Emiss는 default보다 홀드아웃 RMSE가 오히려 나쁘다"),
  T("(Δrmse +0.0061, physics_worse=False). 이는 단일창 equifinality의 한계로, 이전의 window 누수가 만든 "),
  T("낙관적 오해를 걷어낸 정직한 결과다. (참조: 1-step persistence RMSE 0.0078 — 30s 해상도에서 자명해 "),
  T("게이트 baseline으로 부적합.)")]));

push(H2("4.4 상태추정 forecast DA (핵심 결과)"));
push(P([T("k0=2000, 동화창 120스텝, 예보 lead 480스텝의 단일 window:")]));
push(TABLE(
  ["모델", "예보 RMSE(°C)", "gate vs no-DA"],
  [
    ["constant_initial", "0.9180", "baseline"],
    ["no_DA (background)", "0.2210", "baseline"],
    ["DA (state)", "0.2082", "PASS"],
  ], [3760, 3000, 2600]));
push(...FIG("fig_single.png", 430, 231,
  "그림 3. 단일 window 예보 RMSE — DA(state)가 no-DA와 constant_initial을 모두 개선."));
push(P([B("발견: "), T("초기상태를 동화한 자유 예보가 no-DA를 개선(Δrmse −0.0128, degradation 0.954로 overfit "),
  T("아님). 즉 “DA가 default를 못 이긴다”던 이전 결론은 DA 자체의 한계가 아니라 "), B("파라미터 equifinality"),
  T("의 한계였음을 분리해냈다. 상태의 memory가 lead에서 작동한다.")]));

push(H2("4.5 다중 window 재현 검증 (보수적 판정)"));
push(TABLE(
  ["k0", "DA RMSE", "BG RMSE", "Δrmse", "DA 우위"],
  [
    ["1500", "0.6431", "0.5248", "+0.1183", "False"],
    ["2100", "0.2313", "0.2113", "+0.0200", "False"],
    ["2700", "0.4580", "0.4626", "−0.0046", "True"],
    ["3300", "1.1007", "1.1392", "−0.0385", "True"],
  ], [1560, 2000, 2000, 1900, 1900]));
push(...FIG("fig_multi.png", 430, 231,
  "그림 4. 다중 window 재현 — DA는 4개 중 2개에서만 no-DA를 이김(평균 Δrmse +0.0238, REPORT_ONLY)."));
push(P([B("결론: "), T("단일 window 성공은 재현되지 않는다. DA는 4개 중 2개 window에서만 no-DA를 이기고 평균 "),
  T("Δrmse는 +0.0238(악화). promotion은 REPORT_ONLY이며, 그 사유는 “케이스 부족”이 아니라 "),
  B("“모든 window를 이기지 못함”"), T("이다. window는 독립 case가 아니므로 promotion은 n_cases=1로 판정한다.")]));

push(H2("4.6 win/lose regime 분석 (N=4 case-study)"));
push(P([T("이긴 window(2700,3300)와 진 window(1500,2100)의 조건 차이를 3+1 feature family로 분리 분석했다. "),
  T("표본이 작아 통계검정이 아니라 hypothesis generator다.")]));
push(TABLE(
  ["ex-ante forcing", "win 평균", "lose 평균", "방향"],
  [
    ["tair_mean (°C)", "−0.02", "+0.84", "lose가 따뜻"],
    ["sw_mean (일사)", "28.1", "0.06", "win이 큼"],
    ["is_night_fraction", "0.31", "0.69", "win이 주간 비중↑"],
  ], [3360, 2000, 2000, 2000]));
push(P([T("win group은 "), B("평균적으로 더 춥고·일사가 크며·주간 비중이 높은"),
  T(" 쪽으로 치우친다(초기조건 오차가 lead에서 지배적인 regime). 다만 window별 예외가 있어(예: win k0=2700은 "),
  T("야간 비중 0.63) 인과 규칙이 아니라 후속 grid의 탐색 prior로만 사용한다. "),
  CODEF("dx_layer*"), T(" 같은 DA 자체 보정은 내생적(원인 아닌 결과)이라 별도 family로 분리했다.")]));

// 5. 사용자 매뉴얼
push(new Paragraph({ children: [new PageBreak()] }));
push(H1("5. 사용자 매뉴얼"));
push(H2("5.1 환경 준비"));
push(P([T("파이썬 3, NumPy가 필수이고, DA/forecast 계열 도구는 JAX·optax(dev extra)가 필요하다.")]));
push(CODE([
  "cd ~/Claude/Projects/dROAD",
  "pip install -e .            # 코어",
  "pip install -e .[dev]       # jax, optax (DA/forecast 도구용)",
]));
push(H2("5.2 테스트 실행"));
push(P([T("테스트는 코어(NumPy)와 "), CODEF("jax"), T(" 마커로 분리된다. 코어 187개, JAX 34개.")]));
push(CODE([
  "python3 -m pytest -q -m \"not jax\"     # 코어 187 (빠름)",
  "python3 -m pytest -q -m jax            # JAX 34 (느림)",
  "python3 tools/check_raw_primitives.py  # 원시연산 금지 감사 (clean=통과)",
]));
push(H2("5.3 리포트 도구 실행법"));
push(P([T("모든 리포트 도구는 "), CODEF("reports/"), T(" 아래에 "), CODEF(".md"), T(", "),
  CODEF(".csv"), T(", 일부는 "), CODEF("_meta.json"), T(" sidecar를 생성한다.")]));
push(CODE([
  "python3 tools/report_deviation_budget.py            # 질량 감사 기준선",
  "python3 tools/report_skill_gate.py                  # skill 게이트 기준선",
  "python3 tools/report_multiwindow_skill.py           # 6기간 skill 안정성",
  "python3 tools/report_da_evaluation.py --max-steps 4000   # 파라미터 DA",
  "python3 tools/report_forecast_da.py [--k0 --window --lead --bg-w]   # 상태 forecast DA",
  "python3 tools/report_forecast_da_multi.py [--windows N]  # 다중 window 재현",
  "python3 tools/analyze_forecast_da_regimes.py        # win/lose regime 분석",
]));
push(H2("5.4 리포트 해석 가이드"));
push(...BUL([
  [B("잔차(max_primary_residual): "), T("~0(< 1e-9)이어야 PASS. 0이 아니면 물리가 아니라 코드 누출이다.")],
  [B("진단 카운트: "), T("over_melt/overflow/negative-pre-clamp는 실패가 아니라 참조 물리 특성의 발생 빈도다.")],
  [B("gate_vs_bg / gate_vs_default: "), T("DA가 실제 baseline을 이겼는지의 hard 판정(RMSE 기준).")],
  [B("degradation_ratio: "), T("예보RMSE/동화창RMSE. >1이면 overfit 또는 lead가 본질적으로 더 어려운 구간(둘을 분리해 판단).")],
  [B("promotion_verdict: "), T("단일 fixture는 항상 REPORT_ONLY. PROMOTE는 독립 case가 확보돼야 가능.")],
  [B("_meta.json: "), T("게이트 결과·window interval·holdout 잔차/부담·참조값을 Markdown 파싱 없이 재현 가능한 형태로 보존.")],
]));
push(H2("5.5 forecast DA 파라미터 의미"));
push(TABLE(
  ["인자", "의미", "기본값"],
  [
    ["--k0", "분석 window 시작 스텝(spin 종료점, 배경상태 확정)", "2000"],
    ["--window", "동화창 스텝 수(초기상태 fit 구간)", "120"],
    ["--lead", "예보 lead 스텝 수(자유예보 구간)", "480"],
    ["--bg-w", "상태 보정에 대한 background 정규화(overfit 억제)", "0.05"],
  ], [1800, 5760, 1800]));

// 6. 품질보증
push(new Paragraph({ children: [new PageBreak()] }));
push(H1("6. 품질 보증 및 개발 프로세스"));
push(...BUL([
  [B("테스트: "), T("27개 파일, 코어 187 + JAX 34 = 총 221 테스트. 방어적 입력·게이트 의미·직렬화·회귀를 고정.")],
  [B("원시연산 감사: "), CODEF("check_raw_primitives.py"),
    T("가 NumPy 코어에서 where/clip/sqrt 등 무가드 원시연산을 금지(허용목록: branches, jax_model, smoothing, jax_storage).")],
  [B("적대적 검토 사이클: "), T("모든 주요 커밋은 외부 정적 검토(P0/P1/P2/P3)를 거쳐 반영되었다. "),
    T("false-pass 계열 결함(NaN 통과, 타입 우회, 게이트-진단 불일치)을 반복적으로 폐쇄.")],
  [B("정직성 게이트: "), T("단일 성공을 다중 window/case로 재검증하고, 재현되지 않으면 REPORT_ONLY로 남기는 보수성을 유지.")],
]));

// 7. 결론 및 향후
push(H1("7. 결론 및 향후 과제"));
push(P([T("현재 상태는 "), B("forecast DA report-only milestone"),
  T("으로 정직하고 견고하다. 핵심 성과는 (1) 질량 감사부터 forecast 게이트까지 일관된 검증 파이프라인, "),
  T("(2) 파라미터 DA와 상태추정 DA의 명확한 분리, (3) 단일 window의 상태-DA 개선을 다중 window에서 "),
  T("재검증해 과대주장을 방지한 점이다.")]));
push(P([B("핵심 과학적 결론: "),
  T("상태추정 forecast DA는 특정 regime(춥고·일사 큰 구간)에서 예보를 개선하나 4개 window 중 2개에서만 "),
  T("재현되어, “유망한 state-memory 신호”이지 “promote 가능한 일반 성능”은 아니다.")]));
push(H2("7.1 다음 단계 (우선순위)"));
push(...NUM([
  [B("bg_w × window × lead 축소 grid: "), T("regime 분석이 준 prior(냉·일사·짧은 lead)를 탐색 축으로, "),
    T("state-DA가 작동하는 정규화·시간척도 범위 규명.")],
  [B("full-model 확장 설계: "), T("dry 상태 보정 dx를 full 모델 열상태에 주입(설계 A) → 저장/상전이 포함(설계 B/C). "),
    T("이때 deviation_budget 감사가 다시 핵심이 된다.")],
  [B("독립 case 데이터 확보: "), T("promotion_gate 실사용을 위해 다른 station/day/weather regime 3~18 case의 "),
    T("manifest(cases.yaml)와 산출 스키마를 설계.")],
  [B("win/lose regime 심화: "), T("case가 늘면 forcing regime과 DA 이득의 관계를 통계적으로 검증.")],
]));
// 부록 A. 기호 정의
push(new Paragraph({ children: [new PageBreak()] }));
push(H1("부록 A. 기호 정의 (Nomenclature)"));
push(TABLE(
  ["기호", "의미", "단위·비고"],
  [
    ["T_surf, T_j", "노면온도(=(T₁+T₂)/2), j번째 층 온도", "°C"],
    ["α, ε, σ", "알베도, 방사율, Stefan–Boltzmann 상수", "– / – / W·m⁻²K⁻⁴"],
    ["R_net, LE", "표면 순복사속, 잠열속", "W·m⁻²"],
    ["G₀, G_j, h_BLC", "표면·층간 지중 열유속, 경계층 전도도", "W·m⁻² / –"],
    ["C_vsh, W, Δt", "체적 열용량, 함수량, 시간 스텝(=30s)", "J·m⁻³K⁻¹ / – / s"],
    ["M, H", "모델 전파 연산자, 관측 연산자", "–"],
    ["S_ext, K_ext, r", "외부 질량 유입·유출, 질량잔차(코드 누출)", "mm 수당량"],
    ["x₀, x_b, dx", "초기상태, 배경상태, 상태보정(layers 1:5)", "°C"],
    ["B, R, J", "배경·관측 오차공분산, 변분 비용함수", "–"],
    ["bg_w, λ, Σ", "배경 정규화, damping, Laplace 사후공분산", "–"],
    ["RMSE, τ, degradation", "제곱근평균제곱오차, freeze 임계, 예보/동화창 RMSE 비", "°C / °C / –"],
  ], [2500, 4860, 2000]));

// 부록 B. 재현 절차 및 파일 구조
push(H1("부록 B. 재현 절차 및 파일 구조"));
push(H2("B.1 재현 절차"));
push(...NUM([
  [B("설치: "), CODEF("pip install -e .[dev]"), T(" (JAX·optax 포함).")],
  [B("코어 검증: "), CODEF("pytest -m \"not jax\""), T(" → 187 통과, "), CODEF("check_raw_primitives.py"), T(" clean.")],
  [B("리포트 재생성: "), T("§5.3의 각 도구를 실행하면 "), CODEF("reports/"), T(" 산출물이 본문 §4 수치와 일치.")],
  [B("forecast DA 재현: "), CODEF("report_forecast_da.py"), T(" / "), CODEF("report_forecast_da_multi.py"),
    T(" / "), CODEF("analyze_forecast_da_regimes.py"), T(" 순으로 실행.")],
]));
push(H2("B.2 파일 구조"));
push(CODE([
  "dROAD/",
  "├─ droad/      # 17개 코어 모듈 (~2,836 LOC)",
  "│   ├─ ledger.py  deviation.py  skill_gate.py       # 감사·게이트",
  "│   ├─ thermal/boundary/radiation/storage/roadcond   # 물리(NumPy 정밀)",
  "│   ├─ model.py  driver.py                           # 결합·전체 rollout",
  "│   └─ jax_model/jax_storage/smoothing/assimilate/dual # 미분·변분 DA",
  "├─ tools/      # 9개 리포트/분석 도구 + check_raw_primitives",
  "├─ tests/      # 27개 파일 (코어 187 + jax 34 = 221)",
  "└─ reports/    # 6개 리포트 계열 (.md / .csv / _meta.json)",
]));
push(H2("B.3 커밋 이정표 (최근)"));
push(...BUL([
  [CODEF("688b8c3"), T(" — 진짜 forecast DA(상태추정): DA 0.2082 < no-DA 0.2210 PASS")],
  [CODEF("867e100"), T(" — 다중 window 재현검증(2/4 → REPORT_ONLY) + P1 버그·메모리 opt-in")],
  [CODEF("f8b64fc / 8bb1372 / a9f635f"), T(" — win/lose regime 분석 + feature family 분리")],
]));
push(SP());
push(new Paragraph({ border: { top: { style: BorderStyle.SINGLE, size: 6, color: BLUE, space: 6 } },
  spacing: { before: 200 }, children: [new TextRun({
    text: "본 보고서는 커밋 a9f635f 기준 리포지토리 상태를 반영한다. 모든 수치는 reports/ 아래 산출물에서 재현 가능하다.",
    italics: true, size: 18, color: "666666" })] }));

// ---------- assemble ----------
const doc = new Document({
  title: "dROAD 기술보고서 및 사용자 매뉴얼",
  subject: "미분가능 RoadSurf 노면기상 모델 · 자료동화",
  creator: "dROAD project",
  description: "작성일 2026-07-06 (v1.0), 대상 커밋 a9f635f",
  keywords: "RoadSurf, differentiable, data assimilation, JAX, forecast DA",
  styles: {
    default: { document: { run: { font: "Arial", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, font: "Arial", color: BLUE },
        paragraph: { spacing: { before: 300, after: 160 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, font: "Arial", color: "2E5496" },
        paragraph: { spacing: { before: 220, after: 120 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 23, bold: true, font: "Arial", color: "333333" },
        paragraph: { spacing: { before: 160, after: 80 }, outlineLevel: 2 } },
    ],
  },
  numbering: {
    config: [
      { reference: "bul", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•",
        alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 620, hanging: 300 } } } }] },
      { reference: "num", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.",
        alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 620, hanging: 300 } } } }] },
    ],
  },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 },
      margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    headers: { default: new Header({ children: [new Paragraph({
      alignment: AlignmentType.RIGHT, border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: LINE, space: 4 } },
      children: [new TextRun({ text: "dROAD 기술보고서 · 사용자 매뉴얼", size: 16, color: "888888" })] })] }) },
    footers: { default: new Footer({ children: [new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [new TextRun({ text: "— ", size: 16, color: "888888" }),
        new TextRun({ children: [PageNumber.CURRENT], size: 16, color: "888888" }),
        new TextRun({ text: " —", size: 16, color: "888888" })] })] }) },
    children: body,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync("dROAD_기술보고서.docx", buf);
  console.log("wrote dROAD_기술보고서.docx", buf.length, "bytes");
});
