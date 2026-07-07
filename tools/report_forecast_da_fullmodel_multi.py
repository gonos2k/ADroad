#!/usr/bin/env python3
"""A0 full-model forecast DA — multi-window reproduction / generality check.

Runs the full-model A0 cycle (tools.report_forecast_da_fullmodel.build_a0) on N
consecutive analysis windows and asks the honest questions the single-fixture A0
report can't: across windows, how often does DA improve skill, how often does the
lead physics burden worsen (physics_worse), how often is the state correction large,
and does DA beat no-DA — on skill AND physics — in EVERY window?

The per-window gate is the full-model gate (RMSE hard + physics burden non-worse), so
`gate_pass` already folds skill and physics. We still surface skill_improved and
physics_worse separately because the interesting failure mode is skill↑ yet physics↓.

promotion policy (design §11): these windows come from ONE fixture, so — exactly like
the dry multi-window tool — n_cases is pinned to 1 and the verdict stays REPORT_ONLY
even at 4/4. Distinct stations/days would raise n_cases (that is Step 4, cases.yaml).

Compute is heavy (each window spins the full model to k0 + fits dx + 2 full forecasts),
so windows run in SLICES and accumulate into reports/forecast_da_fullmodel_multi_partial.json;
every run re-renders the report from all windows gathered so far.

    python3 tools/report_forecast_da_fullmodel_multi.py --start 0 --count 1   # window 0
    python3 tools/report_forecast_da_fullmodel_multi.py --start 1 --count 1   # window 1 ...
    python3 tools/report_forecast_da_fullmodel_multi.py --render              # re-render only

Writes reports/forecast_da_fullmodel_multi.{md,csv} + _meta.json. Requires jax.
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from tools.report_forecast_da_fullmodel import build_a0, WINDOW, LEAD, BG_WEIGHT  # noqa: E402
from droad.skill_gate import promotion_gate  # noqa: E402

K0_FIRST, N_WINDOWS, STRIDE = 1500, 4, 600     # 1500/2100/2700/3300 (matches dry multi)
PARTIAL = REPO / "reports" / "forecast_da_fullmodel_multi_partial.json"


def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def _case_row(r):
    """Flatten one build_a0 result into a per-window row (pure)."""
    m_bg, dev_bg = r["bg"]; m_da, dev_da = r["da"]
    dev_bg_win, dev_da_win = r["win"]
    delta = r["rmse_delta_da_minus_bg"]
    return {"k0": r["k0"], "da_rmse": m_da["rmse"], "bg_rmse": m_bg["rmse"],
            "rmse_delta": delta, "skill_improved": bool(delta < 0.0),
            "gate_pass": bool(r["gate_da_vs_bg"][0]), "physics_worse": bool(r["physics_worse"]),
            "state_large": bool(r["dx_max_abs"] > 2.0 or r["dx_l2"] > 3.0),
            "lead_diag_bg": dev_bg["diagnostic_steps_rate"],
            "lead_diag_da": dev_da["diagnostic_steps_rate"],
            "win_diag_bg": dev_bg_win["diagnostic_steps_rate"],
            "win_diag_da": dev_da_win["diagnostic_steps_rate"],
            "resid_bg": dev_bg["max_primary_residual"], "resid_da": dev_da["max_primary_residual"],
            "dx_l2": r["dx_l2"], "dx_max_abs": r["dx_max_abs"]}


def summarize_multi(rows, residual_atol=1e-9):
    """Aggregate per-window rows into rates + a promotion verdict (pure).

    windows_beat_baseline = every window's full-model gate PASSes (skill AND physics).
    n_cases is pinned to 1 (single fixture) so the verdict is REPORT_ONLY by policy —
    a robust multi-*case* win (Step 4) is what would raise n_cases toward PROMOTE."""
    n = len(rows)
    if n == 0:
        return {"n_valid": 0, "promotion": ("REPORT_ONLY", ["no windows run yet"])}
    wins = sum(x["gate_pass"] for x in rows)
    all_beat = wins == n
    deltas = [x["rmse_delta"] for x in rows]
    max_resid = max(max(x["resid_bg"], x["resid_da"]) for x in rows)
    residual_clean = bool(max_resid <= residual_atol)
    verdict, reasons = promotion_gate(n_cases=1, windows_beat_baseline=all_beat,
                                      residual_atol=residual_atol)
    # aggregate audit must also block promotion: a dirty residual in ANY window means
    # the code-leak detector fired, so never PROMOTE regardless of skill (design §11).
    if not residual_clean:
        reasons = list(reasons) + [f"aggregate residual {max_resid:.3e} > {residual_atol:.0e}"]
        verdict = "REPORT_ONLY"
    return {"n_valid": n, "gate_pass_n": wins, "gate_pass_rate": wins / n, "all_beat": all_beat,
            "skill_improved_n": sum(x["skill_improved"] for x in rows),
            "skill_improved_rate": sum(x["skill_improved"] for x in rows) / n,
            "physics_worse_n": sum(x["physics_worse"] for x in rows),
            "physics_worse_rate": sum(x["physics_worse"] for x in rows) / n,
            "state_large_n": sum(x["state_large"] for x in rows),
            "mean_delta": _mean(deltas), "worst_delta": max(deltas),
            "max_lead_diag_da": max(x["lead_diag_da"] for x in rows),
            "max_residual": max_resid, "residual_clean": residual_clean,
            "promotion": (verdict, reasons)}


# baseline (bg) diagnostic/residual columns are shown next to DA so a physics_worse=False
# row with non-zero lead_diag_da is legible: the gate compares DA burden vs baseline, not
# vs zero, so the reader needs both columns to see WHY it's not worse.
_COLS = ("k0", "bg_rmse", "da_rmse", "rmse_delta", "skill_improved", "gate_pass",
         "physics_worse", "state_large", "lead_diag_bg", "lead_diag_da",
         "win_diag_bg", "win_diag_da", "resid_bg", "resid_da")


_SCHEMA = 1
_CONFIG = {"k0_first": K0_FIRST, "stride": STRIDE, "window": WINDOW, "lead": LEAD,
           "bg_weight": BG_WEIGHT}


def _load_partial():
    """Load accumulated windows, but IGNORE a stale partial (old schema or different
    config) — a runner resuming across config changes must not mix incompatible rows."""
    if not PARTIAL.exists():
        return {}
    try:
        blob = json.loads(PARTIAL.read_text())
        if blob.get("schema_version") != _SCHEMA or blob.get("config") != _CONFIG:
            print(f"  [warn] stale partial ({PARTIAL.name}) — schema/config mismatch, ignoring")
            return {}
        return {int(r["k0"]): r for r in blob["rows"]}
    except (ValueError, KeyError, TypeError, AttributeError):
        print(f"  [warn] unreadable partial ({PARTIAL.name}) — ignoring")
        return {}


def _save_partial(store):
    PARTIAL.write_text(json.dumps(
        {"schema_version": _SCHEMA, "config": _CONFIG, "rows": [store[k] for k in sorted(store)]},
        indent=2, allow_nan=False), encoding="utf-8")


def render(store):
    rows = [store[k] for k in sorted(store)]
    s = summarize_multi(rows)
    outdir = REPO / "reports"; outdir.mkdir(exist_ok=True)
    import csv as _csv, io as _io
    buf = _io.StringIO(); w = _csv.writer(buf); w.writerow(_COLS)
    for r in rows:
        w.writerow([r.get(c, "") for c in _COLS])

    head = "| " + " | ".join(_COLS) + " |"
    sep = "| " + " | ".join("---" for _ in _COLS) + " |"
    lines = [f"# A0 full-model forecast DA — multi-window ({len(rows)}/{N_WINDOWS} windows)", "",
             "단일 fixture A0가 보여줄 수 없는 것을 본다: 여러 연속 window에서 DA가 skill을 개선하는 빈도, "
             "lead physics burden이 악화되는(physics_worse) 빈도, state correction이 큰 빈도, 그리고 매 window에서 "
             "DA가 no-DA를 **skill과 physics 모두** 이기는지. per-window gate는 full-model gate(RMSE hard + 물리부담 "
             "비악화)라 gate_pass가 둘을 이미 포함한다. skill↑ yet physics↓가 핵심 관심 실패 양상이라 둘을 따로 표기.",
             "",
             f"windows: k0={K0_FIRST}+i×{STRIDE} (i<{N_WINDOWS}) · 동화창 {WINDOW} · lead {LEAD} · bg_w {BG_WEIGHT}. "
             "promotion: 단일 fixture라 n_cases=1 고정 → REPORT_ONLY(정직성). 다수 독립 case는 Step 4(cases.yaml).",
             "", head, sep]
    for r in rows:
        lines.append("| " + " | ".join(
            (f"{r.get(c):.4f}" if isinstance(r.get(c), float) else str(r.get(c, "")))
            for c in _COLS) + " |")

    if rows:
        verdict, reasons = s["promotion"]
        lines += ["", "## 집계 (single fixture → REPORT_ONLY)",
                  f"- gate PASS: {s['gate_pass_n']}/{s['n_valid']} (rate {s['gate_pass_rate']:.2f}) · "
                  f"beats-all={s['all_beat']}",
                  f"- skill 개선: {s['skill_improved_n']}/{s['n_valid']} (rate {s['skill_improved_rate']:.2f}) · "
                  f"physics_worse: {s['physics_worse_n']}/{s['n_valid']} (rate {s['physics_worse_rate']:.2f})",
                  f"- state_correction_large: {s['state_large_n']}/{s['n_valid']} · "
                  f"mean Δrmse {s['mean_delta']:+.4f} · worst Δrmse {s['worst_delta']:+.4f} "
                  "(Δrmse=DA−BG, 클수록 나쁨 → worst=max)",
                  f"- max lead diag_rate(DA): {s['max_lead_diag_da']:.4f} · "
                  f"max residual {s['max_residual']:.2e} (clean={s['residual_clean']})",
                  f"- **promotion: {verdict}** — {'; '.join(reasons)}",
                  "", "해석: gate_pass_rate<1이면 DA가 매 window를 이기지 못한 것(regime-dependent). "
                  "physics_worse_rate>0이면 일부 window에서 열 보정이 물리 부담을 키운 것 — skill이 좋아도 그 window는 "
                  "FAIL해야 정직. residual이 clean하면 실패 원인은 accounting leak이 아니라 deviation burden이다."]
    (outdir / "forecast_da_fullmodel_multi.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (outdir / "forecast_da_fullmodel_multi.csv").write_text(buf.getvalue(), encoding="utf-8")
    meta = {"k0_first": K0_FIRST, "n_windows": N_WINDOWS, "stride": STRIDE,
            "window": WINDOW, "lead": LEAD, "bg_weight": BG_WEIGHT,
            "n_valid": len(rows), "summary": {k: v for k, v in s.items() if k != "promotion"},
            "promotion": s["promotion"][0], "promotion_reasons": s["promotion"][1],
            "windows": rows}
    (outdir / "forecast_da_fullmodel_multi_meta.json").write_text(
        json.dumps(meta, indent=2, allow_nan=False), encoding="utf-8")
    return rows, s


def main():
    import argparse
    ap = argparse.ArgumentParser(description="A0 full-model forecast DA multi-window check")
    ap.add_argument("--start", type=int, default=0, help=f"first window index (0..{N_WINDOWS-1})")
    ap.add_argument("--count", type=int, default=N_WINDOWS, help="how many windows to run this call")
    ap.add_argument("--render", action="store_true", help="only re-render from accumulated partial")
    args = ap.parse_args()
    if args.start < 0 or args.count < 0:
        ap.error("--start and --count must be non-negative")
    if not args.render and args.start >= N_WINDOWS:
        ap.error(f"--start must be < N_WINDOWS ({N_WINDOWS}); use --render to only re-render")

    store = _load_partial()
    if not args.render:
        for i in range(args.start, min(args.start + args.count, N_WINDOWS)):
            k0 = K0_FIRST + i * STRIDE
            try:
                r = build_a0(k0=k0, window=WINDOW, lead=LEAD, bg_w=BG_WEIGHT)
            except RuntimeError as e:                       # too few valid obs in this window
                print(f"  window {i} k0={k0}: SKIP ({e})")
                continue
            row = _case_row(r)
            store[k0] = row
            _save_partial(store)
            print(f"  window {i} k0={k0}: Δrmse={row['rmse_delta']:+.4f} gate_pass={row['gate_pass']} "
                  f"physics_worse={row['physics_worse']} lead_diag_da={row['lead_diag_da']:.4f}")
    rows, s = render(store)
    v = s.get("promotion", ("REPORT_ONLY", []))[0] if rows else "REPORT_ONLY"
    print(f"wrote reports/forecast_da_fullmodel_multi.{{md,csv}} + meta "
          f"({len(rows)}/{N_WINDOWS} windows, promotion={v})")


if __name__ == "__main__":
    main()
