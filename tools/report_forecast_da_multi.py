#!/usr/bin/env python3
"""Multi-window forecast DA: does the single-fixture state-DA improvement REPRODUCE?

Runs the state-estimation forecast-DA cycle (tools/report_forecast_da.build_multi)
on N consecutive analysis windows and asks whether DA beats no-DA(background) in
EVERY window — then feeds that to promotion_gate (design §11): a single lucky
window is REPORT_ONLY; a robust, multi-window win is a candidate to PROMOTE.

    python3 tools/report_forecast_da_multi.py [--k0 K0] [--windows N] [--window W] [--lead L]

Writes reports/forecast_da_multi.md, .csv and forecast_da_multi_meta.json.
Requires jax/optax. The gate is skill-only (dry thermal model evolves no storages).
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from tools.report_forecast_da import build_multi, WINDOW, LEAD  # noqa: E402
from droad.skill_gate import promotion_gate  # noqa: E402

K0_FIRST, N_WINDOWS = 1500, 4


def summarize(results):
    rows = []
    for r in results:
        beats = r["gate_da_vs_bg"][0]
        rows.append({"k0": r["k0"], "da_rmse": r["da"][0]["rmse"], "bg_rmse": r["bg"][0]["rmse"],
                     "delta_da_minus_bg": r["rmse_delta_da_minus_bg"],
                     "train_da": r["da"][1], "train_bg": r["bg"][1],
                     "da_degradation": r["degradation_da"], "dx_l2": r["dx_l2"],
                     "beats_bg": beats})
    all_beat = bool(rows) and all(x["beats_bg"] for x in rows)
    # IMPORTANT: consecutive windows of ONE fixture are NOT independent cases. Window
    # reproducibility (n_beat/N) and independent-case promotion are DIFFERENT gates, so
    # promotion sees n_cases=1 (a single fixture) — it can never PROMOTE from one fixture
    # even at 4/4, matching the conservative report-only philosophy. Distinct
    # stations/days would raise n_cases. (skill-only: no deviation summary in dry model.)
    verdict, reasons = promotion_gate(n_cases=1, windows_beat_baseline=all_beat)
    return rows, all_beat, (verdict, reasons)


_COLS = ("k0", "da_rmse", "bg_rmse", "delta_da_minus_bg", "train_da", "train_bg",
         "da_degradation", "dx_l2", "beats_bg")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Multi-window forecast DA reproduction check")
    ap.add_argument("--k0", type=int, default=K0_FIRST, help="first analysis window start")
    ap.add_argument("--windows", type=int, default=N_WINDOWS, help="number of consecutive windows")
    ap.add_argument("--window", type=int, default=WINDOW, help="assimilation window steps")
    ap.add_argument("--lead", type=int, default=LEAD, help="forecast lead steps")
    ap.add_argument("--bg-w", type=float, default=None, dest="bg_w",
                    help="background regularization (default: report_forecast_da.BG_WEIGHT)")
    args = ap.parse_args()
    if args.k0 < 0:                                   # k0=0 is a valid start (no spin)
        ap.error("--k0 must be non-negative")
    for nm, v in (("windows", args.windows), ("window", args.window), ("lead", args.lead)):
        if v <= 0:
            ap.error(f"--{nm} must be positive")
    import math as _math
    if args.bg_w is not None and (not _math.isfinite(args.bg_w) or args.bg_w < 0):
        ap.error("--bg-w must be a finite non-negative number")

    from tools.report_forecast_da import BG_WEIGHT
    bg_w = BG_WEIGHT if args.bg_w is None else args.bg_w
    results = build_multi(args.k0, args.windows, args.window, args.lead, bg_w)
    if not results:
        raise RuntimeError("no window had enough valid observations")
    rows, all_beat, (verdict, reasons) = summarize(results)
    n_beat = sum(x["beats_bg"] for x in rows)
    mean_delta = sum(x["delta_da_minus_bg"] for x in rows) / len(rows)

    outdir = REPO / "reports"; outdir.mkdir(exist_ok=True)
    import csv as _csv, io as _io
    buf = _io.StringIO(); w = _csv.writer(buf); w.writerow(_COLS)
    for r in rows:
        w.writerow([r[c] for c in _COLS])

    head = "| " + " | ".join(_COLS) + " |"
    sep = "| " + " | ".join("---" for _ in _COLS) + " |"
    lines = [f"# Multi-window forecast DA ({len(rows)} windows)", "",
             "단일 fixture에서 관측된 'state-DA가 no-DA를 이긴다'가 여러 연속 analysis window에서 "
             "재현되는지 검증한다. 매 window에서 DA가 background를 이겨야 신뢰(한 창의 운 아님). "
             "gate: RMSE만 hard, deviation 감사 미적용(dry 모델). promotion_gate: 단일/소수 case는 "
             "REPORT_ONLY, 충분한 case에서 모두 이기면 PROMOTE 후보.",
             "", head, sep]
    for r in rows:
        lines.append("| " + " | ".join(
            (f"{r[c]:+.4f}" if c == "delta_da_minus_bg"
             else f"{r[c]:.4f}" if c in ("da_rmse", "bg_rmse", "train_da", "train_bg",
                                         "da_degradation", "dx_l2")
             else str(r[c])) for c in _COLS) + " |")
    lines += ["", "## Window reproducibility (같은 fixture, 연속 window)",
              f"- windows: {len(rows)}",
              f"- DA beats background in: {n_beat}/{len(rows)} windows",
              f"- mean Δrmse (DA − background): {mean_delta:+.4f}",
              f"- beats background in ALL windows: {all_beat}",
              "", "## Promotion gate (design §11)",
              "**window ≠ 독립 case**: 한 fixture의 연속 window는 서로 독립이 아니므로 promotion은 "
              "`n_cases=1`(단일 fixture)로 판정한다. 따라서 window를 모두 이겨도 단일 fixture로는 "
              "PROMOTE되지 않는다(독립 station/day를 늘려야 n_cases 증가).",
              f"- promotion_cases: 1  ·  window_reproducibility: {n_beat}/{len(rows)}",
              f"- verdict: **{verdict}**",
              *[f"- {x}" for x in reasons]]
    (outdir / "forecast_da_multi.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (outdir / "forecast_da_multi.csv").write_text(buf.getvalue(), encoding="utf-8")

    import json as _json
    meta = {"k0_first": args.k0, "n_windows_requested": args.windows,
            "n_windows_valid": len(rows), "window": args.window, "lead": args.lead,
            "bg_weight": bg_w, "da_beats_background": n_beat, "beats_all": all_beat,
            "window_reproducibility": f"{n_beat}/{len(rows)}", "promotion_cases": 1,
            "mean_delta_da_minus_bg": mean_delta, "promotion_verdict": verdict,
            "promotion_reasons": reasons, "windows": rows}
    (outdir / "forecast_da_multi_meta.json").write_text(_json.dumps(meta, indent=2), encoding="utf-8")
    print("wrote reports/forecast_da_multi.{md,csv} + forecast_da_multi_meta.json")
    for r in rows:
        print(f"  k0={r['k0']}: da={r['da_rmse']:.4f} bg={r['bg_rmse']:.4f} "
              f"Δ={r['delta_da_minus_bg']:+.4f} beats_bg={r['beats_bg']}")
    print(f"  DA beats bg in {n_beat}/{len(rows)} | all_beat={all_beat} | promotion: {verdict}")


if __name__ == "__main__":
    main()
