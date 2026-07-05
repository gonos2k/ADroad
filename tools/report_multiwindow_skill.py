#!/usr/bin/env python3
"""Multi-period forecast skill: split the single fixture's valid-obs timeline into
K consecutive windows and evaluate the default model vs. a constant_initial
baseline (analysis-time obs held over the lead) in each, so skill STABILITY
across periods is visible — not one lucky holdout.

    python3 tools/report_multiwindow_skill.py [--windows K]

Writes reports/multiwindow_skill.md and .csv. This is the multi-case scaffold:
extra models (DA, smooth_compat) or real multi-station cases become extra rows.
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from tools.report_skill_gate import _run_default  # noqa: E402  (default rollout + obs)
from droad.deviation import deviation_budget  # noqa: E402
from droad.skill_gate import (  # noqa: E402
    forecast_metrics, skill_gate, aggregate_metrics, promotion_gate,
)

K = 6


def build(windows=K):
    out, tso = _run_default()
    dev = deviation_budget(out, case_id="default")
    tsurf = np.asarray(out["Tsurf"], float)
    idx = np.flatnonzero(tso > -100.0)
    folds = np.array_split(idx, windows)                 # K consecutive valid-obs windows

    rows, default_metrics = [], []
    for wi, fold in enumerate(folds):
        if len(fold) < 3:
            continue
        obs = tso[fold]
        obs_eval = obs[1:]
        const = np.full_like(obs_eval, obs[0])           # constant-initial baseline (not 1-step)
        m_def = forecast_metrics(tsurf[fold][1:], obs_eval)
        m_con = forecast_metrics(const, obs_eval)
        ok, _ = skill_gate(m_def, m_con)
        default_metrics.append(m_def)
        rows.append({"window": wi, "n": m_def["n"],
                     "default_rmse": m_def["rmse"], "const_initial_rmse": m_con["rmse"],
                     "freeze_thaw_accuracy": m_def["freeze_thaw_accuracy"],
                     "beats_const_initial": ok})
    agg = aggregate_metrics(default_metrics)
    all_beat = all(r["beats_const_initial"] for r in rows)
    # 1 fixture = 1 case: promotion gate correctly stays REPORT_ONLY (design §11)
    verdict, reasons = promotion_gate(n_cases=1, windows_beat_baseline=all_beat, deviation=dev)
    return rows, agg, (verdict, reasons)


_COLS = ("window", "n", "default_rmse", "const_initial_rmse", "freeze_thaw_accuracy", "beats_const_initial")


def main():
    windows = K
    if "--windows" in sys.argv:
        windows = int(sys.argv[sys.argv.index("--windows") + 1])
    rows, agg, (verdict, reasons) = build(windows)
    outdir = REPO / "reports"; outdir.mkdir(exist_ok=True)

    import csv as _csv
    import io as _io
    buf = _io.StringIO(); w = _csv.writer(buf); w.writerow(_COLS)
    for r in rows:
        w.writerow([r[c] for c in _COLS])

    head = "| " + " | ".join(_COLS) + " |"
    sep = "| " + " | ".join("---" for _ in _COLS) + " |"
    all_beat = all(r["beats_const_initial"] for r in rows)
    lines = [f"# Multi-window forecast skill ({len(rows)} periods)", "",
             "기간별 default vs constant_initial baseline(분석시각 obs 고정; 1-step persistence는 30s에서 "
             "자명해 미사용). 여러 기간에서 일관되게 이겨야 신뢰(한 창의 운 아님). gate: RMSE만 hard.",
             "", head, sep]
    for r in rows:
        lines.append("| " + " | ".join(
            (f"{r[c]:.4f}" if c in ("default_rmse", "const_initial_rmse", "freeze_thaw_accuracy")
             else str(r[c])) for c in _COLS) + " |")
    lines += ["", "## Aggregate (default across periods)",
              f"- n_windows: {agg['n_windows']}",
              f"- rmse_mean: {agg['rmse_mean']:.4f}",
              f"- rmse_max (worst window): {agg['rmse_max']:.4f}",
              f"- rmse_min (best window): {agg['rmse_min']:.4f}",
              f"- freeze_thaw_accuracy_mean: {agg['freeze_thaw_accuracy_mean']:.4f}",
              f"- beats constant_initial in ALL windows: {all_beat}",
              "", "## Promotion gate (design §11)",
              f"- verdict: **{verdict}**",
              *[f"- {r}" for r in reasons]]
    (outdir / "multiwindow_skill.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (outdir / "multiwindow_skill.csv").write_text(buf.getvalue(), encoding="utf-8")
    print("wrote reports/multiwindow_skill.{md,csv}")
    for r in rows:
        print(f"  window {r['window']}: default={r['default_rmse']:.4f} "
              f"const={r['const_initial_rmse']:.4f} beats={r['beats_const_initial']}")
    print(f"  rmse mean/worst: {agg['rmse_mean']:.4f} / {agg['rmse_max']:.4f} | all beat: {all_beat}")
    print(f"  promotion: {verdict}" + (f" — {'; '.join(reasons)}" if reasons else ""))


if __name__ == "__main__":
    main()
