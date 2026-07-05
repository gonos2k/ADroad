#!/usr/bin/env python3
"""Win/lose regime analysis for state forecast DA (Step 1 of the experiment plan).

The multi-window run shows state-DA beats no-DA in only some windows (regime-
dependent). This tool asks WHY: for each analysis window it collects forcing / obs
difficulty / DA-internal features and groups them by outcome (DA beat background or
not), so the features that separate wins from losses become visible.

N is tiny (one fixture, a few windows), so this is a CASE-STUDY diagnostic, NOT a
statistical test: it ranks candidate separating features by relative group-mean gap
and leaves the interpretation (overfit vs lead difficulty vs forcing transition) to
the reader. It reuses tools.report_forecast_da.build_multi (no new physics).

    python3 tools/analyze_forecast_da_regimes.py [--k0 K0] [--windows N]
                                                 [--window W] [--lead L] [--bg-w BG]

Writes reports/forecast_da_regimes.md, .csv and forecast_da_regimes_meta.json.
Requires jax/optax.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from tools.report_forecast_da import build_multi, WINDOW, LEAD, BG_WEIGHT  # noqa: E402

K0_FIRST, N_WINDOWS = 1500, 4


def _features(r):
    """Flatten one build_multi result into a single feature row (outcome + regime)."""
    f = {"k0": r["k0"], "beats_bg": bool(r["gate_da_vs_bg"][0]),
         "delta_rmse": r["rmse_delta_da_minus_bg"],
         "dx_l2": r["dx_l2"], "dx_max_abs": r["dx_max_abs"],
         "train_da": r["da"][1], "train_bg": r["bg"][1],
         "train_delta": r["train_delta_da_minus_bg"],
         "degradation_da": r["degradation_da"], "degradation_bg": r["degradation_bg"]}
    for k, v in r["regime"].items():
        if k == "dx_layers":
            for i, dv in enumerate(v):
                f[f"dx_layer{i + 1}"] = dv
        else:
            f[k] = v
    return f


def _numeric_feature_names(row):
    # every numeric column except the identifier (k0) and the label (beats_bg)
    return [k for k, v in row.items()
            if k not in ("k0", "beats_bg") and isinstance(v, (int, float)) and not isinstance(v, bool)]


def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def summarize_regimes(results):
    """Group per-window features by win/lose and rank features by relative separation.
    Returns (rows, win, lose, table). Pure — unit-testable without running any model."""
    if not results:
        raise ValueError("no window results to analyze")
    rows = [_features(r) for r in results]
    win = [x for x in rows if x["beats_bg"]]
    lose = [x for x in rows if not x["beats_bg"]]
    table = []
    # delta_rmse IS the (continuous) outcome label -> excluded from separators (circular).
    for f in _numeric_feature_names(rows[0]):
        if f == "delta_rmse":
            continue
        wm, lm = _mean([x[f] for x in win]), _mean([x[f] for x in lose])
        diff = wm - lm
        scale = max(abs(wm), abs(lm), 1e-9)
        table.append({"feature": f, "win_mean": wm, "lose_mean": lm, "diff": diff,
                      "direction": ("higher in wins" if diff > 0 else "higher in losses"),
                      "separation": abs(diff) / scale})
    table.sort(key=lambda t: t["separation"], reverse=True)   # strongest separators first
    return rows, win, lose, table


_ROW_COLS = ("k0", "beats_bg", "delta_rmse", "bg_init_error", "train_delta",
             "degradation_da", "dx_l2", "obs_std", "tair_trend_abs", "freeze_crossing_count")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Win/lose regime analysis for state forecast DA")
    ap.add_argument("--k0", type=int, default=K0_FIRST)
    ap.add_argument("--windows", type=int, default=N_WINDOWS)
    ap.add_argument("--window", type=int, default=WINDOW)
    ap.add_argument("--lead", type=int, default=LEAD)
    ap.add_argument("--bg-w", type=float, default=BG_WEIGHT, dest="bg_w")
    args = ap.parse_args()

    results, skipped = build_multi(args.k0, args.windows, args.window, args.lead, args.bg_w)
    if not results:
        raise RuntimeError("no window had enough valid observations")
    rows, win, lose, table = summarize_regimes(results)
    deltas = sorted(x["delta_rmse"] for x in rows)
    med = deltas[len(deltas) // 2] if len(deltas) % 2 else _mean(deltas[len(deltas) // 2 - 1:len(deltas) // 2 + 1])

    outdir = REPO / "reports"; outdir.mkdir(exist_ok=True)
    import csv as _csv, io as _io
    buf = _io.StringIO(); w = _csv.writer(buf)
    all_cols = ["k0", "beats_bg"] + _numeric_feature_names(rows[0])
    w.writerow(all_cols)
    for r in rows:
        w.writerow([r[c] for c in all_cols])

    lines = [f"# Forecast DA regime analysis ({len(rows)} windows)", "",
             "state-DA가 어떤 window에서 no-DA를 이기고 어떤 window에서 지는지, forcing/obs/DA 내부 "
             "feature로 설명한다. **표본이 작아 통계검정이 아니라 case-study**다 — win/lose 그룹 평균 "
             "차이가 큰 feature를 후보 신호로 제시할 뿐, 인과는 독자가 판단한다.",
             "", "## Summary",
             f"- windows: {len(rows)}  ·  DA wins: {len(win)}/{len(rows)}",
             f"- win windows (k0): {[x['k0'] for x in win]}",
             f"- lose windows (k0): {[x['k0'] for x in lose]}",
             f"- mean Δrmse (DA−bg): {_mean([x['delta_rmse'] for x in rows]):+.4f}  ·  median: {med:+.4f}",
             ""]
    # per-window snapshot (a readable subset of columns)
    snap = [c for c in _ROW_COLS if c in rows[0]]
    lines += ["## Per-window snapshot", "| " + " | ".join(snap) + " |",
              "| " + " | ".join("---" for _ in snap) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(
            (str(r[c]) if c in ("k0", "beats_bg", "freeze_crossing_count")
             else f"{r[c]:+.4f}" if c in ("delta_rmse", "train_delta")
             else f"{r[c]:.4f}") for c in snap) + " |")
    # candidate separators (only meaningful when we actually have both groups)
    lines += ["", "## Candidate regime signals (win vs lose group means)"]
    if win and lose:
        lines += ["가장 잘 분리하는 feature 상위 12개(상대 gap 기준):", "",
                  "| feature | win_mean | lose_mean | direction | separation |",
                  "| --- | ---: | ---: | --- | ---: |"]
        for t in table[:12]:
            lines.append(f"| {t['feature']} | {t['win_mean']:.4f} | {t['lose_mean']:.4f} "
                         f"| {t['direction']} | {t['separation']:.3f} |")
        lines += ["", "주의: `dx_layer*`/`dx_l2`/`dx_max_abs`는 DA가 **실제로 가한 보정(내생적)**이라 "
                  "win/lose의 원인이 아니라 결과에 가깝다. 독립적인 regime 원인은 forcing/obs feature "
                  "(tair_*, sw_mean, freeze_crossing_count, obs_* 등)에서 찾아야 한다.",
                  "", "## 가설 점검(케이스 스터디)",
                  "- **A (배경오차 큼 + model error 작음 → 이김)**: bg_init_error/train_delta가 win에서 "
                  "더 크게 유리한가?",
                  "- **B (lead 난이도 상승 → 짐)**: degradation_da·obs_std·tair_trend_abs가 lose에서 큰가?",
                  "- **C (dx 과도 → overfit으로 짐)**: dx_l2·dx_max_abs가 lose에서 큰가?"]
    else:
        lines += ["", f"(win {len(win)}, lose {len(lose)} — 두 그룹이 모두 있어야 비교 가능. "
                  "--windows/--k0로 표본을 늘리세요.)"]
    (outdir / "forecast_da_regimes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (outdir / "forecast_da_regimes.csv").write_text(buf.getvalue(), encoding="utf-8")

    import json as _json
    meta = {"k0_first": args.k0, "windows": args.windows, "window": args.window,
            "lead": args.lead, "bg_weight": args.bg_w, "skipped_windows": skipped,
            "n_win": len(win), "n_lose": len(lose),
            "win_k0": [x["k0"] for x in win], "lose_k0": [x["k0"] for x in lose],
            "top_separators": table[:12], "rows": rows}
    (outdir / "forecast_da_regimes_meta.json").write_text(_json.dumps(meta, indent=2), encoding="utf-8")
    print("wrote reports/forecast_da_regimes.{md,csv} + forecast_da_regimes_meta.json")
    print(f"  wins {len(win)}/{len(rows)}  win_k0={[x['k0'] for x in win]}  lose_k0={[x['k0'] for x in lose]}")
    if win and lose:
        for t in table[:5]:
            print(f"  {t['feature']:22s} win={t['win_mean']:+.4f} lose={t['lose_mean']:+.4f} "
                  f"sep={t['separation']:.3f} ({t['direction']})")


if __name__ == "__main__":
    main()
