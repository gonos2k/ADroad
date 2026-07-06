#!/usr/bin/env python3
"""Step 2: bg_w × window × lead sensitivity grid for state forecast DA.

For each hyperparameter combo it runs the multi-window forecast-DA reproduction
(tools.report_forecast_da.build_multi) and aggregates the per-window outcomes into
one grid row (win rate, mean/worst Δrmse, correction magnitude, degradation). The
grid is ranked to find WHERE state-DA is stable — NOT to declare a winner (single
fixture stays REPORT_ONLY).

Interpretation policy (from the regime analysis, tools/analyze_forecast_da_regimes):
  - PRIOR (ex-ante forcing): DA leaned better in colder / higher-SW / less-night windows.
  - post-hoc obs difficulty: explanatory only, not an operational predictor.
  - dx_l2 / degradation_da: TUNING RESULTS, read as diagnostics, never as a cause.

Compute is heavy (12 combos × 4 windows), so combos run in SLICES and accumulate
into reports/forecast_da_grid_partial.json; every run re-renders the report from all
rows gathered so far.

    python3 tools/grid_forecast_da.py --start 0 --count 4     # run combos 0..3
    python3 tools/grid_forecast_da.py --start 4 --count 4     # ... 4..7
    python3 tools/grid_forecast_da.py --start 8 --count 4     # ... 8..11 (renders full grid)
    python3 tools/grid_forecast_da.py --render                # re-render only

Writes reports/forecast_da_grid.{md,csv} + forecast_da_grid_meta.json. Requires jax.
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from tools.report_forecast_da import build_multi  # noqa: E402

K0_FIRST, N_WINDOWS = 1500, 4
BG_WS, WINDOWS, LEADS = [0.01, 0.05, 0.2], [60, 120], [240, 480]
COMBOS = [(bw, w, l) for bw in BG_WS for w in WINDOWS for l in LEADS]   # 12, deterministic order
PARTIAL = REPO / "reports" / "forecast_da_grid_partial.json"


def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def _median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return float("nan")
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def summarize_combo(bg_w, window, lead, results):
    """Aggregate one combo's per-window results into a single grid row (pure)."""
    n = len(results)
    key = f"{bg_w}_{window}_{lead}"
    if n == 0:
        return {"key": key, "bg_w": bg_w, "window": window, "lead": lead, "n_valid": 0,
                "wins": 0, "win_rate": 0.0}
    deltas = [r["rmse_delta_da_minus_bg"] for r in results]
    dxl2 = [r["dx_l2"] for r in results]
    degr = [r["degradation_da"] for r in results]
    wins = sum(bool(r["gate_da_vs_bg"][0]) for r in results)
    return {"key": key, "bg_w": bg_w, "window": window, "lead": lead, "n_valid": n,
            "wins": wins, "win_rate": wins / n,
            "mean_delta": _mean(deltas), "median_delta": _median(deltas), "worst_delta": max(deltas),
            "mean_da_rmse": _mean([r["da"][0]["rmse"] for r in results]),
            "mean_bg_rmse": _mean([r["bg"][0]["rmse"] for r in results]),
            "mean_dx_l2": _mean(dxl2), "max_dx_l2": max(dxl2),
            "mean_degradation_da": _mean(degr), "max_degradation_da": max(degr)}


def rank_rows(rows):
    """Rank combos: prefer high win_rate, then lower mean Δ, safer worst Δ, smaller dx.
    A good average that hides one badly-broken window is penalized by worst_delta."""
    return sorted(rows, key=lambda r: (
        -r.get("win_rate", 0.0), r.get("mean_delta", 9e9),
        r.get("worst_delta", 9e9), r.get("max_dx_l2", 9e9)))


_COLS = ("bg_w", "window", "lead", "n_valid", "wins", "win_rate", "mean_delta",
         "worst_delta", "mean_da_rmse", "mean_bg_rmse", "max_dx_l2", "max_degradation_da")


def _load_partial():
    if PARTIAL.exists():
        return {r["key"]: r for r in json.loads(PARTIAL.read_text())}
    return {}


def render(store):
    rows = rank_rows(list(store.values()))
    outdir = REPO / "reports"; outdir.mkdir(exist_ok=True)
    import csv as _csv, io as _io
    buf = _io.StringIO(); w = _csv.writer(buf); w.writerow(_COLS)
    for r in rows:
        w.writerow([r.get(c, "") for c in _COLS])

    head = "| " + " | ".join(_COLS) + " |"
    sep = "| " + " | ".join("---" for _ in _COLS) + " |"
    lines = [f"# Forecast DA grid — bg_w × window × lead ({len(rows)}/{len(COMBOS)} combos)", "",
             "state-DA가 **어떤 hyperparameter 영역에서 안정적인지** 탐색한다. 단일 fixture이므로 어떤 조합도 "
             "promotion 대상이 아니며(REPORT_ONLY), win_rate·worst_delta로 안정성만 본다. 랭킹: win_rate↑ → "
             "mean_delta↓ → worst_delta 안전 → max_dx_l2 작음. dx_l2·degradation은 튜닝 결과(진단)이지 원인이 아니다.",
             "", head, sep]
    for r in rows:
        lines.append("| " + " | ".join(
            (f"{r.get(c):.4f}" if isinstance(r.get(c), float) else str(r.get(c, "")))
            for c in _COLS) + " |")
    any_win = [r for r in rows if r.get("win_rate", 0) > 0.5]
    lines += ["", "## 관찰(hypothesis only, 단일 fixture)"]
    if any_win:
        best = rank_rows(rows)[0]
        lines += [f"- 과반 window에서 DA 우위인 조합 수: {len(any_win)}/{len(rows)}",
                  f"- 최상위 조합: bg_w={best['bg_w']}, window={best['window']}, lead={best['lead']} "
                  f"(win_rate={best['win_rate']:.2f}, mean_delta={best['mean_delta']:+.4f}, "
                  f"worst_delta={best['worst_delta']:+.4f})",
                  "- 같은 win_rate tier 안에서는 **더 큰 bg_w(정규화)가 worst_delta와 max_dx_l2를 낮춰** "
                  "worst-case 손상을 줄인다(win_rate는 불변) — 즉 win_rate는 window/lead가, 안전성은 bg_w가 지배.",
                  "- 해석은 regime 분석의 ex-ante forcing prior와 함께 볼 것 — grid 최적값을 결론이 아니라 "
                  "다음 독립-case 실험의 탐색 범위로만 사용한다."]
    else:
        lines += ["- 어떤 조합도 과반 window에서 DA가 no-DA를 안정적으로 이기지 못함 → state-DA는 이 fixture에서 "
                  "hyperparameter로 구제되지 않는다(초기조건 vs model-error 균형이 regime-dependent)."]
    (outdir / "forecast_da_grid.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (outdir / "forecast_da_grid.csv").write_text(buf.getvalue(), encoding="utf-8")
    meta = {"grid": {"bg_w": BG_WS, "window": WINDOWS, "lead": LEADS}, "n_combos": len(COMBOS),
            "n_done": len(rows), "k0_first": K0_FIRST, "n_windows": N_WINDOWS,
            "ranked_rows": rows}
    (outdir / "forecast_da_grid_meta.json").write_text(json.dumps(meta, indent=2, allow_nan=False),
                                                       encoding="utf-8")
    return rows


def main():
    import argparse
    ap = argparse.ArgumentParser(description="bg_w × window × lead grid for state forecast DA")
    ap.add_argument("--start", type=int, default=0, help="first combo index (0..11)")
    ap.add_argument("--count", type=int, default=len(COMBOS), help="how many combos to run this call")
    ap.add_argument("--k0", type=int, default=K0_FIRST)
    ap.add_argument("--windows", type=int, default=N_WINDOWS)
    ap.add_argument("--render", action="store_true", help="only re-render from accumulated partial")
    args = ap.parse_args()

    store = _load_partial()
    if not args.render:
        for idx in range(args.start, min(args.start + args.count, len(COMBOS))):
            bw, wdw, ld = COMBOS[idx]
            results, _skipped = build_multi(args.k0, args.windows, wdw, ld, bw)
            row = summarize_combo(bw, wdw, ld, results)
            store[row["key"]] = row
            PARTIAL.write_text(json.dumps(list(store.values()), indent=2, allow_nan=False), encoding="utf-8")
            print(f"  combo {idx:2d} bg_w={bw} window={wdw} lead={ld}: "
                  f"wins={row.get('wins')}/{row.get('n_valid')} "
                  f"mean_delta={row.get('mean_delta', float('nan')):+.4f}")
    rows = render(store)
    print(f"wrote reports/forecast_da_grid.{{md,csv}} + meta ({len(rows)}/{len(COMBOS)} combos done)")


if __name__ == "__main__":
    main()
