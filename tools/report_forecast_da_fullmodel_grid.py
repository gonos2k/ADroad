#!/usr/bin/env python3
"""A0 full-model forecast DA — bg_w × window × lead grid (stability region, not a winner).

For each hyperparameter combo it runs the A0 multi-window sweep (tools.report_forecast_da_
fullmodel_multi.run_windows over the standard k0 windows) and aggregates the per-window
outcomes into one grid row. The point is to find WHERE A0 is stable — not to declare a
promotable setting: this is one fixture, so every combo stays REPORT_ONLY / not eligible.

The honest columns (per the review): rates over windows, not just mean RMSE. A combo with
a good mean that badly breaks ONE window (worst_delta) or raises physics burden
(physics_worse_rate) is penalized by the ranking, so an "average-only" combo can't win.

Ranking (best first): gate_pass_rate↑ → physics_worse_rate↓ → worst_delta_rmse↓ →
mean_delta_rmse↓ → state_large_rate↓ (prefer robust skill AND clean physics, avoid
worst-case damage). ↑=higher-is-better, ↓=lower-is-better.

Compute is heavy (12 combos × 4 windows = 48 A0 runs), so combos run in SLICES and
accumulate into reports/forecast_da_fullmodel_grid_partial.json; every run re-renders from
all combos gathered so far.

    python3 tools/report_forecast_da_fullmodel_grid.py --start 0 --count 1   # combo 0
    python3 tools/report_forecast_da_fullmodel_grid.py --start 1 --count 1   # combo 1 ...
    python3 tools/report_forecast_da_fullmodel_grid.py --render              # re-render only

Writes reports/forecast_da_fullmodel_grid.{md,csv} + _meta.json. Requires jax.
"""
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from tools.report_forecast_da_fullmodel_multi import (  # noqa: E402
    run_windows, summarize_multi, K0_FIRST, N_WINDOWS, STRIDE,
)

BG_WS, WINDOWS, LEADS = [0.01, 0.05, 0.2], [60, 120], [240, 480]
COMBOS = [(bw, w, l) for bw in BG_WS for w in WINDOWS for l in LEADS]   # 12, deterministic
PARTIAL = REPO / "reports" / "forecast_da_fullmodel_grid_partial.json"
_SCHEMA = 1
_CONFIG = {"bg_ws": BG_WS, "windows": WINDOWS, "leads": LEADS,
           "k0_first": K0_FIRST, "n_windows": N_WINDOWS, "stride": STRIDE}


def _combo_key(bg_w, window, lead):
    return f"{bg_w}_{window}_{lead}"


def summarize_combo(bg_w, window, lead, rows):
    """Aggregate one combo's per-window rows into a grid row (pure). Reuses summarize_multi
    for the rate/promotion logic, then adds the combo axes + state_large_rate.

    promotion policy: this is a SINGLE fixture, so promotion_eligible is ALWAYS False here
    (window reproducibility ≠ independent-case promotion, per design §11). The window-level
    precondition (all_beat ∧ residual_clean) is preserved separately as
    window_precondition_met so a 4/4-clean combo is still visible without overclaiming.
    Empty combos (every window skipped) use None for numeric fields — never inf/NaN, which
    json.dumps(allow_nan=False) would reject on save/render."""
    key = _combo_key(bg_w, window, lead)
    base = {"key": key, "bg_w": bg_w, "window": window, "lead": lead, "n_valid": len(rows)}
    if not rows:
        return {**base, "gate_pass_n": 0, "gate_pass_rate": 0.0,
                "skill_improved_n": 0, "skill_improved_rate": 0.0,
                "physics_worse_n": 0, "physics_worse_rate": 1.0,
                "state_large_n": 0, "state_large_rate": 1.0,
                "mean_delta_rmse": None, "worst_delta_rmse": None, "max_lead_diag_delta": None,
                "max_residual": None, "residual_clean": False,
                "window_precondition_met": False, "promotion_eligible": False,
                "promotion": "REPORT_ONLY"}
    s = summarize_multi(rows)
    n = s["n_valid"]
    return {**base,
            "gate_pass_n": s["gate_pass_n"], "gate_pass_rate": s["gate_pass_rate"],
            "skill_improved_n": s["skill_improved_n"], "skill_improved_rate": s["skill_improved_rate"],
            "physics_worse_n": s["physics_worse_n"], "physics_worse_rate": s["physics_worse_rate"],
            "state_large_n": s["state_large_n"], "state_large_rate": s["state_large_n"] / n,
            "mean_delta_rmse": s["mean_delta"], "worst_delta_rmse": s["worst_delta"],
            "max_lead_diag_delta": s["max_lead_diag_delta"],
            "max_residual": s["max_residual"], "residual_clean": s["residual_clean"],
            "window_precondition_met": s["promotion_eligible"],   # all_beat ∧ residual_clean
            "promotion_eligible": False,                          # single-fixture grid: never
            "promotion": s["promotion"][0]}


def _bad_high(x):
    """None (empty combo) sorts as worst on a lower-is-better key."""
    return math.inf if x is None else x


def _fmt_delta(x):
    """Format a Δrmse that may be None (an all-skipped empty combo) without crashing."""
    return "NA" if x is None else f"{x:+.4f}"


def rank_rows(rows):
    """Prefer robust skill AND clean physics; punish worst-case damage. An average-only
    combo that breaks one window (worst_delta) or raises burden (physics_worse) is demoted.
    Ranking: gate_pass_rate↑ → physics_worse_rate↓ → worst_delta↓ → mean_delta↓ → state_large_rate↓."""
    return sorted(rows, key=lambda r: (
        -r.get("gate_pass_rate", 0.0), r.get("physics_worse_rate", 1.0),
        _bad_high(r.get("worst_delta_rmse")), _bad_high(r.get("mean_delta_rmse")),
        r.get("state_large_rate", 1.0)))


_COLS = ("bg_w", "window", "lead", "n_valid", "gate_pass_rate", "skill_improved_rate",
         "physics_worse_rate", "state_large_rate", "mean_delta_rmse", "worst_delta_rmse",
         "max_lead_diag_delta", "max_residual", "residual_clean", "window_precondition_met")


# axes are always present numbers; metric fields may be None for an empty combo, so we only
# require their presence, not finiteness (matching summarize_combo's None-for-empty policy).
_REQUIRED_GRID_KEYS = set(_COLS) | {"key", "promotion", "promotion_eligible"}


# metric fields may be None (empty combo); when present they must be finite numbers.
_NUMERIC_OR_NONE_GRID_KEYS = {"gate_pass_rate", "skill_improved_rate", "physics_worse_rate",
                              "state_large_rate", "mean_delta_rmse", "worst_delta_rmse",
                              "max_lead_diag_delta", "max_residual"}
_BOOL_GRID_KEYS = {"residual_clean", "window_precondition_met", "promotion_eligible"}


def _validate_grid_row(r):
    if not isinstance(r, dict):
        raise ValueError("grid row must be a mapping")
    missing = _REQUIRED_GRID_KEYS - set(r)
    if missing:
        raise ValueError(f"grid row missing keys: {sorted(missing)}")
    for k in ("bg_w", "window", "lead", "n_valid"):     # axes must be real numbers
        v = r[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError(f"grid row {k} must be a finite number")
    for k in _NUMERIC_OR_NONE_GRID_KEYS:                 # None ok (empty combo), else finite
        v = r[k]
        if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float))
                              or not math.isfinite(v)):
            raise ValueError(f"grid row {k} must be a finite number or None")
    for k in _BOOL_GRID_KEYS:
        if not isinstance(r[k], bool):
            raise ValueError(f"grid row {k} must be bool")
    return r


def _load_partial():
    if not PARTIAL.exists():
        return {}
    try:
        blob = json.loads(PARTIAL.read_text())
        if blob.get("schema_version") != _SCHEMA or blob.get("config") != _CONFIG:
            print(f"  [warn] stale grid partial ({PARTIAL.name}) — schema/config mismatch, ignoring")
            return {}
        return {_validate_grid_row(r)["key"]: r for r in blob["rows"]}
    except (ValueError, KeyError, TypeError, AttributeError) as e:
        print(f"  [warn] unreadable/incomplete grid partial ({PARTIAL.name}) — ignoring ({e})")
        return {}


def _save_partial(store):
    PARTIAL.write_text(json.dumps(
        {"schema_version": _SCHEMA, "config": _CONFIG, "rows": [store[k] for k in store]},
        indent=2, allow_nan=False), encoding="utf-8")


def render(store):
    rows = rank_rows(list(store.values()))
    outdir = REPO / "reports"; outdir.mkdir(exist_ok=True)
    import csv as _csv, io as _io
    buf = _io.StringIO(); w = _csv.writer(buf); w.writerow(_COLS)
    for r in rows:
        w.writerow([r.get(c, "") for c in _COLS])

    head = "| " + " | ".join(_COLS) + " |"
    sep = "| " + " | ".join("---" for _ in _COLS) + " |"
    lines = [f"# A0 full-model forecast DA — grid bg_w×window×lead ({len(rows)}/{len(COMBOS)} combos)",
             "",
             "A0가 **어떤 hyperparameter 영역에서 안정적으로 유효한지** 탐색한다(우승자 선정 아님). 단일 fixture라 "
             "어떤 조합도 promotion 대상이 아니다(promotion_eligible=False). 평균 RMSE가 아니라 rate를 본다: "
             "physics_worse_rate>0이거나 worst_delta_rmse가 크면 평균이 좋아도 operational로 위험. 랭킹: "
             "gate_pass_rate↑ → physics_worse_rate↓ → worst_delta↓ → mean_delta↓ → state_large_rate↓.",
             "",
             f"windows: k0={K0_FIRST}+i×{STRIDE} (i<{N_WINDOWS}). combos: bg_w{BG_WS} × window{WINDOWS} × "
             f"lead{LEADS}. Δrmse=DA−BG(클수록 나쁨). residual은 code-leak 게이트(clean 유지 기대).",
             "", head, sep]
    for r in rows:
        lines.append("| " + " | ".join(
            (f"{r.get(c):.4f}" if isinstance(r.get(c), float) else str(r.get(c, "")))
            for c in _COLS) + " |")

    if rows:
        best = rows[0]
        stable = [r for r in rows if r.get("gate_pass_rate", 0) > 0.5 and r.get("physics_worse_rate", 1) == 0]
        lines += ["", "## 관찰 (single fixture → 모두 REPORT_ONLY)",
                  f"- physics_worse 없이 과반 window PASS인 조합: {len(stable)}/{len(rows)}",
                  f"- 최상위: bg_w={best['bg_w']} window={best['window']} lead={best['lead']} "
                  f"(gate_pass_rate={best.get('gate_pass_rate', 0):.2f}, "
                  f"physics_worse_rate={best.get('physics_worse_rate', 0):.2f}, "
                  f"worst_delta={_fmt_delta(best.get('worst_delta_rmse'))})",
                  "- 해석축: 짧은 window vs 긴 window, lead↑에 따른 state memory 소실, bg_w↓의 overfit/state_large, "
                  "bg_w↑의 DA 효과 소실. grid 최적값은 결론이 아니라 독립-case(Step 4) 실험의 탐색 범위."]
    (outdir / "forecast_da_fullmodel_grid.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (outdir / "forecast_da_fullmodel_grid.csv").write_text(buf.getvalue(), encoding="utf-8")
    meta = {"config": _CONFIG, "n_combos": len(COMBOS), "n_done": len(rows),
            "grid": [{"bg_w": r["bg_w"], "window": r["window"], "lead": r["lead"],
                      "summary": {k: v for k, v in r.items()
                                  if k not in ("bg_w", "window", "lead", "key", "windows")},
                      "windows": store[r["key"]].get("windows", [])} for r in rows]}
    (outdir / "forecast_da_fullmodel_grid_meta.json").write_text(
        json.dumps(meta, indent=2, allow_nan=False), encoding="utf-8")
    return rows


def main():
    import argparse
    ap = argparse.ArgumentParser(description="A0 full-model forecast DA grid (bg_w×window×lead)")
    ap.add_argument("--start", type=int, default=0, help=f"first combo index (0..{len(COMBOS)-1})")
    ap.add_argument("--count", type=int, default=len(COMBOS), help="how many combos this call")
    ap.add_argument("--render", action="store_true", help="only re-render from accumulated partial")
    args = ap.parse_args()
    if args.start < 0 or args.count < 0:
        ap.error("--start and --count must be non-negative")
    if not args.render and args.count == 0:
        ap.error("--count must be positive (use --render to only re-render)")
    if not args.render and args.start >= len(COMBOS):
        ap.error(f"--start must be < {len(COMBOS)} (use --render to only re-render)")

    store = _load_partial()
    if args.render and not store:
        ap.error("no valid grid partial to render (missing/stale)")
    if not args.render:
        for idx in range(args.start, min(args.start + args.count, len(COMBOS))):
            bw, wdw, ld = COMBOS[idx]
            rows = run_windows(window=wdw, lead=ld, bg_w=bw)
            combo = summarize_combo(bw, wdw, ld, rows)
            combo["windows"] = rows                       # keep raw windows for meta nesting
            store[combo["key"]] = combo
            _save_partial(store)
            print(f"  combo {idx:2d} bg_w={bw} window={wdw} lead={ld}: "
                  f"gate_pass_rate={combo.get('gate_pass_rate', 0):.2f} "
                  f"physics_worse_rate={combo.get('physics_worse_rate', 0):.2f} "
                  f"worst_delta={_fmt_delta(combo.get('worst_delta_rmse'))}")
    rows = render(store)
    print(f"wrote reports/forecast_da_fullmodel_grid.{{md,csv}} + meta "
          f"({len(rows)}/{len(COMBOS)} combos)")


if __name__ == "__main__":
    main()
