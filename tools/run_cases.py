#!/usr/bin/env python3
"""run_cases — the real promotion path (Step 4): run a fixed A0 setting on every case in an
independent-case manifest and feed the per-case verdicts to promotion_gate with the ACTUAL
n_cases (not the n_cases=1 the single-fixture tools are pinned to).

This is the first place a model can actually be a PROMOTE candidate: promotion_gate promotes
only when there are enough distinct cases AND every case beats baseline on skill+physics AND
residual is clean. Each case's `gate_pass` already comes from skill_gate, so it folds skill,
physics burden, and accounting residual per case; summarize_cases aggregates across cases.

SCOPE (skeleton): the aggregation + driver wiring are implemented and tested. The per-case
execution `run_one(case, setting)` is INJECTABLE — the default raises, because a per-case
forcing/obs loader does not exist yet (we still have one fixture). Wire a real loader in the
next increment; the honest contract (aggregate -> promotion_gate) is locked here.

    # once a per-case loader + real manifest exist:
    #   python3 tools/run_cases.py cases.yaml --bg-w 0.05 --window 60 --lead 480
"""
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from droad.skill_gate import promotion_gate  # noqa: E402
from tools.validate_cases import validate_manifest  # noqa: E402

# case row fields a run_one implementation must return (per case). residual_clean is NOT an
# input — it is derived from max_residual so a row can't disagree with itself.
CASE_FIELDS = ("case_id", "regime", "gate_pass", "physics_worse", "state_large",
               "rmse_delta", "max_residual")
_BOOL_FIELDS = ("gate_pass", "physics_worse", "state_large")


def _finite(name, x):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise ValueError(f"{name} must be a finite number")
    return float(x)


def _strict_bool(name, x):
    """Accept a genuine boolean (Python bool or numpy bool_) and return a Python bool; reject
    strings/ints so a corrupt A0 flag like 'False' can't sneak through as truthy."""
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    raise ValueError(f"{name} must be bool")


def make_setting(bg_w, window, lead):
    """A frozen A0 hyperparameter setting (chosen from the grid stability region)."""
    bg_w = _finite("bg_w", bg_w)
    for name, v in (("window", window), ("lead", lead)):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError(f"setting {name} must be a finite integer number of steps")
        if int(v) != v:                        # finite check first so inf/nan don't OverflowError
            raise ValueError(f"setting {name} must be an integer number of steps")
    if not (bg_w > 0 and window > 0 and lead > 0):
        raise ValueError("setting bg_w/window/lead must be positive")
    return {"bg_w": bg_w, "window": int(window), "lead": int(lead)}


def _validate_case_row(r):
    """Strict per-case validation — this feeds a real PROMOTE verdict, so an injected run_one
    returning an inconsistent row must be rejected, not silently promoted."""
    if not isinstance(r, dict) or set(CASE_FIELDS) - set(r):
        raise ValueError(f"case result must have fields {CASE_FIELDS}")
    for k in ("case_id", "regime"):
        if not isinstance(r[k], str) or not r[k].strip():
            raise ValueError(f"case result {k!r} must be a non-empty string")
    for k in _BOOL_FIELDS:
        if not isinstance(r[k], bool):
            raise ValueError(f"case result {k!r} must be bool")
    # the core honesty contract: a case that worsened physics cannot also 'pass' the gate.
    if r["physics_worse"] and r["gate_pass"]:
        raise ValueError(f"case {r['case_id']!r} inconsistent: physics_worse=True requires "
                         "gate_pass=False")
    _finite("rmse_delta", r["rmse_delta"])
    if _finite("max_residual", r["max_residual"]) < 0:
        raise ValueError(f"case {r['case_id']!r} max_residual must be non-negative")
    return r


def case_row_from_a0(case, a0):
    """Map a build_a0() result dict to a CASE_FIELDS row (pure). This is the reusable core of
    a run_one loader — the part that turns model output into the promotion contract, and the
    part most likely to drift. What still needs real multi-case DATA is only the case ->
    forcing/obs mapping that build_a0 is run on; this extraction is settled and testable now.
    Returns a row already passed through _validate_case_row (raises on any inconsistency)."""
    for k in ("case_id", "regime"):
        if k not in case:
            raise ValueError(f"case missing {k!r}")
    try:
        gate_pass = _strict_bool("a0.gate_da_vs_bg[0]", a0["gate_da_vs_bg"][0])
        physics_worse = _strict_bool("a0.physics_worse", a0["physics_worse"])
        dx_l2 = _finite("a0.dx_l2", a0["dx_l2"])
        dx_max_abs = _finite("a0.dx_max_abs", a0["dx_max_abs"])
        rmse_delta = _finite("a0.rmse_delta_da_minus_bg", a0["rmse_delta_da_minus_bg"])
        # each residual validated separately — max(0.0, nan) can return 0.0 and hide a NaN.
        bg_res = _finite("a0.bg residual", a0["bg"][1]["max_primary_residual"])
        da_res = _finite("a0.da residual", a0["da"][1]["max_primary_residual"])
    except (KeyError, TypeError, IndexError) as e:
        raise ValueError(f"malformed A0 result for case {case.get('case_id')!r}: {e}") from e
    if bg_res < 0.0 or da_res < 0.0:
        raise ValueError(f"case {case['case_id']!r} A0 residuals must be non-negative")
    row = {"case_id": case["case_id"], "regime": case["regime"],
           "gate_pass": gate_pass, "physics_worse": physics_worse,
           "state_large": bool(dx_max_abs > 2.0 or dx_l2 > 3.0),
           "rmse_delta": rmse_delta, "max_residual": max(bg_res, da_res)}
    return _validate_case_row(row)


def summarize_cases(rows, residual_atol=1e-9):
    """Aggregate per-case results and call promotion_gate with the REAL n_cases (pure).

    Unlike the single-fixture multi-window tool (pinned n_cases=1 -> always REPORT_ONLY),
    here n_cases is the number of independent cases, so a robust result can PROMOTE."""
    rows = [_validate_case_row(r) for r in rows]
    n = len(rows)
    if n == 0:
        return {"n_cases": 0, "promotion": ("REPORT_ONLY", ["no cases"])}
    ids = [r["case_id"] for r in rows]
    if len(set(ids)) != n:                      # n_cases must be UNIQUE independent cases
        raise ValueError("duplicate case_id in case results (n_cases must be unique)")
    wins = sum(r["gate_pass"] for r in rows)
    all_beat = wins == n
    max_resid = max(r["max_residual"] for r in rows)
    residual_clean = bool(max_resid <= residual_atol)
    verdict, reasons = promotion_gate(n_cases=n, windows_beat_baseline=all_beat,
                                      residual_atol=residual_atol)
    if not residual_clean:                      # aggregate audit blocks promotion regardless
        reasons = list(reasons) + [f"aggregate residual {max_resid:.3e} > {residual_atol:.0e}"]
        verdict = "REPORT_ONLY"
    return {"n_cases": n, "cases_beat_baseline": wins, "all_beat": all_beat,
            "physics_worse_n": sum(r["physics_worse"] for r in rows),
            "physics_worse_rate": sum(r["physics_worse"] for r in rows) / n,
            "state_large_n": sum(r["state_large"] for r in rows),
            "mean_rmse_delta": sum(r["rmse_delta"] for r in rows) / n,
            "worst_rmse_delta": max(r["rmse_delta"] for r in rows),
            "max_residual": max_resid, "residual_clean": residual_clean,
            "promotion": (verdict, reasons)}


def _run_one_not_implemented(case, setting):
    raise NotImplementedError(
        "per-case forcing/obs loader not implemented yet (only one fixture exists). "
        "Inject run_one=... with a loader that runs A0 for this case and returns a CASE_FIELDS row.")


def run_manifest(manifest, setting, run_one=_run_one_not_implemented, require="minimum"):
    """Validate the manifest, run `run_one(case, setting)` for each case, aggregate.
    Refuses to run unless the manifest is schema-valid and meets the required evidence tier
    (so promotion can't be attempted on a too-thin/invalid base). Returns (summary, rows)."""
    if require not in ("minimum", "recommended"):
        raise ValueError("require must be 'minimum' or 'recommended'")
    rep = validate_manifest(manifest)
    if not rep["ok"]:
        raise ValueError(f"invalid manifest: {rep['errors']}")
    if require == "minimum" and not rep["minimum_evidence_ready"]:
        raise ValueError(f"manifest not minimum_evidence_ready: {rep['minimum_reasons']}")
    if require == "recommended" and not rep["recommended_promotion_ready"]:
        raise ValueError(f"manifest not recommended_promotion_ready: {rep['recommended_reasons']}")
    rows = []
    for c in manifest["cases"]:
        row = run_one(c, setting)
        # loader safety belt: a row must describe the case it was asked to run.
        if not isinstance(row, dict) or row.get("case_id") != c["case_id"]:
            raise ValueError(f"run_one returned a row for {row.get('case_id') if isinstance(row, dict) else '?'!r}, "
                             f"expected {c['case_id']!r}")
        if row.get("regime") != c["regime"]:
            raise ValueError(f"run_one row {c['case_id']!r} regime {row.get('regime')!r} "
                             f"!= manifest {c['regime']!r}")
        rows.append(row)
    return summarize_cases(rows), rows


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Run a fixed A0 setting over a case manifest")
    ap.add_argument("manifest")
    ap.add_argument("--bg-w", type=float, required=True, dest="bg_w")
    ap.add_argument("--window", type=int, required=True)
    ap.add_argument("--lead", type=int, required=True)
    ap.add_argument("--require", choices=("minimum", "recommended"), default="minimum")
    args = ap.parse_args(argv)
    try:                                         # bad setting / unreadable manifest / thin tier
        import yaml
        try:
            text = Path(args.manifest).read_text()
        except OSError as e:
            raise ValueError(f"cannot read manifest: {e}") from e
        try:
            manifest = yaml.safe_load(text)
        except yaml.YAMLError as e:
            raise ValueError(f"cannot parse manifest YAML: {e}") from e
        setting = make_setting(args.bg_w, args.window, args.lead)
        summary, _ = run_manifest(manifest, setting, require=args.require)
    except NotImplementedError as e:            # per-case loader not wired yet (skeleton)
        print(f"ERROR {e}", file=sys.stderr)
        return 2
    except ValueError as e:                      # invalid manifest / thin tier / bad row / setting
        print(f"ERROR {e}", file=sys.stderr)
        return 1
    verdict, reasons = summary["promotion"]
    print(f"n_cases={summary['n_cases']} all_beat={summary['all_beat']} "
          f"physics_worse_rate={summary['physics_worse_rate']:.2f} "
          f"residual_clean={summary['residual_clean']}")
    print(f"promotion: {verdict} — {'; '.join(reasons)}")
    return 0 if verdict == "PROMOTE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
