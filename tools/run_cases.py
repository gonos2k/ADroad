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
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from droad.skill_gate import promotion_gate  # noqa: E402
from tools.validate_cases import validate_manifest  # noqa: E402

# case row fields a run_one implementation must return (per case):
CASE_FIELDS = ("case_id", "regime", "gate_pass", "physics_worse", "residual_clean",
               "state_large", "rmse_delta", "max_residual")


def make_setting(bg_w, window, lead):
    """A frozen A0 hyperparameter setting (chosen from the grid stability region)."""
    for name, v in (("bg_w", bg_w), ("window", window), ("lead", lead)):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"setting {name} must be numeric")
    if not (bg_w > 0 and window > 0 and lead > 0):
        raise ValueError("setting bg_w/window/lead must be positive")
    return {"bg_w": float(bg_w), "window": int(window), "lead": int(lead)}


def _validate_case_row(r):
    if not isinstance(r, dict) or set(CASE_FIELDS) - set(r):
        raise ValueError(f"case result must have fields {CASE_FIELDS}")
    for k in ("gate_pass", "physics_worse", "residual_clean", "state_large"):
        if not isinstance(r[k], bool):
            raise ValueError(f"case result {k!r} must be bool")
    return r


def summarize_cases(rows, residual_atol=1e-9):
    """Aggregate per-case results and call promotion_gate with the REAL n_cases (pure).

    Unlike the single-fixture multi-window tool (pinned n_cases=1 -> always REPORT_ONLY),
    here n_cases is the number of independent cases, so a robust result can PROMOTE."""
    rows = [_validate_case_row(r) for r in rows]
    n = len(rows)
    if n == 0:
        return {"n_cases": 0, "promotion": ("REPORT_ONLY", ["no cases"])}
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
    rep = validate_manifest(manifest)
    if not rep["ok"]:
        raise ValueError(f"invalid manifest: {rep['errors']}")
    if require == "minimum" and not rep["minimum_evidence_ready"]:
        raise ValueError(f"manifest not minimum_evidence_ready: {rep['minimum_reasons']}")
    if require == "recommended" and not rep["recommended_promotion_ready"]:
        raise ValueError(f"manifest not recommended_promotion_ready: {rep['recommended_reasons']}")
    rows = [run_one(c, setting) for c in manifest["cases"]]
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
    import yaml
    manifest = yaml.safe_load(Path(args.manifest).read_text())
    setting = make_setting(args.bg_w, args.window, args.lead)
    summary, _ = run_manifest(manifest, setting, require=args.require)   # will raise until loader wired
    verdict, reasons = summary["promotion"]
    print(f"n_cases={summary['n_cases']} all_beat={summary['all_beat']} "
          f"physics_worse_rate={summary['physics_worse_rate']:.2f} "
          f"residual_clean={summary['residual_clean']}")
    print(f"promotion: {verdict} — {'; '.join(reasons)}")
    return 0 if verdict == "PROMOTE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
