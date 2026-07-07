#!/usr/bin/env python3
"""Validate a dROAD independent-case manifest (Step 4 toward real promotion).

promotion_gate (design §11) counts a CASE = one station over one non-overlapping time
interval, not a window. This turns the case manifest into an executable contract: it checks
each case's schema/semantics and reports whether the manifest has enough DISTINCT,
regime-diverse cases to even attempt promotion (two readiness tiers; independence is enforced
by no overlapping same-station intervals). It never claims a model is promotable — only
whether the EVIDENCE BASE is large enough to ask the question.

    python3 tools/validate_cases.py cases.yaml                     # schema only (exit!=0 on error)
    python3 tools/validate_cases.py cases.yaml --require minimum   # also gate on minimum tier
    python3 tools/validate_cases.py cases.yaml --require recommended

Pure validate_manifest(dict) has no yaml dependency (tests pass dicts directly).
"""
from collections import Counter, defaultdict
import sys
from datetime import datetime
from pathlib import Path

REGIMES = {"dry_cold", "warm_wet", "freeze_transition", "precip_snow", "melt_refreeze"}
REQUIRED = ("case_id", "station", "start", "end", "regime")
OPTIONAL = ("forcing_source", "obs_source", "notes")
NONEMPTY_OPTIONAL = ("forcing_source", "obs_source")   # notes may be an empty string
# two readiness tiers so the name can't be mistaken for "the model is promotable":
MIN_CASES, MIN_REGIMES = 3, 2               # minimum_evidence_ready (matches promotion_gate)
# recommended = the doc's "3 regimes x 3 cases each" target, enforced literally (not just a
# 9-total that a lopsided [7,1,1] could game) since the finding is regime-dependent:
RECOMMENDED_REGIMES, RECOMMENDED_CASES_PER_REGIME = 3, 3


def _parse_dt(s, field):
    if not isinstance(s, str):
        raise ValueError(f"{field} must be an ISO-8601 string, got {type(s).__name__}")
    if "T" not in s:                        # require a time for reproducibility (no date-only)
        raise ValueError(f"{field} must include a time, e.g. 2026-01-01T00:00:00 (got {s!r})")
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        raise ValueError(f"{field} is not a valid ISO-8601 datetime: {s!r}")


def validate_case(c):
    """Validate one case mapping; raise ValueError with a specific reason. Returns the
    parsed (start, end) datetimes so the caller can check independence."""
    if not isinstance(c, dict):
        raise ValueError("case must be a mapping")
    missing = [k for k in REQUIRED if k not in c]
    if missing:
        raise ValueError(f"case missing required fields: {missing}")
    for k in REQUIRED:
        if not isinstance(c[k], str) or not c[k].strip():
            raise ValueError(f"case field {k!r} must be a non-empty string")
    if c["regime"] not in REGIMES:
        raise ValueError(f"case {c['case_id']!r} has unknown regime {c['regime']!r} "
                         f"(allowed: {sorted(REGIMES)})")
    for k in OPTIONAL:                      # if present, must be a string (typed, not null/int)
        if k in c and not isinstance(c[k], str):
            raise ValueError(f"case {c['case_id']!r} optional field {k!r} must be a string")
    for k in NONEMPTY_OPTIONAL:
        if k in c and not c[k].strip():
            raise ValueError(f"case {c['case_id']!r} optional field {k!r} must be non-empty")
    start, end = _parse_dt(c["start"], "start"), _parse_dt(c["end"], "end")
    if (start.tzinfo is None) != (end.tzinfo is None):   # mixed tz would raise on compare
        raise ValueError(f"case {c['case_id']!r} start/end timezone-awareness must match")
    try:
        ordered = start < end
    except TypeError as e:                  # keep validate_manifest's 'never raises' contract
        raise ValueError(f"case {c['case_id']!r} has incomparable start/end datetimes") from e
    if not ordered:
        raise ValueError(f"case {c['case_id']!r} must have start < end")
    unknown = set(c) - set(REQUIRED) - set(OPTIONAL)
    if unknown:
        raise ValueError(f"case {c['case_id']!r} has unknown fields: {sorted(unknown)}")
    return start, end


def _overlap_errors(by_station):
    """Two cases at the SAME station whose [start,end) intervals overlap are not independent.
    Guards against mixed-tz comparisons so it can't raise (would break 'never raises')."""
    errs = []
    for station, ivals in by_station.items():
        try:
            ivals = sorted(ivals, key=lambda t: t[0])
        except TypeError:
            errs.append(f"station {station!r}: cases have incomparable datetimes (mixed tz)")
            continue
        for (_, end_prev, id_prev), (start_next, _, id_next) in zip(ivals, ivals[1:]):
            if start_next < end_prev:
                errs.append(f"overlapping cases at station {station!r}: "
                            f"{id_prev!r} and {id_next!r} (not independent)")
    return errs


def validate_manifest(manifest):
    """Validate a whole manifest (dict with a 'cases' list). Never raises — returns a report:
    {ok, errors, n_cases, n_stations, n_regimes, regimes, minimum_evidence_ready,
    recommended_promotion_ready, readiness_reasons}.

    The readiness flags describe the EVIDENCE BASE (how many distinct, non-overlapping,
    regime-diverse cases exist), NOT whether any model is promotable — skill/physics still
    have to beat baseline in every case at run time."""
    if not isinstance(manifest, dict) or not isinstance(manifest.get("cases"), list):
        return {"ok": False, "errors": ["manifest must be a mapping with a 'cases' list"],
                "n_cases": 0, "minimum_evidence_ready": False,
                "recommended_promotion_ready": False,
                "minimum_reasons": ["no cases"], "recommended_reasons": ["no cases"]}
    errors, ids, stations = [], set(), set()
    regime_counts = Counter()
    by_station = defaultdict(list)
    for i, c in enumerate(manifest["cases"]):
        try:
            start, end = validate_case(c)
        except ValueError as e:
            errors.append(f"[case {i}] {e}")
            continue
        cid = c["case_id"]
        if cid in ids:
            errors.append(f"[case {i}] duplicate case_id: {cid!r}")
        ids.add(cid); regime_counts[c["regime"]] += 1; stations.add(c["station"])
        by_station[c["station"]].append((start, end, cid))
    errors += _overlap_errors(by_station)

    n_cases, n_reg = len(ids), len(regime_counts)
    # minimum and recommended reasons are tracked SEPARATELY so a minimum-ready manifest
    # doesn't confusingly surface recommended-only failures as if it failed.
    min_reasons = ["has errors"] if errors else []
    if n_cases < MIN_CASES:
        min_reasons.append(f"cases {n_cases} < minimum {MIN_CASES}")
    if n_reg < MIN_REGIMES:
        min_reasons.append(f"regimes {n_reg} < minimum {MIN_REGIMES}")
    rec_reasons = list(min_reasons)            # recommended presupposes minimum
    if n_reg < RECOMMENDED_REGIMES:
        rec_reasons.append(f"regimes {n_reg} < recommended {RECOMMENDED_REGIMES}")
    thin = {r: n for r, n in regime_counts.items() if n < RECOMMENDED_CASES_PER_REGIME}
    if thin:                                   # every covered regime needs >=3 cases (not just 9 total)
        rec_reasons.append(f"recommended needs >= {RECOMMENDED_CASES_PER_REGIME} cases per regime; "
                           f"under-covered: {dict(sorted(thin.items()))}")
    return {"ok": not errors, "errors": errors,
            "n_rows": len(manifest["cases"]), "n_cases": n_cases,
            "n_stations": len(stations), "n_regimes": n_reg,
            "regimes": sorted(regime_counts), "regime_counts": dict(sorted(regime_counts.items())),
            "minimum_evidence_ready": not min_reasons,
            "recommended_promotion_ready": not rec_reasons,
            "minimum_reasons": min_reasons or ["meets minimum target"],
            "recommended_reasons": rec_reasons or ["meets recommended target"]}


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Validate a dROAD independent-case manifest")
    ap.add_argument("manifest", help="path to cases.yaml")
    ap.add_argument("--require", choices=("minimum", "recommended"), default=None,
                    help="also fail (exit!=0) if this evidence-base tier is not met")
    args = ap.parse_args(argv)
    import yaml                                          # only the CLI needs yaml
    manifest = yaml.safe_load(Path(args.manifest).read_text())
    rep = validate_manifest(manifest)
    print(f"rows={rep.get('n_rows')} cases={rep['n_cases']} stations={rep.get('n_stations')} "
          f"regimes={rep.get('regimes')}")
    print(f"regime_counts={rep.get('regime_counts')}")
    for e in rep["errors"]:
        print(f"  ERROR {e}")
    print("(readiness = evidence base only, NOT model promotability)")
    print(f"minimum_evidence_ready: {rep['minimum_evidence_ready']} — "
          f"{'; '.join(rep['minimum_reasons'])}")
    print(f"recommended_promotion_ready: {rep['recommended_promotion_ready']} — "
          f"{'; '.join(rep['recommended_reasons'])}")
    if not rep["ok"]:
        return 1
    if args.require == "minimum" and not rep["minimum_evidence_ready"]:
        return 1
    if args.require == "recommended" and not rep["recommended_promotion_ready"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
