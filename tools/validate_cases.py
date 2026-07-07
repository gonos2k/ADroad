#!/usr/bin/env python3
"""Validate a dROAD independent-case manifest (Step 4 toward real promotion).

promotion_gate (design §11) counts a CASE = independent station/day, not a window. This
turns the case manifest into an executable contract: it checks each case's schema/semantics
and reports whether the manifest has enough DISTINCT, regime-diverse cases to even attempt
promotion (n_cases ≥ MIN_CASES, ≥ 2 regimes, no duplicate station-days). It never claims a
model is promotable — only whether the EVIDENCE BASE is large enough to ask the question.

    python3 tools/validate_cases.py cases.yaml        # or cases.example.yaml

Pure validate_manifest(dict) has no yaml dependency (tests pass dicts directly).
"""
from collections import defaultdict
import sys
from datetime import datetime
from pathlib import Path

REGIMES = {"dry_cold", "warm_wet", "freeze_transition", "precip_snow", "melt_refreeze"}
REQUIRED = ("case_id", "station", "start", "end", "regime")
OPTIONAL = ("forcing_source", "obs_source", "notes")
NONEMPTY_OPTIONAL = ("forcing_source", "obs_source")   # notes may be an empty string
# two readiness tiers so the name can't be mistaken for "the model is promotable":
MIN_CASES, MIN_REGIMES = 3, 2               # minimum_evidence_ready (matches promotion_gate)
RECOMMENDED_CASES, RECOMMENDED_REGIMES = 9, 3   # recommended_promotion_ready (design target)


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
                "recommended_promotion_ready": False, "readiness_reasons": ["no cases"]}
    errors, ids, regimes, stations = [], set(), set(), set()
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
        ids.add(cid); regimes.add(c["regime"]); stations.add(c["station"])
        by_station[c["station"]].append((start, end, cid))
    errors += _overlap_errors(by_station)

    n_cases, n_reg = len(ids), len(regimes)
    reasons = []
    if errors:
        reasons.append("has errors")
    if n_cases < MIN_CASES:
        reasons.append(f"cases {n_cases} < minimum {MIN_CASES}")
    if n_reg < MIN_REGIMES:
        reasons.append(f"regimes {n_reg} < minimum {MIN_REGIMES}")
    min_ready = not reasons
    rec_reasons = list(reasons)
    if n_cases < RECOMMENDED_CASES:
        rec_reasons.append(f"cases {n_cases} < recommended {RECOMMENDED_CASES}")
    if n_reg < RECOMMENDED_REGIMES:
        rec_reasons.append(f"regimes {n_reg} < recommended {RECOMMENDED_REGIMES}")
    return {"ok": not errors, "errors": errors, "n_cases": n_cases,
            "n_stations": len(stations), "n_regimes": n_reg, "regimes": sorted(regimes),
            "minimum_evidence_ready": min_ready,
            "recommended_promotion_ready": not rec_reasons,
            "readiness_reasons": rec_reasons if rec_reasons else ["meets recommended target"]}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: validate_cases.py <manifest.yaml>", file=sys.stderr)
        return 2
    import yaml                                          # only the CLI needs yaml
    manifest = yaml.safe_load(Path(argv[0]).read_text())
    rep = validate_manifest(manifest)
    print(f"cases={rep['n_cases']} stations={rep.get('n_stations')} regimes={rep.get('regimes')}")
    for e in rep["errors"]:
        print(f"  ERROR {e}")
    print(f"minimum_evidence_ready: {rep['minimum_evidence_ready']} · "
          f"recommended_promotion_ready: {rep['recommended_promotion_ready']} "
          f"(evidence base only, NOT model promotability) — {'; '.join(rep['readiness_reasons'])}")
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
