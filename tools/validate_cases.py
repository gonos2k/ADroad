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
import sys
from datetime import datetime
from pathlib import Path

REGIMES = {"dry_cold", "warm_wet", "freeze_transition", "precip_snow", "melt_refreeze"}
REQUIRED = ("case_id", "station", "start", "end", "regime")
OPTIONAL = ("forcing_source", "obs_source", "notes")
MIN_CASES = 3          # matches promotion_gate(min_cases=3)
MIN_REGIMES = 2        # a promotion built on a single regime is weak


def _parse_dt(s, field):
    if not isinstance(s, str):
        raise ValueError(f"{field} must be an ISO-8601 string, got {type(s).__name__}")
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        raise ValueError(f"{field} is not a valid ISO-8601 datetime: {s!r}")


def validate_case(c):
    """Validate one case mapping; raise ValueError with a specific reason. Returns the
    parsed (start, end) datetimes so the caller can derive the station-day key."""
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
    start, end = _parse_dt(c["start"], "start"), _parse_dt(c["end"], "end")
    if not start < end:
        raise ValueError(f"case {c['case_id']!r} must have start < end")
    unknown = set(c) - set(REQUIRED) - set(OPTIONAL)
    if unknown:
        raise ValueError(f"case {c['case_id']!r} has unknown fields: {sorted(unknown)}")
    return start, end


def validate_manifest(manifest):
    """Validate a whole manifest (dict with a 'cases' list). Never raises — returns a
    report {ok, errors, n_cases, n_stations, n_regimes, n_distinct_station_days, regimes,
    promotion_ready, promotion_reasons}. promotion_ready means the evidence base is big and
    diverse enough to attempt promotion — NOT that any model passed."""
    errors = []
    if not isinstance(manifest, dict) or not isinstance(manifest.get("cases"), list):
        return {"ok": False, "errors": ["manifest must be a mapping with a 'cases' list"],
                "n_cases": 0, "promotion_ready": False, "promotion_reasons": ["no cases"]}
    cases = manifest["cases"]
    ids, station_days, regimes, stations = set(), set(), set(), set()
    for i, c in enumerate(cases):
        try:
            start, _ = validate_case(c)
        except ValueError as e:
            errors.append(f"[case {i}] {e}")
            continue
        cid = c["case_id"]
        if cid in ids:
            errors.append(f"[case {i}] duplicate case_id: {cid!r}")
        ids.add(cid)
        station_days.add((c["station"], start.date().isoformat()))
        regimes.add(c["regime"]); stations.add(c["station"])

    n_cases = len(ids)
    reasons = []
    if len(station_days) < len(ids):
        reasons.append("duplicate station-day cases (not independent)")
    if n_cases < MIN_CASES:
        reasons.append(f"insufficient cases: {n_cases} < {MIN_CASES}")
    if len(regimes) < MIN_REGIMES:
        reasons.append(f"insufficient regime coverage: {len(regimes)} < {MIN_REGIMES}")
    promotion_ready = not errors and not reasons
    return {"ok": not errors, "errors": errors, "n_cases": n_cases,
            "n_stations": len(stations), "n_regimes": len(regimes),
            "n_distinct_station_days": len(station_days), "regimes": sorted(regimes),
            "promotion_ready": promotion_ready,
            "promotion_reasons": reasons if reasons else ["evidence base sufficient to attempt"]}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: validate_cases.py <manifest.yaml>", file=sys.stderr)
        return 2
    import yaml                                          # only the CLI needs yaml
    manifest = yaml.safe_load(Path(argv[0]).read_text())
    rep = validate_manifest(manifest)
    print(f"cases={rep['n_cases']} stations={rep.get('n_stations')} "
          f"regimes={rep.get('regimes')} distinct_station_days={rep.get('n_distinct_station_days')}")
    for e in rep["errors"]:
        print(f"  ERROR {e}")
    print(f"promotion_ready(evidence base only): {rep['promotion_ready']} "
          f"— {'; '.join(rep['promotion_reasons'])}")
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
