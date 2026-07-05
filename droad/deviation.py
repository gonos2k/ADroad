"""Deviation budget — aggregate a full_rollout audit trail into quality metrics.

Consumes `full_rollout(return_ledger=True)` output (per-step merged ledger +
diagnostics + storage trajectories) and turns it into a per-case report:

  - residual (accounting): max |primary_mass_residual| over the rollout. This is
    the P0 gate — it must stay ~0 (a non-zero value is a code leak, not physics).
  - diagnostics (physics/deviation): how often, and which, reference quirks /
    hard-projection hits fire. These are COUNTED, never treated as failures —
    consistent with the residual↔diagnostics split (droad.ledger).
  - max_storage_jump: largest single-step change across the 5 storages, a proxy
    for numerically abrupt transitions.

The point is to quantify "exact parity is kept, but how often do reference quirks
occur?" so smooth_compat / DA runs can be compared on physics burden, not just RMSE.
"""

from __future__ import annotations

import csv
import io
import math
from collections import Counter
from collections.abc import Mapping as ABCMapping

from .ledger import (
    DIAGNOSTIC_CODES, LedgerError, StorageLedger,
    as_finite_float, normalize_diagnostics,
)

_STORAGES = ("Snow", "Water", "Ice", "Ice2", "Dep")
_NEG_PRE_CLAMP = ("snow_negative_pre_clamp", "ice_negative_pre_clamp",
                  "deposit_negative_pre_clamp", "water_negative_pre_clamp")
_OVER_MELT = ("snow_over_melt", "ice_over_melt")
_OVERFLOW = ("snow_overflow", "ice_overflow", "deposit_overflow", "water_overflow")


def _max_storage_jump(out, n_steps: int) -> float:
    """Largest |x[i+1]-x[i]| across the 5 storage trajectories (0.0 if all absent).

    Each present trajectory must be a sized sequence of length n_steps with only
    finite scalar values — otherwise the jump would be computed on a mismatched
    timeline or silently skip a NaN, so both fail with LedgerError.
    """
    jump = 0.0
    for k in _STORAGES:
        seq = out.get(k)
        if seq is None:
            continue
        if isinstance(seq, (str, bytes, ABCMapping, set, frozenset)):
            raise LedgerError(f"{k} trajectory must be an ordered numeric sequence")
        try:
            L = len(seq)
        except TypeError:
            raise LedgerError(f"{k} trajectory must be a sized sequence") from None
        if L != n_steps:
            raise LedgerError(f"{k} length {L} != n_steps {n_steps}")
        vals = []
        for i, x in enumerate(seq):          # validate every value, incl. length-1
            try:
                v = float(x)
            except (TypeError, ValueError):
                raise LedgerError(f"{k} has non-scalar storage value at step {i}") from None
            if not math.isfinite(v):
                raise LedgerError(f"{k} has non-finite storage value at step {i}")
            vals.append(v)
        for i in range(1, len(vals)):
            d = abs(vals[i] - vals[i - 1])
            if d > jump:
                jump = d
    return jump


def deviation_budget(out, case_id: str = "case") -> dict:
    """Aggregate one full_rollout(return_ledger=True) result into a budget dict.

    Requires the audit keys ("ledger", "diagnostics"); raises LedgerError otherwise.
    """
    if not isinstance(out, ABCMapping):
        raise LedgerError("deviation_budget input must be a full_rollout(return_ledger=True) mapping")
    for key in ("ledger", "diagnostics"):
        if key not in out:
            raise LedgerError(f"deviation_budget needs '{key}' — run full_rollout(return_ledger=True)")
    ledgers = out["ledger"]
    diag_per_step = out["diagnostics"]
    # both must be ORDERED per-step sequences: a mapping/set/str is sized+iterable
    # and would pass len()/iteration on the wrong thing (keys, chars, unordered).
    for nm, seq in (("ledger", ledgers), ("diagnostics", diag_per_step)):
        if isinstance(seq, (ABCMapping, set, frozenset, str, bytes)):
            raise LedgerError(f"{nm} must be an ordered per-step sequence, not {type(seq).__name__}")
    try:
        n_steps = len(ledgers)
        lengths_ok = n_steps == len(diag_per_step)
    except TypeError:
        raise LedgerError("ledger and diagnostics must be sized sequences") from None
    if not lengths_ok:
        raise LedgerError("ledger and diagnostics lengths differ")
    if n_steps == 0:                        # empty rollout = "cannot evaluate", not PASS
        raise LedgerError("deviation_budget requires at least one step")
    max_resid = 0.0
    for lg in ledgers:
        if not isinstance(lg, StorageLedger):
            raise LedgerError("ledger entries must be StorageLedger")
        r = abs(lg.primary_mass_residual)
        if r > max_resid:
            max_resid = r

    counts = Counter()
    diagnostic_steps = 0
    for step in diag_per_step:              # same validation as StorageResult/rollout_audit
        codes = normalize_diagnostics(step)    # None/str/mapping/set/unknown -> LedgerError
        if codes:
            diagnostic_steps += 1
        counts.update(codes)

    per_code = {code: int(counts.get(code, 0)) for code in sorted(DIAGNOSTIC_CODES)}
    return {
        "case_id": str(case_id),                # normalize report identifier
        "n_steps": n_steps,
        "max_primary_residual": max_resid,
        "n_diagnostics_total": int(sum(counts.values())),
        "diagnostic_steps": diagnostic_steps,
        "diagnostic_steps_rate": (diagnostic_steps / n_steps) if n_steps else 0.0,
        "over_melt_count": sum(per_code[c] for c in _OVER_MELT),
        "overflow_count": sum(per_code[c] for c in _OVERFLOW),
        "negative_pre_clamp_count": sum(per_code[c] for c in _NEG_PRE_CLAMP),
        "max_storage_jump": _max_storage_jump(out, n_steps),
        "counts": per_code,
    }


def accounting_gate(summary: dict, residual_atol: float = 1e-9) -> tuple[bool, list[str]]:
    """P0 accounting gate. Diagnostics are NOT failures — only the mass residual
    and (structurally-guaranteed) code validity gate here. Returns (passed, reasons)."""
    if not isinstance(summary, ABCMapping):
        raise LedgerError("summary must be a mapping from deviation_budget")
    if "max_primary_residual" not in summary:
        raise LedgerError("summary missing max_primary_residual")
    residual_atol = as_finite_float("residual_atol", residual_atol)   # NaN would false-PASS
    if residual_atol < 0.0:
        raise LedgerError("residual_atol must be non-negative")
    max_resid = as_finite_float("summary.max_primary_residual", summary["max_primary_residual"])
    if max_resid < 0.0:
        raise LedgerError("summary.max_primary_residual must be non-negative")
    reasons = []
    if max_resid > residual_atol:
        reasons.append(f"max_primary_residual {max_resid:.3e} > {residual_atol:.0e}")
    return (not reasons), reasons


_COLUMNS = ("case_id", "n_steps", "max_primary_residual", "n_diagnostics_total",
            "diagnostic_steps_rate", "over_melt_count", "overflow_count",
            "negative_pre_clamp_count", "max_storage_jump")


def _require_columns(s):
    if not isinstance(s, ABCMapping):
        raise LedgerError("summary must be a mapping")
    missing = set(_COLUMNS) - set(s)
    if missing:
        raise LedgerError(f"summary missing columns: {sorted(missing, key=str)}")


def _fmt(s, c):
    """Human-facing formatting (Markdown only). CSV keeps raw values."""
    if c == "max_primary_residual":
        return f"{s[c]:.3e}"
    if c in ("diagnostic_steps_rate", "max_storage_jump"):
        return f"{s[c]:.4f}"
    return str(s[c])


def budget_to_csv(summaries) -> str:
    """Machine-readable CSV: RAW values (full precision), via csv.writer so a
    case_id with a comma/newline is quoted rather than corrupting the row.
    Use budget_to_markdown for human-facing rounded values."""
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(_COLUMNS)
    for s in summaries:
        _require_columns(s)
        w.writerow([s[c] for c in _COLUMNS])       # raw, not _fmt -> no precision loss
    return buf.getvalue()


def _md_cell(x: str) -> str:
    return str(x).replace("|", "\\|").replace("\n", " ")


def budget_to_markdown(summaries, title: str = "Deviation Budget") -> str:
    """Markdown report: a table plus a P0 gate line per case. Cell values are
    escaped so a stray '|' or newline can't break the table."""
    head = "| " + " | ".join(_COLUMNS) + " |"
    sep = "| " + " | ".join("---" for _ in _COLUMNS) + " |"
    lines = [f"# {_md_cell(title)}", "",
             "residual = 코드 누출 게이트(P0, ~0 필수). diagnostics = 물리/deviation 신호(카운트, 실패 아님).",
             "", head, sep]
    for s in summaries:
        _require_columns(s)
        lines.append("| " + " | ".join(_md_cell(_fmt(s, c)) for c in _COLUMNS) + " |")
    lines += ["", "## P0 accounting gate"]
    for s in summaries:
        ok, reasons = accounting_gate(s)
        status = "PASS" if ok else "FAIL — " + "; ".join(reasons)
        lines.append(f"- {_md_cell(s['case_id'])}: {status}")
    return "\n".join(lines) + "\n"
