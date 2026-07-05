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

from collections import Counter

from .ledger import DIAGNOSTIC_CODES, LedgerError, StorageLedger

_STORAGES = ("Snow", "Water", "Ice", "Ice2", "Dep")
_NEG_PRE_CLAMP = ("snow_negative_pre_clamp", "ice_negative_pre_clamp",
                  "deposit_negative_pre_clamp", "water_negative_pre_clamp")
_OVER_MELT = ("snow_over_melt", "ice_over_melt")
_OVERFLOW = ("snow_overflow", "ice_overflow", "deposit_overflow", "water_overflow")


def _max_storage_jump(out) -> float:
    """Largest |x[i+1]-x[i]| across the 5 storage trajectories (0.0 if absent)."""
    jump = 0.0
    for k in _STORAGES:
        seq = out.get(k)
        if seq is None:
            continue
        for i in range(1, len(seq)):
            d = abs(float(seq[i]) - float(seq[i - 1]))
            if d > jump:
                jump = d
    return jump


def deviation_budget(out, case_id: str = "case") -> dict:
    """Aggregate one full_rollout(return_ledger=True) result into a budget dict.

    Requires the audit keys ("ledger", "diagnostics"); raises LedgerError otherwise.
    """
    for key in ("ledger", "diagnostics"):
        if key not in out:
            raise LedgerError(f"deviation_budget needs '{key}' — run full_rollout(return_ledger=True)")
    ledgers = out["ledger"]
    diag_per_step = out["diagnostics"]
    if len(ledgers) != len(diag_per_step):
        raise LedgerError("ledger and diagnostics lengths differ")

    n_steps = len(ledgers)
    max_resid = 0.0
    for lg in ledgers:
        if not isinstance(lg, StorageLedger):
            raise LedgerError("ledger entries must be StorageLedger")
        r = abs(lg.primary_mass_residual)
        if r > max_resid:
            max_resid = r

    counts = Counter()
    diagnostic_steps = 0
    for step in diag_per_step:
        if step:
            diagnostic_steps += 1
        counts.update(step)
    unknown = set(counts) - DIAGNOSTIC_CODES
    if unknown:
        raise LedgerError(f"unknown diagnostic codes in rollout: {sorted(unknown, key=str)}")

    per_code = {code: int(counts.get(code, 0)) for code in sorted(DIAGNOSTIC_CODES)}
    return {
        "case_id": case_id,
        "n_steps": n_steps,
        "max_primary_residual": max_resid,
        "n_diagnostics_total": int(sum(counts.values())),
        "diagnostic_steps": diagnostic_steps,
        "diagnostic_steps_rate": (diagnostic_steps / n_steps) if n_steps else 0.0,
        "over_melt_count": sum(per_code[c] for c in _OVER_MELT),
        "overflow_count": sum(per_code[c] for c in _OVERFLOW),
        "negative_pre_clamp_count": sum(per_code[c] for c in _NEG_PRE_CLAMP),
        "max_storage_jump": _max_storage_jump(out),
        "counts": per_code,
    }


def accounting_gate(summary: dict, residual_atol: float = 1e-9) -> tuple[bool, list[str]]:
    """P0 accounting gate. Diagnostics are NOT failures — only the mass residual
    and (structurally-guaranteed) code validity gate here. Returns (passed, reasons)."""
    reasons = []
    if summary["max_primary_residual"] > residual_atol:
        reasons.append(
            f"max_primary_residual {summary['max_primary_residual']:.3e} > {residual_atol:.0e}")
    return (not reasons), reasons


_COLUMNS = ("case_id", "n_steps", "max_primary_residual", "n_diagnostics_total",
            "diagnostic_steps_rate", "over_melt_count", "overflow_count",
            "negative_pre_clamp_count", "max_storage_jump")


def budget_to_csv(summaries) -> str:
    """CSV (flat columns) for one or many case summaries."""
    rows = ["\t".join(_COLUMNS).replace("\t", ",")]
    for s in summaries:
        rows.append(",".join(str(s[c]) for c in _COLUMNS))
    return "\n".join(rows) + "\n"


def budget_to_markdown(summaries, title: str = "Deviation Budget") -> str:
    """Markdown report: a table plus a P0 gate line per case."""
    head = "| " + " | ".join(_COLUMNS) + " |"
    sep = "| " + " | ".join("---" for _ in _COLUMNS) + " |"
    lines = [f"# {title}", "",
             "residual = 코드 누출 게이트(P0, ~0 필수). diagnostics = 물리/deviation 신호(카운트, 실패 아님).",
             "", head, sep]
    for s in summaries:
        lines.append("| " + " | ".join(
            (f"{s[c]:.3e}" if c == "max_primary_residual"
             else f"{s[c]:.4f}" if c in ("diagnostic_steps_rate", "max_storage_jump")
             else str(s[c])) for c in _COLUMNS) + " |")
    lines += ["", "## P0 accounting gate"]
    for s in summaries:
        ok, reasons = accounting_gate(s)
        lines.append(f"- {s['case_id']}: {'PASS' if ok else 'FAIL — ' + '; '.join(reasons)}")
    return "\n".join(lines) + "\n"
