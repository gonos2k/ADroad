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


def _validate_storage_container(k: str, seq, expected_len: int) -> None:
    """A storage trajectory must be an ORDERED, sized numeric sequence of the given
    length. Rejecting mapping/str/set here (before any integer indexing) is what stops
    a dict like {0:.., 1:..} from masquerading as a trajectory on the steps= path."""
    if isinstance(seq, (str, bytes, ABCMapping, set, frozenset)):
        raise LedgerError(f"{k} trajectory must be an ordered numeric sequence")
    try:
        L = len(seq)
    except TypeError:
        raise LedgerError(f"{k} trajectory must be a sized sequence") from None
    if L != expected_len:
        raise LedgerError(f"{k} length {L} != expected {expected_len}")


def _max_storage_jump(store, n_steps: int, step_ids=None) -> dict:
    """Largest |x[i+1]-x[i]| across the 5 storage trajectories (0.0 if all absent).

    Each present trajectory must be a sized sequence of length n_steps with only
    finite scalar values — otherwise the jump would be computed on a mismatched
    timeline or silently skip a NaN, so both fail with LedgerError.

    `step_ids` maps local position -> ORIGINAL rollout step so the reported
    max_storage_jump_step is the true rollout index even on a sliced (steps=) window
    (None means local index == original, i.e. the full run).
    """
    jump = 0.0
    key, step, signed = "", -1, 0.0          # provenance of the max jump (which storage/step/direction)
    for k in _STORAGES:
        seq = store.get(k)
        if seq is None:
            continue
        _validate_storage_container(k, seq, n_steps)
        # same numeric policy as ledger fields: rejects str/bool/non-scalar/NaN/Inf
        vals = [as_finite_float(f"{k}[{i}]", x) for i, x in enumerate(seq)]
        for i in range(1, len(vals)):
            d = vals[i] - vals[i - 1]
            if abs(d) > jump:
                orig = step_ids[i] if step_ids is not None else i   # report ORIGINAL rollout step
                jump, key, step, signed = abs(d), k, orig, d
    return {"max_storage_jump": jump, "max_storage_jump_key": key,
            "max_storage_jump_step": step, "max_storage_jump_signed": signed}


def _as_step_index(name: str, x) -> int:
    """One step index: a WHOLE number, not a bool (True->1 would corrupt a window)
    and not a float that silently truncates (1.9->1). Reuses as_finite_float so the
    str/Python-bool/numpy-bool/NaN/Inf policy is IDENTICAL to the ledger/deviation
    numeric contract — then adds the integer-only requirement on top."""
    v = as_finite_float(name, x)              # rejects str/bool/np.bool_/NaN/Inf
    if not v.is_integer():
        raise LedgerError(f"{name} must be a whole-number index, got {x!r}")
    return int(v)


def _resolve_steps(steps, full_n: int) -> list:
    """Validate `steps` into a list of in-range int indices (for holdout-window budgets).

    A holdout skill gate scores only obs-valid steps, so its physics burden should be
    aggregated on the SAME window — not the full run. steps must be a STRICTLY
    INCREASING sequence of whole-number indices in [0, full_n): duplicates would
    double-count diagnostics/jumps and reversal would break the 'interval' meaning."""
    if isinstance(steps, (ABCMapping, set, frozenset, str, bytes)):
        raise LedgerError(f"steps must be an ordered index sequence, not {type(steps).__name__}")
    try:
        raw = list(steps)
    except TypeError:
        raise LedgerError("steps must be an iterable of integer indices") from None
    idx = [_as_step_index(f"steps[{j}]", i) for j, i in enumerate(raw)]
    if not idx:
        raise LedgerError("steps selects zero rollout steps — cannot evaluate")
    for i in idx:
        if not (0 <= i < full_n):
            raise LedgerError(f"steps index {i} out of range [0, {full_n})")
    if any(b <= a for a, b in zip(idx, idx[1:])):
        raise LedgerError("steps must be strictly increasing (no duplicates or reversal)")
    return idx


def deviation_budget(out, case_id: str = "case", *, steps=None) -> dict:
    """Aggregate one full_rollout(return_ledger=True) result into a budget dict.

    Requires the audit keys ("ledger", "diagnostics"); raises LedgerError otherwise.
    When `steps` is given (an index sequence), the budget is aggregated over ONLY those
    rollout steps — used to align diagnostics with a holdout forecast window so a
    skill gate compares like-for-like physics burden (not full-run vs holdout skill).
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
        full_n = len(ledgers)
        lengths_ok = full_n == len(diag_per_step)
    except TypeError:
        raise LedgerError("ledger and diagnostics must be sized sequences") from None
    if not lengths_ok:
        raise LedgerError("ledger and diagnostics lengths differ")
    if full_n == 0:                         # empty rollout = "cannot evaluate", not PASS
        raise LedgerError("deviation_budget requires at least one step")

    if steps is None:
        sel = range(full_n)
        store_src = out                     # full-run: _max_storage_jump reads out directly
        step_ids = None                     # local index already == original rollout step
    else:
        sel = _resolve_steps(steps, full_n)
        step_ids = sel                      # map local jump position -> original rollout step
        # validate ORIGINAL storage containers (against full length) BEFORE integer
        # indexing — otherwise a dict/str trajectory would be silently sliced into a
        # list and bypass the ordered-sequence check _max_storage_jump relies on.
        for k in _STORAGES:
            if k in out:
                _validate_storage_container(k, out[k], full_n)
        # slice storage trajectories to the window; jump is between consecutive SELECTED steps
        store_src = {k: [out[k][i] for i in sel] for k in _STORAGES if k in out}
    ledgers = [ledgers[i] for i in sel]
    diag_per_step = [diag_per_step[i] for i in sel]
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
    for step in diag_per_step:              # same validation as StorageResult/rollout_audit
        codes = normalize_diagnostics(step)    # None/str/mapping/set/unknown -> LedgerError
        if codes:
            diagnostic_steps += 1
        counts.update(codes)

    per_code = {code: int(counts.get(code, 0)) for code in sorted(DIAGNOSTIC_CODES)}
    summary = {
        "case_id": str(case_id),                # normalize report identifier
        "n_steps": n_steps,
        "max_primary_residual": max_resid,
        "n_diagnostics_total": int(sum(counts.values())),
        "diagnostic_steps": diagnostic_steps,
        "diagnostic_steps_rate": (diagnostic_steps / n_steps) if n_steps else 0.0,
        "over_melt_count": sum(per_code[c] for c in _OVER_MELT),
        "overflow_count": sum(per_code[c] for c in _OVERFLOW),
        "negative_pre_clamp_count": sum(per_code[c] for c in _NEG_PRE_CLAMP),
        "counts": per_code,
    }
    summary.update(_max_storage_jump(store_src, n_steps, step_ids))   # jump + original-step provenance
    return summary


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
            "negative_pre_clamp_count", "max_storage_jump",
            "max_storage_jump_key", "max_storage_jump_step", "max_storage_jump_signed")

# per-diagnostic-code breakdown columns (machine-readable CSV appends these)
_DIAG_COLUMNS = tuple(f"diag_{c}" for c in sorted(DIAGNOSTIC_CODES))

# non-numeric columns: case_id (string) and the jump provenance key (storage name)
_NUMERIC_COLUMNS = tuple(c for c in _COLUMNS if c not in ("case_id", "max_storage_jump_key"))


def _require_columns(s):
    if not isinstance(s, ABCMapping):
        raise LedgerError("summary must be a mapping")
    missing = set(_COLUMNS) - set(s)
    if missing:
        raise LedgerError(f"summary missing columns: {sorted(missing, key=str)}")
    if not isinstance(s.get("counts"), ABCMapping):
        raise LedgerError("summary missing per-code 'counts' mapping")
    for c in _NUMERIC_COLUMNS:          # numeric columns must be finite scalars
        as_finite_float(f"summary[{c}]", s[c])


def _fmt(s, c):
    """Human-facing formatting (Markdown only). CSV keeps raw values."""
    if c == "max_primary_residual":
        return f"{s[c]:.3e}"
    if c in ("diagnostic_steps_rate", "max_storage_jump", "max_storage_jump_signed"):
        return f"{s[c]:.4f}"
    return str(s[c])


def budget_to_csv(summaries) -> str:
    """Machine-readable CSV: RAW values (full precision) + per-code diagnostic
    breakdown columns, via csv.writer so a case_id with a comma/newline is quoted
    rather than corrupting the row. Use budget_to_markdown for rounded values."""
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(_COLUMNS + _DIAG_COLUMNS)
    for s in summaries:
        _require_columns(s)
        counts = s["counts"]
        w.writerow([s[c] for c in _COLUMNS]
                   + [int(counts.get(c, 0)) for c in sorted(DIAGNOSTIC_CODES)])
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

    # per-code diagnostic breakdown: only the codes that actually fired anywhere
    active = [c for c in sorted(DIAGNOSTIC_CODES)
              if any(int(s["counts"].get(c, 0)) for s in summaries)]
    if active:
        lines += ["", "## Diagnostic breakdown (counts)",
                  "| case_id | " + " | ".join(active) + " |",
                  "| --- | " + " | ".join("---" for _ in active) + " |"]
        for s in summaries:
            lines.append("| " + _md_cell(s["case_id"]) + " | "
                         + " | ".join(str(int(s["counts"].get(c, 0))) for c in active) + " |")
    return "\n".join(lines) + "\n"
