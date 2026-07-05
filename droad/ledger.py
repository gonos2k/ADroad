"""Mass ledger contract for storage steps (P0 §3).

State-mutating Surf steps (snow/ice/deposit_storage, road_cond) return a
StorageResult(state_next, ledger, diagnostics); scalar helper steps return
value(s) plus ledger, and may additionally return diagnostics — e.g.
precipitation_to_storage returns (SrfWat, SrfSnow, ledger) and water_storage
returns (SrfWat, ledger, diagnostics). The ledger separates
primary mass (water+snow+ice+deposit), auxiliary (ice2), and external
source/sink so conservation is testable — not a single "total water" number.

Key sets are fixed: a process that doesn't touch a key must still report 0,
so a missing transfer can't be silently hidden by a zero residual. Both internal
transfers and external source/sink are accumulated AT their branches (not
inferred from the net delta), so `primary_mass_residual` is a genuine leak
detector — it is ~0 for correct code and non-zero if mass changes without being
recorded. Ledgers are immutable once built (mappings are frozen); use
`ledger_to_dict` for JSON/logging.

Semantics of a non-zero residual: it flags an accounting leak in OUR code (a
branch that changed mass without booking a transfer/external flow) — NOT a
physical-invariant judgement about the reference model. RoadSurf's own quirks
(e.g. melting more snow than exists, then clamping to 0) are mirrored faithfully:
the clamp import/export is booked as external, so the residual stays ~0. Deciding
whether such reference behavior is *physically* desirable is a separate concern
(deviation budget), not something the residual asserts.
"""

from __future__ import annotations

import math
from collections.abc import Mapping as ABCMapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

Number = float  # backend-neutral scalar for M1 (NumPy/plain float)

INTERNAL_TRANSFER_KEYS = (
    "water_to_ice", "ice_to_water", "snow_to_water", "snow_to_ice",
    "deposit_to_water", "deposit_to_ice",
)

# Feasibility diagnostics (physics/numerical flags, SEPARATE from mass accounting).
# Registered like transfer keys so a typo raises instead of silently passing.
DIAG_SNOW_OVER_MELT = "snow_over_melt"
DIAG_SNOW_NEGATIVE_PRE_CLAMP = "snow_negative_pre_clamp"
DIAG_SNOW_OVERFLOW = "snow_overflow"
DIAG_ICE_OVER_MELT = "ice_over_melt"
DIAG_ICE_NEGATIVE_PRE_CLAMP = "ice_negative_pre_clamp"
DIAG_ICE_OVERFLOW = "ice_overflow"
DIAG_DEPOSIT_NEGATIVE_PRE_CLAMP = "deposit_negative_pre_clamp"
DIAG_DEPOSIT_OVERFLOW = "deposit_overflow"
DIAG_WATER_NEGATIVE_PRE_CLAMP = "water_negative_pre_clamp"
DIAG_WATER_OVERFLOW = "water_overflow"
DIAGNOSTIC_CODES = frozenset({
    DIAG_SNOW_OVER_MELT, DIAG_SNOW_NEGATIVE_PRE_CLAMP, DIAG_SNOW_OVERFLOW,
    DIAG_ICE_OVER_MELT, DIAG_ICE_NEGATIVE_PRE_CLAMP, DIAG_ICE_OVERFLOW,
    DIAG_DEPOSIT_NEGATIVE_PRE_CLAMP, DIAG_DEPOSIT_OVERFLOW,
    DIAG_WATER_NEGATIVE_PRE_CLAMP, DIAG_WATER_OVERFLOW,
})
AUXILIARY_UPDATE_KEYS = ("ice2_increase", "ice2_decrease", "ice2_reset")
EVENT_FLAG_KEYS = ("freeze_event", "melt_event", "snow_event", "deposit_melt_event")


class LedgerError(ValueError):
    """Raised on missing/unknown ledger keys."""


def as_finite_float(name: str, value) -> float:
    """Public alias of the finite-float coercion — a cross-module validation
    contract (used by droad.deviation as well)."""
    return _as_finite_float(name, value)


def normalize_diagnostics(diagnostics) -> tuple:
    """Public alias of the diagnostics normalizer — shared validation contract."""
    return _normalize_diagnostics(diagnostics)


def _normalize_diagnostics(diagnostics) -> tuple:
    """Normalize & validate a diagnostics collection (shared by StorageResult and
    rollout_audit_to_dict): a bare string becomes a 1-tuple, every code must be a
    registered string. Returns a tuple; raises LedgerError on anything invalid."""
    if diagnostics is None:
        raise LedgerError("diagnostics must be an iterable of codes (or a str), not None")
    if isinstance(diagnostics, ABCMapping):     # a mapping would silently become its keys
        raise LedgerError("diagnostics must be a str or iterable of codes, not a mapping")
    if isinstance(diagnostics, (set, frozenset)):   # set order is non-deterministic
        raise LedgerError("diagnostics must be an ordered iterable, not a set")
    try:
        d = (diagnostics,) if isinstance(diagnostics, str) else tuple(diagnostics)
    except TypeError:
        raise LedgerError(f"diagnostics must be iterable or str, got {diagnostics!r}") from None
    for code in d:
        if not isinstance(code, str):
            raise LedgerError(f"diagnostic code must be str, got {code!r}")
    unknown = set(d) - DIAGNOSTIC_CODES
    if unknown:
        raise LedgerError(f"unknown diagnostic codes: {sorted(unknown, key=str)}")
    return d


def _check_keys(d: Mapping[str, object], required: tuple[str, ...], name: str) -> None:
    if not isinstance(d, ABCMapping):
        raise LedgerError(f"{name} must be a mapping, got {type(d).__name__}")
    keys = set(d)
    missing = set(required) - keys
    unknown = keys - set(required)
    if missing:
        raise LedgerError(f"{name} missing keys: {sorted(missing, key=str)}")
    if unknown:
        raise LedgerError(f"{name} has unknown keys: {sorted(unknown, key=str)}")


def _is_boolish(v) -> bool:
    """True for a genuine boolean — Python bool or numpy bool_ (detected by
    duck-typing so this backend-neutral module needn't import numpy). Rejects
    str/int/float, so a truthy "False" string can't masquerade as a flag."""
    if isinstance(v, bool):
        return True
    cls = type(v)                                    # numpy bool: name 'bool'/'bool_'
    return (cls.__module__.split(".")[0] == "numpy"
            and cls.__name__ in ("bool_", "bool"))


def _as_finite_float(name: str, value) -> float:
    """Coerce to a finite Python float or raise LedgerError. Returns the value so
    the caller can NORMALIZE the field — every numeric field is stored as a plain
    float, so later `< 0` / arithmetic can't TypeError on a str/array/np/jax scalar.
    Strings and bools are rejected (a mass amount must be numeric, not "1.0" and
    not True/False which would silently coerce to 1.0/0.0)."""
    if _is_boolish(value):
        raise LedgerError(f"{name} must be numeric, not bool: {value!r}")
    if isinstance(value, str):
        raise LedgerError(f"{name} must be a numeric scalar, not a string: {value!r}")
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise LedgerError(f"{name} must be a finite scalar, got {value!r}") from None
    if not math.isfinite(v):
        raise LedgerError(f"{name} must be finite, got {value!r}")
    return v


_CONSISTENCY_TOL = 1e-9


@dataclass(frozen=True)
class StorageLedger:
    primary_before: Number
    external_source: Number
    external_sink: Number
    internal_transfer: Mapping[str, Number]
    auxiliary_update: Mapping[str, Number]
    primary_after_expected: Number
    primary_after_actual: Number
    primary_mass_residual: Number
    event_flags: Mapping[str, bool]

    def __post_init__(self):
        _check_keys(self.internal_transfer, INTERNAL_TRANSFER_KEYS, "internal_transfer")
        _check_keys(self.auxiliary_update, AUXILIARY_UPDATE_KEYS, "auxiliary_update")
        _check_keys(self.event_flags, EVENT_FLAG_KEYS, "event_flags")

        # finiteness + normalization: coerce every numeric field to a plain float
        # (NaN/Inf/str/array rejected). Done first so all later comparisons are safe.
        for nm in ("primary_before", "external_source", "external_sink",
                   "primary_after_expected", "primary_after_actual",
                   "primary_mass_residual"):
            object.__setattr__(self, nm, _as_finite_float(nm, getattr(self, nm)))
        it = {k: _as_finite_float(f"internal_transfer[{k}]", v)
              for k, v in self.internal_transfer.items()}
        aux = {k: _as_finite_float(f"auxiliary_update[{k}]", v)
               for k, v in self.auxiliary_update.items()}
        object.__setattr__(self, "internal_transfer", it)   # frozen at end
        object.__setattr__(self, "auxiliary_update", aux)

        # event_flags must be genuine bools (merge_ledgers uses bool(...) — a
        # string "False" would be truthy and corrupt the aggregated flag)
        for k, v in self.event_flags.items():
            if not _is_boolish(v):
                raise LedgerError(f"event_flags[{k}] must be bool, got {v!r}")

        # non-negativity: source/sink and every transfer/aux amount are magnitudes
        if self.external_source < 0.0 or self.external_sink < 0.0:
            raise LedgerError("external_source/external_sink must be non-negative")
        for k, v in self.internal_transfer.items():
            if v < 0.0:
                raise LedgerError(f"internal_transfer[{k}] must be non-negative")
        for k, v in self.auxiliary_update.items():
            if v < 0.0:
                raise LedgerError(f"auxiliary_update[{k}] must be non-negative")

        # primary mass (water+snow+ice+deposit) is a physical state — non-negative.
        # expected may go negative for residual diagnosis, but before/actual can't.
        if self.primary_before < -_CONSISTENCY_TOL:
            raise LedgerError(f"primary_before must be non-negative, got {self.primary_before}")
        if self.primary_after_actual < -_CONSISTENCY_TOL:
            raise LedgerError(
                f"primary_after_actual must be non-negative, got {self.primary_after_actual}")

        # derived-field consistency: expected & residual must match the flows, so a
        # directly-constructed ledger can't carry a forged residual (only the
        # amounts are trusted; expected/residual are recomputed and checked).
        expected = self.primary_before + self.external_source - self.external_sink
        if abs(self.primary_after_expected - expected) > _CONSISTENCY_TOL:
            raise LedgerError("primary_after_expected inconsistent with flows")
        if abs(self.primary_mass_residual
               - (self.primary_after_actual - expected)) > _CONSISTENCY_TOL:
            raise LedgerError("primary_mass_residual inconsistent with flows")

        # freeze the mappings so an audit record can't be mutated after the fact
        object.__setattr__(self, "internal_transfer", MappingProxyType(dict(self.internal_transfer)))
        object.__setattr__(self, "auxiliary_update", MappingProxyType(dict(self.auxiliary_update)))
        object.__setattr__(self, "event_flags",       # store as plain Python bool
                           MappingProxyType({k: bool(v) for k, v in self.event_flags.items()}))


@dataclass(frozen=True)
class StorageResult:
    state_next: object
    ledger: StorageLedger
    # physics/numerical feasibility diagnostics — SEPARATE from the mass ledger.
    # The ledger residual detects code-accounting leaks; these flag physically
    # notable events (over-melt, negative-before-clamp, hard-projection hits) that
    # are mass-conserving but relevant to the deviation budget. Never affects mass.
    diagnostics: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, "diagnostics", _normalize_diagnostics(self.diagnostics))


def make_ledger(
    primary_before, external_source, external_sink,
    primary_after_actual, internal_transfer, auxiliary_update, event_flags,
) -> StorageLedger:
    """Build a ledger; residual is derived, never passed in."""
    # normalize the scalars first so the pre-construction arithmetic can't raise a
    # raw TypeError (bad input -> LedgerError, consistent with StorageLedger).
    pb = _as_finite_float("primary_before", primary_before)
    src = _as_finite_float("external_source", external_source)
    sink = _as_finite_float("external_sink", external_sink)
    actual = _as_finite_float("primary_after_actual", primary_after_actual)
    expected = pb + src - sink
    # Pass the mappings THROUGH (no dict() here): StorageLedger.__post_init__ runs
    # _check_keys (which rejects non-mapping as LedgerError) and then freezes a
    # copy — pre-converting here would raise a raw TypeError on e.g. None.
    return StorageLedger(
        primary_before=pb,
        external_source=src,
        external_sink=sink,
        internal_transfer=internal_transfer,
        auxiliary_update=auxiliary_update,
        primary_after_expected=expected,
        primary_after_actual=actual,
        primary_mass_residual=actual - expected,
        event_flags=event_flags,
    )


def ledger_to_dict(lg: StorageLedger) -> dict:
    """Plain nested-dict view of a ledger for JSON/logging/reporting.

    The ledger's own mappings are frozen (MappingProxyType); this returns fresh
    mutable dicts so a report can be serialized without touching the audit record.
    """
    return {
        "primary_before": lg.primary_before,
        "external_source": lg.external_source,
        "external_sink": lg.external_sink,
        "internal_transfer": dict(lg.internal_transfer),
        "auxiliary_update": dict(lg.auxiliary_update),
        "primary_after_expected": lg.primary_after_expected,
        "primary_after_actual": lg.primary_after_actual,
        "primary_mass_residual": lg.primary_mass_residual,
        "event_flags": dict(lg.event_flags),
    }


def storage_result_to_dict(r: StorageResult) -> dict:
    """JSON/logging view of a StorageResult: its ledger plus the diagnostics
    (state_next is omitted — it's the model state, not part of the audit record)."""
    return {"ledger": ledger_to_dict(r.ledger), "diagnostics": list(r.diagnostics)}


def rollout_audit_to_dict(out: Mapping) -> dict:
    """JSON/logging view of a full_rollout(return_ledger=True) audit trail:
    per-step merged ledger, (prec, cond) detail, and diagnostics."""
    if not isinstance(out, ABCMapping):
        raise LedgerError("rollout audit must be a mapping from full_rollout(return_ledger=True)")
    missing = {"ledger", "ledger_detail", "diagnostics"} - set(out)
    if missing:
        raise LedgerError(
            f"rollout audit keys missing (need return_ledger=True): {sorted(missing, key=str)}")
    try:
        n = len(out["ledger"])
        lengths_ok = len(out["ledger_detail"]) == n == len(out["diagnostics"])
    except TypeError:
        raise LedgerError("rollout audit entries must be sized sequences") from None
    if not lengths_ok:
        raise LedgerError("rollout audit lists have inconsistent lengths")
    for lg in out["ledger"]:
        if not isinstance(lg, StorageLedger):
            raise LedgerError("ledger entries must be StorageLedger")
    for item in out["ledger_detail"]:
        if not (isinstance(item, tuple) and len(item) == 2):
            raise LedgerError("ledger_detail entries must be (prec_ledger, cond_ledger)")
        if not all(isinstance(x, StorageLedger) for x in item):
            raise LedgerError("ledger_detail entries must contain StorageLedger objects")
    diagnostics = [list(_normalize_diagnostics(d)) for d in out["diagnostics"]]  # validate codes
    return {
        "ledger": [ledger_to_dict(lg) for lg in out["ledger"]],
        "ledger_detail": [{"prec": ledger_to_dict(p), "cond": ledger_to_dict(c)}
                          for p, c in out["ledger_detail"]],
        "diagnostics": diagnostics,
    }


def _sum_by_keys(ledgers, attr, keys):
    return {k: sum(getattr(l, attr)[k] for l in ledgers) for k in keys}


def _or_by_keys(ledgers, keys):
    return {k: any(bool(l.event_flags[k]) for l in ledgers) for k in keys}


def merge_ledgers(*ledgers: StorageLedger, atol: float = 1e-9,
                  residual_atol: float | None = None,
                  continuity_atol: float | None = None) -> StorageLedger:
    """Aggregate child ledgers (P0 §3, no-go #4).

    Key rule: do NOT sum child residuals. Recompute the expected primary mass
    from the first `primary_before` and summed external flows, and take the
    actual from the LAST child (the final state after all sub-steps).

    Fail-fast on child leaks: every child must itself be balanced (|residual| <=
    residual_atol). Otherwise two children with opposite unaccounted mass
    (+0.5 / -0.5) could telescope to a clean aggregate and hide the leak.
    Continuity is enforced separately: each child's `primary_after_actual` must
    equal the next child's `primary_before` within continuity_atol.

    `atol` sets both tolerances; `residual_atol`/`continuity_atol` override each
    independently (the two mean different things — accounting leak vs. float
    join error — and may want to diverge on float32/large-rollout audit paths).
    """
    if not ledgers:
        raise LedgerError("merge_ledgers requires at least one ledger")
    atol = _as_finite_float("merge_ledgers.atol", atol)   # NaN would skip the check
    res_atol = atol if residual_atol is None else _as_finite_float(
        "merge_ledgers.residual_atol", residual_atol)
    con_atol = atol if continuity_atol is None else _as_finite_float(
        "merge_ledgers.continuity_atol", continuity_atol)
    if atol < 0.0 or res_atol < 0.0 or con_atol < 0.0:
        raise LedgerError("merge_ledgers tolerances must be non-negative")
    for i, lg in enumerate(ledgers):
        if not isinstance(lg, StorageLedger):
            raise LedgerError(f"merge_ledgers child {i} must be StorageLedger, got {type(lg).__name__}")
        if abs(lg.primary_mass_residual) > res_atol:
            raise LedgerError(
                f"child ledger {i} has non-zero residual: {lg.primary_mass_residual}")
    for a, b in zip(ledgers, ledgers[1:]):
        if abs(a.primary_after_actual - b.primary_before) > con_atol:
            raise LedgerError(
                "non-contiguous ledgers: "
                f"{a.primary_after_actual} -> {b.primary_before}")
    primary_before = ledgers[0].primary_before
    external_source = sum(l.external_source for l in ledgers)
    external_sink = sum(l.external_sink for l in ledgers)
    primary_after_actual = ledgers[-1].primary_after_actual
    return StorageLedger(
        primary_before=primary_before,
        external_source=external_source,
        external_sink=external_sink,
        internal_transfer=_sum_by_keys(ledgers, "internal_transfer", INTERNAL_TRANSFER_KEYS),
        auxiliary_update=_sum_by_keys(ledgers, "auxiliary_update", AUXILIARY_UPDATE_KEYS),
        primary_after_expected=primary_before + external_source - external_sink,
        primary_after_actual=primary_after_actual,
        primary_mass_residual=primary_after_actual - (primary_before + external_source - external_sink),
        event_flags=_or_by_keys(ledgers, EVENT_FLAG_KEYS),
    )
