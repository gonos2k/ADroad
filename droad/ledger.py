"""Mass ledger contract for storage steps (P0 §3).

State-mutating Surf steps (snow/ice/deposit_storage, road_cond) return a
StorageResult(state_next, ledger); scalar helper steps (water_storage,
precipitation_to_storage) return (value(s), ledger). The ledger separates
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

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

Number = float  # backend-neutral scalar for M1 (NumPy/plain float)

INTERNAL_TRANSFER_KEYS = (
    "water_to_ice", "ice_to_water", "snow_to_water", "snow_to_ice",
    "deposit_to_water", "deposit_to_ice",
)
AUXILIARY_UPDATE_KEYS = ("ice2_increase", "ice2_decrease", "ice2_reset")
EVENT_FLAG_KEYS = ("freeze_event", "melt_event", "snow_event", "deposit_melt_event")


class LedgerError(ValueError):
    """Raised on missing/unknown ledger keys."""


def _check_keys(d: Mapping[str, object], required: tuple[str, ...], name: str) -> None:
    keys = set(d)
    missing = set(required) - keys
    unknown = keys - set(required)
    if missing:
        raise LedgerError(f"{name} missing keys: {sorted(missing)}")
    if unknown:
        raise LedgerError(f"{name} has unknown keys: {sorted(unknown)}")


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

        # non-negativity: source/sink and every transfer/aux amount are magnitudes
        if self.external_source < 0.0 or self.external_sink < 0.0:
            raise LedgerError("external_source/external_sink must be non-negative")
        for k, v in self.internal_transfer.items():
            if v < 0.0:
                raise LedgerError(f"internal_transfer[{k}] must be non-negative")
        for k, v in self.auxiliary_update.items():
            if v < 0.0:
                raise LedgerError(f"auxiliary_update[{k}] must be non-negative")

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
        object.__setattr__(self, "event_flags", MappingProxyType(dict(self.event_flags)))


@dataclass(frozen=True)
class StorageResult:
    state_next: object
    ledger: StorageLedger


def make_ledger(
    primary_before, external_source, external_sink,
    primary_after_actual, internal_transfer, auxiliary_update, event_flags,
) -> StorageLedger:
    """Build a ledger; residual is derived, never passed in."""
    expected = primary_before + external_source - external_sink
    return StorageLedger(
        primary_before=primary_before,
        external_source=external_source,
        external_sink=external_sink,
        internal_transfer=dict(internal_transfer),
        auxiliary_update=dict(auxiliary_update),
        primary_after_expected=expected,
        primary_after_actual=primary_after_actual,
        primary_mass_residual=primary_after_actual - expected,
        event_flags=dict(event_flags),
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


def _sum_by_keys(ledgers, attr, keys):
    return {k: sum(getattr(l, attr)[k] for l in ledgers) for k in keys}


def _or_by_keys(ledgers, keys):
    return {k: any(bool(l.event_flags[k]) for l in ledgers) for k in keys}


def merge_ledgers(*ledgers: StorageLedger, atol: float = 1e-9) -> StorageLedger:
    """Aggregate child ledgers (P0 §3, no-go #4).

    Key rule: do NOT sum child residuals. Recompute the expected primary mass
    from the first `primary_before` and summed external flows, and take the
    actual from the LAST child (the final state after all sub-steps).

    Continuity is enforced: each child's `primary_after_actual` must equal the
    next child's `primary_before`, so a reordered or mis-recorded sub-step can't
    pass silently.
    """
    if not ledgers:
        raise LedgerError("merge_ledgers requires at least one ledger")
    for a, b in zip(ledgers, ledgers[1:]):
        if abs(a.primary_after_actual - b.primary_before) > atol:
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
