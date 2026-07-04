"""Mass ledger contract for storage steps (P0 §3).

Every storage function returns a StorageResult(state_next, ledger). The ledger
separates primary mass (water+snow+ice+deposit), auxiliary (ice2), and external
source/sink so conservation is testable — not a single "total water" number.

Key sets are fixed: a process that doesn't touch a key must still report 0,
so a missing transfer can't be silently hidden by a zero residual.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

Number = float  # backend-neutral scalar for M1 (NumPy/plain float)

INTERNAL_TRANSFER_KEYS = (
    "water_to_ice", "ice_to_water", "snow_to_water", "snow_to_ice", "deposit_to_water",
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


def _sum_by_keys(ledgers, attr, keys):
    return {k: sum(getattr(l, attr)[k] for l in ledgers) for k in keys}


def _or_by_keys(ledgers, keys):
    return {k: any(bool(l.event_flags[k]) for l in ledgers) for k in keys}


def merge_ledgers(*ledgers: StorageLedger) -> StorageLedger:
    """Aggregate child ledgers (P0 §3, no-go #4).

    Key rule: do NOT sum child residuals. Recompute the expected primary mass
    from the first `primary_before` and summed external flows, and take the
    actual from the LAST child (the final state after all sub-steps).
    """
    if not ledgers:
        raise LedgerError("merge_ledgers requires at least one ledger")
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
