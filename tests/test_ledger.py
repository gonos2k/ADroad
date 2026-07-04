import pytest

from droad.ledger import (
    INTERNAL_TRANSFER_KEYS,
    AUXILIARY_UPDATE_KEYS,
    EVENT_FLAG_KEYS,
    LedgerError,
    StorageLedger,
    make_ledger,
    merge_ledgers,
)


def _zero_transfer():
    return {k: 0.0 for k in INTERNAL_TRANSFER_KEYS}


def _zero_aux():
    return {k: 0.0 for k in AUXILIARY_UPDATE_KEYS}


def _no_events():
    return {k: False for k in EVENT_FLAG_KEYS}


def _ledger(before, source, sink, after, **over):
    it = _zero_transfer(); it.update(over.get("internal_transfer", {}))
    aux = _zero_aux(); aux.update(over.get("auxiliary_update", {}))
    ev = _no_events(); ev.update(over.get("event_flags", {}))
    return make_ledger(before, source, sink, after, it, aux, ev)


def test_make_ledger_residual_is_derived():
    # precipitation: +2 mm rain, no internal transfer, actual matches expected
    lg = _ledger(1.0, 2.0, 0.0, 3.0)
    assert lg.primary_after_expected == 3.0
    assert lg.primary_mass_residual == 0.0


def test_freezing_conserves_primary_mass():
    # water -> ice, primary unchanged (source=sink=0)
    lg = _ledger(2.0, 0.0, 0.0, 2.0,
                 internal_transfer={"water_to_ice": 1.0},
                 event_flags={"freeze_event": True})
    assert lg.primary_mass_residual == 0.0
    assert lg.event_flags["freeze_event"] is True


def test_missing_key_rejected():
    with pytest.raises(LedgerError):
        StorageLedger(
            primary_before=0.0, external_source=0.0, external_sink=0.0,
            internal_transfer={"water_to_ice": 0.0},  # missing others
            auxiliary_update=_zero_aux(),
            primary_after_expected=0.0, primary_after_actual=0.0,
            primary_mass_residual=0.0, event_flags=_no_events(),
        )


def test_unknown_key_rejected():
    with pytest.raises(LedgerError):
        _ledger(0.0, 0.0, 0.0, 0.0, event_flags={"unexpected_event": True})


def test_merge_recomputes_residual_not_sums():
    # child A: +2 source, actual 2 (residual 0)
    a = _ledger(0.0, 2.0, 0.0, 2.0)
    # child B: internal transfer only, actual stays 2 (residual 0)
    b = _ledger(2.0, 0.0, 0.0, 2.0, internal_transfer={"water_to_ice": 1.0})
    merged = merge_ledgers(a, b)
    assert merged.primary_before == 0.0
    assert merged.external_source == 2.0
    assert merged.primary_after_actual == 2.0
    assert merged.primary_after_expected == 2.0
    assert merged.primary_mass_residual == 0.0
    assert merged.internal_transfer["water_to_ice"] == 1.0


def test_merge_detects_leak_via_actual_last_child():
    # child B loses 0.5 mm that isn't accounted as external -> residual shows it
    a = _ledger(0.0, 2.0, 0.0, 2.0)
    b = _ledger(2.0, 0.0, 0.0, 1.5)  # actual dropped, no external sink recorded
    merged = merge_ledgers(a, b)
    assert merged.primary_mass_residual == pytest.approx(-0.5)


def test_merge_event_flags_or():
    a = _ledger(0.0, 0.0, 0.0, 0.0, event_flags={"freeze_event": True})
    b = _ledger(0.0, 0.0, 0.0, 0.0, event_flags={"melt_event": True})
    merged = merge_ledgers(a, b)
    assert merged.event_flags["freeze_event"] is True
    assert merged.event_flags["melt_event"] is True
    assert merged.event_flags["snow_event"] is False
