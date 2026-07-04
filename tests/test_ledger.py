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


# --- adversarial regression (3rd review) ---

def test_merge_rejects_non_contiguous_children():
    a = _ledger(0.0, 2.0, 0.0, 2.0)          # ends at 2.0
    b = _ledger(5.0, 0.0, 0.0, 5.0)          # but next starts at 5.0 -> gap
    with pytest.raises(LedgerError):
        merge_ledgers(a, b)


def test_ledger_mappings_are_immutable():
    lg = _ledger(1.0, 2.0, 0.0, 3.0, internal_transfer={"water_to_ice": 1.0})
    with pytest.raises(TypeError):
        lg.internal_transfer["water_to_ice"] = 999.0
    with pytest.raises(TypeError):
        lg.event_flags["freeze_event"] = True


def test_directly_forged_residual_rejected():
    from droad.ledger import StorageLedger
    it = _zero_transfer(); aux = _zero_aux(); ev = _no_events()
    with pytest.raises(LedgerError):        # actual=5 but residual lied as 0
        StorageLedger(
            primary_before=0.0, external_source=2.0, external_sink=0.0,
            internal_transfer=it, auxiliary_update=aux,
            primary_after_expected=2.0, primary_after_actual=5.0,
            primary_mass_residual=0.0,     # true residual is 5-2=3, not 0
            event_flags=ev)


def test_negative_flow_rejected():
    with pytest.raises(LedgerError):
        _ledger(0.0, -1.0, 0.0, -1.0)      # negative external_source


def test_ledger_to_dict_is_json_serializable():
    import json
    from droad.ledger import ledger_to_dict
    lg = _ledger(1.0, 2.0, 0.0, 3.0, internal_transfer={"water_to_ice": 1.0},
                 event_flags={"freeze_event": True})
    d = ledger_to_dict(lg)
    assert d["primary_after_actual"] == 3.0
    assert d["internal_transfer"]["water_to_ice"] == 1.0
    assert d["event_flags"]["freeze_event"] is True
    json.dumps(d)                          # must not raise (plain dicts, not proxies)
    d["internal_transfer"]["water_to_ice"] = 999.0   # mutating the copy is fine
    assert lg.internal_transfer["water_to_ice"] == 1.0   # original untouched


def test_storage_result_rejects_unknown_diagnostic():
    from droad.ledger import StorageResult, DIAG_SNOW_OVERFLOW
    lg = _ledger(1.0, 0.0, 0.0, 1.0)
    StorageResult(object(), lg, (DIAG_SNOW_OVERFLOW,))       # known code ok
    with pytest.raises(LedgerError):
        StorageResult(object(), lg, ("snow_overflowwww",))  # typo rejected


def test_storage_result_to_dict_is_json_serializable():
    import json
    from droad.ledger import StorageResult, storage_result_to_dict, DIAG_ICE_OVER_MELT
    lg = _ledger(1.0, 0.0, 0.0, 1.0)
    d = storage_result_to_dict(StorageResult(object(), lg, (DIAG_ICE_OVER_MELT,)))
    assert d["diagnostics"] == ["ice_over_melt"]
    assert d["ledger"]["primary_after_actual"] == 1.0
    json.dumps(d)                          # state_next omitted -> serializable


def test_ledger_rejects_nan_inf():
    for bad in (float("nan"), float("inf")):
        with pytest.raises(LedgerError):
            _ledger(0.0, bad, 0.0, bad)                 # non-finite external_source/actual
    with pytest.raises(LedgerError):                    # non-finite transfer amount
        _ledger(0.0, 0.0, 0.0, 0.0, internal_transfer={"water_to_ice": float("nan")})


def test_storage_result_diagnostics_frozen_to_tuple():
    from droad.ledger import StorageResult, DIAG_SNOW_OVERFLOW
    lg = _ledger(1.0, 0.0, 0.0, 1.0)
    r = StorageResult(object(), lg, [DIAG_SNOW_OVERFLOW])   # pass a list
    assert isinstance(r.diagnostics, tuple)                 # normalized/frozen
    assert r.diagnostics == (DIAG_SNOW_OVERFLOW,)


def test_rollout_audit_to_dict_is_json_serializable():
    import json
    from droad.ledger import rollout_audit_to_dict, DIAG_SNOW_OVERFLOW
    lg = _ledger(0.0, 1.0, 0.0, 1.0)
    out = {"ledger": [lg], "ledger_detail": [(lg, lg)],
           "diagnostics": [(DIAG_SNOW_OVERFLOW,)]}
    d = rollout_audit_to_dict(out)
    assert len(d["ledger"]) == 1 and d["ledger_detail"][0]["cond"]["external_source"] == 1.0
    assert d["diagnostics"] == [["snow_overflow"]]
    json.dumps(d)
