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


def test_merge_rejects_leaky_child():
    # child B loses 0.5 mm not accounted as external -> its own residual is -0.5.
    # merge is fail-fast: a leaky child is rejected (it could otherwise telescope
    # with an opposite leak and hide in a clean aggregate).
    a = _ledger(0.0, 2.0, 0.0, 2.0)
    b = _ledger(2.0, 0.0, 0.0, 1.5)
    assert b.primary_mass_residual == pytest.approx(-0.5)
    with pytest.raises(LedgerError):
        merge_ledgers(a, b)


def test_merge_rejects_cancelling_child_leaks():
    # +0.5 then -0.5 leaks: aggregate would look clean, but each child is rejected.
    a = _ledger(0.0, 0.0, 0.0, 0.5)   # created 0.5 from nothing (residual +0.5)
    b = _ledger(0.5, 0.0, 0.0, 0.0)   # lost 0.5 (residual -0.5); contiguous
    with pytest.raises(LedgerError):
        merge_ledgers(a, b)


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


def test_event_flag_must_be_bool():
    with pytest.raises(LedgerError):
        _ledger(0.0, 0.0, 0.0, 0.0, event_flags={"freeze_event": "yes"})


def test_primary_mass_must_be_non_negative():
    with pytest.raises(LedgerError):                    # actual = -1 (sink 1 from 0)
        _ledger(0.0, 0.0, 1.0, -1.0)


def test_non_scalar_field_rejected():
    it = _zero_transfer(); aux = _zero_aux(); ev = _no_events()
    with pytest.raises(LedgerError):                    # array-like external_source
        StorageLedger(
            primary_before=0.0, external_source=[1.0], external_sink=0.0,
            internal_transfer=it, auxiliary_update=aux,
            primary_after_expected=0.0, primary_after_actual=0.0,
            primary_mass_residual=0.0, event_flags=ev)


def test_string_number_field_rejected():
    with pytest.raises(LedgerError):                    # "1.0" is a string, not numeric
        _ledger("1.0", 0.0, 0.0, 1.0)


def test_numeric_fields_normalized_to_float():
    lg = _ledger(0, 2, 0, 2, internal_transfer={"water_to_ice": 1})   # ints in
    assert type(lg.primary_before) is float
    assert type(lg.external_source) is float
    assert type(lg.internal_transfer["water_to_ice"]) is float


def test_make_ledger_wraps_bad_arithmetic():
    from droad.ledger import make_ledger
    with pytest.raises(LedgerError):                    # array-like -> LedgerError, not TypeError
        make_ledger(0.0, [1.0], 0.0, 0.0, _zero_transfer(), _zero_aux(), _no_events())


def test_storage_result_accepts_bare_string_diagnostic():
    from droad.ledger import StorageResult, DIAG_SNOW_OVERFLOW
    lg = _ledger(1.0, 0.0, 0.0, 1.0)
    r = StorageResult(object(), lg, DIAG_SNOW_OVERFLOW)   # bare string, not a tuple
    assert r.diagnostics == (DIAG_SNOW_OVERFLOW,)         # wrapped, not char-split


def test_rollout_audit_rejects_missing_keys():
    from droad.ledger import rollout_audit_to_dict
    with pytest.raises(LedgerError):
        rollout_audit_to_dict({"ledger": []})           # return_ledger=False shape


def test_rollout_audit_rejects_bad_detail_entry():
    from droad.ledger import rollout_audit_to_dict
    lg = _ledger(0.0, 0.0, 0.0, 0.0)
    with pytest.raises(LedgerError):                    # detail entry not a 2-tuple
        rollout_audit_to_dict({"ledger": [lg], "ledger_detail": [lg], "diagnostics": [()]})


def test_bool_numeric_field_rejected():
    with pytest.raises(LedgerError):                    # True would coerce to 1.0
        _ledger(0.0, True, 0.0, 1.0)


def test_merge_ledgers_rejects_bad_atol():
    a = _ledger(0.0, 1.0, 0.0, 1.0)
    b = _ledger(1.0, 0.0, 0.0, 1.0)
    with pytest.raises(LedgerError):                    # NaN atol would skip continuity
        merge_ledgers(a, b, atol=float("nan"))
    with pytest.raises(LedgerError):
        merge_ledgers(a, b, atol=-1.0)


def test_rollout_audit_rejects_non_ledger_entries():
    from droad.ledger import rollout_audit_to_dict
    with pytest.raises(LedgerError):
        rollout_audit_to_dict({"ledger": [object()], "ledger_detail": [(object(), object())],
                               "diagnostics": [()]})


def test_diagnostic_code_must_be_str():
    from droad.ledger import StorageResult
    lg = _ledger(1.0, 0.0, 0.0, 1.0)
    with pytest.raises(LedgerError):
        StorageResult(object(), lg, (123,))            # non-str diagnostic code


def test_rollout_audit_validates_diagnostic_codes():
    from droad.ledger import rollout_audit_to_dict
    lg = _ledger(0.0, 0.0, 0.0, 0.0)
    with pytest.raises(LedgerError):                    # unknown diagnostic code per step
        rollout_audit_to_dict({"ledger": [lg], "ledger_detail": [(lg, lg)],
                               "diagnostics": [("not_a_real_code",)]})


# --- defensive wrapping: bad public input -> LedgerError, not raw exception ---

def test_diagnostics_none_and_non_iterable_rejected():
    from droad.ledger import StorageResult
    lg = _ledger(1.0, 0.0, 0.0, 1.0)
    with pytest.raises(LedgerError):
        StorageResult(object(), lg, None)              # None
    with pytest.raises(LedgerError):
        StorageResult(object(), lg, 123)               # non-iterable, non-str


def test_check_keys_non_mapping_rejected():
    it = _zero_transfer(); aux = _zero_aux(); ev = _no_events()
    with pytest.raises(LedgerError):                    # internal_transfer not a mapping
        StorageLedger(
            primary_before=0.0, external_source=0.0, external_sink=0.0,
            internal_transfer=None, auxiliary_update=aux,
            primary_after_expected=0.0, primary_after_actual=0.0,
            primary_mass_residual=0.0, event_flags=ev)


def test_rollout_audit_non_mapping_rejected():
    from droad.ledger import rollout_audit_to_dict
    with pytest.raises(LedgerError):
        rollout_audit_to_dict(None)
    with pytest.raises(LedgerError):
        rollout_audit_to_dict([1, 2, 3])


def test_merge_rejects_non_ledger_child():
    a = _ledger(0.0, 0.0, 0.0, 0.0)
    with pytest.raises(LedgerError):
        merge_ledgers(a, object())


def test_make_ledger_non_mapping_transfer_rejected():
    from droad.ledger import make_ledger
    with pytest.raises(LedgerError):                    # None mapping -> LedgerError, not TypeError
        make_ledger(0.0, 0.0, 0.0, 0.0, None, _zero_aux(), _no_events())


def test_diagnostics_mapping_rejected():
    from droad.ledger import StorageResult, DIAG_SNOW_OVERFLOW
    lg = _ledger(1.0, 0.0, 0.0, 1.0)
    with pytest.raises(LedgerError):                    # mapping would silently become its keys
        StorageResult(object(), lg, {DIAG_SNOW_OVERFLOW: True})


def test_rollout_audit_non_sized_entries_rejected():
    from droad.ledger import rollout_audit_to_dict
    with pytest.raises(LedgerError):                    # entries not sized sequences
        rollout_audit_to_dict({"ledger": 5, "ledger_detail": 5, "diagnostics": 5})


def test_diagnostics_set_rejected():
    from droad.ledger import StorageResult, DIAG_SNOW_OVERFLOW
    lg = _ledger(1.0, 0.0, 0.0, 1.0)
    with pytest.raises(LedgerError):                    # set order is non-deterministic
        StorageResult(object(), lg, {DIAG_SNOW_OVERFLOW})


def test_merge_split_tolerances():
    # continuity gap 0.05, child residuals clean: loose continuity_atol passes,
    # tight continuity_atol rejects; residual_atol is independent.
    a = _ledger(0.0, 1.0, 0.0, 1.0)
    b = _ledger(1.05, 0.0, 0.0, 1.05)                  # before=1.05 vs a.after=1.0 -> 0.05 gap
    merge_ledgers(a, b, continuity_atol=0.1)           # tolerated
    with pytest.raises(LedgerError):
        merge_ledgers(a, b, continuity_atol=1e-9)      # rejected
    with pytest.raises(LedgerError):                   # bad split tolerance still rejected
        merge_ledgers(a, b, residual_atol=float("nan"))


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
