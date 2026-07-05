"""Deviation budget aggregation: residual gate + diagnostics counting + serialization.

Unit tests on synthetic full_rollout(return_ledger=True)-shaped dicts (no heavy
rollout needed). Verifies residual is the P0 gate and diagnostics are counted,
never failing.
"""
import pytest

from droad.ledger import (
    INTERNAL_TRANSFER_KEYS, AUXILIARY_UPDATE_KEYS, EVENT_FLAG_KEYS,
    LedgerError, make_ledger,
)
from droad.deviation import (
    deviation_budget, accounting_gate, budget_to_csv, budget_to_markdown,
)


def _ledger(before, source, sink, after):
    return make_ledger(before, source, sink, after,
                       {k: 0.0 for k in INTERNAL_TRANSFER_KEYS},
                       {k: 0.0 for k in AUXILIARY_UPDATE_KEYS},
                       {k: False for k in EVENT_FLAG_KEYS})


def _clean(n):
    return [_ledger(0.0, 0.0, 0.0, 0.0) for _ in range(n)]


def test_budget_counts_diagnostics_and_residual():
    diags = [("snow_over_melt",), (), ("water_overflow", "snow_negative_pre_clamp"), ()]
    out = {"ledger": _clean(4), "diagnostics": diags,
           "Snow": [0.0, 0.1, 0.05, 0.05], "Water": [0.0, 0.0, 0.3, 0.3]}
    b = deviation_budget(out, case_id="c1")
    assert b["n_steps"] == 4
    assert b["max_primary_residual"] == 0.0
    assert b["n_diagnostics_total"] == 3
    assert b["diagnostic_steps"] == 2
    assert b["diagnostic_steps_rate"] == pytest.approx(0.5)
    assert b["over_melt_count"] == 1                      # snow_over_melt
    assert b["overflow_count"] == 1                       # water_overflow
    assert b["negative_pre_clamp_count"] == 1             # snow_negative_pre_clamp
    assert b["counts"]["snow_over_melt"] == 1
    assert b["max_storage_jump"] == pytest.approx(0.3)    # Water 0.0->0.3


def test_budget_steps_slices_to_holdout_window():
    # 4-step run; only step 2 has a diagnostic and the big Water jump is 0.0->0.3 at step 2.
    diags = [(), (), ("water_overflow",), ()]
    out = {"ledger": _clean(4), "diagnostics": diags,
           "Water": [0.0, 0.0, 0.3, 0.31], "Snow": [0.0, 0.0, 0.0, 0.0]}
    # holdout = steps [2, 3]: the overflow AND the 0.3 jump fall in-window
    b = deviation_budget(out, case_id="hold", steps=[2, 3])
    assert b["n_steps"] == 2
    assert b["overflow_count"] == 1
    assert b["max_storage_jump"] == pytest.approx(0.01)   # within-window: Water 0.30->0.31
    # holdout = steps [0, 1]: quiet window, no diagnostics, no jump
    b0 = deviation_budget(out, case_id="hold0", steps=[0, 1])
    assert b0["overflow_count"] == 0
    assert b0["max_storage_jump"] == pytest.approx(0.0)


def test_budget_steps_validation():
    out = {"ledger": _clean(3), "diagnostics": [(), (), ()],
           "Water": [0.0, 0.1, 0.2]}
    with pytest.raises(LedgerError):
        deviation_budget(out, steps=[0, 3])                  # index out of range
    with pytest.raises(LedgerError):
        deviation_budget(out, steps=[])                      # selects zero steps
    with pytest.raises(LedgerError):
        deviation_budget(out, steps={0: 1})                  # mapping, not ordered indices
    with pytest.raises(LedgerError):
        deviation_budget(out, steps=[0, 1.9])                # fractional index (no silent trunc)
    with pytest.raises(LedgerError):
        deviation_budget(out, steps=[True, 2])               # bool is not an index
    with pytest.raises(LedgerError):
        deviation_budget(out, steps=[1, 1])                  # duplicate would double-count
    with pytest.raises(LedgerError):
        deviation_budget(out, steps=[2, 0])                  # reversal breaks 'interval' meaning


def test_budget_steps_rejects_numpy_bool_index():
    import numpy as np
    out = {"ledger": _clean(3), "diagnostics": [(), (), ()], "Water": [0.0, 0.1, 0.2]}
    with pytest.raises(LedgerError):                  # np.bool_ index, same policy as ledger
        deviation_budget(out, steps=[np.bool_(True), 2])


def test_budget_steps_rejects_dict_trajectory_bypass():
    # a dict trajectory is integer-indexable but NOT an ordered sequence; the steps=
    # path must reject the original container before slicing, not silently accept it.
    out = {"ledger": _clean(3), "diagnostics": [(), (), ()],
           "Water": {0: 0.0, 1: 0.1, 2: 0.2}}
    with pytest.raises(LedgerError):
        deviation_budget(out, steps=[0, 1, 2])


def test_accounting_gate_pass_and_fail():
    ok_out = {"ledger": _clean(2), "diagnostics": [(), ("snow_overflow",)]}
    ok, reasons = accounting_gate(deviation_budget(ok_out))
    assert ok and reasons == []                          # diagnostics don't fail the gate

    leaky = {"ledger": [_ledger(0.0, 0.0, 0.0, 0.5)], "diagnostics": [()]}  # residual 0.5
    ok2, reasons2 = accounting_gate(deviation_budget(leaky))
    assert not ok2 and reasons2                          # residual breaks P0 gate


def test_budget_requires_audit_keys():
    with pytest.raises(LedgerError):
        deviation_budget({"ledger": _clean(1)})          # missing diagnostics
    with pytest.raises(LedgerError):                     # unknown code
        deviation_budget({"ledger": _clean(1), "diagnostics": [("bogus_code",)]})


def test_budget_rejects_empty_rollout():
    with pytest.raises(LedgerError):                     # empty = cannot evaluate, not PASS
        deviation_budget({"ledger": [], "diagnostics": []})


def test_accounting_gate_rejects_nan_atol():
    b = deviation_budget({"ledger": _clean(1), "diagnostics": [()]})
    with pytest.raises(LedgerError):                     # NaN atol would false-PASS
        accounting_gate(b, residual_atol=float("nan"))
    with pytest.raises(LedgerError):
        accounting_gate(b, residual_atol=-1.0)


def test_budget_rejects_nonfinite_storage_jump():
    out = {"ledger": _clean(2), "diagnostics": [(), ()], "Water": [0.0, float("nan")]}
    with pytest.raises(LedgerError):
        deviation_budget(out)


def test_budget_rejects_bad_diagnostic_step_shape():
    with pytest.raises(LedgerError):                     # a step must be normalizable
        deviation_budget({"ledger": _clean(1), "diagnostics": [None]})
    with pytest.raises(LedgerError):                     # unknown code per step
        deviation_budget({"ledger": _clean(1), "diagnostics": [("nope",)]})


def test_budget_rejects_non_mapping_and_bad_shapes():
    with pytest.raises(LedgerError):
        deviation_budget(None)                           # not a mapping
    with pytest.raises(LedgerError):
        deviation_budget([1, 2, 3])
    with pytest.raises(LedgerError):                     # ledger/diagnostics not sized
        deviation_budget({"ledger": 5, "diagnostics": 5})


def test_budget_rejects_storage_length_mismatch():
    out = {"ledger": _clean(3), "diagnostics": [(), (), ()],
           "Water": [0.0, 0.1]}                          # len 2 != n_steps 3
    with pytest.raises(LedgerError):
        deviation_budget(out)


def test_budget_rejects_len1_nonfinite_storage():
    out = {"ledger": _clean(1), "diagnostics": [()], "Snow": [float("inf")]}
    with pytest.raises(LedgerError):                     # length-1 non-finite must fail
        deviation_budget(out)


def test_accounting_gate_rejects_nonfinite_summary():
    with pytest.raises(LedgerError):
        accounting_gate({"max_primary_residual": float("nan")})


def test_budget_rejects_top_level_diagnostics_mapping():
    from droad.ledger import DIAG_SNOW_OVERFLOW
    with pytest.raises(LedgerError):                     # mapping, not per-step sequence
        deviation_budget({"ledger": _clean(1), "diagnostics": {DIAG_SNOW_OVERFLOW: 1}})


def test_csv_keeps_raw_precision():
    b = deviation_budget({"ledger": _clean(3),
                          "diagnostics": [(), ("ice_over_melt",), ()]}, case_id="c2")
    csv = budget_to_csv([b])
    # rate 1/3 must appear at full precision, not rounded to 0.3333
    assert repr(1 / 3) in csv or str(1 / 3) in csv
    md = budget_to_markdown([b])
    assert "0.3333" in md                                # markdown stays human-rounded


def test_serialization_rejects_bad_summary():
    with pytest.raises(LedgerError):
        budget_to_csv([{"case_id": "x"}])                # missing columns
    with pytest.raises(LedgerError):
        budget_to_markdown([{"case_id": "x"}])


def test_accounting_gate_rejects_bad_summary_container():
    with pytest.raises(LedgerError):
        accounting_gate(None)                            # not a mapping
    with pytest.raises(LedgerError):
        accounting_gate({})                              # missing max_primary_residual


def test_budget_rejects_set_and_string_containers():
    with pytest.raises(LedgerError):                     # top-level diagnostics as set
        deviation_budget({"ledger": _clean(1), "diagnostics": {("ice_over_melt",)}})
    with pytest.raises(LedgerError):                     # ledger as string
        deviation_budget({"ledger": "xx", "diagnostics": [(), ()]})


def test_budget_rejects_mapping_storage_trajectory():
    with pytest.raises(LedgerError):
        deviation_budget({"ledger": _clean(2), "diagnostics": [(), ()],
                          "Water": {0: 0.0, 1: 0.1}})    # mapping, not sequence


def test_storage_element_rejects_bool_and_string():
    for bad in ([0.0, "1.0"], [False, True]):
        with pytest.raises(LedgerError):                 # str/bool element rejected like ledger fields
            deviation_budget({"ledger": _clean(2), "diagnostics": [(), ()], "Water": bad})


def test_max_storage_jump_provenance():
    out = {"ledger": _clean(4), "diagnostics": [(), (), (), ()],
           "Snow": [0.0, 0.1, 0.05, 0.05], "Water": [0.0, 0.0, 0.3, 0.0]}
    b = deviation_budget(out)
    assert b["max_storage_jump"] == pytest.approx(0.3)
    assert b["max_storage_jump_key"] == "Water"
    assert b["max_storage_jump_step"] == 2            # 0.0 -> 0.3 at index 2
    assert b["max_storage_jump_signed"] == pytest.approx(0.3)
    # a later drop 0.3 -> 0.0 has equal magnitude but comes after, so first wins
    assert b["max_storage_jump_signed"] > 0


def test_csv_and_markdown_include_per_code_breakdown():
    out = {"ledger": _clean(2), "diagnostics": [("snow_over_melt",), ("water_overflow",)]}
    b = deviation_budget(out, case_id="c")
    csv = budget_to_csv([b])
    header = csv.splitlines()[0]
    assert "diag_snow_over_melt" in header and "diag_water_overflow" in header
    assert "max_storage_jump_key" in header
    md = budget_to_markdown([b])
    assert "Diagnostic breakdown" in md and "snow_over_melt" in md


def test_serialization_rejects_nonfinite_numeric_column():
    b = deviation_budget({"ledger": _clean(1), "diagnostics": [()]})
    b = dict(b); b["max_storage_jump"] = float("nan")    # forge a bad numeric column
    with pytest.raises(LedgerError):
        budget_to_csv([b])
    with pytest.raises(LedgerError):
        budget_to_markdown([b])


def test_budget_serialization():
    b = deviation_budget({"ledger": _clean(3),
                          "diagnostics": [(), ("ice_over_melt",), ()]}, case_id="c2")
    csv = budget_to_csv([b])
    assert csv.splitlines()[0].startswith("case_id,")
    assert "c2" in csv
    md = budget_to_markdown([b])
    assert "P0 accounting gate" in md and "PASS" in md
