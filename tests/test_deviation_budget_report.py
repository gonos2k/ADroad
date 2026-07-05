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


def test_budget_serialization():
    b = deviation_budget({"ledger": _clean(3),
                          "diagnostics": [(), ("ice_over_melt",), ()]}, case_id="c2")
    csv = budget_to_csv([b])
    assert csv.splitlines()[0].startswith("case_id,")
    assert "c2" in csv
    md = budget_to_markdown([b])
    assert "P0 accounting gate" in md and "PASS" in md
