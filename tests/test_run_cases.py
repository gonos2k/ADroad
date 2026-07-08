"""run_cases (Step 4 promotion path) — pure aggregation + driver wiring (no jax)."""
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from tools.run_cases import (  # noqa: E402
    summarize_cases, run_manifest, make_setting, _run_one_not_implemented, CASE_FIELDS,
    case_row_from_a0,
)


def _a0(gate_ok=True, physics_worse=False, dx_l2=0.5, dx_max=0.4, delta=-0.02,
        resid_bg=0.0, resid_da=0.0):
    return {"gate_da_vs_bg": (gate_ok, [] if gate_ok else ["x"]), "physics_worse": physics_worse,
            "dx_l2": dx_l2, "dx_max_abs": dx_max, "rmse_delta_da_minus_bg": delta,
            "bg": ({"rmse": 0.22}, {"max_primary_residual": resid_bg}),
            "da": ({"rmse": 0.20}, {"max_primary_residual": resid_da})}


def _row(cid="c", regime="dry_cold", gate_pass=True, physics_worse=False,
         state_large=False, rmse_delta=-0.02, max_residual=0.0):
    return {"case_id": cid, "regime": regime, "gate_pass": gate_pass,
            "physics_worse": physics_worse, "state_large": state_large,
            "rmse_delta": rmse_delta, "max_residual": max_residual}


def _mcase(cid, station, day, regime="dry_cold"):
    return {"case_id": cid, "station": station, "regime": regime,
            "start": f"2026-01-{day:02d}T00:00:00", "end": f"2026-01-{day:02d}T06:00:00"}


def test_make_setting_validates():
    assert make_setting(0.05, 60, 480) == {"bg_w": 0.05, "window": 60, "lead": 480}
    for bad in [(0.0, 60, 480), (0.05, 0, 480), (True, 60, 480),
                (float("inf"), 60, 480), (0.05, 60.5, 480), (0.05, 60, 480.5),
                (0.05, float("inf"), 480), (0.05, 60, float("nan"))]:
        with pytest.raises(ValueError):
            make_setting(*bad)


def test_three_cases_all_pass_can_PROMOTE():
    # THE point of Step 4: with real n_cases>=3 and every case beating baseline, PROMOTE.
    rows = [_row(cid="a"), _row(cid="b"), _row(cid="c")]
    s = summarize_cases(rows)
    assert s["n_cases"] == 3 and s["all_beat"] is True
    assert s["promotion"][0] == "PROMOTE"


def test_one_case_failing_stays_report_only():
    rows = [_row(cid="a"), _row(cid="b", gate_pass=False), _row(cid="c")]
    s = summarize_cases(rows)
    assert s["all_beat"] is False and s["promotion"][0] == "REPORT_ONLY"
    assert any("every window" in r for r in s["promotion"][1])


def test_too_few_cases_stays_report_only():
    s = summarize_cases([_row(cid="a"), _row(cid="b")])          # 2 < 3
    assert s["promotion"][0] == "REPORT_ONLY"
    assert any("insufficient cases" in r for r in s["promotion"][1])


def test_dirty_residual_blocks_promotion():
    rows = [_row(cid="a"), _row(cid="b"), _row(cid="c", max_residual=1e-6)]
    s = summarize_cases(rows)
    assert s["residual_clean"] is False and s["promotion"][0] == "REPORT_ONLY"
    assert any("aggregate residual" in r for r in s["promotion"][1])


def test_summarize_rejects_duplicate_case_id():
    with pytest.raises(ValueError):
        summarize_cases([_row(cid="dup"), _row(cid="dup"), _row(cid="x")])


def test_case_row_schema_and_consistency_enforced():
    with pytest.raises(ValueError):
        summarize_cases([{"case_id": "a"}])                      # missing fields
    with pytest.raises(ValueError):
        summarize_cases([dict(_row(), gate_pass="yes")])        # non-bool flag
    with pytest.raises(ValueError):                             # physics_worse yet gate_pass
        summarize_cases([dict(_row(), physics_worse=True, gate_pass=True)])
    with pytest.raises(ValueError):                             # non-finite delta
        summarize_cases([dict(_row(), rmse_delta=float("nan"))])
    with pytest.raises(ValueError):                             # negative residual
        summarize_cases([dict(_row(), max_residual=-1.0)])
    with pytest.raises(ValueError):                             # numeric case_id
        summarize_cases([dict(_row(), case_id=123)])
    with pytest.raises(ValueError):                             # empty regime
        summarize_cases([dict(_row(), regime="  ")])


def test_run_manifest_wires_validate_loop_and_gate():
    cases = [_mcase("a", "sa", 1), _mcase("b", "sb", 2, "warm_wet"),
             _mcase("c", "sc", 3, "precip_snow")]
    calls = []

    def fake_run_one(case, setting):
        calls.append((case["case_id"], setting["bg_w"]))
        return _row(cid=case["case_id"], regime=case["regime"])

    summary, rows = run_manifest({"cases": cases}, make_setting(0.05, 60, 480),
                                 run_one=fake_run_one)
    assert len(rows) == 3 and len(calls) == 3
    assert summary["n_cases"] == 3 and summary["promotion"][0] == "PROMOTE"


def test_run_manifest_refuses_invalid_or_thin_manifest():
    with pytest.raises(ValueError):                             # invalid schema
        run_manifest({"cases": [{"case_id": "x"}]}, make_setting(0.05, 60, 480),
                     run_one=lambda c, s: _row())
    thin = {"cases": [_mcase("a", "sa", 1)]}                    # schema-clean but <3 cases
    with pytest.raises(ValueError):
        run_manifest(thin, make_setting(0.05, 60, 480), run_one=lambda c, s: _row())


def test_run_manifest_rejects_mismatched_case_id_or_regime():
    cases = [_mcase("a", "sa", 1), _mcase("b", "sb", 2, "warm_wet"),
             _mcase("c", "sc", 3, "precip_snow")]
    with pytest.raises(ValueError):                              # loader returns wrong case_id
        run_manifest({"cases": cases}, make_setting(0.05, 60, 480),
                     run_one=lambda c, s: _row(cid="WRONG", regime=c["regime"]))
    with pytest.raises(ValueError):                              # loader returns wrong regime
        run_manifest({"cases": cases}, make_setting(0.05, 60, 480),
                     run_one=lambda c, s: _row(cid=c["case_id"], regime="melt_refreeze"))


def test_run_manifest_rejects_bad_require_value():
    cases = [_mcase("a", "sa", 1), _mcase("b", "sb", 2, "warm_wet"),
             _mcase("c", "sc", 3, "precip_snow")]
    with pytest.raises(ValueError):
        run_manifest({"cases": cases}, make_setting(0.05, 60, 480),
                     run_one=lambda c, s: _row(), require="none")


def test_cli_returns_error_code_not_traceback(tmp_path, capsys):
    yaml = pytest.importorskip("yaml")
    from tools.run_cases import main
    p = tmp_path / "thin.yaml"
    p.write_text(yaml.safe_dump({"cases": [_mcase("a", "sa", 1)]}))   # schema-clean but <3 cases
    # invalid/thin manifest -> ValueError is caught, printed, exit 1 (no traceback)
    assert main([str(p), "--bg-w", "0.05", "--window", "60", "--lead", "480"]) == 1
    assert "ERROR" in capsys.readouterr().err


def test_cli_bad_setting_returns_error_code_not_traceback(tmp_path, capsys):
    yaml = pytest.importorskip("yaml")
    from tools.run_cases import main
    p = tmp_path / "cases.yaml"
    p.write_text(yaml.safe_dump({"cases": [_mcase("a", "sa", 1),
                                           _mcase("b", "sb", 2, "warm_wet"),
                                           _mcase("c", "sc", 3, "precip_snow")]}))
    # bad setting (bg_w=0) must be caught in main() -> ERROR + exit 1, not a traceback
    assert main([str(p), "--bg-w", "0", "--window", "60", "--lead", "480"]) == 1
    assert "ERROR" in capsys.readouterr().err


def test_cli_missing_manifest_returns_error_code_not_traceback(capsys):
    from tools.run_cases import main
    assert main(["/no/such/file.yaml", "--bg-w", "0.05", "--window", "60", "--lead", "480"]) == 1
    assert "ERROR" in capsys.readouterr().err


def test_cli_bad_yaml_returns_error_code_not_traceback(tmp_path, capsys):
    pytest.importorskip("yaml")
    from tools.run_cases import main
    p = tmp_path / "bad.yaml"
    p.write_text("cases: [")                                    # malformed YAML
    assert main([str(p), "--bg-w", "0.05", "--window", "60", "--lead", "480"]) == 1
    assert "ERROR" in capsys.readouterr().err


def test_case_row_from_a0_maps_and_validates():
    case = {"case_id": "s_2026_dry_cold", "regime": "dry_cold"}
    row = case_row_from_a0(case, _a0(gate_ok=True, physics_worse=False, delta=-0.03,
                                     resid_bg=0.0, resid_da=1e-12))
    assert row["case_id"] == "s_2026_dry_cold" and row["regime"] == "dry_cold"
    assert row["gate_pass"] is True and row["physics_worse"] is False
    assert row["rmse_delta"] == pytest.approx(-0.03)
    assert row["max_residual"] == pytest.approx(1e-12)
    assert set(CASE_FIELDS).issubset(row.keys())
    # feeds straight into the aggregator
    assert "n_cases" in summarize_cases([row])


def test_case_row_from_a0_flags_state_large_and_stays_consistent():
    r = case_row_from_a0({"case_id": "x", "regime": "warm_wet"},
                         _a0(gate_ok=False, physics_worse=True, dx_max=3.5, dx_l2=40.0))
    assert r["state_large"] is True and r["physics_worse"] is True and r["gate_pass"] is False
    with pytest.raises(ValueError):                             # missing case metadata
        case_row_from_a0({"case_id": "x"}, _a0())


def test_case_row_from_a0_rejects_corrupt_a0():
    case = {"case_id": "x", "regime": "dry_cold"}
    with pytest.raises(ValueError):                             # NaN da residual not hidden by max()
        case_row_from_a0(case, _a0(resid_da=float("nan")))
    with pytest.raises(ValueError):                             # negative residual
        case_row_from_a0(case, _a0(resid_bg=-1.0))
    with pytest.raises(ValueError):                             # non-bool gate flag ('False' truthy)
        case_row_from_a0(case, dict(_a0(), gate_da_vs_bg=("False", [])))
    with pytest.raises(ValueError):                             # non-finite dx
        case_row_from_a0(case, _a0(dx_max=float("nan")))
    with pytest.raises(ValueError):                             # malformed shape -> wrapped
        case_row_from_a0(case, {"physics_worse": False})


@pytest.mark.jax
def test_case_row_from_a0_smoke_on_real_build_a0():
    # end-to-end: real A0 output -> case row -> aggregator, proving the extraction contract
    # holds against actual build_a0 (single fixture -> n_cases=1 -> REPORT_ONLY).
    from tools.report_forecast_da_fullmodel import build_a0
    a0 = build_a0(k0=2000, window=30, lead=60)
    row = case_row_from_a0({"case_id": "fixture_k2000", "regime": "dry_cold"}, a0)
    assert row["max_residual"] < 1e-8                          # full-model audit clean
    s = summarize_cases([row])
    assert s["n_cases"] == 1 and s["promotion"][0] == "REPORT_ONLY"


def test_default_run_one_is_not_implemented():
    with pytest.raises(NotImplementedError):
        _run_one_not_implemented({"case_id": "a"}, make_setting(0.05, 60, 480))
    assert set(CASE_FIELDS)                                     # contract documented
