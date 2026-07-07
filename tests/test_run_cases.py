"""run_cases (Step 4 promotion path) — pure aggregation + driver wiring (no jax)."""
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from tools.run_cases import (  # noqa: E402
    summarize_cases, run_manifest, make_setting, _run_one_not_implemented, CASE_FIELDS,
)


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


def test_default_run_one_is_not_implemented():
    with pytest.raises(NotImplementedError):
        _run_one_not_implemented({"case_id": "a"}, make_setting(0.05, 60, 480))
    assert set(CASE_FIELDS)                                     # contract documented
