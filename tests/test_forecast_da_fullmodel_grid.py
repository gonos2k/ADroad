"""A0 full-model grid: pure aggregation/ranking logic (fast) + jax-marked smoke."""
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from tools.report_forecast_da_fullmodel_multi import _case_row  # noqa: E402
from tools.report_forecast_da_fullmodel_grid import (  # noqa: E402
    summarize_combo, rank_rows, render, _COLS, COMBOS,
)


def _result(k0, da_rmse, bg_rmse, gate_ok, physics_worse, lead_da=0.0, lead_bg=0.0,
            dx_l2=0.5, dx_max=0.4, resid=0.0):
    def _dev(rate, r):
        return {"diagnostic_steps_rate": rate, "max_primary_residual": r,
                "over_melt_count": 0, "overflow_count": 0}
    return {"k0": k0, "bg": ({"rmse": bg_rmse}, _dev(lead_bg, resid)),
            "da": ({"rmse": da_rmse}, _dev(lead_da, resid)),
            "win": (_dev(0.0, 0.0), _dev(0.0, 0.0)),
            "rmse_delta_da_minus_bg": da_rmse - bg_rmse,
            "gate_da_vs_bg": (gate_ok, [] if gate_ok else ["x"]),
            "physics_worse": physics_worse, "dx_l2": dx_l2, "dx_max_abs": dx_max}


def _win(k0, da, bg, ok, pw, **kw):
    return _case_row(_result(k0, da, bg, ok, pw, **kw))


def test_summarize_combo_computes_rates():
    rows = [_win(1500, 0.20, 0.22, True, False), _win(2100, 0.30, 0.22, False, True, lead_da=0.3)]
    c = summarize_combo(0.05, 120, 480, rows)
    assert c["bg_w"] == 0.05 and c["window"] == 120 and c["lead"] == 480 and c["n_valid"] == 2
    assert c["gate_pass_rate"] == 0.5 and c["skill_improved_rate"] == 0.5
    assert c["physics_worse_rate"] == 0.5 and c["state_large_rate"] == 0.0
    assert c["worst_delta_rmse"] == pytest.approx(0.08)     # max(-0.02, +0.08)
    assert c["promotion_eligible"] is False                 # single-fixture grid: always


def test_grid_promotion_eligible_always_false_even_when_windows_clean():
    # a 4/4-clean combo: window precondition met, but promotion_eligible must STAY False
    # (single fixture ≠ independent-case promotion, per design §11).
    rows = [_win(k, 0.18, 0.22, True, False) for k in (1500, 2100, 2700, 3300)]
    c = summarize_combo(0.2, 60, 480, rows)
    assert c["gate_pass_rate"] == 1.0 and c["residual_clean"] is True
    assert c["window_precondition_met"] is True             # all_beat ∧ residual_clean
    assert c["promotion_eligible"] is False                 # ...but still not promotable


def test_summarize_combo_empty_uses_none_not_inf():
    # empty combo must NOT emit inf/nan (json.dumps allow_nan=False would reject them).
    c = summarize_combo(0.2, 60, 240, [])
    assert c["n_valid"] == 0 and c["promotion"] == "REPORT_ONLY"
    assert c["promotion_eligible"] is False and c["gate_pass_rate"] == 0.0
    assert c["worst_delta_rmse"] is None and c["mean_delta_rmse"] is None
    assert c["max_residual"] is None
    json.dumps(c, allow_nan=False)                          # must be JSON-serialisable


def test_rank_prefers_clean_physics_over_average():
    # A: perfect. B: same gate_pass but has physics_worse. C: lower gate_pass.
    a = summarize_combo(0.2, 60, 480, [_win(1500, 0.18, 0.22, True, False),
                                       _win(2100, 0.18, 0.22, True, False)])
    b = summarize_combo(0.05, 60, 480, [_win(1500, 0.18, 0.22, True, False),
                                        _win(2100, 0.17, 0.22, True, True, lead_da=0.3)])
    c = summarize_combo(0.01, 120, 240, [_win(1500, 0.30, 0.22, False, False),
                                         _win(2100, 0.18, 0.22, True, False)])
    ranked = rank_rows([c, b, a])
    assert ranked[0]["bg_w"] == 0.2               # clean + full pass wins
    assert ranked.index(next(r for r in ranked if r["bg_w"] == 0.05)) < \
           ranked.index(next(r for r in ranked if r["bg_w"] == 0.01))  # physics-clean beats lower-pass?
    # b has gate_pass_rate 1.0 (physics_worse is report, gate_pass from _result), c has 0.5 -> b ranks above c
    assert ranked[-1]["bg_w"] == 0.01


def test_dirty_residual_blocks_precondition():
    rows = [_win(1500, 0.18, 0.22, True, False, resid=1e-6)]
    c = summarize_combo(0.05, 120, 480, rows)
    assert c["residual_clean"] is False
    assert c["window_precondition_met"] is False and c["promotion_eligible"] is False


def test_rank_puts_empty_combo_last():
    good = summarize_combo(0.2, 60, 480, [_win(1500, 0.18, 0.22, True, False)])
    empty = summarize_combo(0.05, 120, 240, [])          # None worst_delta -> sorts last
    assert rank_rows([empty, good])[0]["bg_w"] == 0.2
    assert rank_rows([empty, good])[-1]["n_valid"] == 0


def test_load_partial_rejects_incomplete_grid_row(tmp_path, monkeypatch):
    import tools.report_forecast_da_fullmodel_grid as mod
    p = tmp_path / "grid_partial.json"
    monkeypatch.setattr(mod, "PARTIAL", p)
    # schema/config ok but row missing most keys -> ignored (would crash render/rank)
    p.write_text(json.dumps({"schema_version": mod._SCHEMA, "config": mod._CONFIG,
                             "rows": [{"key": "x", "bg_w": 0.05}]}))
    assert mod._load_partial() == {}
    # a complete row loads
    good = summarize_combo(0.05, 120, 480, [_win(1500, 0.18, 0.22, True, False)])
    p.write_text(json.dumps({"schema_version": mod._SCHEMA, "config": mod._CONFIG, "rows": [good]}))
    assert set(mod._load_partial().keys()) == {good["key"]}


def test_render_meta_allow_nan_false(tmp_path, monkeypatch):
    import tools.report_forecast_da_fullmodel_grid as mod
    monkeypatch.setattr(mod, "REPO", tmp_path)
    (tmp_path / "reports").mkdir()
    rows = [_win(1500, 0.18, 0.22, True, False), _win(2100, 0.19, 0.22, True, False)]
    combo = summarize_combo(0.05, 120, 480, rows); combo["windows"] = rows
    empty = summarize_combo(0.2, 60, 240, []); empty["windows"] = []   # None fields must survive
    mod.render({combo["key"]: combo, empty["key"]: empty})             # allow_nan=False, must not raise
    meta = json.loads((tmp_path / "reports" / "forecast_da_fullmodel_grid_meta.json").read_text())
    by_key = {(g["bg_w"], g["window"], g["lead"]): g for g in meta["grid"]}
    assert by_key[(0.05, 120, 480)]["summary"]["gate_pass_rate"] == 1.0
    assert by_key[(0.2, 60, 240)]["summary"]["worst_delta_rmse"] is None   # None survived JSON
    assert len(by_key[(0.05, 120, 480)]["windows"]) == 2


def test_render_only_empty_combo_does_not_crash(tmp_path, monkeypatch):
    # a combo where every window was skipped -> None deltas; render()/format must not crash.
    import tools.report_forecast_da_fullmodel_grid as mod
    monkeypatch.setattr(mod, "REPO", tmp_path)
    (tmp_path / "reports").mkdir()
    empty = summarize_combo(0.2, 60, 240, []); empty["windows"] = []
    mod.render({empty["key"]: empty})                    # must not raise (None worst_delta)
    md = (tmp_path / "reports" / "forecast_da_fullmodel_grid.md").read_text()
    assert "worst_delta=NA" in md


def test_load_partial_rejects_value_corrupt_grid_row(tmp_path, monkeypatch):
    import tools.report_forecast_da_fullmodel_grid as mod
    p = tmp_path / "grid_partial.json"
    monkeypatch.setattr(mod, "PARTIAL", p)
    good = summarize_combo(0.05, 120, 480, [_win(1500, 0.18, 0.22, True, False)])
    bad_bool = dict(good, residual_clean="yes")          # non-bool flag
    p.write_text(json.dumps({"schema_version": mod._SCHEMA, "config": mod._CONFIG,
                             "rows": [bad_bool]}))
    assert mod._load_partial() == {}


def test_cols_present_in_row():
    c = summarize_combo(0.05, 120, 480, [_win(1500, 0.18, 0.22, True, False)])
    assert set(_COLS).issubset(c.keys())
    assert len(COMBOS) == 12


@pytest.mark.jax
def test_grid_smoke_one_combo(tmp_path, monkeypatch):
    # one combo over a SHORT k0 list with tiny window/lead — end-to-end through run_windows.
    import tools.report_forecast_da_fullmodel_multi as multi
    from tools.report_forecast_da_fullmodel_grid import summarize_combo
    rows = multi.run_windows(window=30, lead=60, bg_w=0.05, k0s=[2000])
    c = summarize_combo(0.05, 30, 60, rows)
    assert c["n_valid"] == 1 and isinstance(c["gate_pass_rate"], float)
    assert c["residual_clean"] is True            # full-model audit clean
