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
    assert c["promotion_eligible"] is False


def test_summarize_combo_empty_is_report_only():
    c = summarize_combo(0.2, 60, 240, [])
    assert c["n_valid"] == 0 and c["promotion"] == "REPORT_ONLY"
    assert c["promotion_eligible"] is False and c["gate_pass_rate"] == 0.0


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


def test_dirty_residual_blocks_promotion_eligible():
    rows = [_win(1500, 0.18, 0.22, True, False, resid=1e-6)]
    c = summarize_combo(0.05, 120, 480, rows)
    assert c["residual_clean"] is False and c["promotion_eligible"] is False


def test_render_meta_allow_nan_false(tmp_path, monkeypatch):
    import tools.report_forecast_da_fullmodel_grid as mod
    monkeypatch.setattr(mod, "REPO", tmp_path)
    (tmp_path / "reports").mkdir()
    rows = [_win(1500, 0.18, 0.22, True, False), _win(2100, 0.19, 0.22, True, False)]
    combo = summarize_combo(0.05, 120, 480, rows); combo["windows"] = rows
    mod.render({combo["key"]: combo})
    meta = json.loads((tmp_path / "reports" / "forecast_da_fullmodel_grid_meta.json").read_text())
    assert meta["grid"][0]["summary"]["gate_pass_rate"] == 1.0
    assert "windows" in meta["grid"][0] and len(meta["grid"][0]["windows"]) == 2


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
