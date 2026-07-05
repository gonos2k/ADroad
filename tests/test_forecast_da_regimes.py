"""Regime analysis aggregation: pure win/lose grouping + feature separation ranking
(no model run needed — synthetic build_multi-shaped results)."""
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from tools.analyze_forecast_da_regimes import (  # noqa: E402
    _features, summarize_regimes, group_separators,
)


def _synth(k0, beats, delta, bg_init, obs_std, dx_l2=0.5):
    return {"k0": k0, "gate_da_vs_bg": (beats, [] if beats else ["worse"]),
            "rmse_delta_da_minus_bg": delta, "dx_l2": dx_l2, "dx_max_abs": dx_l2,
            "da": (None, 0.2), "bg": (None, bg_init),
            "train_delta_da_minus_bg": 0.2 - bg_init, "degradation_da": 1.0, "degradation_bg": 1.0,
            "regime": {"tair_std": 1.0, "obs_std": obs_std, "bg_init_error": bg_init,
                       "dx_layers": [0.1, 0.2, 0.3, 0.4], "freeze_crossing_count": 0}}


def test_features_flattens_regime_and_dx_layers():
    f = _features(_synth(2000, True, -0.05, 0.35, 0.9))
    assert f["k0"] == 2000 and f["beats_bg"] is True
    assert f["bg_init_error"] == 0.35 and f["obs_std"] == 0.9
    assert f["dx_layer1"] == 0.1 and f["dx_layer4"] == 0.4   # dx_layers expanded
    assert f["train_bg"] == 0.35                             # from bg tuple


def test_summarize_ranks_separating_feature():
    # wins have LARGER background init error; that feature should surface as a top
    # separator with "higher in wins".
    results = [_synth(2700, True, -0.04, 0.50, 0.5),
               _synth(3300, True, -0.03, 0.55, 0.5),
               _synth(1500, False, 0.12, 0.10, 0.5),
               _synth(2100, False, 0.02, 0.12, 0.5)]
    rows, win, lose, table = summarize_regimes(results)
    assert len(win) == 2 and len(lose) == 2
    bg = next(t for t in table if t["feature"] == "bg_init_error")
    assert bg["win_mean"] > bg["lose_mean"] and bg["direction"] == "higher in wins"
    # obs_std is identical across groups -> near-zero separation, ranked below bg_init_error
    obs = next(t for t in table if t["feature"] == "obs_std")
    assert bg["separation"] > obs["separation"]
    # table is sorted by separation descending
    seps = [t["separation"] for t in table]
    assert seps == sorted(seps, reverse=True)


def test_group_separators_splits_families():
    results = [_synth(2700, True, -0.04, 0.50, 0.5), _synth(1500, False, 0.12, 0.10, 0.5)]
    _rows, _w, _l, table = summarize_regimes(results)
    grouped = group_separators(table)
    # ex-ante forcing must not contain endogenous DA-response features and vice versa
    ex_ante = {t["feature"] for t in grouped["ex_ante_forcing"]}
    da_resp = {t["feature"] for t in grouped["da_response"]}
    assert "tair_std" in ex_ante and "dx_l2" not in ex_ante
    assert "bg_init_error" in da_resp and "dx_layer1" in da_resp
    assert "obs_std" in {t["feature"] for t in grouped["post_hoc_obs"]}
    # every separator carries a family tag
    assert all("family" in t for t in table)


def test_summarize_rejects_empty():
    with pytest.raises(ValueError):
        summarize_regimes([])
