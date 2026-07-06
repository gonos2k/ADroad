"""Grid aggregation/ranking: pure combo summary + ranking (no model run)."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from tools.grid_forecast_da import summarize_combo, rank_rows  # noqa: E402


def _r(delta, dxl2, beats, da=0.4, bg=0.5, degr=1.0):
    return {"gate_da_vs_bg": (beats, [] if beats else ["worse"]),
            "rmse_delta_da_minus_bg": delta, "dx_l2": dxl2,
            "da": ({"rmse": da}, 0.2), "bg": ({"rmse": bg}, 0.3), "degradation_da": degr}


def test_summarize_combo_aggregates():
    row = summarize_combo(0.05, 120, 480, [_r(-0.01, 0.5, True), _r(0.02, 0.8, False)])
    assert row["key"] == "0.05_120_480"
    assert row["n_valid"] == 2 and row["wins"] == 1 and row["win_rate"] == 0.5
    assert row["worst_delta"] == 0.02          # max (worst) Δrmse
    assert row["max_dx_l2"] == 0.8
    assert abs(row["mean_delta"] - 0.005) < 1e-9


def test_summarize_combo_empty():
    row = summarize_combo(0.01, 60, 240, [])
    assert row["n_valid"] == 0 and row["win_rate"] == 0.0


def test_rank_prefers_win_rate_then_mean_delta():
    a = summarize_combo(0.05, 120, 480, [_r(-0.02, 0.5, True), _r(-0.01, 0.5, True)])   # win_rate 1.0
    b = summarize_combo(0.20, 60, 240, [_r(-0.05, 0.5, True), _r(0.03, 0.5, False)])    # win_rate 0.5
    ranked = rank_rows([b, a])
    assert ranked[0]["key"] == a["key"]        # higher win_rate first despite b's lower mean_delta
    # tie on win_rate -> lower mean_delta wins
    c = summarize_combo(0.01, 60, 240, [_r(-0.04, 0.5, True), _r(-0.03, 0.5, True)])
    ranked2 = rank_rows([a, c])
    assert ranked2[0]["key"] == c["key"]       # c mean_delta -0.035 < a -0.015
