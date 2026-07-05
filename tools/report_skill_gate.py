#!/usr/bin/env python3
"""Baseline forecast skill gate on the example case (single-fixture skeleton).

Compares the default dROAD free-running forecast against a 1-step persistence
baseline over a holdout window of valid troad observations, and gates the model
on skill + accounting residual + diagnostic burden (deviation budget).

    python3 tools/report_skill_gate.py

Writes reports/skill_gate_baseline.md and .csv. Multi-case / DA / smooth_compat
rows plug into the same report later.
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from tools.run_no_coupling import build_model, RSP_SRC  # noqa: E402
from tools.report_deviation_budget import _phy, _day, _cp  # noqa: E402
from droad.driver import full_rollout  # noqa: E402
from droad.storage import Surf  # noqa: E402
from droad.deviation import deviation_budget  # noqa: E402
from droad.skill_gate import (  # noqa: E402
    forecast_metrics, skill_gate, skill_report_markdown, skill_report_csv,
)

HOLDOUT = 2000        # last N valid-obs steps used as the forecast holdout


def _run_default():
    sys.path.insert(0, str(RSP_SRC))
    m, objs = build_model()
    mi, mo, phy, g, s, a, coup, st, cpm, _ = objs
    n = st.SimLen - 1
    hours = np.array([t.hour for t in mi.time[:n]], float)
    prec_in = np.array(mi.prec, float) / 3600.0 * st.DTSecs
    surf0 = Surf(SrfWat=s.SrfWatmms, SrfSnow=s.SrfSnowmms, SrfIce=s.SrfIcemms,
                 SrfIce2=s.SrfIce2mms, SrfDep=s.SrfDepmms, TsurfAve=s.TsurfAve,
                 Q2Melt=s.Q2Melt, T4Melt=s.T4Melt, WearSurf=s.WearSurf,
                 SnowType=a.SnowType, WetSnowFrozen=cpm.WetSnowFrozen, VeryCold=s.VeryCold)
    out = full_rollout(
        Tair=np.array(mi.Tair, float), VZ=np.array(mi.VZ, float), Rhz=np.array(mi.Rhz, float),
        SW=np.array(mi.SW, float), LW=np.array(mi.LW, float),
        TSurfObs=np.array(mi.TSurfObs, float), hours=hours,
        prec_phase=np.array(mi.PrecPhase, float), prec_in_tstep=prec_in,
        Tmp0=g.Tmp, TmpNw0=g.TmpNw, WCont=np.array(g.WCont, float),
        CC=np.array(g.CC, float), ZDpth=np.array(g.ZDpth, float),
        DyK=np.array(g.DyK, float), DyC=np.array(g.DyC, float),
        surf0=surf0, Albedo0=g.Albedo, BLCond0=a.BLCond,
        NLayers=st.NLayers, DTSecs=st.DTSecs, MaxPormms=phy.MaxPormms, Tph=st.Tph,
        InitLenI=st.InitLenI, phy=_phy(phy), day=_day(st), cp=_cp(cpm), n_steps=n,
        TsurfObsLast=coup.LastTsurfObs, return_ledger=True)
    tso = np.array(mi.TSurfObs, float)[:n]
    return out, tso


def build_rows():
    out, tso = _run_default()
    tsurf = np.asarray(out["Tsurf"], float)

    valid = tso > -100.0                       # holdout = last HOLDOUT valid-obs steps
    idx = np.flatnonzero(valid)
    idx = idx[-min(HOLDOUT, len(idx)):]
    obs = tso[idx]
    default_pred = tsurf[idx]
    # forecast-lead persistence: hold the analysis-time obs (window start) constant
    # over the whole lead. (1-step persistence is meaningless at 30 s resolution —
    # RMSE ~0.006 — so it can't gate anything; N-step lead is the honest baseline.)
    obs_eval, default_eval = obs[1:], default_pred[1:]
    const_eval = np.full_like(obs_eval, obs[0])

    dev = deviation_budget(out, case_id="default")

    m_default = forecast_metrics(default_eval, obs_eval)
    m_const = forecast_metrics(const_eval, obs_eval)
    ok, reasons = skill_gate(m_default, m_const, deviation=dev)

    def row(model, m, gate):
        return {"model": model, "n": m["n"], "rmse": m["rmse"], "mae": m["mae"],
                "freeze_thaw_accuracy": m["freeze_thaw_accuracy"],
                "cold_n": m["cold_n"], "cold_rmse": m["cold_rmse"], "gate": gate}

    rows = [row("constant_initial", m_const, "baseline"),
            row("default", m_default, "PASS" if ok else "FAIL — " + "; ".join(reasons))]
    return rows, dev


def main():
    rows, dev = build_rows()
    outdir = REPO / "reports"
    outdir.mkdir(exist_ok=True)
    md = skill_report_markdown(rows, "Forecast Skill Gate — default vs constant_initial baseline")
    md += ("\nbaseline = constant_initial(분석시각 obs를 lead 전체에 고정). 1-step persistence는 "
           "30s 해상도에서 RMSE~0.006으로 자명해 gate baseline으로 부적합. gate: RMSE만 hard.\n")
    md += (f"\n## Accounting / deviation (default)\n"
           f"- max_primary_residual: {dev['max_primary_residual']:.3e} "
           f"({'PASS' if dev['max_primary_residual'] < 1e-9 else 'FAIL'})\n"
           f"- diagnostic_steps_rate: {dev['diagnostic_steps_rate']:.4f}\n"
           f"- over_melt_count / overflow_count: {dev['over_melt_count']} / {dev['overflow_count']}\n")
    (outdir / "skill_gate_baseline.md").write_text(md, encoding="utf-8")
    (outdir / "skill_gate_baseline.csv").write_text(skill_report_csv(rows), encoding="utf-8")
    print("wrote reports/skill_gate_baseline.{md,csv}")
    for r in rows:
        print(f"  {r['model']:12s} rmse={r['rmse']:.4f} mae={r['mae']:.4f} "
              f"ft_acc={r['freeze_thaw_accuracy']:.4f} gate={r['gate']}")


if __name__ == "__main__":
    main()
