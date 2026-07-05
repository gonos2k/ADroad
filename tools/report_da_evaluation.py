#!/usr/bin/env python3
"""Diagnostics-aware DA evaluation (single-fixture skeleton).

Calibrates Emiss (+Tair bias) by variational DA on the differentiable JAX dry
model over an init window, then AUDITS the calibrated physics by running the
NumPy storage full_rollout with default vs. calibrated Emiss. The combined table
shows, per model: forecast skill (RMSE/MAE/freeze-thaw) AND physics burden
(accounting residual, diagnostic rate, over-melt/overflow) — so "DA lowered RMSE
but worsened over-melt" is visible in one place, not hidden.

    python3 tools/report_da_evaluation.py [--max-steps N]

Writes reports/da_evaluation.md and .csv. Requires jax/optax (dev extra).
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "examples"))
from tools.run_no_coupling import build_model, RSP_SRC  # noqa: E402
from tools.report_deviation_budget import _phy, _day, _cp  # noqa: E402
from droad.driver import full_rollout  # noqa: E402
from droad.storage import Surf  # noqa: E402
from droad.deviation import deviation_budget  # noqa: E402
from droad.skill_gate import forecast_metrics, skill_gate, diagnostics_delta  # noqa: E402


def calibrate_emiss(k0=2000, na=200, steps=300):
    """Constrained variational DA of Emiss (+Tair bias) on a JAX dry window."""
    sys.path.insert(0, str(RSP_SRC))
    import demo_da  # examples/demo_da.py — reuse its static/forcing builders
    from jax import config
    config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import jax.nn as jnn
    from droad import jax_model as jm
    from droad.assimilate import fit

    m, objs = build_model()
    mi, mo, phy, g, s, a, coup, st, cpm, _ = objs
    for i in range(k0):                                   # spin reference to k0 (physical background)
        m["InputOutput"].SetCurrentValues(i, mi, a, st, s, coup, g)
        m["Storage"].PrecipitationToStorage(st, cpm, mi.PrecPhase[i], a, s)
        m["BalanceModel"].BalanceModelOneStep(mi.SW[i], mi.LW[i], phy, g, s, a, st, coup, mi, i, cpm)
        wf = m["WearingFactors"].WearingFactors(); m["Cond"].WearFactors(cpm, st.Tph, s, wf)
        m["Cond"].RoadCond(phy.MaxPormms, s, a, st, cpm, wf); g.Albedo = m["Cond"].CalcAlbedo(s, cpm)

    static, bf = demo_da._static_forc(mi, g, st, slice(k0, k0 + na), np.zeros(na, bool))
    phy_d = demo_da._phy(phy)
    x0 = (jnp.array(g.Tmp, float), jnp.array(g.TmpNw, float), jnp.float64(a.BLCond))
    obs = jnp.array(np.array(mi.TSurfObs, float)[k0:k0 + na])
    w = jnp.ones(na)
    emap = lambda re: 0.85 + 0.15 * jnn.sigmoid(re)      # constrained to physical [0.85, 1.0]

    def loss(c):
        e = emap(c["re"])
        p = jm.dry_rollout({**phy_d, "Emiss": e}, x0, {**bf, "Tair": bf["Tair"] + c["bias"]}, static)
        return jnp.sum(w * (p - obs) ** 2) / jnp.sum(w) + 5.0 * (e - 0.95) ** 2 + 2.0 * c["bias"] ** 2

    est, _ = fit(loss, {"re": jnp.float64(0.0), "bias": jnp.float64(0.0)}, steps=steps, lr=0.03)
    return float(emap(est["re"])), float(phy.Emiss)


def _run(emiss, n, objs, mi, g, s, a, coup, st, cpm, phy):
    hours = np.array([t.hour for t in mi.time[:n]], float)
    prec_in = np.array(mi.prec, float) / 3600.0 * st.DTSecs
    surf0 = Surf(SrfWat=s.SrfWatmms, SrfSnow=s.SrfSnowmms, SrfIce=s.SrfIcemms,
                 SrfIce2=s.SrfIce2mms, SrfDep=s.SrfDepmms, TsurfAve=s.TsurfAve,
                 Q2Melt=s.Q2Melt, T4Melt=s.T4Melt, WearSurf=s.WearSurf,
                 SnowType=a.SnowType, WetSnowFrozen=cpm.WetSnowFrozen, VeryCold=s.VeryCold)
    phy_d = {**_phy(phy), "Emiss": emiss}
    return full_rollout(
        Tair=np.array(mi.Tair, float), VZ=np.array(mi.VZ, float), Rhz=np.array(mi.Rhz, float),
        SW=np.array(mi.SW, float), LW=np.array(mi.LW, float),
        TSurfObs=np.array(mi.TSurfObs, float), hours=hours,
        prec_phase=np.array(mi.PrecPhase, float), prec_in_tstep=prec_in,
        Tmp0=g.Tmp, TmpNw0=g.TmpNw, WCont=np.array(g.WCont, float),
        CC=np.array(g.CC, float), ZDpth=np.array(g.ZDpth, float),
        DyK=np.array(g.DyK, float), DyC=np.array(g.DyC, float),
        surf0=surf0, Albedo0=g.Albedo, BLCond0=a.BLCond,
        NLayers=st.NLayers, DTSecs=st.DTSecs, MaxPormms=phy.MaxPormms, Tph=st.Tph,
        InitLenI=st.InitLenI, phy=phy_d, day=_day(st), cp=_cp(cpm), n_steps=n,
        TsurfObsLast=coup.LastTsurfObs, return_ledger=True)


def build(max_steps=None):
    cal_emiss, def_emiss = calibrate_emiss()
    m, objs = build_model()
    mi, mo, phy, g, s, a, coup, st, cpm, _ = objs
    n = st.SimLen - 1 if max_steps is None else min(max_steps, st.SimLen - 1)

    out_def = _run(def_emiss, n, objs, mi, g, s, a, coup, st, cpm, phy)
    m2, o2 = build_model(); mi2, _, phy2, g2, s2, a2, coup2, st2, cpm2, _ = o2
    out_da = _run(cal_emiss, n, o2, mi2, g2, s2, a2, coup2, st2, cpm2, phy2)

    tso = np.array(mi.TSurfObs, float)[:n]
    idx = np.flatnonzero(tso > -100.0)
    idx = idx[-min(2000, len(idx)):]
    obs = tso[idx]
    obs_eval = obs[1:]
    persist_eval = np.full_like(obs_eval, obs[0])

    dev_def = deviation_budget(out_def, case_id="default")
    dev_da = deviation_budget(out_da, case_id=f"DA(Emiss={cal_emiss:.3f})")
    m_persist = forecast_metrics(persist_eval, obs_eval)
    m_def = forecast_metrics(np.asarray(out_def["Tsurf"], float)[idx][1:], obs_eval)
    m_da = forecast_metrics(np.asarray(out_da["Tsurf"], float)[idx][1:], obs_eval)

    g_def = skill_gate(m_def, m_persist, deviation=dev_def)
    g_da = skill_gate(m_da, m_persist, deviation=dev_da, baseline_deviation=dev_def)
    delta = diagnostics_delta(dev_da, dev_def)
    return {"n": n, "cal_emiss": cal_emiss, "def_emiss": def_emiss,
            "persist": m_persist, "def": (m_def, dev_def, g_def),
            "da": (m_da, dev_da, g_da), "delta": delta}


def _rows(r):
    rows = []
    def row(model, m, dev, gate):
        return {"model": model, "rmse": m["rmse"], "mae": m["mae"],
                "freeze_thaw_accuracy": m["freeze_thaw_accuracy"],
                "max_primary_residual": (dev or {}).get("max_primary_residual", ""),
                "diagnostic_steps_rate": (dev or {}).get("diagnostic_steps_rate", ""),
                "over_melt_count": (dev or {}).get("over_melt_count", ""),
                "overflow_count": (dev or {}).get("overflow_count", ""), "gate": gate}
    rows.append(row("persistence", r["persist"], None, "baseline"))
    m, dev, (ok, why) = r["def"]
    rows.append(row("default", m, dev, "PASS" if ok else "FAIL — " + "; ".join(why)))
    m, dev, (ok, why) = r["da"]
    rows.append(row(f"DA(Emiss={r['cal_emiss']:.3f})", m, dev, "PASS" if ok else "FAIL — " + "; ".join(why)))
    return rows


_COLS = ("model", "rmse", "mae", "freeze_thaw_accuracy", "max_primary_residual",
         "diagnostic_steps_rate", "over_melt_count", "overflow_count", "gate")


def main():
    max_steps = None
    if "--max-steps" in sys.argv:
        max_steps = int(sys.argv[sys.argv.index("--max-steps") + 1])
    r = build(max_steps)
    rows = _rows(r)
    outdir = REPO / "reports"; outdir.mkdir(exist_ok=True)
    head = "| " + " | ".join(_COLS) + " |"
    sep = "| " + " | ".join("---" for _ in _COLS) + " |"
    lines = [f"# Diagnostics-aware DA evaluation ({r['n']} steps)", "",
             "skill(RMSE↓) + accounting(residual~0) + physics(diagnostics 부담). "
             f"default Emiss={r['def_emiss']:.3f}, calibrated Emiss={r['cal_emiss']:.3f}.",
             "", head, sep]
    import csv as _csv, io as _io
    buf = _io.StringIO(); w = _csv.writer(buf); w.writerow(_COLS)
    for row in rows:
        lines.append("| " + " | ".join(str(row[c]).replace("|", "\\|") for c in _COLS) + " |")
        w.writerow([row[c] for c in _COLS])
    d = r["delta"]
    lines += ["", "## Diagnostics delta (DA − default)",
              f"- Δover_melt_count: {d['delta_over_melt_count']}",
              f"- Δoverflow_count: {d['delta_overflow_count']}",
              f"- Δdiagnostic_steps_rate: {d['delta_diagnostic_steps_rate']:.4f}",
              f"- physics_worse: {d['physics_worse']}"]
    (outdir / "da_evaluation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (outdir / "da_evaluation.csv").write_text(buf.getvalue(), encoding="utf-8")
    print("wrote reports/da_evaluation.{md,csv}")
    for row in rows:
        print(f"  {row['model']:22s} rmse={row['rmse']:.4f} ft_acc={row['freeze_thaw_accuracy']:.4f} "
              f"gate={row['gate']}")
    print(f"  physics_worse(DA vs default): {d['physics_worse']}")


if __name__ == "__main__":
    main()
