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
    # mask missing obs in the calibration window too (evaluation already does this):
    # weight=0 on missing (-9999) so the loss never tries to fit sentinel values.
    obs_np = np.array(mi.TSurfObs, float)[k0:k0 + na]
    valid = obs_np > -100.0
    cal_valid_n = int(valid.sum())
    if cal_valid_n < 3:
        raise RuntimeError(f"calibration window has too few valid observations ({cal_valid_n})")
    # zero out invalid entries so weight=0 * obs never forms 0*NaN (NaN); with a
    # -9999 sentinel this is belt-and-suspenders, but it's cheap and fully safe.
    obs = jnp.array(np.where(valid, obs_np, 0.0))
    w = jnp.array(valid.astype(float))
    emap = lambda re: 0.85 + 0.15 * jnn.sigmoid(re)      # constrained to physical [0.85, 1.0]

    def loss(c):
        e = emap(c["re"])
        p = jm.dry_rollout({**phy_d, "Emiss": e}, x0, {**bf, "Tair": bf["Tair"] + c["bias"]}, static)
        return jnp.sum(w * (p - obs) ** 2) / jnp.sum(w) + 5.0 * (e - 0.95) ** 2 + 2.0 * c["bias"] ** 2

    est, _ = fit(loss, {"re": jnp.float64(0.0), "bias": jnp.float64(0.0)}, steps=steps, lr=0.03)
    return float(emap(est["re"])), float(phy.Emiss), cal_valid_n


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


K0, NA = 2000, 200      # calibration window [K0, K0+NA); evaluation = valid obs AFTER it


def build(max_steps=None):
    cal_emiss, def_emiss, cal_valid_n = calibrate_emiss(k0=K0, na=NA)
    m, objs = build_model()
    mi, mo, phy, g, s, a, coup, st, cpm, _ = objs
    n = st.SimLen - 1 if max_steps is None else min(max_steps, st.SimLen - 1)

    out_def = _run(def_emiss, n, objs, mi, g, s, a, coup, st, cpm, phy)
    m2, o2 = build_model(); mi2, _, phy2, g2, s2, a2, coup2, st2, cpm2, _ = o2
    out_da = _run(cal_emiss, n, o2, mi2, g2, s2, a2, coup2, st2, cpm2, phy2)

    tso = np.array(mi.TSurfObs, float)[:n]
    idx = np.flatnonzero(tso > -100.0)
    idx = idx[idx >= K0 + NA]            # HOLD OUT the calibration window (no train leakage)
    if len(idx) < 3:
        raise RuntimeError(f"holdout too small ({len(idx)}); increase --max-steps beyond {K0 + NA}")
    obs = tso[idx]
    obs_eval = obs[1:]
    # constant-initial baseline: hold the analysis-time obs over the whole lead.
    # (NOT 1-step persistence obs[t-1], which is degenerate at 30 s resolution —
    #  RMSE ~0.006 — so it can't gate anything. Named honestly.)
    const_eval = np.full_like(obs_eval, obs[0])
    # reference-only: 1-step persistence RMSE on the SAME holdout (predict obs[t] with
    # obs[t-1]). ~0 at 30 s resolution — documents WHY it's unfit as a gate baseline.
    m_one_step = forecast_metrics(obs[:-1], obs[1:])

    # two deviation budgets per model: full-run audit (residual = code-leak detector,
    # whole trajectory) AND holdout-aligned so the physics burden the gate weighs comes
    # from the SAME window as the holdout skill — not full-run vs holdout. The holdout
    # window is the CONTIGUOUS interval [first, last] valid-obs step, not only the
    # obs-valid steps: forecast error is measured at obs times, but the physics burden
    # accrues at EVERY model step in between, so all of them must count.
    hold_steps = range(int(idx[0]), int(idx[-1]) + 1)
    dev_def = deviation_budget(out_def, case_id="default")
    dev_da = deviation_budget(out_da, case_id=f"DA(Emiss={cal_emiss:.3f})")
    dev_def_h = deviation_budget(out_def, case_id="default@holdout", steps=hold_steps)
    dev_da_h = deviation_budget(out_da, case_id="DA@holdout", steps=hold_steps)
    m_const = forecast_metrics(const_eval, obs_eval)
    m_def = forecast_metrics(np.asarray(out_def["Tsurf"], float)[idx][1:], obs_eval)
    m_da = forecast_metrics(np.asarray(out_da["Tsurf"], float)[idx][1:], obs_eval)

    # gates use the HOLDOUT-aligned budget (skill window == diagnostics window)
    g_def = skill_gate(m_def, m_const, deviation=dev_def_h)
    g_da_vs_const = skill_gate(m_da, m_const, deviation=dev_da_h, baseline_deviation=dev_def_h)
    g_da_vs_def = skill_gate(m_da, m_def, deviation=dev_da_h, baseline_deviation=dev_def_h)
    delta = diagnostics_delta(dev_da_h, dev_def_h)
    return {"n": n, "holdout_n": len(idx), "cal_window": (K0, K0 + NA),
            "holdout_interval": (int(idx[0]), int(idx[-1])),
            "cal_valid_n": cal_valid_n, "one_step_rmse": m_one_step["rmse"],
            "cal_emiss": cal_emiss, "def_emiss": def_emiss,
            "const": m_const, "def": (m_def, dev_def_h, g_def),
            "da": (m_da, dev_da_h, g_da_vs_const, g_da_vs_def),
            "full_run": {"default": dev_def, "DA": dev_da},
            "rmse_delta_vs_default": m_da["rmse"] - m_def["rmse"],
            "mae_delta_vs_default": m_da["mae"] - m_def["mae"], "delta": delta}


def _rows(r):
    rows = []
    def row(model, m, dev, gate_const, gate_def):
        return {"model": model, "rmse": m["rmse"], "mae": m["mae"],
                "freeze_thaw_accuracy": m["freeze_thaw_accuracy"],
                "max_primary_residual": (dev or {}).get("max_primary_residual", ""),
                "over_melt_count": (dev or {}).get("over_melt_count", ""),
                "overflow_count": (dev or {}).get("overflow_count", ""),
                "gate_vs_const": gate_const, "gate_vs_default": gate_def}
    rows.append(row("constant_initial", r["const"], None, "baseline", "baseline"))
    m, dev, (ok, why) = r["def"]
    rows.append(row("default", m, dev, "PASS" if ok else "FAIL — " + "; ".join(why), "baseline"))
    m, dev, (okc, whyc), (okd, whyd) = r["da"]
    rows.append(row(f"DA(Emiss={r['cal_emiss']:.3f})", m, dev,
                    "PASS" if okc else "FAIL — " + "; ".join(whyc),
                    "PASS" if okd else "FAIL — " + "; ".join(whyd)))
    return rows


_COLS = ("model", "rmse", "mae", "freeze_thaw_accuracy", "max_primary_residual",
         "over_melt_count", "overflow_count", "gate_vs_const", "gate_vs_default")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Diagnostics-aware calibrated-parameter evaluation")
    ap.add_argument("--max-steps", type=int, default=None,
                    help=f"cap rollout length (must exceed calibration window end {K0 + NA})")
    args = ap.parse_args()
    if args.max_steps is not None and args.max_steps <= 0:
        ap.error("--max-steps must be positive")
    r = build(args.max_steps)
    rows = _rows(r)
    outdir = REPO / "reports"; outdir.mkdir(exist_ok=True)
    head = "| " + " | ".join(_COLS) + " |"
    sep = "| " + " | ".join("---" for _ in _COLS) + " |"
    c0, c1 = r["cal_window"]
    lines = [f"# Diagnostics-aware calibrated-parameter evaluation ({r['n']} steps)", "",
             "**범위 주의**: 이것은 완전한 forecast DA가 아니라, JAX dry model에서 보정한 Emiss를 "
             "full model에 넣었을 때의 single-fixture 파라미터 민감도 + 진단 리포트다. 평가는 t=0부터 "
             "전체 trajectory를 다시 도는 free-run(analysis-state forecast 아님)이고, 보정한 Tair bias는 "
             "full model에 적용하지 않는다(Emiss만).",
             "",
             f"calibration window = [{c0}, {c1}) · valid obs {r['cal_valid_n']}개(missing masked) · "
             f"evaluation = 그 이후 valid obs {r['holdout_n']}개(train 누수 없음). "
             f"default Emiss={r['def_emiss']:.3f}, calibrated Emiss={r['cal_emiss']:.3f}. "
             "baseline = constant_initial(분석시각 obs 고정). "
             f"참조: 1-step persistence RMSE = {r['one_step_rmse']:.4f}(30s에서 자명 → gate baseline 부적합, gate에 미사용). "
             "gate: RMSE만 hard, MAE/freeze-thaw는 report-only.",
             "",
             f"표의 residual/over_melt/overflow는 **holdout interval [{r['holdout_interval'][0]}, "
             f"{r['holdout_interval'][1]}] 집계**(skill window와 정렬; forecast 오차는 obs 시각에서, "
             "물리 부담은 그 사이 모든 model step에서 누적 — analysis-time 첫 step 포함). "
             "전체 rollout 감사값은 아래 'Full-run audit' 참조.",
             "", head, sep]
    import csv as _csv, io as _io
    buf = _io.StringIO(); w = _csv.writer(buf); w.writerow(_COLS)
    for row in rows:
        lines.append("| " + " | ".join(str(row[c]).replace("|", "\\|") for c in _COLS) + " |")
        w.writerow([row[c] for c in _COLS])
    d = r["delta"]
    lines += ["", "## DA vs default (직접 비교)",
              f"- Δrmse (DA − default): {r['rmse_delta_vs_default']:+.4f}  ({'개선' if r['rmse_delta_vs_default'] < 0 else '악화'})",
              f"- Δmae  (DA − default): {r['mae_delta_vs_default']:+.4f}",
              "", "## Diagnostics delta (DA − default)",
              f"- Δover_melt_count: {d['delta_over_melt_count']}",
              f"- Δoverflow_count: {d['delta_overflow_count']}",
              f"- Δdiagnostic_steps_rate: {d['delta_diagnostic_steps_rate']:.4f}",
              f"- physics_worse: {d['physics_worse']}",
              "", "## Full-run audit (전체 rollout, residual = 코드 누출 게이트 P0)"]
    for name, dv in r["full_run"].items():
        lines.append(f"- {name}: residual={dv['max_primary_residual']:.3e} "
                     f"({'PASS' if dv['max_primary_residual'] < 1e-9 else 'FAIL'}) · "
                     f"over_melt={dv['over_melt_count']} · overflow={dv['overflow_count']} · "
                     f"diag_rate={dv['diagnostic_steps_rate']:.4f}")
    lines += ["", "기계가독 메타데이터(gate 결과·holdout_interval·holdout residual/rate·"
              "one_step_persistence_rmse)는 `da_evaluation_meta.json` 참조."]
    (outdir / "da_evaluation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (outdir / "da_evaluation.csv").write_text(buf.getvalue(), encoding="utf-8")
    # machine-readable metadata sidecar: reference metrics + window provenance so a
    # downstream reader can reproduce the baseline argument without parsing Markdown.
    import json as _json
    dev_def_h, dev_da_h = r["def"][1], r["da"][1]
    da_row = rows[2]                            # constant_initial, default, DA
    meta = {"n_steps": r["n"], "cal_window": list(r["cal_window"]),
            "holdout_interval": list(r["holdout_interval"]),
            "cal_valid_n": r["cal_valid_n"], "holdout_n": r["holdout_n"],
            "default_emiss": r["def_emiss"], "calibrated_emiss": r["cal_emiss"],
            "one_step_persistence_rmse": r["one_step_rmse"],
            "rmse_delta_da_minus_default": r["rmse_delta_vs_default"],
            "mae_delta_da_minus_default": r["mae_delta_vs_default"],
            "physics_worse": r["delta"]["physics_worse"],
            "da_gate_vs_const": da_row["gate_vs_const"],
            "da_gate_vs_default": da_row["gate_vs_default"],
            "default_holdout_residual": dev_def_h["max_primary_residual"],
            "da_holdout_residual": dev_da_h["max_primary_residual"],
            "default_holdout_diagnostic_steps_rate": dev_def_h["diagnostic_steps_rate"],
            "da_holdout_diagnostic_steps_rate": dev_da_h["diagnostic_steps_rate"],
            "default_holdout_over_melt_count": dev_def_h["over_melt_count"],
            "da_holdout_over_melt_count": dev_da_h["over_melt_count"],
            "default_holdout_overflow_count": dev_def_h["overflow_count"],
            "da_holdout_overflow_count": dev_da_h["overflow_count"]}
    (outdir / "da_evaluation_meta.json").write_text(_json.dumps(meta, indent=2), encoding="utf-8")
    print("wrote reports/da_evaluation.{md,csv} + da_evaluation_meta.json")
    for row in rows:
        print(f"  {row['model']:22s} rmse={row['rmse']:.4f} mae={row['mae']:.4f} "
              f"gate_vs_const={row['gate_vs_const']} gate_vs_default={row['gate_vs_default']}")
    print(f"  Δrmse(DA−default)={r['rmse_delta_vs_default']:+.4f}  physics_worse={d['physics_worse']}")


if __name__ == "__main__":
    main()
