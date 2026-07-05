#!/usr/bin/env python3
"""True forecast data assimilation (STATE estimation), single-fixture.

Unlike tools/report_da_evaluation.py (calibrated-PARAMETER sensitivity, free-run
from t=0), this is an honest forecast-DA cycle carried out ENTIRELY inside the
differentiable dry thermal model, so assimilation and forecast use the same model
(no cross-model transfer artifact):

  1. background : spin the reference to the analysis-window start k0 -> background
     near-surface thermal state x_b (default params, NO obs correction).
  2. assimilate : over the window [k0, k0+W) variationally fit a state correction
     dx (offset to near-surface layers 1:5) that minimizes weighted MSE to obs
     under FREE propagation (no obs insertion) + background reg bg_w*||dx||^2.
     Parameters are FIXED at default -> this isolates STATE estimation, the essence
     of forecast DA (correct the initial condition, not the physics).
  3. end-states : propagate analysis (x_b+dx) and background (x_b) through the
     window to their states at k0+W via dry_rollout_carry.
  4. forecast   : FREE-RUN (no obs) over the lead [k0+W, k0+W+L) from each
     end-state -> pred_da / pred_bg.
  5. score      : on valid obs in the lead, does correcting the initial state
     improve the subsequent FREE forecast vs. not assimilating (background)?

The gate is skill-only (RMSE hard): the dry thermal model evolves no storages, so
the deviation/physics-burden audit (a full-model concept) does not apply here and
the report says so. degradation_ratio (window fit -> forecast) exposes whether the
assimilation overfits the window instead of improving the forecast.

    python3 tools/report_forecast_da.py [--window W] [--lead L] [--k0 K0]

Writes reports/forecast_da.md, .csv and forecast_da_meta.json. Requires jax/optax.
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from tools.run_no_coupling import build_model, RSP_SRC  # noqa: E402
from droad.skill_gate import (  # noqa: E402
    forecast_metrics, skill_gate, degradation_ratio,
)

K0, WINDOW, LEAD = 2000, 120, 480      # analysis start, assimilation steps, forecast steps
BG_WEIGHT = 0.05                       # background regularization on the state correction


def _setup():
    """Import JAX bits + build the NumPy reference model. Returns shared handles."""
    sys.path.insert(0, str(RSP_SRC))
    from jax import config
    config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import examples.demo_da as dd       # reuse _static_forc / _phy builders
    from droad import jax_model as jm
    from droad.assimilate import fit
    m, objs = build_model()
    return m, objs, jnp, dd, jm, fit


def _advance(m, objs, start, count):
    """Step the NumPy reference forward `count` steps from index `start` (in place)."""
    mi, mo, phy, g, s, a, coup, st, cpm, _ = objs
    for i in range(start, start + count):
        m["InputOutput"].SetCurrentValues(i, mi, a, st, s, coup, g)
        m["Storage"].PrecipitationToStorage(st, cpm, mi.PrecPhase[i], a, s)
        m["BalanceModel"].BalanceModelOneStep(mi.SW[i], mi.LW[i], phy, g, s, a, st, coup, mi, i, cpm)
        wf = m["WearingFactors"].WearingFactors(); m["Cond"].WearFactors(cpm, st.Tph, s, wf)
        m["Cond"].RoadCond(phy.MaxPormms, s, a, st, cpm, wf); g.Albedo = m["Cond"].CalcAlbedo(s, cpm)


def _span(dd, objs, jnp, k0, span):
    """Build JAX static + forcings + background state x_b at the CURRENT reference
    state, for the window [k0, k0+span)."""
    mi, mo, phy, g, s, a, coup, st, cpm, _ = objs
    avail = min(len(mi.TSurfObs), len(mi.Tair), len(mi.time))
    if k0 + span > avail:                 # fail-fast: CLI k0/window/lead can overrun the fixture
        raise RuntimeError(f"k0+window+lead={k0 + span} exceeds available data {avail}")
    static, forc = dd._static_forc(mi, g, st, slice(k0, k0 + span), np.zeros(span, bool))
    phy_d = dd._phy(phy)
    x_b = (jnp.array(g.Tmp, float), jnp.array(g.TmpNw, float), jnp.float64(a.BLCond))
    tso = np.array(mi.TSurfObs, float)[k0:k0 + span]
    return static, forc, phy_d, x_b, tso


def _slice_forc(forc, a, b):
    return {k: v[a:b] for k, v in forc.items()}


def build(k0=K0, window=WINDOW, lead=LEAD, bg_w=BG_WEIGHT):
    """Single-window forecast-DA cycle: spin to k0, then run assimilate->forecast."""
    m, objs, jnp, dd, jm, fit = _setup()
    _advance(m, objs, 0, k0)
    static, forc, phy_d, x_b, tso = _span(dd, objs, jnp, k0, window + lead)
    return _da_cycle(jm, fit, jnp, static, forc, phy_d, x_b, tso, window, lead, bg_w, k0)


def build_multi(k0_first, n_windows, window=WINDOW, lead=LEAD, bg_w=BG_WEIGHT):
    """Run the forecast-DA cycle on n_windows CONSECUTIVE analysis windows, spinning
    the reference forward ONCE (not re-spinning per window). Windows with too few
    valid obs are skipped. Returns the list of per-window result dicts."""
    m, objs, jnp, dd, jm, fit = _setup()
    span = window + lead
    _advance(m, objs, 0, k0_first)
    results, cursor = [], k0_first
    for _ in range(n_windows):
        static, forc, phy_d, x_b, tso = _span(dd, objs, jnp, cursor, span)
        try:
            results.append(_da_cycle(jm, fit, jnp, static, forc, phy_d, x_b, tso,
                                     window, lead, bg_w, cursor))
        except RuntimeError:
            pass                              # window without enough valid obs -> not a case
        _advance(m, objs, cursor, span)       # move reference to the next window start
        cursor += span
    return results


def _da_cycle(jm, fit, jnp, static, forc, phy_d, x_b, tso, window, lead, bg_w, k0):
    forc_win = _slice_forc(forc, 0, window)
    forc_lead = _slice_forc(forc, window, window + lead)

    # --- observation masks (real obs carry a -9999 missing sentinel) ---
    obs_win = tso[:window]
    valid_win = obs_win > -100.0
    if int(valid_win.sum()) < 3:
        raise RuntimeError(f"assimilation window has too few valid obs ({int(valid_win.sum())})")
    obs_lead = tso[window:window + lead]
    valid_lead = obs_lead > -100.0
    if int(valid_lead.sum()) < 3:
        raise RuntimeError(f"forecast lead has too few valid obs ({int(valid_lead.sum())})")

    # --- state control: near-surface offset on layers 1:5 (same handle as demo_da) ---
    apply_state = lambda bg, dx: (bg[0].at[1:5].add(dx), bg[1].at[1:5].add(dx), bg[2])
    obs_win_j = jnp.array(np.where(valid_win, obs_win, 0.0))   # zero invalids so 0*NaN can't form
    w_win = jnp.array(valid_win.astype(float))

    def loss(dx):
        pred = jm.dry_rollout(phy_d, apply_state(x_b, dx), forc_win, static)
        misfit = jnp.sum(w_win * (pred - obs_win_j) ** 2) / jnp.sum(w_win)
        return misfit + bg_w * jnp.sum(dx ** 2)      # 3D/4D-Var: misfit + background term

    dx_opt, _ = fit(loss, jnp.zeros(4), steps=400, lr=0.05)

    # --- analysis vs background: propagate through the window to their end-states ---
    carry_a, ts_a_win = jm.dry_rollout_carry(phy_d, apply_state(x_b, dx_opt), forc_win, static)
    carry_b, ts_b_win = jm.dry_rollout_carry(phy_d, x_b, forc_win, static)

    # --- FREE-RUN forecast over the lead from each end-state (no obs insertion) ---
    pred_da = np.asarray(jm.dry_rollout(phy_d, carry_a, forc_lead, static), float)
    pred_bg = np.asarray(jm.dry_rollout(phy_d, carry_b, forc_lead, static), float)

    ol = obs_lead[valid_lead]
    m_da = forecast_metrics(pred_da[valid_lead], ol)
    m_bg = forecast_metrics(pred_bg[valid_lead], ol)
    # constant_initial baseline: hold the LAST valid obs BEFORE the lead over the
    # whole forecast (never reach into the lead's future obs -> no leakage).
    const0 = float(obs_win[valid_win][-1])
    const = np.full_like(ol, const0)
    m_const = forecast_metrics(const, ol)

    # in-sample fit RMSE on the window (train) -> degradation to forecast (holdout)
    aw = np.asarray(ts_a_win, float)[valid_win]
    bw = np.asarray(ts_b_win, float)[valid_win]
    ow = obs_win[valid_win]
    train_da = forecast_metrics(aw, ow)["rmse"]
    train_bg = forecast_metrics(bw, ow)["rmse"]

    g_da_vs_bg = skill_gate(m_da, m_bg)          # DA vs no-DA (skill-only; no storages here)
    g_da_vs_const = skill_gate(m_da, m_const)
    g_bg_vs_const = skill_gate(m_bg, m_const)    # no-DA's OWN gate vs const (not DA's)
    dx = np.asarray(dx_opt, float)
    return {"k0": k0, "window": window, "lead": lead, "bg_w": bg_w,
            "valid_win": int(valid_win.sum()), "valid_lead": int(valid_lead.sum()),
            "dx": [float(v) for v in dx],
            "dx_l2": float(np.sqrt(np.sum(dx ** 2))), "dx_max_abs": float(np.max(np.abs(dx))),
            "const": m_const, "bg": (m_bg, train_bg), "da": (m_da, train_da),
            "gate_da_vs_bg": g_da_vs_bg, "gate_da_vs_const": g_da_vs_const,
            "gate_bg_vs_const": g_bg_vs_const,
            "rmse_delta_da_minus_bg": m_da["rmse"] - m_bg["rmse"],
            "train_delta_da_minus_bg": train_da - train_bg,
            "degradation_da": degradation_ratio(m_da["rmse"], train_da),
            "degradation_bg": degradation_ratio(m_bg["rmse"], train_bg)}


_COLS = ("model", "rmse", "mae", "freeze_thaw_accuracy", "train_rmse",
         "degradation_ratio", "gate_vs_bg", "gate_vs_const")


def _rows(r):
    def row(model, m, train, dr, gate_bg, gate_const):
        return {"model": model, "rmse": m["rmse"], "mae": m["mae"],
                "freeze_thaw_accuracy": m["freeze_thaw_accuracy"],
                "train_rmse": ("" if train is None else train),
                "degradation_ratio": ("" if dr is None else dr),
                "gate_vs_bg": gate_bg, "gate_vs_const": gate_const}
    m_bg, train_bg = r["bg"]
    m_da, train_da = r["da"]
    return [
        row("constant_initial", r["const"], None, None, "baseline", "baseline"),
        row("no_DA(background)", m_bg, train_bg, r["degradation_bg"], "baseline",
            "PASS" if r["gate_bg_vs_const"][0] else "FAIL — " + "; ".join(r["gate_bg_vs_const"][1])),
        row("DA(state)", m_da, train_da, r["degradation_da"],
            "PASS" if r["gate_da_vs_bg"][0] else "FAIL — " + "; ".join(r["gate_da_vs_bg"][1]),
            "PASS" if r["gate_da_vs_const"][0] else "FAIL — " + "; ".join(r["gate_da_vs_const"][1])),
    ]


def main():
    import argparse
    ap = argparse.ArgumentParser(description="True forecast DA (state estimation)")
    ap.add_argument("--window", type=int, default=WINDOW, help="assimilation window steps")
    ap.add_argument("--lead", type=int, default=LEAD, help="forecast lead steps")
    ap.add_argument("--k0", type=int, default=K0, help="analysis window start step")
    ap.add_argument("--bg-w", type=float, default=BG_WEIGHT, dest="bg_w",
                    help="background regularization on the state correction")
    args = ap.parse_args()
    if args.k0 < 0:                                   # k0=0 is a valid start (no spin)
        ap.error("--k0 must be non-negative")
    for nm, v in (("window", args.window), ("lead", args.lead)):
        if v <= 0:
            ap.error(f"--{nm} must be positive")
    import math as _math
    if not _math.isfinite(args.bg_w) or args.bg_w < 0:
        ap.error("--bg-w must be a finite non-negative number")
    r = build(args.k0, args.window, args.lead, args.bg_w)
    rows = _rows(r)
    outdir = REPO / "reports"; outdir.mkdir(exist_ok=True)

    import csv as _csv, io as _io
    buf = _io.StringIO(); w = _csv.writer(buf); w.writerow(_COLS)
    for row in rows:
        w.writerow([row[c] for c in _COLS])

    head = "| " + " | ".join(_COLS) + " |"
    sep = "| " + " | ".join("---" for _ in _COLS) + " |"
    dbg = r["rmse_delta_da_minus_bg"]
    lines = [f"# True forecast DA — state estimation ({r['lead']} step lead)", "",
             "초기 near-surface 온도 상태를 변분동화한 뒤 **자유 예보**(obs 미삽입)가 no-DA(background) "
             "대비 개선되는지 정직하게 판별한다. 파라미터는 default 고정 → 순수 STATE 추정. "
             "dry thermal 모델 내부에서 동화·예보가 같은 모델을 쓰므로 cross-model 아티팩트가 없다.",
             "",
             f"analysis start k0={r['k0']} · 동화창 [k0, k0+{r['window']}) valid obs {r['valid_win']}개 · "
             f"예보 lead {r['lead']}스텝 valid obs {r['valid_lead']}개. "
             "gate: RMSE만 hard. **deviation/physics-burden 감사는 미적용**(dry 모델은 storage를 "
             "진행하지 않음 — full-model 개념). degradation_ratio = 예보RMSE / 동화창RMSE "
             "(>1이면 overfit 또는 lead 구간이 동화창보다 본질적으로 더 어려움 — 둘을 분리해 볼 것).",
             "", head, sep]
    for row in rows:
        lines.append("| " + " | ".join(
            (f"{row[c]:.4f}" if isinstance(row[c], float) else str(row[c])) for c in _COLS) + " |")
    verdict = "개선" if dbg < 0 else "미개선"
    lines += ["", "## DA vs no-DA (초기상태 보정 효과)",
              f"- Δrmse (DA − background): {dbg:+.4f}  ({verdict})",
              f"- DA degradation (예보/동화창): {r['degradation_da']:.3f}",
              f"- background degradation: {r['degradation_bg']:.3f}",
              f"- state correction dx (layers 1:5, degC): "
              f"[{', '.join(f'{v:+.3f}' for v in r['dx'])}]",
              "", "해석: Δrmse<0이면 초기상태 동화가 자유예보를 개선(초기조건 오차가 lead에서 지배적). "
              "Δrmse≥0이면 model error가 지배적이거나 동화가 창에 overfit, 또는 lead가 동화창보다 "
              "본질적으로 어려운 구간일 수 있음(degradation_ratio를 window 난이도와 분리해 판단)."]
    (outdir / "forecast_da.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (outdir / "forecast_da.csv").write_text(buf.getvalue(), encoding="utf-8")

    import json as _json
    meta = {"k0": r["k0"], "window": r["window"], "lead": r["lead"], "bg_weight": r["bg_w"],
            "valid_win": r["valid_win"], "valid_lead": r["valid_lead"],
            "da_forecast_rmse": r["da"][0]["rmse"], "bg_forecast_rmse": r["bg"][0]["rmse"],
            "const_forecast_rmse": r["const"]["rmse"],
            "rmse_delta_da_minus_bg": r["rmse_delta_da_minus_bg"],
            "da_train_rmse": r["da"][1], "bg_train_rmse": r["bg"][1],
            "da_degradation_ratio": r["degradation_da"], "bg_degradation_ratio": r["degradation_bg"],
            "train_delta_da_minus_bg": r["train_delta_da_minus_bg"],
            "da_gate_vs_bg": "PASS" if r["gate_da_vs_bg"][0] else "FAIL",
            "bg_gate_vs_const": "PASS" if r["gate_bg_vs_const"][0] else "FAIL",
            "state_correction_dx": r["dx"], "dx_l2": r["dx_l2"], "dx_max_abs": r["dx_max_abs"]}
    (outdir / "forecast_da_meta.json").write_text(_json.dumps(meta, indent=2), encoding="utf-8")
    print("wrote reports/forecast_da.{md,csv} + forecast_da_meta.json")
    for row in rows:
        print(f"  {row['model']:20s} rmse={row['rmse']:.4f} "
              f"gate_vs_bg={row['gate_vs_bg']} gate_vs_const={row['gate_vs_const']}")
    print(f"  Δrmse(DA−bg)={dbg:+.4f}  DA degradation={r['degradation_da']:.3f}")


if __name__ == "__main__":
    main()
