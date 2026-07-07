#!/usr/bin/env python3
"""Design A0 prototype — full-model forecast DA (thermal-state injection).

dry 상태추정 forecast DA(`report_forecast_da`)는 dry thermal 모델 안에서만 동화·예보하므로
storage/deviation 감사가 적용되지 않는다. 이 도구는 그 dry 보정 `dx`를 **full model**에 주입해
"열 보정이 full 예보에서 살아남는가(skill) + 물리 부담을 악화시키지 않는가(deviation)"를 처음으로
동시에 판정한다. (설계: docs/design/full_model_da.md, A0.)

A0 절차:
  1. full NumPy 모델을 k0까지 spin → 배경 k0 상태(열+저장소)
  2. dry JAX 모델에서 raw dx 추정 (동화창 [k0, k0+window))
  3. full k0 상태에 dx 주입 (Tmp/TmpNw 1:5 += dx, TsurfAve 동기화, 상태 격리)
  4. full model을 [k0, k0+window+lead) 동안 **obs 미삽입 free-run**(InitLenI=-1, TSurfObs=sentinel)
     — BG(무보정)/DA(보정) 각각
  5. lead 구간만 skill_gate + deviation_budget으로 평가(analysis-window diagnostics는 report-only)

raw dx를 k0+window에 직접 더하지 않는다(그건 오적용). A1(evolved-end correction)은 후속 TODO.

    python3 tools/report_forecast_da_fullmodel.py [--k0 --window --lead --bg-w]

Writes reports/forecast_da_fullmodel.{md,csv} + _meta.json. Requires jax/optax.
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from tools.report_deviation_budget import _phy, _day, _cp  # noqa: E402
from droad.driver import full_rollout  # noqa: E402
from droad.storage import Surf  # noqa: E402
from droad.deviation import deviation_budget  # noqa: E402
from droad.skill_gate import forecast_metrics, skill_gate, diagnostics_delta  # noqa: E402

K0, WINDOW, LEAD, BG_WEIGHT = 2000, 120, 480, 0.05


def _validate_dx(dx):
    """A state correction is a finite offset for near-surface layers 1:5 (shape (4,)).
    Reject wrong shape, non-finite, and bool/string/object (matching the project-wide
    numeric policy) so a bad dx can't broadcast or coerce silently."""
    arr = np.asarray(dx)
    if arr.dtype.kind not in ("i", "u", "f"):          # no bool/str/object coercion
        raise ValueError("dx must be a numeric array, not bool/string/object")
    d = arr.astype(float)
    if d.shape != (4,):
        raise ValueError(f"dx must have shape (4,), got {d.shape}")
    if not np.all(np.isfinite(d)):
        raise ValueError("dx must be finite")
    return d


def _inject_dx_state(objs, dx):
    """Return (Tmp0, TmpNw0, surf0) for the forecast start. dx=None → background (no
    correction). When dx is given it is added to a COPIED thermal state and Surf.TsurfAve
    is synced to (Tmp0[1]+Tmp0[2])/2 — never mutates objs, so BG/DA stay isolated."""
    mi, mo, phy, g, s, a, coup, st, cpm, _ = objs
    Tmp0 = np.array(g.Tmp, float).copy()
    TmpNw0 = np.array(g.TmpNw, float).copy()
    tsurf_ave = s.TsurfAve
    if dx is not None:
        dx = _validate_dx(dx)                                 # shape (4,) + finite
        Tmp0[1:5] += dx
        TmpNw0[1:5] += dx
        tsurf_ave = (Tmp0[1] + Tmp0[2]) / 2.0                 # TsurfAve 동기화
    surf0 = Surf(SrfWat=s.SrfWatmms, SrfSnow=s.SrfSnowmms, SrfIce=s.SrfIcemms,
                 SrfIce2=s.SrfIce2mms, SrfDep=s.SrfDepmms, TsurfAve=tsurf_ave,
                 Q2Melt=s.Q2Melt, T4Melt=s.T4Melt, WearSurf=s.WearSurf,
                 SnowType=a.SnowType, WetSnowFrozen=cpm.WetSnowFrozen, VeryCold=s.VeryCold)
    return Tmp0, TmpNw0, surf0


def _forecast_kwargs(objs, k0, span, dx=None):
    """Build the full_rollout kwargs for an A0 forecast over [k0, k0+span) with obs
    insertion DISABLED (InitLenI=-1, TSurfObs=sentinel) and coupling off. Split out so a
    unit test can lock the no-future-obs-leakage contract without running the model."""
    mi, mo, phy, g, s, a, coup, st, cpm, _ = objs
    if int(k0) != k0 or int(span) != span:
        raise RuntimeError("k0/span must be integer steps")
    if k0 < 0:
        raise RuntimeError("k0 must be non-negative")
    if span <= 0:
        raise RuntimeError("span must be positive")
    avail = min(len(mi.TSurfObs), len(mi.Tair), len(mi.VZ), len(mi.Rhz), len(mi.SW),
                len(mi.LW), len(mi.PrecPhase), len(mi.prec), len(mi.time))
    if k0 + span > avail:
        raise RuntimeError(f"k0+span={k0 + span} exceeds available data {avail}")
    sl = slice(k0, k0 + span)
    hours = np.array([t.hour for t in mi.time[sl]], float)
    prec_in = np.array(mi.prec, float)[sl] / 3600.0 * st.DTSecs
    Tmp0, TmpNw0, surf0 = _inject_dx_state(objs, dx)
    return dict(
        Tair=np.array(mi.Tair, float)[sl], VZ=np.array(mi.VZ, float)[sl],
        Rhz=np.array(mi.Rhz, float)[sl], SW=np.array(mi.SW, float)[sl], LW=np.array(mi.LW, float)[sl],
        TSurfObs=np.full(span, -9999.0), hours=hours,               # sentinel → no obs insertion
        prec_phase=np.array(mi.PrecPhase, float)[sl], prec_in_tstep=prec_in,
        Tmp0=Tmp0, TmpNw0=TmpNw0, WCont=np.array(g.WCont, float),
        CC=np.array(g.CC, float), ZDpth=np.array(g.ZDpth, float),
        DyK=np.array(g.DyK, float), DyC=np.array(g.DyC, float),
        surf0=surf0, Albedo0=g.Albedo, BLCond0=a.BLCond,
        NLayers=st.NLayers, DTSecs=st.DTSecs, MaxPormms=phy.MaxPormms, Tph=st.Tph,
        InitLenI=-1, inCouplingPhase=False,                         # obs 삽입·coupling 명시적 off
        phy=_phy(phy), day=_day(st), cp=_cp(cpm), n_steps=span,
        TsurfObsLast=coup.LastTsurfObs, return_ledger=True)


def _full_forecast(objs, k0, span, dx=None):
    return full_rollout(**_forecast_kwargs(objs, k0, span, dx))


def build_a0(k0=K0, window=WINDOW, lead=LEAD, bg_w=BG_WEIGHT, dx_scale=1.0):
    """dx_scale>1 = STRESS mode: inflate the fitted dx to an unphysically large state
    correction. This is a diagnostic stress (not real DA) to probe whether a large
    thermal perturbation increases the lead physics burden and makes the deviation gate
    FAIL — i.e. to verify the lead primary gate actually catches DA-induced burden."""
    from tools.report_forecast_da import _setup, _advance, _span, _slice_forc, _validate_cycle_args
    _validate_cycle_args(k0, window, lead, bg_w)
    if not np.isfinite(dx_scale) or dx_scale <= 0:
        raise ValueError("dx_scale must be a finite positive number")
    m, objs, jnp, dd, jm, fit = _setup()
    _advance(m, objs, 0, k0)                                   # spin full model to k0
    span = window + lead

    # --- dry dx over the assimilation window (raw dx at k0) ---
    static, forc, phy_d, x_b, tso = _span(dd, objs, jnp, k0, span)
    forc_win = _slice_forc(forc, 0, window)
    obs_win = tso[:window]; valid = obs_win > -100.0
    if int(valid.sum()) < 3:
        raise RuntimeError(f"assimilation window has too few valid obs ({int(valid.sum())})")
    obs_j = jnp.array(np.where(valid, obs_win, 0.0)); w = jnp.array(valid.astype(float))
    apply_state = lambda bg, d: (bg[0].at[1:5].add(d), bg[1].at[1:5].add(d), bg[2])

    def loss(d):
        p = jm.dry_rollout(phy_d, apply_state(x_b, d), forc_win, static)
        return jnp.sum(w * (p - obs_j) ** 2) / jnp.sum(w) + bg_w * jnp.sum(d ** 2)

    dx_opt, _ = fit(loss, jnp.zeros(4), steps=400, lr=0.05)
    dx = np.asarray(dx_opt, float) * dx_scale         # dx_scale>1 = stress (unphysical)

    # --- full-model forecasts (obs 미삽입 free-run over window+lead) ---
    out_bg = _full_forecast(objs, k0, span, dx=None)
    out_da = _full_forecast(objs, k0, span, dx=dx)

    # --- evaluate on lead only (out_* are [0, window+lead) LOCAL timelines) ---
    obs = np.array(objs[0].TSurfObs, float)[k0:k0 + span]
    lead_idx = np.flatnonzero(obs > -100.0); lead_idx = lead_idx[lead_idx >= window]
    if len(lead_idx) < 3:
        raise RuntimeError(f"forecast lead has too few valid obs ({len(lead_idx)})")
    ol = obs[lead_idx]
    ts_bg = np.asarray(out_bg["Tsurf"], float); ts_da = np.asarray(out_da["Tsurf"], float)
    m_bg = forecast_metrics(ts_bg[lead_idx], ol)
    m_da = forecast_metrics(ts_da[lead_idx], ol)
    const0 = float(obs_win[valid][-1])                        # 마지막 pre-lead 유효 obs (leakage 없음)
    m_const = forecast_metrics(np.full_like(ol, const0), ol)

    # lead-aligned deviation budget (LOCAL index) = primary physics gate
    budget_steps = range(int(lead_idx[0]), int(lead_idx[-1]) + 1)
    dev_bg = deviation_budget(out_bg, case_id="bg", steps=budget_steps)
    dev_da = deviation_budget(out_da, case_id="da", steps=budget_steps)
    # analysis-window diagnostics — report-only (lead 진입 전 보정이 storage 부담 키웠는지 비교)
    win_steps = range(0, window)
    dev_bg_win = deviation_budget(out_bg, case_id="bg@win", steps=win_steps)
    dev_da_win = deviation_budget(out_da, case_id="da@win", steps=win_steps)

    gate = skill_gate(m_da, m_bg, deviation=dev_da, baseline_deviation=dev_bg)   # skill + 물리부담
    delta = diagnostics_delta(dev_da, dev_bg)
    return {"k0": k0, "window": window, "lead": lead, "bg_w": bg_w, "dx_scale": dx_scale,
            "valid_lead": int(len(lead_idx)), "dx": [float(v) for v in dx],
            "dx_l2": float(np.sqrt(np.sum(dx ** 2))), "dx_max_abs": float(np.max(np.abs(dx))),
            "const": m_const, "bg": (m_bg, dev_bg), "da": (m_da, dev_da),
            "win": (dev_bg_win, dev_da_win), "gate_da_vs_bg": gate,
            "rmse_delta_da_minus_bg": m_da["rmse"] - m_bg["rmse"],
            "physics_worse": delta["physics_worse"], "delta": delta}


_COLS = ("model", "rmse", "mae", "freeze_thaw_accuracy", "max_primary_residual",
         "over_melt_count", "overflow_count", "gate_vs_bg")


def _rows(r):
    def row(model, m, dev, gate):
        return {"model": model, "rmse": m["rmse"], "mae": m["mae"],
                "freeze_thaw_accuracy": m["freeze_thaw_accuracy"],
                "max_primary_residual": (dev or {}).get("max_primary_residual", ""),
                "over_melt_count": (dev or {}).get("over_melt_count", ""),
                "overflow_count": (dev or {}).get("overflow_count", ""), "gate_vs_bg": gate}
    m_bg, dev_bg = r["bg"]; m_da, dev_da = r["da"]
    ok, why = r["gate_da_vs_bg"]
    return [
        row("constant_initial", r["const"], None, "baseline"),
        row("no_DA(background)", m_bg, dev_bg, "baseline"),
        row("DA(state, full)", m_da, dev_da, "PASS" if ok else "FAIL — " + "; ".join(why)),
    ]


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Design A0 full-model forecast DA prototype")
    ap.add_argument("--k0", type=int, default=K0)
    ap.add_argument("--window", type=int, default=WINDOW)
    ap.add_argument("--lead", type=int, default=LEAD)
    ap.add_argument("--bg-w", type=float, default=BG_WEIGHT, dest="bg_w")
    ap.add_argument("--tag", default="", help="artifact filename suffix + case_id (예: storage_active)")
    ap.add_argument("--dx-scale", type=float, default=1.0, dest="dx_scale",
                    help=">1이면 stress 모드(dx를 인위적으로 확대해 lead 물리부담 gate 검증)")
    args = ap.parse_args()
    r = build_a0(args.k0, args.window, args.lead, args.bg_w, args.dx_scale)
    rows = _rows(r)
    outdir = REPO / "reports"; outdir.mkdir(exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    case_id = args.tag or "default"

    import csv as _csv, io as _io
    buf = _io.StringIO(); w = _csv.writer(buf); w.writerow(_COLS)
    for row in rows:
        w.writerow([row[c] for c in _COLS])

    head = "| " + " | ".join(_COLS) + " |"
    sep = "| " + " | ".join("---" for _ in _COLS) + " |"
    dbg = r["rmse_delta_da_minus_bg"]
    dev_bg_win, dev_da_win = r["win"]
    lines = [f"# Full-model forecast DA — 설계 A0 ({r['lead']} step lead)", "",
             "dry에서 추정한 near-surface 상태보정 dx를 **full 모델** k0 상태에 주입하고(TsurfAve 동기화), "
             "[k0, k0+window+lead)를 obs 미삽입 free-run으로 예보한다. dry DA와 달리 storage가 진행되므로 "
             "**deviation 감사(물리 부담)가 forecast DA에 처음 적용**된다. gate: RMSE hard + 물리 부담 비악화.",
             "",
             f"k0={r['k0']} · 동화창 {r['window']} · 예보 lead {r['lead']} valid obs {r['valid_lead']}개. "
             "raw dx at k0 (A0). analysis-window diagnostics는 report-only, lead-aligned budget이 primary gate.",
             "", head, sep]
    for row in rows:
        lines.append("| " + " | ".join(
            (f"{row[c]:.4f}" if isinstance(row[c], float) else str(row[c])) for c in _COLS) + " |")
    dev_bg_lead, dev_da_lead = r["bg"][1], r["da"][1]
    dx_large = r["dx_max_abs"] > 2.0 or r["dx_l2"] > 3.0
    lines += ["", "## DA vs no-DA (핵심)",
              f"- Δrmse (DA − background): {dbg:+.4f}  ({'개선' if dbg < 0 else '미개선'})",
              f"- physics_worse (over_melt/overflow/rate 악화 여부): **{r['physics_worse']}**",
              f"- diag_steps_rate — **lead(primary gate)**: bg {dev_bg_lead['diagnostic_steps_rate']:.4f} / "
              f"da {dev_da_lead['diagnostic_steps_rate']:.4f}  ·  **window(report-only)**: "
              f"bg {dev_bg_win['diagnostic_steps_rate']:.4f} / da {dev_da_win['diagnostic_steps_rate']:.4f}",
              f"- state_correction_large (dx_l2>3 또는 max|dx|>2): {dx_large} "
              f"(dx_l2={r['dx_l2']:.3f}, max|dx|={r['dx_max_abs']:.3f})",
              f"- Δover_melt: {r['delta']['delta_over_melt_count']} · Δoverflow: {r['delta']['delta_overflow_count']} · "
              f"Δdiag_rate: {r['delta']['delta_diagnostic_steps_rate']:+.4f}",
              f"- state correction dx (layers 1:5): [{', '.join(f'{v:+.3f}' for v in r['dx'])}] "
              f"(l2={r['dx_l2']:.3f}, max|dx|={r['dx_max_abs']:.3f})",
              "", "## Analysis-window diagnostics (report-only)",
              f"- background: over_melt={dev_bg_win['over_melt_count']} overflow={dev_bg_win['overflow_count']} "
              f"rate={dev_bg_win['diagnostic_steps_rate']:.4f}",
              f"- DA:         over_melt={dev_da_win['over_melt_count']} overflow={dev_da_win['overflow_count']} "
              f"rate={dev_da_win['diagnostic_steps_rate']:.4f}",
              "", "**주의**: diagnostic 활동이 window(report-only)에서만 발생하고 lead(primary gate)에서 0이면, "
              "이 case는 'storage-active signal은 있으나 lead deviation gate는 clean'이다 — lead gate가 실제 "
              "burden 증가를 처리했다는 증거는 아니다(그건 lead 구간에 diagnostics가 발생하는 window/stress 필요).",
              "", "해석: DA가 lead 예보 RMSE를 낮추면서(gate PASS) physics_worse=False면 열 보정이 full 예보에서 "
              "살아남고 물리 부담도 clean. physics_worse=True면 열을 맞추려다 융해/상전이를 왜곡한 것 → 설계 C 신호."]
    (outdir / f"forecast_da_fullmodel{suffix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (outdir / f"forecast_da_fullmodel{suffix}.csv").write_text(buf.getvalue(), encoding="utf-8")

    import json as _json
    meta = {"case_id": case_id, "k0": r["k0"], "window": r["window"], "lead": r["lead"], "bg_weight": r["bg_w"],
            "dx_scale": r["dx_scale"], "valid_lead": r["valid_lead"],
            "bg_forecast_rmse": r["bg"][0]["rmse"], "da_forecast_rmse": r["da"][0]["rmse"],
            "const_forecast_rmse": r["const"]["rmse"], "rmse_delta_da_minus_bg": dbg,
            "da_gate_vs_bg": "PASS" if r["gate_da_vs_bg"][0] else "FAIL",
            "da_gate_vs_bg_reasons": list(r["gate_da_vs_bg"][1]),   # FAIL 시 RMSE/residual/burden 사유
            "physics_worse": r["physics_worse"],
            "bg_lead_residual": r["bg"][1]["max_primary_residual"],
            "da_lead_residual": r["da"][1]["max_primary_residual"],
            # lead diagnostic burden (the primary physics gate inputs)
            "bg_lead_diagnostic_steps_rate": r["bg"][1]["diagnostic_steps_rate"],
            "da_lead_diagnostic_steps_rate": r["da"][1]["diagnostic_steps_rate"],
            "bg_lead_over_melt_count": r["bg"][1]["over_melt_count"],
            "da_lead_over_melt_count": r["da"][1]["over_melt_count"],
            "bg_lead_overflow_count": r["bg"][1]["overflow_count"],
            "da_lead_overflow_count": r["da"][1]["overflow_count"],
            # analysis-window diagnostics (report-only)
            "bg_window_diagnostic_steps_rate": dev_bg_win["diagnostic_steps_rate"],
            "da_window_diagnostic_steps_rate": dev_da_win["diagnostic_steps_rate"],
            "bg_window_over_melt_count": dev_bg_win["over_melt_count"],
            "da_window_over_melt_count": dev_da_win["over_melt_count"],
            "bg_window_overflow_count": dev_bg_win["overflow_count"],
            "da_window_overflow_count": dev_da_win["overflow_count"],
            "state_correction_dx": r["dx"], "dx_l2": r["dx_l2"], "dx_max_abs": r["dx_max_abs"],
            "state_correction_large": bool(r["dx_max_abs"] > 2.0 or r["dx_l2"] > 3.0)}
    (outdir / f"forecast_da_fullmodel{suffix}_meta.json").write_text(
        _json.dumps(meta, indent=2, allow_nan=False), encoding="utf-8")
    print(f"wrote reports/forecast_da_fullmodel{suffix}.{{md,csv}} + _meta.json")
    for row in rows:
        print(f"  {row['model']:20s} rmse={row['rmse']:.4f} gate_vs_bg={row['gate_vs_bg']}")
    print(f"  Δrmse(DA−bg)={dbg:+.4f}  physics_worse={r['physics_worse']}  "
          f"residual(bg/da)={r['bg'][1]['max_primary_residual']:.2e}/{r['da'][1]['max_primary_residual']:.2e}")


if __name__ == "__main__":
    main()
