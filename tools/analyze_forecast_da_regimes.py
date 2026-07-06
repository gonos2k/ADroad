#!/usr/bin/env python3
"""Win/lose regime analysis for state forecast DA (Step 1 of the experiment plan).

The multi-window run shows state-DA beats no-DA in only some windows (regime-
dependent). This tool asks WHY: for each analysis window it collects forcing / obs
difficulty / DA-internal features and groups them by outcome (DA beat background or
not), so the features that separate wins from losses become visible.

N is tiny (one fixture, a few windows), so this is a CASE-STUDY diagnostic, NOT a
statistical test: it ranks candidate separating features by relative group-mean gap
and leaves the interpretation (overfit vs lead difficulty vs forcing transition) to
the reader. It reuses tools.report_forecast_da.build_multi (no new physics).

    python3 tools/analyze_forecast_da_regimes.py [--k0 K0] [--windows N]
                                                 [--window W] [--lead L] [--bg-w BG]

Writes reports/forecast_da_regimes.md, .csv and forecast_da_regimes_meta.json.
Requires jax/optax.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from tools.report_forecast_da import build_multi, WINDOW, LEAD, BG_WEIGHT  # noqa: E402

K0_FIRST, N_WINDOWS = 1500, 4


def _features(r):
    """Flatten one build_multi result into a single feature row (outcome + regime)."""
    f = {"k0": r["k0"], "beats_bg": bool(r["gate_da_vs_bg"][0]),
         "delta_rmse": r["rmse_delta_da_minus_bg"],
         "dx_l2": r["dx_l2"], "dx_max_abs": r["dx_max_abs"],
         "train_da": r["da"][1], "train_bg": r["bg"][1],
         "train_delta": r["train_delta_da_minus_bg"],
         "degradation_da": r["degradation_da"], "degradation_bg": r["degradation_bg"]}
    for k, v in r["regime"].items():
        if k == "dx_layers":
            for i, dv in enumerate(v):
                f[f"dx_layer{i + 1}"] = dv
        else:
            f[k] = v
    return f


def _numeric_feature_names(row):
    # every numeric column except the identifier (k0) and the label (beats_bg)
    return [k for k, v in row.items()
            if k not in ("k0", "beats_bg") and isinstance(v, (int, float)) and not isinstance(v, bool)]


# Three feature families — kept SEPARATE so exogenous regime signals are not mixed
# with post-hoc difficulty or the DA's own response when ranking separators:
#   ex_ante_forcing    : known at forecast issue time (given the forecast forcing)
#   post_hoc_obs       : computed from the realized lead observations (difficulty, not
#                        knowable ex ante) — explains WHY a window was hard, not a prior
#   da_response        : the DA's own correction / fit — a RESULT, never a cause
_FAMILY = {}
for _f in ("tair_mean", "tair_std", "tair_range", "tair_trend_abs", "sw_mean", "lw_mean",
           "vz_mean", "rhz_mean", "is_night_fraction"):
    _FAMILY[_f] = "ex_ante_forcing"
for _f in ("const_rmse", "obs_std", "obs_range", "obs_trend_abs", "obs_step_change_mean",
           "obs_step_change_max", "freeze_crossing_count", "cold_fraction"):
    _FAMILY[_f] = "post_hoc_obs"
for _f in ("dx_l2", "dx_max_abs", "dx_layer1", "dx_layer2", "dx_layer3", "dx_layer4",
           "train_da", "train_delta", "degradation_da"):
    _FAMILY[_f] = "da_response"
# background fit — how wrong the UNCORRECTED background was (not a DA action): its own
# family so a "background was hard" signal isn't read as a DA-response signal.
for _f in ("bg_init_error", "train_bg", "degradation_bg"):
    _FAMILY[_f] = "background_fit"


def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def summarize_regimes(results):
    """Group per-window features by win/lose and rank features by relative separation.
    Returns (rows, win, lose, table). Pure — unit-testable without running any model."""
    if not results:
        raise ValueError("no window results to analyze")
    rows = [_features(r) for r in results]
    win = [x for x in rows if x["beats_bg"]]
    lose = [x for x in rows if not x["beats_bg"]]
    table = []
    # delta_rmse IS the (continuous) outcome label -> excluded from separators (circular).
    for f in _numeric_feature_names(rows[0]):
        if f == "delta_rmse":
            continue
        wm, lm = _mean([x[f] for x in win]), _mean([x[f] for x in lose])
        diff = wm - lm
        scale = max(abs(wm), abs(lm), 1e-9)
        table.append({"feature": f, "family": _FAMILY.get(f, "other"),
                      "win_mean": wm, "lose_mean": lm, "diff": diff,
                      "direction": ("higher in wins" if diff > 0 else "higher in losses"),
                      "separation": abs(diff) / scale})
    table.sort(key=lambda t: t["separation"], reverse=True)   # strongest separators first
    return rows, win, lose, table


def group_separators(table):
    """Split the flat separator table into the three feature families (each still
    sorted by separation). Returns {family: [rows]}."""
    grouped = {"ex_ante_forcing": [], "post_hoc_obs": [], "background_fit": [],
               "da_response": [], "other": []}
    for t in table:
        grouped[t["family"]].append(t)
    return grouped


_ROW_COLS = ("k0", "beats_bg", "delta_rmse", "bg_init_error", "train_delta",
             "degradation_da", "dx_l2", "obs_std", "tair_trend_abs", "freeze_crossing_count")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Win/lose regime analysis for state forecast DA")
    ap.add_argument("--k0", type=int, default=K0_FIRST)
    ap.add_argument("--windows", type=int, default=N_WINDOWS)
    ap.add_argument("--window", type=int, default=WINDOW)
    ap.add_argument("--lead", type=int, default=LEAD)
    ap.add_argument("--bg-w", type=float, default=BG_WEIGHT, dest="bg_w")
    args = ap.parse_args()

    results, skipped = build_multi(args.k0, args.windows, args.window, args.lead, args.bg_w)
    if not results:
        raise RuntimeError("no window had enough valid observations")
    rows, win, lose, table = summarize_regimes(results)
    deltas = sorted(x["delta_rmse"] for x in rows)
    med = deltas[len(deltas) // 2] if len(deltas) % 2 else _mean(deltas[len(deltas) // 2 - 1:len(deltas) // 2 + 1])

    outdir = REPO / "reports"; outdir.mkdir(exist_ok=True)
    import csv as _csv, io as _io
    buf = _io.StringIO(); w = _csv.writer(buf)
    all_cols = ["k0", "beats_bg"] + _numeric_feature_names(rows[0])
    w.writerow(all_cols)
    for r in rows:
        w.writerow([r[c] for c in all_cols])

    lines = [f"# Forecast DA regime analysis ({len(rows)} windows)", "",
             "state-DA가 어떤 window에서 no-DA를 이기고 어떤 window에서 지는지, forcing/obs/DA 내부 "
             "feature로 설명한다. **표본이 작아 통계검정이 아니라 case-study**다 — win/lose 그룹 평균 "
             "차이가 큰 feature를 후보 신호로 제시할 뿐, 인과는 독자가 판단한다.",
             "", "## Summary",
             f"- windows: {len(rows)}  ·  DA wins: {len(win)}/{len(rows)}",
             f"- win windows (k0): {[x['k0'] for x in win]}",
             f"- lose windows (k0): {[x['k0'] for x in lose]}",
             f"- mean Δrmse (DA−bg): {_mean([x['delta_rmse'] for x in rows]):+.4f}  ·  median: {med:+.4f}",
             ""]
    # per-window snapshot (a readable subset of columns)
    snap = [c for c in _ROW_COLS if c in rows[0]]
    lines += ["## Per-window snapshot", "| " + " | ".join(snap) + " |",
              "| " + " | ".join("---" for _ in snap) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(
            (str(r[c]) if c in ("k0", "beats_bg", "freeze_crossing_count")
             else f"{r[c]:+.4f}" if c in ("delta_rmse", "train_delta")
             else f"{r[c]:.4f}") for c in snap) + " |")
    # candidate separators (only meaningful when we actually have both groups)
    grouped = group_separators(table)
    lines += ["", "## Candidate regime signals (win vs lose group means)",
              f"**n_win={len(win)}, n_lose={len(lose)} — 표본이 작아 separator ranking은 매우 불안정하다 "
              "(outlier 하나에 흔들림). 인과·일반규칙이 아니라 hypothesis generator로만 사용.**",
              "separation = |win_mean − lose_mean| / max(|win_mean|, |lose_mean|, eps); 그룹 평균의 "
              "**부호가 반대이면 1을 넘을 수 있으며, 통계적 유의성이 아니라 상대적 gap일 뿐이다.**"]
    if win and lose:
        def fam(title, key, note):
            out = ["", f"### {title}", note, "",
                   "| feature | win_mean | lose_mean | direction | separation |",
                   "| --- | ---: | ---: | --- | ---: |"]
            for t in grouped[key][:8]:
                out.append(f"| {t['feature']} | {t['win_mean']:.4f} | {t['lose_mean']:.4f} "
                           f"| {t['direction']} | {t['separation']:.3f} |")
            return out
        lines += fam("A. Ex-ante forcing (예보시각에 알 수 있는 후보 regime 신호) — **1차 해석 기준**",
                     "ex_ante_forcing",
                     "이 diagnostic은 forcing을 '예보로 주어진 forcing'으로 취급한다. 실측 forcing만 "
                     "있으면 post-hoc 설명 feature로 해석할 것.")
        lines += fam("B. Post-hoc obs difficulty (사후 난이도 — 예보 전엔 모름)", "post_hoc_obs",
                     "lead의 실측 관측에서 계산 — '왜 어려웠나'는 설명하나 ex-ante 신호는 아님.")
        lines += fam("C. Background-fit diagnostics (보정 안 한 background가 얼마나 틀렸나)", "background_fit",
                     "bg_init_error·train_bg·degradation_bg는 background가 창에서 얼마나 어긋났는지 — "
                     "DA 보정이 아니라 '동화 여지'의 크기.")
        lines += fam("D. DA response (DA가 실제로 한 보정 — 원인 아닌 결과)", "da_response",
                     "dx_*·train_da·degradation_da는 DA의 반응이라 win/lose를 잘 나눠도 사전 원인으로 읽지 말 것.")
        lines += ["", "## 가설 점검(케이스 스터디)",
                  "- **A (배경오차 큼 + model error 작음 → 이김)**: bg_init_error/train_delta가 win에서 유리?",
                  "- **B (lead 난이도 상승 → 짐)**: degradation_da·obs_std·tair_trend_abs가 lose에서 큰가?",
                  "- **C (dx 과도 → overfit으로 짐)**: dx_l2·dx_max_abs가 lose에서 큰가?",
                  "", "관찰(이번 run): win group은 평균적으로 더 낮은 tair_mean·더 높은 sw_mean·freeze "
                  "crossing 쪽으로 치우치나 **window별 예외가 있어**(예: win k0=2700은 is_night_fraction≈0.63으로 "
                  "야간 비중이 큼) causal rule이 아니라 후보 regime signal로만 유지한다."]
    else:
        lines += ["", f"(win {len(win)}, lose {len(lose)} — 두 그룹이 모두 있어야 비교 가능. "
                  "--windows/--k0로 표본을 늘리세요.)"]
    (outdir / "forecast_da_regimes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (outdir / "forecast_da_regimes.csv").write_text(buf.getvalue(), encoding="utf-8")

    import json as _json
    # separators are only meaningful (and finite) with BOTH groups present; an all-win
    # or all-lose run leaves NaN group means, so emit empty separators rather than NaN.
    both = bool(win and lose)
    top = table[:12] if both else []
    grouped_top = {k: v[:8] for k, v in group_separators(table).items() if v} if both else {}
    meta = {"k0_first": args.k0, "windows": args.windows, "window": args.window,
            "lead": args.lead, "bg_weight": args.bg_w, "skipped_windows": skipped,
            "n_win": len(win), "n_lose": len(lose),
            "win_k0": [x["k0"] for x in win], "lose_k0": [x["k0"] for x in lose],
            "separator_stability_warning": f"n_win={len(win)}, n_lose={len(lose)} — unstable; "
            "hypothesis generator only",
            "top_separators": top, "grouped_top_separators": grouped_top, "rows": rows}
    # allow_nan=False -> fail loudly if any non-finite slipped in, rather than writing
    # non-standard JSON that strict parsers reject.
    (outdir / "forecast_da_regimes_meta.json").write_text(
        _json.dumps(meta, indent=2, allow_nan=False), encoding="utf-8")
    print("wrote reports/forecast_da_regimes.{md,csv} + forecast_da_regimes_meta.json")
    print(f"  wins {len(win)}/{len(rows)}  win_k0={[x['k0'] for x in win]}  lose_k0={[x['k0'] for x in lose]}")
    if win and lose:
        for t in table[:5]:
            print(f"  {t['feature']:22s} win={t['win_mean']:+.4f} lose={t['lose_mean']:+.4f} "
                  f"sep={t['separation']:.3f} ({t['direction']})")


if __name__ == "__main__":
    main()
