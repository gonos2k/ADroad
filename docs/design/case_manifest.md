# Independent-case manifest (Step 4 toward real promotion)

## Why this exists

`promotion_gate` (design §11, `droad/skill_gate.py`) intentionally distinguishes a **case**
(one station over one non-overlapping time interval) from a **window** (a slice of one
fixture). Every result so far — dry multi-window, full-model A0 multi-window, the A0 grid —
comes from a **single fixture**, so `promotion_gate` is called with `n_cases=1` and the verdict
is `REPORT_ONLY` no matter how many windows pass. This is deliberate: a model that assimilates
well on four windows of *one* interval has not demonstrated it generalizes.

Independence is enforced by **interval overlap per station**, not by calendar date: two cases
at the same station whose `[start, end)` intervals overlap are not independent (even if their
start dates differ), while two non-overlapping intervals at the same station on the same day
are allowed. Station-days may be summarized separately for reporting, but the *contract* is
non-overlapping station intervals.

The only way to raise `n_cases` honestly is to assemble a manifest of genuinely distinct
cases. This document defines that manifest and its validation contract. Building it does not
make any model promotable — it makes the *evidence base* large enough to even ask the
question.

## Schema

A manifest is a YAML file with a `cases` list. Each case:

Required: `case_id` (unique, non-empty), `station`, `start`, `end` (ISO-8601, `start < end`),
`regime`. Optional: `forcing_source`, `obs_source`, `notes`. Unknown fields are rejected so a
typo can't silently pass.

`regime` is one of: `dry_cold`, `warm_wet`, `freeze_transition`, `precip_snow`,
`melt_refreeze`. The regime tag matters because the forecast-DA finding is **regime-dependent**
(state-DA helped in colder / daytime / freeze-crossing windows and hurt in warm nighttime
ones — see the regime analysis). A promotion assembled only from `dry_cold` days would not be
credible, so coverage across regimes is part of the contract.

See `cases.example.yaml` for the schema in use (example values, not real observations).

## Validation contract

`tools/validate_cases.py` makes the schema an **executable gate**, matching the project rule
that a contract is code, not a doc note (like `promotion_gate` itself). `validate_manifest`
checks each case's schema/semantics and reports two readiness tiers, both describing the
**evidence base** — never model promotability:

- `minimum_evidence_ready`: no errors, `n_cases ≥ 3` (matches `promotion_gate(min_cases=3)`),
  `≥ 2` regimes, and no overlapping cases at the same station. Enough to *run* a first
  promotion attempt.
- `recommended_promotion_ready`: no errors, `≥ 3` regimes, **`≥ 3` cases in every covered
  regime** (so a lopsided `[7, 1, 1]` does not pass), no overlaps. This literally enforces the
  "3 regimes × 3 cases each" target below rather than a bare 9-total.

Datetimes must include a time (no date-only) and start/end must share timezone-awareness so
comparisons are well-defined.

Both flags mean only that the evidence base is large/diverse enough to *attempt* promotion —
never that a model passed. Skill/physics still have to beat baseline in every case at run time.

## Recommended target

A defensible first promotion attempt: **≥ 3 regimes with ≥ 3 non-overlapping cases each
(≥ 9 total)**, which is what `recommended_promotion_ready` enforces. Fewer still allows a run
once `minimum_evidence_ready` holds, but the result stays weak and every case is `REPORT_ONLY`
until it beats baseline.

## What NOT to do

- Do not count windows from one fixture as cases (that is the `n_cases=1` situation the whole
  design guards against).
- Do not fabricate cases to reach the threshold; each must map to real forcing + obs.
- Do not read either readiness flag (`minimum_evidence_ready`, `recommended_promotion_ready`)
  as "the model is promotable" — they are purely about evidence volume/diversity.

## run_cases (the promotion path)

`tools/run_cases.py` implements the aggregation + driver: `run_manifest` validates the
manifest, requires the `minimum`/`recommended` evidence tier, runs `run_one(case, setting)`
per case, and `summarize_cases` feeds the per-case verdicts to `promotion_gate` with the
**real n_cases** — so a robust result can finally be `PROMOTE` (the single-fixture tools are
pinned to `n_cases=1` and can only ever be `REPORT_ONLY`). Every honest block still applies:
one failing case, too few cases, or a dirty aggregate residual keeps it `REPORT_ONLY`.

Still pending (needs data we don't have yet):

1. A per-case forcing/obs loader for `run_one` — the default raises `NotImplementedError`
   because only one fixture exists. The aggregate→gate contract is locked and tested; the
   loader is the remaining wiring once multiple real cases exist.
2. Pick the A0 setting from the grid stability region (Step 3 result) before running cases, so
   the case study isn't confounded by an untuned `bg_w/window/lead`. **Done**: the A0 grid
   (`reports/forecast_da_fullmodel_grid.*`, 12 combos) found no combo exceeds `gate_pass_rate`
   0.50 (A0 stays regime-dependent, REPORT_ONLY everywhere), and the safest combo is
   **`bg_w=0.2, window=60, lead=240`** — top `gate_pass_rate` tier with `physics_worse_rate=0`,
   the smallest `worst_delta_rmse` (+0.019), and no `state_large`. Use this as the fixed A0
   setting for `run_cases`. Note the ceiling: at 0.50 gate-pass a real promotion is still
   unlikely (beats baseline in only 2/4 windows), so independent cases are expected to remain
   regime-dependent — the value of `run_cases` is an honest verdict, not a guaranteed PROMOTE.
