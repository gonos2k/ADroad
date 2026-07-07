# Independent-case manifest (Step 4 toward real promotion)

## Why this exists

`promotion_gate` (design §11, `droad/skill_gate.py`) intentionally distinguishes a **case**
(an independent station/day) from a **window** (a slice of one fixture). Every result so far
— dry multi-window, full-model A0 multi-window, the A0 grid — comes from a **single fixture**,
so `promotion_gate` is called with `n_cases=1` and the verdict is `REPORT_ONLY` no matter how
many windows pass. This is deliberate: a model that assimilates well on four windows of *one*
day/station has not demonstrated it generalizes.

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
checks each case's schema/semantics and reports whether the manifest is *promotion-ready as an
evidence base*:

- no schema/semantic errors,
- `n_cases ≥ 3` (matches `promotion_gate(min_cases=3)`),
- `≥ 2` distinct regimes,
- no duplicate station-day cases (two "cases" on the same station and date are not
  independent).

`promotion_ready = True` means only that the evidence base is big and diverse enough to *run*
promotion — never that a model passed. Skill/physics still have to beat baseline in every case
at run time.

## Minimum target

A defensible first promotion attempt: **≥ 3 regimes × ≥ 3 station-days = ≥ 9 independent
cases**. Fewer is fine for iteration but stays `REPORT_ONLY`.

## What NOT to do

- Do not count windows from one fixture as cases (that is the `n_cases=1` situation the whole
  design guards against).
- Do not fabricate cases to reach the threshold; each must map to real forcing + obs.
- Do not read `promotion_ready` as "the model is promotable" — it is purely about evidence
  volume/diversity.

## Next increments (not in this skeleton)

1. A `run_cases` driver: for each case, run the chosen A0 setting and collect per-case
   gate/physics/residual, then feed `n_cases`, `windows_beat_baseline`, aggregate deviation to
   `promotion_gate` for a real (non-`n_cases=1`) verdict.
2. Pick the A0 setting from the grid stability region (Step 3 result) before running cases, so
   the case study isn't confounded by an untuned `bg_w/window/lead`.
