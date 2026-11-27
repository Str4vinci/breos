# Study 13 — How much NPV does the replacement rule leave on the table?

**Question.** BREOS replaces a battery when its health crosses `eol_percentage` (0.70), optionally
skipping late swaps. #403 proposes an economic rule instead: at each crossing, compare replace,
keep and retire. Before building it: how much is there to gain? This study brute-forces the rule
family BREOS already has, and takes the best as a lower bound on what an economic rule could gain.

**Answer.** A lot, and always in the same direction. In all 126 cases the best rule is to buy no
replacement: either never replace, or set the threshold so low that no battery reaches it. It
beats the default by 0.7–9.7 k€ of projected NPV (median 1.5–9.7 k€ by model and size at
base prices), at a cost of 2–8 points of self-sufficiency. Retiring a worn pack instead of keeping
it never helped. With the default ageing model a 5 kWh battery gains a median 2.9 k€, a 10 kWh one
6.3 k€.

## Setup

Seven sites on their PVGIS TMY, 10 modules with a 5 or 10 kWh battery, the shipped
`projected-optimization.toml` household (Portuguese costs, 20 years, hourly). Ageing models:
field-calibrated v1 (default), v2, and BLAST's LMO cell (the only BLAST chemistry that reaches end
of life within 20 years here; BLAST LFP and the NMC cell tried never do). Cost cases: base storage
500 €/kWh, 300 €/kWh, and base with replacement prices falling 5 %/yr. Policies: `eol_percentage`
0.50–0.80 × `replacement_min_remaining_years` 0–7 × `skipped_replacement_action` keep/retire,
plus `enable_replacement = false`: 92 policies.

## Results

| Model, size (base prices) | Default replacements | Median gain of the best rule | Gain of never replacing | Self-sufficiency lost |
|---|---|---|---|---|
| v1, 5 kWh | 1–2 | 2.9 k€ | 2.9 k€ | 4 pp |
| v1, 10 kWh | 1–2 | 6.3 k€ | 6.3 k€ | 3 pp |
| v2, 5 kWh | 1 | 1.5 k€ | 1.5 k€ | 3 pp |
| v2, 10 kWh | 1–2 | 3.5 k€ | 3.5 k€ | 2 pp |
| BLAST LMO, 5 kWh | 2–3 | 4.6 k€ | 4.6 k€ | 8 pp |
| BLAST LMO, 10 kWh | 2–3 | 9.7 k€ | 9.7 k€ | 7 pp |

Cheaper storage (300 €/kWh) or 5 %/yr price learning halves the gains, but never-replace still wins
every case. Of the existing knobs, lowering the threshold alone recovers about 55–100 % of the gain,
and raising `replacement_min_remaining_years` alone recovers 0–48 %.

Without replacements the battery ends the 20 years at 47–52 % health (v1), 52–65 % (v2) or
29–48 % (BLAST LMO), and BREOS keeps dispatching it at that capacity.

## Why

The battery itself doesn't pay at these prices. Study 11 found the NPV-best design has no battery
at any site, and long-run study 10 found a 5 kWh battery positive nowhere with the default model.
A replacement is a second purchase of the same money-losing asset, with fewer years left to earn.
A worn pack at 50–70 % health still delivers most of the self-consumption benefit for a
household this size, so keeping it is almost free.

## Caveats

- The projected NPV has no terminal value. That treats the default rule's late pack harshly (no
  credit for its remaining life), but it treats never-replacing kindly too (no penalty for a pack
  at 30–50 % health). Long-run study 3 showed terminal value merges the late-replacement bands.
  It wouldn't change the sign here: replacements at year ~10 also lose.
- BREOS doesn't model what happens to a battery far below 70 % (warranty, safety, power fade,
  failure). The v1/v2 fits and the BLAST LMO cell are extrapolated there. "Never replace" means
  "run the worn pack to the end of the study", which real systems may not allow.
- Self-sufficiency falls 2–8 points. A user who values autonomy may still want replacements.
  That's a choice of objective, not something the NPV can settle.

## Takeaways for #403

1. An economic replace/keep/retire decision would change the answer in every case tested. The
   size of the gain (1–10 k€) is much larger than any other battery setting studied
   (cf. studies 3, 9, 11).
2. In this range, "keep" dominates "retire": an economic rule could probably start with
   replace vs keep.
3. Until #403 exists, the docs could recommend a lower `eol_percentage` (0.5) or
   `enable_replacement = false` for NPV-driven studies, and say why the 0.70 default exists
   (warranty conventions, not economics).
