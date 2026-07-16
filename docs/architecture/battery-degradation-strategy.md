# Battery Degradation Strategy for 0.4 and Beyond

This note fixes the scientific and release policy for battery-degradation work
after BREOS 0.3.3. It separates near-term correctness work from the larger BLAST
integration and defines the evidence required before any default changes.

## Release order

Battery work must respect this integration order:

```text
0.3.3
  ↓
PR #58 (validation suite, mid-interval solar position, optimizer/App parity)
  ↓
PR #59 (diffuse IAM and mount-type cell-temperature presets)
  ↓
integrated correctness PR
  ↓
0.3.4
  ↓
small post-release fixes, if needed
  ↓
0.3.5
  ↓
BLAST and other larger changes
  ↓
0.4.0
```

PR #59 is intentionally stacked on PR #58. The BLAST work is a separate 0.4
line and must not be folded into either PR or into the integrated correctness
PR. In particular, changes to shared integration surfaces such as
`app_config.py`, `battery.py`, `app_results.py`, `cli.py`, and the common tests
should be rebased or merged only after the 0.3.4 sequence has landed. This note
and the BLAST design document can evolve independently in the meantime.

## Default policy

Keep the native Naumann/Lam engine as the generic residential-LFP default. Do
not replace it with BLAST, or change the native calibration variant, on the
basis of synthetic trajectories, in-sample fit, or BLAST parity tests.

The current alias `naumann_lam_field_calibrated` continues to select native v1.
Native v2 remains an explicit sensitivity model. The repository records:

| Native calibration | Full-fit RMSE | Leave-one-system-out result | Policy |
|---|---:|---:|---|
| v1 | 4.40 percentage points | mean RMSE 6.00 pp | Current default |
| v2 | 4.65 percentage points | mean RMSE 5.49 pp; pooled RMSE 5.61 pp | Candidate future default |

V2 fits the calibration set slightly worse but generalizes better under
leave-one-system-out testing. That is encouraging, not decisive: the systems
participating in model selection are not a truly held-out validation set. A
default switch requires field data excluded from fitting, calibration choices,
and model selection.

## Comparison program

The primary native-versus-BLAST comparison is native Naumann/Lam against
BLAST's `lfp_gr_sonymurata_3ah` profile. Both ultimately relate to the Naumann
Sony/Murata 3 Ah LFP dataset, making this the most interpretable
apples-to-apples comparison available. The BLAST 250 Ah prismatic LFP profile
is useful for stationary sensitivity studies, but it is not the primary test of
the difference between the two model formulations.

The validation sequence is:

1. Verify the reference Python native path and vendored BLAST path on identical
   absolute-SOC and cell-temperature histories.
2. Compare native v1, native v2, and BLAST Sony/Murata 3 Ah on the common source
   dataset, reporting both calibration error and leave-one-system-out error.
3. Freeze all model choices, then evaluate them on a genuinely held-out field
   dataset with pack metadata and no parameter refitting.
4. Report error by system as well as pooled error so one long or densely sampled
   system cannot dominate the result.
5. Reconsider the generic default only after the held-out result supports a
   change and the practical effect on residential simulations is documented.

Parity against upstream BLAST establishes implementation fidelity; it does not
establish real-world predictive validity.

## Named BLAST profiles

A `blast_model` value selects a specific empirical cell model, not a generic
chemistry curve. User-facing discovery must therefore present each model as a
named cell/chemistry profile and show, at minimum:

- the stable BREOS key and human-readable cell/profile name;
- chemistry, nominal cell capacity or form factor where known;
- the published experimental range used by runtime extrapolation warnings;
- links to the source aging and model-identification studies; and
- whether capacity only or capacity plus resistance was modeled.

The catalog and sources are recorded in
[BLAST Degradation Engine](blast-degradation-engine.md#user-facing-cell-model-profiles).
Future CLI or Python discovery should expose that same registry rather than
maintain a second list. Chemistry-wide operating defaults, if later added,
must remain distinct from the cell-specific degradation equations.

## Make uncertainty visible

A multi-decade SOH value is model-dependent. A result such as `83.4%` must not
be presented as if its decimal precision were predictive certainty.

For long-horizon studies, expose a degradation-model sensitivity range over
plausible models, initially native v1, native v2, and BLAST Sony/Murata 3 Ah
when its experimental range is relevant. Label this a **model sensitivity
range**, not a statistical confidence interval, unless a probabilistic model
has actually been calibrated. Preserve the individual model values and keys so
the range is auditable. Surface experimental-range and aging-horizon warnings
alongside the result.

This should be added after the integrated correctness result schema lands; it
must not be retrofitted independently into the pre-0.3.4 result work.

## Cell-to-pack calibration

BLAST predicts expected cell-level life for the cell and study behind the
selected profile. Residential systems are packs. Thermal gradients,
cell-to-cell variation, imbalance, interconnect losses, and BMS behavior can
make pack-level usable-capacity and power fade worse than the cell model.

Add a pack-level calibration layer only when a suitable dataset includes the
information needed to identify it. Prefer an explicit, documented mapping over
an unexplained degradation multiplier, and validate it on different packs from
those used to fit it. Until then, label BLAST outputs as cell-model projections
and do not claim pack-level calibration.

## Accuracy and screening paths

Retain the reference Python degradation path for accuracy and validation
studies. It uses full rainflow cycle counting and is the comparison basis for
native-versus-BLAST work.

The optional Numba degradation path uses a segment/extrema depth-of-cycle
approximation rather than full rainflow counting. Keep it clearly identified as
a screening approximation. It must not generate calibration evidence or
replace the reference path in published accuracy comparisons merely because it
is faster.

## Resistance fade

Do not feed BLAST resistance output into BREOS efficiency or power limits until
there is a defensible mapping from each model's resistance metric and test
protocol to pack-level round-trip efficiency, charge/discharge power limits,
and BMS behavior. BLAST models expose different resistance states, and a
generic `r → efficiency` conversion would imply unsupported physics.

Until that mapping is documented and validated, `degradation_engine="blast"`
must remain incompatible with BREOS resistance-fade coupling. Resistance may
be retained as diagnostic model output without affecting dispatch.

