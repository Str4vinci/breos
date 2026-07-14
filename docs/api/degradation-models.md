# Battery degradation models

Native Naumann/Lam degradation remains the default. BLAST-Lite is an explicit
opt-in cell-model engine:

```toml
degradation_engine = "blast"
blast_model = "lfp_gr_250ah_prismatic"
```

The same selection is available from the CLI:

```bash
breos run --config config.toml \
  --degradation-engine blast \
  --blast-model lfp_gr_250ah_prismatic
```

Discover model identities and scientific scope from Python or the CLI:

```python
from breos import get_battery_model_profile, list_battery_models

models = list_battery_models()
lfp = get_battery_model_profile("lfp_gr_250ah_prismatic")
```

```bash
breos list battery-models
breos list battery-models --json
```

The registry reports stable keys, readable names, chemistry, cell form factor,
nominal cell capacity, experimental ranges, study citations, capacity and
resistance outputs, upstream BLAST provenance, and whether the model is enabled
in the current integration phase. All 14 vendored models are discoverable;
the profile-layer PR enables only the two core models end to end. The remaining
models are enabled by the separately validated all-model parity phase.

## Configuration precedence

Resolved settings use this order:

1. explicit user configuration;
2. sourced model-profile defaults;
3. global BREOS defaults.

No BLAST paper bundled here defines generic pack operating limits, so the
profiles currently contain no invented SOC, efficiency, power, or replacement
defaults. Existing global settings therefore remain unchanged unless a user
overrides them.

## Migration from `battery_type`

Do not use the legacy `battery_type` selector in `App` configuration. It was
ambiguous: it mixed chemistry identity with degradation-model selection.
Choose `degradation_engine="native"` (or omit it) for the existing LFP
Naumann/Lam model. Choose `degradation_engine="blast"` together with one
stable `blast_model` key for BLAST. Supplying `blast_model` while the native
engine is active raises instead of silently changing behavior.

The lower-level `BatteryConfig.battery_type` field remains temporarily limited
to `"lfp"` for native cycle-aging compatibility; it does not select BLAST.

## Result interpretation

BLAST results include a `degradation` block and matching provenance with the
engine, stable model key, complete model profile, initial/final SOH,
replacement events, cell-model versus pack-calibrated status, warning lists,
and serialized state-schema version. These are empirical **cell models**, not
pack-calibrated field models. Long-horizon BLAST SOH is rounded to one decimal
place to avoid implying unsupported precision.

BLAST plus Monte Carlo is rejected explicitly in 0.4.0. It never falls back to
native degradation. BLAST resistance outputs are reported as capabilities but
do not alter dispatch efficiency or power limits.

