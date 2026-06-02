# Configuration Examples

BREOS packages default presets internally so installed users do not need this directory at runtime. These files are editable examples and templates for experiments, scripts, and agents.

- `base/` mirrors the packaged JSON presets for locations, costs, and emissions.
- `examples/` contains runnable TOML or JSON configs for `breos run --config`.

Keep public examples on the bundled `load_profile = "1"` unless the example explicitly documents an external, user-licensed RLP directory.

For external RLPs, use `configs/examples/external-rlp.toml` as a template and put the licensed CSV files in a local directory such as `external_rlp/`.
