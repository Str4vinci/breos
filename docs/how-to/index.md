# How-to guides

Short, task-focused instructions with a configuration to copy. For worked
case studies with stored results and figures, see the
[case examples](../gallery/index.rst).

Save a TOML block as `config.toml`, then:

```bash
breos validate-config config.toml                  # check resolved choices first
breos run --config config.toml --output result.json
```

Every key works the same in a Python dict passed to {py:class}`~breos.App`.
Valid keys for locations, modules, cost presets, emissions countries and load
profiles are on the [packaged options](../getting-started/options.md) page, or
run `breos list`.

```{toctree}
:maxdepth: 1

custom-location
own-pv-module
partial-year
offline-weather
parameter-sweep
fifteen-minute-resolution
sky-model-and-horizon
external-load-profile
```
