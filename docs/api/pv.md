# PV

DC and AC production, multi-array layouts, the built-in module catalogue,
and inverter sizing.

BREOS uses pvlib for solar position, irradiance transposition, cell
temperature, and PV performance model pieces. The functions below document the
BREOS wrapper surface; for PV modeling background and parameter references,
start with the [pvlib documentation](https://pvlib-python.readthedocs.io/en/stable/)
and the project [Resources](../resources.md) page.

## Production

```{eval-rst}
.. autosummary::
   :toctree: generated/

   breos.solar.calculate_pv_production_dc
   breos.solar.calculate_pv_production_ac
   breos.solar.calculate_multi_array_production
   breos.solar.dc_to_ac
```

## Module catalogue

A built-in dictionary of PV module electrical parameters lives in
`breos.pv_modules.MODULES`. Use the accessor functions below rather than
indexing the dict directly — `get_module` raises a clear error on
unknown keys, and `list_modules` returns the available keys.

```{eval-rst}
.. autosummary::
   :toctree: generated/

   breos.solar.PVModuleParams
   breos.pv_modules.get_module
   breos.pv_modules.list_modules
   breos.pv_modules.get_module_info
```

## Geometry

```{eval-rst}
.. autosummary::
   :toctree: generated/

   breos.solar.estimate_optimal_tilt
   breos.solar.default_azimuth
```

## Inverter

Common inverter configurations live in `breos.inverter.INVERTER_PRESETS`.
Use `get_inverter_preset` to look one up by key.

```{eval-rst}
.. autosummary::
   :toctree: generated/

   breos.inverter.InverterConfig
   breos.inverter.get_inverter_preset
```
