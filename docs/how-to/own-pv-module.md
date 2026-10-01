# Use your own PV module

`pv_module` accepts catalogue keys only. An unknown name fails configuration
validation and lists what is available instead. To simulate hardware BREOS does
not ship, register it with {py:func}`~breos.pv_modules.add_module` first, then
reference it by the name you registered:

```python
import breos
from breos.pv_modules import add_module, PVModuleParams

add_module(
    "MySupplier_540W",
    PVModuleParams(
        Mpp=540.0,          # W at STC
        Vmp=41.3,           # V
        Imp=13.08,          # A
        Voc=49.4,           # V
        Isc=13.86,          # A
        T_Pmax_pct=-0.34,   # %/degC
        T_Voc_pct=-0.25,    # %/degC
        T_Isc_pct=0.045,    # %/degC
        N_Cells=144,
        Name="MySupplier 540W mono PERC",
        Module_Efficiency=0.209,
    ),
)

app = breos.App({
    "location": "porto",
    "n_modules": 10,
    "annual_consumption_kwh": 4000,
    "battery_kwh": 5.0,
    "pv_module": "MySupplier_540W",
    "cost_preset": "residential_pt",
    "emissions_country": "PT",
})
app.simulate()
```

Those nine electrical and thermal values are required, and a datasheet at STC
supplies all of them. `N_Cells` is the cell count, so 144 for a half-cut
72-cell module. The three `T_*_pct` values are temperature coefficients in
percent per degree Celsius, and `T_Pmax_pct` and `T_Voc_pct` are negative on
almost every module.

Everything after `N_Cells` is optional. `Module_Efficiency` feeds the PVsyst and
SAM cell-temperature models, and `noct-sam` refuses to run without both it and a
`NOCT` value. Set `bifaciality` if you also plan to turn on
`bifacial_model`, because the metadata alone changes nothing.

To confirm the registration took effect, resolve the same config in the current
Python process before running a full simulation. Ten 540 W modules give 5.4
kWp:

```python
from breos.app_config import resolve_app_config

resolved = resolve_app_config({
    "location": "porto",
    "n_modules": 10,
    "annual_consumption_kwh": 4000,
    "pv_module": "MySupplier_540W",
})
assert resolved.system_kwp == 5.4
```

`add_module` writes the module into the in-memory catalogue and persists
nothing. Call it before you construct `App`, and call it again in every new
process, including optimizer workers when `n_procs` is above 1. The command-line
interface starts a separate process and has no registration hook, so a custom
module needs the Python API rather than `breos run --config`.
