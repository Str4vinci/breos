# Use an external load profile

Only the demandlib-derived H0 profile (`"1"`, alias `"demandlib_h0"`) ships
with BREOS. For the other standard profiles, download the source CSVs yourself
under terms that permit your use, put them in a local directory, and point
`rlp_directory` at it.
[Load Profile Data](../legal/load-profile-data.md) lists the exact expected
filenames per profile key:

```toml
rlp_directory = "external_rlp"
location = "porto"
n_modules = 10
annual_consumption_kwh = 4000
battery_kwh = 5.0
load_profile = "eredes_btn_c"
resolution = "15min"
cost_preset = "residential_pt"
emissions_country = "PT"
```

A runnable template also ships in the repository as
`configs/examples/external-rlp.toml`.
