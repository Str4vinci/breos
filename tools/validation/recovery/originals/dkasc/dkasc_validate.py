"""Drive the BREOS PV chain from the DKASC Alice Springs measurements.

This is the part the NIST Gaithersburg run could not do. NIST publishes its
pyranometers as raw millivolts with no sensitivity constant, so that validation
had to be driven from *measured* plane-of-array irradiance, leaving
decomposition and transposition completely untested. DKASC logs Global
Horizontal and Diffuse Horizontal radiation in W/m2, so the whole chain runs:

    GHI, DHI -> DNI -> transposition -> POA -> IAM -> cell temp -> DC -> AC

Two independent references are available and both are used:

* A tilted pyranometer at 20 deg / azimuth 0, co-planar with every fixed array
  on the site. Modelled POA can be compared against it with no PV model in the
  way at all.
* Arrays 16A and 16D: physically identical 1.98 kW BP 3165J arrays, same
  inverter model, commissioned the same day, at azimuth 0 (north, sun-facing in
  the southern hemisphere) and azimuth 270 (west). A transposition model has to
  reproduce both from one irradiance record. Module and inverter error is
  common to the pair and cancels in the north/west ratio, which is why that
  ratio is reported alongside the absolute biases.

Measurement reference for AC is ``Active_Power`` (kW, Class 0.5 meter),
cross-checked against the meter's own cumulative ``Active_Energy`` counter --
see dkasc_facts.py, which reports agreement within 0.3-1.0 % per array-year
from 2009 to 2020 and the sharp degradation after 2020 that puts those years
out of bounds.
"""

from __future__ import annotations

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pvlib
from pvlib.location import Location

from breos.inverter import _calculate_dc_ac_power_arrays
from breos.pv.model_options import resolve_pv_model_options
from breos.solar import (
    _compute_irradiance_and_cell_temp_detail,
    _prepare_solarpos_and_weather,
    calculate_pv_production_breakdown,
)

from dkasc_arrays import ARRAYS, SITE_ALT, SITE_LAT, SITE_LON

FREQ = "5min"
STEP_H = 5.0 / 60.0
# Alice Springs is in the Northern Territory, whose IANA zone is
# Australia/Darwin: permanently UTC+9:30 with no daylight saving, i.e. ACST all
# year. This is the site's own zone, not a stand-in, so no DST transition can
# ever be implied. (Australia/Adelaide is also +9:30 but observes DST and would
# be wrong here.) dkasc_facts.py confirms +9:30 against the measured sun.
ACST = ZoneInfo("Australia/Darwin")

# Named loss configurations, matching the NIST run so the two sites can be read
# against each other. "breos-default" is DEFAULT_PVWATTS_LOSSES -- what a BREOS
# user gets untouched. "no-availability" removes the 3 % availability allowance,
# which double-counts when the measured series already contains whatever
# downtime occurred. "module-only" strips the stack to isolate the physics.
LOSS_CASES = {
    "breos-default": None,
    "no-availability": {"availability": 0.0},
    "no-availability-no-shading": {"availability": 0.0, "shading": 0.0},
    "module-only": {k: 0.0 for k in (
        "soiling", "shading", "snow", "mismatch", "wiring",
        "connections", "lid", "nameplate_rating", "availability")},
}

# Every transposition model BREOS exposes. perez-driesse is pvlib's continuous
# reformulation of Perez; the eleven Perez coefficient sets are swept separately
# in dkasc_transposition.py.
TRANSPOSITION_MODELS = ("isotropic", "klucher", "haydavies", "reindl", "king",
                        "perez", "perez-driesse")


def location() -> Location:
    return Location(SITE_LAT, SITE_LON, tz=ACST, altitude=SITE_ALT)


def load(path) -> pd.DataFrame:
    """Load a frame written by dkasc_build.py, localised to ACST."""
    d = pd.read_csv(path, index_col=0, parse_dates=True)
    d.index = d.index.tz_localize(ACST)
    # Wind is absent from late October 2016 onward and carries rare spikes that
    # dkasc_build already screened. Interpolate over short gaps and fall back to
    # the site median rather than discarding otherwise-complete minutes; the
    # sensitivity of the results to this fill is reported in dkasc_analysis.py.
    wind = d["wind_speed"].interpolate(limit=12)
    d["wind"] = wind.fillna(wind.median() if wind.notna().any() else 2.5)
    d["temp_air"] = d["temp_air"].interpolate(limit=12)
    return d


def solar_position(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Sun position evaluated *at* the stamp.

    DKASC's stamps label the centre of the 5-minute averaging window (see
    dkasc_facts.py: fitting measured GHI against clear-sky GHI minimises at
    -0.25 min, where interval-start would give +2.5 and interval-end -2.5), so
    the correct convention is BREOS's default ``solar_position="interval-start"``
    -- no offset. This differs from the NIST array, whose Campbell logger
    stamped 1-minute means at interval end.
    """
    return location().get_solarposition(index)


def derive_dni(d: pd.DataFrame, method: str, solpos: pd.DataFrame | None = None) -> pd.Series:
    """Return DNI in W/m2 by one of four routes.

    ``closure`` is the geometric identity DNI = (GHI - DHI) / cos(zenith). With
    both GHI and DHI measured it involves no model at all, which is what makes
    the transposition test in dkasc_transposition.py clean: any error it leaves
    is transposition error, not decomposition error.

    ``disc``, ``dirint`` and ``erbs`` estimate DNI from GHI alone. They are run
    to quantify what a GHI-only weather source -- the usual case, and the case
    at Esposende -- adds on top.
    """
    solpos = solar_position(d.index) if solpos is None else solpos
    zen = solpos["apparent_zenith"]
    ghi = d["ghi"]

    if method == "closure":
        cosz = np.cos(np.radians(zen.to_numpy()))
        dni = np.divide(
            (ghi - d["dhi"]).to_numpy(), cosz,
            out=np.zeros(len(d)), where=cosz > np.cos(np.radians(85.0)),
        )
        # Cap at the extraterrestrial normal irradiance: the closure is
        # numerically unstable near sunrise and sunset, where a small DHI error
        # divided by a small cosine produces a physically impossible DNI.
        extra = pvlib.irradiance.get_extra_radiation(d.index).to_numpy()
        return pd.Series(np.clip(dni, 0.0, extra), index=d.index)
    if method == "disc":
        return pvlib.irradiance.disc(ghi, zen, d.index)["dni"]
    if method == "dirint":
        return pvlib.irradiance.dirint(ghi, zen, d.index, temp_dew=None)
    if method == "erbs":
        return pvlib.irradiance.erbs(ghi, zen, d.index)["dni"]
    raise ValueError(f"unknown DNI method {method!r}")


def weather_frame(d: pd.DataFrame, dni_method: str = "closure",
                  solpos: pd.DataFrame | None = None) -> pd.DataFrame:
    """Assemble the ghi/dni/dhi/temp_air/wind_speed frame BREOS consumes."""
    return pd.DataFrame(
        {
            "ghi": d["ghi"].fillna(0.0),
            "dni": derive_dni(d, dni_method, solpos).fillna(0.0),
            "dhi": d["dhi"].fillna(0.0),
            "temp_air": d["temp_air"],
            "wind_speed": d["wind"],
        },
        index=d.index,
    )


def modelled_poa(weather: pd.DataFrame, tilt: float, azimuth: float,
                 transposition_model: str, model_perez: str = "allsitescomposite1990",
                 albedo: float | None = None) -> pd.Series:
    """POA global in W/m2 straight out of BREOS's own transposition stage.

    Uses the same private detail path the production chain uses, so this is the
    quantity BREOS would feed its module model -- not a separate pvlib call that
    happens to agree.
    """
    times, solarpos, waligned = _prepare_solarpos_and_weather(weather, location(), FREQ)
    opts = resolve_pv_model_options(
        transposition_model=transposition_model, model_perez=model_perez, albedo=albedo
    )
    detail = _compute_irradiance_and_cell_temp_detail(
        waligned, solarpos, surface_tilt=tilt, surface_azimuth=azimuth,
        pv_params=ARRAYS["16A"].module, model_options=opts,
    )
    return pd.Series(detail.poa_global, index=times, name="poa_model")


def run_chain(d: pd.DataFrame, label: str, transposition_model: str = "isotropic",
              loss_overrides=None, dni_method: str = "closure",
              temperature_model: str = "faiman", model_perez: str = "allsitescomposite1990",
              albedo: float | None = None, degradation_rate: float = 0.0,
              years_aged: float = 0.0, weather: pd.DataFrame | None = None) -> pd.DataFrame:
    """Run the full BREOS chain for one array and return model AC power in kW."""
    arr = ARRAYS[label]
    w = weather_frame(d, dni_method) if weather is None else weather

    breakdown = calculate_pv_production_breakdown(
        w, location(), tilt=arr.tilt, surface_azimuth=arr.azimuth,
        n_modules=arr.n_modules, pv_params=arr.module, freq=FREQ,
        loss_overrides=loss_overrides, transposition_model=transposition_model,
        model_perez=model_perez, albedo=albedo, temperature_model=temperature_model,
        degradation_rate=degradation_rate, current_year=int(years_aged), start_year=0,
    )
    ac_w, _, _ = _calculate_dc_ac_power_arrays(
        breakdown.dc_after_losses.to_numpy(),
        arr.inverter_ac_kw * 1000.0,
        arr.inverter_eta,
    )
    return pd.DataFrame(
        {
            "dc_model_kW": breakdown.dc_after_losses.to_numpy() / 1000.0,
            "ac_model_kW": ac_w / 1000.0,
        },
        index=breakdown.dc_after_losses.index,
    )


def metrics(model_kw: pd.Series, meas_kw: pd.Series) -> dict:
    """Energy bias plus daily and hourly agreement, on the overlap only."""
    ok = model_kw.notna() & meas_kw.notna()
    m, y = model_kw[ok], meas_kw[ok]
    e_model, e_meas = m.sum() * STEP_H, y.sum() * STEP_H
    dm = m.resample("D").sum() * STEP_H
    dd = y.resample("D").sum() * STEP_H
    good = dd > 0.5
    return {
        "model_kWh": e_model,
        "meas_kWh": e_meas,
        "bias_%": (e_model / e_meas - 1.0) * 100.0 if e_meas else np.nan,
        "daily_nRMSE_%": float(np.sqrt(((dm[good] - dd[good]) ** 2).mean()) / dd[good].mean() * 100),
        "daily_r": float(dm[good].corr(dd[good])),
        "hourly_r": float(m.resample("h").mean().corr(y.resample("h").mean())),
        "n_days": int(good.sum()),
    }


def irradiance_metrics(model: pd.Series, meas: pd.Series, mask=None) -> dict:
    """Bias, RMSE and correlation between a modelled and a measured irradiance."""
    ok = model.notna() & meas.notna()
    if mask is not None:
        ok &= mask
    m, y = model[ok], meas[ok]
    err = m - y
    return {
        "n": int(ok.sum()),
        "model_kWh_m2": m.sum() * STEP_H / 1000.0,
        "meas_kWh_m2": y.sum() * STEP_H / 1000.0,
        "bias_%": (m.sum() / y.sum() - 1.0) * 100.0,
        "bias_Wm2": float(err.mean()),
        "RMSE_Wm2": float(np.sqrt((err ** 2).mean())),
        "r": float(m.corr(y)),
    }


# --- screening -------------------------------------------------------------
#
# Two independent failure modes exist at this site and they pull the bias in
# opposite directions, so they are screened separately rather than with one
# rule.
#
# *Weather-station outage.* The pyranometers drop to identically zero for whole
# days while the arrays keep producing normally -- late June and early July 2016
# are the clearest instances. Any irradiance-driven model necessarily predicts
# zero on those days, so leaving them in drags the modelled total down and
# understates the bias. This is the mirror image of the NIST array, where the
# *array* was down and the model over-predicted.
#
# *Array outage or curtailment.* The array underperforms a valid irradiance
# record, as on 2016-01-30, which is depressed for every array on the site at
# once and so reads as a site-wide event rather than an array fault.

WEATHER_MIN_CLEARNESS = 0.05   # daily GHI / clear-sky GHI below this = instrument down
WEATHER_MIN_DAYTIME_ON = 0.90  # fraction of daytime samples that must be non-zero
ARRAY_OUTAGE_RATIO = 0.85      # daily measured/model below this = not normal operation


def clear_sky_ghi(index: pd.DatetimeIndex) -> pd.Series:
    return location().get_clearsky(index, model="ineichen")["ghi"]


def screen_weather_days(d: pd.DataFrame) -> pd.Series:
    """Per-day boolean: is the weather record itself usable on this day?"""
    cs = clear_sky_ghi(d.index)
    clearness = d["ghi"].resample("D").sum() / cs.resample("D").sum()
    # Fraction of genuinely sunlit samples that carry a non-zero reading. A
    # pyranometer that has dropped out reads exactly zero all day; real
    # overcast never does.
    up = cs > 50.0
    lit = (up & (d["ghi"] > 0.0)).resample("D").sum()
    total = up.resample("D").sum()
    frac_on = (lit / total.replace(0, np.nan)).fillna(0.0)
    return (clearness > WEATHER_MIN_CLEARNESS) & (frac_on > WEATHER_MIN_DAYTIME_ON)


def screen_array_days(d: pd.DataFrame, label: str, model_ac_kw: pd.Series,
                      weather_ok: pd.Series) -> pd.Series:
    """Per-day boolean: was the array in normal operation on this day?

    Computed from a loss-free chain so the rule cannot be tuned by the loss
    assumption under test, and only on days the weather record is usable.
    """
    dm = model_ac_kw.resample("D").sum() * STEP_H
    dd = d[f"P_{label}"].resample("D").sum() * STEP_H
    ratio = (dd / dm).replace([np.inf, -np.inf], np.nan)
    # Normalise by the array's own median ratio before thresholding. An
    # absolute threshold would cut the low tail of a distribution that the
    # model's own bias has already shifted -- at this site the model runs ~10 %
    # high on 16A, so a fixed 0.85 cut removes ordinary days and flatters the
    # result. Dividing by the median makes the screen detect *faults*, days the
    # array underperformed its own normal behaviour, independently of how well
    # the model is centred.
    med = ratio[weather_ok].median()
    return weather_ok & (ratio / med >= ARRAY_OUTAGE_RATIO) & ratio.notna()


def day_mask(index: pd.DatetimeIndex, ok_days: pd.Series) -> pd.Series:
    """Broadcast a per-day boolean back onto the 5-minute index."""
    keep = set(ok_days[ok_days].index.date)
    return pd.Series([t.date() in keep for t in index], index=index)
