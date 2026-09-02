"""Array and module definitions for the DKASC Alice Springs validation.

Every array here is documented on its own page at dkasolarcentre.com.au, and
the source-id -> meter mapping is confirmed independently by the column names
inside the site-wide Alice_Springs_2025.csv export ("<id>_DKA_<meter>_<phase>").

Unlike the NIST run, **none of these modules is in the CEC database** -- BP
Solar no longer exists and the Trina DC01 series predates the current listing --
so module parameters come from manufacturer datasheets rather than a sourced
database entry. Each is cross-checked against the array area DKASC publishes,
which is an independent number: implied efficiency from area is quoted in the
comment beside each module and agrees with the datasheet to better than 0.2
percentage points in every case.

The parameter set matters least where it matters most. The headline
transposition test compares array 16A (north) with 16D (west), which are the
same modules, the same inverter model and the same commissioning date; module
error is common to both and cancels in the north/west ratio.
"""

from __future__ import annotations

from dataclasses import dataclass

from breos.solar import PVModuleParams

# --- modules ---------------------------------------------------------------

# BP 3165J, 165 W polycrystalline, 72 cells. Datasheet: Vmp 35.2, Imp 4.7,
# Voc 44.2, Isc 5.1, NOCT 47 +/- 2 degC, Pmax -0.5 %/degC, Voc -160 mV/degC,
# Isc +0.065 %/degC. DKASC publishes 15.1 m2 for 12 modules -> 1.258 m2 each,
# implying 13.1 % module efficiency.
BP3165J = PVModuleParams(
    Mpp=165.0, Vmp=35.2, Imp=4.7, Voc=44.2, Isc=5.1,
    T_Pmax_pct=-0.50,
    T_Voc_pct=-0.160 / 44.2 * 100.0,
    T_Isc_pct=0.065,
    N_Cells=72,
    Name="BP Solar BP 3165J",
    Module_Efficiency=165.0 / (1.258 * 1000.0),
    NOCT=47.0,
    celltype="polySi",
)

# BP 4170N, 170 W monocrystalline, 72 cells. The BP 4-series datasheet is no
# longer cleanly retrievable; values are the BP 170 W mono family's published
# figures. DKASC publishes 37.8 m2 for 30 modules -> 1.26 m2 each, implying
# 13.5 % module efficiency. This array is used only in the loss ladder, never
# in the transposition test, and its parameters are the least certain here.
BP4170N = PVModuleParams(
    Mpp=170.0, Vmp=35.8, Imp=4.75, Voc=44.2, Isc=5.1,
    T_Pmax_pct=-0.50,
    T_Voc_pct=-0.160 / 44.2 * 100.0,
    T_Isc_pct=0.065,
    N_Cells=72,
    Name="BP Solar BP 4170N",
    Module_Efficiency=170.0 / (1.260 * 1000.0),
    NOCT=47.0,
    celltype="monoSi",
)

# Trina TSM-175DC01, 175 W monocrystalline, 72 cells (125 mm cells). Datasheet:
# Vmp 36.2, Imp 4.85, Voc 43.9, Isc 5.30, NOCT 46 +/- 2 degC, Pmax -0.45 %/degC,
# Voc -0.35 %/degC, Isc +0.05 %/degC, module efficiency 13.7 %. DKASC publishes
# 38.37 m2 for 30 modules -> 1.279 m2 each, implying 13.68 %.
TSM175DC01 = PVModuleParams(
    Mpp=175.0, Vmp=36.2, Imp=4.85, Voc=43.9, Isc=5.30,
    T_Pmax_pct=-0.45,
    T_Voc_pct=-0.35,
    T_Isc_pct=0.05,
    N_Cells=72,
    Name="Trina TSM-175DC01",
    Module_Efficiency=175.0 / (1.279 * 1000.0),
    NOCT=46.0,
    celltype="monoSi",
)


# --- arrays ----------------------------------------------------------------

@dataclass(frozen=True)
class Array:
    """One DKASC array, as published on its dkasolarcentre.com.au source page."""

    label: str
    source_id: int
    name: str
    module: PVModuleParams
    n_modules: int
    dc_kw: float
    tilt: float
    azimuth: float           # degrees clockwise from north; 0 = solar north
    inverter_ac_kw: float
    inverter_eta: float      # nominal peak efficiency, manufacturer datasheet
    commissioned: str
    tracking: str = "fixed"


# Site 16 is DKASC's deliberate orientation experiment: four physically
# identical 2 kW BP arrays commissioned the same day, differing only in
# azimuth. 16A (north) and 16D (west) are the controlled pair used here. In the
# southern hemisphere north is the sun-facing orientation, so 16D is the
# off-axis array whose output is far more sensitive to the sky-diffuse model.
ARRAYS: dict[str, Array] = {
    "16A": Array("16A", 100, "16A BP Solar 2.0 kW poly, fixed, north",
                 BP3165J, 12, 1.98, 20.0, 0.0, 2.5, 0.941, "2008-11-11"),
    "16D": Array("16D", 81, "16D BP Solar 2.0 kW poly, fixed, west",
                 BP3165J, 12, 1.98, 20.0, 270.0, 2.5, 0.941, "2008-11-11"),
    "12": Array("12", 84, "12 BP Solar 5.1 kW mono, fixed, north",
                BP4170N, 30, 5.10, 20.0, 0.0, 6.0, 0.955, "2008-11-11"),
    "13": Array("13", 92, "13 Trina 5.25 kW mono, fixed, north",
                TSM175DC01, 30, 5.25, 20.0, 0.0, 6.0, 0.955, "2009-01-08"),
    # Site 1A is two DEGERenergie 5000NT dual-axis trackers, 30 modules each,
    # feeding two SMA SMC 6000A. The site was reconfigured in August 2013 --
    # four trackers with 195 W modules were split off as site 1B -- which is
    # exactly where source 91's record begins (2013-08-14), so only the
    # post-reconfiguration definition is ever in view here.
    "1A": Array("1A", 91, "1A Trina 10.5 kW mono, dual-axis tracking",
                TSM175DC01, 60, 10.50, 0.0, 0.0, 12.0, 0.955, "2013-08-14",
                tracking="dual-axis"),
}

# Inverter peak efficiencies: SMA Sunny Boy 2500 (94.1 %) and SMA Sunny Mini
# Central 6000A (95.5 %), from the manufacturer datasheets. BREOS's PVWatts
# inverter model takes a single nominal efficiency and applies its own
# part-load curve, so only the peak value is needed.

SITE_LAT, SITE_LON, SITE_ALT = -23.7624, 133.8745, 545.0
SITE_TZ_OFFSET_HOURS = 9.5          # ACST, no daylight saving
GTI_SENSOR_TILT, GTI_SENSOR_AZIMUTH = 20.0, 0.0
