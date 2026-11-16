"""Sourced module datasheets shared by the electrical, string-energy, shading and inverter tests.

The catalog does not bundle these modules; tests register them through the
``trina`` and ``jinko`` fixtures in ``tests/conftest.py``.
"""

from breos.pv_modules import PVModuleParams

# Trina Solar TSM-450NEG9R.28, datasheet TSM_EN_2024_C, PDF page 2:
# https://static.trinasolar.com/sites/default/files/Datasheet_Vertex%20S%2B_NEG9R.28_EN_2024_C_web.pdf
TRINA_KEY = "Trina_TSM_450NEG9R28"
TRINA_450 = PVModuleParams(
    Mpp=450,
    Vmp=44.6,
    Imp=10.09,
    Voc=52.9,
    Isc=10.74,
    T_Pmax_pct=-0.29,
    T_Voc_pct=-0.24,
    T_Isc_pct=0.04,
    N_Cells=144,
    Max_System_Voltage=1500,
)

# Jinko Solar JKM440N-54HL4R-B, datasheet EU-JKM430-455N-54HL4R-B-G3-EN, PDF page 2:
# https://jinkosolar.eu/wp-content/uploads/EU-JKM430-455N-54HL4R-B-G3-EN.pdf
# The maximum system voltage was not transcribed, so it stays unknown here.
JINKO_KEY = "Jinko_JKM440N_54HL4R_B"
JINKO_440 = PVModuleParams(
    Mpp=440,
    Vmp=32.99,
    Imp=13.34,
    Voc=39.57,
    Isc=13.8,
    T_Pmax_pct=-0.29,
    T_Voc_pct=-0.25,
    T_Isc_pct=0.045,
    N_Cells=108,
)
