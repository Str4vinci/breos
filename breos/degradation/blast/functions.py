"""Trimmed BLAST-Lite utility helpers required by the degradation base class."""

import numpy as np


def rescale_soc(soc, rescaling_factor):
    """Rescale an SOC vector by scaling first differences."""
    dSOC = np.diff(soc, prepend=soc[0])
    dSOC = dSOC * rescaling_factor
    soc = np.cumsum(dSOC) + soc[0]
    if np.max(soc) > 1 or np.min(soc) < 0:
        soc = np.maximum(0, np.minimum(1, soc))
    return soc
