"""Tests for the PV module catalog."""

import pytest

from breos.pv_modules import MODULES, get_module, list_modules


class TestCatalog:
    def test_get_module_is_case_insensitive_copy(self):
        module = get_module("suntech_stp550s_stc")
        module.Mpp = 1
        assert MODULES["Suntech_STP550S_STC"].Mpp == 550

    def test_unknown_module_error_lists_available(self):
        with pytest.raises(KeyError, match="not found. Available:"):
            get_module("No_Such_Module")

    def test_nomt_entry_removed(self):
        # The Suntech_STP550S_NOMT entry fed NMOT datasheet points (800 W/m2,
        # Mpp=415) into the STC-based CEC fit, which interprets Vmp/Imp/Voc/Isc
        # as STC values — physically wrong, so the entry was removed. Lookups
        # must fail with the actionable catalog error, not fit silently.
        assert "Suntech_STP550S_NOMT" not in list_modules()
        with pytest.raises(KeyError, match="not found. Available:"):
            get_module("Suntech_STP550S_NOMT")
