"""Tests for the PV module catalog."""

import pytest

from breos.pv_modules import MODULES, get_module, get_module_info, list_modules


class TestCatalog:
    def test_get_module_is_case_insensitive_copy(self):
        module = get_module("suntech_stp550s_stc")
        module.Mpp = 1
        assert MODULES["Suntech_STP550S_STC"].Mpp == 550

    def test_unknown_module_error_lists_available(self):
        with pytest.raises(KeyError, match="not found. Available:"):
            get_module("No_Such_Module")

    def test_bifacial_module_exposes_sourced_inert_metadata(self):
        module = get_module("Generic_600W_Bifacial")

        assert module.bifaciality == pytest.approx(0.70)
        assert "Bifaciality: 70.0 %" in get_module_info("Generic_600W_Bifacial")

    def test_sourced_efficiencies_match_their_datasheet_bins(self):
        # Both values come from the module's own datasheet STC table; they are
        # not derived from an assumed frame area.
        assert get_module("Suntech_STP550S_STC").Module_Efficiency == pytest.approx(0.213)
        assert get_module("Generic_600W_Bifacial").Module_Efficiency == pytest.approx(0.212)
        assert "Efficiency: 21.2 %" in get_module_info("Generic_600W_Bifacial")

    def test_catalog_efficiencies_are_physical_when_present(self):
        for key, module in MODULES.items():
            if module.Module_Efficiency is not None:
                assert 0 < module.Module_Efficiency <= 1, key

    def test_unsourced_entries_leave_efficiency_unset_for_the_breos_default(self):
        # These two entries name no datasheet that quotes an efficiency, so they
        # stay None and pick up DEFAULT_MODULE_EFFICIENCY in the PVsyst path
        # rather than carrying a back-derived number.
        assert get_module("Erlangen_445W").Module_Efficiency is None
        assert get_module("Generic_400W").Module_Efficiency is None
        assert "Efficiency: n/a" in get_module_info("Generic_400W")

    def test_catalog_does_not_claim_unsourced_noct_metadata(self):
        assert all(module.NOCT is None for module in MODULES.values())
        assert "NOCT:       n/a (not sourced in bundled catalog)" in get_module_info("Suntech_STP550S_STC")

    def test_nomt_entry_removed(self):
        # The Suntech_STP550S_NOMT entry fed NMOT datasheet points (800 W/m2,
        # Mpp=415) into the STC-based CEC fit, which interprets Vmp/Imp/Voc/Isc
        # as STC values — physically wrong, so the entry was removed. Lookups
        # must fail with the actionable catalog error, not fit silently.
        assert "Suntech_STP550S_NOMT" not in list_modules()
        with pytest.raises(KeyError, match="not found. Available:"):
            get_module("Suntech_STP550S_NOMT")
