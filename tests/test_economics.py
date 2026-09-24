"""Tests for the economics module."""

import numpy as np
import pandas as pd
import pytest

from breos.economics import (
    DEFAULT_REPLACEMENT_YEAR_FRACTION,
    CostParams,
    calculate_costs,
    calculate_lcoe,
    calculate_lcoe_from_projection,
    cost_analysis_projection,
    cost_params_from_config,
    find_payback_year,
    replacement_booking_time,
    replacement_fraction_by_year,
    replacement_fraction_from_steps,
    system_ac_production_power,
)
from breos.optimization import calculate_financials


class TestCostDefaultsSingleSource:
    def test_cost_params_from_config_empty_matches_dataclass_defaults(self):
        # Missing config keys must fall back to the CostParams defaults, so
        # config-driven and direct-construction paths cannot diverge.
        assert cost_params_from_config({}, {}) == CostParams()

    def test_cost_params_from_config_maps_land_cost(self):
        params = cost_params_from_config({"land_cost": 1250.0}, {})

        assert params.land_cost == 1250.0
        assert params.operation_cost == CostParams().operation_cost

    def test_resolve_costs_preset_fallbacks_match_dataclass_defaults(self, monkeypatch):
        from breos import app_config

        monkeypatch.setattr(app_config, "load_json", lambda name: {"minimal": {}})
        cfg = {
            "cost_preset": "minimal",
            "inverter_loading_ratio": 1.25,
            "inflation_rate": 0.02,
            "sell_price_inflation": 0.01,
            "discount_rate": 0.0,
            "pv_degradation_rate": 0.005,
        }

        params = app_config.resolve_costs(cfg)

        # A preset that omits every key behaves exactly like no preset
        assert params == CostParams(
            dc_ac_ratio=1.25,
            inflation_rate=0.02,
            sell_price_inflation=0.01,
            discount_rate=0.0,
            pv_degradation_rate=0.005,
        )

    def test_app_cost_overrides_layer_over_preset_and_defaults(self, monkeypatch):
        from breos import app_config

        monkeypatch.setattr(
            app_config,
            "load_json",
            lambda name: {
                "partial": {
                    "electricity_cost": 0.31,
                    "storage_cost_per_kwh": 450.0,
                    "maintenance_cost": 12.0,
                }
            },
        )
        cfg = {
            "cost_preset": "partial",
            "costs": {
                "electricity_cost": 0.42,
                "storage_cost_per_kwh": 375.0,
                "operation_cost": 20.0,
            },
            "inverter_loading_ratio": 1.4,
            "inflation_rate": 0.025,
            "sell_price_inflation": 0.01,
            "discount_rate": 0.04,
            "pv_degradation_rate": 0.006,
        }

        params = app_config.resolve_costs(cfg)

        assert params.electricity_cost == 0.42  # explicit override
        assert params.battery_cost_per_kwh == 375.0  # catalog name maps to CostParams
        assert params.maintenance_cost_fixed == 12.0  # preset value remains
        assert params.operation_cost == 20.0  # default-only term can be overridden
        assert params.module_cost_per_w == CostParams().module_cost_per_w
        assert params.dc_ac_ratio == 1.4
        assert params.inflation_rate == 0.025

    def test_flat_preset_resolution_is_unchanged(self):
        from breos import app_config

        cfg = {
            "cost_preset": "residential_pt",
            "inverter_loading_ratio": 1.25,
            "inflation_rate": 0.02,
            "sell_price_inflation": 0.0,
            "discount_rate": 0.03,
            "pv_degradation_rate": 0.005,
        }

        assert app_config.resolve_costs(cfg) == CostParams(
            electricity_cost=0.2582,
            electricity_sold_cost=0.04,
            daily_power_cost=0.3,
            module_cost_per_w=0.125,
            battery_cost_per_kwh=500,
            dc_ac_ratio=1.25,
            inverter_cost_per_kw=102.58,
            inverter_cost_per_kw_nobatt=48.37,
            installation_cost_per_module=350,
            battery_installation_cost=350,
            other_cost_per_module=30,
            other_cost_fixed=0,
            maintenance_cost_per_panel=10,
            maintenance_cost_fixed=0,
            inflation_rate=0.02,
            sell_price_inflation=0.0,
            discount_rate=0.03,
            pv_degradation_rate=0.005,
        )


class TestCalculateCosts:
    def test_pv_only(self, cost_params):
        costs = calculate_costs(
            n_modules=10,
            module_power_w=550,
            battery_capacity_wh=0,
            cost_params=cost_params,
        )
        assert costs["battery_cost"] == 0.0
        assert costs["pv_cost"] == pytest.approx(10 * 550 * 0.125)
        assert costs["total_initial_cost"] > 0
        # No battery → simple inverter
        assert costs["inverter_cost"] == pytest.approx(48.37 * (10 * 550 / 1000) / 1.25, rel=0.01)

    def test_with_battery(self, cost_params):
        costs = calculate_costs(
            n_modules=10,
            module_power_w=550,
            battery_capacity_wh=5000,
            cost_params=cost_params,
        )
        assert costs["battery_cost"] == pytest.approx(5 * 500.0)
        # With battery → hybrid inverter (more expensive)
        assert costs["inverter_cost"] == pytest.approx(102.58 * (10 * 550 / 1000) / 1.25, rel=0.01)
        assert costs["total_initial_cost"] > costs["pv_cost"] + costs["battery_cost"]

    def test_cost_breakdown_sums_to_total(self, cost_params):
        costs = calculate_costs(
            n_modules=6,
            module_power_w=550,
            battery_capacity_wh=5000,
            cost_params=cost_params,
        )
        parts = (
            costs["pv_cost"]
            + costs["inverter_cost"]
            + costs["battery_cost"]
            + costs["installation_cost"]
            + costs["other_costs"]
        )
        assert costs["total_initial_cost"] == pytest.approx(parts, rel=0.001)

    def test_optimizer_financials_honor_modern_cost_keys(self):
        base = dict(
            n_modules=10,
            battery_kwh=5.0,
            annual_import_kwh=2000.0,
            annual_export_kwh=1000.0,
            annual_load_kwh=4000.0,
            costs_config={
                "module_cost_per_w": 0.10,
                "storage_cost_per_kwh": 400.0,
                "installation_cost_per_module": 200.0,
                "installation_cost_battery": 500.0,
                "other_costs": 100.0,
            },
            financials_config={
                "electricity_cost": 0.30,
                "electricity_sold_cost": 0.05,
                "inflation_rate": 0.01,
                "discount_rate": 0.0,
                "project_lifespan": 5,
            },
        )

        capex_a, npv_a = calculate_financials(**base)

        modified = dict(base)
        modified["costs_config"] = dict(base["costs_config"], module_cost_per_w=0.30)
        modified["financials_config"] = dict(base["financials_config"], electricity_cost=0.45)
        capex_b, npv_b = calculate_financials(**modified)

        assert capex_b > capex_a
        assert npv_b != npv_a

    def test_optimizer_financials_use_selected_module_mpp(self):
        base = {
            "module_cost_per_w": 0.20,
            "inverter_cost_per_kw_simple": 0.0,
            "installation_cost_per_module": 0.0,
            "other_cost_per_module": 0.0,
        }
        financials = {"project_lifespan": 1}

        capex_400, _ = calculate_financials(10, 0.0, 0.0, 0.0, 0.0, base, financials, module_power_w=400.0)
        capex_550, _ = calculate_financials(10, 0.0, 0.0, 0.0, 0.0, base, financials, module_power_w=550.0)
        assert capex_550 - capex_400 == pytest.approx(10 * 150 * 0.20)

        # The removed costs.panel_wp priced CAPEX at a wattage other than the
        # module's, which let the budget pass a design over budget (#157).
        with pytest.raises(ValueError, match="costs.panel_wp was removed"):
            calculate_financials(10, 0.0, 0.0, 0.0, 0.0, dict(base, panel_wp=500.0), financials, module_power_w=400.0)


def test_system_ac_production_prefers_explicit_ledger_over_legacy_field():
    results = pd.DataFrame(
        {
            "PV_AC_To_Load": [300.0, 100.0],
            "Battery_AC_To_Load_PV": [50.0, 25.0],
            "PV_AC_Export": [200.0, 75.0],
            "Sell_To_Grid": [999.0, 999.0],
            "PV_Production": [9999.0, 9999.0],
        }
    )

    assert system_ac_production_power(results).tolist() == pytest.approx([550.0, 200.0])


def test_system_ac_production_accepts_legacy_field():
    results = pd.DataFrame({"PV_Production": [500.0, 250.0]})

    assert system_ac_production_power(results).tolist() == pytest.approx([500.0, 250.0])


class TestFindPaybackYear:
    def test_known_payback(self):
        import pandas as pd

        # Savings turn positive at year 8
        proj = pd.DataFrame(
            {
                "Year": range(1, 11),
                "Savings_Cumulative_NPV": [-500, -400, -300, -200, -100, -50, -10, 30, 100, 200],
            }
        )
        assert find_payback_year(proj) == 8

    def test_no_payback(self):
        import pandas as pd

        proj = pd.DataFrame(
            {
                "Year": range(1, 6),
                "Savings_Cumulative_NPV": [-500, -400, -300, -200, -100],
            }
        )
        assert find_payback_year(proj) is None


class TestLCOE:
    def test_basic(self):
        lcoe = calculate_lcoe(
            total_investment=5000,
            annual_production_kwh=5000,
            annual_operation_cost=50,
            lifetime_years=20,
            discount_rate=0.0,
            degradation_rate=0.0,
        )
        # (5000 + 50*20) / (5000*20) = 6000/100000 = 0.06
        assert lcoe == pytest.approx(0.06, rel=0.01)

    def test_degradation_increases_lcoe(self):
        lcoe_no_deg = calculate_lcoe(
            total_investment=5000,
            annual_production_kwh=5000,
            annual_operation_cost=50,
            lifetime_years=20,
            discount_rate=0.0,
            degradation_rate=0.0,
        )
        lcoe_with_deg = calculate_lcoe(
            total_investment=5000,
            annual_production_kwh=5000,
            annual_operation_cost=50,
            lifetime_years=20,
            discount_rate=0.0,
            degradation_rate=0.01,
        )
        assert lcoe_with_deg > lcoe_no_deg

    def test_zero_production(self):
        lcoe = calculate_lcoe(
            total_investment=5000,
            annual_production_kwh=0,
            annual_operation_cost=50,
            lifetime_years=20,
        )
        assert lcoe == float("inf")

    def test_projection_lcoe_includes_replacement_costs(self):
        projection = pd.DataFrame(
            {
                "Year": [1, 2],
                "PV_Production_kWh": [1000.0, 1000.0],
                "Cost_Operation": [100.0, 100.0],
                "Cost_Replacement": [0.0, 500.0],
            }
        )

        lcoe = calculate_lcoe_from_projection(projection, total_investment=1000.0, discount_rate=0.0)

        assert lcoe == pytest.approx(0.85)

    def test_cost_projection_exposes_lcoe_attr(self):
        costs = {
            "electricity_cost": 0.30,
            "electricity_sold_cost": 0.05,
            "daily_power_cost": 0.20,
            "total_initial_cost": 1000.0,
            "annual_operation_cost": 100.0,
        }
        yearly_summary = pd.DataFrame(
            {
                "Year": [1, 2],
                "Load_kWh": [1200.0, 1200.0],
                "PV_Production_kWh": [1000.0, 900.0],
                "Import_kWh": [400.0, 450.0],
                "Export_kWh": [200.0, 180.0],
                "PV_Degradation_Factor": [1.0, 0.9],
                "Replacement_Cost": [0.0, 500.0],
            }
        )

        projection = cost_analysis_projection(
            pd.DataFrame(),
            costs,
            num_years=2,
            inflation_rate=0.0,
            discount_rate=0.0,
            yearly_summary_df=yearly_summary,
        )

        assert projection.attrs["lcoe_eur_kwh"] == pytest.approx((1000 + 100 + 100 + 500) / (1000 + 900))

    def test_cost_projection_uses_yearly_load_for_no_system_baseline(self):
        costs = {
            "electricity_cost": 0.30,
            "electricity_sold_cost": 0.05,
            "daily_power_cost": 0.20,
            "total_initial_cost": 1000.0,
            "annual_operation_cost": 100.0,
        }
        yearly_summary = pd.DataFrame(
            {
                "Year": [1, 2],
                "Load_kWh": [1000.0, 2000.0],
                "PV_Production_kWh": [800.0, 800.0],
                "Import_kWh": [300.0, 600.0],
                "Export_kWh": [100.0, 100.0],
                "PV_Degradation_Factor": [1.0, 1.0],
                "Replacement_Cost": [0.0, 0.0],
            }
        )

        projection = cost_analysis_projection(
            pd.DataFrame(),
            costs,
            num_years=2,
            inflation_rate=0.0,
            discount_rate=0.0,
            yearly_summary_df=yearly_summary,
        )

        daily = 365 * costs["daily_power_cost"]
        assert projection["Cost_No_Sys_Annual"].tolist() == pytest.approx(
            [
                1000.0 * costs["electricity_cost"] + daily,
                2000.0 * costs["electricity_cost"] + daily,
            ]
        )


class TestReplacementBookingTime:
    """A replacement is a dated transaction, not a flow spread over its year.

    Booking it at year granularity is what made the two rates disagree: the
    old code inflated by ``(1 + i) ** (year - 1)``, valuing the outlay at the
    start of the replacement year, and discounted by ``(1 + d) ** year``,
    valuing it at the end. Neither is the day the pack was swapped.
    """

    def test_step_fraction_locates_the_swap_within_its_year(self):
        # Hourly year, swap on day 109: step 2616 of 8760.
        assert replacement_fraction_from_steps([2616], 8760) == pytest.approx(0.29863, abs=1e-5)
        # The same instant on a 15-minute timebase is the same fraction.
        assert replacement_fraction_from_steps([2616 * 4], 8760 * 4) == pytest.approx(0.29863, abs=1e-5)

    def test_year_without_a_swap_has_no_fraction(self):
        assert np.isnan(replacement_fraction_from_steps([], 8760))

    def test_booking_time_is_measured_from_commissioning(self):
        # The instant the deferred-fix note pins: a swap 109 days into
        # projection year 11 is t = 10.299, not 10 and not 11.
        booked = replacement_booking_time([11], [replacement_fraction_from_steps([2616], 8760)], [True])

        assert booked[0] == pytest.approx(10.299, abs=1e-3)

    def test_years_without_a_replacement_are_not_booked(self):
        booked = replacement_booking_time([1, 2, 3], [np.nan, np.nan, 0.25], [False, False, True])

        assert np.isnan(booked[:2]).all()
        assert booked[2] == pytest.approx(2.25)

    def test_missing_fraction_falls_back_to_the_documented_default(self):
        booked = replacement_booking_time([3], None, [True])

        assert booked[0] == pytest.approx(2.0 + DEFAULT_REPLACEMENT_YEAR_FRACTION)

    def test_fraction_by_year_reads_the_ledger_column(self):
        years = [2025] * 4 + [2026] * 4
        replaced = [False, False, True, False, False, False, False, False]

        fractions = replacement_fraction_by_year(years, replaced)

        assert fractions.index.tolist() == [2025]
        assert fractions.loc[2025] == pytest.approx(0.5)


class TestReplacementBookingInProjection:
    INFLATION = 0.03
    DISCOUNT = 0.02
    COST = 5000.0

    def _projection(self, fraction):
        years = list(range(1, 13))
        yearly = pd.DataFrame(
            {
                "Year": years,
                "Load_kWh": [4000.0] * 12,
                "PV_Production_kWh": [5000.0] * 12,
                "Import_kWh": [1500.0] * 12,
                "Export_kWh": [2000.0] * 12,
                "PV_Degradation_Factor": [1.0] * 12,
                "Replacement_Cost": [self.COST if y == 11 else 0.0 for y in years],
                "Replacement_Year_Fraction": [fraction if y == 11 else np.nan for y in years],
            }
        )
        return cost_analysis_projection(
            results_df=None,
            costs={
                "electricity_cost": 0.22,
                "electricity_sold_cost": 0.05,
                "daily_power_cost": 0.30,
                "annual_operation_cost": 100.0,
                "total_initial_cost": 12000.0,
            },
            num_years=12,
            inflation_rate=self.INFLATION,
            discount_rate=self.DISCOUNT,
            yearly_summary_df=yearly,
        )

    def test_both_rates_are_applied_at_the_same_instant(self):
        fraction = replacement_fraction_from_steps([2616], 8760)
        projection = self._projection(fraction)
        row = projection[projection["Year"] == 11].iloc[0]
        booked_at = row["Replacement_Time_Years"]

        assert booked_at == pytest.approx(10.299, abs=1e-3)
        assert row["Cost_Replacement"] == pytest.approx(self.COST * (1 + self.INFLATION) ** booked_at)

        # The annual NPV carries that same outlay discounted from the same
        # instant; every other component keeps the year-end convention.
        others = row["Cost_System_Annual"] - row["Cost_Replacement"]
        discounted_outlay = row["Cost_System_Annual_NPV"] - others / (1 + self.DISCOUNT) ** 11
        assert discounted_outlay == pytest.approx(row["Cost_Replacement"] / (1 + self.DISCOUNT) ** booked_at)

    def test_non_replacement_years_keep_the_year_end_convention(self):
        projection = self._projection(0.25)

        assert projection["Replacement_Time_Years"].notna().sum() == 1
        for year in (1, 5, 12):
            row = projection[projection["Year"] == year].iloc[0]
            assert row["Cost_System_Annual_NPV"] == pytest.approx(
                row["Cost_System_Annual"] / (1 + self.DISCOUNT) ** year
            )

    def _present_value(self, projection):
        row = projection[projection["Year"] == 11].iloc[0]
        return row["Cost_Replacement"] / (1 + self.DISCOUNT) ** row["Replacement_Time_Years"]

    def test_booking_at_the_instant_reduces_to_the_two_rates_net_of_each_other(self):
        # Applying both rates at the same t collapses to C * ((1+i)/(1+d))**t,
        # so which way the instant moves the outlay is the sign of i - d and
        # not a property of the calendar.
        for fraction in (0.1, 0.5, 0.9):
            booked_at = 10.0 + fraction
            expected = self.COST * ((1 + self.INFLATION) / (1 + self.DISCOUNT)) ** booked_at
            assert self._present_value(self._projection(fraction)) == pytest.approx(expected)

    def test_the_correction_always_raises_the_outlay_against_the_old_booking(self):
        # The old booking inflated to the start of the replacement year and
        # discounted from its end, so it undervalued the outlay wherever in
        # the year the swap fell. Correcting it cannot make the pack cheaper.
        old_booking = self.COST * (1 + self.INFLATION) ** 10 / (1 + self.DISCOUNT) ** 11

        for fraction in (0.0, 0.1, 0.5, 0.9):
            assert self._present_value(self._projection(fraction)) > old_booking

    def test_lcoe_discounts_the_replacement_from_the_same_instant(self):
        projection = self._projection(replacement_fraction_from_steps([2616], 8760))
        booked_at = projection["Replacement_Time_Years"].dropna().iloc[0]
        outlay = projection["Cost_Replacement"].sum()

        lcoe = calculate_lcoe_from_projection(projection, total_investment=12000.0, discount_rate=self.DISCOUNT)

        years = projection["Year"].to_numpy(dtype=float)
        discount_factors = 1 / (1 + self.DISCOUNT) ** years
        production = float((projection["PV_Production_kWh"].to_numpy() * discount_factors).sum())
        operation = float((projection["Cost_Operation"].to_numpy() * discount_factors).sum())
        expected = (12000.0 + operation + outlay / (1 + self.DISCOUNT) ** booked_at) / production

        assert lcoe == pytest.approx(expected)


class TestFirstYearProjectionCalendar:
    """The first-year path groups the ledger on its own local calendar."""

    COSTS = {
        "electricity_cost": 0.20,
        "electricity_sold_cost": 0.05,
        "daily_power_cost": 0.30,
        "annual_operation_cost": 0.0,
        "total_initial_cost": 5000.0,
    }

    @pytest.mark.parametrize("tz", ["Europe/Berlin", "Australia/Sydney"])
    def test_year_one_is_the_whole_local_year_east_of_utc(self, tz):
        # East of UTC the first local hours fall in the previous UTC year.
        # Grouping in UTC built the projection from that stub alone.
        index = pd.date_range("2023-01-01", periods=8760, freq="h", tz=tz)
        results = pd.DataFrame(
            {
                "Datetime": index,
                "PV_AC_To_Load": 500.0,
                "Battery_AC_To_Load_PV": 0.0,
                "PV_AC_Export": 300.0,
                "Houseload": 1000.0,
                "Import_From_Grid": 500.0,
                "Sell_To_Grid": 300.0,
            }
        )

        projection = cost_analysis_projection(
            results, self.COSTS, num_years=2, inflation_rate=0.0, discount_rate=0.0, degradation_rate=0.0
        )

        year_one = projection.iloc[0]
        assert year_one["Cost_No_Sys_Annual"] == pytest.approx(8760.0 * 0.20 + 365 * 0.30)
        assert year_one["Cost_Import"] == pytest.approx(8760.0 * 0.5 * 0.20)
        assert year_one["Revenue_Export"] == pytest.approx(8760.0 * 0.3 * 0.05)
        assert year_one["Cost_Daily"] == pytest.approx(365 * 0.30)
