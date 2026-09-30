"""Tests for the economics module."""

import numpy as np
import pandas as pd
import pytest

from breos.economics import (
    DEFAULT_REPLACEMENT_YEAR_FRACTION,
    CostParams,
    calculate_costs,
    calculate_lcoe_from_projection,
    cost_analysis_projection,
    cost_params_from_config,
    find_payback_year,
    replacement_booking_time,
    replacement_fraction_from_steps,
)


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

    def test_capex_uses_selected_module_mpp(self):
        base = {
            "module_cost_per_w": 0.20,
            "inverter_cost_per_kw_simple": 0.0,
            "installation_cost_per_module": 0.0,
            "other_cost_per_module": 0.0,
        }
        cost_params = cost_params_from_config(base, {"project_lifespan": 1})

        capex_400 = calculate_costs(10, 400.0, 0.0, cost_params)["total_initial_cost"]
        capex_550 = calculate_costs(10, 550.0, 0.0, cost_params)["total_initial_cost"]
        assert capex_550 - capex_400 == pytest.approx(10 * 150 * 0.20)

        # The removed costs.panel_wp priced CAPEX at a wattage other than the
        # module's, which let the budget pass a design over budget (#157).
        with pytest.raises(ValueError, match="costs.panel_wp was removed"):
            cost_params_from_config(dict(base, panel_wp=500.0), {"project_lifespan": 1})


def test_cost_projection_needs_year_rows():
    with pytest.raises(ValueError, match="needs yearly_summary_df"):
        cost_analysis_projection(None, {"total_initial_cost": 1000.0}, num_years=2)
    with pytest.raises(ValueError, match="needs yearly_summary_df"):
        cost_analysis_projection(pd.DataFrame(), {"total_initial_cost": 1000.0}, num_years=2)
    steps = pd.DataFrame({"PV_AC_Export": [1.0]}, index=pd.date_range("2025-01-01", periods=1, freq="h"))
    with pytest.raises(ValueError, match="not a per-step results frame"):
        cost_analysis_projection(steps, {"total_initial_cost": 1000.0}, num_years=1)


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

    def test_projection_lcoe_matches_the_closed_form_without_inflation(self):
        # An independent 20-year oracle with PV degradation and a non-zero
        # discount rate. With no inflation, O&M escalation or replacement,
        # the projection's LCOE is the closed form
        #   (I + sum_t OM / (1 + d)^t) / sum_t P0 (1 - g)^(t - 1) / (1 + d)^t
        # for t = 1..N.
        years, degradation, discount = 20, 0.005, 0.05
        investment, operation, first_year_kwh = 5000.0, 75.0, 4000.0
        costs = {
            "electricity_cost": 0.30,
            "electricity_sold_cost": 0.05,
            "daily_power_cost": 0.20,
            "total_initial_cost": investment,
            "annual_operation_cost": operation,
        }
        production = first_year_kwh * (1.0 - degradation) ** np.arange(years)
        yearly_summary = pd.DataFrame(
            {
                "Year": range(1, years + 1),
                "Load_kWh": 5000.0,
                "PV_Production_kWh": production,
                "Import_kWh": 2500.0,
                "Export_kWh": 1500.0,
                "PV_Degradation_Factor": production / production[0],
                "Replacement_Cost": 0.0,
            }
        )
        projection = cost_analysis_projection(
            yearly_summary, costs, num_years=years, inflation_rate=0.0, discount_rate=discount
        )

        t = np.arange(1, years + 1)
        discounting = (1.0 + discount) ** t
        npv_costs = investment + np.sum(operation / discounting)
        npv_production = np.sum(first_year_kwh * (1.0 - degradation) ** (t - 1) / discounting)
        closed_form = npv_costs / npv_production

        assert projection.attrs["lcoe_per_kwh"] == pytest.approx(closed_form, rel=1e-12)
        assert calculate_lcoe_from_projection(projection, discount_rate=discount) == pytest.approx(
            closed_form, rel=1e-12
        )

    def test_projection_lcoe_reads_the_recorded_investment_and_does_not_infer_it(self):
        projection = pd.DataFrame(
            {
                "Year": [1, 2],
                "PV_Production_kWh": [1000.0, 1000.0],
                "Cost_Operation": [100.0, 100.0],
                "Cost_System_Annual": [100.0, 100.0],
                "Cost_System_Cumulative": [1100.0, 1200.0],
            }
        )
        with pytest.raises(ValueError, match="total_investment is required"):
            calculate_lcoe_from_projection(projection, discount_rate=0.0)

        projection.attrs["total_investment"] = 1000.0
        assert calculate_lcoe_from_projection(projection, discount_rate=0.0) == pytest.approx(0.6)

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
            yearly_summary,
            costs,
            num_years=2,
            inflation_rate=0.0,
            discount_rate=0.0,
        )

        assert projection.attrs["lcoe_per_kwh"] == pytest.approx((1000 + 100 + 100 + 500) / (1000 + 900))

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
                "Year": [2, 1],
                "Load_kWh": [2000.0, 1000.0],
                "PV_Production_kWh": [800.0, 900.0],
                "Import_kWh": [600.0, 300.0],
                "Export_kWh": [100.0, 200.0],
                "PV_Degradation_Factor": [0.9, 1.0],
                "Replacement_Cost": [500.0, 0.0],
            }
        )

        projection = cost_analysis_projection(
            yearly_summary,
            costs,
            num_years=2,
            inflation_rate=0.0,
            discount_rate=0.0,
        )

        daily = 365 * costs["daily_power_cost"]
        assert projection["Load_kWh"].tolist() == [1000.0, 2000.0]
        assert projection["Cost_No_Sys_Annual"].tolist() == pytest.approx(
            [
                1000.0 * costs["electricity_cost"] + daily,
                2000.0 * costs["electricity_cost"] + daily,
            ]
        )
        assert projection["PV_Production_kWh"].tolist() == [900.0, 800.0]
        assert projection["Export_kWh"].tolist() == [200.0, 100.0]
        assert projection["Cost_Import"].tolist() == [90.0, 180.0]
        assert projection["Cost_Replacement"].tolist() == [0.0, 500.0]

    @pytest.mark.parametrize("years", [[1, 1], [1, 3]])
    def test_cost_projection_rejects_duplicate_or_incomplete_year_labels(self, years):
        with pytest.raises(ValueError, match="yearly_summary_df Year values"):
            cost_analysis_projection(
                pd.DataFrame({"Year": years}),
                costs={},
                num_years=2,
            )


class TestReplacementBookingTime:
    """A replacement is a dated transaction, not a flow spread over its year.

    Booking it at year granularity is what made the two rates disagree: the
    old code inflated by ``(1 + i) ** (year - 1)``, valuing the outlay at the
    start of the replacement year, and discounted by ``(1 + d) ** year``,
    valuing it at the end. Neither is the day the pack was swapped.
    """

    def test_step_fraction_locates_the_swap_within_its_year(self):
        # A replacement flagged on an interval happens at that interval's end.
        assert replacement_fraction_from_steps([0], 4) == pytest.approx(0.25)
        # Hourly year, the interval at index 2616 ends at step 2617 of 8760.
        assert replacement_fraction_from_steps([2616], 8760) == pytest.approx(2617 / 8760)
        # The same instant on a 15-minute timebase is the same fraction.
        assert replacement_fraction_from_steps([2616 * 4 + 3], 8760 * 4) == pytest.approx(2617 / 8760)

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
            yearly,
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
