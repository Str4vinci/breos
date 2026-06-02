# Resources

This page collects external documentation and data sources that are useful
when preparing BREOS inputs. Inclusion here is not a redistribution grant or
endorsement. Always review each source's current license, API terms, citation
requirements, and commercial-use limits before using it in a project.

## PV models and component data

| Resource | Use it for | Notes |
|---|---|---|
| [pvlib Python documentation](https://pvlib-python.readthedocs.io/en/stable/) | Solar position, irradiance transposition, PV temperature, and PV performance model references. | BREOS uses pvlib internally for PV calculations. |
| [pvlib `retrieve_sam`](https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.pvsystem.retrieve_sam.html) | Accessing module and inverter parameter tables bundled with pvlib. | Useful when comparing BREOS module entries against SAM/CEC style data. |
| [NREL SAM photovoltaic module help](https://samrepo.nrelcloud.org/help/pv_module.html) | Understanding CEC, Sandia, IEC 61853, PVWatts, and user-entered module models. | Good background when deciding which module datasheet parameters are needed. |
| Manufacturer datasheets | Final project assumptions for module power, temperature coefficients, dimensions, and inverter ratings. | Prefer datasheets for real designs; catalogue entries can lag current products. |

## Load profiles

BREOS bundles only demandlib-derived H0 example profiles. Other RLPs should be
provided locally through `rlp_directory`; see [Load Profile Data](legal/load-profile-data.md).

| Resource | Use it for | Notes |
|---|---|---|
| [demandlib documentation](https://demandlib.readthedocs.io/en/stable/) | Generating synthetic German standard demand profiles. | Source basis for the bundled example H0 profiles. |
| [BDEW Standardlastprofile Strom](https://www.bdew.de/energie/standardlastprofile-strom/) | German standard load profile references. | Public download page does not by itself imply package redistribution rights. |
| [E-REDES Consumption and Loss Profiles](https://www.e-redes.pt/en/consumption-and-loss-profiles) | Portuguese BTN A/B/C and related consumption/injection profile files. | Use as external user data unless redistribution permission is confirmed. |
| [E-REDES Open Data Portal](https://e-redes.opendatasoft.com/pages/homepage/) | Portuguese distribution-grid open datasets. | Check dataset-specific terms and attribution requirements. |
| [Red Electrica consumption profiles](https://www.ree.es/es/clientes/comercializador/gestion-medidas-electricas/consulta-perfiles-de-consumo) | Spanish final consumption profiles used for non-hourly metering settlement. | Use as external user data unless redistribution permission is confirmed. |

## Weather and solar resource data

| Resource | Use it for | Notes |
|---|---|---|
| [PVGIS online tool](https://joint-research-centre.ec.europa.eu/pvgis-online-tool_en) | EU/JRC solar resource data and PVGIS workflows. | BREOS can fetch PVGIS TMY data through pvlib; acknowledge PVGIS/JRC where required. |
| [PVGIS data download](https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis/pvgis-data-download_en) | PVGIS geodata and source/citation details. | Review the source-specific acknowledgement text. |
| [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api) | Historical weather/reanalysis variables such as temperature, wind, and solar radiation. | Review current pricing/terms for commercial or high-volume use. |
| [NREL NSRDB API](https://developer.nrel.gov/docs/solar/nsrdb/) | Satellite-derived solar radiation and meteorological data, especially for U.S. and covered regions. | Requires an API key for downloads. |
| [Meteonorm API documentation](https://docs.meteonorm.com/api) | Licensed TMY, climate, forecast, and observation data workflows. | Commercial product; use with an appropriate Meteonorm license/API key. |

## Project assumptions to document

For reproducible studies, keep these alongside the BREOS config:

- Weather source, year range, API provider, and API key/account used if applicable.
- Load profile source, license/permission, annual consumption, and scaling method.
- PV module and inverter datasheets, module count, layout, tilt, azimuth, and tracking assumptions.
- Battery datasheet or modelling assumptions, including usable SOC window, efficiency, chemistry, and degradation model.
- Cost, tariff, export-price, inflation, discount-rate, and emissions assumptions.
- BREOS version and any local code changes.
