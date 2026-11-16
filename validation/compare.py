#!/usr/bin/env python3
"""Compare BREOS results against the external references.

Reads ``validation/results/breos_results.json`` (falling back to the
committed baseline) and the reference JSONs, prints a comparison table, and
writes ``validation/REPORT.md``.

The references do NOT share weather with BREOS: BREOS runs on a PVGIS TMY,
PVGIS PVcalc averages the full multi-year satellite record, and PVWatts uses
NSRDB/international station data. Deviations therefore bundle weather-source
differences with model differences. The PVWatts CI band compares performance
ratios (AC / front-side irradiation / DC nameplate) within ±8%; annual AC
deviations remain visible for context.
BREOS AC is normalized by an independent pvlib Perez transposition of its
weather, shared across all variants, to take irradiance out of the comparison.

Usage:
    uv run python validation/compare.py
"""

import datetime
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from validation.common import (  # noqa: E402
    BASELINE_PATH,
    MONTH_LABELS,
    REPORT_PATH,
    RESULTS_PATH,
    load_reference,
    load_spec,
)


def _pct(value: float, reference: float):
    if not reference:
        return None
    return (value / reference - 1.0) * 100.0


def _fmt_pct(p):
    return f"{p:+.1f}%" if p is not None else "—"


def _fmt_kwh(v):
    return f"{v:.0f}" if v is not None else "—"


def _fmt_pr(v):
    return f"{v:.3f}" if v is not None else "—"


def build_rows(results: dict):
    rows = []
    for key, res in results["locations"].items():
        ref = load_reference(key) or {}
        pvgis = ref.get("pvgis_pvcalc") or {}
        pvwatts = ref.get("pvwatts_v8") or {}

        iso = res["models"]["isotropic"]["annual_kwh"]
        perez = res["models"]["perez"]["annual_kwh"]
        perez_mid = (res["models"].get("perez_mid") or {}).get("annual_kwh")
        perez_diffuse = (res["models"].get("perez_diffuse") or {}).get("annual_kwh")
        perez_physical = (res["models"].get("perez_physical") or {}).get("annual_kwh")
        perez_martin_ruiz = (res["models"].get("perez_martin_ruiz") or {}).get("annual_kwh")
        perez_roof = (res["models"].get("perez_roof") or {}).get("annual_kwh")
        perez_sapm_roof = (res["models"].get("perez_sapm_roof") or {}).get("annual_kwh")
        bifacial_front = (res["models"].get("bifacial_front") or {}).get("annual_kwh")
        bifacial_rear = (res["models"].get("bifacial_rear") or {}).get("annual_kwh")
        nameplate_mono = (res["models"].get("nameplate_mono_1200") or {}).get("annual_kwh")
        nameplate_bifacial_front = (res["models"].get("nameplate_bifacial_front_1200") or {}).get("annual_kwh")
        nameplate_bifacial_rear = (res["models"].get("nameplate_bifacial_rear_1200") or {}).get("annual_kwh")
        pvgis_annual = pvgis.get("annual_kwh")
        pvwatts_annual = pvwatts.get("annual_kwh")
        perez_result = res["models"]["perez"]
        breos_poa = perez_result.get("annual_poa_kwh_m2")
        breos_capacity = perez_result.get("dc_nameplate_kw")
        pvwatts_solrad = pvwatts.get("solrad_annual_kwh_m2_day")
        pvwatts_capacity = (pvwatts.get("params") or {}).get("system_capacity")
        breos_pr = perez / (breos_poa * breos_capacity) if breos_poa and breos_capacity else None
        pvwatts_pr = (
            pvwatts_annual / (pvwatts_solrad * 365 * pvwatts_capacity)
            if pvwatts_annual and pvwatts_solrad and pvwatts_capacity
            else None
        )

        monthly_dev_max = None
        if pvgis.get("monthly_kwh"):
            devs = [_pct(b, r) for b, r in zip(res["models"]["perez"]["monthly_kwh"], pvgis["monthly_kwh"]) if r]
            if devs:
                monthly_dev_max = max(devs, key=abs)

        notes = []
        if pvgis:
            notes.append(f"PVGIS {pvgis.get('radiation_db', '?')} {pvgis.get('years', '?')}")
        if pvwatts.get("error"):
            notes.append("PVWatts: fetch failed")
        elif pvwatts:
            dist = pvwatts.get("station_distance_m")
            notes.append(f"PVWatts {pvwatts.get('dataset')} ({dist} m){'' if pvwatts.get('trusted') else ' UNTRUSTED'}")
            if pvwatts.get("mismatched_inputs"):
                notes.append(f"PVWatts mismatched inputs: {', '.join(pvwatts['mismatched_inputs'])}")
            if breos_pr is None:
                notes.append("BREOS POA/nameplate missing: run validation/run_breos.py to populate PR")
            elif pvwatts_capacity and breos_capacity != pvwatts_capacity:
                notes.append(f"PR uses own DC nameplates: BREOS {breos_capacity} kW, PVWatts {pvwatts_capacity} kW")

        rows.append(
            {
                "key": key,
                "name": ref.get("name", key),
                "iso": iso,
                "perez": perez,
                "perez_mid": perez_mid,
                "perez_diffuse": perez_diffuse,
                "physical_delta": _pct(perez_physical, perez_diffuse) if perez_physical is not None else None,
                "martin_ruiz_delta": _pct(perez_martin_ruiz, perez_diffuse) if perez_martin_ruiz is not None else None,
                "pvgis": pvgis_annual,
                "pvgis_dev": _pct(perez, pvgis_annual),
                "pvgis_dev_iso": _pct(iso, pvgis_annual),
                "pvgis_dev_mid": _pct(perez_mid, pvgis_annual) if perez_mid is not None else None,
                "pvgis_dev_diffuse": _pct(perez_diffuse, pvgis_annual) if perez_diffuse is not None else None,
                "perez_roof": perez_roof,
                "roof_delta": _pct(perez_roof, perez) if perez_roof is not None else None,
                "sapm_roof_delta": _pct(perez_sapm_roof, perez) if perez_sapm_roof is not None else None,
                "bifacial_gain": (
                    _pct(bifacial_rear, bifacial_front)
                    if bifacial_front is not None and bifacial_rear is not None
                    else None
                ),
                "nameplate_mono": nameplate_mono,
                "nameplate_bifacial_front": nameplate_bifacial_front,
                "nameplate_bifacial_rear": nameplate_bifacial_rear,
                "nameplate_front_delta": (
                    _pct(nameplate_bifacial_front, nameplate_mono)
                    if nameplate_bifacial_front is not None and nameplate_mono is not None
                    else None
                ),
                "nameplate_rear_delta": (
                    _pct(nameplate_bifacial_rear, nameplate_bifacial_front)
                    if nameplate_bifacial_rear is not None and nameplate_bifacial_front is not None
                    else None
                ),
                "pvwatts": pvwatts_annual,
                "pvwatts_trusted": bool(pvwatts_annual and pvwatts.get("trusted")),
                "pvwatts_dev": _pct(perez, pvwatts_annual),
                "breos_pr": breos_pr,
                "pvwatts_pr": pvwatts_pr,
                "pvwatts_pr_dev": _pct(breos_pr, pvwatts_pr) if breos_pr is not None else None,
                "monthly_dev_max": monthly_dev_max,
                "notes": "; ".join(notes),
                "monthly_breos": res["models"]["perez"]["monthly_kwh"],
                "monthly_pvgis": pvgis.get("monthly_kwh"),
            }
        )
    return rows


def print_table(rows, results):
    print(f"\nBREOS {results['breos_version']} validation — annual AC yield (kWh), 4 kWp system")
    header = (
        f"{'location':<16} {'BREOS iso':>10} {'BREOS perez':>12} {'perez+mid':>10} {'perez+IAM':>10} {'PVGIS':>8} "
        f"{'Δperez':>8} {'Δ+mid':>8} {'Δ+IAM':>8} {'Δphys':>8} {'ΔMR':>8} {'Δiso':>8} "
        f"{'PVsystΔ':>8} {'SAPMΔ':>8} {'rearΔ':>8} "
        f"{'PVWatts':>8} {'Δperez':>8} {'PR BREOS':>9} {'PR PVWatts':>10} {'Δ PR':>8}  notes"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['key']:<16} {r['iso']:>10.0f} {r['perez']:>12.0f} {_fmt_kwh(r['perez_mid']):>10} "
            f"{_fmt_kwh(r['perez_diffuse']):>10} {_fmt_kwh(r['pvgis']):>8} "
            f"{_fmt_pct(r['pvgis_dev']):>8} {_fmt_pct(r['pvgis_dev_mid']):>8} "
            f"{_fmt_pct(r['pvgis_dev_diffuse']):>8} {_fmt_pct(r['physical_delta']):>8} "
            f"{_fmt_pct(r['martin_ruiz_delta']):>8} {_fmt_pct(r['pvgis_dev_iso']):>8} "
            f"{_fmt_pct(r['roof_delta']):>8} {_fmt_pct(r['sapm_roof_delta']):>8} {_fmt_pct(r['bifacial_gain']):>8} "
            f"{_fmt_kwh(r['pvwatts']):>8} {_fmt_pct(r['pvwatts_dev']):>8} "
            f"{_fmt_pr(r['breos_pr']):>9} {_fmt_pr(r['pvwatts_pr']):>10} "
            f"{_fmt_pct(r['pvwatts_pr_dev']):>8}  {r['notes']}"
        )

    print("\nEqual-nameplate comparison — annual AC yield (kWh), 1.2 kWp STC")
    comparison_header = (
        f"{'location':<16} {'3x400 mono':>12} {'2x600 front':>13} {'2x600 rear':>12} "
        f"{'front vs mono':>14} {'rear vs front':>15}"
    )
    print(comparison_header)
    print("-" * len(comparison_header))
    for r in rows:
        print(
            f"{r['key']:<16} {_fmt_kwh(r['nameplate_mono']):>12} "
            f"{_fmt_kwh(r['nameplate_bifacial_front']):>13} {_fmt_kwh(r['nameplate_bifacial_rear']):>12} "
            f"{_fmt_pct(r['nameplate_front_delta']):>14} {_fmt_pct(r['nameplate_rear_delta']):>15}"
        )


def _pvwatts_status(rows):
    """Report which locations CI checks against the ±8% PVWatts PR band."""
    untrusted = [r["name"] for r in rows if not r["pvwatts_trusted"]]
    if not untrusted:
        return []
    if len(untrusted) == len(rows):
        text = (
            "No location has a trusted PVWatts v8 reference in the committed snapshot (the Notes column "
            "gives each fetch result), so the PVWatts PR band test skips all locations. "
            "This report is not a PVWatts validation."
        )
    else:
        text = (
            f"{', '.join(untrusted)} {'has' if len(untrusted) == 1 else 'have'} no trusted PVWatts v8 "
            "reference in the committed snapshot, so the PVWatts PR band test skips "
            f"{'it' if len(untrusted) == 1 else 'them'}."
        )
    return [text, ""]


def write_report(rows, results):
    system, _ = load_spec()
    lines = [
        "# BREOS validation report",
        "",
        f"Generated by `validation/compare.py` on {datetime.date.today().isoformat()} "
        f"against BREOS {results['breos_version']}.",
        "",
        f"System: {system['peak_power_kw']} kWp ({system['n_modules']}× {system['module']}), "
        f"DC/AC {system['dc_ac_ratio']}, inverter η {system['inverter_efficiency']:.2f}, "
        f"albedo {system['albedo']}, free-standing mount, BREOS default loss stack.",
        "",
        "Deviations bundle weather-source differences with model differences "
        "(BREOS runs on a PVGIS TMY; PVGIS PVcalc averages the full multi-year record; "
        "PVWatts uses NSRDB/international station data). The tight regression guarantee "
        "is the baseline test in `tests/test_validation_drift.py`. CI checks PVGIS annual AC within ±10%, "
        "and the PVWatts performance ratio within ±8%: PR = annual AC / (annual front-side POA × DC "
        "nameplate). PVWatts reports its own POA (daily solrad × 365); for BREOS the POA is an independent "
        "pvlib Perez transposition of BREOS's weather, so the ratio takes the weather out without "
        "hiding a BREOS transposition error. PVWatts annual AC is context, not the banded quantity.",
        "",
        *_pvwatts_status(rows),
        "The roof-mount column is the yield delta of `perez_roof` (PVsyst semi-integrated "
        "thermal preset) against `perez` (free-standing) — a documented model-choice effect, "
        "not a comparison against the free-standing references.",
        "",
        "The physical and Martin-Ruiz columns are their respective `perez_*` "
        "IAM-model yield deltas against the Ashrae `perez_diffuse` case. "
        "All three enable Marion diffuse IAM, so beam and diffuse optics use "
        "the same selected model.",
        "",
        "The SAPM roof-mount column is the same comparison for `perez_sapm_roof` "
        "(SAPM close-mount glass/glass). Neither roof-mount column is a reference-model claim.",
        "",
        "The bifacial rear-gain column compares `bifacial_rear` with `bifacial_front` "
        "using the same 600 W module, weather, front-side Perez model, albedo 0.2, GCR 0.35, "
        "row-center height 1.5, and pitch 6.0 (height and pitch in the same arbitrary unit).",
        "",
        "## Annual AC yield (kWh)",
        "",
        "| Location | BREOS isotropic | BREOS perez | BREOS perez+mid-interval | BREOS perez+diffuse-IAM "
        "| PVGIS PVcalc | Δ perez vs PVGIS | Δ perez+mid vs PVGIS | Δ perez+diffuse vs PVGIS "
        "| Physical IAM Δ | Martin-Ruiz IAM Δ | PVsyst roof-mount yield Δ | SAPM roof-mount yield Δ "
        "| Bifacial rear-gain Δ | PVWatts v8 | Δ perez vs PVWatts | PR BREOS | PR PVWatts | Δ PR | Notes |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['name']} | {r['iso']:.0f} | {r['perez']:.0f} | {_fmt_kwh(r['perez_mid'])} "
            f"| {_fmt_kwh(r['perez_diffuse'])} "
            f"| {_fmt_kwh(r['pvgis'])} | {_fmt_pct(r['pvgis_dev'])} | {_fmt_pct(r['pvgis_dev_mid'])} "
            f"| {_fmt_pct(r['pvgis_dev_diffuse'])} "
            f"| {_fmt_pct(r['physical_delta'])} | {_fmt_pct(r['martin_ruiz_delta'])} "
            f"| {_fmt_pct(r['roof_delta'])} "
            f"| {_fmt_pct(r['sapm_roof_delta'])} "
            f"| {_fmt_pct(r['bifacial_gain'])} "
            f"| {_fmt_kwh(r['pvwatts'])} | {_fmt_pct(r['pvwatts_dev'])} "
            f"| {_fmt_pr(r['breos_pr'])} | {_fmt_pr(r['pvwatts_pr'])} | {_fmt_pct(r['pvwatts_pr_dev'])} "
            f"| {r['notes']} |"
        )

    lines += [
        "",
        "## Equal-nameplate module comparison (1.2 kWp STC)",
        "",
        "This comparison holds weather, tilt/azimuth, albedo 0.2, DC/AC ratio 1.2, inverter efficiency 0.96, "
        "losses, Perez transposition, mid-interval solar position, physical beam IAM, Marion diffuse IAM, and "
        "the Faiman open-rack temperature model fixed. It compares 3×400 W monofacial modules with 2×600 W "
        "bifacial modules front-only, then activates infinite-sheds rear irradiance at GCR 0.35, row-center "
        "height 1.5, and pitch 6.0 (height and pitch use the same arbitrary unit).",
        "",
        "Equal STC nameplate does not mean equal module area, string layout, or electrical and temperature "
        "coefficients. Therefore `front vs mono` includes the catalog modules' different parameter sets; "
        "`rear vs front` isolates the modeled rear gain. The front side remains unshaded, so dense-row results "
        "can be optimistic and are not a bankable layout estimate.",
        "",
        "| Location | 3×400 W mono (kWh) | 2×600 W bifacial, front-only (kWh) "
        "| 2×600 W bifacial + rear (kWh) | Front vs mono | Rear vs front |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['name']} | {_fmt_kwh(r['nameplate_mono'])} | {_fmt_kwh(r['nameplate_bifacial_front'])} "
            f"| {_fmt_kwh(r['nameplate_bifacial_rear'])} | {_fmt_pct(r['nameplate_front_delta'])} "
            f"| {_fmt_pct(r['nameplate_rear_delta'])} |"
        )

    lines += ["", "## Monthly shape vs PVGIS (BREOS perez, kWh)", ""]
    for r in rows:
        if not r["monthly_pvgis"]:
            continue
        lines.append(f"### {r['name']}")
        lines.append("")
        lines.append("| | " + " | ".join(MONTH_LABELS) + " |")
        lines.append("|---|" + "---|" * 12)
        lines.append("| BREOS | " + " | ".join(f"{v:.0f}" for v in r["monthly_breos"]) + " |")
        lines.append("| PVGIS | " + " | ".join(f"{v:.0f}" for v in r["monthly_pvgis"]) + " |")
        devs = [_fmt_pct(_pct(b, p)) for b, p in zip(r["monthly_breos"], r["monthly_pvgis"])]
        lines.append("| Δ | " + " | ".join(devs) + " |")
        lines.append("")

    REPORT_PATH.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {REPORT_PATH.relative_to(PROJECT_ROOT)}")


def main():
    results_path = RESULTS_PATH if RESULTS_PATH.exists() else BASELINE_PATH
    if not Path(results_path).exists():
        print("No BREOS results found. Run validation/run_breos.py first.")
        sys.exit(1)
    with open(results_path) as f:
        results = json.load(f)

    rows = build_rows(results)
    if not rows:
        print("No locations to compare.")
        sys.exit(1)

    print_table(rows, results)
    write_report(rows, results)


if __name__ == "__main__":
    main()
