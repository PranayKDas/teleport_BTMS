"""Requirement and limit checker, traceability table and design ranking.

Two kinds of check:
- datasheet limits (hard): the cell would be outside its specified window
- requirements (soft): the application target is missed, edited in the sidebar

The interface checks look across subsystems: charger vs cell, BMS vs requirement,
sensor placement vs hotspot, module voltage vs load window.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import design as D
from .cell import CELL, Cell, i_chg_limit
from .sim import Result


def _fmt(x):
    return x if isinstance(x, str) else f"{x:.4g}"


def _row(cid, kind, text, value, limit, ok, unit="", why=""):
    return {"ID": cid, "Type": kind, "Check": text, "Value": _fmt(value), "Limit": _fmt(limit),
            "Unit": unit, "Status": "PASS" if ok else "FAIL", "Why it matters": why}


def run_checks(res: Result, req: D.Requirements, cell: Cell = CELL) -> pd.DataFrame:
    d, sc, i = res.design, res.scenario, res.ind
    r = []
    # ---- datasheet limits ----
    r.append(_row("L1", "Datasheet", "Peak cell temperature while charging",
                  round(i["peak_temp_charge_c"], 1), cell.t_chg_max, i["peak_temp_charge_c"] <= cell.t_chg_max,
                  "C", "DS 5.1: charge only between 0 and 45 C"))
    r.append(_row("L2", "Datasheet", "Peak cell temperature while discharging",
                  round(i["peak_temp_discharge_c"], 1), cell.t_dis_max, i["peak_temp_discharge_c"] <= cell.t_dis_max,
                  "C", "DS 5.1: discharge between -20 and 60 C"))
    r.append(_row("L3", "Datasheet", "Peak cell discharge current vs limit at its temperature",
                  round(100 * i["peak_limit_ratio_dis"], 0), 100, i["peak_limit_ratio_dis"] <= 1.0, "% of limit",
                  "DS 2.9: 1.5C from 5 to 60 C, 0.3C from -20 to 5 C. Mismatch makes one cell carry more than its share"))
    r.append(_row("L4", "Datasheet", "Peak cell charge current vs limit at its temperature",
                  round(100 * i["peak_limit_ratio_chg"], 0), 100, i["peak_limit_ratio_chg"] <= 1.0, "% of limit",
                  "DS 2.6: 0.7C from 25 to 45 C, 0.3C from 0 to 25 C. Charging at the limit leaves no room for cell spread"))
    r.append(_row("E1", "Electrochemical", "Peak cell charge current vs plating-free current at its SOC (cells >= 25 C)",
                  round(100 * i["peak_plating_ratio"], 0), 100, i["peak_plating_ratio"] <= 1.0, "% of limit",
                  "Chen2020 DFN at 25 C: negative-electrode potential vs lithium stays >= 0 V. Screening criterion, not a plating measurement. "
                  + ("Charge time with a cell below 25 C, outside the map: %.0f min" % (i["plating_unrated_s"] / 60.0)
                     if i["plating_unrated_s"] > 0 else "No charge time below 25 C")))
    r.append(_row("L5", "Datasheet", "Lowest group voltage",
                  round(i["min_group_voltage_v"], 2), cell.v_min, i["min_group_voltage_v"] >= cell.v_min - 1e-3,
                  "V", "DS 2.7: 2.5 V discharge cut-off"))
    r.append(_row("L6", "Datasheet", "Ambient inside the charge window", sc.t_amb,
                  f"{cell.t_chg_min:g} to {cell.t_chg_max:g}", cell.t_chg_min <= sc.t_amb <= cell.t_chg_max,
                  "C", "DS 5.1: no charging below 0 C or above 45 C"))
    # ---- requirements ----
    r.append(_row("R1", "Requirement", "Peak cell temperature", round(i["peak_cell_temp_c"], 1), req.t_peak_max,
                  i["peak_cell_temp_c"] <= req.t_peak_max, "C",
                  "45 C is the top of the datasheet charge window: above it the BMS blocks charging until the module cools"))
    r.append(_row("R2", "Requirement", "Usable energy", round(i["usable_energy_wh"], 1), req.energy_min_wh,
                  i["usable_energy_wh"] >= req.energy_min_wh, "Wh", "Net of fan energy"))
    r.append(_row("R3", "Requirement", "Charge time", round(i["charge_time_min"], 0), req.charge_time_max_min,
                  i["charge_time_min"] <= req.charge_time_max_min, "min", "Includes the CV tail and any BMS pauses"))
    r.append(_row("R4", "Requirement", "Mass", round(i["mass_g"], 0), req.mass_max_g, i["mass_g"] <= req.mass_max_g, "g", ""))
    r.append(_row("R5", "Requirement", "Estimated cost", round(i["cost_inr"], 0), req.cost_max_inr,
                  i["cost_inr"] <= req.cost_max_inr, "INR", "Placeholder prices, replace with quotes"))
    r.append(_row("R6", "Requirement", "Volume", round(i["volume_cm3"], 0), req.volume_max_cm3,
                  i["volume_cm3"] <= req.volume_max_cm3, "cm3", "Cell block plus plate and fan"))
    r.append(_row("R8", "Requirement", "Smallest gap between cells", d.spacing_mm, req.min_gap_mm, d.spacing_mm >= req.min_gap_mm,
                  "mm", "Design rule. DS 5.1 asks for insulation between cells and metal parts; the datasheet gives no spacing"))
    lo, hi = req.v_window
    v_ok = i["module_v_min"] >= lo - 1e-9 and i["module_v_max"] <= hi + 1e-9
    r.append(_row("R7", "Requirement", "Module voltage range fits the load and charger window",
                  f"{i['module_v_min']:.1f} to {i['module_v_max']:.1f}", f"{lo:g} to {hi:g}", v_ok, "V",
                  "Series count sets the voltage window"))
    # ---- interface checks ----
    i_chg_cell = sc.c_chg * cell.cap_ah
    lim_amb = float(i_chg_limit(cell, sc.t_amb))
    r.append(_row("I1", "Interface", "Charger current per cell vs cell limit at ambient temperature", round(i_chg_cell, 2),
                  round(lim_amb, 2), i_chg_cell <= lim_amb + 1e-9, "A",
                  "Charger and cell must agree on the current. A cold cell accepts only 0.3C"))
    r.append(_row("I2", "Interface", "BMS trip level vs temperature requirement",
                  f"trip {cell.t_dis_max:g} (discharge), {cell.t_chg_max:g} (charge)", f"req {req.t_peak_max:g}",
                  i["peak_cell_temp_c"] <= req.t_peak_max,
                  "C", "The discharge trip sits 15 K above the requirement, so a miss raises no alarm during discharge. "
                       "Add a warning level or derate current near the requirement"))
    r.append(_row("I3", "Interface", "BMS sensor under-reads the hottest cell", round(i["sensor_underread_k"], 1), 2.0,
                  i["sensor_underread_k"] <= 2.0, "K", f"Sensor on {d.ntc} cell. Hotspot is cell {i['hottest_cell']}. "
                  "The 45 C charge cut-off acts on the sensor reading, not on the hottest cell"))
    r.append(_row("I4", "Interface", "Fan and pump power as a share of module power" if d.liquid else "Fan power as a share of module power", round(100 * i["fan_power_w"] / max(i["usable_energy_wh"] / (i["discharge_time_min"] / 60.0), 1e-9), 2),
                  2.0, 100 * i["fan_power_w"] / max(i["usable_energy_wh"] / (i["discharge_time_min"] / 60.0), 1e-9) <= 2.0,
                  "%", "Parasitic load lowers usable energy"))
    # ---- liquid loop (Design D only). Engineering-estimate limits, labelled as such ----
    if d.liquid:
        rise, rise_lim = i["coolant_rise_max_k"], 5.0
        r.append(_row("LQ1", "Liquid loop", "Coolant temperature rise across the cold plate", round(rise, 2), rise_lim, rise <= rise_lim,
                      "K", "Flow adequacy: Q = m cp (T_out - T_in). A large rise means too little flow for the heat load. "
                           "5 K is a common design guideline, not a datasheet value"))
        g_rad = i["rad_g_w_k"]
        cap = g_rad * (req.t_peak_max - sc.t_amb)                       # W the radiator can reject with coolant at the requirement temperature
        load = max(i["mean_heat_load_w"], 1e-9)
        ratio = max(cap, 0.0) / load
        r.append(_row("LQ2", "Liquid loop", "Radiator capacity at the requirement temperature vs mean heat load", round(100 * ratio, 0), 100,
                      ratio >= 1.0, "% of load",
                      f"G_rad (T_req - T_amb) = {max(cap, 0.0):.1f} W against {load:.1f} W of cell, bus and pump heat. "
                      "The coolant cannot go below ambient without refrigeration, so this falls to zero when ambient reaches the requirement"))
        gap, gap_lim = i["cell_minus_coolant_max_k"], 10.0
        r.append(_row("LQ3", "Liquid loop", "Hottest cell minus coolant temperature", round(gap, 1), gap_lim, gap <= gap_lim, "K",
                      "Share of the temperature budget used by the pad, plate and plate-to-coolant resistance. 10 K is a guideline"))
        r.append(_row("LQ4", "Liquid loop", "Cell-to-plate pad is an electrical insulator", "yes" if d.pad_dielectric else "no", "yes",
                      d.pad_dielectric, "-", "Design rule, cannot be verified by simulation. Series groups sit at different "
                      "potentials (up to %.1f V), so a conductive pad would short them through the aluminium plate" % i["module_v_max"]))
    return pd.DataFrame(r)


def summarize(df: pd.DataFrame) -> dict:
    return {
        "datasheet_fail": int(((df.Type == "Datasheet") & (df.Status == "FAIL")).sum()),
        "echem_fail": int(((df.Type == "Electrochemical") & (df.Status == "FAIL")).sum()),
        "liquid_fail": int(((df.Type == "Liquid loop") & (df.Status == "FAIL")).sum()),
        "req_fail": int(((df.Type == "Requirement") & (df.Status == "FAIL")).sum()),
        "interface_fail": int(((df.Type == "Interface") & (df.Status == "FAIL")).sum()),
    }


# ---- traceability -------------------------------------------------------------

def traceability(res: Result, checks: pd.DataFrame, cell: Cell = CELL) -> pd.DataFrame:
    """Requirement -> component -> assumption -> model -> result -> validation."""
    d = res.design
    st = {row.ID: f"{row.Value} {row.Unit} vs {row.Limit} ({row.Status})" for row in checks.itertuples()}
    rows = [
        ("R1 Peak cell temperature", f"{cell.name}; {'Al spreader plate; ' if d.spreader else ''}"
         f"{'fan; ' if d.fan_v > 0 else ''}{'cold plate, coolant loop, radiator, pump; ' if d.liquid else ''}{d.spacing_mm:g} mm cell gap",
         "h = %.0f W/m2K (natural), cell cp %.0f J/kgK, %.0f %% resistance spread" % (d.h_nat, cell.cp_j_kgk, 100 * res.scenario.r_sigma),
         "I2R0 + V1^2/R1 heat per cell, %d-node thermal network" % (d.n_cells + (1 if d.spreader else 0)), st["R1"],
         "R(T) from the BD chart, 30 mOhm from DS 4.3.2; no M50 thermal data in the sources, so a bench test is the next step"),
        *([("LQ1 to LQ4 Liquid loop", "Cold plate, pads, water-glycol loop, pump, radiator, radiator fan",
            "Coolant %s: m_dot %.0f g/s, UA plate %.0f W/K, UA radiator %.0f W/K, pad %.1f W/mK x %.1f mm (all ESTIMATES)"
            % (d.coolant_type, 1000 * d.coolant_mdot_kg_s, d.ua_plate_w_k, d.rad_ua_w_k, d.pad_k, d.pad_t_mm),
            "Plate and coolant nodes in the thermal network, effectiveness-NTU conductances, no refrigeration",
            f"{int((checks[checks.Type == 'Liquid loop'].Status == 'FAIL').sum())} fail",
            "Energy balance tested; no measurement. Coolant flow, UA values and pad resistance need a bench test")] if d.liquid else []),
        ("R2 Usable energy", f"{d.n_series}S{d.n_parallel}P of {cell.name}",
         "OCV curve from the Chen2020 electrochemical model, R0 + R1 = 30 mOhm at 30 s (DS 4.3.2)",
         "SOC integration to 2.5 V cut-off, DS energy retention vs temperature", st["R2"],
         "17.8 Wh at 0.2C vs 17.6 min and 18.2 nominal (DS 2.1). Energy at 1C is 2 to 3 points above the electrochemical model"),
        ("R3 Charge time", "Charger CC-CV, BMS", "End current C/20 (DS 4.2.3); polarisation branch fitted to the electrochemical model",
         "CC-CV with BMS temperature pauses", st["R3"],
         "No charge-time data in the datasheet. 10 to 16 % shorter than the electrochemical model at 25 C, so treat it as a lower bound"),
        ("R4 Mass", "Cells, holders, strip, BMS, plate, pads, fan", "Holder 2 g, strip 10 g + 4 g per series group, BMS 12 g",
         "Sum of parts", st["R4"], "DS 2.10 cell mass; other masses are estimates to be weighed"),
        ("R5 Cost", "Same parts list", "Cell INR 450 placeholder, BMS INR 200 + 75 per series group",
         "Sum of parts", st["R5"], "Replace with supplier quotes"),
        ("R6 Volume", "Cell block, plate, fan", "Cells on a square grid at the stated gap", "Bounding box", st["R6"], "Check against CAD"),
        ("R8 Cell gap", "Cell holder", "Design rule, 1 mm minimum", "Direct comparison", st["R8"], "DS 5.1 asks for insulation, no spacing given"),
        ("R7 Voltage window", f"{d.n_series} series groups", "Nominal 3.63 V, 2.5 to 4.2 V per cell (DS 2.2, 2.5, 2.7)",
         "Series count x cell limits", st["R7"], "Datasheet values"),
        ("L1 to L5 Cell limits", "BMS", "Datasheet limits used as BMS trip levels (DS 2.6, 2.9, 2.7, 5.1)",
         "Limit checks on every time step", f"{int((checks[checks.Type == 'Datasheet'].Status == 'FAIL').sum())} fail", "Datasheet"),
        ("I1 to I4 Interfaces", "Charger, BMS, sensor, fan", "Sensor on a %s cell" % d.ntc, "Cross-subsystem checks",
         f"{int((checks[checks.Type == 'Interface'].Status == 'FAIL').sum())} fail", "Model comparison with hotspot"),
    ]
    return pd.DataFrame(rows, columns=["Requirement", "Component", "Assumption", "Model", "Result", "Validation or source"])


# ---- ranking ------------------------------------------------------------------

DEFAULT_WEIGHTS = {"peak_cell_temp_c": 0.30, "usable_energy_wh": 0.15, "charge_time_min": 0.15,
                   "mass_g": 0.15, "cost_inr": 0.15, "volume_cm3": 0.10}
HIGHER_BETTER = {"usable_energy_wh"}


def rank(results: list[Result], checks: list[pd.DataFrame], weights=None) -> pd.DataFrame:
    w = weights or DEFAULT_WEIGHTS
    keys = list(w)
    vals = pd.DataFrame([{k: r.ind[k] for k in keys} for r in results])
    score = np.zeros(len(results))
    for k in keys:
        col = vals[k].astype(float)
        span = col.max() - col.min()
        norm = np.zeros(len(col)) if span < 1e-12 else (col - col.min()) / span
        norm = norm if k in HIGHER_BETTER else 1.0 - norm
        score += w[k] * np.asarray(norm)
    out = pd.DataFrame({
        "Design": [r.design.name for r in results],
        "Datasheet fails": [summarize(c)["datasheet_fail"] for c in checks],
        "Requirement fails": [summarize(c)["req_fail"] for c in checks],
        "Interface fails": [summarize(c)["interface_fail"] for c in checks],
        "Electrochemical fails": [summarize(c)["echem_fail"] for c in checks],
        "Liquid-loop fails": [summarize(c)["liquid_fail"] for c in checks],
        "Score": score.round(3),
    })
    out["Acceptable"] = (out["Datasheet fails"] == 0) & (out["Requirement fails"] == 0) & (out["Electrochemical fails"] == 0) & (out["Liquid-loop fails"] == 0)
    return out.sort_values(["Datasheet fails", "Electrochemical fails", "Requirement fails", "Liquid-loop fails", "Score"],
                           ascending=[True, True, True, True, False]).reset_index(drop=True)
