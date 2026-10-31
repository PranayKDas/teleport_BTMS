"""LG INR21700-M50 cell data and electrical sub-model.

Every number carries a source tag.
  DS    LG Chem product specification INR21700 M50 18.20Wh, 2016-08-23, status "Tentative", with section number.
  BD    batterydesign.net page "LG INR 21700 M50" (5 May 2023). Values read from its DCIR-vs-temperature chart.
  PB    PyBaMM "Chen2020" parameter set (Chen et al., J. Electrochem. Soc. 167, 080534, 2020), used offline
        to generate the open-circuit-voltage curve (tools/make_ocv.py).
  FIT   fitted offline to the PB electrochemical model at 25 C and scaled to the datasheet resistance
        (tools/fit_polarisation.py).
  ASSUMPTION  not in any source above. The reason is given and the value can be edited.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

# ---- open-circuit voltage vs state of charge (PB) -------------------------------------------
# C/50 discharge of the Chen2020 single-particle model at 25 C, voltage read at equal steps of
# discharged capacity. SOC = 1 - discharged fraction. Regenerate with tools/make_ocv.py.
SOC_PTS = np.array([0, 2, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 95, 98, 100]) / 100.0
OCV_PTS = np.array([2.500, 2.859, 3.105, 3.292, 3.430, 3.481, 3.578, 3.663, 3.746, 3.836,
                    3.944, 4.037, 4.093, 4.118, 4.157, 4.196])

# ---- resistance vs temperature (BD chart, 0.5 s DCIR) -----------------------------------------
# Mean of the plotted points at each temperature, divided by the 25 C mean (about 23 mOhm).
# 10 C: 59, 70, 76 mOhm -> 68;  25 C: 21, 23, 25 -> 23;  45 C: 14, 14.5 -> 14.2  (read from the chart, +/- 2 mOhm)
R_T_PTS = np.array([10.0, 25.0, 45.0])
R_T_FACTOR = np.array([68.3 / 23.1, 1.0, 14.2 / 23.1])

# ---- energy retention vs discharge temperature (DS 4.4.4, 0.2C discharge, charged at 25 C) -----
CAP_T_PTS = np.array([-10.0, 0.0, 25.0, 60.0])
CAP_T_RETAIN = np.array([0.70, 0.80, 1.00, 0.95])

def ocv(soc):
    return np.interp(soc, SOC_PTS, OCV_PTS)


def r_temp_factor(t_c):
    """Resistance multiplier against the 25 C value, interpolated in ln(R) between 10, 25 and 45 C.
    Below 10 C the 10-25 C slope is extended. Above 45 C the 45 C value is held (conservative: no further drop)."""
    t = np.asarray(t_c, dtype=float)
    ln = np.log(R_T_FACTOR)
    s_cold = (ln[0] - ln[1]) / (R_T_PTS[1] - R_T_PTS[0])      # per K, positive
    out = np.interp(t, R_T_PTS, ln)
    out = np.where(t < R_T_PTS[0], ln[0] + s_cold * (R_T_PTS[0] - t), out)
    return np.exp(out)


def cap_retention(t_c):
    """Fraction of the 25 C deliverable energy that can be taken at temperature t (DS 4.4.4). Held beyond the table."""
    return np.interp(np.asarray(t_c, dtype=float), CAP_T_PTS, CAP_T_RETAIN)


@dataclass(frozen=True)
class Cell:
    name: str = "LG INR21700-M50"
    cap_ah: float = 4.85          # DS 2.1, 2.9: 1C = 4,850 mA (minimum capacity; nominal energy 18.20 Wh / 3.63 V = 5.0 Ah)
    v_nom: float = 3.63           # DS 2.2
    v_max: float = 4.20           # DS 2.5 (4.20 +/- 0.05 V)
    v_min: float = 2.50           # DS 2.7 discharge cut-off
    mass_kg: float = 0.068        # DS 2.10 (68.0 +/- 1.0 g, without washer)
    dia_m: float = 0.0211         # DS 3.2 (max)
    len_m: float = 0.07015        # DS 3.2 (max)
    i_dis_max: float = 7.275      # DS 2.9: 1.5C for 5 to 60 C
    i_dis_cold: float = 1.455     # DS 2.9: 0.3C for -20 to 5 C
    t_dis_split: float = 5.0      # DS 2.9
    i_chg_std: float = 1.455      # DS 2.4: standard charge 0.3C
    i_chg_cold: float = 1.455     # DS 2.6: 0.3C for 0 to 25 C
    i_chg_max: float = 3.395      # DS 2.6: 0.7C for 25 to 50 C
    t_chg_split: float = 25.0     # DS 2.6
    t_chg_min: float = 0.0        # DS 5.1: charge only between 0 and 45 C
    t_chg_max: float = 45.0       # DS 5.1 (DS 2.6 lists current up to 50 C; the stricter limit is used)
    t_chg_resume: float = 40.0    # ASSUMPTION: datasheet gives no release level; 5 K hysteresis
    t_dis_min: float = -20.0      # DS 5.1: discharge between -20 and 60 C
    t_dis_max: float = 60.0       # DS 5.1
    t_dis_resume: float = 55.0    # ASSUMPTION: datasheet gives no release level; 5 K hysteresis
    i_term_std: float = 0.050     # DS 2.4: standard charge ends at 50 mA
    i_term_fast_c: float = 0.05   # DS 4.2.3: cycle-life test ends at 1/20C
    r_ac_mohm: float = 25.0       # DS 4.3.1: 1 kHz impedance (max)
    r_dc_mohm: float = 30.0       # DS 4.3.2: 30 +/- 6 mOhm, 0.5C for 30 s, 25 C, 50 % SOC
    r0_mohm: float = 24.78        # FIT: series resistance at 25 C (chart 0.5 s DCIR at 25 C: 21 to 25 mOhm)
    r1_mohm: float = 14.07        # FIT: polarisation resistance at 25 C (solid-state diffusion)
    tau_s: float = 65.0           # FIT: polarisation time constant at 25 C; C1 = tau / R1 is held constant
    cp_j_kgk: float = 900.0       # ASSUMPTION: typical cylindrical Li-ion 800 to 1100; no M50 value in DS or BD
    price_inr: float = 450.0      # ASSUMPTION placeholder, replace with a real quote

    @property
    def r_dc30s_model(self) -> float:
        """30 s DC resistance the circuit gives at 25 C, in ohm: R0 + R1 (1 - exp(-30 / tau))."""
        return (self.r0_mohm + self.r1_mohm * (1.0 - np.exp(-30.0 / self.tau_s))) * 1e-3

    @property
    def c1_f(self) -> float:
        """Polarisation capacitance in farad."""
        return self.tau_s / (self.r1_mohm * 1e-3)

    @property
    def e_nom_wh(self) -> float:
        return self.cap_ah * self.v_nom


CELL = Cell()


def i_term(cell: Cell, i_chg_cell: float) -> float:
    """Charge termination current per cell: 50 mA after a standard charge (DS 2.4), C/20 for faster charging (DS 4.2.3)."""
    return cell.i_term_std if i_chg_cell <= cell.i_chg_std + 1e-9 else cell.i_term_fast_c * cell.cap_ah


def i_chg_limit(cell: Cell, t_c):
    """Maximum charge current per cell at cell temperature t (DS 2.6, 5.1). Zero outside 0 to 45 C."""
    t = np.asarray(t_c, dtype=float)
    lim = np.where(t < cell.t_chg_split, cell.i_chg_cold, cell.i_chg_max)
    return np.where((t < cell.t_chg_min) | (t > cell.t_chg_max), 0.0, lim)


def i_dis_limit(cell: Cell, t_c):
    """Maximum continuous discharge current per cell at cell temperature t (DS 2.9, 5.1)."""
    t = np.asarray(t_c, dtype=float)
    lim = np.where(t < cell.t_dis_split, cell.i_dis_cold, cell.i_dis_max)
    return np.where((t < cell.t_dis_min) | (t > cell.t_dis_max), 0.0, lim)


# Source table for the dashboard and the report.
_SOURCES = {
    "cap_ah": ("Capacity (rated minimum, defines 1C)", "Ah", "DS 2.1, 2.9"),
    "v_nom": ("Nominal voltage", "V", "DS 2.2"),
    "v_max": ("Charge voltage", "V", "DS 2.5"),
    "v_min": ("Discharge cut-off", "V", "DS 2.7"),
    "mass_kg": ("Cell mass", "kg", "DS 2.10"),
    "dia_m": ("Diameter (max)", "m", "DS 3.2"),
    "len_m": ("Height (max)", "m", "DS 3.2"),
    "i_dis_max": ("Max continuous discharge, 5 to 60 C (1.5C)", "A", "DS 2.9"),
    "i_dis_cold": ("Max continuous discharge, -20 to 5 C (0.3C)", "A", "DS 2.9"),
    "i_chg_std": ("Standard charge current (0.3C)", "A", "DS 2.4"),
    "i_chg_cold": ("Max charge current, 0 to 25 C (0.3C)", "A", "DS 2.6"),
    "i_chg_max": ("Max charge current, 25 to 50 C (0.7C)", "A", "DS 2.6"),
    "t_chg_min": ("Charge temperature, min", "C", "DS 5.1"),
    "t_chg_max": ("Charge temperature, max", "C", "DS 5.1"),
    "t_chg_resume": ("Charge release below", "C", "ASSUMPTION: not in datasheet, 5 K hysteresis"),
    "t_dis_min": ("Discharge temperature, min", "C", "DS 5.1"),
    "t_dis_max": ("Discharge temperature, max", "C", "DS 5.1"),
    "t_dis_resume": ("Discharge release below", "C", "ASSUMPTION: not in datasheet, 5 K hysteresis"),
    "i_term_std": ("Charge end current, standard charge", "A", "DS 2.4"),
    "i_term_fast_c": ("Charge end current, faster charge (C/20)", "C", "DS 4.2.3"),
    "r_ac_mohm": ("Impedance at 1 kHz (max)", "mOhm", "DS 4.3.1"),
    "r_dc_mohm": ("DC resistance, 25 C, 50 % SOC, 30 s (anchor)", "mOhm", "DS 4.3.2 (30 +/- 6)"),
    "r0_mohm": ("Series resistance R0, 25 C", "mOhm", "FIT to PB, scaled so R0 + R1 (1 - e^(-30/tau)) = 30"),
    "r1_mohm": ("Polarisation resistance R1, 25 C", "mOhm", "FIT to PB, same scaling"),
    "tau_s": ("Polarisation time constant, 25 C", "s", "FIT to PB (tools/fit_polarisation.py)"),
    "cp_j_kgk": ("Specific heat", "J/kg/K", "ASSUMPTION: typical 800 to 1100, no M50 value in sources"),
    "price_inr": ("Cell price", "INR", "ASSUMPTION: placeholder, use a real quote"),
}


def cell_param_table(cell: Cell = CELL):
    rows = []
    for f in fields(cell):
        if f.name in _SOURCES:
            label, unit, src = _SOURCES[f.name]
            v = getattr(cell, f.name)
            rows.append({"Parameter": label, "Value": f"{v:g}" if isinstance(v, float) else str(v), "Unit": unit, "Source": src})
    rows.append({"Parameter": "Open-circuit voltage curve", "Value": f"{len(SOC_PTS)} points", "Unit": "V vs SOC",
                 "Source": "PB: Chen2020 single-particle model at C/50, 25 C (not a datasheet curve)"})
    rows.append({"Parameter": "Resistance and tau vs temperature",
                 "Value": "x%.2f at 10 C, x1 at 25 C, x%.2f at 45 C" % (R_T_FACTOR[0], R_T_FACTOR[2]), "Unit": "-",
                 "Source": "BD: DCIR chart read by eye (+/- 2 mOhm). Below 10 C extended, above 45 C held"})
    rows.append({"Parameter": "Energy retention vs temperature",
                 "Value": "70 % at -10 C, 80 % at 0 C, 100 % at 25 C, 95 % at 60 C", "Unit": "-",
                 "Source": "DS 4.4.4 (0.2C discharge)"})
    return rows
