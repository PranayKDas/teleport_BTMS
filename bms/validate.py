"""Check the single-cell electrical model against what the sources print.

Isothermal tests of one ideal cell (no spread), using the same circuit as the module simulation:
OCV(SOC) - R0 - (R1 || C1). Sources: the LG M50 specification (DS) and the batterydesign.net page (BD).
The M50 specification gives no capacity-vs-rate or charge-time data, so the evidence is thinner than for a
finished datasheet. The table says which rows are independent checks and which only reproduce an input.
Rate and charge-time rows against the electrochemical model come from tools/pybamm_crosscheck.py.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .cell import CELL, Cell, cap_retention, i_term, ocv, r_temp_factor

CROSSCHECK_CSV = Path(__file__).resolve().parent.parent / "results" / "crosscheck_pybamm.csv"


def _rc(cell: Cell, t_c: float):
    f = float(r_temp_factor(t_c))
    r0, r1 = cell.r0_mohm * 1e-3 * f, cell.r1_mohm * 1e-3 * f
    return r0, r1, r1 * cell.c1_f


def discharge_curve(i_a: float, t_c: float = 25.0, cell: Cell = CELL, dt: float = 2.0, retention: bool = False,
                    soc0: float = 1.0):
    """(capacity Ah, energy Wh) of a constant-current discharge from soc0 to the 2.5 V cut-off."""
    r0, r1, tau = _rc(cell, t_c)
    a = np.exp(-dt / tau)
    ret = float(cap_retention(t_c)) if retention else 1.0
    soc, v1, q, e = soc0, 0.0, 0.0, 0.0
    while soc > 0.0:
        v = float(ocv(soc)) - v1 - i_a * r0
        if v <= cell.v_min or (soc0 - soc) >= ret:
            break
        q += i_a * dt / 3600.0
        e += v * i_a * dt / 3600.0
        soc -= i_a * dt / 3600.0 / cell.cap_ah
        v1 = v1 * a + i_a * r1 * (1.0 - a)
    return q, e


def charge_time_min(c_rate: float, t_c: float = 25.0, cell: Cell = CELL, dt: float = 2.0, soc0: float = 0.0) -> float:
    """CC-CV charge from soc0 to the end current, one ideal cell. Time in minutes."""
    r0, r1, tau = _rc(cell, t_c)
    a = np.exp(-dt / tau)
    i_cc = c_rate * cell.cap_ah
    cut = i_term(cell, i_cc)
    soc, v1, t, cv = soc0, 0.0, 0.0, False
    while t < 12 * 3600 and soc < 1.0:
        v_oc = float(ocv(soc))
        if not cv and v_oc + v1 + i_cc * r0 >= cell.v_max:
            cv = True
        i = min(i_cc, max((cell.v_max - v_oc - v1) / r0, 0.0)) if cv else i_cc
        if cv and i <= cut:
            break
        soc = min(1.0, soc + i * dt / 3600.0 / cell.cap_ah)
        v1 = v1 * a + i * r1 * (1.0 - a)      # magnitude of the polarisation voltage; it adds to the terminal voltage on charge
        t += dt
    return t / 60.0


def dcir_mohm(t_pulse_s: float = 30.0, c_rate: float = 0.5, soc: float = 0.5, t_c: float = 25.0, cell: Cell = CELL,
              dt: float = 0.25) -> float:
    """DS 4.3.2 procedure on the model: rest voltage minus voltage at the end of the pulse, divided by the current."""
    r0, r1, tau = _rc(cell, t_c)
    i_a = c_rate * cell.cap_ah
    v_rest = float(ocv(soc))
    v1, s, t = 0.0, soc, 0.0
    a = np.exp(-dt / tau)
    while t < t_pulse_s - 1e-9:
        v1 = v1 * a + i_a * r1 * (1.0 - a)
        s -= i_a * dt / 3600.0 / cell.cap_ah
        t += dt
    v_end = float(ocv(s)) - v1 - i_a * r0
    return (v_rest - v_end) / i_a * 1e3


def validation_table(cell: Cell = CELL) -> pd.DataFrame:
    rows = []
    i02 = 0.2 * cell.cap_ah
    q02, e02 = discharge_curve(i02, 25.0, cell)
    rows.append(("Energy at 0.2C, 25 C", "0.97 A to 2.5 V", "17.6 Wh min, 18.2 Wh nominal", f"{e02:.1f} Wh",
                 f"{e02 - 18.2:+.1f} Wh vs nominal", "DS 2.1, 4.3.3",
                 "independent: OCV from the electrochemical model, R from DS"))
    rows.append(("Mean discharge voltage at 0.2C", "0.97 A to 2.5 V", "3.63 V", f"{e02 / q02:.2f} V",
                 f"{e02 / q02 - 3.63:+.2f} V", "DS 2.2", "independent"))
    d30 = dcir_mohm(30.0, cell=cell)
    rows.append(("DC resistance, 0.5C pulse", "30 s, 25 C, 50 % SOC", "30 +/- 6 mOhm", f"{d30:.1f} mOhm",
                 f"{d30 - 30:+.1f} mOhm", "DS 4.3.2", "input reproduced, not independent"))
    d05 = dcir_mohm(0.5, cell=cell)
    rows.append(("DC resistance, 0.5C pulse", "0.5 s, 25 C, 50 % SOC", "21 to 25 mOhm (3 cells)", f"{d05:.1f} mOhm",
                 "top of range", "BD chart", "independent: R0 is fitted to the electrochemical model, not to these points"))
    for t_c, lab, ds in ((10, "10 C", 68.3 / 23.1), (45, "45 C", 14.2 / 23.1)):
        m = float(r_temp_factor(t_c))
        rows.append(("Resistance ratio to 25 C", lab, f"{ds:.2f} (chart means)", f"{m:.2f}", f"{m - ds:+.2f}", "BD chart",
                     "input reproduced, not independent. Chart read by eye"))
    for t_c, ds in ((0, 80), (-10, 70), (60, 95)):
        _, e_r = discharge_curve(i02, t_c, cell)
        _, e_c = discharge_curve(i02, t_c, cell, retention=True)
        rows.append(("Energy vs 25 C, resistance only", f"{t_c} C, 0.2C", f"{ds} %", f"{100 * e_r / e02:.0f} %",
                     f"{100 * e_r / e02 - ds:+.0f} points", "DS 4.4.4",
                     "independent: resistance alone does not explain the loss, so the datasheet table is added"))
        rows.append(("Energy vs 25 C, with DS table", f"{t_c} C, 0.2C", f"{ds} %", f"{100 * e_c / e02:.0f} %",
                     f"{100 * e_c / e02 - ds:+.0f} points", "DS 4.4.4", "input reproduced, not independent"))
    return pd.DataFrame(rows, columns=["Test", "Condition", "Datasheet or chart", "Model", "Difference", "Source", "Note"])


REPLAY_CSV = Path(__file__).resolve().parent.parent / "results" / "echem_replay.csv"


def replay_table() -> pd.DataFrame | None:
    """Rows written by tools/echem_replay.py, if it has been run."""
    return pd.read_csv(REPLAY_CSV) if REPLAY_CSV.exists() else None


def crosscheck_table() -> pd.DataFrame | None:
    """Rows written by tools/pybamm_crosscheck.py, if it has been run."""
    return pd.read_csv(CROSSCHECK_CSV) if CROSSCHECK_CSV.exists() else None


if __name__ == "__main__":
    print(validation_table().to_string(index=False))
    cc = crosscheck_table()
    if cc is not None:
        print()
        print(cc.to_string(index=False))
