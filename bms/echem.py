"""Electrochemical layer: lithium-plating screen from the PyBaMM Chen2020 model, plus an optional live replay.

What it adds to the equivalent circuit. The circuit knows terminal voltage, heat and temperature. It has no electrode
variables, so it cannot tell whether a charge current is safe for the anode. The electrochemical model does:
the potential of the negative electrode against lithium metal falls below 0 V when plating becomes possible.

Source and limits (read these before using the numbers)
- PB: PyBaMM Chen2020 (LG M50 research parameter set), DFN, 25 C, constant-current charge from empty (tools/make_plating_map.py).
- The Chen2020 set has no temperature dependence, so the map is a 25 C map. Plating risk rises when the cell is colder.
  The platform therefore evaluates the screen only for cells at 25 C or above and reports the charge time spent below that.
- A negative potential is a screening criterion for possible plating, not a measurement of plated lithium.
- The map is for CC charge from empty. Charging from a partly charged state or after a rest changes the concentration gradients.
- The limit is read at the state of charge of the simulation. SOC scales are rated-capacity based in the circuit and
  accepted-charge based in the map; they differ by about 1 %.
The runtime check needs only data/plating_map.csv (no PyBaMM). `replay` needs PyBaMM and is optional.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .cell import CELL

MAP_PATH = Path(__file__).resolve().parent.parent / "data" / "plating_map.csv"
T_VALID_MIN = 25.0          # C, the map is a 25 C map
SOC_GRID = np.linspace(0.0, 1.0, 101)


@lru_cache(maxsize=2)
def _grid(model: str = "DFN"):
    """Return rates (ascending) and phi[rate, soc] in mV on SOC_GRID. Beyond the SOC a rate reaches before 4.2 V the last value is held."""
    df = pd.read_csv(MAP_PATH)
    df = df[df.model == model]
    rates = np.array(sorted(df.c_rate.unique()))
    phi = np.zeros((len(rates), len(SOC_GRID)))
    for k, c in enumerate(rates):
        d = df[df.c_rate == c].sort_values("soc")
        phi[k] = np.interp(SOC_GRID, d.soc.values, d.phi_neg_mv.values)   # np.interp holds the end values
    return rates, phi


def plating_limit_c(soc, model: str = "DFN", margin_mv: float = 0.0) -> np.ndarray:
    """Highest constant charge rate (datasheet C, 1C = 4.85 A) that keeps the negative-electrode potential at or above margin_mv.
    Capped at the highest rate in the map (1.5C)."""
    rates, phi = _grid(model)
    soc = np.atleast_1d(np.asarray(soc, dtype=float))
    out = np.empty(len(soc))
    for n, s in enumerate(soc):
        col = np.array([np.interp(s, SOC_GRID, phi[k]) for k in range(len(rates))]) - margin_mv
        if col[-1] >= 0:
            out[n] = rates[-1]
        elif col[0] < 0:
            out[n] = rates[0]
        else:
            k = int(np.argmax(col < 0))              # first rate that violates
            out[n] = rates[k - 1] + (rates[k] - rates[k - 1]) * col[k - 1] / (col[k - 1] - col[k])
    return out


def plating_limit_a(soc, cell=CELL, model: str = "DFN", margin_mv: float = 0.0) -> np.ndarray:
    return plating_limit_c(soc, model, margin_mv) * cell.cap_ah


def min_potential_mv(c_rate: float, model: str = "DFN") -> float:
    """Lowest anode potential reached during CC charge at this rate (interpolated between the mapped rates)."""
    rates, phi = _grid(model)
    return float(np.interp(c_rate, rates, phi.min(axis=1)))


# ---- optional live replay -------------------------------------------------------------------

def available() -> bool:
    try:
        import pybamm  # noqa: F401
        return True
    except Exception:
        return False


def replay(t_s, i_a, soc0: float, model: str = "SPMe"):
    """Drive the Chen2020 electrochemical model with a cell current profile. i_a is positive on discharge.
    Returns dict with time, terminal voltage and negative-electrode potential at the separator (V). Needs PyBaMM; 25 C."""
    import pybamm

    t_s = np.asarray(t_s, dtype=float)
    t_s = t_s - t_s[0]
    # strictly increasing time axis for the interpolant
    keep = np.concatenate([[True], np.diff(t_s) > 0])
    t_s, i_a = t_s[keep], np.asarray(i_a, dtype=float)[keep]
    pv = pybamm.ParameterValues("Chen2020")
    pv["Current function [A]"] = pybamm.Interpolant(t_s, i_a, pybamm.t, name="module cell current", interpolator="linear")
    m = getattr(pybamm.lithium_ion, model)(options={"thermal": "isothermal"})
    sim = pybamm.Simulation(m, parameter_values=pv)
    sol = sim.solve(t_eval=t_s, initial_soc=float(np.clip(soc0, 0.0, 1.0)))
    key = "Negative electrode surface potential difference at separator interface [V]"
    return {"t_s": sol["Time [s]"].entries, "v": sol["Terminal voltage [V]"].entries, "phi_neg": sol[key].entries,
            "model": model}
