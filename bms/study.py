"""Ambient-temperature and discharge-rate study over repeated cycles, for all designs.

Answers three questions with simulation results, not assumptions:
- does each design hold the temperature requirement at 25, 35 and 45 C ambient and at several discharge rates,
- does the peak keep rising over repeated cycles (heat not rejected between cycles),
- for the liquid design, up to which ambient can the radiator hold the coolant at the requirement temperature.
Run:  python -m bms.study
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from . import design as D
from . import sim


def _cycle_peaks(r: sim.Result):
    """Peak cell temperature in each cycle, from the phase sequence discharge, rest, charge, rest."""
    n = r.n_cells
    tmax = r.temps[:, :n].max(axis=1)
    starts = [0] + [k for k in range(1, len(r.phase)) if r.phase[k] == 0 and r.phase[k - 1] == 3]
    ends = starts[1:] + [len(r.phase)]
    return [float(tmax[a:b].max()) for a, b in zip(starts, ends)]


def ambient_rate_study(designs=None, req: D.Requirements | None = None, ambients=(25.0, 35.0, 45.0),
                       rates=(0.5, 1.0, 1.5), n_cycles: int = 3, c_chg: float = 0.6, dt: float = 10.0) -> pd.DataFrame:
    designs = designs or D.default_designs()
    req = req or D.Requirements()
    rows = []
    for t_amb in ambients:
        for c in rates:
            sc = D.Scenario(t_amb=t_amb, c_dis=c, c_chg=min(c, c_chg), n_cycles=n_cycles)
            for d in designs:
                r = sim.simulate(d, sc, dt=dt)
                i = r.ind
                pk = _cycle_peaks(r)
                row = {
                    "Ambient (C)": t_amb, "Discharge (C-rate)": c, "Design": d.name.split()[0],
                    "Peak cell (C)": round(i["peak_cell_temp_c"], 1),
                    "Mean cell (C)": round(i["avg_cell_temp_c"], 1),
                    "Spread (K)": round(i["temp_spread_k"], 2),
                    "Peak, first cycle (C)": round(pk[0], 1),
                    "Peak, last cycle (C)": round(pk[-1], 1),
                    "Net energy (Wh)": round(i["usable_energy_wh"], 1),
                    "Aux energy (Wh)": round(i["aux_energy_wh"], 1),
                    "Charge time (min)": round(i["charge_time_min"], 0),
                    "Within requirement": "yes" if i["peak_cell_temp_c"] <= req.t_peak_max else "no",
                    "Coolant above ambient (K)": round(i["coolant_above_ambient_max_k"], 1) if d.liquid else np.nan,
                }
                if d.liquid:
                    g = i["rad_g_w_k"]
                    row["Radiator can hold requirement up to ambient (C)"] = round(req.t_peak_max - i["mean_heat_load_w"] / g, 1) if g > 0 else np.nan
                rows.append(row)
    return pd.DataFrame(rows)


def failure_modes(d: D.Design, sc: D.Scenario, req: D.Requirements | None = None, dt: float = 10.0) -> pd.DataFrame:
    """Design D under coolant-side faults, with the BMS temperature trips unchanged. The healthy row is the reference."""
    req = req or D.Requirements()
    cases = [
        ("Healthy loop", d, dict(pump_flow_frac=1.0, pump_fail_min=-1.0)),
        ("Flow reduced to 50 %", d, dict(pump_flow_frac=0.5, pump_fail_min=-1.0)),
        ("Flow reduced to 25 %", d, dict(pump_flow_frac=0.25, pump_fail_min=-1.0)),
        ("Radiator fan failed", replace(d, rad_fan=False), dict(pump_flow_frac=1.0, pump_fail_min=-1.0)),
        ("Pump stops at 10 min", d, dict(pump_flow_frac=1.0, pump_fail_min=10.0)),
        ("No flow from the start", d, dict(pump_flow_frac=0.0, pump_fail_min=-1.0)),
    ]
    rows = []
    for label, dd, kw in cases:
        r = sim.simulate(dd, replace(sc, **kw), dt=dt)
        i = r.ind
        n = r.n_cells
        tmax = r.temps[:, :n].max(axis=1)
        over = float((tmax > req.t_peak_max).sum() * r.dt / 60.0)
        if i["peak_cell_temp_c"] > 60.0:
            verdict = "discharge limit 60 C exceeded"
        elif i["peak_cell_temp_c"] > 45.0:
            verdict = "above the 45 C charge limit, BMS blocks charging"
        elif i["peak_cell_temp_c"] > req.t_peak_max:
            verdict = "above the requirement"
        else:
            verdict = "within the requirement"
        rows.append({
            "Fault": label, "Peak cell (C)": round(i["peak_cell_temp_c"], 1), "Mean cell (C)": round(i["avg_cell_temp_c"], 1),
            "Minutes above requirement": round(over, 0), "BMS blocked time (min)": round(i["bms_trip_time_s"] / 60.0, 0),
            "Discharge time (min)": round(i["discharge_time_min"], 1), "Coolant rise across plate (K)": round(i["coolant_rise_max_k"], 2),
            "Result": verdict})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    print(ambient_rate_study().to_string(index=False))
