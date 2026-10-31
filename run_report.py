"""Run the three default designs and write tables and figures to ./results.

Usage: python run_report.py
Results are repeatable: the cell-to-cell spread uses a fixed random seed.
"""
from pathlib import Path

import pandas as pd

from bms import design as D
from bms.checks import rank, run_checks, summarize, traceability
from bms.plots import (cell_heatmaps, coolant_vs_time, electrical, heat_removal, indicator_bars, plating_plot, sweep_plot,
                       temperature_vs_time)
from bms.sensitivity import run as run_sensitivity
from bms.sim import simulate
from bms.study import ambient_rate_study, failure_modes
from bms.validate import validation_table

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)

sc = D.Scenario()
req = D.Requirements()
designs = D.default_designs()
results = [simulate(d, sc) for d in designs]
checks = [run_checks(r, req) for r in results]

cols = ["usable_energy_wh", "peak_cell_temp_c", "temp_spread_k", "charge_time_min", "discharge_time_min",
        "mass_g", "cost_inr", "volume_cm3", "energy_density_wh_kg", "energy_density_wh_l",
        "round_trip_eff", "peak_cell_current_dis_a", "sensor_underread_k", "fan_power_w", "peak_plating_ratio",
        "avg_cell_temp_c", "aux_energy_wh", "pump_energy_wh", "rad_fan_energy_wh", "air_fan_energy_wh",
        "coolant_peak_c", "coolant_above_ambient_max_k", "coolant_rise_max_k", "peak_radiator_w", "heat_radiator_wh", "heat_air_wh"]
table = pd.DataFrame([{k: r.ind[k] for k in cols} for r in results], index=[r.design.name for r in results]).round(2)
table.to_csv(OUT / "comparison.csv")
ranking = rank(results, checks)
ranking.to_csv(OUT / "ranking.csv", index=False)
for r, c in zip(results, checks):
    tag = r.design.name.split()[0]
    c.to_csv(OUT / f"checks_{tag}.csv", index=False)
    traceability(r, c).to_csv(OUT / f"traceability_{tag}.csv", index=False)
validation_table().to_csv(OUT / "validation.csv", index=False)
run_sensitivity(designs, sc, req).to_csv(OUT / "sensitivity.csv", index=False)

temperature_vs_time(results, req).savefig(OUT / "fig_temperature.png")
indicator_bars(results, req).savefig(OUT / "fig_indicators.png")
cell_heatmaps(results, req).savefig(OUT / "fig_cell_map.png", bbox_inches="tight")
electrical(results[0]).savefig(OUT / "fig_electrical_baseline.png")
plating_plot(results).savefig(OUT / "fig_plating.png")
coolant_vs_time(results).savefig(OUT / "fig_coolant.png", bbox_inches="tight")
heat_removal(results).savefig(OUT / "fig_heat_removal.png")
failure_modes(designs[-1], sc, req).to_csv(OUT / "liquid_fault_cases.csv", index=False) if designs[-1].liquid else None
ambient_rate_study(designs, req).to_csv(OUT / "study_ambient_rate.csv", index=False)

rates = [0.5, 0.75, 1.0, 1.25, 1.5]      # 1.5C is the datasheet discharge maximum
curves = {}
for d in designs:
    ys = []
    for c in rates:
        s = D.Scenario(t_amb=sc.t_amb, c_dis=c, c_chg=min(c, sc.c_chg), n_cycles=1, seed=sc.seed)
        ys.append(simulate(d, s, dt=10.0).ind["peak_cell_temp_c"])
    curves[d.name] = ys
sweep_plot(rates, curves, "discharge rate (C)", req.t_peak_max,
           f"Peak cell temperature vs discharge rate, {sc.t_amb:g} C ambient").savefig(OUT / "fig_sweep_crate.png")

print(table.T.to_string())
print()
print(ranking.to_string(index=False))
print("\nFiles written to", OUT)
