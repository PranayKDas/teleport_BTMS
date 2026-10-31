"""Regenerate the open-circuit-voltage table in bms/cell.py from the PyBaMM Chen2020 (LG M50) parameter set.

Needs PyBaMM (see requirements-extra.txt). Not needed to run the platform.
    python tools/make_ocv.py
"""
import warnings

import numpy as np
import pybamm

warnings.filterwarnings("ignore")
SOC = [0, 2, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 95, 98, 100]

pv = pybamm.ParameterValues("Chen2020")
model = pybamm.lithium_ion.SPM(options={"thermal": "isothermal"})
exp = pybamm.Experiment(["Discharge at 0.1 A for 60 hours or until 2.5 V"], period="10 minutes")  # about C/50
sol = pybamm.Simulation(model, parameter_values=pv, experiment=exp).solve(initial_soc=1.0)
t, v, i = (sol[k].entries for k in ("Time [s]", "Terminal voltage [V]", "Current [A]"))
q = np.concatenate([[0.0], np.cumsum(0.5 * (i[1:] + i[:-1]) * np.diff(t)) / 3600.0])
dod = 1.0 - np.array(SOC) / 100.0
ocv = np.interp(dod * q[-1], q, v)
print(f"C/50 discharge capacity {q[-1]:.3f} Ah")
print("SOC_PTS = np.array(%s) / 100.0" % SOC)
print("OCV_PTS = np.array(%s)" % [round(float(x), 3) for x in ocv])
