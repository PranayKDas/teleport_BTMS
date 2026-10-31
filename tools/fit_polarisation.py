"""Fit the one-RC polarisation branch of the equivalent circuit to the PyBaMM Chen2020 (LG M50) DFN model at 25 C.

Why: a resistance-only cell cannot reproduce the slow tail of a constant-voltage charge or the energy lost at higher
discharge rates, because both come from solid-state diffusion. A series resistance R0 plus one R1 || C1 branch can.
Procedure: 0.5C pulse for 20 min from 50 % SOC, then 30 min rest, simulated in the DFN. The circuit voltage is
fitted to it (R0, R1, tau). R0 and R1 are then scaled by one common factor so that the 30 s DC resistance equals the
datasheet value (DS 4.3.2, 30 mOhm). The fit uses nothing from the charge or high-rate tests, so those stay independent.

Needs PyBaMM and SciPy.    python tools/fit_polarisation.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pybamm
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bms.cell import CELL, ocv  # noqa: E402

warnings.filterwarnings("ignore")
PV = pybamm.ParameterValues("Chen2020")


def run_dfn():
    model = pybamm.lithium_ion.DFN(options={"thermal": "isothermal"})
    q = 5.118
    i_a = 0.5 * q
    exp = pybamm.Experiment(["Rest for 10 minutes", f"Discharge at {i_a:.4f} A for 20 minutes", "Rest for 30 minutes"], period="5 seconds")
    s = pybamm.Simulation(model, parameter_values=PV, experiment=exp).solve(initial_soc=0.5)
    t, v, i = (s[k].entries for k in ("Time [s]", "Terminal voltage [V]", "Current [A]"))
    return t, v, i, q


def circuit(params, t, i, q, soc0=0.5):
    r0, r1, tau = params
    dt = np.diff(t, prepend=t[0])
    v1, soc, out = 0.0, soc0, np.empty_like(t)
    for k in range(len(t)):
        if k:
            a = np.exp(-dt[k] / tau)
            v1 = v1 * a + i[k - 1] * r1 * (1 - a)
            soc -= i[k - 1] * dt[k] / 3600.0 / q
        out[k] = float(ocv(soc)) - v1 - i[k] * r0
    return out


def main():
    t, v, i, q = run_dfn()
    k0 = int(np.where(i > 1e-6)[0][0])
    v_rest = v[k0 - 1]
    mask = t >= t[k0 - 1]
    tt, vv, ii = t[mask] - t[mask][0], v[mask], i[mask]

    def resid(p):
        sim = circuit(p, tt, ii, q)
        return (sim - sim[0]) - (vv - vv[0])

    fit = least_squares(resid, x0=[0.02, 0.01, 200.0], bounds=([0.001, 0.0005, 5.0], [0.1, 0.1, 5000.0]))
    r0, r1, tau = fit.x
    rms = float(np.sqrt(np.mean(fit.fun ** 2))) * 1e3
    r30 = r0 + r1 * (1 - np.exp(-30.0 / tau))
    scale = CELL.r_dc_mohm * 1e-3 / r30
    print(f"fit to DFN: R0 = {r0 * 1e3:.2f} mOhm, R1 = {r1 * 1e3:.2f} mOhm, tau = {tau:.0f} s, rms error {rms:.1f} mV")
    print(f"DFN 30 s resistance from the fit: {r30 * 1e3:.1f} mOhm; scale to datasheet {CELL.r_dc_mohm:g} mOhm: x{scale:.3f}")
    print(f"Cell fields: r0_mohm = {r0 * scale * 1e3:.2f}, r1_mohm = {r1 * scale * 1e3:.2f}, tau_s = {tau:.0f}")
    print(f"0.5 s resistance after scaling: {(r0 + r1 * (1 - np.exp(-0.5 / tau))) * scale * 1e3:.1f} mOhm (chart: 21 to 25)")


if __name__ == "__main__":
    main()
