"""Cross-check the equivalent-circuit cell against the PyBaMM Chen2020 (LG M50) DFN model, 25 C only.

What this is: the Chen2020 set is a research parameterisation of an LG M50 cell, not an LG datasheet model. It has no
temperature dependence (activation energies and entropic coefficients are zero), so it cannot say anything about
cold or hot behaviour. Its 30 s DC resistance at 25 C and 50 % SOC, measured from the rest voltage as in DS 4.3.2,
is about 38 mOhm against 30 +/- 6 mOhm in the datasheet, so it is slightly more resistive than the datasheet cell.
The equivalent circuit is therefore also run with R set to the DFN value, to separate structure from resistance.

Needs PyBaMM (requirements-extra.txt).
    python tools/pybamm_crosscheck.py          compare and write results/crosscheck_pybamm.csv

The polarisation branch (R0, R1, tau) is fitted to a 0.5C pulse only (tools/fit_polarisation.py). The rate and charge
rows below use nothing from that fit, so they are independent checks.
"""
import sys
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pybamm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bms.cell import CELL  # noqa: E402
from bms.validate import charge_time_min, dcir_mohm, discharge_curve  # noqa: E402

warnings.filterwarnings("ignore")
PV = pybamm.ParameterValues("Chen2020")


def _sim(steps, soc, period="10 seconds"):
    model = pybamm.lithium_ion.DFN(options={"thermal": "isothermal"})
    return pybamm.Simulation(model, parameter_values=PV, experiment=pybamm.Experiment(steps, period=period)).solve(initial_soc=soc)


def _arr(sol):
    return (sol[k].entries for k in ("Time [s]", "Terminal voltage [V]", "Current [A]"))


CAP_DFN = None


def cap_dfn():
    global CAP_DFN
    if CAP_DFN is None:
        t, v, i = _arr(_sim(["Discharge at 1.0 A for 8 hours or until 2.5 V"], 1.0, "30 seconds"))
        CAP_DFN = float(np.trapezoid(i, t) / 3600.0)
    return CAP_DFN


def dfn_discharge(c_rate):
    t, v, i = _arr(_sim([f"Discharge at {c_rate * cap_dfn():.4f} A for 8 hours or until 2.5 V"], 1.0, "10 seconds"))
    q = float(np.trapezoid(i, t) / 3600.0)
    return q, float(np.trapezoid(v * i, t) / 3600.0)


def dfn_charge_min(c_rate):
    i_a = c_rate * cap_dfn()
    end = (CELL.i_term_std / CELL.cap_ah) if c_rate <= CELL.i_chg_std / CELL.cap_ah + 1e-9 else CELL.i_term_fast_c
    steps = [(f"Charge at {i_a:.4f} A until 4.2 V", f"Hold at 4.2 V until {end * cap_dfn():.4f} A")]
    t, v, i = _arr(_sim(steps, 0.0))
    return float(t[-1] / 60.0)


def dfn_dcir_mohm(t_s=30.0):
    """DS 4.3.2 procedure: rest, then 0.5C for 30 s from 50 % SOC. Resistance = (rest voltage - voltage at t_s) / current."""
    i_a = 0.5 * cap_dfn()
    t, v, i = _arr(_sim(["Rest for 10 minutes", f"Discharge at {i_a:.4f} A for 30 seconds"], 0.5, "0.5 seconds"))
    k0 = int(np.where(i > 1e-6)[0][0])
    j = int(np.argmin(np.abs(t - (t[k0] + t_s - 0.5))))
    return float((v[k0 - 1] - v[j]) / i_a * 1e3)


def main():
    r30, r05 = dfn_dcir_mohm(30.0), dfn_dcir_mohm(0.5)
    k = r30 / dcir_mohm(30.0)
    like = replace(CELL, r0_mohm=CELL.r0_mohm * k, r1_mohm=CELL.r1_mohm * k)
    rows = []
    rows.append(("DC resistance, 0.5C, 30 s, 50 % SOC", "25 C", f"{r30:.1f} mOhm", f"{dcir_mohm(30.0):.1f} mOhm",
                 f"{dcir_mohm(30.0) - r30:+.1f} mOhm", "-", "DFN is above the datasheet 30 +/- 6 mOhm; circuit is set to the datasheet"))
    rows.append(("DC resistance, 0.5C, 0.5 s, 50 % SOC", "25 C", f"{r05:.1f} mOhm", f"{dcir_mohm(0.5):.1f} mOhm",
                 f"{dcir_mohm(0.5) - r05:+.1f} mOhm", "-", "measured cells: 21 to 25 mOhm (BD chart)"))
    for c in (0.2, 0.5, 1.0, 1.5):
        q, e = dfn_discharge(c)
        qm, em = discharge_curve(c * CELL.cap_ah, 25.0)
        ql, el = discharge_curve(c * like.cap_ah, 25.0, like)
        if c == 0.2:
            q_ref, e_ref, qm_ref, em_ref, ql_ref, el_ref = q, e, qm, em, ql, el
        rows.append(("Capacity vs 0.2C", f"{c}C, 25 C", f"{100 * q / q_ref:.1f} %", f"{100 * qm / qm_ref:.1f} %",
                     f"{100 * qm / qm_ref - 100 * q / q_ref:+.1f}", f"{100 * ql / ql_ref:.1f} %", "independent"))
        rows.append(("Energy vs 0.2C", f"{c}C, 25 C", f"{100 * e / e_ref:.1f} %", f"{100 * em / em_ref:.1f} %",
                     f"{100 * em / em_ref - 100 * e / e_ref:+.1f}", f"{100 * el / el_ref:.1f} %", "independent"))
    for c, kind in ((0.3, "independent, 50 mA end current"), (0.5, "independent"), (0.6, "independent"),
                    (0.7, "independent, C/20 end current")):
        td = dfn_charge_min(c)
        tm = charge_time_min(c, 25.0)
        tl = charge_time_min(c, 25.0, like)
        # compare in C-hours so the 5.12 Ah DFN and the 4.85 Ah rated cell are on the same basis
        rows.append(("Charge time to end current, in C-hours", f"{c}C CC-CV, 25 C", f"{td / 60 * c:.3f} h x C",
                     f"{tm / 60 * c:.3f} h x C", f"{(tm - td) / td * 100:+.0f} %", f"{tl / 60 * c:.3f} h x C ({(tl - td) / td * 100:+.0f} %)", kind))
    df = pd.DataFrame(rows, columns=["Test", "Condition", "Reference (DFN)", "Equivalent circuit (R = 30 mOhm)", "Difference",
                                     "Equivalent circuit (R = DFN value)", "Type"])
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "crosscheck_pybamm.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
