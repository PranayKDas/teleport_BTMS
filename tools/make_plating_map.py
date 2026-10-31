"""Build data/plating_map.csv: anode potential during constant-current charge, from the PyBaMM Chen2020 model.

For each charge current the cell is charged from empty at 25 C with the electrochemical model (DFN, and the faster SPMe
for comparison) until 4.2 V. The potential of the negative electrode against lithium metal, at the separator side, is
recorded against state of charge. A potential below 0 V means lithium plating is thermodynamically possible. This is the
usual screening indicator. It does not say how much lithium plates, and the Chen2020 set has no temperature dependence,
so the map is valid for 25 C. Currents are in datasheet C-rate (1C = 4.85 A).

Needs PyBaMM (requirements-extra.txt).
    python tools/make_plating_map.py
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pybamm

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
PV = pybamm.ParameterValues("Chen2020")
CAP_RATED = 4.85
RATES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.5]
KEY = "Negative electrode surface potential difference at separator interface [V]"


def solve(model_name, i_a, soc0):
    m = getattr(pybamm.lithium_ion, model_name)(options={"thermal": "isothermal"})
    exp = pybamm.Experiment([f"Charge at {i_a:.4f} A for 12 hours or until 4.2 V"], period="10 seconds")
    return pybamm.Simulation(m, parameter_values=PV, experiment=exp).solve(initial_soc=soc0)


def main():
    s = solve("DFN", 0.2 * CAP_RATED, 0.0)   # total charge accepted from empty to 4.2 V, used as the SOC scale
    t = s["Time [s]"].entries
    cap = float(np.trapezoid(np.abs(s["Current [A]"].entries), t) / 3600.0)
    rows = []
    for model in ("DFN", "SPMe"):
        for c in RATES:
            sol = solve(model, c * CAP_RATED, 0.0)
            t = sol["Time [s]"].entries
            i = np.abs(sol["Current [A]"].entries)
            soc = np.concatenate([[0.0], np.cumsum(0.5 * (i[1:] + i[:-1]) * np.diff(t)) / 3600.0 / cap])
            phi = sol[KEY].entries
            for a, b in zip(soc, phi):
                rows.append((model, c, round(float(a), 4), round(float(b) * 1e3, 2)))
    df = pd.DataFrame(rows, columns=["model", "c_rate", "soc", "phi_neg_mv"])
    df.to_csv(ROOT / "data" / "plating_map.csv", index=False)
    print(f"capacity on this scale {cap:.3f} Ah, {len(df)} rows")
    for model in ("DFN", "SPMe"):
        for c in RATES:
            d = df[(df.model == model) & (df.c_rate == c)]
            k = d.phi_neg_mv.idxmin()
            print(f"{model} {c}C: reaches SOC {d.soc.max():.2f}, min potential {d.phi_neg_mv.min():7.1f} mV at SOC {d.soc[k]:.2f}")


if __name__ == "__main__":
    main()
