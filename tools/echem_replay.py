"""Replay the platform's own cell current through the Chen2020 DFN and compare.

The circuit simulation is run at 25 C ambient (the electrochemical set has no temperature dependence). The charge-phase current of
the cell that carries the most current is applied to the DFN. Outputs: terminal voltage of the circuit and of the DFN,
and the DFN negative-electrode potential, which the circuit cannot produce.

The DFN is about 28 % more resistive than the datasheet cell (38 vs 30 mOhm at 30 s), so a voltage offset in the CC phase
is expected. It shows the size of the model gap, it is not an error in the replay.

Needs PyBaMM (requirements-extra.txt).
    python tools/echem_replay.py
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from bms import design as D, echem, sim  # noqa: E402


def main():
    d = D.default_designs()[2]
    rows = []
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for c_chg, col in ((0.6, "tab:blue"), (0.8, "tab:red")):
        r = sim.simulate(d, D.Scenario(t_amb=25.0, c_chg=c_chg, n_cycles=1))
        idx = np.where(r.phase == 2)[0]
        k = int(np.argmax(np.abs(r.i_cells[idx]).max(axis=0)))      # cell with the largest charge current
        g = k // d.n_parallel
        t = (r.t_min[idx] - r.t_min[idx][0]) * 60.0
        i_cell = r.i_cells[idx, k]
        soc0 = float(r.soc[idx][0])
        t0 = time.time()
        out = echem.replay(t, i_cell, soc0, model="DFN")
        dt = time.time() - t0
        # the DFN stops at its own 4.2 V event, so compare on the common time span
        m = t <= out["t_s"][-1]
        tm = t[m]
        v_c = r.v_groups[idx, g][m]
        v_d = np.interp(tm, out["t_s"], out["v"])
        phi = np.interp(tm, out["t_s"], out["phi_neg"]) * 1e3
        i_m = i_cell[m]
        cc = np.abs(i_m) > 0.95 * np.abs(i_m).max()
        lim_c = echem.plating_limit_c(r.soc[idx][m]) * 4.85
        rows.append({"Charge rate": f"{c_chg}C", "Cell": f"{k} (group {g})", "Peak cell current A": round(float(np.abs(i_cell).max()), 2),
                     "Voltage offset circuit - DFN, CC phase mV": round(float(np.mean(v_c[cc] - v_d[cc])) * 1e3, 0),
                     "Voltage RMS difference mV": round(float(np.sqrt(np.mean((v_c - v_d) ** 2))) * 1e3, 0),
                     "DFN min anode potential mV": round(float(phi.min()), 1),
                     "Time below 0 mV, min": round(float((phi < 0).sum() * (tm[1] - tm[0]) / 60.0), 1),
                     "Platform ratio to plating-free current %": round(float(100 * (np.abs(i_m) / lim_c).max()), 0),
                     "DFN replay time s": round(dt, 1)})
        ax[0].plot(tm / 60, v_c, color=col, lw=1.4, label=f"circuit {c_chg}C")
        ax[0].plot(tm / 60, v_d, color=col, lw=1.4, ls="--", label=f"DFN {c_chg}C")
        ax[1].plot(tm / 60, phi, color=col, lw=1.4, label=f"{c_chg}C")
    ax[0].set_xlabel("Time (min)"); ax[0].set_ylabel("Cell terminal voltage (V)"); ax[0].legend(frameon=False, fontsize=8)
    ax[0].set_title("Same cell current, two models (25 C)", fontsize=10)
    ax[1].axhline(0, color="k", lw=0.8); ax[1].set_xlabel("Time (min)"); ax[1].set_ylabel("Anode potential vs Li (mV)")
    ax[1].set_title("DFN: below 0 mV, plating is possible", fontsize=10); ax[1].legend(frameon=False, fontsize=8)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out_dir = ROOT / "results"
    fig.savefig(out_dir / "fig_echem_replay.png", dpi=170)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "echem_replay.csv", index=False)
    print(df.T.to_string())


if __name__ == "__main__":
    main()
