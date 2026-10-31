"""One-at-a-time sensitivity of peak temperature and design ranking to the uncertain inputs.

The uncertain inputs are the ones no source pins down: convection coefficients, gap exposure, cell specific heat,
thermal pad conductivity, the resistance and spread of the real cells (datasheet tolerance is +/- 20 %), and for the
liquid design the radiator and plate conductances, the coolant flow and the placeholder hardware cost.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace

import pandas as pd

from . import design as D
from . import sim
from .cell import CELL
from .checks import rank, run_checks


@contextmanager
def _patched(**kw):
    old = {k: getattr(sim, k) for k in kw}
    try:
        for k, v in kw.items():
            setattr(sim, k, v)
        yield
    finally:
        for k, v in old.items():
            setattr(sim, k, v)


def _scaled_cell(k):
    return replace(CELL, r0_mohm=CELL.r0_mohm * k, r1_mohm=CELL.r1_mohm * k)


def cases():
    """(factor, low label, high label, builder) where builder(level 0/1) -> (designs_fn, scenario_kw, cell, patches)."""
    return [
        ("Natural convection h, +/- 30 %", "7", "13 W/m2K", lambda lv: dict(design=lambda d: replace(d, h_nat=d.h_nat * (0.7, 1.3)[lv]))),
        ("Gap exposure length, 2 / 4.5 mm", "2", "4.5 mm", lambda lv: dict(patch=dict(GAP_SCALE_MM=(2.0, 4.5)[lv]))),
        ("Forced convection coefficient, +/- 30 %", "-30 %", "+30 %", lambda lv: dict(patch=dict(FORCED_K=sim.FORCED_K * (0.7, 1.3)[lv]))),
        ("Cell specific heat, 800 / 1100", "800", "1100 J/kgK", lambda lv: dict(cell=replace(CELL, cp_j_kgk=(800.0, 1100.0)[lv]))),
        ("Cell resistance, +/- 20 % (DS tolerance)", "-20 %", "+20 %", lambda lv: dict(cell=_scaled_cell((0.8, 1.2)[lv]))),
        ("Resistance spread, 0 / 10 %", "0 %", "10 %", lambda lv: dict(scenario=dict(r_sigma=(0.0, 0.10)[lv]))),
        ("Ambient temperature, 30 / 40 C", "30 C", "40 C", lambda lv: dict(scenario=dict(t_amb=(30.0, 40.0)[lv]))),
        ("Thermal pad conductivity, 1.0 / 2.5", "1.0", "2.5 W/mK", lambda lv: dict(design=lambda d: replace(d, pad_k=(1.0, 2.5)[lv]))),
        # liquid loop (Design D only: the fields are unused by the air-cooled designs, so A, B and C do not move)
        ("Radiator UA, +/- 30 %", "-30 %", "+30 %", lambda lv: dict(design=lambda d: replace(d, rad_ua_w_k=d.rad_ua_w_k * (0.7, 1.3)[lv]))),
        ("Plate-to-coolant UA, +/- 30 %", "-30 %", "+30 %", lambda lv: dict(design=lambda d: replace(d, ua_plate_w_k=d.ua_plate_w_k * (0.7, 1.3)[lv]))),
        ("Coolant flow, 50 / 150 %", "50 %", "150 %", lambda lv: dict(design=lambda d: replace(d, coolant_mdot_kg_s=d.coolant_mdot_kg_s * (0.5, 1.5)[lv]))),
        ("Liquid loop hardware cost, 50 / 200 %", "50 %", "200 %", lambda lv: dict(design=lambda d: replace(d, liq_hw_cost_inr=d.liq_hw_cost_inr * (0.5, 2.0)[lv]))),
    ]


def run(designs=None, sc: D.Scenario | None = None, req: D.Requirements | None = None):
    designs = designs or D.default_designs()
    sc = sc or D.Scenario()
    req = req or D.Requirements()
    rows = []

    def evaluate(label, level_label, spec):
        ds = [spec.get("design", lambda d: d)(d) for d in designs]
        s2 = replace(sc, **spec.get("scenario", {}))
        cell = spec.get("cell", CELL)
        with _patched(**spec.get("patch", {})):
            res = [sim.simulate(d, s2, cell=cell, dt=10.0) for d in ds]
        chk = [run_checks(r, req, cell) for r in res]
        rk = rank(res, chk)
        row = {"Factor": label, "Level": level_label}
        for r in res:
            row[r.design.name.split()[0] + " peak (C)"] = round(r.ind["peak_cell_temp_c"], 1)
        row["Best design"] = rk.Design.iloc[0].split()[0]
        row["Order"] = " > ".join(x.split()[0] for x in rk.Design)
        rows.append(row)

    evaluate("Base case", "-", {})
    for label, lo, hi, build in cases():
        for lv, lab in ((0, lo), (1, hi)):
            evaluate(label, lab, build(lv))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print(run().to_string(index=False))
