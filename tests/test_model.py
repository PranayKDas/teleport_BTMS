"""Model self-checks. Run:  python -m pytest tests    or    python tests/test_model.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bms import design as D  # noqa: E402
from bms import echem  # noqa: E402
from bms.cell import CELL, i_chg_limit, i_dis_limit, i_term, ocv, r_temp_factor  # noqa: E402
from bms.checks import rank, run_checks  # noqa: E402
from bms.sim import simulate  # noqa: E402
from bms.validate import charge_time_min, dcir_mohm, discharge_curve  # noqa: E402

SC = D.Scenario()


def test_energy_balance():
    """Terminal energy = OCV energy - heat - change in stored polarisation energy, over the whole run."""
    for d in D.default_designs():
        r = simulate(d, SC)
        voc = np.array([float(ocv(x)) for x in r.soc])
        p_ocv = d.n_series * voc * r.i_mod
        p_term = r.v_mod * r.i_mod
        lhs = float(np.sum(p_ocv - p_term) * r.dt / 3600.0)           # Wh lost between OCV and terminals
        rhs = float(np.sum(r.heat_w) * r.dt / 3600.0 + (r.e_pol_wh[-1] - r.e_pol_wh[0]))
        assert abs(lhs - rhs) < 0.03 * abs(lhs) + 0.05, (d.name, lhs, rhs)


def test_soc_and_voltage_bounds():
    for d in D.default_designs():
        r = simulate(d, SC)
        assert r.soc.min() >= 0.0 and r.soc.max() <= 1.0
        assert r.v_cell_min[r.phase != 1].min() >= CELL.v_min - 1e-3
        assert r.v_mod.max() <= d.n_series * CELL.v_max + d.n_series * 0.05


def test_current_sharing_sums_to_module_current():
    d = D.default_designs()[0]
    r = simulate(d, D.Scenario(n_cycles=1))
    # peak single-cell current cannot be below the mean share and cannot exceed the whole group current
    dis = r.phase == 0
    mean_share = r.i_mod[dis].max() / d.n_parallel
    assert r.i_cell_max[dis].max() >= mean_share - 1e-9
    assert r.i_cell_max[dis].max() <= r.i_mod[dis].max()


def test_repeatable_and_seed_sensitive():
    d = D.default_designs()[0]
    a = simulate(d, SC).ind["peak_cell_temp_c"]
    b = simulate(d, SC).ind["peak_cell_temp_c"]
    c = simulate(d, D.Scenario(seed=8)).ind["peak_cell_temp_c"]
    assert a == b and a != c


def test_no_spread_gives_symmetric_cells():
    d = D.default_designs()[1]                      # plate: strongly coupled
    r = simulate(d, D.Scenario(r_sigma=0.0, n_cycles=1))
    t = r.temps[:, : r.n_cells][-1].reshape(d.n_series, d.n_parallel)
    assert np.ptp(t[:, 0]) < 1.0                    # same temperature within a column up to edge effects


def test_datasheet_anchors():
    assert abs(CELL.cap_ah * 0.2 - 0.97) < 1e-9 and abs(CELL.cap_ah * 0.7 - 3.395) < 1e-9
    assert abs(CELL.cap_ah * 1.5 - 7.275) < 1e-9
    assert 24.0 <= dcir_mohm(30.0) <= 36.0           # DS 4.3.2: 30 +/- 6 mOhm
    q, e = discharge_curve(0.2 * CELL.cap_ah)
    assert 17.6 <= e <= 18.4                         # DS 2.1: 17.6 Wh min, 18.2 Wh nominal
    assert i_term(CELL, 1.455) == 0.05 and abs(i_term(CELL, 3.0) - 0.2425) < 1e-6


def test_limit_functions_follow_datasheet():
    assert float(i_chg_limit(CELL, 20.0)) == CELL.i_chg_cold and float(i_chg_limit(CELL, 30.0)) == CELL.i_chg_max
    assert float(i_chg_limit(CELL, -1.0)) == 0.0 and float(i_chg_limit(CELL, 46.0)) == 0.0
    assert float(i_dis_limit(CELL, 0.0)) == CELL.i_dis_cold and float(i_dis_limit(CELL, 30.0)) == CELL.i_dis_max


def test_resistance_vs_temperature_is_monotonic_and_anchored():
    t = np.linspace(-10, 60, 71)
    f = r_temp_factor(t)
    assert np.all(np.diff(f) <= 1e-12) and abs(float(r_temp_factor(25.0)) - 1.0) < 1e-12


def test_charge_time_trends():
    assert charge_time_min(0.3) > charge_time_min(0.7)
    assert 80 < charge_time_min(0.7) < 160


def test_ranking_orders_default_designs():
    res = [simulate(d, SC) for d in D.default_designs()]
    rk = rank(res, [run_checks(r, D.Requirements()) for r in res])
    assert rk.Design.iloc[0].startswith("C") and rk.Design.iloc[-1].startswith("A")


def test_no_charge_current_while_a_cell_is_below_zero():
    d = D.default_designs()[0]
    r = simulate(d, D.Scenario(t_amb=-5.0, n_cycles=1, c_dis=0.25), max_phase_h=1.0)
    charging = r.i_mod < 0
    if charging.any():
        assert r.temps[charging][:, : r.n_cells].min() >= CELL.t_chg_min - 1e-9


def test_plating_limit_shape_and_datasheet_agreement():
    """Electrochemical map: limit falls with SOC, is the 1.5C cap when empty, and sits near the datasheet 0.7C at high SOC."""
    soc = np.linspace(0.0, 0.95, 20)
    lim = echem.plating_limit_c(soc)
    assert np.all(np.diff(lim) <= 1e-9)
    assert lim[0] >= 1.49
    assert 0.6 < float(echem.plating_limit_c([0.9])[0]) < 0.8
    assert echem.min_potential_mv(0.5) > echem.min_potential_mv(0.8)


def test_plating_check_passes_at_default_and_fails_at_high_rate():
    d = D.default_designs()[2]
    ok = run_checks(simulate(d, D.Scenario(c_chg=0.6)), D.Requirements())
    bad = run_checks(simulate(d, D.Scenario(c_chg=0.8)), D.Requirements())
    assert ok[ok.ID == "E1"].Status.iloc[0] == "PASS"
    assert bad[bad.ID == "E1"].Status.iloc[0] == "FAIL"


def test_plating_ratio_ignores_cold_cells():
    """The map is a 25 C map: with every cell below 25 C the ratio is not evaluated and the time is reported instead."""
    d = D.default_designs()[0]
    r = simulate(d, D.Scenario(t_amb=5.0, n_cycles=1, c_chg=0.3, c_dis=0.25))
    assert r.ind["peak_plating_ratio"] == 0.0
    assert r.ind["plating_unrated_s"] > 0.0


# ---------------------------------------------------------------- liquid cooling (Design D)
# Reference values of designs A, B and C, recorded from the delivered project before Design D was added.
ABC_REFERENCE = {
    "A": dict(peak_cell_temp_c=57.347397, usable_energy_wh=202.254647, charge_time_min=136.5, mass_g=878.0,
              cost_inr=6226.0, volume_cm3=429.957768, temp_spread_k=4.669368, round_trip_eff=0.946208),
    "B": dict(peak_cell_temp_c=47.723137, usable_energy_wh=203.578611, charge_time_min=103.666667, mass_g=951.5,
              cost_inr=6586.25, volume_cm3=467.457768, temp_spread_k=0.21187, round_trip_eff=0.945365),
    "C": dict(peak_cell_temp_c=41.777699, usable_energy_wh=203.9238, charge_time_min=105.25, mass_g=890.0,
              cost_inr=6376.0, volume_cm3=550.342418, temp_spread_k=0.816488, round_trip_eff=0.941823),
}
ABC_PEAK_AMBIENT = {25.0: dict(A=48.926463, B=39.687146, C=33.192751), 45.0: dict(A=62.799713, B=57.356709, C=51.282297)}


def _d():
    return D.default_designs()[3]


def test_four_designs_and_d_is_liquid():
    ds = D.default_designs()
    assert [d.name.split()[0] for d in ds] == ["A", "B", "C", "D"]
    assert ds[3].liquid and not any(d.liquid for d in ds[:3])


def test_designs_a_b_c_are_unchanged():
    """Adding the liquid loop must not move A, B or C (default case and two other ambients)."""
    res = {d.name.split()[0]: simulate(d, SC) for d in D.default_designs()[:3]}
    for tag, ref in ABC_REFERENCE.items():
        for k, v in ref.items():
            assert abs(res[tag].ind[k] - v) < 1e-5, (tag, k, res[tag].ind[k], v)
    for t, refs in ABC_PEAK_AMBIENT.items():
        for d in D.default_designs()[:3]:
            tag = d.name.split()[0]
            assert abs(simulate(d, D.Scenario(t_amb=t)).ind["peak_cell_temp_c"] - refs[tag]) < 1e-5


def test_liquid_heat_balance():
    """Heat made in cells and bus plus pump heat = heat stored in plate, coolant and cells + heat sent to ambient (air and radiator)."""
    from bms.sim import PUMP_HEAT_FRACTION, build_thermal
    for t_amb in (25.0, 35.0, 45.0):
        d = _d()
        r = simulate(d, D.Scenario(t_amb=t_amb, n_cycles=2))
        _, _, C = build_thermal(d)
        gen = (r.heat_w.sum() + PUMP_HEAT_FRACTION * r.pump_w.sum()) * r.dt / 3600.0
        stored = float((C * (r.temps[-1] - r.temps[0])).sum()) / 3600.0
        out = (r.q_air_w.sum() + r.q_rad_w.sum()) * r.dt / 3600.0
        assert abs(gen - (out + stored)) < 0.01 * gen + 0.05, (t_amb, gen, out, stored)


def test_coolant_loop_energy_balance_is_closed_at_every_step():
    """Plate heat into the coolant minus radiator heat (plus pump heat) equals the coolant heating rate."""
    d = _d()
    r = simulate(d, D.Scenario(n_cycles=1))
    from bms.sim import PUMP_HEAT_FRACTION, build_thermal
    _, _, C = build_thermal(d)
    n = r.n_cells
    c_cool = C[n + 1]
    dTdt = np.diff(r.temps[:, n + 1]) / r.dt                       # row k is the state before step k, so this is the step-k rate
    q_net = (r.q_plate_w - r.q_rad_w + PUMP_HEAT_FRACTION * r.pump_w)[1:]
    # the pump switches on and off at phase boundaries, where the logged flux and the logged state belong to different pump states
    mask = np.ones(len(dTdt), dtype=bool)
    for t in np.where(np.diff(r.pump_w) != 0)[0]:
        mask[max(t - 2, 0): t + 4] = False
    assert mask.sum() > 0.9 * len(mask)
    assert np.abs((c_cool * dTdt - q_net)[mask]).max() < 1e-6      # exact for backward Euler away from those steps
    assert abs(float((c_cool * dTdt).sum() - q_net.sum())) * r.dt / 3600.0 < 0.01     # and the whole-run total closes to 0.01 Wh


def test_thermal_network_is_stable():
    """All eigenvalues of C^-1 K have positive real part (decaying modes), with the pump running, stopped and with a weak flow."""
    from bms.sim import build_thermal
    for flow in (1.0, 0.25, 0.0):
        K, _, C = build_thermal(_d(), flow_scale=flow)
        ev = np.linalg.eigvals(np.diag(1.0 / C) @ K)
        # no growing mode. With the pump stopped the coolant is adiabatic and decoupled, which gives one zero eigenvalue
        assert ev.real.min() > (1e-9 if flow > 0 else -1e-9), (flow, ev.real.min())


def test_step_size_does_not_change_the_answer():
    d = _d()
    peaks = [simulate(d, D.Scenario(n_cycles=1), dt=dt).ind["peak_cell_temp_c"] for dt in (2.0, 5.0, 20.0)]
    assert max(peaks) - min(peaks) < 0.6, peaks


def test_coolant_never_below_ambient():
    """No refrigeration: coolant, plate inlet and plate stay at or above ambient (small numerical tolerance)."""
    for t_amb in (15.0, 25.0, 45.0):
        r = simulate(_d(), D.Scenario(t_amb=t_amb, n_cycles=2))
        assert np.nanmin(r.t_cool) >= t_amb - 0.05
        assert np.nanmin(r.t_cool_in) >= t_amb - 0.05
    r = simulate(replace_d(coolant_t_init_c=5.0), D.Scenario(t_amb=35.0, n_cycles=1))
    assert abs(r.t_cool[0] - 35.0) < 0.5          # a requested start below ambient is raised to ambient


def replace_d(**kw):
    from dataclasses import replace
    return replace(_d(), **kw)


def test_liquid_parameters_change_the_temperature():
    sc = D.Scenario(n_cycles=1)
    base = simulate(_d(), sc).ind["peak_cell_temp_c"]
    assert simulate(replace_d(rad_ua_w_k=4.0), sc).ind["peak_cell_temp_c"] > base + 0.3        # smaller radiator
    assert simulate(replace_d(rad_ua_w_k=16.0), sc).ind["peak_cell_temp_c"] < base - 0.1       # larger radiator
    assert simulate(replace_d(ua_plate_w_k=3.0), sc).ind["peak_cell_temp_c"] > base + 0.3      # weaker plate to coolant
    assert simulate(replace_d(pad_k=0.7), sc).ind["peak_cell_temp_c"] > base + 0.5             # worse pad
    assert simulate(replace_d(pad_t_mm=2.0), sc).ind["peak_cell_temp_c"] > base + 0.5          # thicker pad
    assert simulate(replace_d(rad_fan=False), sc).ind["peak_cell_temp_c"] > base + 0.5         # radiator fan off
    # flow barely matters while the loop is oversized (30 g/s moves the coolant by 0.08 K), then bites at very low flow
    assert simulate(replace_d(coolant_mdot_kg_s=0.0005), sc).ind["peak_cell_temp_c"] > base + 0.5
    big = simulate(replace_d(coolant_vol_l=1.5), sc)                                            # more coolant thermal mass
    assert big.t_cool[int(len(big.t_cool) * 0.1)] < simulate(_d(), sc).t_cool[int(len(big.t_cool) * 0.1)]


def test_pump_failure_and_zero_flow():
    sc = D.Scenario(n_cycles=1)
    base = simulate(_d(), sc).ind["peak_cell_temp_c"]
    dead = simulate(_d(), D.Scenario(n_cycles=1, pump_flow_frac=0.0))
    assert np.abs(dead.q_plate_w).max() < 1e-6 and dead.q_rad_w.max() < 1e-6          # no heat reaches the coolant
    assert dead.ind["peak_cell_temp_c"] > base + 5.0
    late = simulate(_d(), D.Scenario(n_cycles=1, pump_fail_min=10.0))
    assert base + 3.0 < late.ind["peak_cell_temp_c"] <= dead.ind["peak_cell_temp_c"] + 1e-6


def test_bms_trips_when_the_pump_fails_in_a_hot_run():
    """Pump stops at 5 min, 45 C ambient, 1.5C: the existing 60 C discharge trip must act."""
    r = simulate(_d(), D.Scenario(t_amb=45.0, c_dis=1.5, n_cycles=1, pump_fail_min=5.0))
    assert r.ind["bms_trip_time_s"] > 0.0
    dis = r.phase == 0
    assert r.temps[dis][:, : r.n_cells].max() < 61.5            # held near the 60 C trip level


def test_auxiliary_energy_is_counted():
    r = simulate(_d(), SC)
    d = _d()
    assert r.ind["pump_energy_wh"] > 0 and r.ind["rad_fan_energy_wh"] > 0
    assert abs(r.ind["aux_energy_wh"] - (r.ind["pump_energy_wh"] + r.ind["rad_fan_energy_wh"])) < 1e-9
    # net usable energy = delivered - pump and radiator fan energy during the first discharge phase
    t_dis_h = r.ind["discharge_time_min"] / 60.0
    aux_dis = (d.pump_power_w + d.rad_fan_power_w) * t_dis_h
    assert abs(r.ind["usable_energy_wh"] - (r.ind["delivered_wh"] - aux_dis)) < 0.05
    big = simulate(replace_d(pump_power_w=20.0), SC)
    assert big.ind["usable_energy_wh"] < r.ind["usable_energy_wh"] - 3.0
    assert simulate(D.default_designs()[0], SC).ind["aux_energy_wh"] == 0.0


def test_liquid_checks_exist_only_for_the_liquid_design():
    res = [simulate(d, SC) for d in D.default_designs()]
    chk = [run_checks(r, D.Requirements()) for r in res]
    for c in chk[:3]:
        assert not (c.Type == "Liquid loop").any()
    ids = set(chk[3][chk[3].Type == "Liquid loop"].ID)
    assert ids == {"LQ1", "LQ2", "LQ3", "LQ4"}
    from dataclasses import replace
    bad = run_checks(simulate(replace(_d(), pad_dielectric=False), SC), D.Requirements())
    assert bad[bad.ID == "LQ4"].Status.iloc[0] == "FAIL"
    weak = run_checks(simulate(replace(_d(), coolant_mdot_kg_s=0.002), SC), D.Requirements())
    assert weak[weak.ID == "LQ1"].Value.astype(float).iloc[0] > chk[3][chk[3].ID == "LQ1"].Value.astype(float).iloc[0]


def test_radiator_cannot_cool_below_ambient():
    """LQ2 falls to zero when ambient reaches the requirement temperature, and the model shows the cells above it."""
    r = simulate(_d(), D.Scenario(t_amb=45.0, n_cycles=1))
    c = run_checks(r, D.Requirements())
    assert c[c.ID == "LQ2"].Status.iloc[0] == "FAIL"
    assert r.ind["peak_cell_temp_c"] > 45.0
    r25 = simulate(_d(), D.Scenario(t_amb=25.0, n_cycles=1))
    assert run_checks(r25, D.Requirements()).query("ID == 'LQ2'").Status.iloc[0] == "PASS"


def test_ranking_uses_simulated_results_and_includes_d():
    res = [simulate(d, SC) for d in D.default_designs()]
    chk = [run_checks(r, D.Requirements()) for r in res]
    rk = rank(res, chk)
    assert len(rk) == 4 and any(x.startswith("D") for x in rk.Design)
    assert not rk.Design.iloc[0].startswith("D")                       # liquid cooling is not declared the winner by default
    # with every weight on temperature the coldest design scores best, and that is D in this case
    w = {"peak_cell_temp_c": 1.0, "usable_energy_wh": 0.0, "charge_time_min": 0.0, "mass_g": 0.0, "cost_inr": 0.0, "volume_cm3": 0.0}
    sc_only = rank(res, chk, w)
    coldest = min(res, key=lambda r: r.ind["peak_cell_temp_c"]).design.name
    assert sc_only.sort_values("Score", ascending=False).Design.iloc[0] == coldest and coldest.startswith("D")


def test_repeated_cycles_do_not_drift_for_the_liquid_design():
    from bms.study import _cycle_peaks
    r = simulate(_d(), D.Scenario(n_cycles=3, t_amb=35.0))
    pk = _cycle_peaks(r)
    assert len(pk) == 3 and max(pk) - min(pk) < 0.3


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok  ", fn.__name__)
    print(f"{len(fns)} tests passed")
