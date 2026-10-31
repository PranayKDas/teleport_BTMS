"""Cycle simulation: electrical + thermal + BMS protection for one module.

Model summary
- Electrical: each parallel cell is OCV(SOC) in series with R0(T) and one R1(T) || C1 branch
  (solid-state diffusion). R0 and R1 carry a seeded cell-to-cell spread. Parallel cells share
  current according to their own R0 and polarisation voltage, so a hotter or lower-resistance
  cell takes more current.
- Heat: I^2 R0 + V1^2 / R1 per cell, plus interconnect loss shared over all cells.
- Thermal: one node per cell (and one for the spreader plate), conduction through
  the air gap between neighbours, convection to ambient on the exposed sides.
  Solved with backward Euler, so it stays stable for any step size.
- BMS: voltage cut-offs, charge and discharge temperature trips with hysteresis
  (datasheet limits), sensor on a chosen cell. Current limits depend on cell temperature
  (0.3C below 25 C and 0.7C above for charge, 0.3C below 5 C and 1.5C above for discharge) and are
  reported as a ratio, not enforced, so the checker can show a charger/cell mismatch.
- Discharge also stops when the energy retention for the cell temperature (DS 4.4.4) is used up.
- Liquid cooling (Design D, optional): two more nodes in the same network, a cold plate and the circulating coolant.
  Cells conduct into the plate through thermal pads, the plate hands heat to the coolant with an effectiveness-NTU
  conductance that depends on the coolant flow, and a radiator rejects heat from the coolant to ambient air. The coolant
  can never fall below ambient (no refrigeration). Pump heat enters the coolant. Pump and radiator fan power are
  counted as auxiliary energy. Equations are in liquid_parts and build_thermal; they are engineering estimates, not
  validated against a measurement.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import design as D
from . import echem
from .cell import CELL, Cell, cap_retention, i_chg_limit, i_dis_limit, i_term, ocv, r_temp_factor

AIR_K = 0.026        # W/m/K
CP_AL = 900.0        # J/kg/K
GAP_SCALE_MM = 3.0   # gap width at which a side surface is ~63% exposed (ASSUMPTION)
FORCED_K = 12.0      # W/m2/K per (m/s)^0.6, bank-derated forced convection (ASSUMPTION)
PHASES = ("discharge", "rest", "charge", "rest")
PUMP_HEAT_FRACTION = 1.0      # share of pump electrical power that ends up as heat in the coolant (ASSUMPTION: wet-rotor pump)
RAD_NATURAL_FRACTION = 0.25   # radiator UA without its fan, as a fraction of the UA with the fan (ASSUMPTION)


def h_effective(d: D.Design) -> float:
    return d.h_nat + (FORCED_K * d.fan_v ** 0.6 if d.fan_v > 0 else 0.0)


@dataclass(frozen=True)
class LiquidParts:
    """Lumped parameters of the coolant loop at one flow rate (Design D)."""
    mcp: float       # W/K, coolant capacity rate m_dot * cp
    ua_rad: float    # W/K, radiator UA at its operating condition (fan on or off)
    eps_rad: float   # radiator effectiveness, 1 - exp(-UA / (m cp))
    g_rad: float     # W/K, radiator heat rejection conductance = m cp eps_rad
    eps_plate: float  # cold plate effectiveness, 1 - exp(-UA_plate / (m cp))
    g_plate: float   # W/K, plate to coolant inlet conductance = m cp eps_plate
    g_pad: float     # W/K, one cell to the plate = k_pad A_pad / t_pad
    g_plate_air: float  # W/K, plate underside to ambient by natural convection
    c_plate: float   # J/K
    c_cool: float    # J/K


def liquid_parts(d: D.Design, flow_scale: float = 1.0) -> LiquidParts:
    """Coolant loop conductances.

    mcp = m_dot cp is the coolant capacity rate (W/K). A heat exchanger of conductance UA between a fluid and a body at
    fixed temperature transfers m cp (1 - exp(-UA / (m cp))) per kelvin of temperature difference (effectiveness-NTU with
    the other side at constant temperature). The same expression is used for the radiator (other side: ambient air)
    and for the cold plate (other side: the plate, treated as one isothermal node). With no flow both go to zero.
    """
    mcp = d.coolant_mdot_kg_s * flow_scale * d.coolant_cp
    ua_r = d.rad_ua_w_k * (1.0 if d.rad_fan else RAD_NATURAL_FRACTION)
    if mcp > 1e-9:
        eps_r = 1.0 - float(np.exp(-ua_r / mcp))
        eps_p = 1.0 - float(np.exp(-d.ua_plate_w_k / mcp))
    else:
        eps_r = eps_p = 0.0
    return LiquidParts(
        mcp=mcp, ua_rad=ua_r, eps_rad=eps_r, g_rad=mcp * eps_r, eps_plate=eps_p, g_plate=mcp * eps_p,
        g_pad=d.pad_k * d.pad_area_cm2 * 1e-4 / (d.pad_t_mm * 1e-3),
        g_plate_air=d.h_nat * d.plate_area_cm2 * 1e-4,
        c_plate=d.plate_mass_kg * CP_AL, c_cool=d.coolant_rho * d.coolant_vol_l * 1e-3 * d.coolant_cp)


def build_thermal(d: D.Design, cell: Cell = CELL, flow_scale: float = 1.0):
    """Return conductance matrix K (W/K), the ambient forcing weights and heat capacities.

    The network obeys C dT/dt = -K T + w T_amb + q, with w the returned ambient vector.
    For the air-cooled designs w is the conductance to ambient of each node. For the liquid design the last two nodes are
    the cold plate (index n) and the coolant (index n + 1), see liquid_parts. flow_scale scales the coolant flow
    (0 = pump stopped)."""
    ns, npar = d.n_series, d.n_parallel
    n = ns * npar
    if d.liquid and d.spreader:
        raise ValueError("Design cannot have both a spreader plate and a liquid cold plate")
    size = n + (1 if d.spreader else 0) + (2 if d.liquid else 0)
    h = h_effective(d)
    a_side = np.pi * cell.dia_m * cell.len_m / 4.0
    phi = 1.0 - np.exp(-d.spacing_mm / GAP_SCALE_MM)
    gap = max(d.spacing_mm, 0.5) / 1000.0
    g_nb = AIR_K * cell.len_m * 0.4 * cell.dia_m / gap

    K = np.zeros((size, size))
    g_amb = np.zeros(size)
    for g in range(ns):
        for i in range(npar):
            k = g * npar + i
            nb = 0
            for dg, di in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                g2, i2 = g + dg, i + di
                if 0 <= g2 < ns and 0 <= i2 < npar:
                    nb += 1
                    if dg > 0 or di > 0:
                        j = g2 * npar + i2
                        K[k, k] += g_nb
                        K[j, j] += g_nb
                        K[k, j] -= g_nb
                        K[j, k] -= g_nb
            g_amb[k] = h * a_side * ((4 - nb) + phi * nb)
    C = np.full(size, cell.mass_kg * cell.cp_j_kgk)
    if d.spreader:
        s = n
        g_cs = d.pad_k * d.pad_area_cm2 * 1e-4 / (d.pad_t_mm * 1e-3)
        for k in range(n):
            K[k, k] += g_cs
            K[s, s] += g_cs
            K[k, s] -= g_cs
            K[s, k] -= g_cs
        g_amb[s] = 2.0 * h * d.sp_area_cm2 * 1e-4
        C[s] = D.spreader_mass_kg(d) * CP_AL
    if d.liquid:
        lp = liquid_parts(d, flow_scale)
        p, c = n, n + 1
        for k in range(n):                       # cell to plate through its pad
            K[k, k] += lp.g_pad
            K[p, p] += lp.g_pad
            K[k, p] -= lp.g_pad
            K[p, k] -= lp.g_pad
        # Plate loses heat to the coolant inlet temperature T_in = (1 - eps_rad) T_c + eps_rad T_amb.
        # Coolant gains the same heat and loses g_rad (T_c - T_amb) to the air. K is not symmetric, the solver does not need it to be.
        K[p, p] += lp.g_plate * (1.0 - lp.eps_rad)
        K[p, c] -= lp.g_plate * (1.0 - lp.eps_rad)
        K[c, p] -= lp.g_plate
        K[c, c] += lp.g_plate
        g_amb[p] = lp.g_plate_air + lp.g_plate * lp.eps_rad
        g_amb[c] = lp.g_rad - lp.g_plate * lp.eps_rad
        C[p] = lp.c_plate
        C[c] = lp.c_cool
    K = K + np.diag(g_amb)
    return K, g_amb, C


def sensor_index(d: D.Design) -> int | None:
    if d.ntc == "corner":
        return 0
    if d.ntc == "center":
        return (d.n_series // 2) * d.n_parallel + d.n_parallel // 2
    return None  # hottest cell, ideal sensing


@dataclass
class Result:
    design: D.Design
    scenario: D.Scenario
    t_min: np.ndarray
    temps: np.ndarray          # (steps, nodes)
    i_mod: np.ndarray          # A, + discharge, - charge
    v_mod: np.ndarray
    soc: np.ndarray
    phase: np.ndarray          # 0 discharge, 1 rest, 2 charge, 3 rest
    i_cell_max: np.ndarray     # per step, largest single-cell current magnitude
    v_cell_min: np.ndarray     # per step, lowest group voltage
    lim_ratio: np.ndarray      # per step, largest cell current / temperature-dependent datasheet limit
    heat_w: np.ndarray         # per step, total heat in the module (cells plus interconnect)
    e_pol_wh: np.ndarray       # per step, energy stored in the polarisation capacitors
    plating_ratio: np.ndarray  # per charge step, largest cell current / plating-free current at the SOC (cells >= 25 C), else 0
    plating_unrated: np.ndarray  # per step, 1 if charging with any cell below 25 C (outside the electrochemical map)
    i_cells: np.ndarray        # (steps, cells) signed cell current, + discharge
    v_groups: np.ndarray       # (steps, series groups) group terminal voltage
    q_air_w: np.ndarray        # per step, heat leaving to ambient air from cells, spreader or plate (all designs)
    q_plate_w: np.ndarray      # per step, heat passed from the cold plate to the coolant (liquid design, else 0)
    q_rad_w: np.ndarray        # per step, heat rejected by the radiator to ambient air (liquid design, else 0)
    t_plate: np.ndarray        # cold plate temperature (liquid design, else NaN)
    t_cool: np.ndarray         # coolant bulk temperature (liquid design, else NaN)
    t_cool_in: np.ndarray      # coolant temperature entering the cold plate (liquid design, else NaN)
    t_cool_out: np.ndarray     # coolant temperature leaving the cold plate (liquid design, else NaN)
    pump_w: np.ndarray         # per step, pump electrical power
    fan_rad_w: np.ndarray      # per step, radiator fan electrical power
    n_cells: int
    dt: float = 5.0
    ind: dict = field(default_factory=dict)


def simulate(d: D.Design, sc: D.Scenario, cell: Cell = CELL, dt: float = 5.0,
             max_phase_h: float = 8.0) -> Result:
    ns, npar = d.n_series, d.n_parallel
    n = ns * npar
    liq = d.liquid
    K, g_amb, C = build_thermal(d, cell, sc.pump_flow_frac if liq else 1.0)
    size = len(C)
    Ainv = np.linalg.inv(np.diag(C / dt) + K)
    if liq:
        lp_run = liquid_parts(d, sc.pump_flow_frac)
        lp_off = liquid_parts(d, 0.0)
        K_off, g_amb_off, _ = build_thermal(d, cell, 0.0)
        Ainv_off = np.linalg.inv(np.diag(C / dt) + K_off)
        g_side = g_amb[:ns * npar].copy()                  # cell side convection, unchanged by the coolant loop
        pump_w_nom = d.pump_power_w
        fan_rad_nom = d.rad_fan_power_w if d.rad_fan else 0.0
    rng = np.random.default_rng(sc.seed)
    mism = np.clip(1.0 + sc.r_sigma * rng.standard_normal(n), 0.7, 1.6)
    r0_25 = cell.r0_mohm * 1e-3 * mism
    r1_25 = cell.r1_mohm * 1e-3 * mism
    c1 = cell.c1_f
    r_bus = d.interconnect_mohm * 1e-3
    sidx = sensor_index(d)

    T = np.full(size, sc.t_amb, dtype=float)
    if liq and d.coolant_t_init_c is not None:
        T[n + 1] = max(float(d.coolant_t_init_c), sc.t_amb)   # no refrigeration: the loop cannot start below ambient
    v1 = np.zeros(n)                       # polarisation voltage of every cell
    soc = 1.0
    q_cell = cell.cap_ah
    i_dis = sc.c_dis * cell.cap_ah * npar
    i_chg_cell = sc.c_chg * cell.cap_ah
    i_chg = i_chg_cell * npar
    i_cut = i_term(cell, i_chg_cell) * npar

    rows_t, rows_T, rows_I, rows_V, rows_soc, rows_ph, rows_ic, rows_vg, rows_lr = [], [], [], [], [], [], [], [], []
    rows_q, rows_ep = [], []
    rows_pr, rows_pu, rows_icells, rows_vgs = [], [], [], []
    rows_qair, rows_qpl, rows_qrad, rows_tpl, rows_tc, rows_tin, rows_tout, rows_pump, rows_fanr = [], [], [], [], [], [], [], [], []
    a_dis = []                             # auxiliary (pump + radiator fan) energy of each discharge phase, Wh
    lim_plate_a = echem.plating_limit_a(echem.SOC_GRID, cell)      # A per cell vs SOC, from the electrochemical map
    e_dis, e_chg, t_dis, t_chg, trips = [], [], [], [], []
    t = 0.0
    fan_w = d.fan_power_w if d.fan_v > 0 else 0.0

    def sense(Tc):
        return Tc.max() if sidx is None else Tc[sidx]

    for _ in range(sc.n_cycles):
        for ph, kind in enumerate(PHASES):
            t_ph = 0.0
            e_ph = 0.0
            aux_ph = 0.0
            trip_s = 0.0
            tripped = False
            cv = False
            while t_ph < max_phase_h * 3600:
                Tc = T[:n]
                f = r_temp_factor(Tc)
                r0 = r0_25 * f
                r1 = r1_25 * f
                voc = float(ocv(soc))
                # Each series group: terminal voltage = A - B * I (I positive on discharge)
                s_g = (1.0 / r0).reshape(ns, npar).sum(axis=1)
                u_g = (v1 / r0).reshape(ns, npar).sum(axis=1)
                a_g = voc - u_g / s_g
                b_g = 1.0 / s_g
                ts = sense(Tc)
                isg = 0.0                  # signed module current, + discharge, - charge
                if kind == "discharge":
                    if tripped and ts <= cell.t_dis_resume:
                        tripped = False
                    if ts >= cell.t_dis_max:
                        tripped = True
                    i_volt = float(np.min((a_g - cell.v_min) / b_g))
                    if (soc <= 0.0 or (1.0 - soc) >= float(cap_retention(Tc.mean()))
                            or (not tripped and i_volt < i_dis)):
                        break
                    isg = 0.0 if tripped else i_dis
                elif kind == "charge":
                    if tripped and ts <= cell.t_chg_resume:
                        tripped = False
                    if ts >= cell.t_chg_max:
                        tripped = True
                    if soc >= 1.0:
                        break
                    i_volt = float(np.min((cell.v_max - a_g) / b_g))   # largest charge current that keeps every group <= 4.2 V
                    if not cv and i_volt <= i_chg:
                        cv = True
                    i_now = min(i_chg, max(i_volt, 0.0)) if cv else i_chg
                    if cv and i_now <= i_cut:
                        break
                    isg = 0.0 if (tripped or Tc.min() < cell.t_chg_min) else -i_now
                else:  # rest
                    if t_ph >= sc.rest_min * 60:
                        break
                i_abs = abs(isg)
                if i_abs == 0.0 and kind != "rest":
                    trip_s += dt
                vg = a_g - b_g * isg
                i_cell = (voc - v1 - np.repeat(vg, npar)) / r0     # signed, + discharge
                v_mod = vg.sum() - isg * r_bus
                tau = r1 * c1
                a = np.exp(-dt / tau)
                v1_new = v1 * a + i_cell * r1 * (1.0 - a)
                q = i_cell ** 2 * r0 + 0.5 * (v1 ** 2 + v1_new ** 2) / r1 + isg ** 2 * r_bus / n
                rows_t.append(t / 60.0)
                rows_T.append(T.copy())
                rows_I.append(isg)
                rows_V.append(v_mod)
                rows_soc.append(soc)
                rows_ph.append(ph)
                rows_ic.append(np.abs(i_cell).max() if i_abs > 0 else 0.0)
                rows_vg.append(vg.min())
                rows_q.append(float(q.sum()))
                rows_ep.append(0.5 * c1 * float(np.sum(v1 ** 2)) / 3600.0)
                rows_icells.append(i_cell.copy())
                rows_vgs.append(vg.copy())
                failed = liq and sc.pump_fail_min >= 0.0 and t / 60.0 >= sc.pump_fail_min   # pump fault
                idle = liq and d.pump_rest_off and kind == "rest"                          # controlled stop, fan stops too
                pump_off = failed or idle
                if liq:
                    lp = lp_off if pump_off else lp_run
                    tp, tc = T[n], T[n + 1]
                    t_in = tc - lp.eps_rad * (tc - sc.t_amb)           # radiator outlet = plate inlet
                    q_pl = lp.g_plate * (tp - t_in)                    # plate to coolant
                    rows_qair.append(float(g_side @ (T[:n] - sc.t_amb) + lp.g_plate_air * (tp - sc.t_amb)))
                    rows_qpl.append(q_pl)
                    rows_qrad.append(lp.g_rad * (tc - sc.t_amb))
                    rows_tpl.append(tp)
                    rows_tc.append(tc)
                    rows_tin.append(t_in)
                    rows_tout.append(t_in + q_pl / lp.mcp if lp.mcp > 1e-9 else tc)   # Q = m cp (T_out - T_in)
                    pw = 0.0 if pump_off else pump_w_nom
                    fw = 0.0 if idle else fan_rad_nom
                    rows_pump.append(pw)
                    rows_fanr.append(fw)
                    aux_ph += (pw + fw) * dt / 3600.0
                else:
                    rows_qair.append(float(g_amb @ (T - sc.t_amb)))
                    rows_qpl.append(0.0)
                    rows_qrad.append(0.0)
                    rows_tpl.append(np.nan)
                    rows_tc.append(np.nan)
                    rows_tin.append(np.nan)
                    rows_tout.append(np.nan)
                    rows_pump.append(0.0)
                    rows_fanr.append(0.0)
                if kind == "charge" and i_abs > 0:
                    ok_t = Tc >= echem.T_VALID_MIN
                    lim_p = float(np.interp(soc, echem.SOC_GRID, lim_plate_a))
                    rows_pr.append(float(np.max(np.abs(i_cell)[ok_t]) / lim_p) if ok_t.any() else 0.0)
                    rows_pu.append(0.0 if ok_t.all() else 1.0)
                else:
                    rows_pr.append(0.0)
                    rows_pu.append(0.0)
                if i_abs > 0:
                    # Temperature is clipped into the datasheet window so this ratio measures the current-vs-band
                    # mismatch only. Out-of-window temperature is reported by the temperature checks.
                    if kind == "discharge":
                        lim = i_dis_limit(cell, np.clip(Tc, cell.t_dis_min, cell.t_dis_max))
                    else:
                        lim = i_chg_limit(cell, np.clip(Tc, cell.t_chg_min, cell.t_chg_max))
                    rows_lr.append(float(np.max(np.abs(i_cell) / lim)))
                else:
                    rows_lr.append(0.0)
                e_ph += v_mod * i_abs * dt / 3600.0
                soc = min(1.0, max(0.0, soc - isg * dt / 3600.0 / (npar * q_cell)))
                v1 = v1_new
                qvec = np.zeros(size)
                qvec[:n] = q
                if liq:
                    if not pump_off:
                        qvec[n + 1] = PUMP_HEAT_FRACTION * pump_w_nom      # pump losses end up in the coolant
                    T = (Ainv_off if pump_off else Ainv) @ (C / dt * T + qvec + (g_amb_off if pump_off else g_amb) * sc.t_amb)
                else:
                    T = Ainv @ (C / dt * T + qvec + g_amb * sc.t_amb)
                t += dt
                t_ph += dt
            if kind == "discharge":
                e_dis.append(e_ph)
                t_dis.append(t_ph)
                a_dis.append(aux_ph)
                trips.append(trip_s)
            elif kind == "charge":
                e_chg.append(e_ph)
                t_chg.append(t_ph)
                trips.append(trip_s)

    res = Result(
        design=d, scenario=sc,
        t_min=np.array(rows_t), temps=np.array(rows_T), i_mod=np.array(rows_I),
        v_mod=np.array(rows_V), soc=np.array(rows_soc), phase=np.array(rows_ph),
        i_cell_max=np.array(rows_ic), v_cell_min=np.array(rows_vg), lim_ratio=np.array(rows_lr), heat_w=np.array(rows_q),
        e_pol_wh=np.array(rows_ep), plating_ratio=np.array(rows_pr), plating_unrated=np.array(rows_pu),
        i_cells=np.array(rows_icells), v_groups=np.array(rows_vgs),
        q_air_w=np.array(rows_qair), q_plate_w=np.array(rows_qpl), q_rad_w=np.array(rows_qrad), t_plate=np.array(rows_tpl),
        t_cool=np.array(rows_tc), t_cool_in=np.array(rows_tin), t_cool_out=np.array(rows_tout), pump_w=np.array(rows_pump),
        fan_rad_w=np.array(rows_fanr), dt=dt, n_cells=n,
    )
    aux_w = (d.pump_power_w + (d.rad_fan_power_w if d.rad_fan else 0.0)) if liq else 0.0
    res.ind = indicators(res, e_dis, e_chg, t_dis, t_chg, trips, fan_w, sidx, cell,
                         aux_dis_wh=a_dis[0] if (liq and a_dis) else 0.0, aux_w=aux_w)
    return res


def indicators(res: Result, e_dis, e_chg, t_dis, t_chg, trips, fan_w, sidx, cell: Cell,
               aux_dis_wh: float = 0.0, aux_w: float = 0.0) -> dict:
    d = res.design
    n = res.n_cells
    Tc = res.temps[:, :n]
    tmax = Tc.max(axis=1)
    chg = res.phase == 2
    dis = res.phase == 0
    fan_wh = fan_w * t_dis[0] / 3600.0 if t_dis else 0.0
    e_net = (e_dis[0] - fan_wh - aux_dis_wh) if e_dis else 0.0     # aux_dis_wh: pump and radiator fan during the first discharge
    spread = (tmax - Tc.min(axis=1)).max()
    hot_idx = int(Tc[np.argmax(tmax)].argmax())
    sensor_err = 0.0
    if sidx is not None:
        sensor_err = float((tmax - Tc[:, sidx]).max())
    m, c, v = D.mass_g(d, cell), D.cost_inr(d, cell), D.volume_cm3(d, cell)
    heat_wh = None
    ind = {
        "usable_energy_wh": e_net,
        "delivered_wh": e_dis[0] if e_dis else 0.0,
        "peak_cell_temp_c": float(tmax.max()),
        "peak_temp_charge_c": float(tmax[chg].max()) if chg.any() else float("nan"),
        "peak_temp_discharge_c": float(tmax[dis].max()) if dis.any() else float("nan"),
        "temp_spread_k": float(spread),
        "hottest_cell": hot_idx,
        "charge_time_min": t_chg[0] / 60.0 if t_chg else float("nan"),
        "discharge_time_min": t_dis[0] / 60.0 if t_dis else float("nan"),
        "mass_g": m,
        "cost_inr": c,
        "volume_cm3": v,
        "energy_density_wh_kg": e_net / (m / 1000.0),
        "energy_density_wh_l": e_net / (v / 1000.0),
        "round_trip_eff": (e_dis[0] / e_chg[0]) if e_dis and e_chg and e_chg[0] > 0 else float("nan"),
        "peak_cell_current_dis_a": float(res.i_cell_max[dis].max()) if dis.any() else 0.0,
        "peak_cell_current_chg_a": float(res.i_cell_max[chg].max()) if chg.any() else 0.0,
        "peak_limit_ratio_dis": float(res.lim_ratio[dis].max()) if dis.any() else 0.0,
        "peak_limit_ratio_chg": float(res.lim_ratio[chg].max()) if chg.any() else 0.0,
        "min_group_voltage_v": float(res.v_cell_min[dis | chg].min()),
        "bms_trip_time_s": float(sum(trips)),
        "sensor_underread_k": sensor_err,
        "fan_power_w": fan_w + aux_w,
        "module_v_nom": d.n_series * cell.v_nom,
        "module_v_min": d.n_series * cell.v_min,
        "module_v_max": d.n_series * cell.v_max,
        "module_wh_nom": d.n_cells * cell.v_nom * cell.cap_ah,
        "heat_total_wh": float(res.heat_w.sum() * res.dt / 3600.0),
        **liquid_indicators(res, d),
        "peak_plating_ratio": float(res.plating_ratio.max()) if res.plating_ratio.size else 0.0,
        "plating_unrated_s": float(res.plating_unrated.sum() * res.dt),
    }
    return ind


def liquid_indicators(res: Result, d: D.Design) -> dict:
    """Indicators added with the liquid design. They are also defined for the air-cooled designs (zeros), so one table covers all."""
    dt_h = res.dt / 3600.0
    n = res.n_cells
    Tc = res.temps[:, :n]
    act = (res.phase == 0) | (res.phase == 2)
    t_total_h = len(res.t_min) * dt_h
    pump_wh = float(res.pump_w.sum() * dt_h)
    fanr_wh = float(res.fan_rad_w.sum() * dt_h)
    air_fan_wh = (d.fan_power_w if d.fan_v > 0 else 0.0) * t_total_h
    liq = d.liquid
    rise = res.t_cool_out - res.t_cool_in if liq else np.zeros(1)
    return {
        "avg_cell_temp_c": float(Tc.mean(axis=1)[act].mean()) if act.any() else float("nan"),
        "pump_energy_wh": pump_wh,
        "rad_fan_energy_wh": fanr_wh,
        "air_fan_energy_wh": air_fan_wh,
        "aux_energy_wh": pump_wh + fanr_wh + air_fan_wh,
        "pump_heat_wh": PUMP_HEAT_FRACTION * pump_wh if liq else 0.0,
        "coolant_peak_c": float(np.nanmax(res.t_cool)) if liq else float("nan"),
        "plate_peak_c": float(np.nanmax(res.t_plate)) if liq else float("nan"),
        "coolant_rise_max_k": float(np.nanmax(rise)) if liq else 0.0,
        "coolant_above_ambient_max_k": float(np.nanmax(res.t_cool) - res.scenario.t_amb) if liq else 0.0,
        "peak_radiator_w": float(res.q_rad_w.max()) if liq else 0.0,
        "heat_to_coolant_wh": float(res.q_plate_w.sum() * dt_h),
        "heat_radiator_wh": float(res.q_rad_w.sum() * dt_h),
        "heat_air_wh": float(res.q_air_w.sum() * dt_h),
        "peak_heat_to_coolant_w": float(res.q_plate_w.max()) if liq else 0.0,
        # mean heat the loop has to remove while the module is active: cell and bus heat plus pump heat
        "mean_heat_load_w": float((res.heat_w + PUMP_HEAT_FRACTION * res.pump_w)[act].mean()) if (liq and act.any()) else 0.0,
        # largest gap between the hottest cell and the coolant: the pad and plate share of the temperature budget
        "cell_minus_coolant_max_k": float(np.nanmax(Tc.max(axis=1) - res.t_cool)) if liq else 0.0,
        "rad_g_w_k": liquid_parts(d, res.scenario.pump_flow_frac).g_rad if liq else 0.0,
    }
