"""Battery Module Studio: Streamlit dashboard.  Run with:  streamlit run app.py"""
from dataclasses import replace

import numpy as np
import pandas as pd
import streamlit as st

from bms import design as D
from bms import echem
from bms.cell import CELL, cell_param_table
from bms.checks import DEFAULT_WEIGHTS, rank, run_checks, summarize, traceability
from bms.plots import (cell_heatmaps, coolant_vs_time, electrical, heat_removal, indicator_bars, plating_plot, sweep_plot,
                       temperature_vs_time)
from bms.sim import simulate
from bms.sensitivity import run as run_sensitivity
from bms.study import ambient_rate_study, failure_modes
from bms.validate import crosscheck_table, replay_table, validation_table

st.set_page_config(page_title="Battery Module Studio", layout="wide")

INK = "#12263A"
st.markdown(
    f"""<style>
    .block-container {{padding-top: 1.6rem; max-width: 1400px;}}
    h1, h2, h3 {{letter-spacing: -0.01em;}}
    .pill {{display:inline-block; padding:2px 10px; border-radius:999px; font-size:0.82rem; font-weight:600;}}
    .ok {{background:#DDF0E4; color:#14532D;}}
    .bad {{background:#F9DED2; color:#7A2E0A;}}
    </style>""",
    unsafe_allow_html=True,
)

st.title("Battery Module Studio")
st.caption("Reusable platform to configure, simulate and compare battery module designs, from cell to module. "
           "Cell: LG INR21700-M50. Simulation only.")


@st.cache_data(show_spinner=False)
def run_sim(d: D.Design, sc: D.Scenario, dt: float = 5.0):
    return simulate(d, sc, dt=dt)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Scenario")
    t_amb = st.slider("Ambient temperature (C)", 10, 50, 35, help="The resistance model is anchored between 10 and 45 C (chart points). Datasheet charge window is 0 to 45 C.")
    c_dis = st.slider("Discharge rate (C)", 0.25, 2.0, 1.0, 0.05,
                      help="Multiples of 4.85 A. Datasheet maximum is 1.5C (7.3 A) from 5 to 60 C.")
    c_chg = st.slider("Charge rate (C)", 0.25, 1.0, 0.6, 0.05,
                      help="Datasheet maximum is 0.7C (3.4 A) from 25 to 45 C and 0.3C (1.5 A) from 0 to 25 C.")
    n_cycles = st.slider("Back-to-back cycles", 1, 3, 2, help="Each cycle is discharge, rest, charge, rest.")
    r_sigma = st.slider("Cell-to-cell resistance spread (%)", 0, 15, 5) / 100.0
    seed = st.number_input("Random seed", 0, 9999, 7, help="Same seed gives the same result every time.")
    with st.expander("Liquid cooling fault case (Design D only)"):
        flow_pct = st.slider("Coolant flow delivered (% of design flow)", 0, 100, 100, 5,
                             help="100 = healthy pump. Lower values model a weak pump, a blocked channel or an air lock. 0 = no flow.")
        pump_stop = st.number_input("Pump stops at minute (-1 = never)", -1.0, 600.0, -1.0, 1.0,
                                    help="Simulates a pump failure part-way through the run. The BMS keeps its temperature trips.")
    st.header("Requirements")
    rq = D.Requirements()
    t_req = st.number_input("Peak cell temperature, max (C)", 30.0, 60.0, rq.t_peak_max, 1.0)
    e_req = st.number_input("Usable energy, min (Wh)", 20.0, 800.0, rq.energy_min_wh, 5.0)
    ct_req = st.number_input("Charge time, max (min)", 20.0, 400.0, rq.charge_time_max_min, 5.0)
    m_req = st.number_input("Mass, max (g)", 200.0, 3000.0, rq.mass_max_g, 50.0)
    c_req = st.number_input("Cost, max (INR)", 1000.0, 30000.0, rq.cost_max_inr, 250.0)
    v_req = st.number_input("Volume, max (cm3)", 100.0, 2000.0, rq.volume_max_cm3, 25.0)
    vlo = st.number_input("Load window, low (V)", 3.0, 60.0, rq.v_window[0], 0.5)
    vhi = st.number_input("Load window, high (V)", 3.0, 80.0, rq.v_window[1], 0.5)
    with st.expander("Ranking weights"):
        weights = {}
        labels = {"peak_cell_temp_c": "Temperature", "usable_energy_wh": "Energy", "charge_time_min": "Charge time",
                  "mass_g": "Mass", "cost_inr": "Cost", "volume_cm3": "Volume"}
        for k, w in DEFAULT_WEIGHTS.items():
            weights[k] = st.slider(labels[k], 0.0, 1.0, float(w), 0.05)
        tot = sum(weights.values()) or 1.0
        weights = {k: v / tot for k, v in weights.items()}

sc = D.Scenario(t_amb=float(t_amb), c_dis=c_dis, c_chg=c_chg, n_cycles=int(n_cycles), r_sigma=r_sigma, seed=int(seed),
                pump_flow_frac=flow_pct / 100.0, pump_fail_min=float(pump_stop))
req = D.Requirements(t_peak_max=t_req, energy_min_wh=e_req, charge_time_max_min=ct_req, mass_max_g=m_req,
                     cost_max_inr=c_req, v_window=(vlo, vhi), volume_max_cm3=v_req)

tabs = st.tabs(["Compare", "Design editor", "Liquid cooling", "Sensitivity", "Checks", "Traceability",
                "Assumptions and validation", "About"])
t_cmp, t_edit, t_liq, t_sens, t_chk, t_trace, t_assume, t_about = tabs

# ---------------------------------------------------------------- design editor
defaults = D.default_designs()
designs = []
with t_edit:
    st.subheader("Design options")
    st.caption("Each column is one design. Change a value and every tab updates. "
               "Design A is the baseline that the others are compared against.")
    cols = st.columns(len(defaults))
    base_kw = []
    for k, (col, d0) in enumerate(zip(cols, defaults)):
        with col:
            st.markdown(f"**{d0.name}**")
            ns = st.number_input("Series groups (S)", 2, 8, d0.n_series, key=f"ns{k}")
            npar = st.number_input("Cells in parallel (P)", 1, 6, d0.n_parallel, key=f"np{k}")
            sp = st.number_input("Gap between cells (mm)", 0.5, 12.0, d0.spacing_mm, 0.5, key=f"sp{k}",
                                 help="Design rule: at least 1 mm. The datasheet gives no spacing.")
            h = st.number_input("Natural convection h (W/m2/K)", 4.0, 20.0, d0.h_nat, 1.0, key=f"h{k}")
            if d0.liquid:
                st.caption("Cooled by the cold plate loop below. No module fan and no spreader plate.")
                fv, fp, spr, sa, stn = 0.0, 0.0, False, d0.sp_area_cm2, d0.sp_thick_mm
            else:
                fv = st.number_input("Fan air speed (m/s, 0 = no fan)", 0.0, 6.0, d0.fan_v, 0.5, key=f"fv{k}")
                fp = st.number_input("Fan power (W)", 0.0, 5.0, d0.fan_power_w, 0.1, key=f"fp{k}")
                spr = st.checkbox("Aluminium spreader plate", d0.spreader, key=f"spr{k}")
                sa = st.number_input("Plate area (cm2)", 20.0, 400.0, d0.sp_area_cm2, 10.0, key=f"sa{k}", disabled=not spr)
                stn = st.number_input("Plate thickness (mm)", 0.5, 5.0, d0.sp_thick_mm, 0.5, key=f"st{k}", disabled=not spr)
            ntc = st.selectbox("BMS temperature sensor on", ["corner", "center", "hottest"],
                               ["corner", "center", "hottest"].index(d0.ntc), key=f"ntc{k}",
                               help="'hottest' is ideal sensing, only possible with a sensor on every cell.")
            base_kw.append(dict(n_series=int(ns), n_parallel=int(npar), spacing_mm=sp, h_nat=h, fan_v=fv,
                                fan_power_w=fp, spreader=spr, sp_area_cm2=sa, sp_thick_mm=stn, ntc=ntc))
    liq_kw = {}
    for k, d0 in enumerate(defaults):
        if not d0.liquid:
            continue
        st.markdown("---")
        st.subheader(f"{d0.name}: liquid cooling parameters")
        st.caption("Every value is an engineering estimate or a placeholder, not a measurement and not a manufacturer "
                   "specification (source of each value in the Liquid cooling tab). The coolant cannot start below ambient "
                   "and the radiator cannot take it below ambient: there is no refrigeration.")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown("**Coolant**")
            ctypes = list(D.COOLANTS) + ["Custom"]
            ct = st.selectbox("Coolant type", ctypes, ctypes.index(d0.coolant_type) if d0.coolant_type in ctypes else 0, key=f"ct{k}")
            cp0 = D.COOLANTS[ct]["cp"] if ct in D.COOLANTS else d0.coolant_cp
            rho0 = D.COOLANTS[ct]["rho"] if ct in D.COOLANTS else d0.coolant_rho
            cp = st.number_input("Specific heat (J/kg/K)", 1000.0, 5000.0, float(cp0), 50.0, key=f"cp_{k}_{ct}")
            rho = st.number_input("Density (kg/m3)", 800.0, 1300.0, float(rho0), 5.0, key=f"rho_{k}_{ct}")
            vol = st.number_input("Coolant volume, whole loop (L)", 0.05, 2.0, d0.coolant_vol_l, 0.05, key=f"vol{k}")
            mdot = st.number_input("Design mass flow (g/s)", 1.0, 200.0, 1000 * d0.coolant_mdot_kg_s, 1.0, key=f"mdot{k}",
                                   help="Flow actually delivered is this times the fault-case slider in the sidebar.")
            t0_amb = st.checkbox("Coolant starts at ambient", d0.coolant_t_init_c is None, key=f"t0a{k}")
            t0 = None if t0_amb else st.number_input("Coolant start temperature (C)", 0.0, 80.0, 35.0, 1.0, key=f"t0{k}",
                                                    help="Values below ambient are raised to ambient.")
        with c2:
            st.markdown("**Cold plate**")
            parea = st.number_input("Area (cm2)", 20.0, 300.0, d0.plate_area_cm2, 5.0, key=f"pa{k}")
            pthk = st.number_input("Thickness (mm)", 2.0, 15.0, d0.plate_thick_mm, 0.5, key=f"pt{k}")
            pmass = st.number_input("Mass (g)", 20.0, 800.0, 1000 * d0.plate_mass_kg, 5.0, key=f"pm{k}")
            uap = st.number_input("Plate-to-coolant conductance UA (W/K)", 0.5, 100.0, d0.ua_plate_w_k, 0.5, key=f"uap{k}")
        with c3:
            st.markdown("**Radiator and pump**")
            uar = st.number_input("Radiator UA, fan on (W/K)", 0.5, 100.0, d0.rad_ua_w_k, 0.5, key=f"uar{k}")
            rfan = st.checkbox("Radiator fan fitted and running", d0.rad_fan, key=f"rf{k}",
                               help="Off: radiator UA drops to 25 % (ASSUMPTION, natural convection).")
            pw = st.number_input("Pump power (W)", 0.0, 30.0, d0.pump_power_w, 0.5, key=f"pw{k}")
            fw = st.number_input("Radiator fan power (W)", 0.0, 20.0, d0.rad_fan_power_w, 0.5, key=f"fw{k}")
            rest_off = st.checkbox("Pump and fan stop during rest", d0.pump_rest_off, key=f"ro{k}")
        with c4:
            st.markdown("**Thermal pad and hardware**")
            pk = st.number_input("Pad conductivity (W/m/K)", 0.5, 10.0, d0.pad_k, 0.1, key=f"pk{k}")
            pt = st.number_input("Pad thickness (mm)", 0.2, 3.0, d0.pad_t_mm, 0.1, key=f"ptm{k}")
            pad_a = st.number_input("Pad area per cell (cm2)", 1.0, 10.0, d0.pad_area_cm2, 0.5, key=f"paa{k}")
            diel = st.checkbox("Pad is an electrical insulator", d0.pad_dielectric, key=f"die{k}",
                               help="Design rule. Cell cans sit at different potentials, so a conductive pad would short the string.")
            hwm = st.number_input("Hardware mass (g) PLACEHOLDER", 0.0, 2000.0, d0.liq_hw_mass_g, 10.0, key=f"hwm{k}")
            hwv = st.number_input("Hardware volume (cm3) PLACEHOLDER", 0.0, 2000.0, d0.liq_hw_volume_cm3, 10.0, key=f"hwv{k}")
            hwc = st.number_input("Hardware cost (INR) PLACEHOLDER", 0.0, 30000.0, d0.liq_hw_cost_inr, 50.0, key=f"hwc{k}",
                                  help="Not a quote. Replace with supplier prices.")
        liq_kw[k] = dict(coolant_type=ct, coolant_cp=cp, coolant_rho=rho, coolant_vol_l=vol, coolant_mdot_kg_s=mdot / 1000.0,
                         coolant_t_init_c=t0, plate_area_cm2=parea, plate_thick_mm=pthk, plate_mass_kg=pmass / 1000.0,
                         ua_plate_w_k=uap, rad_ua_w_k=uar, rad_fan=rfan, pump_power_w=pw, rad_fan_power_w=fw,
                         pump_rest_off=rest_off, pad_k=pk, pad_t_mm=pt, pad_area_cm2=pad_a, pad_dielectric=diel,
                         liq_hw_mass_g=hwm, liq_hw_volume_cm3=hwv, liq_hw_cost_inr=hwc)
    for k, d0 in enumerate(defaults):
        kw = dict(base_kw[k])
        if d0.liquid:
            kw.update(liq_kw[k])
        designs.append(replace(d0, **kw))

with st.spinner("Simulating"):
    results = [run_sim(d, sc) for d in designs]
checks = [run_checks(r, req) for r in results]
ranking = rank(results, checks, weights)

def _style_status(v):
    return ("background-color:#DDF0E4;color:#14532D" if v == "PASS" else "background-color:#F9DED2;color:#7A2E0A;font-weight:600")


# ---------------------------------------------------------------- liquid cooling messages
def liquid_messages(results, checks, req, sc):
    """(level, text) pairs for the liquid-cooled designs. Level is error, warning or info."""
    out = []
    for r, c in zip(results, checks):
        if not r.design.liquid:
            continue
        name, i = r.design.name, r.ind
        row = c.set_index("ID")
        if sc.pump_flow_frac < 1.0 or sc.pump_fail_min >= 0:
            out.append(("info", f"{name}: fault case active ({100 * sc.pump_flow_frac:.0f} % flow"
                                + (f", pump stops at minute {sc.pump_fail_min:g}" if sc.pump_fail_min >= 0 else "") + ")."))
        if row.loc["LQ1", "Status"] == "FAIL":
            out.append(("error", f"{name}: inadequate coolant flow. The coolant warms by {i['coolant_rise_max_k']:.1f} K across the "
                                 "cold plate, above the 5 K guideline."))
        if i["peak_cell_temp_c"] > 60.0:
            out.append(("error", f"{name}: overheating. Peak cell temperature {i['peak_cell_temp_c']:.1f} C is above the 60 C discharge limit."))
        elif i["peak_cell_temp_c"] > 45.0:
            out.append(("warning", f"{name}: peak cell temperature {i['peak_cell_temp_c']:.1f} C is above the 45 C charge limit, "
                                   "so the BMS blocks charging."))
        elif i["peak_cell_temp_c"] > req.t_peak_max:
            out.append(("warning", f"{name}: peak cell temperature {i['peak_cell_temp_c']:.1f} C is above the {req.t_peak_max:g} C requirement."))
        if row.loc["LQ2", "Status"] == "FAIL":
            out.append(("warning", f"{name}: the radiator cannot hold the coolant at the requirement temperature at {sc.t_amb:g} C ambient. "
                                   "Without refrigeration the coolant cannot go below ambient."))
        if row.loc["LQ3", "Status"] == "FAIL":
            out.append(("warning", f"{name}: the hottest cell is {i['cell_minus_coolant_max_k']:.1f} K above the coolant. "
                                   "Check pad thickness, pad conductivity and plate-to-coolant conductance."))
        if row.loc["LQ4", "Status"] == "FAIL":
            out.append(("error", f"{name}: the cell-to-plate pad is not an insulator. Cells at different potentials would be shorted through the plate."))
    return out


# ---------------------------------------------------------------- compare
with t_cmp:
    best = ranking.iloc[0]
    if bool(best["Acceptable"]):
        st.success(f"Recommended: **{best['Design']}**. It meets every datasheet limit and requirement and has the "
                   f"best weighted score among the designs that do.")
    else:
        st.warning(f"No design meets every requirement. Best compromise: **{best['Design']}** "
                   f"({int(best['Requirement fails'])} requirement and {int(best['Datasheet fails'])} datasheet fails).")
    for lvl, txt in liquid_messages(results, checks, req, sc):
        getattr(st, lvl)(txt)
    if any(r.design.liquid for r in results):
        st.caption("Liquid cooling results are engineering estimates from a lumped model. They are not validated measurements.")
    st.markdown("**Ranking** (datasheet fails first, then electrochemical, requirement and liquid-loop fails, then weighted score from 0 to 1)")
    st.dataframe(ranking, hide_index=True, width="stretch")
    if True:
        st.markdown("**What failed**")
        for r, c in zip(results, checks):
            bad = c[c.Status == "FAIL"]
            if bad.empty:
                st.markdown(f"{r.design.name}: <span class='pill ok'>all checks pass</span>", unsafe_allow_html=True)
            else:
                items = "; ".join(f"{x.ID} {x.Check.lower()} ({x.Value} vs {x.Limit} {x.Unit})" for x in bad.itertuples())
                st.markdown(f"{r.design.name}: <span class='pill bad'>{len(bad)} fail</span> {items}",
                            unsafe_allow_html=True)

    names = [r.design.name for r in results]
    rows = {
        "Usable energy (Wh)": [r.ind["usable_energy_wh"] for r in results],
        "Peak cell temperature (C)": [r.ind["peak_cell_temp_c"] for r in results],
        "Average cell temperature (C)": [r.ind["avg_cell_temp_c"] for r in results],
        "Temperature spread across cells (K)": [r.ind["temp_spread_k"] for r in results],
        "Charge time (min)": [r.ind["charge_time_min"] for r in results],
        "Mass (g)": [r.ind["mass_g"] for r in results],
        "Estimated cost (INR)": [r.ind["cost_inr"] for r in results],
        "Volume (cm3)": [r.ind["volume_cm3"] for r in results],
        "Energy density (Wh/kg)": [r.ind["energy_density_wh_kg"] for r in results],
        "Energy density (Wh/L)": [r.ind["energy_density_wh_l"] for r in results],
        "Round-trip efficiency (%)": [100 * r.ind["round_trip_eff"] for r in results],
        "Module voltage (V, nominal)": [r.ind["module_v_nom"] for r in results],
        "Peak cell current, discharge (A)": [r.ind["peak_cell_current_dis_a"] for r in results],
        "Auxiliary cooling energy, whole run (Wh)": [r.ind["aux_energy_wh"] for r in results],
        "  of which pump (Wh)": [r.ind["pump_energy_wh"] for r in results],
        "  of which radiator fan (Wh)": [r.ind["rad_fan_energy_wh"] for r in results],
        "  of which module fan (Wh)": [r.ind["air_fan_energy_wh"] for r in results],
        "Coolant peak temperature (C)": [r.ind["coolant_peak_c"] if r.design.liquid else np.nan for r in results],
        "Heat rejected by the radiator (Wh)": [r.ind["heat_radiator_wh"] for r in results],
    }
    ind_df = pd.DataFrame(rows, index=names).T.round(1)
    st.markdown("**Indicators**")
    st.dataframe(ind_df, width="stretch")
    st.download_button("Download indicators (CSV)", ind_df.to_csv().encode(), "indicators.csv", "text/csv")

    st.pyplot(temperature_vs_time(results, req), width="stretch")
    st.pyplot(indicator_bars(results, req), width="stretch")
    st.pyplot(heat_removal(results), width="stretch")
    st.markdown("**Temperature of every cell at the moment of peak temperature**")
    st.pyplot(cell_heatmaps(results, req), width="stretch")
    with st.expander("Electrochemical plating screen (PyBaMM Chen2020, 25 C map)"):
        st.caption("Green area: charge current at which the DFN keeps the negative electrode above 0 V against lithium, at that "
                   "state of charge. Valid for cells at 25 C and above only, because the Chen2020 set has no temperature "
                   "dependence. A screening criterion, not a measurement of plated lithium.")
        st.pyplot(plating_plot(results), width="stretch")
        for r in results:
            if r.ind["plating_unrated_s"] > 0:
                st.markdown(f"{r.design.name}: charged for {r.ind['plating_unrated_s'] / 60:.0f} min with a cell below 25 C, outside the map.")
        if echem.available():
            pick_e = st.selectbox("Replay with the live electrochemical model", [r.design.name for r in results], key="echem_pick")
            if st.button("Replay hottest-current cell in the DFN"):
                r = results[[x.design.name for x in results].index(pick_e)]
                c = np.where(r.phase == 2)[0]
                c = c[: np.argmax(np.diff(c) > 1) + 1] if np.any(np.diff(c) > 1) else c
                k = int(np.argmax(np.abs(r.i_cells[c]).max(axis=0)))
                with st.spinner("Running DFN"):
                    out = echem.replay((r.t_min[c] - r.t_min[c][0]) * 60.0, r.i_cells[c, k], float(r.soc[c][0]), "DFN")
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(9, 3.2), dpi=130)
                ax.plot(out["t_s"] / 60, out["phi_neg"] * 1e3, color="#1F6FB5")
                ax.axhline(0, color="k", lw=0.8)
                ax.set_xlabel("Time (min)")
                ax.set_ylabel("Anode potential vs Li (mV)")
                ax.set_title(f"Cell {k}, DFN at 25 C parameters", fontsize=10)
                st.pyplot(fig, width="stretch")
        else:
            st.info("The live replay needs PyBaMM (pip install -r requirements-extra.txt). The screen above does not.")
    with st.expander("Electrical behaviour of one design"):
        pick = st.selectbox("Design", names, key="elec_pick")
        st.pyplot(electrical(results[names.index(pick)]), width="stretch")

# ---------------------------------------------------------------- liquid cooling
with t_liq:
    st.subheader("Liquid cooling: cold plate, coolant loop and radiator")
    liq_idx = [k for k, r in enumerate(results) if r.design.liquid]
    if not liq_idx:
        st.info("No liquid-cooled design in the comparison.")
    else:
        rl, dl, cl = results[liq_idx[0]], designs[liq_idx[0]], checks[liq_idx[0]]
        st.caption("Lumped model: cells, thermal pads, cold plate, coolant and radiator in one thermal network. The plate and "
                   "radiator use effectiveness conductances G = m cp (1 - exp(-UA / (m cp))). Every parameter is an engineering "
                   "estimate or a placeholder. Results are estimates, not validated measurements.")
        for lvl, txt in liquid_messages(results, checks, req, sc):
            getattr(st, lvl)(txt)
        i = rl.ind
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Peak cell temperature", f"{i['peak_cell_temp_c']:.1f} C")
        m2.metric("Coolant peak", f"{i['coolant_peak_c']:.1f} C", f"{i['coolant_above_ambient_max_k']:.1f} K above ambient", delta_color="off")
        m3.metric("Radiator, peak heat rejection", f"{i['peak_radiator_w']:.1f} W")
        m4.metric("Coolant rise across plate", f"{i['coolant_rise_max_k']:.2f} K")
        m5, m6, m7, m8 = st.columns(4)
        m5.metric("Pump power", f"{dl.pump_power_w:.1f} W")
        m6.metric("Radiator fan power", f"{dl.rad_fan_power_w if dl.rad_fan else 0.0:.1f} W")
        m7.metric("Auxiliary energy, whole run", f"{i['aux_energy_wh']:.1f} Wh",
                  f"{100 * i['aux_energy_wh'] / max(i['heat_to_coolant_wh'], 1e-9):.0f} % of heat moved", delta_color="off")
        m8.metric("Net usable energy", f"{i['usable_energy_wh']:.1f} Wh", f"{i['usable_energy_wh'] - i['delivered_wh']:.1f} Wh pump and fan", delta_color="off")
        st.pyplot(coolant_vs_time(results), width="stretch")
        st.pyplot(heat_removal(results), width="stretch")
        st.markdown("**Loop checks**")
        lq = cl[cl.Type == "Liquid loop"]
        st.dataframe(lq.style.map(_style_status, subset=["Status"]), hide_index=True, width="stretch",
                     column_config={"Why it matters": st.column_config.TextColumn(width="large")})
        with st.expander("Parameters and where each value comes from"):
            st.dataframe(pd.DataFrame(D.liquid_param_table(dl)), hide_index=True, width="stretch")
        st.markdown("**Fault cases** (pump and radiator-fan failures, BMS temperature trips unchanged)")
        if st.button("Run fault cases"):
            with st.spinner("Running six cases"):
                fm = failure_modes(dl, sc, req)
            st.dataframe(fm, hide_index=True, width="stretch")
            st.caption("The BMS only sees cell temperature. It does not measure coolant flow, so a pump fault is caught late, by the "
                       "temperature trips. A flow or pump-current sensor is a design change this model suggests.")
        else:
            st.info("Press Run fault cases. It takes a few seconds.")
        st.markdown("**Ambient temperature and discharge rate over repeated cycles** (all designs, 3 cycles, 0.6C charge)")
        a1, a2 = st.columns(2)
        amb_pick = a1.multiselect("Ambient temperatures (C)", [25, 30, 35, 40, 45], [25, 35, 45])
        rate_pick = a2.multiselect("Discharge rates (C)", [0.5, 0.75, 1.0, 1.25, 1.5], [0.5, 1.0, 1.5])
        if st.button("Run ambient and rate study") and amb_pick and rate_pick:
            with st.spinner("Simulating"):
                study = ambient_rate_study(designs, req, tuple(float(x) for x in amb_pick), tuple(float(x) for x in rate_pick))
            st.dataframe(study, hide_index=True, width="stretch")
            st.download_button("Download study (CSV)", study.to_csv(index=False).encode(), "study_ambient_rate.csv", "text/csv")
            st.caption("At or above the 45 C charge limit no design can charge: the BMS trip does not release until the cells fall below 40 C, "
                       "and the radiator cannot cool the coolant below ambient.")
        with st.expander("Engineering risks the simulation does not capture"):
            st.markdown(
                """
- **Electrical isolation.** The cans of a 21700 cell are at the potential of their series group (up to about 16.8 V across the module).
  The pad between the cell and the aluminium plate must be an insulator with a rated breakdown voltage and must not be pierced by
  cell corners or sleeve damage. Check LQ4 only records the design rule. Test with an insulation-resistance meter.
- **Coolant leakage.** A leak next to the cells and the busbars is a short-circuit and corrosion risk. Water-glycol is conductive. Use a
  sealed plate (O-ring or brazed), keep fittings away from the electrical side, add a drip tray or a conductivity sensor, and pressure-test the loop. Not modelled.
- **Flow failure and pump failure.** The model shows the temperature consequence (fault cases above). The BMS has no flow sensor here.
- **Air locks, blocked channels and fouling** lower UA over time. Not modelled, see the UA sensitivity.
- **Thermal interface resistance** is lumped into the pad conductivity and thickness. Real pads need compression force, and their
  resistance rises with age and with poor contact. Not measured.
- **Over-temperature.** The existing BMS trips (charge 45 C, discharge 60 C) act on the sensor cell. Liquid cooling keeps cells uniform, so a
  corner sensor reads the module well (spread below 1 K in the default case).
- **Freezing and boiling.** Water-glycol protects against freezing only at the right mixture. Not modelled.
"""
            )

# ---------------------------------------------------------------- sensitivity
with t_sens:
    st.subheader("Where does each design stop working?")
    st.caption("Runs one discharge and charge cycle per point, with your design settings and scenario. "
               "Charge rate follows the discharge rate up to the charge rate in the sidebar. The sweep stops at 1.5C, the datasheet discharge maximum.")
    mode = st.radio("Sweep", ["Discharge rate", "Ambient temperature", "Uncertain inputs"], horizontal=True)
    if mode == "Uncertain inputs":
        st.caption("Changes one uncertain input at a time (convection, gap exposure, specific heat, resistance, spread, "
                   "ambient, pad) and shows the peak temperature of each design and the resulting order.")
        if st.button("Run sensitivity"):
            with st.spinner("Running about 17 cases"):
                sens = run_sensitivity(designs, sc, req)
            st.dataframe(sens, hide_index=True, width="stretch")
            st.download_button("Download sensitivity (CSV)", sens.to_csv(index=False).encode(), "sensitivity.csv", "text/csv")
        else:
            st.info("Press Run sensitivity. It takes about 20 seconds.")
    elif st.button("Run sweep"):
        with st.spinner("Running"):
            curves = {}
            if mode == "Discharge rate":
                xs = [0.5, 0.75, 1.0, 1.25, 1.5]
                for d in designs:
                    curves[d.name] = [run_sim(d, replace(sc, c_dis=x, c_chg=min(x, c_chg), n_cycles=1), 10.0)
                                      .ind["peak_cell_temp_c"] for x in xs]
                fig = sweep_plot(xs, curves, "discharge rate (C)", t_req, f"Peak cell temperature vs discharge rate at {t_amb} C")
            else:
                xs = [15, 20, 25, 30, 35, 40, 45]
                for d in designs:
                    curves[d.name] = [run_sim(d, replace(sc, t_amb=float(x), n_cycles=1), 10.0)
                                      .ind["peak_cell_temp_c"] for x in xs]
                fig = sweep_plot(xs, curves, "ambient temperature (C)", t_req, f"Peak cell temperature vs ambient at {c_dis}C discharge")
        st.pyplot(fig, width="stretch")
        st.dataframe(pd.DataFrame(curves, index=xs).round(1).rename_axis(mode), width="stretch")
    else:
        st.info("Press Run sweep. Each sweep takes a few seconds.")

# ---------------------------------------------------------------- checks



with t_chk:
    st.subheader("Limit, requirement and interface checks")
    st.caption("Datasheet limits come from the cell specification. Requirements are the targets in the sidebar. "
               "Interface checks compare subsystems against each other.")
    for r, c in zip(results, checks):
        s = summarize(c)
        with st.expander(f"{r.design.name}: {s['datasheet_fail']} datasheet, {s['req_fail']} requirement, "
                         f"{s['interface_fail']} interface, {s['echem_fail']} electrochemical, {s['liquid_fail']} liquid-loop fails", expanded=(sum(s.values()) > 0)):
            st.dataframe(c.style.map(_style_status, subset=["Status"]), hide_index=True, width="stretch",
                         column_config={"Why it matters": st.column_config.TextColumn(width="large")})

# ---------------------------------------------------------------- traceability
with t_trace:
    st.subheader("Traceability")
    st.caption("Each requirement is linked to the component, the assumption, the model, the result and the evidence.")
    for r, c in zip(results, checks):
        tr = traceability(r, c)
        st.markdown(f"**{r.design.name}**")
        st.dataframe(tr, hide_index=True, width="stretch")
        st.download_button(f"Download traceability, {r.design.name.split()[0]}", tr.to_csv(index=False).encode(),
                           f"traceability_{r.design.name.split()[0]}.csv", "text/csv", key=f"dl{r.design.name}")

# ---------------------------------------------------------------- assumptions and validation
with t_assume:
    st.subheader("Cell data and assumptions")
    st.caption("DS = LG Chem product specification INR21700 M50 18.20Wh (2016-08-23, status Tentative). BD = batterydesign.net "
               "M50 page. PB = PyBaMM Chen2020 parameter set. FIT = fitted to PB. Anything marked ASSUMPTION is in none of them.")
    st.dataframe(pd.DataFrame(cell_param_table(CELL)), hide_index=True, width="stretch")
    st.subheader("Model vs datasheet and measured points")
    st.caption("The M50 specification has no capacity-vs-rate or charge-time data, so each row says whether it is an "
               "independent check or only reproduces an input.")
    st.dataframe(validation_table(), hide_index=True, width="stretch")
    cc = crosscheck_table()
    if cc is not None:
        st.subheader("Cross-check against the electrochemical model (25 C only)")
        st.caption("PyBaMM Chen2020 DFN, run offline by tools/pybamm_crosscheck.py. Its parameter set has no temperature "
                   "dependence, so it cannot check cold or hot behaviour.")
        st.dataframe(cc, hide_index=True, width="stretch")
    rp = replay_table()
    if rp is not None:
        st.subheader("Electrochemical replay of the platform's own cell current (25 C)")
        st.caption("Design C at 25 C ambient. The charge current of the most heavily loaded cell is applied to the Chen2020 DFN "
                   "by tools/echem_replay.py. The DFN is about 28 % more resistive than the datasheet cell, so its voltage is higher.")
        st.dataframe(rp, hide_index=True, width="stretch")
    st.subheader("Limitations")
    st.markdown(
        """
- Lumped thermal model: one temperature per cell. No gradients inside a cell, no CFD.
- Liquid cooling (Design D) is an engineering estimate. Coolant properties, flow, plate and radiator conductances, pump and fan power,
  pad properties and the hardware mass, volume and cost are estimates or placeholders (source of each in the Liquid cooling tab). The
  loop is two thermal nodes (plate and coolant) with effectiveness conductances, with no pressure-drop, flow-network, leakage or ageing
  model. Nothing in it has been validated against a measurement. The coolant cannot fall below ambient (no refrigeration).
- Equivalent circuit with one polarisation branch, fitted at 25 C to an electrochemical model. Charge time is 10 to 16 %
  shorter than that model, and energy at 1C is 2 to 3 points higher. Treat both as optimistic bounds.
- The M50 specification is a 2016 tentative draft. It has no capacity-vs-rate, charge-time or OCV data. The OCV curve comes
  from a research parameter set (Chen2020), and the resistance-vs-temperature ratios are read by eye from a chart.
- The electrochemical layer (Chen2020 DFN) is a 25 C parameter set with no temperature dependence. The plating screen is
  evaluated only for cells at 25 C or above, and it is a screening criterion (anode potential below 0 V), not a measured
  plating amount. It is not used to set the temperature, energy or charge-time results.
- No ageing. The datasheet only states 500 cycles to 80 % of minimum energy at about 0.33C, which this model does not track.
- Cold and hot energy loss comes from the datasheet table (0.2C only). Resistance below 10 C is extrapolated.
- Reversible (entropic) heat is not modelled; no M50 value is available in the sources.
- Cooling coefficients (natural convection, forced air, gap exposure) are engineering estimates and need a bench test.
- Heat conduction through the nickel strips and the cell ends is ignored, so hotspots are slightly pessimistic.
- Cost and mass of parts other than the cell are placeholders.
- Simulation only. Nothing here replaces a safety test of a real pack.
"""
    )

# ---------------------------------------------------------------- about
with t_about:
    st.subheader("What this platform does")
    st.markdown(
        """
**Scope.** One battery module of LG INR21700-M50 cells. Default is 4S3P: 12 cells, 14.5 V nominal, about 210 Wh.
It is the building block of a larger pack.

**Workflow.** Pick a layout and cooling option, set the operating case and the requirements, read the indicators,
checks and ranking, then change one thing and see the effect.

**Subsystems and how they connect.**
1. Electrical: open-circuit voltage, series resistance and one polarisation branch per cell, current sharing between parallel cells.
2. Thermal: heat from I squared R feeds a network of cell nodes, air gaps, an optional plate and convection.
3. BMS: voltage cut-offs and temperature trips at the datasheet limits (charge 0 to 45 C, discharge -20 to 60 C), with a sensor on a chosen cell. Current limits depend on cell temperature and are checked, not enforced.
4. Parts: mass, cost and volume from the design.
   Design D adds a cold plate, coolant loop, radiator, pump and radiator fan to the same thermal network (Liquid cooling tab).
5. Checker: compares results with datasheet limits and requirements, and flags interface conflicts.

The same parameter set feeds all five, so a change in one place shows up everywhere.
"""
    )
