# Battery Module Studio

A reusable, parameter-driven platform to configure, simulate and compare battery module designs, from cell to module.
Built for the TELIPORT Season 4 problem "Reusable Digital Platform for Designing Battery Energy System".
Simulation only. Cell: LG INR21700-M50 (product specification 2016-08-23, status "Tentative").

## Setup (virtual environment)

Needs Python 3.10 or newer. Run these from inside the project folder.

Windows (PowerShell):
```
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```
If PowerShell blocks the activate script, run `Set-ExecutionPolicy -Scope Process Bypass` once, or use Command Prompt and `.venv\Scripts\activate.bat`.

macOS or Linux:
```
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Leave the environment with `deactivate`. Next time, only activate it again.

## Run

```
streamlit run app.py              # interactive dashboard
python run_report.py              # writes tables and figures to ./results
python -m bms.validate            # model vs datasheet table
python -m bms.sensitivity         # one-at-a-time sensitivity table
python tests/test_model.py        # 29 self-checks (or: python -m pytest tests)
```

Results are repeatable. The cell-to-cell spread uses a fixed random seed (default 7).

Optional, needs PyBaMM and SciPy (`python -m pip install -r requirements-extra.txt`). These regenerate numbers that are already stored in the code and in `results/`:
```
python tools/make_ocv.py              # OCV table in bms/cell.py
python tools/fit_polarisation.py      # R0, R1, tau in bms/cell.py
python tools/pybamm_crosscheck.py     # results/crosscheck_pybamm.csv
python tools/make_plating_map.py      # data/plating_map.csv (read by the platform at run time)
python tools/echem_replay.py          # results/echem_replay.csv, fig_echem_replay.png
```

## Documentation

`docs/Formulas_and_Calculations.pdf` (source `.tex`, 32 pages) lists every formula, parameter, indicator, check, ranking rule and graph definition, with worked numbers, the validation tables and the sensitivity table.

## Scope

One module, default 4S3P: 12 cells, 14.5 V nominal, about 210 Wh. It is the building block of a larger pack.
Default operating case: 35 C ambient, 1C discharge (4.85 A per cell), 0.6C charge, two back-to-back cycles.

## How the subsystems connect

| Subsystem | File | What it does |
|---|---|---|
| Cell and electrical | `bms/cell.py`, `bms/sim.py` | OCV(SOC), R0 and an R1 \|\| C1 polarisation branch with temperature dependence, current sharing between parallel cells, CC-CV charge |
| Thermal | `bms/sim.py` | One node per cell plus an optional plate. Ohmic and polarisation heat, air-gap conduction, convection. Backward Euler |
| BMS and protection | `bms/sim.py` | 2.5 V and 4.2 V limits, charge trip at 45 C, discharge trip at 60 C, sensor on a chosen cell. Temperature-dependent current limits are checked |
| Parts | `bms/design.py` | Mass, cost and volume from the design, including the liquid loop hardware |
| Liquid cooling (Design D) | `bms/sim.py`, `bms/design.py`, `bms/study.py` | Cold plate and coolant nodes in the same thermal network, effectiveness-based plate and radiator conductances, pump and fan energy, fault cases, ambient and rate study |
| Checker and traceability | `bms/checks.py` | Datasheet limits, requirements, interface conflicts, ranking, traceability table |
| Electrochemical layer | `bms/echem.py`, `data/plating_map.csv` | Lithium-plating screen from the PyBaMM Chen2020 DFN: plating-free charge current against SOC, check E1. Optional live DFN replay of a cell current (needs PyBaMM) |
| Validation and sensitivity | `bms/validate.py`, `bms/sensitivity.py` | Model against datasheet and electrochemical model, one-at-a-time sensitivity |

One `Design` and one `Scenario` object feed all of them, so a change shows up in every result.

## Result with the default settings

35 C ambient, 1C discharge, 0.6C charge, two cycles. Designs A, B and C are unchanged from the version before Design D was added (tested against recorded reference values).

| | A Baseline | B Spreader plate | C Fan, wide gaps | D Liquid cold plate |
|---|---|---|---|---|
| Peak cell temperature (C) | 57.3 | 47.7 | 41.8 | 38.8 |
| Average cell temperature (C) | 44.8 | 42.1 | 38.5 | 37.3 |
| Cell-to-cell spread (K) | 4.67 | 0.21 | 0.82 | 0.12 |
| Net usable energy (Wh) | 202.3 | 203.6 | 203.9 | 198.9 |
| Charge time (min) | 137 | 104 | 105 | 106 |
| Round-trip efficiency (%) | 94.6 | 94.5 | 94.2 | 94.0 |
| Auxiliary cooling energy, whole run (Wh) | 0 | 0 | 4.9 | 32.9 (pump 21.9, radiator fan 11.0) |
| Mass (g) | 878 | 952 | 890 | 1592 |
| Cost (INR, placeholder prices) | 6226 | 6586 | 6376 | 8116 |
| Volume (cm3) | 430 | 467 | 550 | 728 |
| Energy density (Wh/kg, Wh/L) | 230, 470 | 214, 436 | 229, 371 | 125, 273 |
| Fails: datasheet, requirement, interface, liquid loop | 1, 2, 2, 0 | 0, 1, 1, 0 | 0, 0, 0, 0 | 0, 3, 1, 0 |
| Weighted score (four designs) | 0.500 | 0.789 | 0.890 | 0.440 |

The ranking recommends C. D is the coolest and most uniform design, but it fails the mass, cost and volume requirements (all from placeholder part values) and the fan-and-pump power share check, and it spends 5.9 Wh of the first discharge on the pump and radiator fan. If the mass, volume and cost limits are relaxed to 1700 g, 800 cm3 and INR 9000, D passes everything and ranks second behind C. The ranking is computed from the simulation and the editable limits; no design is preferred in advance.

At 1C and 35 C ambient the M50 needs active cooling. Design A passes none of the temperature checks: its hottest cell reaches 57 C, the BMS holds
the string near 45 C on the corner sensor, and the hottest cell is 4.7 K higher, so it charges at 53 C against the 45 C datasheet limit.
B removes the hotspot (spread 0.2 K) but still peaks at 47.7 C. C is the only design inside the 45 C requirement, by about 3 K, and is the largest.
The sweep in `results/fig_sweep_crate.png` (now with D) shows the highest discharge rate that stays at 45 C or below (one cycle, 35 C): 0.55C for A, 0.85C for B, 1.25C for C, 1.75C for D (the datasheet maximum is 1.5C, so D is not limited by temperature there).

Sensitivity (`results/sensitivity.csv`, now 25 cases with four liquid-loop factors added): C ranks first in all 25 cases, B second in 23 and D second in 2, and A is last in every case. D has the lowest peak in all 25 but stays behind because of its mass, volume and cost fails. C stays at or below 45 C unless the ambient is 40 C (46.3 C); D stays below 45 C in every case (highest 43.5 C at 40 C ambient). Radiator UA and plate UA of +/-30 % move the peak of D by -0.2 to +0.5 K and the coolant flow by under 0.1 K.
Cell resistance within the datasheet tolerance moves the peak of A by about 4 K and the natural convection coefficient moves it by about 3.5 K, and a 5 K change of ambient moves C by about 4.5 K, so the margin of C needs a bench test.

Charging at the datasheet maximum of 0.7C fails the charge-current limit check in every design, because a 5 % resistance spread gives
one parallel cell about 4 % more than its share. The default charge rate is therefore 0.6C.

## Design D: liquid cooling

A fourth cooling option: aluminium cold plate with internal channels, water-glycol coolant, DC pump, compact radiator with an optional fan, and thermal pads between the cells and the plate. It is added to the existing thermal network in `bms/sim.py`; it does not replace anything. The electrical model, BMS, plating screen and designs A, B and C are untouched.

**Status.** Every liquid-loop parameter is an engineering estimate or a placeholder. None is a measurement or a manufacturer specification, and the model has not been validated against a test. Prices are placeholders, not quotes.

**Model.** Two nodes are added after the cell nodes: the cold plate (`T_p`) and the coolant (`T_c`).

- Pad, per cell: `G_pad = k_pad A_pad / t_pad`
- Coolant capacity rate: `C_dot = m_dot cp`
- Plate and radiator conductances: `eps_x = 1 - exp(-UA_x / C_dot)`, `G_x = C_dot eps_x` (effectiveness-NTU, other side at fixed temperature). Zero flow gives zero conductance.
- Radiator outlet = plate inlet: `T_in = T_c - eps_r (T_c - T_amb)`
- Plate heat: `Q_p = G_p (T_p - T_in)`, `T_out = T_in + Q_p / C_dot`, so `Q = m_dot cp (T_out - T_in)` holds
- Plate: `C_p dT_p/dt = sum_k G_pad (T_k - T_p) - Q_p - G_air (T_p - T_amb)`
- Coolant: `C_c dT_c/dt = Q_p - G_r (T_c - T_amb) + eta_pump P_pump` (pump heat enters the coolant)
- Cells: the existing equation plus `- G_pad (T_k - T_p)`
- Same backward Euler step and precomputed inverse as before. A second inverse, with the pump stopped, is used in rest phases and after a pump fault.
- Radiator UA without its fan is 25 % of the fan-on value (ASSUMPTION).
- Auxiliary energy: pump and radiator fan electrical energy. Net usable energy = discharge energy minus pump and radiator fan energy during the first discharge (same rule as the fan of design C). The pump heat is thermal and the pump energy is electrical: two views of one energy, not counted twice.
- Steady state: `G_eff = C_dot eps_r eps_p / (eps_r + eps_p - eps_r eps_p)`, which tends to `UA_r UA_p / (UA_r + UA_p)` at high flow (3.43 W/K by default).
- Nothing cools the coolant below ambient: there is no refrigeration. At or above the requirement temperature the radiator cannot cool the cells to it.

**Parameters (editable in the Design editor, source tag of each in the Liquid cooling tab).**

| Parameter | Default | Source |
|---|---|---|
| Coolant type | water-glycol 50/50 (also 30/70, water, custom) | typical handbook values, not a supplier specification |
| Specific heat, density | 3350 J/kg/K, 1065 kg/m3 | ESTIMATE |
| Coolant volume | 0.35 L | ESTIMATE |
| Design mass flow | 30 g/s | ESTIMATE, oversized for about 9 W of heat |
| Coolant start temperature | ambient (never below ambient) | input |
| Cold plate area, thickness, mass | 80 cm2, 6 mm, 85 g | ESTIMATE (65 % solid fraction) |
| Plate-to-coolant conductance | 6 W/K | ESTIMATE (laminar channel) |
| Radiator UA, fan on | 8 W/K | ESTIMATE |
| Pump power, radiator fan power | 4 W, 2 W | ESTIMATE |
| Pump and fan stop during rest | on | ASSUMPTION (changes no peak temperature in the default case) |
| Pad conductivity, thickness, area per cell | 1.5 W/m/K, 1 mm, 3.5 cm2 | ASSUMPTION |
| Pad is an electrical insulator | yes | design rule |
| Loop hardware mass, volume, cost | 250 g, 250 cm3, INR 1650 | PLACEHOLDER, not a quote |

**Checks added (design D only).** LQ1 coolant rise across the plate at most 5 K (flow adequacy); LQ2 radiator capacity at the requirement temperature at least the mean heat load (falls to zero when ambient reaches the requirement); LQ3 hottest cell minus coolant at most 10 K (pad and plate resistance); LQ4 pad is an insulator (design rule, not simulated). The 5 K and 10 K limits are guidelines. The dashboard shows warnings for inadequate flow, overheating, a radiator that cannot hold the requirement and a conductive pad.

**Ambient, rate and repeated cycles** (`results/study_ambient_rate.csv`, three cycles; also in the Liquid cooling tab). Peak cell temperature in C:

| Ambient | Discharge | A | B | C | D |
|---|---|---|---|---|---|
| 25 | 0.5C / 1C / 1.5C | 34.0 / 48.9 / 65.7 | 29.7 / 39.7 / 51.0 | 27.3 / 33.2 / 40.6 | 26.5 / 29.6 / 34.0 |
| 35 | 0.5C / 1C / 1.5C | 42.5 / 57.4 / 66.1 | 38.9 / 47.7 / 60.0 | 36.9 / 41.8 / 48.9 | 36.3 / 38.8 / 42.5 |
| 45 | 0.5C / 1C / 1.5C | 51.5 / 62.8 / 63.6 | 48.4 / 57.4 / 60.3 | 46.6 / 51.3 / 58.8 | 46.2 / 48.4 / 52.1 |

D is the coolest design in all nine cases and stays within 45 C in six of them (C five, B three, A two). At 45 C ambient no design holds the requirement or completes a charge: the BMS charge trip releases only below 40 C. The radiator can hold the coolant at 45 C only up to an ambient of 43.4 to 44.5 C (mean heat load of each case), so a radiator to ambient air does not solve a 45 C environment. The peak does not rise from cycle to cycle for D below 45 C.

**Fault cases** (`results/liquid_fault_cases.csv`, 35 C, 1C): flow cut to 50 % or 25 % changes the peak by under 0.1 K (the flow is oversized); radiator fan failure raises the peak from 38.8 to 41.7 C; a pump that stops at 10 min or never runs gives 51.6 C, above the 45 C charge limit, and the BMS blocks charging. The BMS sees only cell temperature, so a pump fault is caught late.

**Risks the simulation does not capture.** Electrical isolation (the cell cans sit at the potential of their series group, so the pad needs a rated dielectric and an insulation-resistance test), coolant leakage onto cells and busbars (conductive coolant; use a sealed plate, keep fittings off the electrical side, add leak detection), air locks, blocked channels and fouling, pad contact pressure and ageing, freezing and boiling. Not modelled.

**Limits.** Two thermal nodes for the loop with effectiveness conductances: no flow network, pressure drop or transport delay. Mass, volume and cost of the loop hardware are placeholders, so the result that D is heavy and expensive must be re-run with real parts. The flow is oversized: the peak barely moves from 30 down to 3 g/s and rises only below about 1 g/s, so a smaller pump is a possible saving this model does not cost.

**Tests added** (`tests/test_model.py`, 15 new, 29 in total): A, B and C unchanged against recorded values; heat balance over the run and coolant balance at every step; stability of the thermal network with the pump running, weak and stopped; step-size independence; coolant never below ambient; every liquid parameter moves the temperature in the right direction; pump failure and zero flow; BMS trip on a pump failure; auxiliary energy counted in net energy; LQ checks only for D; the radiator cannot cool below ambient; ranking follows the simulated results and does not declare D the winner by default; no drift over three cycles.

## Electrochemical layer

The circuit has no electrode variables, so it cannot say whether a charge current is safe for the anode. `bms/echem.py` adds that from the PyBaMM Chen2020 model (DFN, with SPMe for comparison). `tools/make_plating_map.py` charges the model from empty at 0.1C to 1.5C and stores the negative-electrode potential against lithium. A potential below 0 V means plating is possible. The platform reads that table (no PyBaMM needed at run time), turns it into a plating-free current against SOC, and check E1 compares every cell's charge current with it.

- At 25 C the DFN limit settles at about 0.7C above 80 % SOC, where the datasheet puts its 0.7C charge limit. Below 80 % SOC it allows more, but the datasheet limit still applies, so the layer never raises the charge current.
- Default 0.6C passes E1 in all three designs (peak 88 % of the limit, set by the lowest-resistance parallel cell). 0.7C gives 103 % and 0.8C gives 118 %.
- `tools/echem_replay.py` replays the circuit's own cell current through the DFN at 25 C ambient. At 0.6C the DFN anode potential stays above 0 mV (lowest 6 mV). At 0.8C it is below 0 mV for 2.8 min. The map-based check and the direct replay agree.
- The circuit's terminal voltage is 61 to 81 mV below the DFN during CC charge, because Chen2020 is about 28 % more resistive than the datasheet cell.
- Limits: Chen2020 has no temperature dependence, so the screen is evaluated only for cells at 25 C or above and the charge time below that is reported. It is a screening criterion, not a plated-lithium measurement, and it does not feed temperature, energy or charge time.

## Validation

The M50 specification has no capacity-vs-rate, charge-time or OCV data, so the table in `results/validation.csv` marks every row as an independent check or an input reproduced.
- Independent: energy at 0.2C is 17.8 Wh against 17.6 Wh minimum and 18.2 Wh nominal, mean voltage 3.68 V against 3.63 V, 0.5 s resistance 24.9 mOhm against 21 to 25 mOhm measured points.
- Against the PyBaMM electrochemical model at 25 C (`results/crosscheck_pybamm.csv`): energy at 1C is 2.4 points above it and at 1.5C 3.7 points above. Charge time is 10 to 16 % shorter at 0.3C to 0.7C, so charge times are lower bounds.
- Cold and hot energy loss is taken from the datasheet table (0.2C). Resistance alone would predict 94 % at 0 C against 80 %.

## Assumptions and limitations

Every parameter carries a source tag in the dashboard tab "Assumptions and validation" and in the PDF appendix. Main limits:

- The OCV curve is from the PyBaMM Chen2020 parameter set (a research fit of an M50 cell), not a datasheet curve. That set has no temperature dependence.
- Resistance against temperature is read by eye from a chart (10, 25, 45 C points), with about 2 mOhm of reading error. Below 10 C it is extended, above 45 C it is held.
- Lumped thermal model, no CFD. Convection coefficients, gap exposure and cell specific heat are estimates and need a bench test.
- One polarisation branch fitted at 25 C. No ageing, no reversible heat.
- Cost and non-cell mass are placeholders. Replace with supplier quotes.
- The electrochemical layer is a 25 C screen from a research parameter set, see above.
- Liquid cooling (Design D) is an engineering estimate with no measurement behind it, see its section.
- Simulation only. This is not a safety certification of a real pack.

## Suggested next step

Validate the thermal model on a 3-cell group at low current with thermocouples, fit the convection coefficients and the cell specific heat, then extend to the full module.
# teleport_BTMS
