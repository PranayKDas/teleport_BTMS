"""Design, scenario and requirement definitions, plus mass, cost and size roll-up."""
from __future__ import annotations

from dataclasses import dataclass, replace

from .cell import CELL, Cell

RHO_AL = 2700.0  # kg/m3, aluminium

# Coolant presets for Design D. Typical handbook values near 30 to 40 C, not a supplier specification (ASSUMPTION).
# Every value stays editable in the dashboard.
COOLANTS = {
    "Water-glycol 50/50": {"cp": 3350.0, "rho": 1065.0},
    "Water-glycol 30/70": {"cp": 3800.0, "rho": 1040.0},   # 30 % glycol by volume
    "Water (reference)": {"cp": 4180.0, "rho": 995.0},
}


@dataclass(frozen=True)
class Design:
    name: str = "A  Baseline"
    n_series: int = 4
    n_parallel: int = 3
    spacing_mm: float = 1.5        # gap between neighbouring cells
    h_nat: float = 10.0            # W/m2/K, natural convection plus radiation (ASSUMPTION)
    fan_v: float = 0.0             # m/s air speed through the module, 0 = no fan
    fan_power_w: float = 0.0
    spreader: bool = False         # aluminium plate bonded to the cells with thermal pads
    sp_area_cm2: float = 100.0     # plate area, both faces see the air (default design B uses 125)
    sp_thick_mm: float = 2.0
    pad_k: float = 1.5             # W/m/K, thermal pad (ASSUMPTION)
    pad_t_mm: float = 1.0
    pad_area_cm2: float = 4.0      # contact patch per cell
    ntc: str = "corner"            # BMS temperature sensor: corner, center or hottest
    interconnect_mohm: float = 4.0  # total series-link resistance (ASSUMPTION)
    # ---- liquid cooling (Design D). Ignored unless liquid = True. All values are engineering estimates (ASSUMPTION). ----
    liquid: bool = False
    coolant_type: str = "Water-glycol 50/50"
    coolant_t_init_c: float | None = None   # C, coolant start temperature. None = ambient. Cannot fall below ambient (no refrigeration)
    coolant_mdot_kg_s: float = 0.03         # about 1.7 L/min of 50/50 glycol; a small DC circulation pump class
    coolant_cp: float = 3350.0              # J/kg/K
    coolant_rho: float = 1065.0             # kg/m3
    coolant_vol_l: float = 0.35             # litres in the whole loop (plate channels, hoses, radiator, reservoir)
    plate_area_cm2: float = 80.0            # cold plate area (the 4S3P cell footprint is about 61 cm2)
    plate_thick_mm: float = 6.0             # milled channels inside
    plate_mass_kg: float = 0.085            # 80 cm2 x 6 mm x 2700 kg/m3 x 65 % solid fraction = 0.084 kg
    ua_plate_w_k: float = 6.0               # W/K, plate to coolant. Laminar channel estimate, h about 400 W/m2/K x 0.015 m2 wetted
    rad_ua_w_k: float = 8.0                 # W/K, radiator to ambient air with its fan running
    rad_fan: bool = True                    # False = natural convection only, UA is reduced (see sim.RAD_NATURAL_FRACTION)
    pump_power_w: float = 4.0               # W electrical, small brushless DC pump class
    rad_fan_power_w: float = 2.0            # W electrical, 80 to 92 mm radiator fan class
    pump_rest_off: bool = True              # pump and radiator fan stop during the rest phases (simple BMS control, ASSUMPTION)
    pad_dielectric: bool = True             # cell-to-plate pad is an electrical insulator (design rule, DS 5.1)
    liq_hw_mass_g: float = 250.0            # pump, radiator, fan, hoses, fittings (PLACEHOLDER, weigh the real parts)
    liq_hw_volume_cm3: float = 250.0        # same parts, packaged (PLACEHOLDER)
    liq_hw_cost_inr: float = 1650.0         # plate machining, pump, radiator, fan, hoses, coolant (PLACEHOLDER, not a quote)

    @property
    def n_cells(self) -> int:
        return self.n_series * self.n_parallel


@dataclass(frozen=True)
class Scenario:
    t_amb: float = 35.0       # C
    c_dis: float = 1.0        # discharge rate, multiples of cell capacity
    c_chg: float = 0.6        # charge rate, multiples of cell capacity. 0.7C is the datasheet maximum (25 to 45 C),
                              # and a 5 % resistance spread already pushes one parallel cell above it
    n_cycles: int = 2         # discharge, rest, charge, rest
    rest_min: float = 10.0
    r_sigma: float = 0.05     # cell-to-cell resistance spread (1 sigma, fraction)
    seed: int = 7
    pump_flow_frac: float = 1.0   # fraction of the design coolant flow actually delivered (liquid designs only)
    pump_fail_min: float = -1.0   # minute at which the pump stops, -1 = never (liquid designs only)


@dataclass(frozen=True)
class Requirements:
    t_peak_max: float = 45.0        # C, top of the datasheet charge window (DS 5.1): above it the BMS blocks charging
    energy_min_wh: float = 190.0
    charge_time_max_min: float = 120.0
    mass_max_g: float = 1100.0
    cost_max_inr: float = 8000.0
    v_window: tuple = (10.0, 16.8)  # V, window the load and charger accept
    volume_max_cm3: float = 600.0
    min_gap_mm: float = 1.0         # design rule: DS 5.1 asks for insulation between cells and metal parts


def default_designs() -> list[Design]:
    a = Design(name="A  Baseline")
    b = Design(name="B  Spreader plate", spreader=True, sp_area_cm2=125.0, pad_area_cm2=5.0)  # plate and pad scaled with cell side area
    c = Design(name="C  Fan, wide gaps", spacing_mm=4.0, fan_v=2.0, fan_power_w=0.8)
    d = Design(name="D  Liquid cold plate", liquid=True, pad_area_cm2=3.5)   # pads on the cell base (21.1 mm cell: 3.5 cm2)
    return [a, b, c, d]


# ---- mass, cost, size ---------------------------------------------------------

def spreader_mass_kg(d: Design) -> float:
    return d.sp_area_cm2 * 1e-4 * d.sp_thick_mm * 1e-3 * RHO_AL if d.spreader else 0.0


def coolant_mass_kg(d: Design) -> float:
    return d.coolant_rho * d.coolant_vol_l * 1e-3 if d.liquid else 0.0


def mass_g(d: Design, cell: Cell = CELL) -> float:
    m = d.n_cells * (cell.mass_kg * 1000 + 2.0)      # cell + holder (ASSUMPTION 2 g)
    m += 10.0 + 4.0 * d.n_series                      # nickel strip and wiring (ASSUMPTION)
    m += 12.0                                         # BMS board (ASSUMPTION)
    m += spreader_mass_kg(d) * 1000 + (0.5 * d.n_cells if d.spreader else 0.0)
    m += 12.0 if d.fan_v > 0 else 0.0                 # 40 mm fan (ASSUMPTION)
    if d.liquid:
        m += d.plate_mass_kg * 1000 + 0.5 * d.n_cells + coolant_mass_kg(d) * 1000 + d.liq_hw_mass_g
    return m


def cost_inr(d: Design, cell: Cell = CELL) -> float:
    c = d.n_cells * (cell.price_inr + 8.0)            # cell + holder
    c += 150.0 + 20.0 * d.n_series                    # strip, wiring
    c += 200.0 + 75.0 * d.n_series                    # BMS board scales with series count
    if d.spreader:
        c += spreader_mass_kg(d) * 1000 * 0.3 + 100.0 + 20.0 * d.n_cells  # plate, machining, pads
    if d.fan_v > 0:
        c += 150.0
    if d.liquid:
        c += d.liq_hw_cost_inr + 20.0 * d.n_cells     # loop hardware placeholder + pads
    return c


def volume_cm3(d: Design, cell: Cell = CELL) -> float:
    pitch = cell.dia_m * 1000 + d.spacing_mm
    v = (d.n_series * pitch) * (d.n_parallel * pitch) * cell.len_m * 1000 / 1000.0  # mm3 -> cm3
    if d.spreader:
        v += d.sp_area_cm2 * 0.3                      # plate plus pad, 3 mm stack
    if d.fan_v > 0:
        v += 20.0                                     # 40 x 40 x 12 mm fan with duct
    if d.liquid:
        v += d.plate_area_cm2 * d.plate_thick_mm * 0.1 + d.liq_hw_volume_cm3
    return v


def with_(d: Design, **kw) -> Design:
    return replace(d, **kw)


# Source tag of every Design D parameter. None of these is a measurement or a manufacturer specification.
LIQUID_SOURCES = {
    "coolant_type": ("Coolant type", "-", "preset names a typical mixture"),
    "coolant_cp": ("Coolant specific heat", "J/kg/K", "ESTIMATE: typical handbook value near 30 to 40 C, not a supplier specification"),
    "coolant_rho": ("Coolant density", "kg/m3", "ESTIMATE: typical handbook value, not a supplier specification"),
    "coolant_vol_l": ("Coolant volume in the whole loop", "L", "ESTIMATE: plate channels, hoses, radiator, reservoir"),
    "coolant_mdot_kg_s": ("Coolant mass flow rate", "kg/s", "ESTIMATE: small DC pump class, oversized for about 9 W of heat"),
    "coolant_t_init_c": ("Coolant temperature at the start", "C", "input: clipped to ambient or above (no refrigeration)"),
    "plate_area_cm2": ("Cold plate area", "cm2", "ESTIMATE: 4S3P footprint is about 61 cm2"),
    "plate_thick_mm": ("Cold plate thickness", "mm", "ESTIMATE: milled channels inside"),
    "plate_mass_kg": ("Cold plate mass", "kg", "ESTIMATE: area x thickness x 2700 kg/m3 x 65 % solid fraction"),
    "ua_plate_w_k": ("Cold plate to coolant conductance UA", "W/K", "ESTIMATE: laminar channel, h about 400 W/m2/K x 0.015 m2 wetted"),
    "rad_ua_w_k": ("Radiator UA, fan on", "W/K", "ESTIMATE: compact radiator with an 80 to 92 mm fan, no supplier data"),
    "pump_power_w": ("Pump electrical power", "W", "ESTIMATE: small brushless DC pump class"),
    "rad_fan_power_w": ("Radiator fan electrical power", "W", "ESTIMATE: 80 to 92 mm fan class"),
    "pad_k": ("Thermal pad conductivity", "W/m/K", "ASSUMPTION: shared with the spreader design"),
    "pad_t_mm": ("Thermal pad thickness", "mm", "ASSUMPTION"),
    "pad_area_cm2": ("Pad contact area per cell", "cm2", "cell base area, 21.1 mm diameter (DS 3.2)"),
    "liq_hw_mass_g": ("Pump, radiator, fan, hoses, fittings: mass", "g", "PLACEHOLDER: weigh the real parts"),
    "liq_hw_volume_cm3": ("Same parts: packaged volume", "cm3", "PLACEHOLDER"),
    "liq_hw_cost_inr": ("Loop hardware cost", "INR", "PLACEHOLDER, not a quote. Replace with supplier prices"),
}


def liquid_param_table(d: Design):
    rows = []
    for k, (label, unit, src) in LIQUID_SOURCES.items():
        v = getattr(d, k)
        rows.append({"Parameter": label, "Value": "ambient" if v is None else (f"{v:g}" if isinstance(v, float) else str(v)),
                     "Unit": unit, "Source": src})
    rows.append({"Parameter": "Radiator UA without its fan, share of the fan-on value", "Value": "0.25", "Unit": "-",
                 "Source": "ASSUMPTION: sim.RAD_NATURAL_FRACTION"})
    rows.append({"Parameter": "Pump heat entering the coolant, share of pump power", "Value": "1.0", "Unit": "-",
                 "Source": "ASSUMPTION: wet-rotor pump, sim.PUMP_HEAT_FRACTION"})
    return rows
