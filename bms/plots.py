"""Matplotlib figures shared by the dashboard and the report script."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import design as D
from . import echem
from .cell import CELL
from .sim import Result

COLORS = ["#1F6FB5", "#C75A12", "#2A9D8F", "#7B5EA7", "#8A8A8A"]
STYLES = ["-", "--", "-.", ":", "-"]
INK = "#12263A"
GRID = "#D3DEE7"
PHASE_NAMES = {0: "discharge", 2: "charge"}


def _style(ax):
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)


def temperature_vs_time(results: list[Result], req: D.Requirements, cell_limit_chg: float = 45.0):
    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=130)
    r0 = results[0]
    ph = r0.phase
    t = r0.t_min
    start = 0
    for k in range(1, len(ph) + 1):
        if k == len(ph) or ph[k] != ph[start]:
            if ph[start] in PHASE_NAMES:
                ax.axvspan(t[start], t[k - 1], color="#EAF1F7" if ph[start] == 0 else "#FBEFE6", lw=0)
                ax.text((t[start] + t[k - 1]) / 2, 0.02, PHASE_NAMES[ph[start]], transform=ax.get_xaxis_transform(),
                        ha="center", va="bottom", fontsize=8, color="#5A6B7B")
            start = k
    for k, r in enumerate(results):
        Tm = r.temps[:, : r.n_cells].max(axis=1)
        ax.plot(r.t_min, Tm, color=COLORS[k % 5], ls=STYLES[k % 5], lw=2, label=f"{r.design.name} (peak {Tm.max():.1f} C)")
    if abs(req.t_peak_max - cell_limit_chg) < 1e-9:
        ax.axhline(cell_limit_chg, color="#8A1C1C", lw=1.2, ls=(0, (6, 3)))
        ax.text(0.01, cell_limit_chg, f" requirement = datasheet charge limit {cell_limit_chg:g} C",
                transform=ax.get_yaxis_transform(), va="bottom", fontsize=8, color="#8A1C1C")
    else:
        ax.axhline(req.t_peak_max, color="#B04A0C", lw=1.2, ls=(0, (6, 3)))
        ax.text(0.01, req.t_peak_max, f" requirement {req.t_peak_max:g} C", transform=ax.get_yaxis_transform(),
                va="bottom", fontsize=8, color="#B04A0C")
        ax.axhline(cell_limit_chg, color="#8A1C1C", lw=1.2, ls=(0, (2, 2)))
        ax.text(0.01, cell_limit_chg, f" datasheet charge limit {cell_limit_chg:g} C", transform=ax.get_yaxis_transform(),
                va="bottom", fontsize=8, color="#8A1C1C")
    ax.set_xlabel("time (min)")
    ax.set_ylabel("hottest cell (C)")
    ax.set_title("Hottest cell temperature over the duty cycle")
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    _style(ax)
    fig.tight_layout()
    return fig


def indicator_bars(results: list[Result], req: D.Requirements):
    spec = [
        ("peak_cell_temp_c", "Peak cell temperature (C)", req.t_peak_max, "max"),
        ("usable_energy_wh", "Usable energy (Wh)", req.energy_min_wh, "min"),
        ("charge_time_min", "Charge time (min)", req.charge_time_max_min, "max"),
        ("mass_g", "Mass (g)", req.mass_max_g, "max"),
        ("cost_inr", "Estimated cost (INR)", req.cost_max_inr, "max"),
        ("volume_cm3", "Volume (cm3)", req.volume_max_cm3, "max"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.4), dpi=130)
    names = [r.design.name.split("  ")[0] if "  " in r.design.name else r.design.name for r in results]
    for ax, (key, title, lim, kind) in zip(axes.ravel(), spec):
        vals = [r.ind[key] for r in results]
        bars = ax.bar(names, vals, color=[COLORS[k % 5] for k in range(len(results))], width=0.6)
        ax.axhline(lim, color="#B04A0C", lw=1.2, ls=(0, (6, 3)))
        ax.set_title(title, fontsize=10)
        top = max(max(vals), lim) * 1.15
        bottom = 0
        if key in ("peak_cell_temp_c",):
            bottom = min(min(vals), lim) - 8
        ax.set_ylim(bottom, top)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}" if v > 100 else f"{v:.1f}", ha="center", va="bottom", fontsize=8, color=INK)
        _style(ax)
    fig.suptitle("Indicators vs requirement (dashed line)", color=INK, fontsize=11)
    fig.tight_layout()
    return fig


def cell_heatmaps(results: list[Result], req: D.Requirements):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.1), dpi=130, squeeze=False)
    vmin = min(r.temps[:, : r.n_cells].min() for r in results)
    vmax = max(r.temps[:, : r.n_cells].max() for r in results)
    for ax, r in zip(axes[0], results):
        d = r.design
        tm = r.temps[:, : r.n_cells].max(axis=1)
        k = int(tm.argmax())
        grid = r.temps[k, : r.n_cells].reshape(d.n_series, d.n_parallel).T
        im = ax.imshow(grid, cmap="YlOrRd", vmin=vmin, vmax=vmax, aspect="equal")
        for (i, g), v in np.ndenumerate(grid):
            ax.text(g, i, f"{v:.1f}", ha="center", va="center", fontsize=8, color="#12263A")
        ax.set_xticks(range(d.n_series), [f"S{g + 1}" for g in range(d.n_series)], fontsize=8)
        ax.set_yticks(range(d.n_parallel), [f"P{i + 1}" for i in range(d.n_parallel)], fontsize=8)
        ax.set_title(f"{d.name}\nat peak, t = {r.t_min[k]:.0f} min", fontsize=9, color=INK)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="C")
    return fig


def electrical(res: Result):
    fig, axes = plt.subplots(3, 1, figsize=(9, 6), dpi=130, sharex=True)
    axes[0].plot(res.t_min, res.v_mod, color=COLORS[0], lw=1.8)
    axes[0].set_ylabel("module voltage (V)")
    axes[1].plot(res.t_min, res.i_mod, color=COLORS[1], lw=1.8)
    axes[1].set_ylabel("module current (A)\n+ discharge, - charge")
    axes[2].plot(res.t_min, 100 * res.soc, color=COLORS[2], lw=1.8)
    axes[2].set_ylabel("SOC (%)")
    axes[2].set_xlabel("time (min)")
    for ax in axes:
        _style(ax)
    fig.suptitle(f"Electrical behaviour, {res.design.name}", color=INK, fontsize=11)
    fig.tight_layout()
    return fig


def sweep_plot(x, curves: dict, xlabel: str, req_t: float, title: str):
    fig, ax = plt.subplots(figsize=(7.5, 4), dpi=130)
    for k, (name, y) in enumerate(curves.items()):
        ax.plot(x, y, color=COLORS[k % 5], ls=STYLES[k % 5], lw=2, marker="o", ms=4, label=name)
    ax.axhline(req_t, color="#B04A0C", lw=1.2, ls=(0, (6, 3)))
    ax.text(0.01, req_t, f" requirement {req_t:g} C", transform=ax.get_yaxis_transform(), va="bottom", fontsize=8, color="#B04A0C")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("peak cell temperature (C)")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout()
    return fig


def plating_plot(results: list[Result]):
    """Largest cell current during the first charge, against SOC, with the plating-free current from the electrochemical map
    and the datasheet charge limit. The map is for 25 C and above."""
    fig, ax = plt.subplots(figsize=(9, 4.0), dpi=130)
    soc = np.linspace(0.0, 1.0, 101)
    ax.fill_between(soc * 100, 0, echem.plating_limit_a(soc, model="DFN"), color="#E3F1EC", lw=0, label="below the plating-free current (DFN, 25 C)")
    ax.plot(soc * 100, echem.plating_limit_a(soc, model="SPMe"), color="#2A9D8F", lw=1.0, ls=":", label="SPMe, same map")
    ax.axhline(CELL.i_chg_max, color="#8A8A8A", lw=1.0, ls="--", label="datasheet 0.7C (25 to 45 C)")
    for k, r in enumerate(results):
        c = np.where(r.phase == 2)[0]
        if len(c) == 0:
            continue
        first = c[: np.argmax(np.diff(c) > 1) + 1] if np.any(np.diff(c) > 1) else c
        ax.plot(100 * r.soc[first], r.i_cell_max[first], color=COLORS[k % len(COLORS)], ls=STYLES[k % len(STYLES)], lw=1.8,
                label=r.design.name)
    ax.set_xlabel("State of charge (%)")
    ax.set_ylabel("Largest cell charge current (A)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 7.6)
    ax.set_title("Charge current against the electrochemical plating limit", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    _style(ax)
    fig.tight_layout()
    return fig


def _phase_bands(ax, r: Result):
    ph, t = r.phase, r.t_min
    start = 0
    for k in range(1, len(ph) + 1):
        if k == len(ph) or ph[k] != ph[start]:
            if ph[start] in PHASE_NAMES:
                ax.axvspan(t[start], t[k - 1], color="#EAF1F7" if ph[start] == 0 else "#FBEFE6", lw=0)
            start = k


def coolant_vs_time(results: list[Result]):
    """Coolant, cold plate and hottest cell temperature, and the heat flows of the loop, for every liquid design."""
    liq = [r for r in results if r.design.liquid]
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), dpi=130, sharex=True)
    if not liq:
        axes[0].text(0.5, 0.5, "No liquid-cooled design in the comparison", ha="center", va="center", transform=axes[0].transAxes)
        return fig
    r = liq[0]
    _phase_bands(axes[0], r)
    _phase_bands(axes[1], r)
    for k, r in enumerate(liq):
        n = r.n_cells
        tag = r.design.name.split()[0]
        axes[0].plot(r.t_min, r.temps[:, :n].max(axis=1), color=COLORS[3], lw=2, label=f"{tag} hottest cell")
        axes[0].plot(r.t_min, r.t_plate, color=COLORS[1], lw=1.6, ls="--", label=f"{tag} cold plate")
        axes[0].plot(r.t_min, r.t_cool, color=COLORS[0], lw=1.8, label=f"{tag} coolant, plate outlet / radiator inlet")
        axes[0].plot(r.t_min, r.t_cool_in, color=COLORS[2], lw=1.4, ls=":", label=f"{tag} coolant, radiator outlet / plate inlet")
        axes[1].plot(r.t_min, r.q_plate_w, color=COLORS[1], lw=1.8, label=f"{tag} cold plate to coolant")
        axes[1].plot(r.t_min, r.q_rad_w, color=COLORS[0], lw=1.8, ls="--", label=f"{tag} radiator to ambient air")
        axes[1].plot(r.t_min, r.heat_w, color="#8A8A8A", lw=1.0, label=f"{tag} heat generated in the module")
    axes[0].axhline(liq[0].scenario.t_amb, color="#12263A", lw=1.0, ls=(0, (2, 2)))
    axes[0].text(0.01, liq[0].scenario.t_amb, " ambient (coolant cannot go below it)", transform=axes[0].get_yaxis_transform(),
                 va="bottom", fontsize=8, color=INK)
    axes[0].set_ylabel("temperature (C)")
    axes[1].set_ylabel("heat flow (W)")
    axes[1].set_xlabel("time (min)")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    axes[1].legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    axes[0].set_title("Liquid loop temperatures and heat flows (engineering estimate, not a measurement)", fontsize=10, color=INK)
    for ax in axes:
        _style(ax)
    fig.tight_layout()
    return fig


def heat_removal(results: list[Result]):
    """Where the heat goes over the whole run (left) and what the cooling hardware costs in electrical energy (right)."""
    names = [r.design.name.split("  ")[0] if "  " in r.design.name else r.design.name for r in results]
    x = np.arange(len(results))
    dt_h = [r.dt / 3600.0 for r in results]
    air = np.array([r.q_air_w.sum() * h for r, h in zip(results, dt_h)])
    rad = np.array([r.q_rad_w.sum() * h for r, h in zip(results, dt_h)])
    gen = np.array([(r.heat_w.sum() + (r.pump_w.sum() if r.design.liquid else 0.0)) * h for r, h in zip(results, dt_h)])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=130)
    ax = axes[0]
    ax.bar(x, air, 0.55, color=COLORS[0], label="to ambient air from cells, plate or spreader")
    ax.bar(x, rad, 0.55, bottom=air, color=COLORS[1], label="radiator to ambient air")
    ax.plot(x, gen, "k_", ms=26, mew=2, label="generated (cells, bus, pump heat)")
    ax.set_xticks(x, names)
    ax.set_ylabel("energy over the run (Wh)")
    ax.set_title("Heat rejected to ambient", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    ax = axes[1]
    pump = np.array([r.ind["pump_energy_wh"] for r in results])
    fanr = np.array([r.ind["rad_fan_energy_wh"] for r in results])
    fana = np.array([r.ind["air_fan_energy_wh"] for r in results])
    ax.bar(x, pump, 0.55, color=COLORS[3], label="pump")
    ax.bar(x, fanr, 0.55, bottom=pump, color=COLORS[1], label="radiator fan")
    ax.bar(x, fana, 0.55, bottom=pump + fanr, color=COLORS[2], label="module fan")
    for k, tot in enumerate(pump + fanr + fana):
        ax.text(k, tot, f"{tot:.1f}", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(x, names)
    ax.set_ylabel("electrical energy over the run (Wh)")
    ax.set_title("Auxiliary cooling energy", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout()
    return fig
