"""Stage 3 evidence figures (5 panels).

Usage: python stage3/plots/make_evidence.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from convection_mlt.radiation import (
    DEFAULT_DIFFUSIVITY,
    STEFAN_BOLTZMANN,
    SolveRoute,
    radiation_core,
)

OUT = Path(__file__).resolve().parent / "generated"
OUT.mkdir(parents=True, exist_ok=True)
D = DEFAULT_DIFFUSIVITY
DISPLAY_FLOOR = 1e-16


def _run(n, T_bot, T_top, kappa0, g, P_bot, P_top, F_down_top, F_up_bot,
         route=SolveRoute.THOMAS):
    dp = (P_bot - P_top) / n
    mass_path = np.full(n, dp / g)
    frac = (np.arange(n) + 0.5) / n
    temp = T_bot + (T_top - T_bot) * frac
    kappa = np.full((1, n), kappa0)
    w = np.array([1.0])
    return radiation_core(temp, mass_path, kappa, w,
                          np.array([F_down_top]), np.array([F_up_bot]), D, route), temp, mass_path


# ── Figure 1: Analytic-limit flux profiles ───────────────────────────
def fig1_analytic_limits():
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("Stage 3 — Analytic-limit flux profiles", fontsize=13)

    n = 20
    ifaces = np.arange(n + 1)

    # transparent
    ax = axes[0, 0]
    temp = np.full(n, 1500.0); mp = np.full(n, 1000.0)
    kap = np.full((1, n), 0.0); w = np.array([1.0])
    r = radiation_core(temp, mp, kap, w, np.array([300.0]), np.array([500.0]), D)
    ax.plot(ifaces, r.flux_up[0], 'b-o', ms=3, label='F↑')
    ax.plot(ifaces, r.flux_down[0], 'r-s', ms=3, label='F↓')
    ax.set_title("Transparent (κ=0)"); ax.set_ylabel("Flux [W m⁻²]")
    ax.legend(fontsize=8)

    # isothermal equilibrium
    ax = axes[0, 1]
    T0 = 2000.0; B0 = STEFAN_BOLTZMANN * T0**4
    temp = np.full(n, T0); mp = np.full(n, 1000.0)
    kap = np.full((1, n), 0.1); w = np.array([1.0])
    r_iso = radiation_core(temp, mp, kap, w, np.array([B0]), np.array([B0]), D)
    ax.plot(ifaces, r_iso.flux_up[0], 'b-o', ms=3, label='F↑')
    ax.plot(ifaces, r_iso.flux_down[0], 'r-s', ms=3, label='F↓')
    ax.axhline(B0, color='k', ls='--', lw=0.8, label=f'σT₀⁴={B0:.0f}')
    ax.set_title("Isothermal equilibrium"); ax.legend(fontsize=8)

    # single layer (asymmetric)
    ax = axes[0, 2]
    T0s = 1000.0; B0s = STEFAN_BOLTZMANN * T0s**4
    kappa0s = 0.05; dm = 1e4; dtau = kappa0s * dm
    trans = np.exp(-D * dtau); ef = -np.expm1(-D * dtau)
    r_s1 = radiation_core(np.array([T0s]), np.array([dm]),
                          np.array([[kappa0s]]), np.array([1.0]),
                          np.array([300.0]), np.array([500.0]), D)
    ax.bar([0, 1], [r_s1.flux_up[0, 0], r_s1.flux_up[0, 1]], width=0.3, label='F↑', color='b', alpha=0.7)
    ax.bar([0.3, 1.3], [r_s1.flux_down[0, 0], r_s1.flux_down[0, 1]], width=0.3, label='F↓', color='r', alpha=0.7)
    expected_fd0 = trans * 300.0 + ef * B0s
    expected_fu1 = trans * 500.0 + ef * B0s
    ax.axhline(expected_fd0, color='r', ls=':', lw=1, label=f'analytic F↓[0]={expected_fd0:.1f}')
    ax.axhline(expected_fu1, color='b', ls=':', lw=1, label=f'analytic F↑[1]={expected_fu1:.1f}')
    ax.set_title("Single layer (N=1)"); ax.set_xticks([0.15, 1.15]); ax.set_xticklabels(["bot", "top"])
    ax.legend(fontsize=7)

    # optically thin — with total τ and residual annotation
    ax = axes[1, 0]
    kap_thin = 1e-15
    temp = np.full(n, 1500.0); mp = np.full(n, 1000.0)
    kap = np.full((1, n), kap_thin); w = np.array([1.0])
    r_thin = radiation_core(temp, mp, kap, w, np.array([100.0]), np.array([200.0]), D)
    ax.plot(ifaces, r_thin.flux_up[0], 'b-o', ms=3, label='F↑')
    ax.plot(ifaces, r_thin.flux_down[0], 'r-s', ms=3, label='F↓')
    total_tau = kap_thin * float(np.sum(mp))
    # thin-limit residual: deviation from transparent
    thin_resid_fd = float(np.max(np.abs(r_thin.flux_down[0] - 100.0)))
    thin_resid_fu = float(np.max(np.abs(r_thin.flux_up[0] - 200.0)))
    ax.set_title("Optically thin"); ax.set_ylabel("Flux [W m⁻²]")
    ax.set_xlabel("Interface"); ax.legend(fontsize=8)
    ax.annotate(f"Σ Δτ = {total_tau:.1e}\n"
                f"|F↓ − F↓_top|_max = {thin_resid_fd:.2e}\n"
                f"|F↑ − F↑_bot|_max = {thin_resid_fu:.2e}\n"
                f"(expm1 accuracy)",
                xy=(0.03, 0.03), xycoords='axes fraction', fontsize=7,
                verticalalignment='bottom',
                bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.8))

    # optically thick
    ax = axes[1, 1]
    T0t = 2500.0; B0t = STEFAN_BOLTZMANN * T0t**4
    temp = np.full(n, T0t); mp = np.full(n, 1000.0)
    kap = np.full((1, n), 1e6); w = np.array([1.0])
    r_thick = radiation_core(temp, mp, kap, w, np.array([100.0]), np.array([100.0]), D)
    ax.plot(ifaces, r_thick.flux_up[0], 'b-o', ms=3, label='F↑')
    ax.plot(ifaces, r_thick.flux_down[0], 'r-s', ms=3, label='F↓')
    ax.axhline(B0t, color='k', ls='--', lw=0.8, label=f'B={B0t:.0f}')
    ax.set_title("Optically thick"); ax.set_xlabel("Interface"); ax.legend(fontsize=8)

    # isothermal-equilibrium residual at display floor
    ax = axes[1, 2]
    Ns_iso = [1, 2, 5, 10, 20, 50]
    resids = []
    for nn in Ns_iso:
        T0r = 2000.0; B0r = STEFAN_BOLTZMANN * T0r**4
        temp_r = np.full(nn, T0r); mp_r = np.full(nn, 1000.0)
        kap_r = np.full((1, nn), 0.1); w_r = np.array([1.0])
        r_r = radiation_core(temp_r, mp_r, kap_r, w_r, np.array([B0r]), np.array([B0r]), D)
        fnet_max = float(np.max(np.abs(r_r.flux_net)))
        resids.append(max(fnet_max / B0r, DISPLAY_FLOOR))
    ax.semilogy(Ns_iso, resids, 'ko-', ms=5)
    ax.axhline(1e-12, color='r', ls='--', lw=0.8, label='Gate (10⁻¹²)')
    ax.axhline(DISPLAY_FLOOR, color='gray', ls=':', lw=0.8, label=f'Display floor ({DISPLAY_FLOOR:.0e})')
    ax.set_title("Isothermal equilibrium: |F_net|/σT₀⁴")
    ax.set_xlabel("N"); ax.set_ylabel("Normalized |F_net|_max")
    ax.legend(fontsize=7)
    ax.annotate("F↑[0]=F↓[N]=σT₀⁴, T_i=T₀\nExact: F_net=0 ∀ N",
                xy=(0.03, 0.55), xycoords='axes fraction', fontsize=7,
                bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.8))

    fig.tight_layout()
    fig.savefig(OUT / "fig01_analytic_limits.png", dpi=150)
    plt.close(fig)
    print(f"  → {OUT / 'fig01_analytic_limits.png'}")


# ── Figure 2: Three-route agreement ─────────────────────────────────
def fig2_three_route():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Stage 3 — Three-route solver agreement", fontsize=13)

    Ns = [2, 5, 10, 20, 50, 100]

    td_flux, ts_flux = [], []
    td_heat, ts_heat = [], []
    for n in Ns:
        temp, mp, kap, w = np.full(n, 1500.0), np.full(n, 1000.0), np.full((1, n), 0.05), np.array([1.0])
        top, bot = np.array([200.0]), np.array([600.0])
        rt = radiation_core(temp, mp, kap, w, top, bot, D, SolveRoute.THOMAS)
        rd = radiation_core(temp, mp, kap, w, top, bot, D, SolveRoute.DENSE)
        rs = radiation_core(temp, mp, kap, w, top, bot, D, SolveRoute.SWEEP)

        scale_f = max(1e-30, float(np.max(np.abs(rt.flux_up))))
        scale_h = max(1e-30, float(np.max(np.abs(rt.heating))))

        td_f = float(np.max(np.abs(rt.flux_up - rd.flux_up))) / scale_f
        ts_f = float(np.max(np.abs(rt.flux_up - rs.flux_up))) / scale_f
        td_h = float(np.max(np.abs(rt.heating - rd.heating))) / scale_h
        ts_h = float(np.max(np.abs(rt.heating - rs.heating))) / scale_h

        td_flux.append(max(td_f, DISPLAY_FLOOR))
        ts_flux.append(max(ts_f, DISPLAY_FLOOR))
        td_heat.append(max(td_h, DISPLAY_FLOOR))
        ts_heat.append(max(ts_h, DISPLAY_FLOOR))

    ax = axes[0]
    ax.semilogy(Ns, td_flux, 'ro-', label='Thomas–Dense', ms=5)
    ax.semilogy(Ns, ts_flux, 'bs-', label='Thomas–Sweep', ms=5)
    ax.axhline(1e-12, color='k', ls='--', lw=0.8, label='Gate (10⁻¹²)')
    ax.axhline(DISPLAY_FLOOR, color='gray', ls=':', lw=0.8, label=f'Display floor ({DISPLAY_FLOOR:.0e})')
    ax.set_xlabel("N layers"); ax.set_ylabel("Normalized flux difference")
    ax.set_title("Flux agreement"); ax.legend(fontsize=8)

    n_exact_td = sum(1 for v in td_flux if v <= DISPLAY_FLOOR)
    n_exact_ts = sum(1 for v in ts_flux if v <= DISPLAY_FLOOR)
    ax.annotate(f"Thomas–Dense exact: {n_exact_td}/{len(Ns)}\nThomas–Sweep exact: {n_exact_ts}/{len(Ns)}",
                xy=(0.02, 0.02), xycoords='axes fraction', fontsize=7,
                verticalalignment='bottom', bbox=dict(boxstyle='round', fc='wheat', alpha=0.5))

    ax = axes[1]
    ax.semilogy(Ns, td_heat, 'ro-', label='Thomas–Dense', ms=5)
    ax.semilogy(Ns, ts_heat, 'bs-', label='Thomas–Sweep', ms=5)
    ax.axhline(1e-12, color='k', ls='--', lw=0.8, label='Gate (10⁻¹²)')
    ax.axhline(DISPLAY_FLOOR, color='gray', ls=':', lw=0.8, label=f'Display floor ({DISPLAY_FLOOR:.0e})')
    ax.set_xlabel("N layers"); ax.set_ylabel("Normalized heating difference")
    ax.set_title("Heating agreement"); ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT / "fig02_three_route_agreement.png", dpi=150)
    plt.close(fig)
    print(f"  → {OUT / 'fig02_three_route_agreement.png'}")


# ── Figure 3: Grid convergence ──────────────────────────────────────
def fig3_convergence():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Stage 3 — Nonisothermal grid convergence (vs N_ref=4096)", fontsize=13)

    kappa0 = 0.02; g = 10.0; P_bot, P_top = 1e6, 1e4
    T_bot, T_top = 3000.0, 1500.0
    B_bot = STEFAN_BOLTZMANN * T_bot**4

    Ns = [4, 8, 16, 32, 64, 128, 256, 512]

    N_ref = 4096
    r_ref, _, _ = _run(N_ref, T_bot, T_top, kappa0, g, P_bot, P_top, 50.0, B_bot)
    fd_ref = float(r_ref.flux_down[0, 0])
    fu_ref = float(r_ref.flux_up[0, N_ref])

    # reference sensitivity: compare N_ref=4096 vs 8192
    N_ref2 = 8192
    r_ref2, _, _ = _run(N_ref2, T_bot, T_top, kappa0, g, P_bot, P_top, 50.0, B_bot)
    fd_ref2 = float(r_ref2.flux_down[0, 0])
    ref_sensitivity = abs(fd_ref - fd_ref2)

    err_fd, err_fu = [], []
    for n in Ns:
        r, _, _ = _run(n, T_bot, T_top, kappa0, g, P_bot, P_top, 50.0, B_bot)
        err_fd.append(abs(float(r.flux_down[0, 0]) - fd_ref))
        err_fu.append(abs(float(r.flux_up[0, n]) - fu_ref))

    # fit order from plotted data only (last two of Ns)
    order_fd = np.log(err_fd[-2] / err_fd[-1]) / np.log(Ns[-1] / Ns[-2])

    ax = axes[0]
    ax.loglog(Ns, err_fd, 'ro-', ms=5, label='|F↓[0] − ref|')
    ax.loglog(Ns, err_fu, 'bs-', ms=5, label='|F↑[N] − ref|')
    ref_line = err_fd[-1] * (np.array(Ns, dtype=float) / Ns[-1]) ** (-1)
    ax.loglog(Ns, ref_line, 'k--', lw=0.8, label=f'O(1/N), fitted={order_fd:.2f}')
    ax.axhline(ref_sensitivity, color='gray', ls=':', lw=0.8,
               label=f'Ref sensitivity (4096 vs 8192) = {ref_sensitivity:.1e}')
    ax.set_xlabel("N"); ax.set_ylabel("Absolute error [W m⁻²]")
    ax.set_title("Pressure-spaced grid"); ax.legend(fontsize=7)

    # τ-spaced
    kappa0b = 0.05; P_botb, P_topb = 5e5, 1e3
    T_botb, T_topb = 2500.0, 1200.0
    B_botb = STEFAN_BOLTZMANN * T_botb**4

    r_reft, _, _ = _run(4096, T_botb, T_topb, kappa0b, g, P_botb, P_topb, 0.0, B_botb)
    fd_reft = float(r_reft.flux_down[0, 0])
    fu_reft = float(r_reft.flux_up[0, 4096])

    err_fd_t, err_fu_t = [], []
    for n in Ns:
        r, _, _ = _run(n, T_botb, T_topb, kappa0b, g, P_botb, P_topb, 0.0, B_botb)
        err_fd_t.append(abs(float(r.flux_down[0, 0]) - fd_reft))
        err_fu_t.append(abs(float(r.flux_up[0, n]) - fu_reft))

    order_tau = np.log(err_fu_t[-2] / err_fu_t[-1]) / np.log(Ns[-1] / Ns[-2])

    ax = axes[1]
    ax.loglog(Ns, err_fd_t, 'ro-', ms=5, label='|F↓[0] − ref|')
    ax.loglog(Ns, err_fu_t, 'bs-', ms=5, label='|F↑[N] − ref|')
    ref_t = err_fu_t[-1] * (np.array(Ns, dtype=float) / Ns[-1]) ** (-1)
    ax.loglog(Ns, ref_t, 'k--', lw=0.8, label=f'O(1/N), fitted={order_tau:.2f}')
    ax.set_xlabel("N"); ax.set_ylabel("Absolute error [W m⁻²]")
    ax.set_title("τ-spaced grid"); ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(OUT / "fig03_grid_convergence.png", dpi=150)
    plt.close(fig)
    print(f"  → {OUT / 'fig03_grid_convergence.png'}")


# ── Figure 4: Energy closure ────────────────────────────────────────
def fig4_energy_closure():
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Stage 3 — Energy closure (layer 0 = bottom, layer N−1 = top)", fontsize=13)

    n = 20
    temp = np.linspace(1500.0, 2500.0, n)
    mp = np.full(n, 500.0)
    kap = np.full((1, n), 0.03)
    w = np.array([1.0])
    B = STEFAN_BOLTZMANN * temp ** 4
    B_bot = B[0]
    r = radiation_core(temp, mp, kap, w, np.array([100.0]), np.array([B_bot]), D)

    # layer heating from flux divergence
    ax = axes[0, 0]
    ax.bar(range(n), r.heating, color='steelblue', alpha=0.8)
    ax.set_xlabel("Layer (0=bottom)"); ax.set_ylabel("dh/dt [W kg⁻¹]")
    ax.set_title("Flux-divergence heating")

    # independent absorption-emission with residual inset
    ax = axes[0, 1]
    ef = 1.0 - r.transmissivity[0]
    q_ae = np.array([ef[i] * (r.flux_up[0, i] + r.flux_down[0, i + 1] - 2.0 * B[i]) for i in range(n)])
    q_div = np.array([r.flux_net[i] - r.flux_net[i + 1] for i in range(n)])
    ax.bar(np.arange(n) - 0.15, q_ae / mp, width=0.3, label='Abs-Emit Q/(Δm)', color='coral', alpha=0.8)
    ax.bar(np.arange(n) + 0.15, q_div / mp, width=0.3, label='Flux-div Q/(Δm)', color='steelblue', alpha=0.8)
    ax.set_xlabel("Layer (0=bottom)"); ax.set_ylabel("dh/dt [W kg⁻¹]")
    ax.set_title("Abs-emit vs flux div"); ax.legend(fontsize=8)

    # layer identity residual
    layer_resids = np.abs(q_ae - q_div)
    q_scale = np.maximum(np.abs(q_ae), np.abs(q_div))
    q_scale = np.where(q_scale > 0, q_scale, 1.0)
    layer_norm_resids = layer_resids / q_scale
    max_layer_err = float(np.max(layer_norm_resids))

    ax_inset = ax.inset_axes([0.55, 0.55, 0.42, 0.40])
    ax_inset.semilogy(range(n), np.maximum(layer_norm_resids, 1e-17), 'k.-', ms=3)
    ax_inset.axhline(1e-12, color='r', ls='--', lw=0.6)
    ax_inset.set_title(f"Layer identity residual\nmax = {max_layer_err:.2e}", fontsize=7)
    ax_inset.set_xlabel("Layer", fontsize=6); ax_inset.tick_params(labelsize=6)

    # cumulative column heating
    ax = axes[1, 0]
    cumulative = np.cumsum(mp * r.heating)
    ax.plot(range(1, n + 1), cumulative, 'k-o', ms=3)
    total_boundary = float(r.flux_net[0] - r.flux_net[n])
    ax.axhline(total_boundary, color='r', ls='--', label=f'F_net[0]−F_net[N]={total_boundary:.1f}')
    ax.set_xlabel("Layers included (bottom up)"); ax.set_ylabel("Σ Δm·Q [W m⁻²]")
    ax.set_title("Cumulative column heating"); ax.legend(fontsize=8)

    # global residual vs N
    ax = axes[1, 1]
    Ns = [2, 5, 10, 20, 50, 100]
    resids = []
    for nn in Ns:
        temp_n = np.linspace(1500.0, 2500.0, nn)
        mp_n = np.full(nn, 500.0)
        kap_n = np.full((1, nn), 0.03)
        B_n = STEFAN_BOLTZMANN * temp_n[0] ** 4
        rr = radiation_core(temp_n, mp_n, kap_n, w, np.array([100.0]), np.array([B_n]), D)
        lhs = float(np.sum(mp_n * rr.heating))
        rhs = float(rr.flux_net[0] - rr.flux_net[nn])
        scale = max(1e-30, abs(rhs))
        resids.append(abs(lhs - rhs) / scale)
    ax.semilogy(Ns, [max(rv, DISPLAY_FLOOR) for rv in resids], 'ko-', ms=5)
    ax.axhline(1e-12, color='r', ls='--', label='Gate (10⁻¹²)')
    ax.axhline(DISPLAY_FLOOR, color='gray', ls=':', lw=0.8, label=f'Display floor ({DISPLAY_FLOOR:.0e})')
    ax.set_xlabel("N"); ax.set_ylabel("Normalized residual")
    ax.set_title("Telescoping residual"); ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT / "fig04_energy_closure.png", dpi=150)
    plt.close(fig)
    print(f"  → {OUT / 'fig04_energy_closure.png'}")


# ── Figure 5: JAX parity ────────────────────────────────────────────
def fig5_jax_parity():
    try:
        os.environ["JAX_ENABLE_X64"] = "True"
        import jax
        import jax.numpy as jnp
        from convection_mlt.radiation_jax import radiation_core_jax
    except ImportError:
        print("  JAX not available — skipping fig5")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Stage 3 — NumPy/JAX float64 parity", fontsize=13)

    fields = ["transmissivity", "flux_up", "flux_down", "flux_net", "heating"]

    # uniform column
    n = 20
    temp = np.full(n, 1500.0); mp = np.full(n, 1000.0)
    kap = np.full((1, n), 0.05); w = np.array([1.0])
    top, bot = np.array([200.0]), np.array([600.0])

    r_np = radiation_core(temp, mp, kap, w, top, bot, D)
    j = lambda a: jnp.array(a, dtype=jnp.float64)
    r_jax_eager = radiation_core_jax(j(temp), j(mp), j(kap), j(w), j(top), j(bot), D)
    jitted = jax.jit(radiation_core_jax, static_argnames=("diffusivity_factor",))
    r_jax_jit = jitted(j(temp), j(mp), j(kap), j(w), j(top), j(bot), diffusivity_factor=D)

    ax = axes[0]
    x = np.arange(len(fields))
    eager_diffs = []
    jit_diffs = []
    for f in fields:
        np_v = getattr(r_np, f)
        jax_e = np.asarray(getattr(r_jax_eager, f))
        jax_j = np.asarray(getattr(r_jax_jit, f))
        scale = max(1e-30, float(np.max(np.abs(np_v))))
        eager_diffs.append(max(float(np.max(np.abs(np_v - jax_e))) / scale, DISPLAY_FLOOR))
        jit_diffs.append(max(float(np.max(np.abs(np_v - jax_j))) / scale, DISPLAY_FLOOR))

    ax.bar(x - 0.15, eager_diffs, width=0.3, label='Eager JAX', color='steelblue', alpha=0.8)
    ax.bar(x + 0.15, jit_diffs, width=0.3, label='JIT JAX', color='coral', alpha=0.8)
    ax.set_yscale('log')
    ax.axhline(1e-12, color='k', ls='--', lw=0.8, label='Gate (10⁻¹²)')
    ax.axhline(DISPLAY_FLOOR, color='gray', ls=':', lw=0.8, label=f'Display floor ({DISPLAY_FLOOR:.0e})')
    ax.set_xticks(x); ax.set_xticklabels(fields, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel("Normalized difference"); ax.set_title("Uniform column (eager & JIT)")
    ax.legend(fontsize=7)

    # varied column — max of eager and JIT
    temp_v = np.linspace(1200.0, 3000.0, n)
    mp_v = np.linspace(300.0, 1500.0, n)
    kap_v = np.linspace(0.005, 0.1, n)[np.newaxis, :]
    B_v = STEFAN_BOLTZMANN * temp_v[0] ** 4
    r_np2 = radiation_core(temp_v, mp_v, kap_v, np.array([1.0]),
                           np.array([150.0]), np.array([B_v]), D)
    r_jax2_eager = radiation_core_jax(j(temp_v), j(mp_v), j(kap_v), j(np.array([1.0])),
                                       j(np.array([150.0])), j(np.array([B_v])), D)
    r_jax2_jit = jitted(j(temp_v), j(mp_v), j(kap_v), j(np.array([1.0])),
                         j(np.array([150.0])), j(np.array([B_v])), diffusivity_factor=D)

    ax = axes[1]
    eager_v, jit_v = [], []
    for f in fields:
        np_val = getattr(r_np2, f)
        scale = max(1e-30, float(np.max(np.abs(np_val))))
        de = float(np.max(np.abs(np_val - np.asarray(getattr(r_jax2_eager, f))))) / scale
        dj = float(np.max(np.abs(np_val - np.asarray(getattr(r_jax2_jit, f))))) / scale
        eager_v.append(max(de, DISPLAY_FLOOR))
        jit_v.append(max(dj, DISPLAY_FLOOR))

    ax.bar(x - 0.15, eager_v, width=0.3, label='Eager JAX', color='steelblue', alpha=0.8)
    ax.bar(x + 0.15, jit_v, width=0.3, label='JIT JAX', color='coral', alpha=0.8)
    ax.set_yscale('log')
    ax.axhline(1e-12, color='k', ls='--', lw=0.8, label='Gate (10⁻¹²)')
    ax.axhline(DISPLAY_FLOOR, color='gray', ls=':', lw=0.8, label=f'Display floor ({DISPLAY_FLOOR:.0e})')
    ax.set_xticks(x); ax.set_xticklabels(fields, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel("Normalized difference"); ax.set_title("Varied column (eager & JIT)")
    ax.legend(fontsize=7)

    info_text = (f"JAX {jax.__version__}, backend={jax.default_backend()}, "
                 f"x64={jax.config.x64_enabled}, dtype={r_jax_eager.flux_up.dtype}")
    fig.text(0.5, 0.01, info_text, ha='center', fontsize=8, style='italic')

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(OUT / "fig05_jax_parity.png", dpi=150)
    plt.close(fig)
    print(f"  → {OUT / 'fig05_jax_parity.png'}")


if __name__ == "__main__":
    print("Stage 3 evidence figures:")
    fig1_analytic_limits()
    fig2_three_route()
    fig3_convergence()
    fig4_energy_closure()
    fig5_jax_parity()
    print("Done.")
