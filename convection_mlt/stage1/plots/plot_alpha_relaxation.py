"""Alpha relaxation and closure scaling figures."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import (
    DATA_DIR,
    apply_style,
    fit_log_log_slope,
    read_json,
    require_finite,
    save_figure,
)


def _trajectory_history(traj: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    history = traj.get("history", [])
    if not history:
        raise ValueError(f"α={traj['alpha']}: empty trajectory history")
    t = []
    theta_rms = []
    dt_accepted = []
    rejections = []
    for item in history:
        t.append(require_finite("simulated_time_s", item["simulated_time_s"]))
        theta_rms.append(require_finite("theta_rms", item["theta_rms"]))
        dt_accepted.append(require_finite("dt_accepted_s", item["dt_accepted_s"]))
        rejections.append(float(item.get("rejections_this_step", 0)))
    return (
        np.asarray(t),
        np.asarray(theta_rms),
        np.asarray(dt_accepted),
        np.asarray(rejections),
    )


def main() -> None:
    path = DATA_DIR / "alpha_trajectories.json"
    if not path.exists():
        raise SystemExit(f"missing data: {path}")

    data = read_json(path)
    trajectories = data.get("trajectories", [])
    closure_scaling = data.get("closure_scaling", [])
    if not trajectories:
        raise ValueError("alpha_trajectories.json: trajectories[] is empty")
    if not closure_scaling:
        raise ValueError("alpha_trajectories.json: closure_scaling[] is empty")

    threshold = require_finite("threshold_theta_rms", data["threshold_theta_rms"])
    apply_style()

    # --- Figure 5: relaxation ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax_t, ax_scaled = axes[0]
    ax_steps, ax_dt = axes[1]
    cmap = plt.cm.tab10

    threshold_times: list[tuple[float, float | None]] = []

    for idx, traj in enumerate(trajectories):
        alpha = require_finite("alpha", traj["alpha"])
        if alpha <= 0.0:
            raise ValueError("alpha trajectories must exclude α=0")
        color = cmap(idx % 10)
        t, theta_rms, dt_acc, rejections = _trajectory_history(traj)

        ax_t.semilogy(t, theta_rms, color=color, lw=1.5, label=f"α={alpha:g}")
        scaled_t = alpha**2 * t
        ax_scaled.semilogy(scaled_t, theta_rms, color=color, lw=1.5, label=f"α={alpha:g}")

        accepted_mask = rejections == 0
        rejected_mask = rejections > 0
        ax_steps.scatter(
            t[accepted_mask],
            np.arange(len(t))[accepted_mask],
            color=color,
            marker="o",
            s=18,
            alpha=0.8,
        )
        ax_steps.scatter(
            t[rejected_mask],
            np.arange(len(t))[rejected_mask],
            color=color,
            marker="x",
            s=28,
        )

        ax_dt.semilogy(t, dt_acc, color=color, lw=1.2, label=f"α={alpha:g}")

        t_thresh = traj.get("threshold_time_s")
        if t_thresh is not None:
            require_finite("threshold_time_s", t_thresh)
        threshold_times.append((alpha, t_thresh))

    ax_t.axhline(threshold, color="k", ls="--", lw=0.9, alpha=0.6, label="threshold")
    ax_scaled.axhline(threshold, color="k", ls="--", lw=0.9, alpha=0.6)
    ax_t.set_xlabel("t [s]")
    ax_t.set_ylabel("θ RMS [K]")
    ax_t.legend(fontsize=7)
    ax_t.set_title("θ RMS vs time")

    ax_scaled.set_xlabel("α² t [s]")
    ax_scaled.set_ylabel("θ RMS [K]")
    ax_scaled.legend(fontsize=7)
    ax_scaled.set_title("θ RMS vs α²t (α⁻² scaling guide)")

    # α⁻² guide for threshold crossing time
    alphas_with_time = [(a, tt) for a, tt in threshold_times if tt is not None]
    if len(alphas_with_time) >= 2:
        a_arr = np.array([a for a, _ in alphas_with_time])
        t_arr = np.array([tt for _, tt in alphas_with_time])
        fit = fit_log_log_slope(a_arr, t_arr)
        inv_sq = 1.0 / a_arr**2
        coef = np.polyfit(inv_sq, t_arr, 1)
        inv_line = np.linspace(inv_sq.min(), inv_sq.max(), 50)
        guide = np.polyval(coef, inv_line)
        inset_ax = ax_t.inset_axes([0.55, 0.55, 0.42, 0.42])
        inset_ax.plot(inv_sq, t_arr, "ko", ms=5)
        inset_ax.plot(
            inv_line,
            guide,
            "r--",
            lw=1.0,
            label=f"Fit: t_threshold ∝ α^{{{fit['slope']:.2f}}}",
        )
        inset_ax.set_xlabel("α⁻²")
        inset_ax.set_ylabel("threshold time [s]")
        inset_ax.legend(fontsize=5.5)
        inset_ax.set_title("threshold time scaling", fontsize=8)

    total_rejections = 0
    for traj in trajectories:
        total_rejections += int(
            sum(float(item.get("rejections_this_step", 0)) for item in traj["history"])
        )

    ax_steps.set_xlabel("t [s]")
    ax_steps.set_ylabel("accepted step index")
    if total_rejections == 0:
        ax_steps.set_title("Accepted-step accumulation (0 rejected)")
    else:
        ax_steps.set_title("Accepted (•) vs rejected (×) steps")

    ax_dt.set_xlabel("t [s]")
    ax_dt.set_ylabel("accepted Δt [s]")
    ax_dt.legend(fontsize=7)
    ax_dt.set_title("Accepted timestep history")

    fig.suptitle(f"Alpha relaxation (N={data.get('n_layers', '?')})", fontsize=12)
    out = save_figure(fig, "05_alpha_relaxation.png")
    plt.close(fig)
    print(f"wrote {out}")

    # --- Figure 5b: closure scaling ---
    alphas = np.array([require_finite("alpha", c["alpha"]) for c in closure_scaling])
    velocities = np.array([require_finite("mean_velocity", c["mean_velocity"]) for c in closure_scaling])
    fluxes = np.array([require_finite("mean_flux", c["mean_flux"]) for c in closure_scaling])
    kzz = np.array([require_finite("mean_kzz", c["mean_kzz"]) for c in closure_scaling])

    fig2, axs = plt.subplots(1, 3, figsize=(12, 4))
    panels = (
        (velocities, "⟨w⟩ [m s⁻¹]", "w ∝ α", 1.0),
        (fluxes, "⟨F_conv⟩ [W m⁻²]", "F_conv ∝ α²", 2.0),
        (kzz, "⟨K_zz⟩ [m² s⁻¹]", "K_zz ∝ α²", 2.0),
    )
    for ax, (y, ylabel, title, expected_slope) in zip(axs, panels):
        positive = (alphas > 0) & (y > 0)
        if np.count_nonzero(positive) < 2:
            raise ValueError(f"closure scaling {title}: insufficient positive points")
        fit = fit_log_log_slope(alphas[positive], y[positive])
        ax.loglog(alphas[positive], y[positive], "o", ms=7, label="fixed-state mean")
        ref_alpha = alphas[positive]
        ref_y = y[positive][0] * (ref_alpha / ref_alpha[0]) ** expected_slope
        ax.loglog(ref_alpha, ref_y, "--", color="gray", lw=1.0, label=f"guide ∝ α^{expected_slope:g}")
        ax.set_xlabel("α")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} (fit slope={fit['slope']:.2f})")
        ax.legend(fontsize=7)

    fig2.suptitle("Fixed-state closure scaling (log–log)", fontsize=12)
    out2 = save_figure(fig2, "05b_closure_scaling.png")
    plt.close(fig2)
    print(f"wrote {out2}")


if __name__ == "__main__":
    main()
