"""Global column profile figure: T, relative error, θ, and F_conv vs pressure."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import (
    DATA_DIR,
    apply_style,
    exact_zero_display,
    pressure_axis,
    read_json,
    save_figure,
)

FLUX_FLOOR = 1.0e-30


def _require_profile_array(name: str, values) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains nonfinite values")
    return array


def main() -> None:
    path = DATA_DIR / "global_profile.json"
    if not path.exists():
        raise SystemExit(f"missing data: {path}")

    data = read_json(path)
    case = data["case"]
    pressure = _require_profile_array("pressure_centres_pa", case["pressure_centres_pa"])
    pressure_edges = _require_profile_array("pressure_edges_pa", case["pressure_edges_pa"])
    reference = _require_profile_array("reference_temperature_k", data["reference_temperature_k"])
    profiles = data.get("profiles", [])
    if not profiles:
        raise ValueError("global_profile.json: profiles[] is empty")

    apply_style()
    fig, axes = plt.subplots(1, 4, figsize=(14, 6), sharey=True)
    colors = plt.cm.viridis(np.linspace(0.15, 0.95, len(profiles)))
    boundary_zero_hits: list[tuple[float, float]] = []

    for idx, snap in enumerate(profiles):
        temperature = _require_profile_array("temperature_k", snap["temperature_k"])
        theta = _require_profile_array("potential_temperature_k", snap["potential_temperature_k"])
        flux = _require_profile_array("flux_w_m2", snap["flux_w_m2"])
        if len(temperature) != len(pressure):
            raise ValueError(f"profile {idx}: temperature length mismatch")

        rel = (temperature - reference) / reference
        label = f"step {snap['accepted_step']}, t={snap['simulated_time_s']:.3g} s"

        axes[0].plot(temperature, pressure, color=colors[idx], lw=1.2, label=label)
        axes[1].plot(rel, pressure, color=colors[idx], lw=1.2)
        axes[2].plot(theta, pressure, color=colors[idx], lw=1.2)

        flux_plot = np.empty_like(flux)
        for layer, value in enumerate(flux):
            display, is_exact_zero = exact_zero_display(float(value), FLUX_FLOOR)
            flux_plot[layer] = display
            if is_exact_zero and layer in (0, len(flux) - 1):
                p_mark = float(pressure_edges[0] if layer == 0 else pressure_edges[-1])
                boundary_zero_hits.append((display, p_mark))

        if len(flux) == len(pressure) + 2:
            flux_y = pressure
            flux_values = flux_plot[1:-1]
        elif len(flux) == len(pressure_edges):
            flux_y = pressure_edges
            flux_values = flux_plot
        elif len(flux) == len(pressure):
            flux_y = pressure
            flux_values = flux_plot
        else:
            raise ValueError(
                f"profile {idx}: flux length {len(flux)} incompatible with "
                f"N={len(pressure)} centres / {len(pressure_edges)} edges"
            )

        axes[3].plot(flux_values, flux_y, color=colors[idx], lw=1.2)

    axes[0].plot(reference, pressure, "k--", lw=1.5, alpha=0.75, label="analytic T_ref")

    axes[0].set_xlabel("T [K]")
    axes[1].set_xlabel("(T − T_ref) / T_ref")
    axes[2].set_xlabel("θ [K]")
    axes[3].set_xlabel("F_conv [W m⁻²]")
    axes[3].set_xscale("log")

    for ax in axes:
        pressure_axis(ax)

    if boundary_zero_hits:
        for xval, yval in boundary_zero_hits:
            axes[3].plot(xval, yval, "x", color="crimson", ms=7, mew=1.5, zorder=5)
        axes[3].text(
            FLUX_FLOOR,
            pressure_edges[-1],
            f"exact zero (floor={FLUX_FLOOR:.0e})",
            fontsize=7,
            color="crimson",
            va="top",
        )

    axes[0].legend(loc="best", fontsize=6)
    outcome = data["outcome"]
    fig.suptitle(
        f"Global column trace (N={case['n_layers']}, α={case['alpha']}, "
        f"status={outcome['status']})",
        fontsize=12,
    )
    out = save_figure(fig, "01_global_column.png")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
