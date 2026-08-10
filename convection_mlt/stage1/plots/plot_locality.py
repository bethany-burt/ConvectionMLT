"""Localized barrier figure: piecewise θ reference, regions, merges, transfer."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import (
    DATA_DIR,
    apply_style,
    pressure_axis,
    read_json,
    require_finite,
    save_figure,
)


def _array(name: str, values) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains nonfinite values")
    return array


def _shade_regions(ax, pressure, labels, alpha=0.12) -> None:
    labels = np.asarray(labels, dtype=int)
    if labels.size != pressure.size:
        raise ValueError("region_labels length mismatch with pressure")
    cmap = plt.cm.tab10
    for region_id in np.unique(labels):
        mask = labels == region_id
        if not np.any(mask):
            continue
        ax.axhspan(
            float(np.min(pressure[mask])),
            float(np.max(pressure[mask])),
            color=cmap(int(region_id) % 10),
            alpha=alpha,
        )


def _annotate_regions(ax, pressure, labels, prefix: str) -> None:
    for region_id in np.unique(labels):
        mask = labels == region_id
        p_mid = float(np.median(pressure[mask]))
        ax.text(
            0.02,
            p_mid,
            f"{prefix}{int(region_id)}",
            transform=ax.get_yaxis_transform(),
            fontsize=7,
            alpha=0.85,
            va="center",
        )


def _plot_piecewise_theta_ref(ax, pressure, theta_ref, labels, **style) -> None:
    """Draw θ_ref as vertical segments within each region (no cross-region joins)."""
    # Fix style mutation bug: don't pop from shared kwargs across regions.
    labels = np.asarray(labels, dtype=int)
    theta_ref = np.asarray(theta_ref, dtype=float)
    pressure = np.asarray(pressure, dtype=float)
    label = style.get("label")
    draw_style = {k: v for k, v in style.items() if k != "label"}
    first = True
    for region_id in np.unique(labels):
        mask = labels == region_id
        p = pressure[mask]
        th = float(np.mean(theta_ref[mask]))
        ax.plot(
            [th, th],
            [float(np.min(p)), float(np.max(p))],
            label=(label if first else None),
            **draw_style,
        )
        first = False


def main() -> None:
    path = DATA_DIR / "locality.json"
    if not path.exists():
        raise SystemExit(f"missing data: {path}")

    data = read_json(path)
    case = data["case"]
    pressure = _array("pressure_centres_pa", case["pressure_centres_pa"])
    pressure_edges = _array(
        "pressure_edges_pa",
        case.get("pressure_edges_pa", []),
    ) if case.get("pressure_edges_pa") else None

    initial_theta = _array(
        "initial_potential_temperature_k", data["initial_potential_temperature_k"]
    )
    final_theta = _array(
        "final_potential_temperature_k", data["final_potential_temperature_k"]
    )
    # Prefer explicit θ references; fall back only if regenerating older data.
    if "initial_piecewise_reference_potential_temperature_k" in data:
        initial_theta_ref = _array(
            "initial_piecewise_reference_potential_temperature_k",
            data["initial_piecewise_reference_potential_temperature_k"],
        )
        final_theta_ref = _array(
            "final_piecewise_reference_potential_temperature_k",
            data["final_piecewise_reference_potential_temperature_k"],
        )
    else:
        raise ValueError(
            "locality.json missing piecewise θ references; regenerate with "
            "generate_data.py --only locality"
        )
    residuals = _array("final_piecewise_residuals", data["final_piecewise_residuals"])

    initial_labels = np.asarray(case["initial_region_labels"], dtype=int)
    final_labels = np.asarray(data["outcome"]["region_labels"], dtype=int)
    n_initial = int(case.get("n_initial_regions", np.unique(initial_labels).size))
    n_final = int(np.unique(final_labels).size)
    merge_events = data.get("merge_events", [])
    transfer_tol = require_finite(
        "transfer_merge_tolerance",
        case.get("transfer_merge_tolerance", 1.0e-9),
    )

    if "normalized_unmerged_transfer" in data:
        fractions = _array(
            "normalized_unmerged_transfer", data["normalized_unmerged_transfer"]
        )
    else:
        raise ValueError(
            "locality.json missing normalized_unmerged_transfer; regenerate data"
        )
    if pressure_edges is None or len(pressure_edges) != len(fractions):
        raise ValueError("pressure_edges_pa must align with normalized transfer")

    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 6), sharey=True)

    ax0 = axes[0]
    _shade_regions(ax0, pressure, initial_labels, alpha=0.10)
    ax0.plot(initial_theta, pressure, "C0", lw=1.8, label="initial θ")
    ax0.plot(final_theta, pressure, "C1", lw=1.8, label="final θ")
    _plot_piecewise_theta_ref(
        ax0,
        pressure,
        initial_theta_ref,
        initial_labels,
        color="C0",
        ls="--",
        lw=1.4,
        alpha=0.9,
        label="initial θ_ref (piecewise)",
    )
    _plot_piecewise_theta_ref(
        ax0,
        pressure,
        final_theta_ref,
        final_labels,
        color="C1",
        ls="--",
        lw=1.4,
        alpha=0.9,
        label="final θ_ref (piecewise)",
    )
    ax0.set_xlabel("θ [K]")
    pressure_axis(ax0)
    ax0.legend(fontsize=7, loc="best")
    ax0.set_title(
        f"Potential temperature\n"
        f"(initial regions={n_initial}, final={n_final})"
    )
    _annotate_regions(ax0, pressure, initial_labels, "init R")

    ax1 = axes[1]
    _shade_regions(ax1, pressure, final_labels, alpha=0.12)
    ax1.plot(residuals, pressure, "C2", lw=1.8)
    ax1.axvline(0.0, color="k", lw=0.8, alpha=0.5)
    if merge_events:
        ax1.text(
            0.98,
            0.02,
            f"{len(merge_events)} merge events\n"
            f"{n_initial}→{n_final} regions",
            transform=ax1.transAxes,
            ha="right",
            va="bottom",
            fontsize=7,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )
    ax1.set_xlabel("T − T_ref,piecewise [K]")
    pressure_axis(ax1)
    ax1.set_title("Piecewise-reference temperature residuals")
    _annotate_regions(ax1, pressure, final_labels, "final R")

    ax2 = axes[2]
    # Plot absolute normalized transfer on log scale; zeros stay at the floor.
    floor = 1.0e-20
    display = np.maximum(np.abs(fractions), floor)
    ax2.plot(display, pressure_edges, "C3", lw=1.8, drawstyle="steps-mid")
    ax2.axvline(
        transfer_tol,
        color="k",
        ls="--",
        lw=1.0,
        label=f"merge threshold={transfer_tol:g}",
    )
    ax2.set_xscale("log")
    ax2.set_xlabel(r"$|Q_j|/H_{\mathrm{adjacent}}$")
    pressure_axis(ax2)
    ax2.set_title("Normalized separating-interface transfer")
    ax2.legend(fontsize=7, loc="best")

    outcome = data["outcome"]
    fig.suptitle(
        f"Localized barrier (N={case['n_layers']}, α={case['alpha']}, "
        f"status={outcome['status']}; "
        f"init R# = initial labels, final R# = post-merge labels)",
        fontsize=11,
    )
    out = save_figure(fig, "02_localized_barrier.png")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
