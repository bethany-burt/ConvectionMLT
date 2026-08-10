"""Resolution scaling figure: timing, steps, and rejection fraction vs N."""

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


def _loglog_guides(ax, ns: np.ndarray, y: np.ndarray, guides: list[tuple[float, str]]) -> None:
    fit = fit_log_log_slope(ns, y)
    if fit["n_points"] >= 2 and np.all(y > 0):
        ref = y[0] * (ns / ns[0]) ** fit["slope"]
        ax.loglog(ns, ref, "--", color="C1", lw=1.0, label=f"fit ∝ N^{fit['slope']:.2f}")
    for slope, label in guides:
        guide = y[0] * (ns / ns[0]) ** slope
        ax.loglog(ns, guide, ":", color="gray", lw=0.9, alpha=0.7, label=f"provisional {label}")


def _plot_series(ax, ns: np.ndarray, y: np.ndarray, ylabel: str, title: str, guides) -> None:
    ax.loglog(ns, y, "o-", lw=1.3, ms=6)
    _loglog_guides(ax, ns, y, guides)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("N")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=6)


def _plot_timing_iqr(
    ax,
    ns: np.ndarray,
    medians: np.ndarray,
    q25: np.ndarray,
    q75: np.ndarray,
    ylabel: str,
    title: str,
    guides,
) -> None:
    lower = np.maximum(medians - q25, 0.0)
    upper = np.maximum(q75 - medians, 0.0)
    ax.errorbar(ns, medians, yerr=[lower, upper], fmt="o-", capsize=3, lw=1.3)
    _loglog_guides(ax, ns, medians, guides)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("N")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=6)


def main() -> None:
    path = DATA_DIR / "resolution_scaling.json"
    if not path.exists():
        raise SystemExit(f"missing data: {path}")

    data = read_json(path)
    records = data.get("records", [])
    if not records:
        raise ValueError("resolution_scaling.json: records[] is empty")

    ns = []
    medians = []
    q25 = []
    q75 = []
    steps = []
    wall_per_step = []
    total_wall = []
    reject_frac = []
    min_dt = []
    median_dt = []

    for record in records:
        n = int(record["n_layers"])
        ns.append(float(n))
        timing = record["timing"]
        medians.append(require_finite("median_s", timing["median_s"]))
        q25.append(require_finite("q25_s", timing["q25_s"]))
        q75.append(require_finite("q75_s", timing["q75_s"]))
        outcome = record["outcome"]
        steps.append(float(int(outcome["steps"])))
        wall_per_step.append(require_finite("wall_time_per_step_s", record["wall_time_per_step_s"]))
        total_wall.append(medians[-1])
        reject_frac.append(require_finite("rejection_fraction", record["rejection_fraction"]))
        min_dt.append(
            require_finite("min_accepted_dt_s", outcome["min_accepted_dt_s"])
        )
        median_dt.append(
            require_finite(
                "median_accepted_dt_s", outcome["median_accepted_dt_s"]
            )
        )

    ns_arr = np.asarray(ns)
    apply_style()
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    ax_dt_min, ax_dt_med, ax_steps = axes[0]
    ax_wps, ax_total, ax_reject = axes[1]

    _plot_series(
        ax_dt_min,
        ns_arr,
        np.asarray(min_dt),
        "min accepted Δt [s]",
        "Min accepted Δt",
        [(-2, "N⁻²")],
    )
    _plot_series(
        ax_dt_med,
        ns_arr,
        np.asarray(median_dt),
        "median accepted Δt [s]",
        "Median accepted Δt",
        [(-2, "N⁻²")],
    )

    _plot_series(
        ax_steps,
        ns_arr,
        np.asarray(steps),
        "accepted steps",
        "Steps to common finite endpoint",
        [(1, "O(N)"), (3, "O(N³)")],
    )

    _plot_series(
        ax_wps,
        ns_arr,
        np.asarray(wall_per_step),
        "wall time / step [s]",
        "Wall time per step (median)",
        [(1, "O(N)"), (3, "O(N³)")],
    )
    _plot_timing_iqr(
        ax_total,
        ns_arr,
        np.asarray(total_wall),
        np.asarray(q25),
        np.asarray(q75),
        "total wall time [s]",
        "Total wall time (median/IQR)",
        [(1, "O(N)"), (3, "O(N³)")],
    )

    ax_reject.plot(ns_arr, reject_frac, "o-", lw=1.3)
    ax_reject.set_xscale("log", base=2)
    ax_reject.set_xlabel("N")
    ax_reject.set_ylabel("rejection fraction")
    ax_reject.set_title("Rejection fraction")
    ax_reject.set_ylim(0.0, min(1.0, max(reject_frac) * 1.1 + 0.01))

    fig.suptitle(
        "Resolution scaling (provisional exponent guides; NOT required convergence order)",
        fontsize=12,
    )
    out = save_figure(fig, "06_resolution_scaling.png")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
