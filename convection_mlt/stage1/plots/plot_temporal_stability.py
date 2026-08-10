"""Temporal stability figures: controller behaviour (07a) and convergence (07b)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from common import (
    DATA_DIR,
    apply_style,
    fit_log_log_slope,
    read_json,
    require_finite,
    save_figure,
)

OUTCOME_COLORS = {
    "direct_acceptance": "#2ca02c",
    "adaptive_acceptance_after_backtracking": "#98df8a",
    "fixed_step_failure": "#d62728",
    "adaptive_failure": "#ff9896",
}
OUTCOME_LABELS = {
    "direct_acceptance": "direct acceptance",
    "adaptive_acceptance_after_backtracking": "adaptive after backtracking",
    "fixed_step_failure": "fixed-step failure",
    "adaptive_failure": "adaptive failure",
}


def _validate_safety_case(case: dict) -> None:
    outcome = case["outcome_class"]
    if outcome not in OUTCOME_COLORS:
        raise ValueError(f"unexpected safety outcome_class: {outcome!r}")


def _adaptive_x_labels(cases: list[dict]) -> tuple[np.ndarray, list[str]]:
    """Categorical positions so the intentional unrecoverable probe is visible."""
    labels = []
    for case in cases:
        c_diff = require_finite("c_diff", case["c_diff"])
        if c_diff >= 1.0e6:
            labels.append("unrecov.\nprobe")
        else:
            labels.append(f"{c_diff:g}")
    return np.arange(len(cases), dtype=float), labels


def _scatter_outcomes_categorical(ax, cases: list[dict], title: str) -> None:
    xs, labels = _adaptive_x_labels(cases)
    for x, case in zip(xs, cases):
        color = OUTCOME_COLORS[case["outcome_class"]]
        ax.scatter(x, 0.0, c=color, s=110, edgecolors="k", linewidths=0.5, zorder=3)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticks([])
    ax.set_xlabel(r"$c_{\mathrm{diff}}$ (dimensionless; probe = unrecoverable)")
    ax.set_title(title)
    ax.set_ylim(-0.7, 0.7)
    ax.set_xlim(-0.6, len(cases) - 0.4)


def _scatter_outcomes_fixed(ax, cases: list[dict], title: str) -> None:
    for case in cases:
        x = np.log10(require_finite("dt_s", case["dt_s"]))
        color = OUTCOME_COLORS[case["outcome_class"]]
        ax.scatter(x, 0.0, c=color, s=110, edgecolors="k", linewidths=0.5, zorder=3)
    ax.set_yticks([])
    ax.set_xlabel(r"$\log_{10}(\Delta t\,[\mathrm{s}])$")
    ax.set_title(title)
    ax.set_ylim(-0.7, 0.7)


def _plot_accepted_margins(ax, cases: list[dict], x_values, xlabel: str, title: str, c_cross: float) -> None:
    xs = []
    ys = []
    for x, case in zip(x_values, cases):
        accepted = case.get("min_accepted_trial_delta_over_epsilon")
        if accepted is None:
            continue
        xs.append(x)
        ys.append(require_finite("min_accepted_trial_delta_over_epsilon", accepted))
    if not xs:
        ax.text(0.5, 0.5, "no accepted trials", transform=ax.transAxes, ha="center")
        ax.set_title(title)
        return
    ax.plot(xs, ys, "o-", ms=7, lw=1.3, color="C0", label="final accepted margin")
    ax.axhline(-c_cross, color="k", ls="--", lw=1.1, label=f"−c_cross={-c_cross:g}")
    ax.axhline(0.0, color="gray", lw=0.6, alpha=0.5)
    # Signed symlog keeps −1 visible while showing large positive safety margins.
    ax.set_yscale("symlog", linthresh=2.0, linscale=0.75)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"min accepted $\delta_{\mathrm{trial}}/\varepsilon_\nabla$")
    ax.set_title(title)
    ax.legend(fontsize=6, loc="best")


def _plot_retry_counts(ax, cases: list[dict], x_values, labels: list[str]) -> None:
    counts = [int(case.get("rejections", 0) or 0) for case in cases]
    colors = [OUTCOME_COLORS[case["outcome_class"]] for case in cases]
    ax.bar(x_values, counts, color=colors, edgecolor="k", linewidth=0.4, width=0.65)
    ax.set_xticks(x_values)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("rejected trials before accept/fail")
    ax.set_xlabel(r"$c_{\mathrm{diff}}$")
    ax.set_title("Adaptive retry counts")
    ax.set_ylim(0.0, max(counts) * 1.15 + 1.0)


def _rejected_extrema_table(ax, cases: list[dict]) -> None:
    ax.axis("off")
    rows = []
    for case in cases:
        mode = case["mode"]
        if mode == "adaptive":
            key = f"c_diff={case['c_diff']:g}"
        else:
            key = f"Δt={case['dt_s']:g} s"
        rejected = case.get("min_rejected_trial_delta_over_epsilon")
        accepted = case.get("min_accepted_trial_delta_over_epsilon")
        rej_text = "—" if rejected is None else f"{float(rejected):.3g}"
        acc_text = "—" if accepted is None else f"{float(accepted):.3g}"
        rows.append(
            [
                key,
                OUTCOME_LABELS[case["outcome_class"]],
                acc_text,
                rej_text,
                str(int(case.get("rejections", 0) or 0)),
            ]
        )
    table = ax.table(
        cellText=rows,
        colLabels=["case", "outcome", "accepted min", "rejected min", "retries"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.05, 1.35)
    ax.set_title("Rejected extrema (table; not forced onto the accepted-margin axis)", fontsize=10)


def _legend_for_present_outcomes(ax, cases: list[dict]) -> None:
    present = {case["outcome_class"] for case in cases}
    handles = [
        mpatches.Patch(color=OUTCOME_COLORS[key], label=OUTCOME_LABELS[key])
        for key in OUTCOME_COLORS
        if key in present
    ]
    ax.legend(handles=handles, fontsize=6, loc="upper left", ncol=2)


def plot_controller(data: dict) -> None:
    safety_cases = data["safety_cases"]
    for case in safety_cases:
        _validate_safety_case(case)

    adaptive_cases = [c for c in safety_cases if c["mode"] == "adaptive"]
    fixed_cases = [c for c in safety_cases if c["mode"] == "fixed_like"]
    if not adaptive_cases:
        raise ValueError("no adaptive safety cases")
    c_cross = require_finite("c_cross", safety_cases[0]["c_cross"])

    apply_style()
    fig = plt.figure(figsize=(13.5, 9.5))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.9, 1.2, 1.35], hspace=0.45, wspace=0.28)

    ax_adapt_map = fig.add_subplot(gs[0, 0])
    ax_fixed_map = fig.add_subplot(gs[0, 1])
    ax_adapt_margin = fig.add_subplot(gs[1, 0])
    ax_retries = fig.add_subplot(gs[1, 1])
    ax_table = fig.add_subplot(gs[2, :])

    adapt_x, adapt_labels = _adaptive_x_labels(adaptive_cases)
    _scatter_outcomes_categorical(
        ax_adapt_map, adaptive_cases, "Adaptive safety-factor outcomes"
    )
    _legend_for_present_outcomes(ax_adapt_map, safety_cases)

    if fixed_cases:
        _scatter_outcomes_fixed(ax_fixed_map, fixed_cases, "Fixed-step probe outcomes")
    else:
        ax_fixed_map.axis("off")

    _plot_accepted_margins(
        ax_adapt_margin,
        adaptive_cases,
        adapt_x,
        r"$c_{\mathrm{diff}}$ index",
        "Adaptive final accepted margins",
        c_cross,
    )
    ax_adapt_margin.set_xticks(adapt_x)
    ax_adapt_margin.set_xticklabels(adapt_labels, fontsize=8)

    _plot_retry_counts(ax_retries, adaptive_cases, adapt_x, adapt_labels)
    _rejected_extrema_table(ax_table, safety_cases)

    recovered = all(
        c["outcome_class"] != "adaptive_failure" or c.get("expected_unstable", False)
        for c in adaptive_cases
        if require_finite("c_diff", c["c_diff"]) < 1.0e6
    )
    recovery_note = (
        "Every routine adaptive overshoot recovered through backtracking; "
        "the unrecoverable probe forces max_rejections=0."
        if recovered
        else "Adaptive recovery was incomplete for one or more routine c_diff values."
    )
    fig.suptitle(
        "07a · Stability-controller behaviour\n"
        f"Accepted trials must satisfy margin ≥ −c_cross. {recovery_note}",
        fontsize=11,
    )
    out = save_figure(fig, "07a_stability_controller.png")
    plt.close(fig)
    print(f"wrote {out}")


def _plot_order_panel(ax, record: dict) -> None:
    n_layers = record["n_layers"]
    alpha = record["alpha"]
    points = record["points"]
    if not points:
        raise ValueError(f"order record N={n_layers}: no points")

    dts = []
    errors = []
    for point in points:
        dt = require_finite("dt_s", point["dt_s"])
        if point.get("status") != "completed":
            continue
        err = point.get("relative_temperature_rms")
        if err is None:
            raise ValueError(
                f"completed point at dt={dt} missing relative_temperature_rms"
            )
        errors.append(require_finite("relative_temperature_rms", err))
        dts.append(dt)

    if len(dts) < 2:
        raise ValueError(f"order record N={n_layers}: insufficient completed points")

    dts_arr = np.asarray(dts)
    err_arr = np.asarray(errors)
    fit_points = [p for p in points if p.get("used_in_fit")]
    fit_dts = [require_finite("dt_s", p["dt_s"]) for p in fit_points]
    fit_errs = [
        require_finite("relative_temperature_rms", p["relative_temperature_rms"])
        for p in fit_points
        if p.get("relative_temperature_rms") is not None
    ]

    ax.loglog(dts_arr, err_arr, "o", ms=6, label="all stable points")
    if fit_points:
        ax.loglog(
            [require_finite("dt_s", p["dt_s"]) for p in fit_points],
            [
                require_finite(
                    "relative_temperature_rms", p["relative_temperature_rms"]
                )
                for p in fit_points
            ],
            "s",
            ms=8,
            mfc="none",
            mec="C1",
            mew=1.5,
            label="asymptotic fit subset",
        )

    if len(fit_dts) >= 2:
        fit = fit_log_log_slope(np.asarray(fit_dts), np.asarray(fit_errs))
        guide = fit_errs[0] * (dts_arr / fit_dts[0]) ** 1.0
        ax.loglog(dts_arr, guide, "--", color="gray", lw=1.0, label="slope 1 guide")
        slope = record.get("fitted_slope")
        if slope is not None and np.isfinite(slope):
            ax.set_title(f"N={n_layers}, α={alpha:g} (fit slope={slope:.2f})")
        else:
            ax.set_title(f"N={n_layers}, α={alpha:g} (fit slope={fit['slope']:.2f})")
    else:
        ax.set_title(f"N={n_layers}, α={alpha:g}")

    ax.set_xlabel("Δt [s]")
    ax.set_ylabel("mass-weighted relative T RMS")
    ax.legend(fontsize=7)


def plot_convergence(data: dict) -> None:
    order_records = data.get("order_records", [])
    if not order_records:
        raise ValueError("temporal_stability.json: order_records[] is empty")

    apply_style()
    n = len(order_records)
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 4.8), squeeze=False)
    for ax, record in zip(axes[0], order_records):
        _plot_order_panel(ax, record)
    fig.suptitle("07b · Temporal convergence at common finite time", fontsize=12)
    out = save_figure(fig, "07b_temporal_convergence.png")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    path = DATA_DIR / "temporal_stability.json"
    if not path.exists():
        raise SystemExit(f"missing data: {path}")

    data = read_json(path)
    if not data.get("safety_cases"):
        raise ValueError("temporal_stability.json: safety_cases[] is empty")
    plot_controller(data)
    plot_convergence(data)


if __name__ == "__main__":
    main()
