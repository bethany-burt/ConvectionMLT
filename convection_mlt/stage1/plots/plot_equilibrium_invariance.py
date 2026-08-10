"""Resolution invariance figure: equilibrium metrics vs N and score heatmap."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np

from common import (
    DATA_DIR,
    apply_style,
    exact_zero_display,
    format_significant,
    read_json,
    require_finite,
    save_figure,
)

METRICS = (
    ("temperature_rms", "relative T RMS"),
    ("temperature_max", "max relative T"),
    ("potential_temperature_rms", "θ RMS"),
    ("max_superadiabaticity", "positive superadiabaticity"),
    ("normalized_tendency_max", "normalized tendency max"),
    ("convective_flux_max", "max F_conv"),
    ("enthalpy_drift", "max |enthalpy drift|"),
)
METRIC_SHORT = {
    "temperature_rms": "Trms",
    "temperature_max": "Tmax",
    "potential_temperature_rms": "θrms",
    "max_superadiabaticity": "∇+",
    "normalized_tendency_max": "tend",
    "convective_flux_max": "F",
    "enthalpy_drift": "H",
}
MARKERS = ("o", "s", "^", "D", "v", "P", "X")
LOG_FLOOR = 1.0e-30


def _controlling_metric(record: dict) -> str:
    score = record.get("score", {})
    controlling = score.get("controlling_metric")
    if controlling:
        return str(controlling)
    ratios = score.get("ratios", {})
    if not ratios:
        return "?"
    return max(ratios, key=lambda key: float(ratios[key]))


def _pass_score_cmap(vmax: float):
    """Green/amber for S<1; red reserved for S≥1."""
    colors = ["#1a9850", "#66bd63", "#a6d96a", "#fee08b", "#d73027"]
    # Last bin starts exactly at 1.0 so near-pass scores stay amber, not red.
    upper = float(max(1.01, vmax))
    bounds = [0.0, 0.5, 0.85, 0.97, 1.0, upper]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bounds, cmap.N)
    return cmap, norm, bounds


def _validate_record(record: dict) -> None:
    alpha = require_finite("alpha", record["alpha"])
    if alpha <= 0.0:
        return
    n_layers = int(record["n_layers"])
    status = record["outcome"]["status"]
    if status not in ("converged", "no_active_convection"):
        raise ValueError(
            f"N={n_layers} α={alpha}: unexpected terminal status {status!r}"
        )
    metrics = record["metrics_for_score"]
    tolerances = record["tolerances"]
    for key, _ in METRICS:
        require_finite(key, metrics[key])
        require_finite(f"tol:{key}", tolerances[key])


def _plot_metric(ax, records, resolutions, alphas, key, title) -> int:
    cmap = plt.cm.tab10
    zero_count = 0
    tol_ref = None
    for alpha_idx, alpha in enumerate(alphas):
        xs, ys = [], []
        for n in resolutions:
            rec = next(
                r
                for r in records
                if int(r["n_layers"]) == n and require_finite("alpha", r["alpha"]) == alpha
            )
            value = require_finite(key, rec["metrics_for_score"][key])
            tol = require_finite(f"tol:{key}", rec["tolerances"][key])
            if tol_ref is None:
                tol_ref = tol
            display, is_zero = exact_zero_display(value, LOG_FLOOR)
            xs.append(float(n))
            ys.append(display)
            if is_zero:
                zero_count += 1
        ax.plot(
            xs,
            ys,
            color=cmap(alpha_idx % 10),
            lw=1.4,
            marker=MARKERS[alpha_idx % len(MARKERS)],
            ms=5,
            label=f"α={alpha:g}",
        )
    if tol_ref is not None:
        ax.axhline(
            exact_zero_display(tol_ref, LOG_FLOOR)[0],
            color="k",
            ls="--",
            lw=0.9,
            alpha=0.6,
            label="acceptance tol",
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("N (resolution)")
    ax.set_ylabel(title)
    ax.set_title(title)
    ax.legend(fontsize=6, loc="best")
    return zero_count


def main() -> None:
    path = DATA_DIR / "equilibrium_matrix.json"
    if not path.exists():
        raise SystemExit(f"missing data: {path}")

    data = read_json(path)
    records = [r for r in data.get("records", []) if require_finite("alpha", r["alpha"]) > 0.0]
    if not records:
        raise ValueError("equilibrium_matrix.json: no records with α > 0")

    for record in records:
        _validate_record(record)

    alphas = sorted({require_finite("alpha", r["alpha"]) for r in records})
    resolutions = sorted({int(r["n_layers"]) for r in records})

    apply_style()
    fig = plt.figure(figsize=(16, 11.5))
    gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 1.25])

    metric_axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[1, 2]),
        fig.add_subplot(gs[2, 0]),
    ]
    ax_heat = fig.add_subplot(gs[3, :])

    zero_total = 0
    for ax, (key, title) in zip(metric_axes, METRICS):
        zero_total += _plot_metric(ax, records, resolutions, alphas, key, title)

    # Note on overlapping curves in the unused lower-right area of the metric grid.
    ax_note = fig.add_subplot(gs[2, 1:])
    ax_note.axis("off")
    ax_note.text(
        0.0,
        0.55,
        "Metric panels use distinct markers per α.\n"
        "Exact overlap of α curves is intentional when the equilibrium\n"
        "is resolution-invariant (same residual against the same tolerance).",
        fontsize=9,
        va="center",
    )

    heat = np.full((len(alphas), len(resolutions)), np.nan)
    controlling = [[""] * len(resolutions) for _ in alphas]
    for ai, alpha in enumerate(alphas):
        for ni, n in enumerate(resolutions):
            rec = next(
                r
                for r in records
                if int(r["n_layers"]) == n and require_finite("alpha", r["alpha"]) == alpha
            )
            heat[ai, ni] = require_finite("score", rec["score"]["score"])
            controlling[ai][ni] = _controlling_metric(rec)

    finite = heat[np.isfinite(heat)]
    vmax = float(np.max(finite)) if finite.size else 1.0
    cmap, norm, _bounds = _pass_score_cmap(vmax)
    im = ax_heat.imshow(heat, aspect="auto", cmap=cmap, norm=norm, origin="upper")
    ax_heat.set_xticks(range(len(resolutions)))
    ax_heat.set_xticklabels([str(n) for n in resolutions])
    ax_heat.set_yticks(range(len(alphas)))
    ax_heat.set_yticklabels([f"α={a:g}" for a in alphas])
    ax_heat.set_xlabel("N (resolution)")
    ax_heat.set_ylabel("α")
    ax_heat.set_title(
        "Score S = max(metric / tolerance); pass requires S < 1 "
        "(cell text: S and controlling metric; red only for S ≥ 1)"
    )
    for ai in range(len(alphas)):
        for ni in range(len(resolutions)):
            val = heat[ai, ni]
            if not np.isfinite(val):
                continue
            short = METRIC_SHORT.get(controlling[ai][ni], controlling[ai][ni][:4])
            label = format_significant(val, digits=4)
            if val < 1.0 and label in {"1", "1.0", "1.00", "1.000"}:
                label = "<1.000"
            ax_heat.text(
                ni,
                ai,
                f"{label}\n({short})",
                ha="center",
                va="center",
                fontsize=6.5,
                color="black" if val < 1.0 else "white",
            )
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.015, pad=0.01)
    cbar.set_label("S (green/amber: pass; red: fail)")

    subtitle = "Resolution invariance (NOT spatial order convergence)"
    if zero_total:
        subtitle += f"; {zero_total} exact-zero floor markers"
    fig.suptitle(subtitle, fontsize=12)
    out = save_figure(fig, "04_equilibrium_invariance.png")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
