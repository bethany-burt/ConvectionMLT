"""Robustness summary: tolerance-normalized metrics and terminal status."""

from __future__ import annotations

import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

from common import (
    DATA_DIR,
    GENERATED_DIR,
    apply_style,
    ensure_dirs,
    format_score_ratio,
    read_json,
    require_finite,
    save_figure,
)

METRIC_KEYS = (
    "temperature_rms",
    "temperature_max",
    "potential_temperature_rms",
    "max_superadiabaticity",
    "normalized_tendency_max",
    "convective_flux_max",
    "enthalpy_drift",
)
METRIC_LABELS = {
    "temperature_rms": "T RMS / tol",
    "temperature_max": "T max / tol",
    "potential_temperature_rms": "θ RMS / tol",
    "max_superadiabaticity": "∇ super / tol",
    "normalized_tendency_max": "tendency / tol",
    "convective_flux_max": "F_conv max / tol",
    "enthalpy_drift": "enthalpy / tol",
}


def _validate_record(record: dict) -> tuple[dict[str, float | None], list[str]]:
    name = record["name"]
    expected = record.get("expected_status")
    status = record["outcome"]["status"]
    if expected is not None and status != expected:
        raise ValueError(f"{name}: unexpected status {status!r}, expected {expected!r}")
    if not record.get("status_ok", True):
        raise ValueError(f"{name}: status_ok is false")

    metrics = record["metrics_for_score"]
    tolerances = record["tolerances"]
    applicable = set(record.get("applicable_metrics", METRIC_KEYS))
    normalized: dict[str, float | None] = {}
    for key in METRIC_KEYS:
        if key not in applicable:
            normalized[key] = None
            continue
        if key not in metrics:
            raise KeyError(f"{name}: missing metric {key}")
        if key not in tolerances:
            raise KeyError(f"{name}: missing tolerance {key}")
        value = require_finite(f"{name}.{key}", metrics[key])
        tol = require_finite(f"{name}.tol.{key}", tolerances[key])
        if tol <= 0.0:
            raise ValueError(f"{name}: nonpositive tolerance for {key}: {tol}")
        normalized[key] = value / tol

    score = record.get("score", {})
    if "score" in score and score["score"] is not None:
        require_finite(f"{name}.score", score["score"])
    return normalized, sorted(applicable)


def _write_csv(rows: list[dict]) -> None:
    ensure_dirs()
    path = GENERATED_DIR / "robustness_summary.csv"
    fieldnames = ["name", "status", "pass", "score", "applicable_metrics"] + list(
        METRIC_KEYS
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"wrote {path}")


def main() -> None:
    path = DATA_DIR / "robustness.json"
    if not path.exists():
        raise SystemExit(f"missing data: {path}")

    data = read_json(path)
    records = data.get("records", [])
    if not records:
        raise ValueError("robustness.json: records[] is empty")

    normalized_rows: list[list[float]] = []
    mask_rows: list[list[bool]] = []
    names: list[str] = []
    statuses: list[str] = []
    csv_rows: list[dict] = []

    for record in records:
        norm, applicable = _validate_record(record)
        values = []
        masked = []
        for key in METRIC_KEYS:
            value = norm[key]
            if value is None:
                values.append(np.nan)
                masked.append(True)
            else:
                values.append(float(value))
                masked.append(False)
        normalized_rows.append(values)
        mask_rows.append(masked)
        names.append(record["name"])
        statuses.append(record["outcome"]["status"])
        csv_rows.append(
            {
                "name": record["name"],
                "status": record["outcome"]["status"],
                "pass": record.get("pass", False),
                "score": record.get("score", {}).get("score", float("nan")),
                "applicable_metrics": ";".join(applicable),
                **{
                    k: ("" if norm[k] is None else norm[k])
                    for k in METRIC_KEYS
                },
            }
        )

    _write_csv(csv_rows)

    apply_style()
    n_rows = len(names)
    n_cols = len(METRIC_KEYS) + 1  # trailing status column
    display = np.zeros((n_rows, n_cols), dtype=float)
    mask = np.zeros((n_rows, n_cols), dtype=bool)
    for i, values in enumerate(normalized_rows):
        for j, value in enumerate(values):
            if mask_rows[i][j]:
                display[i, j] = np.nan
                mask[i, j] = True
            else:
                display[i, j] = value
        # Status column is categorical; keep masked from metric cmap.
        display[i, -1] = np.nan
        mask[i, -1] = True

    matrix = np.ma.array(display, mask=mask)
    finite = matrix[:, :-1].compressed()
    if finite.size == 0 or not np.all(np.isfinite(finite)):
        raise ValueError("robustness matrix contains nonfinite applicable metrics")

    fig, ax = plt.subplots(figsize=(15, max(4.5, 0.48 * n_rows + 2.2)))
    vmax = float(max(1.01, float(np.max(finite))))
    colors = ["#1a9850", "#66bd63", "#a6d96a", "#fee08b", "#d73027"]
    bounds = [0.0, 0.5, 0.85, 0.97, 1.0, vmax]
    cmap = ListedColormap(colors)
    cmap.set_bad(color="#d9d9d9")
    norm = BoundaryNorm(bounds, cmap.N)
    im = ax.imshow(
        matrix,
        aspect="auto",
        cmap=cmap,
        norm=norm,
        origin="upper",
        interpolation="nearest",
        extent=(-0.5, n_cols - 0.5, n_rows - 0.5, -0.5),
    )

    status_colors = {
        "converged": "tab:green",
        "no_active_convection": "tab:blue",
        "failed": "tab:red",
    }
    for i, status in enumerate(statuses):
        ax.fill_between(
            [n_cols - 1.5, n_cols - 0.5],
            i - 0.5,
            i + 0.5,
            color=status_colors.get(status, "tab:gray"),
            alpha=0.8,
            linewidth=0,
            zorder=2,
        )
        ax.text(
            n_cols - 1,
            i,
            status,
            ha="center",
            va="center",
            fontsize=7,
            color="white",
            zorder=3,
        )

    ax.set_xticks(list(range(len(METRIC_KEYS))) + [n_cols - 1])
    ax.set_xticklabels(
        [METRIC_LABELS[k] for k in METRIC_KEYS] + ["terminal status"],
        rotation=35,
        ha="right",
    )
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)
    for i in range(n_rows):
        for j in range(len(METRIC_KEYS)):
            if mask[i, j]:
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=7)
            else:
                ax.text(
                    j,
                    i,
                    format_score_ratio(float(display[i, j]), decimals=3),
                    ha="center",
                    va="center",
                    fontsize=7,
                )
    ax.set_title(
        "Tolerance-normalized metrics (n/a = inapplicable; pass requires ratio < 1)"
    )
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="metric / tolerance")
    ax.legend(
        handles=[Patch(facecolor="#d9d9d9", edgecolor="k", label="masked / n/a")],
        loc="upper right",
        fontsize=7,
    )

    n_pass = sum(1 for r in records if r.get("pass"))
    fig.suptitle(f"Robustness matrix ({n_pass}/{len(records)} pass)", fontsize=12)
    out = save_figure(fig, "02b_robustness_summary.png")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
