"""Conservation, analytic-reference, convergence, and timescale diagnostics."""

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .config import SolverConfig
from .grid import PressureGrid
from .validate import finite_1d, positive, temperatures


@dataclass(frozen=True)
class ConvergenceMetrics:
    max_superadiabaticity: float
    potential_temperature_rms: float
    temperature_rms: float
    temperature_max: float
    normalized_tendency_max: float
    convective_flux_max: float
    enthalpy_drift: float

    def converged(self, config: SolverConfig) -> bool:
        return (
            self.max_superadiabaticity <= config.epsilon_gradient
            and self.potential_temperature_rms <= config.theta_rms_tolerance
            and self.temperature_rms <= config.temperature_rms_tolerance
            and self.temperature_max <= config.temperature_max_tolerance
            and self.normalized_tendency_max <= config.tendency_tolerance
            and self.convective_flux_max <= config.flux_tolerance
            and self.enthalpy_drift <= config.enthalpy_drift_tolerance
        )

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def column_enthalpy(grid: PressureGrid, temperature: ArrayLike, cp: float) -> float:
    """Legacy Stage 0/1 column enthalpy using constant cp: Σ cp T Δm."""
    t = temperatures(temperature, grid.n_layers)
    positive("cp", cp)
    return float(np.sum(cp * t * grid.layer_mass))


def column_enthalpy_per_area_from_state(
    mass_path: ArrayLike, enthalpy: ArrayLike
) -> float:
    """Stage 2 column enthalpy per unit area H = Σ Δm_i h_i."""
    from .energy import column_enthalpy_per_area

    return column_enthalpy_per_area(mass_path, enthalpy)


def enthalpy_normalized_adiabat(
    grid: PressureGrid,
    initial_temperature: ArrayLike,
    cp: float,
    nabla_ad: float,
    reference_pressure: float | None = None,
) -> NDArray[np.float64]:
    t0 = temperatures(initial_temperature, grid.n_layers)
    positive("cp", cp)
    p0 = float(reference_pressure or grid.pressure_centres[0])
    positive("reference_pressure", p0)
    shape = (grid.pressure_centres / p0) ** nabla_ad
    amplitude = np.sum(cp * t0 * grid.layer_mass) / np.sum(
        cp * shape * grid.layer_mass
    )
    return amplitude * shape


def piecewise_enthalpy_reference(
    grid: PressureGrid,
    initial_temperature: ArrayLike,
    cp: float,
    nabla_ad: float,
    region_labels: ArrayLike,
) -> NDArray[np.float64]:
    """Return one enthalpy-normalized adiabat per connected mixing region."""
    t0 = temperatures(initial_temperature, grid.n_layers)
    labels = np.asarray(region_labels)
    if labels.ndim != 1 or labels.size != grid.n_layers:
        raise ValueError("region_labels must have one entry per layer")
    reference = np.empty_like(t0)
    for label in np.unique(labels):
        region = labels == label
        p0 = grid.pressure_centres[np.flatnonzero(region)[0]]
        shape = (grid.pressure_centres[region] / p0) ** nabla_ad
        amplitude = np.sum(cp * t0[region] * grid.layer_mass[region]) / np.sum(
            cp * shape * grid.layer_mass[region]
        )
        reference[region] = amplitude * shape
    return reference


def reference_enthalpy_residuals(
    grid: PressureGrid,
    initial_temperature: ArrayLike,
    reference_temperature: ArrayLike,
    cp: float,
    region_labels: ArrayLike,
) -> dict[int, float]:
    """Return relative initial-versus-reference enthalpy residual per region."""
    initial = temperatures(initial_temperature, grid.n_layers)
    reference = temperatures(reference_temperature, grid.n_layers)
    positive("cp", cp)
    labels = np.asarray(region_labels)
    if labels.ndim != 1 or labels.size != grid.n_layers:
        raise ValueError("region_labels must have one entry per layer")
    residuals: dict[int, float] = {}
    for label in np.unique(labels):
        region = labels == label
        initial_h = np.sum(cp * initial[region] * grid.layer_mass[region])
        reference_h = np.sum(cp * reference[region] * grid.layer_mass[region])
        residuals[int(label)] = float(abs(reference_h - initial_h) / initial_h)
    return residuals


def mixing_region_labels(
    grid: PressureGrid,
    temperature: ArrayLike,
    nabla_ad: float,
    active_threshold: float,
) -> NDArray[np.int64]:
    """Label layers joined by initially active internal interfaces."""
    t = temperatures(temperature, grid.n_layers)
    positive("active_threshold", active_threshold)
    gradient = (np.log(t[:-1]) - np.log(t[1:])) / (
        np.log(grid.pressure_centres[:-1])
        - np.log(grid.pressure_centres[1:])
    )
    joins = gradient - nabla_ad > active_threshold
    labels = np.zeros(grid.n_layers, dtype=np.int64)
    for index, joined in enumerate(joins, start=1):
        labels[index] = labels[index - 1] if joined else labels[index - 1] + 1
    return labels


def potential_temperature(
    pressure: ArrayLike,
    temperature: ArrayLike,
    nabla_ad: float,
    reference_pressure: float,
) -> NDArray[np.float64]:
    p = finite_1d("pressure", pressure)
    t = temperatures(temperature, p.size)
    positive("reference_pressure", reference_pressure)
    return t * (reference_pressure / p) ** nabla_ad


def convergence_metrics(
    grid: PressureGrid,
    temperature: ArrayLike,
    reference_temperature: ArrayLike,
    tendency: ArrayLike,
    interface_flux: ArrayLike,
    superadiabaticity: ArrayLike,
    cp: float,
    initial_enthalpy: float,
    nabla_ad: float = 2.0 / 7.0,
    region_labels: ArrayLike | None = None,
) -> ConvergenceMetrics:
    t = temperatures(temperature, grid.n_layers)
    tref = temperatures(reference_temperature, grid.n_layers)
    tdot = finite_1d("tendency", tendency)
    flux = finite_1d("interface_flux", interface_flux)
    delta = finite_1d("superadiabaticity", superadiabaticity)
    if tdot.size != grid.n_layers:
        raise ValueError("tendency length does not match grid")
    if flux.size != grid.n_layers + 1:
        raise ValueError("interface_flux length does not match grid")
    positive("initial_enthalpy", initial_enthalpy)
    weights = grid.layer_mass
    weight_sum = np.sum(weights)
    labels = (
        np.zeros(grid.n_layers, dtype=np.int64)
        if region_labels is None
        else np.asarray(region_labels)
    )
    if labels.ndim != 1 or labels.size != grid.n_layers:
        raise ValueError("region_labels must have one entry per layer")
    theta_squared_error = np.zeros(grid.n_layers)
    for label in np.unique(labels):
        region = labels == label
        p0 = grid.pressure_centres[np.flatnonzero(region)[0]]
        theta = potential_temperature(
            grid.pressure_centres[region], t[region], nabla_ad, p0
        )
        theta_mean = np.sum(weights[region] * theta) / np.sum(weights[region])
        theta_squared_error[region] = ((theta - theta_mean) / theta_mean) ** 2
    theta_rms = np.sqrt(np.sum(weights * theta_squared_error) / weight_sum)
    relative_t = (t - tref) / tref
    current_enthalpy = column_enthalpy(grid, t, cp)
    return ConvergenceMetrics(
        max_superadiabaticity=float(np.max(delta, initial=0.0)),
        potential_temperature_rms=float(theta_rms),
        temperature_rms=float(
            np.sqrt(np.sum(weights * relative_t**2) / weight_sum)
        ),
        temperature_max=float(np.max(np.abs(relative_t), initial=0.0)),
        normalized_tendency_max=float(np.max(np.abs(tdot) / t, initial=0.0)),
        convective_flux_max=float(np.max(np.abs(flux), initial=0.0)),
        enthalpy_drift=float(
            abs(current_enthalpy - initial_enthalpy) / initial_enthalpy
        ),
    )


def mixing_timescales(
    mixing_length: ArrayLike,
    velocity: ArrayLike,
    scale_height: ArrayLike,
    kzz: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    ell = finite_1d("mixing_length", mixing_length)
    w = finite_1d("velocity", velocity)
    hp = finite_1d("scale_height", scale_height)
    diffusion = finite_1d("kzz", kzz)
    if not (ell.shape == w.shape == hp.shape == diffusion.shape):
        raise ValueError("timescale fields must have matching shapes")
    active = (w > 0.0) & (diffusion > 0.0)
    turn = np.full_like(w, np.inf)
    mix = np.full_like(w, np.inf)
    turn[active] = ell[active] / w[active]
    mix[active] = hp[active] ** 2 / diffusion[active]
    return turn, mix, active


def numerical_isentrope(
    grid: PressureGrid,
    initial_temperature: ArrayLike,
    thermo,
    mass_path: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Build an enthalpy-normalized numerical isentrope on the pressure centres."""
    from .energy import column_enthalpy_per_area

    t0 = temperatures(initial_temperature, grid.n_layers)
    mass = (
        np.asarray(grid.layer_mass, dtype=float)
        if mass_path is None
        else finite_1d("mass_path", mass_path)
    )
    if mass.size != grid.n_layers:
        raise ValueError("mass_path length mismatch")

    t_min = float(getattr(thermo, "t_min", 200.0))
    t_max = float(getattr(thermo, "t_max", 6000.0))
    # Keep a small margin inside closed endpoints for robust inversion.
    t_lo = t_min * (1.0 + 1.0e-9) if t_min > 0.0 else t_min + 1.0e-9
    t_hi = t_max * (1.0 - 1.0e-9)
    psi_lo = float(thermo.psi(np.asarray([t_lo]))[0])
    psi_hi = float(thermo.psi(np.asarray([t_hi]))[0])
    r_mix = float(thermo.gas_constant)
    p_ref = float(thermo.p_ref)

    # Valid constant-s values are the intersection over layers of
    # s ∈ [Ψ_min - R ln(P/P_ref), Ψ_max - R ln(P/P_ref)].
    s_lo = -np.inf
    s_hi = np.inf
    for pressure in grid.pressure_centres:
        offset = r_mix * np.log(float(pressure) / p_ref)
        s_lo = max(s_lo, psi_lo - offset)
        s_hi = min(s_hi, psi_hi - offset)
    if not np.isfinite(s_lo) or not np.isfinite(s_hi) or s_lo >= s_hi:
        raise ValueError("no common entropy is reachable on this pressure grid")

    def profile_for_entropy(s_target: float) -> NDArray[np.float64]:
        from .thermodynamics import invert_psi_newton

        target_psi = s_target + r_mix * np.log(grid.pressure_centres / p_ref)
        return invert_psi_newton(thermo, target_psi, t_min=t_lo, t_max=t_hi)


    def enthalpy_of_s(s_value: float) -> float:
        return column_enthalpy_per_area(
            mass, thermo.enthalpy(profile_for_entropy(s_value))
        )

    h_target = column_enthalpy_per_area(mass, thermo.enthalpy(t0))
    h_lo = enthalpy_of_s(s_lo)
    h_hi = enthalpy_of_s(s_hi)
    if h_target < h_lo or h_target > h_hi:
        raise ValueError(
            "initial column enthalpy is outside the reachable isentrope range "
            f"[{h_lo}, {h_hi}] for domain [{t_lo}, {t_hi}] K"
        )

    lo = float(s_lo)
    hi = float(s_hi)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if enthalpy_of_s(mid) < h_target:
            lo = mid
        else:
            hi = mid
    return profile_for_entropy(0.5 * (lo + hi))
