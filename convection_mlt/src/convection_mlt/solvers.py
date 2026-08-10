"""Forward-Euler reference solvers with conservative global timesteps."""

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .closure import ClosureResult, mixing_length_flux
from .config import PhysicsConfig, SolverConfig
from .diagnostics import (
    ConvergenceMetrics,
    column_enthalpy,
    convergence_metrics,
    enthalpy_normalized_adiabat,
    piecewise_enthalpy_reference,
    reference_enthalpy_residuals,
)
from .energy import temperature_tendency
from .grid import PressureGrid, hydrostatic_layer_thickness
from .thermodynamics import IdealH2
from .trace import IntegrationTrace
from .validate import temperatures


class TerminalStatus(str, Enum):
    CONVERGED = "converged"
    NO_ACTIVE_CONVECTION = "no_active_convection"
    FAILED = "failed"


@dataclass(frozen=True)
class StepResult:
    temperature: NDArray[np.float64]
    trial_temperature: NDArray[np.float64]
    dt: float
    accepted: bool
    reason: str | None
    closure: ClosureResult
    tendency: NDArray[np.float64]
    min_active_trial_delta_over_epsilon: float | None = None


@dataclass(frozen=True)
class IntegrationResult:
    temperature: NDArray[np.float64]
    status: TerminalStatus
    reason: str
    steps: int
    rejections: int
    simulated_time: float
    final_dt: float | None
    metrics: ConvergenceMetrics
    region_labels: NDArray[np.int64]
    cumulative_unmerged_transfer: NDArray[np.float64]
    max_unmerged_transfer_fraction: float


class SolverFailure(RuntimeError):
    """Typed failure carrying a terminal `failed` result."""

    def __init__(self, result: IntegrationResult):
        super().__init__(result.reason)
        self.result = result


def _evaluate(
    grid: PressureGrid,
    temperature: NDArray[np.float64],
    physics: PhysicsConfig,
    thermo: IdealH2,
) -> tuple[ClosureResult, NDArray[np.float64]]:
    closure = mixing_length_flux(
        grid,
        temperature,
        physics.gravity,
        physics.alpha,
        thermo,
        physics.closure_prefactor,
    )
    return closure, temperature_tendency(grid, closure.flux, thermo.cp)


def _min_active_trial_delta_over_epsilon(
    old_closure: ClosureResult,
    trial_closure: ClosureResult,
    thermo: IdealH2,
    config: SolverConfig,
) -> float | None:
    old_delta = old_closure.gradient[1:-1] - thermo.nabla_ad
    trial_delta = trial_closure.gradient[1:-1] - thermo.nabla_ad
    actively_unstable = old_delta > config.c_active * config.epsilon_gradient
    if not np.any(actively_unstable):
        return None
    return float(
        np.min(trial_delta[actively_unstable] / config.epsilon_gradient)
    )


def _crossing_reason(
    old_closure: ClosureResult,
    trial_closure: ClosureResult,
    thermo: IdealH2,
    config: SolverConfig,
) -> str | None:
    old_delta = old_closure.gradient[1:-1] - thermo.nabla_ad
    trial_delta = trial_closure.gradient[1:-1] - thermo.nabla_ad
    actively_unstable = old_delta > config.c_active * config.epsilon_gradient
    crossed = actively_unstable & (
        trial_delta < -config.c_cross * config.epsilon_gradient
    )
    if np.any(crossed):
        interface = int(np.flatnonzero(crossed)[0] + 1)
        return f"active interface {interface} crossed the neutral hysteresis band"
    return None


def _trial_step(
    grid: PressureGrid,
    temperature: NDArray[np.float64],
    dt: float,
    physics: PhysicsConfig,
    config: SolverConfig,
    thermo: IdealH2,
) -> StepResult:
    closure, tendency = _evaluate(grid, temperature, physics, thermo)
    trial = temperature + dt * tendency
    reason: str | None = None
    min_delta: float | None = None
    if not np.all(np.isfinite(trial)):
        reason = "trial state contains nonfinite temperature"
    elif np.any(trial <= 0.0):
        reason = "trial state contains nonpositive temperature"
    elif np.max(np.abs(dt * tendency) / temperature, initial=0.0) > (
        config.epsilon_temperature * (1.0 + 1.0e-12)
    ):
        reason = "fractional-temperature bound exceeded"
    else:
        trial_closure, _ = _evaluate(grid, trial, physics, thermo)
        min_delta = _min_active_trial_delta_over_epsilon(
            closure, trial_closure, thermo, config
        )
        reason = _crossing_reason(closure, trial_closure, thermo, config)
    return StepResult(
        temperature.copy() if reason else trial,
        trial,
        dt,
        reason is None,
        reason,
        closure,
        tendency,
        min_delta,
    )


def fixed_step(
    grid: PressureGrid,
    temperature: ArrayLike,
    dt: float,
    physics: PhysicsConfig,
    config: SolverConfig | None = None,
    thermo: IdealH2 | None = None,
) -> StepResult:
    """Attempt exactly the requested dt; report failure without changing it."""
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")
    gas = thermo or IdealH2()
    settings = config or SolverConfig(epsilon_temperature=1.0e300)
    t = temperatures(temperature, grid.n_layers)
    return _trial_step(grid, t, dt, physics, settings, gas)


def adaptive_timestep(
    grid: PressureGrid,
    temperature: ArrayLike,
    closure: ClosureResult,
    tendency: ArrayLike,
    physics: PhysicsConfig,
    config: SolverConfig,
    thermo: IdealH2 | None = None,
) -> tuple[float, float]:
    """Return (dt_diff, dt_T), with infinity for inactive empty sets."""
    gas = thermo or IdealH2()
    t = temperatures(temperature, grid.n_layers)
    tdot = np.asarray(tendency, dtype=float)
    dz = hydrostatic_layer_thickness(
        grid, t, gas.gas_constant, physics.gravity
    )
    adjacent_kh = np.maximum(
        closure.thermal_diffusivity[:-1],
        closure.thermal_diffusivity[1:],
    )
    diff_bounds = np.full(grid.n_layers, np.inf)
    active_diffusion = adjacent_kh > 0.0
    diff_bounds[active_diffusion] = (
        dz[active_diffusion] ** 2 / adjacent_kh[active_diffusion]
    )
    dt_diff = config.c_diff * float(np.min(diff_bounds, initial=np.inf))
    active_tendency = np.abs(tdot) > 0.0
    if np.any(active_tendency):
        dt_temperature = config.epsilon_temperature * float(
            np.min(t[active_tendency] / np.abs(tdot[active_tendency]))
        )
    else:
        dt_temperature = np.inf
    return dt_diff, dt_temperature


def _result(
    temperature: NDArray[np.float64],
    status: TerminalStatus,
    reason: str,
    steps: int,
    rejections: int,
    simulated_time: float,
    final_dt: float | None,
    metrics: ConvergenceMetrics,
    region_labels: NDArray[np.int64],
    cumulative_unmerged_transfer: NDArray[np.float64],
    max_unmerged_transfer_fraction: float,
) -> IntegrationResult:
    return IntegrationResult(
        temperature.copy(),
        status,
        reason,
        steps,
        rejections,
        simulated_time,
        final_dt,
        metrics,
        region_labels.copy(),
        cumulative_unmerged_transfer.copy(),
        max_unmerged_transfer_fraction,
    )


def _merge_active_regions(
    labels: NDArray,
    closure: ClosureResult,
    thermo: IdealH2,
    config: SolverConfig,
) -> tuple[NDArray, bool]:
    """Merge region labels when an interface becomes actively convective."""
    merged = labels.copy()
    changed = False
    active_edges = (
        closure.gradient[1:-1] - thermo.nabla_ad
        > config.c_active * config.epsilon_gradient
    )
    for upper_layer, active in enumerate(active_edges, start=1):
        if active and merged[upper_layer - 1] != merged[upper_layer]:
            old_label = merged[upper_layer]
            merged[merged == old_label] = merged[upper_layer - 1]
            changed = True
    if changed:
        canonical = np.empty_like(merged, dtype=np.int64)
        mapping: dict[object, int] = {}
        for index, label in enumerate(merged):
            mapping.setdefault(label.item(), len(mapping))
            canonical[index] = mapping[label.item()]
        merged = canonical
    return merged, changed


def _piecewise_reference_checked(
    grid: PressureGrid,
    initial_temperature: NDArray[np.float64],
    thermo: IdealH2,
    labels: NDArray,
) -> NDArray[np.float64]:
    reference = piecewise_enthalpy_reference(
        grid, initial_temperature, thermo.cp, thermo.nabla_ad, labels
    )
    residuals = reference_enthalpy_residuals(
        grid, initial_temperature, reference, thermo.cp, labels
    )
    roundoff_limit = 64.0 * np.finfo(float).eps
    if max(residuals.values(), default=0.0) > roundoff_limit:
        raise ArithmeticError(
            "piecewise reference does not conserve regional enthalpy "
            f"to roundoff: {residuals}"
        )
    return reference


def _unmerged_transfer_fractions(
    grid: PressureGrid,
    initial_temperature: NDArray[np.float64],
    thermo: IdealH2,
    labels: NDArray,
    cumulative_transfer: NDArray[np.float64],
) -> NDArray[np.float64]:
    fractions = np.zeros(grid.n_layers + 1)
    enthalpy_by_label = {
        label.item(): float(
            np.sum(
                thermo.cp
                * initial_temperature[labels == label]
                * grid.layer_mass[labels == label]
            )
        )
        for label in np.unique(labels)
    }
    for edge in range(1, grid.n_layers):
        lower_label = labels[edge - 1].item()
        upper_label = labels[edge].item()
        if lower_label != upper_label:
            adjacent_enthalpy = min(
                enthalpy_by_label[lower_label],
                enthalpy_by_label[upper_label],
            )
            fractions[edge] = (
                abs(cumulative_transfer[edge]) / adjacent_enthalpy
            )
    return fractions


def _merge_transferred_regions(
    grid: PressureGrid,
    initial_temperature: NDArray[np.float64],
    thermo: IdealH2,
    labels: NDArray,
    cumulative_transfer: NDArray[np.float64],
    tolerance: float,
) -> tuple[NDArray, bool, NDArray[np.float64]]:
    fractions = _unmerged_transfer_fractions(
        grid, initial_temperature, thermo, labels, cumulative_transfer
    )
    merged = labels.copy()
    changed = False
    for edge in range(1, grid.n_layers):
        if (
            fractions[edge] > tolerance
            and merged[edge - 1] != merged[edge]
        ):
            old_label = merged[edge]
            merged[merged == old_label] = merged[edge - 1]
            changed = True
    if changed:
        canonical = np.empty_like(merged, dtype=np.int64)
        mapping: dict[object, int] = {}
        for index, label in enumerate(merged):
            mapping.setdefault(label.item(), len(mapping))
            canonical[index] = mapping[label.item()]
        merged = canonical
        fractions = _unmerged_transfer_fractions(
            grid, initial_temperature, thermo, merged, cumulative_transfer
        )
    return merged, changed, fractions


def solve_adaptive(
    grid: PressureGrid,
    initial_temperature: ArrayLike,
    physics: PhysicsConfig,
    config: SolverConfig | None = None,
    thermo: IdealH2 | None = None,
    region_labels: ArrayLike | None = None,
    trace: IntegrationTrace | None = None,
) -> IntegrationResult:
    """Relax a closed column, raising `SolverFailure` on explicit failure.

    Status precedence is deliberate: test convergence metrics first, then apply
    the exact-zero no-active-convection shortcut. Thus a roundoff-level analytic
    adiabat is `converged`, while a stationary nonadiabatic alpha=0 profile is
    `no_active_convection`.
    """
    settings = config or SolverConfig()
    gas = thermo or IdealH2()
    initial = temperatures(initial_temperature, grid.n_layers).copy()
    t = initial.copy()
    initial_enthalpy = column_enthalpy(grid, t, gas.cp)
    labels = (
        None
        if region_labels is None
        else np.asarray(region_labels).copy()
    )
    reference = (
        enthalpy_normalized_adiabat(grid, t, gas.cp, gas.nabla_ad)
        if labels is None
        else _piecewise_reference_checked(grid, initial, gas, labels)
    )
    result_labels = (
        np.zeros(grid.n_layers, dtype=np.int64)
        if labels is None
        else labels
    )
    cumulative_transfer = np.zeros(grid.n_layers + 1)
    transfer_fractions = np.zeros(grid.n_layers + 1)
    simulated_time = 0.0
    total_rejections = 0
    last_dt: float | None = None
    recorder = trace
    initial_recorded = False

    for step in range(settings.max_steps + 1):
        closure, tendency = _evaluate(grid, t, physics, gas)
        # Candidate merges are computed from the accepted pre-step state whose
        # flux will drive the update, but they are committed only after a trial
        # step is accepted. Rejected trials and no-step terminals leave labels
        # unchanged.
        candidate_labels = labels
        candidate_active_merged = False
        if labels is not None:
            candidate_labels, candidate_active_merged = _merge_active_regions(
                labels, closure, gas, settings
            )
            transfer_fractions = _unmerged_transfer_fractions(
                grid, initial, gas, labels, cumulative_transfer
            )
        metrics = convergence_metrics(
            grid,
            t,
            reference,
            tendency,
            closure.flux,
            closure.superadiabaticity,
            gas.cp,
            initial_enthalpy,
            gas.nabla_ad,
            labels,
        )
        if recorder is not None and not initial_recorded:
            recorder.record_initial(
                t,
                metrics,
                result_labels,
                closure.flux,
                settings.theta_rms_tolerance,
            )
            initial_recorded = True
        reference_is_informative = labels is None or any(
            np.count_nonzero(labels == label) > 1
            for label in np.unique(labels)
        )
        # Required precedence: convergence before exact-zero shortcut.
        # All-singleton partitions are not evidence of mixed equilibrium:
        # their reference agreement is true by construction.
        if reference_is_informative and metrics.converged(settings):
            if recorder is not None:
                recorder.record_final(
                    t,
                    closure.flux,
                    metrics,
                    result_labels,
                    simulated_time,
                    step,
                )
            return _result(
                t,
                TerminalStatus.CONVERGED,
                "all acceptance metrics are within tolerance",
                step,
                total_rejections,
                simulated_time,
                last_dt,
                metrics,
                result_labels,
                cumulative_transfer,
                float(np.max(transfer_fractions, initial=0.0)),
            )
        if np.all(closure.thermal_diffusivity == 0.0) and np.all(
            tendency == 0.0
        ):
            if recorder is not None:
                recorder.record_final(
                    t,
                    closure.flux,
                    metrics,
                    result_labels,
                    simulated_time,
                    step,
                )
            return _result(
                t,
                TerminalStatus.NO_ACTIVE_CONVECTION,
                "all diffusivities and tendencies are exactly zero",
                step,
                total_rejections,
                simulated_time,
                last_dt,
                metrics,
                result_labels,
                cumulative_transfer,
                float(np.max(transfer_fractions, initial=0.0)),
            )
        if step == settings.max_steps:
            if recorder is not None:
                recorder.record_final(
                    t,
                    closure.flux,
                    metrics,
                    result_labels,
                    simulated_time,
                    step,
                )
            result = _result(
                t,
                TerminalStatus.FAILED,
                "maximum accepted-step limit reached",
                step,
                total_rejections,
                simulated_time,
                last_dt,
                metrics,
                result_labels,
                cumulative_transfer,
                float(np.max(transfer_fractions, initial=0.0)),
            )
            raise SolverFailure(result)

        dt_diff, dt_temperature = adaptive_timestep(
            grid, t, closure, tendency, physics, settings, gas
        )
        dt = min(dt_diff, dt_temperature)
        rejection_reason = "no finite adaptive timestep"
        accepted = False
        rejections_this_step = 0
        last_attempted_dt: float | None = None
        for attempt in range(settings.max_rejections + 1):
            if not np.isfinite(dt) or dt < settings.dt_min:
                rejection_reason = (
                    f"timestep {dt!r} is below dt_min={settings.dt_min}"
                )
                if recorder is not None:
                    recorder.record_trial(
                        step,
                        attempt,
                        dt if np.isfinite(dt) else float("nan"),
                        False,
                        rejection_reason,
                        None,
                    )
                break
            last_attempted_dt = dt
            trial = _trial_step(grid, t, dt, physics, settings, gas)
            if recorder is not None:
                recorder.record_trial(
                    step,
                    attempt,
                    dt,
                    trial.accepted,
                    trial.reason,
                    trial.min_active_trial_delta_over_epsilon,
                )
            if trial.accepted:
                if labels is not None:
                    # Commit pre-step active merges before transfer accounting
                    # so active-before/neutral-after interfaces stay merged.
                    labels = candidate_labels
                    if candidate_active_merged:
                        reference = _piecewise_reference_checked(
                            grid, initial, gas, labels
                        )
                    separated = labels[:-1] != labels[1:]
                    internal_edges = np.flatnonzero(separated) + 1
                    cumulative_transfer[internal_edges] += (
                        trial.closure.flux[internal_edges] * dt
                    )
                    labels, transfer_merged, transfer_fractions = (
                        _merge_transferred_regions(
                            grid,
                            initial,
                            gas,
                            labels,
                            cumulative_transfer,
                            settings.transfer_merge_tolerance,
                        )
                    )
                    if transfer_merged:
                        reference = _piecewise_reference_checked(
                            grid, initial, gas, labels
                        )
                    result_labels = labels
                t = trial.temperature
                simulated_time += dt
                last_dt = dt
                total_rejections += rejections_this_step
                accepted = True
                if recorder is not None:
                    post_closure, post_tendency = _evaluate(
                        grid, t, physics, gas
                    )
                    post_metrics = convergence_metrics(
                        grid,
                        t,
                        reference,
                        post_tendency,
                        post_closure.flux,
                        post_closure.superadiabaticity,
                        gas.cp,
                        initial_enthalpy,
                        gas.nabla_ad,
                        labels,
                    )
                    signed_drift = (
                        column_enthalpy(grid, t, gas.cp) - initial_enthalpy
                    ) / initial_enthalpy
                    recorder.record_accepted(
                        step + 1,
                        simulated_time,
                        dt,
                        rejections_this_step,
                        post_metrics,
                        signed_drift,
                        result_labels,
                        cumulative_transfer,
                        t,
                        post_closure.flux,
                    )
                break
            rejection_reason = trial.reason or "unknown rejection"
            rejections_this_step += 1
            dt *= settings.f_back
        if not accepted:
            total_rejections += rejections_this_step
            failure_dt = (
                last_attempted_dt if last_attempted_dt is not None else dt
            )
            failure_metrics = convergence_metrics(
                grid,
                t,
                reference,
                tendency,
                closure.flux,
                closure.superadiabaticity,
                gas.cp,
                initial_enthalpy,
                gas.nabla_ad,
                labels,
            )
            if recorder is not None:
                recorder.record_final(
                    t,
                    closure.flux,
                    failure_metrics,
                    result_labels,
                    simulated_time,
                    step,
                )
            result = _result(
                t,
                TerminalStatus.FAILED,
                (
                    f"backtracking failed: {rejection_reason}; "
                    f"final attempted dt={failure_dt}"
                ),
                step,
                total_rejections,
                simulated_time,
                failure_dt,
                failure_metrics,
                result_labels,
                cumulative_transfer,
                float(np.max(transfer_fractions, initial=0.0)),
            )
            raise SolverFailure(result)

    raise AssertionError("unreachable")
