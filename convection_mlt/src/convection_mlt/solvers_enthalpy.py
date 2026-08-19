"""Stage 2 enthalpy-based adaptive solver with atomic column-state commits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .closure import ClosureResult, mixing_length_flux
from .config import PhysicsConfig, SolverConfig
from .diagnostics import ConvergenceMetrics, numerical_isentrope
from .energy import (
    apply_enthalpy_step,
    column_enthalpy_per_area,
    enthalpy_tendency,
    flux_divergence_identity_residual,
)
from .gravity import ConstantGravity, GravityLaw
from .grid import PressureGrid
from .hydrostatics import HydrostaticDomainError
from .solvers import IntegrationResult, SolverFailure, TerminalStatus, _result
from .state import ColumnState, build_column_state
from .thermodynamics import (
    EnthalpyInversionError,
    ThermoDomainError,
    ThermoProvider,
)
from .trace import IntegrationTrace
from .validate import temperatures


@dataclass(frozen=True)
class EnthalpyStepResult:
    state: ColumnState
    trial_state: ColumnState | None
    dt: float
    accepted: bool
    reason: str | None
    closure: ClosureResult
    enthalpy_tendency: NDArray[np.float64]
    identity_residual: float | None
    apparent_budget_change: float | None
    geometric_residual: float | None
    min_active_trial_delta_over_epsilon: float | None = None


def _evaluate_closure(
    grid: PressureGrid,
    state: ColumnState,
    physics: PhysicsConfig,
    thermo: ThermoProvider,
) -> ClosureResult:
    return mixing_length_flux(
        grid,
        state.temperature,
        state.g_edges,
        physics.alpha,
        thermo,
        physics.closure_prefactor,
        use_entropy_instability=True,
    )


def _crossing_reason(
    old_closure: ClosureResult,
    trial_closure: ClosureResult,
    config: SolverConfig,
) -> str | None:
    old_delta = old_closure.superadiabaticity[1:-1]
    trial_delta = trial_closure.superadiabaticity[1:-1]
    actively_unstable = old_delta > config.c_active * config.epsilon_gradient
    crossed = actively_unstable & (
        trial_delta < -config.c_cross * config.epsilon_gradient
    )
    if np.any(crossed):
        interface = int(np.flatnonzero(crossed)[0] + 1)
        return f"active interface {interface} crossed the neutral hysteresis band"
    return None


def _min_active_margin(
    old_closure: ClosureResult,
    trial_closure: ClosureResult,
    config: SolverConfig,
) -> float | None:
    old_delta = old_closure.superadiabaticity[1:-1]
    trial_delta = trial_closure.superadiabaticity[1:-1]
    actively_unstable = old_delta > config.c_active * config.epsilon_gradient
    if not np.any(actively_unstable):
        return None
    return float(np.min(trial_delta[actively_unstable] / config.epsilon_gradient))


def _metrics(
    grid: PressureGrid,
    state: ColumnState,
    reference: NDArray[np.float64],
    tendency_proxy: NDArray[np.float64],
    closure: ClosureResult,
    thermo: ThermoProvider,
    initial_enthalpy: float,
    *,
    enforce_enthalpy_drift: bool,
    config: SolverConfig | None = None,
) -> ConvergenceMetrics:
    weights = state.mass_path
    weight_sum = float(np.sum(weights))
    t = state.temperature
    if enforce_enthalpy_drift:
        tref = reference
        relative_t = (t - tref) / tref
        temperature_rms = float(
            np.sqrt(np.sum(weights * relative_t**2) / weight_sum)
        )
        temperature_max = float(np.max(np.abs(relative_t), initial=0.0))
    else:
        # Variable-g: constant-s flatness is the thermodynamic exit metric.
        entropy = thermo.entropy(t, grid.pressure_centres)
        entropy_span = float(np.max(entropy) - np.min(entropy))
        scale = max(abs(float(np.mean(entropy))), 1.0)
        temperature_rms = entropy_span / scale
        temperature_max = temperature_rms
    current_h = column_enthalpy_per_area(state.mass_path, state.enthalpy)
    apparent_drift = float(
        abs(current_h - initial_enthalpy) / max(abs(initial_enthalpy), 1.0)
    )
    # Variable-g updates change Δm; Stage 2 does not claim apparent-budget closure.
    enthalpy_drift = apparent_drift if enforce_enthalpy_drift else 0.0
    max_superadiabaticity = float(np.max(closure.superadiabaticity, initial=0.0))
    normalized_tendency_max = float(
        np.max(np.abs(tendency_proxy) / np.maximum(t, 1.0e-300), initial=0.0)
    )
    convective_flux_max = float(np.max(np.abs(closure.flux), initial=0.0))
    # θ needs a Newton invert per layer; defer until other exit gates pass.
    need_theta = True
    if config is not None:
        need_theta = (
            max_superadiabaticity <= config.epsilon_gradient
            and temperature_rms <= config.temperature_rms_tolerance
            and temperature_max <= config.temperature_max_tolerance
            and normalized_tendency_max <= config.tendency_tolerance
            and convective_flux_max <= config.flux_tolerance
            and enthalpy_drift <= config.enthalpy_drift_tolerance
        )
    if need_theta:
        theta = thermo.potential_temperature(t, grid.pressure_centres)
        theta_mean = float(np.sum(weights * theta) / weight_sum)
        theta_rms = float(
            np.sqrt(
                np.sum(weights * ((theta - theta_mean) / theta_mean) ** 2) / weight_sum
            )
        )
    else:
        theta_rms = float("inf")
    return ConvergenceMetrics(
        max_superadiabaticity=max_superadiabaticity,
        potential_temperature_rms=theta_rms,
        temperature_rms=temperature_rms,
        temperature_max=temperature_max,
        normalized_tendency_max=normalized_tendency_max,
        convective_flux_max=convective_flux_max,
        enthalpy_drift=enthalpy_drift,
    )


def trial_enthalpy_step(
    grid: PressureGrid,
    state: ColumnState,
    dt: float,
    physics: PhysicsConfig,
    config: SolverConfig,
    thermo: ThermoProvider,
    gravity: GravityLaw,
) -> EnthalpyStepResult:
    """Atomic trial: h→T→domain→ρ→hydro→mass→acceptance; commit nothing on failure."""
    closure = _evaluate_closure(grid, state, physics, thermo)
    dhdt = enthalpy_tendency(grid, closure.flux, state.mass_path)
    # Temperature-proxy tendency for fractional bound.
    cp_centres = thermo.specific_heat(state.temperature)
    t_tendency = dhdt / cp_centres

    reason: str | None = None
    trial_state: ColumnState | None = None
    identity = None
    apparent = None
    geometric = None
    min_delta = None

    try:
        h_trial = apply_enthalpy_step(state.enthalpy, dhdt, dt)
        t_trial = thermo.invert_enthalpy(h_trial)
        if not np.all(np.isfinite(t_trial)) or np.any(t_trial <= 0.0):
            raise ThermoDomainError("nonfinite or nonpositive trial temperature")
        # Domain check via thermo evaluation.
        _ = thermo.specific_heat(t_trial)
        if np.max(np.abs(dt * t_tendency) / state.temperature, initial=0.0) > (
            config.epsilon_temperature * (1.0 + 1.0e-12)
        ):
            reason = "fractional-temperature bound exceeded"
        else:
            # Keep the updated enthalpy; do not replace with h(T*) after invert.
            trial_state = build_column_state(
                grid, t_trial, thermo, gravity, enthalpy=h_trial
            )
            identity = flux_divergence_identity_residual(
                state.mass_path,
                state.enthalpy,
                h_trial,
                dt,
                float(closure.flux[0]),
                float(closure.flux[-1]),
            )
            h_old = column_enthalpy_per_area(state.mass_path, state.enthalpy)
            h_new_old_mass = column_enthalpy_per_area(
                state.mass_path, trial_state.enthalpy
            )
            h_new_new_mass = column_enthalpy_per_area(
                trial_state.mass_path, trial_state.enthalpy
            )
            apparent = h_new_new_mass - h_old
            geometric = (h_new_new_mass - h_old) - (h_new_old_mass - h_old)
            trial_closure = _evaluate_closure(grid, trial_state, physics, thermo)
            min_delta = _min_active_margin(closure, trial_closure, config)
            reason = _crossing_reason(closure, trial_closure, config)
    except (ThermoDomainError, EnthalpyInversionError, HydrostaticDomainError) as exc:
        reason = f"{type(exc).__name__}: {exc}"
        trial_state = None

    accepted = reason is None and trial_state is not None
    return EnthalpyStepResult(
        state=state if not accepted else trial_state,  # type: ignore[arg-type]
        trial_state=trial_state,
        dt=dt,
        accepted=accepted,
        reason=reason,
        closure=closure,
        enthalpy_tendency=dhdt,
        identity_residual=identity,
        apparent_budget_change=apparent,
        geometric_residual=geometric,
        min_active_trial_delta_over_epsilon=min_delta,
    )


def adaptive_timestep_enthalpy(
    grid: PressureGrid,
    state: ColumnState,
    closure: ClosureResult,
    dhdt: NDArray[np.float64],
    physics: PhysicsConfig,
    config: SolverConfig,
    thermo: ThermoProvider,
) -> tuple[float, float]:
    dz = np.diff(state.z_edges)
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
    cp = thermo.specific_heat(state.temperature)
    tdot = dhdt / cp
    active_tendency = np.abs(tdot) > 0.0
    if np.any(active_tendency):
        dt_temperature = config.epsilon_temperature * float(
            np.min(
                state.temperature[active_tendency]
                / np.abs(tdot[active_tendency])
            )
        )
    else:
        dt_temperature = np.inf
    return dt_diff, dt_temperature


def _entropy_span(thermo: ThermoProvider, temperature, pressure) -> float:
    entropy = thermo.entropy(temperature, pressure)
    return float(np.max(entropy) - np.min(entropy))


def _metric_decade(value: float) -> int | None:
    if not np.isfinite(value) or value <= 0.0:
        return None
    return int(np.floor(np.log10(value)))


def _signed_enthalpy_drift(current_h: float, initial_h: float) -> float:
    return float((current_h - initial_h) / max(abs(initial_h), 1.0))


def _record_enthalpy_checkpoint(
    recorder: IntegrationTrace | None,
    *,
    accepted_step: int,
    simulated_time: float,
    dt_accepted: float,
    rejections_this_step: int,
    metrics: ConvergenceMetrics,
    signed_drift: float,
    labels: NDArray[np.int64],
    cumulative_transfer: NDArray[np.float64],
    temperature: NDArray[np.float64],
    flux: NDArray[np.float64],
    entropy_span: float,
    force_summary: bool,
) -> None:
    if recorder is None or not recorder.enabled:
        return
    recorder.record_accepted(
        accepted_step,
        simulated_time,
        dt_accepted,
        rejections_this_step,
        metrics,
        signed_drift,
        labels,
        cumulative_transfer,
        temperature,
        flux,
        snapshot_profile=False,
        entropy_span=entropy_span,
        force_summary=force_summary,
    )


def solve_adaptive_enthalpy(
    grid: PressureGrid,
    initial_temperature: ArrayLike,
    physics: PhysicsConfig,
    thermo: ThermoProvider,
    gravity: GravityLaw | None = None,
    config: SolverConfig | None = None,
    trace: IntegrationTrace | None = None,
) -> IntegrationResult:
    """Relax a closed column with enthalpy updates and atomic state commits."""
    settings = config or SolverConfig()
    grav = gravity or ConstantGravity(physics.gravity)
    enforce_enthalpy_drift = isinstance(grav, ConstantGravity)
    recorder = trace
    # Hard-fail invalid initial state immediately.
    t0 = temperatures(initial_temperature, grid.n_layers)
    state = build_column_state(grid, t0, thermo, grav)
    initial_enthalpy = column_enthalpy_per_area(state.mass_path, state.enthalpy)
    reference = numerical_isentrope(grid, t0, thermo, state.mass_path)
    simulated_time = 0.0
    total_rejections = 0
    last_dt: float | None = None
    labels = np.zeros(grid.n_layers, dtype=np.int64)
    cumulative_transfer = np.zeros(grid.n_layers + 1)
    last_metric_decade: int | None = None
    last_span_decade: int | None = None
    initial_recorded = False

    def _emit_result(
        status: TerminalStatus,
        reason: str,
        step: int,
        metrics: ConvergenceMetrics,
        closure: ClosureResult,
        *,
        failed: bool = False,
        final_dt: float | None = None,
    ) -> IntegrationResult:
        if recorder is not None and recorder.enabled:
            already = (
                recorder.accepted_steps
                and recorder.accepted_steps[-1].accepted_step == step
            )
            if not already:
                signed_drift = _signed_enthalpy_drift(
                    column_enthalpy_per_area(state.mass_path, state.enthalpy),
                    initial_enthalpy,
                )
                span = _entropy_span(thermo, state.temperature, grid.pressure_centres)
                _record_enthalpy_checkpoint(
                    recorder,
                    accepted_step=step,
                    simulated_time=simulated_time,
                    dt_accepted=0.0 if last_dt is None else last_dt,
                    rejections_this_step=0,
                    metrics=metrics,
                    signed_drift=signed_drift,
                    labels=labels,
                    cumulative_transfer=cumulative_transfer,
                    temperature=state.temperature,
                    flux=closure.flux,
                    entropy_span=span,
                    force_summary=True,
                )
            recorder.record_final(
                state.temperature,
                closure.flux,
                metrics,
                labels,
                simulated_time,
                step,
            )
        result = _result(
            state.temperature,
            status,
            reason,
            step,
            total_rejections,
            simulated_time,
            last_dt if final_dt is None else final_dt,
            metrics,
            labels,
            cumulative_transfer,
            0.0,
        )
        if failed:
            raise SolverFailure(result)
        return result

    for step in range(settings.max_steps + 1):
        # Closure is always evaluated on the currently committed state.
        # This is the post-accept flux (not the flux that produced the last update).
        closure = _evaluate_closure(grid, state, physics, thermo)
        dhdt = enthalpy_tendency(grid, closure.flux, state.mass_path)
        cp = thermo.specific_heat(state.temperature)
        t_tendency = dhdt / cp
        metrics = _metrics(
            grid,
            state,
            reference,
            t_tendency,
            closure,
            thermo,
            initial_enthalpy,
            enforce_enthalpy_drift=enforce_enthalpy_drift,
            config=settings,
        )
        if recorder is not None and recorder.enabled:
            entropy_span = _entropy_span(thermo, state.temperature, grid.pressure_centres)
            signed_drift = _signed_enthalpy_drift(
                column_enthalpy_per_area(state.mass_path, state.enthalpy),
                initial_enthalpy,
            )
            if not initial_recorded:
                recorder.record_initial(
                    state.temperature,
                    metrics,
                    labels,
                    closure.flux,
                    settings.theta_rms_tolerance,
                )
                _record_enthalpy_checkpoint(
                    recorder,
                    accepted_step=0,
                    simulated_time=0.0,
                    dt_accepted=0.0,
                    rejections_this_step=0,
                    metrics=metrics,
                    signed_drift=signed_drift,
                    labels=labels,
                    cumulative_transfer=cumulative_transfer,
                    temperature=state.temperature,
                    flux=closure.flux,
                    entropy_span=entropy_span,
                    force_summary=True,
                )
                last_metric_decade = _metric_decade(metrics.max_superadiabaticity)
                last_span_decade = _metric_decade(entropy_span)
                initial_recorded = True
            elif step >= 1:
                metric_decade = _metric_decade(metrics.max_superadiabaticity)
                span_decade = _metric_decade(entropy_span)
                decade_crossed = (
                    metric_decade is not None
                    and last_metric_decade is not None
                    and metric_decade < last_metric_decade
                ) or (
                    span_decade is not None
                    and last_span_decade is not None
                    and span_decade < last_span_decade
                )
                stride = max(recorder.summary_stride, 1)
                keep = decade_crossed or step == 1 or step % stride == 0
                if keep:
                    if decade_crossed:
                        if metric_decade is not None:
                            last_metric_decade = metric_decade
                        if span_decade is not None:
                            last_span_decade = span_decade
                    _record_enthalpy_checkpoint(
                        recorder,
                        accepted_step=step,
                        simulated_time=simulated_time,
                        dt_accepted=0.0 if last_dt is None else last_dt,
                        rejections_this_step=0,
                        metrics=metrics,
                        signed_drift=signed_drift,
                        labels=labels,
                        cumulative_transfer=cumulative_transfer,
                        temperature=state.temperature,
                        flux=closure.flux,
                        entropy_span=entropy_span,
                        force_summary=True,
                    )
        if metrics.converged(settings):
            return _emit_result(
                TerminalStatus.CONVERGED,
                "all acceptance metrics are within tolerance",
                step,
                metrics,
                closure,
            )
        if np.all(closure.thermal_diffusivity == 0.0) and np.all(dhdt == 0.0):
            return _emit_result(
                TerminalStatus.NO_ACTIVE_CONVECTION,
                "all diffusivities and tendencies are exactly zero",
                step,
                metrics,
                closure,
            )
        if step == settings.max_steps:
            # Full metrics (including θ) for the failure record.
            metrics = _metrics(
                grid,
                state,
                reference,
                t_tendency,
                closure,
                thermo,
                initial_enthalpy,
                enforce_enthalpy_drift=enforce_enthalpy_drift,
                config=None,
            )
            return _emit_result(
                TerminalStatus.FAILED,
                "maximum accepted-step limit reached",
                step,
                metrics,
                closure,
                failed=True,
            )

        dt_diff, dt_temperature = adaptive_timestep_enthalpy(
            grid, state, closure, dhdt, physics, settings, thermo
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
            # Keep a pure copy of the accepted state for rejected-trial purity.
            prior = state.copy()
            trial = trial_enthalpy_step(
                grid, state, dt, physics, settings, thermo, grav
            )
            if recorder is not None:
                recorder.record_trial(
                    step,
                    attempt,
                    dt,
                    trial.accepted,
                    trial.reason,
                    trial.min_active_trial_delta_over_epsilon,
                )
            if trial.accepted and trial.trial_state is not None:
                state = trial.trial_state
                simulated_time += dt
                last_dt = dt
                total_rejections += rejections_this_step
                accepted = True
                break
            # Rejected: ensure accepted state unchanged.
            if (
                not np.array_equal(prior.temperature, state.temperature)
                or not np.array_equal(prior.enthalpy, state.enthalpy)
                or not np.array_equal(prior.mass_path, state.mass_path)
            ):
                raise AssertionError("rejected trial mutated accepted column state")
            rejection_reason = trial.reason or "unknown rejection"
            rejections_this_step += 1
            dt *= settings.f_back
        if not accepted:
            total_rejections += rejections_this_step
            failure_dt = last_attempted_dt if last_attempted_dt is not None else dt
            _emit_result(
                TerminalStatus.FAILED,
                (
                    f"backtracking failed: {rejection_reason}; "
                    f"final attempted dt={failure_dt}"
                ),
                step,
                metrics,
                closure,
                failed=True,
                final_dt=failure_dt,
            )

    raise AssertionError("unreachable")
