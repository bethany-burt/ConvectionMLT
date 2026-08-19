"""Bounded observational tracing for Stage 1 validation plots.

Trace levels never mutate solver physics. Rejected trials may be recorded, but
they must not alter accepted temperature, labels, transfers, or references.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .diagnostics import ConvergenceMetrics


class TraceLevel(str, Enum):
    NONE = "none"
    SUMMARY = "summary"
    PROFILES = "profiles"
    TRIALS = "trials"


@dataclass
class TrialRecord:
    accepted_step: int
    attempt: int
    dt: float
    accepted: bool
    reason: str | None
    min_active_trial_delta_over_epsilon: float | None


@dataclass
class AcceptedStepRecord:
    accepted_step: int
    simulated_time: float
    dt_accepted: float
    rejections_this_step: int
    metrics: ConvergenceMetrics
    signed_enthalpy_drift: float
    region_labels: NDArray[np.int64]
    cumulative_transfer: NDArray[np.float64]
    entropy_span: float | None = None


@dataclass
class ProfileSnapshot:
    accepted_step: int
    simulated_time: float
    temperature: NDArray[np.float64]
    flux: NDArray[np.float64]
    potential_temperature_rms: float
    region_labels: NDArray[np.int64]


@dataclass
class IntegrationTrace:
    """Mutable recorder attached optionally to ``solve_adaptive``."""

    level: TraceLevel = TraceLevel.NONE
    summary_stride: int = 1
    max_summary_records: int = 4096
    theta_rms_targets: list[float] = field(default_factory=list)
    initial_temperature: NDArray[np.float64] | None = None
    final_temperature: NDArray[np.float64] | None = None
    final_flux: NDArray[np.float64] | None = None
    accepted_steps: list[AcceptedStepRecord] = field(default_factory=list)
    trials: list[TrialRecord] = field(default_factory=list)
    profiles: list[ProfileSnapshot] = field(default_factory=list)
    extrema: dict[str, float] = field(default_factory=dict)
    totals: dict[str, float | int] = field(default_factory=dict)
    _targets_hit: set[int] = field(default_factory=set, repr=False)
    _initial_theta_rms: float | None = field(default=None, repr=False)

    @property
    def enabled(self) -> bool:
        return self.level is not TraceLevel.NONE

    def record_initial(
        self,
        temperature: NDArray[np.float64],
        metrics: ConvergenceMetrics,
        labels: NDArray[np.int64],
        flux: NDArray[np.float64],
        theta_tolerance: float,
    ) -> None:
        if not self.enabled:
            return
        self.initial_temperature = temperature.copy()
        self._initial_theta_rms = max(metrics.potential_temperature_rms, theta_tolerance)
        if self.level in (TraceLevel.PROFILES, TraceLevel.TRIALS):
            if not self.theta_rms_targets:
                start = self._initial_theta_rms
                if start is not None and np.isfinite(start) and start > 0.0:
                    self.theta_rms_targets = list(
                        np.geomspace(
                            start,
                            max(theta_tolerance, np.finfo(float).tiny),
                            8,
                        )
                    )
            self.profiles.append(
                ProfileSnapshot(
                    0,
                    0.0,
                    temperature.copy(),
                    flux.copy(),
                    metrics.potential_temperature_rms,
                    labels.copy(),
                )
            )
        self._update_extrema(metrics, metrics.enthalpy_drift)
        self.totals.setdefault("accepted_steps", 0)
        self.totals.setdefault("rejections", 0)

    def record_trial(
        self,
        accepted_step: int,
        attempt: int,
        dt: float,
        accepted: bool,
        reason: str | None,
        min_active_trial_delta_over_epsilon: float | None,
    ) -> None:
        if self.level is not TraceLevel.TRIALS:
            return
        self.trials.append(
            TrialRecord(
                accepted_step,
                attempt,
                float(dt),
                accepted,
                reason,
                min_active_trial_delta_over_epsilon,
            )
        )

    def record_accepted(
        self,
        accepted_step: int,
        simulated_time: float,
        dt_accepted: float,
        rejections_this_step: int,
        metrics: ConvergenceMetrics,
        signed_enthalpy_drift: float,
        labels: NDArray[np.int64],
        cumulative_transfer: NDArray[np.float64],
        temperature: NDArray[np.float64],
        flux: NDArray[np.float64],
        *,
        snapshot_profile: bool = True,
        entropy_span: float | None = None,
        force_summary: bool = False,
    ) -> None:
        if not self.enabled:
            return
        self.totals["accepted_steps"] = int(accepted_step)
        self.totals["rejections"] = int(
            self.totals.get("rejections", 0)
        ) + int(rejections_this_step)
        self.totals["simulated_time"] = float(simulated_time)
        self._update_extrema(metrics, signed_enthalpy_drift)
        keep_summary = (
            force_summary
            or accepted_step % max(self.summary_stride, 1) == 0
            or accepted_step == 1
            or accepted_step == 0
        )
        if keep_summary:
            self.accepted_steps.append(
                AcceptedStepRecord(
                    accepted_step,
                    float(simulated_time),
                    float(dt_accepted),
                    int(rejections_this_step),
                    metrics,
                    float(signed_enthalpy_drift),
                    labels.copy(),
                    cumulative_transfer.copy(),
                    None if entropy_span is None else float(entropy_span),
                )
            )
            self._bound_summary_history()
        if snapshot_profile and self.level in (TraceLevel.PROFILES, TraceLevel.TRIALS):
            self._maybe_snapshot_profile(
                accepted_step,
                simulated_time,
                temperature,
                flux,
                metrics.potential_temperature_rms,
                labels,
            )

    def record_final(
        self,
        temperature: NDArray[np.float64],
        flux: NDArray[np.float64],
        metrics: ConvergenceMetrics,
        labels: NDArray[np.int64],
        simulated_time: float,
        accepted_step: int,
    ) -> None:
        if not self.enabled:
            return
        self.final_temperature = temperature.copy()
        self.final_flux = flux.copy()
        self._update_extrema(metrics, metrics.enthalpy_drift)
        if self.level in (TraceLevel.PROFILES, TraceLevel.TRIALS):
            if not self.profiles or self.profiles[-1].accepted_step != accepted_step:
                self.profiles.append(
                    ProfileSnapshot(
                        accepted_step,
                        float(simulated_time),
                        temperature.copy(),
                        flux.copy(),
                        metrics.potential_temperature_rms,
                        labels.copy(),
                    )
                )

    def _maybe_snapshot_profile(
        self,
        accepted_step: int,
        simulated_time: float,
        temperature: NDArray[np.float64],
        flux: NDArray[np.float64],
        theta_rms: float,
        labels: NDArray[np.int64],
    ) -> None:
        for index, target in enumerate(self.theta_rms_targets):
            if index in self._targets_hit:
                continue
            if theta_rms <= target:
                self._targets_hit.add(index)
                self.profiles.append(
                    ProfileSnapshot(
                        accepted_step,
                        float(simulated_time),
                        temperature.copy(),
                        flux.copy(),
                        float(theta_rms),
                        labels.copy(),
                    )
                )

    def _bound_summary_history(self) -> None:
        """Increase decimation as needed while retaining the first record."""
        limit = max(int(self.max_summary_records), 2)
        while len(self.accepted_steps) > limit:
            self.summary_stride *= 2
            self.accepted_steps = [
                item
                for item in self.accepted_steps
                if item.accepted_step in (0, 1)
                or item.accepted_step % self.summary_stride == 0
            ]

    def _update_extrema(
        self, metrics: ConvergenceMetrics, signed_enthalpy_drift: float
    ) -> None:
        values = {
            "max_abs_enthalpy_drift": abs(signed_enthalpy_drift),
            "max_superadiabaticity": metrics.max_superadiabaticity,
            "max_potential_temperature_rms": metrics.potential_temperature_rms,
            "max_temperature_rms": metrics.temperature_rms,
            "max_temperature_max": metrics.temperature_max,
            "max_normalized_tendency": metrics.normalized_tendency_max,
            "max_convective_flux": metrics.convective_flux_max,
        }
        for key, value in values.items():
            previous = self.extrema.get(key)
            self.extrema[key] = (
                float(value) if previous is None else max(float(previous), float(value))
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level.value,
            "summary_stride": self.summary_stride,
            "max_summary_records": self.max_summary_records,
            "theta_rms_targets": list(self.theta_rms_targets),
            "extrema": dict(self.extrema),
            "totals": dict(self.totals),
            "accepted_steps": [
                {
                    "accepted_step": item.accepted_step,
                    "simulated_time": item.simulated_time,
                    "dt_accepted": item.dt_accepted,
                    "rejections_this_step": item.rejections_this_step,
                    "metrics": item.metrics.as_dict(),
                    "signed_enthalpy_drift": item.signed_enthalpy_drift,
                    "region_labels": item.region_labels.tolist(),
                    "cumulative_transfer": item.cumulative_transfer.tolist(),
                    "entropy_span": item.entropy_span,
                }
                for item in self.accepted_steps
            ],
            "trials": [asdict(item) for item in self.trials],
            "profiles": [
                {
                    "accepted_step": item.accepted_step,
                    "simulated_time": item.simulated_time,
                    "temperature": item.temperature.tolist(),
                    "flux": item.flux.tolist(),
                    "potential_temperature_rms": item.potential_temperature_rms,
                    "region_labels": item.region_labels.tolist(),
                }
                for item in self.profiles
            ],
            "initial_temperature": (
                None
                if self.initial_temperature is None
                else self.initial_temperature.tolist()
            ),
            "final_temperature": (
                None
                if self.final_temperature is None
                else self.final_temperature.tolist()
            ),
            "final_flux": (
                None if self.final_flux is None else self.final_flux.tolist()
            ),
        }


def make_trace(level: TraceLevel | str = TraceLevel.NONE) -> IntegrationTrace | None:
    resolved = TraceLevel(level)
    if resolved is TraceLevel.NONE:
        return None
    return IntegrationTrace(level=resolved)
