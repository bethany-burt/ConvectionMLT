"""HELIOS adapter: units/orientation/shape conversion only.

No physics duplication. HELIOS integrated-flux and tp.dat use bottom-first
indexing (interface 0 = BOA), matching canonical convection_mlt orientation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
from numpy.typing import NDArray

from .helios_contracts import (
    FLUX_CGS_TO_SI,
    FLUX_SI_TO_CGS,
    GRAVITY_CGS,
    GRAVITY_SI,
    GRAVITY_SI_TO_CGS,
    MICROBAR_TO_PA,
    OPACITY_CGS_TO_SI,
    OPACITY_SI_TO_CGS,
    PA_TO_MICROBAR,
    PINNED_HELIOS_COMMIT,
    TP_PRESSURE_UNIT,
)
from .helios_grid import HeliosPressureGrid, interpolate_log_pressure

__all__ = [
    "HeliosAdapter",
    "HeliosArrays",
    "HeliosFixtureMetadata",
    "HeliosFluxProfile",
    "HeliosTPProfile",
    "PINNED_HELIOS_COMMIT",
    "assert_helios_bottom_first",
    "flux_cgs_to_si",
    "flux_si_to_cgs",
    "gravity_cgs_to_si",
    "gravity_si_to_cgs",
    "format_integrated_flux_row",
    "heating_from_net_flux",
    "layer_energy_increment",
    "load_integrated_flux",
    "load_tp_profile",
    "make_fixture_metadata",
    "opacity_cgs_to_si",
    "opacity_si_to_cgs",
    "pressure_microbar_to_pa",
    "pressure_pa_to_microbar",
    "simulate_helios_tp_read",
    "to_canonical_interfaces",
    "write_integrated_flux_stub",
    "write_param_dat",
    "write_tp_profile",
]


def flux_si_to_cgs(flux: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(flux, dtype=np.float64) * FLUX_SI_TO_CGS


def flux_cgs_to_si(flux: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(flux, dtype=np.float64) * FLUX_CGS_TO_SI


def opacity_si_to_cgs(kappa: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(kappa, dtype=np.float64) * OPACITY_SI_TO_CGS


def opacity_cgs_to_si(kappa: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(kappa, dtype=np.float64) * OPACITY_CGS_TO_SI


def pressure_pa_to_microbar(p: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(p, dtype=np.float64) * PA_TO_MICROBAR


def pressure_microbar_to_pa(p: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(p, dtype=np.float64) * MICROBAR_TO_PA


def gravity_si_to_cgs(g: float) -> float:
    return float(g) * GRAVITY_SI_TO_CGS


def gravity_cgs_to_si(g: float) -> float:
    return float(g) / GRAVITY_SI_TO_CGS


def layer_energy_increment(
    flux_net: NDArray[np.float64],
) -> NDArray[np.float64]:
    """ΔF_i = F_net,i − F_net,i+1 [W m⁻²] on canonical bottom-to-top indexing."""
    f = np.asarray(flux_net, dtype=np.float64)
    if f.size < 2:
        raise ValueError("flux_net must have at least two interfaces")
    return f[:-1] - f[1:]


def heating_from_net_flux(
    flux_net: NDArray[np.float64],
    mass_path: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Per-mass heating Q_i = ΔF_i / Δm_i [W kg⁻¹] (canonical bottom-to-top)."""
    dm = np.asarray(mass_path, dtype=np.float64)
    df = layer_energy_increment(flux_net)
    if df.size != dm.size:
        raise ValueError("flux_net must have length n_layer + 1")
    return df / dm


@dataclass(frozen=True)
class HeliosFixtureMetadata:
    helios_commit: str
    helios_config: dict
    units: dict
    orientation: str
    checksum_sha256: str


@dataclass(frozen=True)
class HeliosArrays:
    pressure_centres: NDArray[np.float64]
    temperature_centres: NDArray[np.float64]
    flux_interfaces_upward: NDArray[np.float64]


def assert_helios_bottom_first(
    pressure_microbar: NDArray[np.float64],
    values: NDArray[np.float64] | None = None,
    *,
    n_layers: int | None = None,
) -> None:
    """Verify HELIOS bottom-first interface ordering by monotonic pressure."""
    p = np.asarray(pressure_microbar, dtype=np.float64)
    if p.ndim != 1 or p.size < 2:
        raise ValueError("pressure_microbar must be a 1D array with at least 2 interfaces")
    if n_layers is not None and p.size != int(n_layers) + 1:
        raise ValueError(f"expected {int(n_layers) + 1} interface pressures, got {p.size}")
    if not np.all(np.isfinite(p)) or np.any(p <= 0):
        raise ValueError("interface pressures must be finite and strictly positive")
    if not np.all(p[:-1] > p[1:]):
        raise ValueError("HELIOS interface pressures must strictly decrease with index (bottom-first)")
    if not np.isclose(p[0], np.max(p)):
        raise ValueError("interface index 0 must be BOA (maximum pressure)")
    if values is not None:
        v = np.asarray(values, dtype=np.float64)
        if v.shape != p.shape:
            raise ValueError("values must have the same length as pressure_microbar")


def to_canonical_interfaces(
    values: NDArray[np.float64],
    pressure_microbar: NDArray[np.float64],
    *,
    n_layers: int | None = None,
) -> NDArray[np.float64]:
    """Return HELIOS interface values in canonical bottom-first order after checks."""
    assert_helios_bottom_first(pressure_microbar, values, n_layers=n_layers)
    return np.asarray(values, dtype=np.float64).copy()


class HeliosAdapter:
    """Deprecated compatibility wrapper.

    HELIOS native ordering is already canonical bottom-first. Reversal is kept
    only for the unmatched legacy pilot when ``legacy_reverse=True``.
    """

    def __init__(self, legacy_reverse: bool = False) -> None:
        self.legacy_reverse = legacy_reverse
        if legacy_reverse:
            warnings.warn(
                "HeliosAdapter(legacy_reverse=True) reverses arrays for the "
                "unmatched pilot only; frozen parity must not use reversal.",
                DeprecationWarning,
                stacklevel=2,
            )

    def to_canonical_interfaces(
        self,
        values: NDArray[np.float64],
        pressure_microbar: NDArray[np.float64] | None = None,
        *,
        n_layers: int | None = None,
    ) -> NDArray[np.float64]:
        v = np.asarray(values, dtype=np.float64)
        if self.legacy_reverse:
            return v[::-1].copy()
        if pressure_microbar is None:
            raise ValueError("pressure_microbar required when legacy_reverse=False")
        return to_canonical_interfaces(v, pressure_microbar, n_layers=n_layers)

    def to_canonical_layers(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        v = np.asarray(values, dtype=np.float64)
        return v[::-1].copy() if self.legacy_reverse else v.copy()

    def from_canonical_layers(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        v = np.asarray(values, dtype=np.float64)
        return v[::-1].copy() if self.legacy_reverse else v.copy()

    def from_canonical_interfaces(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        v = np.asarray(values, dtype=np.float64)
        return v[::-1].copy() if self.legacy_reverse else v.copy()

    def roundtrip_interfaces(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.legacy_reverse:
            return values[::-1][::-1].copy()
        return values.copy()


@dataclass(frozen=True)
class HeliosFluxProfile:
    interface_index: NDArray[np.int64]
    pressure_microbar: NDArray[np.float64]
    flux_down_cgs: NDArray[np.float64]
    flux_up_cgs: NDArray[np.float64]
    flux_net_cgs: NDArray[np.float64]
    flux_conv_net_cgs: NDArray[np.float64]
    flux_intern_cgs: NDArray[np.float64]


@dataclass(frozen=True)
class HeliosTPProfile:
    layer_index: NDArray[np.int64]
    temperature_k: NDArray[np.float64]
    pressure_microbar: NDArray[np.float64]
    conv_unstable_flag: NDArray[np.float64]
    conv_lapse_flag: NDArray[np.float64]


HELIOS_FLUX_SENTINELS = frozenset(
    {"not_avail.", "not_avail", "n/a", "nan", "none", "null", ""}
)
HELIOS_FLUX_COLUMNS = (
    "interface",
    "pressure",
    "F_down",
    "F_up",
    "F_net",
    "F_dir",
    "delta_F_net",
    "F_net_conv",
    "F_add_heat",
    "F_intern",
)
_REQUIRED_FLUX_COLUMNS = ("interface", "pressure", "F_down", "F_up", "F_net")
_FLUX_HEADER_ALIASES = {
    "interface": "interface",
    "press.[10^-6bar]": "pressure",
    "press.[10^-6 bar]": "pressure",
    "pressure": "pressure",
    "f_down": "F_down",
    "f_up": "F_up",
    "f_net": "F_net",
    "f_dir": "F_dir",
    "delta_f_net": "delta_F_net",
    "f_net_conv": "F_net_conv",
    "f_add_heat": "F_add_heat",
    "f_intern": "F_intern",
}


def parse_flux_token(token: str) -> float:
    """Parse a HELIOS flux-table token; sentinels and blanks become NaN."""
    text = token.strip()
    if text.lower() in HELIOS_FLUX_SENTINELS:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def format_integrated_flux_row(
    interface: int,
    pressure: float,
    f_down: float,
    f_up: float,
    f_net: float,
    f_dir: float = 0.0,
    delta_f_net: float | None = None,
    f_net_conv: float = 0.0,
    f_add_heat: float | None = 0.0,
    f_intern: float | None = None,
) -> str:
    """Space-delimited HELIOS integrated-flux row; width is irrelevant."""
    def _num(value: float) -> str:
        return f"{float(value):.17e}"

    def _optional(value: float | None) -> str:
        if value is None or not np.isfinite(value):
            return "not_avail."
        return _num(value)

    fields = [
        str(int(interface)),
        _num(pressure),
        _num(f_down),
        _num(f_up),
        _num(f_net),
        _num(f_dir),
        _optional(delta_f_net),
        _num(f_net_conv),
        _optional(f_add_heat),
    ]
    if f_intern is not None and np.isfinite(f_intern):
        fields.append(_num(f_intern))
    return " ".join(fields)


def _collapse_flux_header(fields: list[str]) -> list[str]:
    collapsed: list[str] = []
    i = 0
    while i < len(fields):
        token = fields[i]
        if (
            token.lower().startswith("delta_f_net")
            and i + 2 < len(fields)
            and fields[i + 1].lower() == "(layer"
            and fields[i + 2].lower().startswith("quantity")
        ):
            collapsed.append("delta_F_net")
            i += 3
            continue
        collapsed.append(token)
        i += 1
    return collapsed


def _flux_header_names(fields: list[str]) -> list[str] | None:
    names = []
    for token in _collapse_flux_header(fields):
        key = token.lower()
        if key not in _FLUX_HEADER_ALIASES:
            return None
        names.append(_FLUX_HEADER_ALIASES[key])
    if "interface" not in names or "F_net" not in names:
        return None
    return names


def _numeric_rows(path: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        fields = text.split()
        if not fields:
            continue
        token = fields[0]
        if token == "BOA":
            rows.append(fields)
            continue
        try:
            float(token)
        except ValueError:
            continue
        rows.append(fields)
    return rows


def load_integrated_flux(path: str | Path) -> HeliosFluxProfile:
    """Parse HELIOS integrated_flux.dat by header names.

    Optional diagnostics (F_dir, delta_F_net, F_net_conv, F_add_heat, F_intern)
    may be absent, blank, or ``not_avail.``; those become NaN. Comparison code
    must require only the fields it actually uses.
    """
    file_path = Path(path)
    header_names: list[str] | None = None
    rows: list[dict[str, float]] = []
    for line in file_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0].lower() == "interface":
            header_names = _flux_header_names(fields)
            continue
        if fields[0] == "BOA":
            continue
        try:
            float(fields[0])
        except ValueError:
            continue
        names = header_names if header_names is not None else list(HELIOS_FLUX_COLUMNS)
        record = {name: float("nan") for name in HELIOS_FLUX_COLUMNS}
        for name, token in zip(names, fields):
            record[name] = parse_flux_token(token)
        if any(not np.isfinite(record[name]) for name in _REQUIRED_FLUX_COLUMNS):
            raise ValueError(
                f"required flux columns {_REQUIRED_FLUX_COLUMNS} must be numeric in {file_path}: {fields}"
            )
        rows.append(record)
    if not rows:
        raise ValueError(f"no flux rows parsed from {file_path}")
    return HeliosFluxProfile(
        interface_index=np.asarray([int(r["interface"]) for r in rows], dtype=np.int64),
        pressure_microbar=np.asarray([r["pressure"] for r in rows], dtype=np.float64),
        flux_down_cgs=np.asarray([r["F_down"] for r in rows], dtype=np.float64),
        flux_up_cgs=np.asarray([r["F_up"] for r in rows], dtype=np.float64),
        flux_net_cgs=np.asarray([r["F_net"] for r in rows], dtype=np.float64),
        flux_conv_net_cgs=np.asarray([r["F_net_conv"] for r in rows], dtype=np.float64),
        flux_intern_cgs=np.asarray([r["F_intern"] for r in rows], dtype=np.float64),
    )


def load_tp_profile(path: str | Path) -> HeliosTPProfile:
    file_path = Path(path)
    rows = _numeric_rows(file_path)
    layer_index = []
    temperature = []
    pressure = []
    conv_unstable = []
    conv_lapse = []
    for row in rows:
        idx = -1 if row[0] == "BOA" else int(float(row[0]))
        layer_index.append(idx)
        if len(row) >= 3:
            temperature.append(float(row[1]))
            pressure.append(float(row[2]))
        else:
            temperature.append(np.nan)
            pressure.append(np.nan)
        if len(row) >= 7:
            conv_unstable.append(float(row[5]) if row[5] != "not_calculated" else np.nan)
            conv_lapse.append(float(row[6]) if row[6] != "not_calculated" else np.nan)
        else:
            conv_unstable.append(np.nan)
            conv_lapse.append(np.nan)
    return HeliosTPProfile(
        layer_index=np.asarray(layer_index, dtype=np.int64),
        temperature_k=np.asarray(temperature, dtype=np.float64),
        pressure_microbar=np.asarray(pressure, dtype=np.float64),
        conv_unstable_flag=np.asarray(conv_unstable, dtype=np.float64),
        conv_lapse_flag=np.asarray(conv_lapse, dtype=np.float64),
    )


def make_fixture_metadata(
    helios_commit: str,
    helios_config: dict,
    units: dict,
    orientation: str,
    arrays_for_checksum: dict[str, NDArray[np.float64]],
) -> HeliosFixtureMetadata:
    payload = {
        "helios_commit": helios_commit,
        "helios_config": helios_config,
        "units": units,
        "orientation": orientation,
    }
    h = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8"))
    for key in sorted(arrays_for_checksum.keys()):
        arr = np.asarray(arrays_for_checksum[key], dtype=np.float64)
        h.update(key.encode("utf-8"))
        h.update(arr.tobytes())
    return HeliosFixtureMetadata(
        helios_commit=helios_commit,
        helios_config=helios_config,
        units=units,
        orientation=orientation,
        checksum_sha256=h.hexdigest(),
    )


def simulate_helios_tp_read(
    file_press_microbar: NDArray[np.float64],
    file_temp_k: NDArray[np.float64],
    grid: HeliosPressureGrid,
) -> NDArray[np.float64]:
    """Simulate HELIOS read_temperature_file interpolation onto the RT grid."""
    target_pa = grid.tp_read_pressures_microbar() * MICROBAR_TO_PA
    source_pa = np.asarray(file_press_microbar, dtype=np.float64) * MICROBAR_TO_PA
    source_t = np.asarray(file_temp_k, dtype=np.float64)
    return interpolate_log_pressure(source_pa, source_t, target_pa)


def write_tp_profile(
    path: str | Path,
    *,
    temperature_boa_k: float,
    temperature_lay_k: NDArray[np.float64],
    p_int_microbar: NDArray[np.float64],
    p_lay_microbar: NDArray[np.float64],
) -> None:
    """Write HELIOS tp.dat bottom-first with exact target pressures."""
    p_int = np.asarray(p_int_microbar, dtype=np.float64)
    p_lay = np.asarray(p_lay_microbar, dtype=np.float64)
    t_lay = np.asarray(temperature_lay_k, dtype=np.float64)
    if p_lay.size != t_lay.size:
        raise ValueError("temperature_lay_k and p_lay_microbar must have equal length")
    assert_helios_bottom_first(p_int, n_layers=p_lay.size)
    lines = [
        "This file contains the corresponding layer temperatures and pressures, "
        "and the altitude and the height of each layer.",
        "layer   temp.[K]           press.[10^-6bar]          altitude[cm]         "
        "height.of.layer[cm]     conv.unstable?[1:yes,0:no]    conv.lapse-rate?[1:yes,0:no]  "
        "pl.eff.temp.[K]",
        f"BOA     {float(temperature_boa_k):.6g}               {float(p_int[0]):.6g}                   "
        f"0                    not_avail.              not_calculated                 "
        f"not_calculated                              not_avail.",
    ]
    for i, (t, p) in enumerate(zip(t_lay, p_lay)):
        lines.append(
            f"{i}       {t:.6g}                {p:.6g}                    0                    "
            f"0                        not_calculated                 not_calculated"
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_tp_profile_from_grid(
    path: str | Path,
    *,
    grid: HeliosPressureGrid,
    temperature_boa_k: float,
    temperature_lay_k: NDArray[np.float64],
) -> None:
    write_tp_profile(
        path,
        temperature_boa_k=temperature_boa_k,
        temperature_lay_k=temperature_lay_k,
        p_int_microbar=grid.p_int_microbar,
        p_lay_microbar=grid.p_lay_microbar,
    )


FIXTURES_HELIOS = Path(__file__).resolve().parents[3] / "stage4" / "fixtures" / "helios"
DEFAULT_PARAM_TEMPLATE = FIXTURES_HELIOS / "helios_param_template.dat"


def _patch_param_lines(lines: list[str], replacements: dict[str, str]) -> list[str]:
    """Replace lines by exact leading key prefix (HELIOS param.dat format)."""
    out: list[str] = []
    for ln in lines:
        replaced = False
        for prefix, new_line in replacements.items():
            if ln.startswith(prefix):
                out.append(new_line)
                replaced = True
                break
        if not replaced:
            out.append(ln)
    return out


def write_param_dat(
    path: str | Path,
    *,
    case_name: str,
    output_dir: str,
    toa_pressure_microbar: float,
    boa_pressure_microbar: float,
    opacity_path: str,
    tp_profile_path: str,
    gravity_cgs: float = GRAVITY_CGS,
    t_int_k: float,
    diffusivity_factor: float,
    scattering: bool = False,
    convective_adjustment: bool = False,
    direct_irradiation: bool = False,
    f_factor: float = 0.0,
    post_processing: bool = True,
    template_path: str | Path | None = None,
    stellar_model: str = "blackbody",
    n_layers: int | None = None,
    planet_type: str = "rocky",
    isothermal_layers: bool = True,
    energy_budget_correction: bool = False,
    surface_albedo: float = 0.0,
    use_f_approximation: bool = False,
    tp_profile_smoothing: bool = False,
    include_additional_heating: bool = False,
    coupling_mode: bool = False,
    improved_two_stream: bool = False,
    geometric_zenith_correction: bool = False,
    max_iterations: int | None = None,
) -> None:
    """Write HELIOS param.dat by patching the pinned template (keeps bracket annotations)."""
    template = Path(template_path) if template_path else DEFAULT_PARAM_TEMPLATE
    if not template.exists():
        raise FileNotFoundError(f"HELIOS param template not found: {template}")
    run_type = "post-processing" if post_processing else "iterative"
    tp_line = tp_profile_path if post_processing else "./output/0/0_tp.dat"
    lines = template.read_text(encoding="utf-8").splitlines()
    replacements = {
        "name =": f"name =                                                {case_name}                               [any string]                                 (CL: Y)",
        "output directory =": f"output directory =                                    {output_dir}                       [directory path]                             (CL: Y)",
        "realtime plotting =": "realtime plotting =                                   no                             [yes, no, number > 0]                        (CL: Y)",
        "TOA pressure [10^-6 bar] =": (
            f"TOA pressure [10^-6 bar] =                            "
            f"{float(toa_pressure_microbar):.17g}                            [number > 0]                                 (CL: Y)"
        ),
        "BOA pressure [10^-6 bar] =": (
            f"BOA pressure [10^-6 bar] =                            "
            f"{float(boa_pressure_microbar):.17g}                            [number > 0]                                 (CL: Y)"
        ),
        "run type =": f"run type =                                            {run_type}                       [iterative, post-processing]                 (CL: Y)",
        "  post-proc. --> path to temperature file =": f"  post-proc. --> path to temperature file =           {tp_line}             [file path]                                  (CL: Y)",
        "scattering =": f"scattering =                                          {'yes' if scattering else 'no'}                             [yes, no]                                    (CL: Y)",
        "direct irradiation beam =": f"direct irradiation beam =                             {'yes' if direct_irradiation else 'no'}                              [yes, no]                                    (CL: Y)",
        "  no  --> f factor =": f"  no  --> f factor =                                  {f_factor:g}                             [number: 0.25 - 1]                           (CL: Y)",
        "internal temperature [K] =": (
            f"internal temperature [K] =                            "
            f"{float(t_int_k):.17g}                              [number > 0]                                 (CL: Y)"
        ),
        "  premixed   --> path to opacity file =": f"  premixed   --> path to opacity file =               {opacity_path}  [file path]                                  (CL: Y)",
        "convective adjustment =": f"convective adjustment =                               {'yes' if convective_adjustment else 'no'}                             [yes, no]                                    (CL: Y)",
        "stellar spectral model =": f"stellar spectral model =                              {stellar_model}                            [blackbody, file]                            (CL: Y)",
        "  manual --> surface gravity [cm s^-2] =": f"  manual --> surface gravity [cm s^-2] =              {gravity_cgs:g}                            [number > 0]                                 (CL: Y)",
        "planet =": "planet =                                              manual                        [manual, name of planet]                     (CL: Y)",
        "planet type =": f"planet type =                                         {planet_type}                             [rocky, gas, no_atmosphere]                  (CL: Y)",
        "  manual --> temperature star [K] =": "  manual --> temperature star [K] =                   0                               [number >= 0]                                (CL: Y)",
        "diffusivity factor =": f"diffusivity factor =                             {diffusivity_factor:g}                          [number between 1 and 2]                                       (CL: Y)",
        "isothermal layers =": f"isothermal layers =                              {'yes' if isothermal_layers else 'no'}                         [automatic, yes, no]                                           (CL: Y)",
        "energy budget correction =": f"energy budget correction =                       {'yes' if energy_budget_correction else 'no'}                         [automatic, yes, no]                                           (CL: Y)",
        "surface albedo =": f"surface albedo =                                      {surface_albedo:g}                             [file, number: 0 - 1]                        (CL: Y)",
        "rocky planet --> use f approximation formula =": f"rocky planet --> use f approximation formula =        {'yes' if use_f_approximation else 'no'}                             [yes, no]                                    (CL: Y)",
        "TP profile smoothing =": f"TP profile smoothing =                           {'yes' if tp_profile_smoothing else 'no'}                         [yes, no]                                                      (CL: Y)",
        "include additional heating =": f"include additional heating =                     {'yes' if include_additional_heating else 'no'}                         [yes, no]                                                      (CL: Y)",
        "coupling mode =": f"coupling mode =                                       {'yes' if coupling_mode else 'no'}                              [yes, no]                                    (CL: Y)",
        "improved two stream correction =": f"improved two stream correction =                 {'yes' if improved_two_stream else 'no'}                         [yes, no]                                                      (CL: Y)",
        "geometric zenith angle correction =": f"geometric zenith angle correction =              {'no' if not geometric_zenith_correction else 'yes'}                         [automatic, yes, no]                                           (CL: Y)",
        "flux calculation method =": "flux calculation method =                        iteration                  [iteration, matrix]                                            (CL: Y)",
        "precision =": "precision =                                      double                     [double, single]                                               (CL: Y)",
    }
    if n_layers is not None:
        replacements["number of layers ="] = (
            f"number of layers =                               {int(n_layers)}                  [automatic, number > 0]                                        (CL: Y)"
        )
    if max_iterations is not None:
        replacements["maximum number of iterations ="] = (
            f"maximum number of iterations =                   {int(max_iterations)}                     [number > 0]                                                   (CL: Y)"
        )
    Path(path).write_text("\n".join(_patch_param_lines(lines, replacements)) + "\n", encoding="utf-8")


def write_integrated_flux_stub(
    path: str | Path,
    *,
    pressure_microbar: NDArray[np.float64],
    flux_down_cgs: NDArray[np.float64],
    flux_up_cgs: NDArray[np.float64],
    flux_net_cgs: NDArray[np.float64],
    flux_intern_cgs: NDArray[np.float64] | None = None,
) -> None:
    """Write space-delimited integrated_flux.dat for fixture round-trip tests."""
    intern = flux_intern_cgs
    if intern is None:
        intern = np.full(np.asarray(flux_net_cgs).shape, np.nan, dtype=np.float64)
    n = len(pressure_microbar)
    lines = [
        "This file contains the integrated total and net fluxes at each interface resp. layer.",
        "Fluxes given in [erg s^-1 cm^-2].",
        "interface press.[10^-6bar] F_down F_up F_net F_dir delta_F_net F_net_conv F_add_heat F_intern",
    ]
    for i, (p, fd, fu, fn, fi) in enumerate(
        zip(pressure_microbar, flux_down_cgs, flux_up_cgs, flux_net_cgs, intern)
    ):
        add_heat = 0.0 if i < n - 1 else None
        intern_val = None if not np.isfinite(fi) else float(fi)
        intern_val = intern_val if i == 0 else None
        lines.append(
            format_integrated_flux_row(
                i, float(p), float(fd), float(fu), float(fn),
                f_dir=0.0,
                delta_f_net=None,
                f_net_conv=0.0,
                f_add_heat=add_heat,
                f_intern=intern_val,
            )
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
