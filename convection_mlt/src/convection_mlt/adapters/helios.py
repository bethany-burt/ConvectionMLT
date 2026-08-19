"""HELIOS adapter: units/orientation/shape conversion only.

No physics duplication. Converts between canonical convection_mlt orientation
(bottom-to-top layers, interfaces 0=bottom,N=top) and a HELIOS-facing layout.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


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


class HeliosAdapter:
    """Thin mapping layer.

    By default assumes HELIOS arrays are top-to-bottom and converts to canonical
    bottom-to-top. If HELIOS orientation is already canonical, set
    helios_top_to_bottom=False.
    """

    def __init__(self, helios_top_to_bottom: bool = True) -> None:
        self.helios_top_to_bottom = helios_top_to_bottom

    def to_canonical_layers(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        v = np.asarray(values, dtype=np.float64)
        return v[::-1].copy() if self.helios_top_to_bottom else v.copy()

    def from_canonical_layers(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        v = np.asarray(values, dtype=np.float64)
        return v[::-1].copy() if self.helios_top_to_bottom else v.copy()

    def to_canonical_interfaces(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        v = np.asarray(values, dtype=np.float64)
        return v[::-1].copy() if self.helios_top_to_bottom else v.copy()

    def from_canonical_interfaces(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        v = np.asarray(values, dtype=np.float64)
        return v[::-1].copy() if self.helios_top_to_bottom else v.copy()

    def roundtrip_layers(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.to_canonical_layers(self.from_canonical_layers(values))

    def roundtrip_interfaces(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.to_canonical_interfaces(self.from_canonical_interfaces(values))


@dataclass(frozen=True)
class HeliosFluxProfile:
    interface_index: NDArray[np.int64]
    pressure_microbar: NDArray[np.float64]
    flux_down_cgs: NDArray[np.float64]
    flux_up_cgs: NDArray[np.float64]
    flux_net_cgs: NDArray[np.float64]
    flux_conv_net_cgs: NDArray[np.float64]


@dataclass(frozen=True)
class HeliosTPProfile:
    layer_index: NDArray[np.int64]
    temperature_k: NDArray[np.float64]
    pressure_microbar: NDArray[np.float64]
    conv_unstable_flag: NDArray[np.float64]
    conv_lapse_flag: NDArray[np.float64]


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
    file_path = Path(path)
    rows = _numeric_rows(file_path)
    interface_index = []
    pressure = []
    flux_down = []
    flux_up = []
    flux_net = []
    flux_conv = []
    for row in rows:
        if row[0] == "BOA":
            # Not expected here, but keep parser robust.
            continue
        if len(row) < 8:
            continue
        interface_index.append(int(float(row[0])))
        pressure.append(float(row[1]))
        flux_down.append(float(row[2]))
        flux_up.append(float(row[3]))
        flux_net.append(float(row[4]))
        flux_conv.append(float(row[7]))
    return HeliosFluxProfile(
        interface_index=np.asarray(interface_index, dtype=np.int64),
        pressure_microbar=np.asarray(pressure, dtype=np.float64),
        flux_down_cgs=np.asarray(flux_down, dtype=np.float64),
        flux_up_cgs=np.asarray(flux_up, dtype=np.float64),
        flux_net_cgs=np.asarray(flux_net, dtype=np.float64),
        flux_conv_net_cgs=np.asarray(flux_conv, dtype=np.float64),
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
