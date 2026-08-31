#!/usr/bin/env bash
# Frozen radiation-only HELIOS parity on shared HELIOS grid (N96-A / N192 thermal).
set -euo pipefail

LAYERS="${1:-96}"
MODE="${2:-thermal}"
ROOT="${STAGE4_ROOT:-/project/ls-heng/Bethany.Burt/convection_mlt}"
HELIOS_ROOT="${HELIOS_ROOT:-/project/ls-heng/Bethany.Burt/HELIOS}"
VENV_DIR="${HELIOS_VENV:-/project/ls-heng/Bethany.Burt/venvs/stage4-helios-py312}"
PIN="b0800f9ea4366263241c13bb926e8ca68f266cc5"
OUT_ROOT="${HELIOS_FROZEN_OUT:-/project/ls-heng/Bethany.Burt/helios_stage4_frozen}"
CASE="stage4_frozen_n${LAYERS}_${MODE}"
CASE_DIR="${OUT_ROOT}/${CASE}"
EXPORT_MODE="$([[ "${MODE}" == thermal ]] && echo thermal-only || echo "${MODE}")"
SMOKE_JSON="${ROOT}/stage4/fixtures/helios/helios_contract_smoke_n8.cluster.json"
if [[ ! -f "${SMOKE_JSON}" ]]; then
  SMOKE_JSON="${ROOT}/stage4/results/helios_contract_smoke_n8.json"
fi

if command -v module >/dev/null 2>&1; then
  module purge >/dev/null 2>&1 || true
  module load python/3.12-2024.10 >/dev/null 2>&1 || true
fi
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ "${LAYERS}" != "8" && -f "${SMOKE_JSON}" ]]; then
  smoke_status="$("${PYTHON_BIN}" - <<PY
import json
print(json.loads(open("${SMOKE_JSON}").read()).get("status", "NOT_RUN"))
PY
)"
  if [[ "${smoke_status}" != "PASS" ]]; then
    echo "Refusing N${LAYERS}: helios_contract_smoke_n8 status=${smoke_status} (require PASS)" >&2
    exit 2
  fi
fi

mkdir -p "${CASE_DIR}" "${ROOT}/stage4/fixtures/helios" "${ROOT}/stage4/results"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
cd "${ROOT}"

"${PYTHON_BIN}" stage4/experiments/verify_frozen_inputs.py \
  --require analytic_grey_nested.h5 radiation_only_tolerances.json \
            helios_write_integrated_flux_b0800f9.patch \
            helios_write_integrated_flux_b0800f9.patch.json

FIXTURE_REF="${ROOT}/stage4/fixtures/helios/helios_grid_n${LAYERS}_${MODE}_reference.json"
RESULT_REF="${ROOT}/stage4/results/helios_grid_n${LAYERS}_${MODE}_reference.json"
if [[ -f "${FIXTURE_REF}" ]]; then
  REF_JSON="${FIXTURE_REF}"
elif [[ -f "${RESULT_REF}" ]]; then
  REF_JSON="${RESULT_REF}"
else
  echo "missing frozen reference ${FIXTURE_REF} (and ${RESULT_REF})" >&2
  exit 2
fi
RUNTIME_JSON="${CASE_DIR}/helios_runtime_config.json"
"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path
import numpy as np
from convection_mlt.adapters.helios import write_param_dat, write_tp_profile
from convection_mlt.adapters.helios_grid import build_helios_pressure_grid

root = Path("${ROOT}")
ref = json.loads(Path("${REF_JSON}").read_text())
grid = build_helios_pressure_grid(
    p_boa_microbar=ref["grid"]["p_boa_microbar"],
    p_toa_microbar=ref["grid"]["p_toa_microbar"],
    n_layers=int(ref["n_layers"]),
)
write_tp_profile(
    "${CASE_DIR}/${CASE}_tp.dat",
    temperature_boa_k=float(ref["frozen"]["temperature_boa_k"]),
    temperature_lay_k=np.asarray(ref["frozen"]["temperature_lay_k"], dtype=float),
    p_int_microbar=grid.p_int_microbar,
    p_lay_microbar=grid.p_lay_microbar,
)
write_param_dat(
    "${CASE_DIR}/param.dat",
    case_name="${CASE}",
    output_dir="${CASE_DIR}/",
    toa_pressure_microbar=float(ref["grid"]["p_toa_microbar"]),
    boa_pressure_microbar=float(ref["grid"]["p_boa_microbar"]),
    opacity_path="${ROOT}/stage4/fixtures/helios/analytic_grey_nested.h5",
    tp_profile_path="${CASE_DIR}/${CASE}_tp.dat",
    t_int_k=float(ref["contracts"]["internal_flux_temperature_k"]),
    diffusivity_factor=float(ref["contracts"]["diffusivity_factor"]),
    scattering=False,
    convective_adjustment=False,
    direct_irradiation=False,
    post_processing=True,
    n_layers=int(ref["n_layers"]),
    planet_type="rocky",
)
PY

cd "${HELIOS_ROOT}"
git fetch --all --tags
git checkout "${PIN}"
git checkout "${PIN}" -- source/write.py
"${PYTHON_BIN}" "${ROOT}/stage4/experiments/apply_helios_write_precision.py" \
  --write-py "${HELIOS_ROOT}/source/write.py" \
  --require-patch-checksum

if [[ ! -d "${VENV_DIR}" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
python -c "import pycuda.autoinit" 2>/dev/null || {
  if command -v module >/dev/null 2>&1; then
    module load spack/2024.07 >/dev/null 2>&1 || true
  fi
  spack_setup="/software/opt/el_9/x86_64/spack/2024.07/spack/share/spack/setup-env.sh"
  if [[ -f "${spack_setup}" ]]; then
    # shellcheck disable=SC1090
    . "${spack_setup}"
    eval "$(spack load --sh cuda@12.4.0)"
  fi
  python -m pip install --upgrade pip
  python -c "import pycuda.autoinit; print('pycuda ok after install')"
}

cd "${HELIOS_ROOT}"
PARAM_NAME="param_${CASE}.dat"
cp "${CASE_DIR}/param.dat" "${HELIOS_ROOT}/${PARAM_NAME}"
python "${HELIOS_ROOT}/helios.py" -parameter_file "${PARAM_NAME}"

FLUX="${CASE_DIR}/${CASE}/${CASE}_integrated_flux.dat"
if [[ ! -f "${FLUX}" ]]; then
  FLUX="${CASE_DIR}/${CASE}_integrated_flux.dat"
fi

cd "${ROOT}"
"${PYTHON_BIN}" - <<PY > "${RUNTIME_JSON}"
import json, re, sys
from pathlib import Path
sys.path.insert(0, "${ROOT}/stage4/experiments")
from apply_helios_write_precision import patch_provenance
text = Path("${CASE_DIR}/param.dat").read_text()
def grab(key):
    import re
    m = re.search(rf"^{re.escape(key)}\s+(.+?)\s+\[", text, flags=re.MULTILINE)
    return m.group(1).strip() if m else None
payload = {
    "helios_commit": "${PIN}",
    "n_layers": int(float(grab("number of layers =").split()[0])),
    "diffusivity_factor": float(grab("diffusivity factor =").split()[0]),
    "precision": grab("precision ="),
    "scattering": grab("scattering ="),
    "convective_adjustment": grab("convective adjustment ="),
    "direct_irradiation": grab("direct irradiation beam ="),
    "planet_type": grab("planet type ="),
    "internal_flux_temperature_k": float(grab("internal temperature [K] =").split()[0]),
    "p_boa_microbar": float(grab("BOA pressure [10^-6 bar] =").split()[0]),
    "p_toa_microbar": float(grab("TOA pressure [10^-6 bar] =").split()[0]),
    "run_type": grab("run type ="),
    "stellar_model": grab("stellar spectral model ="),
}
payload.update(patch_provenance())
print(json.dumps(payload, indent=2))
PY

STRUCTURAL=""
if [[ "${MODE}" == irradiated ]]; then
  STRUCTURAL="--structural-only"
fi
python stage4/experiments/compare_frozen_radiation.py \
  --layers "${LAYERS}" \
  --mode "${MODE}" \
  --helios-flux "${FLUX}" \
  --reference "${REF_JSON}" \
  --runtime-config "${RUNTIME_JSON}" \
  --output "stage4/results/helios_frozen_rad_n${LAYERS}_${MODE}.json" \
  ${STRUCTURAL}

echo "Frozen HELIOS radiation-only ${CASE} completed"
