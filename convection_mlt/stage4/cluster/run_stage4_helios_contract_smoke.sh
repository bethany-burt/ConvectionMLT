#!/usr/bin/env bash
# N=8 HELIOS contract smoke (must PASS before formal N96-A).
set -euo pipefail

ROOT="${STAGE4_ROOT:-/project/ls-heng/Bethany.Burt/convection_mlt}"
HELIOS_ROOT="${HELIOS_ROOT:-/project/ls-heng/Bethany.Burt/HELIOS}"
VENV_DIR="${HELIOS_VENV:-/project/ls-heng/Bethany.Burt/venvs/stage4-helios-py312}"
PIN="b0800f9ea4366263241c13bb926e8ca68f266cc5"
OUT_ROOT="${HELIOS_FROZEN_OUT:-/project/ls-heng/Bethany.Burt/helios_stage4_frozen}"
CASE_DIR="${OUT_ROOT}/helios_contract_smoke_n8"

setup_cuda_toolchain() {
  if command -v module >/dev/null 2>&1; then
    module load spack/2024.07 >/dev/null 2>&1 || true
  fi
  local spack_setup="/software/opt/el_9/x86_64/spack/2024.07/spack/share/spack/setup-env.sh"
  if [[ -f "${spack_setup}" ]]; then
    # shellcheck disable=SC1090
    . "${spack_setup}"
    eval "$(spack load --sh cuda@12.4.0)"
  fi
}

if command -v module >/dev/null 2>&1; then
  module purge >/dev/null 2>&1 || true
  module load python/3.12-2024.10 >/dev/null 2>&1 || true
fi
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p "${CASE_DIR}" "${ROOT}/stage4/fixtures/helios" "${ROOT}/stage4/results"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
cd "${ROOT}"

"${PYTHON_BIN}" stage4/experiments/verify_frozen_inputs.py \
  --require analytic_grey_nested.h5 radiation_only_tolerances.json

"${PYTHON_BIN}" stage4/experiments/helios_contract_smoke.py \
  --case-dir "${CASE_DIR}" \
  --prepare-only \
  --output "${ROOT}/stage4/results/helios_contract_smoke_n8_prepare.json"

cd "${HELIOS_ROOT}"
git fetch --all --tags
git checkout "${PIN}"

if [[ ! -d "${VENV_DIR}" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
setup_cuda_toolchain
python -c "import pycuda.autoinit; print('pycuda ok')" || {
  python -m pip install --upgrade pip
  python -m pip install numpy scipy astropy h5py numba pycuda
  setup_cuda_toolchain
  python -c "import pycuda.autoinit; print('pycuda ok after install')"
}

cd "${HELIOS_ROOT}"
PARAM_NAME="param_helios_contract_smoke_n8.dat"
cp "${CASE_DIR}/param.dat" "${HELIOS_ROOT}/${PARAM_NAME}"
python "${HELIOS_ROOT}/helios.py" -parameter_file "${PARAM_NAME}"

FLUX="${CASE_DIR}/helios_contract_smoke_n8/helios_contract_smoke_n8_integrated_flux.dat"
if [[ ! -f "${FLUX}" ]]; then
  FLUX="${CASE_DIR}/helios_contract_smoke_n8_integrated_flux.dat"
fi

cd "${ROOT}"
python stage4/experiments/helios_contract_smoke.py \
  --case-dir "${CASE_DIR}" \
  --helios-flux "${FLUX}" \
  --output "stage4/results/helios_contract_smoke_n8.json"

echo "HELIOS contract smoke completed"
