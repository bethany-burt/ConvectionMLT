#!/usr/bin/env bash
# N=96 coupled HELIOS RCE pilot: iterative + convective adjustment.
# Benchmark against nested MLT, not solver identity. Does not claim full Stage 4.
set -euo pipefail

LAYERS="${1:-96}"
ROOT="${STAGE4_ROOT:-/project/ls-heng/Bethany.Burt/convection_mlt}"
HELIOS_ROOT="${HELIOS_ROOT:-/project/ls-heng/Bethany.Burt/HELIOS}"
VENV_DIR="${HELIOS_VENV:-/project/ls-heng/Bethany.Burt/venvs/stage4-helios-py312}"
PIN="b0800f9ea4366263241c13bb926e8ca68f266cc5"
OUT_ROOT="${HELIOS_COUPLED_OUT:-/project/ls-heng/Bethany.Burt/helios_stage4_coupled}"
CASE="stage4_coupled_n${LAYERS}"
CASE_DIR="${OUT_ROOT}/${CASE}"
OPACITY="${ROOT}/stage4/fixtures/helios/analytic_grey_nested.h5"
TOL="${ROOT}/stage4/fixtures/helios/coupled_rce_benchmark_tolerances.json"
SMOKE_JSON="${ROOT}/stage4/fixtures/helios/helios_contract_smoke_n8.cluster.json"

if command -v module >/dev/null 2>&1; then
  module purge >/dev/null 2>&1 || true
  module load python/3.12-2024.10 >/dev/null 2>&1 || true
fi
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ ! -f "${TOL}" ]]; then
  echo "missing frozen coupled tolerances ${TOL}" >&2
  exit 2
fi
frozen="$("${PYTHON_BIN}" - <<PY
import json
print(json.loads(open("${TOL}").read()).get("frozen_before_live", False))
PY
)"
if [[ "${frozen}" != "True" ]]; then
  echo "Refusing coupled HELIOS: tolerances not frozen_before_live" >&2
  exit 2
fi

if [[ -f "${SMOKE_JSON}" ]]; then
  smoke_status="$("${PYTHON_BIN}" - <<PY
import json
print(json.loads(open("${SMOKE_JSON}").read()).get("status", "NOT_RUN"))
PY
)"
  if [[ "${smoke_status}" != "PASS" ]]; then
    echo "Refusing N${LAYERS} coupled: helios_contract_smoke_n8 status=${smoke_status}" >&2
    exit 2
  fi
fi

mkdir -p "${CASE_DIR}" "${ROOT}/stage4/results"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
cd "${ROOT}"

"${PYTHON_BIN}" stage4/experiments/verify_frozen_inputs.py \
  --require analytic_grey_nested.h5 \
            helios_write_integrated_flux_b0800f9.patch \
            helios_write_integrated_flux_b0800f9.patch.json

"${PYTHON_BIN}" stage4/experiments/export_coupled_helios_case.py \
  --layers "${LAYERS}" \
  --case-dir "${CASE_DIR}" \
  --opacity "${OPACITY}"

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
}

cd "${HELIOS_ROOT}"
PARAM_NAME="param_${CASE}.dat"
cp "${CASE_DIR}/param.dat" "${HELIOS_ROOT}/${PARAM_NAME}"
python "${HELIOS_ROOT}/helios.py" -parameter_file "${PARAM_NAME}"

TP="${CASE_DIR}/${CASE}/${CASE}_tp.dat"
FLUX="${CASE_DIR}/${CASE}/${CASE}_integrated_flux.dat"
if [[ ! -f "${TP}" ]]; then
  TP="${CASE_DIR}/${CASE}_tp.dat"
fi
if [[ ! -f "${FLUX}" ]]; then
  FLUX="${CASE_DIR}/${CASE}_integrated_flux.dat"
fi

cd "${ROOT}"
python stage4/experiments/compare_coupled_helios_rce.py \
  --layers "${LAYERS}" \
  --helios-tp "${TP}" \
  --helios-flux "${FLUX}" \
  --runtime-config "${CASE_DIR}/helios_runtime_config.json" \
  --tolerances "${TOL}" \
  --output "stage4/results/helios_coupled_rce_n${LAYERS}.json"

echo "Coupled HELIOS N=${LAYERS} pilot completed (benchmark, not Stage-4 headline)"
