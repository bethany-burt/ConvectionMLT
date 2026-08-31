#!/usr/bin/env bash
# Gated HELIOS radiation-only sequence.
# N=96: tagged layer → analytic layer → constant-κ flux → analytic flux.
# N=192: require N=96 flux PASS; tagged layer → analytic layer → analytic flux.
# Constant-κ is not repeated at N=192. Coupled RCE is not launched.
set -euo pipefail

ROOT="${STAGE4_ROOT:-/project/ls-heng/Bethany.Burt/convection_mlt}"
LAYERS="${HELIOS_FROZEN_LAYERS:-96}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
cd "${ROOT}"

if command -v module >/dev/null 2>&1; then
  module purge >/dev/null 2>&1 || true
  module load python/3.12-2024.10 >/dev/null 2>&1 || true
fi
PYTHON_BIN="${PYTHON_BIN:-python3}"

gate() {
  "${PYTHON_BIN}" stage4/experiments/gate_helios_diag_step.py --result "$1" --label "$2"
}

if [[ "${LAYERS}" == "192" ]]; then
  echo "=== gated prerequisite: N=96 radiation-only PASS ==="
  gate "${ROOT}/stage4/results/helios_frozen_rad_n96_thermal.json" n96_radiation_only
fi

echo "=== gated step 0: HDF5 HELIOS-index round-trip ==="
"${PYTHON_BIN}" stage4/experiments/verify_helios_hdf5_index.py \
  --table "${ROOT}/stage4/fixtures/helios/pressure_tagged.h5"
"${PYTHON_BIN}" stage4/experiments/verify_helios_hdf5_index.py \
  --table "${ROOT}/stage4/fixtures/helios/analytic_grey_nested.h5"
"${PYTHON_BIN}" stage4/experiments/verify_helios_hdf5_index.py \
  --table "${ROOT}/stage4/fixtures/helios/constant_grey.h5"

echo "=== gated step 1: pressure-tagged layer opacity (κ∝P, Δτ∝P²) ==="
bash stage4/cluster/run_stage4_helios_opacity_diag.sh tagged "${LAYERS}"
gate "${ROOT}/stage4/results/helios_layer_opacity_n${LAYERS}_tagged.json" tagged_layer

echo "=== gated step 2: analytic layer opacity (κ∝P^{1/2}, Δτ∝P^{3/2}) ==="
bash stage4/cluster/run_stage4_helios_opacity_diag.sh analytic "${LAYERS}"
gate "${ROOT}/stage4/results/helios_layer_opacity_n${LAYERS}_analytic.json" analytic_layer

if [[ "${LAYERS}" != "192" ]]; then
  echo "=== gated step 3: constant-κ flux + layer opacity ==="
  bash stage4/cluster/run_stage4_helios_opacity_diag.sh constant "${LAYERS}"
  gate "${ROOT}/stage4/results/helios_layer_opacity_n${LAYERS}_constant.json" constant_layer
  gate "${ROOT}/stage4/results/helios_opacity_diag_n${LAYERS}_constant.json" constant_flux
else
  echo "=== gated step 3 skipped: constant-κ already scored at N=96 ==="
fi

echo "=== gated step 4: formal analytic N=${LAYERS} flux/heating/energy-increment parity ==="
bash stage4/cluster/run_stage4_helios_frozen_rad.sh "${LAYERS}" thermal
gate "${ROOT}/stage4/results/helios_frozen_rad_n${LAYERS}_thermal.json" analytic_n${LAYERS}_flux

echo "Gated HELIOS radiation-only sequence completed for N=${LAYERS}"
