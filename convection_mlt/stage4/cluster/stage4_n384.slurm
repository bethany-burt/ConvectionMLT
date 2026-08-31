#!/usr/bin/env bash
#SBATCH --job-name=stage4-n384
#SBATCH --partition=cluster
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/project/ls-heng/Bethany.Burt/convection_mlt/stage4/results/slurm-stage4-n384-%j.out
#SBATCH --error=/project/ls-heng/Bethany.Burt/convection_mlt/stage4/results/slurm-stage4-n384-%j.err

set -euo pipefail

ROOT="${STAGE4_ROOT:-/project/ls-heng/Bethany.Burt/convection_mlt}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"

if command -v module >/dev/null 2>&1; then
  module purge >/dev/null 2>&1 || true
  module load python/3.12-2024.10 >/dev/null 2>&1 || module load python/3.12-2024.06 >/dev/null 2>&1 || true
fi

python stage4/experiments/continue_n384.py
python stage4/experiments/build_current_exit_audit.py

PYTEST=(python -m pytest)
if ! python -c "import pytest" >/dev/null 2>&1; then
  if [[ -x "${ROOT}/.venv/bin/pytest" ]]; then
    PYTEST=("${ROOT}/.venv/bin/pytest")
  fi
fi
"${PYTEST[@]}" -q \
  stage4/tests/test_rce_record_integrity.py \
  stage4/tests/test_current_exit_audit.py
