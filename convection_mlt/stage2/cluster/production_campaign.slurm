#!/bin/bash
#SBATCH --job-name=mlt-stage2-prod
#SBATCH --output=stage2/results/slurm-%j.out
#SBATCH --error=stage2/results/slurm-%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#
# Fill these for your Munich site before first submit (examples only):
# #SBATCH --partition=YOUR_PARTITION
# #SBATCH --account=YOUR_ACCOUNT
# #SBATCH --qos=YOUR_QOS
#
# Usage (submit from the remote convection_mlt/ directory, never interactively
# on the login node):
#   sbatch stage2/cluster/production_campaign.slurm
#
# Or from your laptop:
#   export CLUSTER_HOST=Bethany.Burt@cluster.hpc.physik.uni-muenchen.de
#   export CLUSTER_DIR=/project/ls-heng/Bethany.Burt/convection_mlt
#   ./stage2/cluster/submit_from_local.sh

set -euo pipefail

echo "host=$(hostname)  job=${SLURM_JOB_ID:-none}  start=$(date -Is)"
echo "cwd=$(pwd)"

# Munich physics HPC: system /usr/bin/python3 is 3.9; package needs >=3.10.
if command -v module >/dev/null 2>&1; then
  module purge 2>/dev/null || true
  module load python/3.12-2024.10
fi

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "${REPO_ROOT}"

if [[ ! -d convection_mlt ]]; then
  # Job was submitted from inside convection_mlt/
  PKG_ROOT="${REPO_ROOT}"
else
  PKG_ROOT="${REPO_ROOT}/convection_mlt"
fi
cd "${PKG_ROOT}"

mkdir -p stage2/results

PYTHON="${PYTHON:-python3}"
if [[ -x .venv/bin/python ]]; then
  # Refuse a non-Linux / wrong-arch venv (e.g. synced from macOS).
  if ! .venv/bin/python -c 'import sys; assert sys.version_info >= (3, 10)' 2>/dev/null; then
    echo "warning: .venv is unusable on this node; falling back to module python3" >&2
    PYTHON=python3
  else
    PYTHON=".venv/bin/python"
  fi
elif [[ -x venv/bin/python ]]; then
  if ! venv/bin/python -c 'import sys; assert sys.version_info >= (3, 10)' 2>/dev/null; then
    echo "warning: venv is unusable on this node; falling back to module python3" >&2
    PYTHON=python3
  else
    PYTHON="venv/bin/python"
  fi
fi

echo "python=${PYTHON}  version=$(${PYTHON} --version 2>&1)"
${PYTHON} -c "import numpy; import convection_mlt; print('numpy', numpy.__version__)"

export PYTHONUNBUFFERED=1
${PYTHON} stage2/experiments/production_campaign.py

echo "done=$(date -Is)"
ls -la stage2/results/production_campaign.json
