#!/usr/bin/env bash
#SBATCH --job-name=stage4-helios-n192-rcb
#SBATCH --partition=cluster
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=slurm-helios-n192-rcb-%j.out
#SBATCH --error=slurm-helios-n192-rcb-%j.err

set -euo pipefail
ROOT="${STAGE4_ROOT:-/project/ls-heng/Bethany.Burt/convection_mlt}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "${ROOT}/stage4/results"
cd "${ROOT}"
bash stage4/cluster/run_stage4_helios_rcb_resolution_n192.sh
