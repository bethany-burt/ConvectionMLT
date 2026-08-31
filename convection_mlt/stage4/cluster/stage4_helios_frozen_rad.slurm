#!/usr/bin/env bash
#SBATCH --job-name=stage4-helios-frozen
#SBATCH --partition=cluster
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/project/ls-heng/Bethany.Burt/convection_mlt/stage4/results/slurm-helios-frozen-%j.out
#SBATCH --error=/project/ls-heng/Bethany.Burt/convection_mlt/stage4/results/slurm-helios-frozen-%j.err

set -euo pipefail

ROOT="${STAGE4_ROOT:-/project/ls-heng/Bethany.Burt/convection_mlt}"
LAYERS="${HELIOS_FROZEN_LAYERS:-96}"
MODE="${HELIOS_FROZEN_MODE:-thermal}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"

bash stage4/cluster/run_stage4_helios_frozen_rad.sh "${LAYERS}" "${MODE}"
