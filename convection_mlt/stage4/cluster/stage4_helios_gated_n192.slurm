#!/usr/bin/env bash
#SBATCH --job-name=stage4-helios-n192
#SBATCH --partition=cluster
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/project/ls-heng/Bethany.Burt/helios_stage4_frozen/slurm-helios-gated-n192-%j.out
#SBATCH --error=/project/ls-heng/Bethany.Burt/helios_stage4_frozen/slurm-helios-gated-n192-%j.err

set -euo pipefail

ROOT="${STAGE4_ROOT:-/project/ls-heng/Bethany.Burt/convection_mlt}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
export HELIOS_FROZEN_LAYERS=192
cd "$ROOT"

bash stage4/cluster/run_stage4_helios_gated_diag.sh
