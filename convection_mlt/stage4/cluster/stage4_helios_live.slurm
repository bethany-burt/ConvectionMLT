#!/usr/bin/env bash
#SBATCH --job-name=stage4-helios-live
#SBATCH --partition=cluster
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/project/ls-heng/Bethany.Burt/convection_mlt/stage4/results/slurm-stage4-helios-live-%j.out
#SBATCH --error=/project/ls-heng/Bethany.Burt/convection_mlt/stage4/results/slurm-stage4-helios-live-%j.err

set -euo pipefail

if command -v module >/dev/null 2>&1; then
  module purge >/dev/null 2>&1 || true
  module load python/3.12-2024.10 >/dev/null 2>&1 || module load python/3.12-2024.06 >/dev/null 2>&1 || true
  module load spack/2024.07 >/dev/null 2>&1 || true
fi

bash /project/ls-heng/Bethany.Burt/convection_mlt/stage4/cluster/run_stage4_helios_live.sh
