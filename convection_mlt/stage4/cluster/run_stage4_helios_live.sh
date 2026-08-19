#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/project/ls-heng/Bethany.Burt"
CM_ROOT="$PROJECT_ROOT/convection_mlt"
HELIOS_ROOT="$PROJECT_ROOT/HELIOS"
VENV_DIR="$PROJECT_ROOT/venvs/stage4-helios-py312"
OUT_ROOT="$PROJECT_ROOT/helios_stage4_output"
PIN="b0800f9ea4366263241c13bb926e8ca68f266cc5"

mkdir -p "$OUT_ROOT"

# Munich physics HPC: system python3 is 3.9; PyCUDA 2026.x needs >=3.10.
if command -v module >/dev/null 2>&1; then
  module purge >/dev/null 2>&1 || true
  module load python/3.12-2024.10 >/dev/null 2>&1 || module load python/3.12-2024.06 >/dev/null 2>&1 || true
fi
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! "$PYTHON_BIN" -c 'import sys; assert sys.version_info >= (3, 10)' 2>/dev/null; then
  echo "ERROR: need Python >=3.10 for PyCUDA 2026.x; got $("$PYTHON_BIN" --version 2>&1)" >&2
  exit 1
fi
echo "Using $("$PYTHON_BIN" --version 2>&1)"

cd "$HELIOS_ROOT"
git fetch --all --tags
git checkout "$PIN"

if [ ! -f input/r50_kdistr_solar_eq.h5 ] || [ ! -f input/star_2022.h5 ]; then
  echo "Installing HELIOS input data from Zenodo..."
  if [ -x install_input_files.bash ]; then
    bash install_input_files.bash
  else
    wget -np -nH -O HELIOS_input_data.zip https://zenodo.org/records/17425932/files/HELIOS_input_data.zip
    unzip -o HELIOS_input_data.zip
  fi
fi
if [ ! -f input/r50_kdistr_solar_eq.h5 ]; then
  echo "ERROR: missing HELIOS opacity file input/r50_kdistr_solar_eq.h5 after install." >&2
  exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# Ensure CUDA toolchain is visible before building PyCUDA.
setup_cuda_toolchain() {
  if command -v module >/dev/null 2>&1; then
    module load spack/2024.07 >/dev/null 2>&1 || true
  fi
  local spack_setup="/software/opt/el_9/x86_64/spack/2024.07/spack/share/spack/setup-env.sh"
  if [ -f "$spack_setup" ]; then
    # shellcheck disable=SC1090
    . "$spack_setup"
    eval "$(spack load --sh cuda@12.4.0)"
  fi
  if ! command -v nvcc >/dev/null 2>&1; then
    for croot in /usr/local/cuda /usr/local/cuda-12.4 /usr/local/cuda-12.2 /usr/local/cuda-12.1 /usr/local/cuda-11.8 /sw/cuda /opt/cuda; do
      if [ -x "$croot/bin/nvcc" ]; then
        export PATH="$croot/bin:$PATH"
        export CUDA_HOME="$croot"
        export CUDA_ROOT="$croot"
        break
      fi
    done
  fi
  if [ -z "${CUDA_HOME:-}" ] && command -v nvcc >/dev/null 2>&1; then
    CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
    export CUDA_HOME CUDA_ROOT="$CUDA_HOME"
  fi
  if [ -n "${CUDA_HOME:-}" ]; then
    export CUDA_INC_DIR="$CUDA_HOME/include"
    export CPATH="$CUDA_HOME/include:${CPATH:-}"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
  fi
  if [ ! -e "${CUDA_HOME:-/nonexistent}/include/cuda.h" ]; then
    echo "ERROR: cuda.h not found; PyCUDA cannot build without CUDA toolkit." >&2
    echo "CUDA_HOME=${CUDA_HOME:-unset}" >&2
    echo "PATH=$PATH" >&2
    exit 1
  fi
  echo "CUDA_HOME=$CUDA_HOME"
  nvcc --version | head -1
}
setup_cuda_toolchain

python -m pip install --upgrade pip
python -m pip install numpy scipy astropy h5py matplotlib numba wget
python -m pip install --no-cache-dir pycuda
python -c "import pycuda.autoinit; print('pycuda import ok')"
python -m pip install -e "$CM_ROOT[plot,test]"

PARAM_BASE="param.dat"
PARAM_ON="param_stage4_conv_on.dat"
PARAM_OFF="param_stage4_conv_off.dat"
cp "$PARAM_BASE" "$PARAM_ON"
cp "$PARAM_BASE" "$PARAM_OFF"

python - <<'PY'
from pathlib import Path

def patch(path: Path, pairs: dict[str, str]) -> None:
    lines = path.read_text(encoding='utf-8').splitlines()
    out = []
    for ln in lines:
        done = False
        for key, value in pairs.items():
            if ln.startswith(key):
                out.append(f"{key}{value}")
                done = True
                break
        if not done:
            out.append(ln)
    path.write_text("\n".join(out) + "\n", encoding='utf-8')

on = Path('param_stage4_conv_on.dat')
off = Path('param_stage4_conv_off.dat')
common = {
    "name =                                                ": "stage4_live_conv_on",
    "output directory =                                    ": "/project/ls-heng/Bethany.Burt/helios_stage4_output/",
    "realtime plotting =                                   ": "no",
    "maximum number of iterations =                   ": "3000",
}
patch(on, common)
common_off = common.copy()
common_off["name =                                                "] = "stage4_live_conv_off"
patch(off, common_off)
patch(off, {"convective adjustment =                               ": "no"})
PY

python "$HELIOS_ROOT/helios.py" -parameter_file "$PARAM_ON"
python "$HELIOS_ROOT/helios.py" -parameter_file "$PARAM_OFF"

CONV_ON_DIR="$OUT_ROOT/stage4_live_conv_on"
CONV_OFF_DIR="$OUT_ROOT/stage4_live_conv_off"

PYTHONPATH="$CM_ROOT/src" python "$CM_ROOT/stage4/experiments/live_helios_compare.py" \
  --helios-output-dir "$CONV_ON_DIR" \
  --helios-case "stage4_live_conv_on" \
  --helios-conv-off-flux-file "$CONV_OFF_DIR/stage4_live_conv_off_integrated_flux.dat" \
  --helios-commit "$PIN" \
  --output "$CM_ROOT/stage4/results/live_helios_comparison.json"

PYTHONPATH="$CM_ROOT/src" python "$CM_ROOT/stage4/experiments/build_exit_dossier.py"

echo "Stage4 live HELIOS completed"
