#!/usr/bin/env bash
# Labelled HELIOS N=192 resolution test for RCB attribution.
# Same physics as N=96 pilot (iso=no, convective adjustment, F_irr=0).
# Seeds from N=96 MLT sampled onto N=192 HELIOS grid. Not a Stage-4 headline.
set -euo pipefail

LAYERS=192
ROOT="${STAGE4_ROOT:-/project/ls-heng/Bethany.Burt/convection_mlt}"
HELIOS_ROOT="${HELIOS_ROOT:-/project/ls-heng/Bethany.Burt/HELIOS}"
VENV_DIR="${HELIOS_VENV:-/project/ls-heng/Bethany.Burt/venvs/stage4-helios-py312}"
PIN="b0800f9ea4366263241c13bb926e8ca68f266cc5"
OUT_ROOT="${HELIOS_COUPLED_OUT:-/project/ls-heng/Bethany.Burt/helios_stage4_rcb_attrib}"
CASE="stage4_coupled_n${LAYERS}"
CASE_DIR="${OUT_ROOT}/${CASE}_resolution"
OPACITY="${ROOT}/stage4/fixtures/helios/analytic_grey_nested.h5"
ATTRIB_JSON="${ROOT}/stage4/results/helios_coupled_n96_rcb_attribution.json"
RESULT_JSON="${ROOT}/stage4/results/helios_coupled_n192_resolution_rcb.json"

if command -v module >/dev/null 2>&1; then
  module purge >/dev/null 2>&1 || true
  module load python/3.12-2024.10 >/dev/null 2>&1 || true
fi
if [[ -f "${VENV_DIR}/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_DIR}/bin/activate"
fi
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "${HELIOS_ROOT}"
git rev-parse HEAD | grep -q "^${PIN}$" || {
  echo "HELIOS not at pinned commit ${PIN}" >&2
  exit 2
}

mkdir -p "${CASE_DIR}" "${ROOT}/stage4/results"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
cd "${ROOT}"

"${PYTHON_BIN}" stage4/experiments/export_coupled_helios_case.py \
  --layers "${LAYERS}" \
  --case-dir "${CASE_DIR}" \
  --opacity "${OPACITY}"

# Patch absolute paths for HELIOS cwd.
python3 - <<PY
from pathlib import Path
import re
param = Path("${CASE_DIR}/param.dat")
tp = Path("${CASE_DIR}/stage4_coupled_n${LAYERS}_tp.dat")
# export names case stage4_coupled_n192
cands = list(Path("${CASE_DIR}").glob("*_tp.dat"))
tp = cands[0] if cands else tp
text = param.read_text()
lines = []
for ln in text.splitlines():
    if re.match(r"^output directory\s*=", ln):
        m = re.search(r"(\s{2,}\[.*)$", ln)
        lines.append("output directory =                                    ${CASE_DIR}/" + (m.group(1) if m else ""))
    elif re.match(r"^path to temperature file\s*=", ln):
        m = re.search(r"(\s{2,}\[.*)$", ln)
        lines.append(f"path to temperature file =                           {tp}" + (m.group(1) if m else ""))
    elif re.match(r"^  premixed   --> path to opacity file\s*=", ln):
        m = re.search(r"(\s{2,}\[.*)$", ln)
        lines.append("  premixed   --> path to opacity file =               ${OPACITY}" + (m.group(1) if m else ""))
    else:
        lines.append(ln)
param.write_text("\n".join(lines) + "\n")
print("patched", param, "tp", tp)
PY

run_helios_param() {
  local param_path="$1"
  local log_path="$2"
  local stamp_name="n192_resolution_param.dat"
  cp -f "${param_path}" "${HELIOS_ROOT}/${stamp_name}"
  cd "${HELIOS_ROOT}"
  set +e
  python helios.py -parameter_file "${stamp_name}" 2>&1 | tee "${log_path}"
  local rc=${PIPESTATUS[0]}
  set -e
  cd "${ROOT}"
  return "${rc}"
}

HELIOS_LOG="${CASE_DIR}/helios_stdout.log"
set +e
run_helios_param "${CASE_DIR}/param.dat" "${HELIOS_LOG}"
HELIOS_RC=$?
set -e

# Locate outputs
TP=""
FLUX=""
for cand in \
  "${CASE_DIR}/stage4_coupled_n${LAYERS}/stage4_coupled_n${LAYERS}_tp.dat" \
  "${CASE_DIR}/stage4_coupled_n${LAYERS}_resolution/stage4_coupled_n${LAYERS}_resolution_tp.dat" \
  "${CASE_DIR}"/*/stage4_coupled_n${LAYERS}*_tp.dat \
  "${CASE_DIR}"/*_tp.dat
do
  if [[ -f "${cand}" && "${cand}" != *_seed* && "$(basename "${cand}")" != stage4_coupled_n${LAYERS}_tp.dat ]]; then
    # prefer HELIOS output dirs over the seed tp
    if [[ "${cand}" == *"/${CASE}/"* ]] || [[ "${cand}" == *"_resolution/"* ]] || [[ "$(dirname "${cand}")" != "${CASE_DIR}" ]]; then
      TP="${cand}"
      break
    fi
  fi
done
# Fallback: any non-seed tp under case subdirs
if [[ -z "${TP}" ]]; then
  TP="$(find "${CASE_DIR}" -name '*_tp.dat' ! -path "${CASE_DIR}/stage4_coupled_n${LAYERS}_tp.dat" | head -1 || true)"
fi
FLUX="$(find "${CASE_DIR}" -name '*_integrated_flux.dat' | head -1 || true)"

"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path
import sys
sys.path.insert(0, "src")
sys.path.insert(0, "stage4/experiments")
from convection_mlt.adapters.helios import load_tp_profile
from convection_mlt.adapters.helios_contracts import MICROBAR_TO_PA
import numpy as np

tp_path = Path("${TP}") if "${TP}" else None
flux_path = Path("${FLUX}") if "${FLUX}" else None
out = Path("${RESULT_JSON}")
attrib_path = Path("${ATTRIB_JSON}")
payload = {
    "purpose": "labelled_helios_n192_resolution_rcb",
    "n_layers": 192,
    "helios_process_returncode": int("${HELIOS_RC}"),
    "helios_tp": str(tp_path) if tp_path else None,
    "helios_flux": str(flux_path) if flux_path else None,
    "labelled_resolution_test": True,
    "helios_parity_headline": False,
    "full_stage4_claim": False,
}
if tp_path is None or not tp_path.is_file() or int("${HELIOS_RC}") != 0:
    payload["status"] = "FAIL"
    payload["rcb_log10p"] = None
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    raise SystemExit(4)

tp = load_tp_profile(tp_path)
lay = tp.layer_index != -1
p = np.asarray(tp.pressure_microbar[lay], dtype=np.float64) * MICROBAR_TO_PA
flag_u = np.asarray(tp.conv_unstable_flag[lay], dtype=np.float64)
flag_l = np.asarray(tp.conv_lapse_flag[lay], dtype=np.float64)
flag = flag_u if np.any(flag_u > 0.5) else flag_l
unstable = flag > 0.5
rcb = None
n_cz = 0
if unstable.size and bool(unstable[0]):
    i_hi = 0
    while i_hi + 1 < unstable.size and unstable[i_hi + 1]:
        i_hi += 1
    n_cz = i_hi + 1
    rcb = float(np.log10(float(p[i_hi])))
n96 = 4.7187499048239445
payload.update({
    "status": "COMPLETE",
    "rcb_log10p": rcb,
    "n_cz_layers": n_cz,
    "n96_rcb_log10p": n96,
    "rcb_dex_vs_n96": None if rcb is None else abs(rcb - n96),
    "toward_mlt_1p07bar": None if rcb is None else abs(rcb - 5.028032313236911) < abs(n96 - 5.028032313236911),
})
out.write_text(json.dumps(payload, indent=2) + "\n")

if attrib_path.exists():
    attrib = json.loads(attrib_path.read_text())
    attrib.setdefault("resolution", {})["n192"] = {
        "rcb_log10p": rcb,
        "status": "COMPLETE",
        "source": str(tp_path),
        "rcb_dex_vs_n96": payload["rcb_dex_vs_n96"],
        "note": "Labelled HELIOS geometric-grid resolution test; not headline benchmark.",
    }
    from attribute_coupled_rcb_discrepancy import build_attribution_table
    attrib["attribution_table"] = build_attribution_table(attrib)
    attrib_path.write_text(json.dumps(attrib, indent=2) + "\n")

print(json.dumps(payload, indent=2))
PY

echo "N=192 resolution RCB -> ${RESULT_JSON}"
