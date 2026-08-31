#!/usr/bin/env bash
# Cheap HELIOS post-processing for RCB attribution: iso=yes and iso=no on the
# final coupled N=96 T(P). Not iterative coupled; not Stage-4 headline.
set -euo pipefail

LAYERS="${LAYERS:-96}"
ROOT="${STAGE4_ROOT:-/project/ls-heng/Bethany.Burt/convection_mlt}"
HELIOS_ROOT="${HELIOS_ROOT:-/project/ls-heng/Bethany.Burt/HELIOS}"
VENV_DIR="${HELIOS_VENV:-/project/ls-heng/Bethany.Burt/venvs/stage4-helios-py312}"
PIN="b0800f9ea4366263241c13bb926e8ca68f266cc5"
OUT_ROOT="${HELIOS_ATTRIB_OUT:-/project/ls-heng/Bethany.Burt/helios_stage4_rcb_attrib}"
ATTRIB_ROOT="${ATTRIB_ROOT:-${OUT_ROOT}/n${LAYERS}_final_tp_postproc}"
TP_SRC="${TP_SRC:-${ROOT}/stage4/results/helios_coupled_n96_job16015698_debug/iterative/final_tp.dat}"
FLUX_ITER="${FLUX_ITER:-${ROOT}/stage4/results/helios_coupled_n96_job16015698_debug/iterative/final_integrated_flux.dat}"
OPACITY="${ROOT}/stage4/fixtures/helios/analytic_grey_nested.h5"
OUT_JSON="${OUT_JSON:-${ROOT}/stage4/results/helios_coupled_n96_rcb_attribution.json}"

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

mkdir -p "${ATTRIB_ROOT}" "${ROOT}/stage4/results"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
cd "${ROOT}"

if [[ ! -f "${TP_SRC}" ]]; then
  echo "missing final HELIOS TP ${TP_SRC}" >&2
  exit 2
fi

"${PYTHON_BIN}" stage4/experiments/attribute_coupled_rcb_discrepancy.py \
  --tp "${TP_SRC}" \
  --flux "${FLUX_ITER}" \
  --layers "${LAYERS}" \
  --export-postproc-dir "${ATTRIB_ROOT}" \
  --output "${OUT_JSON}.partial_local.json"

# Rewrite opacity / TP / output paths to absolute cluster locations.
for tag in iso_no iso_yes; do
  param="${ATTRIB_ROOT}/${tag}/param.dat"
  tp="${ATTRIB_ROOT}/${tag}/stage4_rcb_attrib_n${LAYERS}_${tag}_tp.dat"
  python3 - <<PY
from pathlib import Path
import re
param = Path("${param}")
text = param.read_text()
repl = {
    r"^output directory\s*=": "output directory =                                    ${ATTRIB_ROOT}/${tag}/",
    r"^path to temperature file\s*=": "path to temperature file =                           ${tp}",
    r"^  premixed   --> path to opacity file\s*=": "  premixed   --> path to opacity file =               ${OPACITY}",
}
lines = []
for ln in text.splitlines():
    done = False
    for pat, new in repl.items():
        if re.match(pat, ln):
            # keep trailing comment column if present
            m = re.search(r"(\s{2,}\[.*)$", ln)
            comment = m.group(1) if m else ""
            lines.append(new + comment)
            done = True
            break
    if not done:
        lines.append(ln)
param.write_text("\n".join(lines) + "\n")
print("patched", param)
PY
done

run_helios_param() {
  local param_path="$1"
  local log_path="$2"
  local param_name
  param_name="$(basename "${param_path}")"
  # Unique name so iso_yes/iso_no do not clobber each other in HELIOS_ROOT.
  local stamp_name
  stamp_name="$(basename "$(dirname "${param_path}")")_${param_name}"
  cp -f "${param_path}" "${HELIOS_ROOT}/${stamp_name}"
  cd "${HELIOS_ROOT}"
  set +e
  python helios.py -parameter_file "${stamp_name}" 2>&1 | tee "${log_path}"
  local rc=${PIPESTATUS[0]}
  set -e
  cd "${ROOT}"
  return "${rc}"
}

find_flux() {
  local case_dir="$1"
  local case_name="$2"
  local cand
  for cand in \
    "${case_dir}/${case_name}/${case_name}_integrated_flux.dat" \
    "${case_dir}/${case_name}_integrated_flux.dat" \
    "${case_dir}"/*_integrated_flux.dat
  do
    if [[ -f "${cand}" ]]; then
      echo "${cand}"
      return 0
    fi
  done
  return 1
}

YES_DIR="${ATTRIB_ROOT}/iso_yes"
NO_DIR="${ATTRIB_ROOT}/iso_no"
YES_NAME="stage4_rcb_attrib_n${LAYERS}_iso_yes"
NO_NAME="stage4_rcb_attrib_n${LAYERS}_iso_no"

set +e
run_helios_param "${NO_DIR}/param.dat" "${NO_DIR}/helios_stdout.log"
RC_NO=$?
run_helios_param "${YES_DIR}/param.dat" "${YES_DIR}/helios_stdout.log"
RC_YES=$?
set -e

FLUX_NO="$(find_flux "${NO_DIR}" "${NO_NAME}" || true)"
FLUX_YES="$(find_flux "${YES_DIR}" "${YES_NAME}" || true)"

"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path
import sys
sys.path.insert(0, "stage4/experiments")
sys.path.insert(0, "src")
from attribute_coupled_rcb_discrepancy import run_local

payload = run_local(
    tp_path=Path("${TP_SRC}"),
    flux_path=Path("${FLUX_ITER}"),
    iso_yes_flux=Path("${FLUX_YES}") if "${FLUX_YES}" else None,
    iso_no_flux=Path("${FLUX_NO}") if "${FLUX_NO}" else None,
    n_layers=int("${LAYERS}"),
    out_path=Path("${OUT_JSON}"),
)
summary = {
    "out": "${OUT_JSON}",
    "helios_rc_iso_no": int("${RC_NO}"),
    "helios_rc_iso_yes": int("${RC_YES}"),
    "flux_iso_no": "${FLUX_NO}",
    "flux_iso_yes": "${FLUX_YES}",
    "stage3_approx_iso_yes": payload["radiation_source"]["stage3_approx_helios_iso_yes"],
    "rcb_from_radiation": {
        k: {
            "field_F": (v.get("using_field_F_rad") or {}).get("rcb_log10p"),
            "F_int": (v.get("using_F_int") or {}).get("rcb_log10p"),
        }
        for k, v in payload["rcb_from_radiation"].items()
    },
    "convection_closure_dex": payload["convection_closure"]["rcb_dex_mlt_vs_exact_adj"],
    "attribution_rows": payload["attribution_table"]["rows"],
}
print(json.dumps(summary, indent=2))
meta = {
    "helios_process_returncode_iso_no": int("${RC_NO}"),
    "helios_process_returncode_iso_yes": int("${RC_YES}"),
    "flux_iso_no": "${FLUX_NO}",
    "flux_iso_yes": "${FLUX_YES}",
}
payload["helios_postproc_meta"] = meta
Path("${OUT_JSON}").write_text(json.dumps(payload, indent=2) + "\n")
if int("${RC_NO}") != 0 or int("${RC_YES}") != 0:
    raise SystemExit(4)
if not ("${FLUX_NO}" and Path("${FLUX_NO}").is_file() and "${FLUX_YES}" and Path("${FLUX_YES}").is_file()):
    raise SystemExit(5)
PY

echo "RCB attribution post-proc complete -> ${OUT_JSON}"
