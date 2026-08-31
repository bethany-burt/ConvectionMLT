#!/usr/bin/env bash
# Labelled HELIOS N=96 counterfactual: convective adjustment + isothermal_layers=yes.
# Uses a dedicated HELIOS checkout with the minimal ISO1 conv_check patch.
# Isolates layer-source treatment vs the iso=no pilot (job 16015698).
# Does NOT claim Stage-4 headline; frozen 0.15-dex RCB gate remains failed.
set -euo pipefail

LAYERS=96
ROOT="${STAGE4_ROOT:-/project/ls-heng/Bethany.Burt/convection_mlt}"
HELIOS_STOCK="${HELIOS_STOCK:-/project/ls-heng/Bethany.Burt/HELIOS}"
HELIOS_ROOT="${HELIOS_ROOT:-/project/ls-heng/Bethany.Burt/HELIOS_iso1_counterfactual}"
VENV_DIR="${HELIOS_VENV:-/project/ls-heng/Bethany.Burt/venvs/stage4-helios-py312}"
PIN="b0800f9ea4366263241c13bb926e8ca68f266cc5"
OUT_ROOT="${HELIOS_ISO1_OUT:-/project/ls-heng/Bethany.Burt/helios_stage4_iso1_counterfactual}"
CASE="stage4_coupled_n${LAYERS}_iso_yes_adj"
CASE_DIR="${OUT_ROOT}/${CASE}"
OPACITY="${ROOT}/stage4/fixtures/helios/analytic_grey_nested.h5"
RESULT_JSON="${ROOT}/stage4/results/helios_coupled_n96_iso_yes_adj_rcb.json"
ISO_NO_RCB_LOG10P="${ISO_NO_RCB_LOG10P:-4.7187499048239445}"
MLT_RCB_LOG10P="${MLT_RCB_LOG10P:-5.028032313236911}"

if command -v module >/dev/null 2>&1; then
  module purge >/dev/null 2>&1 || true
  module load python/3.12-2024.10 >/dev/null 2>&1 || true
fi
if [[ -f "${VENV_DIR}/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_DIR}/bin/activate"
fi
PYTHON_BIN="${PYTHON_BIN:-python3}"

# Dedicated HELIOS tree: copy stock pin, apply write-precision + ISO1 patches.
if [[ ! -d "${HELIOS_ROOT}/.git" ]]; then
  echo "Creating HELIOS_iso1_counterfactual from ${HELIOS_STOCK}"
  rm -rf "${HELIOS_ROOT}"
  git clone --no-local "${HELIOS_STOCK}" "${HELIOS_ROOT}"
fi
cd "${HELIOS_ROOT}"
git fetch --all >/dev/null 2>&1 || true
git checkout -f "${PIN}"
git clean -fdx -e 'param*.dat' -e 'output' >/dev/null 2>&1 || true

mkdir -p "${CASE_DIR}" "${ROOT}/stage4/results"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
cd "${ROOT}"

"${PYTHON_BIN}" stage4/experiments/apply_helios_write_precision.py \
  --write-py "${HELIOS_ROOT}/source/write.py" \
  --require-patch-checksum

"${PYTHON_BIN}" stage4/experiments/apply_helios_iso1_conv_fix.py \
  --helios-root "${HELIOS_ROOT}" \
  --require-patch-checksum

"${PYTHON_BIN}" stage4/experiments/export_coupled_helios_case.py \
  --layers "${LAYERS}" \
  --case-dir "${CASE_DIR}" \
  --opacity "${OPACITY}" \
  --iso-yes-adj-counterfactual

# Absolute paths for HELIOS cwd
python3 - <<PY
from pathlib import Path
import re
param = Path("${CASE_DIR}/param.dat")
tps = list(Path("${CASE_DIR}").glob("*_tp.dat"))
tp = tps[0]
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
    elif re.match(r"^isothermal layers\s*=", ln):
        if "yes" not in ln.split("=")[1].split()[0]:
            raise SystemExit(f"expected isothermal layers=yes, got {ln!r}")
        lines.append(ln)
    else:
        lines.append(ln)
param.write_text("\n".join(lines) + "\n")
print("patched", param, "tp", tp)
PY

run_helios_param() {
  local param_path="$1"
  local log_path="$2"
  local stamp_name="iso1_n96_param.dat"
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

TP="$(find "${CASE_DIR}" -path "*/${CASE}/${CASE}_tp.dat" | head -1 || true)"
FLUX="$(find "${CASE_DIR}" -path "*/${CASE}/${CASE}_integrated_flux.dat" | head -1 || true)"
if [[ -z "${TP}" ]]; then
  TP="$(find "${CASE_DIR}" -name '*_tp.dat' ! -name "${CASE}_tp.dat" | head -1 || true)"
fi
if [[ -z "${FLUX}" ]]; then
  FLUX="$(find "${CASE_DIR}" -name '*_integrated_flux.dat' | head -1 || true)"
fi

"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, "src")
sys.path.insert(0, "stage4/experiments")
from convection_mlt.adapters.helios import load_tp_profile
from convection_mlt.adapters.helios_contracts import MICROBAR_TO_PA

tp_path = Path("${TP}") if "${TP}" else None
flux_path = Path("${FLUX}") if "${FLUX}" else None
out = Path("${RESULT_JSON}")
iso_no = float("${ISO_NO_RCB_LOG10P}")
mlt = float("${MLT_RCB_LOG10P}")
payload = {
    "purpose": "labelled_helios_n96_iso_yes_adj_source_counterfactual",
    "n_layers": 96,
    "isothermal_layers": True,
    "convective_adjustment": True,
    "iso1_patch": "helios_iso1_conv_check_b0800f9.patch",
    "helios_process_returncode": int("${HELIOS_RC}"),
    "helios_tp": str(tp_path) if tp_path else None,
    "helios_flux": str(flux_path) if flux_path else None,
    "helios_log": "${HELIOS_LOG}",
    "isolates": "layer_source_treatment_only",
    "held_constant": [
        "helios_geometric_grid",
        "convective_adjustment",
        "opacity",
        "forcing_F_int_300_F_irr_0",
        "n_layers_96",
    ],
    "compare_to_iso_no_pilot_rcb_log10p": iso_no,
    "mlt_reference_rcb_log10p": mlt,
    "helios_parity_headline": False,
    "full_stage4_claim": False,
    "frozen_rcb_gate_0p15_dex": "FAIL_UNCHANGED",
    "note": (
        "Counterfactual only. Quantifies RCB shift from iso=yes vs iso=no with "
        "adjustment held on. Does not relabel the coupled benchmark as PASS."
    ),
}
log_text = Path("${HELIOS_LOG}").read_text(errors="replace") if Path("${HELIOS_LOG}").exists() else ""
if "Traceback (most recent call last):" in log_text:
    payload["status"] = "HELIOS_CRASH"
    payload["rcb_log10p"] = None
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    raise SystemExit(4)
if tp_path is None or not tp_path.is_file() or int("${HELIOS_RC}") != 0:
    payload["status"] = "FAIL"
    payload["rcb_log10p"] = None
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    raise SystemExit(4)

tp = load_tp_profile(tp_path)
lay = tp.layer_index != -1
p = np.asarray(tp.pressure_microbar[lay], dtype=np.float64) * MICROBAR_TO_PA
T = np.asarray(tp.temperature_k[lay], dtype=np.float64)
flag_u = np.asarray(tp.conv_unstable_flag[lay], dtype=np.float64)
flag_l = np.asarray(tp.conv_lapse_flag[lay], dtype=np.float64)
# HELIOS layer 0 = deep
if p.size and p[0] < p[-1]:
    p = p[::-1]
    T = T[::-1]
    flag_u = flag_u[::-1]
    flag_l = flag_l[::-1]

def _bottom_cz(mask):
    rcb = None
    n_cz = 0
    if mask.size and bool(mask[0]):
        i_hi = 0
        while i_hi + 1 < mask.size and mask[i_hi + 1]:
            i_hi += 1
        n_cz = i_hi + 1
        rcb = float(np.log10(float(p[i_hi])))
    return rcb, n_cz

scoring_method = "lapse_flags"
with np.errstate(all="ignore"):
    use_u = bool(np.nanmax(flag_u) > 0.5) if flag_u.size else False
    use_l = bool(np.nanmax(flag_l) > 0.5) if flag_l.size else False
if use_u or use_l:
    flag = flag_u if use_u else flag_l
    rcb, n_cz = _bottom_cz(flag > 0.5)
else:
    # Stock write blanks flags under iso=yes; fall back to T(P) near-adiabat.
    scoring_method = "near_adiabat_T_P"
    nabla_ad = 2.0 / 7.0
    dlnT = np.diff(np.log(T))
    dlnP = np.diff(np.log(p))
    nabla = dlnT / dlnP
    on_ad = nabla >= (nabla_ad - 1.0e-3)
    # Map interface i (between layer i and i+1) onto deeper layer i for CZ count.
    layer_on = np.zeros(T.size, dtype=bool)
    if on_ad.size:
        layer_on[: on_ad.size] = on_ad
        if on_ad[-1]:
            layer_on[-1] = True
    rcb, n_cz = _bottom_cz(layer_on)

payload.update({
    "status": "COMPLETE",
    "scoring_method": scoring_method,
    "rcb_log10p": rcb,
    "n_cz_layers": n_cz,
    "rcb_dex_vs_iso_no_pilot": None if rcb is None else abs(rcb - iso_no),
    "rcb_dex_vs_mlt": None if rcb is None else abs(rcb - mlt),
    "source_treatment_rcb_shift_dex": None if rcb is None else abs(rcb - iso_no),
    "source_treatment_structural": None if rcb is not None else {
        "iso_no_adj_rcb": iso_no,
        "iso_yes_adj_rcb": None,
        "note": "No bottom-connected CZ under iso=yes+adj; not a scalar ΔRCB.",
    },
    "T_deep_K": float(T[0]) if T.size else None,
})
out.write_text(json.dumps(payload, indent=2) + "\n")

# Merge into attribution JSON if present.
attrib_path = Path("${ROOT}/stage4/results/helios_coupled_n96_rcb_attribution.json")
if attrib_path.exists():
    attrib = json.loads(attrib_path.read_text())
    shift = {
        "status": "COMPLETE",
        "iso_yes_adj_rcb_log10p": rcb,
        "iso_no_pilot_rcb_log10p": iso_no,
        "iso_yes_bottom_connected_cz": bool(n_cz > 0),
        "rcb_dex": None if rcb is None else abs(rcb - iso_no),
        "scoring_method": scoring_method,
        "result_json": str(out),
        "note": (
            "Same HELIOS grid/adjustment/opacity/forcing/N=96; only iso yes↔no. "
            "Frozen 0.15-dex gate remains FAIL."
        ),
    }
    attrib["source_treatment_counterfactual"] = shift
    if "radiation_source" in attrib:
        attrib["radiation_source"]["radiation_source_rcb_shift"] = shift
    attrib["helios_parity_headline"] = False
    attrib["full_stage4_claim"] = False
    table = attrib.get("attribution_table") or {}
    rows = list(table.get("rows") or [])
    rows = [r for r in rows if r.get("test") != "HELIOS iso=yes+adj vs iso=no+adj"]
    rows.append({
        "test": "HELIOS iso=yes+adj vs iso=no+adj",
        "isolates": "Layer-source treatment (coupled RCB)",
        "metric": "bottom-connected CZ / RCB under N=96 adjustment",
        "value": {
            "iso_yes_adj_rcb": rcb,
            "iso_no_adj_rcb": iso_no,
            "iso_yes_bottom_connected_cz": bool(n_cz > 0),
            "rcb_dex": None if rcb is None else abs(rcb - iso_no),
            "frozen_0p15_gate": "FAIL_UNCHANGED",
        },
        "status": "COMPLETE",
    })
    table["rows"] = rows
    if rcb is None:
        table["defensible_conclusion"] = (
            "HELIOS non-isothermal source treatment changes the coupled N=96 "
            "adjusted RCE from a bottom-connected CZ (RCB≈4.719 under iso=no) to "
            "no bottom-connected CZ under iso=yes, with grid/adjustment/opacity/"
            "forcing held fixed. The pilot gap vs MLT is a combined source-treatment "
            "and closure discrepancy—not solely iso=no. The frozen 0.15-dex RCB "
            "gate remains failed; explained disagreement is not a numerical PASS."
        )
    else:
        table["defensible_conclusion"] = (
            "HELIOS non-isothermal source treatment shifts the coupled RCB by "
            f"{abs(rcb - iso_no):.3f} dex at N=96 with adjustment held fixed. "
            "The frozen 0.15-dex RCB gate remains failed; this is an explained "
            "benchmark disagreement, not a numerical PASS."
        )
    attrib["attribution_table"] = table
    attrib_path.write_text(json.dumps(attrib, indent=2) + "\n")

print(json.dumps(payload, indent=2))
PY

echo "ISO1 counterfactual complete -> ${RESULT_JSON}"
