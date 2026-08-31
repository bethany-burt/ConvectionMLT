#!/usr/bin/env bash
# N=96 coupled HELIOS RCE pilot: frozen-T iso=no preflight, then stock iterative.
# isothermal_layers=false (required for HELIOS convective adjustment at b0800f9).
# Matched F_irr=0 nested τ-grid MLT reference. Does not claim full Stage 4.
# Do NOT enlarge the Planck table if T approaches the ceiling — diagnose runaway.
set -euo pipefail

LAYERS="${1:-96}"
ROOT="${STAGE4_ROOT:-/project/ls-heng/Bethany.Burt/convection_mlt}"
HELIOS_ROOT="${HELIOS_ROOT:-/project/ls-heng/Bethany.Burt/HELIOS}"
VENV_DIR="${HELIOS_VENV:-/project/ls-heng/Bethany.Burt/venvs/stage4-helios-py312}"
PIN="b0800f9ea4366263241c13bb926e8ca68f266cc5"
OUT_ROOT="${HELIOS_COUPLED_OUT:-/project/ls-heng/Bethany.Burt/helios_stage4_coupled}"
CASE="stage4_coupled_n${LAYERS}"
CASE_DIR="${OUT_ROOT}/${CASE}"
FROZEN_DIAG_ROOT="${OUT_ROOT}/${CASE}_frozen_iso_diag"
OPACITY="${ROOT}/stage4/fixtures/helios/analytic_grey_nested.h5"
TOL="${ROOT}/stage4/fixtures/helios/coupled_rce_benchmark_tolerances.json"
SMOKE_JSON="${ROOT}/stage4/fixtures/helios/helios_contract_smoke_n8.cluster.json"
RESULT_JSON="${ROOT}/stage4/results/helios_coupled_rce_n${LAYERS}.json"
FROZEN_DIAG_JSON="${ROOT}/stage4/results/helios_coupled_frozen_iso_diag_n${LAYERS}.json"

if command -v module >/dev/null 2>&1; then
  module purge >/dev/null 2>&1 || true
  module load python/3.12-2024.10 >/dev/null 2>&1 || true
fi
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ ! -f "${TOL}" ]]; then
  echo "missing frozen coupled tolerances ${TOL}" >&2
  exit 2
fi
frozen="$("${PYTHON_BIN}" - <<PY
import json
print(json.loads(open("${TOL}").read()).get("frozen_before_live", False))
PY
)"
if [[ "${frozen}" != "True" ]]; then
  echo "Refusing coupled HELIOS: tolerances not frozen_before_live" >&2
  exit 2
fi

if [[ -f "${SMOKE_JSON}" ]]; then
  smoke_status="$("${PYTHON_BIN}" - <<PY
import json
print(json.loads(open("${SMOKE_JSON}").read()).get("status", "NOT_RUN"))
PY
)"
  if [[ "${smoke_status}" != "PASS" ]]; then
    echo "Refusing N${LAYERS} coupled: helios_contract_smoke_n8 status=${smoke_status}" >&2
    exit 2
  fi
fi

mkdir -p "${CASE_DIR}" "${FROZEN_DIAG_ROOT}" "${ROOT}/stage4/results"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
cd "${ROOT}"

"${PYTHON_BIN}" stage4/experiments/verify_frozen_inputs.py \
  --manifest "${ROOT}/stage4/fixtures/helios/coupled_input_manifest.json"

"${PYTHON_BIN}" stage4/experiments/export_coupled_helios_case.py \
  --layers "${LAYERS}" \
  --case-dir "${CASE_DIR}" \
  --opacity "${OPACITY}" \
  --frozen-iso-diag-dir "${FROZEN_DIAG_ROOT}"

# Preflight exported iterative param/TP before any HELIOS launch.
python - <<PY
import re, sys
from pathlib import Path
param = Path("${CASE_DIR}/param.dat").read_text()
tp = Path("${CASE_DIR}/${CASE}_tp.dat")
if not tp.exists():
    print("missing exported tp.dat", file=sys.stderr)
    raise SystemExit(2)
n_tp = 0
for ln in tp.read_text().splitlines():
    fields = ln.split()
    if not fields or fields[0] in ("BOA",) or not fields[0][0].isdigit():
        continue
    n_tp += 1
if n_tp != int("${LAYERS}"):
    print(f"tp layer count {n_tp} != ${LAYERS}", file=sys.stderr)
    raise SystemExit(2)
for key, want in {
    "direct irradiation beam": "no",
    "convective adjustment": "yes",
    "run type": "iterative",
    "physical timestep [s]": "no",
    "isothermal layers": "no",
}.items():
    m = re.search(rf"^{re.escape(key)}\s*=\s*(\S+)", param, re.M)
    got = m.group(1) if m else ""
    if got != want:
        print(f"exported param mismatch: {key} got={got!r} want={want!r}", file=sys.stderr)
        raise SystemExit(2)
m = re.search(r"^number of layers\s*=\s*(\S+)", param, re.M)
if not m or m.group(1) != "${LAYERS}":
    print(f"number of layers mismatch: {None if not m else m.group(1)}", file=sys.stderr)
    raise SystemExit(2)
m = re.search(r"^maximum number of iterations\s*=\s*(\S+)", param, re.M)
if not m or int(float(m.group(1))) != 50000:
    print(f"maximum iterations mismatch: {None if not m else m.group(1)} (want 50000)", file=sys.stderr)
    raise SystemExit(2)
m = re.search(r"^relax radiative criterion at\s*=\s*(\S+)\s+(\S+)", param, re.M)
if not m or abs(float(m.group(1)) - 1.0e4) > 1.0 or abs(float(m.group(2)) - 2.0e4) > 1.0:
    print(f"criterion relaxation mismatch: {None if not m else m.groups()}", file=sys.stderr)
    raise SystemExit(2)
m = re.search(r"^plancktable dimension and stepsize\s*=\s*(\S+)\s+(\S+)", param, re.M)
if not m or int(float(m.group(1))) != 8000 or int(float(m.group(2))) != 2:
    print(f"Planck table must stay 8000 2; got {None if not m else m.groups()}", file=sys.stderr)
    raise SystemExit(2)
m = re.search(r"^diffusivity factor\s*=\s*(\S+)", param, re.M)
if not m or abs(float(m.group(1)) - 2.0) > 1e-12:
    print(f"diffusivity factor mismatch: {None if not m else m.group(1)}", file=sys.stderr)
    raise SystemExit(2)
m = re.search(r"^internal temperature \[K\]\s*=\s*(\S+)", param, re.M)
tint = float(m.group(1)) if m else float("nan")
sigma = 5.670374419e-8
f_int = sigma * tint**4
if abs(f_int - 300.0) / 300.0 > 1e-9:
    print(f"F_int from T_int mismatch: T={tint} F={f_int}", file=sys.stderr)
    raise SystemExit(2)
m = re.search(r"^kappa value\s*=\s*(\S+)", param, re.M)
if not m or abs(float(m.group(1)) - (2.0 / 7.0)) > 1e-15:
    print(f"nabla_ad/kappa mismatch: {None if not m else m.group(1)}", file=sys.stderr)
    raise SystemExit(2)
if "analytic_grey_nested.h5" not in param:
    print("exported param does not reference analytic_grey_nested.h5", file=sys.stderr)
    raise SystemExit(2)
rt = __import__("json").loads(Path("${CASE_DIR}/helios_runtime_config.json").read_text())
if rt.get("isothermal_layers") is not False:
    print(f"runtime isothermal_layers must be false; got {rt.get('isothermal_layers')!r}", file=sys.stderr)
    raise SystemExit(2)
print("exported case preflight PASS", {"n_layers": n_tp, "T_int": tint, "F_int": f_int, "iso": "no", "max_iters": 50000})
PY

cd "${HELIOS_ROOT}"
git fetch --all --tags
git checkout "${PIN}"
git checkout "${PIN}" -- source/write.py
"${PYTHON_BIN}" "${ROOT}/stage4/experiments/apply_helios_write_precision.py" \
  --write-py "${HELIOS_ROOT}/source/write.py" \
  --require-patch-checksum

if [[ ! -d "${VENV_DIR}" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
python -c "import pycuda.autoinit" 2>/dev/null || {
  if command -v module >/dev/null 2>&1; then
    module load spack/2024.07 >/dev/null 2>&1 || true
  fi
  spack_setup="/software/opt/el_9/x86_64/spack/2024.07/spack/share/spack/setup-env.sh"
  if [[ -f "${spack_setup}" ]]; then
    # shellcheck disable=SC1090
    . "${spack_setup}"
    eval "$(spack load --sh cuda@12.4.0)"
  fi
}

run_helios_param() {
  local param_src="$1"
  local log_path="$2"
  local param_name
  param_name="$(basename "${param_src}")"
  cp "${param_src}" "${HELIOS_ROOT}/${param_name}"
  set +e
  python "${HELIOS_ROOT}/helios.py" -parameter_file "${param_name}" 2>&1 | tee "${log_path}"
  local rc=${PIPESTATUS[0]}
  set -e
  return "${rc}"
}

# --- Frozen-T iso=no radiation preflight (plus iso=yes baseline on same T) ---
cd "${HELIOS_ROOT}"
FROZEN_NO_DIR="${FROZEN_DIAG_ROOT}/iso_no"
FROZEN_YES_DIR="${FROZEN_DIAG_ROOT}/iso_yes"
FROZEN_NO_CASE="stage4_coupled_n${LAYERS}_frozen_iso_no"
FROZEN_YES_CASE="stage4_coupled_n${LAYERS}_frozen_iso_yes"

set +e
run_helios_param "${FROZEN_NO_DIR}/param.dat" "${FROZEN_NO_DIR}/helios_stdout.log"
FROZEN_NO_RC=$?
run_helios_param "${FROZEN_YES_DIR}/param.dat" "${FROZEN_YES_DIR}/helios_stdout.log"
FROZEN_YES_RC=$?
set -e

FROZEN_NO_FLUX="${FROZEN_NO_DIR}/${FROZEN_NO_CASE}/${FROZEN_NO_CASE}_integrated_flux.dat"
FROZEN_YES_FLUX="${FROZEN_YES_DIR}/${FROZEN_YES_CASE}/${FROZEN_YES_CASE}_integrated_flux.dat"
if [[ ! -f "${FROZEN_NO_FLUX}" ]]; then
  FROZEN_NO_FLUX="${FROZEN_NO_DIR}/${FROZEN_NO_CASE}_integrated_flux.dat"
fi
if [[ ! -f "${FROZEN_YES_FLUX}" ]]; then
  FROZEN_YES_FLUX="${FROZEN_YES_DIR}/${FROZEN_YES_CASE}_integrated_flux.dat"
fi

cd "${ROOT}"
python - <<PY
import json, sys
from pathlib import Path
sys.path.insert(0, "stage4/experiments")
sys.path.insert(0, "src")
from compare_coupled_helios_rce import find_helios_abort, helios_abort_payload
from diagnose_coupled_frozen_iso_no import analyze_frozen_iso_diag

flux_no = Path("${FROZEN_NO_FLUX}")
flux_yes = Path("${FROZEN_YES_FLUX}")
log_no = Path("${FROZEN_NO_DIR}/helios_stdout.log")
out = Path("${FROZEN_DIAG_JSON}")
abort = find_helios_abort(Path("${FROZEN_NO_DIR}"), "${FROZEN_NO_CASE}")
traceback = log_no.exists() and "Traceback (most recent call last):" in log_no.read_text(errors="replace")
if abort is not None or traceback or not flux_no.is_file() or int("${FROZEN_NO_RC}") != 0:
    payload = helios_abort_payload(
        n_layers=int("${LAYERS}"),
        abort_path=abort,
        helios_tp=None,
        helios_flux=flux_no if flux_no.is_file() else None,
        helios_log=log_no if log_no.exists() else None,
    )
    payload["status"] = "HELIOS_CRASH" if traceback else payload["status"]
    payload["execution_status"] = payload["status"]
    payload["failure_stage"] = "frozen_iso_no_preflight"
    payload["purpose"] = "frozen_T_iso_no_radiation_preflight"
    payload["helios_process_returncode"] = int("${FROZEN_NO_RC}")
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"frozen_preflight": "FAIL", "out": str(out)}, indent=2))
    raise SystemExit(4)
payload = analyze_frozen_iso_diag(
    n_layers=int("${LAYERS}"),
    flux_iso_no=flux_no,
    flux_iso_yes=flux_yes if flux_yes.is_file() else None,
)
payload["helios_process_returncode_iso_no"] = int("${FROZEN_NO_RC}")
payload["helios_process_returncode_iso_yes"] = int("${FROZEN_YES_RC}")
out.write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps({"frozen_preflight": payload["status"], "out": str(out), "delta": payload.get("iso_yes_vs_iso_no")}, indent=2))
if payload["status"] != "PASS":
    raise SystemExit(4)
PY

# --- Stock iterative coupled HELIOS (iso=no, maxiters=50000) ---
cd "${HELIOS_ROOT}"
HELIOS_LOG="${CASE_DIR}/helios_stdout.log"
set +e
run_helios_param "${CASE_DIR}/param.dat" "${HELIOS_LOG}"
HELIOS_RC=$?
set -e

cd "${ROOT}"
# Abort / crash / missing outputs / Planck runaway → no physical score.
python - <<PY
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "stage4/experiments")
sys.path.insert(0, "src")
from compare_coupled_helios_rce import (
    find_helios_abort,
    helios_abort_payload,
    score_structural_irradiated,
    load_mlt_reference,
)

case_dir = Path("${CASE_DIR}")
case_name = "${CASE}"
log = Path("${HELIOS_LOG}")
out = Path("${RESULT_JSON}")
abort = find_helios_abort(case_dir, case_name)
helios_tp = case_dir / case_name / f"{case_name}_tp.dat"
helios_flux = case_dir / case_name / f"{case_name}_integrated_flux.dat"
missing_outputs = (not helios_tp.is_file()) or (not helios_flux.is_file())
log_text = log.read_text(errors="replace") if log.exists() else ""
traceback = "Traceback (most recent call last):" in log_text
planck_jump = (
    "plancktable" in log_text.lower()
    or "exceeds planck" in log_text.lower()
    or re.search(r"surface reaches too high", log_text, re.I) is not None
)
# HELIOS silently jumps to convection when BOA T hits the table ceiling; detect via log pattern used in source.
# Also treat explicit abort / missing outputs / nonzero crash as unscored.
if abort is not None or missing_outputs or traceback or int("${HELIOS_RC}") not in (0,):
    runtime = {}
    rt = case_dir / "helios_runtime_config.json"
    if rt.exists():
        runtime = json.loads(rt.read_text())
    try:
        structural = score_structural_irradiated(int("${LAYERS}"), load_mlt_reference(int("${LAYERS}")))
    except Exception as exc:  # noqa: BLE001
        structural = {"status": "STRUCTURAL_NOT_SCORED", "note": f"optional structural skipped: {exc}"}
    payload = helios_abort_payload(
        n_layers=int("${LAYERS}"),
        abort_path=abort,
        helios_tp=helios_tp if helios_tp.is_file() else None,
        helios_flux=helios_flux if helios_flux.is_file() else None,
        helios_log=log if log.exists() else None,
        runtime=runtime,
        structural=structural,
    )
    payload["helios_process_returncode"] = int("${HELIOS_RC}")
    payload["initialization_mode"] = "stock_helios_isothermal_500K_iso_layers_false"
    payload["isothermal_layers"] = False
    if planck_jump and not traceback:
        payload["failure_stage"] = "numerical_runaway_planck_ceiling"
        payload["note"] = (
            "Iteration approached the pinned Planck-table ceiling (~15998 K). "
            "Do not enlarge the table; diagnose stability/damping. "
            "Physical reference T is <700 K. " + payload.get("note", "")
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "execution_status": payload["execution_status"],
        "helios_coupled_rce_n96_status": payload.get("helios_coupled_rce_n96_status"),
        "failure_stage": payload["failure_stage"],
        "helios_process_returncode": payload["helios_process_returncode"],
        "out": str(out),
        "physical_scorer": "SKIPPED",
    }, indent=2))
    raise SystemExit(3)

print("HELIOS outputs present; invoking physical coupled scorer")
PY

python stage4/experiments/compare_coupled_helios_rce.py \
  --layers "${LAYERS}" \
  --helios-tp "${CASE_DIR}/${CASE}/${CASE}_tp.dat" \
  --helios-flux "${CASE_DIR}/${CASE}/${CASE}_integrated_flux.dat" \
  --helios-log "${HELIOS_LOG}" \
  --case-dir "${CASE_DIR}" \
  --case-name "${CASE}" \
  --runtime-config "${CASE_DIR}/helios_runtime_config.json" \
  --tolerances "${TOL}" \
  --output "${RESULT_JSON}"

# Scorer may still return NOT_RUN/INFRASTRUCTURE if not converged — treat non-PASS as unscored exit.
python - <<PY
import json, sys
from pathlib import Path
p = json.loads(Path("${RESULT_JSON}").read_text())
st = p.get("status")
print(json.dumps({"scored_status": st, "helios_coupled_rce_n96_status": p.get("helios_coupled_rce_n96_status")}, indent=2))
if st != "PASS":
    # FAIL is a real scored physical disagreement; still a completed benchmark attempt.
    if st in ("HELIOS_ABORT", "HELIOS_CRASH", "NOT_RUN", "INFRASTRUCTURE_FAIL"):
        raise SystemExit(3)
print("Coupled HELIOS N=${LAYERS} pilot completed (independently discretized RCE; not Stage-4 headline)")
PY
