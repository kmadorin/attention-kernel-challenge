#!/usr/bin/env bash
# _run_eval.sh — shared runner for the autoresearch tier scripts.
#
# Usage: _run_eval.sh <tier_label> <suite> <backend> [extra cli args...]
#
# Calls `python -m attention_kernel_challenge.cli eval-submission` with the
# given suite/backend, captures the JSON summary, and emits ONE
# tab-separated line on stdout that the loop driver can parse:
#
#   tier=<label> suite=<suite> backend=<backend> valid=<true|false> \
#   geo=<float|NA> worst=<float|NA> \
#   sliding=<float|NA> sw_global=<float|NA> sw_retrieval=<float|NA> \
#   speedup=<float|NA> wall_s=<float> reason=<short|none>
#
# The full JSON summary is written to autoresearch/run.log for debugging.
# Exit code: 0 if overall_valid=true, 1 otherwise (or on harness errors).

set -uo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 <tier_label> <suite> <backend> [extra args...]" >&2
  exit 2
fi

TIER="$1"; SUITE="$2"; BACKEND="$3"; shift 3

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AR_DIR="${REPO_ROOT}/autoresearch"
SUB_DIR="${AR_DIR}/submission"
LOG="${AR_DIR}/run.log"
BASELINE="${AR_DIR}/baseline_${SUITE}.json"

PY="${PY:-python}"
if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  PY="${REPO_ROOT}/.venv/bin/python"
fi

CLI_ARGS=(
  -m attention_kernel_challenge.cli eval-submission
  --submission-dir "${SUB_DIR}"
  --suite "${SUITE}"
  --backend "${BACKEND}"
  --emit-json
  --redact-case-details
)

# Local CPU runs need --device cpu (the default for local is cpu already, but
# we're explicit). Modal runs ignore --device.
if [[ "${BACKEND}" == "local" ]]; then
  CLI_ARGS+=(--device cpu)
fi

CLI_ARGS+=("$@")

START_NS=$(date +%s%N)
JSON="$(cd "${REPO_ROOT}" && "${PY}" "${CLI_ARGS[@]}" 2>"${LOG}.stderr")"
RC=$?
END_NS=$(date +%s%N)
WALL_S=$(awk -v a="${START_NS}" -v b="${END_NS}" 'BEGIN{printf "%.3f", (b-a)/1e9}')

# Persist the full JSON + stderr for debugging.
{
  echo "=== tier=${TIER} suite=${SUITE} backend=${BACKEND} wall_s=${WALL_S} rc=${RC} ==="
  if [[ -s "${LOG}.stderr" ]]; then
    echo "--- stderr ---"
    cat "${LOG}.stderr"
  fi
  echo "--- json ---"
  if [[ -n "${JSON}" ]]; then
    echo "${JSON}"
  else
    echo "(no json — see stderr above)"
  fi
} > "${LOG}"
rm -f "${LOG}.stderr"

if [[ -z "${JSON}" || ${RC} -ne 0 ]]; then
  REASON="harness_error_rc${RC}"
  printf 'tier=%s\tsuite=%s\tbackend=%s\tvalid=false\tgeo=NA\tworst=NA\tsliding=NA\tsw_global=NA\tsw_retrieval=NA\tspeedup=NA\twall_s=%s\treason=%s\n' \
    "${TIER}" "${SUITE}" "${BACKEND}" "${WALL_S}" "${REASON}"
  exit 1
fi

# Parse the JSON summary in Python (no jq dependency).
JSON_TMP="$(mktemp)"
trap 'rm -f "${JSON_TMP}"' EXIT
printf '%s' "${JSON}" > "${JSON_TMP}"

PARSE_OUT="$(BASELINE_PATH="${BASELINE}" "${PY}" - "${TIER}" "${SUITE}" "${BACKEND}" "${WALL_S}" "${JSON_TMP}" <<'PYEOF'
import json, os, sys, statistics

tier, suite, backend, wall_s, json_path = sys.argv[1:6]
with open(json_path) as f:
    data = json.load(f)

valid = bool(data.get("overall_valid"))
geo = data.get("geometric_mean_family_latency_ms")
worst = data.get("worst_family_latency_ms")
reason = data.get("failure_reason") or "none"
# Compress reason to a single short token (no tabs/newlines).
reason = " ".join(str(reason).split())[:80] or "none"

per_family = {}
for case in data.get("case_results", []) or []:
    fam = case.get("family")
    lat = case.get("latency_ms")
    if fam and lat is not None:
        per_family.setdefault(fam, []).append(float(lat))
fam_med = {f: statistics.median(v) for f, v in per_family.items()}

def fmt(x):
    return "NA" if x is None else f"{float(x):.4f}"

speedup = "NA"
bp = os.environ.get("BASELINE_PATH", "")
if bp and os.path.exists(bp) and geo:
    try:
        with open(bp) as f:
            ref = json.load(f)
        ref_geo = ref.get("geometric_mean_family_latency_ms")
        if ref_geo:
            speedup = f"{float(ref_geo) / float(geo):.4f}"
    except Exception:
        pass

print("\t".join([
    f"tier={tier}",
    f"suite={suite}",
    f"backend={backend}",
    f"valid={'true' if valid else 'false'}",
    f"geo={fmt(geo)}",
    f"worst={fmt(worst)}",
    f"sliding={fmt(fam_med.get('sliding_window'))}",
    f"sw_global={fmt(fam_med.get('sliding_window_global'))}",
    f"sw_retrieval={fmt(fam_med.get('sliding_window_retrieval'))}",
    f"speedup={speedup}",
    f"wall_s={wall_s}",
    f"reason={reason}",
]))
PYEOF
)"
PARSE_RC=$?

if [[ ${PARSE_RC} -ne 0 || -z "${PARSE_OUT}" ]]; then
  printf 'tier=%s\tsuite=%s\tbackend=%s\tvalid=false\tgeo=NA\tworst=NA\tsliding=NA\tsw_global=NA\tsw_retrieval=NA\tspeedup=NA\twall_s=%s\treason=parse_error\n' \
    "${TIER}" "${SUITE}" "${BACKEND}" "${WALL_S}"
  exit 1
fi

echo "${PARSE_OUT}"

# Exit code reflects validity so callers can `if ! run_quick.sh; then ...`.
if echo "${PARSE_OUT}" | grep -q $'\tvalid=true\t'; then
  exit 0
else
  exit 1
fi
