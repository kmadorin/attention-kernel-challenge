#!/usr/bin/env bash
# Tier 0 — local CPU correctness on the smoke suite. Free, runs in seconds.
# Validates the torch fallback path against the FP32 reference. The Triton
# path is NOT exercised here (no GPU on the inner-loop box).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${HERE}/_run_eval.sh" check smoke local "$@"
