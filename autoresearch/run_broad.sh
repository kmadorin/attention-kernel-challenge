#!/usr/bin/env bash
# Tier 4 — Modal H100 on the broad suite. ~$5–10. The held-out overfitting
# check. Run sparingly: at most once per ~10 kept commits, only on milestone
# candidates that already won on `full`. If broad_geo regresses by >5% vs the
# previous broad milestone, STOP — the kernel is overfitting to the
# quick/full case distributions.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${HERE}/_run_eval.sh" broad broad modal "$@"
