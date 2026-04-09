#!/usr/bin/env bash
# Tier 3 — Modal H100 on the full suite. ~$1–2. The decision point. Only run
# if tier 2 (quick) shows improvement. The full_geo here is the
# leaderboard-relevant number; speedup vs baseline_full.json is the headline.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${HERE}/_run_eval.sh" full full modal "$@"
