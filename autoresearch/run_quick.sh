#!/usr/bin/env bash
# Tier 2 — Modal H100 on the quick suite. ~$0.30. The dev loop: cheap enough
# to run dozens of times per night, gives the first real geomean signal.
# Decide keep/discard candidates here before paying for `full`.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${HERE}/_run_eval.sh" quick quick modal "$@"
