#!/usr/bin/env bash
# Tier 1 — Modal H100 on the smoke suite. ~$0.10. First tier that exercises
# the actual Triton path, sandbox, compile-cache freeze, and serverlike
# isolation. Use it as the cheap "did this even build" check after every
# nontrivial edit.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${HERE}/_run_eval.sh" smoke smoke modal "$@"
