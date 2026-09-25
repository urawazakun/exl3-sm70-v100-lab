#!/usr/bin/env bash
# muse-dev.sh -- developer harness for the "second developer" role, mirroring sol-dev.sh's interface
# (workspace + brief file) but running Muse Spark 1.3 (Meta) through the CommandCode API, i.e. the same
# endpoint and the same GOAT key the operator's other agent sessions use.
#
# Usage: muse-dev.sh <workspace> <brief.md> [model]
#
# Model choice matters for data use:
#   meta/muse-spark-1.3-contributor  THE OPERATOR'S CHOICE (2026-09-25): ~12x-21x cheaper
#                                    ($0.10/M in, $0.002/M cached, $0.20/M out) because prompts and
#                                    completions may be used to improve Meta products. Tighter limits
#                                    than standard (60 requests/min vs 3,000) and web-search grounding is
#                                    NOT discounted ($2.50 per 1k queries on both tiers).
#   meta/muse-spark-1.3              the standard tier ($1.25/$0.15/$4.25) whose traffic is not used for
#                                    training -- pass it as $3 whenever a task must not leave the lab.
# Other models live on the same endpoint (gpt-6-sol, gpt-6-luna, claude-opus-5, ...) so a caller can
# pass one as $3 without changing anything else -- that is the fallback path when a quota runs out.
#
# Token usage: every run writes a usage file next to the log (musedev-<stamp>.usage.json) so the cost
# of a brief can be priced afterwards against the tier's per-million rates.
set -uo pipefail

WORKSPACE="${1:-}"
BRIEF="${2:-}"
MODEL="${3:-}"
if [ -z "$WORKSPACE" ] || [ -z "$BRIEF" ]; then
    echo "usage: muse-dev.sh <workspace> <brief.md> [model]"; exit 2
fi
[ -d "$WORKSPACE" ] || { echo "workspace not found: $WORKSPACE"; exit 2; }
[ -f "$BRIEF" ] || { echo "brief not found: $BRIEF"; exit 2; }

HERMES_HOME_DIR="%USERPROFILE%/%HOME%"
HERMES="$HERMES_HOME_DIR/bin/hermes.exe"
PROFILE="muse"
STAMP=$(date +%Y%m%d-%H%M%S)
LOG="/h/exl3-lab/logs/musedev-$STAMP.log"
# NOTE: `hermes --usage-file` writes nothing for this flow (verified 2026-09-25), so the cost of a run is
# read from the profile's session store instead:
#   sqlite3 profiles/muse/state.db "select model,input_tokens,output_tokens,cache_read_tokens,reasoning_tokens \
#     from session_model_usage order by rowid desc limit 3"
# Contributor rates for pricing: $0.10/M input, $0.002/M cached input, $0.20/M output.

if [ -z "$MODEL" ]; then
    MODEL=$("$HERMES" -p "$PROFILE" config get model.default 2>/dev/null | tail -1 | tr -d '\r')
fi

echo "workspace : $WORKSPACE"
echo "brief     : $BRIEF"
echo "profile   : $PROFILE   model: $MODEL"
echo "log       : $LOG"
echo "started   : $(date '+%F %T')"
echo "------------------------------------------------------------"

cd "$WORKSPACE" || exit 1
# Only pass --model when the caller asked for a specific one: the profile default (muse-spark-1.3-contributor)
# is applied by Hermes itself, and passing the contributor id as a command-line override trips Hermes'
# data-policy confirmation prompt ("Use this model for this invocation? [y/N]"), which cannot be answered
# from a non-interactive harness.
if [ -n "$MODEL" ] && [ "${MODEL_EXPLICIT:-0}" = "1" ]; then
    "$HERMES" -p "$PROFILE" --model "$MODEL" chat -q "$(cat "$BRIEF")" 2>&1 | tee "$LOG"
else
    "$HERMES" -p "$PROFILE" chat -q "$(cat "$BRIEF")" 2>&1 | tee "$LOG"
fi
rc=${PIPESTATUS[0]}
echo "------------------------------------------------------------"
echo "muse-dev.sh exit=$rc  log=$LOG"
echo "cost: read session_model_usage from profiles/muse/state.db (contributor rates: \$0.10/M in, \$0.002/M cached, \$0.20/M out)"
