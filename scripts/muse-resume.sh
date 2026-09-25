#!/usr/bin/env bash
# muse-resume.sh -- resume a Muse (Contributor) session that was cut off by the 5-hour API limit.
# Same environment as muse-dev.sh, but with `--resume <session_id>` so the session keeps its context
# (a 591-tool-call session that lost its context is expensive to redo; the worktree/build survive anyway).
#
# Usage: muse-resume.sh <session_id> <instructions_file>
set -uo pipefail
SID="${1:?usage: muse-resume.sh <session_id> <instructions_file>}"
BRIEF="${2:?usage: muse-resume.sh <session_id> <instructions_file>}"

HERMES_HOME_DIR="%USERPROFILE%/%HOME%"
HERMES="$HERMES_HOME_DIR/bin/hermes.exe"
PROFILE="${MUSE_PROFILE:-muse}"
LOG="/h/exl3-lab/logs/museresume-$SID-$(date +%Y%m%d-%H%M%S).log"

[ -f "$BRIEF" ] || { echo "brief not found: $BRIEF"; exit 2; }

echo "session : $SID"
echo "profile : $PROFILE"
echo "brief   : $BRIEF"
echo "log     : $LOG"

"$HERMES" -p "$PROFILE" --resume "$SID" chat -q "$(cat "$BRIEF")" 2>&1 | tee "$LOG"
