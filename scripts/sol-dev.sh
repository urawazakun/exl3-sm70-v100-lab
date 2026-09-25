#!/usr/bin/env bash
# sol-dev.sh -- the "developer with full process access" launcher: Codex profile 'sol'
# (gpt-6-sol, reasoning medium, the cheaper pin the user asked for) with `--sandbox
# danger-full-access`, because workspace-write denies every process operation on this host
# (Stop-Process / taskkill / Get-NetTCPConnection all returned access denied), which left Sol unable
# to take the GPU at all.
#
# Safety comes from the process boundary, not the sandbox: one git workspace (H:\exl3-lab), a narrow
# brief, the live tree/ports declared off limits in the brief, and the operator reviewing `git diff`
# and the lab logs afterwards.
#
#   bash sol-dev.sh [WORKDIR] [BRIEF_FILE]
set -u
WORKDIR="${1:-H:/exl3-lab}"
BRIEF="${2:-H:/exl3-lab/PROMPT5.md}"
LOGDIR="/h/exl3-lab/logs"
mkdir -p "$LOGDIR"
STAMP=$(date +%Y%m%d-%H%M%S)
LOG="$LOGDIR/soldev-$STAMP.log"

[ -f "$BRIEF" ] || { echo "brief not found: $BRIEF"; exit 1; }
cd /h/exl3-lab 2>/dev/null || true
[ -d .git ] || git init -q .

echo "workdir : $WORKDIR"
echo "brief   : $BRIEF  ($(wc -c < "$BRIEF") bytes)"
echo "log     : $LOG"
echo "model   : $(grep -m1 '^model' "$HOME/.codex/sol.config.toml" 2>/dev/null)"
echo "sandbox : danger-full-access"
echo "------------------------------------------------------------"

codex exec --profile sol --sandbox danger-full-access -C "$WORKDIR" "$(cat "$BRIEF")" 2>&1 | tee "$LOG"
echo "------------------------------------------------------------"
echo "sol-dev.sh exit=$?  log=$LOG"
