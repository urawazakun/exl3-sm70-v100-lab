#!/usr/bin/env bash
# muse-swarm.sh -- launch N Muse Spark 1.3 (contributor) developer sessions in parallel, one per brief,
# each isolated so they cannot trip over each other.
#
# Why this exists: the contributor tier is cheap enough that several developer sessions are affordable,
# but this machine has exactly one V100 and one git workspace. So the swarm enforces two rules:
#
#   1. WORKSPACE ISOLATION -- every session gets its own copy-on-write git worktree of the lab repo
#      (or its own subdirectory when the brief is not code work), so parallel writes never collide.
#   2. GPU SERIALIZATION -- one card, so briefs that need the GPU must take
#      `H:\exl3-lab\scripts\gpu-lock.sh acquire <min>` before starting a server/bench and release it
#      afterwards. Non-GPU work (source reading, docs, diffing, analysis) runs fully parallel.
#
# Usage:
#   muse-swarm.sh <name>:<brief.md> [<name>:<brief.md> ...]
#   e.g. muse-swarm.sh prefill:PROMPT15.md ctxvalid:PROMPT16.md sourceaudit:PROMPT-audit.md
#
# Each session logs to H:\exl3-lab\logs\musedev-<stamp>-<name>.log (+ .usage.json) and, for code work,
# works in H:\exl3-lab\worktrees\<name> (created from the lab repo's HEAD).
set -uo pipefail

LAB="/h/exl3-lab"
HERMES_HOME_DIR="%USERPROFILE%/%HOME%"
HERMES="$HERMES_HOME_DIR/bin/hermes.exe"
PROFILE="muse"
STAMP=$(date +%Y%m%d-%H%M%S)
MAX_PARALLEL="${MUSE_SWARM_MAX:-3}"

if [ $# -eq 0 ]; then
    echo "usage: muse-swarm.sh <name>:<brief.md> [<name>:<brief.md> ...]"; exit 2
fi

running=0
for spec in "$@"; do
    name="${spec%%:*}"
    brief="${spec#*:}"
    case "$name" in
        -*|"") echo "bad spec: $spec"; exit 2 ;;
    esac
    if [ ! -f "$brief" ]; then brief="$LAB/$brief"; fi
    if [ ! -f "$brief" ]; then echo "brief not found: $brief (looked in cwd and $LAB)"; exit 2; fi
    # Make it absolute now: the session subshell cd's into its worktree, where a relative brief path
    # no longer resolves (and untracked briefs are not in the worktree at all).
    brief="$(cd "$(dirname "$brief")" && pwd)/$(basename "$brief")"

    workdir="$LAB/worktrees/$name"
    # native git needs Windows-style paths (MSYS /h/... is not resolvable for it). A worktree left over
    # from an earlier run is still registered in git, and `worktree add -f` refuses to reuse it: remove
    # the registration + the directory first, otherwise every session silently falls back to $LAB and
    # they all share one directory (observed 2026-09-25 03:54).
    if [ -d "$LAB/.git" ]; then
        git -C H:/exl3-lab worktree remove --force "H:/exl3-lab/worktrees/$name" >/dev/null 2>&1
        rm -rf "$workdir" 2>/dev/null
        git -C H:/exl3-lab worktree add -f "H:/exl3-lab/worktrees/$name" HEAD >/dev/null 2>&1 \
            && echo "[swarm] worktree for $name: $workdir" \
            || { echo "[swarm] worktree failed for $name, falling back to \$LAB"; workdir="$LAB"; }
    else
        workdir="$LAB"
    fi

    log="$LAB/logs/musedev-$STAMP-$name.log"
    echo "[swarm] launching $name  brief=$brief  workdir=$workdir  log=$log"

    (
        cd "$workdir" || exit 1
        # The brief text is prefixed (not replaced) so each session knows the swarm rules; the brief
        # itself still carries the task. GPU rule is stated in every session, not only in code briefs.
        {
          printf '%s\n\n' "SWARM CONTEXT: you are one of up to $MAX_PARALLEL developer sessions running in parallel on a single V100 machine (H: drive, Windows, MSYS bash). Two rules: (1) stay inside your worktree/workspace; do not write to another session's directory; (2) before starting any llama-server, benchmark, or anything that touches the GPU, run \`bash /h/exl3-lab/scripts/gpu-lock.sh acquire 45\` and release it with \`bash /h/exl3-lab/scripts/gpu-lock.sh release\` when finished -- exactly one session may hold the card at a time. The live deployment on port 8081 is read-only; stop/restore it only while holding the lock. Work source code in your own worktree; put reports under /h/exl3-lab/logs/<your-name>/."
          cat "$brief"
        } > "$log.brief"
        cp "$log.brief" "$log"
        "$HERMES" -p "$PROFILE" chat -q "$(cat "$log.brief")" >> "$log" 2>&1
        echo "[swarm] $name finished rc=$? log=$log" >> "$log"
    ) &
    running=$(( running + 1 ))
    if [ "$running" -ge "$MAX_PARALLEL" ]; then
        wait
        running=0
    fi
done
wait
echo "[swarm] all sessions finished; logs: $LAB/logs/musedev-$STAMP-*.log"
