#!/usr/bin/env bash
# gpu-lock.sh -- advisory lock so any number of Muse/Sol sessions can run in parallel while only one
# of them ever touches the V100. The card is the one resource that cannot be shared: two servers, two
# benches or a bench plus the live deployment end in an OOM or a silently wrong number.
#
# Usage:
#   gpu-lock.sh acquire [minutes]   # blocks (default wait 60 min) then prints LOCK_HELD
#   gpu-lock.sh release
#   gpu-lock.sh status
#
# Rule for briefs: acquire before starting a server or a bench, release in the same command sequence
# (or via the trap pattern shown below), and never hold it while merely editing source.
#
#   bash /h/exl3-lab/scripts/gpu-lock.sh acquire 30 || exit 1
#   ... bench ...
#   bash /h/exl3-lab/scripts/gpu-lock.sh release
set -uo pipefail

LOCKDIR="/h/exl3-lab/locks/gpu.lock"
INFO="$LOCKDIR/owner"
WAIT_DEFAULT_MIN=60
mkdir -p "$(dirname "$LOCKDIR")"

now() { date '+%F %T'; }

acquire() {
    local minutes="${1:-$WAIT_DEFAULT_MIN}"
    local deadline=$(( $(date +%s) + minutes*60 ))
    local waited=0
    while :; do
        if mkdir "$LOCKDIR" 2>/dev/null; then
            printf '%s pid=%s session=%s\n' "$(now)" "$$" "${HERMES_SESSION_ID:-unknown}" > "$INFO"
            echo "LOCK_HELD owner=$$ at $(now)"
            return 0
        fi
        # stale lock? the owner writes its pid; if that pid is gone, take it over.
        local owner
        owner=$(grep -oE 'pid=[0-9]+' "$INFO" 2>/dev/null | head -1 | cut -d= -f2)
        if [ -n "${owner:-}" ] && ! kill -0 "$owner" 2>/dev/null; then
            echo "taking over stale lock (owner pid $owner is gone)"
            rm -rf "$LOCKDIR"
            continue
        fi
        if [ $(( $(date +%s) )) -ge "$deadline" ]; then
            echo "LOCK_TIMEOUT after ${minutes} min; current holder:"
            cat "$INFO" 2>/dev/null
            return 1
        fi
        [ $(( waited % 60 )) -eq 0 ] && echo "waiting for the GPU lock: $(cat "$INFO" 2>/dev/null)"
        waited=$(( waited + 10 ))
        sleep 10
    done
}

release() {
    if [ -d "$LOCKDIR" ]; then
        rm -rf "$LOCKDIR"
        echo "LOCK_RELEASED at $(now)"
    else
        echo "no lock held"
    fi
}

case "${1:-}" in
    acquire) shift; acquire "${1:-}" ;;
    release) release ;;
    status)  if [ -d "$LOCKDIR" ]; then echo "HELD"; cat "$INFO" 2>/dev/null; else echo "FREE"; fi ;;
    *) echo "usage: gpu-lock.sh acquire [minutes] | release | status"; exit 2 ;;
esac
