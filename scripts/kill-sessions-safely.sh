#!/usr/bin/env bash
# kill-sessions-safely.sh -- stop lab sessions WITHOUT leaving the box in a broken state.
#
# Why: killing a session can orphan its llama-server (the child survives), which then holds ~13 GB of VRAM
# while the live deployment on 8081 is down — measured twice (2026-09-25). This script does the whole sweep:
#   1. kill the given pids (and their trees)
#   2. kill any orphaned llama-server
#   3. free the GPU lock if it is stale (holder pid dead)
#   4. wait for the GPU to drain, then restore the live deployment on 8081 (hidden window, KI-4)
#   5. verify health=200 and print the VRAM footprint
#
# Usage: kill-sessions-safely.sh <pid> [<pid> ...]
set -u
LAB=/h/exl3-lab
LOG="$LAB/logs/kill-sessions.log"
say() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }

[ "$#" -ge 1 ] || echo "(no pids given: doing the sweep + deployment restore only)"
say "=== killing pids: $* ==="
for pid in "$@"; do
    MSYS_NO_PATHCONV=1 taskkill /PID "$pid" /T /F 2>&1 | head -3 | sed 's/^/  /' | tee -a "$LOG"
done
sleep 3

say "--- sweeping orphaned llama-server processes"
n=$(MSYS_NO_PATHCONV=1 tasklist /FO CSV 2>/dev/null | grep -ci llama-server)
if [ "$n" != "0" ]; then
    MSYS_NO_PATHCONV=1 taskkill /IM llama-server.exe /F 2>&1 | head -4 | sed 's/^/  /' | tee -a "$LOG"
    sleep 5
fi

say "--- releasing the GPU lock if stale"
bash "$LAB/scripts/gpu-lock.sh" release 2>&1 | tail -1 | sed 's/^/  /' | tee -a "$LOG"

say "--- waiting for the GPU to drain"
for i in $(seq 1 20); do
    used=$(MSYS_NO_PATHCONV=1 nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d '\r')
    [ "${used:-9999}" -lt 200 ] && { say "  GPU drained (used=${used} MiB) after $((i*5))s"; break; }
    sleep 5
done

h=$(curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:8081/health 2>/dev/null)
if [ "$h" = "200" ]; then
    say "deployment already healthy (health=200)"
else
    say "--- restoring the live deployment on 8081 (hidden window)"
    MSYS_NO_PATHCONV=1 powershell.exe -NoProfile -Command "Start-Process -FilePath 'powershell.exe' -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File','H:\exl3-local\scripts\start-server-exl3.ps1') -WindowStyle Hidden" >>"$LOG" 2>&1
    for i in $(seq 1 24); do
        sleep 15
        if [ "$(curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:8081/health 2>/dev/null)" = "200" ]; then
            say "  restored: health=200 after $((i*15))s"
            break
        fi
    done
fi
say "final: health=$(curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:8081/health 2>/dev/null) VRAM=$(MSYS_NO_PATHCONV=1 nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader 2>/dev/null | tr -d '\r') servers=$(MSYS_NO_PATHCONV=1 tasklist /FO CSV 2>/dev/null | grep -ci llama-server)"
