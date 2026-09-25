#!/usr/bin/env bash
# deploy-watchdog.sh -- restore the live EXL3 deployment (8081) once the lab sessions are finished.
#
# Why: sessions legitimately stop the deployment to take the GPU for a benchmark and are supposed to
# restore it themselves. When one forgets (or dies), the operator's model stays unavailable while the GPU
# sits idle. This watcher restores it only when the coast is clear:
#   * the GPU lock is FREE, AND
#   * no llama-server process is running, AND
#   * that has been true continuously for GRACE seconds (default 300),
# so it can never fight a session for the GPU (the KI-2 failure mode).
#
# It also never steals the screen (KI-4): the deployment is started with a hidden PowerShell process.
#
# Usage: deploy-watchdog.sh [grace_seconds] [max_hours]
set -u
GRACE="${1:-300}"
MAXH="${2:-4}"
LAB=/h/exl3-lab
LOG="$LAB/logs/deploy-watchdog.log"
BASE=http://127.0.0.1:8081
START_PS='H:\exl3-local\scripts\start-server-exl3.ps1'

say() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }
health() { curl -s -m 5 -o /dev/null -w '%{http_code}' "$BASE/health" 2>/dev/null; }
servers() { MSYS_NO_PATHCONV=1 tasklist /FO CSV 2>/dev/null | grep -ci llama-server; }
lockstate() { bash "$LAB/scripts/gpu-lock.sh" status 2>/dev/null | head -1; }

say "=== watchdog start (grace ${GRACE}s, max ${MAXH}h) ==="
deadline=$(( $(date +%s) + MAXH * 3600 ))
quiet_since=""
healthy_noted=""

while [ "$(date +%s)" -lt "$deadline" ]; do
    h=$(health); st=$(lockstate); sv=$(servers)
    if [ "$h" = "200" ]; then
        [ "$healthy_noted" != "1" ] && { say "deployment healthy (health=200); staying on watch"; healthy_noted=1; }
        quiet_since=""
        sleep 60
        continue
    fi
    case "$st" in
        FREE*)
            if [ "$sv" = "0" ]; then
                [ -z "$quiet_since" ] && { quiet_since=$(date +%s); say "coast clear: lock FREE, no llama-server, 8081 down -> starting ${GRACE}s grace timer"; }
                if [ $(( $(date +%s) - quiet_since )) -ge "$GRACE" ]; then
                    say "grace elapsed; restoring the deployment (hidden window)"
                    MSYS_NO_PATHCONV=1 powershell.exe -NoProfile -Command "Start-Process -FilePath 'powershell.exe' -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File','$START_PS') -WindowStyle Hidden" >>"$LOG" 2>&1
                    for i in $(seq 1 24); do
                        sleep 15
                        if [ "$(health)" = "200" ]; then
                            say "restored: health=200 after $((i*15))s; VRAM=$(MSYS_NO_PATHCONV=1 nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader 2>/dev/null | tr -d '\r')"
                            healthy_noted=1
                            quiet_since=""
                            break
                        fi
                    done
                    say "restore attempt did not reach health=200 within 6 min (will retry while the coast stays clear)"
                    quiet_since=""
                fi
            else
                [ -n "$quiet_since" ] && { say "a llama-server appeared ($sv) -> cancelling grace timer"; quiet_since=""; }
            fi
            ;;
        *)
            [ -n "$quiet_since" ] && { say "lock is held again ($st) -> cancelling grace timer"; quiet_since=""; }
            ;;
    esac
    sleep 60
done
say "=== watchdog deadline reached without restoring the deployment ==="
