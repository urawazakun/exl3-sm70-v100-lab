#!/usr/bin/env bash
# watch-brief28.sh -- when the n-gram re-test (#28) is done, hand the GPU back:
#   1. release the assistant's protective lock   (so #29 can measure)
#   2. restore the live deployment on 8081       (hidden window, KI-4)
#   3. verify health=200 + the operator's model answers one prompt
#   4. write /h/exl3-lab/logs/brief28/handback.txt and say what happened
#
# Why: the assistant holds the lock during #28's run so #29 cannot collide with its
# benchmark (two 15 GB servers cannot share a 16 GB card), and the deployment stays
# down while the bench owns the GPU. Nobody else releases it if the assistant is not
# watching, so this watcher does it on completion or after a hard deadline.
#
# Usage: watch-brief28.sh [deadline_minutes]   (default 75)
set -u
LAB=/h/exl3-lab
DEADLINE=${1:-75}
LOG="$LAB/logs/brief28/watch-handback.log"
mkdir -p "$LAB/logs/brief28"
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

say "=== watcher start (deadline ${DEADLINE}m) ==="
report="$LAB/logs/brief28/REPORT.md"
done_flag=0
for i in $(seq 1 $((DEADLINE * 2))); do
    # #28 is done when its REPORT.md exists, or when its console log has not grown for 6 minutes
    if [ -s "$report" ]; then done_flag=1; say "REPORT.md appeared"; break; fi
    m=$(stat -c %Y "$LAB/logs/musedev-ngram2-console.log" 2>/dev/null || echo 0)
    now=$(date +%s)
    if [ "$m" != "0" ] && [ $((now - m)) -gt 360 ]; then done_flag=1; say "console idle >6 min"; break; fi
    sleep 30
done
[ "$done_flag" = "1" ] || say "DEADLINE reached — acting anyway"

say "--- releasing the assistant's lock (if still held)"
bash "$LAB/scripts/gpu-lock.sh" release 2>&1 | tail -2 | tee -a "$LOG"

say "--- current servers"
MSYS_NO_PATHCONV=1 tasklist /FO CSV 2>/dev/null | grep -i llama-server | sed 's/^/  /' | tee -a "$LOG"

if [ "$(curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:8081/health 2>/dev/null)" != "200" ]; then
    say "--- killing any llama-server and restoring 8081"
    MSYS_NO_PATHCONV=1 taskkill /IM llama-server.exe /F 2>&1 | head -2 | sed 's/^/  /' | tee -a "$LOG"
    sleep 6
    MSYS_NO_PATHCONV=1 powershell.exe -NoProfile -Command "Start-Process -FilePath 'powershell.exe' -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File','H:\exl3-local\scripts\start-server-exl3.ps1') -WindowStyle Hidden" >>"$LOG" 2>&1
    for k in $(seq 1 20); do
        sleep 15
        [ "$(curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:8081/health 2>/dev/null)" = "200" ] && { say "restored: health=200 after $((k*15))s"; break; }
    done
else
    say "8081 already healthy"
fi

# verify the operator's model actually answers
ans=$(curl -s -m 120 -X POST http://127.0.0.1:8081/v1/chat/completions -H 'Content-Type: application/json' \
      -d '{"messages":[{"role":"user","content":"Reply with exactly: OK"}],"max_tokens":24,"temperature":0}' 2>/dev/null \
      | python -c "import sys,json;d=json.load(sys.stdin);m=d['choices'][0]['message'];print((m.get('content') or m.get('reasoning_content') or '')[:60].replace(chr(10),' '))" 2>/dev/null)
vram=$(MSYS_NO_PATHCONV=1 nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader 2>/dev/null | tr -d '\r')
say "final: health=$(curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:8081/health 2>/dev/null) vram=[$vram] servers=$(MSYS_NO_PATHCONV=1 tasklist /FO CSV 2>/dev/null | grep -ci llama-server) model_says=[$ans]"
rm -f "$LAB/logs/RESTORE-NEEDED"
{
  echo "watcher finished $(date '+%F %T')"
  echo "done_flag=$done_flag"
  echo "8081_health=$(curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:8081/health 2>/dev/null)"
  echo "vram=$vram"
  echo "model_says=$ans"
  echo "lock_status: $(bash "$LAB/scripts/gpu-lock.sh" status 2>&1 | head -1)"
} > "$LAB/logs/brief28/handback.txt"
say "=== watcher end ==="
