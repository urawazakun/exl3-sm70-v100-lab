#!/usr/bin/env bash
# run-deploy-probes.sh -- wait for the LIVE deployment (8081) to come back, then run the corrected
# needle probes against it (v3: max_tokens=1024, reasoning_content also searched).
#
# Why the live server: the brief-16 run measured the lab build and its 64-token cap made every probe
# score MISS (all tokens went to the reasoning channel). These probes measure the configuration the
# operator actually uses: 128k ctx, -fa on, MTP k=3, q4_0 KV.
#
# Usage: run-deploy-probes.sh [health_wait_minutes]
set -u
WAIT_MIN="${1:-60}"
PY="python"
BASE="http://127.0.0.1:8081"
SRC="/h/exl3-lab/logs/brief16"
OUT="/h/exl3-lab/logs/deploy-probes"
J="$OUT/journal.txt"
mkdir -p "$OUT"

say() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$J"; }

touch "$OUT/RUNNING"
say "=== step 0: wait for the GPU lock to be FREE (max 60 min) ==="
# A lab session legitimately stops the deployment to take the GPU (run16d did exactly that at
# 04:41:39 on 2026-09-25 and cut a probe in half). So: do not probe while someone else holds the
# lock -- wait for them to finish, THEN take the lock ourselves so nobody cuts us in turn.
for i in $(seq 1 120); do
  st=$(bash /h/exl3-lab/scripts/gpu-lock.sh status 2>/dev/null | head -1)
  case "$st" in
    FREE*) say "lock FREE after $((i * 30))s"; break ;;
  esac
  sleep 30
done
say "=== step 1: acquire the GPU lock ==="
bash /h/exl3-lab/scripts/gpu-lock.sh acquire 45 2>&1 | tail -2 | tee -a "$J"
say "=== step 2: wait for deployment /health (max ${WAIT_MIN} min) ==="
up=0
for i in $(seq 1 $((WAIT_MIN * 2))); do
  h=$(curl -s -m 5 -o /dev/null -w '%{http_code}' "$BASE/health" 2>/dev/null)
  if [ "$h" = "200" ]; then up=1; say "deployment UP after $((i * 30))s (health=200)"; break; fi
  sleep 30
done
if [ "$up" != "1" ]; then say "ABORT: deployment did not come up within ${WAIT_MIN} min"; exit 1; fi
curl -s -m 10 "$BASE/props" 2>/dev/null | head -c 300 | tr -d '\r' >> "$J"; echo >> "$J"
say "GPU: $(MSYS_NO_PATHCONV=1 nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader 2>/dev/null | tr -d '\r')"

# needle -> expected code (from the brief-16 prompt files)
run_one() {
  name="$1"; code="$2"; f="$SRC/needle-$name.prompt.json"
  [ -f "$f" ] || { say "skip $name (no prompt file)"; return; }
  say "--- probe $name (expect $code, max_tokens=1024) ---"
  timeout 1500 $PY "H:/exl3-lab/scripts/probe3.py" "$BASE" "$code" "$(cygpath -m "$f")" "$(cygpath -m "$OUT/needle-$name")" 1024 2>&1 | tee -a "$J"
}

run_one "8k-mid"   "TOKEN-3129"
run_one "32k-mid"  "PLATE-8072"
run_one "96k-mid"  "BADGE-8973"
run_one "96k-early" "PASS-2372"
run_one "96k-late"  "TOKEN-1137"

bash /h/exl3-lab/scripts/gpu-lock.sh release 2>&1 | tail -1 | tee -a "$J"
rm -f "$OUT/RUNNING"
say "=== deploy probes done ==="
grep -a "^RESULT" "$J" | tail -12 | sed 's/^/  /'
