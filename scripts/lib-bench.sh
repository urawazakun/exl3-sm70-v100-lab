#!/usr/bin/env bash
# lib-bench.sh -- one shared harness library for every V100 benchmark script.
#
# Source it, do not execute it:
#   LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#   # shellcheck source=/dev/null
#   source "$LIB_DIR/lib-bench.sh"
#
# Sourcing has NO side effects: no traps installed, no servers started or killed,
# no directories created. All behaviour lives in the functions below.
#
# Standard lab invocation encoded here (== the live deployment as of 2026-09-25,
# see H:/exl3-local/scripts/start-server-exl3.ps1):
#   live binary, live 3bpw MTP GGUF, MMA8, -c 131072, KV q4_0, MTP k=3,
#   -b 2048 -ub 512, -t 6, --jinja --no-reasoning-preserve.
# Override per-script via environment (keeps old scripts' measurement semantics
# identical while they share the code paths):
#   LAB_BIN / LAB_MODEL / LAB_CTX, or pass extra flags to start_lab_server
#   (extra flags are appended AFTER the defaults; llama.cpp last-flag-wins, so
#   passing '-c 65536' there reverts the context size for that call only).
#
# The three pitfalls this library encodes (see HARNESS.md for the why):
#   1. MSYS paths handed to native Windows python/exe  -> win_path everywhere.
#   2. '$!' is not a Windows PID under MSYS            -> kill_servers by image name.
#   3. restore must be idempotent                       -> restore_deployment checks first.

# --- guard: refuse to run as a script ---------------------------------------------------------------
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo 'lib-bench.sh must be sourced, not executed' >&2
    exit 2
fi

# --- configuration (environment-overridable) --------------------------------------------------------
: "${LAB_BIN:=H:/exl3-local/build-v100/bin/llama-server.exe}"                                       # live binary
: "${LAB_MODEL:=H:/exl3-local/models/exl3-gguf/huihui-qwen3.8-27b-abliterated-exl3-3bpw-mtp.gguf}"   # live 3bpw MTP GGUF
: "${LAB_CTX:=131072}"
: "${LAB_PORT:=8082}"
: "${LAB_DEPLOY_PORT:=8081}"
: "${LAB_DEPLOY_START:=H:\\exl3-local\\scripts\\start-server-exl3.ps1}"
: "${LAB_PROMPT_JSON:=H:/exl3-lab/bench/prompt.json}"       # 512-token greedy decode prompt
: "${LAB_SMI:=/c/Windows/System32/nvidia-smi.exe}"
: "${LAB_LOCK:=/h/exl3-lab/scripts/gpu-lock.sh}"

# Standard lab invocation pieces. Built at CALL time (not source time) so a script can
# override per case, e.g.  LAB_CTX=65536 start_lab_server ...  or
# LAB_MODEL=... LAB_BIN=... start_lab_server ... . Extra flags are appended AFTER the
# defaults; llama.cpp scalar options are last-wins, so an extra '-c 65536' / '-b 4096' /
# '--spec-type none' overrides the default for that call only. Measurement-critical
# flags (-c, -b, -ub, --spec-*) are therefore safest set via LAB_* env (no duplication),
# free-form extras (e.g. -md, -lv) via "$@".
start_lab_server() {
    local port=$1 slog=$2; shift 2
    EXL3_EXPERIMENTAL_MMA8=1 "$LAB_BIN" \
        -m "$LAB_MODEL" --alias bench -ngl 99 -sm none -mg 0 \
        -c "$LAB_CTX" -np 1 -fa on -b 2048 -ub 512 -ctk q4_0 -ctv q4_0 -t 6 \
        --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-n-min 1 \
        --host 127.0.0.1 --port "$port" --jinja --no-reasoning-preserve "$@" >"$slog" 2>&1 &
    LAB_SERVER_LOG="$slog"
}

# --- 1. path conversion -----------------------------------------------------------------------------
# win_path <msys-path>: convert to a Windows path for native python/exe.
# Pitfall #1: a bare /h/... string is NOT absolute for the Windows interpreter -- it resolves to
# <current-drive>\h\... and open(...,'w') dies with FileNotFoundError. One cygpath -m before every
# python block fixes it; do it here, once.
win_path() {
    if command -v cygpath >/dev/null 2>&1; then
        cygpath -m "$1"
    else
        printf '%s' "$1"
    fi
}

# --- 2. killing and waiting -------------------------------------------------------------------------
# kill_servers: kill by IMAGE NAME, never by $!.
# Pitfall #2: under MSYS, $! is an MSYS pid, not a Windows pid -- `taskkill /PID $!` and even bash
# `kill $PID` on a spawned .exe silently kill nothing, leaving a 13 GB server holding the card while
# the machine looks idle. The bench owns the card exclusively, so kill by image name.
kill_servers() {
    MSYS_NO_PATHCONV=1 taskkill /IM llama-server.exe /F >/dev/null 2>&1 || true
}

# wait_gpu_free [seconds]: poll nvidia-smi until no compute app holds the card.
# Prints the leftover table on timeout and returns 1.
wait_gpu_free() {
    local budget="${1:-60}" i n
    for ((i = 0; i < budget; i += 3)); do
        n=$(MSYS_NO_PATHCONV=1 "$LAB_SMI" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
        if [ "$n" -eq 0 ]; then return 0; fi
        sleep 3
    done
    echo 'GPU still busy:' >&2
    MSYS_NO_PATHCONV=1 "$LAB_SMI" --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader >&2 || true
    return 1
}

# --- server lifecycle -------------------------------------------------------------------------------
# wait_health <port> <seconds>: return 0 once /health answers 200, else 1.
wait_health() {
    local port=$1 budget=${2:-180} code
    for ((i = 0; i < budget; i += 3)); do
        code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 4 "http://127.0.0.1:$port/health" || true)
        if [ "$code" = "200" ]; then return 0; fi
        sleep 3
    done
    return 1
}

# restore_deployment: IDEMPOTENT -- if 8081/health already answers 200, do nothing.
# Pitfall #3: every old script unconditionally started the deployment at the end, so two overlapping
# scripts could start two servers on one card (OOM). Checking first is safe under any interleaving.
restore_deployment() {
    local code
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 4 "http://127.0.0.1:$LAB_DEPLOY_PORT/health" || true)
    if [ "$code" = "200" ]; then
        echo "deployment already live on $LAB_DEPLOY_PORT (/health 200), nothing to do"
        return 0
    fi
    echo "restoring the live deployment on $LAB_DEPLOY_PORT"
    MSYS_NO_PATHCONV=1 powershell.exe -NoProfile -ExecutionPolicy Bypass -Command \
        "Start-Process -FilePath 'powershell.exe' -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File','$LAB_DEPLOY_START') -WindowStyle Hidden" >/dev/null 2>&1
    if wait_health "$LAB_DEPLOY_PORT" 120; then
        echo "live server back: $LAB_DEPLOY_PORT /health 200, VRAM: $(vram_used)"
        return 0
    fi
    echo "WARNING: live server did not answer 200; check H:/exl3-local/logs/server-exl3.err.log" >&2
    return 1
}

# --- 3. standard probes ------------------------------------------------------------------------------
# probe_decode <port> <out.json>: the fixed 512-token greedy decode probe (bench/prompt.json).
# All Windows paths are converted with win_path inside -- never hand an MSYS path to python.
probe_decode() {
    local port=$1 out=$2
    python - "$(win_path "$LAB_PROMPT_JSON")" "$(win_path "$out")" "$port" <<'PY'
import json, sys, urllib.request
prompt_f, out_f, port = sys.argv[1], sys.argv[2], sys.argv[3]
with open(prompt_f, encoding='utf-8') as f:
    body = json.load(f)
body.update(max_tokens=512, temperature=0.0, stream=False)
req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions',
                             data=json.dumps(body).encode('utf-8'),
                             headers={'Content-Type': 'application/json'})
with urllib.request.urlopen(req, timeout=1800) as r:
    result = json.load(r)
with open(out_f, 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
t = result.get('timings', {})
acc = (t['draft_n_accepted'] / t['draft_n']) if t.get('draft_n') else None
print(f"decode: {t.get('predicted_per_second')} tok/s, {t.get('predicted_n')} tokens, acceptance {acc}")
PY
}

# probe_prefill <port> <prefix.txt> <out.json> [suffix]: one post of the long stable
# prefix + tiny suffix (default suffix "\n\nQuestion: reply with the single word ok.").
# (The 4,564-token prefix: bench/prefix-8k.txt posts as prompt_n ~4520 plus the suffix line.)
probe_prefill() {
    local port=$1 prefix=$2 out=$3 suffix="${4:-$'\n\nQuestion: reply with the single word ok.'}"
    python - "$(win_path "$prefix")" "$(win_path "$out")" "$port" "$suffix" <<'PY'
import json, sys, urllib.request
prefix = open(sys.argv[1], encoding='utf-8').read()
out, port, suffix = sys.argv[2], sys.argv[3], sys.argv[4]
body = {"messages": [{"role": "user", "content": prefix + suffix}],
        "max_tokens": 8, "temperature": 0.0, "stream": False}
req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions',
                             data=json.dumps(body).encode('utf-8'),
                             headers={'Content-Type': 'application/json'})
with urllib.request.urlopen(req, timeout=1800) as r:
    result = json.load(r)
with open(out, 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
t = result.get('timings', {})
ptps = (t['prompt_n'] / (t['prompt_ms'] / 1000.0)) if t.get('prompt_ms') else None
print(f"prefill: {ptps} tok/s, prompt_n {t.get('prompt_n')}, cache_n {t.get('cache_n')}")
PY
}

# --- telemetry ------------------------------------------------------------------------------------------------
vram_used() { MSYS_NO_PATHCONV=1 "$LAB_SMI" --query-gpu=memory.used --format=csv,noheader 2>/dev/null | tr -d '\r '; }
vram_free() { MSYS_NO_PATHCONV=1 "$LAB_SMI" --query-gpu=memory.free --format=csv,noheader 2>/dev/null | tr -d '\r '; }

# host_ws_mb [process-name]: server's host working set in MiB via Get-Process (for --no-mmproj-offload accounting).
host_ws_mb() {
    local name="${1:-llama-server}"
    MSYS_NO_PATHCONV=1 powershell.exe -NoProfile -ExecutionPolicy Bypass -Command \
        "(Get-Process -Name '$name' -ErrorAction SilentlyContinue | Measure-Object -Property WorkingSet64 -Sum).Sum / 1MB" 2>/dev/null | tr -d '\r '
}

# --- lock + exit -----------------------------------------------------------------------------------------------
gpu_lock_acquire() { bash "$LAB_LOCK" acquire "${1:-45}"; }
gpu_lock_release() { bash "$LAB_LOCK" release; }

# on_exit: trap-friendly cleanup -- kill bench servers, restore the deployment, release the lock.
# Usage: trap on_exit EXIT INT TERM   (after acquiring the lock; safe to call without it too)
on_exit() {
    kill_servers
    wait_gpu_free 60 || true
    restore_deployment || true
    bash "$LAB_LOCK" release || true
}
