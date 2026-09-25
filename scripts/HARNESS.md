# HARNESS.md -- the one-page contract for every V100 benchmark session

Every benchmark script sources `scripts/lib-bench.sh` and follows one recipe. This page
exists so later sessions stop re-learning the same three failure modes -- two of them
already cost a full cycle each.

## The recipe: one GPU, one lock, restore at the end

```bash
LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$LIB_DIR/lib-bench.sh"          # no side effects on source

gpu_lock_acquire 45 || exit 1           # exactly one session holds the card
trap on_exit EXIT INT TERM              # kill bench servers, restore 8081, release lock
kill_servers; wait_gpu_free 60          # the deployment holds the card: kill FIRST, then wait
start_lab_server 8082 "$OUT/case.server.log" -lv 4   # standard lab invocation + extras
wait_health 8082 180                    # /health 200 before probing
probe_decode 8082 "$OUT/case.json"      # 512-token greedy decode probe
probe_prefill 8082 bench/prefix-8k.txt "$OUT/case.prefill.json"  # long-prefix probe
# ... on_exit handles the rest: kill, idempotent restore, lock release
```

Rules:

- GPU only under the lock: `bash /h/exl3-lab/scripts/gpu-lock.sh acquire 45`, release after.
  Never hold it while merely editing source.
- The live deployment on 8081 is read-only: stop/restore it only while holding the lock.
- Never change a measurement flag, prompt, or threshold to make a number look better. If a
  refactor changes a number, that is a regression: fix it and report it.
- Reports go to `/h/exl3-lab/logs/<brief-name>/`; every claim carries its command and log/JSON.

## The library API (`source scripts/lib-bench.sh`; sourcing is side-effect free)

| function | what it does |
|---|---|
| `win_path <msys-path>` | `cygpath -m` with fallback; use for EVERYTHING handed to native python/exe |
| `kill_servers` | `taskkill /IM llama-server.exe /F` (image name, never a PID) |
| `wait_gpu_free [sec]` | poll `nvidia-smi --query-compute-apps` until empty; prints leftovers on timeout |
| `start_lab_server <port> <log> [flags...]` | standard lab invocation (live binary, 3bpw MTP GGUF, MMA8, `-c 131072`, KV q4_0, MTP k=3), backgrounded; extra flags appended after defaults (last-wins) |
| `wait_health <port> [sec]` | 0 once `/health` answers 200 |
| `restore_deployment` | idempotent: no-op if 8081 already 200, else starts it via `start-server-exl3.ps1` and waits |
| `probe_decode <port> <out.json>` | fixed 512-token greedy decode probe from `bench/prompt.json` |
| `probe_prefill <port> <prefix.txt> <out.json> [suffix]` | long-prefix probe (4,564-token prefix + tiny suffix; default suffix the standard question line) |
| `vram_used` / `vram_free` | `nvidia-smi` MiB readings |
| `host_ws_mb [name]` | server host working set in MiB via `Get-Process` (for `--no-mmproj-offload` accounting) |
| `gpu_lock_acquire <min>` / `gpu_lock_release` | thin wrappers over `scripts/gpu-lock.sh` |
| `on_exit` | trap-friendly: kill servers, restore deployment, release lock (safe without the lock too) |

Standard lab invocation = the live deployment as of 2026-09-25 (`H:/exl3-local/scripts/start-server-exl3.ps1`):
live binary, live 3bpw MTP GGUF, `EXL3_EXPERIMENTAL_MMA8=1`, `-c 131072`, KV q4_0, MTP k=3,
`-b 2048 -ub 512`, `-t 6`, `--jinja --no-reasoning-preserve`. Campaigns that measured at
64k context set `LAB_CTX=65536` before `start_lab_server`; per-case `-b`/`-ub`/`--spec-*`
travel as extra args. Overrides: `LAB_BIN`, `LAB_MODEL`, `LAB_CTX`, `LAB_PORT`,
`LAB_DEPLOY_PORT`, `LAB_PROMPT_JSON`, `LAB_SMI`, `LAB_LOCK`.

## The three pitfalls it encodes

1. **MSYS paths handed to native Windows Python.**
   `python - "$OUT/..."` with `$OUT=/h/exl3-lab/...` is NOT absolute for the Windows
   interpreter: it resolves to `<current-drive>\h\...` and `json.dump`/`open(...,'w')` dies
   with `FileNotFoundError`. Worse, a *read* can silently succeed against a stale file while
   the write goes elsewhere, so the run looks fine and the numbers are old.
   It looks like a GPU bug because the failure surfaces mid-bench as a missing/empty JSON
   next to a healthy server log. Fix: `win_path` (one `cygpath -m`) before every python block;
   the probes do it internally, and one-off blocks must do it at the call site.

2. **`$!` is not a Windows PID under MSYS.**
   `taskkill /PID $!` and even bash `kill $PID` on a spawned `llama-server.exe` silently kill
   nothing. The 13 GB server keeps holding the card, the next case OOMs or the deployment
   restore never answers 200 -- while `nvidia-smi` (if nobody runs it) makes the machine look
   idle, so the session blames VRAM, the model, or the driver instead of its own dead kill.
   Fix: `kill_servers` kills by image name (`taskkill /IM llama-server.exe /F`) because the
   bench owns the card exclusively, then `wait_gpu_free` polls the driver until the card is
   actually free. Never gate on a PID; always confirm with `nvidia-smi`.

3. **Restore is not idempotent.**
   Every old script unconditionally started the deployment at the end, so two overlapping
   sessions (or a retry after a partial restore) could start TWO servers on one 16 GB card:
   instant OOM, and the `/health` that answers may belong to the wrong server. It looks like
   flaky hardware because each run is fine in isolation and only overlapping runs die.
   Fix: `restore_deployment` first checks `8081/health` and starts the server only when it is
   NOT 200 -- safe under any interleaving, and safe to call twice in a row.

## Verification band (the number a refactor must reproduce)

MTP k=3 512-token greedy decode probe, deployment flags: **32.66-34.44 tok/s**,
acceptance (`draft_n_accepted/draft_n`) **0.66667-0.68327** depending on the run.
A refactor that lands outside this band changed the measurement: fix the harness, not the flags.
After every GPU run the deployment must be back on 8081 with `/health` 200.
