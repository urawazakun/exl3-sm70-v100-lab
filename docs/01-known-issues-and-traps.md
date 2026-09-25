# KNOWN ISSUES — lab harness

## KI-5 (fixed): killing a session orphaned its llama-server and left the deployment down

**Measured 2026-09-25 (twice).** A session stops the 8081 deployment to take the GPU for a benchmark —
correct per its brief — but if the session is killed before its restore step runs, its `llama-server`
**survives as an orphan**, keeps ~13.5 GB of VRAM and the deployment stays at `health=000`. The second
occurrence happened when brief 19B was stopped after the API-limit/priority decision: an orphan
`worktrees/fixA/build-fixA/bin/llama-server.exe` (pid 184) held 13,495 MiB.

**Fix:** `scripts/kill-sessions-safely.sh <pid> [...]` does the whole sweep — kill the session tree → kill
any orphan `llama-server` → release a stale GPU lock → wait for the GPU to drain → restore the deployment
with a hidden window (KI-4) → verify `health=200` and print the final VRAM/server count. **Use it instead of
a bare `taskkill`.** `deploy-watchdog.sh` (patched to stay on watch instead of exiting when healthy) also
restores the deployment whenever the lock is FREE, no server runs, and 8081 has been down for 5 minutes.

## KI-5b (open): the watchdog does not rescue the deployment when an orphan holds a DIFFERENT port

**Measured 2026-09-25 (3rd occurrence of the KI-5 family).** #21B finished with the lab server still healthy on
**8082** while **8081 was refused**, and its session was blocked from `taskkill`ing it. `deploy-watchdog.sh`
did not restore 8081 because its restore condition includes "no llama-server process runs at all" — the 8082
orphan failed that test. **Interim rule:** when the lock is FREE and 8081 has been down for 5 minutes, kill
any `llama-server.exe` (nothing legitimate can be holding the GPU without the lock) and then restore. Sessions
cannot be relied on for the restore step: their own tool policy may block `taskkill`, so **the assistant or the
watchdog owns the restore**, and every brief that stops the deployment must also drop a marker file
`logs/RESTORE-NEEDED` that the restore path deletes.

## KI-8 (fixed): every session's SASS step has been failing — CUDA 13.1's `nvdisasm` refuses sm_70

**Measured 2026-09-25.** `nvdisasm.exe` (and therefore `cuobjdump -sass`, which calls it) from
`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin` answers
`CUDA architecture "SM70" is deprecated. Please use CUDA toolkit 13.0 or earlier` — **so no SASS of our
sm_70 kernels can be produced with the installed toolchain.** Earlier sessions hit this and moved on
silently: `logs/brief21/nvdisasm-23.err`, `logsbrief21/nvdisasm-k6.err`, `logs/brief21b/nvdisasm-stock.err`
and brief 26's `sass-mma8-fixed.err` are all the same error. A working **CUDA 11.8 `nvdisasm` already exists
on this box**, extracted by an earlier session:

```
ND="/c/llm-local/cuda118-extract/cuda_nvdisasm/nvdisasm/bin/nvdisasm.exe"
MSYS_NO_PATHCONV=1 "$ND" "H:/exl3-lab/logs/brief26/deployed-cubins/ggml-cuda.23.sm_70.cubin" > whole.sass
```

Notes that cost this lab several detours: the local build toolchain lives in a **stub** dir
`C:/llm-local/cuda-11.8/bin` that contains only `nvcc.exe`/`nvcc.profile` (no disassembler); `nvdisasm -fun`
takes a **function index**, not a mangled name (that is `cuobjdump -fun`); and the whole-cubin dump must be
split on `Function : <mangled>` lines before counting anything. `cuobjdump -res-usage` works fine and gives
the per-kernel register/shared numbers (deployed cubin 23: `exl3_mma8_splitk_kernel<0|1|2>` REG 96/95/95 with
SHARED 768 B, `exl3_gemm_id_kernel<8,*>` REG 72, `<7|6,*>` REG 91, `<5,*>` REG 77).

## KI-6 (fixed): parallel consultations overwrote each other's answer

**Measured 2026-09-25.** `claude-advisor.sh` named its outputs `advisor-<YYYYmmdd-HHMMSS>.md`/`.json`. Five
audits launched in parallel finished inside the same second, so four of five answers were overwritten and two
(`unparseable answer` on top of the collision) were unrecoverable. **Fix:** the stamp now includes `$$` and a
slug of the question file name, and the caller should still copy each run's `.md` aside immediately when
running sequentially. **Rule: never run two advisor consultations concurrently anymore — the script tolerates
it now, but the answers are too expensive to lose.**

## KI-7 (fixed): chained launcher hung, and a killed advisor left 11 orphan `claude.exe`

**Measured 2026-09-25.** A background launcher that chained `taskkill` → wait-for-brief → `git worktree add`
→ two `muse-dev.sh` calls never started either session (the `git worktree add` almost certainly blocked on a
git lock held by a session that was already running in the same repo), and separately, killing the advisor
wrapper left **11 orphaned `claude.exe`** processes burning quota until they were force-killed.
**Rules:** create worktrees **before** launching anything and never concurrently with a running session that
uses git; kill advisor runs by process name (`Get-Process claude | Stop-Process -Force`) and verify the count
is 0; verify a launcher actually started by checking its console log exists within ~30 s.

## 測定と推論の規律（2026-09-25、外部レビューで判明した我々の誤り3件の再発防止）

1. **Volta に uniform datapath は無い**。uniform register/datapath は **Turing (sm_75) 以降**の機能で、
   sm_70 では `UR` オペランドが構造的に存在しない。**SASS で UR=0 なのは正常**であり、
   「未活用資産」として追ってはならない（以前のブリーフがこの誤りを含んでいた）。
2. **命令数を wall-clock の代理にしない**。Volta は FP32 と INT32 を**別パイプで並行発行**するので、
   LOP3/SEL/ISETP が命令の56%でも実行時間の56%とは限らない。**判定はカーネル実時間（CUDA event /
   engine の print_timing）で行う**。
3. **Amdahl の f を測る前に E2E 倍率を語らない**。例: 46.7→30 ms/token には対象部分が約83%必要、
   46.7→25 ms は f=100% でも到達不能（理想 ≈26.7 ms）。命令数比 1.75× から 1.6〜1.9× を導くのは不可。
4. **この箱ではプロファイラが使えない**: `ncu 2025.4.0` は *"Skipping unsupported chip GV100"* で拒否、
   `nsys` 未導入（brief 18 の実測）。stall 系カウンタ（long scoreboard / math-pipe throttle / MIO
   throttle / bank conflict / spill）は**取得不能**。使えるのは CUDA event、engine counters、
   逆アセンブリの静的カウント、wall-clock A/B のみ。
5. **SASS の静的カウントは「同一カーネル・同一PC範囲・同一フラグ」でしか比較しない**。
   別 kernel が混ざった PC 範囲の比較（baseline 11.22 vs fixA 19.31 のような逆転）は無効。
6. **採用基準はカーネル実時間**（例: K3 projection 群の合計が何%下がったか）。命令数や UR 数は
   診断材料であって採用根拠にしない。

## KI-4 (fixed): a session's `cmd /c start` popped a Windows dialog on the operator's screen

**Measured 2026-09-25 05:31.** Muse session `20260925_050419_4f87dc` launched its bench server with

```
cmd.exe /c start /min "brief24b" "H:\exl3-lab\logs\brief24\start-8082.bat" baseline
```

`start` treats the **first quoted argument as the window title**; the quoting did not survive the layers
in between, so Windows tried to execute a program literally named `brief24b` and showed
*"Windows cannot find 'brief24b'"* on the operator's desktop. Nothing was broken (GPU idle, no
llama-server, all sessions alive) — but a GUI popup steals the screen, and the operator's standing rule is
that background work must be silent.

**Rule for briefs and for the agent:** never use `cmd /c start` to launch a background process. Use
`MSYS_NO_PATHCONV=1 powershell.exe -NoProfile -WindowStyle Hidden -Command "Start-Process ... -WindowStyle Hidden"`
(or plain `nohup ... &` / the fork's own `start-lab-server` helper from `lib-bench.sh`, which is already
silent). If a dialog titled with a strange name appears on the operator's screen, look it up in the running
session's transcript first (the `messages.tool_calls` column of `profiles/*/state.db` records every command)
— it is almost always a launch/quoting bug, not an OS fault.

## KI-3 (fixed): a consultation that hits the turn cap returns an empty answer and loses all its work

**Measured 2026-09-25 04:54-04:58.** The "hunt for other finds" consultation returned **no answer at all**:
the saved `.md` had `## Answer` followed by nothing. The JSON explained it:

```
subtype            : error_max_turns      is_error : True      num_turns : 17
stop_reason        : tool_use            (i.e. it wanted one more tool call when the loop was cut)
permission_denials : [Bash(git ls-tree --name-only HEAD engine/src/models/ | grep ...)]
```

Two harness faults, both mine, not the model's:

1. `--max-turns` was hardcoded to **16** while the question had six sub-parts that required reading the
   fork *and* searching the web. The run burned **17 turns and USD 1.76** and produced nothing.
2. `git` (read-only) was **not** in the allowlist, so "is this in upstream, which commit?" — an explicitly
   asked question — was denied and the model had to route around it, wasting turns.

**Fixes applied:** `claude-advisor.sh` now takes `max_turns` as its 4th argument (default **40**), allows
`Bash(git log|show|diff|ls-tree|rev-parse|status *)` and `Bash(find *)`, and the script header records this
failure. Consultation questions now also state an explicit tool-call budget ("at most N tool calls, then
answer; mark unchecked rows 'not checked' — an incomplete table is valid, an empty answer is not") and
multi-part research questions are **split into separate consultations** (A: artifacts/runtime support,
B: techniques/dead ends).

**Rule:** never treat a consultation as done until the saved `.md` contains a non-empty answer; check
`subtype` in the JSON (`error_max_turns` = the question was too big, split it and raise the cap).

## KI-2 (mitigated): a lab session took the GPU while a probe was in flight, cutting it in half

**Measured 2026-09-25 ~04:41.** A probe run against the live deployment was in progress when session
`20260925_040011_6f4794` (brief 16's `run16d.sh`) began a new benchmark: it stopped the deployment on
8081 (its log ends abruptly mid-prefill, ~04:41:39, no error line → a kill, not a crash) and started its
own server on 8082 at 04:41:53. The in-flight probe died with `WinError 10054` (connection reset) and the
following ones with `10061` (refused). The session did hold the GPU lock from 04:41:54, i.e. it followed
the brief's protocol — the *probe* was the one running without the lock.

**This was first misread as "the deployment crashes on a 32k prompt". It does not.** Evidence against a
config bug: no error/assert/OOM line anywhere in `server-exl3.err.log` or the stdout log, and the lab
build with the same `-c 131072` handled 8k/23k/71k prompts repeatedly minutes earlier.

**Fix:** `scripts/run-deploy-probes.sh` now (a) waits until `gpu-lock.sh status` is `FREE`, (b) acquires
the lock itself, (c) waits for `/health`=200, (d) runs the probes, (e) releases. It also drops a
`logs/deploy-probes/RUNNING` marker while it holds the GPU, so a session that wants the GPU can see that
the deployment is in use.

**Rule for every future brief:** before stopping the 8081 deployment to take the GPU, check for
`logs/deploy-probes/RUNNING` (and for a held lock you did not take) — if present, wait, do not kill it.

## KI-1 (open, high): the "byte-identical greedy" gate compared empty strings in every A/B run

**Measured 2026-09-25.** In every response captured by the lab harness, the `content` field is **0
characters long** — the model's entire output lands in `reasoning_content` (this fork serves the
reasoning channel separately) and the request's `max_tokens` (512, from `bench/prompt.json`) is consumed
by the thinking phase, so `content` is empty and `finish_reason` is `length`.

Evidence, from the raw responses in the log directories:

| run | file | content | reasoning | sha256(content) | tok/s |
|---|---|---:|---:|---|---:|
| DFlash2 A/B | `logs/brief10/run-20260925-015403/mtp_k3.json` | 0 | 1657 | `e3b0c44298fc…` | 34.44 |
| DFlash2 A/B | `…/none.json` | 0 | 1816 | `e3b0c44298fc…` | 21.42 |
| DFlash2 A/B | `…/dflash_5.json` | 0 | 1875 | `e3b0c44298fc…` | 20.73 |
| split-K | `logs/brief14/baseline.decode.json` | 0 | 1657 | `e3b0c44298fc…` | 34.77 |
| split-K | `logs/brief14/split480.decode.json` | 0 | 1995 | `e3b0c44298fc…` | 29.83 |
| split-K | `logs/brief14/split960.decode.json` | 0 | 1821 | `e3b0c44298fc…` | 35.43 |

`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` is the SHA-256 of the **empty string**
(verified locally). So:

* **Invalidated:** every "greedy output is byte-identical" / "byte-identity regression gate" statement in
  `LAB_REPORT.md` (split-K sweep, DFlash2 A/B, axes, prefill). They were comparisons of empty strings.
* **Still valid:** all tok/s, acceptance rate, mean committed length and VRAM figures — those come from
  engine counters (`timings`, `print_timing`) and are independent of this bug.
* **Additional finding:** the `reasoning` lengths *differ* between configurations (1657 vs 1816 vs 1821 vs
  1995), i.e. the real streams were not identical either. Any future identity check must hash
  `content + reasoning_content`, and greedy equivalence between a spec-decode config and a plain config is
  not guaranteed anyway (batch shape / kernel differ, so FP rounding differs — see the advisor note that
  byte-identity is the wrong gate for kernels that change summation order).

**Fix (applied 2026-09-25):**

1. `scripts/probe3.py` — posts with `max_tokens=1024`, searches **both** `content` and
   `reasoning_content`, prints the head of both channels and the timings. Use this for any probe that must
   produce a *usable answer*.
2. `bench/prompt-nothink.json` — the same fixed prompt text with **`/no_think`** appended (Qwen3 honours it)
   and `max_tokens=256`, so the model answers directly and `content` is non-empty. **All new quality /
   identity gates must use this prompt**; keep `bench/prompt.json` only for tok/s comparability with the
   older runs.
3. Rule for every future brief: **a gate is only a gate if the compared payload is provably non-empty.**
   Report the payload length and its hash next to the verdict — `sha256(content)=e3b0c442…` must be treated
   as "no data", never as "identical".
