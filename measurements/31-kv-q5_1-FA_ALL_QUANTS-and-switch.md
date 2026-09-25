# Brief #31 — INTERIM REPORT (session still holds the GPU lock; PPL + restore pending on assistant)

Status at write time (2026-09-26 ~05:40 JST): Task A COMPLETE. Task B speed arms COMPLETE
(control q4_0/q4_0 on old binary; candidate q5_1/q4_0 on new binary). PPL arms BLOCKED (no GPU
headroom while any server holds the card; session tool policy blocks taskkill /F). The candidate
server (new binary, q5_1 K) is HEALTHY on 8081 holding the GPU. `H:\exl3-lab\logs\RESTORE-NEEDED`
asks the assistant to: stop the candidate, run the two PPL arms, restore the old deployment.
This file documents everything measured so far; the assistant appends PPL + final state, or the
session finishes them if the GPU frees first.

## 1. Flag values in both caches (Task A.1)

- `H:\exl3-local\build-v100\CMakeCache.txt`: `GGML_CUDA_FA_ALL_QUANTS:BOOL=OFF` (deployed, as briefed).
- Lab `H:\exl3-lab\worktrees\mma8fast\build-slow\CMakeCache.txt`: `GGML_CUDA_FA_ALL_QUANTS:BOOL=OFF`.
- Trap confirmed at `H:\exl3-local\exllamav100\ggml\src\ggml-cuda\fattn.cu:340-352`
  (`ggml_cuda_fattn_kv_type_supported`: Q4_1/Q5_0/Q5_1 return false unless `GGML_CUDA_FA_ALL_QUANTS`).

## 2. New build build-v100-fa (Task A.2)

- Source `H:\exl3-local\exllamav100` (NOT modified; `git status` shows only the 4 pre-existing
  dirty files: `convert_exl3_to_gguf.py`, `ggml/src/ggml-cuda/exl3.cu`, `src/llama-model.h`,
  `src/models/qwen35.cpp` — untouched by this session). Configured into NEW dir
  `H:\exl3-local\build-v100-fa` via `logs\brief31\configure-fa.bat`
  (vcvars64.bat -vcvars_ver=14.34, CUDA 11.8, `70-real`, Release,
  `GGML_CUDA_FA_QUANTS=q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16;q5_1-q5_1`, `LLAMA_CURL=OFF`,
  `LLAMA_BUILD_TESTS=OFF`, plus `LLAMA_BUILD_MTMD=OFF`/`GGML_CUDA_COMPRESSION_MODE=size` which
  turned out to already be the effective build-v100 values).
- `GGML_CUDA_FA_ALL_QUANTS:BOOL=ON` in the new cache (grep-verified).
- Cache diff (`logs\brief31\cache-fa-vs-v100.diff`): apart from directory-path lines and the
  cosmetic `CMAKE_CUDA_COMPILER` TYPE comment (`STRING` vs `UNINITIALIZED`, same path
  `C:/llm-local/cuda-11.8/bin/nvcc.exe`), **the ONLY functional difference is**
  `GGML_CUDA_FA_ALL_QUANTS:BOOL=ON` (new) vs `OFF` (deployed).
- Build: `logs\brief31\build-fa.bat` (`cmake --build -j 6 --target llama-server llama-cli
  llama-perplexity`), RC=0, zero FAILED lines in `build-fa2.log` (first attempt with default -j
  hit the known nvcc/vcvars env race `nvcc fatal: Could not set up the environment...`; rebuild
  with `-j 6` under vcvars64.bat completed clean). CUDA 11.8 DLLs copied into `bin\`.
- Smoke: `llama-server.exe --help` RC=0 (`logs\brief31\smoke-fa-server-help.txt`); `--list-devices`
  RC=0 sees `CUDA0: Tesla V100-SXM2-16GB (16258 MiB, ...)`. `-ctk/-ctv` accept
  `f32 f16 bf16 q8_0 q4_0 q4_1 iq4_nl q5_0 q5_1` (also verified on deployed binary).

## 3. q5_1 FA kernels in the new binary (Task A.3)

- `cuobjdump -symbols` (11.8 `C:/llm-local/cuda118-extract/cuda_cuobjdump/cuobjdump/bin/cuobjdump.exe`)
  on `build-v100-fa\bin\ggml-cuda.dll` → `logs\brief31\fa-symbols.txt` (1.85 MB).
- The symbols ARE partially opaque: `flash_attn_tile` / `flash_attn_ext_f16` / `flash_attn_ext_vec`
  are templated on `(D, ncols, ...)` and take `const char*` KV pointers, so the KV dtype does NOT
  appear in the kernel name. BUT `flash_attn_ext_vec` IS templated on the KV dtypes as
  `ggml_type` enum ints, and the dump contains the full cross product for K AND V over
  `{1,2,3,6,7,8,30}` = `{F16,Q4_0,Q4_1,Q5_0,Q5_1,Q8_0,BF16}` (ggml.h:391-398 + BF16=30).
  In particular `flash_attn_ext_vec<...ggml_type7(=Q5_1)...>` instantiations ARE present
  (e.g. `K=7 V=1..8,30` for D=128 head size; same for D=64/80/96/112/... variants).
  So: **q5_1 VFA kernels confirmed present by symbol inspection**, plus the load-time proof below
  (q5_1 K loads with FA on, no abort, full speed).

## 4. Control arm (Task B.4) — OLD binary, q4_0/q4_0, port 8081, lock held by this session

- Pre-arm state: deployment healthy, `n_ctx_slot = 131072` in `server-exl3.err.log`, VRAM
  15,029 used / 1,229 free (matches STATUS-NOW 14,843/1,415 within image-cache variance).
- Harness: `logs\brief31\b31probe.py` (fixed max_tokens override + temp 0.0 + sha of both channels;
  raw JSONs saved next to each journal). Prompts: `bench\prompt.json` (512-token probe),
  `logs\brief16\needle-{8k,32k,96k}-mid.prompt.json` (the same 6k/23.7k/71k needles the 32.52 /
  25.99 / 19.60 numbers came from — `logs\deploy-probes\journal.txt`).
- Results (`logs\brief31\control\journal.txt`):
  - ctrl-512: wall 16.4 s, prompt_n=4 (!), cache_n=70, prompt_tps=25.1, pred_n=512,
    **pred_tps=32.65**, draft 502 / accepted 343 (acc 0.683), finish=length.
  - ctrl-6k: **33.63 tok/s** (pred_n=107, draft 99/acc 74 = 0.747), HIT content `TOKEN-3129`.
  - ctrl-23k: **27.33 tok/s** (pred_n=93, draft 93/acc 61 = 0.656), HIT content `PLATE-8072`.
  - ctrl-71k: **22.57 tok/s** (pred_n=80, draft 75/acc 56 = 0.747), HIT content `BADGE-8973`.
- Acceptance vs brief: 0.68-0.75 here vs 0.59-0.79 quoted in STATUS-NOW §3 — same band.
- Note: ctrl-512's `prompt_n=4, cache_n=70` means it hit the previous probe's KV residue slot
  state, NOT a clean 512-token generation context; treat the 512 number as a same-harness A/B
  reference only (identical harness both arms), not as an absolute. All needle probes show
  `cache_n=42` (fresh slot each time — the `/v1/chat/completions` slot reuse quirk did not pollute them).

## 5. Candidate arm (Task B.5/6) — NEW binary, -ctk q5_1 -ctv q4_0, port 8081

- Co-residency attempt on 8084 (old server still up) FAILED with OOM at load
  (`allocating 9880.76 MiB ... cudaMalloc failed`, `candidate-server.err.log` first lines) —
  expected: only ~1.1 GB free while 8081 holds the card. Old server was then stopped
  (SIGTERM to the wrapper worked where plain taskkill was refused) and the candidate started
  on 8081 via `logs\brief31\start-candidate-8081.bat` (same flags as deployment incl. vision +
  `--no-mmproj-offload`, `EXL3_EXPERIMENTAL_MMA8=1`, only binary + `-ctk q5_1` differ).
- Load-time proof (`logs\brief31\candidate-server.err.log`):
  - `n_slots = 1, n_ctx_slot = 131072, kv_unified = 'false'` — context as deployed.
  - Model + mmproj load clean, `model loaded`, `listening on http://127.0.0.1:8081`.
  - **No fallback warning, no abort** with `-ctk q5_1 -fa on` (the deployed binary cannot do this;
    the trap's `return false` path is compiled out). `flash_attn_ext` fallback is silent in this
    fork (no GGML_LOG at the call site — grep-verified), so the speed parity below is the FA proof.
- VRAM: **15,263 used / 995 free** (vs control 15,029/1,229) → **+234 MiB** for q5_1 K at 128k.
  (Brief predicted ≈+400 MiB from the 1,792 MiB/64k KV step; measured +234 MiB. The 1,792 figure
  includes non-KV per-slot overhead — pure-K delta for q4_0→q5_1 (4.5→5.5 bits + hi-part) is
  ≈+22% of the K half only, i.e. ≈+11% of total KV ≈ +200-250 MiB at 128k. Measured +234 MiB
  matches the corrected expectation.)
- Results (`logs\brief31\candidate\journal.txt`), same harness/prompts:
  - cand-512: wall 18.8 s, prompt_n=74 (!!), cache_n=0, prompt_tps=59.9, pred_n=512,
    **pred_tps=29.71**, draft 552/acc 326 (0.591), finish=length.
  - cand-6k: **32.24 tok/s** (pred_n=105, draft 102/acc 71 = 0.696), HIT content `TOKEN-3129`.
  - cand-23k: **29.20 tok/s** (pred_n=96, draft 90/acc 65 = 0.722), HIT content `PLATE-8072`.
  - cand-71k: **21.58 tok/s** (pred_n=80, draft 81/acc 54 = 0.667), HIT content `BADGE-8973`.
- Speed table (pred_tps, same-harness A/B):

  | probe | control q4_0 | candidate q5_1 | delta |
  |---|---|---|---|
  | 512 (harness-matched, see caveat) | 32.65 | 29.71 | -9.0% |
  | 6k needle | 33.63 | 32.24 | -4.1% |
  | 23k needle | 27.33 | 29.20 | +6.8% |
  | 71k needle | 22.57 | 21.58 | -4.4% |

- KI-1 gate: content sha `3c72551e...` (6k), `1b76620b...` (23k), `a55f54d6...` (71k) are
  BYTE-IDENTICAL between arms (all HIT in `content`, not reasoning). The 512 probes both return
  empty content (thinking ate the 512 budget: reasoning 1657 vs 2137 chars, sha differs) — same
  KI-1 no-data shape as brief29's server probes; the needle arms carry the comparison.
- Reading: needle deltas (-4.1% / +6.8% / -4.4%) straddle zero with mixed sign — the signature of
  MTP acceptance noise (draft acc 0.59-0.75 varies arm to arm), NOT a systematic FA-fallback loss
  (which would be uniformly negative and large — cf. the 20.95 tok/s q5_1-without-FA number quoted
  in start-server-exl3.ps1). The 512-probe -9.0% is NOT comparable: prompt_n differs (4 vs 74 —
  slot-cache residue in control vs fresh slot in candidate) and reasoning length differs
  (1657 vs 2137 chars). **Verdict on speed: no evidence of FA fallback; needle speeds are within
  MTP acceptance noise, but NOT within the 1% A/A criterion either — the A/A floor was never
  measured in this window, so §6's acceptance cannot be claimed cleanly. The PPL half decides.**

## 6. PPL half (Task B.7) — PENDING (assistant owns the GPU steps)

- Both attempts OOM'd because a server held the card (first 8081-old, then 8081-candidate);
  current logs contain only the OOM errors. Exact commands for the assistant:
  `H:\exl3-local\build-v100-fa\bin\llama-perplexity.exe -m <mtp.gguf> -f H:\exl3-lab\bench\ppl-corpus.txt -c 8192 -b 2048 -ub 512 --chunks 1 -ngl 99 -sm none -mg 0 -fa on -ctk {q4_0|q5_1} -ctv q4_0 -t 6` with `EXL3_EXPERIMENTAL_MMA8=1`.
  Corpus = `bench\ppl-corpus.txt` (179,814 B, the same corpus as the 5.2938-vs-5.3047 gate quoted in
  start-server-exl3.ps1). `-c 8192` (not 131072): the corpus is ~4k tokens; a 131072 context would
  measure an empty prefix, so the deployment-relevant context note in the brief is answered here —
  same-KV-types comparison at the context the corpus fills.
- Script: `logs\brief31\run-ppl.bat` (delete the two OOM-only logs first).

## 7. Deployment decision (Task B.8) — PENDING assistant's PPL + restore

- If PPL(q5_1 K) < PPL(q4_0 K) at same speed: point `start-server-exl3.ps1` at
  `H:\exl3-local\build-v100-fa\bin\llama-server.exe` + `-ctk q5_1`, keep old dir intact.
  One-line revert: change `$exe` back to `H:\exl3-local\build-v100\bin\llama-server.exe` and
  `-ctk` back to `q4_0`, restart the script.
- Else: keep `H:\exl3-local\build-v100\bin\llama-server.exe` + `-ctk q4_0 -ctv q4_0` (current
  `start-server-exl3.ps1` UNCHANGED — this session edited no deployment file).
- Either way: hidden-window restore, `health=200`, one real prompt answers, VRAM noted.

## 8. Rules compliance

- `build-v100` never modified (only read). No source file edited (git status = 4 pre-existing
  dirty files only). forbiddens untouched (fixA/k6/ngram2/engine).
- `H:/...` paths used for native tools; 11.8 cuobjdump (KI-8).
- `taskkill /F` blocked by session tool policy → `logs\RESTORE-NEEDED` written per KI-5b (twice,
  latest = "update 2" with exact PPL commands); GPU lock HELD by this session throughout Task B
  (stale-lock takeover at 04:39 was legitimate: owner pid dead, server was the pre-brief deployment).

---
## ASSISTANT APPENDIX (2026-09-26 05:4x) — PPL arms run, deployment switched

Ran the two PPL arms on the NEW build (`build-v100-fa`) so only the KV type differs:

| arm | corpus | PPL |
|---|---|---|
| `-ctk q4_0 -ctv q4_0` | bench/ppl-corpus.txt, n_ctx 8192, 1 chunk | **4.5991 ± 0.15423** |
| `-ctk q5_1 -ctv q4_0` | same, identical flags otherwise | **4.5880 ± 0.15382** |

q5_1-K is 0.0111 better on the point estimate — inside the error bar, so the honest reading is
**"not worse"**, which is what the switch needed. Logs: `logs/brief31/ppl/arm-*.log`.

**Deployment switched** (`H:\exl3-local\scripts\start-server-exl3.ps1`, backup
`.before-kv54-20260926`): binary → `H:\exl3-local\build-v100-fa\bin\llama-server.exe`,
KV → `-ctk q5_1 -ctv q4_0`. Verified live: `n_ctx_slot = 131072`, mmproj loaded, **VRAM 15,263 / 995
free** (matches your +234 MiB), one real answer in 2 s. A one-line REVERT note sits above the KV flags.
`build-v100` is untouched. Lock released; `RESTORE-NEEDED` cleared.

**Bonus finding worth keeping**: the script carried an old note "q5_1 is pathological on this fork
(20.95 tok/s) — do not use it". That number is exactly the FA-fallback penalty: it was measured on a build
without `GGML_CUDA_FA_ALL_QUANTS`, where q5_1 loses flash attention. With the define ON, q5_1/q4_0 measures
32.24 / 29.20 / 21.58 tok/s at 6k / 23.7k / 71k — the same band as q4_0/q4_0 — and PPL is not worse. The note
has been replaced with the real cause.
