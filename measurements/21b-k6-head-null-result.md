# Brief 21B — K=6 specialized decode path (re-run in the k6 worktree)

Re-run of brief 21 (whose shared-tree edits were reverted by a parallel session).
Source of truth: `H:/exl3-lab/worktrees/k6` (HEAD `07928d8`, the baseline all
published numbers come from). Hint (not gospel):
`H:/exl3-lab/logs/brief23/exl3cu-k6fast.diff`. Final diff lives in the k6
worktree: `engine/ggml/src/ggml-cuda/exl3.cu` (+113/-6 vs `07928d8`) and
`engine/tools/server/server-context.cpp` (+71, dispatch/phase logging only).
Archived copies: `exl3.21b-k6.cu`, `server-context.21b-k6.cpp` in this directory.
Switch `EXL3_FAST_K6` is OFF by default (both paths in one binary).

## 1. Dispatch decision today (instrumented first)

`EXL3_LOG_DISPATCH=1` logs one line per tensor (shape, K, cb, kernel, extract).
Standard 512-token greedy probe, deployed flags (`EXL3_EXPERIMENTAL_MMA8=1`,
`-c 65536`, KV q4_0, MTP k=3), 14207 dispatch lines per case:

| K | cb | kernel | extract | rows | count |
|---|---|--------|---------|------|-------|
| 3 | 2 | gemm/splitk | specialized | 2/4/28/32/42 | 7200 |
| 4 | 2 | gemm/splitk | specialized | 1/2/4/28/32/42 | 6456 |
| 6 | 2 | gemv | generic-wi0wi1 (base) / k6fast (fast) | 1/2/4 | 551 |

The ONLY generic-path tensor is the 6-bit output projection
(`output.trellis`, K=6 cb=2, 248320x5120 = 0.954 GB per full read):
540x rows=1 gemv (draft steps) + 1x rows=2 + 10x rows=4 gemv per 512-token run.
Full tables: `bench21b.journal.txt` lines 17-40 (base) and 56-79 (k6fast).

Baseline (base, `EXL3_FAST_K6=0`): 15.78 tok/s decode, acceptance
0.61524 (331/538), mean committed ~2.46; spec phases over 180 rounds:
draft mean 11.01 ms vs verify mean 4.00 ms (totals 1981.1/720.3 ms).

## 2. The change

K=6 closed form, re-derived in this tree (`k6_equiv.py`, 0/256 value +
0/256 index mismatches): lane L's 8 windows touch words {q-1,q,q+1},
q = lane*3/2; 3 shared loads per tile, `wi0/wi1/wsh` tables skipped
(`if constexpr (!K6FAST)`), behind `EXL3_FAST_K6=1`. GEMV (rows=1 draft)
and split-k (rows 2..8 verify) both covered; GEMM rows>8 untouched
(dispatch-only logging there).

## 3. Evidence

SASS (brief-8 method: `cuobjdump -xelf all` + `nvdisasm -c`, H:/ paths;
full bodies `sass-stock-23.txt` / `sass-k6fast-23.txt`, tallies
`tally-*.txt`, delta `sass-delta.txt`):

| kernel (K=6 cb=2) | stock generic | k6fast | delta |
|---|---|---|---|
| splitk body | 1364 static (21.31/64w, regs 112) | 1196 (18.69/64w, regs 108) | -168 (-12.3%) |
| gemv body | 1626 (25.41/64w, regs 98) | 1515 (23.67/64w, regs 82) | -111 (-6.8%) |

LDS 136->32 (splitk), 151->47 (gemv); SHF 115->83 / 121->78;
IMAD 245->212 / 318->285.

Timers (same binary, switch only; `b21b_ab.py`): decode 15.78 -> 15.16
tok/s (-3.9%); nothink 15.06 -> 14.45 (-4.0%); draft counts identical
(538/331, 258/169). The specialization does NOT pay at the round level:
the head is ~0.95 GB x ~4 reads/round, but the fused loop is LDG-bound
(LDG unchanged 9/9 splitk, 9->37 gemv) and the saved instructions are
address arithmetic the scheduler already hides.

## 4. Correctness gate

* Greedy streams byte-identical: `reasoning_content` len 1802
  sha `2e2cfd1d…` both paths (decode), len 853 sha `74de731a…` both paths
  (nothink). `content` is EMPTY in all four runs (len 0,
  sha `e3b0c442…` = empty string) — per KI-1 this is NO DATA, not a gate.
  The model thinks instead of answering (`/no_think` not honoured by this
  fork/model combo at these budgets; even max_tokens=32 leaves
  `content` empty — see `quality21b.journal.txt`).
* Retrieval probe (`b21b_retrieval.py`, deployment 8081 + base + k6fast):
  all three return identical non-thinking behaviour — `content` empty,
  `reasoning_content` len 165 sha `b1385bc2…` byte-identical across all
  three. No retrieval answer string exists to compare (KI-1 reported
  honestly: no non-empty payload could be produced).
* Per-token numeric check (`/v1/completions`, echo+logprobs, same prompt):
  top-1 token ` a` logprob `-0.1655101180076599` identical;
  `b21b_logdiff.py`: ndiff=0, max-abs-diff=0.0, max-rel-diff=0.0 over the
  top-5 vectors. The raw-JSON byte difference (byte 601) is the `created`
  timestamp only (`b21b_rawdiff.py`).

## Verdict: REJECT (adopt = no)

The change is numerically clean (host proof + zero logprob diff + identical
streams) but it does not speed up the round it targets (-3.9% decode,
within-run noise direction wrong, draft/verify means 11.01/4.00 ->
10.50/4.62 ms). The K=6 loop is memory-bound, not ALU-bound; removing
address arithmetic cannot help. Switch stays OFF (default). Null result
with instruction counts: a successful session per the brief.

## Repro

* Sources: k6 worktree diff (`git -C H:/exl3-lab/worktrees/k6 diff`), archived
  here as `exl3.21b-k6.cu` / `server-context.21b-k6.cpp`.
* Build: copy the 2 files into `H:/exl3-lab/engine`, `build21b-lab.bat`
  (targets `ggml-cuda llama-server` only), restore + `rebuild-stock.log`.
* Bench: `bench21b-cases.sh` (A/B), `bench21b-quality.sh` (retrieval),
  `bench21b-logprob.sh` (numeric), analysis `b21b_ab.py` /
  `b21b_dispatch.py` / `b21b_logdiff.py` / `b21b_rawdiff.py`.
* Build note: the k6 worktree at `07928d8` predates the `src/models/`
  split, so a private `build-k6/` dir cannot link (`llama-model.cpp`
  needs `models/models.h`; backfilling pulls unbounded version drift).
  Build in the shared lab tree instead (already configured); own the
  *sources*, not the build dir. `build-k6/` was abandoned
  (`build21b-k6fast*.log`); `build/` restored to stock (`rebuild-stock.log`
  BUILD_RC=0, dll timestamp matched pre-brief state).

## State left behind

* Shared `engine/` sources + `build/bin/` restored to pre-brief state
  (stock rebuild BUILD_RC=0). GPU lock FREE.
* LEFTOVER (operator note): a lab server (k6 binary, `EXL3_FAST_K6=1`
  process env) is listening on **8082** (health 200) holding the GPU;
  the live deployment on **8081 is NOT up** (connection refused at
  handover). `taskkill` is blocked in this session mode, so restoring
  8081 needs one operator command: restart
  `H:/exl3-local/scripts/start-server-exl3.ps1` (it self-kills :8082
  or start after `taskkill /IM llama-server.exe /F`).
