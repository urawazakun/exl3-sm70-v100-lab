# Consultation to Claude Code (opus, effort high) -- 20260925-191426

Question asked:
```
# Consultation (3) — you are now directing the optimization campaign. Tell us what to do next.

You have read-only access to `H:\exl3-lab` (lab) and `H:\exl3-local` (deployment), plus web search/fetch.
**Budget: at most 20 tool calls, then write the answer.** An incomplete but concrete plan beats a
beautiful empty one (a previous consultation died on a turn cap and returned nothing — see
`KNOWN-ISSUES.md` KI-3).

## Your own critique has been adopted — three of my claims were wrong

1. **The 1.6-1.9× end-to-end projection was unsupported (Amdahl).** It came from an earlier advisor
   estimate (46.7 → 25-30 ms/token) resting on the 17.27 → 9.86 instructions/weight ratio, with **f never
   measured**. Your arithmetic is accepted: 46.7→30 ms needs f≈83%; 46.7→25 ms is unreachable even at
   f=100% (ideal ≈26.7 ms). The projection is withdrawn until f is measured.
2. **"Volta's uniform datapath is an unexploited asset" was wrong.** Uniform registers/datapath arrive with
   **Turing (sm_75)**; on sm_70 `UR` operands are structurally absent, so the UR=0 we measured is *normal*,
   not an opportunity. (I had it in a brief; correcting the record now.)
3. **Instruction count ≠ cycles on Volta.** FP32 and INT32 issue on separate pipes, so 56% LOP3/SEL/ISETP
   does not imply 56% of wall-clock. Treating the SASS mix as a performance model was too strong.
   Also accepted: 79 registers unchanged means **no occupancy gain** on that kernel; the 11.22 vs 19.31
   recount means **all** tallies (including 9.86 and the 17.27 baseline) are on hold until a same-kernel,
   same-PC-range, same-flags A/B is done.

## Constraints you did not have (these are measured facts, please plan around them)

* **No profiler is available.** `ncu` (2025.4.0) refuses this GPU with *"Skipping unsupported chip GV100"*
  and `nsys` is not installed. So stall-level counters (long scoreboard, math-pipe throttle, MIO throttle,
  bank conflicts, spills) **cannot be collected on this box**. Anything you recommend must be measurable
  with: CUDA events / engine `print_timing`, instruction counts from a disassembly, wall-clock A/Bs, and
  the fork's own counters (acceptance, mean committed, per-phase ms). If you insist on stall data, say
  what we would have to build or buy.
* **Our weights are not the public 3.5bpw file.** Deployed: `huihui-qwen3.8-27b-abliterated-exl3-3bpw-mtp.gguf`,
  **EXL3 3.0bpw, `exl3.codebook=2` (=mul1, measured 15.59 cycles/weight vs 21.70/22.08 for cb0/cb1),
  ~12.7 GB**, 128k context, KV q4_0, `-fa on`, MTP k=3. Measured: **plain decode 46.68 ms/token; MTP k=3
  34.77 tok/s / acceptance 0.683 / mean committed 3.04 / round 28.76 ms/token**; decode falls 36.1 → 19.6
  tok/s from 6k to 71k context; prefill 345 → 248 tok/s over the same range.
* **Hardware/OS**: single Tesla V100-SXM2-16GB, TCC mode, driver 472.12, **6 physical cores (HT off)**,
  127.8 GiB RAM (92.8 free), 1,415 MiB free VRAM at 128k context.
* **Quality and file size are locked.** Only *inference-side* changes (layout at load time, kernels,
  scheduling) are in scope; **no re-quantisation of the 27B** and no format change that alters outputs.
  Load-time prepacking is in scope (it reorders bytes, it does not re-quantise) — which is one of your
  recommendations and is back on the table.
* **Deployment availability**: the operator uses the 8081 server; long GPU jobs must take a lock and
  restore it. Two dev sessions run at once at most, in separate git worktrees (a shared tree already cost
  us a session's work).

## What is already running / available

* `worktrees/fixA` — the constant-shift extraction implementation (11 modified files, `EXL3_FAST_EXTRACT`
  ×3 in `exl3.cu`, binaries built: llama-bench/cli/perplexity). Its SASS tallies exist but conflict.
* `logs/brief19b/` — build logs, the cubin, `sass_tally.py`, `run19b.py`, a started `bench-off/` run.
* Fork capability notes: `--spec-type none|draft-mtp|draft-dflash|draft-eagle3|ngram-*|draft-dspark`
  (DSpark ported locally from upstream PR #25173; Q4 drafter built), `EXL3_EXPERIMENTAL_MMA8=1` (K==3 &&
  rows 4/8 mma8 path), `EXL3_FAST_K6` work starting in `worktrees/k6`.
* Closed with measurements (do not reopen): DFlash2, DSpark (acceptance 0.430 vs MTP 0.615, +68% round
  time, +1.95 GB VRAM), EAGLE3, CPU-resident drafters, n-gram/lookup (8 configurations, none beat MTP;
  free-form regressions −5…−19.5%), the jusko Volta attention fork (q8_0-KV-gated, not an EXL3 fork).
* `RESEARCH-prior-art-specdec.md` and `RESEARCH-sol-specdec.md` summarise the external evidence.

## What we need from you — the direction, not reassurance

1. **The measurement protocol.** Write the exact sequence for the next GPU session: how to fix
   same-kernel/same-PC-range/same-flags, how to get **kernel-level wall time** without ncu, and how to
   measure **f** (the share of the 46.7 ms token that the K3 projection kernels occupy) with tools we have.
   Include the commands if you can infer them from the harness (`scripts/`, `logs/brief19b/run19b.py`).
2. **Two candidates, one decision.** (a) finish the constant-shift extraction (in flight), (b) go to a
   **K3-specific prepacked layout** so the lane→word mapping is natural — plus (c) multi-row reuse of one
   decoded weight stream for MTP verification. Rank them **by expected wall-clock effect per hour of work
   on THIS box**, and state the *adoption criterion* (you suggested kernel wall-time, not instruction
   counts — give the threshold).
3. **What would make you abandon fix A?** Give falsification criteria up front.
4. **Prepack design, concretely.** If prepacking wins, specify the layout for the K=3 `mul1` trellis
   (where each lane's 8 windows live, how a warp reads them, how the 2-word funnel-shift becomes a fixed
   offset), what changes in the loader vs the kernel, and how to verify equivalence cheaply.
5. **Anything cheaper we are ignoring** for a 46.7 ms/token, 12.7 GB, 128k-context, single-user V100
   (e.g. things the recent V100-specific implementations do that we have not: persistent kernels, L2
   residency tricks, weight-stream reuse across rows, cp.async substitutes on sm_70, warp-specialised
   loaders). One line each, with the source.
```

## Answer

# Direction for the next cycle: move fix A's extraction into the M=4 verify kernel

## The key finding: fix A cannot speed up the deployed setup

The verify pass that dominates each MTP round does not run through the kernel fix A changes.

- **Deployment** (`H:\exl3-local\exllamav100\ggml\src\ggml-cuda\exl3.cu:1018`): at `K==3` with rows 4 or 8, verify uses `exl3_mma8_splitk_kernel` (`:427`). M=2..8 falls back to the `GROUP2` "dense2" split-K kernel.
- **fixA** (`worktrees/fixA/.../exl3.cu`): it branched from a lab tree that has **neither MMA8 nor GROUP2**. The fast path only changes the split-K GEMV, which in deployment runs at M=1 only.
- **Round budget.** One round is 28.76 × 3.04 ≈ **87 ms**. About 70 ms of that is M=4 verify through MMA8 (brief 14, `llama-bench -p 4` = 70.25 ms). About 17.5 ms is the three draft steps, and most of that is the K=6 head, which fix A doesn't touch either.
- **What fix A can reach in deployment:** only the MTP layer's own K=3 matmuls at M=1. That is about 1/64 of the body, so roughly 2 ms per round, **at most about 1–2% of tok/s**. It could only matter for plain decode, which is not what the operator runs.
- **`run19b.py` is also invalid as an A/B against deployment.** It sets `EXL3_EXPERIMENTAL_MMA8=1` on a binary that has no MMA8 code. Both arms therefore run the old one-row-at-a-time loop at M=4, and neither number can be compared with 34.77 tok/s.

Candidate (c), reusing one decoded weight stream across rows, **already exists**: MMA8 decodes each weight once for all 4 rows. What remains is making MMA8's decode cheap. Its inner loop does three expensive things:
- a generic `exl3_window<3>` call per weight (`:467`);
- a fp32 `xh` reload from global memory plus `__float2half` for every weight on every k-tile (`:476-480`);
- no prefetch, only 4 warps per block, and about 192 B in flight per warp (`:450-454`).

---

## 1. Measurement protocol for the next GPU session

**Step 0: get onto the right source (about 2 h, no GPU).**
- Create a new worktree, `worktrees/mma8fast`, and make its first commit "sync ggml-cuda/exl3.cu to deployed". Then diff the whole `ggml-cuda/` directory against `H:\exl3-local\exllamav100`. Until this is done, every A/B measures the wrong kernel.
- Replace the runtime `__constant__` flag with a **compile-time** macro, `-DEXL3_FAST_EXTRACT=0/1`. Build both variants from one `CMakeCache.txt` (copy it, then diff the two caches: identical flags). This removes the "both arms live" confound behind the 17.78 vs 11.22 vs 19.31 tallies.

**Step 1: same-kernel, same-range SASS (no GPU).**
- `cuobjdump -sass -fun <mangled name> ggml-cuda.*.sm_70.cubin` for the **deployed** kernels: `exl3_mma8_splitk_kernel<2>`, `exl3_gemv_splitk_kernel<3,2,false>` and `<6,2,*>`.
- Define the loop as running from the target of the backward `BRA` to that `BRA`, and normalise per weight decoded per lane (16 for MMA8, 64 for split-K).
- Report registers with `cuobjdump -res-usage`.
- Use this only as a diagnostic, never as the adoption test.

**Step 2: kernel-level wall time without ncu (about 3 h to build).**
- Add an `EXL3_KTIMING=1` path in `ggml_cuda_op_exl3_matmul`. It records a `cudaEventRecord` pair around each of `had_in`, the main kernel and the epilogue, and bins elapsed time by `(K, cb, rows, kernel)`. Accumulate at each graph-compute sync and print the table at exit.
- Run it with `GGML_CUDA_DISABLE_GRAPHS=1`. Also measure `tg` with timing off vs on, both with graphs disabled, and report that overhead.
- As a cross-check, add a **knock-out** switch, `EXL3_NULL_MATMUL=1`: skip all three kernels and `cudaMemsetAsync(dst)`.
- The knock-out is for `llama-bench` only. The output is garbage, so never use it with speculative decoding.

**Step 3: measure f (one lock, about 60–75 min).**
1. `bash scripts/gpu-lock.sh acquire 75`, then stop 8081.
2. **A/A noise:** deployed binary, `llama-bench ... -ctk q4_0 -ctv q4_0 -fa on -t 6 -n 128 -p 0 -r 5`, and `-p 4 -n 0 -r 5`, run twice each. The noise floor is the spread between the two runs.
3. Same commands with `EXL3_KTIMING=1 GGML_CUDA_DISABLE_GRAPHS=1`. From these:
   - f₁ = Σ EXL3 bins / tg ms-per-token;
   - f₄ = Σ bins / pp4 time.
4. Same commands with `EXL3_NULL_MATMUL=1`. Then f = 1 − t_null / t_base. If this disagrees with step 3 by more than 5%, launch gaps or serialisation matter. In that case, count the kernels per token.
5. Add `-d 0,32768,65536` if your llama-bench build has `--n-depth`, so f can be read against context length.
6. Server arm with MTP k=3 on port 8083 (`run19b.py` with `BIN` pointed at the synced build), using `EXL3_KTIMING=1`. This gives **f_round**, the share of the 87 ms round spent in `mma8`, `splitk<3>`, `splitk<6>`, `had_in` and `epilogue`. **f_round is the number that ranks the work.**
7. Restore 8081 and release the lock.

**Stall data** would need an **older Nsight Compute; nothing has to be bought.** Your 2025.4 dropped GV100; the last releases with Volta are around 2025.1. Driver 472.12 is the R470 branch (CUDA 11.4 era), so the version most likely to accept it is the **2021.2.x / 2022.3 generation** (2022.3 matches the 11.8 toolkit you already build with). This is not verified. Profiling also needs admin rights or `RmProfilingAdminOnly=0` under TCC. Try it once: `ncu --kernel-name regex:mma8 --launch-count 3 --set full llama-bench.exe ... -p 4 -n 0 -r 1`. If it refuses the driver, don't spend more than an hour on it; the timing bins above are sufficient.

## 2. Ranking by expected wall-clock gain per hour on this box

| Rank | Work | What it targets | Effort | Expected effect |
|---|---|---|---|---|
| **1** | **MMA8-fast** (details below) | ~80% of the round | 1–1.5 days | If verify drops 70 → 55 ms, the round drops 87 → 72 ms: 34.8 → **~42 tok/s (+20%)**. This is a target; the forecast waits on the f_round bins. |
| 2 | **K=6 head:** constant-shift K=6 extraction plus a GROUP4/MMA-style route at M=4, so the head is decoded once per round rather than twice through GROUP2 | ~17.5 ms of draft steps plus the head in verify | ~1 day | +5–10% |
| 3 | **(a) Finish fix A as a standalone item** | M=1 only | ~0.5 day | ≤2% deployed. **Stop working on fix A by itself.** Its proven arithmetic (`span_proof_final.py`) moves into rank 1. |
| 4 | **(b) Prepack** | — | ~1 day | ~0 for this kernel (section 4) |

**What MMA8-fast is.** Three changes that preserve summation order, all inside `exl3_mma8_splitk_kernel`:
1. **Native-lane decode.** Each lane decodes its own EXL3 run with fix A's constant shifts: 2 tiles, so 16 weights, which is the same count as today. Each even/odd pair `(j, j+1)` is already a B-register half2, for column `c0 + 8·(j≥4)` and k-pair `(r0 + 8·((j>>1)&1))/2`. Write the 8 half2 values to a per-warp `sB[32][9]` (padded; 1.1 KB), `__syncwarp`, then read back `b[jj] = sB[mycol][jj]`. This costs about 1 STS/LDS per 2 weights, and the per-weight index arithmetic disappears.
2. **fp16 `xh`.** `had_in` also writes a half copy. This removes about one LDG plus one F2F per weight, on half the lanes.
3. **Register prefetch** of the next k-tile, as the split-K kernel already does, and 8 warps per block.

Because B and A hold exactly the same half values in the same registers, the output should be **byte-identical to deployed MMA8**. That keeps a cheap and strict gate.

**Adoption criterion**, all of which must hold:
- the `mma8` kernel bin is **≥15% lower** (median of 5 runs, CV <2%);
- `llama-bench -p 4` is **≥5% (≥3.5 ms) faster**, and the gain exceeds 3× the A/A spread;
- server MTP tok/s is **≥+3%** on a 20-prompt set;
- greedy content **and** `reasoning_content` are byte-identical (KI-1), and acceptance is unchanged.

For any variant that changes summation order, use this gate instead: max relative error vs fp32 ≤1e-3 per kernel, PPL within +0.1% on `bench/ppl-corpus.txt`, and acceptance within ±0.01 over 20 prompts.

## 3. What would make me abandon fix A (and MMA8-fast)

- **Fix A as a standalone deliverable is already falsified for deployment** by the dispatch structure above. Keep it only if the operator also cares about plain decode.
- **Abandon the extraction idea entirely** if either of these holds on the deployed kernels:
  - a **load-only** MMA8 variant (same loads; decode replaced by an XOR-sum into `b[]`) runs within 15% of the full kernel's time, meaning it is latency/memory-bound and extraction savings can't show; or
  - the compile-time fast variant's kernel bin improves by less than 10% even though SASS per weight drops by at least 25%, meaning instructions are not the limiter.
- **Pivot to prefetch/occupancy work instead** if the `mma8` bin is under 35 ms of the 70 ms verify pass. In that case the budget sits elsewhere: `had_in`/`epilogue` launches, DeltaNet, attention.

## 4. Prepack design: it does not win

- **Byte cost.** A lane's 8 windows need a 37-bit span, and neighbouring spans share 13 bits. Making spans lane-private costs **+33% to +54% bytes**. With 1,415 MiB of VRAM free at 128k context, that cannot be done.
- **Pure word permutation doesn't help.** 32 lanes need overlapping 2-word pairs `(W1−1, W1)` out of 24 words per tile, so no permutation makes the lane→word mapping natural. Fix A already turns the funnel shift into a fixed offset per lane class (`A = {29,37,45,21}[lane&3] − 3j`) without touching bytes.
- **The one zero-byte layout worth considering is split-major tile order**, `[n-block][k-row][8 tiles]`.
  - Loader: permute tiles in the GGUF load path.
  - Kernels: a one-line address change in each of the 5 kernels.
  - Equivalence check: de-permute on the host and compare the SHA with the original buffer, then require byte-identical greedy output.
  - Gain: DRAM locality, 0–10%, and only once a kernel is bandwidth-bound. At 70 ms for about 10 GB, verify runs at roughly 145 GB/s, far from bandwidth-bound. **Defer it.**

## 5. Cheaper things you may be missing (one line each)

- **Launch count.** Every EXL3 matmul is 3 kernels (`had_in` → main → epilogue, `exl3.cu:994/1022/1033`). Count kernels per token in the KTIMING table. Folding `had_in` into the prologue and doing the epilogue with a last-block counter removes 2/3 of those launches. *Source: code; prior advice item D.*
- **CUDA graphs during spec decoding.** Check whether graphs are actually active for M=4 verify: an A/B with `GGML_CUDA_DISABLE_GRAPHS=1` that shows no difference means they aren't. *Source: llama.cpp graph gating for batch >1.*
- **L2 persistence (`cudaAccessPolicyWindow`)** is sm_80+ and **not available on V100**; don't plan around it.
- **`cp.async`** is also sm_80+. On sm_70 the substitute is register double-buffering, which split-K has (`:290-321`) and **MMA8 lacks**. It is part of rank 1.
- **Warp-specialised loaders** have little value on sm_70 without asynchronous copies; register prefetch does the same job.
- **Persistent kernels:** only worth it after the launch fusion, and only if the KTIMING table shows inter-kernel gaps above ~3 ms per token.
- **A compact draft head.** Load-time requantisation of a truncated vocabulary (`bench/draft-vocab-32k.json` exists) into q4_0 is about 90 MB and runs on stock MMVQ. Verify still uses the exact 6-bit head, so outputs are unchanged. It targets most of the 17.5 ms draft cost; it is an alternative to rank 2.

**Order of work:** Step 0 → Steps 1–3 (one GPU lock) → MMA8-fast → K=6 head. Do not run `run19b.py` in its current form.

Sources:
- [Nsight Compute Release Notes](https://docs.nvidia.com/nsight-compute/ReleaseNotes/index.html)
- [Getting Started with Nsight Compute 2025.1](https://developer.nvidia.com/tools-overview/nsight-compute/get-started-2025_1)
- [Getting Started with Nsight Compute 2025.3](https://developer.nvidia.com/tools-overview/nsight-compute/get-started-2025_3)

---
model: claude-opus-5-5, claude-haiku-4-5-20251001 | turns: 20 | notional cost USD: 1.2343506 | 302369 ms | session: faa88616-8bab-4186-888d-8ca48098999f
