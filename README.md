# EXL3 on a Tesla V100 16 GB (sm_70): a 27B with 128k context on a cheap used card

Notes from running a **Qwen3.8-27B EXL3 3.0 bpw** model on a single **Tesla V100-SXM2-16GB** (Volta,
`sm_70`) with a 128k context, a vision tower, and MTP speculative decoding at k=3 — on Windows, with a
Triton-free CUDA 11.8 toolchain.

Everything here is **measured on this machine** unless it is explicitly labelled otherwise. The negative
results are the point of the repository: most of the "obvious" optimizations for this shape of problem do
not pay here, and several confident-looking estimates turned out to be artefacts.

## Scope: this is the "16 GB Volta card" case — and only that case

**The whole point is "a 27B model, 128k context, on a used 16 GB V100"** (on the order of ¥40k second-hand
at the time of writing, 2026-09). Every choice below follows from that: a **3.0 bpw EXL3 GGUF (12.7 GB)**, a
**q4_0 KV cache**, and the vision tower pinned to the **CPU** so that 128k fits in 16 GB at all.

**If you are choosing hardware today, do not do this.** On any FP4-capable GPU you would store the model in
a native 4-bit block format (NVFP4/MXFP4) whose values feed the FP4 tensor cores **directly** — no
per-weight decode, no index arithmetic, far higher throughput at comparable quality. EXL3 is a *storage*
format whose decode targets **fp16**; it never touches FP4 silicon, so on an FP4 card it buys you the bytes
but not the speed. What is left for a 16 GB Volta card is exactly the kernel-and-latency work in this
repository — the trap list in §3 applies to any card, the kernel findings do not.

Engine: the `exllamav100` fork (not ours; linked, not redistributed): <https://github.com/Vendetta1871/exllamav100>
Model: a public EXL3-quantised GGUF (`huihui-…-exl3-3bpw-mtp.gguf`, 12.7 GB) — weights are not included here.

---

## 1. The recipe that works (128k context, vision, MTP k=3)

```
llama-server -m <exl3 3bpw gguf> -c 131072 -ngl 99 -fa on -b 2048 -ub 512
             -ctk q4_0 -ctv q4_0
             -mm <mmproj-model-bf16.gguf> --no-mmproj-offload
             --jinja --no-reasoning-preserve
             (env) EXL3_EXPERIMENTAL_MMA8=1
```

| setting | why it is what it is (measured) |
|---|---|
| `-c 131072` | 128k is the biggest that fits: 64k → 14,189 MiB, 128k → 15,981 MiB of 16,258 (clip on GPU); 262,144 segfaults. With `--no-mmproj-offload` (clip on CPU) 128k lands at **14,843 MiB used / 1,415 free**. |
| `--no-mmproj-offload` | moves the vision tower to the CPU and costs ~0 VRAM; without it, 128k does not fit. |
| `-ctk/ctv q4_0` | fp16 KV would not fit at 128k. KV costs ≈ **+1,792 MiB per 64k** at q4_0 (see the caveat in `docs/01`). |
| `-fa on` | required for the context; see the `FA_ALL_QUANTS` trap below before you touch the KV types. |
| MTP k=3 | 1.60–1.62× over no draft on this model; acceptance 0.68327 (343/502), mean commit 3.04. |
| `EXL3_EXPERIMENTAL_MMA8=1` | selects the 4-row tensor-core verifier. Without it the fork has neither MMA8 nor GROUP2 (verified by grep on both source trees). |

## 2. What it actually does here

**Decode** (matched prompts, 128k deployment): **32.52 tok/s @6k → 25.99 @23.7k → 19.60 @71k → 20.36 @96k**.
**Prefill**: 345 → 248 tok/s from 6k to 71k.
**Lab baseline** (same model, `-c 65536`, MTP k=3): 28.76 ms/round = 34.77 tok/s; no-draft 46.68 ms/token.

**Where the time goes** (one GPU lock, cudaEvent bins + a compile-time knock-out):

| measurement | value |
|---|---|
| EXL3 share of a decode token (NULL_MATMUL knock-out) | **86.4 %** (47.85 → 6.525 ms/tok) |
| verify breakdown at M=4 (mma8-main / had_in / epilogue) | **34.8 / 32.6 / 32.6 %** |
| EXL3 matmuls per decode token | **~401** (×3 kernels ≈ 1,200 launches/token) |
| CUDA graphs | worth **+31 %** decode throughput (20.9 vs 15.94 tok/s) |
| effective bandwidth vs practical ceiling | **~272 GB/s vs ~830 GB/s = 33 %** |

**The conclusion that followed from those numbers**: this workload is **latency/occupancy-bound, not
bandwidth-bound or instruction-bound**. Cutting instructions (measured: −12.3 % SASS, LDS 136→32) bought
**nothing**; the roofline says a 128k MTP round could be ~19 ms (≈157 tok/s) but the practical ceiling with
the current issue rate is ~25–30 ms/round (≈100 tok/s), and we sit at 87.7 ms.

**SASS of the deployed kernels** (from the deployed cubins, same function / same loop / same flags):

| kernel | loop instructions | per weight | registers |
|---|---|---|---|
| `exl3_gemv_splitk_kernel<3,2,false>` | 711 | 11.11 | 142 |
| `exl3_gemv_splitk_kernel<6,2,false>` | 586 | 9.16 | 112 |
| `exl3_mma8_splitk_kernel<2>` (verify) | 234 | 14.63 | 95 |

In the MMA8 verify loop, **~76 % of the instructions are index/address arithmetic and only 2.4 % are HMMA**.

## 3. Traps (each one cost real time here)

1. **`GGML_CUDA_FA_ALL_QUANTS=OFF` (the default) silently disables `q5_0`/`q5_1`/`q4_1` for flash
   attention.** `fattn.cu` returns `false` for those types unless the define is on, so `-ctk q5_1` either
   aborts at load or drops FA. `-ctk/-ctv` themselves accept
   `f32 f16 bf16 q8_0 q4_0 q4_1 iq4_nl q5_0 q5_1`. Both builders here were built with it **OFF**.
2. **CUDA 13.x cannot disassemble `sm_70`.** `nvdisasm` (and therefore `cuobjdump -sass`) answers
   `CUDA architecture "SM70" is deprecated. Please use CUDA toolkit 13.0 or earlier`. A CUDA 11.8
   `nvdisasm`/`cuobjdump` is required to read Volta SASS. `-fun` on `nvdisasm` takes an **index**;
   mangled names go to `cuobjdump -fun`.
3. **The server ignores `GGML_CUDA_DISABLE_GRAPHS`** — only `llama-bench` honours it, and the server reuses
   graphs across requests. Any "graphs off" server measurement is really a graphs-on measurement.
4. **A tool that stops the deployment can leave an orphaned `llama-server`** holding ~13 GB while the
   deployment stays down. Sweep by process name and restore explicitly.
5. **An `atexit`-printed table is lost if you `terminate()` the server.** Use a graceful unload/exit.
6. **Byte-identity as a correctness gate is only as good as the field you compare.** A gate that compared
   `content` passed for a while because both arms returned empty content while the answer sat in
   `reasoning_content`; compare **both** fields.
7. **Parallel advisor/LLM runs overwrite each other's output files** if the filename is a bare timestamp.
8. **`git worktree add` concurrent with a live session on the same repo can hang**; create worktrees before
   launching work.
9. **Regression gates**: use a fp32 reference, perplexity and a ≥20-prompt acceptance average. "Greedy bytes
   match" is not a gate — adding in a different order changes the trajectory.
10. **The `FA_ALL_QUANTS` trap has already fooled an operator.** This lab's launch script carried the note
    *"q5_1 is pathological on this fork (20.95 tok/s) — do not use it"*. That 20.95 tok/s **is** the
    flash-attention fallback penalty: `q5_1` is not FA-capable in a default build, so the server silently drops
    FA and loses about a third of its decode speed. Rebuilt with `GGML_CUDA_FA_ALL_QUANTS=ON` (trap 1), the
    same `-ctk q5_1 -ctv q4_0` runs **32.24 / 29.20 / 21.58 tok/s at 6k / 23.7k / 71k** — the same band as
    `q4_0/q4_0` — costs **+234 MiB** of KV at 128k (15,263 / 995 free), and scores **PPL 4.5880 vs 4.5991** on
    the same corpus. The verdict "this cache type is bad" was really "this build cannot feed it".

## 4. What did **not** work (all measured here)

| idea | result |
|---|---|
| K=6 specialised decode path (constant-shift + MMA at the head) | SASS −12.3 %, LDS 136→32, **decode 15.78 → 15.16 tok/s (−3.9 %)** — the loop is memory-bound |
| n-gram/lookup speculative decoding | **loses 24.4–24.7 %** to MTP k=3 on copy-heavy prompts with acceptance 0.879 vs 0.963; three sanity gates passed, so this is not a harness artefact (an earlier "the offline simulation said great" number came from measuring the *prompt* stream, not the output) |
| Constant-shift extraction in the M=1 GEMV only | touches ~1/64 of the work under MTP (rows ∈ {2,4,28,32,42}, never 1) → ≤2 % |
| Load-time prepack of a K=3 layout | not the lever: the loop is not bandwidth-bound |
| K=4 tensors through the 4-row MMA8 path | K=4 is **2.3 % of the weight bytes** (24 of 1,684 tensors) — dead by the pre-registered 15 % criterion |
| DSpark / DFlash2 / EAGLE3 / CPU-resident drafters / a Volta attention fork | all net losses or unavailable in this fork (details in `measurements/`) |
| "8 warps/block" for the MMA8 kernel | at REG 95 it *lowers* occupancy (256 threads × 95 regs = 2 blocks = 16 warps/SM vs 20 today) |

## 5. What measuring harder changed (and what is still open)

A single GPU lock with a synchronized spec-phase timer and a floor of bench arms resolved most of the
contradictions this repository started with:

* **Draft cost is ≈ 10.56 ms per round** (synchronized timer, 180 rounds). The old "17.5 ms, mostly the K=6
  head" figure was a subtraction from a different era. The decomposition still does not close cleanly — the
  remainder swallows async verify + sampling — so a corrected timer run is queued.
* **"The deployment is slower" is real**, and the cause is **context size and mmproj shape, not kernels**: with
  matched flags the lab and deployed arms produce **byte-identical acceptance streams (343/502)**, so the
  earlier "0.615 vs 0.683 acceptance" discrepancy was a **kernel-order artefact**, and the config-delta
  hypothesis (build flags) is dead.
* **Target-side sampling is not the missing time** (`-bs` Δ < 1 %, identical output streams).
* **CUDA graphs are worth +2.6 % in the bench only** — the server ignores `GGML_CUDA_DISABLE_GRAPHS` and keeps
  reusing graphs (177–253 reuses observed), so server-side "graphs off" measurements measure graphs on.
* **The long-context slope is 2.2× the kill threshold** (0.15 ms/1k) — attention at depth really does cost,
  which reopens the case for porting a Volta attention kernel.
* **Draft-only vocabulary cap is worth building**: 99 % of the head's mass sits in **35.8 %** of the 128-column
  blocks (B=694 of 1940), i.e. ≈ 2.0 GB of the ~16.1 GB round, with verification on the full head so outputs
  are unchanged. Not implemented yet.
* **The MMA8 kernel's own bandwidth and its true share of the round are still unmeasured** (the event-bin
  round-robin aliases, so absolute bin times overcount 1.4–1.7×; ratios are the usable part), and the
  "affine fold" (−40 instructions per iteration, see `measurements/advisor-tensorcore-decode-verdict.md`) is
  untested.

## 6. Reproducing the measurements

`scripts/` holds the harness we used: a GPU lock (`gpu-lock.sh`), the deployment probe set
(`run-deploy-probes.sh`, `probe3.py`), a serving watchdog, and the session wrappers. Numbers in this
repository come from `measurements/`; the toolchain quirks are in `docs/01-known-issues-and-traps.md`.

## 7. Credits and licence

* Engine: the `exllamav100` fork (upstream), not redistributed here.
* Text, scripts and measurements in this repository: **MIT** (see `LICENSE`). Corrections welcome — several
  of these numbers replaced earlier beliefs, and one audit of our own claims changed three of them.
