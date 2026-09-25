# Consultation to Claude Code (opus, effort high) -- 20260926-053724-69755-consult-tensorcore-decode

Question asked:
```
# CONSULT — can EXL3's decode/expansion be moved onto the tensor cores? (organise the argument, then answer)

We want your verdict on a specific structural idea, plus an organised issue list. Read-only lab access;
you may read `H:\exl3-local\exllamav100\ggml\src\ggml-cuda\exl3.cu` (the **deployed** source — the lab tree
has no MMA8 kernel at all) and `H:\exl3-lab\STATUS-NOW.md` §2/§9/§10/§11, `logs/brief29/REPORT.md`,
`logs/brief30/REPORT.md`, `logs/advisor-20260926-023039-62719-consult-score-all-options.md` (your own
scoring table). Tool budget ~30 calls.

## The proposal to evaluate

EXL3 stores weights as a **bit-packed trellis**; the kernel expands each weight to fp16 at run time and feeds
fp16 tensor cores. The operator asks: **can that expansion be done *by* the tensor cores instead of in SIMT
integer/DP4A work?**

## What the deployed kernel actually does (I checked; correct me if I misread)

`exl3_mma8_splitk_kernel` (`exl3.cu:427-503`), inner loop, per weight:

```cpp
const int i0 = (4*(cc & 7) + ((rr0 & 7) >> 1))*8 + 4*(cc >> 3) + (rr0 & 1) + 2*(rr0 >> 3);
const half h0 = __float2half(exl3_decode_cb<cb>(exl3_window<3>(pw + tc*8*3, i0)));
```

So the expansion is **not** a cumulative/scan structure (my first guess was wrong): it is
**(a) a lane/row/col-dependent bit-index computation, (b) a K-bit window extract from shared memory,
(c) a codebook map (`cb=2` is `mul1` = one IDP4A per weight), (d) an fp16 convert**. There is no prefix sum
along the trellis in the kernel.

Measured shape of that loop (deployed cubin, same-function/same-range tally): **234 instructions per lane
per k-tile iteration = 14.63 per model weight; ~76 % of them index/address arithmetic (IMAD/SHF/IADD3/LOP3),
HMMA 2.4 % (16 per iteration), F2F exactly 1 per weight.**

## The measured context you must fit your answer into

* **f = 86.4 %**: the three EXL3 kernels are 86 % of decode token time (NULL_MATMUL knock-out, 47.85 → 6.525
  ms/token). Inside verify (M=4) the split is **mma8-main 34.8 % / had_in 32.6 % / epilogue 32.6 %**, so the
  MMA8 kernel *proper* is ≈ 30 % of the round (~26 of 87.4 ms).
* **Instruction reduction has already failed once**: the K=6 head rewrite cut SASS −12.3 % and LDS 136→32 and
  the measured decode went **15.78 → 15.16 tok/s (−3.9 %)** — memory-bound. Your scoring table therefore put
  "A. MMA8-fast" at 38 and said the value sits in G (register prefetch) and NEW-2 (grid fill), i.e. **bytes in
  flight / occupancy**, not instruction count.
* This question is *not* the same as "remove instructions": it is "**move work to a different execution unit**".
  The SIMT issue slots would stop doing the expansion, and the TC would do it — but sm_70's TC is
  **fp16 `m8n8k4` only** (no INT8, no binary/1-bit MMA; `cp.async`/`ldmatrix` absent), the kernel already runs
  at **REG 95** with 4 warps/block, and every extra fragment costs registers.
* Anchors: 12.7 GB weights, 3.0 bpw, cb=mul1; round 87.66 ms lab / ~141 ms @71k deployed; effective bandwidth
  272 GB/s against a practical ~830 GB/s ceiling; CUDA graphs worth +31 %.

## What I need from you

1. **Verdict on the tensor-core expansion**, with the mechanism spelled out. Specifically address: the decode
   is a **gather** (data-dependent bit extract + codebook map), and dense MAC engines do not gather. Can any of
   these actually work on sm_70, and at what cost?
   * `w = one_hot(state) · Codebook` as an fp16 MMA — the one-hot must be built from the packed bits: is that
     build cheaper than the current ~14 instructions/weight, or does it just move the cost into SIMT?
   * bit-slicing / binary encoding: exploit that a K-bit state is a **sum of K bits times powers of two**, so
     `w = Σ_b bit_b · (2^b · basis)` — can the K bit-planes be produced with fewer instructions than the
     current window+codebook path, and can the accumulation go to the TC?
   * **LUT-GEMM-family prior art** ("quantised matmul without dequantisation", LUT-based TC tricks): does it
     apply to a *bit-packed trellis* rather than uniform blocks, and are there measured numbers on V100-class
     hardware? Cite what you find.
   * anything else you know: `PRMT`-based register LUTs, `__byte_perm` tricks, warp-shuffle broadcasts of
     codebook entries, `lop3`-based select trees.
2. **Organise the argument** as a table: `claim | what would have to be true | cheapest test that could
   falsify it | your confidence (with the reason)`. Include the claims you think are *wrong* in the framing
   above.
3. **The one-hour falsification test** we should run first: a micro-benchmark on this V100 that compares the
   current expansion against the best TC-based alternative for the same 64 weights, with the exact code sketch
   (no model, no GGUF — a standalone .cu is fine). Give the pass/fail criterion in cycles per weight.
4. **Consequence for the plan**: if the answer is "the expansion cannot move", say what that does to your
   scoring of A / G / NEW-2 / L, and what the single highest-value remaining item becomes. If the answer is
   "it can, here is how", re-score accordingly and name the first implementation step.
5. **Quality constraint**: fp16 inputs with fp32 accumulation is acceptable (the same order as today's fp16
   add chain); anything that changes the *values* (e.g. coarser codebook) is out of scope — say so if your
   proposal crosses that line.

Answer in English. Numbers and file:line over prose.
```

## Answer

# Can EXL3's weight expansion run on the V100 tensor cores?

## Verdict

**No, not the expansion itself.** Two of its four steps cannot move to sm_70 tensor cores without changing the weight values. The one step that can move is small: the scale-and-offset at the end of the codebook map, about 21% of the loop. A separate change on the activation side is bit-exact and saves about as much. Together they would take the loop from 234 to about 150 instructions. Whether that buys wall time depends on occupancy, and a one-hour micro-benchmark can settle it.

### One premise is wrong

The loop is not 76% index arithmetic. I read the loop directly (`logs/brief26/deployed-cubin23.sass.txt:1226-1461`, PC 0x15c0–0x2450, 234 instructions). There is no per-weight index computation inside it: the compiler precomputed about 24 shared-memory addresses in the prologue (`:1200-1224`, the run of `IMAD.SHL …, 0x4`), which is part of why the kernel uses 95 registers. The 76% figure came from the 658-instruction tally that brief29 §1c withdrew. 16 of the loop's 22 IMAD instructions are the codebook multiply (`IMAD R, R, -0x7c232ed3` = `0x83DCD12D`). I found no LEA in this PC range, so brief29's histogram line "LEA 17" is miscounted; the 234 total is correct.

Actual breakdown per lane per k-tile (16 weights):

| Part | Instructions | Per weight | Can it move? |
|---|---|---|---|
| **Window gather**: `LDS`×29 + `SHF.R.U64`×16 + `LOP3 &0xffff`×16 | 61 | 3.8 | No. A lane-dependent shared-memory gather; tensor cores don't gather |
| **Hash**: `IMAD ×C`×16 + `IDP.4A`×16 | 32 | 2.0 | No. Nonlinear in the bits (see below) |
| **Affine tail**: `HADD2.F32`×16 + `FFMA`(α,β)×16 + `F2F`×16 | 48 | 3.0 | **Yes**, by linearity through the MMA |
| B pack `PRMT` | 8 | 0.5 | Stays |
| **A side** (activations): `LDG`×16 + `F2F`×16 + `PRMT`×8 + `CS2R`×4 (`exl3.cu:474-482`) | ~44 | 2.75 | Yes, bit-exact (fp16 `xh`); not a tensor-core question |
| HMMA | 16 | 1.0 | — |
| Staging and loop control | ~25 | — | — |

### Each tensor-core mechanism, checked

- **`one_hot(state)·Codebook`: dead.** The codebook input is the full 16-bit trellis state (`& 0xffff`, `exl3.cu:39`), not a 3-bit symbol. That means a 65,536-wide one-hot operand, with 1 useful multiply-add per 65,536. Building the one-hot also costs more SIMT work than the 2-instruction hash it would replace.
- **Bit-slicing, `w = Σ bit_b·2^b·basis`: dead.** The value is `α·(1024 + bytesum((w·0x83DCD12D) mod 2³²)) + β` (`exl3.cu:19-22`). The mod-2³² multiply and the byte sum with carries make it nonlinear in the bits of w; this is by design, since QTIP's computed codes are meant to decorrelate. Any linear bit-plane version is a different codebook, which your constraint rules out. sm_70 also has no 1-bit MMA (that starts at sm_75).
- **Byte-plane variant (the most tensor-core-heavy exact option): loses.** `PRMT(x, 0x64646464, sel)` gives an exact half2 of `1024+byte` for two bytes, so the tensor core could do the byte sum. But that is 4× the K dimension: 64 HMMA + 32 PRMT per iteration instead of 16 IDP, and A replicated ×4. That adds roughly 48 registers, dropping from 5 to about 3 blocks per SM. It is exact but worse on both instruction count and registers.
- **LUT-GEMM family: does not apply.**
  - LUT-GEMM (Park et al.) needs weights that are linear in bit-planes (its BCQ format). It precomputes activation partial sums per 8-bit pattern and runs on CUDA cores, not tensor cores. I found no V100 numbers.
  - FLUTE targets small non-uniform lookup tables (16 entries or fewer) on Ampere/Hopper-class GPUs.
  - LUT Tensor Core concludes that conventional LUT kernels fail to deliver on current tensor cores and proposes new hardware.
  - QTIP designed 1MAD/3INST (EXL3's cb2 is the 1MAD-style one) specifically so decoding costs 2–4 ALU instructions instead of a lookup. A 2¹⁶-entry fp16 table (128 KB) would also cost a gather, the same as today.
- **PRMT / shuffle / LOP3 tables: no gain.** A PRMT is an 8-entry byte table, and SHFL is quarter-rate on Volta with 32 entries. Both lose to `IMAD`+`IDP` = 2 instructions for a 65,536-entry computed codebook.
- **What does move: the affine fold.** Feed `h − 1534` straight into the MMA. It is an exact integer in [−510, 510], so it is exact in fp16, and one `HSUB2` handles a weight pair. Then:
  - `y = α·acc + β′·Σ_k a_k`, where `β′ = β + 1534α ≈ −0.0018`, computed exactly in fp32.
  - α goes into the epilogue's fp32 scale.
  - β′·Σa is a per-row constant added before the output Hadamard; after the transform it only touches index 0 of each 128-block.
  - `Σ_k xh` has to be computed deterministically: have `had_in` write per-block sums and let the epilogue add them in fixed order. No `atomicAdd`, or runs stop being reproducible.
  - Saving: 48 − 8 = **40 instructions per iteration**.
- **A-side fp16 `xh` for the MMA8 path ("A1"): bit-exact.** Today the loop does `__float2half(xr[j])` on values `had_in` already rounded in fp32. Having `had_in` write `__float2half(s*SCALE)` gives identical bits, and all 4 warps stop redoing 16 LDG + 16 F2F + 8 PRMT every k-tile. Saving: **about 42 per iteration**. The M=1 gemv path keeps fp32 `xh`.

## Claims, tested

| # | Claim | What would have to be true | Cheapest falsifier | Confidence |
|---|---|---|---|---|
| 1 | Loop is ~76% index arithmetic | Per-weight index math inside the loop | Read SASS `:1226-1461` (done) | **Wrong, 95%**. Indices are hoisted; the IMADs are the hash multiply |
| 2 | Expansion = gather + codebook map | Both steps are data-dependent loads | SASS | **Half right, 95%**. The window is a gather; the codebook is a computed hash with no table |
| 3 | one-hot·Codebook on tensor cores | State space small enough for an MMA operand | Arithmetic: state is 16 bits | **Infeasible, 99%** |
| 4 | Bit-plane linear decomposition | `f(a⊕b) = f(a)+f(b)−f(0)` for disjoint bit masks | 2¹⁶ host loop, 1 minute | **False, 99%**. Linearising changes the codebook (out of scope) |
| 5 | LUT-GEMM family applies to a trellis | Weights linear in bit-planes, or ≤4-bit index | Read the papers | **No, 85%**. No V100 numbers found |
| 6 | Affine tail must stay in SIMT | α,β can't pass through the matmul | Linearity; bench arm B | **Wrong, 85%**. −40 instr/iter |
| 7 | HMMA at 2.4% means idle tensor-core capacity to use | Tensor-core throughput is the binding resource | Byte-plane arithmetic: more issue, more registers | **Misleading, 90%**. HMMA uses the same issue slot |
| 8 | K=6's −3.9% proves instruction count doesn't matter | K=6 and MMA8 have similar instructions per byte | Arithmetic: 586/1536 B = **0.38 instr/B** vs 234/192 B = **1.22 instr/B**; K=6 is ~1 matmul/token (~2–3% of token time, so Amdahl caps it below 1%) | **Doesn't transfer, 80%** |
| 9 | MMA8 proper ≈ 30% of the round (26 ms) | had_in (~80 KB/call) takes as long as mma8 (~23 MB/call) | Time each kernel alone (bench arm E) | **Probably wrong, 70%**. Equal thirds on bins that brief29 says overcount 1.4–1.7× look like per-launch cadence, not work |
| 10 | 272 GB/s is the MMA8 number | — | — | **No, 75%**. It's the M=1 gemv decode average; MMA8's own GB/s is unmeasured. pp4 suggests about 150–250 |
| 11 | MMA8 is latency-bound only (my earlier table) | Issue time far below load latency | Bench arm D grid × variant | **Partly wrong, 60%**. See model below |
| 12 | sm_70 tensor cores are fp16 m8n8k4 only; no int8/b1, `cp.async` or `ldmatrix` | — | — | **True, 99%** |
| 13 | Extra fragments cost registers; REG 95 limits occupancy | — | `--resource-usage` | **True, 85%**. 95 → 5 blocks/SM, but grid 240 gives only 3 blocks/SM |

### Model behind claim 11

This is a toy model (inference, not measurement). Assumptions: DRAM latency L ≈ 700 cycles, 1.53 GHz, per scheduler period `T = max(L + c, w·c)`, where w is warps per scheduler and c is instructions per iteration.

| Configuration | Model GB/s | vs now |
|---|---|---|
| Now (w=3, c=234) | ~300 | 1.00× |
| NEW-2 only (w=5) | ~400 | 1.33× |
| Fold + A1 only (c≈150) | ~330 | 1.10× |
| Fold + A1 + NEW-2 | ~550 | 1.8× |
| … + G (prefetch) | ~630 | 2.1× |

The kernel sits where issue rate and latency are about equal, so each lever alone gives little and the combination is where the gain is. This also explains why the K=6 cut did nothing: at 0.38 instructions per byte it is latency-bound. These absolute GB/s values are model output only.

## The one-hour falsification test

Standalone `tc_fold_bench.cu`, `nvcc -O3 -arch=sm_70 --resource-usage`. No model, no GGUF. Take the GPU lock: the deployed server uses 14.8 of 16 GB, so keep buffers ≤ 512 MB.

```cpp
#include <cstdio>
#include <cstdint>
#include <cuda_fp16.h>

__device__ __forceinline__ uint32_t win3(const volatile uint32_t* w, int i) {   // == exl3_window<3>
    const int b2 = (i + 1)*3 + 768, i0 = (b2 - 16)/32, i1 = (b2 - 1)/32, s = (i1 + 1)*32 - b2;
    const uint32_t a = w[i0 % 24], b = w[i1 % 24];
    return (uint32_t)((((uint64_t)a << 32) | b) >> s) & 0xffff;
}
__device__ __forceinline__ half cb2_val(uint32_t w) {                  // deployed exl3.cu:19-22 + F2F
    const uint32_t s = __dp4a(w*0x83DCD12Du, 0x01010101u, 0x6400u);
    const float h = __half2float(__ushort_as_half((unsigned short)(s & 0xffff)));
    return __float2half(h*__half2float(__ushort_as_half(0x1eee)) + __half2float(__ushort_as_half(0xc931)));
}
__device__ __forceinline__ uint32_t pair_ref(uint32_t w0, uint32_t w1) {
    half2 p = __halves2half2(cb2_val(w0), cb2_val(w1)); return *(uint32_t*)&p;
}
__device__ __forceinline__ uint32_t pair_fold(uint32_t w0, uint32_t w1) { // B = h-1534, exact in fp16
    const uint32_t s0 = __dp4a(w0*0x83DCD12Du, 0x01010101u, 0x6400u);
    const uint32_t s1 = __dp4a(w1*0x83DCD12Du, 0x01010101u, 0x6400u);
    uint32_t p = __byte_perm(s0, s1, 0x5410);                              // {s0.lo, s1.lo}
    half2 h = __hsub2(*(half2*)&p, __half2half2(__ushort_as_half(0x65fe)));  // 0x65fe = 1534.0
    return *(uint32_t*)&h;
}
// MODE 0 = deployed decode, 1 = fold, 2 = floor (raw bits, no codebook)
template <int MODE>
__global__ void __launch_bounds__(128) arm_shared(const uint32_t* __restrict__ src,
        const uint32_t* __restrict__ a_h, float* out, int iters, unsigned long long* t) {
    __shared__ uint32_t pk[4*2*8*3];
    const int warp = threadIdx.x/32, lane = threadIdx.x%32;
    const int qp = (lane >> 2) & 3, ar = (lane & 3) + ((lane & 16) ? 4 : 0), col = qp*8 + ar;
    const int tc = col >> 4, cc = col & 15;
    volatile uint32_t* pw = pk + warp*48;                   // volatile: forces LDS every iteration
    if (lane < 48)      pw[lane]      = src[(blockIdx.x*4 + warp)*48 + lane];
    if (lane + 32 < 48) pw[lane + 32] = src[(blockIdx.x*4 + warp)*48 + lane + 32];
    __syncwarp();
    uint32_t a[8]; for (int j = 0; j < 8; ++j) a[j] = a_h[(ar & 3)*8 + j];   // A excluded: B-side only
    float c[8] = {};
    const unsigned long long t0 = clock64();
    for (int it = 0; it < iters; ++it) {
        uint32_t b[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int r0 = 2*j, r1 = r0 + 1;
            const int i0 = (4*(cc & 7) + ((r0 & 7) >> 1))*8 + 4*(cc >> 3) + (r0 & 1) + 2*(r0 >> 3);
            const int i1 = (4*(cc & 7) + ((r1 & 7) >> 1))*8 + 4*(cc >> 3) + (r1 & 1) + 2*(r1 >> 3);
            const uint32_t w0 = win3(pw + tc*24, i0), w1 = win3(pw + tc*24, i1);
            b[j] = MODE == 0 ? pair_ref(w0, w1) : MODE == 1 ? pair_fold(w0, w1) : __byte_perm(w0, w1, 0x5410);
        }
#pragma unroll
        for (int j = 0; j < 4; ++j)                          // verbatim exl3.cu:484-492
            asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};\n"
                : "+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7])
                : "r"(a[2*j]),"r"(a[2*j+1]),"r"(b[2*j]),"r"(b[2*j+1]));
    }
    const unsigned long long t1 = clock64();
    unsigned smid; asm("mov.u32 %0, %%smid;" : "=r"(smid));
    if (lane == 0) { const int g = blockIdx.x*4 + warp; t[3*g] = t0; t[3*g+1] = t1; t[3*g+2] = smid; }
    for (int i = 0; i < 8; ++i) out[(blockIdx.x*128 + threadIdx.x)*8 + i] = c[i];    // defeat DCE
}
```

### Arms

- **S, shared-resident (the decision):**
  - Grid = 80 × {1, 2, 3, 5} blocks, one wave, `iters = 1<<16`, random `src`.
  - Metric: **SM-cycles per weight** = (max t1 − min t0 per SM) / (warps on that SM × 512 × iters).
  - Expectations: MODE 0 ≥ 0.114 (issue bound: 234/512/4; measured probably 0.13–0.16), MODE 1 ≈ 0.09, MODE 2 ≈ 0.05.
- **Correctness:** at `iters=1`, compare `α·c_fold + β′·Σa` (β′ = β + 1534α, fp32) against `c_ref` and against a host fp64 reference. Expected: fold error ≤ reference error, because the fold drops the per-weight fp16 rounding (~8e-4).
- **D, DRAM:** copy `exl3_mma8_splitk_kernel` (`exl3.cu:426-503`) verbatim as MODE 0. Add:
  - MODE 1: `pair_fold`
  - MODE 3: fold + fp16 `xh` (A1)
  - MODE 4: load-only (`b[j] = pw[j] ^ pw[j+24]`)

  Use the K=3 shapes from the brief30 task-2 census (e.g. n=5120 with k = 5120 and 17408), 8 rotating copies to defeat the 6 MB L2, and split-K S chosen for **240 / 480 / 960 blocks**. Report GB/s.
- **E:** `had_in` and `epilogue` alone (verbatim, rows=4, S=6), 1000 launches in a CUDA graph → µs per call × 400 = ms per round. This settles claim 9.

### Pass/fail

- **PASS** (build the fold): arm S MODE 1 ≤ **0.80×** MODE 0 in SM-cycles per weight at ≥16 warps/SM, **and** arm D MODE 3 at 480 blocks ≥ **1.15×** MODE 0 at 480.
- **KILL** the instruction line for MMA8 (A1 and the fold) if arm S MODE 1 > **0.90×** MODE 0, or if arm D MODE 0 at 480 is within 15% of MODE 4 at 480 (load/latency-bound).
- **NEW-2 is real** if arm D MODE 0 at 480 ≥ 1.20× MODE 0 at 240.
- **E (launch fusion)** rises only if arm E shows `had_in`+`epilogue` ≥ 10 ms/round. My prediction is 2–4 ms.

## Consequence for the plan

"The expansion cannot move" changes the scores as follows:

| Item | Old | New | Why |
|---|---|---|---|
| Tensor-core expansion (one-hot / bit-plane / LUT) | — | **2** | Infeasible or value-changing |
| A, MMA8-fast as one item | 38 | **retire** | Split into A1 and A2 |
| A1: fp16 `xh` into the MMA8 path | (was I, 15) | **55** (65 bundled with NEW-2) | Bit-exact; −42 instr/iter |
| A2: affine fold | — | **50**, conditional on arm S passing | −40 instr/iter; changes rounding, not values |
| NEW-2: grid fill | 62 | **68** | It's what makes A1/A2 pay (issue and latency are within ~30%) |
| G: prefetch, MMA8-specific | 55 | **62** | MMA8 staging is 1.5 words per lane, so one prefetch stage costs ~2 registers per lane, not a 4-register `uint4` |
| L: draft vocabulary cap | 70 | **70** | Independent: −2.04 GB/round, output unchanged |
| E: launch fusion | 30 | 20–60 | Decided by arm E |

- **Highest-value next item:** the one-hour benchmark itself. It reorders A1/A2/NEW-2/G/E and needs no engine build.
- **First build step:**
  - If it passes: A1 plus NEW-2. A1 is bit-exact, so the byte-identity gate still applies. NEW-2 should be done by splitting N more finely, which keeps the fp32 reduction order; adding split-K does not.
  - If it fails: **L**.

## Quality constraint

- **A1:** bit-identical.
- **Fold (A2):** inputs stay fp16 with fp32 accumulation, and per-weight values become slightly *more* exact, so it is in scope. But results will not be byte-identical to today: brief30 showed rounding order alone moved acceptance from 0.683 to 0.615. Gate it on the fp64 error check plus PPL on `bench/ppl-corpus.txt`, not on byte identity.
- **Out of scope:** any bit-plane or linearised codebook, because it changes the values.

Sources:
- [LUT-GEMM (arXiv 2206.09557)](https://arxiv.org/abs/2206.09557)
- [LUT-GEMM (ICLR 2024)](https://proceedings.iclr.cc/paper_files/paper/2024/file/a4f98ce85f440ee269b0df57b4368719-Paper-Conference.pdf)
- [FLUTE (EMNLP Findings 2024)](https://aclanthology.org/2024.findings-emnlp.724.pdf)
- [LUT Tensor Core (arXiv 2408.06003)](https://arxiv.org/abs/2408.06003)
- [LUT Tensor Core (ISCA 2025)](https://dl.acm.org/doi/10.1145/3695053.3731057)
- [QTIP (arXiv 2406.11235)](https://arxiv.org/abs/2406.11235)
- [QTIP (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/6de2e84b8da47bb2eb5e2ac96c63d2b0-Paper-Conference.pdf)

---
model: claude-opus-5-5, claude-haiku-4-5-20251001 | turns: 13 | notional cost USD: 1.9623822 | 517393 ms | session: 0f629097-a2cc-4427-9c75-8c4e2aff330c
