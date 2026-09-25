# Consultation to Claude Code (opus, effort high) -- 20260926-023039-62719-consult-score-all-options

Question asked:
```
# CONSULT — score EVERY candidate optimization 1–100, on a corrected fact base

You advised this lab earlier (`H:\exl3-lab\logs\advisor-20260925-191426.md`). Since then **two of our own
audits invalidated the numbers your plan rested on**, one of your ranked items was **measured and lost**, and
we found several errors in our own record. Read the material below before scoring, then produce **one
consolidated table** with a 1–100 effectiveness score per item.

## Read first (read-only lab access; cite file:line in your answer)

* `H:\exl3-lab\logs\brief27\AUDIT-B-answer.md` — **does our performance decomposition close?** (short answer:
  no, 10.6% is unaccounted, and the "MMA8 verify ≈70 ms" number your plan used is a **whole-model
  `llama-bench -p 4` forward on a lab binary that contains no MMA8 at all**).
* `H:\exl3-lab\logs\brief27\AUDIT-C-answer.md` — your code-reading claims checked against the **deployed**
  source. Almost all TRUE there, FALSE in the lab tree; two of your supporting statements are wrong
  (`M=2..8 → GROUP2`; `xh` is **global pool**, not shared memory).
* `H:\exl3-lab\logs\brief21b\REPORT.md` — **K=6 (your rank 2) was implemented and lost**: SASS −12.3%,
  LDS 136→32, registers 112→108, and decode went **15.78 → 15.16 tok/s (−3.9%)**; the loop is
  **memory-bound**, so instruction reduction bought nothing.
* `H:\exl3-lab\logs\brief26\REPORT.md` (**including the ADDENDUM at the end**) — Step 0 synced the lab tree
  to the deployed source (`exl3.cu` was the only differing file), `EXL3_FAST_EXTRACT` is now a compile-time
  macro with two cache-identical builds, and a deployed-cubin SASS tally exists (K3 loop 711 → 486 with the
  flag, REG 142 → 102; MMA8 loop measured twice with **unreconciled** normalisations: 234 instr/14.63 per
  weight vs 658 instr/20.6 per weight — treat any per-weight SASS figure as ±2× until reconciled).
* `H:\exl3-lab\STATUS-NOW.md` §2, §8, §9, §10 · `H:\exl3-lab\KNOWN-ISSUES.md` (KI-1…KI-8 + 測定と推論の規律).

## The anchors that ARE solid (all from lab kernels; use these to calibrate, not the dead numbers)

* no-draft decode **46.68 ms/token**; MTP k=3 **28.76 ms/token = 34.77 tok/s**, acceptance 0.68327 (343/502),
  mean commit 3.04; within-run speedup **1.60–1.62×**.
* Deployment (128k ctx, vision on CPU, MTP k=3, KV q4_0, `-fa on`, MMA8 flag on) decodes **32.52 tok/s @6k
  → 19.60 @71k** with matched prompts; prefill 345 → 248 tok/s.
* Weights 12.7 GB EXL3 3.0 bpw cb=mul1; V100 16 GB, driver 472.12, sm_70, ~900 GB/s theoretical, and we
  measured ~272 GB/s equivalent at 46.68 ms/token. Wrapper: `GGML_CUDA_FA_ALL_QUANTS=OFF` in the deployed
  build (the lab build differs — a config delta nobody has tested).
* **The metric that decides everything is still unmeasured**: in-server **draft mean vs verify mean vs round
  time**. #29 is now adding `llama_synchronize(ctx_tgt)` before the verify-timer stop and running
  `EXL3_SPEC_PHASES=1` with a 512-token greedy probe so the sum closes in one run.

## Score ALL of these (add any we missed — that is the most valuable part of your answer)

A. MMA8-fast: constant-shift decode inside the MMA8 verify loop, fp16 `xh` (drop the per-weight
   `__float2half`), register prefetch (the split-K kernel already has it), 8 warps/block instead of 4.
B. fix A standalone (constant-shift in the M=1 path only). We estimated ≤2% because it touches ~1/64 of the
   body — does the "K3 loop 711 → 486 (−31.6%)" SASS change that?
C. K=6 specialised head — **already measured: no gain** (see brief 21b).
D. Load-time prepack of a K=3 layout (you rejected it; audit E says the "1.94×" that supported part of that
   judgement came from **v100-skinny**, not EXL3/QLoRA, and may hide an untested idea).
E. Launch fusion: fold `had_in` into the prologue, epilogue via a last-block counter (your item D — 3 launches →
   1). Nobody has counted kernels per token in this fork yet.
F. Check that CUDA graphs are actually active for M=4 verify (`GGML_CUDA_DISABLE_GRAPHS=1` A/B).
G. Register double-buffering (software pipelining) in the MMA8 staging loop (`__ldcs` → `__syncwarp` → decode
   is serial today; `cp.async` does not exist on sm_70).
H. Raise MMA8 occupancy (4 warps/block, REG 95 today).
I. Keep `xh` in fp16 end-to-end (the loop converts per weight per k-tile).
J. **"Split N, not K"** — share activations rather than output rows across the verified tokens.
K. Compact draft head: requantise a truncated vocabulary (`bench/draft-vocab-32k.json`) to q4_0 (~90 MB) and
   run it on stock MMVQ; verification keeps the exact 6-bit head so outputs are unchanged.
L. 32k draft-vocabulary limiting (the hook exists in source, the lab binary lacks it).
M. Deeper speculation / trees now that multi-row verification exists (k=4/5/7, DFlash2's 4.09 commits/round).
N. Re-open DFlash2 *after* MMA8-fast (it loses only on verify-row cost).
O. n-gram / lookup drafter — a re-test is running right now on the deployed binary with thinking off and three
   sanity gates; earlier verdict was likely a harness artifact.
P. Port the Volta attention fork (jusko) — closed as "q8_0-KV only, 16–24 h"; audit E says our long-context
   decode drop (32.5 → 19.6 tok/s) is exactly what it targets.
Q. Deployment-vs-lab config parity (`FA_ALL_QUANTS=OFF`, flag set, same prompts) — there is still **no matched
   A/B** for our "deployment is 10–12% slower" claim.
R. The measurement itself (in-server synchronized phases) — score it as an item: it is the only thing that can
   turn the remaining estimates into numbers.
S. Anything else: persistent kernels, L2 residency (we believe sm_80+ only — confirm), batch/MB tricks, KV
   compression at the same quality, `-ngl`/`-ot` splits, draft-model CPU offload, anything we have not listed.

## Output format (exactly this)

1. **Table**: `item | what changes concretely | expected effect (tok/s @128k or ms/round, with units) |
   evidence class (MEASURED-BY-US / MEASURED-BY-OTHERS / INFERENCE) | cost (hours) | SCORE 1–100 |
   cheapest falsification test`.
2. **Calibration note**: state what 100 and 50 mean in your scale, and the roofline ceiling you are scoring
   against (derive it from the anchors above).
3. **Today's order of work** (≤3 items) and **what to delete from the plan**.
4. **Your own corrections**: which numbers in your earlier directive you now withdraw.
5. **The missing idea**: at most two candidates we did not list, with the cheap test for each.

Constraints: **same quality, same capacity** (no requantisation, no KV type change, no larger VRAM budget);
one V100, one GPU lock at a time; sessions are ~1–1.5 h of agent time each, so cost estimates must be honest.
Tool budget: ~30 calls, then answer.
```

## Answer

# Consultation 4: every candidate scored 1–100, rebuilt on the corrected facts

**Headline:** the audits leave two numbers that matter most:

1. **Draft cost per round is 8.14–11.0 ms**, and most of it is reading the 0.954 GB head three times.
2. **Verify is latency-bound, not instruction-bound.** Its kernels don't keep enough bytes in flight.

Brief 21B's null result (K=6), the register tallies and the grid arithmetic all point the same way. The best remaining work therefore adds bytes in flight or removes bytes. Cutting instructions scores low.

Separately, two things were missing from the plan entirely:
- **K=4 tensors never take MMA8.** They make up about 30% of the verify matmuls by count.
- **Target sampling runs on the CPU over the full 248k vocabulary.** This is the first suspect for the missing 9.27 ms.

## 1. Table

Units: Δ is ms per MTP round. The lab round is 87.66 ms with 3.04 commits (`AUDIT-B-answer.md:55-65`). The ~128k round is about 177 ms. That figure is INFERENCE: I extrapolated the deployment's round time, 3.37/32.52 ≈ 104 ms at 6k and ≈141 ms at 71k (from `STATUS-NOW.md:33`, acceptance 0.79→0.59), roughly linearly. % gain = Δ/(round−Δ).

| item | what changes concretely | expected effect | evidence class | cost (h) | SCORE | cheapest falsification test |
|---|---|---|---|---|---|---|
| **R** measurement | Add `llama_synchronize(ctx_tgt)` before the verify-timer stop, then run `EXL3_SPEC_PHASES=1` with a 512-token greedy probe on the **brief26 `build-slow`** (deployed-identical kernels, `brief26/REPORT.md:212-215`) | 0 ms by itself. It closes draft + verify + remainder in one run, and every row below that is INFERENCE depends on it | — | 1.5 | **92** | Fails if draft + verify + remainder still leaves >5% of the round unexplained. In that case add `cudaEventRecord` around the sampler as well |
| **Q** parity A/B | Matched `llama-bench -p 4 -n 0 -r 5` and `-n 128 -p 0`, plus the 512-token server probe, on: deployed exe with MMA8 on and off; lab exe `07928d8`; `build-slow` | Direct: 0. Possible free **+5%**. The deployed build did 32.90 tok/s (`H:\exl3-local\STATUS.md:18`) and the lab build did 34.77, on **the same 343/502 acceptance stream**. The lab binary has **no MMA8 and no GROUP2**, yet its pp4 of 70.2 ms (`brief14/baseline.m4.json:44`) is below the deployed-era MMA8 figure of 79.0 ms (`H:\exl3-local\STATUS.md:143`). No kernel story is consistent with both until this is re-measured | MEASURED-BY-US (but unmatched) | 1 | **80** | If all four arms land within 2% at pp4 and tg, the "deployment is slower" claim dies and so does the config-delta hypothesis |
| **L** vocabulary limit on the **exact 6-bit head** (draft only) | Launch the head `gemv` over only the first B 128-column blocks: grid `dim3(B)` plus an `ld=n` stride parameter, because `exl3.cu:986` asserts a contiguous tensor. Restrict the draft argmax to `[0, 128·B)`. Verify keeps the full head, so **outputs are unchanged** and **no bytes or VRAM are added**. Granularity is forced to whole 128-column blocks because svh and the output Hadamard act per 128-column block; picking arbitrary token ids would amount to requantising | Head reads fall from 3×0.954 GB to about 3×0.12–0.35 GB per round → **−5.5 to −7.5 ms/round**, minus 1–3% of commits for tokens outside the kept blocks → **net +4…+7% short context, +2…+4% @128k** | INFERENCE on MEASURED-BY-US inputs: head bytes `bench21b.journal.txt:37`, draft 8.14–11.0 ms | 4–6 | **70** | No GPU needed: build the coverage-vs-B curve from **Japanese and English** operator outputs. The file is misnamed: it holds **5,797 ids** covering 100% at N=8192 on 60k tokens, dense below ~20k, plus ~45 outliers up to 247,438 (`bench/draft-vocab-32k.json:1`). Kill if 99% coverage needs >50% of the 1,940 blocks |
| **NEW-1** MMA8 for **K=4** | Template `exl3_mma8_splitk_kernel` on K (staging `2*8*K` words, `exl3_window<K>`) and widen the gate at `exl3.cu:1018` to `K==3‖K==4`. Today K=4 at rows=4 falls to `dense2` (`:1027`): it decodes twice (two row pairs), and its K=3 sibling needs 168 registers, which means **1 block/SM** | Per forward: about **206 K=4 vs 480 K=3** matmuls at rows=4 (`bench21b.journal.txt:22,28`, ÷10 captures). The K=3 switch from GROUP2 to MMA8 took 97.3 → 79.0 ms (`H:\exl3-local\STATUS.md:142-143`). By analogy: **−3 to −10 ms/round → +4…+13% short, +2…+6% @128k** | MEASURED-BY-US analog + INFERENCE | 5–8 | **66** | ① Census of bytes by K from the GGUF tensor list (no GPU, 15 min); kill if K=4 is <15% of verify bytes. ② With only the gate changed, pp4 must fall ≥3 ms |
| **NEW-2** fill the GPU with the MMA8 grid | `exl3_gemv_splits` targets about 240 blocks (`exl3.cu:958-967`). That target suits the 256-thread split-K kernel, but MMA8 uses 128-thread blocks (`:1022`). The result is about 3 blocks, i.e. about 12 warps, per SM, against a register limit of 5 blocks (20 warps) at REG 95. Sweep an MMA8-only target of 480/960 | By Little's law: ~13 warps × 192 B × 80 SMs ≈ 200 KB in flight, against ≈ 415 KB needed at 830 GB/s × ~0.5 µs. That caps loads at ~400 GB/s. Estimate **0 to −6 ms/round, 0…+7%** | INFERENCE (arithmetic on `brief26/REPORT.md:202`, `AUDIT-C-answer.md:25-30`) | 1–1.5 | **62** | pp4 A/B on `build-slow`. Drop it if no target beats 240 by ≥2% (3× the A/A spread) |
| **G** register prefetch in the MMA8 staging loop | Copy split-K's `uint4 pf[PF]` pipeline (`exl3.cu:264-296`) into MMA8, so load(k+1) overlaps decode(k) | Doubles bytes in flight per warp → **−2 to −10 ms/round, +2…+13%** | INFERENCE (split-K has it; the K=6 null says the bottleneck is bytes, not instructions) | 4–6 | **55** | Run the "load-only" MMA8 variant from the earlier directive first. If full ≈ load-only, G is the lever; if full ≫ load-only, the decode is, and G drops to ~20 |
| **S-bs** target-side backend sampling | Add `-bs` (`common/arg.cpp:2094-2099`). It is off in deployment (`start-server-exl3.ps1:65-67,99`); only the draft uses it (`brief14/...err.log:251`). Today the target moves up to 4×1 MB of logits to the host per round and runs `top-k → top-p → min-p → temp → dist` on the CPU over 248,320 entries (`:290`) | Top suspect for part of the unexplained 9.27 ms: **0 to −4 ms/round, 0…+5%**. Distribution unchanged; greedy stays argmax | INFERENCE | 0.5 | **50** | One probe each with and without `-bs`: compare eval time, acceptance and `sha256(content+reasoning)` using `prompt-nothink.json` (KI-1). Kill if Δ<1% or it breaks the speculative-decoding accept path |
| **P′** depth slope (not the port) | `llama-bench -p 4 -n 0` and `-n 32 -p 0` with `-d 0,32768,65536,114688` on the deployed exe (`n_depth` exists: `baseline.m4.json:38`) | 0 ms. It decides P. At 71k the round grows about 37 ms over 6k, while the q4_0 KV floor is a few ms | MEASURED-BY-US (deploy probes, confounded by acceptance) | 0.5 | **55** | Keep P closed if the M=4 attention slope is <0.15 ms per 1k tokens |
| **A** MMA8-fast bundle | Constant-shift decode + fp16 `xh` + prefetch + 8 warps/block | The value sits in G and NEW-2. The decode-cost part is 0 to −4 ms. **"8 warps/block" is a trap at REG 95**: 256 threads × 95 registers gives 2 blocks = 16 warps/SM, *fewer* than today's 20. It only pays at REG ≤ 85 | INFERENCE | 10–14 | **38** | Do G and NEW-2 first. Attempt the decode part only if load-only ≪ full |
| **H** raise MMA8 occupancy | — | The **grid**, not registers, limits occupancy today, so merge H into NEW-2 | INFERENCE | — | **45** (via NEW-2) | Same as NEW-2 |
| **O** n-gram / lookup drafter | Re-test running on the deployed binary with thinking off | Content-dependent. Plausibly +5–15% on copy-heavy agent turns, regressions on free-form text | MEASURED-BY-US (old result probably a harness artifact) | sunk | **30** | Its own three gates. Adopt only per request type, never globally |
| **E** launch fusion | Fold `had_in` into the prologue; epilogue via a last-block counter that sums the S partials in a **fixed order**, so output stays byte-identical | About 690 matmuls/forward × 2 fewer kernels. Verify is already graph-replayed, so the saving is GPU bubbles, not CPU launches: **−1.5 to −3 ms/round, +2…+3.5%** | INFERENCE | 8–12 | **30** | Run the brief26 KTIMING build and compare Σ(bins) with the pp4 wall time. If the gap is <3 ms/forward, drop E |
| **F** CUDA graphs for verify | — | **Mostly answered already.** `graphs reused = 165` (`brief14/...err.log:311`), and only **10** host dispatches at rows=4 across 180 rounds (`bench21b.journal.txt:31`). Verify replays. Draft steps are re-dispatched every step (540 = 3×180, `:29`): **0 to −1 ms/round** | MEASURED-BY-US (indirect) | 0.3 | **30** | One `GGML_CUDA_DISABLE_GRAPHS=1` run inside Q, which also gives an upper bound for E's CPU-side share |
| **M** deeper speculation | k=7 so that verify is rows=8 (MMA8). k=4/5/6 give rows 5–7, which fall to dense2 and should be avoided | Cumulative acceptance by position is 0.857/0.685/0.500 (`:314`); extrapolated, 7 drafts give about 3.9 commits. Today it costs ~+31 ms/round, a **net loss**. After L, roughly −1…+3% | MEASURED-BY-US + INFERENCE | 0.3 | **22** now, **35** after L | One flag run after L lands |
| **J** "split N, not K" | Finer n-tiles and no split-K, which removes the partial buffer and the epilogue | Overlaps with E, **−1 to −3 ms/round** | INFERENCE (source is v100-skinny, not EXL3) | 6–10 | **25** | Only if E's KTIMING shows the epilogue ≥2 ms/forward |
| **P** jusko attention port | Port a q8_0-only kernel to q4_0 (a KV type change is forbidden) | Maybe −18 ms @71k, −30 ms @128k (**+20% @128k**) if attention is really ~2× off | MEASURED-BY-OTHERS (36.47 tok/s @100k, q8_0) | 24–32 | **22** | Run P′ first |
| **K** q4_0 compact head | Requantise a truncated head (~90 MB) | Same saving as L, but draft numerics change and it **adds VRAM** (breaks the "same capacity" constraint) | INFERENCE | 5–8 | **18** | L dominates it; skip |
| **D** load-time K=3 prepack | Permute tiles so each lane can do a 16-B load | Only relevant as a bytes-per-load lever *after* G and NEW-2, 0…+5% | INFERENCE ("1.94×" = v100-skinny) | 8–12 | **15** | Only if G+NEW-2 leave MMA8 below 50% of 830 GB/s |
| **I** fp16 `xh` end-to-end | `had_in` also writes a half copy | `xh` is small and L2-resident. Removes F2F/LDG work that is not the bottleneck: 0 to −1.5 ms | INFERENCE | 3 | **15** | Only inside A |
| **B** fix A standalone | Constant shift in `<3,2,false>` | **No: 711→486 doesn't change the answer for MTP.** K=3 never runs at rows=1 under MTP (rows ∈ {2,4,28,32,42}, `bench21b.journal.txt:18-22`). Its real effect is REG 142→102, which crosses **1→2 blocks/SM** (`brief26/REPORT.md:159-162,225`), so it matters for **plain decode only**. GROUP2 goes 168→130, still 1 block/SM | MEASURED-BY-US (SASS/registers) + INFERENCE | 1 (built) | **12** | A tg128 A/B of `build-slow` vs `build-fast`, only if plain decode matters to the operator |
| **N** re-open DFlash2 | — | Its verify is rows > 8, i.e. the GEMM path, not MMA8. MMA8-fast doesn't help it; acceptance 0.446 | MEASURED-BY-US | 2 | **8** | Never, unless the GEMM path changes |
| persistent kernels | — | Verify is already graph-replayed | INFERENCE | 20+ | **6** | — |
| **C** K=6 head extraction | — | Measured: no gain. **But** it was measured on a binary running at 15.78 tok/s with a ~180 ms round, about 2× slower than baseline for unknown reasons (`bench21b.journal.txt:5,38`). The null direction still holds (draft −0.5 ms, verify +0.6 ms) | MEASURED-BY-US | — | **3** | — |
| L2 persistence | `cudaAccessPolicyWindow` is **sm_80+ only**, confirmed. V100's 6 MB L2 is irrelevant for a 12.7 GB stream anyway | 0 | MEASURED-BY-OTHERS (CUDA guide) | — | **1** | — |
| `-ngl`/`-ot` splits, CPU drafter | Anything over PCIe (~12 GB/s) loses; the CPU drafter measured 8.68 tok/s | negative | MEASURED-BY-US | — | **2** | — |
| KV compression | Excluded by the constraints | — | — | — | **0** | — |

## 2. Calibration

- **What the scores mean:**
  - **100:** ≥+10% deployed tok/s (or it decides ≥3 other rows), ≤2 h, output provably unchanged, high confidence.
  - **50:** expected ≈+3–5% for about a day, or a ≤1 h test with roughly a 1-in-3 chance of revealing ≥5%.
  - **<20:** don't schedule it.
  - The score is roughly P(works) × gain ÷ hours, plus information value.
- **Bandwidth roofline.** Per MTP k=3 round the GPU must read:
  - the verify weights once: 12.7 GB;
  - 3 × (head 0.954 GB + MTP layer ~0.17 GB): ≈3.4 GB.
  - Total ≈ **16.1 GB**. At a practical ~830 GB/s that is **19.4 ms/round → ~157 tok/s** at 3.04 commits.
  - Plain decode floor: 12.7/0.83 = **15.3 ms/token**. Measured: 46.68, i.e. 33%.
- **Practical ceiling.** An issue floor of about 27e9 weights × ~10 instructions/weight ÷ ~1.4e13 thread-instructions/s ≈ 19 ms per full forward. Combined with the bandwidth floor, that gives a practical ceiling of **~25–30 ms/round → ~100 tok/s** short-context.
- **Where we are.** Today 87.66 ms/round is about 22% of the bandwidth roofline and about 1/3 of the practical ceiling. The gap is **bytes in flight and occupancy**, not arithmetic. I scored against that ~100 tok/s ceiling.

## 3. Today's order of work, and what to delete

1. **One GPU lock, about 1.5 h: R + Q + S-bs + F + P′ together.** Run the synchronized spec-phases probe on `build-slow`. Then the matched pp4/tg runs on the deployed exe (MMA8 on/off), the lab exe and `build-slow`. Then one `-bs` probe, one `GGML_CUDA_DISABLE_GRAPHS=1` probe, and the `-d` depth sweep. This resolves the 70.2 vs 79.0 ms contradiction and places the 9.27 ms.
2. **No GPU, about 30 min:** the byte census by K from the GGUF (decides NEW-1) and the head coverage-vs-B curve on Japanese and English output (decides L).
3. **Then build one thing:** L if coverage holds, otherwise NEW-2 followed by NEW-1.

**Delete from the plan:**
- C and the "K=6 rank 2" line.
- B as a deployment item.
- I and "8 warps/block" as standalone steps.
- K (L dominates it).
- N.
- The 16–24 h jusko port, until P′ justifies it.
- `run19b.py`.
- Every lab-binary A/B of EXL3 kernels, including the brief21/21b/25 binary family, until its ~2× slowdown and its 0.615 acceptance are explained.

## 4. Numbers from my earlier directive that I withdraw

- **"~70 ms of the 87 ms round is M=4 verify through MMA8" and "~80%"** (`advisor-20260925-191426.md:94,140`). Wrong kernel, wrong harness, empty KV.
- **"17.5 ms draft, mostly K=6 head"** (`:94`). Replace with 8.14–11.0 ms (measured, unsynchronized).
- **"+20%, 34.8 → ~42 tok/s"** (`:140`) and **"K=6 head +5–10%"** (`:141`).
- **"M=2..8 falls back to GROUP2"** (`:92`). Rows 4/8 take MMA8. Rows 2/3/5/6/7 **and every K≠3 tensor at rows>1** take dense2. **I missed K=4 entirely**, and it is the larger oversight.
- **"8 warps/block" as unconditional** (`:148`). It lowers occupancy at REG 95.
- **Prepack "~0 for this kernel"** (`:143,168-176`). Downgraded to "deferred": in a latency-bound regime, the bytes carried per load instruction do matter.
- **"Launches: remove 2/3"** (`:180`). The saving is per matmul, only when S>1, and with verify already graph-replayed it saves GPU bubbles, not launch overhead.
- **Line ref `:290-321`** should be `:264-296`.
- On `xh`: my saved directive says "reload from **global** memory" (`:100`). The audit's "shared" charge doesn't match that text. `xh` is the global pool (`exl3.cu:992`).

## 5. The missing ideas

1. **NEW-1: MMA8 for K=4 at verify.** About 206 of the ~686 rows=4 matmuls per forward are K=4 and go through dense2: decoded twice, at about 1 block/SM (`bench21b.journal.txt:22,28`; `exl3.cu:1018,1027`).
   - **Test:** GGUF byte census by K (15 min, no GPU). If K=4 is ≥15% of verify bytes, template MMA8 on K. Change only the gate, then run pp4 A/B; it must be ≥3 ms faster.
2. **NEW-2: grid fill and bytes in flight, as the unifying model.** MMA8 launches about 240 blocks of 4 warps (~12 warps/SM, ~200 KB in flight vs ~415 KB needed). This single constraint explains the K=6 null, the 272 GB/s plain-decode figure and why cutting instructions doesn't pay.
   - **Test:** a 1-line MMA8-only split target of 480/960, then pp4 A/B on `build-slow`, 1 h. If nothing beats 240 by ≥2%, the latency model is wrong and G drops to about 20.

---
model: claude-opus-5-5 | turns: 23 | notional cost USD: 1.9969678 | 496360 ms | session: 46cece2c-d79b-49c2-90be-ae280715e85f
