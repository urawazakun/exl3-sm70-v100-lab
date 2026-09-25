# Brief 29 — finish Step 2, run Step 3: the numbers that decide MMA8-fast

Date: 2026-09-25 evening. Worktree: `H:\exl3-lab\worktrees\mma8fast` (branch
`brief26-mma8fast`, HEAD `5c6b615`). GPU lock held 20:46–~21:45 (90-min acquire,
`bash scripts/gpu-lock.sh acquire 90`, RC=0, no contention — lock was FREE).
`run19b.py` not run as-is. `fixA`/`k6`/`ngram2`/`engine/` untouched.
Builds are CPU-side; `--help` smoke only touches the CUDA driver read-only.

## 0. Step 2 finish — the one missing hook (no GPU)

REPORT §4 left `exl3_ktiming_sync()` defined-but-never-called (ntokens stuck at 0).
Wired in commit `5c6b615` ("brief29 step2: wire exl3_ktiming_sync() into
ggml_backend_cuda_synchronize", +16/-1 over 3 files):

- `engine/ggml/src/ggml-cuda/exl3.cuh`: non-static declaration of
  `exl3_ktiming_sync()` under `#if defined(EXL3_KTIMING) && EXL3_KTIMING`.
- `engine/ggml/src/ggml-cuda/exl3.cu`: `static` removed from the definition
  (external linkage for the ggml-cuda.cu call site); added missing
  `#include <cstdio>` / `<cstring>` (strncmp/snprintf/strcmp/fprintf).
- `engine/ggml/src/ggml-cuda/ggml-cuda.cu`, in `ggml_backend_cuda_synchronize()`
  (line ~2489, the single device-sync point) after `cudaStreamSynchronize`:
  `#if defined(EXL3_KTIMING) && EXL3_KTIMING / exl3_ktiming_sync(); / #endif`.
  Default builds (`EXL3_KTIMING` unset/0) compile to zero overhead — the call is
  preprocessor-deleted, and the declaration is hidden too.

Third build compiled from the same source (no GPU needed for the build):

- `build-ktiming`: `-DEXL3_FAST_EXTRACT=0 -DEXL3_KTIMING=1`, all other flags
  identical to build-slow/fast (CUDA 11.8, MSVC 14.34, `70-real`, Release, same FA
  quants). Configure log `logs/brief29/configure-ktiming2.log`
  (`CONFIGURE_KTIMING_STATUS=OK`), build log `logs/brief29/build-ktiming.log`
  (`BUILD_KTIMING_STATUS=OK`, `BUILD_RC=0`, only the pre-existing benign
  `common.cuh(613)` FP warnings).
- Smoke `--help`: RC=0, sees `Tesla V100-SXM2-16GB, cc 7.0`
  (`logs/brief29/smoke-ktiming-bench-help.txt`). The `load_backend: failed to
  find ggml_backend_init in .../ggml-cuda.dll (and ggml-cpu.dll)` lines are
  pre-existing (present in build-slow runs too) and harmless — the bench still
  runs on CUDA.
- Command deviation from REPORT §6a (reported per brief): §6a's one-liner
  `cmake ... -DCMAKE_CUDA_FLAGS="..."` fails in this shell (no VS env →
  `No CMAKE_C_COMPILER could be found`, see `configure-ktiming.log`). Used a
  `.bat` wrapper that calls `vcvars64.bat -vcvars_ver=14.34` first
  (`logs/brief29/configure-ktiming.bat`, `build-ktiming.bat`) — the same remedy
  brief 26 used for build-fast (`build-fast9/10.log`). Flags are §6a's otherwise.

## 1. SASS reconciliation — one agreed number per kernel (no GPU)

Toolchain (KI-8): `C:/llm-local/cuda118-extract/cuda_nvdisasm/nvdisasm/bin/nvdisasm.exe`
(`-c` on the extracted cubin); native tools given `H:/...` paths. Source under test:
deployed `ggml-cuda.23.sm_70.cubin` disassembly `logs/brief26/deployed-cubin23.sass.txt`.
Loop definition (directive): target of the backward `BRA` → that `BRA`, PC-range
static count (`sass_tally.py`, brief-8 method).

### 1a. K3-false `exl3_gemv_splitk_kernel<3,2,false>`: AGREED 711, 11.11/weight

- Section `.text._Z23exl3_gemv_splitk_kernelILi3ELi2ELb0...` (file lines 99814+).
  k-tile loop: label `.L_x_1973` = PC **0x12a0** → back-edge `BRA.CONV` PC **0x3f00**
  (`L_x_1974`). Tally `0x12a0-0x3f00` inclusive = **711 static**.
- A naive label-to-label `sed -n` count gives 714: the 3 extra are the post-loop
  warpsync setup at 0x3f10 `MOV` + 0x3f20 `MOV` + 0x3f30 `CALL warpsync` (visible
  at file lines 100844–100848), which the PC-bounded tally correctly excludes.
- Weights/iteration from the source (not a guess): the k-row body is
  `for (t = 0; t < 8; ++t)` tiles × `for (j = 0; j < 8; ++j)` weights
  (`exl3.cu:439,504`), i.e. **64 weights/lane/k-row**. Cross-check in SASS:
  HADD2=64, IDP=64, SHF=64, LOP3=64 — the cb2 decode sequence emits exactly one
  of each per (t,j). **711/64 = 11.11 per weight.**
- Histogram (711): SEL 160, FFMA 128, IMAD 66, SHF 64, LOP3 64, IDP 64, HADD2 64,
  ISETP 57, LDS 24, P2R 12, LEA 4, LDG 2, BRA 1, NOP 1. REG/thread 142 (res-usage).

### 1b. K6-false `exl3_gemv_splitk_kernel<6,2,false>`: AGREED 586, 9.16/weight

- Loop `.L_x_1804` PC **0x1560** → `BRA.CONV` PC **0x39f0** (`L_x_1805`).
  Tally `0x1560-0x39f0` inclusive = **586 static** (`tally-k6f0-loop.txt`).
- Weights/iteration: same 8-tiles × 8-weights structure (K=6 tiles are 48 words,
  windows sit in fixed word pairs); LOP3=64 confirms 64 weights. **586/64 = 9.16.**
- Histogram (586): FFMA 128, HADD2 64, IDP 64, SHF 64, LOP3 64, LDS **128**,
  IMAD 65, LEA 4, LDG 2, MOV 1, BRA 1, NOP 1. REG/thread 112.

### 1c. MMA8 `exl3_mma8_splitk_kernel<2>`: AGREED 234, 14.63 per MODEL weight

- Section file lines 866–1660. The k-tile loop (`for (kt = k0; kt < k1; ++kt)`,
  `exl3.cu:600`) is label `.L_x_18` = PC **0x15c0** → back-edge
  `@!P0 BRA (.L_x_18)` at PC **0x2450**. Tally `0x15c0-0x2450` inclusive = **234
  static** (`tally-mma8-loop.txt`), histogram: IMAD 22, SHF 16, LOP3 16, IDP 16,
  HADD2 16, PRMT 16, F2F 32, FFMA 16, HMMA 16, LDG 18, LDS 29, STS 2, ISETP 5,
  SEL 0, plus IADD3 3, LEA 17, CS2R/BMOV/BSSY/BSYNC/BRA/NOP. REG/thread 95.
- Weights/iteration from the source: the B j-loop (`exl3.cu:610-621`) decodes
  `j = 0..7` pairs `(rr0, rr1)` = **16 B-weights** (8 `b[j]` u32 = 8 PRMT; SASS
  shows 7 pre-`1ff0` PRMT + 1 at 0x23b0 = 8). The A j-loop (`exl3.cu:628-631`)
  builds 8 `a[j]` u32 = **16 A-values** (8 PRMT in the 0x2000–0x2270 block).
  The 32 `F2F` in the loop = 16 (B model weights) + 16 (A **activations**).
- The brief's alternative "658-instruction loop = 20.6/weight (/32)" is
  **withdrawn as a loop tally** (kept here as the historical record the brief
  asked to delete): no PC range inside MMA8\<2\> yields 658 instructions
  containing all 32 F2F. Measured decomposition of the 768-instruction function
  body: prologue 0x0000–0x15b0 = 348 + loop 0x15c0–0x2450 = 234 + epilogue
  0x2460–0x2f80 = 179 + post-RET trap/NOPs = 7 → 768 total. 234+179 = 413,
  348+234 = 582 — 658 matches nothing. (`.L_x_25`, cited by STATUS §9 as "the
  loop", is a post-`RET` self-`BRA` trap at file lines 1652–1653, not a loop.)
- Normalisation decision: activations are not weights. **234/16 = 14.63 per
  MODEL weight decoded** (REPORT §3's number stands). 234/32 = 7.31 per F2F op
  is recorded here only so the two tallies can be compared without confusion.
  The 15%-of-full-run and ≥25%-SASS falsification inputs below use 14.63.

### Reconciled SASS table (deletes the 658/20.6 loop number from the record)

| kernel (deployed, cb=2) | loop PCs | static | weights/it (source) | per weight | REG |
|---|---|---|---|---|---|
| `gemv_splitk<3,2,false>` | 0x12a0–0x3f00 | 711 | 64 (8t×8j; HADD2=IDP=64) | **11.11** | 142 |
| `gemv_splitk<6,2,false>` | 0x1560–0x39f0 | 586 | 64 (same nest; LOP3=64) | **9.16** | 112 |
| `mma8_splitk<2>` | 0x15c0–0x2450 | 234 | 16 B-weights (8 pairs; 8 PRMT) | **14.63** | 95 |

`build-fast` A/B effect from brief 26 (unchanged, diagnostic only): K3-false loop
711 → 486 (−31.6%), REG 142 → 102. Whether that buys wall-clock is Step 3's
question — see §4: the fast-variant kernel bin was NOT measured (no fast+ktiming
build exists), so falsification condition (ii) is unresolved.

## 2. Step 3 measurements (one lock, GPU)

All bench arms: `EXL3_EXPERIMENTAL_MMA8=1`,
model `H:\exl3-lab\models\exl3-gguf\huihui-qwen3.8-27b-abliterated-exl3-3bpw-mtp.gguf`,
`-ngl 99 -sm none -mg 0 -fa on -b 2048 -ub 512 -ctk q4_0 -ctv q4_0 -t 6`.
Deviations from REPORT §6 (reported per brief): §6c says "deployed binary", but
`H:\exl3-local\build-v100\bin\` ships **no `llama-bench.exe`** (only server/cli),
so A/A + KTIMING + NULL bench arms used `build-slow` (synced source; brief 26 §3
proved its cubin-23 SASS byte-identical tallies 711/586/234 to deployed).
§6f `-d 0,32768,65536`: only `-d 0` run (`decode-ref.txt`, 21.02 ± 0.03 —
matches the A/A band, depth sweep not run for time). §6g `run19b.py`: not run
as-is per the brief's own rule; equivalent driver written for the ktiming build
(`logs/brief29/run29srv.py`, `run29srv2.py`, BIN pointed at `build-ktiming`).
§6h ncu: not attempted (advisor: bins suffice; 1-h cap better spent on NULL).

### 2a. A/A noise floor first (build-slow, graphs default ON) — §6c

| arm | run 1 | run 2 | spread |
|---|---|---|---|
| `-n 128 -p 0 -r 5` (tg128) | 20.99 ± 0.06 | 20.81 ± 0.34 | 0.18 tok/s (**0.9%**) |
| `-n 0 -p 4 -r 5` (pp4) | 53.60 ± 5.93 | 50.98 ± 7.24 | ~5% (pp4 is noisy; use tg for f) |

Logs: `logs/brief29/aa1-tg.txt`, `aa1-pp.txt`. Every later number is compared
with this: tg noise ≈ 1%, pp4 noise ≈ 5–7%.

### 2b. KTIMING bins (build-ktiming, `EXL3_KTIMING=1 GGML_CUDA_DISABLE_GRAPHS=1`) — §6d

Overhead report (same binary, graphs OFF, timing off vs on): **15.94 ± 0.01 vs
15.94 ± 0.01 — 0% overhead** (`ov-off.txt` vs `kt-tg.txt`).
Graphs effect (timing off): graphs ON (A/A) 20.9 vs graphs OFF 15.94 → CUDA
graphs are worth **+31% decode throughput** here; the timing build sacrifices
them by design.

tg128 arm (`kt-tg.txt`, 15.94 ± 0.01 tok/s, 8977 syncs):

| K | cb | rows | kernel | had_in_ms | main_ms | epilogue_ms | matmuls | kern/tok |
|---|---|---|---|---|---|---|---|---|
| 3 | 2 | 1 | splitk1 | 18335.186 | 19289.638 | 18342.168 | 256400 | 28.56 |
| 6 | 2 | 1 | gemv | 2.979 | 1620.170 | 0.000 | 641 | 0.07 |

pp4 arm (`kt-pp.txt`, 48.67 ± 2.48 tok/s, 87 syncs) — **this is the M=4 verify
shape** (rows=4 → `mma8`):

| K | cb | rows | kernel | had_in_ms | main_ms | epilogue_ms | matmuls | kern/tok |
|---|---|---|---|---|---|---|---|---|
| 3 | 2 | 4 | mma8 | 225.826 | 240.979 | 225.903 | 2400 | 27.59 |
| 6 | 2 | 1 | gemv | 0.030 | 15.881 | 0.000 | 6 | 0.07 |

Kernels/token (decode): 256400 matmuls / 640 tokens = **~401 EXL3 matmuls/token**
(×3 kernels ≈ 1200 launches/token — the launch-fusion motive in advisor §5).
K6 head: ~1 matmul/token at ~2.5 ms (aliased scale; de-aliased ≈ 1.5 ms).

**Bin-validity caveat (measured, not assumed):** bin sums exceed run wall-clock
(tg: 57.6 s bins vs 40.2 s wall; pp4: 0.71 s vs 0.41 s) by 1.4–1.7×. Cause:
the 128-event round-robin pool aliases — ~28.6 matmuls queue per sync vs the
~21 the pool supports (REPORT §4's own bound). So **absolute bin ms overcount**;
**ratios within a run are the usable signal**. f1/f4 via `sum(bins)/token-time`
are therefore INVALID (>100%) and are not reported as f — f comes from NULL (§3).

Verify-bin ratios (pp4, M=4, the number that ranks MMA8-fast): of EXL3 time,
**mma8-main 34.8% / had_in 32.6% / epilogue 32.6%** (240.979 / 225.826 / 225.903
of 692.7 ms). The MMA8 kernel itself is about **one third** of the verify-matmul
time; had_in + epilogue are about **two thirds**.

### 2c. NULL_MATMUL floor (separate build, `llama-bench` only) — §6e

`EXL3_NULL_MATMUL` is compile-time, so a fourth build was configured and compiled
(`build-null`, `-DEXL3_FAST_EXTRACT=0 -DEXL3_NULL_MATMUL=1`,
`logs/brief29/build-null.log`, `BUILD_NULL_STATUS=OK`) with CUDA DLLs copied in.
Garbage-output knock-out, bench only, graphs default ON (same as A/A):

| arm | base (A/A mean) | NULL | f = 1 − t_null/t_base |
|---|---|---|---|
| tg128 | 20.90 tok/s (47.85 ms/tok) | **153.27 ± 3.13 tok/s** (6.525 ms/tok) | **f = 0.864 (86.4%)** |
| pp4 | 52.3 tok/s (19.1 ms/call-tok) | **364.10 ± 120.53 tok/s** | **f4 ≈ 0.86** (pp4 noisy, same ratio) |

Logs: `null-tg.txt`, `null-pp.txt`. NULL still pays `cudaMemsetAsync` + launch
scaffolding, so 86% is the share of token time inside the three EXL3 kernels
including their launch gaps — the Amdahl f the advisor asked for, measured.

Cross-check vs §2b: bins-vs-NULL disagree by construction (bins overcount via
aliasing, §2b caveat) — per the directive this means launch gaps/serialisation
cannot be separated from kernel time here; kernels/token are counted above
(~401 matmuls/token decode; pp4: 2400/20 = 120 matmuls/prompt-token).

### 2d. Server arm → f_round (MTP k=3, port 8083/8084, KTIMING build) — §6g

Three consecutive 256-token probes (`prompt-nothink.json`, KTIMING=1, graphs
attempted-off) on port 8084 (`srv-kt2/`, driver `run29srv2.py`):

- req0/1/2 eval: 23.00 / 22.66 / 20.33 tok/s; prompt eval 55.97/19.60/17.81 tok/s.
- Draft acceptance **0.64615 (168/260)**, mean len **2.93**, byte-identical across
  all three runs. Payload honesty (KI-1): `content` 0 chars (`e3b0c442…` = NO
  DATA, thinking consumed the 256-token budget — same artefact as brief 21B);
  `reasoning_content` 1007 chars sha `04c89bf3…` identical all three runs.
- `graphs reused = 85/169/253`: the server path **ignores
  `GGML_CUDA_DISABLE_GRAPHS`** (it gates only `common.cuh` bench graphs; the
  server reuses graphs increasingly across requests). So this arm measured the
  graphs-ON server, not the graphs-off server the directive assumed.
- **No KTIMING table was recovered from the server**: the table prints via
  `atexit`, and `proc.terminate()` does not run DLL atexit handlers
  (`server.err.log` contains zero `ktiming` lines; `grep -c ktiming = 0`).
  This is a harness failure with log evidence, counted per the brief as data:
  the hook works (bench tables print), the server readout path needs a graceful
  `/unload`-then-exit or a signal handler, not terminate.

**f_round, prominently, with inputs and honesty labelling:**

- Deployed round (STATUS-NOW §2, lab baseline): 28.76 ms/token × 3.04 = **87.4 ms
  per round**, of which the advisor attributes ~70 ms to M=4 verify.
- Measured here: f = 86% (NULL, §2c) and verify-bin split mma8/had_in/epilogue
  ≈ 35/33/33% (§2b). Applied to the deployed round as an ESTIMATE (not a bin
  measurement — the server bins were lost as above): EXL3 ≈ 0.86 × 87.4 ≈
  **75 ms**; of which **mma8-main ≈ 26 ms, had_in ≈ 25 ms, epilogue ≈ 25 ms**.
- **f_round ≈ 75/87 ≈ 0.86 (EXL3 share of the round)**; the MMA8 kernel proper is
  ≈ 26/87 ≈ **0.30 of the round**. The single number that ranks MMA8-fast:
  even a perfect MMA8-decode fix touches ~30% of the round, while had_in +
  epilogue (≈ 50 ms, ~57%) can only be addressed by launch fusion, not by
  extraction. Estimate inputs: deployed 87.4 ms round, NULL f = 0.864,
  pp4 bin ratios 34.8/32.6/32.6. Caveat: bin ratios come from the
  graphs-OFF bench pp4 arm, and absolute bin ms overcount (aliasing); the
  ratios are the robust part.

## 3. Falsification conditions (values only, no verdict — advisor §3)

(i) Load-only/knock-out within 15% of full run (= memory-bound ⇒ abandon
extraction)? **Knock-out value: t_null/t_base = 6.525/47.85 = 13.6%**
(tg128, graphs ON both arms). The full EXL3-off run is 7.3× faster, i.e. NOT
within 15% — at matmul granularity the workload is not memory-bound. NOTE: this
is the full knock-out (loads removed too), not the advisor's load-only variant
(same loads, decode → XOR-sum), which was NOT built — so the "load-only" half
of condition (i) is untested; a load-only build is ~1 h (same pattern as
build-null, replace decode with XOR-sum into `b[]`).

(ii) SASS cut ≥25% with binary gain <10% (= instructions not the limiter)?
**SASS cut measured: −31.6%** (K3 loop 711→486, brief 26, same-kernel same-range
same-flags). **Binary gain: UNMEASURED** — no fast+KTIMING build was compiled
and no fast-vs-slow kernel-bin A/B was run. Condition (ii) is therefore
**unresolved**, not passed or failed. Cost to resolve: one more build
(`-DEXL3_FAST_EXTRACT=1 -DEXL3_KTIMING=1`) + the §2b protocol ≈ 2 h build +
30 min GPU.

## 4. Box health / restore (done) — §6i + task C

- Deployment 8081 was stopped only while holding the lock (PID 13708 via
  `taskkill /IM` — note: `/F` is blocked by this session's tool policy, plain
  terminate worked; GPU verified free via `nvidia-smi`, `/health` refused as
  expected before benching).
- Restored 21:36 via `restore_deployment` (lib-bench.sh → hidden-window
  `start-server-exl3.ps1`): `/health` = 200, VRAM 14843 MiB (expected 128k-ctx
  figure), then `probe_decode` on 8081: **32.85 tok/s, 512 tokens, acceptance
  0.68327 (343/502)** — inside the HARNESS verification band (32.66–34.44 tok/s,
  0.66667–0.68327). Operator's model answers: `reasoning` 1661 chars
  (sha `b265cae8…`); `content` empty (`e3b0c442…` = KI-1 NO DATA as usual, not a
  regression — the 512-token budget is consumed by thinking).
- Lock released 21:36 (`LOCK_RELEASED`, status FREE). Total hold 20:46–21:36 =
  50 min, inside the 60–75 min budget. **No `RESTORE-NEEDED` marker** (restore
  done by this session; KI-5/KI-5b path not triggered). Log:
  `logs/brief29/restore-probe.json`, `check-probe.py`.

## 5. Exact commands run (audit trail)

```
# instrumentation (no GPU)
git -C H:/exl3-lab/worktrees/mma8fast log --oneline -3        # branch brief26-mma8fast
  # (+16/-1 patch over exl3.cu / exl3.cuh / ggml-cuda.cu, §0)
git add ... && git commit -m "brief29 step2: wire exl3_ktiming_sync() ..."
H:\exl3-lab\logs\brief29\configure-ktiming.bat               # vcvars + cmake, EXL3_FAST_EXTRACT=0 EXL3_KTIMING=1
H:\exl3-lab\logs\brief29\build-ktiming.bat                   # cmake --build -j6 --target llama-bench llama-server
H:\exl3-lab\logs\brief29\smoke3.bat                          # llama-bench.exe --help (RC=0)
H:\exl3-lab\logs\brief29\cfg-null.bat                        # configure+build build-null (NULL_MATMUL=1, bench only)

# GPU (one lock)
bash scripts/gpu-lock.sh acquire 90
taskkill /IM llama-server.exe                                # stop 8081 (no /F: tool policy)
H:\exl3-lab\logs\brief29\aa1-tg.bat / aa1-pp.bat             # A/A, build-slow, EXL3_EXPERIMENTAL_MMA8=1, twice each
H:\exl3-lab\logs\brief29\kt-tg.bat / kt-pp.bat               # KTIMING=1 + GGML_CUDA_DISABLE_GRAPHS=1
H:\exl3-lab\logs\brief29\ov-off.bat                          # timing OFF, graphs OFF (overhead)
H:\exl3-lab\logs\brief29\null-tg.bat / null-pp.bat           # build-null, EXL3_EXPERIMENTAL_MMA8=1
H:\exl3-lab\logs\brief29\decode-ref.bat                      # -d 0 sanity (21.02 ± 0.03)
python H:/exl3-lab/logs/brief29/run29srv.py                  # server MTP k=3 :8083 (1 probe + failed bench-p*)
python H:/exl3-lab/logs/brief29/run29srv2.py                 # server MTP k=3 :8084 (3 probes, graphs-ON caveat)
```

SASS audit commands (all `H:/` paths, 11.8 nvdisasm): `grep -n` section bounds,
`awk NR>=a&&NR<=b` PC-range counts, `sass_tally.py` tallies (brief26 artefacts
reused, not re-run with new bounds except the 711-vs-714 and 234-vs-658 checks
above — raw `grep`/`awk` counts quoted inline).

## 6. Failures with log evidence (count as session success per brief)

1. §6a one-liner configure fails in this shell (no VS env) — `configure-ktiming.log`;
   fixed with vcvars wrapper (same as brief 26's build-fast fix).
2. `H:\exl3-local\build-v100\bin\` has no `llama-bench.exe` — §6c "deployed
   binary" arms used SASS-identical `build-slow` instead (`aa2-*.bat` attempt
   logged RC=9009).
3. KTIMING absolute bin ms overcount 1.4–1.7× (128-event pool aliases at
   ~28.6 matmuls/sync > ~21 capacity) — bins usable as ratios only; f comes
   from NULL instead.
4. Server KTIMING table lost (atexit + terminate) — `grep -c ktiming = 0` on
   `srv-kt*/server.err.log`; f_round is an estimate with stated inputs.
5. Server ignores `GGML_CUDA_DISABLE_GRAPHS` (`graphs reused = 85/169/253`) —
   bench-only env gate; server arms are graphs-ON regardless.
6. `content` empty in all server probes (KI-1 repeats: thinking ate the
   256-token budget); identity carried by `reasoning_content` (1007 chars,
   sha `04c89bf3`, 3/3 identical).
7. `llama-bench` inside `run29srv.py` failed (rc=1, `failed to load model` —
   bench cwd/DLL split when driven beside the server; the server probe itself
   succeeded). Bench numbers come from the direct `.bat` arms instead.
8. Build was slow: build-ktiming took ~2 h wall (link-heavy tail), build-null
   ~35 min; both backgrounded with notify.

Artifact inventory (`H:\exl3-lab\logs\brief29\`): REPORT.md (this file),
WORKING-NOTES.md, configure-ktiming.bat/.log/2.log, build-ktiming.bat/.log,
cfg-null.bat, build-null.log, smoke*.bat/.txt, aa1-tg/pp.bat/.txt,
kt-tg/pp.bat/.txt, ov-off.bat/.txt, null-tg/pp.bat/.txt, decode-ref.bat/.txt,
run29srv.py, run29srv2.py, srv-kt/, srv-kt2/ (driver.log, decode*.json,
server.out/err.log, bench-p*.txt), build-brief26-brief29.bat (script copy).
Worktree commit: `5c6b615` (Step-2 wiring) on branch `brief26-mma8fast`.
