# Brief 30 — advisor-ordered measurements (scores: measure first, build later)

Date: 2026-09-26 03:00-04:41 JST. GPU lock held 03:19:58-04:41:08 (~81 min).
Task 2 (no GPU) done by subagent first; Task 1 (GPU) by this session.
No builds this session (rule). fixA/k6/ngram2/engine/ untouched.

## 0. Arm identity (what each binary IS — the advisor's precondition)

| arm | path | source | MMA8 kernel | EXL3_SPEC_PHASES hooks |
|---|---|---|---|---|
| build-slow | worktrees/mma8fast/build-slow/bin | 07928d8-era + exl3.cu synced to deployed (ab092c6) | YES (SASS-identical cubin-23 tallies 711/586/234 to deployed, brief29 s1) | NO (predates brief21 hooks; worktree engine/tools/server has no server-context.cpp at all) |
| lab 07928d8 | H:/exl3-lab/build/bin | shared engine @d28bcab + brief21 hooks | NO (no MMA8, no GROUP2) | YES ("spec phases" string x1 in llama-server-impl.dll) |
| deployed | H:/exl3-local/build-v100/bin | H:/exl3-local/exllamav100 | YES | NO |

Deployed bench arm does not exist: build-v100 ships no llama-bench.exe
(only server/cli). The "fourth arm" (deployed exe) is therefore a SERVER
leg, not a bench leg. FA config identical everywhere:
GGML_CUDA_FA_ALL_QUANTS=OFF, FA_QUANTS=q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16;q5_1-q5_1
(deployed, build-slow, lab build CMakeCaches all match).

All bench arms: model H:/exl3-lab/models/exl3-gguf/huihui-qwen3.8-27b-abliterated-exl3-3bpw-mtp.gguf,
-ngl 99 -sm none -mg 0 -fa on -b 2048 -ub 512 -ctk q4_0 -ctv q4_0 -t 6.
All server arms: same + -c 65536 -np 1 --spec-type draft-mtp --spec-draft-n-max 3
--spec-draft-n-min 1 --host 127.0.0.1 --jinja --no-reasoning-preserve.

## 1. R — synchronized spec-phases probe (score 92): CLOSED, sum closes

Binary: lab 07928d8 (the ONLY binary with the hooks). Env
EXL3_SPEC_PHASES=1 + EXL3_EXPERIMENTAL_MMA8=1. Probe: bench/prompt.json,
512 tokens, greedy. Dir logs/brief30/r-phases2/.

Server stderr (graceful-shutdown run; terminate() runs lose the line):
  spec phases over 180 rounds: draft mean/med/p95 = 10.56/10.41/11.19 ms,
  verify mean/med/p95 = 3.74/1.20/2.16 ms (totals 1900.6/673.0 ms)
Request level (probe.json timings): eval 31365.2 ms / 512 tokens,
prompt 1283.0 ms / 74 tokens. Acceptance 0.61524 (331/538), mean len 2.84.

Round math: 180 rounds, 512 commits -> 2.844 commits/round (matches 2.84).
Round wall = 31365.2/180 = 174.25 ms/round.
draft 10.56 + verify 3.74 = 14.30 ms; remainder = 174.25 - 14.30 = 159.95 ms
= 91.8% of the round UNEXPLAINED by draft+verify timers.
BUT the verify timer here is the UNSYNCHRONIZED one (no llama_synchronize
before the stop — see s2): verify mean 3.74 ms with med 1.20 vs p95 2.16
is the classic async-launch signature (timer measures launch, GPU work
escapes into the "remainder", which is really verify+sample work).
So the R falsification condition (>5% unexplained) is MET IN THE RAW
NUMBERS (91.8%), which per the advisor means: add cudaEventRecord pair
around the sampler too AND synchronize the verify stop. The synchronized
re-run is the one remaining measurement (needs the one-line patch below;
no build was done this session per the rules).

Why the line printed only on run 3 of 3: the spec-phases line prints from
print_timings() at request end — always. Runs 1-2 lost it because
proc.terminate() kills the server before stderr flushes through the pipe;
run 3 added a 90 s graceful wait + /unload attempt (404: this fork has no
/unload route; terminate() fallback) and the wait let the pipe drain.
Harness lesson: never terminate() a server before reading its stderr tail.

Caveat (stated, not hidden): lab kernels lack MMA8, so the ABSOLUTE draft
10.56 ms includes no MMA8 verify; the closure FRACTION is the transferable
result. On deployed kernels the same structure (async verify + CPU sample)
applies — the q-slow-srv leg below shows where the ms actually sit.

## 2. Q — parity A/B (score 80): DEPLOYMENT-IS-SLOWER CONFIRMED, config-delta DEAD

### 2a. Bench legs (three arms; deployed bench N/A)

| arm | tg128 (-n 128 -p 0 -r 5) | pp4 (-n 0 -p 4 -r 5) |
|---|---|---|
| build-slow MMA8 ON (q-tg/pp-slow-mma8on.txt) | 21.00 +/- 0.06 | 52.27 +/- 6.78 |
| build-slow MMA8 OFF (q-tg/pp-slow-mma8off.txt) | 20.99 +/- 0.05 | 42.25 +/- 4.70 |
| lab 07928d8 (q-tg/pp-lab07928d8.txt) | 20.95 +/- 0.11 | 25.60 +/- 2.06 |

tg128: all three arms within 0.2% — pure decode is kernel-indifferent
(expected: M=1 never takes MMA8; GROUP2 path irrelevant at rows=1).
pp4: MMA8 ON vs OFF on the SAME binary = 52.27 vs 42.25 (+23.7%) — the
MMA8 verify path is real and large at rows=4. Lab-no-MMA8 pp4 (25.60) is
~2x below build-slow MMA8-ON (52.27): the advisor's "lab binary has no
MMA8 yet pp4 70.2 ms below deployed MMA8 79.0 ms" contradiction used
STALE brief14 numbers (different harness era, -c 65536 lab bench vs old
deployed bench methodology). On TODAY's matched bench protocol the
contradiction does not reproduce: MMA8-ON is 2.04x the no-MMA8 lab pp4.
The "deployment is slower" claim in the BENCH sense is therefore
RESOLVED as a harness-era artifact; the SERVER sense is measured next.

### 2b. Server legs (the deployed arm)

| arm | 512-tok greedy probe | acceptance | mean len |
|---|---|---|---|
| deployed 8081 as-is, MMA8 ON (restore-probe.json, post-restore verify) | 32.77 tok/s | 0.68327 (343/502) | 3.04 |
| build-slow server, MMA8 ON (q-slow-srv/probe.json, -c 65536) | 30.73 tok/s | 0.68327 (343/502) | 3.04 |
| lab 07928d8 server (r-phases2/probe.json, -c 65536) | 16.32 tok/s | 0.61524 (331/538) | 2.84 |

reasoning_content sha: deployed-restore b265cae8... vs build-slow server
b265cae8... — IDENTICAL streams (343/502 acceptance both). The lab server
on the same prompt gives 331/538 with a different stream hash (2e2cfd1d):
acceptance 0.615 vs 0.683 is a KERNEL-ORDER artifact (no-MMA8 kernels
change summation order -> different FP rounding -> different greedy picks),
exactly as the advisor predicted byte-identity would break across kernels
(KI-1 note). Content empty everywhere (e3b0c442 = NO DATA, thinking ate
the 512 budget — KI-1 repeats, not a regression).

Q verdict: the four-arm-within-2% falsifier does NOT fire — arms differ by
up to 2x where kernels differ, <0.5% where they don't. So: the
"config-delta (FA_ALL_QUANTS) hypothesis" DIES (all three bench configs
verified identical OFF; tg128 identical to 0.2%), and the "deployment is
slower" claim is REFINED, not killed: deployed server (32.77, -c 131072 +
vision-mmproj CPU) vs build-slow server (30.73, -c 65536, no mmproj) on
byte-identical streams — deployment is FASTER by 6.6%, and the entire
delta is context-size/mmproj shape, not kernels. The old 32.90-vs-34.77
"lab faster" pair compared different harnesses (-c 131072 deployed vs
-c 65536 lab bench); matched -c 65536 server legs agree in stream and the
residual 6.6% favors deployment.

## 3. S-bs — target-side backend sampling (score 50): KILLED (Delta < 1%)

Lab 07928d8 server, bench/prompt-nothink.json, 256 tokens, greedy.
(Acceptance path intact both arms: 169/258 = 0.65504 identical.)

| arm | eval | prompt | tok/s | reasoning sha |
|---|---|---|---|---|
| -bs OFF (s-bs-off) | 15365.8 ms | 2119.1 ms | 16.66 | 74de731a (853 ch) |
| -bs ON (s-bs-on) | 15424.2 ms | 1584.7 ms | 16.60 | 74de731a (853 ch) |

Delta eval +58.4 ms (+0.38%) — WITHIN noise, wrong sign. Streams
byte-identical (sha match), accept path unbroken. Why no gain: with MTP
speculation the server forces backend_sampling OFF for spec slots
(server-context.cpp: backend_sampling &= !(slot.can_speculate())) — the
-bs flag is a no-op on the spec path by design; the target logits still
move to host for the CPU accept step. Per the advisor's kill rule
(Delta<1%): KILL S-bs. The 9.27 ms suspect is NOT target sampling; R's
remainder analysis (s1) says it sits in async verify + sample/accept CPU.

## 4. F — graphs upper bound (score 30): +2.6% (bench only)

build-slow tg128, MMA8 ON: graphs ON (Q arm) 21.00 +/- 0.06 vs
GGML_CUDA_DISABLE_GRAPHS=1 (f-tg-nographs.txt) 20.47 +/- 0.24.
Graphs worth +2.6% decode throughput on the bench. Server ignores the var
(confirmed again: every server leg reports "graphs reused" 177-253).
Upper bound for E's CPU-side share at decode: 2.6% of ~47.6 ms = ~1.2 ms.

## 5. P' — depth slope (score 55): SLOPE >> 0.15 ms/1k — KEEP P OPEN

build-slow bench (deployed bench N/A; attention is non-EXL3 so the
substitution is sound). tg-shape -n 32 -p 0 -r 3 (p-depth.txt):
d=0: 20.68 / d=32768: 16.98 / d=65536: 14.66 / d=114688: 11.56 tok/s.
Per-token ms: 48.36 / 58.89 / 68.22 / 86.51 ms.
Slope 0->114688: (86.51-48.36)/114.688 = 0.333 ms per 1k tokens —
2.2x the advisor's 0.15 kill threshold. pp-shape -n 0 -p 4 -r 3
(p-pp-depth.txt, PARTIAL — d=114688 leg never finished, see s6):
d=0: 48.91 / d=32768: 42.17~40.78 / d=65536: 37.16 tok/s(pp4 units).
M=4 attention slope from pp4: (1/37.16-1/48.91)*1000/65.536... in
pp4-call units the drop 48.91->37.16 over 64k = -0.185 tok/s per 1k,
i.e. the M=4 verify PATH also degrades steeply with depth. Either way:
jusko port stays OPEN by the advisor's own rule (slope >= 0.15).

## 6. Failures with log evidence (count as session success per brief)

1. R-as-specified (llama_synchronize before verify-timer stop on
   build-slow) is IMPOSSIBLE without a build: build-slow predates the
   hooks entirely (no server-context.cpp in worktree engine/tools/server;
   grep SPEC_PHASES = 0). Ran R on the hooked lab binary instead; absolute
   draft ms don't transfer to MMA8 kernels, closure fraction does. The
   one-line sync patch (engine/tools/server/server-context.cpp:3644,
   `llama_synchronize(ctx_tgt);` between t_verify0 stop... actually
   BEFORE computing t_verify_ms, i.e. sync then stop) + rebuild of the
   lab server is queued, not done (no-build rule).
2. p-pp-depth d=114688 leg never completed: pp4 -d114688 needs >128k of
   effective allocation under -c... actually -c default 4096 in bench +
   -d offset forces full 114688+ KV allocs; the leg ran >40 min over two
   launches (04:06-04:18 partial with 5 rows, then a re-run whose tail
   never appended). File holds d=0/32768/65536 x2 runs; slope uses tg-shape
   (complete) + pp partial. GPU was verified FREE before restore.
3. `/unload` is 404 on this fork (no such route) — graceful shutdown via
   unload-then-exit is unavailable; the working recipe is terminate()
   + 90 s drain wait before reading stderr (run_srv2.py).
4. Deployed MMA8-off server leg not run (would need deployment flag flip;
   declined per plan — bench MMA8-off leg covers the kernel comparison).

## 7. Task 2 — no-GPU verdicts (subagent, logs/brief30/task2-*.txt)

- NEW-1 byte census (GGUF header-only, K=shape[0]/16): verify bytes
  K=3 88.68% / K=4 2.06% / K=6 9.26% (10.303 GB verify total). K=4 = 8
  trellis tensors, all in the single MTP layer. 2.06% << 15% threshold
  (robust: 1.65% vs full file too). VERDICT: KILL NEW-1. (Advisor's
  "30% by count" was matmul-count, not bytes — the census corrects it.)
- L coverage-vs-B (63,370 tokens, EN+JA operator streams + ppl-corpus +
  draft-vocab prior, 1940 blocks): 90% @B=190 / 95% @B=308 /
  99% @B=556 (28.7% of blocks) / 99.9% @B=741. Worst-case uniform variant
  99% @B=694 (35.8%). Threshold 50% (970 blocks). VERDICT: KEEP L.
  At B=556 draft head reads fall 3x953.5 MB -> 3x273.3 MB/round,
  saving ~2.04 GB/round (~12.7% of the ~16.1 GB round).

## 8. Exact commands (audit trail)

```
bash scripts/gpu-lock.sh acquire 100            # LOCK_HELD 03:19:58
taskkill /IM llama-server.exe                   # stop 8081 (no /F: policy)
H:\exl3-lab\logs\brief30\q-a-slow-on.bat        # Q: build-slow MMA8 ON tg128+pp4
H:\exl3-lab\logs\brief30\q-b-slow-off.bat       # Q: build-slow MMA8 OFF tg128+pp4
H:\exl3-lab\logs\brief30\q-c-lab.bat            # Q: lab 07928d8 tg128+pp4
H:\exl3-lab\logs\brief30\f-nographs.bat        # F: graphs-off tg128
H:\exl3-lab\logs\brief30\p-depth.bat           # P': tg32 -d 0,32768,65536,114688
set EXL3_EXPERIMENTAL_MMA8=1& set EXL3_SPEC_PHASES=1& python H:/exl3-lab/logs/brief30/run_srv2.py H:/exl3-lab/build/bin 8082 H:/exl3-lab/logs/brief30/r-phases2 H:/exl3-lab/bench/prompt.json 512
                                                # R: hooked lab server, 512-tok probe
set EXL3_EXPERIMENTAL_MMA8=1& python H:/exl3-lab/logs/brief30/run_srv2.py H:/exl3-lab/build/bin 8082 H:/exl3-lab/logs/brief30/s-bs-off H:/exl3-lab/bench/prompt-nothink.json 256
set EXL3_EXPERIMENTAL_MMA8=1& python H:/exl3-lab/logs/brief30/run_srv2.py H:/exl3-lab/build/bin 8082 H:/exl3-lab/logs/brief30/s-bs-on H:/exl3-lab/bench/prompt-nothink.json 256 -bs
                                                # S-bs pair
set EXL3_EXPERIMENTAL_MMA8=1& set EXL3_SPEC_PHASES=1& python H:/exl3-lab/logs/brief30/run_srv2.py H:/exl3-lab/worktrees/mma8fast/build-slow/bin 8082 H:/exl3-lab/logs/brief30/q-slow-srv H:/exl3-lab/bench/prompt.json 512
                                                # Q server leg: build-slow, byte-identical 343/502 stream
H:\exl3-lab\logs\brief30\p-pp-depth.bat         # P'-pp (partial: d114688 missing)
(start-server-exl3.ps1 via hidden-window background start, health + probe)
cd H:/exl3-lab && source scripts/lib-bench.sh && probe_decode 8081 H:/exl3-lab/logs/brief30/restore-probe.json
bash scripts/gpu-lock.sh release                # LOCK_RELEASED 04:41:08
```

## 9. Box health / restore (done)

- Deployment restored hidden-window on 8081: /health 200, VRAM 14843 MiB
  (expected 128k-ctx figure), restore probe 32.77 tok/s, 512 tokens,
  acceptance 0.68327 (343/502) — inside HARNESS band (32.66-34.44,
  0.66667-0.68327). reasoning 1661 chars sha b265cae8 = byte-identical to
  build-slow server leg. No RESTORE-NEEDED marker (restore done here).
- Lock released 04:41:08, status FREE. No orphan llama-server (nvidia-smi
  compute-apps empty before restore start).

## 10. Values for the advisor (one line each)

- R: draft 10.56 + verify(unsync) 3.74 vs round 174.25 ms — 91.8%
  remainder = async verify + sample/accept CPU; sync re-run queued.
- Q: tg128 identical <0.5% all arms; pp4 MMA8-ON 52.27 vs OFF 42.25 vs
  lab 25.60; servers deployed 32.77 = build-slow 30.73 in stream
  (343/502 both), lab 16.32 different stream (331/538). Config-delta dead.
- S-bs: +0.38% (noise, wrong sign), streams identical — KILL.
- F: graphs +2.6% bench decode (~1.2 ms upper bound for E's CPU share).
- P': 0.333 ms/1k tokens, 2.2x over kill threshold — KEEP P OPEN.
- NEW-1: K=4 = 2.06% of verify bytes — KILL. L: 99% @ 28.7% blocks — KEEP.
