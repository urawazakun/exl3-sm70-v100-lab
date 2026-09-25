# AUDIT B — answer: does the decomposition close? (read 2026-09-25, no GPU)

All arithmetic below was recomputed from the cited lines. "Deployed tree" =
`H:/exl3-local/exllamav100`, "lab tree" = `H:/exl3-lab/engine`. Lab-tree
binaries contain **neither** MMA8 **nor** GROUP2 (0 grep hits), so every
llama-bench/server number from a lab `build/bin` binary measured the old
one-row-at-a-time kernel — including the 70.246 ms "MMA8" figure.

## Flag list (claim -> verdict -> file:line -> what it changes)

1. No-draft baseline 46.68 ms/token -> SOLID (measured, lab kernel).
   `logs/brief10/run-20260925-015403/none.server.log:265`:
   `eval time = 23898.59 ms / 512 tokens (46.68 ms/token)`. 23898.59/512 =
   46.671 ✓. Reference rep 46.56 (`run-20260925-014647/none.server.log:265`).
2. MTP k=3 round 28.76 ms/token = 34.77 tok/s -> SOLID (measured, lab kernel).
   `logs/brief14/baseline.server.err.log:309`: `eval time = 14726.12 ms /
   512 tokens (28.76 ms/token, 34.77)`. 14726.12/512 = 28.760 ✓,
   512/14.72612 = 34.769 ✓. Same-config reps span 34.28-34.83
   (`LAB_REPORT.md:356,362,374`), so 34.77 sits at the top of a ±0.8% band.
3. Acceptance 0.68327 (343/502), mean commit 3.04 -> SOLID (measured).
   `logs/brief14/baseline.server.err.log:312,314`. 343/502 = 0.683267 ✓,
   (343+168)/168 = 3.042 ✓.
4. Speedup 1.62x -> DERIVED-ONLY (across two different runs).
   46.68 (brief11 `none`, #1) / 28.76 (brief14, #2) = 1.6235 ✓ arithmetic,
   but different dates/binaries. Within-run ratios: 1.608 (`LAB_REPORT.md:356-357`),
   1.596 (`:362`). Honest range 1.60-1.62.
5. "MMA8 verify kernel ~70 ms" -> DERIVED-ONLY + WRONG KERNEL.
   70.246 = mean of samples 2-5 in `logs/brief14/baseline.m4.json:44`
   ((70.2596+70.2219+70.2372+70.2636)/4 = 70.2456 ✓; json `avg_ns` 74.93 ms
   includes the 93.66 ms warmup `:40`). It is a **whole-model** `llama-bench
   -p 4` forward pass, empty KV cache, and the lab binary has no MMA8 code,
   so it timed the old kernel. Not a kernel time, not in-server, not deployed.
6. Draft ~17.5 ms/round -> INCONSISTENT (derived by subtraction, contradicted
   by direct measurement). Source: `PLAN.md:214-215`, `PROMPT8.md:133-137`:
   92 ms round − 74.5 ms verify, from the q8_0-KV era (commit 2.75). Direct
   measurement: `logs/brief14/baseline.server.err.log:314`,
   `dur(b,g,a) = 0.001, 1367.109, 0.239 ms` over 168 calls ->
   1367.109/168 = **8.14 ms/round**. Caveat: the `dur g` timer wraps
   `impl->draft()` with no `llama_synchronize`, so async GPU work may escape it.
7. Codebook cb0 21.70 / cb1 22.08 / cb2 15.59 cycles -> SOLID for the
   synthetic microbench only. `logs/brief18/cb-clock.out:1-3` (per-thread
   minimums; avgs 21.705/22.087/15.598). `LAB_REPORT.md:388` "3 repeats
   agree" is UNKNOWN — the .out file holds one run.
8. Staging ~1.5 ms vs 12-37 ms ≈ 1/10 -> DERIVED-ONLY, synthetic.
   `logs/brief18/cb-microbench.out`: staging 1.512 ms; decode 11.976 (cb2/4row)
   – 37.054 ms. Ratio runs 1/7.9 (production cb2, 4 rows) to 1/24.5, not 1/10.
   Nothing ties it to in-server time.
9. Deployment 10-12% slower at 4 matched points -> UNSUPPORTED.
   No file contains this claim except the audit question (`AUDIT-B-numbers-close.md:19`).
   `logs/deploy-probes/journal.txt` decode figures (8k-mid 32.52/32.4;
   32k-mid 25.99; 96k-mid 19.6/17.92/20.36) use needle-retrieval prompts,
   max_tokens=1024, default temperature 1.0 (`journal.txt:3`), 77-107 tokens
   generated — none matched to the lab's 74-token greedy 512-token probe. UNKNOWN.

## Closure sum (k=3, brief14 lab baseline, 168 rounds; round = 14726.12/168 = 87.66 ms)

| term | ms/round | source | status |
|---|---|---|---|
| draft (`dur g`/calls) | 8.14 | `brief14/baseline.server.err.log:314` | measured, host wall, async caveat |
| accept (`dur a`/calls) | 0.0014 | same line | measured |
| verify M=4 | 70.246 | `brief14/baseline.m4.json:44` | measured, DIFFERENT harness + wrong kernel |
| sampling / server loop | UNKNOWN | — | not logged anywhere |
| sum of known terms | 78.39 | | |
| round (eval/calls) | 87.66 | `:309` + `:314` | measured |
| **missing** | **9.27 (10.6%)** | | **over the ~10% bar** |

With the claimed 17.5 ms draft, 70.25+17.5 = 87.75 closes to 0.1% — but that
is circular: 17.5 was produced by subtraction (92 − 74.5). The missing term is
most plausibly (a) verify-with-filled-KV + all-rows logits vs empty-KV bench,
(b) sampling + server loop, (c) GPU work escaping the async `dur g` timer.

## Measured vs derived

- Measured: #1, #2, #3, draft 8.14 (`dur g`/calls), accept 0.0014,
  brief21 phase timers (draft mean 10.55-11.22, verify mean 3.60-4.19;
  `logs/brief21/baseline.server.err.log:9578,17921,32798`), 70.246 (bench harness),
  cb cycles, staging/decode ms, deploy-probe tok/s.
- Derived: 1.62x (from #1/#2 across runs); 17.5 (92−74.5); "80%" (70.246/87.66);
  round 87.4 (28.76×3.04, `RESEARCH-prior-art-specdec.md:37`); staging ratio;
  "M=4 costs 1.5x" (70.25/46.68, `ADVICE-20260925-opus-cycle.md:16`).

## Mutual inconsistencies (same quantity, different value)

- a. Verify M=4: 74.5 (`PLAN.md:215`, `PROMPT8.md:130`; older build,
  `-p 1,2,4,8` curve) vs 70.246 (`LAB_REPORT.md:374`,
  `brief14/baseline.m4.json:44`) vs 79.0 (`H:/exl3-local/STATUS.md:143`,
  deployed-era table). Three values, three build/config combos.
- b. Round: 92 (`PLAN.md:214`, `PROMPT8.md:133`, q8_0 era) vs 88.2
  (`PROMPT14.md:11`: 29.03×3.04 = 88.25 ✓) vs ~88 (`PROMPT18.md:21`,
  `PROMPT21.md:18`) vs 87.4 (28.76×3.04 = 87.43 ✓) vs **87.66 measured**.
- c. Same-flag MTP acceptance: 0.68327/3.04 (brief14, `baseline.server.err.log:312`)
  vs 0.61524/2.84 (brief21 task0, `logs/brief21/baseline.server.err.log:9577`;
  also brief25 control `logs/brief25/REPORT.md:60`). Same prompt, same flags —
  cause UNKNOWN (different binaries; brief21 was an instrumented rebuild).
- d. Brief21 phase timers sum to draft ~11.07 + verify ~3.76 = ~14.8 ms/round
  (`baseline.server.err.log:32798`) while its own round is 32986.99/179 =
  184.3 ms — only ~8% attributed. Both timers lack `llama_synchronize`
  (verify wraps async `llama_decode` at
  `logs/brief21/server-context.instrumented.cpp:3642-3645`), so the "verify
  row is cheap" reading in `logs/brief24/REPORT.md:101-108` compares queue
  time against draft-step time that includes synchronous sampling. c
  contradicts the 80% narrative from inside the server; the timers are the
  unreliable party, not the round clock.
- e. `LAB_REPORT.md:370` honestly scopes the M=4 number (no draft steps, no
  populated KV) — later consumers (`STATUS-NOW.md:23`, advice §2) dropped the caveat.

## "Verify is ~80% of the round": INFERRED, not grounded

It is exactly 70.246/87.66 = 80.1% — a cross-harness ratio whose numerator is
the wrong kernel. From measured in-server numbers alone, all that is grounded
is: non-draft work = (87.66−8.14)/87.66 = **90.7%** of the round (verify +
sampling + loop + unattributed GPU). No log isolates in-server verify.

## 3 weakest numbers + cheapest fix each

1. Verify share / missing 9.27 ms: in the brief21 instrumented copy, add
   `llama_synchronize(ctx_tgt);` before stopping the verify timer
   (`logs/brief21/server-context.instrumented.cpp:3645`), rebuild, run one
   512-token probe with `EXL3_SPEC_PHASES=1`, read the `spec phases` line.
   One GPU session; closes the sum for real.
2. Draft 17.5 vs 8.14 vs ~11: no new run needed — replace 17.5 with
   `dur_g/#calls` (8.14) everywhere; the same sync-aware run as (1) confirms
   whether GPU work was escaping the timer.
3. Deployment gap 10-12%: `POST bench/prompt.json` (temp 0, 512 tok) to 8081
   then 8082 back-to-back, 3 reps each, compare `eval time` lines. Only then
   state a percentage.
