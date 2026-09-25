# Brief #27 REPORT (2026-09-25, read-only, no GPU)

1. Closure: 8.14 (draft, measured) + 70.246 (bench, wrong kernel)
   + 0.0014 = 78.39 vs round 87.66 -> 9.27 ms (10.6%) missing.
2. "Verify ~80%" is inferred (70.246/87.66 across harnesses); grounded
   in-server claim is only "non-draft = 90.7% of round".
3. Lab binaries lack MMA8+GROUP2: 34.77, 70.246, cb cycles, staging all
   measured the OLD kernel. Deployment numbers live only in STATUS.md
   and deploy-probes (unmatched prompts).
4. Could NOT verify: in-server verify share (no sync'd timer exists);
   true draft cost (`dur g` lacks sync, brief21 timers lack it too);
   why same-flag acceptance is 0.683 (brief14) vs 0.615 (brief21/25);
   deployment 10-12% gap (no matched A/B exists); EXL3_FAST_EXTRACT
   (in forbidden fixA tree — not read).
5. Advice errors: "M=2..8 -> GROUP2" (rows 4/8 take MMA8 when flagged);
   claim-2 "xh from shared memory" (xh is global pool memory).
6. FIRST measurement (one GPU lock, instrumented mma8fast-synced build):
   add `llama_synchronize(ctx_tgt);` before the verify-timer stop
   (`server-context.instrumented.cpp:3645` pattern), rebuild,
   `EXL3_SPEC_PHASES=1` + 512-token greedy probe on 8082, read the
   `spec phases` line: draft mean + verify mean + round time closes
   the sum and settles draft cost, verify share, and sampling/loop
   remainder in one run. Then restore 8081, health 200, release.
