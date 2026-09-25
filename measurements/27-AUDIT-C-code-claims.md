# AUDIT C — answer: are the code claims true? (read 2026-09-25, no GPU)

Deployed tree = `H:/exl3-local/exllamav100/ggml/src/ggml-cuda/exl3.cu` (1131
lines). Lab tree = `H:/exl3-lab/engine/ggml/src/ggml-cuda/exl3.cu` (1095
lines, 0 hits for `EXL3_EXPERIMENTAL_MMA8`/`GROUP2`/`mma8`). Deployed binary
`H:/exl3-local/build-v100/bin/ggml-cuda.dll` contains the
`EXL3_EXPERIMENTAL_MMA8` string (1 hit); deployment sets it to `1` in
`H:/exl3-local/scripts/start-server-exl3.ps1:99`.

## Verdicts (claim -> verdict -> real lines -> consequence)

1. `exl3_window<3>` per weight in MMA8 inner loop (cited `:467`) ->
   TRUE in deployed, FALSE in lab (no MMA8 kernel at all).
   Deployed `H:/exl3-local/exllamav100/ggml/src/ggml-cuda/exl3.cu:467-468`,
   inside `for j 0..7` (`:460`) inside `for kt` (`:450`): two
   `exl3_window<3>` calls per j = 16 weights/lane/k-tile. Line ref exact.
   No change to MMA8-fast: this is its target #1.
2. fp32 `xh` reload + `__float2half` per weight per k-tile (cited `:476-480`) ->
   PARTLY in deployed, FALSE in lab. The conversion loop is real at `:475-482`
   (`xr = xh + ar*k + kt*16`, 8 iters x 2 `__float2half`), but `xh` is a
   **global-memory** pool allocation (`:992`, `xh.get()`), NOT shared memory —
   the advice mislabels the source. `packed` (shared) holds trellis, not `xh`.
   Consequence: the fp16-`xh` proposal must add a half copy path in `had_in`,
   not a shared-memory reuse; cost model (1 LDG + 1 F2F/weight) stands.
3. MMA8: no prefetch, 4 warps/block, ~192 B in flight (cited `:450-454`) ->
   TRUE in deployed, FALSE in lab. Launch `:1022`: 128 threads = 4 warps
   (split-K `:1029`: 256 = 8 warps). No `pf[]`/double-buffering in MMA8
   (`:427-503` has no `pf`; contrast split-K `:264-296`). 192 B =
   per-warp staging `2*8*3` words (`:447`, `packed[4*2*8*3]` `:436`).
   Cited range covers the kt-loop/staging (`:450-454`) — close enough.
   No change to MMA8-fast: its target #3 confirmed.
4. Split-K HAS register double-buffering (cited `:290-321`) ->
   TRUE in both trees, STALE-LINE-REF. The prefetch pipeline is at deployed
   `:264-296` (`uint4 pf[PF]` `:266`, fill `:271-279`, refill `:287-296`,
   comment "prefetch pipeline" `:264`), not `:290-321` (that range is the
   x-load + K==4 decode arm). Same code in lab at `:265-296`. Consequence:
   none — the asymmetry (split-K prefetches, MMA8 does not) is real; fix the
   line ref to `:264-296` in the plan.
5. Launch structure had_in + main + epilogue (cited `:994/:1022/:1033`),
   folding removes 2 of 3 -> PARTLY in deployed, PARTLY in lab.
   Deployed: `had_in` `:994`, MMA8 main `:1022` (or split-K `:1029`),
   `epilogue` `:1033` — line refs exact — BUT only on the S>1 path. S==1
   launches had_in + single `gemv` (2 kernels, no epilogue, `:1042`); rows>8
   launches had_in + `gemm` (2 kernels, `:1002`). Lab: same 3-launch shape at
   `:874/:929/:932`-ish, also S-gated. And "per token" is wrong granularity:
   it is 3 launches per EXL3 matmul op x dozens of layers per token.
   Consequence: the "2/3 launches removed" saving is per-matmul, and only for
   S>1 tensors — re-estimate with the KTIMING kernel-count table before
   ranking it; do not multiply by tokens.
6. "MMA8 already decodes each weight once for 4 rows" -> TRUE in deployed,
   FALSE in lab (old kernel decodes inside `for row` `:126`/`:282`, i.e. once
   per row). Deployed: `b[j]` decoded once per lane per k-tile (`:458-471`),
   then the `for j 0..3` mma loop (`:484-492`) applies it across the row
   dimension at warp level; each weight belongs to one warp's n-tiles, so one
   decode serves all 4 rows. Consequence: (c) "multi-row reuse" is indeed
   already implemented — kill that work item, keep "make the decode cheap".
7. Constant-shift extraction under `EXL3_FAST_EXTRACT`, M=1-only ->
   NOT FOUND in deployed, NOT FOUND in lab (0 hits for `FAST_EXTRACT` in
   either file; lab has only the different `EXL3_FAST_K6` flag). The flag
   presumably lives in `worktrees/fixA`, which this brief must not read, so:
   UNKNOWN in permitted trees. The second half is consistent with the deployed
   dispatch (a split-K GEMV change would run at M=1 only — see #8), but that
   does not verify the flag exists. Consequence: the plan's Step-0 item
   "replace runtime flag with `-DEXL3_FAST_EXTRACT=0/1`" cannot be checked
   until the mma8fast sync; do not cite the flag as established fact.
8. `EXL3_EXPERIMENTAL_MMA8=1` exists + MMA8 is the M=4 path under deployment
   flags -> TRUE in deployed (code + binary + starter all agree), FALSE in lab.
   Code `:1014-1024` (`getenv` `:1015`, gate `mma8_enabled && K==3 &&
   (rows==4||rows==8)` `:1018`); binary string present (1 hit in
   `ggml-cuda.dll`); starter sets it (`start-server-exl3.ps1:99`); model is
   K=3 3.0bpw and MTP k=3 verifies at rows=4 with S>1 (splits fn `:959-968`
   gives S>>1 at 27B shapes). Lab: no flag, no kernel, no GROUP2 — every lab
   `build/bin` number (34.77, 70.246, cb cycles) measured the old kernel.
   Consequence: no lab-binary A/B is valid for MMA8-fast; use the deployed
   binary or the synced mma8fast build only.

## Two advice statements that are wrong on the deployed source

- a. "M=2..8 falls back to GROUP2" (advice §0): FALSE for 4/8. With the flag
  set, rows 4/8 take MMA8 (`:1018-1024`); GROUP2/dense2 takes rows
  2,3,5,6,7 (`:1027`); rows==1 takes plain split-K; rows>8 takes GEMM
  (`:999-1008`). The K=6 draft-head work must target GROUP2-at-M=4 only where
  MMA8 does NOT fire — recheck which tensors actually take that path.
- b. Claim-2's "from shared memory" (see #2 above): `xh` is global.

## Code facts in the plan not found anywhere readable

- `EXL3_FAST_EXTRACT` (see #7).
- `EXL3_KTIMING=1` / `EXL3_NULL_MATMUL=1` (advice Steps 2-3): 0 hits in both
  trees — correctly absent (proposed, not implemented); no correction needed,
  just do not grep for them as if they exist.

## Checking commands used (all native tools got `H:/` paths)

- `grep -n "EXL3_EXPERIMENTAL_MMA8" H:/exl3-local/exllamav100/ggml/src/ggml-cuda/exl3.cu` -> `:1015`
- `grep -n "GROUP2" H:/exl3-local/exllamav100/ggml/src/ggml-cuda/exl3.cu` -> `:215,:253,...`
- `grep -c "EXL3_EXPERIMENTAL_MMA8\|GROUP2" H:/exl3-lab/engine/ggml/src/ggml-cuda/exl3.cu` -> `0`
- `grep -n "exl3_window\|__float2half\|had_in\|EXL3_FAST_EXTRACT" H:/exl3-local/exllamav100/ggml/src/ggml-cuda/exl3.cu`
- `read_file` deployed `:215-339` (split-K+prefetch), `:427-503` (MMA8),
  `:980-1048` (dispatch), lab `:960-1012` (dispatch, FAST_K6 only)
- `grep -a -c "EXL3_EXPERIMENTAL_MMA8" H:/exl3-local/build-v100/bin/ggml-cuda.dll` -> `1`
- `git -C H:/exl3-lab log --oneline -5 -- engine/ggml/src/ggml-cuda/exl3.cu` ->
  only `6e932e1` + scaffold: lab exl3.cu never received MMA8/GROUP2
