# start-server-exl3.ps1  (ASCII ONLY) -- run the exllamav100 (EXL3-for-V100) build as an
# OpenAI-compatible server for the abliterated EXL3 3.0bpw Qwen3.8-27B, with the model's own MTP
# (nextn) head driving speculative decoding.
#
# Engine : H:\exl3-local\build-v100\bin\llama-server.exe
#          llama.cpp fork by Vendetta1871/exllamav100 -- CUDA 11.8 + MSVC 14.34, sm_70-real, native
#          EXL3 trellis kernels (upstream exllamav3 cannot execute on V100/sm_70).
#          Since 2026-09-24 it also carries a LOCAL kernel patch (Sol/gpt-6-sol, brief #6): the dense
#          split-K GEMV decodes each trellis weight once and applies it to two rows (GROUP2 in
#          ggml/src/ggml-cuda/exl3.cu). That patch is what makes the speculative setting below
#          profitable; plain single-token decode is unaffected (+0.18%, within noise).
#          Pre-patch backup: H:\exl3-local\backup-prepatch-<ts>\{llama-server.exe,exl3.cu}.prepatch
#          Rollback: restore exl3.cu.prepatch + rebuild with scripts\build-exl3-v100.ps1, then start
#          this script again (the MTP model also works without speculation: see the numbers below).
# Model  : H:\exl3-local\models\exl3-gguf\huihui-qwen3.8-27b-abliterated-exl3-3bpw-mtp.gguf
#          Same 3.0 bpw EXL3 body plus the nextn/MTP block exported from the source repo's mtp.*
#          tensors (lm_head 6-bit, vision tower dropped). The non-MTP GGUF stays side by side.
#
# Measured on this V100 (fixed prompt, 512 generated tokens, -c 32768, q8_0 KV, matched harness):
#   no speculation        20.96 tok/s   <- live until 2026-09-24
#   THIS setting          24.24 / 24.26 / 24.38 tok/s   (+15.6..16.3%, three samples)
#   MTP k=7 instead       14.81         (verification batch too large on this kernel)
#   draft model 0.8B k=4  11.76
# Acceptance at k=3: 57.68% (323/560), mean committed length 2.73 tokens per round.
#
# Port 8081 (the stock llama.cpp stack on port 8080 stays untouched).
# Hidden window: a visible console would steal focus while the user is typing.
#
# Stop the server:  taskkill /IM llama-server.exe /F    (stops both stacks)
#                   or kill only this one: stop the PID that listens on 8081
# Server log:       H:\exl3-local\logs\server-exl3.err.log

$ErrorActionPreference = 'Continue'
$exe  = 'H:\exl3-local\build-v100\bin\llama-server.exe'
$mdl  = 'H:\exl3-local\models\exl3-gguf\huihui-qwen3.8-27b-abliterated-exl3-3bpw-mtp.gguf'
$mmproj = 'H:\exl3-local\models\mmproj\mmproj-model-bf16.gguf'
$log  = 'H:\exl3-local\logs\server-exl3.log'
$errl = 'H:\exl3-local\logs\server-exl3.err.log'
New-Item -ItemType Directory -Path 'H:\exl3-local\logs' -Force | Out-Null

if (-not (Test-Path -LiteralPath $exe)) { Write-Output ('ERROR: engine not found: ' + $exe); exit 1 }
if (-not (Test-Path -LiteralPath $mdl)) { Write-Output ('ERROR: model not found: ' + $mdl); exit 1 }
if (-not (Test-Path -LiteralPath $mmproj)) { Write-Output ('ERROR: mmproj not found: ' + $mmproj); exit 1 }

# KV cost measured 2026-09-25 (q4_0, MMA8, MTP k=3, GPU clip): 64k -> 14,189 MiB used, 128k ->
# 15,981 MiB used, i.e. ~1,792 MiB per 64k of context; 262,144 crashes the engine (segfault, OOM is not
# reported cleanly -- do not set it). With clip moved to the CPU (-c 131072) the same 128k costs
# 14,843 MiB, leaving 1,415 MiB for image work. Speculation adds a small draft context on top;
# 16.2 GB is the card's usable total.
$a = @(
  '-m', $mdl,
  '--alias', 'huihui-qwen3.8-27b-abliterated-exl3',
  '-ngl', '99',
  '-sm', 'none', '-mg', '0',
  '-c', '131072',
  '-np', '1',
  '-fa', 'on', '-b', '2048', '-ub', '512',
  # KV dtype: q4_0 since 2026-09-24 ~15:50. Measured at -c 65536 with MMA8 + MTP k=3 on the fixed
  # prompt: q4_0 32.82 / 32.93 / 32.88 tok/s vs q8_0 29.44 / 29.62 / 29.60 (+11.1 %, four samples,
  # no overlap), and 1 GB less VRAM (13,244 vs 14,268 MiB). f16 (2x the bytes) matched q8_0, so the
  # gain is NOT bandwidth: the draft acceptance rises from 0.5817 (mean len 2.74) to 0.6833 (3.04)
  # because drafter and verifier agree more often at coarser KV. Quality gate: llama-perplexity on the
  # same corpus gives 5.2938 +/- 0.0989 (q8_0) vs 5.3047 +/- 0.0991 (q4_0) -- 0.21 %, inside the
  # error bars. q5_1 is pathological on this fork (20.95 tok/s) -- do not use it.
  '-ctk', 'q4_0', '-ctv', 'q4_0',
  '-t', '6',
  '--spec-type', 'draft-mtp', '--spec-draft-n-max', '3', '--spec-draft-n-min', '1',
  # Vision: mmproj for the Huihui Qwen3.8-27B-abliterated vision tower (qwen3vl_merger). Measured on
  # this card 2026-09-25: 8/8 recognition cases correct including an exact OCR read ("PLATE-7381"),
  # a left/right relation, a bar-chart comparison, a caption-vs-picture conflict and a Japanese
  # prompt; VRAM 13,051 -> 14,189 MiB of 16,258 at -c 65536 (clip worst-case estimate 1,161 MiB);
  # +1.6 s per image (clip encode, ~330 image tokens); text throughput unchanged (512-token greedy
  # probe 31.21 -> 31.17 tok/s). Without it the same image request returns an empty completion.
  # Revert = delete these two lines (the deployment then behaves exactly as before).
  '-mm', $mmproj,
  # Clip on the CPU (--no-mmproj-offload) instead of the GPU: the mmproj then costs 0 MiB of VRAM
  # (weights + workspace live in host RAM, measured host working set +1.4 GB), which is what makes
  # 131072 context possible at all. Measured 2026-09-25, -c 131072 with this flag: idle VRAM
  # 14,843 MiB (1,415 free) vs 15,981 MiB (277 free) with clip on GPU -- and at 277 free a normal
  # image request (measured +408 MiB spike with clip on GPU) kills the server. Clip on CPU works:
  # small image answers correctly in 11.9 s (GPU: 1.6-2.3 s), the SAME image again is cached and
  # costs 0.8 s (cache_n 407), a 1920x1080 screenshot answers correctly in 95.5 s and consumes
  # 2,067 prompt tokens of context. Text throughput is unaffected by the flag.
  '--no-mmproj-offload',
  '--host', '127.0.0.1',
  '--port', '8081',
  '--jinja',
  '--no-reasoning-preserve'
)

# Kernel selection for the M=4/M=8 verification band (the speculative batch):
#   EXL3_EXPERIMENTAL_MMA8=1  -> Volta mma.sync.m8n8k4 path, trellis staged compressed and expanded
#                                to transient FP16 fragments. Measured 2026-09-24: M=4 97.3 -> 79.0 ms,
#                                M=8 170.7 -> 97.9 ms; live k=3 MTP 24.12 -> 28.96 tok/s (temp 0.6) and
#                                21.64 -> 25.72 (temp 1.0); greedy token stream identical to GROUP2 on
#                                the same prompt+seed (content and reasoning md5 equal).
#   unset / 0                -> the GROUP2 dense split-K kernel (the previous deployment).
# M=1 decode is unaffected either way. Rollback = remove this line (binary stays valid).
$env:EXL3_EXPERIMENTAL_MMA8 = '1'

$p = Start-Process -FilePath $exe -ArgumentList $a -RedirectStandardOutput $log -RedirectStandardError $errl -WindowStyle Hidden -PassThru
Write-Output ('started pid=' + $p.Id)
