# ラボ状態の正本（コンパクション後の復帰はこれを読む）— 2026-09-25 19:xx
実測ログは `H:\exl3-lab\logs\`、失敗と規律は `H:\exl3-lab\KNOWN-ISSUES.md`、方針は Claude指示書
`logs\advisor-20260925-191426.md`。デスクトップの「ラボ状況-確認.cmd」でも一覧可。

## 0. いま走っているもの
| セッション | 内容 | ワークツリー | ログ |
|---|---|---|---|
| **#26** | Claude指示書 Step 0-2（**配備ソースとの同期**＋コンパイル時マクロ化＋`EXL3_KTIMING`/`EXL3_NULL_MATMUL` 計装） | `worktrees/mma8fast` | `logs/musedev-mma8fast-console.log` |
| **#21B** | K=6 専用経路（Claude順位2） | `worktrees/k6` | `logs/musedev-k6b-console.log` |
| 配備見張り | ロックFREE＋サーバ0＋8081が5分down → 自動復帰（8h常駐） | — | `logs/deploy-watchdog.log` |
- **配備**: 8081=`start-server-exl3.ps1`（**128k ctx + vision(clip CPU)** + MTP k=3 + `-fa on` + KV q4_0）✓ health=200、VRAM 14,843/1,415 ✓
- セッション停止は **`scripts/kill-sessions-safely.sh <pid>`** を使う（孤児サーバ掃除＋配備復帰込み、KI-5）

## 1. 指揮系統（2026-09-25 に本人が委任）
- **方針決定 = Claude Code (Opus 5.5)**（`scripts/claude-advisor.sh <consult.md> opus high 40`）。判断は指示書として `logs/advisor-*.md` に固定し、それに従う
- **実装 = Muse Spark 1.3 Contributor**（`muse-dev.sh` / `muse-resume.sh`、プロファイル `muse`）
- **予備 = Sol (gpt-6-sol)**（`sol-dev.sh`。先行知見調査はこれで実施済み → `RESEARCH-sol-specdec.md`）
- **検証・実行・採用判断 = assistant**（委譲結果は必ず生データで検証、同一ツリー衝突を避けワークツリー分離）

## 2. Claude の指示（順位と基準。逐語は指示書参照）
| 順位 | 作業 | 対象 | 工数 | 期待 | 状態 |
|---|---|---|---|---|---|
| **1** | **MMA8-fast**（native-lane定数シフト復号／fp16 `xh`／レジスタ先読み／8warp/block） | ラウンドの約80% | 1〜1.5日 | verify 70→55ms なら 34.8→**約42 tok/s (+20%)** | 実装前（Step 0-2 を #26 が構築中） |
| 2 | K=6 ヘッド（定数シフトK=6＋M=4でGROUP4/MMA、ヘッドを1ラウンド1回に） | ドラフト17.5ms＋検証 | 1日 | +5〜10% | **✗ 実測で否（#21B）: SASSは splitk −12.3%/gemv −6.8%（LDS 136→32,151→47）と減ったのに、同バイナリ・スイッチのみの実測で decode 15.78→15.16 tok/s（−3.9%）。K=6ループはメモリ律速で命令削減が効かない。reasoning_content はバイト一致** |
| 3 | **fix A 単体** | M=1のみ＝本体の1/64 | 0.5日 | **≤2%（単体では停止）** | **停止済み**（算術証明は順位1へ移設） |
| 4 | prepack | — | 1日 | **~0（却下）** | 却下 |
- **採用基準（順位1）**: `mma8` bin ≥15%減（5回中央値/CV<2%）／`-p 4` ≥5%かつA/Aばらつきの3倍超／MTP tok/s ≥+3%（20プロンプト）／**content と reasoning_content がバイト一致**
- **反証条件**: ①load-only版がフル実行の15%以内（＝メモリ律速）②SASS25%減でも bin 改善<10%（＝命令は律速でない）→ どちらかで**抽出アイデア自体を放棄**
- **次の実測（指示書 Step 3）**: A/Aノイズ→`EXL3_KTIMING=1 GGML_CUDA_DISABLE_GRAPHS=1`→`EXL3_NULL_MATMUL=1`→**f と f_round を算出**（f_round が作業順位を決める）。`run19b.py` は現状のまま走らせない（MMA8無しバイナリにMMA8を立てていた＝無効）
- 参考: 古い Nsight Compute（2022.3世代）なら GV100 を掴める可能性（未検証・1時間以内で見切り）

## 3. 実測値（すべて我々の生データ）
- **配備decode**: 6.0k=32.52 / 23.7k=25.99 / 71.1k=19.60 / 71.1k(early)=17.92 / 71.2k(late)=20.36 tok/s、受理率 0.79→0.59
- **配備prefill**: 345→248 tok/s（6k→71k）
- **基準線（MTP k=3, ラボ, -c 65536, MMA8, KV q4_0）**: 34.77 tok/s / 受理0.68327(343/502) / 平均コミット3.04 / ラウンド28.76ms ＝ ドラフト無し46.68ms比 **1.62×**
- **長文脈品質**: 6k/23.7k/71k×3/96k のニードル = **全HIT**（配備）、ラボで94.6k HIT。以前のMISSはハーネスのバグ（KI-1）
- **コードブック実測**: cb0=21.70 / cb1=22.08 / **cb2(mul1)=15.59 cycles/weight（最安）**
- **ステージングは decode の約1/10**（1.5ms vs 12〜37ms）
- **プロファイラ不可**（`ncu` は "Skipping unsupported chip GV100"、`nsys` なし）
- **ビルド差**: 同一条件で配備ビルドがラボより一貫して10〜12%遅い（4点ずつ、clean A/B 未実施）

## 4. 閉じた軸（我々の実測）
DFlash2(0.446<0.683)／**DSpark(0.430<0.615, ラウンド+68%, VRAM+1.95GB=128kで入らない)**／CPU常駐ドラフタ(8.68tok/s, 受理bit一致)／EAGLE3(ヘッド不在＋フォーク衝突)／jusko Volta attention(q8 KV専用・EXL3非対応・16-24h)／**n-gram(#24B: 8セル全敗, 自由記述−5〜−19.5%)**

## 5. 生成物・資産
- DSpark: `models/dspark-q4_k_m.gguf` 1,104,585,888 B／`dspark-q8_0.gguf` 1,984,570,528 B／`dspark-f16.gguf` 3,725,780,128 B（不採用だが再利用可）
- 救出差分: `logs/brief23/exl3cu-k6fast.diff`／fix A のスパン証明: `logs/brief19b/span_proof_final.py`／#24のハーネス: `logs/brief24/prompts/`
- 調査: `RESEARCH-prior-art-specdec.md`（出典10本）、`RESEARCH-sol-specdec.md`（Sol、163k tokens）
- 分析ツール: `scripts/offline-ngram.py`、`scripts/probe3.py`、`scripts/run-deploy-probes.sh`、`scripts/lib-bench.sh`＋`HARNESS.md`、`scripts/claude-advisor.sh`、`scripts/muse-resume.sh`、`scripts/kill-sessions-safely.sh`、`scripts/deploy-watchdog.sh`

## 6. 訂正済みの誤り（再発防止・KNOWN-ISSUES 冒頭に固定）
1. **Volta に uniform datapath は無い**（Turing sm_75 以降）→ UR=0 は正常、追わない
2. **命令数 ≠ サイクル**（VoltaはFP32/INT32を別パイプで並行発行）→ 判定はカーネル実時間
3. **Amdahl の f を測る前に E2E 倍率を語らない**（46.7→30msには f≈83%必要、25msは f=100%でも不可）
4. **SASS比較は同一カーネル・同一PC範囲・同一フラグのみ**（11.22 vs 19.31 の逆転は無効）
5. 採用基準は**カーネル実時間**（命令数・UR数は診断）

## 7. 直近の運営事故と対策（KNOW-ISSUES KI-1〜KI-5）
KI-1 空contentをゲートにしていた／KI-2 セッションがGPUを奪ってプローブ切断／KI-3 相談がターン上限で空回答／KI-4 `cmd /c start` がデスクトップにダイアログ／KI-5 セッションkillで孤児サーバ＋配備停止

## 8. 認識合わせ（Claude監査 A/D/E、2026-09-25）で判明した「危うい前提」
**最重要: ラボツリーには MMA8 も GROUP2 も無い。** `H:\exl3-lab\engine\ggml\src\ggml-cuda\exl3.cu` は
`EXL3_EXPERIMENTAL_MMA8` / `GROUP2` の grep ヒット **0**（監査A）。配備側（`H:\exl3-local\exllamav100`）には
両方あり、`getenv("EXL3_EXPERIMENTAL_MMA8")` は 248行目、`exl3_gemv_splitk_kernel<K,cb,GROUP2>` は2行ずつ処理。
⇒ **ラボ `build/bin` で出した数値（34.77 tok/s・28.76 ms・cb別サイクル等）は配備カーネルを測っていない。**
⇒ **fix A / MMA8-fast の A/B に `build/bin` を使わない。** 配備＝`H:\exl3-local\build-v100\bin\llama-server.exe`。
- 配備は起動スクリプト line 99 で **`EXL3_EXPERIMENTAL_MMA8=1`** を設定（配備の記述に MMA8 を必ず含める）
- 配備の CMakeCache は **`GGML_CUDA_FA_ALL_QUANTS:BOOL=OFF`**（ラボと差＝測定時に揃える）
- `start-server-exl3.ps1` 冒頭の **24.24 tok/s / 57.68% は `-c 3276` 時代の古い数字**（引用しない）
- 64k=14,189 / 128k=15,981 MiB は**clip をGPUに載せていた時代**の値（差の1,792 MiBは純KVではない）
- **DFlash2 は「verify行のコスト」で負けている**（コミットは 4.09 > MTP 3.04）⇒ **MMA8-fast 後に30分で再判定**
- **n-gram の敗北は人工物の疑いが濃い**: 256トークン予算を thinking が食い、n-gram 段に到達していない。オフライン8.98は「99.4%反復のフィラー文書」で測った値。⇒ 配備バイナリ＋thinking off で**1時間の再試験**（ゲート: ctrl copy-23k ≈29.9 tok/s／ctrl > no-spec／引用がバイト一致）
- **jusko は正しい理由で閉じていない**: 我々の最大損失＝長文脈decode（32.5→19.6）で、jusko はまさにそこを狙う（100k で 36.47 tok/s の報告あり）。ただし再開は16-24h ⇒ 閉じたままにするなら**理由を書き直す**。`RESEARCH-prior-art-specdec.md:50` が「本命」、STATUS §4 が closed で**我々のファイル内で矛盾**
- **「1.94×」の出典は EXL3/QLoRA ではなく v100-skinny**（監査E）⇒ 「**split N, not K**（活性共有）」という未検証アイデアを隠している ⇒ `exl3_mma8_splitk` のオペランド対応を確認する
- 「~200 tok/s」は**目標にしない**（NInfer の no-spec decode と我々の 21.42 を比較してから）
- **KI-6**: `claude-advisor.sh` の出力が同一秒で衝突し、並列相談の回答が消える（B/C が消失）⇒ ファイル名に pid＋質問名スラグを追加済み。並列時は各回で退避コピーも取る

## 11. Claude の1〜100採点（`logs/advisor-20260926-023039-62719-consult-score-all-options.md`、$2.00・23ターン）
**較正**: 100＝「配備tok/sが≥+10%（または他の3行を決める）、≤2h、出力が証明付きで不変、高信頼」／50＝「約1日で+3〜5%、または1h以内・1/3の確率で≥5%」／20未満＝やらない。スコア≒P(成功)×利得÷時間＋情報価値。
**Roofline（採点の基準）**: 1ラウンドで読む量＝検証重み12.7GB＋3×(ヘッド0.954GB＋MTP層0.17GB)≒**16.1GB** ⇒ 実効830GB/sで**19.4ms/round＝約157tok/s**。ドラフトなしの床は15.3ms/token（実測46.68＝33%）✓ 命令律速の床を含めた**実用上限は25〜30ms/round＝約100tok/s** ✓ **現在87.66ms/roundは roofline の22%** ✓ 結論: **ボトルネックは「in-flight バイト数と占有数」であって演算ではない** ✓
| 順位 | 項目 | スコア | 期待 | 工数 |
|---|---|---|---|---|
| 1 | **R 計測**（verify 停止直前に `llama_synchronize` ＋ `EXL3_SPEC_PHASES=1`・build-slow） | **92** | 0ms（draft＋verify＋残りを1回で閉じる＝他行の前提） | 1.5h |
| 2 | **Q 条件一致A/B**（配備MMA8 on/off・lab 07928d8・build-slow の4腕） | **80** | 0（無料の+5%があり得る／「配備が遅い」矛盾の解消） | 1h |
| 3 | **L 語彙制限（6bitヘッド・ドラフトのみ）** | **70** | ヘッド読取 3×0.954GB → 3×0.12〜0.35GB ＝ **−5.5〜−7.5ms/round、正味+4〜7%（128kで+2〜4%）**／**検証はフルヘッド＝出力不変・VRAM増ゼロ** | 4-6h |
| 4 | **NEW-1 K=4 を MMA8 化** | **66** | K=4は1forwardで約206/686本が dense2（2回復号・1block/SM）⇒ **−3〜−10ms/round、+4〜13%** | 5-8h |
| 5 | **NEW-2 MMA8のグリッド充填**（240→480/960） | **62** | Littleの法則: 約13warp×192B×80SM≒200KB in flight 対 必要415KB ⇒ **0〜−6ms/round**／K=6のnullと272GB/sを同じ式で説明 | 1-1.5h |
| 6 | G 登録済みプリフェッチ（split-Kの `uint4 pf[PF]` を移植） | 55 | −2〜−10ms/round | 4-6h |
| 7 | S-bs ターゲット側サンプリング（`-bs`） | 50 | 0〜−4ms（**行方不明の9.27msの第一容疑**＝毎round 最大4×1MBのlogitsをCPUへ移し248,320件でtop-k→top-p→min-p→temp） | 0.5h |
| 8 | P′ 深さスロープ（`-d 0,32768,65536,114688`） | 55 | 0ms（jusko移植の可否を決める） | 0.5h |
| 9 | H 占有数を上げる | 45（NEW-2経由） | NEW-2に統合 | — |
| 10 | **A MMA8-fast 本体** | **38**（旧+20%から降格） | 価値はGとNEW-2に移った／復号部分は0〜−4ms／**「8 warp/block」は罠**（REG95で256threads＝2blocks＝16warp/SM＜現行20） | 10-14h |
| 11 | O n-gram | 30 → **実測で5〜10に下方修正**（#28: ゲート通過で −24.4%） | コピー系のみ+5〜15% | sunk |
| 12 | E カーネル融合 | 30 | verify は既にグラフ再生なので節約はGPUバブル分のみ＝**−1.5〜−3ms** | 8-12h |
| 13 | F CUDA graphs | 30 | ほぼ解決済（verify は再生・ホストdispatchは180roundで10回のみ） | 0.3h |
| 14 | M 深い投機（k=7でrows=8） | 22（L後35） | 現状は正味損失 | 0.3h |
| 15 | J split N, not K | 25 | −1〜−3ms | 6-10h |
| 16 | P jusko移植（P′次第） | 22 | 128kで+20%があり得る | 24-32h |
| 17 | K 小型ヘッド再量子化 | 18 | Lが上位互換（VRAM増＝容量制約違反） | — |
| 18 | D prepack | 15（「延期」へ降格） | G+NEW-2の後に効く bytes/load の話 | 8-12h |
| 19 | I `xh` のfp16化 | 15 | 0〜−1.5ms（Aの中で） | 3h |
| 20 | **B fix A 単体** | **12** | **MTPでK=3はrows=1を通らない**（rows∈{2,4,28,32,42}）⇒ MTPには無効／REG 142→102 は**ドラフトなしdecodeのみ**に効く | 1h（構築済） |
| 21 | N DFlash2再開 | 8 | 検証はrows>8＝GEMM経路なのでMMA8-fastは効かない | 2h |
| 22 | 永続カーネル / C K=6 / L2永続化 / PCIe系 / KV圧縮 | 6 / **3** / 1 / 2 / 0 | やらない | — |
**順序（Claude指定）**: ①**1ロック1.5hで R+Q+S-bs+F+P′ をまとめて**（70.2 vs 79.0ms の矛盾を解消し9.27msの所在を決める）②**GPU不要30分**でK別バイト国勢調査（NEW-1の可否）＋ヘッドのカバレッジ対B曲線（Lの可否、日本語と英語で）③**それから1つだけ作る**（カバレッジが持てばL、無ければNEW-2→NEW-1）
**削除**: C と「K=6は順位2」/ B の配備項目化 / I と 8warp の単独項目 / K / N / jusko（P′が正当化するまで）/ `run19b.py` / **EXL3カーネルのラボバイナリA/B一切**（約2倍遅さと受理0.615の謎が解けるまで）
**Claude の自己撤回8件**: 「verify≈70msと80%」／「draft 17.5ms」（→8.14〜11.0ms）／「+20%（34.8→42）」と「K=6 +5〜10%」／「M=2..8はGROUP2」（rows4/8はMMA8、**K=4を見落としていた**）／「8 warp/blockは無条件に良い」／prepack「〜0」（延期へ）／「起動を2/3削減」／行番号 `:290-321`→`:264-296`
**我々が完全に見落としていた2件**: ①**K=4はMMA8を通らない**（検証行列の約30%＝1forwardで206本が dense2 で2回復号）②**ターゲット側サンプリングがCPUで248k語彙を回している**（9.27msの第一容疑）
**訂正（我々側）**: 「`xh`が共有メモリ」は**我々の要約が誤り**（Claudeの指示書は "global memory" ✓ `exl3.cu:992` ✓）— 監査の指摘を我々が誤って一般化していた。また `bench/draft-vocab-32k.json` は**名前が誤り**で、実際は5,797 id（N=8192でカバレッジ100%、密集は〜20k以下、外れ値45個が247,438まで）

## 13. 方向の記録（2026-09-26 本人指示）「転送速度が余ってるので KV を切り詰めてもメモリしか美味しくない」
- **実測の裏づけ**: 実効 **272 GB/s** 対 実用上限 **~830 GB/s ＝ roofline の 33%** ✓ 律速は**帯域ではなくレイテンシ（in-flight バイト数と占有数）** ✓（Claude採点 §2 と同じ結論 ✓）
- ⇒ **KV をこれ以上削っても速度は買えない**（メモリだけ浮く ✗）→ **余った帯域は精度側に使う** ✓ 第一歩 = **`-ctk q5_1 -ctv q4_0`**
- **落とし穴（特定済み）**: **配備もラボも `GGML_CUDA_FA_ALL_QUANTS:BOOL=OFF`（2026-09-26 実測、両キャッシュ）** ⇒ `fattn.cu:340-352` が **q5_0/q5_1/q4_1 を FA 非対応**と判定 ✓ → `-ctk q5_1` を渡すと**ロード拒否か FA の黙落ち** ✗ ⇒ **`FA_ALL_QUANTS=ON` で新規ビルドした `build-v100-fa` が前提** ✓（`build-v100` は触らない ✓）
  - 訂正: 監査Aは「配備とラボでこのフラグが違う」と示唆していたが、**実測では両方 OFF** ✗ → ラボ/配備のビルド差は**このフラグではない**（差の正体は未確定 ✓ 採点表の Q で測る ✓）
- **KV 型の履歴（訂正）**: 当初の **24.24 tok/s は `-c 32768`・`q8_0 KV`** で測った記録 ✓（監査Aの「3276」は読み違い ✓）→ **2026-09-24 15:50 頃に q8_0→q4_0 へ変更** ✓ ＝「量子化サイズが変わった」は事実 ✓
- **判定基準（#31）**: 速度は A/A ノイズ（~1%）以内 ✓／`n_ctx=131072` と FA の生存をログで確認 ✓／**同一固定コーパスでの PPL 2本（q4_0/q4_0 と q5_1/q4_0）** で「+400 MiB が何を買うか」を実測 ✓ → 勝てば配備を新バイナリ＋q5_1 に切替（1行で戻せる形で記録 ✓）

## 12. 打ち切りの判断材料（2026-09-26、本人「全体で20%以上超えないならここでバラす」への回答）
**測定済みの現在地**: 87.66ms/round（配備 6k=32.52 → 71k=19.60 tok/s、prefill 345→248）。roofline 19.4ms/round（157tok/s）、実用上限 25〜30ms/round（約100tok/s）⇒ **理論余地は約3倍** ✓
**しかし「実装して実測した」項目は正の数字が1つも無い**: fix A単体 ✗・K=6 −3.9% ✗・n-gram −24.4%（ゲート通過後）✗・DFlash2/DSpark/CPU常駐/Volta fork ✗。採点表の +X% はすべて INFERENCE か MEASURED-BY-OTHERS ✓
**128k（実運用の文脈長）での現実的な積み上げ**: L +2〜4%／NEW-1 +2〜6%／Q 0〜+5%（無料かもしれない）／S-bs 0〜+5%／G+NEW-2 0〜+7% ⇒ **現実線 +5〜12%** ✗（各1〜8h・単桁%）
**結論**: 本人の条件（20%未満なら畳む）に**該当する公算が高い** ✓
**畳む前に絞って取る候補（2つだけ）**:
1. **#30 の CPU 側30分**（K別バイト国勢調査＋ヘッドのカバレッジ曲線）→ **L（ドラフト語彙制限）**の可否が決まる ✓ Lは「**検証はフルヘッド＝出力不変・VRAM増ゼロ・容量も品質も動かさない**」唯一の正の候補（−5.5〜−7.5ms/round 見込み ✓）
2. **Q の4腕A/B**（1h）→ **無料の+5%（ビルド/設定差）**が出るかもしれない ✓ 出れば配備を直すだけ ✓
**畳むときの残し方**: 配備 8081=200 ✓／ロック FREE ✓／稼働セッションなし ✓／`STATUS-NOW.md`（§0〜§12）＋`KNOWN-ISSUES.md`（KI-1〜KI-8＋測定規律）＋ Claude採点表（`logs/advisor-20260926-023039-*.md`）＋ `logs/brief26..30/REPORT.md` で完全に再現可能 ✓

## 10. 監査 B/C の成果（Muse #27、`logs/brief27/`）— 数値の土台が2つ崩れた
**B（数値は閉じるか）**: `AUDIT-B-answer.md`
- **「MMA8 verify ≈70ms」はカーネル計測ではない** ✗: 70.246ms は `llama-bench -p 4` の**モデル全体**forward（**KV空**）のサンプル2-5の平均で、しかも**MMA8を含まないラボバイナリ**（`brief14/baseline.m4.json:44`）
- **分解が閉じない**: draft 8.14（実測 `brief14/...server.err.log:314`、1367.109ms/168回）＋70.246＋0.0014 ＝ **78.39 vs ラウンド87.66＝10.6%が未計上** ✓ かつ**その8.14msも `llama_synchronize` 無しの計測**＝非同期が逃げている疑い
- in-serverで言えるのは**「非ドラフト＝ラウンドの90.7%」**まで ✓（「verify 80%」は推論）
- **堅い数字**: ドラフトなし46.68ms/token、MTP k=3 28.76ms=34.77tok/s、受理0.68327(343/502)、平均コミット3.04（すべて**ラボ**カーネル）／速度比は同一ラウンド内で **1.60〜1.62×**
- **未解明**: 同一フラグで受理率が 0.683(brief14) vs 0.615(brief21/25) と違う理由／「配備が10〜12%遅い」は**同一条件のA/Bが存在しない**／コードブックのサイクルは合成1回のみ
- 助言側の誤り: 「M=2..8 は GROUP2」✗（rows 4/8 はフラグで MMA8 経路 ✓）／「`xh` は共有メモリから再ロード」✗（`xh` は **global pool** ✓）
**C（コード読解の真偽）**: `AUDIT-C-answer.md` — 外部アドバイザの主張は**配備ソースで TRUE** ✓（4 warp/192B/二重バッファ無し/重みごと変換/4行で1回復号/`EXL3_EXPERIMENTAL_MMA8=1` が M=4 経路 ✓）。**ラボツリーでは全て FALSE**（MMA8カーネル自体が無い ✓）。行番号のドリフト2件 ✓
**推奨される最初の Step 3 計測**（#29 のブリーフに追記済み）: verify タイマー停止の直前に `llama_synchronize(ctx_tgt);` を入れ（`server-context.instrumented.cpp:3645` の型 ✓）、`EXL3_SPEC_PHASES=1` ＋512トークンのgreedy プローブ1回で **draft平均＋verify平均＋ラウンド**が同時に取れて和が閉じる ✓

## 9. 配備カーネルのSASS実測（2026-09-25、初の配備バイナリ直接測定）
**手段**: `cuobjdump -xelf all` で配備バイナリから cubin を抜き、**11.8版 `nvdisasm -c`**（`C:/llm-local/cuda118-extract/cuda_nvdisasm/nvdisasm/bin/`、cuobjdump も同所の `cuda_cuobjdump/` に）で逆アセンブル。13.1版は sm_70 を拒否するので使えない（KI-8）。**`/h/...` をネイティブ python に渡すと死ぬ**ので `H:/` 形式で。
- **`exl3_mma8_splitk_kernel<2>`（K=3, cb=2, 配備 = M=4検証カーネル）**（`ggml-cuda.23.sm_70.cubin`、REG 95 / SHARED 768 B / 128 threads=4 warp）
  - **関数本体 768 命令、メインループ（`.L_x_25`）＝ 658 命令**。ループ内の `F2F` がちょうど 32（＝ソースの「重みごとに1回 fp16 変換」✓）なのでループ1周＝**32重み/レーン相当** ⇒ **約 20.6 命令/重み**
  - 内訳（658）: **IMAD 221 (33.6%) / SHF 77 (11.7%) / IADD3 68 (10.3%) / LOP3 42 (6.4%) / F2F 32 / LDS 29 / LDG 18 / LEA 17 / ISETP 16 / IDP 16 / HADD2 16 / FFMA 16 / PRMT 16 / HMMA 16** ⇒ **帳簿系 76.1% / 数学系 14.7%、HMMA は 2.4%**
  - ⇒ **verify 側は「重み1個あたり約20命令のうち約15命令が添字計算」**。ここが MMA8-fast の伸びしろの実測値
  - ロードは LDG 18 + LDS 29 ＝ 47 に対し 658 命令 ⇒ **静的に見て明らかに命令発行律速**（メモリ律速ではない）
- 参考（関数全体、ループ未分離なので per-weight には使わない）: `gemv_splitk_kernel<3,2,GROUP2=true>` 1,848命令（IDP 64＋HADD2 64＝64重み相当）/ `gemv_splitk_kernel<3,2,GROUP2=false>` 1,456 / `gemv_kernel<3,2>` 1,608
- **`exl3_gemv_splitk_kernel<6,2>`/`gemv_kernel<6,2>`（K=6、配備）**: 1,364 / 1,626 命令、**21.31 / 25.41 命令/重み**（64重み/レーン/1K行）、帳簿系 56%
- **注（自己訂正）**: 初回に「832命令・帳簿80.9%」と報告したのは **cb=0 のカーネルを拾っていた**（セクション境界の取り違え）。`.size <name>,(.L_x_N - <name>)` で境界を厳密化した上記が正しい値
- **配備ソース（`exl3.cu:427` の mma8）で確認できた事実**（外部アドバイザの主張と一致）: 1ブロック **4 warp（128 threads）**／ステージングは `2*8*3` words＝**192 B**／`__ldcs` で読み `__syncwarp()` してから復号＝**二重バッファ無し**／`xh` を**重みごとに fp32 で再ロード**／**8重み/レーンを `#pragma unroll`**／`mma.sync.m8n8k4` を kタイルあたり4発
- ⇒ **含意**: 「定数シフト抽出は M=1 経路しか触れない」は*コード配置*の話で、**同じアイデアを MMA8 の復号ループに持ち込む（=MMA8-fast）のが本命**という順位付けは、この 80.9% という実測で支持される
