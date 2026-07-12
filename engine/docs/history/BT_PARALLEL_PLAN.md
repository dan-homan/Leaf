# Plan: Threaded Batch Trainer + Sharding Cleanup

**Status: IMPLEMENTED 2026-07-08.**  Written for implementation by Claude (Opus)
and carried out the same day.  Read `OFFLINE_TRAINING.md` first for background on
the batch trainer and the hybrid loop.

## Outcome (2026-07-08)

Delivered `--bt-threads N` (within-batch thread parallelism) and removed the
`--bt-sync` multi-process sharding, per this plan.  Measured on a 570k-row
game-ply corpus (v12 net, 2 epochs, 8-core M-series): **T=8 ≈ 2.85× T=1**
(30k → 86k pos/s; epoch 18s → 6s).  **`--bt-threads 1` is bit-for-bit identical
to the pre-change trainer** (`.tdleaf.bin` and `.nnue` both `cmp`-clean in a git
A/B), and T=8 matches T=1 to ~5 decimals in val-MSE (float reduction order only).

Two things the plan under-estimated, both discovered by the per-epoch phase
timing this plan mandated, and both fixed here rather than deferred:

1. **The serial tail was ~50–70% of batch time, not "small."**  It was dominated
   by (a) `nnue_requantize_fc` re-rounding the *entire* 23M-weight FT array every
   batch, and (b) the FC-stack + FT-row Adam in `nnue_apply_gradients`.  Fixes:
   - **Targeted requantize** (`nnue_requantize_fc_applied`): re-quantize only the
     rows the Adam step touched (`g_applied_ft_rows`).  Bit-identical to the full
     requantize (untouched rows already hold an int16 matching their FP32 shadow).
   - **Parallel apply** (`nnue_apply_gradients_parallel`): the 8 FC stacks and the
     FT/PSQT rows are disjoint, so they parallelize across the same pool.  The
     serial `nnue_apply_gradients` is left **untouched** (online path + T=1 use it);
     the parallel path mirrors its per-stack/per-row bodies via shared helpers
     (`nnue_adam_step`, `nnue_apply_fc_stack`, `nnue_apply_ft_rows`).  FT rows are
     partitioned by the dirty-row **list** (not the feature-index range) because
     dirty rows cluster by king-bucket — this balancing was worth ~0.6× on its own
     (67k → 86k pos/s).
2. **This was the plan's "stage 2," which the plan said to defer.**  The 70% tail
   measurement justified doing it now; deferring would have capped the speedup at
   ~1.5×.

Remaining lever (not done, diminishing returns): the reduce phase (~1.3s at T=8)
has the same king-bucket index-range imbalance as the apply FT rows did; a
list-partitioned reduce would recover perhaps another ~0.3–0.4× but touches the
delicate deterministic-merge path.  Left for a future pass.

---

## Original plan (as written before implementation)

---

## 1. Goal and rationale

Replace the failed multi-process sharding (`--bt-sync`) with **within-batch
thread parallelism** in a single trainer process, then delete the sharding
machinery from the trainer, `train.py`, and the docs.

Why this is safe where sharding was not: `--bt-sync` ran N independent Adam
optimizers, each taking ~256 full steps on its own diverging weight copy
between delta-merges.  That gradient *staleness* destroyed the subtle gen-2+
consolidation signal (every 8-way arm regressed; the identical unsharded run
gained — see `OFFLINE_TRAINING.md` "Frontier caveat").  Within-batch
parallelism is **synchronous data parallelism**: all threads compute gradients
for positions of ONE batch against the SAME frozen weights, gradients are
summed, and ONE optimizer takes ONE step.  It is mathematically identical to
the current single-process trainer — same weights seen by every gradient, same
sequence of Adam steps, same batch size (512), same LR calibration — up to
floating-point summation order, which we fix (see determinism, §4.5).

Target: ≥4× throughput at 8 threads (31k → 130k+ pos/s), turning the ~3–4 h
single-process consolidation run into under an hour, with `--shards 1`
semantics.

**Explicit non-goals** (do not do these):
- GPU training, PyTorch port, or changing `--bt-batch` (batch size is a tuned
  hyperparameter; threading at fixed batch changes nothing that was tuned).
- Threading the *online* TDLeaf path (game play + `tdleaf_update_after_game`).
  Online training stays exactly as it is.
- Removing the POSIX-locked delta-merge save protocol in
  `nnue_save_fc_weights` / `nnue_load_fc_weights` (`nnue_training.cpp`).
  **Online multi-process self-play training depends on it** (10 concurrent
  writers on one `.tdleaf.bin`).  Only the batch trainer's *use* of it for
  sharded sync (`--bt-sync`) is removed.

---

## 2. Current structure (verified 2026-07-08)

Epoch loop, `src/nnue_batch_train.cpp:475-515`:

```
for each shuffled train position:
    bt_eval_record(r, K, &act)        # decode FEN → nnue_init_accumulator →
                                      # nnue_evaluate_acc_raw (int8, score) →
                                      # nnue_forward_fp32 (FP32 activations) →
                                      # feature-index harvest
    nnue_accumulate_gradients(act, grad_scale, false)   # writes global grad arrays
    every 512:
        nnue_clip_gradients(TDLEAF_GRAD_CLIP_NORM)      # global L2 norm + scale
        nnue_apply_gradients(lr_scale)                  # Adam, dirty rows only
        nnue_requantize_fc()                            # FP32 → int8 inference arrays
```

Thread-safety audit of the compute phase (all verified clean — they read only
weight arrays that are constant within a batch, and use stack/argument scratch):

| Function | File | Shared state touched |
|---|---|---|
| `nnue_init_accumulator` | `nnue.cpp:179` | reads int16 FT weights only |
| `nnue_evaluate_acc_raw` → `nnue_evaluate` | `nnue_training.cpp:2409`, `nnue.cpp:491` | reads int8 inference arrays only; local `l0_in` |
| `nnue_forward_fp32` | `nnue_training.cpp:619` | reads `l*_weights_f32` only; writes caller's `act` |
| `halfkav2_feature` | `nnue.cpp:121` | pure |
| `nnue_accumulate_gradients` | `nnue_training.cpp:705` | **writes global grad arrays — the one thing to fix** |

Global gradient state in `nnue_training.cpp` (all file-scope statics):

| Array | Decl | Size |
|---|---|---|
| `grad_l0_w[8][16×1024]`, `grad_l0_b[8][16]` | :36-37 | 512 KB + tiny |
| `grad_l1_w[8][32×32]`, `grad_l1_b[8][32]` | :38-39 | 32 KB + tiny |
| `grad_l2_w[8][32]`, `grad_l2_b[8]` | :40-41 | ~1 KB |
| `grad_ft_w` (heap, 22528×1024) | :75 | **92 MB** |
| `grad_psqt_w` (heap, 22528×8) | :76 | 720 KB |
| `ft_dirty` (heap, 22528 bool) | :77 | 22 KB |
| `grad_ft_bias[1024]` | :84 | 4 KB |

Total per-thread copy ≈ **93 MB** (8 threads ≈ 750 MB — acceptable).

Trainer-local statics that must become per-thread: `bt_eval_record`'s
`static position pos` and `static NNUEAccumulator acc`
(`nnue_batch_train.cpp:248,251`), and the epoch loop's
`static NNUEActivations act` (:461).

Readers/consumers of the global grad arrays that stay **serial and unchanged**
in stage 1: `nnue_clip_gradients` (:959), `nnue_apply_gradients` (:1070,
iterates `ft_dirty` rows, clears them at :1275), `nnue_requantize_fc` (:1346),
and the zeroing memsets in the two init paths (:440-442, :559-561, :584).

---

## 3. Part A — engine: `NNUEGradBuf` refactor (`nnue_training.cpp`)

Mechanical refactor; **online TDLeaf behaviour must be bit-identical after it.**

1. Introduce a struct owning everything a gradient-accumulation target needs:

   ```cpp
   struct NNUEGradBuf {
       float grad_l0_w[NNUE_LAYER_STACKS][NNUE_L0_SIZE * NNUE_L0_INPUT];
       float grad_l0_b[NNUE_LAYER_STACKS][NNUE_L0_SIZE];
       float grad_l1_w[NNUE_LAYER_STACKS][NNUE_L1_SIZE * NNUE_L1_PADDED];
       float grad_l1_b[NNUE_LAYER_STACKS][NNUE_L1_SIZE];
       float grad_l2_w[NNUE_LAYER_STACKS][NNUE_L2_PADDED];
       float grad_l2_b[NNUE_LAYER_STACKS];
       float grad_ft_bias[NNUE_HALF_DIMS];
       float *grad_ft_w;     // heap: FT_INPUTS × HALF_DIMS
       float *grad_psqt_w;   // heap: FT_INPUTS × PSQT_BKTS
       bool  *ft_dirty;      // heap: FT_INPUTS
   };
   ```

   The existing statics collapse into one global instance (`g_grad`), heap
   parts allocated where `grad_ft_w` is allocated today
   (`nnue_init_fp32_weights`, :543-545).  Every existing reference in
   `nnue_accumulate_gradients` / `nnue_clip_gradients` /
   `nnue_apply_gradients` / the init memsets becomes `g_grad.<field>` —
   a rename, no logic change.

2. Re-signature the accumulator:

   ```cpp
   void nnue_accumulate_gradients(const NNUEActivations &act, float grad_scale,
                                  bool replay_mode, NNUEGradBuf *gb = nullptr);
   // gb == nullptr → &g_grad  (online callers unchanged, zero risk)
   ```

   Inside, replace direct global references with `gb->`.  The online call
   sites (`tdleaf.cpp`) don't change at all.

3. Add helpers used by the batch trainer only:
   - `nnue_gradbuf_alloc/free(NNUEGradBuf&)` — heap parts, zeroed once.
   - Per-thread **dirty-row list**: alongside the `ft_dirty` bitmap, worker
     buffers also keep a `std::vector<int> dirty_rows` appended on first
     touch of a row.  This makes both the reduction and the post-reduce
     clearing O(dirty), **never** a 92 MB memset per batch (that would erase
     the speedup).  Simplest home: a thin wrapper struct in
     `nnue_batch_train.cpp` (`struct BTWorker { NNUEGradBuf gb;
     std::vector<int> dirty_rows; position pos; NNUEAccumulator acc;
     NNUEActivations act; double se; ... }`), with the dirty-list append done
     in the trainer-side accumulate wrapper — or pass the list into
     `nnue_accumulate_gradients` via the buf struct.  Implementer's choice;
     keep `nnue_training.cpp` changes minimal and put orchestration in
     `nnue_batch_train.cpp`.

`nnue_clip_gradients`, `nnue_apply_gradients`, `nnue_requantize_fc` keep
operating on `g_grad` only, serial, exactly as today (stage 1).

---

## 4. Part A — trainer: thread pool and batch pipeline (`nnue_batch_train.cpp`)

### 4.1 New flag

```
--bt-threads N      worker threads for gradient computation (default 1)
```

Default **1** = exact current behaviour (and the code path is the same one,
just with one worker).  `train.py` passes 8 by default (§5).  Clamp to
[1, 16], print the value in the startup banner alongside lambda/K/etc.
No OpenMP — plain `std::thread` (already used elsewhere in the engine; no
new build flags in `comp.pl`).

### 4.2 Pool

A persistent pool created once before the epoch loop, destroyed at exit:
T worker threads parked on a condition variable; the main thread publishes a
job (phase tag + batch range), wakes workers, waits on a completion counter.
~60–80 lines.  Do not spawn threads per batch (660k batches on a full run).

### 4.3 Per-batch pipeline

Batch = the next `batch` (512) indices of the shuffled `train_idx`, exactly as
now.  Two parallel phases, then the existing serial tail:

**Phase 1 — compute (parallel).**  Worker t takes the contiguous slice
`[t·512/T, (t+1)·512/T)` of the batch (contiguous, not strided — so T=1
reproduces today's accumulation order exactly).  For each position:
`bt_eval_record` (using the worker's own `pos`/`acc`/`act`) → target → error →
`nnue_accumulate_gradients(act, scale, false, &worker.gb)`.  Squared errors
accumulate into `worker.se`.

**Phase 2 — reduce (parallel, deterministic).**  Partition work so no two
threads write the same global memory, and always sum workers in index order
0..T−1:

- **FT + PSQT rows**: partition feature-index space `[0, 22528)` into T
  contiguous ranges.  Reducer thread p walks each worker's `dirty_rows` (in
  worker order), and for rows inside its range adds the worker's
  `grad_ft_w` row (1024 floats) and `grad_psqt_w` row (8 floats) into
  `g_grad`, setting `g_grad.ft_dirty[fi]`.  (Per-worker dirty lists are short
  — ≤ 64 positions × ~64 rows — so scanning all T lists per reducer is cheap;
  alternatively pre-bucket by range.  Either way the *addition order* per row
  must be worker 0, 1, … T−1.)
- **FC grads + `grad_ft_bias`** (~560 KB total): one designated thread sums
  all workers in order.  Tiny.
- **Epoch MSE**: main thread sums `worker.se` in worker order into `se`.

**Phase 3 — clear (parallel, O(dirty)).**  Each worker zeroes only its own
touched state: rows in its `dirty_rows` (FT + PSQT), its FC grad arrays
(memset, small), `grad_ft_bias`, clears the list and its bitmap entries.

**Phase 4 — serial tail (unchanged).**  `nnue_clip_gradients` →
`nnue_apply_gradients(lr_scale)` → `nnue_requantize_fc` on `g_grad`, exactly
the current three calls, including the tail-batch flush after the epoch loop.

Add lightweight per-epoch timing of the four phases (accumulated
`steady_clock` sums, one line per epoch).  This tells us whether a stage 2 is
warranted:

**Stage 2 (only if the serial tail dominates the profile — do not build
pre-emptively):** parallelize the FT dirty-row loops inside
`nnue_clip_gradients` and `nnue_apply_gradients` with the same row-range
partition (each row's clip contribution / Adam update is independent).
Watch the serial bits inside apply: `t_adam`, delta counts, save-dirty
bitmaps, and the `TDLEAF_LOG_STEP_CLIPS` telemetry (if telemetry is enabled,
fall back to the serial loop).

### 4.4 Validation-loss parallelism

`val_loss` (:441) is read-only → trivially parallel with the same pool:
contiguous per-worker slices of `val_idx`, per-worker double partial sums,
summed in worker order.  Worth doing — validation runs over 5% of the corpus
every epoch.

### 4.5 Determinism contract

- Fixed `(--bt-seed, --bt-threads)` ⇒ bit-identical run.  Shuffle order,
  slice assignment, and reduction order are all deterministic by
  construction.
- `--bt-threads 1` must reproduce the **current** trainer: identical printed
  per-epoch train/val MSE, and identical weights in the epoch snapshots.
  (Pedantic caveat: adding a worker buffer into zeroed globals can flip a
  +0/−0 sign on a gradient; the `!= 0.0f` guards treat them identically, so
  weights are unaffected.  Gate on weights + printed MSE, not on a byte-diff
  of intermediate gradient memory.)
- Different T values will differ in float summation order → different
  low-order bits.  That's expected; strength is checked statistically (§7).

### 4.6 Remove `--bt-sync` (the failed sharding experiment)

In `nnue_batch_train.cpp`:
- Flags `--bt-sync` / `--bt-sync-every` (:52-54 doc block, :325-326, :347-348,
  :360-362).
- `do_sync` lambda, `batches_since_sync`, the periodic call at :501-502, the
  epoch-end call at :518, and the header's "Multi-process data parallelism"
  paragraph (:58-66).
- Per-epoch snapshot writes (`nnue_write_nnue` + `nnue_save_fc_weights`)
  **stay** — they're the normal output path.

**Keep** everything in `nnue_training.cpp`'s save/load path: the POSIX
locking, delta-merge, deferral logic all serve concurrent *online* training.

---

## 5. Part B — `scripts/train.py` simplification

1. **Arguments**: delete `--shards` (:225) and `--sync-every` (:231).  Add
   `--bt-threads` (default 8), forwarded to the trainer.  Delete the
   `--gauntlet-epochs requires --shards 1` check (:264-267) — the ladder now
   always works (one process, snapshots are the full state).

2. **Phase 4 (corpus assembly)**: keep the axis detection/refusal and the
   concatenation loop, but write ONE file `work/corpus.tsv` (re-emitting the
   `# tdleaf-corpus axis=game-ply` marker at the top) instead of N shard
   files.  Don't pass the raw dump-file list on the trainer command line —
   a 400k-game generation can leave hundreds of per-pid dump files and the
   trainer's comma-list buffer is 4 KB.

3. **Phase 5 (training)**: one `Leaf_vbt` invocation —
   `--batch-train ../corpus.tsv --bt-out <tag> --bt-threads N ...` — replacing
   the `procs` loop (:411-427).  Snapshot prefix becomes `<tag>_ep<N>.*`
   (drop the `_p0` shard-worker infix; update `rate_epoch` :434 and the
   watcher globs :461, :484 to match).  The SIGSTOP/SIGCONT pause around
   ladder matches (:465-471) stays — it's about core contention, not
   sharding.  The non-ladder branch (:487-495) waits on the single process.

4. **Final-net export**: delete the zero-LR merged-export block (:507-528,
   including `tiny.tsv` and the sync-file copy) — it existed only because the
   merged state lived in the sync file.  Replace with a copy of a chosen
   epoch snapshot to `learn/<tag>_final.{nnue,tdleaf.bin}`:
   - with `--gauntlet-epochs`: the ladder's best epoch (max Elo; on a tie,
     the later epoch), logged as e.g. `final = epoch 4 of 6 (ladder best)` —
     this encodes the "pick the epoch by ladder" practice that has been
     manual until now (both gen-2 runs peaked at epoch 4 of 6);
   - without: the last epoch (previous behaviour).

5. Docstring (:6-7, :19, :27, :44) and log messages updated accordingly.
   `_corpus_is_game_ply` stays (still used by phase 4).

---

## 6. Part C — docs and references

- `docs/OFFLINE_TRAINING.md`: replace the "Multi-process data parallelism —
  `--bt-sync`" section with a short "Threaded training — `--bt-threads`"
  section (synchronous within-batch parallelism, mathematically identical to
  single-threaded, measured throughput).  Rewrite the "Frontier caveat" as a
  brief historical note: *why* sharding was removed (staleness vs the gen-2
  signal) — keep the lesson, drop the workaround advice.  Update the
  hybrid-loop invocation examples (`--shards 1` → nothing needed) and the
  "Practical guidance" bullet.
- `docs/SCRIPT_USE.md` train.py section (:801-858): phase list, option
  table (`--shards`/`--sync-every` out, `--bt-threads` in, `--gauntlet-epochs`
  no longer says "requires --shards 1"), artifact names (`shards` out).
- `engine/CLAUDE.md`: `nnue_batch_train.cpp` row (:103) — replace
  "multi-process `--bt-sync`" with "threaded (`--bt-threads`)".
- `docs/TODO.md`: drop the "`--bt-sync` frontier fix" item (:79-80) — resolved
  by removal; adjust the recipe mention at :55 if it names `--shards 1`.
- `docs/change_log.txt`: one entry covering both the threaded trainer and the
  sharding removal.
- Delete this plan file (or mark it DONE) at the end.

---

## 7. Validation gates (run in order; stop and investigate on any failure)

1. **Build**: `perl src/comp.pl bt_thr NNUE=1 NNUE_NET=learn/nn-fresh-260628.nnue
   TDLEAF=1 OVERWRITE` compiles clean.  Also build once *without* TDLEAF
   (`NNUE=1` only) to confirm the unity build isn't broken for play binaries.
2. **Online-path regression (refactor safety)**: the `NNUEGradBuf` rename must
   not change online TDLeaf.  Quick check: a few self-play games with a
   TDLEAF binary (e.g. `match.py` 4 games as in the smoke workflow), confirm
   normal learning output and a loadable `.tdleaf.bin`.
3. **T=1 equivalence gate**: on a ~2M-position subset (`--bt-max 2000000`,
   fixed seed, 2 epochs), the new binary at `--bt-threads 1` vs the
   *pre-change* binary: identical printed train/val MSE lines, and epoch-
   snapshot weights identical (`scripts/compare_nnue_learning.py` /
   `diff_tdleaf_checkpoints.py` show zero movement between the two ep2
   `.tdleaf.bin`s).  This is the non-negotiable gate.
4. **T=8 correctness + throughput**: same subset, `--bt-threads 8`: val-MSE
   trajectory matches T=1 to ~4 decimal places (bit-exactness NOT expected),
   throughput reported; target ≥4× (stretch 6×).  Check the phase-timing line:
   if the serial tail (clip+apply+requantize) exceeds ~25% of batch time,
   note it as the stage-2 trigger, but do not implement stage 2 in this pass.
5. **Strength (optional but cheap, recommended)**: train 1 epoch at T=1 and
   T=8 on the same corpus from the same state; 1000-game ladder between the
   two snapshot nets at 1+0.01 → expect ≈ 0 ± 19.
6. **train.py smoke**: `--skip-online --corpus <small.tsv> --force` with
   `--gauntlet-epochs`, 1 epoch, tiny game counts — corpus concat, single
   trainer, ladder, final-net copy, gauntlet all fire; artifact names correct.

## 8. Landmines (read before coding)

- **Do not touch the multi-writer save protocol** (locking/delta-merge in
  `nnue_save_fc_weights`/`nnue_load_fc_weights`) — online training uses it.
- **Never memset a worker's 92 MB FT grad buffer per batch** — clear via the
  dirty-row list only.  This is the difference between 6× and 0.8×.
- The `grad != 0.0f` skip-guards in clip/apply exist so untouched weights keep
  no Adam state; per-worker buffers must be *exactly* zero outside dirty rows
  (guaranteed by the clear-via-list discipline + zeroed allocation).
- Contiguous (not strided) batch slices, reduction in worker order — that's
  what makes T=1 bit-reproduce today's trainer and any fixed (seed, T) run
  reproducible.
- `bt_eval_record`'s statics (`pos`, `acc`) and the loop's static `act` must
  move into per-worker storage — they're the only hidden mutable state in the
  compute phase (audit table in §2; everything else reads batch-constant
  weight arrays).
- `nnue_apply_gradients` consumes and clears the **global** `ft_dirty`
  (:1227, :1275) — the reduce phase must set it for every merged row, or Adam
  silently skips those rows.
- train.py's epoch watcher keys on `_ep<N>.tdleaf.bin` appearing (written
  after the `.nnue`) — keep that write order in the trainer.
