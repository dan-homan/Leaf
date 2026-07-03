# Offline Training & the Hybrid Loop

Leaf's second training mode, added 2026-07-02: **offline supervised consolidation**
of quiet positions harvested from games the engine has already played, complementing
the online TDLeaf(λ) system (see `TDLEAF.md`).  Together they form the **hybrid
loop**: online self-play generates games (and learns as it goes); offline training
extracts the full information content of those games with multi-epoch, shuffled,
all-layer gradient descent; the consolidated net re-enters online play to generate
better games.

---

## Motivation and Results

Online TDLeaf is single-pass and within-game correlated: each game is seen once, in
order, and per-game weight movement decays as Adam's second moment accumulates.
Analysis of the 260628 training run showed the depth-6 self-play Elo curve
saturating ~150–200 Elo below the classical hand-crafted eval, with data *quality*
(not volume, LR, or capacity) as the binding constraint.

Offline consolidation converts the question from "can single-pass TD extract more?"
to "is the information in the data?"  Measured results (2026-07-02, all matches
3+0.05, FRC openings):

| Experiment | Corpus | vs its online endpoint | vs classic_eval |
|---|---|---|---|
| Pilot (4 epochs, single process) | 41.5M positions (d8 + rotation) | +143 ± 22 | −90 ± 18 |
| Pure self-play (6 epochs, 8-way sharded) | 183M positions (260628 d6+d8 only) | **+139 ± 22** | **−87 ± 18** |
| *online endpoint baseline (2.4e6g)* | — | 0 | −214 |

Both runs recovered **+125–130 Elo of genuine cross-family strength** from games the
engine had already played — roughly 60% of the then-remaining gap to classic_eval —
in ~1.5–2 hours of training and zero new games.  Notably the pure self-play corpus
alone matched the mixed corpus: the binding factor was extraction depth, not data
diversity.  A further finding, replicated in both runs: **validation MSE is a poor
proxy for playing strength** (it oscillated while strength rose monotonically) —
rate snapshots by gauntlet, never by validation loss.

Two data-scaling caveats established empirically:
- Elo gain is strongly sublinear in position count: the 183M-position run matched
  the 41.5M pilot because the added d6-era positions (from much weaker net
  generations) carried little information.  A generation of self-play data supports
  roughly +130–145 over its online endpoint, extractable from its best ~40M
  positions.  The route to more is *better games* (the hybrid loop), not more
  epochs over old ones.
- Sharded training does **not** multiply effective LR (total Adam step mass is
  conserved); it adds gradient staleness between syncs, visible as validation
  oscillation but benign in the measured outcomes.

---

## Corpus Format

Tab-separated text, one quiet position per row (produced by both corpus sources
below; `#` comment lines and a `fen`-prefixed header are skipped by consumers):

```
fen  cp  result  ply  depth  gid
```

| Column | Meaning |
|---|---|
| `fen` | Shredder-FEN of the position (castling/ep unused by the trainer) |
| `cp` | eval label, **white POV**, centipawns |
| `result` | game outcome, **white POV**: `1` / `0.5` / `0` |
| `ply` | 1-based ply index within the game |
| `depth` | search depth of the eval label; **0 = no search label** (see below) |
| `gid` | stable game id — the trainer splits train/validation **by game** |

**The depth-0 rule:** records with `depth == 0` (leaf-dump rows, whose `cp` is the
net's *own static eval* — self-distillation, no information) are trained
**outcome-only regardless of `--bt-lambda`**.  Records with `depth > 0` carry a
search-amplified label and get the full λ-blend.  This lets root and leaf corpora
mix freely in one training run.

---

## Building Corpora

### 1. From PGNs — `scripts/extract_quiet_positions.py`

Replays existing games (python-chess, Chess960-aware, multiprocessed ~2,300
games/s) and emits root positions with their search-score labels from the move
comments.  Quiet filters: in-check, played-move-is-capture/promotion/check,
missing/mate eval, |eval| cap, min-ply, fifty-move clock; duplicate control via
Zobrist hash with a per-position record cap (FRC openings repeat massively).
See `SCRIPT_USE.md` for the option table.

### 2. During play — `TDLEAF_DUMP_TSV` (in-engine)

Any TDLEAF build dumps corpora as a by-product of online play — PGN retention
becomes optional:

```sh
TDLEAF_DUMP_TSV=iter2 python3 match.py Leaf_vtrain_a Leaf_vtrain_b ...
```

writes two per-process files at game end:

- `iter2.<pid>.root.tsv` — played root positions, **search-score labels**
  (white POV), `depth` = achieved ID depth.  Root quietness test:
  |root static − root search| ≤ `TDLEAF_DUMP_QUIET_CP` (default 60 cp) — an
  operational filter (unresolved tactics show up as static-vs-search
  disagreement).
- `iter2.<pid>.leaf.tsv` — PV-leaf positions, static labels, `depth` 0.  Leaves
  are distribution-matched to what the net actually evaluates in search; their
  training signal is the outcome label.

`TDLEAF_DUMP_MAX_CP` (default 1500) caps |eval| for both.  Results are labelled
via the engine's normal game-outcome path (protocol result or UCI
self-adjudication), so only games that feed the TD update are dumped.

---

## The Batch Trainer — `--batch-train`

Built into any `NNUE=1 TDLEAF=1` binary (`src/nnue_batch_train.cpp`).  Loads the
base `.nnue` + `.tdleaf.bin` next to the binary exactly like a normal training
session, trains on the given TSVs, writes per-epoch snapshots, and exits:

```sh
./Leaf_vbt --batch-train corpus_a.tsv,corpus_b.tsv --bt-out myrun \
           [--bt-epochs 3] [--bt-lambda 0.7] [--bt-K 220] [--bt-lr 0.25] \
           [--bt-batch 512] [--bt-val 0.05] [--bt-seed 42] [--bt-max 0] \
           [--bt-sync shared.tdleaf.bin] [--bt-sync-every 256]
```

Design principles:

- **Maximum reuse of the online machinery.**  The per-position update goes through
  `nnue_init_accumulator` → `nnue_forward_fp32` → `nnue_accumulate_gradients` →
  per-batch PSQT mean-centering / gradient clipping / per-section Adam /
  requantize.  The gradient-scale formula mirrors `tdleaf_accumulate_game`, so the
  online LR calibration and evaluation-gauge anchors (PAWN pin, PSQT slot-means)
  carry over unchanged.
- **Target:** `p = λ·result + (1−λ)·sigmoid(cp/K)`, squared error in probability
  space (the same update form as the TD rule).  λ and K are training-time
  hyperparameters — never baked into the data.
- **Packed records** (occupancy bitboard + piece nibbles, ~40 B each): a 200M-
  position corpus fits in ~8 GB RAM.
- **Validation split by game** (hashed `gid`), reported per epoch — but see the
  rate-by-gauntlet finding above.
- Throughput ≈ 31k positions/s single-threaded (≈ 18 min/epoch per 34M positions).

### Per-epoch snapshots and pairing rules

- `<prefix>_epN.nnue` — piece_val **baked into PSQT**; use to compile rating
  binaries (plain `NNUE=1` build).  Not a valid base for further training.
- `<prefix>_epN.tdleaf.bin` — pairs with the **original base `.nnue`** to resume
  or seed further training.  The v10 content-hash check refuses a mispairing at
  load and save (loudly), so the two snapshot types cannot be confused silently.

The base `.nnue` never changes across iterations: every consolidated `.tdleaf.bin`
of every generation pairs with the original base.

### Multi-process data parallelism — `--bt-sync`

Shard the corpus across N processes pointing at one shared file:

```sh
for N in 0..7:  ./Leaf_vbt --batch-train shard_$N.tsv --bt-sync shared.tdleaf.bin ...
```

Every `--bt-sync-every` batches (default 256), each process runs the existing
POSIX-locked delta-merge save — the same multi-writer protocol as concurrent
online training — then requantizes, pulling co-workers' merged updates into its
own inference arrays.  Busy locks defer with deltas retained.  Measured: ~5×
effective speedup at 8 workers (153k positions/s aggregate); merged state clean
throughout.  The final merged net is exported by a zero-LR pass over the sync file
(automated by `hybrid_loop.py`).

---

## The Hybrid Loop — `scripts/hybrid_loop.py`

One command per iteration (see `SCRIPT_USE.md` for the option table, and the
Obsidian `Hybrid_Loop_Runbook` note for the manual procedure it encodes):

```sh
cd engine/learn/
python3 hybrid_loop.py --tag iter2 --games 400000 --depth 8 \
    --state <consolidated>.tdleaf.bin \
    --gauntlet Leaf_vbtsp-final Leaf_v260628-2.4e6g Leaf_vclassic_eval
```

Phases: promote `--state` to the live training state (with backup and a
content-hash pairing pre-flight) → online self-play generation with corpus dumping
(TDLeaf keeps learning throughout) → post-generation checkpoint → shard →
sharded consolidation → merged-net export → rating-binary compile → gauntlet with
an Elo table.

Skippable phases cover the general workflows:
- `--skip-online` + `--corpus ...` — consolidate-only on existing corpora (also how
  LR/sync probe arms rerun on the same dumps without regenerating games).
- `--skip-train` — generate-only (games + corpora, no offline training).

Artifacts (all named by `--tag`): `<tag>_final.nnue` (rating binaries),
`<tag>_final.tdleaf.bin` (seeds the next iteration), `<netbase>.tdleaf.bin-<tag>-online`
(post-generation online checkpoint), `<tag>_work/` (dumps, shards, logs).

### Practical guidance

- **Rate by gauntlet**, 300–400 games vs a fixed panel; never by validation MSE.
- Depth 8 generation recommended (data-quality lever; save-I/O overhead ~10% at d8
  vs ~13% at d6 after the 2026-07-02 buffered-I/O fix).
- Fresh-generation data first; add older corpora only as controlled arms.
- Optimizer probe arms if a generation under-delivers: `--sync-every 128`
  (staleness) before `--bt-lr 0.1` with more epochs (step size).
- Check the drift canaries between checkpoints with
  `diff_tdleaf_checkpoints.py` (per-stack fc2_bias, stack-0 fc2_w[13]/[27],
  FC0 passthrough-row mean, R/Q piece_val).
