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
  conserved); it adds gradient staleness between syncs.  This was benign for the
  large generation-1 backlog signal, but **destroys the subtler generation-2
  signal** — see below.

### Generation 2 (iteration 2, 2026-07-03/04)

The second loop iteration — 400k d8 online games from the consolidated net
(`iter2-online` = +27 over the gen-1 consolidation, −79.5 vs classic measured
directly), then re-consolidation on the 57M-position in-play dump corpus —
initially regressed under the gen-1 settings, and a systematic arm series
resolved why.  Final matrix (all arms trained from the iter2-online state):

| Arm | Sharding | K | λ | Leaf rows | vs gen-1 net | vs classic_eval | Q piece_val drift |
|---|---|---|---|---|---|---|---|
| all 7 initial arms | 8-way | 165–220 | 0.3–0.7 | various | −87 … | −123 … −279 | up to +339 cp |
| iter2s | **none** | 220 | 0.7 | blend | +39 | −90 | +254 cp |
| iter2ks | none | 165 | 0.7 | blend | +12 | −119 | +25 cp |
| iter2ks2 | none | 165 | 0.7 | outcome-only | +9 | −136 | +47 cp |
| **iter2s2** | **none** | **220** | **0.3** | **blend** | **+55 ± 18** | **−64 ± 18** | +89 cp |

Lessons, in order of importance:

1. **Single-process training at the frontier.**  Every 8-way sharded arm regressed;
   the identical unsharded control succeeded.  Sync-merge staleness that the large
   gen-1 backlog signal absorbed destroys the subtle gen-2 signal.  Live tell: the
   validation-MSE *trajectory shape* — smooth when healthy, oscillating under
   staleness (the absolute level still doesn't map to Elo).
2. **λ, not K, is the knob for outcome-driven piece-value inflation.**  Consolidation
   sharpens evals, so gen-2 labels fit a smaller K (165 vs 220); training with
   too-large K makes the outcome term inflate eval magnitudes, which lands in
   `piece_val` (the only free material channel).  But refitting K to 165 removed
   the overshoot *and* the productive material correction hiding inside it
   (iter2s's minors moved toward classical values), costing 30–55 Elo.  Cutting
   λ 0.7 → 0.3 instead tames the drift ~3× while keeping the correction.
3. **Leaf rows need the blend anchor**: outcome-only leaves cost ~46 Elo vs
   leaves trained on the λ-blend with their dump-time static as a magnitude
   anchor.  (Blended leaves are now the trainer default; the run-time knob is
   `--bt-leaf-lambda`, with 1.0 recovering outcome-only.)

**Settled gen-2+ recipe:** `--shards 1 --bt-K 220 --bt-lambda 0.3` (leaf rows
follow λ by default; ~4 h single-process on a 57M corpus).  Consolidation
remains gauntlet-positive per generation: iter2s2 is +55 over the gen-1 net and
+28 over its own online endpoint, cross-family.  Epoch count matters at gen-2+:
iter2s2 peaked at **epoch 4** of 6 in the in-family ladder (gen-1 was still
improving at 6) — select the epoch by a fast ladder (`hybrid_loop.py
--gauntlet-epochs`: 1000 games at 1+0.01 per epoch snapshot, ±19, minutes each)
rather than assuming the last epoch.  Direct classic anchor: ep4 −62 ± 30 vs
ep6 −64 ± 18 — statistically identical cross-family (the +24 in-family edge is
inside the error bars at 400 games), so the ladder pick costs nothing and may
gain.  (Superseded as the iteration-3 seed by the decayed λ-return net below,
which edged it on the direct anchor.)

### The λ sweep and the distance-decayed result weight (2026-07-04)

A single-epoch λ × leaf-λ ladder sweep (1000 games at 1+0.01 each, vs a shared
anchor) showed that **the corpus-mean outcome weight is the knob, not the
root/leaf split**: arms with the same mean `0.43·λ_root + 0.57·λ_leaf` (the
corpus is 43% roots / 57% leaves) were statistically identical, with a plateau
at mean weight ≈ 0.2–0.3 and falloff on both sides.  A flat λ can only set that
mean; TD(λ) says the *per-position* weight should depend on distance from the
game end.  That motivated the decayed target above (`--bt-td-lambda`, default
`TDLEAF_LAMBDA`): near-terminal positions trust the result, early positions
lean on the eval bootstrap.  Under decay the nominal ceilings roughly double
(mean decay 0.502 on the iter2 corpus), so the plateau maps to diagonal
ceilings λ ≈ 0.4–0.65 — swept before iteration 3 (`learn/sweep_td.sh`:
diagonal λ = leaf-λ ∈ {0.3, 0.5, 0.7, 1.0} plus two crossed arms that test
whether the mean-is-the-knob result still holds under decay).

### Sweep results: the pure λ-return is the settled gen-3+ recipe (2026-07-05)

The decay sweep's winner was the **pure λ-return end**: λ = leaf-λ = 1.0, all
moderation supplied by the td_λ = 0.98 distance decay.  Decay *shape* beats the
flat mean — the decayed arm at corpus-mean weight 0.50 won where the flat family
was already declining at 0.44 — and the crossed arms were again inert (root/leaf
split doesn't matter under decay either).  A 3000-game head-to-head vs the
λ = 0.5 diagonal arm was a tie (+3 ± 10), so 1.0 wins on simplicity.  A 6-epoch
confirmation run (`tdL10F10x6`, per-epoch ladder) showed **no multi-epoch
rollover** and **self-limiting piece-value drift**: per-epoch Q increments
+57/+49/+34/+25/+19/+15 cp — geometric convergence (~×0.73/epoch, asymptote
≈ +240 cp) toward a new material equilibrium, not iter2's compounding spiral.
The ladder peaked at **epoch 4** again (second time, after iter2s2), and the
direct classic anchor on ep4 measured **−58.6 ± 20** (1000 games at 3+0.05) —
the best of any net, vs iter2s2-ep4's −62 ± 30.  **`tdL10F10x6_p0_ep4.tdleaf.bin`
seeds iteration 3.**

**Settled gen-3+ recipe:** `--shards 1 --bt-K 220` — `--bt-lambda` now
*defaults to 1.0* (trainer and hybrid_loop), so **`--bt-td-lambda` (default
`TDLEAF_LAMBDA` = 0.98) is the single knob of record**.  The λ ceilings remain
as dormant scale knobs: they decouple overall outcome weight from decay shape
(useful when a corpus's ply-gap distribution differs from iter2's mean decay of
0.502, e.g. different dump depth or PGN-extracted game plies) and keep past
runs reproducible.

Measurement note, learned the hard way: a single 1000-game ladder point can
swing ~±30 Elo (the *bit-identical* ep1 net measured +29 at 1+0.1 and −5 at
1+0.01 vs the same opponent).  Read ladders by shape, replicate before trusting
any single-point peak.

---

## Corpus Format

Tab-separated text, one quiet position per row (produced by both corpus sources
below; `#` comment lines and a `fen`-prefixed header are skipped by consumers):

```
fen  cp  result  ply  depth  gid  [endply]
```

| Column | Meaning |
|---|---|
| `fen` | Shredder-FEN of the position (castling/ep unused by the trainer) |
| `cp` | eval label, **white POV**, centipawns |
| `result` | game outcome, **white POV**: `1` / `0.5` / `0` |
| `ply` | 1-based ply index within the game |
| `depth` | search depth of the eval label; **0 = no search label** (see below) |
| `gid` | stable game id — the trainer splits train/validation **by game** |
| `endply` | *(optional)* the game's true final ply — exact distance base for the result decay.  When absent, the trainer falls back to the per-`gid` max ply seen in the corpus (short by the quiet-filtered game tail, a mild ~uniform over-weighting of the result).  Both corpus producers write it since 2026-07-04. |

**Ply scale caveat:** the in-engine dump counts *recorded engine plies* (one per
own move — the same per-TD-step scale the online `TDLEAF_LAMBDA` trace uses),
while PGN extraction counts *game plies* (every half-move).  Each corpus is
internally consistent, but the result decay runs ~2× faster per game on
PGN-extracted corpora — for those, the scale-matched value is
`--bt-td-lambda ≈ √0.98 ≈ 0.99`.  Avoid mixing the two sources in one decayed
training run.  (The K/λ calibration pipeline — `extract_positions.py` /
`analyze_calibration.py` — also fits λ per *game ply*: its 0.986–0.995 band
equals the engine's 0.98-per-recorded-ply ≈ 0.990-per-game-ply, so online λ
matches the calibration exactly in consistent units.)

**The depth-0 rule:** records with `depth == 0` (leaf-dump rows, whose `cp` is the
net's *own static eval* at dump time, acting as a magnitude anchor) get their own
outcome weight, `--bt-leaf-lambda` — **default: the same λ as roots** (the
recommended setting, worth ~46 Elo over outcome-only leaves in A/B, iteration 2).
Set `--bt-leaf-lambda 1.0` for the old outcome-only behaviour, or any other value
to sweep the leaf blend independently of the root blend.  Records with `depth > 0`
carry a search-amplified label and always use `--bt-lambda`.  This lets root and
leaf corpora mix freely in one training run.

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
           [--bt-epochs 3] [--bt-lambda 0.7] [--bt-leaf-lambda <λ>] \
           [--bt-td-lambda 0.98] [--bt-K 220] [--bt-lr 0.25] \
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
- **Target:** `p = w·result + (1−w)·sigmoid(cp/K)` with the distance-decayed
  result weight `w = λ_eff · td_λ^(N−ply)` — the outcome's credibility decays
  with distance from the game end, exactly as the terminal outcome's weight
  decays in the TD(λ) forward-view λ-return, and the freed weight returns to
  the eval bootstrap so the two weights always sum to 1 (targets stay
  calibrated at any decay).  `λ_eff` is `--bt-lambda` for root rows and
  `--bt-leaf-lambda` for leaf rows; `N` is the game's final ply (`endply`
  column, or per-game corpus max as fallback); `--bt-td-lambda` defaults to
  `TDLEAF_LAMBDA` (0.98, `tdleaf.h`) so offline targets reconstruct the same
  λ-returns the online games were trained on, and `1.0` reproduces the flat
  blend `p = λ·result + (1−λ)·σ` bit-for-bit.  Squared error in probability
  space (the same update form as the TD rule); all of λ/td_λ/K are
  training-time hyperparameters — never baked into the data.  Measured on the
  iter2 corpus: mean decay 0.502 (mean gap ~42 recorded plies), i.e. a nominal
  ceiling λ delivers about half its flat-blend outcome mass — nominal λ values
  under decay run ~2× the flat-blend equivalents.
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

**Frontier caveat (iteration 2):** sync-merge staleness that was benign for the
large generation-1 backlog signal **destroyed the subtler generation-2 signal** —
every 8-way arm regressed while the identical single-process run gained.  Until a
frontier-appropriate configuration is validated (fewer workers, much more frequent
syncs — e.g. 2 workers `--bt-sync-every 64`), use `--shards 1` for gen-2+
consolidation and accept the single-process overnight run.  Oscillating per-epoch
validation MSE is the live symptom of staleness trouble.

---

## The Hybrid Loop — `scripts/hybrid_loop.py`

One command per iteration (see `SCRIPT_USE.md` for the option table, and the
Obsidian `Hybrid_Loop_Runbook` note for the manual procedure it encodes):

```sh
cd engine/learn/
python3 hybrid_loop.py --tag iter3 --games 400000 --depth 8 \
    --state <consolidated>.tdleaf.bin --recompile \
    --shards 1 --bt-K 220 \
    --gauntlet-epochs --gauntlet Leaf_vtdL10F10x6-ep4 Leaf_vclassic_eval
```

(`--shards 1 --bt-K 220` with the default pure λ-return target is the settled
gen-3+ consolidation recipe — see the sweep-results section above.  `--gauntlet-epochs` rates each epoch snapshot
vs the first gauntlet opponent as soon as that epoch finishes training — 1000
games at 1+0.01 by default, `--epoch-games`/`--epoch-tc` to change — and prints
an epoch-ladder table; requires `--shards 1`.  The trainer is paused (SIGSTOP/
SIGCONT) while each ladder match runs, so training never contends with the
games for cores.  The interleaved design leaves a hook for a future
auto-decider that stops a run whose ladder is trending down.)

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

- **Rate by gauntlet**, 300–400 games vs a fixed panel; never by validation-MSE
  *level*.  The MSE trajectory *shape* is still useful live: smooth = healthy
  optimizer, oscillating = staleness/step-size trouble.
- Consolidation recipe (settled 2026-07-05): `--shards 1 --bt-K 220` with the
  default pure λ-return target — `--bt-td-lambda` (0.98) is the knob of record;
  `--bt-lambda`/`--bt-leaf-lambda` default to 1.0 and stay dormant.  Sharding
  is currently for gen-1-scale backlogs only.
- Select the epoch by ladder (`--gauntlet-epochs`), not by assuming the last
  epoch: both gen-2 multi-epoch runs peaked at epoch 4 of 6.  A single
  1000-game ladder point can swing ~±30 — read the shape, replicate peaks.
- Depth 8 generation recommended (data-quality lever; save-I/O overhead ~10% at d8
  vs ~13% at d6 after the 2026-07-02 buffered-I/O fix).
- Fresh-generation data first; add older corpora only as controlled arms.
- Check the drift canaries between checkpoints with
  `diff_tdleaf_checkpoints.py` (per-stack fc2_bias, stack-0 fc2_w[13]/[27],
  FC0 passthrough-row mean, R/Q piece_val).  Some piece_val movement is
  *productive* (material learning) — judge by gauntlet, not by drift alone.
- Measure the promoted net vs the fixed cross-family anchor **directly**:
  Elo chained through a family member reads ~20–30 optimistic (measured twice
  in iteration 2).
