# Leaf Script Reference

All Python scripts live in `scripts/`.  Symlinks in `run/` and `learn/` allow
them to be invoked in-place from those directories, which is the normal workflow
since engines, `.nnue` files, and `.tdleaf.bin` files live there.

---

## match.py

Run a head-to-head match or gauntlet between Leaf executables using
cutechess-cli.  **Invoke from `run/`** (engines and `.nnue` files must be in the
working directory).

```sh
cd run/

# Head-to-head, 200 games
python3 match.py Leaf_vA Leaf_vB -n 200 -c 4 -tc 5+0.05

# Multi-iteration symmetric self-play (both engines learn; restart between
# iterations so each picks up the merged .tdleaf.bin weights)
python3 match.py Leaf_vtrain_a Leaf_vtrain_b -n 500 -i 10 --wait 500

# Gauntlet: probe engine vs multiple opponents; all games appended to one PGN
python3 match.py Leaf_vnew Leaf_v1 Leaf_v2 Leaf_v3 \
    -n 100 --pgn results/gauntlet.pgn
```

### Key options

| Flag | Default | Description |
|------|---------|-------------|
| `-n`, `--games` | 100 | Games per iteration per opponent |
| `-i`, `--iterations` | 1 | Iterations per opponent; engines restart between each |
| `-c`, `--concurrency` | cpu_count/2 | Simultaneous games |
| `-tc`, `--time-control` | `10+0.1` | Time control (`moves/time+inc` or `time+inc`, seconds) |
| `--pgn FILE` | — | Persistent PGN; all games appended across opponents/iterations |
| `--pgn-out FILE` | auto | Per-iteration PGN (default: `match_<e1>_vs_<e2>.pgn`) |
| `--fischer-random` | off | Chess960 starting positions |
| `--wait MS` | 0 | Milliseconds between games (useful when sharing a `.tdleaf.bin`) |
| `--depth1 N` | — | Limit engine1 search to depth N |
| `--depth2 N` | — | Limit engine2 search to depth N |
| `--openings FILE` | — | Openings file: `.epd`, `.pgn`, or `.bin` (polyglot book) |
| `--no-repeat` | off | Play each opening once (`-rounds N`, no `-games 2 -repeat`).  Increases position diversity; recommended for symmetric self-play. |
| `--noswap` | off | Pass `-noswap` to cutechess-cli; engine1 always plays white.  Off by default (correct for training). |

When more than one opponent is supplied the script enters **gauntlet mode** and
prints a summary table (Opponent, Games, W, D, L, Score%, Elo diff) at the end.

### UCI vs xboard cross-protocol match

`match.py` always uses `proto=xboard`.  To run a cross-protocol parity test (UCI
engine vs xboard engine, same binary) call `cutechess-cli` directly:

```sh
cd run/
../tools/cutechess-1.4.0/build/cutechess-cli \
  -engine cmd=./Leaf_vX name=LeafUCI  proto=uci    dir=$(pwd) \
  -engine cmd=./Leaf_vX name=LeafXB   proto=xboard dir=$(pwd) \
  -each tc=10+0.1 \
  -games 2 -rounds 100 -repeat \
  -concurrency 4 \
  -openings file=../testing/testsuites/wac.epd format=epd order=random \
  -pgnout /tmp/uci_vs_xboard.pgn \
  -resign movecount=5 score=800 \
  -draw movenumber=40 movecount=8 score=20
```

Expected result: ~50% score, Elo difference within ±50 (same engine, different wire
protocol).

---

## make_training_epd.py

Generate a combined opening EPD file for TDLeaf training:
- All 960 Chess960 starting positions (FRC), with optional random suffix moves
- ~N positions sampled from a Polyglot opening book at a given ply depth,
  with optional random suffix moves

The output is shuffled and ready for use with
`-openings file=training_openings.epd format=epd order=random -variant fischerandom -noswap`.
`training_run.py` auto-detects and uses this file when it is present in `learn/`.

**Why random suffix moves?**  normbk02.bin at ply 8 converges to only ~2500 unique
positions due to transpositions.  Adding 1–2 random moves after each book/FRC leaf
explodes the unique count into the hundreds of thousands, preventing game replication
in training.  Use `--quiet-only` to restrict suffix moves to non-captures (keeps
material balanced; recommended).

| Suffix | Book unique | FRC unique (per replicate) |
|--------|-------------|---------------------------|
| 0 | ~2,500 | 960 |
| 1 | ~60,000 | ~19,000 |
| 2 | ~1,000,000+ | ~300,000+ |

Two sizing modes are available (mutually exclusive):

**Explicit** (`--frc-replicates` / `--book-positions`):
```sh
cd learn/

# Default: 960 FRC + 2000 book, no suffix (2,960 total)
python3 make_training_epd.py

# 2 quiet suffix moves, 21 FRC replicates + 80k book positions
python3 make_training_epd.py --frc-replicates 21 --book-positions 80000 \
    --random-suffix 2 --quiet-only
```

**Fraction-based** (`--total` / `--frc-fraction`):
```sh
cd learn/

# 100k total, ~20% FRC-derived, 2 quiet suffix moves
# → frc_replicates=21, 20160 FRC-derived + 79840 book = 100000
python3 make_training_epd.py --total 100000 --frc-fraction 0.20 \
    --random-suffix 2 --quiet-only
```

Fraction-based sizing: `frc_replicates = max(1, round(total × frc_fraction / 960))`;
`book_positions = total − 960 × frc_replicates`.  Actual totals may differ by up to
960 from `--total` because replicates must be an integer.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--book FILE` | `normbk02.bin` in script dir | Polyglot `.bin` book to sample from |
| `--book-positions N` | 2000 | Number of book positions to sample (explicit mode) |
| `--frc-replicates K` | 1 | Samples per FRC position (explicit mode).  Without `--random-suffix`: K identical copies.  With `--random-suffix`: K unique suffix-varied samples per FRC position. |
| `--total N` | — | Target total output size (fraction mode; use with `--frc-fraction`) |
| `--frc-fraction F` | — | Desired FRC fraction 0.0–1.0 (fraction mode; use with `--total`) |
| `--random-suffix K` | 0 | Random moves to play after each book/FRC position; greatly increases unique position count |
| `--quiet-only` | off | Restrict random suffix moves to non-captures **and** filter positions by eval balance (see below) |
| `--eval-binary FILE` | `Leaf_vclassic_eval` in script dir | Leaf binary for eval filtering; only used with `--quiet-only` |
| `--eval-limit CP` | 50 | Discard positions where `\|score\| > CP` centipawns (default: 50 = 0.5 pawns); only used with `--quiet-only` |
| `--eval-depth N` | 10 | Search depth for eval filtering; only used with `--quiet-only` |
| `--eval-workers N` | cpu_count/2 | Parallel eval engine processes; only used with `--quiet-only` |
| `--ply N` | 8 | Ply depth for book random walks |
| `--output FILE` | `training_openings.epd` in script dir | Output EPD file |
| `--seed N` | 42 | Random seed for reproducibility |

Book positions are selected by weighted random walks (move probability ∝ Polyglot
`weight` field), then deduplicated.  With `--random-suffix`, each walk's leaf gets
additional random (or quiet) moves before deduplication, multiplying the unique count.
FRC replication without suffix preserves intentional duplicates (for position weighting);
with suffix, duplicates across replicates are silently dropped (rare).

**`--quiet-only` eval filter:** when a `Leaf_vclassic_eval` binary is present, every
generated position is scored at `--eval-depth` via xboard protocol and positions with
`|score| > --eval-limit` cp are discarded.  Chess960 castling rights are stripped from
the EPD before sending to the engine.  `--eval-workers` parallel engine processes run
concurrently (~75 pos/sec at depth 10 with 4 workers).  Compile the eval binary with
`perl src/comp.pl classic_eval OVERWRITE` (no NNUE, no TDLEAF).  If the binary is
absent, a warning is printed and the filter is skipped.

---

## training_run.py

Interactive TDLeaf(λ) training run manager.  **Invoke from `learn/`** so that
all working files (`.nnue`, `.tdleaf.bin`, `.games`, built binaries, PGN output)
land in `learn/`.

> **TDLeaf requires xboard protocol.**  Learning hooks are called from inside
> `make_move()`, which is only reached in the xboard game loop.  Matches driven
> through a UCI GUI will not update any weights even if the binary was compiled
> with `TDLEAF=1`.  `training_run.py` and `match.py` both use `proto=xboard` and
> are the correct tools for all training workflows.

```sh
cd learn/
python3 training_run.py
```

### Prompt sequence

1. **Starting network** — existing `.nnue` file or a freshly random-initialised one
   (classical material prior or uniform 100cp via `--init-nnue-noprior`)
2. **Opponent roster** — build a rotation of one or more opponent types:
   - `[s]` Self-play — both `_a` and `_b` instances learn (symmetric)
   - `[p]` Previous checkpoint — learner vs. its own recent snapshot (read-only)
   - `[e]` External engine — learner vs. user-supplied executable

   When the roster has multiple entries (or includes `prev-checkpoint`), the user
   sets a **rotation interval** — games are split into segments of that many games,
   cycling through the roster.  A `.nnue` checkpoint is exported at every rotation
   boundary.  The `prev-checkpoint` opponent always loads the most recently exported
   checkpoint (or the base net for the first segment).
3. **Train-validate loop** — optional; see below
4. **Build** — compiles only the binaries the roster requires:
   - Learner (`_a`, TDLEAF=1) — always built
   - Self-play partner (`_b`, TDLEAF=1) — built if roster includes self-play
   - Checkpoint opponent (`_ro`, NNUE-only, no TDLEAF) — built with a separate
     `NNUE_NET` so its `.nnue` can be swapped at rotation boundaries
5. **Continuity** — continue from existing `.tdleaf.bin` or start fresh
6. **Match parameters** — TC, concurrency, wait, opening selection, per-engine
   depth limits; per-engine TCs (`--tc1` / `--tc2`) when the opponent runs at a
   different speed.  Opening selection priority:
   - If `learn/training_openings.epd` exists: use it with Fischer Random variant
     (no question asked — EPD file encodes the intent).
   - Else if Fischer Random is chosen: use random Chess960 positions.
   - Else if `normbk02.bin` is in `learn/`: use it as the Polyglot opening book.
   See `make_training_epd.py` to generate `training_openings.epd`.

On completion, trained weights are exported to `<net_base>-<total_games>g.nnue` and
a copy of the current `.tdleaf.bin` is saved as `<net_base>.tdleaf.bin-<total_games>g`
for archival and rollback.  If terminated early with Ctrl-C, the export uses a
`-partial` suffix to avoid overwriting an existing game-count checkpoint.
The `<net_base>.games` sidecar is always written to `current_games` at the start of
Step 6 (before any filename is determined), so sidecar counts and file names stay
in sync on all exit paths.  Game counts accumulate across runs.

**Startup backup:** when continuing from an existing `.tdleaf.bin`, a copy is saved
as `.tdleaf.bin.bak` before any training begins.  This allows recovery of the
pre-run weights if a training session produces bad results or is interrupted at an
inopportune moment.

**Adam momentum persistence (.tdleaf.bin v7):** the Adam first-moment (m) arrays are
now persisted across sessions alongside the second-moment (v) arrays introduced in v6.
Previously m cold-started at zero each run while v was restored, causing the optimizer
to rediscover gradient directions for the first few thousand games — visible as a slow
or negative Elo trend at the start of every run.  With v7, the optimizer resumes with
full directional momentum.  Concurrent-writer merge uses element-wise average
`(m_file + m_local) / 2`.  Storage overhead: ~1.5 MB.  Existing v6 files upgrade
automatically on the first save.

**Self-play opening diversity:** when the current opponent segment is symmetric
self-play (both engines learn), `training_run.py` automatically passes `--no-repeat`
to `match.py` so each opening is played once rather than twice, maximising the variety
of positions seen per N games.  Non-self-play segments (prev-checkpoint, external
engine) retain `-games 2 -repeat` for fairer W/L/D statistics.

### Train-validate loop

When enabled, the script runs repeated train → validate cycles instead of a single
match block:

```
setup (once before first cycle):
  export current .tdleaf.bin → <net>-best.nnue
  (or copy base .nnue if no .tdleaf.bin exists yet)

repeat N cycles (0 = forever until Ctrl-C):
  1. Checkpoint current .tdleaf.bin
  2. Train for X games (single iteration — Adam state preserved throughout)
  3. Export new weights → <net>-cand.nnue
  4. Run Y-game validation match: eval_cand vs eval_best
  5. Accept if LOS ≥ threshold →
       bank games, export accepted weights → best.nnue
       save snapshot → <net>-<total_games>g.nnue  (for tournament use)
     Reject → revert .tdleaf.bin to pre-cycle checkpoint
```

**Snapshots:** Each accepted cycle saves a game-count-stamped `.nnue` file (e.g.
`nn-training1a-5000g.nnue`, `nn-training1a-10000g.nnue`).  These can later be
entered in a tournament via `bayeselo_ratings.py` to chart Elo progression over
training.

Loop-mode prompts (Step 3):

| Prompt | Default | Notes |
|--------|---------|-------|
| Cycles | 0 (∞) | Number of train-validate cycles; 0 = run until Ctrl-C |
| Validation games | 200 | Games per validation match |
| LOS acceptance threshold | 70% | Candidate accepted if LOS ≥ this |
| Early-stop high | 90% | Terminate validation early if LOS ≥ this (clear win) |
| Early-stop low  | 10% | Terminate validation early if LOS ≤ this (clear loss) |
| Validation TC | (same as training TC) | Separate TC for the validation match |
| Games per cycle | 5000 | Training games per cycle (single iteration; no engine restart) |

Two eval-only binaries (`NNUE=1`, no `TDLEAF`) are compiled once at setup:
`eval_best` loads `<net>-best.nnue` and `eval_cand` loads `<net>-cand.nnue`.

**Why single iteration in loop mode**: multiple iterations restart the engine
processes between blocks, discarding Adam momentum and variance state.  A single
long iteration preserves the full optimiser state across the training block.

Ctrl-C exits cleanly: current weights are exported and a per-cycle result table
is printed.

---

## compare_nnue_learning.py

Visualise NNUE weight changes after TDLeaf training.  **Invoke from `learn/`**
(where the `.nnue` baseline and `.tdleaf.bin` live).

```sh
cd learn/
python3 compare_nnue_learning.py nn-fresh-260309.nnue nn-fresh-260309.tdleaf.bin
```

Produces a four-page matplotlib figure:

- **Page 1 — FC weights**: FC0/FC1/FC2 weight distributions (baseline vs learned),
  per-output delta histograms, per-stack % changed + max |Δ|
- **Page 2 — FC biases**: FC0/FC1/FC2 bias distributions (baseline vs learned, int32),
  delta histograms, per-stack scatter of individual Δ values (every bias visible so
  no outlier can hide in an aggregate)
- **Page 3 — Feature transformer**: FT bias distributions (baseline vs learned, v4
  `.tdleaf.bin` only), FT weight distributions, delta and update counts
- **Page 4 — PSQT**: baseline vs learned distributions, delta histogram,
  per-bucket mean delta bar chart ±1σ

Optional flags:

```sh
# Save pages to PNG files instead of displaying
python3 compare_nnue_learning.py baseline.nnue weights.tdleaf.bin --save out_prefix

# Include full FT weight arrays (slow; requires ~92 MB of memory per perspective)
python3 compare_nnue_learning.py baseline.nnue weights.tdleaf.bin --ft-weights
```

---

## verify_fc0.py

Debugging tool: recomputes FC0 outputs from a manually specified `l0_in` vector
and compares against raw weights read directly from the `.nnue` file.  Used to
verify the Leaf forward pass against the reference network.

Edit the `L0_IN` dict near the top of the script with values captured from
`NNUE_DEBUG_VERBOSE=1` output, then run from `run/`:

```sh
cd run/
python3 verify_fc0.py
```

---

## launch_epoch.py

Registers an Leaf binary in cutechess's `engines.json` and launches the
cutechess GUI.  **Invoke from `run/`**.

```sh
cd run/
python3 launch_epoch.py Leaf_v2026_03_09a
```

If no executable name is given, defaults to `Leaf`.

---

## run_epoch.py

Wrapper that launches an Leaf binary with `run/` as its working directory,
ensuring it finds `search.par`, `main_bk.dat`, and the `.nnue` network file.
Useful as the engine command inside cutechess or other GUIs.

```sh
# As engine command in cutechess:
python3 /path/to/run/run_epoch.py Leaf_v2026_03_09a
```

---

## bayeselo_ratings.py

Compute a Bayesian Elo rating list for all players in a PGN file, using the
`tools/BayesElo/bayeselo` binary.  Can be invoked from anywhere.

```sh
# Basic usage
python3 scripts/bayeselo_ratings.py learn/pgn/fresh-260309-testing.pgn

# Exclude players with fewer than 50 games
python3 scripts/bayeselo_ratings.py results.pgn --min 50

# Also optimise first-move advantage and draw-Elo
python3 scripts/bayeselo_ratings.py results.pgn --advantage --drawelo

# Use a non-default bayeselo binary
python3 scripts/bayeselo_ratings.py results.pgn --bayeselo /usr/local/bin/bayeselo
```

Example output:

```
Bayesian Elo ratings — fresh-260309-testing.pgn
5612 games loaded, 7 players rated

Rank  Name                            Elo     ±  Games   Score   Oppo  Draws
----------------------------------------------------------------------------
   1  EXchess_classic             +1031   213    500    100%    +66     0%
   2  Leaf_vnn-fresh-260309-4000g    +66    17   2612     68%    -11    10%
   3  EXchess_classic_material         -8    56    112     40%    +66    24%
   4  Leaf_vnn-fresh-260309-2000g    -79    15   2000     68%   -236    18%
   5  Leaf_vnn-fresh-260309-1000g   -219    15   2000     48%   -201    20%
   6  Leaf_vnn-fresh-260309-500g    -324    15   2000     33%   -175    19%
   7  Leaf_vnn-fresh-260309         -467    18   2000     16%   -139    12%
```

Ratings are relative (zero-sum); absolute scale depends on which players are
included.  The `±` column is the larger of the two asymmetric confidence
intervals returned by BayesElo.  `Oppo` is the average Elo of opponents faced.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `pgn` | *(required)* | PGN file(s) to analyse (combined before analysis) |
| `--bayeselo PATH` | `tools/BayesElo/bayeselo` | Path to the bayeselo binary |
| `--min N` | 0 | Exclude players with fewer than N games |
| `--advantage` | off | Optimise first-move advantage alongside ratings |
| `--drawelo` | off | Optimise draw-Elo alongside ratings |

Multiple PGN files can be supplied; BayesElo reads them sequentially and combines all games:

```sh
python3 scripts/bayeselo_ratings.py learn/pgn/run1.pgn learn/pgn/run2.pgn --min 20
```

---

## pgn_dedup.py

Remove duplicate games from one or more PGN files.  Two games are considered
identical when their move sequences match after stripping move numbers, comments,
NAG annotations, and result tokens.

```sh
# Deduplicate a single file
python3 scripts/pgn_dedup.py input.pgn --output deduped.pgn --report

# Combine and deduplicate multiple files
python3 scripts/pgn_dedup.py run1.pgn run2.pgn run3.pgn --output combined.pgn --report
```

Example `--report` output (to stderr):

```
  duplicate game #312 (Leaf_vA vs Leaf_vB) in run2.pgn — matches game #87 in run1.pgn

pgn_dedup: 4500 games read, 4487 written, 13 duplicates removed.
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `pgn` | *(required)* | PGN file(s) to process |
| `--output FILE` | stdout | Write deduplicated games to FILE |
| `--report` | off | Print per-duplicate details and a summary to stderr |
| `--players` | off | Include White and Black headers in the identity key |

By default only the move sequence is used for comparison, so games that differ
only in headers (date, round, event) are treated as duplicates.  With `--players`,
two games must also have matching `White` and `Black` tags to be considered duplicates
— useful when the same position was played by different engine pairs.

---

## merge_tdleaf.py

Merge multiple `.tdleaf.bin` files with count-weighted averaging.  Each weight
in the output is the weighted average of the corresponding weights across all
input files, where the per-element weight is its update count (`cnt`).

The `-o` argument is a filename base: produces `<base>.tdleaf.bin` always,
and `<base>.nnue` when `--baseline` is given.

```sh
# Merge two training runs (.tdleaf.bin only)
python3 scripts/merge_tdleaf.py run1.tdleaf.bin run2.tdleaf.bin -o merged

# Also produce a merged .nnue file from a baseline network
python3 scripts/merge_tdleaf.py run1.tdleaf.bin run2.tdleaf.bin \
    -o merged --baseline nn-start.nnue

# Merge with summary statistics
python3 scripts/merge_tdleaf.py a.tdleaf.bin b.tdleaf.bin c.tdleaf.bin \
    -o merged --report
```

### Key options

| Flag | Default | Description |
|------|---------|-------------|
| `files` (positional) | *(required)* | Two or more `.tdleaf.bin` input files |
| `-o`, `--output` | *(required)* | Output filename base (produces `<base>.tdleaf.bin` and optionally `<base>.nnue`) |
| `--baseline` | *(none)* | Baseline `.nnue` file; when given, also writes `<output>.nnue` with merged weights applied |
| `--report` | off | Print per-file and merged update-count statistics |

### Merge algorithm

For each weight element `i` across N input files:

```
if sum(cnt[i]) > 0:
    merged[i] = sum(value[i] * cnt[i]) / sum(cnt[i])
else:
    merged[i] = value[i] from first file
merged_cnt[i] = sum(cnt[i])
```

FC layers, FT weights, PSQT weights, and FT biases are all merged with this
scheme.  Sparse FT/PSQT rows are unioned: a feature row present in any input
file appears in the output.  The output is always v6 format; input files v2–v6
are all accepted.  v6 files include persistent Adam second-moment (v) arrays,
which are max-merged across inputs.

### Use cases

- **Combining independent training runs** that started from the same baseline
  `.nnue` file but diverged.  The count-weighting ensures each run's
  contribution is proportional to how much training it performed.
- **Averaging checkpoints** from different stages of a single training run
  (e.g., merging the 1000-game and 2000-game snapshots).

---

## tdleaf_selfplay.py

*(Not in active use.)*  Earlier self-play driver for TDLeaf training, predating
`training_run.py`.  Kept in `scripts/` for reference; no symlinks are provided
in `run/` or `learn/`.
