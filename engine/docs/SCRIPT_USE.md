# Leaf Script Reference

All Python scripts live in `scripts/`.  Symlinks in `run/` and `learn/` allow
them to be invoked in-place from those directories, which is the normal workflow
since engines, `.nnue` files, and `.tdleaf.bin` files live there.

---

## match.py

Run a head-to-head match or gauntlet between chess engines using cutechess-cli.
Supports Leaf binaries and external UCI engines (e.g. Stockfish, placed in
`tools/engines/<name>/`).  **Invoke from `run/`** (symlink) or `scripts/`.

### Interactive mode

Run with no arguments for a fully interactive session — the script discovers
available engines from `engine/run/` (Leaf binaries) and `tools/engines/`
(external engines), presents numbered menus, and prompts for all match options:

```sh
cd run/
python3 match.py                    # fully interactive
python3 match.py Leaf_vA            # interactive for opponent and options
```

### CLI mode

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

# External engine match (engine binary in tools/engines/stockfish/)
python3 match.py Leaf_vA stockfish -n 100 -tc 10+0.1
```

### Engine discovery

When running interactively (or to resolve bare engine names), the script scans:

- **`engine/run/Leaf_v*`** — Leaf binaries (executables matching the `Leaf_v` prefix)
- **`tools/engines/<name>/`** — external engines; within each subdirectory the
  script picks the executable whose filename best matches the directory name (or
  the largest executable if none match)

Each engine's **working directory** is automatically set to the directory
containing its binary, so engines can find their data files (books, NNUE nets,
etc.) without manual `dir=` configuration.

### Key options

| Flag | Default | Description |
|------|---------|-------------|
| `-n`, `--games` | 100 | Games per iteration per opponent |
| `-i`, `--iterations` | 1 | Iterations per opponent; engines restart between each |
| `-c`, `--concurrency` | cpu_count/2 | Simultaneous games |
| `-tc`, `--time-control` | `10+0.1` | Time control (`moves/time+inc` or `time+inc`, seconds) |
| `--proto` | `uci` | Protocol for both engines (`uci` or `xboard`) |
| `--proto1` | (from `--proto`) | Override protocol for engine1 only |
| `--proto2` | (from `--proto`) | Override protocol for engine2 only |
| `--pgn FILE` | — | Persistent PGN; all games appended across opponents/iterations |
| `--pgn-out FILE` | auto | Per-iteration PGN (default: `match_<e1>_vs_<e2>.pgn`) |
| `--fischer-random` | off | Chess960 starting positions |
| `--wait MS` | 0 | Milliseconds between games (useful when sharing a `.tdleaf.bin`) |
| `--depth1 N` | — | Limit engine1 search to depth N |
| `--depth2 N` | — | Limit engine2 search to depth N |
| `--openings FILE` | — | Openings file: `.epd`, `.pgn`, or `.bin` (polyglot book) |
| `--no-repeat` | off | Play each opening once (`-rounds N`, no `-games 2 -repeat`).  Increases position diversity; recommended for symmetric self-play. |
| `--noswap` | off | Pass `-noswap` to cutechess-cli; engine1 always plays white.  Off by default (correct for training). |
| `--error-log FILE` | — | Append cutechess-cli stderr (and inherited engine stderr) to FILE.  Captures per-batch `[tdleaf step-clip]` telemetry from all concurrent training engines in one file; lines are atomic per `fprintf` (<4KB), so interleaving is line-granular. |

When more than one opponent is supplied the script enters **gauntlet mode** and
prints a summary table (Opponent, Games, W, D, L, Score%, Elo diff) at the end.

### Protocol notes

The default protocol is **UCI**.  Leaf auto-detects UCI, so no special flags are
needed.  Use `--proto xboard` (or `--proto1`/`--proto2` for per-engine overrides)
when running xboard-only engines or TDLeaf training (which requires the xboard
game loop).

For a cross-protocol parity test (same Leaf binary, UCI vs xboard):

```sh
python3 match.py Leaf_vX Leaf_vX --proto1 uci --proto2 xboard -n 200
```

Expected result: ~50% score, Elo difference within ±50 (same engine, different
wire protocol).

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
> with `TDLEAF=1`.  `training_run.py` automatically passes `--proto1 xboard`
> for the learner and selects the opponent protocol per fixed engine (xboard
> for Leaf binaries, the engine's own protocol for external executables).

```sh
cd learn/
python3 training_run.py
```

### Prompt sequence

1. **Starting network** — existing `.nnue` file or a freshly random-initialised one
   (classical material prior or uniform 100cp via `--init-nnue-noprior`)
2. **Opponent roster** — build a rotation of one or more opponent types:
   - `[s]` Self-play — both `_a` and `_b` instances learn (symmetric)
   - `[r]` Read-only mirror — learner vs. a TDLEAF_READONLY copy of itself,
     frozen at the start of each rotation segment
   - `[f]` Fixed engine — any Leaf binary or external executable; presents a
     numbered list of engines discovered under `tools/engines/`, plus the
     option to enter a custom path

   When the roster has multiple entries (or includes a read-only mirror), the
   user sets a **rotation interval** — games are split into segments of that
   many games, cycling through the roster.  A `.nnue` checkpoint is exported at
   every rotation boundary.  The read-only mirror loads the most recently
   exported checkpoint (or the base net for the first segment).
3. **Train-validate loop** — optional; see below
4. **Build** — compiles only the binaries the roster requires:
   - Learner (`_a`, TDLEAF=1) — always built
   - Self-play partner (`_b`, TDLEAF=1) — built if roster includes self-play
   - Read-only mirror (`_ro`, TDLEAF_READONLY=1) — built if roster includes
     a read-only mirror; loads weights but skips updates
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
of positions seen per N games.  Non-self-play segments (read-only mirror, fixed
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

Also prints a console summary of weight changes, then produces a four-page matplotlib figure:

**Console output:**
- FC layer change table (% changed, Δ range, mean ± std), per-stack FC1 breakdown,
  update count summary
- FT / PSQT statistics (rows trained, weight range, update counts, per-bucket Δ table)
- Dense piece values table (centipawns per piece type, update counts)
- Adam optimizer state (t_adam, v/m/FT-v loaded flags)

**Matplotlib pages:**
- **Page 1 — FC weights**: FC0/FC1/FC2 weight distributions (baseline vs learned),
  per-output delta histograms, per-stack % changed + max |Δ|
- **Page 2 — FC biases**: FC0/FC1/FC2 bias distributions (baseline vs learned, int32),
  delta histograms, per-stack scatter of individual Δ values (every bias visible so
  no outlier can hide in an aggregate)
- **Page 3 — Feature transformer**: FT bias distributions (baseline vs learned, v4+
  `.tdleaf.bin` only), FT weight distributions, delta and update counts
- **Page 4 — PSQT**: baseline vs learned distributions, delta histogram,
  per-bucket mean delta bar chart ±1σ

Supports `.tdleaf.bin` versions 2–9.  V9 stores `piece_val[6]` (one value per piece
type); V5–V8 stored `piece_val[6][8]` (per bucket), which is averaged on read.

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

## pgn_winrate.py

Analyse win/draw/loss rates per N-game window for one player in a PGN file.
Auto-detects the non-baseline player (anything that is not `*material_eval*`).
Useful for spotting training collapses, peak performance windows, and whether
the engine recovers after a crash.

### Usage

```sh
# Auto-detect player, 100-game windows (default)
python3 scripts/pgn_winrate.py learn/pgn/run1/match_run1_0g.pgn

# Explicit player and window size
python3 scripts/pgn_winrate.py learn/pgn/run1/match_run1_0g.pgn \
    --player Leaf_vtrain_nn-fresh_a --window 200

# CSV output (for plotting / further processing)
python3 scripts/pgn_winrate.py learn/pgn/run1/match_run1_0g.pgn --csv
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `pgn_file` (positional) | *(required)* | PGN file to analyse |
| `--player <name>` | auto-detect | Player name to track |
| `--opponent <name>` | *(none)* | Opponent name; used to auto-detect the other player |
| `--window <N>` | 100 | Number of games per analysis window |
| `--csv` | off | Emit CSV instead of a formatted table |

### Output

Formatted table with columns W / D / L / Win% / Draw% / Loss% / Score% per
window, followed by totals and a summary that reports:

- Starting win rate (first window)
- Peak win rate and which window it occurred in
- First window where win rate drops below 5%
- Whether/when the win rate recovers above 5% after a crash

---

## pgn_elo_progress.py

Track Elo progress across training by splitting a PGN into fixed-size windows
and running bayeselo on each window.  Useful for plotting strength over time
during a long training run.

```sh
# Default: 10,000-game windows, auto-detect training engine
python3 scripts/pgn_elo_progress.py learn/pgn/run1/combined.pgn

# Custom window size and explicit player
python3 scripts/pgn_elo_progress.py learn/pgn/run1/combined.pgn --window 5000 --player Leaf_vtrain_a

# Cumulative (each window includes all prior games)
python3 scripts/pgn_elo_progress.py learn/pgn/run1/combined.pgn --cumulative
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `pgn_file` (positional) | *(required)* | PGN file to analyse |
| `--window N` | 10000 | Games per window |
| `--player NAME` | auto-detect | Player name to track |
| `--bayeselo PATH` | `tools/BayesElo/bayeselo` | Path to bayeselo binary |
| `--cumulative` | off | Use cumulative windows instead of sliding |

---

## reset_adam.py

Zero (or decay) the Adam optimizer state in a `.tdleaf.bin` file.  All weight
data (FC, FT, PSQT, piece_val) is preserved exactly — only the first-moment (m)
and second-moment (v) arrays are modified.  Useful when training has plateaued
due to accumulated v values damping all updates.

```sh
# Full zero reset (backup kept as .bak)
python3 scripts/reset_adam.py learn/nn-fresh.tdleaf.bin

# Soft reset — keep 10% of accumulated v/m
python3 scripts/reset_adam.py learn/nn-fresh.tdleaf.bin --decay 0.1

# Write to a new file, leave original untouched
python3 scripts/reset_adam.py learn/nn-fresh.tdleaf.bin --out learn/nn-fresh-reset.tdleaf.bin
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `file` (positional) | *(required)* | `.tdleaf.bin` file to modify |
| `--decay F` | 0 (full zero) | Multiply v and m by F instead of zeroing (0 < F < 1) |
| `--out PATH` | *(overwrite input)* | Write result to PATH instead of overwriting |

---

## extract_positions.py

Stream Leaf self-play PGN files and emit a per-position parquet dataset for
calibration analysis.  Scores are read from inline move comments (`{+1.23/6 0.01s}`),
converted to centipawns from the White perspective, and written alongside the game
outcome and ply metadata.  Mate scores are capped at ±2000 cp.

The output parquet is the input for `analyze_calibration.py`.

```sh
# Default: sample ~200K games from learn/pgn/nn-fresh-260410/
python3 scripts/extract_positions.py

# Explicit paths and options
python3 scripts/extract_positions.py \
    --pgn-dir  learn/pgn/nn-fresh-260410 \
    --out      learn/positions.parquet \
    --max-games 200000 \
    --min-plies 8 \
    --seed 42
```

### Output columns

| Column | Type | Description |
|--------|------|-------------|
| `game_id` | int32 | Sequential game counter across all files |
| `training_stage` | int8 | 0=untrained … 6=1.5M-game-trained (inferred from filename) |
| `ply` | int16 | 0-based ply index (0 = White's first move) |
| `white_score_cp` | int16 | Eval from White's POV, centipawns, capped ±2000 |
| `result` | float32 | Game result from White's POV: 1.0 / 0.5 / 0.0 |
| `n_plies` | int16 | Total plies in the game |

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--pgn-dir PATH` | `learn/pgn/nn-fresh-260410` | Directory containing `.pgn` files |
| `--out PATH` | `learn/positions.parquet` | Output parquet file |
| `--max-games N` | 200000 | Approximate number of games to sample (0 = all ~1.6M) |
| `--min-plies N` | 8 | Skip games shorter than this (matches `TDLEAF_MIN_PLIES`) |
| `--seed N` | 42 | Random seed for game sampling |

---

## analyze_calibration.py

Calibrate TDLeaf hyperparameters from the per-position parquet produced by
`extract_positions.py`.  Implements two analyses:

**Goal 1A — Sigmoid temperature K:** MLE search for the K that maximises
log-likelihood of game outcomes under `P(White wins) = σ(score / K)`.  Produces
an NLL-vs-K curve, a reliability diagram, and a sigmoid comparison plot.

**Goal 2A — Lambda decay:** Autocorrelation of `d_t = σ(score / K)` vs lag,
split by decisive/draw games.  Even-lag pairs (same side-to-move) remove the
ply-alternation oscillation.  Also computes `corr(d_t, result)` vs distance to
game end.  Fits `λ^k` to both curves.

```sh
# Default: stages 5–6, max-lag 60, output to learn/calibration_plots/
python3 scripts/analyze_calibration.py

# Explicit options
python3 scripts/analyze_calibration.py \
    --input   learn/positions.parquet \
    --out-dir learn/calibration_plots \
    --stage 5 6 \
    --max-lag 60

# Include all training stages (shows how K and λ evolve over training)
python3 scripts/analyze_calibration.py --all-stages
```

### Output files

| File | Contents |
|------|----------|
| `calibration_K.png` | NLL vs K, reliability diagram, sigmoid comparison |
| `lambda_decay.png` | 4-panel: full/even-lag autocorr, full/even-parity d_t-vs-result |
| `summary.txt` | K_opt, delta from current, Brier scores, fitted λ values (both methods) |

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--input PATH` | `learn/positions.parquet` | Parquet file from `extract_positions.py` |
| `--out-dir PATH` | `learn/calibration_plots` | Output directory for plots and summary |
| `--stage N [N …]` | `5 6` | Training stage(s) to include in analysis |
| `--all-stages` | off | Include all stages (overrides `--stage`) |
| `--max-lag N` | 60 | Maximum lag for autocorrelation plots |

---

## extract_quiet_positions.py

Build an offline-training position set from existing PGNs (see
`OFFLINE_TRAINING.md`).  Replays each game (python-chess, Chess960-aware,
multiprocessed, ~2,300 games/s) and emits one TSV record per QUIET position:
`fen  cp  result  ply  depth  gid  endply` — Shredder-FEN, search eval from the
move comment (white POV, cp), game result (white POV), ply, eval depth, a stable
game id so the trainer can split train/validation by game, and the game's true
final ply (distance base for the trainer's result decay).  Eval and outcome are
stored separately: the trainer's decayed λ-blend keeps λ, td_λ, and K as
training-time hyperparameters.  Note: `ply`/`endply` here count game plies
(every half-move), vs recorded engine plies in the in-engine dump — see the ply
scale caveat in `OFFLINE_TRAINING.md`.

Quiet filters: side-to-move in check, played move is a capture/promotion/check,
missing or mate eval, |eval| cap, min-ply, fifty-move clock.  Duplicate control
via polyglot Zobrist hash with a per-position record cap (FRC book openings
repeat massively).  Requires the `python-chess` package.

```sh
# One file (d8 self-play), both sides are the learner
python3 extract_quiet_positions.py \
    --pgn-file pgn/nn-fresh-260628/match_nn-fresh-260628_2e6g.pgn \
    --out quiet_d8.tsv

# A directory of PGNs, keeping only positions where a named engine moved
python3 extract_quiet_positions.py --pgn-dir rotation_segs \
    --player Leaf_vtrain_nn-fresh-260628_a --out quiet_rotation.tsv

# Random game sample for a fast pilot
python3 extract_quiet_positions.py --pgn-file big.pgn --max-games 50000 \
    --out quiet_sample.tsv
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--pgn-file PATH` | — | PGN file to extract from (repeatable) |
| `--pgn-dir PATH` | — | Directory of `.pgn` files (all processed, sorted) |
| `--out PATH` | required | Output TSV (`.gz` suffix → gzip) |
| `--player NAME` | all | Keep only positions where this engine (header substring) is to move |
| `--max-games N` | 0 = all | Random game sample across the whole input |
| `--min-ply N` | 8 | Skip the first N plies of each game |
| `--max-eval CP` | 1500 | Skip positions with \|eval\| above this |
| `--max-fifty N` | 80 | Skip positions with halfmove clock ≥ N |
| `--max-dups N` | 4 | Max records per unique position (0 = off; dedup table is in RAM) |
| `--workers N` | cores−2 | Parallel parser processes |
| `--seed N` | 42 | Sampling seed |

---

## diff_tdleaf_checkpoints.py

Diff two `.tdleaf.bin` checkpoints section by section: piece values (raw and
cp-equivalent), per-section FC weight/bias movement (median/mean/max |dw|),
FT bias, and FT/PSQT rows matched by feature index.  The standard monitor for
the outcome-imbalance drift canaries (per-stack fc2_bias, stack-0 fc2_w[13]/[27],
FC0 passthrough-row mean, R/Q piece_val — see `TDLEAF.md`).

```sh
# From learn/: compare consecutive checkpoints
python3 diff_tdleaf_checkpoints.py nn-fresh.tdleaf.bin-1e6g nn-fresh.tdleaf.bin-2e6g
```

Two positional arguments (old, new); no options.  Uses the
`compare_nnue_learning.py` reader (supports `.tdleaf.bin` v2–v11).

---

## hybrid_loop.py

One command per hybrid-loop iteration: promote a consolidated state → online
self-play generation with leaf/root corpus dumping → checkpoint → shard →
sharded multi-process offline training → merged-net export → gauntlet with an
Elo table.  Non-interactive; run from `learn/`.  Helper binaries (`Leaf_vbt`,
`Leaf_vtrain_hl_a/b`) are auto-compiled.  See `OFFLINE_TRAINING.md` for the
concepts and the manual runbook it encodes.

```sh
# Full iteration: generate 400k d8 games, consolidate (settled gen-2+ recipe),
# rate every epoch as it completes, then the final full gauntlet
python3 hybrid_loop.py --tag iter3 --games 400000 --depth 8 \
    --state iter2s2_final.tdleaf.bin \
    --shards 1 --bt-K 220 --bt-lambda 0.3 \
    --gauntlet-epochs --gauntlet Leaf_viter2s2-final Leaf_vclassic_eval

# Consolidate-only on existing corpora (e.g. a hyperparameter arm on the same dumps)
python3 hybrid_loop.py --tag iter2s2 --skip-online --shards 1 \
    --bt-K 220 --bt-lambda 0.3 \
    $(for f in iter2_work/iter2.*.tsv; do echo --corpus $f; done) \
    --gauntlet Leaf_vbtsp-final Leaf_vclassic_eval

# Generate-only (games + corpora, no offline training)
python3 hybrid_loop.py --tag gen3 --games 200000 --depth 8 --skip-train
```

Artifacts (named by `--tag`): `<tag>_final.nnue` (piece_val baked — compile
rating binaries from this), `<tag>_final.tdleaf.bin` (seeds the next iteration;
pairs with the ORIGINAL base `.nnue`), `<netbase>.tdleaf.bin-<tag>-online`
(post-generation checkpoint), `<tag>_work/` (dumps, shards, training logs).

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--tag NAME` | required | Iteration name; prefixes all artifacts |
| `--net FILE` | `nn-fresh-260628.nnue` | Base `.nnue` in `learn/` (never changes across iterations) |
| `--state FILE` | keep live | `.tdleaf.bin` to promote to the live state (backed up + hash-checked against the base) |
| `--skip-online` | off | Consolidate-only; train on `--corpus` files |
| `--games N` | 400000 | Games to generate |
| `--depth N` | 8 | Fixed search depth for generation |
| `--concurrency N` | 9 | Concurrent games |
| `--openings FILE` | `training_openings.epd` | Opening set (FRC) |
| `--quiet-cp N` | 60 | `TDLEAF_DUMP_QUIET_CP` for the dump |
| `--skip-train` | off | Generate-only |
| `--corpus TSV` | — | Extra corpus file(s) for training (repeatable) |
| `--shards N` | 8 | Parallel trainer processes (**use 1 for gen-2+ consolidation** — sync staleness destroys subtle signal; see `OFFLINE_TRAINING.md`) |
| `--epochs N` | 6 | Training epochs |
| `--bt-lr X` | 0.25 | LR scale on all category LRs |
| `--bt-lambda X` | 0.7 | Outcome weight in the blend target (0.3 recommended for gen-2+) |
| `--bt-K X` | 220 | Sigmoid temperature |
| `--bt-batch N` | 512 | Positions per Adam step |
| `--bt-leaf-lambda X` | = `--bt-lambda` | Outcome-weight ceiling for depth-0 leaf rows (default follows the root λ, the recommended setting) |
| `--bt-td-lambda X` | trainer default (`TDLEAF_LAMBDA`) | Result decay per ply from the game end: `w = λ_eff·td_λ^(N−ply)`; `1.0` = flat blend |
| `--sync-every N` | 256 | Batches between delta-merge syncs |
| `--gauntlet OPP …` | none | Opponent binaries in `learn/` (empty = skip) |
| `--gauntlet-games N` | 400 | Games per opponent |
| `--tc TC` | `3+0.05` | Gauntlet time control |
| `--gauntlet-epochs` | off | Per-epoch ladder: rate each epoch snapshot vs the first `--gauntlet` opponent as soon as its epoch finishes training; prints an epoch table (requires `--shards 1`) |
| `--epoch-games N` | 1000 | Games per epoch-ladder match |
| `--epoch-tc TC` | `1+0.01` | Epoch-ladder time control |
| `--no-final-gauntlet` | off | Skip the final full gauntlet (ladder-only runs; `--gauntlet` then names just the ladder opponent) |
| `--force` | off | Reuse an existing `<tag>_work` directory |
| `--recompile` | off | Force recompile of helper binaries |
