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
| `--openings FILE` | — | EPD or PGN openings file, randomly ordered |

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
2. **Training partner** — choose one of three modes:
   - `[1]` Symmetric self-play — both `_a` and `_b` instances learn (default)
   - `[2]` Read-only opponent — `_a` learns; `_ro` holds weights fixed (`TDLEAF_READONLY=1`)
   - `[3]` External opponent — `_a` learns; user supplies path to any executable
3. **Train-validate loop** — optional; see below
4. **Build** — compiles the required binaries via `src/comp.pl` and moves them to `learn/`
5. **Continuity** — continue from existing `.tdleaf.bin` or start fresh (with optional
   Adam `cnt` priming for pre-trained nets)
6. **Match parameters** — TC, concurrency, wait, Fischer Random, per-engine depth limits;
   per-engine TCs (`--tc1` / `--tc2`) when the opponent runs at a different speed

On completion, trained weights are exported to `<net_base>-<total_games>g.nnue`.
Game counts accumulate in a `<net_base>.games` sidecar file across runs.

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
  5. Accept if LOS ≥ threshold → bank games, export accepted weights → best.nnue
     Reject → revert .tdleaf.bin to pre-cycle checkpoint
```

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

```sh
# Merge two training runs
python3 scripts/merge_tdleaf.py run1.tdleaf.bin run2.tdleaf.bin -o merged.tdleaf.bin

# Merge with summary statistics
python3 scripts/merge_tdleaf.py a.tdleaf.bin b.tdleaf.bin c.tdleaf.bin \
    -o merged.tdleaf.bin --report
```

### Key options

| Flag | Default | Description |
|------|---------|-------------|
| `files` (positional) | *(required)* | Two or more `.tdleaf.bin` input files |
| `-o`, `--output` | *(required)* | Output `.tdleaf.bin` path |
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
file appears in the output.  The output is always v4 format regardless of input
versions (v2/v3/v4 are all accepted).

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
