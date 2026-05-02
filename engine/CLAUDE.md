# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Project Overview

Leaf is a C++ chess engine (GPL v3) by Daniel C. Homan, originally developed as EXchess (1997–2017).
The 2026 restart adds NNUE evaluation (Stockfish 15.1–compatible HalfKAv2_hm architecture) and
TDLeaf(λ) online learning from self-play.  Supports UCI, xboard/CECP, and an interactive CLI;
protocol is auto-detected from the first command received on stdin.

---

## Build System

Compilation is managed by `src/comp.pl`.  The build uses a **unity build** pattern: `src/Leaf.cc`
includes every other `.cpp` file as a single translation unit.  Binaries land in `run/`.

```sh
# Classical eval (no NNUE)
perl src/comp.pl <version>

# NNUE eval
perl src/comp.pl <version> NNUE=1

# NNUE with a specific net file
perl src/comp.pl <version> NNUE=1 NNUE_NET=nn-to-be-trained.nnue

# NNUE with net embedded in binary (no external .nnue file needed at runtime)
perl src/comp.pl <version> NNUE=1 NNUE_EMBED=1 NNUE_NET=nn-to-be-trained.nnue

# NNUE + TDLeaf(λ) training
perl src/comp.pl <version> NNUE=1 TDLEAF=1

# Read-only weights (inference only)
perl src/comp.pl <version> NNUE=1 TDLEAF=1 TDLEAF_READONLY=1

# Skip interactive overwrite prompt
perl src/comp.pl <version> NNUE=1 OVERWRITE
```

Binary naming: `run/Leaf_v<version>` — e.g. `Leaf_v2026_03_09a`, `Leaf_vtrain_nn-fresh`.

### Key compile flags

| Flag | Effect |
|------|--------|
| `NNUE=1` | Enable NNUE evaluation |
| `TDLEAF=1` | Enable TDLeaf(λ) learning (requires NNUE=1) |
| `TDLEAF_READONLY=1` | Load `.tdleaf.bin` weights but skip updates |
| `TDLEAF_LOG_STEP_CLIPS=1` | Enable per-batch step-clip telemetry (writes `tdleaf_telemetry.log` next to the binary; default off) |
| `MATERIAL_ONLY=1` | `score_pos()` returns raw material balance only |
| `NNUE_NET=<file>` | Override default network file (`to-be-trained.nnue`) |
| `NNUE_EMBED=1` | Embed the `.nnue` file into the binary via incbin (requires `NNUE=1` and `NNUE_NET=<file>`). The net file must exist in `run/` or the current directory at compile time. At runtime, no external `.nnue` file is needed. |
| `OVERWRITE` | Skip overwrite prompt |
| `NATIVE=1` | Compile with `-march=native -mtune=native` (max perf, non-portable). Default uses `-march=x86-64-v3` (AVX2, portable across Intel Haswell+ and AMD Zen 1+). |

The `.nnue` network file and `.tdleaf.bin` weights file must reside in the same directory as the binary (unless `NNUE_EMBED=1` was used, in which case no external `.nnue` file is needed).

### CLI flags

| Flag | Effect |
|------|--------|
| `--uci` | Force UCI protocol mode |
| `--xboard` | Force xboard/CECP protocol mode |
| `--log` | Enable logging to `run.log` (default: off) |
| `hash <MB>` | Set hash table size |
| `cores <N>` | Set thread count |

---

## Architecture

### Unity build (`src/Leaf.cc` include order)

`main.cpp` → `uci.cpp` → `attacks.cpp` → `exmove.cpp` → `swap.cpp` → `moves.cpp` → `captures.cpp` →
`captchecks.cpp` → `hash.cpp` → `smp.cpp` → `search.cpp` → `score.cpp` →
`#if NNUE nnue.cpp` → `#if NNUE_EMBED nnue_embed.cpp` → `#if TDLEAF tdleaf.cpp` → `check.cpp` → `book.cpp` → `sort.cpp` →
`util.cpp` → `support.cpp` → `setup.cpp` → `game_rec.cpp` →
`tree_search_functions.cpp`

Because of the unity build, LSP tools that analyse files individually will emit many false-positive
errors (unknown types, undeclared identifiers).  These are expected and can be ignored.

### Key source files

| File | Role |
|------|------|
| `src/main.cpp` | Protocol detection, xboard/CECP loop, CLI loop, game loop, TDLeaf hooks (`tdleaf_record_ply`, `tdleaf_update_after_game`, `tdleaf_replay`) |
| `src/uci.cpp` | Full UCI implementation: I/O reader thread, command queue, `uci_loop()`, `uci_dispatch_go()`, `uci_set_position()`, `uci_check_interrupt()`, `uci_send_info()` |
| `src/search.cpp` | PVS alpha-beta, null-move pruning, LMR, lazy SMP, iterative deepening; tracks `id_scores[]` for TDLeaf |
| `src/score.cpp` | Classical hand-crafted eval + NNUE dispatch; NNUE/pawn/score hash probe/store |
| `src/nnue.cpp` | NNUE forward pass (int8 inference + NEON), FP32 shadow weights, gradient accumulation, `.tdleaf.bin` I/O |
| `src/tdleaf.cpp` | PV walking, TD error computation (with score-change clipping + ID-stability weighting), gradient backprop |
| `src/chess.h` | All major structs: `position`, `move`, `move_list`, `tree_search`, `game_rec` |
| `src/define.h` | Compile-time constants and flag defaults (`NNUE`, `TDLEAF`, `MATERIAL_ONLY`, piece encodings, `MAXD`, `MAX_GAME_PLY`) |
| `src/tdleaf.h` | TDLeaf hyperparameters, `TDRecord`, `TDGameRecord`, function declarations |

### NNUE architecture (HalfKAv2_hm, Stockfish 15.1–compatible)

- **Feature transformer:** 22,528 → 1,024 int16 accumulators per perspective + 8 int32 PSQT buckets
- **FC layers (×8 material-bucket stacks):** FC0 (1,024→16, SqrCReLU) → FC1 (30→32) → FC2 (32→1)
- **Score:** `(psqt_diff/2 + positional) × 100/5776` centipawns
- **Accumulators:** lazily updated at `exec_move` sites via `nnue_record_delta` / `nnue_apply_delta`
- **SIMD:** NEON (Apple M1/arm64) and AVX2/SSE4.1 (x86-64) hot paths for SqrCReLU, FC0, FC1; scalar fallback

### TDLeaf(λ) learning

See `docs/TDLEAF.md` for the full algorithm reference, hyperparameters, and gradient flow.

**Call flow:**
- After each search: `tdleaf_record_ply()` snapshots the PV leaf accumulator, feature indices, and ID score history.
- After each game: `tdleaf_update_after_game()` computes backward TD errors, backpropagates through all layers, and applies Adam/RMSProp updates.
- Optional replay: `tdleaf_replay()` runs additional passes over recent games (currently disabled, `TDLEAF_REPLAY_K=0`).

**Key hyperparameters:** `TDLEAF_K = 200 cp` (sigmoid temperature), `TDLEAF_LAMBDA = 0.98` (single eligibility trace decay — decisive and draw games use the same value).  Five separate Adam LRs: FC (0.10), FT (1.0), FT bias (0.01), PSQT (10.0), piece_val (50.0).  Gradient clipping (L2 norm, 1.0) and AdamW weight decay (1e-4, FC+FT weights only).

**Critical gotchas for code changes:**
- `material` in `score.cpp` is **already STM (side-to-move) POV** — do not flip it.
- `piece_val` is **clamped ≥ 0** after each Adam step: negative piece_val inverts material evaluation, creating an unrecoverable death spiral.
- PSQT must be initialized **symmetrically** (own=+V, enemy=-V).  Asymmetric init (own=+2V, enemy=0) produces the same score but causes FT biases to explode to int16 saturation during training.
- FC0 passthrough row (output 15) is **zero-initialized**; gradient flows through it normally.
- FT uses RMSProp (no m), not full Adam.  Session-local `t_ft_session` prevents oversized steps after restart.
- Weights persist to `.tdleaf.bin` (v8 format); POSIX file locking + delta merging for concurrent training.
- TDLeaf is inactive in UCI mode — hooks are in `make_move()` which UCI never calls.

### Protocol support

Leaf supports three interface modes, selected at runtime by the first command received:

| Mode | Trigger | Force flag |
|------|---------|------------|
| UCI | `"uci"` | `--uci` |
| xboard/CECP | `"xboard"` | `--xboard` |
| Interactive CLI | anything else | *(default)* |

- `uci_mode` (int) — set when running under UCI.
- `xboard` (int) — set when running under xboard/CECP.
- `interface_mode` (int) — set under **either** GUI protocol; used wherever the behaviour is "suppress console output / use GUI time management" rather than being xboard-specific.

UCI pondering: `go ponder` sets `uci_in_ponder=1` + `analysis_mode=1` rather than `game.ts.ponder=1`, because the GUI has already applied the expected opponent move in the `position` command. `ponderhit` clears `uci_in_ponder` and switches to a time-limited search on the same position. `stop` terminates the ponder search and emits `bestmove`.

UCI_Chess960: when `setoption name UCI_Chess960 value true` is sent, castling notation is handled by two boundary translation functions in `uci.cpp`, leaving `exec_move` and search untouched:
- `uci_960_input(s, pos)` → translates incoming king-captures-own-rook to Leaf's internal king-to-destination format.  Returns `true` if translated (castling detected).  Uses `PSIDE(sq[from])` for side detection, not rank — prevents false translation when a king captures an opponent's rook.  Only translates when the target square has an own rook matching `Krook[side]` or `Qrook[side]`.
- `uci_960_output(m, out, pos)` → translates outgoing castling moves (CASTLE flag) from king-destination to king-captures-rook via `Krook[side]`/`Qrook[side]`.
- `uci_parse_move` accepts `require_castle` flag from `uci_960_input` to disambiguate castling from regular king moves that share the same from/to squares.
- `exec_move` exempts CASTLE moves from the "can't capture own king" sanity check, handling from==to castling when king starts on the destination square.

### Important conventions

- `material` is already from the side-to-move's perspective (see `exmove.cpp:223`).
- Piece encoding: `PAWN=1 … KING=6`; white pieces are `+8` (e.g. `WPAWN=9`).
- `NOMOVE` terminates PV arrays.
- All docs live in `docs/`; update `docs/change_log.txt` when making notable changes.

---

## Scripts

All scripts live in `scripts/`; `run/` and `learn/` have symlinks for in-place invocation.
See `docs/SCRIPT_USE.md` for full option tables.

```sh
# Run a match — interactive mode (discovers engines, prompts for everything)
python3 scripts/match.py

# Run a match — CLI mode (from run/)
python3 scripts/match.py Leaf_vA Leaf_vB -n 200 -c 4 -tc 5+0.05

# Interactive training run (from learn/)
python3 scripts/training_run.py

# Bayesian Elo ratings (one or more PGN files combined)
python3 scripts/bayeselo_ratings.py file1.pgn file2.pgn --min 20 --report

# Remove duplicate games from PGN files
python3 scripts/pgn_dedup.py input.pgn --output deduped.pgn --report

# Generate combined FRC + book opening EPD (run once from learn/)
python3 scripts/make_training_epd.py

# Visualise weight changes after training
python3 scripts/compare_nnue_learning.py learn/nn-fresh.nnue learn/nn-fresh.tdleaf.bin

# Merge multiple .tdleaf.bin files (count-weighted averaging)
python3 scripts/merge_tdleaf.py run1.tdleaf.bin run2.tdleaf.bin -o merged --report

# Merge and also produce a .nnue file from a baseline network
python3 scripts/merge_tdleaf.py run1.tdleaf.bin run2.tdleaf.bin -o merged --baseline nn-start.nnue

# Win/draw/loss rate analysis per N-game window (default 100) for one player in a PGN
python3 scripts/pgn_winrate.py learn/pgn/run1/match_run1_0g.pgn
python3 scripts/pgn_winrate.py learn/pgn/run1/match_run1_0g.pgn --player Leaf_vtrain_nn-fresh_a --window 200
python3 scripts/pgn_winrate.py learn/pgn/run1/match_run1_0g.pgn --csv

# Extract per-position dataset from PGN files (for K/λ calibration)
python3 scripts/extract_positions.py --pgn-dir learn/pgn/nn-fresh-260410 --out learn/positions.parquet

# Calibrate sigmoid temperature K and lambda decay from extracted data
python3 scripts/analyze_calibration.py --input learn/positions.parquet --out-dir learn/calibration_plots --stage 5 6
```

### Training workflow summary

The recommended way to train is via `training_run.py`, which handles network init,
binary compilation, opponent rotation, checkpointing, and optional train-validate loops:

```sh
cd learn/
python3 training_run.py
```

Manual workflow (equivalent to what the script automates):

```sh
# 1. Initialise a fresh random network (optional)
perl src/comp.pl init_nnue NNUE=1 TDLEAF=1 OVERWRITE
./run/Leaf_vinit_nnue --init-nnue --write-nnue learn/nn-fresh.nnue
# Or with no material prior at all (learns piece values from scratch):
./run/Leaf_vinit_nnue --init-nnue-noprior --write-nnue learn/nn-fresh.nnue

# 2. Build training binaries (symmetric self-play: both engines learn)
perl src/comp.pl train_fresh_a NNUE=1 NNUE_NET=learn/nn-fresh.nnue TDLEAF=1 OVERWRITE
perl src/comp.pl train_fresh_b NNUE=1 NNUE_NET=learn/nn-fresh.nnue TDLEAF=1 OVERWRITE

# 3. Run training matches (from learn/)
python3 match.py Leaf_vtrain_fresh_a Leaf_vtrain_fresh_b -c 5 -tc 0:03+0.05 --wait 500 -n 500
```

---

## Directory Layout

```
engine/
  src/          Source code (unity-built via Leaf.cc)
  docs/         Documentation (NNUE.md, TDLEAF.md, TODO.md, TRAINING_RUN1.md, SCRIPT_USE.md, change_log.txt)
  scripts/      Python automation scripts
  run/          Compiled binaries + runtime config (opening book)
  learn/        Training artifacts: .nnue, .tdleaf.bin, .games, pgn/
gui/            LeafGUI Flutter chess GUI (see gui/CLAUDE.md)
logos/          Shared logo assets
tools/          Third-party tools (cutechess-1.4.0/, BayesElo/)
TB/, TB_34/     Tablebase data (not in git)
testing/        Test suites and opening books
archives/       Historical EXchess source snapshots
```
