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
| `MATERIAL_ONLY=1` | `score_pos()` returns raw material balance only |
| `NNUE_NET=<file>` | Override default network file (`to-be-trained.nnue`) |
| `OVERWRITE` | Skip overwrite prompt |
| `NATIVE=1` | Compile with `-march=native -mtune=native` (max perf, non-portable). Default uses `-march=x86-64-v3` (AVX2, portable across Intel Haswell+ and AMD Zen 1+). |

The `.nnue` network file and `.tdleaf.bin` weights file must reside in the same directory as the binary.

---

## Architecture

### Unity build (`src/Leaf.cc` include order)

`main.cpp` → `uci.cpp` → `attacks.cpp` → `exmove.cpp` → `swap.cpp` → `moves.cpp` → `captures.cpp` →
`captchecks.cpp` → `hash.cpp` → `smp.cpp` → `search.cpp` → `score.cpp` →
`#if NNUE nnue.cpp` → `#if TDLEAF tdleaf.cpp` → `check.cpp` → `book.cpp` → `sort.cpp` →
`util.cpp` → `support.cpp` → `probe.cpp` → `setup.cpp` → `game_rec.cpp` →
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

- After each search, `tdleaf_record_ply()` walks the PV to the leaf, snapshots the accumulator, active feature indices, and iterative-deepening score history.
- After each game, `tdleaf_update_after_game()` computes backward TD errors (λ=0.8 decisive / 0.5 draw), applies score-change clipping (TDLEAF_SCORE_CLIP_CP=200 cp) and ID-stability weighting (TDLEAF_ID_VAR_SIGMA2=10,000 cp²), then backpropagates through FC/FT/PSQT/piece_val layers.  Five separate Adam LRs: FC (0.1), FT (1.0), FT bias (0.01), PSQT (1.6), piece_val (1.6).  Gradient clipping (L2 norm, threshold 1.0) and AdamW weight decay (1e-4, FC+FT weights only) are applied.  FT bias uses a reduced LR to prevent dying-ReLU from update frequency asymmetry.
- `tdleaf_replay()` then runs `TDLEAF_REPLAY_K` (default 0, disabled) additional passes over the last `TDLEAF_REPLAY_BUF_N` (default 8) completed games stored in a ring buffer, refreshing scores from current weights before each pass.  Replay uses FC-only gradients (`fc_only=true`) to avoid FT feedback divergence.
- Dense piece values (`piece_val[6][8]`, 48 floats) hold the base material prior and learn material corrections via dense gradient updates (~200/game); added to eval as `nnue_dense_piece_val()`.  PSQT starts at zero and learns only positional corrections; PSQT gradients are mean-centered per piece-type slot to prevent PSQT from absorbing material shifts (handled exclusively by piece_val, active when `piece_val_active`).  For fresh networks (`--init-nnue`): PSQT=0, piece_val=classical material (P=100,N=377,B=399,R=596,Q=1197 cp).  For `--init-nnue-noprior`: PSQT=0, piece_val=0 (learns all values from scratch).  Initialising PSQT with material values caused a 5000–7000-game training crash: large material evaluations saturate the sigmoid (sigmoid(1197/290)≈0.98 for queen), requiring impossibly high win rates from fresh engines to avoid net-corrosive PSQT gradients.
- **FC0 passthrough initialization:** FC0 output 15 (`NNUE_L0_DIRECT`) bypasses FC1/FC2 and contributes `fc0_raw[15] × 9600/8128` directly to the positional score before the `×100/5776` conversion.  With random FC0 weights (std=4) this creates ≈81 cp score noise at initialization, overwhelming the 100 cp/pawn piece_val signal.  The passthrough row is zero-initialized; gradient still flows through it so it learns normally from game 1.
- Weights persist to `<net>.tdleaf.bin` (v8 format); POSIX file locking + delta merging allows concurrent multi-instance training.  v6 adds persistent Adam second-moment (v) arrays and `t_adam`; multi-writer merge uses `max(v_file, v_local)`.  v7 adds persistent Adam first-moment (m) arrays; multi-writer merge uses element-wise average `(m_file + m_local) / 2`.  v8 adds sparse FT v persistence (same dirty-row set as FT weights); merge uses `max(v_file, v_local)`.  FT weight m not persisted (FT uses RMSProp, no m).
- FT RMSProp uses a session-local counter `t_ft_session` (not persisted) for bc2 via `min(t_adam, t_ft_session)` for cold rows.  Rows whose `v_ft_w` was restored from disk (`ft_v_warmed[fi]==true`) use `t_adam` directly for bc2 since their saved v is already calibrated.  This prevents ~31× oversized FT updates on the first Adam step after restart.
- `v_ft_w` (FT per-weight second moment, ~92 MB) is now sparsely persisted in .tdleaf.bin v8 for feature rows where v is **non-zero** (rows with zero v — e.g. first session from v7 — are explicitly excluded: saving a zero-v row as "warmed" causes bc2_warm≈1 with v=0, giving sv≈ε and ~10,000× oversized steps).  This eliminates the per-restart FT v cold-start that caused a 10–20 Elo dip for the first ~1000 Adam steps.  A per-session FT LR warmup (`TDLEAF_FT_SESSION_WARMUP=100` Adam steps) damps FT updates while v accumulates, covering both loaded (warmed) and fresh rows.
- `material` in `score.cpp` is **already STM (side-to-move) POV** — do not flip it.

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

TDLeaf learning is inactive in UCI mode (the engine never calls `make_move()`, so the TDLeaf hooks in `make_move()` are never reached).

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
# Run a match (from run/)
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
src/          Source code (unity-built via Leaf.cc)
docs/         Documentation (NNUE.md, TDLEAF.md, TODO.md, TRAINING.md, SCRIPT_USE.md, change_log.txt)
scripts/      Python automation scripts
run/          Compiled binaries + runtime config (search.par, opening book)
learn/        Training artifacts: .nnue, .tdleaf.bin, .games, pgn/
tools/        Third-party tools (cutechess-1.4.0/, BayesElo/)
TB/, TB_34/   Tablebase data (not in git)
testing/      Test suites and opening books
archives/     Historical EXchess source snapshots
```
