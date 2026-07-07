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
perl src/comp.pl <version> NNUE=1 NNUE_NET=nn-leaf-260414.nnue

# NNUE with net embedded in binary (no external .nnue file needed at runtime)
perl src/comp.pl <version> NNUE=1 NNUE_EMBED=1 NNUE_NET=nn-leaf-260414.nnue

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
| `NNUE_NET=<file>` | Override default network file (`nn-leaf-260414.nnue`) |
| `NNUE_EMBED=1` | Embed the `.nnue` file into the binary via incbin (requires `NNUE=1` and `NNUE_NET=<file>`). The net file must exist in `run/` or the current directory at compile time. At runtime, no external `.nnue` file is needed. |
| `OVERWRITE` | Skip overwrite prompt |
| `NATIVE=1` | Compile with `-march=native -mtune=native` (max perf, non-portable). Default uses `-march=x86-64-v3` (AVX2, portable across Intel Haswell+ and AMD Zen 1+). |
| `KNOWLEDGE=<N>` | Set compile-time default for `chess_skill` / `game.knowledge_scale` (1-100, default 100 = full strength).  Equivalent to setting the Skill slider to `N` at runtime; useful for building binaries pinned to a strength level for automated testing. |

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

`main.cpp` → `protocol.cpp` → `uci.cpp` → `attacks.cpp` → `exmove.cpp` → `swap.cpp` → `moves.cpp` → `captures.cpp` →
`captchecks.cpp` → `hash.cpp` → `smp.cpp` → `search.cpp` → `score.cpp` →
`#if NNUE nnue.cpp` → `#if NNUE nnue_io.cpp` → `#if NNUE_EMBED nnue_embed.cpp` →
`#if TDLEAF nnue_training.cpp` → `#if TDLEAF tdleaf.cpp` → `#if TDLEAF nnue_batch_train.cpp` → `check.cpp` → `book.cpp` → `sort.cpp` →
`util.cpp` → `support.cpp` → `setup.cpp` → `game_rec.cpp` →
`tree_search_functions.cpp`

Because of the unity build, LSP tools that analyse files individually will emit many false-positive
errors (unknown types, undeclared identifiers).  These are expected and can be ignored.

### Key source files

| File | Role |
|------|------|
| `src/main.cpp` | Protocol detection, xboard/CECP loop, CLI loop, game loop, TDLeaf hooks; defines grouped global state (`proto`, `engine_cfg`, `search_cfg`, `thread_cfg`) |
| `src/protocol.cpp` | xboard/CECP and interactive CLI command dispatch (`parse_command`), interrupt polling (`inter`) |
| `src/uci.cpp` | Full UCI implementation: I/O reader thread, command queue, `uci_loop()`, `uci_dispatch_go()`, `uci_set_position()`, `uci_check_interrupt()`, `uci_send_info()` |
| `src/engine_globals.h` | Grouped global state structs: `ProtocolState proto`, `EngineConfig engine_cfg`, `SearchConfig search_cfg`, `ThreadConfig thread_cfg` |
| `src/search.cpp` | PVS alpha-beta, null-move pruning, LMR, lazy SMP, iterative deepening; tracks `id_scores[]` for TDLeaf |
| `src/score.cpp` | Classical hand-crafted eval + NNUE dispatch; NNUE/pawn/score hash probe/store |
| `src/nnue.cpp` | NNUE runtime: weight arrays, accumulator init/update/delta, int8 forward pass (SIMD) |
| `src/nnue_io.cpp` | NNUE file I/O: `.nnue` load (file + embedded), LEB128, `nnue_write_nnue` |
| `src/nnue_training.cpp` | TDLeaf training: FP32 shadow weights, gradients, Adam optimizer, `.tdleaf.bin` I/O (TDLEAF=1 only) |
| `src/tdleaf.cpp` | PV walking, TD error computation (with score-change clipping + ID-stability weighting), gradient backprop, leaf/root corpus dump (`TDLEAF_DUMP_TSV`) |
| `src/nnue_batch_train.cpp` | Offline batch trainer (`--batch-train`): supervised training on quiet-position TSV corpora, λ-blend targets, multi-process `--bt-sync` (TDLEAF=1 only) |
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

**Two training modes** — online TDLeaf(λ) (below) and offline supervised consolidation of quiet-position corpora (`--batch-train` in any TDLEAF binary; corpora from `extract_quiet_positions.py` or the in-play `TDLEAF_DUMP_TSV` dump).  Together they form the hybrid loop (online generates games, offline extracts them fully, consolidated net re-enters online play) — driven end-to-end by `scripts/hybrid_loop.py`.  See `docs/OFFLINE_TRAINING.md`.

**Call flow:**
- After each search: `tdleaf_record_ply()` snapshots the PV leaf accumulator, feature indices, and ID score history.
- After each game: `tdleaf_update_after_game()` computes backward TD errors, backpropagates through all layers, and applies Adam/RMSProp updates.
- Optional replay: `tdleaf_replay()` runs additional passes over recent games (currently disabled, `TDLEAF_REPLAY_K=0`).

**Key hyperparameters:** `TDLEAF_K = 220 cp` (sigmoid temperature), `TDLEAF_LAMBDA = 0.98` (single eligibility trace decay — decisive and draw games use the same value).  Seven separate Adam LRs sized to ~0.001 × median(|w|) per section: FC0/FC1 weights (0.005), FC2 weights (0.07), FC biases (1.5), FT weights (0.015), FT bias (0.02), PSQT (13.0), piece_val (13.0).  Runtime sweep via env vars `TDLEAF_LR_{FC,FC2,FC_BIAS,FT,FT_BIAS,PSQT,PV}`.  Gradient clipping (L2 norm, 1.0) and AdamW weight decay (1e-4, FC+FT weights only).  Horizon noise mitigation: `TDLEAF_SCORE_CLIP_PAWNS = 1.0` (×max(value[PAWN], 100 cp); hand-tuned 0.5 → 1.0) and `TDLEAF_ID_VAR_SIGMA2 = 10000 cp²` (hand-tuned 625 → 10000, equivalent to a 100 cp std-dev reference).  Per-step safety: `TDLEAF_ADAM_STEP_CLIP = 30.0` uniform across categories.  Mini-batch size `TDLEAF_BATCH_SIZE = 8`.

**Material representation — pure-PSQT (current default).**  The bucketed PSQT is the *sole* trainable material channel; `piece_val` and the gauge machinery are disabled.  Defaults: `TDLEAF_PURE_PSQT = true` (tdleaf.h), `NNUE_FIXED_PIECE_VALUES 1` (define.h — search keeps classical `value[]`; `nnue_extract_piece_values()` is report-only).  Seed with `--init-nnue-classical` (classical material + phase-interpolated PST baked into PSQT) or `--init-nnue-noprior` (uniform 100 cp).  With one material channel there is no gauge null direction, so nothing for the multi-writer merge to amplify; absolute scale is loss-anchored by `TDLEAF_K = 220`.  Validated by an 8k-game two-writer merge smoke test (pawn pinned at 100 cp across all merge cycles, FC biases converge).  Full rationale in `docs/TDLEAF.md` ("Evaluation Gauge — Pure-PSQT"); planned code cleanup in `docs/MAINSTREAM_PLAN.md` Phase B.
- **⚠️ Do NOT freeze PSQT.**  Freezing (phase 1 = PSQT+piece_val, phase 1b = PSQT only) fails ~−200 Elo.  Outcome-scaling pressure is phase-dependent per material bucket; the bucketed PSQT is the only *linear* absorber; `piece_val` is bucket-independent and cannot substitute (1b proved this).  With PSQT frozen the pressure blows out the int8 FC output scale (won KQP-vs-K read +4033 cp vs reference +1286 cp).  Pure-PSQT works precisely because it keeps the bucketed PSQT *free* and removes only the redundant second channel.
- Gauge-machinery internals (`TDLEAF_PIN_PAWN_VALUE`, `nnue_mean_center_psqt_gradients`, Pass-2 dw centering, `nnue_recenter_psqt_slot_means`, persisted v11 slot-means) remain in the code but are gated off by `TDLEAF_PURE_PSQT`.  Retained for history; slated for deletion + `.tdleaf.bin` v12 in Phase B.
- Cosmetic drift: extracted pawn 100→107 cp per 500k online games — harmless because search values are fixed.  `TDLEAF_SCORE_CLIP_PAWNS` is in units of `max(value[PAWN], 100 cp)`, which stays 100 cp exactly under fixed values, so the clip does not stretch with drift.

**Critical gotchas for code changes:**
- `material` in `score.cpp` is **already STM (side-to-move) POV** — do not flip it.
- `piece_val` is **clamped ≥ 0** after each Adam step: negative piece_val inverts material evaluation, creating an unrecoverable death spiral.
- PSQT must be initialized **symmetrically** (own=+V, enemy=-V).  Asymmetric init (own=+2V, enemy=0) produces the same score but causes FT biases to explode to int16 saturation during training.
- FC0 passthrough row (output 15) is **zero-initialized**; gradient flows through it normally.
- FT uses RMSProp (no m), not full Adam.  Session-local `t_ft_session` prevents oversized steps after restart.
- Weights persist to `.tdleaf.bin` (v11 format: adds source-`.nnue` content hash (v10) and persisted PSQT init slot-means (v11)); POSIX file locking + delta merging for concurrent training.  The save path uses section-level buffered I/O + cached dirty-row bitmaps (learning overhead ≈ +13% wall clock at depth 6).
- TDLeaf works under BOTH protocols: xboard/CECP gets results via the protocol `result` command (`make_move()` hooks); UCI derives outcomes via `tdleaf_self_adjudicate()` in `uci_finish_game()` (terminal-position checks + score-history fallback; ambiguous games skip learning).
- Sustained training score far from 50% (e.g. vs a fixed stronger opponent) causes outcome-imbalance drift — the constant component of the TD outcome term is absorbed by FC output biases and other constant-capable channels, collapsing the net.  Self-play is immune and balanced play actively heals drift.  See "Outcome-Imbalance Drift" in `docs/TDLEAF.md`; canary monitor: `scripts/diff_tdleaf_checkpoints.py`.
- Setting `TDLEAF_DUMP_TSV=<prefix>` dumps quiet leaf+root training corpora during play (see `docs/OFFLINE_TRAINING.md`).

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
