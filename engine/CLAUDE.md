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
includes every other `.cpp` file as a single translation unit.  **`comp.pl` compiles into the
current directory** — run it from wherever the binary belongs:

- **`run/`** for *durable* executables: the end point of a line of work, kept for regular play.
- **`learn/`** for training and rating binaries — `cd engine/learn && perl comp.pl <version> ...`
  builds straight into `learn/`, where they run.  Do **not** build in `run/` and copy; and never
  execute anything with `run/` as the working directory, because `main_bk.dat` lives there and a
  training or rating binary would silently pick up book moves (`docs/TRAINING.md`, "The `run/`
  invariant").

Both directories carry a `comp.pl` symlink, so `perl comp.pl ...` works in either.
⚠️ It must be invoked **from `run/` or `learn/`** — `comp.pl` resolves `../src/Leaf.cc`
against the current directory, so `perl src/comp.pl` from `engine/` fails.

The examples below build a playable engine, so they run from `run/`:

```sh
cd engine/run/

# Classical eval (no NNUE)
perl comp.pl <version>

# NNUE eval
perl comp.pl <version> NNUE=1

# NNUE with a specific net file
perl comp.pl <version> NNUE=1 NNUE_NET=nn-leaf-260414.nnue

# NNUE with net embedded in binary (no external .nnue file needed at runtime)
perl comp.pl <version> NNUE=1 NNUE_EMBED=1 NNUE_NET=nn-leaf-260414.nnue

# NNUE + TDLeaf(λ) training
perl comp.pl <version> NNUE=1 TDLEAF=1

# Read-only weights (inference only)
perl comp.pl <version> NNUE=1 TDLEAF=1 TDLEAF_READONLY=1

# Skip interactive overwrite prompt
perl comp.pl <version> NNUE=1 OVERWRITE
```

Binary naming: `Leaf_v<version>` in whichever directory you build from — e.g.
`run/Leaf_v2026_03_09a` for a durable engine, `learn/Leaf_vtrain_nn-fresh` for a training binary.

### Key compile flags

| Flag | Effect |
|------|--------|
| `NNUE=1` | Enable NNUE evaluation |
| `TDLEAF=1` | Enable TDLeaf(λ) learning (requires NNUE=1) |
| `TDLEAF_READONLY=1` | Load `.tdleaf.bin` weights but skip updates |
| `TDLEAF_LOG_STEP_CLIPS=1` | Enable per-batch step-clip telemetry (writes `tdleaf_telemetry.log` next to the binary; default off) |
| `TDLEAF_BATCH_SIZE_DEFAULT=N` | Override the mini-batch size (default 50; counts GAMES) |
| `TDLEAF_LEAF_MATCH_CP_DEFAULT=N` | Override the PV leaf-match gate in cp (default 10; 0 disables) |
| `PV_LAST_RESOLVED=0` / `PV_NO_TT_CUTOFF=0` | Disable a PV repair (both default 1, learning-only) |
| `PVTRUNC_DIAG=1` | PV-walk / leaf-match telemetry: pv_len histogram, root PV provenance, leaf-vs-root correspondence, aspiration exit reasons (default off) |
| `SCORE_HASH_OFF=1`, `FAIL_SOFT_DIAG=1`, `WIDEWIN=N` | Diagnostics retained from the PV investigation; all default no-ops.  Each encodes a NEGATIVE result — see `docs/Learning_Investigation.md` §1 L before re-testing |
| `MATERIAL_ONLY=1` | `score_pos()` returns raw material balance only |
| `NNUE_NET=<file>` | Override default network file (`nn-leaf-260414.nnue`) |
| `NNUE_EMBED=1` | Embed the `.nnue` file into the binary via incbin (requires `NNUE=1` and `NNUE_NET=<file>`). The net file must exist in `run/` or the current directory at compile time. At runtime, no external `.nnue` file is needed. |
| `OVERWRITE` | Skip overwrite prompt |
| `NATIVE=1` | Tune for the build machine (`-march=native`, or `-mcpu=native` on arm64). Fast but **non-portable — never use for a distributed binary**. Default is portable: `-march=x86-64-v3` on x86-64 targets, and untuned on macOS (measured identical in NPS and node count on Apple Silicon) |
| `CXX=<compiler>` | Compiler to invoke (default `g++`). Set to a cross compiler, e.g. `CXX=x86_64-w64-mingw32-g++` |
| `WINDOWS=1` | Target Windows: appends `.exe`, defines `MINGW=1`, links `-static` (one self-contained exe, no MinGW DLLs). Auto-detected when building under MSYS2/Cygwin |
| `STATIC=1` | Static-link libstdc++/libgcc (Linux; implied on Windows) so release binaries run on older distros |
| `MACOS_MIN=<ver>` | macOS deployment target (default `11.0`, the first macOS with Apple Silicon). **Do not remove the default** — without it clang stamps the binary with the build host's OS version and it refuses to launch on anything older |
| `KNOWLEDGE=<N>` | Set compile-time default for `chess_skill` / `game.knowledge_scale` (1-100, default 100 = full strength).  Equivalent to setting the Skill slider to `N` at runtime; useful for building binaries pinned to a strength level for automated testing. |

The `.nnue` network file and `.tdleaf.bin` weights file must reside in the same directory as the binary (unless `NNUE_EMBED=1` was used, in which case no external `.nnue` file is needed).

### CLI flags

| Flag | Effect |
|------|--------|
| `--uci` | Force UCI protocol mode |
| `--xboard` | Force xboard/CECP protocol mode |
| `--log` | Enable logging to `run.log` (default: off) |
| `--no-pv-learning` | Disable the learning-only PV repairs, restoring competitive search behaviour.  For rating a TDLEAF binary; plain `NNUE=1` binaries never enable them |
| `hash <MB>` | Set hash table size |
| `cores <N>` | Set thread count |

---

## Architecture

### Unity build (`src/Leaf.cc` include order)

`main.cpp` → `protocol.cpp` → `uci.cpp` → `attacks.cpp` → `exmove.cpp` → `swap.cpp` → `moves.cpp` → `captures.cpp` →
`captchecks.cpp` → `hash.cpp` → `smp.cpp` → `search.cpp` → `score.cpp` →
`#if NNUE nnue.cpp` → `#if NNUE nnue_io.cpp` → `#if NNUE_EMBED nnue_embed.cpp` →
`#if TDLEAF nnue_training.cpp` → `#if TDLEAF tdleaf.cpp` → `#if TDLEAF nnue_batch_train.cpp` →
`#if TDLEAF selfplay.cpp` → `check.cpp` → `book.cpp` → `sort.cpp` →
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
| `src/nnue_batch_train.cpp` | Offline batch trainer (`--batch-train`): supervised training on quiet-position TSV corpora, λ-blend targets, within-batch thread parallelism (`--bt-threads`) (TDLEAF=1 only) |
| `src/selfplay.cpp` | Internal self-play driver (`--selfplay`: whole games in one process, both sides recorded every ply, exact in-engine results, optional fastchess-format PGN via `--pgn-out`) and the trajectory learner (`--learn-stream`: consumes actor-emitted `.tdg` files with ONE optimizer) (TDLEAF=1 only) |
| `src/selfplay_traj.h` | Binary `.tdg` per-game trajectory format (actor → learner): positions + search outputs + POV/gate metadata; accumulators/features/stack rebuilt exactly by `tdleaf_rebuild_record` |
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

See `docs/TRAINING.md` for the full algorithm reference, hyperparameters, and gradient flow.

**Two training modes** — online TDLeaf(λ) (below) and offline supervised consolidation of quiet-position corpora (`--batch-train` in any TDLEAF binary; corpora from `extract_quiet_positions.py` or the in-play `TDLEAF_DUMP_TSV` dump).  Together they form the hybrid loop (online generates games, offline extracts them fully, consolidated net re-enters online play) — driven end-to-end by `scripts/train.py`, including chained iterations via `--continue`.  See `docs/TRAINING.md`.

**Generation for the online phase** (`docs/TRAINING.md` "Generation Modes").  The **only mode is the actor/learner split** (`train.py` always uses it — no generation-mode flag): N−1 frozen actors emit `.tdg` trajectories, ONE learner owns the optimizer — sole `.tdleaf.bin` writer, bit-exact vs single-process online given same state/env.  Driver: `scripts/selfplay_run.py` (epoch-style weight refresh via actor respawn).  The two former modes (`--uci-pair-gen`, the `match.py`/fastchess UCI engine pair; and `--selfplay-gen`, N striped independent engines) were removed — both were multi-writer, merging N optimizers into one `.tdleaf.bin`.  The engine's internal `--selfplay` driver they used is still the per-actor engine of the split; `match.py` remains as the gauntlet/rating tool.

**Call flow:**
- After each search: `tdleaf_record_ply()` snapshots the PV leaf accumulator, feature indices, and ID score history.
- After each game: `tdleaf_update_after_game()` computes backward TD errors, backpropagates through all layers, and applies Adam/RMSProp updates.
- (The epoch-based experience-replay ring buffer, disabled at `TDLEAF_REPLAY_K=0`, was removed in the Phase-3 cleanup.)

**Key hyperparameters:** `TDLEAF_K = 220 cp` (sigmoid temperature), `TDLEAF_LAMBDA = 0.985` — eligibility trace decay expressed **per game-ply**: the trace applies `λ^dply` where `dply` is the game-ply gap between consecutive records (1 under internal self-play, 2 under UCI play where only the engine's own moves are recorded).  **One LR set is shared by both phases at scale 1.0** (online `--lr-scale`, offline `--bt-lr`, both default 1.0): FC0 0.00035, FC1 0.00125, FC2 0.0175, FC biases 0.375, FT 0.00375, FT bias 0.005, PSQT 3.25.  ⚠️ These are **0.25× the pre-2026-09-15 constants** — divide any absolute LR from the chain history by 4 before comparing, and note every "online cost" figure in the research record was measured at 4× the current online LR.  Sections split into **stationary** (fc0_w, ft_w, psqt_w — a magnitude rule is well defined) and **scale-finding** (fc2_w plus the five bias sections, which start at zero; their LR is a growth rate, so never size them off a converged magnitude).  Gradient clipping (L2 norm, **2.0** — rescale with √B if the batch changes) and AdamW weight decay (1e-4, FC+FT weights only).  `TDLEAF_ADAM_EPS = 1e-12` (raising it makes the optimizer sensitive to accumulated gradient scale).  Horizon noise mitigation: `TDLEAF_SCORE_CLIP_PAWNS = 1.0` (constant 100 cp under the default `NNUE_FIXED_PIECE_VALUES`) and `TDLEAF_ID_VAR_SIGMA2 = 10000 cp²`.  Per-step safety: `TDLEAF_ADAM_STEP_CLIP = 30.0`.  Mini-batch size **`TDLEAF_BATCH_SIZE = 50`** (counts GAMES, not records; compile-time overridable) and `TDLEAF_ADAM_WARMUP = 100` steps — counted in steps, so 5,000 games at batch 50.

**PV quality (2026-09-16):** TDLeaf trains at the PV leaf, so the recorded PV must locate the position the search valued.  `PV_LAST_RESOLVED = 1` returns the last *resolved* iteration's PV instead of the root fail-high stub (which wrote a 2-ply PV explicitly, 35.9% of searches); `PV_NO_TT_CUTOFF = 1` suppresses TT **cutoffs** at PV nodes only; `TDLEAF_LEAF_MATCH_CP = 10` gates both the online trace and the dumped **leaf** rows on `|leaf_static − propagated root search| ≤ 10 cp` (root rows are gated on root *quietness*, a different quantity, and are untouched).  All three are **learning-only** — competitive search is verified node-identical, and `--no-pv-learning` disables them for rating with a TDLEAF binary.  Combined: coherent label bias −24.88 → +0.15 cp, sd 185.9 → 79.7, ~74% of records retained.  **No Elo measured yet.**  ⚠️ `TDRecord::leaf_ok` is not a `.tdg` field — it is recomputed in `tdleaf_rebuild_record`, because the learner never calls `tdleaf_record_ply`.  Any new derived field needs the same treatment and an **actor/learner** smoke test, not just `--selfplay`.

**Material representation — pure-PSQT (only mode).**  The bucketed PSQT is the *sole* trainable material channel.  There is no separate dense `piece_val` channel and no gauge-anchoring machinery in the codebase — both were fully deleted (not just disabled by a flag) once pure-PSQT was mainstreamed.  Defaults: `NNUE_FIXED_PIECE_VALUES 1` (define.h — search keeps classical `value[]`; `nnue_extract_piece_values()` is report-only).  Seed with `--init-nnue-classical` (classical material + phase-interpolated PST baked into PSQT) or `--init-nnue-noprior` (uniform 100 cp).  With one material channel there is no gauge null direction, so nothing for the multi-writer merge to amplify; absolute scale is loss-anchored by `TDLEAF_K = 220`.  Validated by an 8k-game two-writer merge smoke test (pawn pinned at 100 cp across all merge cycles, FC biases converge).  Full rationale and the retired gauge machinery in `docs/history/TRAINING_HISTORY.md`.
- **⚠️ Do NOT freeze PSQT.**  Freezing (phase 1 = PSQT+piece_val, phase 1b = PSQT only) fails ~−200 Elo.  Outcome-scaling pressure is phase-dependent per material bucket; the bucketed PSQT is the only *linear* absorber; the old `piece_val` channel was bucket-independent and could not substitute (1b proved this).  With PSQT frozen the pressure blows out the int8 FC output scale (won KQP-vs-K read +4033 cp vs reference +1286 cp).  Pure-PSQT works precisely because it keeps the bucketed PSQT *free*.
- The old gauge-machinery internals (pawn pin, gradient mean-centering, post-Adam dw centering, persisted PSQT slot-means) and the dense `piece_val` channel itself were deleted from the code, along with the `.tdleaf.bin` sections they used — current format is v12. Full mechanism writeup in `docs/history/TRAINING_HISTORY.md`.
- Cosmetic drift: extracted pawn 100→107 cp per 500k online games — harmless because search values are fixed.  `TDLEAF_SCORE_CLIP_PAWNS` is in units of `max(value[PAWN], 100 cp)`, which stays 100 cp exactly under fixed values, so the clip does not stretch with drift.

**Critical gotchas for code changes:**
- `material` in `score.cpp` is **already STM (side-to-move) POV** — do not flip it.
- PSQT must be initialized **symmetrically** (own=+V, enemy=-V).  Asymmetric init (own=+2V, enemy=0) produces the same score but causes FT biases to explode to int16 saturation during training.
- FC0 passthrough row (output 15) is **zero-initialized**; gradient flows through it normally.
- FT uses RMSProp (no m), not full Adam.  Session-local `t_ft_session` prevents oversized steps after restart.
- Weights persist to `.tdleaf.bin` (v12 format: adds source-`.nnue` content hash (v10); the loader still accepts v2–v11, discarding the now-removed piece-value and PSQT-slot-means fields from those older files).  Since the actor/learner learner is the sole writer, the save path is a direct atomic write (write to `.tmp`, `rename()`) of the in-memory weights / counts / Adam v,m — the former multi-writer POSIX-lock + delta-merge machinery was removed in the Phase-3 cleanup (a `flock` guard against accidental concurrent writers is kept).  On save the in-memory Adam v,m are written as-is (the old max/avg cross-writer merge is gone), so a reload continues from the faithful state.  Save uses section-level buffered I/O (learning overhead ≈ +13% wall clock at depth 6).
- TDLeaf works under BOTH protocols: xboard/CECP gets results via the protocol `result` command (`make_move()` hooks); UCI derives outcomes via `tdleaf_self_adjudicate()` in `uci_finish_game()` (terminal-position checks + score-history fallback; ambiguous games skip learning).  The training path (actor/learner internal self-play) uses exact in-process results, not UCI adjudication — the self-adjudication path now matters only for UCI play outside training (e.g. GUI/analysis).
- Sustained training score far from 50% (e.g. vs a fixed stronger opponent) causes outcome-imbalance drift — the constant component of the TD outcome term is absorbed by FC output biases and other constant-capable channels, collapsing the net.  Self-play is immune and balanced play actively heals drift.  See "Outcome-Imbalance Drift" in `docs/TRAINING.md`; canary monitor: `scripts/diff_tdleaf_checkpoints.py`.
- **Online-stability rules** (each violation collapsed a full iteration; `docs/TRAINING.md` "Online-stability rules"): (1) online-learning self-play must play to **natural termination** — resign/draw adjudication + learning is a runaway decisiveness spiral; (2) TD targets must be on **current weights** — the actor/learner path requires the learner's `--refresh-scores` (trajectory scores otherwise lag by a refresh cycle).  The health canary is the **draw rate** (healthy ≈ 35–40% at d8), NOT gradient norms — both collapses kept nominal grad telemetry throughout.
- Env diagnostics: `TDLEAF_CHECK_ACC=1` verifies walked-vs-rebuilt leaf accumulators per record (this check caught the FRC castle phantom-capture bug in `nnue_record_delta`); `TDLEAF_TRACE_UPDATE=<file>` dumps a per-record gradient trace in exact hex.
- **Env guardrail** (`tdleaf_check_env()`, runs at `main()` entry in TDLEAF builds): the binary **hard-errors on any `TDLEAF_*` env var outside the allowlist** (`TDLEAF_FREEZE`, `TDLEAF_DUMP_TSV`, `TDLEAF_DUMP_QUIET_CP`, `TDLEAF_DUMP_MAX_CP`, `TDLEAF_CHECK_ACC`, `TDLEAF_TRACE_UPDATE`) and logs the effective training config (K, λ, LRs, batch) at startup.  **Adding a new `TDLEAF_*` env var requires adding it to this allowlist** or the binary refuses to start.  The experimental target-mode / root-learning / `TDLEAF_LR_*` knobs were removed in Phase 1 (`docs/SIMPLIFICATION_PLAN.md`).
- Setting `TDLEAF_DUMP_TSV=<prefix>` dumps quiet leaf+root training corpora during play (see `docs/TRAINING.md`).  Under `--learn-stream`, set it on the learner to dump the corpus from consumed trajectories.

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

All scripts live in `scripts/`; `run/` and `learn/` have symlinks for in-place invocation
(including `comp.pl`, so `perl comp.pl <version> ...` builds into whichever of the two you are in).
See `docs/SCRIPT_USE.md` for full option tables.

```sh
# Run a match — interactive mode (discovers engines, prompts for everything)
python3 scripts/match.py

# Run a match — CLI mode (from run/)
python3 scripts/match.py Leaf_vA Leaf_vB -n 200 -c 4 -tc 5+0.05

# One full hybrid-loop iteration: generate -> consolidate -> gauntlet (from learn/)
# Generation defaults to the actor/learner split (frozen actors + ONE learner)
python3 scripts/train.py --tag iter1 --games 188000 --depth 8 \
    --state <net>.tdleaf.bin --bt-K 220 --bt-threads 8 \
    --gauntlet-epochs

# Chained iteration: --continue reads the previous run's sidecar JSON
python3 scripts/train.py --tag iter2 --continue iter1 --games 188000 --depth 8

# Standalone actor/learner run (what train.py's generation phase wraps; from learn/)
python3 scripts/selfplay_run.py --binary Leaf_vtrain_hl_a --epd training_openings.epd \
    --actors 8 --depth 8 --games-per-actor 1000 --total-games 100000 \
    --traj-dir traj_run1 --refresh-scores

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

For the hybrid loop (online generation + offline consolidation + gauntlet, the
recommended path for iterative training), `train.py` drives one full iteration
per invocation — defaulting to the actor/learner generation split — with
`--continue` to chain iterations without re-specifying `--net`/`--state`/opponent
lists by hand:

```sh
cd learn/
python3 train.py --tag iter1 --games 188000 --depth 8 \
    --bt-K 220 --bt-threads 8 --gauntlet-epochs --gauntlet Leaf_vclassic_eval
```

For standalone online-only generation (no offline consolidation), use
`scripts/selfplay_run.py` directly (the actor/learner driver that `train.py`
wraps).  The older interactive `training_run.py` manager is archived under
`scripts/older/`.

See `docs/TRAINING.md` for the full workflow and `docs/SCRIPT_USE.md` for the option
tables.

Manual online-only workflow (equivalent to what `selfplay_run.py` automates):

```sh
# All of this runs from engine/learn/ — training binaries are built and run
# there.  run/ is only for durable engines, and nothing executes with run/ as
# its cwd (main_bk.dat would feed it book moves).

# 1. Initialise a fresh random network (optional)
perl comp.pl init_nnue NNUE=1 TDLEAF=1 OVERWRITE
./Leaf_vinit_nnue --init-nnue --write-nnue nn-fresh.nnue
# Or with no material prior at all (learns piece values from scratch):
./Leaf_vinit_nnue --init-nnue-noprior --write-nnue nn-fresh.nnue

# 2. Build training binaries (symmetric self-play: both engines learn)
perl comp.pl train_fresh_a NNUE=1 NNUE_NET=nn-fresh.nnue TDLEAF=1 OVERWRITE
perl comp.pl train_fresh_b NNUE=1 NNUE_NET=nn-fresh.nnue TDLEAF=1 OVERWRITE

# 3. Run training matches
python3 match.py Leaf_vtrain_fresh_a Leaf_vtrain_fresh_b -c 5 -tc 0:03+0.05 --wait 500 -n 500
```

---

## Directory Layout

```
engine/
  src/          Source code (unity-built via Leaf.cc)
  docs/         Documentation (NNUE.md, TRAINING.md, SCRIPT_USE.md, TODO.md,
                Learning_Investigation.md — the distilled synthesis of what is
                known about the hybrid loop, READ THIS ONE; the chronological
                records it distils, Online_Learning_Investigation.md and
                Offline_Learning_Investigation.md; Generation_Throughput.md,
                SIMPLIFICATION_PLAN.md, change_log.txt, history/ for retired
                designs and experiment write-ups)
  scripts/      Python automation scripts
  run/          DURABLE binaries + runtime config (opening book, incl.
                main_bk.dat).  End-of-line-of-work engines for regular play.
                Nothing ever EXECUTES with run/ as its working directory —
                main_bk.dat would silently feed book moves to a training or
                rating binary.  Build training binaries in learn/ instead
  learn/        Training artifacts: .nnue, .tdleaf.bin, per-run <tag>_work/
                archives, PGNs
gui/            LeafGUI Flutter chess GUI (see gui/CLAUDE.md)
logos/          Shared logo assets
tools/          Third-party tools (cutechess-1.4.0/, BayesElo/)
TB/, TB_34/     Tablebase data (not in git)
testing/        Test suites and opening books
archives/       Historical EXchess source snapshots
```
