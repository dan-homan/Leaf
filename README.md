<p align="center">
  <img src="logos/leaf.svg" alt="Leaf Chess Engine" width="500"/>
</p>

# Leaf

**Leaf** is an open-source chess engine written in C++ by Daniel C. Homan, an astrophysicist at Denison University in Granville, Ohio.  Originally developed under the name **EXchess**, development began in the late 1990s and the engine was actively maintained and released through early 2017.  After a long hiatus, the project was restarted in 2026 with significant new features developed in collaboration with [Claude Code](https://claude.ai/claude-code) (Anthropic) and renamed to Leaf.

---

## Quick Start

```sh
# Build with NNUE evaluation (from engine/run/)
cd engine/run/
perl comp.pl myversion NNUE=1 NNUE_NET=nn-to-be-trained.nnue

# Run (auto-detects UCI, xboard, or interactive CLI)
./Leaf_vmyversion
```

The `.nnue` network file must be in the same directory as the binary.  To embed it directly into the binary (no external file needed):

```sh
perl comp.pl myversion NNUE=1 NNUE_EMBED=1 NNUE_NET=nn-to-be-trained.nnue
```

See [Build Options](#build-options) for all compile flags.

---

## History

EXchess (now Leaf) first appeared around 1997–1998 and was one of a handful of serious open-source engines of that era.  Over two decades of on-and-off development produced a series of increasingly capable versions — from the early v2/v3 series through the v6 and v7 lines released in 2011–2017.

The engine includes a classical hand-crafted evaluation function, principal variation search (PVS), null-move pruning, late move reductions, static exchange evaluation, history heuristics, and a lazy SMP implementation.  It communicates via UCI, xboard/CECP, and an interactive CLI, with protocol auto-detection.

The final pre-hiatus release was **v7.97b** (February 2017), rated around **2,772 Elo** on CCRL 40/40.  More history can be found on the [Chessprogramming wiki](https://www.chessprogramming.org/EXchess).

---

## 2026 Restart — Collaboration with Claude Code

In early 2026 the project was restarted with a focus on two major new capabilities: a **Stockfish-compatible NNUE evaluation** and a **TDLeaf(λ) online learning** system that can train the NNUE weights from self-play.

This work was developed interactively with [Claude Code](https://claude.ai/claude-code), Anthropic's AI coding assistant.  The collaboration covered design, implementation, debugging, and tuning — from the initial NNUE forward-pass implementation through to verifying the evaluation matched Stockfish 15.1 exactly and training the network from self-play games.

---

## Features

### NNUE Evaluation

Leaf supports **HalfKAv2_hm** NNUE evaluation compatible with Stockfish 15.1 era networks (22,528 features, 1,024-unit accumulator, 8 layer stacks with FC0→FC1→FC2).  The forward pass matches Stockfish 15.1 evaluation exactly (within 1 cp rounding) on every tested position.  NEON (Apple M-series) and AVX2 (x86-64) SIMD acceleration is included.

Current result with the Stockfish 15.1 net: **92W–8D–0L (96.0%)** vs the classical Leaf eval at 10+0.1s.

See [`engine/docs/NNUE.md`](engine/docs/NNUE.md) for full architecture notes, score formula, and optimization history.

### TDLeaf(λ) Online Learning

Leaf includes a complete **TDLeaf(λ)** reinforcement learning system that trains all NNUE layers from self-play games — FC weights, the 46 MB feature transformer, PSQT weights, and dense piece values.  The long-term goal is for Leaf to develop its own network, tuned to its own search, entirely through self-play.

Key features:
- PV leaf scores as the TD signal with full NNUE backpropagation
- Sparse FT/PSQT updates (only active feature rows touched per position)
- Adam optimizer with persistent momentum across training sessions (`.tdleaf.bin` v8 format)
- Concurrent multi-instance training via POSIX file locking and delta merging
- Automated training via `scripts/training_run.py` with opponent rotation, checkpointing, and train-validate loops

Build with `NNUE=1 TDLEAF=1`.  See [`engine/docs/TDLEAF.md`](engine/docs/TDLEAF.md) for the full algorithm, hyperparameter reference, and training workflow.

### Chess960 / Fischer Random

Full Chess960 support in both UCI and xboard protocols.  UCI_Chess960 castling notation is handled by boundary translation at the I/O layer, leaving the search and move execution untouched.

### LeafGUI

**LeafGUI** is a cross-platform Flutter chess GUI included in the `gui/` directory.  It provides a graphical interface for playing against Leaf (or any UCI engine), watching engine-vs-engine matches, and analyzing positions.

Key features: Chess960 support with engine capability detection, engine registry with persistent storage, engine-vs-engine mode with dual output, multiple time controls, move list navigation, FEN copy/load, and per-engine skill level adjustment.

```sh
cd gui/
export PATH="$HOME/develop/flutter/bin:$PATH"
flutter pub get && flutter build macos --release
```

See [`gui/CLAUDE.md`](gui/CLAUDE.md) for full GUI documentation.

---

## Build Options

Compilation is managed by `comp.pl` (in both `engine/src/` and `engine/run/`).  Built binaries land in `engine/run/` as `Leaf_v<version>`.

```sh
cd engine/run/

# Classical eval (no NNUE)
perl comp.pl <version>

# NNUE eval
perl comp.pl <version> NNUE=1

# NNUE with a specific net file
perl comp.pl <version> NNUE=1 NNUE_NET=nn-to-be-trained.nnue

# NNUE with net embedded in binary
perl comp.pl <version> NNUE=1 NNUE_EMBED=1 NNUE_NET=nn-to-be-trained.nnue

# NNUE + TDLeaf(λ) training
perl comp.pl <version> NNUE=1 TDLEAF=1

# Skip interactive overwrite prompt
perl comp.pl <version> NNUE=1 OVERWRITE
```

| Flag | Effect |
|------|--------|
| `NNUE=1` | Enable NNUE evaluation |
| `NNUE_NET=<file>` | Override default network file (`to-be-trained.nnue`) |
| `NNUE_EMBED=1` | Embed `.nnue` into binary via incbin (requires `NNUE=1` + `NNUE_NET`) |
| `TDLEAF=1` | Enable TDLeaf(λ) learning (requires `NNUE=1`) |
| `TDLEAF_READONLY=1` | Load trained weights but skip updates |
| `MATERIAL_ONLY=1` | `score_pos()` returns raw material balance only |
| `OVERWRITE` | Skip overwrite prompt |
| `NATIVE=1` | `-march=native` (max perf, non-portable) |

The network file must be in the same directory as the binary (unless `NNUE_EMBED=1`).  Runtime data files (`search.par`, `main_bk.dat`) must also be present.

---

## Running

Leaf auto-detects UCI, xboard/CECP, or interactive CLI from the first command on stdin.  Point any compatible GUI at the binary, or run directly:

```sh
cd engine/run/
./Leaf_v<version>
```

### Engine Matches

Matches between engines require [cutechess-cli](https://github.com/cutechess/cutechess):

```sh
cd engine/run/

# Interactive mode — discovers engines, prompts for options
python3 match.py

# CLI mode
python3 match.py Leaf_vA Leaf_vB -n 200 -c 4 -tc 10+0.1

# Chess960
python3 match.py Leaf_vA Leaf_vB -n 200 --fischer-random
```

External UCI engines (e.g. Stockfish) can be placed in `tools/engines/<name>/` and will be auto-discovered.  See [`engine/docs/SCRIPT_USE.md`](engine/docs/SCRIPT_USE.md) for full script documentation.

### Training

The recommended training workflow is `scripts/training_run.py` (run from `engine/learn/`):

```sh
cd engine/learn/
python3 training_run.py
```

This handles network initialization, binary compilation, opponent rotation, checkpointing, and optional train-validate loops.  See [`engine/docs/TDLEAF.md`](engine/docs/TDLEAF.md) for details.

---

## Directory Layout

```
engine/
  src/          C++ source code (unity build via Leaf.cc)
  docs/         Documentation (NNUE.md, TDLEAF.md, SCRIPT_USE.md, change_log.txt)
  scripts/      Python automation scripts
  run/          Compiled binaries + runtime data (search.par, opening book)
  learn/        Training artifacts (.nnue, .tdleaf.bin, PGN)
gui/            LeafGUI Flutter chess GUI
logos/           Shared logo assets
tools/           Third-party tools (cutechess, BayesElo, external engines)
testing/         Test suites and opening books
archives/        Historical EXchess source snapshots
```

---

## License

GNU General Public License.  See [`engine/docs/license.txt`](engine/docs/license.txt).

---

## Acknowledgements

- Classical search and evaluation by **Daniel C. Homan** (1997–2017, 2026–present)
- NNUE architecture and network statistics from the [Stockfish](https://stockfishchess.org) project (GPL v3)
- NNUE implementation, TDLeaf(λ) learning system, and 2026 restart developed in collaboration with **[Claude Code](https://claude.ai/claude-code)** (Anthropic)
- [Chessprogramming wiki](https://www.chessprogramming.org) for algorithm references
