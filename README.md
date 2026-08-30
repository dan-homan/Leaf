<p align="center">
  <img src="logos/leaf.svg" alt="Leaf Chess Engine" width="500"/>
</p>

# Leaf

**Leaf** is an open-source chess engine written in C++ by Daniel C. Homan, an astrophysicist at Denison University in Granville, Ohio.  Originally developed under the name **EXchess**, development began in the late 1990s and the engine was actively maintained and released through early 2017.  After a long hiatus, the project was restarted in 2026 with significant new features developed in collaboration with [Claude Code](https://claude.ai/claude-code) (Anthropic) and renamed to Leaf.

---

## Quick Start

```sh
# Build with NNUE evaluation, net embedded in the binary (from engine/run/)
cd engine/run/
perl comp.pl myversion NNUE=1 NNUE_EMBED=1 NNUE_NET=nn-leaf-260414.nnue

# Run (auto-detects UCI, xboard, or interactive CLI)
./Leaf_vmyversion
```

**`NNUE_EMBED=1` is the canonical form for distribution** — the network is compiled into the binary, so a released engine carries no external `.nnue` and cannot be run against a mismatched one.  Only the opening book (`main_bk.dat`) ships alongside it.

For development you may prefer to load the net from disk, which lets you swap networks without recompiling:

```sh
perl comp.pl myversion NNUE=1 NNUE_NET=nn-leaf-260414.nnue
```

In that form the `.nnue` file must be in the same directory as the binary.

See [Build Options](#build-options) for all compile flags.

---

## History

EXchess by Dan Homan (now Leaf) was first released to the public in early 1998 and was one of a handful of serious open-source engines of that era.  Over two decades of on-and-off development produced a series of increasingly capable versions — from the early v2/v3 series through the v6 and v7 lines released in 2011–2017.

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

Leaf includes a complete **TDLeaf(λ)** reinforcement learning system that trains all NNUE layers from self-play games — FC weights, the 46 MB feature transformer, and PSQT weights (the sole trainable material channel).  The long-term goal is for Leaf to develop its own network, tuned to its own search, entirely through self-play.

Key features:
- PV leaf scores as the TD signal with full NNUE backpropagation
- Sparse FT/PSQT updates (only active feature rows touched per position)
- Adam optimizer with per-section learning rates and persistent momentum across sessions (`.tdleaf.bin` v12 format)
- Pure-PSQT material representation — a single trainable material channel, loss-anchored, with no gauge-anchoring machinery needed (an earlier dense second material channel and its anchoring mechanism were tried and later fully removed; see `engine/docs/history/TRAINING_HISTORY.md`)
- Works under both xboard/CECP (protocol results) and UCI (in-engine self-adjudication of game outcomes); the default actor/learner path plays in-engine self-play with exact in-process results
- Single-writer `.tdleaf.bin` persistence via direct atomic write (the actor/learner learner is the sole optimizer)
- Automated hybrid-loop iterations (online generation + offline consolidation + gauntlet, chainable via `--continue`) via `scripts/train.py`, which defaults to the actor/learner self-play split (`scripts/selfplay_run.py`)

Build with `NNUE=1 TDLEAF=1`.  See [`engine/docs/TRAINING.md`](engine/docs/TRAINING.md) for the full algorithm, hyperparameter reference, and training workflow.

### Offline Training & the Hybrid Loop

Online learning is complemented by an **offline supervised consolidation** mode: quiet positions are harvested from games the engine has already played (extracted from PGNs, or dumped directly by the engine during play via `TDLEAF_DUMP_TSV`) and trained with multi-epoch, shuffled, all-layer gradient descent on a λ-blend of game outcome and search score.  Together the two modes form the **hybrid loop** — online self-play generates games and learns as it goes; offline training extracts the full information content of those games; the consolidated net re-enters online play to generate better games.  Training targets use a TD(λ)-style **distance-decayed result weight** — the game outcome carries weight `0.985^(N−ply)` and the position's own eval takes the rest — so late positions learn from the result and early positions from the bootstrap.

Measured on the longest from-scratch chain to date (`m260720`, seven chained iterations, 2.5M self-play games from a freshly-seeded net — random feature-transformer weights, with nothing baked in but the classical material values in PSQT): the consolidated network **passed the classical hand-crafted eval**, scoring **+17 ± 11 Elo** against it over 1000 games at 3+0.05 — starting from −404 at the 100k-game mark.  Every game was generated by the engine playing itself from an opening-position book; no external networks and no imported game data were used at any point.  Each iteration contributes roughly +24 to +95 Elo against that fixed foreign anchor.  Consolidation uses single-process, within-batch thread parallelism (`--bt-threads`); `scripts/train.py` drives one-command iterations, chainable with `--continue`.

A per-iteration finding worth knowing before tuning: the online phase reliably *loses* Elo in its own right (−17 to −32) while the iteration as a whole gains — its product is the corpus, not its own strength.  The full research record, including the knobs that were tried and rejected, is in [`engine/docs/Online_Learning_Investigation.md`](engine/docs/Online_Learning_Investigation.md).

See [`engine/docs/TRAINING.md`](engine/docs/TRAINING.md) for the corpus format, trainer reference, and hybrid-loop workflow.

### Chess960 / Fischer Random

Full Chess960 support in both UCI and xboard protocols.  UCI_Chess960 castling notation is handled by boundary translation at the I/O layer, leaving the search and move execution untouched.

### LeafGUI

**LeafGUI** is a cross-platform Flutter chess GUI included in the `gui/` directory.  It provides a graphical interface for playing against Leaf (or any UCI engine), watching engine-vs-engine matches, and analyzing positions.

Key features: Chess960 support with engine capability detection, engine registry with persistent storage, engine-vs-engine mode with dual output, multiple time controls, move list navigation, FEN copy/load, and per-engine skill level adjustment.

```sh
# One-command release build: Flutter app + embedded-net engine + book + LICENSE
./scripts/build_release.sh

# Or step by step, from gui/
flutter pub get && flutter build macos --release
./bundle_engine.sh
```

`bundle_engine.sh` builds the engine with `NNUE_EMBED=1` and is the single source of truth for how the engine gets into the `.app`; `scripts/build_release.sh` drives the Flutter build around it.

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
perl comp.pl <version> NNUE=1 NNUE_NET=nn-leaf-260414.nnue

# NNUE with net embedded in binary
perl comp.pl <version> NNUE=1 NNUE_EMBED=1 NNUE_NET=nn-leaf-260414.nnue

# NNUE + TDLeaf(λ) training
perl comp.pl <version> NNUE=1 TDLEAF=1

# Skip interactive overwrite prompt
perl comp.pl <version> NNUE=1 OVERWRITE
```

| Flag | Effect |
|------|--------|
| `NNUE=1` | Enable NNUE evaluation |
| `NNUE_NET=<file>` | Override default network file (`nn-leaf-260414.nnue`) |
| `NNUE_EMBED=1` | Embed `.nnue` into binary via incbin (requires `NNUE=1` + `NNUE_NET`) |
| `TDLEAF=1` | Enable TDLeaf(λ) learning (requires `NNUE=1`) |
| `TDLEAF_READONLY=1` | Load trained weights but skip updates |
| `MATERIAL_ONLY=1` | `score_pos()` returns raw material balance only |
| `OVERWRITE` | Skip overwrite prompt |
| `NATIVE=1` | Tune for the build machine (`-march=native`, or `-mcpu=native` on arm64). Fast but **non-portable — never use for a distributed binary**. Default is portable: `-march=x86-64-v3` on x86-64 targets, and untuned on macOS (measured identical in NPS and node count on Apple Silicon) |
| `CXX=<compiler>` | Compiler to invoke (default `g++`). Set to a cross compiler, e.g. `CXX=x86_64-w64-mingw32-g++` |
| `WINDOWS=1` | Target Windows: appends `.exe`, defines `MINGW=1`, links `-static` (one self-contained exe, no MinGW DLLs). Auto-detected when building under MSYS2/Cygwin |
| `STATIC=1` | Static-link libstdc++/libgcc (Linux; implied on Windows) so release binaries run on older distros |
| `MACOS_MIN=<ver>` | macOS deployment target (default `11.0`, the first macOS with Apple Silicon). **Do not remove the default** — without it clang stamps the binary with the build host's OS version and it refuses to launch on anything older |
| `KNOWLEDGE=<N>` | Compile-time default for the Skill level (1–100, default 100 = full strength) |

Builds are **portable by default** — the flags above only need changing to cross-compile
or to deliberately trade portability for speed on a machine you control.

```sh
# Portable Linux x86-64 release binary
perl comp.pl 1.0 NNUE=1 NNUE_EMBED=1 NNUE_NET=<net>.nnue STATIC=1

# Windows x86-64, cross-compiled from macOS or Linux with MinGW-w64
perl comp.pl 1.0 NNUE=1 NNUE_EMBED=1 NNUE_NET=<net>.nnue \
    WINDOWS=1 CXX=x86_64-w64-mingw32-g++
```

The network file must be in the same directory as the binary (unless `NNUE_EMBED=1`).  The opening book (`main_bk.dat`) must also be present.

---

## Running

Leaf auto-detects UCI, xboard/CECP, or interactive CLI from the first command on stdin.  Point any compatible GUI at the binary, or run directly:

```sh
cd engine/run/
./Leaf_v<version>
```

### Engine Matches

Matches between engines use [fastchess](https://github.com/Disservin/fastchess) by default ([cutechess-cli](https://github.com/cutechess/cutechess) available via `--driver=cutechess`):

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

The recommended training workflow is `scripts/train.py` (run from `engine/learn/`), which drives one full hybrid-loop iteration — online generation with corpus dumping → offline consolidation → gauntlet — in a single command, and defaults to the actor/learner self-play split:

```sh
cd engine/learn/
python3 train.py --tag iter2 --games 400000 --depth 8 \
    --state <consolidated>.tdleaf.bin --gauntlet Leaf_vclassic_eval
```

For a standalone actor/learner generation run (no offline consolidation), use `scripts/selfplay_run.py`.  (The older interactive `training_run.py` manager has been archived under `scripts/older/`.)  See [`engine/docs/TRAINING.md`](engine/docs/TRAINING.md) and [`engine/docs/SCRIPT_USE.md`](engine/docs/SCRIPT_USE.md).

---

## Directory Layout

```
engine/
  src/          C++ source code (unity build via Leaf.cc)
  docs/         Documentation (NNUE.md, TRAINING.md, SCRIPT_USE.md, TODO.md,
                Online_Learning_Investigation.md, change_log.txt, history/)
  scripts/      Python automation scripts
  run/          Compiled binaries + runtime data (opening book)
  learn/        Training artifacts (.nnue, .tdleaf.bin, PGN)
gui/            LeafGUI Flutter chess GUI
logos/           Shared logo assets
tools/           Third-party tools (cutechess, BayesElo, external engines)
testing/         Test suites and opening books
archives/        Historical EXchess source snapshots
```

---

## License

**GNU General Public License, version 3 or later.**  This covers everything in this
repository authored here — the Leaf engine (`engine/`), its Python tooling
(`engine/scripts/`), and the LeafGUI Flutter application (`gui/`).  See
[`LICENSE`](LICENSE) for the full text.

Third-party components keep their own licences and are not relicensed here: the
Flutter packages LeafGUI depends on (`bishop`, `squares`, `square_bishop`,
`flutter_riverpod`) are MIT, and the tools under `tools/` (cutechess, fastchess,
BayesElo, external engines) carry their own terms.  All are GPL-compatible for
distribution alongside Leaf.

---

## Acknowledgements

- Classical search and evaluation by **Daniel C. Homan** (1997–2017, 2026–present)
- NNUE architecture and network statistics from the [Stockfish](https://stockfishchess.org) project (GPL v3)
- NNUE implementation, TDLeaf(λ) learning system, and 2026 restart developed in collaboration with **[Claude Code](https://claude.ai/claude-code)** (Anthropic)
- [Chessprogramming wiki](https://www.chessprogramming.org) for algorithm references
