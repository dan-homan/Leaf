# Leaf Script Reference

All Python scripts live in `scripts/`.  One-off experiment drivers — the arms
behind specific published results — live in `scripts/arms/` with their own
README; they are reproduction handles, not tooling.  Symlinks in `run/` and `learn/` allow
them to be invoked in-place from those directories, which is the normal workflow
since engines, `.nnue` files, and `.tdleaf.bin` files live there.

| Script | Description |
| --- | --- |
| [`train.py`](#trainpy) | One-command hybrid-loop iteration: online self-play generation, offline threaded batch training, and gauntlet rating, with `--continue` chaining across iterations. |
| [`selfplay_run.py`](#selfplay_runpy) | Actor/learner self-play driver: N frozen actors emit `.tdg` trajectories, one learner owns the optimizer.  The generation engine `train.py` wraps. |
| [`match.py`](#matchpy) | Runs head-to-head or gauntlet matches between engines via fastchess/cutechess, interactively or from the CLI. |
| [`make_training_epd.py`](#make_training_epdpy) | Generates a combined FRC + Polyglot-book opening EPD file for TDLeaf training. |
| [`compare_nnue_learning.py`](#compare_nnue_learningpy) | Visualises NNUE weight changes between a baseline `.nnue` and a trained `.tdleaf.bin`. |
| [`psqt_decomp.py`](#psqt_decomppy) | Decomposes a net's PSQT into material and positional parts on real positions (and the static into PSQT vs FC); tracks net composition across nets; writes the `--psqt-noise` material reference. |
| [`verify_fc0.py`](#verify_fc0py) | Debugging tool that recomputes FC0 outputs from a manually specified input vector to verify the forward pass. |
| [`bayeselo_ratings.py`](#bayeselo_ratingspy) | Computes a Bayesian Elo rating list for all players in a PGN file. |
| [`pgn_dedup.py`](#pgn_deduppy) | Removes duplicate games from one or more PGN files. |
| [`merge_tdleaf.py`](#merge_tdleafpy) | Merges multiple `.tdleaf.bin` files with count-weighted averaging. |
| [`pgn_winrate.py`](#pgn_winratepy) | Analyses win/draw/loss rates per N-game window for one player in a PGN file. |
| [`pgn_elo_progress.py`](#pgn_elo_progresspy) | Tracks Elo progress over training by running bayeselo on fixed-size PGN windows. |
| [`reset_adam.py`](#reset_adampy) | Zeroes or decays the Adam optimizer state in a `.tdleaf.bin` file. |
| [`extract_positions.py`](#extract_positionspy) | Extracts a per-position parquet dataset from self-play PGNs for calibration analysis. |
| [`analyze_calibration.py`](#analyze_calibrationpy) | Calibrates sigmoid temperature K and lambda decay from the parquet produced by `extract_positions.py`. |
| [`extract_quiet_positions.py`](#extract_quiet_positionspy) | Builds an offline-training TSV position corpus from existing self-play PGNs. |
| [`diff_tdleaf_checkpoints.py`](#diff_tdleaf_checkpointspy) | Diffs two `.tdleaf.bin` checkpoints section by section to monitor training drift. |
| `analyze_tdleaf.py` | Parameter coverage and quantisation diagnostics for a `.tdleaf.bin` + `.nnue` pair. |
| `analyze_fc0_passthrough.py` | Per-row FC0 drift analysis, isolating the passthrough row (output 15) from the 15 regular rows. |
| `bucket_phase_analysis.py` | Per-material-bucket phase analysis of a seed → post-online → final chain (first used in `history/Online_Learning_Investigation.md` Part 3). |
| `label_quality_by_bucket.py` | Per-material-bucket label quality from a hybrid-loop corpus TSV. |
| `distill_alignment.py` | Control analysis: does consolidating the *seed* on the hybrid corpus move it toward the online endpoint? |
| `engine_discovery.py` | Shared library (not a CLI): engine discovery/resolution and the interactive picker used by `match.py`. |
| [`older/training_run.py`](#oldertraining_runpy) | **Archived.** Interactive TDLeaf(λ) training-run manager (legacy UCI-pair workflow); superseded by `train.py` / `selfplay_run.py`. |
| [`older/migrate_legacy_work.py`](#oldermigrate_legacy_workpy) | One-time migration of pre-`train.py` training iterations into the current per-run archive layout. |

The six unlinked analysis scripts near the bottom of the table are research
one-offs with no section below — run them with `--help`, and see
`docs/history/Online_Learning_Investigation.md` for the analyses that motivated them.

---

## train.py

One command per hybrid-loop iteration: promote a consolidated state → online
self-play generation with leaf/root corpus dumping → checkpoint → assemble
corpus → threaded single-process offline training → best-epoch promotion →
gauntlet with an Elo table.  Non-interactive; run from `learn/`.  Generation is
always the **actor/learner split** (frozen actors + one learner, via
`selfplay_run.py`) — there is no generation-mode flag.  Helper binaries
(`Leaf_vbt`, `Leaf_vtrain_hl_a`) are auto-compiled.  See `TRAINING.md` for the
concepts and the manual runbook it encodes.

```sh
# Full iteration: generate 400k d8 games, consolidate (settled gen-3+ recipe:
# pure λ-return defaults, td_λ decay is the knob of record), rate every epoch
# as it completes, then the final full gauntlet
python3 train.py --tag iter3 --games 400000 --depth 8 \
    --state tdL10F10x6_ep4.tdleaf.bin --recompile \
    --bt-K 220 --bt-threads 8 \
    --gauntlet-epochs --gauntlet Leaf_vtdL10F10x6-ep4 Leaf_vclassic_eval

# Chained iteration: --continue reads iter3_final.json and defaults
# --net/--state/--gauntlet-anchors from it, tracking cumulative_games
python3 train.py --tag iter4 --continue iter3 --games 1000000 --depth 8 \
    --bt-K 220 --bt-threads 8 --gauntlet-epochs

# Consolidate-only on existing corpora (e.g. a hyperparameter arm on the same dumps)
python3 train.py --tag arm1 --skip-online \
    --bt-K 220 \
    $(for f in iter2_work/iter2.*.tsv; do echo --corpus $f; done) \
    --gauntlet Leaf_vbtsp-final Leaf_vclassic_eval

# Generate-only (games + corpora, no offline training)
python3 train.py --tag gen3 --games 200000 --depth 8 --skip-train
```

**Artifacts** (named by `--tag`), always flat in `learn/`: `<tag>_final.nnue`
(compile rating binaries from this; with `--gauntlet-epochs` this is the ladder's
best epoch, else the last epoch), `<tag>_final.tdleaf.bin` (seeds the next
iteration; pairs with the ORIGINAL base `.nnue`), `<tag>_final.json` (sidecar:
cumulative games, gauntlet anchors, epoch-ladder/gauntlet results — read by
`--continue`), and `Leaf_v<tag>-final` (compiled rating binary — always built,
even with no gauntlet opponents, so it's ready as a future `--continue` anchor;
never left resident in `run/`).

**`<tag>_work/`** is a permanent per-run archive — never deleted on a successful
run. It keeps `corpus.tsv` (gzip'd), the online-generation and final-gauntlet
PGNs (online one gzip'd), and `train/train.log` (with the epoch-ladder and
gauntlet tables appended); it prunes raw per-shard dumps, non-winning epoch
`.nnue` files, epoch-ladder PGNs, and epoch rating binaries. `--keep-epoch-states`
additionally keeps every epoch's `.tdleaf.bin`; `--keep-work` disables all
pruning for one run. A failed run's `<tag>_work/` is never touched. See
`TRAINING.md` for the full artifact-lifecycle reference.

### Options

| Flag                                         | Default                                                | Description                                                  |
| -------------------------------------------- | ------------------------------------------------------ | ------------------------------------------------------------ |
| `--tag NAME`                                 | required                                               | Iteration name; prefixes all artifacts                       |
| `--net FILE`                                 | `nn-fresh-260628.nnue`, or from `--continue`'s sidecar | Base `.nnue` in `learn/` (never changes across iterations)   |
| `--continue PREV_TAG`                        | —                                                      | Chain from a prior run: read `learn/PREV_TAG_final.json` and default `--net`/`--state`/`--gauntlet-anchors` from it, track `cumulative_games` across the chain, and auto-add `Leaf_v<PREV_TAG>-final` to the final gauntlet |
| `--init-nnue [material\|classical\|noprior]` | off                                                    | Initialise a fresh `--net` (+ companion `.tdleaf.bin`) before generating — turns the run into a start-to-finish iteration. Bare flag = `material`. Fails if `--net` already exists |
| `--state FILE`                               | keep live, or from `--continue`'s sidecar              | `.tdleaf.bin` to promote to the live state (backed up + hash-checked against the base) |
| `--skip-online`                              | off                                                    | Consolidate-only; train on `--corpus` files                  |
| `--games N`                                  | 400000                                                 | Games to generate                                            |
| `--depth N`                                  | 8                                                      | Fixed search depth for generation                            |
| `--nodes N`                                  | 0 (off)                                                | Node budget per move for generation.  `--depth` becomes the **floor**, so the search is never shallower than the fixed-depth run it replaces.  Adapts effort like a clock (extends on a failing-low root, halves on a singular reply) but deterministic.  `--depth 8 --nodes 4000` → mean d8.87, min d8, at ~1.85× the wall clock — see `docs/TRAINING.md` |
| `--lr-scale K`                               | 1.0                                                    | Uniform multiplier on every **online** TDLeaf step during generation (all six weight categories).  The offline phase is unaffected — that is `--bt-lr`.  Recorded in the sidecar as `lr_scale`; **not** inherited through `--continue`.  `K=0` does not mean "no generation": actors are frozen either way, so it means generating from a net that stops drifting (`history/Online_Learning_Investigation.md` 7.9.5) |
| `--psqt-noise FRAC` / `--psqt-opponent` / `--psqt-noise-ref FILE` | 0 (off) / `anti` / auto | PSQT hypothesis play for the actors (see `selfplay_run.py` below).  Without `--psqt-noise-ref` the material reference is built from the `--continue` parent's final `.nnue` and root corpus into `<tag>_work/psqt_ref.txt` (~75 s).  Recorded in the sidecar as `psqt_noise` / `psqt_opponent` |
| `--concurrency N`                            | 9                                                      | Concurrent games                                             |
| `--hash N`                                   | 128                                                    | Per-actor hash size (MB) for generation.  16 MB is ~25% faster at depth 8 but measured **+8.9 ± 11.4 Elo weaker at fixed depth** (`history/Online_Learning_Investigation.md` 7.5), so the default reverted to 128.  See `docs/history/Generation_Throughput.md` |
| `--openings FILE`                            | `training_openings.epd`                                | Opening set (FRC)                                            |
| `--no-gen-pgn`                               | off (PGN is **on**)                                    | Skip the generation PGN.  On by default: the corpus is quiet-gated and the `.tdg` stream is consumed, so the PGN is the only complete record of the games played.  Lands as `<tag>_work/<tag>_gen.pgn.gz` (~304 MB per 300k games) |
| `--games-per-actor N`                        | 1000                                                   | Actor respawn cadence / weight-refresh interval for the actor/learner split |
| `--no-repeat`                                | deprecated no-op                                       | Kept for backward compatibility; the actor/learner split plays each opening once (striped across actors), so there is no fastchess pairing to suppress |
| `--dedup-corpus`                             | **always on** (flag = deprecated no-op)                | Corpus assembly always drops duplicate rows (identical in every field except gid). Duplicate games straddle the trainer's by-game train/val split (different gids), so training on them both overfits and leaks validation; frozen deterministic pairs are the worst case (one unique game per opening) |
| `--quiet-cp N`                               | 1000                                                   | `TDLEAF_DUMP_QUIET_CP` for the dump — effectively open, since every row carries its `gate` and the width is chosen at assembly by `--bt-quiet-cp`.  Pass 60 to reproduce the historical corpora |
| `--bt-quiet-cp CP`                           | 60                                                     | Quiet gate applied at corpus **assembly**: keep a row only when `|cp - gate| <= CP`.  This is the width that actually trains.  Budget and quotas count only surviving rows (~57% of a wide dump), so the re-cut shrinks the corpus rather than silently shrinking the training set.  A2: 60/120/200 flat, ungated −28 Elo.  0 = train on the dump as-is |
| `--bt-rescore [FRAC]`                        | off                                                    | Retarget every root label to its **PV leaf, re-evaluated on the weights training will start from**, before corpus assembly.  The stored label is the generator's search score and goes stale as the net passes the generator; the leaf sits ~8 ply down the PV, so `eval_now(leaf) − eval_now(root)` still carries the search's verdict.  Bare flag = full retarget; `FRAC` blends `cp' = (1−f)·cp + f·eval_now(leaf)`.  Needs `--bt-rows root`, and every source must carry leaf rows.  **Measured null on a mature chain** (−2.5 ± 21.0, `history/Online_Learning_Investigation.md` 7.7) where the bootstrap was saturated — untested below saturation |
| `--skip-train`                               | off                                                    | Generate-only                                                |
| `--corpus TSV`                               | —                                                      | Extra corpus file(s) for training (repeatable)               |
| `--corpus-window N`                          | `0`                                                    | Dilute this run's dump with the archived `corpus.tsv.gz` of up to N prior iterations from the `--continue` chain, holding the **total** row count fixed (`--corpus-rows`) so epoch cost is unchanged. Worth ~45 Elo — row-matched arms differing only in game diversity scored +112.9 (500k games) vs +148.7 (2.5M games) vs the classical anchor (`history/Offline_Learning_Investigation.md` Part 2). `0` disables it (pre-A1 behaviour). Needs `--continue` |  Windows on each prior leg's **raw dumps** when they exist (kept since 2026-09-06), falling back to its assembled `corpus.tsv.gz` otherwise — the assembly of a leg that itself used a window already contains older legs, so windowing on it double-counts them.  A warning is logged when a fallback happens with >1 other source.
| `--corpus-rows N`                            | `0` (auto)                                             | Total row budget, split evenly across this run's dump and each window corpus (capped at each source's size, leftovers redistributed). `0` = match this run's own dump row count, so the window changes *which* games the rows come from, not how many. With no window and no explicit budget, nothing is thinned |
| `--corpus-weight source\|game`                | `source`                                               | How the row budget splits across sources.  `source` gives every **iteration** an equal share; `game` splits proportionally so every **game** carries equal weight.  Identical when the legs are the same size (the whole m260720 chain); they diverge sharply otherwise — a 300k-game leg windowed with two 100k-game legs gives the *fresh* games 30 rows/game against 91 for the stale ones under `source`.  Neither is universally right, and neither changes how many distinct games reach the corpus — only their relative weight |
| `--corpus-window-max-stale ELO`              | `0` (off)                                              | Drop window corpora whose **generator** rates more than ELO below the freshest generator in the window. A corpus is labelled by its generator — the *previous* iteration's promoted net — so a stale one drags the student back through the eval-bootstrap term. The generator Elo of every source is logged either way |
| `--bt-threads N`                             | 8                                                      | Worker threads for within-batch gradient compute (single process; synchronous data parallelism, identical to 1 thread up to float summation order — see `TRAINING.md`) |
| `--epochs N`                                 | 2                                                      | Training epochs                                              |
| `--bt-lr X`                                  | 1.0                                                   | LR scale on all category LRs                                 |
| `--bt-lambda X`                              | 1.0                                                    | Outcome-weight ceiling in the decayed blend target (`w = λ_eff·td_λ^(N−ply)`).  Default 1.0 = pure λ-return, the settled gen-3+ recipe: `--bt-td-lambda` is the knob of record; this stays a dormant scale knob (decouples overall outcome weight from decay shape across corpora with different ply-gap distributions) |
| `--bt-K X`                                   | 220                                                    | Sigmoid temperature                                          |
| `--bt-batch N`                               | 512                                                    | Positions per Adam step                                      |
| `--bt-leaf-lambda X`                         | = `--bt-lambda`                                        | Outcome-weight ceiling for depth-0 leaf rows (default follows the root λ, the recommended setting) |
| `--bt-td-lambda X`                           | trainer default (`TDLEAF_LAMBDA`)                      | Result decay per ply from the game end: `w = λ_eff·td_λ^(N−ply)`; `1.0` = flat blend |
| `--bt-loss-gamma X`                          | 1.0                                                    | Focal-loss exponent `(d·(1−d))^γ/K`: `1.0` = standard MSE (default), `0.0` = cross-entropy, `0.5` = between |
| `--bt-rows R`                                | `root`                                                 | Which corpus rows to train on: `root` (search-score rows, depth > 0), `leaf` (static-eval rows, depth 0), or `both`. **Default `root` since 2026-09-03** — at a fixed row budget root-only beat leaf-only by +35.6 ± 11.0 head-to-head and the natural mix by +46 Elo on the foreign anchor (`history/Offline_Learning_Investigation.md` Part 3). The budget and window quotas count only the selected type, so the filter shrinks the corpus, not the training set; the archived `corpus.tsv.gz` then holds only that type, so use `both` to keep the full mix |
| `--bt-diag`                                  | *(off)*                                                | Read-only diagnostic instead of training: evaluate every corpus row once and print, per row class (root/leaf) and binned by ply-from-end, piece count and `|label − net|`, how much better the corpus label predicts the game outcome than the net does (`ΔMSE_out`). `ΔMSE_out ≈ 0` means that channel is saturated. Writes nothing |
| `--bt-rescore FILE`                          | *(off)*                                                | Read-only: write this net's evaluation of every corpus row (white-POV cp, one per line) to FILE **in input order**, then exit.  Pastes line-for-line onto a parallel file — built to relabel `(root, leaf)` pairs with current weights.  Refuses `--bt-rows` / `--bt-quiet-cp`, which drop rows at load and would break alignment.  See `history/Online_Learning_Investigation.md` 7.7 |
| `--bt-quiet-cp N`                            | `0` (keep dump gate)                                   | Re-apply the quietness gate at training time: drop rows with `|cp − gate| > N`, using the corpus's `gate` column. Lets one widely-dumped corpus be re-cut to any narrower gate — a paired offline experiment over the same games instead of two divergent generation runs. Rows without a `gate` column are always kept. `0` keeps whatever the dump applied |
| `--gauntlet OPP …`                           | none                                                   | One-off opponent binaries in `learn/` for this run's final gauntlet (combined with `--gauntlet-anchors`) |
| `--gauntlet-anchors OPP …`                   | inherited from `--continue`'s sidecar, or empty        | Fixed opponent list carried forward automatically across a `--continue` chain (e.g. `Leaf_vclassic_eval`). Pass with no arguments to explicitly clear an inherited list |
| `--gauntlet-games N`                         | 1000                                                   | Games per opponent                                           |
| `--tc TC`                                    | `3+0.05`                                               | Gauntlet time control when `--gauntlet-depth` is not set, and always for the `tc-anchor` continuity match |
| `--gauntlet-depth N`                         | 0 (off)                                                | Run the final gauntlet at fixed depth N instead of `--tc`.  Isolates **eval quality**: fixed depth quotients out nps and nodes-to-depth, leaving how good the evaluation is at a fixed search.  Reproducible and load-immune, so concurrency is free.  At depth 8, ~10× cheaper per game than `3+0.05`: 4000 games ≈ 4.5 min and carry the same signal-to-noise the 1000-game `3+0.05` gauntlet gets in ~22 min.  ⚠️ Compressed, depth-dependent scale — never mix with a TC number (`Learning_Investigation.md` §1 P) |
| `--gauntlet-concurrency N`                   | 8 under `--tc`, `os.cpu_count()` under `--gauntlet-depth` | Concurrency for final-gauntlet matches |
| `--tc-anchor-games N`                        | 1000                                                   | Games for the `tc-anchor` continuity match: with `--gauntlet-depth`, `<tag>-final` is additionally rated at `--tc` against **every `--gauntlet-anchors` opponent** (usually `classic_eval`, but whatever the anchors are, and all of them if several).  This is what keeps the chain's recorded `3+0.05` ladder going and the only thing that catches a change improving eval-per-node while costing search speed.  Recorded in the sidecar as `tc_anchor_gauntlet` |
| `--no-tc-anchor`                             | off                                                    | Skip the `tc-anchor` continuity match |
| `--gauntlet-epochs`                          | off                                                    | Per-epoch ladder: rate each epoch snapshot vs the net as it stood before offline training, as soon as that epoch finishes; the trainer is paused (SIGSTOP/SIGCONT) during each match so games never contend with training for cores; prints an epoch table and promotes the best epoch as the final net |
| `--gauntlet-tdleaf`                          | off                                                    | Also rate the net as it *entered* offline training (post-online checkpoint, or the incoming state under `--skip-online`): saves it permanently to `learn/` as `<tag>-tdleaf.nnue` + `Leaf_v<tag>-tdleaf` and runs it through the same final gauntlet (opponents, `--gauntlet-games`, and the same `--gauntlet-depth`/`--tc` budget), recorded in the sidecar as `tdleaf_gauntlet`. Gives every run a same-conditions baseline so per-iteration deltas read directly. Under `TDLEAF_FREEZE` this net equals the incoming seed — the gauntlet still measures the baseline under *this run's* conditions |
| `--epoch-games N`                            | 1000                                                   | Games per epoch-ladder match                                 |
| `--epoch-tc TC`                              | `1+0.01`                                               | Epoch-ladder time control, used when `--epoch-depth` is not set |
| `--epoch-depth N`                            | 0 (off)                                                | Run the epoch ladder at fixed depth N instead of `--epoch-tc`.  Same instrument as `--gauntlet-depth`; the ladder is a pure within-family contrast, so it is the natural first thing to convert.  The trainer is still SIGSTOPped for each ladder match |
| `--epoch-concurrency N`                      | 8 under `--epoch-tc`, `os.cpu_count()` under `--epoch-depth` | Concurrency for epoch-ladder matches |
| `--no-final-gauntlet`                        | off                                                    | Skip the final full gauntlet matches (the rating binary is still always built) |
| `--force`                                    | off                                                    | Reuse an existing `<tag>_work` directory                     |
| `--recompile`                                | off                                                    | Force recompile of helper binaries                           |
| `--keep-epoch-states`                        | off                                                    | Keep every epoch's `.tdleaf.bin` in `<tag>_work/train/` (default: only the promoted epoch's state survives) |
| `--keep-work`                                | off                                                    | Skip all end-of-run pruning inside `<tag>_work/` for this run (raw dumps, non-winning epoch `.nnue`, epoch-ladder PGNs, epoch rating binaries all stay; `corpus.tsv` and the per-actor generation PGNs in `<tag>_work/pgn/` stay uncompressed and unconcatenated) |

---

## selfplay_run.py

Stage-1 actor/learner training driver: N frozen actor engines play internal
self-play (`--selfplay --traj-out`) emitting binary `.tdg` per-game
trajectories; one learner engine (`--learn-stream`) consumes them in arrival
order and owns the optimizer (single `.tdleaf.bin` writer).  Epoch-style
weight refresh: actors exit every `--games-per-actor` games and respawn,
reloading the learner's latest atomic state save.  Run from `learn/`:

```sh
python3 selfplay_run.py --binary Leaf_vtrain_hl_a --epd training_openings.epd \
    --actors 8 --depth 8 --games-per-actor 1000 --total-games 100000 \
    --traj-dir traj_run1
```

| Option | Default | Meaning |
|--------|---------|---------|
| `--binary NAME` | required | Training binary in cwd (needs `--selfplay`/`--learn-stream`) |
| `--epd FILE` | required | Opening book |
| `--actors N` | 4 | Frozen actor processes (TDLEAF_FREEZE forced) |
| `--depth D` | 8 | Fixed search depth |
| `--nodes N` | 0 (off) | Node budget per move; `--depth` becomes the **floor** ("at least this deep, then spend up to N nodes").  Reports extends/reductions per run so you can see whether adaptation is firing |
| `--hash N` | 128 | Per-process hash size (MB), passed to actors and learner alike.  Briefly defaulted to 16 for throughput; reverted after a fixed-depth A/B — see `docs/history/Online_Learning_Investigation.md` 7.5 |
| `--games-per-actor M` | 1000 | Actor respawn cadence = weight-refresh interval |
| `--total-games N` | required | Learner stop budget |
| `--traj-dir DIR` | `traj` | `.tdg` handoff dir (`STOP` sentinel stops early) |
| `--pgn-dir DIR` | off | Write the games played to PGN, one file per actor generation (`actor_<slot>_g<gen>.pgn`), with fastchess-shaped `{score/depth time}` comments.  <0.5% of actor wall clock, ~3.9 KB/game.  `train.py` turns this on by default |
| `--tdleaf-out PATH` | live companion | Learner state file |
| `--publish PATH` / `--publish-every G` | off / 512 | Bake a `.nnue` every G games |
| `--seed N` / `--delete-consumed` | 1 / archive | Shuffle seed base / delete instead of archive |
| `--refresh-scores` | off (**always pass it for online runs**) | Learner re-evaluates leaf statics with current weights at consume time. Without it, trajectory scores lag the learner by a refresh cycle and online TD drifts toward extreme decisiveness (d8t-3al2: 37%→12% draws by 40k games) |
| `--lr-scale K` | 1.0 | Multiplier on every online Adam/RMSProp step, mirroring `--bt-lr` offline.  Range-checked to `0 <= K <= 4`.  At 1.0 no flag reaches the learner and the step is bit-identical to previous behaviour |
| `--eval-noise CP` | 0 (off) | Actors only: zero-mean offset of sd CP on the static eval, keyed on the pawn structure (salt = seed + slot).  A diversity knob; measured not to add TD signal at 20–30 cp (`Learning_Investigation.md`).  Needs a binary built on or after 2026_09_23a for clean labels |
| `--psqt-noise FRAC` | 0 (off) | Actors only: scale the **positional** part of each (piece type, PSQT bucket) group of PSQT entries by (1+ε), ε ~ N(0, FRAC) clamped to [−1, 3·FRAC], material untouched.  One draw per refresh **generation**, shared by every actor.  Requires `TDLEAF_FREEZE` (forced on actors) and `--psqt-noise-ref`.  Labels stay clean: the actor's statics include the perturbation and `--refresh-scores` removes it exactly |
| `--psqt-opponent MODE` | `anti` | What the hypothesis plays against: `anti` = +ε vs −ε (antithetic; outcomes balanced by construction), `clean` = +ε vs the current net (side A alternates colour over paired openings), `same` = both sides hold it (only **enacts** the hypothesis — reproduction only).  Two-sided modes keep a second TT/score hash per actor; side A alternates colour game by game |
| `--pair-openings` | off | Two-sided modes only: play each opening twice, side A on either colour, for a pentanomial Elo of the hypothesis (`scripts/arms/psqt_noise_tderr.sh` uses it).  Off, a leg keeps exactly the opening sequence of an unperturbed leg with the same `--seed` — so seed-pairing a hypothesis leg against its sibling leaves the hypotheses as the only variable |
| `--psqt-noise-seed N` | `--seed` | Base seed for the draws; generation g uses N + g.  Vary it to change the hypotheses while keeping openings paired |
| `--psqt-noise-ref FILE` | — | Usage-weighted material reference from `psqt_decomp.py --write-ref` |
| `--adjudicate` | off | Enable actor resign/draw adjudication. Leave OFF for online learning — adjudication + learning is a runaway spiral (d8t-3al: 97% resignations, 27-ply games, dead net) |

The learner inherits the parent env (e.g. set `TDLEAF_DUMP_TSV` to have the
learner dump the corpus).  Bit-exactness property: with identical starting state
and env, `--learn-stream` over an online run's trajectories reproduces that run's
`.tdleaf.bin` byte-for-byte.  `train.py`'s generation phase wraps this driver with
the safe defaults.

## sample_corpus.py

Build a **game-stratified composite corpus** from several legs' raw TSV dumps —
for offline experiments that span more than one leg's generation window.

```sh
# four legs, 19 rows per game, root rows at the gate-60 cut
python3 sample_corpus.py --source m260916-{2,3,4,5}e6g_work \
    --rows root --quiet-cp 60 --quota 19 --out arms/corpus_base.tsv
```

Build from the **raw** `<tag>.<pid>.{root,leaf}.tsv.gz` dumps, not the assembled
`corpus.tsv`.  The latter carries only the row type and gate its leg happened to
use, so a wider gate or the leaf rows cannot be recovered from it; and its `gid`
column is **renumbered per leg from 0**, so game 5 of one leg and game 5 of
another are indistinguishable — which breaks both cross-leg game stratification
and root↔leaf pairing.

The sampling rule holds two properties at once that the obvious approaches
each destroy.  Uniform over rows gives an exact row count and weights games by
length (mean 77 rows/game at gate 60, sd 47, min 2, max 389 — a 195:1 spread);
uniform over games gives equal weight and a variable row count.  So: a fixed
**quota of rows per game**, drawn uniformly within the game.  At quota 19 about
98.8% of games fill it, so game weights are equal to ~1% and the total is
quota × games to the same tolerance; `--target-rows` then trims the residual
exactly.

| Option | Default | Meaning |
|---|---|---|
| `--source` | — | work directories (or explicit `.tsv[.gz]` files) |
| `--rows` | `root` | which raw dump to sample |
| `--quiet-cp` | 60 | load-time gate `\|cp − gate\|`, matching `--bt-quiet-cp` |
| `--quota` | 19 | rows drawn per game |
| `--games-per-source` | min across sources | so no leg dominates |
| `--target-rows` | natural total | exact total; for matching an earlier run |
| `--restrict-gids` | — | file (or corpus TSV) of game ids to restrict to |
| `--count-only` | off | pass 1 only; print `{games, eligible_rows}` as JSON |

⚠️ `--target-rows` is for trimming a **small** surplus.  Shedding a large one
flattens the quota past the short games and leaves only the long ones (at quota
100000 against a 3.9% target it kept 121 games of 1244).  To take a small
fraction of a corpus, lower the **quota** — `--count-only` sizes it — rather
than leaning on `--target-rows`.

Use `--restrict-gids` whenever an arm changes `--rows`: leaf rows survive in
games whose root rows the quiet gate removed entirely, so an unrestricted leaf
sample draws ~45% more games at correspondingly fewer rows each, and the
row-type contrast would carry a game-set difference inside it.

## calibrate_from_corpus.py

Fit the sigmoid temperature **K** and the trace decay **λ** straight from a
leg's raw root dump, split by **game stage** (material remaining).

```sh
cd engine/learn
python3 calibrate_from_corpus.py --source m260916-5e6g_work \
    --games 60000 --max-lag 60 --quiet-cp 60
```

Not to be confused with `analyze_calibration.py`, whose `--stage` means
*training* stage (net maturity), which needs a PGN-derived parquet, and whose
constants predate the current recipe (K=290, λ 0.8/0.5 per *record*).  This one
reads the raw `*.root.tsv.gz` dumps — `cp`, `result`, `ply`, `endply` and the
FEN are all there — and needs no intermediate file.

| Section | Measures |
|---|---|
| (1) | K by maximum likelihood, plus a reliability table, by material stack and by pawn count |
| (2A) | `corr(ev, outcome)` by plies-to-end — outcome informativeness |
| (2B) | `corr(ev_t, ev_{t+k})` by lag → λ.  **Position to position; the result never enters** |
| (2B2/B3/B4) | λ by material stack, by game ply, and the cross-tab that decides which one λ tracks |
| (3) | `Var(outcome − ev)` by material and by \|cp\|, the input to a reliability-based outcome weight |

**What is and isn't identified.**  K is a straight calibration and is identified.
**λ is not** — the outcome and the eval estimate the same unobserved value, so
choosing between them needs a bias/variance assumption; each measurement states
the assumption that turns it into a λ.  §5 of `Learning_Investigation.md` says
why (2B) is the right one: under the martingale property of a calibrated value
function, `corr(ev_t, ev_{t+k}) = sqrt(Var_t/Var_{t+k})`, so the decorrelation
*is* the rate new information arrives.

⚠️ **The results of this tool are not a guide to training hyperparameters.**
Seven arms derived from these fits all lost on the foreign anchor — see
`Learning_Investigation.md` §1 O and the §4 closure.  The measurements are real
facts about the corpus; the inference to a hyperparameter is refuted.

## run_consolidation_arms.py

Drives the **composite-corpus consolidation arms**: five one-epoch offline runs
from one seed, matched row-for-row, each rated against the seed and a foreign
anchor.  Answers whether a periodic consolidation round over several legs'
games beats another epoch over the newest leg alone.

```sh
cd engine/learn
python3 run_consolidation_arms.py --tag cons1 \
    --seed-net m260916.nnue --seed-state m260916-5e6g_final.tdleaf.bin \
    --sources m260916-{2,3,4,5}e6g_work
```

| arm | corpus | differs from `base` by |
|---|---|---|
| `null` | newest leg only, quota-sized to the same rows | — (it is the control) |
| `base` | four legs, quota 19 | game diversity + label age |
| `bout` | base's **exact** row file | `--bt-td-lambda 0.9925` (more outcome) |
| `bcp` | base's **exact** row file | `--bt-td-lambda 0.97` (more cp) |
| `leaf` | four legs, leaf rows, **same games** as base | row type |

`bout`/`bcp` reuse `base`'s row file byte for byte, so those three differ only
in the target and carry no sampling noise between them.  They also **bound**
what `--bt-rescore` could buy: if the target curve slopes toward the outcome,
stale cp labels are hurting and rescoring is worth its cost; if it peaks at or
below the default, cp labels are not the binding constraint.

⚠️ **`--seed-net` must be the chain's BASE `.nnue`** (the constant one, e.g.
`m260916.nnue`), never a baked `<tag>_final.nnue` export.  A `.tdleaf.bin`
records the content hash of the net it was trained against; hand the trainer
the wrong one and it **refuses the state with a warning and a zero exit**, then
trains from the base net with no learned weights and no Adam moments.  Every
arm still "works" and none of them means anything.  The driver runs each
binary once before use and dies on that warning, and on the matching
silent failure where a missing `.nnue` falls back to classical eval.

⚠️ **Time control.**  `--tc` (default `1+0.01`) applies to every arm and both
opponents, so the programme is internally consistent — but `classic_eval` is a
classical-eval engine with a different nps profile, so its column cannot be
compared against anchor Elos recorded at another TC (the chain's
`final_gauntlet` figures are `3+0.05`).  The paired column, against the seed, is
the primary reading and is unaffected.  A **fixed depth** is a third condition
again, not a TC at all: it moves the `classic_eval` gap by ~85 Elo and rescales
family gaps too (`Learning_Investigation.md` §1 P).

Everything is resumable: an arm whose `_ep1.nnue` exists is not retrained, a
match whose PGN exists is not replayed, and pass-1 corpus counts are cached.
`--only` runs a subset, `--corpus-only` stops after assembly.

## match.py

Run a head-to-head match or gauntlet between chess engines using a tournament
driver — **fastchess by default** (`--driver=cutechess` selects cutechess-cli
instead).  Supports Leaf binaries and external UCI engines (e.g. Stockfish,
placed in `tools/engines/<name>/`).  **Invoke from `run/` or `learn/`**
(symlinked into both) or `scripts/`.

`match.py` is the **rating / gauntlet tool only** — it is no longer a training
generator.  Training generation runs through `selfplay_run.py` (actor/learner),
which uses the engine's own in-process results, not UCI self-adjudication.
`tdleaf_self_adjudicate()` still exists for a learning binary played under UCI
outside training (GUI/analysis).  cutechess-cli supports
xboard as well, so the engine receives explicit `result` commands instead;
pass `--driver=cutechess --proto xboard` for that path.

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

# Multi-iteration gauntlet; engines restart between iterations
python3 match.py Leaf_vA Leaf_vB -n 500 -i 10

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
| `--driver` | `fastchess` | Tournament driver: `fastchess` or `cutechess` |
| `-n`, `--games` | 100 | Games per iteration per opponent |
| `-i`, `--iterations` | 1 | Iterations per opponent; engines restart between each |
| `-c`, `--concurrency` | cpu_count/2 | Simultaneous games |
| `-tc`, `--time-control` | `3+0.05`, or `inf` under a fixed budget | Time control for both engines (`moves/time+inc` or `time+inc`, seconds), or `inf` for no clock |
| `--tc1` / `--tc2` | (from `-tc`) | Override time control for engine1/engine2 only |
| `--proto` | `uci` | Protocol for both engines (`uci` or `xboard`; fastchess requires `uci` for both) |
| `--proto1` / `--proto2` | (from `--proto`) | Override protocol for engine1/engine2 only |
| `--pgn FILE` | — | Persistent PGN; all games appended across opponents/iterations |
| `--pgn-out FILE` | auto | Per-iteration PGN (default: `match_<e1>_vs_<e2>.pgn`) |
| `--fischer-random` | off | Chess960 starting positions |
| `--no-adjudication` | off | Disable score-based early adjudication (`-draw`/`-resign`); games run to a natural ending (mate/stalemate/repetition/50-move/insufficient material), capped by `-maxmoves 400`. Useful early in training when evals are noisy. |
| `--ponder` | off | Enable pondering (cutechess only; fastchess doesn't expose it — a warning is printed if combined with `--driver=fastchess`) |
| `--wait MS` | 0 | Milliseconds between games (a legacy throttle from the multi-writer era; not needed for gauntlets) |
| `--depth N` | — | Limit **both** engines to depth N: the fixed-budget match.  Reproducible and load-immune (0/20 positions differ across repeat runs, solo and under 16-way load), so concurrency can be the full core count without changing the result.  Implies `-tc inf`.  Measures **eval quality at a fixed search**, not playing strength — it quotients out both nps and nodes-to-depth, and its Elo is on a compressed, depth-dependent scale that must never be read against a time-control rating (`Learning_Investigation.md` §1 P) |
| `--nodes N` | — | Limit **both** engines to N nodes/move.  ⚠️ `go nodes` is **not reproducible** in the NNUE build — 5–8 of 20 positions differ across repeat runs, one changing the best move, with swings to 7× in node count under load (the classical binary is clean at 0/20).  Prefer `--depth`.  Implies `-tc inf` |
| `--depth1 N` / `--depth2 N` | — | Limit engine1/engine2 search to depth N (asymmetric form of `--depth`) |
| `--nodes1 N` / `--nodes2 N` | none | Limit each engine to N nodes/move (`go nodes N`).  Spends effort where the position needs it, unlike fixed depth — but see the `--nodes` reproducibility warning |
| `--openings FILE` | — | Openings file: `.epd`, `.pgn`, or `.bin` (polyglot book; fastchess doesn't support `.bin`) |
| `--no-repeat` | off | One game per round (`-rounds N`, no `-games 2 -repeat`): removes the driver's color-swapped duplicate pair per opening, at the cost of per-opening color balance; recommended for symmetric self-play.  Does **not** guarantee opening uniqueness by itself — fastchess cycles a shuffled book order, so openings recycle once total games exceed the book size. |
| `--noswap` | off | Pass `-noswap` to the driver; engine1 always plays white.  Off by default (correct for training). |
| `--error-log FILE` | — | Append driver stderr (and inherited engine stderr) to FILE.  Captures per-batch `[tdleaf step-clip]` telemetry from all concurrent training engines in one file; lines are atomic per `fprintf` (<4KB), so interleaving is line-granular. |
| `--stall-timeout SEC` | 600 | Kill the driver + engines if no output appears for this long (guards a known cutechess-cli 1.4.0 deadlock under high concurrency; exits with code 124 on stall) |
| `--option1` / `--option2 KEY=VALUE` | — | Pass a UCI/xboard option to engine1/engine2 (repeatable), e.g. `--option1 'UCI_Elo=2000'` |
| `--name1` / `--name2` | basename of engine | Override the display name in PGN/output |

When more than one opponent is supplied the script enters **gauntlet mode** and
prints a summary table (Opponent, Games, W, D, L, Score%, Elo diff) at the end.

### Protocol notes

The default protocol is **UCI**, and the default driver **fastchess** requires
UCI for both engines. Leaf auto-detects UCI, so no special flags are needed. To
run under xboard instead (e.g. for external xboard-only engines), pass `--driver=cutechess
--proto xboard` (or `--proto1`/`--proto2` for per-engine overrides).

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
`train.py` and `selfplay_run.py` default to `training_openings.epd` (in `learn/`) as
their opening set (`--openings`).

**Why random suffix moves?**  normbk02.bin at ply 8 converges to only ~2500 unique
positions due to transpositions.  Adding 1–2 random moves after each book/FRC leaf
explodes the unique count into the hundreds of thousands, preventing game replication
in training.  Use `--quiet-only` to restrict suffix moves to non-captures (keeps
material balanced; recommended).

`--random-suffix` accepts one or more values; each position draws its suffix length
uniformly at random from the set.  This applies to both book and FRC positions.

| Suffix | Book unique | FRC unique (per replicate) |
|--------|-------------|---------------------------|
| 0 | ~2,500 | 960 |
| 1 | ~60,000 | ~19,000 |
| 2 | ~1,000,000+ | ~300,000+ |
| 0 1 2 (mixed) | varies | varies |

**Book sampling** uses a two-phase approach for speed at large targets:
1. BFS enumerates all unique positions reachable through the book up to `--ply` depth
   (including shallower lines where the book runs out early) — fast, done once.
2. Weighted-samples from that leaf pool, applying suffix moves, with progress printed
   every 10 k positions and an early-stop saturation guard (`--saturation`).

Two sizing modes are available (mutually exclusive):

**Explicit** (`--frc-replicates` / `--book-positions`):
```sh
cd learn/

# Default: 960 FRC + 2000 book, no suffix (2,960 total)
python3 make_training_epd.py

# 2 quiet suffix moves, 21 FRC replicates + 80k book positions
python3 make_training_epd.py --frc-replicates 21 --book-positions 80000 \
    --random-suffix 2 --quiet-only

# Mixed suffix: each position gets 0, 1, or 2 moves chosen at random
python3 make_training_epd.py --book-positions 500000 --random-suffix 0 1 2
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
| `--random-suffix K [K ...]` | 0 | Suffix move count(s) per position.  Pass one value for a fixed length; pass multiple to draw at random per position (e.g. `--random-suffix 0 1 2 3`).  Applies to both book and FRC positions.  Greatly increases unique position count. |
| `--quiet-only` | off | Restrict random suffix moves to non-captures **and** filter positions by eval balance (see below) |
| `--eval-binary FILE` | `Leaf_vclassic_eval` in script dir | Leaf binary for eval filtering; only used with `--quiet-only` |
| `--eval-limit CP` | 50 | Discard positions where `\|score\| > CP` centipawns (default: 50 = 0.5 pawns); only used with `--quiet-only` |
| `--eval-depth N` | 10 | Search depth for eval filtering; only used with `--quiet-only` |
| `--eval-workers N` | cpu_count/2 | Parallel eval engine processes; only used with `--quiet-only` |
| `--ply N` | 8 | Ply depth for book BFS / walks |
| `--saturation F` | 0.002 | Stop generation when a full pool pass yields fewer than `pool_size × F` new positions; lower values allow more exhaustive search near saturation |
| `--output FILE` | `training_openings.epd` in script dir | Output EPD file |
| `--seed N` | 42 | Random seed for reproducibility |

Book positions are sampled by weighted pool draw (move probability ∝ Polyglot `weight`
field), then deduplicated.  With `--random-suffix`, each drawn leaf gets additional
random (or quiet) moves before deduplication, multiplying the unique count.
FRC replication without suffix preserves intentional duplicates (for position weighting);
with suffix, duplicates across replicates are silently dropped (rare).

**`--quiet-only` eval filter:** when a `Leaf_vclassic_eval` binary is present, every
generated position is scored at `--eval-depth` via xboard protocol and positions with
`|score| > --eval-limit` cp are discarded.  Chess960 castling rights are stripped from
the EPD before sending to the engine.  `--eval-workers` parallel engine processes run
concurrently (~75 pos/sec at depth 10 with 4 workers).  Compile the eval binary with
`perl comp.pl classic_eval OVERWRITE` from `run/` or `learn/` (no NNUE, no TDLEAF).  If the binary is
absent, a warning is printed and the filter is skipped.

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

Supports `.tdleaf.bin` versions 2–12.  Version 12 (pure-PSQT) removed the dense
piece-value channel and the v11 PSQT gauge-anchoring slot-means; earlier versions'
now-dropped fields are read and discarded to stay aligned.

Optional flags:

```sh
# Save pages to PNG files instead of displaying
python3 compare_nnue_learning.py baseline.nnue weights.tdleaf.bin --save out_prefix

# Include full FT weight arrays (slow; requires ~92 MB of memory per perspective)
python3 compare_nnue_learning.py baseline.nnue weights.tdleaf.bin --ft-weights
```

---

## psqt_decomp.py

Decompose a net's PSQT into **material** and **positional** parts, measured on
positions the net actually plays.  Each PSQT entry belongs to a (plane, PSQT
bucket) group; the group's **usage-weighted** mean over (king bucket, square) is
that piece's material value in that bucket, and each entry's deviation from it
is positional — split further into a king-independent part (per-square mean
over king buckets) and a king-dependent remainder.  Weighting by how often each
entry is active keeps unreachable and never-trained entries from distorting
anything.  Runs in ~10 s on 330k positions.

```sh
# One net, with the static split into PSQT vs FC (the TSV must be this net's dump)
python3 scripts/psqt_decomp.py learn/m260921-2.5e6g_final.nnue \
    --positions 'learn/tderr_noise/td_s0/dump.*.root.tsv' --fc

# Composition over time: several nets, on one common position set
python3 scripts/psqt_decomp.py learn/m260921-1e6g_final.nnue \
    learn/m260921-2e6g_final.nnue learn/m260921-2.5e6g_final.nnue \
    --positions 'learn/m260921-2.5e6g_work/*root.tsv*'

# Write the material reference the engine's --psqt-noise needs
python3 scripts/psqt_decomp.py learn/<net>_final.nnue \
    --positions '<that net's root dump>' --write-ref psqt_ref.txt
```

| Option | Default | Meaning |
|--------|---------|---------|
| `nets...` | required | One or more `.nnue` files |
| `--positions GLOB...` | required | Root-row TSVs (`TDLEAF_DUMP_TSV` / `train.py` corpus format, `.tsv` or `.tsv.gz`) |
| `--games N` | 2500 | Whole games sampled (consecutive plies are needed for the per-move numbers) |
| `--fc` | off | Also split the whole static into PSQT and FC.  Column 8 of a root row is the **dumping net's** root static, so this is valid only for the single net that dumped the TSV |
| `--detail` | single net: on | Per-(plane, bucket) table: uses, material cp, positional sd, king-independent sd, and the plain-mean error |
| `--write-ref FILE` | — | Write the 88-line usage-weighted material reference (raw PSQT units) for `--psqt-noise-ref` |

**Output.**  Per net: material per piece (usage-weighted over buckets); how far a
*plain* mean over reachable entries misses the usage-weighted material (the
error an in-engine perturbation would leak into piece values); and for each
component the sd across positions, per move, and per **quiet** move (material
unchanged — a proxy for the sibling-move differences that decide move choice).
With `--fc`, the quiet-move positional variance split between PSQT and FC and
their correlation.

**Reading (m260921-2.5e6g, 2026-09-24).**  PSQT positional sd 88 cp across
positions and 35 cp per quiet move; FC 78 cp per quiet move; the two
uncorrelated (−0.001), so quiet-move positional variance is **17% PSQT / 83%
FC**.  Positional spread per (piece, bucket) mostly 15–70 cp.  Plain-mean
material misses usage-weighted material by 18.5 cp rms (max 69), and Adam
update counts by 6.3 rms (max 35) — hence the reference file.

**Monitoring.**  Given several nets it reports each on the same positions, so
material drift, the growth of the positional part, and the king-dependent share
can be followed across a chain.  Without `--fc` every number is a pure function
of the weights and the position sample, valid for any net.

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
`TRAINING.md`).  Replays each game (python-chess, Chess960-aware,
multiprocessed, ~2,300 games/s) and emits one TSV record per QUIET position:
`fen  cp  result  ply  depth  gid  endply` — Shredder-FEN, search eval from the
move comment (white POV, cp), game result (white POV), ply, eval depth, a stable
game id so the trainer can split train/validation by game, and the game's true
final ply (distance base for the trainer's result decay).  Eval and outcome are
stored separately: the trainer's decayed λ-blend keeps λ, td_λ, and K as
training-time hyperparameters.  `ply`/`endply` here count game plies (every
half-move), matching the in-engine dump's own game-ply counting — see
"Ply units" in `TRAINING.md`.

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
FC0 passthrough-row mean, R/Q piece_val — see `TRAINING.md`).

```sh
# From learn/: compare consecutive checkpoints
python3 diff_tdleaf_checkpoints.py nn-fresh.tdleaf.bin-1e6g nn-fresh.tdleaf.bin-2e6g
```

Two positional arguments (old, new); no options.  Uses the
`compare_nnue_learning.py` reader (supports `.tdleaf.bin` v2–v12).

---

## older/training_run.py

> **ARCHIVED.**  This interactive manager predates the hybrid loop and drives
> the legacy `match.py`/fastchess UCI-pair generation path.  The supported
> workflow is `train.py` (which defaults to the actor/learner split); use
> `selfplay_run.py` for a standalone actor/learner run.  The reference below is
> kept for historical runs only.

Interactive TDLeaf(λ) training run manager.  **Invoke from `learn/`** so that
all working files (`.nnue`, `.tdleaf.bin`, `.games`, built binaries, PGN output)
land in `learn/`.

> **TDLeaf runs under UCI by default.**  All engines in `training_run.py` now
> run under `--proto uci`: the default driver (fastchess) is UCI-only, and
> Leaf's UCI loop triggers TDLeaf learning via self-adjudicated outcomes
> (`uci_finish_game()` → `tdleaf_self_adjudicate()` — terminal-position checks
> with a score-history fallback).  xboard/CECP remains supported (learning
> hooks also run from `make_move()` there, driven by the protocol `result`
> command) for `--driver=cutechess --proto xboard`, but is no longer required.

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

**Adam momentum persistence:** both Adam moment arrays (m and v) persist across
sessions in `.tdleaf.bin`, so the optimizer resumes with full directional momentum
rather than cold-starting — see `TRAINING.md`'s Adam Optimizer section for the
concurrent-writer merge rules and format detail.

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

| Prompt                   | Default               | Notes                                                        |
| ------------------------ | --------------------- | ------------------------------------------------------------ |
| Cycles                   | 0 (∞)                 | Number of train-validate cycles; 0 = run until Ctrl-C        |
| Validation games         | 200                   | Games per validation match                                   |
| LOS acceptance threshold | 70%                   | Candidate accepted if LOS ≥ this                             |
| Early-stop high          | 90%                   | Terminate validation early if LOS ≥ this (clear win)         |
| Early-stop low           | 10%                   | Terminate validation early if LOS ≤ this (clear loss)        |
| Validation TC            | (same as training TC) | Separate TC for the validation match                         |
| Games per cycle          | 5000                  | Training games per cycle (single iteration; no engine restart) |

Two eval-only binaries (`NNUE=1`, no `TDLEAF`) are compiled once at setup:
`eval_best` loads `<net>-best.nnue` and `eval_cand` loads `<net>-cand.nnue`.

**Why single iteration in loop mode**: multiple iterations restart the engine
processes between blocks, discarding Adam momentum and variance state.  A single
long iteration preserves the full optimiser state across the training block.

Ctrl-C exits cleanly: current weights are exported and a per-cycle result table
is printed.

---

## older/migrate_legacy_work.py

One-time backlog migration for training iterations produced before `train.py`'s
per-run archive model existed (the pre-rename `hybrid_loop.py`, and any early
`train.py` runs predating this script). For each `--tags` entry, in the order
given: reorganizes the tag's scattered flat-`learn/` artifacts (per-epoch rating
binaries, PGNs previously written flat instead of into `<tag>_work/`, checkpoint
backups) into the current archive layout and prunes them the same way a live run
would, then backfills a `<tag>_final.json` sidecar — reconstructed from whatever
PGNs/snapshots are still on disk — so `--continue` works on the net going forward.

**Defaults to a dry run.** Prints every action without touching anything; pass
`--apply` to actually execute. Always dry-run first and read the output before
`--apply` — this operates on irreplaceable training data.

```sh
cd learn/

# Review what would happen (default: dry run)
python3 ../scripts/migrate_legacy_work.py --tags \
    material_260708-1e5g material_260708-5e5g material_260708-1e6g \
    material_260708-2e6g material_260708-3e6g

# Then actually perform it
python3 ../scripts/migrate_legacy_work.py --tags \
    material_260708-1e5g material_260708-5e5g material_260708-1e6g \
    material_260708-2e6g material_260708-3e6g --apply
```

`bt_lr`/`bt_lambda`/`bt_K`/`bt_td_lambda` can't be recovered from on-disk
artifacts alone (the orchestrator's own hyperparameter log line went to the
terminal, not to any file) — they stay `null` in the reconstructed sidecar
unless `--transcript <file>` points at a saved copy of the original invocations,
in which case they're parsed from the actual flags used.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--tags TAG [TAG ...]` | required | Tags in chain order (each tag's parent = the previous one) |
| `--transcript FILE` | — | Saved copy of the original invocations, to recover `bt_lr`/`bt_lambda`/`bt_K`/`bt_td_lambda`/`gauntlet_anchors` precisely |
| `--keep-epoch-states` | off | Keep every epoch's `.tdleaf.bin` (same meaning as `train.py`'s flag) |
| `--apply` | off | Actually perform the migration (default: dry run only) |
