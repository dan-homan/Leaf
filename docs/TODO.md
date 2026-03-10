# Epoch TODO

Planned investigations, improvements, and open questions.

---

## TDLeaf(λ) Training

### Learning rate tuning
The current LR scales are initial guesses and have not been systematically tuned.
PSQT in particular appears to learn very slowly — `NNUE_PSQT_LR_SCALE` (currently 1000)
may need to be raised further, or the grad computation for PSQT reviewed.  FT bias
(`NNUE_FT_BIAS_LR_SCALE = 10`) and FC bias (`NNUE_FC_BIAS_LR_SCALE = 1000`) scales
similarly benefit from empirical testing.  A short grid search varying each scale
independently across 500–1000-game runs would establish good defaults.

### Epoch-based replay for TDLeaf training

The current TDLeaf implementation is fully online: weights are updated after
each game and the game record is discarded.  A cheap alternative is to store
each game's leaf positions after self-play and make K additional passes through
them, recomputing the NNUE evaluation at each stored leaf with the updated
weights and applying the TD error on each pass.  The search is not re-run —
only the forward and backward NNUE passes at the fixed stored positions.

**Cost estimate (Flavor B — fixed leaves, recompute NNUE only):**
- Per-position: ~1–5 μs forward pass + ~10–15 μs backward pass
- Per-game (~80 leaf positions): ~1 ms per epoch
- 4,000 games × 10 epochs ≈ 40 seconds of additional compute — negligible

**Memory:**
- Storing board states only (accumulator recomputed each pass): ~160 MB for 4,000 games
- Storing accumulators (2 × 1024 int16 per position): ~1.3 GB for 4,000 games

**Caveats:**
- TDLeaf is on-policy: the stored leaf positions were found by an earlier
  version of the network.  Re-using them with updated weights is an off-policy
  approximation.  Empirically (cf. experience replay in DQN) this works well
  for small K, but degrades as the stored positions become stale relative to
  the current network.  K = 2–5 is a reasonable starting range; beyond ~5
  epochs over the same data instability is likely.
- The leaf *positions* themselves would change if the search were re-run with
  updated weights.  Flavor B accepts this as an approximation in exchange for
  the large reduction in compute cost vs. re-running the full search (Flavor A),
  which would cost the same as replaying all games × number of epochs.

**Recommendation:** implement Flavor B with K configurable at build or run
time.  Run a 500-game ablation comparing K=1 (current), K=2, and K=4 on the
same starting network and measure Elo gain per wall-clock hour.

### Bias initialisation
FC biases and FT biases are currently initialised from the SF15.1 distribution (random
N(μ,σ)).  Consider initialising all biases to 0 and letting TDLeaf learn them from
scratch — this removes any dependence on the SF15.1 starting point and may converge to
better values, especially for a freshly-initialised network.

### Search parameter tuning
The search's pruning parameters (null-move margins, futility thresholds, aspiration
windows, LMR reduction tables) were tuned for the classical eval.  The NNUE eval has a
different score distribution and may benefit from re-tuning these constants.  CLOP or
a self-play tournament with systematic variation would be the appropriate approach.

---

## NNUE Infrastructure

### Pawn hash under NNUE
The classical eval stores pawn structure scores in a pawn hash table.  The NNUE eval
bypasses classical eval entirely, so `pawn hash hits` is always 0 in NNUE mode and the
pawn hash memory (≈19 MB) is wasted.  Disabling or shrinking it at build time when
`NNUE=1` would recover that memory (no effect on playing strength).

### search.par revision
Review whether `search.par` is still the right mechanism for runtime configuration —
consider what parameters are still relevant, whether the file format should be updated,
and how it interacts with xboard/cutechess invocation.

### Linux compilation
Verify that Epoch compiles and runs correctly on Linux.  The NEON SIMD optimisations
(`vdotq_s32` etc.) are ARM-specific and will need `#ifdef` guards or x86 fallbacks
(SSE/AVX via `_mm_dp_epi8` or plain scalar) for x86-64 Linux builds.

### Multi-thread accumulator correctness
The SMP search allocates one `ts_thread_data` per thread, each with its own
`search_node n[MAXD+1]` stack including per-node accumulators.  Each thread's root
accumulator is independently initialised.  Thread interactions have not been tested
under NNUE; correctness is expected but unverified with `THREADS > 1`.
