# Generation Throughput on Linux — why self-play scaled badly, and what fixed it

**Question asked (2026-09-03).**  Since generation moved from the fastchess
driver to internal self-play (the actor/learner split), per-game generation on
the Linux box has felt ~30–50% slower for large runs, while on the Macintosh
internal self-play is clearly *faster* than fastchess.  Small runs look fine.
Working hypothesis: memory conflict / paging between processors, which fastchess
avoided via `numactl`-style binding that internal self-play does not do.  Second
question: is the default hash too large for depth-8 generation?

**Answer, short form.**  The memory/NUMA hypothesis does not hold on this
machine — there is nothing to bind, and the multi-process slowdown reproduces
exactly in a workload with no memory traffic at all.  The hash hypothesis is
right, and it is worth ~25%: the engine wipes its hash tables once per game, so
the default 128 MB is a per-game tax paid 500,000 times per training iteration.

Test machine: AMD Ryzen 9 8940HX (16 physical cores / 32 threads, one socket),
31.9 GB RAM, `performance` governor, Linux 6.17.

---

## 1. NUMA / memory binding — ruled out

```
NUMA node(s):    1
NUMA node0 CPU(s): 0-31
$ numactl -H
numactl: command not found
```

Single NUMA node.  There is no second memory domain for a process to be placed
badly against, nothing for `numactl --membind` to express, and the tool is not
even installed — so whatever fastchess was doing here, it was not NUMA binding.
The machine does have two L3 domains (cores 0–7 and 8–15, 32 MB each, SMT
siblings at +16), but that is a cache-locality boundary, not a memory-affinity
one.

## 2. The multi-process slowdown is clock throttling, not memory contention

Actor-scaling sweep, depth 6, every actor playing the **identical** 20-game set
(same shuffle, stride 1, offset 0) so per-actor work is constant and wall-time
differences are pure contention.  Alongside it, `spin` — a pure-ALU,
cache-resident loop with no memory traffic and no syscalls at all:

| actors | self-play per-actor, rel. to N=1 | `spin` per-core, rel. to N=1 |
|--------|--------------------------------|------------------------------|
| 1      | 1.00                           | 1.00                         |
| 2      | 0.75                           | 0.74                         |
| 4      | 0.56                           | 0.55                         |
| 8      | 0.41                           | 0.43                         |
| 16     | 0.34                           | 0.41                         |

The curves are the same.  A workload that touches no memory cannot be losing
throughput to memory conflict, so neither is self-play: this is the CPU giving
back clock as more cores light up.  Measured effective frequency falls from
~3.5 GHz at one busy core to ~1.3–1.5 GHz all-core — expected for a laptop-class
`HX` part, and paid identically by fastchess.  **This is not a regression and
not fixable in software.**  It does mean aggregate throughput on this box tops
out around 5–6× the single-actor rate no matter how many actors are launched.

## 3. The hash tables were being reallocated once per game

`selfplay_new_game_reset()` (and the UCI `ucinewgame` handler) called
`set_hash_size(engine_cfg.hash_size)` to start a clean game.  `set_hash_size()`
is `close_hash()` + `open_hash()` — `free()` all four tables, `aligned_alloc()`
them again at *exactly the same size*, and walk every entry writing its initial
value.  With the 128 MB default that is a `munmap`/`mmap` round trip plus a
full page-fault-in and 128 MB of init writes, **every game**, in every actor.

Two independent costs hide in there: the allocator round trip, and the init
writes over the whole table.  Both were measured.  Depth 8, 30 games/actor, all
arms playing the identical game set:

| arm | 1 actor (games/s) | 14 actors (aggregate games/s) |
|-----|-------------------|-------------------------------|
| realloc / 128 MB *(as shipped)* | 0.949 | 5.013 |
| realloc / 16 MB                 | 1.123 (+18%) | 6.302 (+26%) |
| clear / 128 MB                  | 1.020 (+7%)  | 5.166 (+3%)  |
| clear / 16 MB                   | 1.220 (**+29%**) | 6.279 (**+25%**) |

Reading this: the allocator round trip is worth ~7% at one actor and ~3% at
fourteen; the **table size** is worth ~25% at either.  At 14 actors everything
is clock-throttled, which compresses the allocator effect but not the cost of
writing 128 MB.  The two fixes compose, and the size is the larger one.

Hash size sweep at 14 actors, depth 6, for the shape:

| hash | aggregate games/s |
|------|-------------------|
| 128 MB | 16.3 |
| 64 MB  | 16.9 |
| 32 MB  | 17.1 |
| 16 MB  | 19.4 |

### Is 16 MB big enough?

> **⚠️ RETRACTED (2026-09-05).  The measurement below cannot answer the question
> it was asked.**  It compares *self-play* W/D/L and termination mix across hash
> sizes — but in self-play both sides run the same net at the same hash, so a
> uniform strength change cancels exactly and is invisible by construction.  It
> shows only that 16 MB does not change the *character* of self-play games.
>
> Measured properly — same binary and net on both sides, **fixed depth 8** so the
> smaller table's speed advantage cannot pay for a quality loss, `Hash=128` vs
> `Hash=16` over 2000 games — **Hash 128 reads +8.9 ± 11.4 Elo** (W/L/D
> 669/618/713).  That is 0.8σ, consistent with zero, but the point estimate
> favours the larger table, and hash size genuinely reaches search quality at
> fixed depth: TT scores feed move ordering, null-move, LMR and singular
> extensions.
>
> **The default reverted to 128 MB.**  16 MB buys ~25% generation throughput,
> but the corpora that generation produces currently measure worth *zero*
> (`Online_Learning_Investigation.md` 7.3), so the throughput is not worth
> leaving a confound in every comparison against the pre-`6e6g` chain.
> `clear_hash()` below is unaffected and stays — it is bit-identical output.
> Full account in `Online_Learning_Investigation.md` 7.5.

The original (insufficient) evidence, kept for the record — 600 games per arm,
same openings, patched binary:

| | 128 MB | 16 MB |
|---|---|---|
| W / D / L | 193 / 220 / 187 | 193 / 229 / 178 |
| draw rate | 36.7% | 38.2% |
| mate / 3-rep / 50-move / material | 380 / 149 / 24 / 45 | 371 / 151 / 29 / 47 |

The draw rates differ by ~0.5σ and both sit inside the healthy 35–40% band the
online-stability canary wants.  The termination mix is flat.  (A 40-game pilot
appeared to show a mate→repetition drift; it did not survive n=600.)  16 MB is
0.75 × 16 MB / 64 B ≈ 196k buckets ≈ 786k entries, comfortably more than a
depth-8 search touches.  Raise `--hash` if generating at much greater depth.

### What changed

- `clear_hash()` (new, `hash.cpp`) wipes the tables in place.  `open_hash()`
  now allocates and delegates to it, so there is one copy of the
  initialisation and the post-condition is identical to before.
- `selfplay_new_game_reset()` and the UCI `ucinewgame` handler call
  `clear_hash()` instead of `set_hash_size()`.  Geometry never changes between
  games, so the realloc bought nothing.
- `selfplay_run.py --hash` (default **128**, briefly 16 — see the retraction
  above) passes `hash <MB>` to actors and learner; `train.py --hash` forwards
  it.  `hash` must precede
  `--selfplay`/`--learn-stream` on the command line — `main()` dispatches to the
  driver as soon as it sees that flag and never returns to the argument loop.

Requires a rebuild of the training binary (`train.py --recompile`).

## 4. The learner is a hard throughput ceiling — and it is depth-dependent

This is the one structural difference from the fastchess era, and worth knowing
even though it is not currently binding.  Under fastchess, learning happened
inside each engine process and therefore scaled with concurrency.  Under the
actor/learner split, **one** single-threaded learner consumes every trajectory.
Measured on a fixed bank of 477 depth-6 trajectories:

| competing CPU load | learner games/s |
|--------------------|-----------------|
| idle               | 17.9            |
| 4 busy cores       | 12.3            |
| 8 busy cores       | 13.0            |

Its work per game is *depth-independent* — under `--refresh-scores` it rebuilds
two accumulators per record (leaf and root) and backpropagates; it never
searches.  So the ceiling sits at ~13–18 games/s regardless of generation depth,
while actor supply falls steeply with depth:

- **Depth 6:** 14 actors supply ~16 games/s > the ceiling.  The learner binds.
  Confirmed directly: a live pipeline run at 8 actors sat at 449 of 500 pending
  `.tdg` files with the learner consuming 9.2 games/s — actors were sleeping in
  the backpressure loop (`usleep(500000)`, so 0.5 s granularity) rather than
  playing.
- **Depth 8:** 14 actors supply ~5–6 games/s, well under the ceiling.  Not
  binding.
- **Depth 10:** further under.  Not binding.

So for the current depth-8/10 recipe the learner is not the problem, and no
change was made.  If generation ever moves back to shallow depth, or actor count
rises substantially, the learner becomes the constraint and the fix is on the
learner side (parallelise the rebuild, or shard the trajectory stream), not more
actors.  Note also that raising `--actors` past the point where the learner
saturates makes things *worse*, because the extra actors take clock away from
the learner that is setting the rate.

## 5. What this does and does not explain

- **Ruled out:** NUMA placement, memory-bandwidth conflict, paging.  The box has
  one memory domain and free RAM throughout; a zero-memory workload shows the
  same scaling curve.
- **Inherent:** all-core clock throttling.  Costs ~60% of ideal scaling at 16
  actors and applies to any driver, fastchess included.
- **Fixed:** the per-game hash reallocation.  The oversized-table half was also
  worth +25% at depth 8 / 14 actors, but the table size was **reverted to 128 MB**
  on 2026-09-05 after a fixed-depth A/B put 16 MB at −8.9 ± 11.4 Elo of search
  quality (see the retraction in §3).  What remains is `clear_hash()`: ~+7% at
  one actor, ~+3% at fourteen, bit-identical output.
- **Understood but not currently binding:** the single-learner ceiling, which
  *is* a genuine structural difference from fastchess and would dominate at
  depth 6.

### Reading the historical actor logs

Per-actor `games/s` lines in `<tag>_work/traj/actor_*.log` are the cleanest
record, but they are not a work rate: an actor blocked on learner backpressure
reports a low rate while doing nothing.  Mean game length is flat at ~149 plies
across every depth-8 iteration (`m260720-2.2e6g` … `-5e6g`) and 158.6 at depth
10, so game length is not a confound; depth and blocking are.
