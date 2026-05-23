#!/usr/bin/env python3
"""analyze_tdleaf.py — diagnose parameter coverage and quantisation in .tdleaf.bin + .nnue.

Reports per-section statistics for the FP32 shadow-weights file (.tdleaf.bin) and,
optionally, cross-checks the int8/int16 .nnue exported from it.

Sections inspected:
    FT weights (sparse, 22 528 × 1 024 fp32 in int16 scale)
    FT biases  (1 024 fp32 in int16 scale)
    PSQT      (sparse, 22 528 × 8 fp32 in int32 scale)
    FC0/FC1/FC2 (8 layer stacks; FP32 in int8 scale)
    Dense piece values (6 fp32)

Per section we emit:
  - total params
  - "untouched" count (cnt == 0)        — never received a gradient step
  - "v-zero" count                       — Adam second-moment never accumulated
  - "near init" count                    — |w_now - w_init| < tol
  - "at quant clamp" count               — |w| >= int8/int16 max after rounding
  - magnitude percentiles (10/50/90/99/max)

Usage:
  python3 analyze_tdleaf.py --bin <bin> [--nnue <nnue>] [--init-nnue <init>] [--out report.md]
  python3 analyze_tdleaf.py --learn-dir /Users/.../engine/learn --out report.md  # full sweep

Run from repo root or with PYTHONPATH=engine/scripts.
"""
from __future__ import annotations

import argparse
import os
import sys
import textwrap
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Reuse the parser from merge_tdleaf.py.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from merge_tdleaf import (  # type: ignore  # noqa: E402
    TDLeafFile, NNUEFile,
    FT_INPUTS, HALF_DIMS, PSQT_BKTS,
    L0_SIZE, L0_INPUT, L1_SIZE, L1_PADDED, L2_PADDED, LAYER_STACKS,
    TDLEAF_SCALE,
)

# ---------------------------------------------------------------------------
# HalfKAv2_hm slot classification (matches mean-centering in nnue_training.cpp)
# ---------------------------------------------------------------------------
SLOT_NAMES = [
    "own pawn", "opp pawn",
    "own knight", "opp knight",
    "own bishop", "opp bishop",
    "own rook",   "opp rook",
    "own queen",  "opp queen",
    "king",                                 # slot 10 (covers king feature)
]

def fi_to_slot(fi: int) -> int:
    """Mean-centering slot index: 11 buckets of 64 features each (within a king bucket)."""
    return (fi % 704) // 64

def fi_to_king_bucket(fi: int) -> int:
    return fi // 704

def fi_to_sq(fi: int) -> int:
    """Square index 0..63 within the slot."""
    return fi % 64

def fi_unreachable_pawn(fi: int) -> bool:
    """Pawn on rank 1 or rank 8 — never legally occurs."""
    slot = fi_to_slot(fi)
    if slot not in (0, 1):  # own pawn or opp pawn
        return False
    sq = fi_to_sq(fi)
    rank = sq // 8
    return rank == 0 or rank == 7

# ---------------------------------------------------------------------------
# Pretty-printing helpers
# ---------------------------------------------------------------------------
def pct(n: int, total: int) -> str:
    if total == 0:
        return "—"
    return f"{100.0 * n / total:6.2f}%"

def magnitude_stats(arr: np.ndarray) -> dict:
    """Magnitude percentile and clamp stats."""
    if arr.size == 0:
        return {"min": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0,
                "mean_abs": 0.0, "std": 0.0}
    a = np.abs(arr).astype(np.float64)
    return {
        "min": float(a.min()),
        "p10": float(np.percentile(a, 10)),
        "p50": float(np.percentile(a, 50)),
        "p90": float(np.percentile(a, 90)),
        "p99": float(np.percentile(a, 99)),
        "max": float(a.max()),
        "mean_abs": float(a.mean()),
        "std": float(arr.std()),
    }

def fmt_stats(s: dict, suffix: str = "") -> str:
    return (f"|w| mean={s['mean_abs']:.3g} std={s['std']:.3g}  "
            f"p10={s['p10']:.3g} p50={s['p50']:.3g} p90={s['p90']:.3g} p99={s['p99']:.3g} "
            f"max={s['max']:.3g}{suffix}")

# ---------------------------------------------------------------------------
# Section analysers
# ---------------------------------------------------------------------------

@dataclass
class SectionStats:
    name: str
    total: int = 0
    untouched: int = 0            # cnt == 0
    v_zero: int = 0               # Adam v == 0
    m_zero: int = 0               # Adam m == 0 (FC layers)
    near_zero: int = 0            # |w| < small_tol (in *target* scale)
    at_clamp: int = 0             # |w| >= quant max (post-quantisation)
    mag_stats: dict = field(default_factory=dict)
    init_match: Optional[int] = None  # vs init: count exactly equal
    near_init: Optional[int] = None   # |w - w_init| < tol
    notes: list = field(default_factory=list)

def stats_dense_fc(w_scaled_f32: np.ndarray, c_u32: np.ndarray,
                   v_f32: np.ndarray, m_f32: Optional[np.ndarray],
                   name: str, init_w_i8: Optional[np.ndarray] = None) -> SectionStats:
    """FC bias/weight section.

    w_scaled_f32 is at int8 scale × TDLEAF_SCALE (=128); divide to get the actual int8-scaled fp32.
    """
    w = w_scaled_f32 / TDLEAF_SCALE  # float in int8 weight scale
    n = w.size
    s = SectionStats(name=name, total=n)
    s.untouched = int((c_u32 == 0).sum())
    s.v_zero    = int((v_f32 == 0.0).sum())
    if m_f32 is not None:
        s.m_zero = int((m_f32 == 0.0).sum())
    s.near_zero = int((np.abs(w) < 0.5).sum())          # would quantise to 0 (int8)
    rounded = np.round(w).astype(np.int32)
    s.at_clamp  = int(((rounded <= -127) | (rounded >= 127)).sum())
    s.mag_stats = magnitude_stats(w)
    if init_w_i8 is not None:
        s.init_match = int((rounded == init_w_i8.astype(np.int32)).sum())
        s.near_init  = int((np.abs(w - init_w_i8.astype(np.float32)) < 1e-3).sum())
    return s

def stats_ft_bias(td: TDLeafFile, nn_init: Optional[NNUEFile]) -> SectionStats:
    """FT biases: int16-scale FP32 in .tdleaf.bin, compared against int16 in .nnue."""
    w = td.ft_bias_w / TDLEAF_SCALE
    s = SectionStats(name="FT biases (1024)", total=HALF_DIMS)
    s.untouched = int((td.ft_bias_c == 0).sum())
    s.v_zero    = int((td.v_ft_bias == 0.0).sum())
    s.m_zero    = int((td.m_ft_bias == 0.0).sum())
    s.near_zero = int((np.abs(w) < 0.5).sum())
    rounded = np.clip(np.round(w), -32767, 32767).astype(np.int32)
    s.at_clamp  = int((np.abs(rounded) >= 32767).sum())
    s.mag_stats = magnitude_stats(w)
    if nn_init is not None:
        init = nn_init.ft_biases.astype(np.int32)
        s.init_match = int((rounded == init).sum())
        s.near_init  = int((np.abs(rounded - init) < 1).sum())
    return s

def stats_ft_weights(td: TDLeafFile, nn_init: Optional[NNUEFile]) -> tuple[SectionStats, dict]:
    """Sparse FT weights, also report per-slot dirty coverage."""
    total = FT_INPUTS * HALF_DIMS
    s = SectionStats(name=f"FT weights (sparse: {FT_INPUTS}×{HALF_DIMS} = {total:,})", total=total)

    n_dirty_rows = len(td.ft_rows)
    n_dead_rows  = FT_INPUTS - n_dirty_rows

    # Coverage by slot/king-bucket.
    dirty_set   = set(td.ft_rows.keys())
    slot_total  = [0] * 11
    slot_dirty  = [0] * 11
    slot_unreach= [0] * 11
    kb_total    = [0] * 32
    kb_dirty    = [0] * 32
    for fi in range(FT_INPUTS):
        slot = fi_to_slot(fi)
        slot_total[slot] += 1
        if fi in dirty_set:
            slot_dirty[slot] += 1
        if fi_unreachable_pawn(fi):
            slot_unreach[slot] += 1
        kb = fi_to_king_bucket(fi)
        kb_total[kb] += 1
        if fi in dirty_set:
            kb_dirty[kb] += 1

    # Per-cell statistics across dirty rows.
    if n_dirty_rows > 0:
        n = n_dirty_rows * HALF_DIMS
        all_w  = np.empty(n, dtype=np.float32)
        all_c  = np.empty(n, dtype=np.uint32)
        all_v  = np.empty(n, dtype=np.float32)
        # ft_v_rows can have a subset of dirty rows (rows with v_nonzero only).
        idx = 0
        v_missing_rows = 0
        for fi, (fw, fc, _ps_w, _ps_c) in td.ft_rows.items():
            all_w[idx:idx+HALF_DIMS] = fw / TDLEAF_SCALE     # int16 scale
            all_c[idx:idx+HALF_DIMS] = fc
            if fi in td.ft_v_rows:
                all_v[idx:idx+HALF_DIMS] = td.ft_v_rows[fi]
            else:
                all_v[idx:idx+HALF_DIMS] = 0.0
                v_missing_rows += 1
            idx += HALF_DIMS

        # Among the dirty rows: how many individual weights were touched?
        s.untouched = int((all_c == 0).sum())  # weights inside a dirty row whose own count is 0
        s.v_zero    = int((all_v == 0.0).sum())
        s.near_zero = int((np.abs(all_w) < 0.5).sum())
        rounded = np.clip(np.round(all_w), -32767, 32767).astype(np.int32)
        s.at_clamp = int((np.abs(rounded) >= 32767).sum())
        s.mag_stats = magnitude_stats(all_w)

        if nn_init is not None:
            # Compare against init values from .nnue (int16). Index back into ft_weights.
            ft_init = nn_init.ft_weights
            equal = 0
            close = 0
            ii = 0
            for fi in td.ft_rows.keys():
                base = fi * HALF_DIMS
                init_row = ft_init[base:base+HALF_DIMS].astype(np.int32)
                cur = rounded[ii:ii+HALF_DIMS]
                equal += int((cur == init_row).sum())
                close += int((np.abs(cur - init_row) <= 1).sum())
                ii += HALF_DIMS
            s.init_match = equal
            s.near_init  = close
        # Add the implicitly-untouched rows: every weight in a non-dirty row equals init.
        if n_dead_rows > 0:
            s.untouched += n_dead_rows * HALF_DIMS
            s.v_zero    += n_dead_rows * HALF_DIMS
            s.near_zero += n_dead_rows * HALF_DIMS   # init values are tiny, ~He
            # All non-dirty rows trivially match init.
            if s.init_match is not None:
                s.init_match += n_dead_rows * HALF_DIMS
                s.near_init  += n_dead_rows * HALF_DIMS
        s.notes.append(f"dirty rows: {n_dirty_rows:,} / {FT_INPUTS:,} "
                       f"({100.0*n_dirty_rows/FT_INPUTS:.2f}%);  "
                       f"dead rows: {n_dead_rows:,}")
        s.notes.append(f"ft_v_rows: {len(td.ft_v_rows):,}; "
                       f"dirty rows without v: {v_missing_rows:,}")
    else:
        s.notes.append("no dirty rows — file appears uninitialised")

    coverage = {
        "n_dirty_rows": n_dirty_rows,
        "n_dead_rows": n_dead_rows,
        "slot_total": slot_total,
        "slot_dirty": slot_dirty,
        "slot_unreach_pawn": slot_unreach,
        "kb_total": kb_total,
        "kb_dirty": kb_dirty,
    }
    return s, coverage

def stats_psqt(td: TDLeafFile, nn_init: Optional[NNUEFile]) -> SectionStats:
    """Sparse PSQT — 22 528 × 8 in int32 scale (FP32 / 128)."""
    n_dirty_rows = len(td.ft_rows)  # PSQT dirty == FT dirty (set is shared)
    total = FT_INPUTS * PSQT_BKTS
    s = SectionStats(name=f"PSQT (sparse: {FT_INPUTS}×{PSQT_BKTS} = {total:,})", total=total)
    n_dead_rows = FT_INPUTS - n_dirty_rows
    if n_dirty_rows == 0:
        s.untouched = total
        return s

    n = n_dirty_rows * PSQT_BKTS
    all_w = np.empty(n, dtype=np.float32)
    all_c = np.empty(n, dtype=np.uint32)
    all_v = np.empty(n, dtype=np.float32)
    all_m = np.empty(n, dtype=np.float32)
    idx = 0
    for fi, (_ftw, _ftc, ps_w, ps_c) in td.ft_rows.items():
        all_w[idx:idx+PSQT_BKTS] = ps_w / TDLEAF_SCALE      # int32 scale
        all_c[idx:idx+PSQT_BKTS] = ps_c
        if fi in td.psqt_v_rows:
            all_v[idx:idx+PSQT_BKTS] = td.psqt_v_rows[fi]
        if fi in td.psqt_m_rows:
            all_m[idx:idx+PSQT_BKTS] = td.psqt_m_rows[fi]
        idx += PSQT_BKTS

    s.untouched = int((all_c == 0).sum()) + n_dead_rows * PSQT_BKTS
    s.v_zero    = int((all_v == 0.0).sum()) + n_dead_rows * PSQT_BKTS
    s.m_zero    = int((all_m == 0.0).sum()) + n_dead_rows * PSQT_BKTS
    s.near_zero = int((np.abs(all_w) < 0.5).sum())
    s.mag_stats = magnitude_stats(all_w)
    if nn_init is not None:
        init_full = nn_init.psqt_weights.reshape(FT_INPUTS, PSQT_BKTS)
        equal = 0; close = 0
        ii = 0
        for fi in td.ft_rows.keys():
            init_row = init_full[fi].astype(np.float64)
            cur = np.round(all_w[ii:ii+PSQT_BKTS]).astype(np.float64)
            equal += int((cur == init_row).sum())
            close += int((np.abs(cur - init_row) < 1).sum())
            ii += PSQT_BKTS
        s.init_match = equal + n_dead_rows * PSQT_BKTS
        s.near_init  = close + n_dead_rows * PSQT_BKTS
    s.notes.append(f"dirty rows: {n_dirty_rows:,}; dead rows: {n_dead_rows:,}")
    return s

def stats_piece_val(td: TDLeafFile) -> SectionStats:
    s = SectionStats(name="Dense piece values (6)", total=6)
    pv = td.piece_val_w / TDLEAF_SCALE  # in int16/PSQT scale (centipawns × 5776/100 scaling lives in PSQT)
    s.untouched = int((td.piece_val_c == 0).sum())
    s.v_zero    = int((td.v_piece_val == 0.0).sum())
    s.m_zero    = int((td.m_piece_val == 0.0).sum())
    s.near_zero = int((np.abs(pv) < 0.5).sum())
    # piece_val is clamped >= 0 in the engine: flag any that have nonzero count
    # but landed exactly at 0 (clamp-stuck).
    pv_at_clamp = int(((td.piece_val_c > 0) & (np.round(pv) == 0)).sum())
    s.at_clamp = pv_at_clamp
    s.mag_stats = magnitude_stats(pv)
    s.notes.append("piece_val_w (scaled): " + ", ".join(
        f"{name}={float(pv[i]):.2f}" for i, name in
        enumerate(["P","N","B","R","Q","K"])))
    s.notes.append("counts: " + ", ".join(
        f"{name}={int(td.piece_val_c[i])}" for i, name in
        enumerate(["P","N","B","R","Q","K"])))
    return s

# ---------------------------------------------------------------------------
# .tdleaf.bin ↔ .nnue cross-check
# ---------------------------------------------------------------------------

def cross_check(td: TDLeafFile, nn: NNUEFile, scope: str = "exported") -> list[str]:
    """Compare the float shadow in td against the int8/int16 quantised values in nn.

    For each section, report whether nn[i] == clamp(round(td[i] / TDLEAF_SCALE)) where applicable.
    `scope` is just a label for the report.
    """
    lines = []
    lines.append(f"### Cross-check ({scope}) — .tdleaf.bin float shadow ↔ .nnue quantised\n")

    # FT biases (int16)
    shadow = td.ft_bias_w / TDLEAF_SCALE
    rounded = np.clip(np.round(shadow), -32767, 32767).astype(np.int32)
    diff = (rounded != nn.ft_biases.astype(np.int32))
    if diff.any():
        d = (rounded - nn.ft_biases.astype(np.int32))[diff]
        lines.append(f"- FT biases: **{int(diff.sum())} / {HALF_DIMS} mismatch** "
                     f"(max |Δ|={int(np.abs(d).max())}, mean |Δ|={float(np.abs(d).mean()):.2g})")
    else:
        lines.append(f"- FT biases: all 1024 match ✓")

    # FT weights (int16, sparse) — only inspect dirty rows.
    ft_init = nn.ft_weights
    mism = 0
    max_abs = 0
    for fi, (fw, _fc, _pw, _pc) in td.ft_rows.items():
        shadow = fw / TDLEAF_SCALE
        rounded = np.clip(np.round(shadow), -32767, 32767).astype(np.int32)
        base = fi * HALF_DIMS
        nnv = ft_init[base:base+HALF_DIMS].astype(np.int32)
        d = rounded - nnv
        mism += int((d != 0).sum())
        if d.size:
            m = int(np.abs(d).max())
            if m > max_abs:
                max_abs = m
    n_dirty = len(td.ft_rows) * HALF_DIMS
    if mism == 0:
        lines.append(f"- FT weights (dirty rows: {n_dirty:,}): all match ✓")
    else:
        lines.append(f"- FT weights: **{mism:,} / {n_dirty:,} dirty cells mismatch** "
                     f"(max |Δ|={max_abs}).  Indicates the .nnue was written from a "
                     f"different shadow state, OR quantisation saturated.")

    # PSQT (int32, sparse)
    psqt_init = nn.psqt_weights.reshape(FT_INPUTS, PSQT_BKTS)
    psqt_mism = 0; psqt_max = 0
    for fi, (_fw, _fc, pw, _pc) in td.ft_rows.items():
        shadow = pw / TDLEAF_SCALE
        rounded = np.round(shadow).astype(np.int64)
        d = rounded - psqt_init[fi].astype(np.int64)
        psqt_mism += int((d != 0).sum())
        if d.size:
            m = int(np.abs(d).max())
            if m > psqt_max:
                psqt_max = m
    n_dirty_psqt = len(td.ft_rows) * PSQT_BKTS
    if psqt_mism == 0:
        lines.append(f"- PSQT (dirty rows: {n_dirty_psqt:,}): all match ✓")
    else:
        lines.append(f"- PSQT: **{psqt_mism:,} / {n_dirty_psqt:,} dirty cells mismatch** "
                     f"(max |Δ|={psqt_max}).  N.B. the .nnue PSQT bakes piece_val into it; "
                     f"a non-zero piece_val explains a ±round(piece_val/2) systematic shift.")

    # FC layers (int8)
    fc_total = 0; fc_mism = 0; fc_max = 0
    for s in range(LAYER_STACKS):
        for (shadow_arr, init_arr, label) in [
            (td.fc[s].l0_weight_w, nn.l0_weights[s], f"FC0[{s}]"),
            (td.fc[s].l1_weight_w, nn.l1_weights[s], f"FC1[{s}]"),
            (td.fc[s].l2_weight_w, nn.l2_weights[s], f"FC2[{s}]"),
        ]:
            shadow = shadow_arr / TDLEAF_SCALE
            r = np.clip(np.round(shadow), -127, 127).astype(np.int32)
            d = r - init_arr.astype(np.int32)
            n_mism = int((d != 0).sum())
            fc_total += shadow.size
            fc_mism += n_mism
            if d.size:
                m = int(np.abs(d).max())
                if m > fc_max:
                    fc_max = m
    if fc_mism == 0:
        lines.append(f"- FC weights (all 8 stacks): all {fc_total:,} match ✓")
    else:
        lines.append(f"- FC weights: **{fc_mism:,} / {fc_total:,} mismatch** "
                     f"(max |Δ|={fc_max}).  Probably exported .nnue lags the current shadow.")

    return lines

# ---------------------------------------------------------------------------
# Single-file analysis
# ---------------------------------------------------------------------------

def analyze_single(bin_path: str, nnue_path: Optional[str],
                   init_nnue_path: Optional[str]) -> str:
    out = []
    out.append(f"# tdleaf.bin analysis — `{os.path.basename(bin_path)}`\n")
    sz = os.path.getsize(bin_path)
    out.append(f"File size: **{sz:,} bytes** ({sz/1e6:.2f} MB)\n")

    print(f"Loading {bin_path}...", file=sys.stderr)
    td = TDLeafFile.load(bin_path)
    out.append(f"- Version: {td.version}")
    out.append(f"- Adam step counter (t_adam): {td.t_adam:,}")
    out.append(f"- nnue_content_hash: 0x{td.nnue_content_hash:08X}\n")

    nn_init = None
    if init_nnue_path:
        print(f"Loading init {init_nnue_path}...", file=sys.stderr)
        nn_init = NNUEFile.load(init_nnue_path)
        out.append(f"Init .nnue: `{os.path.basename(init_nnue_path)}`\n")

    nn = None
    if nnue_path:
        print(f"Loading exported {nnue_path}...", file=sys.stderr)
        nn = NNUEFile.load(nnue_path)

    # --- Per-section ---
    out.append("## Per-section statistics\n")
    out.append("| Section | Total | Untouched (cnt=0) | v=0 (Adam) | At clamp | Match init |")
    out.append("|---|---:|---:|---:|---:|---:|")

    def row(s: SectionStats) -> str:
        cells = [
            s.name,
            f"{s.total:,}",
            f"{s.untouched:,} ({pct(s.untouched, s.total)})",
            f"{s.v_zero:,} ({pct(s.v_zero, s.total)})",
            f"{s.at_clamp:,} ({pct(s.at_clamp, s.total)})",
        ]
        cells.append("—" if s.init_match is None
                     else f"{s.init_match:,} ({pct(s.init_match, s.total)})")
        return "| " + " | ".join(cells) + " |"

    # FC ×8 stacks: aggregate (each stack's params are tiny; user wants the picture)
    fc0_b_w = np.concatenate([td.fc[s].l0_bias_w   for s in range(LAYER_STACKS)])
    fc0_b_c = np.concatenate([td.fc[s].l0_bias_c   for s in range(LAYER_STACKS)])
    fc0_b_v = np.concatenate([td.fc[s].v_l0_bias   for s in range(LAYER_STACKS)])
    fc0_b_m = np.concatenate([td.fc[s].m_l0_bias   for s in range(LAYER_STACKS)])
    fc0_w_w = np.concatenate([td.fc[s].l0_weight_w for s in range(LAYER_STACKS)])
    fc0_w_c = np.concatenate([td.fc[s].l0_weight_c for s in range(LAYER_STACKS)])
    fc0_w_v = np.concatenate([td.fc[s].v_l0_weight for s in range(LAYER_STACKS)])
    fc0_w_m = np.concatenate([td.fc[s].m_l0_weight for s in range(LAYER_STACKS)])
    fc1_b_w = np.concatenate([td.fc[s].l1_bias_w   for s in range(LAYER_STACKS)])
    fc1_b_c = np.concatenate([td.fc[s].l1_bias_c   for s in range(LAYER_STACKS)])
    fc1_b_v = np.concatenate([td.fc[s].v_l1_bias   for s in range(LAYER_STACKS)])
    fc1_b_m = np.concatenate([td.fc[s].m_l1_bias   for s in range(LAYER_STACKS)])
    fc1_w_w = np.concatenate([td.fc[s].l1_weight_w for s in range(LAYER_STACKS)])
    fc1_w_c = np.concatenate([td.fc[s].l1_weight_c for s in range(LAYER_STACKS)])
    fc1_w_v = np.concatenate([td.fc[s].v_l1_weight for s in range(LAYER_STACKS)])
    fc1_w_m = np.concatenate([td.fc[s].m_l1_weight for s in range(LAYER_STACKS)])
    fc2_b_w = np.concatenate([td.fc[s].l2_bias_w   for s in range(LAYER_STACKS)])
    fc2_b_c = np.concatenate([td.fc[s].l2_bias_c   for s in range(LAYER_STACKS)])
    fc2_b_v = np.concatenate([td.fc[s].v_l2_bias   for s in range(LAYER_STACKS)])
    fc2_b_m = np.concatenate([td.fc[s].m_l2_bias   for s in range(LAYER_STACKS)])
    fc2_w_w = np.concatenate([td.fc[s].l2_weight_w for s in range(LAYER_STACKS)])
    fc2_w_c = np.concatenate([td.fc[s].l2_weight_c for s in range(LAYER_STACKS)])
    fc2_w_v = np.concatenate([td.fc[s].v_l2_weight for s in range(LAYER_STACKS)])
    fc2_w_m = np.concatenate([td.fc[s].m_l2_weight for s in range(LAYER_STACKS)])

    sec_fc0_b = stats_dense_fc(fc0_b_w, fc0_b_c, fc0_b_v, fc0_b_m, "FC0 biases (8×16=128)",
                                init_w_i8=(nn_init.l0_biases.flatten().astype(np.int32) if nn_init else None) if False else None)
    sec_fc0_w = stats_dense_fc(fc0_w_w, fc0_w_c, fc0_w_v, fc0_w_m, "FC0 weights (8×16×1024=131072)",
                                init_w_i8=(nn_init.l0_weights.flatten().astype(np.int32) if nn_init else None))
    sec_fc1_b = stats_dense_fc(fc1_b_w, fc1_b_c, fc1_b_v, fc1_b_m, "FC1 biases (8×32=256)")
    sec_fc1_w = stats_dense_fc(fc1_w_w, fc1_w_c, fc1_w_v, fc1_w_m, "FC1 weights (8×32×32=8192)",
                                init_w_i8=(nn_init.l1_weights.flatten().astype(np.int32) if nn_init else None))
    sec_fc2_b = stats_dense_fc(fc2_b_w, fc2_b_c, fc2_b_v, fc2_b_m, "FC2 biases (8)")
    sec_fc2_w = stats_dense_fc(fc2_w_w, fc2_w_c, fc2_w_v, fc2_w_m, "FC2 weights (8×32=256)",
                                init_w_i8=(nn_init.l2_weights.flatten().astype(np.int32) if nn_init else None))
    sec_ftb   = stats_ft_bias(td, nn_init)
    sec_ftw, coverage = stats_ft_weights(td, nn_init)
    sec_psqt  = stats_psqt(td, nn_init)
    sec_pv    = stats_piece_val(td)

    for s in [sec_fc0_b, sec_fc0_w, sec_fc1_b, sec_fc1_w, sec_fc2_b, sec_fc2_w,
              sec_ftb, sec_ftw, sec_psqt, sec_pv]:
        out.append(row(s))

    # --- Magnitude detail ---
    out.append("\n### Magnitude distribution per section\n")
    out.append("```")
    for s in [sec_fc0_b, sec_fc0_w, sec_fc1_b, sec_fc1_w, sec_fc2_b, sec_fc2_w,
              sec_ftb, sec_ftw, sec_psqt, sec_pv]:
        if s.mag_stats:
            out.append(f"{s.name:42s} " + fmt_stats(s.mag_stats))
    out.append("```")

    # --- Notes ---
    out.append("\n### Notes per section\n")
    for s in [sec_fc0_b, sec_fc0_w, sec_fc1_b, sec_fc1_w, sec_fc2_b, sec_fc2_w,
              sec_ftb, sec_ftw, sec_psqt, sec_pv]:
        if s.notes:
            out.append(f"- **{s.name}**: " + "; ".join(s.notes))

    # --- FT coverage breakdown ---
    out.append("\n### FT feature coverage\n")
    n_dirty = coverage["n_dirty_rows"]
    n_dead  = coverage["n_dead_rows"]
    out.append(f"Total dirty FT rows: **{n_dirty:,} / {FT_INPUTS:,}** "
               f"(dead rows: {n_dead:,}, {100.0*n_dead/FT_INPUTS:.2f}%)\n")
    out.append("Coverage per slot (within all 32 king-buckets):\n")
    out.append("| Slot | Name | Total | Dirty | Unreachable (rank 1/8) | Dirty/(Total-Unreach) |")
    out.append("|---|---|---:|---:|---:|---:|")
    for i in range(11):
        t = coverage["slot_total"][i]
        d = coverage["slot_dirty"][i]
        u = coverage["slot_unreach_pawn"][i]
        reach = max(t - u, 1)
        out.append(f"| {i} | {SLOT_NAMES[i]} | {t:,} | {d:,} | {u:,} | {pct(d, reach)} |")
    out.append("")
    out.append("Coverage per king bucket (0..31):\n")
    out.append("```")
    for kb in range(32):
        t = coverage["kb_total"][kb]
        d = coverage["kb_dirty"][kb]
        bar_w = 40
        filled = int(round(bar_w * d / t)) if t else 0
        out.append(f"  kb {kb:2d}: {d:5d}/{t:5d}  "
                   f"[{'#'*filled}{'.'*(bar_w-filled)}]  {pct(d,t)}")
    out.append("```")

    # --- Cross-check ---
    if nn is not None:
        out.append("")
        out.extend(cross_check(td, nn, scope=os.path.basename(nnue_path)))

    out.append("")
    return "\n".join(out)

# ---------------------------------------------------------------------------
# Multi-snapshot sweep
# ---------------------------------------------------------------------------

def find_snapshots(learn_dir: str) -> list[tuple[str, str, Optional[str], Optional[str]]]:
    """Return list of (label, bin_path, nnue_path_or_None, init_nnue_path_or_None)."""
    files = os.listdir(learn_dir)
    bins = sorted(f for f in files if ".tdleaf.bin" in f and not f.endswith(".lock"))

    # Group by base name, e.g. "nn-fresh-260514" or "nn-psqt-prior".
    snapshots = []
    for b in bins:
        # path like nn-fresh-260514.tdleaf.bin-1.39e6g
        base, _, suffix = b.partition(".tdleaf.bin")
        suffix = suffix.lstrip("-")        # "1.39e6g" or empty
        bin_path = os.path.join(learn_dir, b)
        # Look for matching .nnue (snapshot version).
        nnue_path = None
        # try f"{base}-{suffix}.nnue" then f"{base}.nnue"
        for cand in [f"{base}-{suffix}.nnue", f"{base}.nnue"]:
            if cand in files:
                nnue_path = os.path.join(learn_dir, cand)
                break
        init_path = os.path.join(learn_dir, f"{base}.nnue")
        if not os.path.exists(init_path):
            init_path = None
        label = f"{base} @ {suffix or 'live'}"
        snapshots.append((label, bin_path, nnue_path, init_path))
    return snapshots

def sweep_report(learn_dir: str) -> str:
    snaps = find_snapshots(learn_dir)
    out = []
    out.append(f"# Multi-snapshot sweep — `{learn_dir}`\n")
    out.append(f"Found {len(snaps)} .tdleaf.bin file(s).\n")
    out.append("## Summary table\n")
    out.append("| Snapshot | bytes | t_adam | n_dirty | dead rows | piece_val (P/N/B/R/Q) | n_ft_v_rows |")
    out.append("|---|---:|---:|---:|---:|---|---:|")

    rows = []
    detail_sections = []
    for label, bp, np_, ip in snaps:
        try:
            td = TDLeafFile.load(bp)
        except Exception as e:
            out.append(f"| {label} | ERROR: {e} | | | | | |")
            continue
        sz = os.path.getsize(bp)
        pv = td.piece_val_w / TDLEAF_SCALE
        pv_str = "/".join(f"{float(pv[i]):.0f}" for i in range(5))
        n_dirty = len(td.ft_rows)
        n_dead  = FT_INPUTS - n_dirty
        n_ft_v = len(td.ft_v_rows)
        rows.append((label, sz, td.t_adam, n_dirty, n_dead, pv_str, n_ft_v))
        out.append(f"| {label} | {sz:,} | {td.t_adam:,} | {n_dirty:,} | {n_dead:,} | {pv_str} | {n_ft_v:,} |")

    out.append("")
    out.append("## Detail per snapshot\n")
    for label, bp, np_, ip in snaps:
        out.append("---")
        out.append(f"## {label}\n")
        out.append(f"- bin:  `{bp}`")
        out.append(f"- nnue: `{np_ or '—'}`")
        out.append(f"- init: `{ip or '—'}`\n")
        try:
            detail = analyze_single(bp, np_, ip)
            out.append(detail)
        except Exception as e:
            out.append(f"ERROR: {e}\n")

    return "\n".join(out)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin",  help=".tdleaf.bin to analyse")
    ap.add_argument("--nnue", help="optional companion .nnue (exported from this bin)")
    ap.add_argument("--init-nnue", help="optional initial .nnue for compare-to-init")
    ap.add_argument("--learn-dir",
                    help="instead of --bin, sweep all .tdleaf.bin files in this directory")
    ap.add_argument("--out", default=None, help="write report to file (default: stdout)")
    args = ap.parse_args()

    if args.learn_dir:
        text = sweep_report(args.learn_dir)
    elif args.bin:
        text = analyze_single(args.bin, args.nnue, args.init_nnue)
    else:
        ap.error("provide --bin or --learn-dir")
        return 2

    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
        print(f"Wrote {args.out}", file=sys.stderr)
    else:
        print(text)

if __name__ == "__main__":
    main()
