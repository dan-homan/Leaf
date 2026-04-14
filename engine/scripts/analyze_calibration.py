#!/usr/bin/env python3
"""
analyze_calibration.py  —  Sigmoid temperature (K) and lambda decay analysis
                           for the Leaf TDLeaf(λ) training calibration.

Goal 1A — Sigmoid temperature K
  Find the K (centipawns) that maximises the log-likelihood of game outcomes
  under the model  P(white wins | score) = sigmoid(score / K).
  Produce a reliability diagram showing calibration at the optimal K.

Goal 2A — Lambda decay (autocorrelation of d_t)
  For each game compute  d_t = sigmoid(white_score_cp_t / K_opt).
  Plot the Pearson correlation of d_t with d_{t+k} as a function of lag k,
  split by decisive (win/loss) vs draw games.
  Overlay the current  λ^k  curves for λ_decisive=0.8 and λ_draw=0.5 and
  fit empirical λ values from the data.

Input
-----
  Parquet file produced by extract_positions.py  (default: ../learn/positions.parquet)

Usage
-----
  python3 analyze_calibration.py
  python3 analyze_calibration.py --input ../learn/positions.parquet \\
                                  --stage 5 6 \\
                                  --out-dir ../learn/calibration_plots
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # no display needed; saves to file
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import minimize_scalar
from scipy.special import expit as sigmoid   # fast vectorised sigmoid

# ---------------------------------------------------------------------------
# Current Leaf TDLeaf hyperparameters (for comparison lines)
# ---------------------------------------------------------------------------
K_CURRENT          = 290.0   # centipawns
LAMBDA_DECISIVE    = 0.8
LAMBDA_DRAW        = 0.5

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_INPUT   = Path(__file__).parent.parent / 'learn' / 'positions.parquet'
DEFAULT_OUT_DIR = Path(__file__).parent.parent / 'learn' / 'calibration_plots'

# Training stages to include in calibration (0=untrained … 6=1.5M-trained).
# Default: stages 5 and 6 (the most-trained network — 800K and 1.5M games).
DEFAULT_STAGES = [5, 6]

# Max lag (plies) for the autocorrelation analysis.
MAX_LAG = 60

# Number of reliability-diagram bins.
N_REL_BINS = 40


# ===========================================================================
# Section 1A — Sigmoid temperature K
# ===========================================================================

def neg_log_likelihood(K: float, scores: np.ndarray,
                       results: np.ndarray) -> float:
    """Negative log-likelihood of outcomes under sigmoid(score/K)."""
    p = sigmoid(scores / K)
    # Clip to avoid log(0); results are in {0, 0.5, 1}
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return -np.sum(results * np.log(p) + (1 - results) * np.log(1 - p))


def find_optimal_K(scores: np.ndarray, results: np.ndarray,
                   K_min: float = 50.0, K_max: float = 800.0) -> dict:
    """Grid search + Brent refinement to find MLE temperature K."""
    grid = np.linspace(K_min, K_max, 300)
    nlls = [neg_log_likelihood(k, scores, results) for k in grid]
    k0 = grid[int(np.argmin(nlls))]

    res = minimize_scalar(
        neg_log_likelihood,
        bounds=(max(K_min, k0 - 50), min(K_max, k0 + 50)),
        args=(scores, results),
        method='bounded',
        options={'xatol': 0.1},
    )
    return {
        'K_opt':     float(res.x),
        'nll_opt':   float(res.fun),
        'nll_current': float(neg_log_likelihood(K_CURRENT, scores, results)),
        'grid_K':    grid,
        'grid_nll':  np.array(nlls),
        'n_pos':     len(scores),
    }


def reliability_diagram(scores: np.ndarray, results: np.ndarray,
                        K: float, n_bins: int = N_REL_BINS) -> dict:
    """
    Bin positions by predicted win probability; compare to empirical win rate.
    Returns bin centres, empirical rates, counts, and overall Brier score.
    """
    p_pred = sigmoid(scores / K)
    # Brier score (lower = better)
    brier = float(np.mean((p_pred - results) ** 2))

    edges = np.linspace(0, 1, n_bins + 1)
    centres, empirical, counts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p_pred >= lo) & (p_pred < hi)
        if mask.sum() == 0:
            continue
        centres.append((lo + hi) / 2)
        empirical.append(float(results[mask].mean()))
        counts.append(int(mask.sum()))

    return {
        'bin_centres': np.array(centres),
        'empirical':   np.array(empirical),
        'counts':      np.array(counts),
        'brier':       brier,
    }


def plot_calibration(fit: dict, rel_opt: dict, rel_cur: dict,
                     K_opt: float, out_path: Path):
    """Three-panel calibration figure."""
    fig = plt.figure(figsize=(14, 5))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # --- panel 1: NLL vs K ---------------------------------------------------
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(fit['grid_K'], fit['grid_nll'] / fit['n_pos'], color='steelblue',
             lw=1.5, label='NLL / position')
    ax1.axvline(K_opt,       color='firebrick',  lw=1.5, ls='--',
                label=f'K_opt = {K_opt:.1f} cp')
    ax1.axvline(K_CURRENT,   color='darkorange', lw=1.5, ls=':',
                label=f'K_current = {K_CURRENT:.0f} cp')
    ax1.set_xlabel('Temperature K  (centipawns)')
    ax1.set_ylabel('Neg. log-likelihood per position')
    ax1.set_title('MLE Temperature Search')
    ax1.legend(fontsize=8)
    ax1.set_xlim(fit['grid_K'][0], fit['grid_K'][-1])

    # --- panel 2: reliability diagram ----------------------------------------
    ax2 = fig.add_subplot(gs[1])
    ax2.plot([0, 1], [0, 1], 'k--', lw=0.8, label='Perfect calibration')
    ax2.scatter(rel_opt['bin_centres'], rel_opt['empirical'],
                s=rel_opt['counts'] / rel_opt['counts'].max() * 80,
                color='firebrick', alpha=0.8, zorder=3,
                label=f'K_opt={K_opt:.1f}  Brier={rel_opt["brier"]:.4f}')
    ax2.scatter(rel_cur['bin_centres'], rel_cur['empirical'],
                s=rel_cur['counts'] / rel_cur['counts'].max() * 80,
                color='darkorange', alpha=0.5, marker='^', zorder=2,
                label=f'K={K_CURRENT:.0f}  Brier={rel_cur["brier"]:.4f}')
    ax2.set_xlabel('Predicted win probability  σ(score/K)')
    ax2.set_ylabel('Empirical win rate')
    ax2.set_title('Reliability Diagram')
    ax2.legend(fontsize=8)
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)

    # --- panel 3: sigmoid curves at K_opt vs K_current -----------------------
    ax3 = fig.add_subplot(gs[2])
    s = np.linspace(-600, 600, 600)
    ax3.plot(s, sigmoid(s / K_opt),     color='firebrick',  lw=2,
             label=f'K_opt = {K_opt:.1f} cp')
    ax3.plot(s, sigmoid(s / K_CURRENT), color='darkorange', lw=2, ls='--',
             label=f'K_current = {K_CURRENT:.0f} cp')
    ax3.axhline(0.5, color='gray', lw=0.7, ls=':')
    ax3.axvline(0,   color='gray', lw=0.7, ls=':')
    ax3.set_xlabel('White score  (centipawns)')
    ax3.set_ylabel('P(White wins)')
    ax3.set_title('Sigmoid Comparison')
    ax3.legend(fontsize=8)

    fig.suptitle(
        f'Sigmoid Temperature Calibration  '
        f'(n={fit["n_pos"]:,} positions,  stages {DEFAULT_STAGES})',
        fontsize=11, y=1.01)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path}')


# ===========================================================================
# Section 2A — Lambda decay (autocorrelation of d_t)
# ===========================================================================

def _pearson_from_pairs(pairs_list: list, n_lags: int) -> np.ndarray:
    """Compute Pearson correlation from a list of (x_array, y_array) pairs per lag."""
    corrs = np.full(n_lags, np.nan)
    for k_idx in range(n_lags):
        pairs = pairs_list[k_idx]
        if not pairs:
            continue
        x = np.concatenate([p[0] for p in pairs])
        y = np.concatenate([p[1] for p in pairs])
        if len(x) < 10:
            continue
        xm = x - x.mean(); ym = y - y.mean()
        denom = np.sqrt((xm**2).sum() * (ym**2).sum())
        corrs[k_idx] = float(np.dot(xm, ym) / denom) if denom > 0 else np.nan
    return corrs


def _fit_lambda(corr_array: np.ndarray, lags: np.ndarray) -> float:
    """
    Fit λ such that λ^k best approximates corr_array (least-squares in log space,
    no intercept: log corr = k·log λ).  Returns nan if too few valid points.
    """
    valid = ~np.isnan(corr_array) & (corr_array > 1e-6)
    if valid.sum() < 3:
        return np.nan
    log_c  = np.log(corr_array[valid])
    k_vals = lags[valid].astype(float)
    lam_log = np.dot(k_vals, log_c) / np.dot(k_vals, k_vals)
    return float(np.exp(lam_log))


def compute_lag_correlation(df: pd.DataFrame, K: float,
                            max_lag: int = MAX_LAG) -> dict:
    """
    For each game compute d_t = sigmoid(white_score_cp / K).

    Returns autocorrelation of d_t at lags 1..max_lag (all lags and even-only),
    and corr(d_t, result) vs distance-to-game-end (all and even-parity), split
    by decisive/draw.  Lambda is fitted from the even-lag curves which are free
    of the ply-alternation oscillation.

    The oscillation arises because the moving player's score is systematically
    more optimistic from their own side (tempo effect), so consecutive plies
    carry opposite sign biases in white-POV space, causing corr(d_t, d_{t+1})
    to be lower than corr(d_t, d_{t+2}).  Even-lag pairs are same-parity
    (same side to move) and remove this effect.

    Keys returned
    -------------
    lags                  — int array 1..max_lag
    corr_decisive         — corr(d_t, d_{t+k}), decisive, all lags
    corr_draw             — corr(d_t, d_{t+k}), draw, all lags
    even_lags             — int array 2,4,..max_lag (even subset)
    corr_decisive_even    — corr at even lags, decisive
    corr_draw_even        — corr at even lags, draw
    lags_to_end           — int array 0..max_lag
    corr_result_dec       — corr(d_t, result) vs dist-to-end, decisive, all
    corr_result_draw      — corr(d_t, result) vs dist-to-end, draw, all
    corr_result_dec_ep    — same, even-parity-ply positions only
    corr_result_draw_ep   — same, even-parity-ply positions only
    lambda_ac_dec         — λ fitted from even-lag autocorrelation, decisive
    lambda_ac_draw        — λ fitted from even-lag autocorrelation, draw
    lambda_result_dec     — λ fitted from even-parity d_t-vs-result decay, decisive
    lambda_result_draw    — λ fitted from even-parity d_t-vs-result decay, draw
    """
    df = df.copy()
    df['d'] = sigmoid(df['white_score_cp'].to_numpy(float) / K)

    all_lags  = np.arange(1, max_lag + 1)
    even_lags = np.arange(2, max_lag + 1, 2)          # 2, 4, 6, …
    even_idx  = even_lags - 1                          # indices into all_lags arrays

    # ---- per-game autocorrelation pairs ------------------------------------
    print('  Computing within-game lag correlations...')
    games = df.sort_values(['game_id', 'ply']).groupby('game_id')

    pairs_dec  = [[] for _ in range(max_lag)]
    pairs_draw = [[] for _ in range(max_lag)]

    game_count = 0
    for gid, grp in games:
        result = grp['result'].iloc[0]
        is_dec = (result != 0.5)
        d      = grp['d'].to_numpy(np.float32)
        n      = len(d)
        target = pairs_dec if is_dec else pairs_draw
        for k_idx, k in enumerate(all_lags):
            if n > k:
                target[k_idx].append((d[:n-k], d[k:]))
        game_count += 1
        if game_count % 10_000 == 0:
            print(f'\r    {game_count:,} games processed...', end='', flush=True)

    print(f'\r    {game_count:,} games processed.      ')

    corr_dec  = _pearson_from_pairs(pairs_dec,  max_lag)
    corr_draw = _pearson_from_pairs(pairs_draw, max_lag)

    # Even-lag subset: same side-to-move, removes tempo oscillation
    corr_dec_even  = corr_dec[even_idx]
    corr_draw_even = corr_draw[even_idx]

    lambda_ac_dec  = _fit_lambda(corr_dec_even,  even_lags)
    lambda_ac_draw = _fit_lambda(corr_draw_even, even_lags)

    # ---- predictiveness vs plies-to-end ------------------------------------
    # Decisive games: corr(d_t, result) — valid because result ∈ {0,1}.
    # Draw games:     mean(|d_t − 0.5|) — measures how non-draw-like the score
    #   looked at distance n.  Near game end this should be small (engine has
    #   converged to ≈0.5); far from end it is larger.  We normalise to the
    #   value at distance max_lag (farthest point) so the curve starts below 1
    #   and rises, then invert to get a decay-like signal for λ fitting.
    #   Both metrics use even-parity (white-to-move) plies for the clean version
    #   to remove the mover-optimism oscillation.
    print('  Computing d_t vs result correlation by distance-to-game-end...')
    df['plies_to_end'] = df['n_plies'] - 1 - df['ply']
    df['even_ply']     = (df['ply'] % 2 == 0)

    lags_to_end = np.arange(0, max_lag + 1)
    # Decisive: Pearson corr(d_t, result)
    corr_result_dec    = np.full(max_lag + 1, np.nan)
    corr_result_dec_ep = np.full(max_lag + 1, np.nan)
    # Draw: mean |d_t − 0.5|  (raw value; normalised later for λ fit)
    bias_draw          = np.full(max_lag + 1, np.nan)   # mixed parity
    bias_draw_ep       = np.full(max_lag + 1, np.nan)   # even-parity ply

    def _pearson_dr(d_vals, r_vals):
        if len(d_vals) < 10:
            return np.nan
        dm = d_vals - d_vals.mean(); rm = r_vals - r_vals.mean()
        denom = np.sqrt((dm**2).sum() * (rm**2).sum())
        return float(np.dot(dm, rm) / denom) if denom > 1e-12 else np.nan

    for dte in lags_to_end:
        sub = df[df['plies_to_end'] == dte]

        # Decisive
        s_dec    = sub[sub['result'] != 0.5]
        s_dec_ep = s_dec[s_dec['even_ply']]
        corr_result_dec[dte]    = _pearson_dr(s_dec['d'].to_numpy(float),
                                               s_dec['result'].to_numpy(float))
        corr_result_dec_ep[dte] = _pearson_dr(s_dec_ep['d'].to_numpy(float),
                                               s_dec_ep['result'].to_numpy(float))

        # Draw: |d_t − 0.5| mean
        s_draw    = sub[sub['result'] == 0.5]
        s_draw_ep = s_draw[s_draw['even_ply']]
        if len(s_draw) >= 10:
            bias_draw[dte]    = float(np.abs(s_draw['d'].to_numpy(float) - 0.5).mean())
        if len(s_draw_ep) >= 10:
            bias_draw_ep[dte] = float(np.abs(s_draw_ep['d'].to_numpy(float) - 0.5).mean())

    # Normalise draw bias so it looks like a decay curve starting at 1.
    # Use the farthest valid point as the normalisation reference.
    valid_bias = ~np.isnan(bias_draw_ep)
    if valid_bias.any():
        ref = float(bias_draw_ep[valid_bias][-1])   # value at largest distance
        if ref > 1e-6:
            bias_draw_norm    = bias_draw    / ref
            bias_draw_ep_norm = bias_draw_ep / ref
        else:
            bias_draw_norm = bias_draw_ep_norm = bias_draw_ep.copy()
    else:
        bias_draw_norm = bias_draw_ep_norm = bias_draw_ep.copy()

    # Fit λ from even-parity curves
    # Decisive: corr_result_dec_ep[n] ≈ c0·λ^n  (normalise by n=0 value)
    def _fit_lambda_result_dec(corr_arr):
        c0 = corr_arr[0] if not np.isnan(corr_arr[0]) and corr_arr[0] > 1e-6 else np.nan
        if np.isnan(c0):
            return np.nan
        return _fit_lambda(corr_arr / c0, lags_to_end)

    # Draw: bias_draw_ep_norm[n] should decay from max (far end) toward min
    # (near end).  It is normalised to 1 at the far end and decreases toward 0.
    # Invert to get a conventional decay-from-1 curve: 1 - (bias_norm - bias_norm[0])
    # Actually simpler: treat (bias_norm[max_lag] - bias_norm[n]) as the "signal"
    # that rises then levels; instead just fit directly to 1 - bias_norm_ep (centred).
    def _fit_lambda_draw_bias(bias_norm_arr):
        # Flip and normalise: signal = max_val - bias_norm → starts low, rises.
        # We want the decay of "non-draw-ness" from max distance to 0.
        # Treat bias_norm_ep as a decay curve from bias at far end (≈1) to bias
        # at near end (→0).  This is already a decay; fit directly.
        valid = ~np.isnan(bias_norm_arr) & (bias_norm_arr > 1e-6)
        if valid.sum() < 3:
            return np.nan
        # bias_norm ≈ λ^(max_lag - n) * bias_norm[max_lag]; reverse the lag axis
        # so we have a conventional λ^k form with k = distance from start.
        return _fit_lambda(bias_norm_arr[valid][::-1],
                           lags_to_end[:valid.sum()])

    lambda_result_dec  = _fit_lambda_result_dec(corr_result_dec_ep)
    lambda_result_draw = _fit_lambda_draw_bias(bias_draw_ep_norm)

    return {
        'lags':               all_lags,
        'corr_decisive':      corr_dec,
        'corr_draw':          corr_draw,
        'even_lags':          even_lags,
        'corr_decisive_even': corr_dec_even,
        'corr_draw_even':     corr_draw_even,
        'lags_to_end':        lags_to_end,
        # Decisive: corr(d_t, result)
        'corr_result_dec':    corr_result_dec,
        'corr_result_dec_ep': corr_result_dec_ep,
        # Draw: mean|d_t − 0.5| (normalised so far end ≈ 1)
        'bias_draw':          bias_draw_norm,
        'bias_draw_ep':       bias_draw_ep_norm,
        'lambda_ac_dec':      lambda_ac_dec,
        'lambda_ac_draw':     lambda_ac_draw,
        'lambda_result_dec':  lambda_result_dec,
        'lambda_result_draw': lambda_result_draw,
    }


def plot_lambda(lc: dict, K_opt: float, n_decisive: int, n_draw: int,
                stages: list, out_path: Path):
    """
    Four-panel lambda analysis figure.

    Top row: autocorrelation of d_t
      Left  — all lags (shows ply-alternation oscillation)
      Right — even lags only (same side-to-move pairs; oscillation removed)

    Bottom row: corr(d_t, result) vs distance-to-game-end
      Left  — all distances (oscillation visible)
      Right — even-parity-ply positions only (cleaner; λ fitted here)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.subplots_adjust(hspace=0.38, wspace=0.32)
    (ax1, ax2), (ax3, ax4) = axes

    lags      = lc['lags']
    even_lags = lc['even_lags']
    l2e       = lc['lags_to_end']

    # Shared colour / style helpers
    COL_DEC  = 'steelblue'
    COL_DRAW = 'darkorange'

    def _ref_curve(ax, lags_arr, c0, lam, col, label):
        ax.plot(lags_arr, c0 * lam ** lags_arr, '--', color=col, alpha=0.55,
                lw=1.3, label=label)

    def _fit_curve(ax, lags_arr, c0, lam, col, label):
        if not np.isnan(lam):
            ax.plot(lags_arr, c0 * lam ** lags_arr, ':', color=col, lw=2.2,
                    label=label)

    # ── Panel 1: all-lag autocorrelation (shows oscillation) ─────────────────
    ax1.plot(lags, lc['corr_decisive'], 'o-', color=COL_DEC,  ms=2.5, lw=1.2,
             label=f'Decisive (n={n_decisive:,})', alpha=0.9)
    ax1.plot(lags, lc['corr_draw'],     's-', color=COL_DRAW, ms=2.5, lw=1.2,
             label=f'Draw (n={n_draw:,})',    alpha=0.9)
    ax1.axhline(0, color='gray', lw=0.6, ls=':')
    ax1.set_xlabel('Lag k  (plies)')
    ax1.set_ylabel('corr(d_t, d_{t+k})')
    ax1.set_title('Autocorrelation — all lags\n'
                  '(oscillation from ply-alternation visible)')
    ax1.legend(fontsize=8)

    # ── Panel 2: even-lag autocorrelation (clean; fit λ here) ────────────────
    c0_dec_e  = lc['corr_decisive_even'][0] if not np.isnan(lc['corr_decisive_even'][0])  else 1.0
    c0_draw_e = lc['corr_draw_even'][0]     if not np.isnan(lc['corr_draw_even'][0])      else 1.0

    ax2.plot(even_lags, lc['corr_decisive_even'], 'o-', color=COL_DEC,  ms=4, lw=1.8,
             label='Decisive (even lags)')
    ax2.plot(even_lags, lc['corr_draw_even'],     's-', color=COL_DRAW, ms=4, lw=1.8,
             label='Draw (even lags)')

    _ref_curve(ax2, even_lags, c0_dec_e,  LAMBDA_DECISIVE,
               COL_DEC,  f'λ={LAMBDA_DECISIVE} (current)')
    _ref_curve(ax2, even_lags, c0_draw_e, LAMBDA_DRAW,
               COL_DRAW, f'λ={LAMBDA_DRAW} (current)')
    _fit_curve(ax2, even_lags, c0_dec_e,  lc['lambda_ac_dec'],
               COL_DEC,  f'λ_fit={lc["lambda_ac_dec"]:.3f}')
    _fit_curve(ax2, even_lags, c0_draw_e, lc['lambda_ac_draw'],
               COL_DRAW, f'λ_fit={lc["lambda_ac_draw"]:.3f}')

    ax2.axhline(0, color='gray', lw=0.6, ls=':')
    ax2.set_xlabel('Lag k  (even plies)')
    ax2.set_ylabel('corr(d_t, d_{t+k})')
    ax2.set_title('Autocorrelation — even lags only\n'
                  '(same side-to-move; oscillation removed)')
    ax2.legend(fontsize=8)

    # ── Panel 3: d_t signal vs result, all distances (shows oscillation) ────────
    # Decisive: corr(d_t, result).  Draw: mean|d_t−0.5| (normalised).
    ax3.plot(l2e, lc['corr_result_dec'], 'o-', color=COL_DEC,  ms=2.5, lw=1.2, alpha=0.9,
             label='Decisive: corr(d_t, result)')
    ax3.plot(l2e, lc['bias_draw'],       's-', color=COL_DRAW, ms=2.5, lw=1.2, alpha=0.9,
             label='Draw: mean|d_t−0.5| (norm.)')
    ax3.axhline(0, color='gray', lw=0.6, ls=':')
    ax3.set_xlabel('Plies to game end')
    ax3.set_ylabel('Predictive signal (normalised)')
    ax3.set_title('Predictiveness — all distances\n'
                  '(oscillation from parity mixing visible)')
    ax3.legend(fontsize=8)

    # ── Panel 4: even-parity ply only (clean; fit λ here) ────────────────────
    # Decisive: corr(d_t, result) normalised to 1 at distance 0.
    # Draw: mean|d_t−0.5| normalised so farthest point ≈ 1, decays to 0.
    r0_dec_ep = lc['corr_result_dec_ep'][0]
    r0_dec_ep = r0_dec_ep if (not np.isnan(r0_dec_ep) and r0_dec_ep > 1e-6) else 1.0
    dec_ep_norm = lc['corr_result_dec_ep'] / r0_dec_ep

    ax4.plot(l2e, dec_ep_norm,         'o-', color=COL_DEC,  ms=4, lw=1.8,
             label='Decisive: corr/corr_0 (even-ply)')
    ax4.plot(l2e, lc['bias_draw_ep'],  's-', color=COL_DRAW, ms=4, lw=1.8,
             label='Draw: mean|d_t−0.5| (even-ply, norm.)')

    # Reference λ^n curves anchored at 1
    _ref_curve(ax4, l2e, 1.0, LAMBDA_DECISIVE,
               COL_DEC,  f'λ={LAMBDA_DECISIVE} (current)')
    _ref_curve(ax4, l2e, 1.0, LAMBDA_DRAW,
               COL_DRAW, f'λ={LAMBDA_DRAW} (current)')
    lrd = lc['lambda_result_dec']
    lrw = lc['lambda_result_draw']
    if not np.isnan(lrd):
        ax4.plot(l2e, lrd ** l2e, ':', color=COL_DEC,  lw=2.2,
                 label=f'λ_fit_dec={lrd:.3f}')
    if not np.isnan(lrw):
        ax4.plot(l2e, lrw ** l2e, ':', color=COL_DRAW, lw=2.2,
                 label=f'λ_fit_draw={lrw:.3f}')

    ax4.axhline(0, color='gray', lw=0.6, ls=':')
    ax4.set_xlabel('Plies to game end')
    ax4.set_ylabel('Normalised signal')
    ax4.set_title('Predictiveness — even-parity ply only\n'
                  '(white-to-move positions; oscillation removed)')
    ax4.legend(fontsize=8)

    fig.suptitle(
        f'Lambda Decay Analysis  (K={K_opt:.1f} cp,  stages {stages})',
        fontsize=12, y=1.01)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path}')


# ===========================================================================
# Main
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(
        description='Sigmoid K and lambda decay analysis for Leaf TDLeaf.')
    ap.add_argument('--input',   type=Path, default=DEFAULT_INPUT,
                    help='Parquet file from extract_positions.py')
    ap.add_argument('--out-dir', type=Path, default=DEFAULT_OUT_DIR,
                    help='Directory for output plots and summary')
    ap.add_argument('--stage',   type=int,  nargs='+', default=DEFAULT_STAGES,
                    help='Training stage(s) to include (0–6); default: 5 6')
    ap.add_argument('--max-lag', type=int,  default=MAX_LAG,
                    help='Maximum lag for autocorrelation analysis')
    ap.add_argument('--all-stages', action='store_true',
                    help='Include all training stages (overrides --stage)')
    args = ap.parse_args()

    if not args.input.exists():
        sys.exit(f'ERROR: input file not found: {args.input}\n'
                 f'       Run extract_positions.py first.')

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print(f'Loading {args.input} ...', flush=True)
    df = pd.read_parquet(args.input)
    print(f'  {len(df):,} rows,  {df["game_id"].nunique():,} games,  '
          f'stages present: {sorted(df["training_stage"].unique())}')

    stages = sorted(df['training_stage'].unique()) if args.all_stages else args.stage
    df = df[df['training_stage'].isin(stages)].copy()
    print(f'  After stage filter ({stages}): {len(df):,} rows,  '
          f'{df["game_id"].nunique():,} games')

    if len(df) == 0:
        sys.exit('ERROR: no data remaining after stage filter.')

    scores  = df['white_score_cp'].to_numpy(float)
    results = df['result'].to_numpy(float)

    # ------------------------------------------------------------------
    # 1A — Sigmoid temperature
    # ------------------------------------------------------------------
    print('\n--- Goal 1A: Sigmoid temperature K ---')
    fit = find_optimal_K(scores, results)
    K_opt = fit['K_opt']

    print(f'  K_opt          = {K_opt:.2f} cp')
    print(f'  K_current      = {K_CURRENT:.0f} cp')
    print(f'  NLL/pos (opt)  = {fit["nll_opt"] / fit["n_pos"]:.5f}')
    print(f'  NLL/pos (cur)  = {fit["nll_current"] / fit["n_pos"]:.5f}')
    print(f'  n positions    = {fit["n_pos"]:,}')

    rel_opt = reliability_diagram(scores, results, K_opt)
    rel_cur = reliability_diagram(scores, results, K_CURRENT)
    print(f'  Brier (opt)    = {rel_opt["brier"]:.5f}')
    print(f'  Brier (cur)    = {rel_cur["brier"]:.5f}')

    plot_calibration(fit, rel_opt, rel_cur, K_opt,
                     args.out_dir / 'calibration_K.png')

    # ------------------------------------------------------------------
    # 2A — Lambda decay
    # ------------------------------------------------------------------
    print('\n--- Goal 2A: Lambda decay (autocorrelation of d_t) ---')
    n_dec  = df[df['result'] != 0.5]['game_id'].nunique()
    n_draw = df[df['result'] == 0.5]['game_id'].nunique()
    print(f'  Decisive games : {n_dec:,}')
    print(f'  Draw games     : {n_draw:,}')

    lc = compute_lag_correlation(df, K_opt, max_lag=args.max_lag)

    print(f'\n  From even-lag autocorrelation:')
    print(f'    λ_decisive = {lc["lambda_ac_dec"]:.4f}  (current = {LAMBDA_DECISIVE})')
    print(f'    λ_draw     = {lc["lambda_ac_draw"]:.4f}  (current = {LAMBDA_DRAW})')
    print(f'  From even-parity d_t-vs-result decay:')
    print(f'    λ_decisive = {lc["lambda_result_dec"]:.4f}  (current = {LAMBDA_DECISIVE})')
    print(f'    λ_draw     = {lc["lambda_result_draw"]:.4f}  (current = {LAMBDA_DRAW})')

    plot_lambda(lc, K_opt, n_dec, n_draw, stages,
                args.out_dir / 'lambda_decay.png')

    # ------------------------------------------------------------------
    # Summary text file
    # ------------------------------------------------------------------
    summary_path = args.out_dir / 'summary.txt'
    with open(summary_path, 'w') as f:
        f.write('Leaf TDLeaf Calibration Analysis\n')
        f.write('=' * 50 + '\n\n')
        f.write(f'Input          : {args.input}\n')
        f.write(f'Training stages: {stages}\n')
        f.write(f'Positions      : {fit["n_pos"]:,}\n')
        f.write(f'Games (decisive): {n_dec:,}\n')
        f.write(f'Games (draw)    : {n_draw:,}\n\n')
        f.write('--- Sigmoid Temperature K ---\n')
        f.write(f'  K_opt          = {K_opt:.2f} cp\n')
        f.write(f'  K_current      = {K_CURRENT:.0f} cp\n')
        f.write(f'  delta          = {K_opt - K_CURRENT:+.2f} cp\n')
        f.write(f'  NLL/pos (opt)  = {fit["nll_opt"] / fit["n_pos"]:.6f}\n')
        f.write(f'  NLL/pos (cur)  = {fit["nll_current"] / fit["n_pos"]:.6f}\n')
        f.write(f'  Brier (opt)    = {rel_opt["brier"]:.6f}\n')
        f.write(f'  Brier (cur)    = {rel_cur["brier"]:.6f}\n\n')
        f.write('--- Lambda Decay ---\n')
        f.write('  Method: even-lag autocorrelation (removes ply-alternation oscillation)\n')
        f.write(f'    lambda_decisive = {lc["lambda_ac_dec"]:.4f}  '
                f'(current = {LAMBDA_DECISIVE})\n')
        f.write(f'    lambda_draw     = {lc["lambda_ac_draw"]:.4f}  '
                f'(current = {LAMBDA_DRAW})\n')
        f.write('  Method: even-parity d_t-vs-result decay\n')
        f.write(f'    lambda_decisive = {lc["lambda_result_dec"]:.4f}  '
                f'(current = {LAMBDA_DECISIVE})\n')
        f.write(f'    lambda_draw     = {lc["lambda_result_draw"]:.4f}  '
                f'(current = {LAMBDA_DRAW})\n')
    print(f'  Saved: {summary_path}')

    print('\nAll done.')


if __name__ == '__main__':
    main()
