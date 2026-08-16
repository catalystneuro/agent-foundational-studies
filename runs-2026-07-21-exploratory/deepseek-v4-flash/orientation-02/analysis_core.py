"""Core orientation-selectivity analysis for the Allen Visual Coding session.

Reads the drifting-gratings stimulus presentations, computes per-unit,
per-condition firing rates (8 drift directions x 5 temporal frequencies),
the global orientation selectivity index (gOSI), and a permutation-based
significance test whose null recomputes the FULL statistic (per-TF gOSI
followed by max over temporal frequencies).
"""

import json
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------- stimulus
DIRS = np.array([0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0])
TFS = np.array([1.0, 2.0, 4.0, 8.0, 15.0])
SWEEP_MS = 2.0  # seconds per presentation

# regions we analyze, and which label groups each into
CORTEX_VIS = {"VISp", "VISl", "VISrl", "VISpm", "VISam", "VISal"}
ALL_REGIONS = ["VISp", "VISl", "VISrl", "VISpm", "VISal", "LGd", "LP", "CA1", "CA3", "DG", "APN", "PO"]


def fold_orient(rate8):
    """Average the two opposite drift directions -> 4 orientation rates."""
    r = np.asarray(rate8, dtype=float)
    return (r[:4] + r[4:]) / 2.0


def gsi(rate, fold=True):
    """Global selectivity index (orientation if fold, direction if not).

    gSI = |sum_k r_k e^(2i theta_k)| / sum_k r_k  over K directions,
    where the period-180 flag folds opposite directions by doubling the
    angle. If fold=False the angle is not doubled (period 360).
    """
    r = np.asarray(rate, dtype=float)
    if fold:
        ang = np.deg2rad(np.linspace(0, 135, 4) * 2)
    else:
        ang = np.deg2rad(np.linspace(0, 315, 8))
    s_c = np.sum(r * np.cos(ang))
    s_s = np.sum(r * np.sin(ang))
    denom = np.sum(r)
    if denom <= 0:
        return np.nan
    return np.hypot(s_c, s_s) / denom


def per_tf_statistic(rates, fold=True):
    """gSI at the unit's best temporal frequency (the FULL statistic).

    rates : ndarray (n_dirs, n_tfs). Returns nan if no TF has data.
    """
    if np.isnan(rates).all():
        return np.nan
    stat = np.array([gsi(rates[:, j], fold=fold) for j in range(rates.shape[1])])
    with np.errstate(invalid="ignore"):
        return np.nanmax(stat)


def rate_matrix_from_sweeps(counts, cond_idx):
    """Convert per-sweep spike counts into an (n_dir, n_tf) rate matrix."""
    n_dir, n_tf = len(DIRS), len(TFS)
    sums = np.zeros((n_dir, n_tf))
    ns = np.zeros((n_dir, n_tf), dtype=int)
    for j in range(len(counts)):
        d, tf = int(cond_idx[j, 0]), int(cond_idx[j, 1])
        sums[d, tf] += counts[j]
        ns[d, tf] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        rates = sums / np.maximum(ns, 1) / SWEEP_MS  # hz
    rates[ns == 0] = np.nan
    return rates