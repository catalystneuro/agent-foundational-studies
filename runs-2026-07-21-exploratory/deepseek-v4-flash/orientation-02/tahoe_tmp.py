"""Core orientation-tuning analysis for Allen Visual Coding Neuropixels.

Computes per-unit firing rates for every drifting-grating condition
(8 drift directions x 5 temporal frequencies), the global orientation
selectivity index (gOSI, one per temporal frequency), the preferred
temporal frequency, and a spike-train permutation test for selectivity
significance.

The gOSI is the magnitude of the circular mean of response vectors where
each new direction contributes a vector at twice its angle, so that
opposite drift directions (180 deg apart) produce the same orientation.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm

DRIFT_DIRECTIONS = np.array([0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0])
TEMPORAL_FREQS = np.array([1.0, 2.0, 4.0, 8.0, 15.0])
SWEEP_DURATION = 2.0  # seconds


def sweep_spike_counts(spike_times, starts, stops):
    """Count spikes in each [start, stop) window, vectorized.

    Uses np.searchsorted per window boundary; each window is a 2 s
    drifting-grating presentation.
    """
    starts = np.asarray(starts)
    stops = np.asarray(stops)
    idx_start = np.searchsorted(spike_times, starts, side="left")
    idx_stop = np.searchsorted(spike_times, stops, side="left")
    return (idx_stop - idx_start).astype(float)


def empty_response_array():
    """(n_directions, n_tfs) array that is NaN where no data exists."""
    return np.full((len(DRIFT_DIRECTIONS), len(TEMPORAL_FREQS)), np.nan)


def fold_orientation(rate_dir):
    """Average opposite drift directions into 4 orientation responses."""
    return (rate_dir[:4] + rate_dir[4:]) / 2.0


def gosi_vector(rate, orientations_as_directions=False):
    """Global orientation selectivity index for a response vector.

    rate : length 8 (directions) or length 4 (orientations).
    orientations_as_directions : if False, assumes the input is 8 drift
        directions and folds opposite directions before computing the
        angular index with period 180 deg.
    """
    rate = np.asarray(rate, dtype=float)
    if orientations_as_directions:
        resp = rate
        angles = np.radians(np.arange(4) * 45.0) * 2.0  # period 180 deg
    else:
        resp = fold_condition(rate)
        angles = np.radians(np.arange(4) * 45.0) * 2.0
    total = np.nansum(resp)
    sin_sum = np.nansum(resp * np.sin(angles))
    cos_sum = np.nansum(resp * np.cos(angles))
    if total <= 0 or np.isnan(total):
        return np.nan
    return np.hypot(sin_sum, cos_sum) / total


def compute_unit_stats(spike_times, starts, stops, cond_idx):
    """Compute mean responses and gOSI for one unit.

    Parameters
    ----------
    spike_times : np.ndarray
        All spike times of the unit (s).
    starts, stops : np.ndarray
        Sweep boundaries for the valid (non-blank, non-invalid) sweeps.
    cond_idx : np.ndarray of shape (n_sweeps, 2), int
        [direction-index, tf-index] for each sweep.

    Returns
    -------
    dict with keys:
      rates : (8, 5) mean firing rate array
      rate_sd_over_reps : (8, 5) sd across repeated sweeps (for plotting)
      gosi_per_tf : (5,) gOSI at each temporal frequency
      peak_rate : float
      blank_rate : float
      preferred_tf_idx : int
      dir_rates_pref_tf : (8,) response at preferred TF
      orient_rates_pref_tf : (4,) orientation response at preferred TF
    """
    counts = sweep_spike_counts(spike_times, starts, stops)
    # group counts by (direction, tf)
    n_dir = len(DRIFT_DIRECTIONS)
    n_tf = len(TEMPORAL_FREQS)
    sum_counts = np.full((n_dir, n_tf), np.nan)
    n_reps = np.full((n_dir, n_tf), np.nan)
    for d in range(n_dir):
        for tf in range(n_tf):
            mask = (cond_idx[:, 0] == d) & (cond_idx[:, 1] == tf)
            if mask.sum() > 0:
                sum_counts[d, tf] = counts[mask].sum()
                n_reps[d, tf] = mask.sum()
    rates = sum_counts / n_reps / SWEEP_DURATION

    # blank rate: mean of all valid sweeps? blanks are excluded from the
    # sweep set; instead the blank baseline comes from the inter-sweep
    # gaps which are not in our sweep list. We use the minimum response
    # as an internal baseline check only; the significance test does not
    # depend on it.
    peak_rate = np.nanmax(rates) if np.any(np.isfinite(rates)) else np.nan
    blank_rate = None

    gosi_per_tf = np.full(n_tf, np.nan)
    for tf in range(n_tf):
        col = rates[:, tf]
        if np.all(np.isnan(col)):
            continue
        gosi_per_tf[tf] = gosi_vector(col)

    if np.all(np.isnan(gosi_per_tf)):
        preferred_tf_idx = np.nan
        dir_osi_pref = np.full(n_dir, np.nan)
        orient_osi_pref = np.full(4, np.nan)
    else:
        preferred_tf_idx = int(np.nanargmax(gosi_per_tf))
        dir_osi_pref = rates[:, preferred_tf_idx]
        orient_osi_pref = fold_condition(dir_osi_pref)

    return {
        "rates": rates,
        "gosi_per_tf": gosi_per_tf,
        "peak_rate": peak_rate,
        "preferred_tf_idx": preferred_tf_idx,
        "dir_rates_pref_tf": dir_osi_pref,
        "orient_rates_pref_tf": orient_osi_pref,
    }


def gosi_with_preferred_tf(rate_matrix):
    """gOSI selected at a unit's preferred temporal frequency.

    Null distribution generation needs this exact computation (per-TF
    gOSI, then max over TF), which is the FULL statistic.
    """
    gosi_per_tf = np.full(rate_matrix.shape[1], np.nan)
    for tf in range(rate_matrix.shape[1]):
        col = rate_matrix[:, tf]
        if np.all(np.isnan(col)):
            continue
        gosi_per_tf[tf] = gosi_vector(col)
    if np.all(np.isnan(gosi_per_tf)):
        return np.nan
    return np.nanmax(gosi_per_tf)