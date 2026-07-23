"""Core analysis routines: per-trial spike counts, tuning curves, selectivity stats."""
import numpy as np
import pandas as pd
import pynapple as nap
from scipy import stats

BASELINE = (-0.155, -0.005)   # s relative to tone onset
RESPONSE = (0.005, 0.105)     # s relative to tone onset (25 ms tone + offset response)


def trial_counts(units, onsets, window):
    """Spike counts per (trial, unit) inside `window` seconds relative to each onset.

    Uses searchsorted on the sorted spike-time vectors, which is exact and fast
    for the ~7.4k trials x ~200 units in one session.
    """
    lo = onsets + window[0]
    hi = onsets + window[1]
    out = np.empty((len(onsets), len(units)), dtype=np.int32)
    for j, k in enumerate(units.keys()):
        t = units[k].t
        out[:, j] = np.searchsorted(t, hi) - np.searchsorted(t, lo)
    return out


def evoked_matrix(units, onsets, baseline=BASELINE, response=RESPONSE):
    """Baseline-subtracted per-trial firing rate, shape (n_trials, n_units)."""
    n_resp = trial_counts(units, onsets, response)
    n_base = trial_counts(units, onsets, baseline)
    rate_r = n_resp / (response[1] - response[0])
    rate_b = n_base / (baseline[1] - baseline[0])
    return rate_r - rate_b, rate_r, rate_b


def sparseness(tuning):
    """Lifetime sparseness of a rectified tuning curve: 0 = flat, 1 = one frequency."""
    r = np.clip(np.atleast_2d(tuning), 0, None)
    n = r.shape[1]
    with np.errstate(invalid="ignore", divide="ignore"):
        return (1 - (r.sum(1) / n) ** 2 / ((r ** 2).sum(1) / n)) / (1 - 1 / n)


def session_tuning(units, onsets, frequency, baseline=BASELINE, response=RESPONSE):
    """Per-unit frequency tuning + selectivity statistics for one session.

    Returns (tuning DataFrame indexed by unit with one column per frequency,
             stats DataFrame with baseline rate, best frequency, p-values, etc.)
    """
    freqs = np.unique(frequency)
    evoked, rate_r, rate_b = evoked_matrix(units, onsets, baseline, response)

    tuning = np.stack([evoked[frequency == f].mean(0) for f in freqs], axis=1)
    tuning_sem = np.stack(
        [stats.sem(evoked[frequency == f], axis=0) for f in freqs], axis=1)
    driven = np.stack([rate_r[frequency == f].mean(0) for f in freqs], axis=1)

    # Frequency selectivity: Kruskal-Wallis across the 5 frequency groups on
    # per-trial evoked counts. Sound-responsiveness: response vs baseline counts.
    p_freq, p_sound = [], []
    for j in range(evoked.shape[1]):
        groups = [evoked[frequency == f, j] for f in freqs]
        p_freq.append(stats.kruskal(*groups).pvalue)
        p_sound.append(stats.wilcoxon(evoked[:, j], alternative="two-sided").pvalue)

    bf_idx = np.argmax(tuning, axis=1)
    sparse = sparseness(tuning)

    idx = list(units.keys())
    tuning_df = pd.DataFrame(tuning, index=idx, columns=freqs)
    sem_df = pd.DataFrame(tuning_sem, index=idx, columns=freqs)
    driven_df = pd.DataFrame(driven, index=idx, columns=freqs)
    stats_df = pd.DataFrame(
        dict(baseline_rate=rate_b.mean(0),
             best_freq=freqs[bf_idx],
             peak_evoked=tuning.max(1),
             mean_evoked=tuning.mean(1),
             sparseness=sparse,
             p_freq=p_freq,
             p_sound=p_sound),
        index=idx)
    return tuning_df, sem_df, driven_df, stats_df, evoked


def fdr(pvals, alpha=0.05):
    """Benjamini-Hochberg; returns boolean mask of significant tests."""
    p = np.asarray(pvals)
    order = np.argsort(p)
    m = len(p)
    thresh = alpha * (np.arange(1, m + 1) / m)
    passed = p[order] <= thresh
    k = np.max(np.where(passed)[0]) + 1 if passed.any() else 0
    out = np.zeros(m, bool)
    out[order[:k]] = True
    return out


def psth(unit_ts, onsets, window=(-0.15, 0.30), bin_size=0.005, sigma_bins=1.5):
    """Peri-stimulus time histogram (spikes/s) for one unit, using pynapple's
    perievent alignment. Returns (time_centres, rate)."""
    peth = nap.compute_perievent(unit_ts, nap.Ts(np.asarray(onsets)), window)
    cnt = peth.count(bin_size).sum(axis=1)
    rate = cnt.values / (bin_size * len(onsets))
    if sigma_bins:
        k = np.exp(-0.5 * (np.arange(-4, 5) / sigma_bins) ** 2)
        rate = np.convolve(rate, k / k.sum(), mode="same")
    return cnt.index.values, rate
