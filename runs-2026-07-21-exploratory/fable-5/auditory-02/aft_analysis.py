"""Frequency-tuning analysis for dandiset 000986.

All quantities are derived from trial-aligned spike counts.  Two windows are
used throughout: a pre-tone baseline and a post-onset response window chosen
from the pooled population PSTH (the tone-evoked transient in this dataset runs
from roughly 5 to 60 ms after onset).
"""

import numpy as np
from scipy import stats

BASELINE_WIN = (-0.105, -0.005)   # s relative to tone onset
RESPONSE_WIN = (0.005, 0.055)     # s relative to tone onset


def trial_counts(units, onsets, window):
    """Spike counts in `window` around each onset.

    Returns an (n_trials, n_units) integer array.  Counting is done with
    searchsorted on the sorted spike-time vector of each unit, which is exact
    and fast enough to run over every unit of every session.
    """
    lo = onsets + window[0]
    hi = onsets + window[1]
    out = np.empty((len(onsets), len(units)), dtype=np.int32)
    for j, uid in enumerate(units.keys()):
        t = units[uid].times()
        out[:, j] = np.searchsorted(t, hi) - np.searchsorted(t, lo)
    return out


def benjamini_hochberg(pvals, alpha=0.05):
    """Return a boolean mask of hypotheses rejected at FDR `alpha`."""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    thresh = alpha * (np.arange(1, n + 1) / n)
    passed = p[order] <= thresh
    reject = np.zeros(n, dtype=bool)
    if passed.any():
        kmax = np.max(np.nonzero(passed)[0])
        reject[order[: kmax + 1]] = True
    return reject


def analyse_session(units, trials, n_perm=1000, seed=0):
    """Per-unit tone responsiveness and frequency tuning for one session.

    Returns a dict with the per-unit summary arrays and the (n_units, n_freq)
    tuning curves in spikes/s, both raw and baseline-subtracted.
    """
    rng = np.random.default_rng(seed)
    onsets = trials.start_time.values
    freqs = np.unique(trials.stim_frequency.values)
    fidx = np.searchsorted(freqs, trials.stim_frequency.values)

    base = trial_counts(units, onsets, BASELINE_WIN)
    resp = trial_counts(units, onsets, RESPONSE_WIN)
    base_dur = BASELINE_WIN[1] - BASELINE_WIN[0]
    resp_dur = RESPONSE_WIN[1] - RESPONSE_WIN[0]

    n_units = base.shape[1]
    n_freq = len(freqs)

    # Rate in each window, and the tone-evoked difference on every trial.
    base_rate = base / base_dur
    resp_rate = resp / resp_dur
    evoked_rate = resp_rate - base_rate

    # --- tuning curves: mean rate per frequency -----------------------------
    tuning = np.empty((n_units, n_freq))
    tuning_evoked = np.empty((n_units, n_freq))
    tuning_sem = np.empty((n_units, n_freq))
    for k in range(n_freq):
        m = fidx == k
        tuning[:, k] = resp_rate[m].mean(axis=0)
        tuning_evoked[:, k] = evoked_rate[m].mean(axis=0)
        tuning_sem[:, k] = evoked_rate[m].std(axis=0) / np.sqrt(m.sum())

    baseline_rate = base_rate.mean(axis=0)

    # --- is the unit driven by tones at all? --------------------------------
    p_resp = np.array([
        stats.wilcoxon(resp[:, j], base[:, j] * resp_dur / base_dur,
                       zero_method="zsplit").pvalue
        for j in range(n_units)
    ])
    driven = benjamini_hochberg(p_resp) & (tuning_evoked.mean(axis=1) > 0)

    # --- does the response depend on frequency? -----------------------------
    # Kruskal-Wallis across the five frequency groups, on the single-trial
    # response counts.
    groups = [resp[fidx == k] for k in range(n_freq)]
    p_freq = np.array([
        stats.kruskal(*[g[:, j] for g in groups]).pvalue for j in range(n_units)
    ])
    freq_tuned = benjamini_hochberg(p_freq) & driven

    # --- effect size: permutation-calibrated tuning depth -------------------
    # Depth = (max - min) of the evoked tuning curve.  Its null distribution is
    # obtained by shuffling the frequency label across trials.
    depth = tuning_evoked.max(axis=1) - tuning_evoked.min(axis=1)
    null = np.empty((n_perm, n_units))
    counts = np.array([(fidx == k).sum() for k in range(n_freq)])
    for i in range(n_perm):
        perm = rng.permutation(fidx)
        tc = np.empty((n_units, n_freq))
        for k in range(n_freq):
            tc[:, k] = evoked_rate[perm == k].mean(axis=0)
        null[i] = tc.max(axis=1) - tc.min(axis=1)
    p_perm = (1 + (null >= depth).sum(axis=0)) / (n_perm + 1)
    depth_z = (depth - null.mean(axis=0)) / null.std(axis=0)

    # --- descriptive tuning parameters --------------------------------------
    bf_idx = np.argmax(tuning_evoked, axis=1)
    best_freq = freqs[bf_idx]
    # Selectivity index on the rectified evoked curve: 1 = responds to one
    # frequency only, 0 = equal response to all five.
    pos = np.clip(tuning_evoked, 0, None)
    with np.errstate(invalid="ignore", divide="ignore"):
        selectivity = (1 - (pos.sum(axis=1) ** 2) / (n_freq * (pos ** 2).sum(axis=1))) \
            / (1 - 1 / n_freq)
    # Centre of mass of the rectified curve in octaves re 1 kHz.
    oct_axis = np.log2(freqs / 1000.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        centroid = (pos * oct_axis).sum(axis=1) / pos.sum(axis=1)

    return dict(
        freqs=freqs, fidx=fidx, n_trials_per_freq=counts,
        unit_ids=np.array(list(units.keys())),
        tuning=tuning, tuning_evoked=tuning_evoked, tuning_sem=tuning_sem,
        baseline_rate=baseline_rate, mean_evoked=tuning_evoked.mean(axis=1),
        p_resp=p_resp, driven=driven,
        p_freq=p_freq, freq_tuned=freq_tuned,
        depth=depth, depth_z=depth_z, p_perm=p_perm,
        best_freq=best_freq, bf_idx=bf_idx,
        selectivity=selectivity, centroid=centroid,
        resp_counts=resp, base_counts=base,
    )


def psth_by_frequency(units, trials, bin_size=0.005, window=(-0.10, 0.20),
                      unit_ids=None):
    """Mean firing rate (Hz) per unit, per frequency, per time bin.

    Returns (edges, rates) where rates has shape (n_units, n_freq, n_bins).
    """
    onsets = trials.start_time.values
    freqs = np.unique(trials.stim_frequency.values)
    fidx = np.searchsorted(freqs, trials.stim_frequency.values)
    edges = np.arange(window[0], window[1] + bin_size / 2, bin_size)
    keys = list(units.keys()) if unit_ids is None else list(unit_ids)

    rates = np.zeros((len(keys), len(freqs), len(edges) - 1))
    for j, uid in enumerate(keys):
        t = units[uid].times()
        # Offsets of every spike relative to its nearest preceding onset window.
        for k in range(len(freqs)):
            ons = onsets[fidx == k]
            lo = np.searchsorted(t, ons + window[0])
            hi = np.searchsorted(t, ons + window[1])
            rel = np.concatenate([t[a:b] - o for a, b, o in zip(lo, hi, ons)]) \
                if len(ons) else np.array([])
            h, _ = np.histogram(rel, bins=edges)
            rates[j, k] = h / (len(ons) * bin_size)
    return edges, rates


def gaussian_bf(tuning_evoked, freqs):
    """Interpolated best frequency from a parabolic fit in log-frequency.

    Fits a parabola to the peak and its two neighbours on the evoked tuning
    curve; falls back to the discrete peak at the edges of the sampled range.
    """
    oct_axis = np.log2(freqs / 1000.0)
    out = np.empty(tuning_evoked.shape[0])
    for j, tc in enumerate(tuning_evoked):
        k = int(np.argmax(tc))
        if k == 0 or k == len(tc) - 1:
            out[j] = oct_axis[k]
            continue
        y0, y1, y2 = tc[k - 1], tc[k], tc[k + 1]
        denom = y0 - 2 * y1 + y2
        shift = 0.0 if denom == 0 else 0.5 * (y0 - y2) / denom
        out[j] = oct_axis[k] + np.clip(shift, -1, 1) * (oct_axis[1] - oct_axis[0])
    return out
