"""Frequency-tuning analysis primitives.

Everything here operates on a pynapple TsGroup of spike trains plus an array of
tone onset times and the frequency presented on each trial.
"""

import numpy as np
from scipy import stats

# response window relative to tone onset (s); chosen from the grand PSTH
RESP_WIN = (0.010, 0.060)
BASE_WIN = (-0.110, -0.010)


def _sorted_times(tsgroup):
    return [np.asarray(tsgroup[u].times()) for u in tsgroup.keys()]


def window_counts(tsgroup, onsets, window):
    """Spike counts in `window` around each onset.

    Returns an (n_trials, n_units) integer array.
    """
    a, b = window
    out = np.empty((len(onsets), len(tsgroup)), dtype=np.int32)
    for j, st in enumerate(_sorted_times(tsgroup)):
        lo = np.searchsorted(st, onsets + a, side="left")
        hi = np.searchsorted(st, onsets + b, side="left")
        out[:, j] = hi - lo
    return out


def relative_spike_times(spike_times, onsets, window):
    """Spike times relative to each onset, plus the trial index of each spike."""
    a, b = window
    lo = np.searchsorted(spike_times, onsets + a, side="left")
    hi = np.searchsorted(spike_times, onsets + b, side="left")
    n = hi - lo
    total = int(n.sum())
    if total == 0:
        return np.array([]), np.array([], dtype=int)
    trial_idx = np.repeat(np.arange(len(onsets)), n)
    starts = np.repeat(lo, n)
    offs = np.arange(total) - np.repeat(np.cumsum(n) - n, n)
    rel = spike_times[starts + offs] - onsets[trial_idx]
    return rel, trial_idx


def psth(spike_times, onsets, bins):
    """Mean firing rate (Hz) in each bin, averaged over onsets."""
    rel, _ = relative_spike_times(spike_times, onsets, (bins[0], bins[-1]))
    counts = np.histogram(rel, bins=bins)[0]
    return counts / (len(onsets) * np.diff(bins))


def evoked_rates(tsgroup, onsets, frequency, resp_win=RESP_WIN, base_win=BASE_WIN):
    """Per-unit, per-frequency evoked firing rate and its standard error.

    Returns a dict with:
      freqs        (n_freq,)             sorted unique frequencies
      resp_rate    (n_units, n_freq)     mean rate in the response window (Hz)
      base_rate    (n_units,)            mean rate in the baseline window (Hz)
      evoked       (n_units, n_freq)     response minus baseline (Hz)
      evoked_sem   (n_units, n_freq)     s.e.m. of the evoked rate across trials
      resp_counts  (n_trials, n_units)   raw response-window counts
      base_counts  (n_trials, n_units)   raw baseline-window counts
    """
    freqs = np.unique(frequency)
    rc = window_counts(tsgroup, onsets, resp_win)
    bc = window_counts(tsgroup, onsets, base_win)
    rdur = resp_win[1] - resp_win[0]
    bdur = base_win[1] - base_win[0]

    base_rate = bc.mean(axis=0) / bdur
    resp_rate = np.empty((rc.shape[1], len(freqs)))
    evoked = np.empty_like(resp_rate)
    evoked_sem = np.empty_like(resp_rate)
    for k, f in enumerate(freqs):
        m = frequency == f
        trial_evoked = rc[m] / rdur - bc[m] / bdur
        resp_rate[:, k] = rc[m].mean(axis=0) / rdur
        evoked[:, k] = trial_evoked.mean(axis=0)
        evoked_sem[:, k] = trial_evoked.std(axis=0, ddof=1) / np.sqrt(m.sum())

    return dict(freqs=freqs, resp_rate=resp_rate, base_rate=base_rate, evoked=evoked,
                evoked_sem=evoked_sem, resp_counts=rc, base_counts=bc,
                resp_dur=rdur, base_dur=bdur)


def fdr_bh(pvals, q=0.05):
    """Benjamini-Hochberg FDR control. Returns a boolean reject mask."""
    p = np.asarray(pvals)
    n = len(p)
    order = np.argsort(p)
    thresh = q * np.arange(1, n + 1) / n
    passed = p[order] <= thresh
    reject = np.zeros(n, dtype=bool)
    if passed.any():
        kmax = np.max(np.where(passed)[0])
        reject[order[: kmax + 1]] = True
    return reject


def tuning_statistics(res, frequency, q=0.05):
    """Sound-responsiveness and frequency-selectivity tests for every unit.

    Sound responsiveness: Wilcoxon signed-rank on (response - baseline) counts
    across all trials, corrected for the 100/50 ms window mismatch by rate.
    Frequency selectivity: Kruskal-Wallis across the frequency groups on the
    response-window counts.
    """
    rc, bc = res["resp_counts"], res["base_counts"]
    n_units = rc.shape[1]
    freqs = res["freqs"]
    groups = [frequency == f for f in freqs]

    p_resp = np.ones(n_units)
    p_freq = np.ones(n_units)
    for j in range(n_units):
        d = rc[:, j] / res["resp_dur"] - bc[:, j] / res["base_dur"]
        if np.any(d != 0):
            p_resp[j] = stats.wilcoxon(d, zero_method="zsplit").pvalue
        samples = [rc[m, j] for m in groups]
        if len(np.unique(np.concatenate(samples))) > 1:
            p_freq[j] = stats.kruskal(*samples).pvalue

    responsive = fdr_bh(p_resp, q)
    tuned = fdr_bh(p_freq, q)

    ev = res["evoked"]
    best_idx = np.argmax(ev, axis=1)
    best_freq = freqs[best_idx]

    # frequency selectivity index on the non-negative part of the tuning curve
    pos = np.clip(ev, 0, None)
    with np.errstate(invalid="ignore", divide="ignore"):
        peak = pos.max(axis=1)
        sparseness = _sparseness(pos)
        mi = _mutual_information(res, frequency)

    # units whose best frequency actually drives them above baseline; the rest
    # are suppressed by every tone tested and have no meaningful "best frequency"
    enhanced = responsive & (ev.max(axis=1) > 0)
    suppressed = responsive & ~enhanced
    snr = peak / np.clip(res["evoked_sem"].mean(axis=1), 1e-9, None)

    return dict(p_resp=p_resp, p_freq=p_freq, responsive=responsive, tuned=tuned,
                enhanced=enhanced, suppressed=suppressed, snr=snr,
                best_idx=best_idx, best_freq=best_freq, peak_evoked=peak,
                sparseness=sparseness, mutual_info=mi)


def _sparseness(pos):
    """Lifetime sparseness across frequencies (0 = flat, 1 = one frequency only)."""
    n = pos.shape[1]
    num = (pos.mean(axis=1)) ** 2
    den = (pos ** 2).mean(axis=1)
    s = (1 - num / den) / (1 - 1.0 / n)
    return np.where(den > 0, s, np.nan)


def _mutual_information(res, frequency):
    """Mutual information between tone frequency and spike count, in bits/spike.

    Uses the Skaggs formulation with the frequency distribution as the prior.
    """
    freqs = res["freqs"]
    p_f = np.array([(frequency == f).mean() for f in freqs])
    lam = res["resp_rate"]                       # (n_units, n_freq)
    lam_bar = (lam * p_f[None, :]).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = lam / lam_bar[:, None]
        terms = p_f[None, :] * ratio * np.log2(np.where(ratio > 0, ratio, 1))
    mi = terms.sum(axis=1)
    return np.where(lam_bar > 0, mi, np.nan)


def split_half(tsgroup, onsets, frequency, resp_win=RESP_WIN, base_win=BASE_WIN):
    """Split trials into interleaved halves and compute tuning in each.

    Best frequency estimated on one half is unbiased when applied to the other,
    which matters because argmax over a noisy tuning curve is biased upwards.
    """
    idx = np.arange(len(onsets))
    a, b = idx % 2 == 0, idx % 2 == 1
    ra = evoked_rates(tsgroup, onsets[a], frequency[a], resp_win, base_win)
    rb = evoked_rates(tsgroup, onsets[b], frequency[b], resp_win, base_win)
    bf_a, bf_b = np.argmax(ra["evoked"], axis=1), np.argmax(rb["evoked"], axis=1)
    return dict(A=ra, B=rb, bf_a=bf_a, bf_b=bf_b, match=bf_a == bf_b)


def bf_aligned(res_test, bf_train, peak_train, n_freq, keep):
    """Stack held-out tuning curves on an axis of octaves from the training BF.

    Curves are scaled by the training-half peak so the value at offset 0 is an
    unbiased estimate of tuning reliability (1.0 = fully reproducible).
    """
    offsets = np.arange(-(n_freq - 1), n_freq)
    stack = np.full((int(keep.sum()), len(offsets)), np.nan)
    for row, i in enumerate(np.where(keep)[0]):
        shift = (n_freq - 1) - bf_train[i]
        stack[row, shift:shift + n_freq] = res_test["evoked"][i] / peak_train[i]
    return offsets, stack


def half_max_width_octaves(evoked, freqs):
    """Tuning width in octaves at half the peak evoked rate.

    Linear interpolation on a log2-frequency axis; returns NaN when the tuning
    curve does not fall below half maximum inside the tested range.
    """
    lf = np.log2(freqs)
    widths = np.full(evoked.shape[0], np.nan)
    for j in range(evoked.shape[0]):
        y = np.clip(evoked[j], 0, None)
        if y.max() <= 0:
            continue
        half = y.max() / 2
        k = int(np.argmax(y))
        # left crossing
        left = lf[0]
        ok_l = False
        for i in range(k, 0, -1):
            if y[i - 1] < half:
                frac = (y[i] - half) / (y[i] - y[i - 1])
                left = lf[i] - frac * (lf[i] - lf[i - 1])
                ok_l = True
                break
        right = lf[-1]
        ok_r = False
        for i in range(k, len(y) - 1):
            if y[i + 1] < half:
                frac = (y[i] - half) / (y[i] - y[i + 1])
                right = lf[i] + frac * (lf[i + 1] - lf[i])
                ok_r = True
                break
        if ok_l and ok_r:
            widths[j] = right - left
    return widths
