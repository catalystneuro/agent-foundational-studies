"""Core tuning analysis for DANDI 000986 pure-tone responses.

For each unit and each trial we count spikes in a short post-onset response
window and in a pre-onset baseline window. Tone responsiveness is tested with
a Wilcoxon signed-rank test (response vs baseline across all trials);
frequency selectivity with a Kruskal-Wallis test on per-trial response counts
grouped by stimulus frequency. Tuning curves are the mean evoked-minus-baseline
rate per frequency.
"""
import numpy as np
import scipy.stats

FREQS = np.array([2000.0, 4000.0, 8000.0, 16000.0, 32000.0])
RESP_WIN = (0.005, 0.060)   # post-onset response window (s)
BASE_WIN = (-0.050, 0.0)    # pre-onset baseline window (s)
TRIM = 1e-6                 # numerical tolerance


def per_trial_counts(spike_times, starts, resp_win=RESP_WIN, base_win=BASE_WIN):
    """Spike counts in response and baseline windows, per trial.

    ``spike_times`` must be sorted. ``starts`` are returned in sorted order
    together with the permutation so the caller can align other trial metadata.
    """
    starts = np.asarray(starts, dtype=float)
    order = np.argsort(starts)
    starts = starts[order]
    idx = np.searchsorted(spike_times, starts)
    n = len(starts)
    resp = np.zeros(n)
    base = np.zeros(n)
    for i in range(n):
        lo, hi = idx[i], (idx[i + 1] if i + 1 < n else len(spike_times))
        st = spike_times[lo:hi]
        resp[i] = ((st >= starts[i] + resp_win[0]) & (st < starts[i] + resp_win[1])).sum()
        base[i] = ((st >= starts[i] + base_win[0]) & (st < starts[i] + base_win[1])).sum()
    return resp, base, order


def unit_stats(spike_times, starts, freqs):
    """Wilcoxon responsiveness and Kruskal-Wallis frequency modulation for one unit."""
    resp, base, order = per_trial_counts(np.sort(spike_times), starts)
    freqs = np.asarray(freqs, dtype=float)[order]
    w_p = 1.0
    if np.any(resp != base):
        try:
            _, w_p = scipy.stats.wilcoxon(resp, base)
        except ValueError:
            w_p = 1.0
    kw_p = 1.0
    groups = [resp[freqs == f] for f in FREQS]
    if all(len(g) > 1 for g in groups) and np.any(resp > 0):
        kw_p = scipy.stats.kruskal(*groups).pvalue
    return resp, base, order, w_p, kw_p


def tuning_and_rate(resp, base, freqs, order):
    """Evolved-baseline net rate per frequency (spikes/s) and overall baseline rate."""
    counts = (resp - base)[order]
    fr = np.asarray(freqs, dtype=float)[order]
    win_len = RESP_WIN[1] - RESP_WIN[0]
    baseline_len = BASE_WIN[1] - BASE_WIN[0]
    tuning = np.array([counts[fr == f].mean() / win_len for f in FREQS])
    base_rate = base.mean() / baseline_len
    return tuning, base_rate


def raster_relative(spike_times, starts, freq_ordered, pre=0.050, post=0.260):
    """Trial index and relative spike time (s, onset-relative) for a raster plot.

    ``freq_ordered`` are the trial frequencies already aligned to sorted starts
    (as returned by ``unit_stats``). Returns trial index per spike, relative
    time per spike, and the sorted starts + frequency per trial.
    """
    starts = np.asarray(starts, dtype=float)
    order = np.argsort(starts)
    starts = starts[order]
    freq_ordered = np.asarray(freq_ordered, dtype=float)[order]
    st = np.sort(spike_times)
    # Build per-trial spike sets with a bounded loop; 7.4k trials x ~30 spikes max.
    trials = []
    times = []
    idx = np.searchsorted(st, starts)
    for i, s0 in enumerate(starts):
        lo, hi = idx[i], (idx[i + 1] if i + 1 < len(starts) else len(st))
        sp = st[lo:hi]
        d = sp[(sp >= s0 - pre) & (sp < s0 + post)] - s0
        trials.append(np.full(len(d), i, dtype=np.int64))
        times.append(d)
    return (np.concatenate(trials), np.concatenate(times),
            starts, freq_ordered)