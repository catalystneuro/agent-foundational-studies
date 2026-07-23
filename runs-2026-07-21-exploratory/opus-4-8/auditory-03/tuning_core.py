"""Core computations for tone-evoked frequency tuning.

Spike times arrive as a pynapple TsGroup; the routines below turn them into
trial-by-trial spike counts, PSTHs, frequency tuning curves and the summary
statistics (best frequency, tuning bandwidth, significance) used in the figures.
"""

import numpy as np
from scipy import stats


def trial_counts(spike_times, onsets, t0, t1):
    """Spikes fired by one unit in [onset+t0, onset+t1) for every trial."""
    st = np.asarray(spike_times)
    lo = np.searchsorted(st, onsets + t0, side="left")
    hi = np.searchsorted(st, onsets + t1, side="left")
    return (hi - lo).astype(float)


def psth(spike_times, onsets, window=(-0.1, 0.2), binsize=0.002):
    """Mean firing rate (Hz) of one unit around events, plus the bin centers."""
    edges = np.arange(window[0], window[1] + binsize / 2, binsize)
    st = np.asarray(spike_times)
    lo = np.searchsorted(st, onsets + window[0])
    hi = np.searchsorted(st, onsets + window[1])
    rel = np.concatenate([st[a:b] - o for a, b, o in zip(lo, hi, onsets)]) if len(onsets) else np.array([])
    h, _ = np.histogram(rel, edges)
    return edges[:-1] + binsize / 2, h / len(onsets) / binsize


def population_psth(spikes, onsets, window=(-0.1, 0.2), binsize=0.002):
    """(n_units, n_bins) array of PSTHs plus bin centers."""
    out = []
    for u in spikes.keys():
        t, r = psth(spikes[u].t, onsets, window, binsize)
        out.append(r)
    return t, np.array(out)


def session_tuning(spikes, onsets, freq, evoked=(0.010, 0.060), base=(-0.100, -0.010)):
    """Per-unit frequency tuning for one session.

    Returns a dict of arrays indexed by (unit, frequency) or (unit,):
        ufreq        : the unique tone frequencies, ascending
        evoked_rate  : baseline-subtracted rate (Hz) in the response window
        raw_rate     : rate (Hz) in the response window
        base_rate    : rate (Hz) in the pre-tone baseline window
        sem          : s.e.m. of the evoked rate across trials
        p_anova      : one-way ANOVA across frequencies on per-trial evoked counts
        p_driven     : paired test, response window vs baseline, pooled over tones
        bf           : best frequency (Hz), argmax of evoked_rate
        n_trials     : trials per frequency
    """
    ufreq = np.unique(freq)
    dur_e = evoked[1] - evoked[0]
    dur_b = base[1] - base[0]
    uids = list(spikes.keys())

    evoked_rate = np.zeros((len(uids), len(ufreq)))
    raw_rate = np.zeros_like(evoked_rate)
    sem = np.zeros_like(evoked_rate)
    base_rate = np.zeros(len(uids))
    p_anova = np.ones(len(uids))
    p_driven = np.ones(len(uids))

    masks = [freq == f for f in ufreq]
    for i, u in enumerate(uids):
        st = spikes[u].t
        ce = trial_counts(st, onsets, *evoked) / dur_e
        cb = trial_counts(st, onsets, *base) / dur_b
        d = ce - cb
        base_rate[i] = cb.mean()
        groups = [d[m] for m in masks]
        raw_rate[i] = [ce[m].mean() for m in masks]
        evoked_rate[i] = [g.mean() for g in groups]
        sem[i] = [g.std(ddof=1) / np.sqrt(len(g)) for g in groups]
        # A unit that never fires in either window has no F statistic to report.
        # When one tone's group is constant the within-group variance is zero and
        # F overflows to infinity, which is the correct answer (p = 0) but makes
        # numpy complain, so the overflow is silenced for this call only.
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            p_anova[i] = 1.0 if np.ptp(d) == 0 else stats.f_oneway(*groups).pvalue
        p_driven[i] = stats.wilcoxon(ce, cb, zero_method="zsplit",
                                     method="approx").pvalue

    return dict(
        ufreq=ufreq,
        evoked_rate=evoked_rate,
        raw_rate=raw_rate,
        base_rate=base_rate,
        sem=sem,
        p_anova=p_anova,
        p_driven=p_driven,
        bf=ufreq[np.argmax(evoked_rate, axis=1)],
        n_trials=np.array([m.sum() for m in masks]),
        unit_ids=np.array(uids),
    )


def bandwidth_octaves(evoked_rate, ufreq, thresh=0.5):
    """Width (octaves) of the tuning curve at `thresh` of the peak evoked rate.

    Interpolates the crossing points on a log2-frequency axis on either side of
    the peak. Returns NaN when the curve does not fall below threshold inside
    the tested range (i.e. the width is not bounded by the stimulus set).
    """
    x = np.log2(ufreq)
    out = np.full(evoked_rate.shape[0], np.nan)
    for i, y in enumerate(evoked_rate):
        pk = int(np.argmax(y))
        if y[pk] <= 0:
            continue
        lvl = thresh * y[pk]
        # walk left
        left = None
        for j in range(pk, 0, -1):
            if y[j - 1] < lvl:
                left = x[j] - (y[j] - lvl) / (y[j] - y[j - 1]) * (x[j] - x[j - 1])
                break
        right = None
        for j in range(pk, len(y) - 1):
            if y[j + 1] < lvl:
                right = x[j] + (y[j] - lvl) / (y[j] - y[j + 1]) * (x[j + 1] - x[j])
                break
        if left is not None and right is not None:
            out[i] = right - left
    return out


def permutation_tuning_test(spike_times, onsets, freq, evoked, base, n_perm=1000, rng=None):
    """Shuffle tone labels to get a null distribution for the tuning modulation.

    Statistic: range of the mean evoked rate across frequencies. Returns
    (observed, p_value).
    """
    rng = np.random.default_rng(0 if rng is None else rng)
    ce = trial_counts(spike_times, onsets, *evoked) / (evoked[1] - evoked[0])
    cb = trial_counts(spike_times, onsets, *base) / (base[1] - base[0])
    d = ce - cb
    ufreq = np.unique(freq)
    codes = np.searchsorted(ufreq, freq)
    n = np.bincount(codes, minlength=len(ufreq))

    def stat(c):
        m = np.bincount(c, weights=d, minlength=len(ufreq)) / n
        return m.max() - m.min()

    obs = stat(codes)
    null = np.array([stat(rng.permutation(codes)) for _ in range(n_perm)])
    return obs, (1 + (null >= obs).sum()) / (1 + n_perm)
