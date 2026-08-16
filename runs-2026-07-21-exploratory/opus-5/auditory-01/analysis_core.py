"""Core computations for tone-evoked frequency tuning in auditory cortex."""
import numpy as np
import pandas as pd
from scipy import stats


def trial_counts(units, onsets, win):
    """Spike counts for every (trial, unit) in a window relative to tone onset.

    units  : pynapple TsGroup
    onsets : 1-D array of tone onset times (s)
    win    : (t_start, t_stop) in seconds relative to onset
    returns (n_trials, n_units) integer array
    """
    a, b = np.asarray(onsets) + win[0], np.asarray(onsets) + win[1]
    out = np.empty((len(a), len(units)), dtype=np.int32)
    for j, u in enumerate(units.index):
        st = units[u].t
        out[:, j] = np.searchsorted(st, b) - np.searchsorted(st, a)
    return out


def psth(units, onsets, tmin=-0.1, tmax=0.3, bin_size=0.005):
    """Trial-averaged firing rate (Hz) per unit on a common time axis."""
    edges = np.arange(tmin, tmax + bin_size / 2, bin_size)
    centers = edges[:-1] + bin_size / 2
    rates = np.zeros((len(units), len(centers)))
    onsets = np.asarray(onsets)
    for j, u in enumerate(units.index):
        st = units[u].t
        rel = []
        lo = np.searchsorted(st, onsets + tmin)
        hi = np.searchsorted(st, onsets + tmax)
        for k in range(len(onsets)):
            rel.append(st[lo[k]:hi[k]] - onsets[k])
        rel = np.concatenate(rel) if len(rel) else np.array([])
        h, _ = np.histogram(rel, bins=edges)
        rates[j] = h / (len(onsets) * bin_size)
    return centers, rates


def tuning_table(counts_ev, counts_bl, freqs, win_ev, win_bl, unit_ids):
    """Per-unit tuning curves and statistics.

    counts_ev/counts_bl : (n_trials, n_units) spike counts
    freqs : (n_trials,) stimulus frequency per trial
    returns (curves DataFrame [units x freqs, Hz evoked], stats DataFrame)
    """
    ufreq = np.unique(freqs)
    dur_ev, dur_bl = win_ev[1] - win_ev[0], win_bl[1] - win_bl[0]
    rate_ev = counts_ev / dur_ev
    rate_bl = counts_bl / dur_bl
    delta = rate_ev - rate_bl  # baseline-subtracted, per trial

    curves = np.zeros((counts_ev.shape[1], len(ufreq)))
    sems = np.zeros_like(curves)
    for i, f in enumerate(ufreq):
        m = freqs == f
        curves[:, i] = delta[m].mean(0)
        sems[:, i] = delta[m].std(0) / np.sqrt(m.sum())

    rows = []
    for j in range(counts_ev.shape[1]):
        # responsiveness: evoked vs baseline across all trials
        try:
            _, p_resp = stats.wilcoxon(counts_ev[:, j], counts_bl[:, j])
        except ValueError:                       # all differences zero
            p_resp = 1.0
        groups = [delta[freqs == f, j] for f in ufreq]
        _, p_tune = stats.kruskal(*groups)
        c = curves[j]
        bf = ufreq[np.argmax(c)]
        pos = np.clip(c, 0, None)
        # lifetime sparseness (Rolls & Tovee) on rectified tuning curve
        n = len(pos)
        sparseness = ((1 - (pos.sum() / n) ** 2 / max((pos ** 2).sum() / n, 1e-12)) /
                      (1 - 1.0 / n)) if pos.sum() > 0 else np.nan
        rows.append(dict(
            unit=unit_ids[j], baseline_hz=rate_bl[:, j].mean(),
            peak_evoked_hz=c.max(), min_evoked_hz=c.min(),
            best_frequency=bf, p_responsive=p_resp, p_tuned=p_tune,
            sparseness=sparseness,
            depth_of_tuning=(c.max() - c.min()) / (abs(c.max()) + abs(c.min()) + 1e-12),
        ))
    curves_df = pd.DataFrame(curves, index=unit_ids, columns=ufreq)
    sems_df = pd.DataFrame(sems, index=unit_ids, columns=ufreq)
    return curves_df, sems_df, pd.DataFrame(rows).set_index("unit")


def fdr(pvals, q=0.05):
    """Benjamini-Hochberg: returns boolean array of rejections."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    thresh = q * (np.arange(1, n + 1)) / n
    passed = p[order] <= thresh
    k = np.where(passed)[0].max() + 1 if passed.any() else 0
    rej = np.zeros(n, bool)
    rej[order[:k]] = True
    return rej
