"""Per-unit frequency-tuning statistics for DANDI:000986."""
import numpy as np
import pandas as pd
from scipy import stats

from common import (BASELINE_WIN, EVOKED_WIN, FREQS, evoked_rates,
                    lifetime_sparseness, load_session, tuning_from_rates)


def fdr_bh(p):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, dtype=float)
    n = p.size
    order = np.argsort(p)
    adj = np.empty(n)
    adj[order] = np.minimum.accumulate((p[order] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.clip(adj, 0, 1)


def psth_matrix(spikes, onsets, edges):
    """(n_units, n_bins) spike counts summed over `onsets`."""
    out = np.zeros((len(spikes), edges.size - 1))
    for row, u in enumerate(spikes.index):
        st = spikes[u].t
        idx = np.searchsorted(st, onsets[:, None] + edges[None, :])
        out[row] = np.diff(idx, axis=1).sum(0)
    return out


def onset_latency(spikes, onsets_by_bf, bin_size=0.002, thresh_sd=3.0, n_consec=3):
    """First post-onset bin where the BF response exceeds baseline + 3 SD.

    `onsets_by_bf` maps unit index -> tone onsets at that unit's best frequency.
    """
    edges = np.arange(-0.100, 0.100 + bin_size / 2, bin_size)
    centers = edges[:-1] + bin_size / 2
    pre = centers < 0
    lat = np.full(len(spikes), np.nan)
    for row, u in enumerate(spikes.index):
        st = spikes[u].t
        ons = onsets_by_bf[u]
        idx = np.searchsorted(st, ons[:, None] + edges[None, :])
        counts = np.diff(idx, axis=1).sum(0) / (len(ons) * bin_size)
        mu, sd = counts[pre].mean(), counts[pre].std()
        if sd == 0:
            continue
        above = (counts > mu + thresh_sd * sd) & (centers > 0)
        run = np.convolve(above.astype(int), np.ones(n_consec, int), "valid")
        hit = np.flatnonzero(run == n_consec)
        if hit.size:
            lat[row] = centers[hit[0]]
    return lat


def analyze_session(name, asset_id, sess=None, rng_seed=0, with_latency=True):
    """Frequency tuning for every unit in one session.

    Returns (units_df, tuning_mean, tuning_sem, session_dict) where the tuning
    arrays are (n_freqs, n_units) baseline-subtracted evoked rates in Hz.
    """
    if sess is None:
        sess = load_session(name, asset_id)
    trials, spikes = sess["trials"], sess["spikes"]
    freq_of_trial = trials.stim_frequency.values
    ev, bl = evoked_rates(spikes, trials)
    driven = ev - bl                                  # per-trial baseline subtraction
    tmean, tsem = tuning_from_rates(driven, freq_of_trial)
    tabs, _ = tuning_from_rates(ev, freq_of_trial)    # absolute evoked rate

    n_units = ev.shape[1]
    groups = [driven[freq_of_trial == f] for f in FREQS]

    p_resp = np.array([stats.wilcoxon(ev[:, i], bl[:, i],
                                      zero_method="zsplit").pvalue
                       for i in range(n_units)])
    p_freq = np.array([stats.kruskal(*[g[:, i] for g in groups]).pvalue
                       for i in range(n_units)])

    bf_idx = np.argmax(tmean, axis=0)
    cols = np.arange(n_units)
    best = tmean[bf_idx, cols]
    # Frequency selectivity index on absolute evoked rates, bounded in [0, 1].
    hi, lo = tabs.max(axis=0), tabs.min(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        fsi = np.where(hi + lo > 0, (hi - lo) / (hi + lo), np.nan)
    sparse = np.array([lifetime_sparseness(tmean[:, i]) for i in range(n_units)])

    # Split-half reliability: is the preferred frequency stable across trials?
    rng = np.random.default_rng(rng_seed)
    perm = rng.permutation(len(trials))
    half_a, half_b = perm[::2], perm[1::2]
    ta, _ = tuning_from_rates(driven[half_a], freq_of_trial[half_a])
    tb, _ = tuning_from_rates(driven[half_b], freq_of_trial[half_b])
    bf_a, bf_b = np.argmax(ta, axis=0), np.argmax(tb, axis=0)
    with np.errstate(invalid="ignore"):
        half_r = np.array([np.corrcoef(ta[:, i], tb[:, i])[0, 1] for i in range(n_units)])

    units = pd.DataFrame({
        "session": name,
        "subject": sess["subject"],
        "unit": np.asarray(spikes.index),
        "mean_rate_hz": [len(spikes[u]) / spikes.time_support.tot_length()
                         for u in spikes.index],
        "baseline_hz": bl.mean(0),
        "evoked_hz": ev.mean(0),
        "driven_hz": driven.mean(0),
        "p_responsive": p_resp,
        "p_frequency": p_freq,
        "bf_hz": FREQS[bf_idx],
        "bf_idx": bf_idx,
        "best_hz": best,
        "selectivity_index": fsi,
        "sparseness": sparse,
        "bf_split_a": FREQS[bf_a],
        "bf_split_b": FREQS[bf_b],
        "bf_split_match": bf_a == bf_b,
        "split_half_r": half_r,
    })
    units["q_responsive"] = fdr_bh(p_resp)
    units["q_frequency"] = fdr_bh(p_freq)
    units["responsive"] = units.q_responsive < 0.01
    units["enhanced"] = units.responsive & (units.driven_hz > 0)
    units["suppressed"] = units.responsive & (units.driven_hz <= 0)
    units["freq_tuned"] = units.enhanced & (units.q_frequency < 0.01)

    if with_latency:
        onsets_by_bf = {u: trials.start_time.values[freq_of_trial == FREQS[b]]
                        for u, b in zip(spikes.index, bf_idx)}
        units["latency_s"] = onset_latency(spikes, onsets_by_bf)
    return units, tmean, tsem, sess
