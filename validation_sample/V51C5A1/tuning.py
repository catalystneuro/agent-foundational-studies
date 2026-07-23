"""Core computations for auditory frequency tuning in DANDI:000986.

The stimulus set is five pure tones (2, 4, 8, 16, 32 kHz), 25 ms long, 60 dB SPL,
delivered in random order roughly every 0.8 s while the mouse sits passively.

Everything downstream is derived from one object: a per-unit, per-trial matrix of
spike counts in fine time bins around tone onset.  Evoked and baseline windows are
chosen to fall exactly on bin edges so the same matrix serves both the PSTHs and
the per-trial rate measurements.
"""

import numpy as np
import pynapple as nap
from scipy import stats

FREQS = np.array([2000.0, 4000.0, 8000.0, 16000.0, 32000.0])

PSTH_WIN = (-0.100, 0.300)  # s relative to tone onset
PSTH_BIN = 0.005  # s
BASELINE_WIN = (-0.100, 0.000)
EVOKED_WIN = (0.010, 0.060)  # A1 onset response to a 25 ms tone

MIN_RATE_HZ = 0.5  # units below this session-wide rate are dropped


def psth_edges():
    n = int(round((PSTH_WIN[1] - PSTH_WIN[0]) / PSTH_BIN))
    return PSTH_WIN[0] + PSTH_BIN * np.arange(n + 1)


def _window_bins(win):
    """Bin indices covering `win`; both windows land exactly on 5 ms edges."""
    edges = psth_edges()
    lo = int(np.round((win[0] - PSTH_WIN[0]) / PSTH_BIN))
    hi = int(np.round((win[1] - PSTH_WIN[0]) / PSTH_BIN))
    assert np.isclose(edges[lo], win[0]) and np.isclose(edges[hi], win[1])
    return lo, hi


def trial_bin_counts(spike_times, onsets):
    """Spike counts in (n_trials, n_bins) around each onset, for one unit.

    Tones are >=0.8 s apart and the window is 0.4 s wide, so every spike falls in
    at most one trial window; that lets us assign spikes to trials in one pass
    instead of looping over 7000+ trials.
    """
    edges = psth_edges()
    n_trials, n_bins = len(onsets), len(edges) - 1
    if spike_times.size == 0:
        return np.zeros((n_trials, n_bins), dtype=np.int16)

    j = np.searchsorted(onsets, spike_times)
    # nearest onset is either the one before or the one after each spike
    j_lo = np.clip(j - 1, 0, n_trials - 1)
    j_hi = np.clip(j, 0, n_trials - 1)
    d_lo = np.abs(spike_times - onsets[j_lo])
    d_hi = np.abs(spike_times - onsets[j_hi])
    j_near = np.where(d_lo <= d_hi, j_lo, j_hi)

    rel = spike_times - onsets[j_near]
    keep = (rel >= edges[0]) & (rel < edges[-1])
    if not keep.any():
        return np.zeros((n_trials, n_bins), dtype=np.int16)

    bin_idx = np.floor((rel[keep] - edges[0]) / PSTH_BIN).astype(np.int64)
    flat = np.bincount(
        j_near[keep] * n_bins + bin_idx, minlength=n_trials * n_bins
    )
    return flat.reshape(n_trials, n_bins).astype(np.int16)


def session_response_matrices(spikes, onsets, freqs):
    """Reduce a session to the arrays every later analysis needs.

    Returns a dict with
      psth      (n_units, n_freqs, n_bins)  trial-averaged firing rate, Hz
      evoked    (n_trials, n_units)         spike count in EVOKED_WIN
      baseline  (n_trials, n_units)         spike count in BASELINE_WIN
    """
    lo_e, hi_e = _window_bins(EVOKED_WIN)
    lo_b, hi_b = _window_bins(BASELINE_WIN)
    unit_ids = np.asarray(spikes.index)
    n_units, n_bins = len(unit_ids), len(psth_edges()) - 1

    freq_masks = [freqs == f for f in FREQS]
    psth = np.zeros((n_units, len(FREQS), n_bins))
    evoked = np.zeros((len(onsets), n_units), dtype=np.int16)
    baseline = np.zeros((len(onsets), n_units), dtype=np.int16)

    for u, uid in enumerate(unit_ids):
        m = trial_bin_counts(np.asarray(spikes[uid].t), onsets)
        for k, mask in enumerate(freq_masks):
            psth[u, k] = m[mask].mean(axis=0) / PSTH_BIN
        evoked[:, u] = m[:, lo_e:hi_e].sum(axis=1)
        baseline[:, u] = m[:, lo_b:hi_b].sum(axis=1)

    return {
        "unit_ids": unit_ids,
        "psth": psth,
        "evoked": evoked,
        "baseline": baseline,
        "freqs": freqs,
        "onsets": onsets,
    }


def unit_statistics(res):
    """Per-unit tuning descriptors and significance tests."""
    ev_dur = EVOKED_WIN[1] - EVOKED_WIN[0]
    bl_dur = BASELINE_WIN[1] - BASELINE_WIN[0]
    ev_rate = res["evoked"] / ev_dur  # Hz, per trial
    bl_rate = res["baseline"] / bl_dur
    dr = ev_rate - bl_rate  # baseline-subtracted evoked rate
    freqs = res["freqs"]
    groups = [np.flatnonzero(freqs == f) for f in FREQS]

    n_units = ev_rate.shape[1]
    out = {
        "unit_ids": res["unit_ids"],
        "tc_evoked": np.zeros((n_units, len(FREQS))),  # raw evoked rate
        "tc_delta": np.zeros((n_units, len(FREQS))),  # baseline-subtracted
        "tc_sem": np.zeros((n_units, len(FREQS))),
        "baseline_rate": bl_rate.mean(axis=0),
        "p_tuning": np.ones(n_units),
        "p_responsive": np.ones(n_units),
        "omega2": np.zeros(n_units),
        "bf_idx": np.zeros(n_units, dtype=int),
        "selectivity": np.zeros(n_units),
    }

    for u in range(n_units):
        by_freq = [dr[g, u] for g in groups]
        out["tc_evoked"][u] = [ev_rate[g, u].mean() for g in groups]
        out["tc_delta"][u] = [x.mean() for x in by_freq]
        out["tc_sem"][u] = [x.std(ddof=1) / np.sqrt(len(x)) for x in by_freq]

        # tone-evoked at all? (paired, across every trial)
        out["p_responsive"][u] = stats.wilcoxon(
            ev_rate[:, u], bl_rate[:, u], zero_method="zsplit"
        ).pvalue
        # does the response depend on frequency?
        out["p_tuning"][u] = stats.kruskal(*by_freq).pvalue
        out["omega2"][u] = _omega_squared(by_freq)

        tc = out["tc_evoked"][u]
        out["bf_idx"][u] = int(np.argmax(out["tc_delta"][u]))
        out["selectivity"][u] = (
            (tc.max() - tc.min()) / tc.max() if tc.max() > 0 else 0.0
        )

    out["bf_hz"] = FREQS[out["bf_idx"]]
    return out


def _omega_squared(groups):
    """Effect size of frequency on firing rate (fraction of variance explained)."""
    allv = np.concatenate(groups)
    n, k = allv.size, len(groups)
    grand = allv.mean()
    ss_between = sum(g.size * (g.mean() - grand) ** 2 for g in groups)
    ss_total = ((allv - grand) ** 2).sum()
    ms_within = (ss_total - ss_between) / (n - k)
    denom = ss_total + ms_within
    return max(0.0, (ss_between - (k - 1) * ms_within) / denom) if denom > 0 else 0.0


def fdr_bh(pvals, alpha=0.05):
    """Benjamini-Hochberg; returns boolean reject vector."""
    p = np.asarray(pvals)
    order = np.argsort(p)
    m = p.size
    thresh = alpha * np.arange(1, m + 1) / m
    passed = p[order] <= thresh
    reject = np.zeros(m, dtype=bool)
    if passed.any():
        reject[order[: np.flatnonzero(passed)[-1] + 1]] = True
    return reject


def poisson_template_decoder(counts, labels, n_folds=5, seed=0):
    """Cross-validated naive-Bayes decoding of tone frequency from population counts.

    The templates are exactly the tuning curves measured on the training trials, so
    decoding performance is a direct read-out of how much frequency information the
    tuning curves carry.
    """
    counts = np.asarray(counts, dtype=float)
    y = np.searchsorted(FREQS, labels)
    rng = np.random.default_rng(seed)
    fold = rng.permutation(len(y)) % n_folds
    pred = np.empty_like(y)

    for f in range(n_folds):
        tr, te = fold != f, fold == f
        # mean count per frequency per unit, on training trials only
        lam = np.stack([counts[tr & (y == k)].mean(axis=0) for k in range(len(FREQS))])
        lam = np.clip(lam, 1e-3, None)
        prior = np.log([np.mean(y[tr] == k) for k in range(len(FREQS))])
        # log P(counts | freq) up to a term constant across frequencies.
        # errstate: numpy on Apple Accelerate raises spurious FP flags from matmul;
        # the products are verified finite, so the flags are ignored here only.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            ll = counts[te] @ np.log(lam).T - lam.sum(axis=1)[None, :] + prior[None, :]
        assert np.isfinite(ll).all()
        pred[te] = np.argmax(ll, axis=1)

    conf = np.zeros((len(FREQS), len(FREQS)))
    for t, p in zip(y, pred):
        conf[t, p] += 1
    conf /= conf.sum(axis=1, keepdims=True)
    return {"accuracy": float(np.mean(pred == y)), "confusion": conf,
            "y_true": y, "y_pred": pred}


def decoding_vs_population_size(counts, labels, sizes, n_rep=10, seed=0):
    """Decoding accuracy as a function of how many units the decoder sees."""
    rng = np.random.default_rng(seed)
    n_units = counts.shape[1]
    out = {}
    for s in sizes:
        if s > n_units:
            continue
        accs = [
            poisson_template_decoder(
                counts[:, rng.choice(n_units, s, replace=False)], labels, seed=r
            )["accuracy"]
            for r in range(n_rep)
        ]
        out[s] = (float(np.mean(accs)), float(np.std(accs)))
    return out


def split_half_reliability(res, seed=0):
    """Correlate each unit's tuning curve between two random halves of the trials.

    A real tuning curve reproduces on independent trials; noise does not.  The
    shuffled version breaks the frequency labels and gives the null distribution.
    """
    rng = np.random.default_rng(seed)
    freqs = res["freqs"]
    ev = res["evoked"].astype(float)
    half = rng.random(len(freqs)) < 0.5

    def _tc(mask, f):
        return np.stack([ev[mask & (f == k)].mean(axis=0) for k in FREQS])

    def _corr(f):
        a, b = _tc(half, f), _tc(~half, f)
        a = a - a.mean(axis=0)
        b = b - b.mean(axis=0)
        denom = np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0)
        return np.where(denom > 0, (a * b).sum(axis=0) / np.where(denom > 0, denom, 1), 0.0)

    return _corr(freqs), _corr(rng.permutation(freqs))


def load_session_arrays(nwb, nwbfile):
    """Pull the pieces of a session we analyse, as pynapple objects / arrays."""
    trials = nwb["trials"]
    spikes = nwb["units"]
    onsets = np.asarray(trials.start)
    freqs = np.asarray(trials.stim_frequency)
    order = np.argsort(onsets)
    onsets, freqs = onsets[order], freqs[order]

    rates = np.array([len(spikes[i]) / spikes.time_support.tot_length() for i in spikes.index])
    keep = rates >= MIN_RATE_HZ
    spikes = spikes[list(np.asarray(spikes.index)[keep])]
    return spikes, onsets, freqs, rates[keep]
