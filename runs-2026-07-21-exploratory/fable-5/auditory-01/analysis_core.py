"""Per-session tuning analysis for DANDI 000986, shared by the scripts and the notebook."""

import numpy as np
from scipy import stats

import dandi_auditory as da


def sparseness(r):
    """Lifetime sparseness (Rolls & Tovee) over the n frequencies. 0 = flat, 1 = one freq.

    Computed on absolute evoked firing rates, which are non-negative. Applying it to
    baseline-subtracted rates would require clipping the suppressed frequencies to zero
    and would push most units to exactly 1.
    """
    r = np.clip(r, 0, None)
    n = len(r)
    denom = (r ** 2).mean()
    if denom == 0:
        return np.nan
    return (1 - (r.mean() ** 2) / denom) / (1 - 1 / n)


def analyze_session(asset_id):
    """Stream one session and return its tuning statistics.

    The split-half fields (bf_odd / bf_even / tuning_even) support the control against
    best-frequency selection bias: pick the BF on odd trials, measure tuning on even ones.
    """
    s = da.load_session(asset_id)
    units, freqs, freq = s["units"], s["freqs"], s["frequency"]
    onsets = s["trials"].start
    n_units = len(units)

    base = da.trial_spike_counts(units, onsets, da.BASELINE_WINDOW)
    evok = da.trial_spike_counts(units, onsets, da.EVOKED_WINDOW)
    base_rate = base / (da.BASELINE_WINDOW[1] - da.BASELINE_WINDOW[0])
    evok_rate = evok / (da.EVOKED_WINDOW[1] - da.EVOKED_WINDOW[0])
    delta = evok_rate - base_rate

    tuning = np.stack([delta[freq == f].mean(axis=0) for f in freqs])
    tuning_sem = np.stack([stats.sem(delta[freq == f], axis=0) for f in freqs])
    tuning_abs = np.stack([evok_rate[freq == f].mean(axis=0) for f in freqs])

    responsive_p = np.array([
        stats.wilcoxon(evok[:, i], base[:, i])[1]
        if (evok[:, i] - base[:, i]).any() else 1.0
        for i in range(n_units)
    ])
    tuned_p = np.array([
        stats.kruskal(*[delta[freq == f, i] for f in freqs])[1] for i in range(n_units)
    ])

    odd = np.zeros(len(onsets), bool)
    odd[::2] = True
    tuning_odd = np.stack([delta[odd & (freq == f)].mean(axis=0) for f in freqs])
    tuning_even = np.stack([delta[~odd & (freq == f)].mean(axis=0) for f in freqs])

    return dict(
        subject=s["subject"], session=s["session_id"], n_units=n_units,
        freqs=freqs, tuning=tuning, tuning_sem=tuning_sem, tuning_abs=tuning_abs,
        responsive_p=responsive_p, tuned_p=tuned_p,
        bf=np.argmax(tuning, axis=0),
        bf_odd=np.argmax(tuning_odd, axis=0), bf_even=np.argmax(tuning_even, axis=0),
        tuning_even=tuning_even,
        psth=da.psth_by_frequency(units, onsets, freq, freqs),
        baseline=base_rate.mean(axis=0), evoked=evok_rate.mean(axis=0),
        sparseness=np.array([sparseness(tuning_abs[:, i]) for i in range(n_units)]),
        unit_ids=np.array(list(units.keys())),
    )
