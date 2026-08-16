"""Session-level auditory frequency tuning analysis for DANDI 000986.

Computes per-unit per-trial spike counts in baseline and response windows,
tuning curves, tone responsiveness (Wilcoxon), frequency selectivity
(Kruskal-Wallis), best frequency, and selectivity indices.
"""

import numpy as np
import pandas as pd
from scipy import stats

# Analysis windows (seconds relative to tone onset)
RESP_WIN = (0.005, 0.060)   # 55 ms response window
BASE_WIN = (-0.050, 0.005)  # 55 ms baseline window
RESP_WIDTH = RESP_WIN[1] - RESP_WIN[0]
BASE_WIDTH = BASE_WIN[1] - BASE_WIN[0]

ALPHA_RESP = 0.01   # Wilcoxon p for tone-responsive
ALPHA_TUNED = 0.01  # KW p for frequency-tuned


def trial_spike_counts(sorted_spikes, onsets, win):
    """Vectorized per-trial spike counts via searchsorted.

    sorted_spikes: sorted 1-D array of spike times for one unit.
    onsets: sorted 1-D array of trial onset times.
    """
    lo = np.searchsorted(sorted_spikes, onsets + win[0], side="left")
    hi = np.searchsorted(sorted_spikes, onsets + win[1], side="left")
    return hi - lo


def compute_session_tuning(spike_times_list, onsets, freq_per_trial, frqs,
                           subject_id="-", session_id="-"):
    """Compute per-unit tuning metrics for one session.

    spike_times_list: list of sorted spike arrays, one per unit.
    onsets: (n_trials,) ascending trial onset times.
    freq_per_trial: (n_trials,) frequency of each trial.
    frqs: (n_freqs,) the set of stimulus frequencies.

    Returns dict with per-session arrays and a metrics DataFrame.
    """
    n_units = len(spike_times_list)
    n_trials = len(onsets)
    if frqs is None:
        frqs = np.sort(np.unique(freq_per_trial))
    n_freqs = len(frqs)

    counts_resp = np.zeros((n_units, n_trials), dtype=int)
    counts_base = np.zeros((n_units, n_trials), dtype=int)
    for ku, st in enumerate(spike_times_list):
        st = np.sort(st)
        counts_resp[ku] = trial_spike_counts(st, onsets, RESP_WIN)
        counts_base[ku] = trial_spike_counts(st, onsets, BASE_WIN)
    rates_resp = counts_resp / RESP_WIDTH
    rates_base = counts_base / BASE_WIDTH

    # per-frequency mean net rate (baseline-subtracted)
    rates_net = np.zeros((n_units, n_freqs))
    rates_resp_f = np.zeros((n_units, n_freqs))
    rates_base_f = np.zeros((n_units, n_freqs))
    for j, frq in enumerate(frqs):
        mask = freq_per_trial == frq
        rates_resp_f[:, j] = rates_resp[:, mask].mean(axis=1)
        rates_base_f[:, j] = rates_base[:, mask].mean(axis=1)
        rates_net[:, j] = rates_resp_f[:, j] - rates_base_f[:, j]

    # per-trial net counts for statistics
    net_counts = counts_resp - counts_base

    # ---- per-unit statistics ----
    resp_p = np.full(n_units, np.nan)
    tuned_p = np.full(n_units, np.nan)
    for i in range(n_units):
        with np.errstate(invalid="ignore"):
            if net_counts[i].var() > 0 and counts_base[i].var() > 0:
                resp_p[i] = stats.wilcoxon(counts_resp[i], counts_base[i],
                                           zero_method="wilcox").pvalue
            else:
                resp_p[i] = 1.0
            if net_counts[i].var() > 0:
                grp = [net_counts[i][freq_per_trial == frq] for frq in frqs]
                tuned_p[i] = stats.kruskal(*grp).pvalue
            else:
                tuned_p[i] = 1.0

    responsive = resp_p < ALPHA_RESP
    tuned = (tuned_p < ALPHA_TUNED) & responsive

    bf_idx = np.argmax(rates_net, axis=1)
    bf_freq = frqs[bf_idx]
    max_rate = rates_net[np.arange(n_units), bf_idx]

    # selectivity: number of freqs at/above 50% of peak net rate
    half_max = max_rate[:, None] / 2
    n_above_half = (rates_net >= half_max).sum(axis=1)

    # suppression: tuned unit whose min net rate < 0 and peak is positive
    min_rate = rates_net.min(axis=1)
    suppressed = tuned & (min_rate < -max_rate * 0.1)

    # spontaneous / baseline activity
    base_rate_all = rates_base.mean(axis=1)

    metrics = pd.DataFrame({
        "subject": subject_id,
        "session": session_id,
        "unit": np.arange(n_units),
        "n_spikes": [len(s) for s in spike_times_list],
        "spontaneous": base_rate_all,
        "peak_rate": max_rate,
        "bf_idx": bf_idx,
        "bf_freq": bf_freq,
        "tuned_p": tuned_p,
        "responsive_p": resp_p,
        "responsive": responsive,
        "tuned": tuned,
        "n_above_half": n_above_half,
        "suppressed": suppressed,
    })

    return {
        "frqs": frqs,
        "n_trials": n_trials,
        "counts_resp": counts_resp,
        "counts_base": counts_base,
        "net_counts": net_counts,
        "rates_base": rates_base_f,
        "rates_resp": rates_resp_f,
        "rates_net": rates_net,
        "bf_idx": bf_idx,
        "bf_freq": bf_freq,
        "metrics": metrics,
    }