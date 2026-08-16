"""Frequency tuning of mouse auditory cortex units (DANDI:000986).

Head-fixed mice were passively exposed to 25 ms pure tones at 2, 4, 8, 16 and 32 kHz
(60 dB SPL, ~1500 repeats each per session) while Neuropixels probes recorded auditory
cortex. For every unit we measure the tone-evoked rate as a function of frequency, test
whether that rate depends on frequency, and read off the best frequency.
"""

import numpy as np
import pynapple as nap
from scipy import stats

import dandi_io as dio

nap.nap_config.suppress_conversion_warnings = True

EVOKED = (0.010, 0.060)    # tone-evoked window, from the population PSTH
BASELINE = (-0.100, 0.0)   # pre-tone baseline (inter-tone interval is ~0.8 s)
MIN_RATE = 0.5             # Hz, exclude near-silent units
N_PERM = 2000              # permutations for the frequency-selectivity test


def session_to_pynapple(sess):
    """TsGroup of units plus IntervalSets for the evoked and baseline windows."""
    spikes = nap.TsGroup({int(u): nap.Ts(t) for u, t in sess["spikes"].items() if len(t) > 1})
    onset = sess["tone_onset"]
    evoked_ep = nap.IntervalSet(start=onset + EVOKED[0], end=onset + EVOKED[1])
    base_ep = nap.IntervalSet(start=onset + BASELINE[0], end=onset + BASELINE[1])
    return spikes, evoked_ep, base_ep


def trial_counts(spikes, ep, win):
    """(n_trials, n_units) spike counts, one row per interval of `ep`."""
    width = win[1] - win[0]
    return np.asarray(spikes.count(bin_size=width + 1e-9, ep=ep).values)


def population_psth(spikes, onset, window=(-0.15, 0.25), bin_size=0.002):
    """Mean firing rate of every unit around tone onset. Returns (t, rates[n_units, n_bins])."""
    peri = nap.compute_perievent(spikes, nap.Ts(onset), window=window)
    rates, t = [], None
    for u in spikes.index:
        c = peri[u].count(bin_size)
        t = c.t
        rates.append(np.asarray(c.values).sum(axis=1) / (len(onset) * bin_size))
    return t, np.array(rates)


def psth_by_frequency(spikes, onset, freq, window=(-0.05, 0.15), bin_size=0.002):
    """Per-unit PSTH computed separately for each tone frequency."""
    freqs = np.unique(freq)
    out = {}
    for fq in freqs:
        t, r = population_psth(spikes, onset[freq == fq], window=window, bin_size=bin_size)
        out[fq] = r
    return t, out, freqs


def selectivity(driven):
    """Sparseness-based selectivity index over a unit's frequency response profile.

    1 - (sum r)^2 / (n sum r^2), normalised to [0, 1]; 0 = equal response to every
    frequency, 1 = response to a single frequency. Negative rates are clipped to 0.
    """
    r = np.clip(driven, 0, None)
    n = len(r)
    if r.sum() <= 0:
        return np.nan
    s = 1 - (r.sum() ** 2) / (n * np.sum(r**2))
    return s / (1 - 1 / n)


def analyze_session(sess, rng=None):
    """Per-unit tuning curves and statistics for one session."""
    rng = rng or np.random.default_rng(0)
    spikes, evoked_ep, base_ep = session_to_pynapple(sess)
    freq = sess["frequency"]
    freqs = np.unique(freq)

    n_ev = trial_counts(spikes, evoked_ep, EVOKED)
    n_bs = trial_counts(spikes, base_ep, BASELINE)
    assert np.isfinite(n_ev).all() and np.isfinite(n_bs).all(), "non-finite spike counts"
    rate_ev = n_ev / (EVOKED[1] - EVOKED[0])
    rate_bs = n_bs / (BASELINE[1] - BASELINE[0])

    # group-mean operator: G @ rates gives the (n_freq, n_units) matrix of mean rates
    G = np.array([(freq == f) / np.sum(freq == f) for f in freqs])
    # Accelerate (the BLAS numpy uses on macOS) raises spurious FP status warnings on
    # these matmuls; the results agree with explicit group means to ~1e-12
    with np.errstate(all="ignore"):
        obs_var = np.var(G @ rate_ev, axis=0)

    # permutation null for the frequency-selectivity test, shared across units: shuffling
    # the frequency labels is equivalent to shuffling trial order, so one permuted trial
    # order can be applied to every unit at once
    exceed = np.zeros(rate_ev.shape[1])
    with np.errstate(all="ignore"):
        for _ in range(N_PERM):
            perm = rng.permutation(len(freq))
            exceed += np.var(G @ rate_ev[perm], axis=0) >= obs_var
    p_tuned_all = (exceed + 1) / (N_PERM + 1)

    units = []
    for k, u in enumerate(spikes.index):
        overall = len(spikes[u]) / spikes.time_support.tot_length()
        if overall < MIN_RATE:
            continue
        ev, bs = rate_ev[:, k], rate_bs[:, k]
        tuning = np.array([ev[freq == f].mean() for f in freqs])
        tuning_sem = np.array([stats.sem(ev[freq == f]) for f in freqs])
        baseline = bs.mean()
        driven = tuning - baseline

        # is the unit driven by tones at all?
        _, p_resp = stats.wilcoxon(ev, bs, zero_method="zsplit")
        p_tuned = p_tuned_all[k]

        units.append(dict(
            session=sess["session"], subject=sess["subject"], unit=int(u),
            overall_rate=overall, baseline=baseline, tuning=tuning, tuning_sem=tuning_sem,
            driven=driven, bf=float(freqs[np.argmax(driven)]),
            peak_driven=float(driven.max()), selectivity=selectivity(driven),
            p_responsive=float(p_resp), p_tuned=float(p_tuned),
        ))
    return units, freqs, spikes, evoked_ep


def decode_frequency(spikes, sess, n_folds=5, rng=None, shuffle=False):
    """Cross-validated Bayesian decoding of tone frequency from population activity.

    Tuning curves are estimated with pynapple on the training trials and used to decode
    the held-out trials; returns the confusion matrix and accuracy.
    """
    rng = rng or np.random.default_rng(0)
    onset, freq = sess["tone_onset"], sess["frequency"]
    if shuffle:  # control: destroy the tone-frequency / spike-count relationship
        freq = rng.permutation(freq)
    freqs = np.unique(freq)
    log_f = np.log2(freq / 1000.0)
    levels = np.log2(freqs / 1000.0)

    evoked_ep = nap.IntervalSet(start=onset + EVOKED[0], end=onset + EVOKED[1])
    # feature: the tone's log-frequency, sampled through each evoked window
    dt = 0.005
    ts, vals = [], []
    for (s, e), lf in zip(zip(evoked_ep.start, evoked_ep.end), log_f):
        grid = np.arange(s, e, dt)
        ts.append(grid)
        vals.append(np.full(len(grid), lf))
    feature = nap.Tsd(t=np.concatenate(ts), d=np.concatenate(vals), time_support=evoked_ep)

    order = rng.permutation(len(onset))
    folds = np.array_split(order, n_folds)
    conf = np.zeros((len(freqs), len(freqs)))
    for i in range(n_folds):
        test = np.sort(folds[i])
        train = np.sort(np.concatenate([folds[j] for j in range(n_folds) if j != i]))
        tc = nap.compute_tuning_curves(
            spikes, feature, bins=len(freqs),
            range=(levels[0] - 0.5, levels[-1] + 0.5),
            epochs=evoked_ep[train], feature_names=["log_frequency"],
        )
        decoded, _ = nap.decode_bayes(
            tc, spikes, evoked_ep[test], bin_size=EVOKED[1] - EVOKED[0] + 1e-9
        )
        pred = np.asarray(decoded.values).ravel()
        assert len(pred) == len(test), f"decoder returned {len(pred)} bins for {len(test)} trials"
        true = log_f[test]
        for a, b in zip(true, pred):
            conf[np.argmin(np.abs(levels - a)), np.argmin(np.abs(levels - b))] += 1
    acc = np.trace(conf) / conf.sum()
    return conf, acc, freqs


if __name__ == "__main__":
    files = dio.list_assets("000986")
    url = [u for p, u in files if "LA9_ses-1" in p][0]
    sess = dio.load_tone_session_000986(url)
    units, freqs, spikes, _ = analyze_session(sess)
    print(sess["session"], sess["subject"], len(units), "units analysed")
    n_resp = sum(u["p_responsive"] < 0.01 for u in units)
    n_tuned = sum(u["p_tuned"] < 0.01 for u in units)
    print(f"responsive: {n_resp}/{len(units)}   frequency-tuned: {n_tuned}/{len(units)}")
    for u in sorted(units, key=lambda u: -u["peak_driven"])[:5]:
        print(f"  unit {u['unit']:4d} BF={u['bf']/1000:5.1f} kHz  peak driven "
              f"{u['peak_driven']:6.1f} Hz  sel={u['selectivity']:.2f}  p={u['p_tuned']:.4f}")
