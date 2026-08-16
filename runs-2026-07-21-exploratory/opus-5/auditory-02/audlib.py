"""Shared loading / analysis helpers for the DANDI 000986 auditory tuning analysis.

Dandiset 000986: "Auditory cortex Neuropixels recordings and pupil diameter traces
from mice during passive exposure to pure tones."  Each session presents 25 ms
pure tones at 60 dB SPL, drawn randomly from {2, 4, 8, 16, 32} kHz, roughly every
0.8 s, interleaved with 300 s blocks of silence.
"""

import numpy as np
import requests
import h5py
import remfile
import pynapple as nap
from pynwb import NWBHDF5IO

DANDISET = "000986"
API = "https://api.dandiarchive.org/api/dandisets"
CACHE = "/tmp/remfile_cache_000986"

# response windows, in seconds relative to tone onset (validated in 01_load_inspect.py)
EVOKED_WIN = (0.010, 0.060)
BASELINE_WIN = (-0.100, -0.005)


def list_assets():
    """Return [(path, download_url), ...] for every session, sorted by path."""
    r = requests.get(
        f"{API}/{DANDISET}/versions/draft/assets/", params={"page_size": 500}
    ).json()
    out = []
    for a in sorted(r["results"], key=lambda x: x["path"]):
        url = f"{API}/{DANDISET}/versions/draft/assets/{a['asset_id']}/download/"
        out.append((a["path"], url))
    return out


def load_session(url):
    """Stream one NWB file (cached on disk) and return (nwbfile, pynapple wrapper)."""
    rem = remfile.File(url, disk_cache=remfile.DiskCache(CACHE))
    h5 = h5py.File(rem, "r")
    io = NWBHDF5IO(file=h5, load_namespaces=True)
    nwbfile = io.read()
    return nwbfile, nap.NWBFile(nwbfile)


def trial_table(nwbfile):
    """Trial onsets and stimulus parameters as plain arrays."""
    tr = nwbfile.trials
    return dict(
        onset=np.asarray(tr["start_time"].data[:]),
        stop=np.asarray(tr["stop_time"].data[:]),
        frequency=np.asarray(tr["stim_frequency"].data[:]),
        amplitude=np.asarray(tr["stim_amplitude"].data[:]),
        duration=np.asarray(tr["stim_duration"].data[:]),
    )


# ------------------------------------------------------------ spike counts --


def window_counts(spikes, onsets, win):
    """Spike counts per (unit, trial) in `win` seconds relative to tone onset.

    Returns an (n_units, n_trials) integer array.  searchsorted on each unit's
    sorted spike times is exact and fast enough for ~7500 trials x ~200 units.
    """
    lo, hi = onsets + win[0], onsets + win[1]
    out = np.zeros((len(spikes), len(onsets)))
    for i, k in enumerate(spikes.keys()):
        t = spikes[k].t
        out[i] = np.searchsorted(t, hi) - np.searchsorted(t, lo)
    return out


def evoked_rates(spikes, onsets, evoked_win=EVOKED_WIN, base_win=BASELINE_WIN):
    """Per-trial evoked and baseline firing rates (spikes/s).

    Returns (evoked, baseline), each (n_units, n_trials).
    """
    ev = window_counts(spikes, onsets, evoked_win) / (evoked_win[1] - evoked_win[0])
    bl = window_counts(spikes, onsets, base_win) / (base_win[1] - base_win[0])
    return ev, bl


def peri_event_times(spike_times, onsets, tmin, tmax):
    """Spike times relative to each onset, plus the trial index of each spike.

    Fully vectorised gather: for every onset we take the slice of the (sorted)
    spike train that falls in [onset+tmin, onset+tmax).  Windows shorter than the
    inter-tone interval are assumed, so each spike lands in at most one window.
    """
    lo = np.searchsorted(spike_times, onsets + tmin)
    hi = np.searchsorted(spike_times, onsets + tmax)
    n = hi - lo
    if n.sum() == 0:
        return np.array([]), np.array([], dtype=int)
    starts = np.concatenate([[0], np.cumsum(n)[:-1]])
    idx = np.repeat(lo - starts, n) + np.arange(n.sum())
    trial = np.repeat(np.arange(len(onsets)), n)
    return spike_times[idx] - onsets[trial], trial


def psth(spike_times, onsets, bins):
    """Trial-averaged firing rate (spikes/s) of one unit, aligned to `onsets`."""
    rel, _ = peri_event_times(spike_times, onsets, bins[0], bins[-1])
    h = np.histogram(rel, bins=bins)[0]
    return h / (len(onsets) * np.diff(bins))


def group_means(x, labels, levels):
    """Mean of x (n_units, n_trials) within each stimulus level -> (n_units, n_levels)."""
    return np.column_stack([x[:, labels == lv].mean(axis=1) for lv in levels])


def group_sems(x, labels, levels):
    out = []
    for lv in levels:
        s = x[:, labels == lv]
        out.append(s.std(axis=1, ddof=1) / np.sqrt(s.shape[1]))
    return np.column_stack(out)


# ------------------------------------------------------------ tuning stats --


def sparseness(tc):
    """Treves-Rolls lifetime sparseness of a tuning curve, rescaled to [0, 1].

    0 = uniform response across frequencies, 1 = responds to a single frequency.
    Computed on the rectified tuning curve.
    """
    r = np.clip(tc, 0, None)
    n = r.shape[-1]
    num = (r.mean(axis=-1)) ** 2
    den = (r**2).mean(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        s = (1 - num / den) / (1 - 1.0 / n)
    return s


def anova_f(x, labels, levels):
    """Vectorised one-way ANOVA F statistic over trials, for every unit at once.

    x: (n_units, n_trials); returns F (n_units,) and the dfs.
    """
    n_units, n_tot = x.shape
    grand = x.mean(axis=1, keepdims=True)
    ss_b = np.zeros(n_units)
    ss_w = np.zeros(n_units)
    for lv in levels:
        g = x[:, labels == lv]
        m = g.mean(axis=1, keepdims=True)
        ss_b += g.shape[1] * (m[:, 0] - grand[:, 0]) ** 2
        ss_w += ((g - m) ** 2).sum(axis=1)
    df_b = len(levels) - 1
    df_w = n_tot - len(levels)
    with np.errstate(invalid="ignore", divide="ignore"):
        f = (ss_b / df_b) / (ss_w / df_w)
    return f, df_b, df_w


def permutation_p(x, labels, levels, n_perm=1000, seed=0):
    """Permutation p-value for frequency tuning, per unit.

    Shuffles the frequency label across trials and recomputes the ANOVA F.
    Returns (p, f_obs, f_null_matrix).
    """
    rng = np.random.default_rng(seed)
    f_obs, _, _ = anova_f(x, labels, levels)
    ge = np.zeros(x.shape[0])
    null = np.zeros((n_perm, x.shape[0]))
    for i in range(n_perm):
        sh = rng.permutation(labels)
        f_null, _, _ = anova_f(x, sh, levels)
        null[i] = f_null
        ge += f_null >= f_obs
    p = (ge + 1) / (n_perm + 1)
    return p, f_obs, null


def bh_fdr(p):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return out


# ----------------------------------------------------- smooth tuning (GLM) --


def glm_tuning(counts, flab, freqs, n_basis=4, n_grid=81, n_folds=5, seed=0):
    """Smooth frequency tuning from a Poisson population GLM (NeMoS).

    counts : (n_units, n_trials) evoked spike counts in the response window
    Design matrix is a raised-cosine basis over log2(frequency), so the fitted
    tuning curve is smooth in octaves and can be read out between the five
    sampled tones.  Returns a dict with the dense grid, per-unit fitted rate,
    continuous best frequency, half-max bandwidth in octaves, and a
    cross-validated McFadden pseudo-R2 (full model vs. constant rate).
    """
    import nemos as nmo
    from sklearn.model_selection import KFold

    oct_trial = np.log2(flab / freqs[0])
    span = np.log2(freqs[-1] / freqs[0])
    basis = nmo.basis.RaisedCosineLinearEval(n_basis_funcs=n_basis, bounds=(0, span))
    X = np.asarray(basis.compute_features(oct_trial), dtype=float)
    y = counts.T.astype(float)  # (n_trials, n_units)

    model = nmo.glm.PopulationGLM(solver_name="LBFGS").fit(X, y)
    grid_oct = np.linspace(0, span, n_grid)
    Xg = np.asarray(basis.compute_features(grid_oct), dtype=float)
    rate = np.asarray(model.predict(Xg))  # (n_grid, n_units) counts per window

    # cross-validated McFadden pseudo-R2, per unit
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    ll_full = np.zeros(y.shape[1])
    ll_null = np.zeros(y.shape[1])
    for tr_i, te_i in kf.split(X):
        m = nmo.glm.PopulationGLM(solver_name="LBFGS").fit(X[tr_i], y[tr_i])
        lam = np.asarray(m.predict(X[te_i]))
        lam0 = np.broadcast_to(y[tr_i].mean(axis=0), lam.shape)
        yt = y[te_i]
        ll_full += _pois_ll(yt, lam)
        ll_null += _pois_ll(yt, lam0)
    pr2 = 1 - ll_full / ll_null  # both log-likelihoods are negative

    # diagnostic: does the fitted curve reproduce the empirical mean counts at
    # the five tested frequencies?  (r ~ 1 means the basis is rich enough)
    Xt = np.asarray(basis.compute_features(np.log2(freqs / freqs[0])), dtype=float)
    pred_at_tested = np.asarray(model.predict(Xt))
    emp = group_means(counts, flab, freqs).T
    fit_corr = np.corrcoef(pred_at_tested.ravel(), emp.ravel())[0, 1]

    bf_oct = grid_oct[np.argmax(rate, axis=0)]
    bw = _half_max_width(rate, grid_oct)
    return dict(grid_oct=grid_oct, rate=rate.T, bf_oct=bf_oct, bandwidth_oct=bw,
                pseudo_r2=pr2, span=span, fit_corr=fit_corr)


def _pois_ll(y, lam):
    """Summed Poisson log-likelihood per unit (constant terms dropped consistently)."""
    from scipy.special import gammaln
    lam = np.clip(lam, 1e-9, None)
    return (y * np.log(lam) - lam - gammaln(y + 1)).sum(axis=0)


def _half_max_width(rate, grid_oct):
    """Width in octaves of the contiguous region above half of the peak drive.

    Drive is the fitted rate minus its minimum over the tested range, so the
    width measures tuning sharpness on top of the unit's floor response.
    """
    r = rate - rate.min(axis=0, keepdims=True)
    peak = r.max(axis=0)
    out = np.full(r.shape[1], np.nan)
    step = grid_oct[1] - grid_oct[0]
    for i in range(r.shape[1]):
        if peak[i] <= 0:
            continue
        above = r[:, i] >= 0.5 * peak[i]
        j = int(np.argmax(r[:, i]))
        lo = j
        while lo > 0 and above[lo - 1]:
            lo -= 1
        hi = j
        while hi < len(above) - 1 and above[hi + 1]:
            hi += 1
        out[i] = (hi - lo) * step
    return out


# ------------------------------------------------------------- decoding ----


def decode_frequency(counts, flab, freqs, n_folds=5, seed=0):
    """Cross-validated decoding of the tone frequency from a single trial.

    counts : (n_units, n_trials) evoked counts.  A shrinkage LDA is used because
    the number of units is comparable to the number of trials per class; the
    Ledoit-Wolf shrinkage keeps the class covariance estimate well conditioned.
    Returns (confusion, accuracy), confusion row-normalised (rows = true freq).
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import StratifiedKFold

    X = counts.T.astype(float)
    y = np.searchsorted(freqs, flab)
    pred = np.zeros_like(y)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr_i, te_i in skf.split(X, y):
        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        clf.fit(X[tr_i], y[tr_i])
        pred[te_i] = clf.predict(X[te_i])
    conf = np.zeros((len(freqs), len(freqs)))
    for a, b in zip(y, pred):
        conf[a, b] += 1
    acc = (pred == y).mean()
    return conf / conf.sum(axis=1, keepdims=True), acc


def split_half_bf(dev, flab, freqs, seed=0):
    """Best frequency computed independently on two random halves of the trials."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(dev.shape[1])
    h1, h2 = order[::2], order[1::2]
    bf1 = freqs[np.argmax(group_means(dev[:, h1], flab[h1], freqs), axis=1)]
    bf2 = freqs[np.argmax(group_means(dev[:, h2], flab[h2], freqs), axis=1)]
    return bf1, bf2
