"""Poisson GLM encoding model of cortical tone responses (NeMoS).

The trial-averaged tuning curve says how much a unit fires for each frequency but not how
that response unfolds in time, and it ignores the unit's own spiking dynamics. Here each
unit's binned spike train is modelled as a Poisson process driven by five frequency-specific
tone kernels plus a spike-history filter. The area under each tone kernel is a
model-based tuning curve, estimated jointly with the refractoriness and bursting that
otherwise contaminate simple spike counts.
"""

import numpy as np
import nemos as nmo
import pynapple as nap

import cortex_analysis as cx

nap.nap_config.suppress_conversion_warnings = True

BIN = 0.005            # s
STIM_WINDOW = 30       # bins (150 ms) of tone kernel
HIST_WINDOW = 20       # bins (100 ms) of spike history
N_STIM_BASIS = 6
N_HIST_BASIS = 5


def build_design(sess, spikes, span=1800.0, train_frac=0.75):
    """Bin a stretch of the session and build the GLM design matrix.

    Returns the feature matrix, the count matrix, the bases and the train/test split.
    """
    onset, freq = sess["tone_onset"], sess["frequency"]
    freqs = np.unique(freq)
    ep = nap.IntervalSet(start=onset[0] - 1.0, end=onset[0] + span)

    counts = spikes.count(BIN, ep=ep)                      # (n_bins, n_units)
    stim = np.zeros((len(counts), len(freqs)))
    m = (onset >= ep.start[0]) & (onset <= ep.end[0])
    idx = np.searchsorted(counts.t, onset[m])
    for i, fq in zip(idx, freq[m]):
        if i < len(counts):
            stim[i, list(freqs).index(fq)] += 1
    stim = nap.TsdFrame(t=counts.t, d=stim, time_support=ep)

    stim_basis = nmo.basis.RaisedCosineLogConv(
        n_basis_funcs=N_STIM_BASIS, window_size=STIM_WINDOW, label="tone")
    hist_basis = nmo.basis.RaisedCosineLogConv(
        n_basis_funcs=N_HIST_BASIS, window_size=HIST_WINDOW, label="history")
    stim_basis.set_input_shape(stim)
    X_stim = stim_basis.compute_features(stim)

    n_train = int(train_frac * len(counts))
    return dict(counts=counts, stim=stim, X_stim=np.asarray(X_stim), freqs=freqs,
                stim_basis=stim_basis, hist_basis=hist_basis, n_train=n_train, ep=ep)


def fit_unit(design, unit):
    """Fit the Poisson GLM for one unit and return kernels and held-out performance."""
    y = np.asarray(design["counts"][:, design["counts"].columns.get_loc(unit)])
    own = nap.Tsd(t=design["counts"].t, d=y, time_support=design["ep"])
    X_hist = np.asarray(design["hist_basis"].compute_features(own))
    X = np.hstack([design["X_stim"], X_hist])

    n = design["n_train"]
    ok_tr = np.isfinite(X[:n]).all(1)
    ok_te = np.isfinite(X[n:]).all(1)

    model = nmo.glm.GLM(solver_name="LBFGS", regularizer="Ridge",
                        regularizer_strength=1e-4)
    model.fit(X[:n][ok_tr], y[:n][ok_tr])
    r2_test = float(model.score(X[n:][ok_te], y[n:][ok_te],
                                score_type="pseudo-r2-McFadden"))
    r2_train = float(model.score(X[:n][ok_tr], y[:n][ok_tr],
                                 score_type="pseudo-r2-McFadden"))

    # reconstruct the frequency-specific tone kernels and the history filter
    _, stim_kern = design["stim_basis"].evaluate_on_grid(STIM_WINDOW)
    _, hist_kern = design["hist_basis"].evaluate_on_grid(HIST_WINDOW)
    n_f = len(design["freqs"])
    w_stim = model.coef_[: n_f * N_STIM_BASIS].reshape(n_f, N_STIM_BASIS)
    w_hist = model.coef_[n_f * N_STIM_BASIS:]
    kernels = np.array([stim_kern @ w for w in w_stim])     # (n_freq, window)
    history = hist_kern @ w_hist

    return dict(unit=unit, kernels=kernels, history=history,
                gain=kernels.sum(1), r2_test=r2_test, r2_train=r2_train,
                intercept=float(model.intercept_[0]),
                t_stim=np.arange(STIM_WINDOW) * BIN, t_hist=np.arange(HIST_WINDOW) * BIN)


def most_responsive_units(sess, spikes, n=3):
    """Units with the largest tone-evoked rate increase, without the permutation test."""
    _, ev_ep, bs_ep = cx.session_to_pynapple(sess)
    ev = cx.trial_counts(spikes, ev_ep, cx.EVOKED).mean(0) / (cx.EVOKED[1] - cx.EVOKED[0])
    bs = cx.trial_counts(spikes, bs_ep, cx.BASELINE).mean(0) / (cx.BASELINE[1] - cx.BASELINE[0])
    order = np.argsort(bs - ev)  # most positive driven rate first
    return [int(spikes.index[k]) for k in order[:n]]


def fit_session(sess, spikes, unit_ids, span=1800.0):
    design = build_design(sess, spikes, span=span)
    return [fit_unit(design, int(u)) for u in unit_ids], design


if __name__ == "__main__":
    import dandi_io as dio

    files = dio.list_assets("000986")
    sess = dio.load_tone_session_000986([u for p, u in files if "LA9_ses-1" in p][0])
    spikes, _, _ = cx.session_to_pynapple(sess)
    unit_ids = most_responsive_units(sess, spikes, n=3)
    print("units:", unit_ids)
    fits, design = fit_session(sess, spikes, unit_ids)
    for f in fits:
        print(f"unit {f['unit']}: pseudo-R2 train {f['r2_train']:.4f} "
              f"test {f['r2_test']:.4f}  gain {np.round(f['gain'], 2)}")
