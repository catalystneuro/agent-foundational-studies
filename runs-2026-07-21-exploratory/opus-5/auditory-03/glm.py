"""Poisson GLM of tone-evoked spiking with NeMoS.

Each unit's binned spike count is modelled as

    rate(t) = exp( b0 + sum_f (k_f * s_f)(t) + (h * y)(t) )

where s_f is the onset train of tones at frequency f, k_f is a temporal filter
expanded in log-spaced raised cosines, and h is a spike-history filter. The
history term absorbs adaptation and refractoriness so the frequency filters are
not fit to the unit's own autocorrelation.
"""
import nemos as nmo
import numpy as np
import pynapple as nap

from common import EVOKED_WIN, FREQS

BIN = 0.005          # 5 ms bins
WINDOW = 40          # 200 ms stimulus / history window
N_STIM_BASIS = 7
N_HIST_BASIS = 5


def build_basis():
    stim = None
    for f in FREQS:
        b = nmo.basis.RaisedCosineLogConv(n_basis_funcs=N_STIM_BASIS,
                                          window_size=WINDOW,
                                          label=f"{int(f / 1000)}kHz")
        stim = b if stim is None else stim + b
    hist = nmo.basis.RaisedCosineLogConv(n_basis_funcs=N_HIST_BASIS,
                                         window_size=WINDOW, label="history")
    return stim, hist


def design(spikes, trials, ep=None):
    """Binned counts, per-frequency stimulus features, and the basis objects."""
    if ep is None:
        t0 = trials.start_time.values[0] - 1.0
        t1 = trials.stop_time.values[-1] + 1.0
        ep = nap.IntervalSet(start=t0, end=t1)
    counts = spikes.count(BIN, ep=ep)
    onsets = trials.start_time.values
    events = [nap.Ts(onsets[trials.stim_frequency.values == f]).count(BIN, ep=ep)
              for f in FREQS]
    stim_basis, hist_basis = build_basis()
    X_stim = np.asarray(stim_basis.compute_features(*events))
    return counts, X_stim, stim_basis, hist_basis


def fit_unit(counts_u, X_stim, hist_basis):
    """Fit one unit; return the model and its history features."""
    X_hist = np.asarray(hist_basis.compute_features(counts_u))
    X = np.hstack([X_stim, X_hist])
    y = np.asarray(counts_u, dtype=float)
    ok = ~np.isnan(X).any(axis=1)
    model = nmo.glm.GLM(solver_name="LBFGS",
                        regularizer="Ridge", regularizer_strength=1e-4)
    model.fit(X[ok], y[ok])
    return model, X, ok


def filters(model, stim_basis, hist_basis):
    """Reconstruct the frequency and history filters in units of log-rate.

    The five frequency bases are identical, so one kernel matrix serves all of
    them; the additive basis itself expects one grid argument per component.
    """
    _, stim_k = nmo.basis.RaisedCosineLogConv(
        n_basis_funcs=N_STIM_BASIS, window_size=WINDOW).evaluate_on_grid(WINDOW)
    _, hist_k = hist_basis.evaluate_on_grid(WINDOW)
    n_stim = len(FREQS) * N_STIM_BASIS
    coef = np.asarray(model.coef_)
    freq_filters = np.zeros((len(FREQS), WINDOW))
    for i in range(len(FREQS)):
        sl = slice(i * N_STIM_BASIS, (i + 1) * N_STIM_BASIS)
        freq_filters[i] = stim_k @ coef[sl]
    hist_filter = hist_k @ coef[n_stim:]
    return freq_filters, hist_filter


def glm_tuning(model, freq_filters, window=EVOKED_WIN):
    """Predicted evoked rate per frequency (Hz above baseline), no history term.

    The filters act on log rate, so the isolated tone response is
    exp(b0 + k_f(t)) and the baseline is exp(b0), both converted to Hz.
    """
    lags = (np.arange(WINDOW) + 1) * BIN          # causal conv starts one bin out
    sel = (lags >= window[0]) & (lags < window[1])
    b0 = float(np.asarray(model.intercept_)[0])
    base = np.exp(b0) / BIN
    rate = np.exp(b0 + freq_filters) / BIN
    return rate[:, sel].mean(axis=1) - base, base, lags
