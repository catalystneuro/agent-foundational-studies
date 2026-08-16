"""Poisson GLM STRF with 2D basis (frequency x latency) using nemos."""
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import nemos as nmo

FREQ_BASIS = nmo.basis.BSplineEval(n_basis_funcs=4)
TIME_BASIS = nmo.basis.BSplineEval(n_basis_funcs=9)

LOGF = np.log2(np.array([2000., 4000., 8000., 16000., 32000.]) / 1000.)  # 1..5

def build_design(freqs, bin_c_ms):
    """X: (n_trials*n_bins, 1 + 4*6); also return per-freq/time design for prediction."""
    fb = FREQ_BASIS.compute_features(LOGF)          # (5, 4)
    tb = TIME_BASIS.compute_features(bin_c_ms)      # (n_bins, 6)
    fidx = np.searchsorted(np.array([2000.,4000.,8000.,16000.,32000.]), freqs)
    n_trials, n_bins = len(freqs), len(bin_c_ms)
    Ft = fb[fidx]                                    # (n_trials, 4)
    # full 2D: kron per row -> (n_trials, n_bins, 24)
    X3 = np.einsum('ta,nb->tnab', Ft, tb).reshape(n_trials, n_bins, -1)
    X = X3.reshape(-1, X3.shape[-1])
    return X, fb, tb

def fit_glm_strf(counts, freqs, bin_c_ms, rng_seed=0):
    """Fit Poisson GLM; return weights, test pseudo-R2, predicted rate map (5, n_bins)."""
    X, fb, tb = build_design(freqs, bin_c_ms)
    y = counts.reshape(-1).astype(float)
    n_trials, n_bins = counts.shape
    rng = np.random.default_rng(rng_seed)
    tr_mask_trials = rng.random(n_trials) < 0.8
    m = np.repeat(tr_mask_trials, n_bins)
    model = nmo.glm.GLM(regularizer=nmo.regularizer.Ridge(), regularizer_strength=1e-4,
                        solver_name="LBFGS", solver_kwargs=dict(maxiter=2000, tol=1e-8))
    model.fit(X[m], y[m])
    null = nmo.glm.GLM(solver_name="LBFGS", solver_kwargs=dict(maxiter=2000))
    # null = intercept only: use a single column of ones via feature matrix of ones
    X0 = np.ones((X.shape[0], 1))
    null.fit(X0[m], y[m])
    ll_m = model.score(X[~m], y[~m])
    ll_0 = null.score(X0[~m], y[~m])
    pseudo_r2 = 1.0 - ll_m / ll_0
    # predicted rate map per (freq, timebin)
    Xp = np.einsum('fa,nb->fnab', fb, tb).reshape(len(fb), len(tb), -1).reshape(-1, X.shape[-1])
    rate = model.predict(Xp).reshape(len(fb), len(tb)) / 0.005  # Hz (counts per 5 ms bin)
    return model, pseudo_r2, rate
