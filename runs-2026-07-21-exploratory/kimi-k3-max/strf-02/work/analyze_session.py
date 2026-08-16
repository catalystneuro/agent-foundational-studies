"""Full single-session STRF analysis: rate maps + stats + PopulationGLM."""
import numpy as np, time, sys
import jax
jax.config.update("jax_enable_x64", True)
import nemos as nmo
from session_strf import (load_session, trial_bin_counts, rate_map_strf,
                          responsiveness_stats, characterize_strf,
                          FREQ_VALS, BIN_C, BASE_MASK, EVOKED_MASK, ASSETS)
from glm_strf import build_design

def analyze_session(asset_id, label):
    t0 = time.time()
    unit_spikes, onsets, freqs, nwbfile = load_session(asset_id)
    n_units = len(unit_spikes)
    print(f'[{label}] {n_units} units, {len(onsets)} trials, loaded {time.time()-t0:.0f}s', flush=True)

    counts_all = np.stack([trial_bin_counts(s, onsets) for s in unit_spikes])  # (U, T, B)
    print(f'[{label}] binned {time.time()-t0:.0f}s', flush=True)

    strfs = np.stack([rate_map_strf(counts_all[u], freqs) for u in range(n_units)])
    p_resp = np.zeros(n_units); p_tune = np.zeros(n_units)
    for u in range(n_units):
        p_resp[u], p_tune[u] = responsiveness_stats(counts_all[u], freqs)
    tuned = (p_resp < 0.01) & (p_tune < 0.01)
    print(f'[{label}] tuned: {tuned.sum()}/{n_units}', flush=True)

    chars = [characterize_strf(strfs[u]) for u in range(n_units)]

    # --- PopulationGLM on tuned units (shared design matrix)
    glm_rate = np.full_like(strfs, np.nan)
    pseudo_r2 = np.full(n_units, np.nan)
    tu = np.where(tuned)[0]
    if len(tu) > 0:
        X, fb, tb = build_design(freqs, BIN_C * 1000)
        n_trials, n_bins = len(onsets), len(BIN_C)
        rng = np.random.default_rng(0)
        tr_trials = rng.random(n_trials) < 0.8
        m = np.repeat(tr_trials, n_bins)
        Y = counts_all[tu].transpose(1, 2, 0).reshape(-1, len(tu)).astype(float)  # (T*B, U_tuned)
        import warnings
        model = nmo.glm.PopulationGLM(regularizer=nmo.regularizer.Ridge(), regularizer_strength=1e-5,
                                      solver_name="LBFGS", solver_kwargs=dict(maxiter=5000, tol=1e-8))
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            model.fit(X[m], Y[m])
        converged = not any('did not converge' in str(w.message) for w in wlist)
        print(f'[{label}] GLM converged: {converged}', flush=True)
        X0 = np.ones((X.shape[0], 1))
        null = nmo.glm.PopulationGLM(solver_name="LBFGS", solver_kwargs=dict(maxiter=2000))
        null.fit(X0[m], Y[m])
        # per-unit Poisson log-likelihood on held-out trials (gammaln included in both)
        from scipy.special import gammaln
        Yte = Y[~m]
        rm_ = np.asarray(model.predict(X[~m]))
        r0_ = np.asarray(null.predict(X0[~m]))
        def pll(y, r):
            r = np.clip(r, 1e-12, None)
            return (y * np.log(r) - r - gammaln(y + 1)).mean(axis=0)
        pr2 = 1.0 - pll(Yte, rm_) / pll(Yte, r0_)
        pseudo_r2[tu] = pr2
        Xp = np.einsum('fa,nb->fnab', fb, tb).reshape(len(fb), len(tb), -1).reshape(-1, X.shape[-1])
        rate = model.predict(Xp)  # (5*n_bins, U_tuned) counts per bin
        glm_rate[tu] = (rate / 0.005).reshape(len(fb), len(tb), len(tu)).transpose(2, 0, 1)
        print(f'[{label}] GLM done {time.time()-t0:.0f}s, median pseudo-R2 {np.nanmedian(pseudo_r2):.3f}', flush=True)

    out = dict(strfs=strfs, glm_rate=glm_rate, pseudo_r2=pseudo_r2,
               p_resp=p_resp, p_tune=p_tune, tuned=tuned,
               chars=chars, label=label, n_units=n_units)
    np.savez(f'session_{label}.npz', strfs=strfs, glm_rate=glm_rate, pseudo_r2=pseudo_r2,
             p_resp=p_resp, p_tune=p_tune, tuned=tuned,
             bf=np.array([c['bf'] for c in chars]),
             peak_lat=np.array([c['peak_lat'] for c in chars]),
             peak_hz=np.array([c['peak_hz'] for c in chars]),
             min_hz=np.array([c['min_hz'] for c in chars]),
             separability=np.array([c['separability'] for c in chars]))
    return out

if __name__ == '__main__':
    label = sys.argv[1] if len(sys.argv) > 1 else 'LA11_ses2'
    analyze_session(ASSETS[label], label)
