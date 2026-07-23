"""Poisson GLM of frequency-specific tone responses (NeMoS), example session.

The PSTH answers "does the mean response differ by frequency".  The GLM asks a
stricter question on held-out data: does knowing *which* tone was played predict
spiking better than knowing only *that* a tone was played?

  full model      log lambda(t) = b + sum_k (h_k * s_k)(t)     (one kernel per frequency)
  reduced model   log lambda(t) = b + (h * s_all)(t)           (one kernel for any tone)

Both are fit on interleaved 60 s chunks and scored on the held-out chunks.
"""

import pickle

import numpy as np
import nemos as nmo
import pynapple as nap
from tqdm import trange

import dandi_io
import tuning as T

BIN = 0.005  # s
WINDOW_BINS = 60  # 300 ms kernel
N_BASIS = 8
CHUNK_S = 60.0  # train/test chunks, alternated to be robust to slow drift
EXAMPLE_SESSION = "sub-LA11/sub-LA11_ses-1_behavior.nwb"


def build_design(onsets, freqs, counts, collapse_frequency=False):
    """One-hot tone-onset streams convolved with a raised-cosine basis."""
    t = counts.t
    streams = np.zeros((len(t), 1 if collapse_frequency else len(T.FREQS)))
    idx = np.searchsorted(t, onsets)
    idx = idx[idx < len(t)]
    if collapse_frequency:
        np.add.at(streams[:, 0], idx, 1.0)
    else:
        for k, f in enumerate(T.FREQS):
            sel = np.searchsorted(t, onsets[freqs == f])
            np.add.at(streams[:, k], sel[sel < len(t)], 1.0)

    stim = nap.TsdFrame(t=t, d=streams, time_support=counts.time_support)
    basis = nmo.basis.RaisedCosineLogConv(n_basis_funcs=N_BASIS, window_size=WINDOW_BINS)
    basis.set_input_shape(stim)
    return basis, basis.compute_features(stim)


def poisson_ll_per_unit(y, mu):
    """Poisson log-likelihood per time bin, per unit (constant term dropped)."""
    from scipy.special import gammaln

    mu = np.clip(np.asarray(mu), 1e-10, None)
    y = np.asarray(y)
    return (y * np.log(mu) - mu - gammaln(y + 1)).mean(axis=0)


UNIT_BATCH = 48  # fit the population in batches to bound peak memory


def fit_and_score(X, y, train, test):
    """Fit a Poisson PopulationGLM in unit batches; return coefficients and held-out LL.

    The population log-likelihood is a sum over units, so batching changes nothing
    about the fit, it only bounds memory.
    """
    Xtr, Xte = np.asarray(X[train], dtype=np.float32), np.asarray(X[test], dtype=np.float32)
    coefs, intercepts, ll = [], [], []
    for lo in trange(0, y.shape[1], UNIT_BATCH, desc="  unit batches", leave=False):
        yb = y[:, lo:lo + UNIT_BATCH].astype(np.float32)
        model = nmo.glm.PopulationGLM(
            observation_model="Poisson", regularizer="Ridge",
            regularizer_strength=1e-4, solver_name="LBFGS",
        )
        model.fit(Xtr, yb[train])
        coefs.append(np.asarray(model.coef_))
        intercepts.append(np.asarray(model.intercept_))
        ll.append(poisson_ll_per_unit(yb[test], model.predict(Xte)))
    return np.concatenate(coefs, axis=1), np.concatenate(intercepts), np.concatenate(ll)


def main():
    asset = next(a for a in dandi_io.list_assets() if a["path"] == EXAMPLE_SESSION)
    nwb, nwbfile = dandi_io.open_session(asset)
    spikes, onsets, freqs, _ = T.load_session_arrays(nwb, nwbfile)

    # Restrict to the tone-presentation blocks (drop the spontaneous blocks).
    trials = nap.IntervalSet(start=onsets, end=onsets + 0.8).merge_close_intervals(1.0)
    counts = spikes.count(BIN, ep=trials)
    print(f"{counts.shape[0]} bins x {counts.shape[1]} units over "
          f"{trials.tot_length():.0f} s in {len(trials)} blocks")

    basis, X_full = build_design(onsets, freqs, counts)
    _, X_red = build_design(onsets, freqs, counts, collapse_frequency=True)

    ok = np.isfinite(np.asarray(X_full)).all(axis=1) & np.isfinite(np.asarray(X_red)).all(axis=1)
    chunk = np.floor((counts.t - counts.t[0]) / CHUNK_S).astype(int)
    train, test = ok & (chunk % 2 == 0), ok & (chunk % 2 == 1)
    print(f"train bins {train.sum()}, test bins {test.sum()}")

    y = np.asarray(counts)
    print("fitting frequency-specific model")
    coef_full, intercept, ll_full = fit_and_score(np.asarray(X_full), y, train, test)
    print("fitting tone-only (reduced) model")
    _, _, ll_red = fit_and_score(np.asarray(X_red), y, train, test)

    # Frequency-specific kernels, as a gain on log firing rate.
    _, kernels = basis.evaluate_on_grid(WINDOW_BINS)  # (window, n_basis)
    w = basis.split_by_feature(coef_full, axis=0)["RaisedCosineLogConv"]
    # w: (n_streams, n_basis, n_units) -> (n_units, n_streams, window)
    filters = np.einsum("kbu,tb->ukt", np.asarray(w), kernels)

    out = {
        "unit_ids": np.asarray(spikes.index),
        "filters": filters,
        "intercept": intercept,
        "ll_full": ll_full,
        "ll_reduced": ll_red,
        "kernel_time": np.arange(WINDOW_BINS) * BIN,
        "bin_size": BIN,
    }
    with open("glm_results.pkl", "wb") as f:
        pickle.dump(out, f)

    d = (ll_full - ll_red) / BIN  # per second of recording
    print(f"held-out delta log-likelihood (full - reduced), nats/s: "
          f"median {np.median(d):.3f}, {np.mean(d > 0):.0%} of units improved")


if __name__ == "__main__":
    main()
