"""Poisson GLM of tone-evoked activity with NeMoS (session sub-LA11_ses-1, 000986).

Each unit's binned spike train is modelled as a Poisson process whose rate is
driven by five frequency-specific stimulus filters, one per tone, each expanded
in a raised-cosine basis. The fitted filters give a model-based estimate of both
the response dynamics and the frequency tuning, and comparing the held-out
likelihood against a frequency-blind model (a single filter driven by every tone
regardless of its frequency) tests whether frequency tuning improves the
prediction of single-trial spiking at all.

The fit is restricted to the 80 most active units of one session, which is what
the run time of the population fit scales with. Cross-validation uses interleaved
30 s blocks rather than a held-out tail: firing rates drift over the two hours of
recording, so a tail split scores that drift, which shows up as a negative
pseudo-R2 for reasons that have nothing to do with tuning.
"""

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)   # float32 stops LBFGS short of convergence

import nemos as nmo
import pynapple as nap
from tqdm import tqdm

from dandi_io import list_assets, load_tone_session

nap.nap_config.suppress_conversion_warnings = True

SESSION = "sub-LA11/sub-LA11_ses-1_behavior.nwb"
BIN = 0.010          # s
WINDOW = 20          # bins -> 200 ms of stimulus filter
N_BASIS = 6
N_FOLDS = 3
BLOCK = 30.0         # s, size of the interleaved cross-validation blocks
N_UNITS = 60         # most active units; the population fit scales with unit count
REG = 1e-3
SOLVER = dict(maxiter=2000, tol=1e-8)


def build_design(onsets, freq, ufreq, time_grid):
    """One binary channel per tone frequency, 1 in the bin containing the onset."""
    stim = np.zeros((len(time_grid), len(ufreq)))
    idx = np.clip(np.searchsorted(time_grid, onsets) - 1, 0, len(time_grid) - 1)
    for k, f in enumerate(ufreq):
        stim[idx[freq == f], k] = 1.0
    return stim


def poisson_ll(y, mu):
    """Per-unit Poisson log-likelihood, dropping the y! constant."""
    mu = np.clip(mu, 1e-9, None)
    return (y * np.log(mu) - mu).sum(0)


def main():
    url = dict(list_assets("000986"))[SESSION]
    S = load_tone_session(url)
    spikes, onsets, freq = S["spikes"], S["onsets"], S["freq"]
    ufreq = np.unique(freq)

    ep = nap.IntervalSet(start=onsets[0] - 1.0, end=onsets[-1] + 1.0)
    counts = spikes.count(BIN, ep=ep)
    time_grid = counts.t
    keep = np.argsort(np.asarray(spikes.rate))[::-1][:N_UNITS]
    keep = np.sort(keep)
    y = np.asarray(counts).astype(float)[:, keep]
    unit_ids = np.array(list(spikes.keys()))[keep]
    stim = build_design(onsets, freq, ufreq, time_grid)

    basis = nmo.basis.RaisedCosineLogConv(n_basis_funcs=N_BASIS, window_size=WINDOW)
    X_full = np.asarray(basis.compute_features(stim))
    X_blind = np.asarray(basis.compute_features(stim.sum(1, keepdims=True)))
    _, kernels = basis.evaluate_on_grid(WINDOW)          # (WINDOW, N_BASIS)

    valid = np.isfinite(X_full).all(1) & np.isfinite(X_blind).all(1)
    block = ((time_grid - time_grid[0]) // BLOCK).astype(int)
    fold = block % N_FOLDS

    ll_full = np.zeros(y.shape[1])
    ll_blind = np.zeros(y.shape[1])
    ll_null = np.zeros(y.shape[1])
    n_spk = np.zeros(y.shape[1])
    coefs = []

    for k in tqdm(range(N_FOLDS), desc="CV folds"):
        tr = valid & (fold != k)
        te = valid & (fold == k)
        m_full = nmo.glm.PopulationGLM(regularizer="Ridge", regularizer_strength=REG,
                                       solver_name="LBFGS", solver_kwargs=SOLVER)
        m_full.fit(X_full[tr], y[tr])
        m_blind = nmo.glm.PopulationGLM(regularizer="Ridge", regularizer_strength=REG,
                                        solver_name="LBFGS", solver_kwargs=SOLVER)
        m_blind.fit(X_blind[tr], y[tr])

        ll_full += poisson_ll(y[te], np.asarray(m_full.predict(X_full[te])))
        ll_blind += poisson_ll(y[te], np.asarray(m_blind.predict(X_blind[te])))
        ll_null += poisson_ll(y[te], np.broadcast_to(y[tr].mean(0), y[te].shape))
        n_spk += y[te].sum(0)
        coefs.append(np.asarray(m_full.coef_))           # (n_features, n_units)

    coef = np.mean(coefs, 0).T.reshape(len(unit_ids), len(ufreq), N_BASIS)
    filters = np.einsum("wb,ufb->uwf", kernels, coef)    # (n_units, WINDOW, n_freq)
    intercept = np.asarray(m_full.intercept_)
    base_rate = np.exp(intercept) / BIN
    tuning_glm = np.exp(intercept[:, None] + filters.max(1)) / BIN - base_rate[:, None]

    np.savez_compressed(
        "results_glm_000986.npz", unit_ids=unit_ids, ufreq=ufreq, filters=filters,
        kernel_t=np.arange(WINDOW) * BIN, tuning_glm=tuning_glm,
        base_rate=base_rate, ll_full=ll_full, ll_blind=ll_blind, ll_null=ll_null,
        n_spikes=n_spk, session=SESSION, bin_size=BIN)
    S["io"].close()

    r2_full = 1 - ll_full / ll_null
    r2_blind = 1 - ll_blind / ll_null
    bits = (ll_full - ll_blind) / np.maximum(n_spk, 1) / np.log(2)
    print(f"held-out McFadden pseudo-R2: frequency-specific {np.median(r2_full):.4f}, "
          f"frequency-blind {np.median(r2_blind):.4f} (median over {len(unit_ids)} units)")
    print(f"frequency-specific model better in "
          f"{np.mean(ll_full > ll_blind):.0%} of units; "
          f"median gain {np.median(bits):.4f} bits/spike")
    print("saved results_glm_000986.npz")


if __name__ == "__main__":
    main()
