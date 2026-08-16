"""Single-neuron encoding of the prior block in the pre-stimulus window (NeMoS).

The decoding analysis says the population carries the upcoming bias.  This asks
the encoding-side question: how many individual neurons change their firing rate
in the 400 ms before stimulus onset depending on which prior block the animal is
in, once slow drift and behavioural history are accounted for?

Each session is fit with one Poisson ``PopulationGLM`` whose design matrix is
  * a B-spline basis over trial index (absorbs slow drift in firing rate, which
    is the main thing that could masquerade as a block effect),
  * the block identity,
  * previous choice and previous reward.

Significance is assessed against pseudo-sessions: the same design with the block
column replaced by a surrogate block sequence drawn from the IBL block-length
distribution.
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pandas as pd
import nemos as nmo
from tqdm import tqdm

import decoding as dec
from importlib import import_module

an = import_module("03_analyze")

N_PSEUDO = 20
DRIFT_BASIS = 6


def design(tr, block):
    """(n_trials, n_features) design matrix; column -3 is the block regressor."""
    n = len(tr)
    drift = nmo.basis.BSplineEval(n_basis_funcs=DRIFT_BASIS).compute_features(
        np.linspace(0, 1, n)
    )
    prev_choice = np.nan_to_num(tr["prev_choice_right"].to_numpy(), nan=0.5)
    prev_rew = np.nan_to_num(tr["prev_rewarded"].to_numpy(), nan=0.5)
    return np.column_stack(
        [np.asarray(drift), block.astype(float), prev_choice, prev_rew]
    ).astype(np.float32)


def fit_block_coefs(X, y):
    model = nmo.glm.PopulationGLM(
        observation_model="Poisson",
        regularizer="Ridge",
        regularizer_strength=0.01,
        solver_name="LBFGS",
        solver_kwargs={"tol": 1e-8, "maxiter": 400},
    )
    model.fit(X, y)
    return np.asarray(model.coef_[-3])  # block coefficient per neuron


def run(sessions):
    real_rows, null_rows = [], []
    for sid, d in tqdm(sessions.items(), desc="GLM encoding"):
        tr = d["trials"]
        block = tr["block_right"].to_numpy()
        if len(block) < an.MIN_TRIALS or len(np.unique(block)) < 2:
            continue
        y = an.window_sum(d, -0.4, 0.0).astype(np.float32)
        # A Poisson GLM cannot initialise on a unit that never fires in the
        # window; those units carry no information here anyway.
        active = y.sum(0) >= 5
        y = y[:, active]
        regions, locations = d["regions"][active], d["locations"][active]

        beta = fit_block_coefs(design(tr, block), y)
        real_rows.append(
            pd.DataFrame(dict(session=sid, region=regions, location=locations,
                              firing_rate=d["firing_rate"][active], beta=beta))
        )

        rng = np.random.default_rng(abs(hash(sid)) % 10000)
        for j in range(N_PSEUDO):
            pb = dec.pseudo_blocks(len(block), rng=rng)
            if len(np.unique(pb)) < 2:
                continue
            null_rows.append(
                pd.DataFrame(dict(session=sid, draw=j, region=regions,
                                  beta=fit_block_coefs(design(tr, pb), y)))
            )
    return pd.concat(real_rows, ignore_index=True), pd.concat(null_rows, ignore_index=True)


if __name__ == "__main__":
    sessions = an.load_sessions()
    real, null = run(sessions)
    real.to_csv("results/glm_block_coefs.csv", index=False)
    null.to_csv("results/glm_block_coefs_null.csv", index=False)

    thr = np.percentile(np.abs(null["beta"]), 97.5)
    real["significant"] = np.abs(real["beta"]) > thr
    print(f"null |beta| 97.5th pct = {thr:.4f}")
    print(f"units with block modulation: {real.significant.mean():.1%} "
          f"({real.significant.sum()}/{len(real)}); expected under null 2.5%")
    print(real.groupby("region").significant.agg(["mean", "sum", "count"]).sort_values("mean"))
    real.to_csv("results/glm_block_coefs.csv", index=False)
