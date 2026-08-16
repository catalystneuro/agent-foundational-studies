"""Encoding (NeMoS Poisson GLM) and decoding analyses of grating orientation."""

import pickle

import jax
import numpy as np
import pandas as pd

jax.config.update("jax_enable_x64", True)  # 32-bit LBFGS does not converge on these designs
import nemos as nmo
from scipy.special import gammaln
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

import dandi_io as dio
import tuning as tn
from run_analysis import SESSIONS, SG_WINDOW

N_FOLDS = 5
RNG = np.random.default_rng(0)


# --------------------------------------------------------------------------
# trial-level design matrix
# --------------------------------------------------------------------------
def session_trials(session, tsgroup):
    """Static-grating trials: spike counts, orientation, SF and running speed."""
    sg = session["static_gratings"]
    keep = sg[["orientation", "spatial_frequency"]].notna().all(axis=1).values
    tab = sg.loc[keep].reset_index(drop=True)
    rates = tn.trial_rates(tsgroup, tab, window=SG_WINDOW)
    dur = SG_WINDOW[1] - SG_WINDOW[0]
    counts = np.round(rates * dur).astype(int)

    t = session["running_time"]
    v = session["running_speed"]
    lo = np.searchsorted(t, tab["start_time"].values + SG_WINDOW[0])
    hi = np.searchsorted(t, tab["start_time"].values + SG_WINDOW[1])
    csum = np.concatenate([[0], np.cumsum(v)])
    n = np.maximum(hi - lo, 1)
    speed = (csum[hi] - csum[lo]) / n
    return tab, counts, speed


def poisson_ll(y, mu):
    """Per-unit Poisson log-likelihood, summed over trials."""
    mu = np.clip(mu, 1e-9, None)
    return np.sum(y * np.log(mu) - mu - gammaln(y + 1), axis=0)


def build_design(tab, speed, with_orientation=True):
    ori = tab["orientation"].values.astype(float)
    logsf = np.log2(tab["spatial_frequency"].values.astype(float))
    sp = np.clip(speed, 0, np.percentile(speed, 99))

    sf_basis = nmo.basis.BSplineEval(n_basis_funcs=4, bounds=(logsf.min(), logsf.max()), label="sf")
    speed_basis = nmo.basis.BSplineEval(n_basis_funcs=4, bounds=(sp.min(), sp.max()), label="speed")
    if with_orientation:
        ori_basis = nmo.basis.CyclicBSplineEval(n_basis_funcs=8, bounds=(0.0, 180.0), label="ori")
        basis = ori_basis + sf_basis + speed_basis
        X = basis.compute_features(ori, logsf, sp)
    else:
        basis = sf_basis + speed_basis
        X = basis.compute_features(logsf, sp)
    return np.asarray(X)


def glm_encoding(session, tsgroup, meta, n_folds=N_FOLDS):
    """Cross-validated Poisson GLM: how much does orientation improve prediction?

    A population GLM is fit twice per fold, once with a cyclic B-spline basis over
    orientation and once without. The per-unit difference in held-out
    log-likelihood isolates the orientation contribution from spatial frequency
    and running speed.
    """
    tab, counts, speed = session_trials(session, tsgroup)
    X_full = build_design(tab, speed, with_orientation=True)
    X_red = build_design(tab, speed, with_orientation=False)
    n_trials, n_units = counts.shape

    ll_full = np.zeros(n_units)
    ll_red = np.zeros(n_units)
    ll_null = np.zeros(n_units)
    folds = np.array_split(RNG.permutation(n_trials), n_folds)
    for f in tqdm(range(n_folds), desc=f"GLM {session['session_id']}", leave=False):
        test = folds[f]
        train = np.concatenate([folds[g] for g in range(n_folds) if g != f])
        y_tr, y_te = counts[train], counts[test]
        for X, acc in ((X_full, "full"), (X_red, "red")):
            model = nmo.glm.PopulationGLM(
                regularizer="Ridge", regularizer_strength=1e-4, solver_name="LBFGS",
                solver_kwargs={"maxiter": 1000, "tol": 1e-6}
            )
            model.fit(X[train], y_tr)
            mu = np.asarray(model.predict(X[test]))
            if acc == "full":
                ll_full += poisson_ll(y_te, mu)
            else:
                ll_red += poisson_ll(y_te, mu)
        ll_null += poisson_ll(y_te, np.tile(y_tr.mean(axis=0), (len(test), 1)))

    total = counts.sum(axis=0)
    return pd.DataFrame(dict(
        unit_id=meta["unit_id"].values,
        session_id=session["session_id"],
        location=meta["location"].values,
        region_group=meta["region_group"].values,
        pseudo_r2_full=1 - ll_full / ll_null,
        pseudo_r2_reduced=1 - ll_red / ll_null,
        d_ll_orientation=(ll_full - ll_red) / np.maximum(total, 1),  # nats per spike
        n_spikes=total,
    ))


# --------------------------------------------------------------------------
# population decoding
# --------------------------------------------------------------------------
def decode_orientation(counts, labels, group_mask, sizes, n_folds=N_FOLDS, n_draws=8, rng=RNG):
    """Cross-validated multinomial decoding of orientation from population counts.

    Random subsets of `sizes` units are drawn from the units flagged by
    `group_mask` so that regions with different yields are compared at matched
    population size.
    """
    active = counts.std(axis=0) > 0  # constant units carry no information and break scaling
    idx_pool = np.where(group_mask & active)[0]
    rows = []
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
    for size in sizes:
        if len(idx_pool) < size:
            continue
        for draw in range(n_draws):
            cols = rng.choice(idx_pool, size=size, replace=False)
            X = np.sqrt(counts[:, cols])  # variance-stabilising transform
            acc, acc_shuf = [], []
            for train, test in skf.split(X, labels):
                sc = StandardScaler().fit(X[train])
                clf = LogisticRegression(max_iter=2000, C=0.1)
                clf.fit(sc.transform(X[train]), labels[train])
                acc.append(clf.score(sc.transform(X[test]), labels[test]))
                y_shuf = rng.permutation(labels[train])
                clf2 = LogisticRegression(max_iter=2000, C=0.1)
                clf2.fit(sc.transform(X[train]), y_shuf)
                acc_shuf.append(clf2.score(sc.transform(X[test]), labels[test]))
            rows.append(dict(size=size, draw=draw, accuracy=np.mean(acc),
                             accuracy_shuffled=np.mean(acc_shuf)))
    return pd.DataFrame(rows)


# Sessions used for the GLM: the four with simultaneous cortical, thalamic and
# hippocampal coverage. The population GLM is the expensive analysis (10 fits per
# session over ~6000 trials), and these sessions already contain all three
# comparison groups, so extending it to all ten adds cost without adding contrast.
GLM_SESSIONS = [
    "sub-699733573/sub-699733573_ses-715093703.nwb",
    "sub-703279277/sub-703279277_ses-719161530.nwb",
    "sub-726298249/sub-726298249_ses-754829445.nwb",
    "sub-744915196/sub-744915196_ses-762602078.nwb",
]


def _glm_one(path):
    s = dio.extract_session(path)
    tsg, meta = tn.make_tsgroup(s)
    return glm_encoding(s, tsg, meta)


def _decode_one(path, sizes):
    s = dio.extract_session(path)
    tsg, meta = tn.make_tsgroup(s)
    tab, counts, _ = session_trials(s, tsg)
    labels = tab["orientation"].values.astype(int)
    out = []
    for grp in ["visual cortex", "visual thalamus", "hippocampus"]:
        mask = (meta["region_group"] == grp).values
        if mask.sum() < min(sizes):
            continue
        df = decode_orientation(counts, labels, mask, sizes)
        df["region_group"] = grp
        df["session_id"] = s["session_id"]
        out.append(df)
    return pd.concat(out, ignore_index=True)


if __name__ == "__main__":
    import sys
    from concurrent.futures import ProcessPoolExecutor

    what = sys.argv[1] if len(sys.argv) > 1 else "both"
    sizes = [5, 10, 20, 40, 80]

    if what in ("decode", "both"):
        dec = pd.concat([_decode_one(p, sizes) for p in tqdm(SESSIONS, desc="decoding")],
                        ignore_index=True)
        dec.to_pickle("results_decoding.pkl")
        print(dec[dec["size"] == 40].groupby("region_group")[["accuracy", "accuracy_shuffled"]].mean().round(3))

    if what in ("glm", "both"):
        with ProcessPoolExecutor(max_workers=4) as ex:
            glm = pd.concat(list(ex.map(_glm_one, GLM_SESSIONS)), ignore_index=True)
        glm.to_pickle("results_glm.pkl")
        print(glm.groupby("region_group")[["pseudo_r2_full", "pseudo_r2_reduced",
                                           "d_ll_orientation"]].median().round(4))
