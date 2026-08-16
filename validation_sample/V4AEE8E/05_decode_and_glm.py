"""
Two population-level tests of orientation coding:

1. Cross-validated Poisson naive-Bayes decoding of grating orientation from single-trial
   population activity, as a function of population size, separately for visual cortex,
   visual thalamus and hippocampus.
2. A NeMoS Poisson GLM in which direction and temporal frequency are separate additive
   terms, so the contribution of direction can be isolated by cross-validated
   log-likelihood.
"""
import pickle
import numpy as np
import pandas as pd
from scipy.special import gammaln
from tqdm import tqdm
import nemos as nmo

RNG = np.random.default_rng(1)
POP_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]
N_REPEATS = 20
N_FOLDS = 5


# --------------------------------------------------------------------------------------
# Poisson naive-Bayes decoder
# --------------------------------------------------------------------------------------
def poisson_nb_accuracy(counts, labels, levels, n_folds=N_FOLDS, seed=0):
    """
    Leave-fold-out decoding accuracy.  counts: (n_trials, n_units) of spike counts.

    Class conditional means are estimated on the training trials; a test trial is
    assigned the class maximising the Poisson log-likelihood summed over units.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(labels))
    folds = np.array_split(order, n_folds)
    y = np.searchsorted(levels, labels)

    correct = 0
    conf = np.zeros((len(levels), len(levels)), int)
    for f in range(n_folds):
        test = folds[f]
        train = np.concatenate([folds[g] for g in range(n_folds) if g != f])
        lam = np.stack([counts[train][y[train] == c].mean(axis=0) for c in range(len(levels))])
        lam = np.clip(lam, 1e-3, None)                       # (n_levels, n_units)
        ll = counts[test] @ np.log(lam).T - lam.sum(axis=1)  # (n_test, n_levels)
        pred = np.argmax(ll, axis=1)
        correct += np.sum(pred == y[test])
        np.add.at(conf, (y[test], pred), 1)
    return correct / len(labels), conf


def decoding_curve(counts, labels, levels, unit_pool, sizes=POP_SIZES, n_repeats=N_REPEATS, seed=0):
    """Accuracy vs. number of randomly chosen units."""
    rng = np.random.default_rng(seed)
    rows = []
    for n in sizes:
        if n > len(unit_pool):
            continue
        for rep in range(n_repeats):
            cols = rng.choice(unit_pool, size=n, replace=False)
            acc, _ = poisson_nb_accuracy(counts[:, cols], labels, levels, seed=seed + rep)
            rows.append(dict(n_units=n, repeat=rep, accuracy=acc))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# NeMoS population GLM
# --------------------------------------------------------------------------------------
def poisson_ll(counts, rate):
    """
    Poisson log-likelihood per neuron.

    The ``log(k!)`` normalising term is kept.  It is constant across models and so
    cancels in a likelihood *difference*, but McFadden's pseudo-R^2 is a likelihood
    *ratio*; dropping the term leaves the log-likelihood positive for units firing more
    than about one spike per trial and makes the ratio meaningless.
    """
    rate = np.clip(rate, 1e-8, None)
    return np.sum(counts * np.log(rate) - rate - gammaln(counts + 1), axis=0)


def glm_direction_contribution(counts, direction, temporal_freq, n_folds=N_FOLDS, seed=0):
    """
    Cross-validated McFadden pseudo-R^2 for nested GLMs of trial spike counts.

    Returns a dict of per-neuron pseudo-R^2 for a TF-only model, a direction-only model,
    and the full additive model, all relative to an intercept-only null.
    """
    dir_basis = nmo.basis.CyclicBSplineEval(n_basis_funcs=8, bounds=(0.0, 360.0), label="direction")
    tf_basis = nmo.basis.BSplineEval(n_basis_funcs=4, bounds=(0.0, 4.0), label="log2_tf")

    X_dir = np.asarray(dir_basis.compute_features(direction))
    X_tf = np.asarray(tf_basis.compute_features(np.log2(temporal_freq)))
    designs = {
        "tf_only": X_tf,
        "direction_only": X_dir,
        "direction_plus_tf": np.hstack([X_dir, X_tf]),
    }

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(direction))
    folds = np.array_split(order, n_folds)

    ll = {k: np.zeros(counts.shape[1]) for k in list(designs) + ["null"]}
    for f in range(n_folds):
        test = folds[f]
        train = np.concatenate([folds[g] for g in range(n_folds) if g != f])
        null_rate = np.broadcast_to(counts[train].mean(axis=0), (len(test), counts.shape[1]))
        ll["null"] += poisson_ll(counts[test], null_rate)
        for name, X in designs.items():
            model = nmo.glm.PopulationGLM(regularizer="Ridge", regularizer_strength=1e-3,
                                          solver_name="LBFGS",
                                          solver_kwargs=dict(maxiter=5000, tol=1e-6))
            model.fit(X[train], counts[train])
            ll[name] += poisson_ll(counts[test], np.asarray(model.predict(X[test])))

    n_trials = len(direction)
    denom = ll["null"]
    out = {}
    for name in designs:
        with np.errstate(invalid="ignore", divide="ignore"):
            out[f"pr2_{name}"] = 1 - ll[name] / denom
        # Held-out log-likelihood gain over the intercept-only model, in nats per trial.
        # Unlike the ratio this stays interpretable for units with very few spikes.
        out[f"nats_{name}"] = (ll[name] - denom) / n_trials

    # Fit the full model once on all trials so the tuning curve can be visualised.
    model = nmo.glm.PopulationGLM(regularizer="Ridge", regularizer_strength=1e-3,
                                  solver_name="LBFGS", solver_kwargs=dict(maxiter=5000, tol=1e-6))
    model.fit(designs["direction_plus_tf"], counts)
    grid = np.linspace(0, 360, 181, endpoint=False)
    Xg = np.hstack([
        np.asarray(dir_basis.compute_features(grid)),
        np.tile(np.asarray(tf_basis.compute_features(np.log2(temporal_freq))).mean(axis=0), (len(grid), 1)),
    ])
    out["glm_curve"] = np.asarray(model.predict(Xg))  # (n_grid, n_units)
    out["glm_grid"] = grid
    return out


# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    with open("responses.pkl", "rb") as f:
        results = pickle.load(f)
    units = pd.read_parquet("unit_metrics.parquet")

    decode_rows, conf_store, glm_rows, glm_curves = [], {}, [], {}

    for res in tqdm(results, desc="sessions"):
        sid = res["session"]
        meta = res["meta"]
        assert list(meta["uid"]) == list(units.loc[units.session == sid, "uid"]), "unit order mismatch"

        # ---- static gratings: 6-way orientation decoding ----
        sg = res["sg_table"]
        sg_counts = np.rint(res["sg_rates"] * 0.25).astype(int)   # SG_WINDOW spans 0.25 s
        oris = np.sort(sg["orientation"].unique())
        # ---- drifting gratings: 8-way direction decoding ----
        dg = res["dg_table"]
        dg_counts = np.rint(res["dg_rates"] * 2.0).astype(int)
        dirs = np.sort(dg["orientation"].unique())

        for group in ["visual cortex", "visual thalamus", "hippocampus"]:
            pool = np.flatnonzero(meta["area_group"].values == group)
            if len(pool) < 4:
                continue
            for stim, counts, labels, levels in [
                ("static gratings (6 orientations)", sg_counts, sg["orientation"].values, oris),
                ("drifting gratings (8 directions)", dg_counts, dg["orientation"].values, dirs),
            ]:
                curve = decoding_curve(counts, labels, levels, pool, seed=int(sid) % 1000)
                curve["session"], curve["area_group"], curve["stimulus"] = sid, group, stim
                curve["chance"] = 1 / len(levels)
                decode_rows.append(curve)

        # full-population confusion matrix for V1 on static gratings
        v1 = np.flatnonzero(meta["area"].values == "VISp")
        if len(v1) >= 10:
            acc, conf = poisson_nb_accuracy(sg_counts[:, v1], sg["orientation"].values, oris)
            conf_store[sid] = dict(conf=conf, acc=acc, n_units=len(v1), levels=oris)

        # ---- GLM on drifting gratings, cortical units only ----
        ctx = np.flatnonzero(meta["area_group"].values == "visual cortex")
        g = glm_direction_contribution(dg_counts[:, ctx], dg["orientation"].values,
                                       dg["temporal_frequency"].values)
        glm_rows.append(pd.DataFrame({
            "uid": meta["uid"].values[ctx],
            "area": meta["area"].values[ctx],
            "session": sid,
            "pr2_tf_only": g["pr2_tf_only"],
            "pr2_direction_only": g["pr2_direction_only"],
            "pr2_direction_plus_tf": g["pr2_direction_plus_tf"],
            "nats_tf_only": g["nats_tf_only"],
            "nats_direction_only": g["nats_direction_only"],
            "nats_direction_plus_tf": g["nats_direction_plus_tf"],
        }))
        glm_curves[sid] = dict(curve=g["glm_curve"], grid=g["glm_grid"], uid=meta["uid"].values[ctx])

    decode = pd.concat(decode_rows, ignore_index=True)
    decode.to_parquet("decoding.parquet")
    glm = pd.concat(glm_rows, ignore_index=True)
    glm.to_parquet("glm_scores.parquet")
    with open("decode_extras.pkl", "wb") as f:
        pickle.dump(dict(conf=conf_store, glm_curves=glm_curves), f)

    print("\nDecoding accuracy at the largest common population size:")
    print(decode.groupby(["stimulus", "area_group", "n_units"])["accuracy"].mean().unstack(0).round(3))
    print("\nMedian cross-validated pseudo-R^2 (cortical units, drifting gratings):")
    print(glm[["pr2_tf_only", "pr2_direction_only", "pr2_direction_plus_tf"]].median().round(4))
    print("\nMedian held-out log-likelihood gain (nats/trial):")
    print(glm[["nats_tf_only", "nats_direction_only", "nats_direction_plus_tf"]].median().round(4))
