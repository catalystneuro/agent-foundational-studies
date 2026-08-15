"""Stage 5: model-based analyses.

(a) NeMoS Poisson GLM: per-trial spike counts as a function of grating direction
    (cyclic B-spline basis), cross-validated against a mean-rate-only model.
(b) Population decoding of grating direction / orientation from spike counts.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

import jax

jax.config.update("jax_enable_x64", True)
import nemos as nmo
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix

import orientation_lib as ol

N_SESSIONS = int(sys.argv[1]) if len(sys.argv) > 1 else 4
sessions = ol.list_session_assets().head(N_SESSIONS)


def session_rates(row):
    """Trial-rate matrices for one session, cached on disk."""
    p = f"cache/rates_{row.session_id}.npz"
    if os.path.exists(p):
        return np.load(p, allow_pickle=True)
    D = ol.load_and_prepare(row["url"])
    np.savez_compressed(
        p,
        dg_counts=(D["dg_rates"] * D["dg"].duration.values[:, None]),
        dg_rates=D["dg_rates"], dgb_rates=D["dgb_rates"],
        direction=D["dg"].direction.values, tf=D["dg"].temporal_frequency.values,
        duration=D["dg"].duration.values,
        unit_id=D["units"].unit_id.values, region=D["units"].region.values.astype(str),
    )
    del D
    return np.load(p, allow_pickle=True)


# ===========================================================================
# (a) NeMoS GLM: direction encoding, cross-validated
# ===========================================================================
def glm_cv(counts, direction, n_basis=8, n_folds=5, seed=0, orientation_only=False):
    """5-fold cross-validated Poisson GLM of trial spike counts on grating angle.

    Returns the per-unit deviance explained on held-out trials, relative to a
    model that only knows the mean rate.
    """
    # The orientation model folds direction modulo 180 and rescales onto the full
    # circle, leaving only 4 distinct stimulus values; it therefore gets 4 basis
    # functions. Giving it 8 would make the design rank-deficient (and the solver
    # would run to max iterations without converging).
    ang = (direction % 180.0) * 2 if orientation_only else direction
    n_basis = 4 if orientation_only else n_basis
    basis = nmo.basis.CyclicBSplineEval(n_basis_funcs=n_basis, bounds=(0.0, 360.0),
                                        label="direction")
    X = np.asarray(basis.compute_features(ang), dtype=float)
    # The cyclic B-spline design is a partition of unity, so its rows sum to one
    # and it is exactly collinear with the intercept. Centring the columns removes
    # that flat direction; without it LBFGS crawls along the degenerate axis.
    X = X - X.mean(0)
    y = np.asarray(counts, dtype=float)
    n_units = y.shape[1]

    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    ll_model = np.zeros(n_units)
    ll_null = np.zeros(n_units)
    ll_sat = np.zeros(n_units)
    for tr, te in kf.split(X, direction):
        model = nmo.glm.PopulationGLM(
            observation_model="Poisson", solver_name="LBFGS",
            regularizer="Ridge", regularizer_strength=1e-2,
            solver_kwargs=dict(maxiter=3000, tol=1e-10))
        model.fit(X[tr], y[tr])
        mu = np.asarray(model.predict(X[te]))
        mu = np.clip(mu, 1e-8, None)
        mu0 = np.clip(y[tr].mean(0)[None, :], 1e-8, None)
        yte = y[te]
        ll_model += (yte * np.log(mu) - mu).sum(0)
        ll_null += (yte * np.log(mu0) - mu0).sum(0)
        ll_sat += (yte * np.log(np.clip(yte, 1e-8, None)) - yte).sum(0)
    dev_expl = 1 - (ll_sat - ll_model) / (ll_sat - ll_null)
    return dev_expl, basis


rows = []
glm_curves = {}
for _, row in sessions.iterrows():
    z = session_rates(row)
    counts = z["dg_counts"]
    direction = z["direction"]
    region = z["region"].astype(str)
    print("session %s: GLM on %d units x %d trials" % (row.session_id, counts.shape[1],
                                                       counts.shape[0]))
    de_dir, basis = glm_cv(counts, direction)
    de_ori, _ = glm_cv(counts, direction, orientation_only=True)
    rows.append(pd.DataFrame(dict(session_id=row.session_id, unit_id=z["unit_id"],
                                  region=region, dev_dir=de_dir, dev_ori=de_ori)))
    if row.session_id == sessions.session_id.iloc[0]:
        # smooth GLM tuning curves for a few example units, fitted on all trials
        Xraw = np.asarray(basis.compute_features(direction), dtype=float)
        Xmean = Xraw.mean(0)
        X = Xraw - Xmean
        m = nmo.glm.PopulationGLM(
            observation_model="Poisson", solver_name="LBFGS",
            regularizer="Ridge", regularizer_strength=1e-2,
            solver_kwargs=dict(maxiter=3000, tol=1e-10))
        m.fit(X, counts.astype(float))
        grid = np.linspace(0, 359.9, 361)
        Xg = np.asarray(basis.compute_features(grid), dtype=float) - Xmean
        glm_curves = dict(grid=grid, pred=np.asarray(m.predict(Xg)) / 2.0,
                          unit_id=z["unit_id"], region=region,
                          direction=direction, rates=z["dg_rates"])

glm = pd.concat(rows, ignore_index=True)
glm.to_csv("cache/glm_results.csv", index=False)
print("\nmedian cross-validated deviance explained by grating direction:")
print(glm.groupby("region")[["dev_dir", "dev_ori"]].median().sort_values(
    "dev_dir", ascending=False).to_string())

# ===========================================================================
# (b) Population decoding
# ===========================================================================
def decode(counts, labels, groups_mask, n_folds=5, seed=0):
    """Multinomial logistic decoding of `labels` from population counts."""
    X = counts[:, groups_mask]
    if X.shape[1] < 5:
        return np.nan, None, None
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, C=0.05))
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    pred = np.empty_like(labels)
    for tr, te in kf.split(X, labels):
        clf.fit(X[tr], labels[tr])
        pred[te] = clf.predict(X[te])
    acc = (pred == labels).mean()
    cm = confusion_matrix(labels, pred, normalize="true")
    return acc, cm, pred


dec_rows = []
cms = {}
for _, row in sessions.iterrows():
    z = session_rates(row)
    counts, direction = z["dg_counts"], z["direction"]
    region = z["region"].astype(str)
    ori = direction % 180.0
    for name, mask in [("visual cortex", np.isin(region, ol.VISUAL_CORTEX)),
                       ("thalamus", np.isin(region, ol.THALAMUS)),
                       ("CA1", region == "CA1")]:
        acc_d, cm_d, _ = decode(counts, direction.astype(int), mask)
        acc_o, cm_o, _ = decode(counts, ori.astype(int), mask)
        dec_rows.append(dict(session_id=row.session_id, group=name, n_units=int(mask.sum()),
                             acc_dir=acc_d, acc_ori=acc_o))
        if row.session_id == sessions.session_id.iloc[0]:
            cms[name] = (cm_d, cm_o)
        print("  %-14s n=%3d  direction %.2f (chance 0.125)  orientation %.2f (chance 0.25)"
              % (name, mask.sum(), acc_d, acc_o))

dec = pd.DataFrame(dec_rows)
dec.to_csv("cache/decoding_results.csv", index=False)
print(dec.groupby("group")[["acc_dir", "acc_ori"]].mean().to_string())

# --- accuracy as a function of population size (cortex, first session) ---
z = session_rates(sessions.iloc[0])
counts, direction = z["dg_counts"], z["direction"]
region = z["region"].astype(str)
rng = np.random.default_rng(0)
sizes = [2, 5, 10, 20, 40, 80, 160]
curve = {}
for name, mask in [("visual cortex", np.isin(region, ol.VISUAL_CORTEX)),
                   ("thalamus", np.isin(region, ol.THALAMUS)),
                   ("CA1", region == "CA1")]:
    idx = np.where(mask)[0]
    accs = []
    for s in sizes:
        if s > len(idx):
            accs.append(np.nan)
            continue
        a = [decode(counts, direction.astype(int),
                    rng.choice(idx, s, replace=False))[0] for _ in range(5)]
        accs.append(np.mean(a))
    curve[name] = accs
    print(name, np.round(accs, 3))
np.savez("cache/decoding_curve.npz", sizes=sizes, **{k: v for k, v in curve.items()})

# ===========================================================================
# FIGURES
# ===========================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

ax = axes[0]
order = ["VISp", "VISl", "VISrl", "VISal", "VISam", "VISpm", "LGd", "LP", "CA1"]
order = [r for r in order if (glm.region == r).sum() >= 20]
data = [glm[glm.region == r].dev_dir.clip(-0.02, None).values for r in order]
bp = ax.boxplot(data, showfliers=False, patch_artist=True, widths=0.6)
for patch, r in zip(bp["boxes"], order):
    patch.set_facecolor("C0" if r.startswith("VIS") else ("C1" if r in ol.THALAMUS else "0.7"))
ax.axhline(0, color="k", ls="--", lw=0.8)
ax.set_xticklabels(order, rotation=45)
ax.set_ylabel("cross-validated deviance explained")
ax.set_title("NeMoS Poisson GLM:\nhow much of the spiking direction explains", fontsize=10)

ax = axes[1]
ctx = glm[glm.region.isin(ol.VISUAL_CORTEX)]
ax.plot(ctx.dev_dir, ctx.dev_ori, ".", ms=3, alpha=0.5, color="C0")
lim = [-0.02, max(0.05, np.nanpercentile(ctx.dev_dir, 99))]
ax.plot(lim, lim, "k--", lw=0.8)
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_xlabel("deviance explained, direction model (360$\\degree$)")
ax.set_ylabel("orientation model (180$\\degree$)")
ax.set_title("Most direction tuning is\n180$\\degree$-periodic (orientation)", fontsize=10)

ax = axes[2]
for name, c in [("visual cortex", "C0"), ("thalamus", "C1"), ("CA1", "0.5")]:
    ax.plot(sizes, curve[name], "o-", color=c, label=name)
ax.axhline(1 / 8, color="k", ls="--", lw=0.8, label="chance")
ax.set_xscale("log")
ax.set_xlabel("number of units in decoded population")
ax.set_ylabel("8-way direction decoding accuracy")
ax.set_title("Population decoding of grating direction", fontsize=10)
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("figures/fig06_glm_decoding.png", dpi=140, bbox_inches="tight")
plt.close()

# confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
dirs8 = np.arange(0, 360, 45)
for ax, name in zip(axes, ["visual cortex", "thalamus", "CA1"]):
    cm = cms[name][0]
    im = ax.imshow(cm, vmin=0, vmax=max(0.3, cm.max()), cmap="magma")
    ax.set_xticks(range(8))
    ax.set_xticklabels(dirs8, rotation=45, fontsize=8)
    ax.set_yticks(range(8))
    ax.set_yticklabels(dirs8, fontsize=8)
    ax.set_xlabel("decoded direction (deg)")
    ax.set_ylabel("true direction (deg)")
    acc = np.trace(cm) / 8
    ax.set_title("%s\naccuracy %.2f (chance 0.125)" % (name, acc), fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
plt.savefig("figures/fig07_confusion.png", dpi=140, bbox_inches="tight")
plt.close()

# GLM tuning curves for example units
if glm_curves:
    g0 = glm.groupby(["session_id", "unit_id"]).first().reset_index()
    first = glm[glm.session_id == sessions.session_id.iloc[0]]
    ctx_idx = np.where(np.isin(glm_curves["region"], ol.VISUAL_CORTEX))[0]
    best = ctx_idx[np.argsort(first.dev_dir.values[ctx_idx])[::-1][:6]]
    dirs = np.unique(glm_curves["direction"])
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, j in zip(axes.ravel(), best):
        _, mu, sem, _ = ol.condition_means(glm_curves["rates"][:, [j]],
                                           glm_curves["direction"])
        ax.errorbar(dirs, mu[:, 0], yerr=sem[:, 0], fmt="o", color="k", ms=4,
                    capsize=2, label="measured (all TFs)")
        ax.plot(glm_curves["grid"], glm_curves["pred"][:, j], color="C3", lw=2,
                label="GLM (cyclic B-spline)")
        ax.set_title("unit %d (%s), dev. expl. %.2f"
                     % (glm_curves["unit_id"][j], glm_curves["region"][j],
                        first.dev_dir.values[j]), fontsize=9)
        ax.set_xticks(dirs)
        ax.set_xlabel("direction (deg)")
        ax.set_ylabel("rate (Hz)")
    axes[0, 0].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig("figures/fig08_glm_tuning.png", dpi=140, bbox_inches="tight")
    plt.close()
print("stage 5 done")
