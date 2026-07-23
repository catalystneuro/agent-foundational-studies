"""Poisson GLM encoding models (NeMoS): how much of the spiking do direction and speed explain?

Five nested feature sets are compared on held-out data:
    constant      intercept only
    speed         B-spline over hand speed
    direction     cyclic B-spline over the direction of the velocity vector
    dir + speed   the two additively
    dir x speed   their outer product, i.e. an unconstrained velocity tuning surface

Cross-validated McFadden pseudo-R^2 is computed per unit from the held-out Poisson
log-likelihood, so the extra parameters of the larger models are not free.
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nemos as nmo
from tqdm import tqdm

from s02_behavior import build
from s04_velocity_tuning import BIN, MOVE_SPEED, binned, moving_epochs, shift_velocity

LAG = 0.10       # s, population-optimal lead time measured in s04
N_FOLDS = 5
N_DIR_BASIS = 8
N_SPEED_BASIS = 5


def design(S, lag=LAG):
    """Spike counts and the velocity features that predict them, restricted to moving bins."""
    counts, _, vel = binned(S, S["obs"])
    a, b = shift_velocity(counts, vel, lag)
    Y = np.asarray(counts, dtype=np.float64)[a]
    V = np.asarray(vel, dtype=np.float64)[b]
    sp = np.linalg.norm(V, axis=1)
    keep = sp > MOVE_SPEED
    Y, V, sp = Y[keep], V[keep], sp[keep]
    ang = np.arctan2(V[:, 1], V[:, 0])
    return Y, ang, sp


def feature_sets(ang, sp):
    dir_basis = nmo.basis.CyclicBSplineEval(n_basis_funcs=N_DIR_BASIS, bounds=(-np.pi, np.pi),
                                            label="direction")
    # Eval bases return NaN outside `bounds`, so cover the full observed speed range.
    speed_basis = nmo.basis.BSplineEval(n_basis_funcs=N_SPEED_BASIS,
                                        bounds=(float(sp.min()), float(sp.max())),
                                        label="speed")
    return {
        "constant": np.zeros((len(ang), 0)),
        "speed": speed_basis.compute_features(sp),
        "direction": dir_basis.compute_features(ang),
        "dir + speed": (dir_basis + speed_basis).compute_features(ang, sp),
        "dir x speed": (dir_basis * speed_basis).compute_features(ang, sp),
    }, dir_basis, speed_basis


def poisson_ll(y, rate):
    """Poisson log-likelihood per unit, up to the y! term that cancels in pseudo-R^2."""
    r = np.clip(rate, 1e-9, None)
    return (y * np.log(r) - r).sum(0)


def cv_pseudo_r2(X, Y, n_folds=N_FOLDS):
    """Per-unit McFadden pseudo-R^2 on held-out contiguous blocks."""
    folds = np.array_split(np.arange(len(Y)), n_folds)
    ll_model = np.zeros(Y.shape[1])
    ll_null = np.zeros(Y.shape[1])
    for f in folds:
        tr = np.setdiff1d(np.arange(len(Y)), f)
        if X.shape[1] == 0:
            rate = np.repeat(Y[tr].mean(0)[None, :], len(f), axis=0)
        else:
            glm = nmo.glm.PopulationGLM(
                observation_model="Poisson", regularizer="Ridge",
                regularizer_strength=1e-4, solver_name="LBFGS",
            )
            glm.fit(X[tr], Y[tr])
            rate = np.asarray(glm.predict(X[f]))
        ll_model += poisson_ll(Y[f], rate)
        ll_null += poisson_ll(Y[f], np.repeat(Y[tr].mean(0)[None, :], len(f), axis=0))
    return 1 - ll_model / ll_null


def run(S):
    Y, ang, sp = design(S)
    print(f"GLM design: {Y.shape[0]} bins x {Y.shape[1]} units, "
          f"{Y.sum():.0f} spikes total, mean rate {Y.mean()/BIN:.1f} Hz")

    feats, dir_basis, speed_basis = feature_sets(ang, sp)
    for name, X in feats.items():
        assert np.isfinite(np.asarray(X)).all(), f"non-finite features in {name}"
    scores = {}
    for name, X in tqdm(feats.items(), desc="GLM model comparison"):
        scores[name] = cv_pseudo_r2(np.asarray(X, dtype=np.float64), Y)
        print(f"  {name:12s} median pseudo-R2 = {np.median(scores[name]):.4f}")

    # Fit the full interaction model once on all data to read out tuning surfaces.
    Xfull = np.asarray(feats["dir x speed"], dtype=np.float64)
    glm = nmo.glm.PopulationGLM(
        observation_model="Poisson", regularizer="Ridge",
        regularizer_strength=1e-4, solver_name="LBFGS",
    ).fit(Xfull, Y)

    # Predicted rate over a (direction, speed) grid
    grid_ang = np.linspace(-np.pi, np.pi, 49)
    grid_sp = np.linspace(float(sp.min()), float(np.percentile(sp, 99)), 25)
    A, Sp = np.meshgrid(grid_ang, grid_sp, indexing="ij")
    Xg = np.asarray((dir_basis * speed_basis).compute_features(A.ravel(), Sp.ravel()),
                    dtype=np.float64)
    pred = np.asarray(glm.predict(Xg)) / BIN            # Hz
    pred = pred.reshape(len(grid_ang), len(grid_sp), Y.shape[1])

    return dict(scores=scores, grid_ang=grid_ang, grid_sp=grid_sp, pred=pred,
                Y=Y, ang=ang, sp=sp)


def figure_glm(G, path="figures/fig06_glm_encoding.png"):
    names = list(G["scores"].keys())[1:]  # drop the constant model, whose score is 0 by definition
    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.62, wspace=0.32)

    ax = fig.add_subplot(gs[0, 0])
    data = [G["scores"][n] for n in names]
    bp = ax.boxplot(data, tick_labels=names, showfliers=False, patch_artist=True)
    for patch, c in zip(bp["boxes"], ["0.7", "tab:blue", "tab:purple", "crimson"]):
        patch.set_facecolor(c)
    ax.set_ylabel("cross-validated pseudo-$R^2$")
    ax.set_title("Encoding model comparison (n=%d units)" % len(data[0]), fontsize=11)
    ax.tick_params(axis="x", rotation=20)

    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(G["scores"]["direction"], G["scores"]["dir x speed"], s=16,
               color="crimson", alpha=0.75)
    lim = [0, max(G["scores"]["dir x speed"].max(), G["scores"]["direction"].max()) * 1.1]
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("pseudo-$R^2$, direction only")
    ax.set_ylabel("pseudo-$R^2$, direction $\\times$ speed")
    win = np.mean(G["scores"]["dir x speed"] > G["scores"]["direction"]) * 100
    ax.set_title(f"Speed adds to direction for {win:.0f}% of units", fontsize=11)

    ax = fig.add_subplot(gs[0, 2])
    gain = G["scores"]["dir x speed"] - G["scores"]["direction"]
    ax.hist(gain, bins=30, color="crimson", alpha=0.8)
    ax.axvline(0, color="k", lw=0.8)
    ax.axvline(np.median(gain), color="k", ls="--",
               label=f"median +{np.median(gain):.4f}")
    ax.set_xlabel("$\\Delta$ pseudo-$R^2$ from adding speed")
    ax.set_ylabel("units")
    ax.set_title("Improvement from the speed term", fontsize=11)
    ax.legend(fontsize=9)

    # GLM-predicted direction tuning at several speeds. Examples are the best-fit units
    # among those whose peak rate grows with speed (a minority of units are instead most
    # active when the hand is slow).
    peak = G["pred"].max(axis=0)           # (speed, unit)
    grows = peak[-1] > 1.5 * peak[0]
    order = np.where(grows)[0][np.argsort(-G["scores"]["dir x speed"][grows])]
    cmap = plt.get_cmap("viridis")
    speeds_to_show = [1, len(G["grid_sp"]) // 3, 2 * len(G["grid_sp"]) // 3, len(G["grid_sp"]) - 1]
    for j, u in enumerate(order[:3]):
        ax = fig.add_subplot(gs[1, j], projection="polar")
        for si in speeds_to_show:
            ax.plot(G["grid_ang"], G["pred"][:, si, u],
                    color=cmap(si / (len(G["grid_sp"]) - 1)),
                    label=f"{G['grid_sp'][si]:.0f} cm/s")
        ax.set_title(f"unit {u}: GLM tuning vs speed\npseudo-$R^2$={G['scores']['dir x speed'][u]:.3f}",
                     fontsize=10, pad=22)
        ax.tick_params(labelsize=7)
        if j == 2:
            ax.legend(fontsize=7.5, loc="center left", bbox_to_anchor=(1.15, 0.5),
                      title="hand speed", title_fontsize=8)

    fig.suptitle("Poisson GLM: direction and speed as predictors of motor cortical spiking",
                 fontsize=13)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


CACHE = "cache/glm_results.npz"


def run_cached(S, path=CACHE):
    import os
    if os.path.exists(path):
        d = np.load(path, allow_pickle=False)
        names = ["constant", "speed", "direction", "dir + speed", "dir x speed"]
        G = dict(scores={n: d[f"score::{n}"] for n in names},
                 grid_ang=d["grid_ang"], grid_sp=d["grid_sp"], pred=d["pred"])
        print("loaded cached GLM results from", path)
        return G
    G = run(S)
    np.savez(path, grid_ang=G["grid_ang"], grid_sp=G["grid_sp"], pred=G["pred"],
             **{f"score::{k}": v for k, v in G["scores"].items()})
    return G


if __name__ == "__main__":
    S = build()
    G = run_cached(S)
    for k, v in G["scores"].items():
        print(f"  {k:12s} median pseudo-R2 = {np.median(v):.4f}")
    figure_glm(G)
