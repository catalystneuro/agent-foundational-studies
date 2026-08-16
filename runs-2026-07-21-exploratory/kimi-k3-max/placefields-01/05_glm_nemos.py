"""Model-based tuning curves with a Poisson GLM (nemos) + B-spline position basis.

Fits count ~ splines(position) for the example place cells and compares the
GLM-predicted tuning curve with the empirical occupancy-normalized rate map.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pynapple as nap
import jax
jax.config.update("jax_enable_x64", True)
import nemos as nmo

CACHE = "achilles_maze_cache.npz"
RES = "achilles_placefields.npz"

N_BASIS = 12
BIN_SIZE = 0.0256  # ~ position sampling period (39.06 Hz)


def load_all():
    c = np.load(CACHE, allow_pickle=False)
    r = np.load(RES, allow_pickle=False)
    D = {k: c[k] for k in c.files}
    D.update({k: r[k] for k in r.files})
    D["spikes"] = {u: c[f"spk_{u}"] for u in c["unit_ids"]}
    return D


def pick_example_cells(D, n=8):
    """Same selection as fig3: one high-SI place cell per eighth of the track."""
    place_idx = np.where(D["is_place"])[0]
    centers = D["centers"]
    peaks = np.array([centers[np.nanargmax(D["ratemaps_sm"][i])] for i in place_idx])
    picks = []
    for b in range(n):
        lo, hi = b * 1.6 / n, (b + 1) * 1.6 / n
        cand = place_idx[(peaks >= lo) & (peaks < hi)]
        if len(cand) == 0:
            continue
        picks.append(cand[np.argmax(D["si"][cand])])
    return picks


def main():
    D = load_all()
    t, lin = D["t"], D["lin"]
    run_ep = nap.IntervalSet(start=D["run_starts"], end=D["run_ends"])
    pos_tsd = nap.Tsd(t=t, d=lin).restrict(run_ep)

    basis = nmo.basis.BSplineEval(n_basis_funcs=N_BASIS)
    grid = np.linspace(0, 1.6, 200)
    X_grid = basis.compute_features(grid)

    picks = pick_example_cells(D)
    unit_ids = D["unit_ids"]
    centers = D["centers"]

    fig, axes = plt.subplots(2, 4, figsize=(11.5, 5))
    axes = axes.ravel()
    fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.11,
                        hspace=0.5, wspace=0.35)

    for k, ui in enumerate(picks):
        u = unit_ids[ui]
        spk = D["spikes"][u]
        count = nap.Ts(spk).count(BIN_SIZE, ep=run_ep)
        # align position to count timebase, drop NaN-position bins
        pos_matched = pos_tsd.interpolate(count)
        m = ~np.isnan(pos_matched.values)
        X = basis.compute_features(pos_matched[m])
        y = count[m]

        # small ridge penalty keeps coefficients finite for cells with
        # near-zero rates over part of the track (Poisson separability)
        model = nmo.glm.GLM(solver_name="LBFGS", regularizer="Ridge",
                            regularizer_strength=1e-4,
                            solver_kwargs={"maxiter": 1000, "tol": 1e-6})
        model.fit(X, y)
        rate_grid = np.asarray(model.predict(X_grid)) / BIN_SIZE

        ax = axes[k]
        emp = D["ratemaps_sm"][ui]
        ax.fill_between(centers, emp, color="0.7", alpha=0.5, lw=0,
                        label="empirical (smoothed)")
        ax.plot(grid, rate_grid, color="tab:purple", lw=1.6, label="GLM (Poisson, B-splines)")
        ax.set_xlim(0, 1.6)
        ax.set_xticks([0, 0.8, 1.6])
        ax.set_title(f"unit {u}", fontsize=9)
        if k == 0:
            ax.set_ylabel("firing rate (Hz)")
            ax.legend(frameon=False, fontsize=7, loc="upper right")
        ax.set_xlabel("position (m)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # hide unused axes if fewer than 8 picks
    for k in range(len(picks), len(axes)):
        axes[k].axis("off")

    fig.suptitle("Poisson GLM tuning curves (B-spline basis on position) vs empirical rate maps",
                 fontsize=11, y=0.97)
    fig.savefig("fig7_glm_tuning.png")
    plt.close(fig)
    print("saved fig7_glm_tuning.png")


if __name__ == "__main__":
    main()
