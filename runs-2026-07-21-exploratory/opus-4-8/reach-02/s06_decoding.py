"""Decoding: can reach direction and speed be read back out of the population?

Two decoders are used. The first is the classical population vector, which sums each unit's
preferred direction weighted by its normalized firing rate on that trial and needs nothing but
the direction tuning measured in s03. The second is a ridge regression from a short window of
population activity onto instantaneous hand velocity, cross-validated over held-out blocks.
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from tqdm import tqdm

from s02_behavior import build
from s03_direction_tuning import run as run_direction, MOVE_EP, trial_rates
from s04_velocity_tuning import BIN, binned

DECODE_LAGS = np.arange(0, 5) * BIN   # 0 .. 80 ms of population history
N_FOLDS = 5


def population_vector(R):
    """Per-trial decoded direction from PD-weighted normalized rates."""
    fit = R["fit_move"]
    sig = fit["p"] < 0.01
    rates = R["rates_move"][:, sig]
    pd = fit["pref_dir"][sig]
    # z-score each unit so that high-rate units do not dominate
    z = (rates - rates.mean(0)) / (rates.std(0) + 1e-9)
    vec = z @ np.column_stack([np.cos(pd), np.sin(pd)])
    decoded = np.arctan2(vec[:, 1], vec[:, 0])
    err = np.degrees((decoded - R["angles"] + np.pi) % (2 * np.pi) - np.pi)
    return decoded, err, int(sig.sum())


def lagged_design(rate, lags=DECODE_LAGS):
    """Stack population activity at several lags; drop rows that would span a trial gap."""
    t = np.asarray(rate.t)
    tk = np.round(t * 1e6).astype(np.int64)
    step = int(round(BIN * 1e6))
    C = np.asarray(rate, dtype=float)
    ks = [int(round(l / BIN)) for l in lags]
    kmax = max(ks)
    base = np.arange(0, len(t) - kmax)
    ok = np.ones(len(base), dtype=bool)
    cols = []
    for k in ks:
        idx = base + k
        ok &= (tk[idx] - tk[base]) == k * step
        cols.append(C[idx])
    X = np.concatenate([c[ok] for c in cols], axis=1)
    return X, base[ok] + kmax  # predict the velocity at the last lag in the window


def ridge_decode(X, Yv, n_folds=N_FOLDS, alpha=10.0):
    folds = np.array_split(np.arange(len(X)), n_folds)
    pred = np.empty_like(Yv)
    for f in tqdm(folds, desc="ridge folds", leave=False):
        tr = np.setdiff1d(np.arange(len(X)), f)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        m = Ridge(alpha=alpha).fit((X[tr] - mu) / sd, Yv[tr])
        pred[f] = m.predict((X[f] - mu) / sd)
    ss_res = ((Yv - pred) ** 2).sum(0)
    ss_tot = ((Yv - Yv.mean(0)) ** 2).sum(0)
    return pred, 1 - ss_res / ss_tot


def run(S, R=None):
    if R is None:
        R = run_direction(S)
    decoded, pv_err, n_used = population_vector(R)
    print(f"population vector: {n_used} tuned units, median |error| = "
          f"{np.median(np.abs(pv_err)):.1f} deg")

    counts, rate, vel = binned(S, S["obs"])
    X, idx = lagged_design(rate)
    V = np.asarray(vel, dtype=float)[idx]
    pred, r2 = ridge_decode(X, V)
    print(f"ridge decoding of velocity over all trial bins, held-out R^2: "
          f"vx={r2[0]:.3f}, vy={r2[1]:.3f}")

    # Direction and speed accuracy are only meaningful while the hand is actually moving.
    moving = np.linalg.norm(V, axis=1) > 5.0
    ssr = ((V[moving] - pred[moving]) ** 2).sum(0)
    sst = ((V[moving] - V[moving].mean(0)) ** 2).sum(0)
    r2_moving = 1 - ssr / sst
    print(f"restricted to moving bins, the same decoder gives R^2: "
          f"vx={r2_moving[0]:.3f}, vy={r2_moving[1]:.3f}")
    ang_true = np.arctan2(V[moving, 1], V[moving, 0])
    ang_dec = np.arctan2(pred[moving, 1], pred[moving, 0])
    ang_err = np.degrees((ang_dec - ang_true + np.pi) % (2 * np.pi) - np.pi)
    sp_true = np.linalg.norm(V[moving], axis=1)
    sp_dec = np.linalg.norm(pred[moving], axis=1)
    sp_r = np.corrcoef(sp_true, sp_dec)[0, 1]
    print(f"during movement: median |direction error| = {np.median(np.abs(ang_err)):.1f} deg, "
          f"speed correlation r = {sp_r:.2f}")

    # decoding accuracy as a function of population size
    rng = np.random.default_rng(2)
    n_units = X.shape[1] // len(DECODE_LAGS)
    sizes = [2, 5, 10, 20, 40, 80, 120, n_units]
    curve = np.full((len(sizes), 5), np.nan)
    for i, n in enumerate(tqdm(sizes, desc="population size sweep")):
        for rep in range(5 if n < n_units else 1):
            sub = rng.choice(n_units, size=n, replace=False)
            cols = np.concatenate([sub + j * n_units for j in range(len(DECODE_LAGS))])
            _, r2n = ridge_decode(X[:, cols], V)
            curve[i, rep] = r2n.mean()

    return dict(R=R, pv_err=pv_err, pv_decoded=decoded, n_pv_units=n_used,
                pred=pred, r2=r2, ang_err=ang_err, sp_true=sp_true, sp_dec=sp_dec,
                sp_r=sp_r, V=V, moving=moving, r2_moving=r2_moving, sizes=sizes, curve=curve,
                rate_t=np.asarray(rate.t)[idx])


def figure_decoding(S, D, path="figures/fig07_decoding.png"):
    fig = plt.figure(figsize=(15, 8.5))
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.42)

    # population vector, per trial
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(np.degrees(D["R"]["angles"]), np.degrees(D["pv_decoded"]),
               s=5, alpha=0.25, color="tab:blue")
    ax.plot([-180, 180], [-180, 180], "k--", lw=0.8)
    ax.set_xlabel("actual reach direction ($\\degree$)")
    ax.set_ylabel("population vector direction ($\\degree$)")
    ax.set_title(f"Population vector decoding\n({D['n_pv_units']} tuned units, "
                 f"{len(D['pv_err'])} trials)", fontsize=11)

    ax = fig.add_subplot(gs[0, 1])
    ax.hist(D["pv_err"], bins=np.arange(-180, 185, 7.5), color="tab:blue", alpha=0.85)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("population vector error ($\\degree$)")
    ax.set_ylabel("trials")
    ax.set_title(f"Median |error| = {np.median(np.abs(D['pv_err'])):.0f}$\\degree$ "
                 f"(chance 90$\\degree$)", fontsize=11)

    # ridge decoding traces over a contiguous stretch of real session time
    ax = fig.add_subplot(gs[0, 2])
    t = D["rate_t"]
    t0 = S["trials"]["start"].values[1200]
    w = (t >= t0) & (t < t0 + 14)
    ax.plot(t[w] - t0, D["V"][w, 0], color="k", lw=1.1, label="true $v_x$")
    ax.plot(t[w] - t0, D["pred"][w, 0], color="crimson", lw=1.1, label="decoded $v_x$")
    ax.plot(t[w] - t0, D["V"][w, 1] - 150, color="0.45", lw=1.1, label="true $v_y$")
    ax.plot(t[w] - t0, D["pred"][w, 1] - 150, color="tab:blue", lw=1.1, label="decoded $v_y$")
    ax.set_xlabel("session time (s)")
    ax.set_ylabel("velocity (cm/s, $v_y$ offset by -150)")
    ax.set_title(f"Held-out velocity decoding: $R^2$ = {D['r2'][0]:.2f} / {D['r2'][1]:.2f} "
                 f"($v_x$/$v_y$)\nover moving bins only: "
                 f"{D['r2_moving'][0]:.2f} / {D['r2_moving'][1]:.2f}", fontsize=10.5)
    ax.legend(fontsize=7.5, ncol=2, loc="upper right")

    ax = fig.add_subplot(gs[1, 0])
    ax.hist(D["ang_err"], bins=np.arange(-180, 185, 7.5), color="crimson", alpha=0.85)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("instantaneous direction error ($\\degree$)")
    ax.set_ylabel("20 ms bins")
    ax.set_title(f"Direction from the ridge decoder\nmedian |error| = "
                 f"{np.median(np.abs(D['ang_err'])):.0f}$\\degree$", fontsize=11)

    ax = fig.add_subplot(gs[1, 1])
    h = ax.hist2d(D["sp_true"], D["sp_dec"], bins=[50, 50],
                  range=[[0, 110], [0, 110]], cmap="magma")
    ax.plot([0, 110], [0, 110], "w--", lw=0.8)
    ax.set_xlabel("true speed (cm/s)")
    ax.set_ylabel("decoded speed (cm/s)")
    ax.set_title(f"Speed is recovered too (r = {D['sp_r']:.2f})", fontsize=11)
    fig.colorbar(h[3], ax=ax, label="bins", fraction=0.046)

    ax = fig.add_subplot(gs[1, 2])
    m = np.nanmean(D["curve"], axis=1)
    s = np.nanstd(D["curve"], axis=1)
    ax.errorbar(D["sizes"], m, yerr=s, fmt="o-", color="tab:green", capsize=3)
    ax.set_xscale("log")
    ax.set_xlabel("number of units in the decoder")
    ax.set_ylabel("mean held-out $R^2$ ($v_x$, $v_y$)")
    ax.set_title("Decoding improves with population size", fontsize=11)

    fig.suptitle("Reading reach direction and speed out of the population", fontsize=13)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    S = build()
    D = run(S)
    figure_decoding(S, D)
