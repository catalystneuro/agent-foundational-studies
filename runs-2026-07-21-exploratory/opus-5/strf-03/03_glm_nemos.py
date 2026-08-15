"""Model-based STRFs: Poisson GLMs (NeMoS) with a raised-cosine lag basis.

The reverse-correlation STRF is unbiased only for a white stimulus.  Here we fit the
same receptive field as the linear filter of a Poisson LNP model, which handles the
non-white (fixed 0.805 s inter-onset interval) tone sequence properly, gives a
regularised low-dimensional filter, and can be scored on held-out data.
"""

import os
import time

import jax
import matplotlib.pyplot as plt
import nemos as nmo

jax.config.update("jax_enable_x64", True)   # LBFGS converges reliably in float64
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import strf_lib as sl

PROTO = "sub-LA11/sub-LA11_ses-1_behavior.nwb"
BIN = 0.005
W_STIM = 60          # 300 ms of stimulus lags
N_BASIS = 9
W_HIST = 40          # 200 ms of spike history
N_BASIS_HIST = 5
N_UNITS = 24         # units fitted (most responsive first)
ALPHAS = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]

assets = sl.list_assets()
ses = sl.load_session(assets[PROTO], with_behavior=False)
units, trials, blocks = ses["units"], ses["trials"], ses["blocks"]
metrics = pd.read_csv(f"{sl.FIGDIR}/proto_metrics.csv")

stim = sl.stimulus_tsdframe(trials, blocks, binsize=BIN)
counts = units.count(BIN, ep=stim.time_support)

stim_basis = nmo.basis.RaisedCosineLogConv(n_basis_funcs=N_BASIS, window_size=W_STIM,
                                           label="tone")
hist_basis = nmo.basis.RaisedCosineLogConv(n_basis_funcs=N_BASIS_HIST, window_size=W_HIST,
                                           label="history")
X_stim = np.asarray(stim_basis.compute_features(stim), dtype=np.float64)
Y = np.asarray(counts, dtype=np.float64)
tvec = np.asarray(counts.t)
print("design", X_stim.shape, "counts", Y.shape)

# nemos convolution is causal and excludes the current bin, so feature column for
# basis lag index k corresponds to a delay of (k + 1) bins.
_, kern = stim_basis.evaluate_on_grid(W_STIM)          # (W_STIM, N_BASIS)
lags = (np.arange(W_STIM) + 1) * BIN

train_ep = blocks[0:3]
test_ep = blocks[3:4]
in_train = np.zeros(tvec.size, bool)
for s, e in zip(train_ep.start, train_ep.end):
    in_train |= (tvec >= s) & (tvec < e)
in_test = (tvec >= test_ep.start[0]) & (tvec < test_ep.end[0])
valid = ~np.isnan(X_stim).any(axis=1)
tr = in_train & valid
te = in_test & valid
print("train bins", tr.sum(), "test bins", te.sum())

sel = np.argsort(-metrics.peak_evoked_hz.values)
sel = [i for i in sel if metrics.pval.values[i] < 0.05][:N_UNITS]
print("fitting", len(sel), "units")


def fit_one(X_tr, y_tr, alpha):
    m = nmo.glm.GLM(observation_model="Poisson", regularizer="Ridge",
                    regularizer_strength=alpha, solver_name="LBFGS",
                    solver_kwargs={"tol": 1e-9, "maxiter": 300})
    m.fit(X_tr, y_tr)
    return m


# --------------------------------------------------------------------------- #
# Regularisation strength: held-out Poisson pseudo-R2 on a few units
# --------------------------------------------------------------------------- #
CACHE = f"{sl.FIGDIR}/glm_results.npz"
if not os.path.exists(CACHE):
    sweep = np.zeros((len(ALPHAS), 5))
    for ai, alpha in enumerate(tqdm(ALPHAS, desc="alpha sweep")):
        for ui, i in enumerate(sel[:5]):
            m = fit_one(X_stim[tr], Y[tr, i], alpha)
            sweep[ai, ui] = m.score(X_stim[te], Y[te, i],
                                    score_type="pseudo-r2-McFadden")
    best_alpha = ALPHAS[int(np.argmax(sweep.mean(axis=1)))]
    print("sweep (mean held-out pseudo-R2):", sweep.mean(axis=1), "-> alpha", best_alpha)

    # ----------------------------------------------------------------------- #
    # Fit stimulus-only and stimulus + spike-history models
    # ----------------------------------------------------------------------- #
    filters = np.zeros((len(sel), len(sl.FREQS), W_STIM))
    r2_stim = np.zeros(len(sel))
    r2_hist = np.zeros(len(sel))
    intercepts = np.zeros(len(sel))
    pred_test = np.zeros((len(sel), int(te.sum())))
    t0 = time.time()
    for ui, i in enumerate(tqdm(sel, desc="GLM fits")):
        m = fit_one(X_stim[tr], Y[tr, i], best_alpha)
        c = np.asarray(m.coef_).reshape(len(sl.FREQS), N_BASIS)
        filters[ui] = (kern @ c.T).T
        intercepts[ui] = float(np.asarray(m.intercept_)[0])
        r2_stim[ui] = m.score(X_stim[te], Y[te, i], score_type="pseudo-r2-McFadden")
        pred_test[ui] = np.asarray(m.predict(X_stim[te]))

        h = np.asarray(hist_basis.compute_features(counts[:, i]), dtype=np.float64)
        Xh = np.hstack([X_stim, h])
        vh = ~np.isnan(Xh).any(axis=1)
        mh = fit_one(Xh[tr & vh], Y[tr & vh, i], best_alpha)
        r2_hist[ui] = mh.score(Xh[te & vh], Y[te & vh, i],
                               score_type="pseudo-r2-McFadden")
    print("fits took %.0f s" % (time.time() - t0))
    np.savez(CACHE, filters=filters, r2_stim=r2_stim, r2_hist=r2_hist, lags=lags,
             sweep=sweep, best_alpha=best_alpha, sel=np.array(sel),
             intercepts=intercepts, pred_test=pred_test, t_test=tvec[te])
else:
    z = np.load(CACHE)
    filters, r2_stim, r2_hist, lags = z["filters"], z["r2_stim"], z["r2_hist"], z["lags"]
    sweep, best_alpha, sel = z["sweep"], float(z["best_alpha"]), list(z["sel"])
    intercepts, pred_test = z["intercepts"], z["pred_test"]

print("held-out pseudo-R2: stim %.4f +- %.4f, stim+history %.4f +- %.4f"
      % (r2_stim.mean(), r2_stim.std(), r2_hist.mean(), r2_hist.std()))

# --------------------------------------------------------------------------- #
# Figure: GLM STRFs next to the reverse-correlation STRFs
# --------------------------------------------------------------------------- #
z = np.load(f"{sl.FIGDIR}/proto_strf.npz")
centers, R = z["centers"], z["R"]
show = list(range(6))
fig, axes = plt.subplots(2, 6, figsize=(16, 5.4), sharex=True, sharey=True)
for col, ui in enumerate(show):
    i = sel[ui]
    M = R[i] - metrics.baseline_hz.values[i]
    v = np.abs(M).max()
    axes[0, col].pcolormesh(centers * 1000, np.arange(6) - 0.5, np.vstack([M, M[-1]]),
                            cmap="RdBu_r", vmin=-v, vmax=v, shading="auto")
    axes[0, col].set_title(f"unit {metrics.unit.values[i]}", fontsize=9, pad=4)
    F = filters[ui]
    v = np.abs(F).max()
    axes[1, col].pcolormesh(lags * 1000, np.arange(6) - 0.5, np.vstack([F, F[-1]]),
                            cmap="RdBu_r", vmin=-v, vmax=v, shading="auto")
    axes[1, col].set_title(f"pseudo-$R^2$ = {r2_stim[ui]:.3f}", fontsize=9, pad=4)
    axes[1, col].set_xlabel("lag (ms)")
for row, lab in enumerate(["reverse correlation\n(evoked rate)", "Poisson GLM\n(filter weight)"]):
    axes[row, 0].set_ylabel(lab, fontsize=9)
    for ax in axes[row]:
        ax.set_yticks(range(5), [f"{f/1000:g}" for f in sl.FREQS], fontsize=7)
axes[0, 0].set_xlim(-50, 300)
fig.suptitle("Spectrotemporal receptive fields: reverse correlation (top) vs "
             "regularised Poisson GLM (bottom)", y=1.0)
fig.tight_layout()
fig.savefig(f"{sl.FIGDIR}/fig08_glm_vs_revcorr.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# --------------------------------------------------------------------------- #
# Figure: agreement, regularisation sweep, model comparison, prediction
# --------------------------------------------------------------------------- #
fig, axes = plt.subplots(1, 4, figsize=(16.5, 3.9))

# (a) The GLM filter acts on 5 ms stimulus bins in log-rate units, so to compare it
# with the measured STRF we push a single 25 ms tone through the model: the predicted
# log gain at lag tau is the filter summed over the tone boxcar, and the measured log
# gain is log(rate / spontaneous rate).
K = int(round(sl.TONE_DUR / BIN))
a, b = [], []
for ui, i in enumerate(sel):
    bl = metrics.baseline_hz.values[i]
    for j in range(len(sl.FREQS)):
        F = filters[ui][j]
        pred = np.array([F[max(0, k - K + 1): k + 1].sum() for k in range(F.size)])
        meas = np.interp(lags, centers, R[i, j]) / bl
        keep = meas > 0            # drop empty bins, where the log gain is undefined
        a.append(np.log(meas[keep]))
        b.append(pred[keep])
a = np.concatenate(a)
b = np.concatenate(b)
rr = np.corrcoef(a, b)[0, 1]
axes[0].plot(a, b, ".", ms=2, alpha=0.25, color="tab:blue")
lim = [min(a.min(), b.min()), max(a.max(), b.max())]
axes[0].plot(lim, lim, "k--", lw=1)
axes[0].set_xlabel("measured log gain, log(rate / spont.)")
axes[0].set_ylabel("GLM log gain (filter x tone boxcar)")
axes[0].set_title(f"tone response, measured vs GLM\nr = {rr:.2f}, {len(sel)} units", pad=8)

axes[1].plot(ALPHAS, sweep.mean(axis=1), "o-", color="k")
for ui in range(sweep.shape[1]):
    axes[1].plot(ALPHAS, sweep[:, ui], "-", color="0.75", lw=0.8, zorder=0)
axes[1].set_xscale("log")
axes[1].set_xlabel("ridge strength")
axes[1].set_ylabel("held-out pseudo-$R^2$")
axes[1].set_title(f"regularisation sweep (chosen {best_alpha:g})", pad=8)

axes[2].plot([0, max(r2_stim.max(), r2_hist.max()) * 1.05],
             [0, max(r2_stim.max(), r2_hist.max()) * 1.05], "k--", lw=1)
axes[2].scatter(r2_stim, r2_hist, s=18, color="tab:red")
axes[2].set_xlabel("stimulus only")
axes[2].set_ylabel("stimulus + spike history")
axes[2].set_title("held-out pseudo-$R^2$", pad=8)

# (d) observed vs GLM-predicted peri-tone PSTH on the held-out block
ui = int(np.argmax(r2_stim))
i = sel[ui]
t_test = tvec[te]
edges = np.arange(-0.05, 0.3 + BIN / 2, BIN)
obs_psth = np.zeros((len(sl.FREQS), edges.size - 1))
pred_psth = np.zeros_like(obs_psth)
y_obs = Y[te, i]
for j, f in enumerate(sl.FREQS):
    ev = trials.start_time.values[(trials.stim_frequency.values == f)
                                  & (trials.start_time.values >= test_ep.start[0])
                                  & (trials.start_time.values < test_ep.end[0])]
    idx0 = np.searchsorted(t_test, ev + edges[0])
    keep = (idx0 + edges.size - 1) < t_test.size
    idx0 = idx0[keep]
    win = idx0[:, None] + np.arange(edges.size - 1)[None, :]
    obs_psth[j] = y_obs[win].mean(axis=0) / BIN
    pred_psth[j] = pred_test[ui][win].mean(axis=0) / BIN
colors = plt.get_cmap("viridis")(np.linspace(0, 1, 5))
cc = edges[:-1] + BIN / 2
for j in range(len(sl.FREQS)):
    axes[3].plot(cc * 1000, obs_psth[j], color=colors[j], lw=1.2)
    axes[3].plot(cc * 1000, pred_psth[j], color=colors[j], lw=1.2, ls="--")
axes[3].set_xlabel("time from onset (ms)")
axes[3].set_ylabel("rate (Hz)")
r_psth = np.corrcoef(obs_psth.ravel(), pred_psth.ravel())[0, 1]
axes[3].set_title(f"unit {metrics.unit.values[i]}, held-out block\n"
                  f"solid observed, dashed GLM (r = {r_psth:.2f})", pad=8)
fig.tight_layout()
fig.savefig(f"{sl.FIGDIR}/fig09_glm_quality.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("done; filter/STRF agreement r =", round(rr, 3), "PSTH r =", round(r_psth, 3))
