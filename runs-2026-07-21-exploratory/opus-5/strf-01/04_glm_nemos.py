"""Poisson GLM version of the STRF (NeMoS), with and without a spike-history filter.

Reverse correlation estimates a filter that predicts a *rate*. A Poisson GLM instead
models the spike train itself: the linear filter is expanded in a raised-cosine basis
(far fewer free parameters than one weight per frequency x lag), and refractoriness
enters explicitly as a spike-history filter.

Time within the frozen noise token is split into contiguous blocks: 60% train,
20% validation (to pick the regularization strength) and 20% test. The same split is
applied to every repeat, so test blocks are stimulus the model has never seen.
"""
import time

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nemos as nmo
import numpy as np

import anf_lib as al

ASSET = "5817b204-bf2b-40fb-b852-21072af4b628"
TOKEN = "NOISE_NOISE_4"
N_BASIS_STIM, WIN_STIM = 10, 26       # 25 ms of stimulus history
N_BASIS_HIST, WIN_HIST = 6, 21        # 20 ms of spike history
STRENGTHS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]

fb = al.read_fibre(ASSET)
cfs = al.band_cfs()
coch = al.cochleagram(fb["stim"]["NOISE_1"])
d = al.build_pynapple(fb, TOKEN, coch)
counts = d["counts"]
n_reps, n_bins = counts.shape
print("repeats", n_reps, "bins", n_bins, "mean rate", counts.mean() / al.BIN, "Hz")

stim_basis = nmo.basis.RaisedCosineLinearConv(n_basis_funcs=N_BASIS_STIM,
                                              window_size=WIN_STIM, label="stimulus")
stim_basis.set_input_shape(coch)
F_stim = np.asarray(stim_basis.compute_features(coch))
hist_basis = nmo.basis.RaisedCosineLogConv(n_basis_funcs=N_BASIS_HIST,
                                           window_size=WIN_HIST, label="history")

# ongoing noise only, and far enough into the token that both convolution windows
# are filled
valid = al.analysis_mask(coch, lag_min=0.0, lag_max=WIN_STIM * al.BIN)
valid &= np.arange(n_bins) >= max(WIN_STIM, WIN_HIST)
period = al.token_period(coch)
blocks = al.blocked_folds(n_bins, n_folds=5, n_blocks=25, period=period)[valid]
split = np.select([blocks < 3, blocks == 3], ["train", "val"], "test")

X_stim, X_hist, y = [], [], []
for r in range(n_reps):
    X_stim.append(F_stim[valid])
    X_hist.append(np.asarray(
        hist_basis.compute_features(counts[r].astype(float)[:, None]))[valid])
    y.append(counts[r][valid])
X_stim = np.concatenate(X_stim)
X_hist = np.concatenate(X_hist)
y = np.concatenate(y).astype(np.float32)
part = np.tile(split, n_reps)
print("design", X_stim.shape, X_hist.shape, y.shape,
      {k: int((part == k).sum()) for k in ["train", "val", "test"]})


def fit_select(X, label):
    """Standardize, sweep the ridge strength on the validation blocks, score on test."""
    mu, sd = X[part == "train"].mean(0), X[part == "train"].std(0)
    sd[sd == 0] = 1.0
    Z = ((X - mu) / sd).astype(np.float32)
    best = None
    for s in STRENGTHS:
        t0 = time.time()
        glm = nmo.glm.GLM(solver_name="LBFGS", regularizer="Ridge",
                          regularizer_strength=s)
        glm.fit(Z[part == "train"], y[part == "train"])
        v = float(glm.score(Z[part == "val"], y[part == "val"],
                            score_type="pseudo-r2-McFadden"))
        print(f"  {label:20s} strength {s:.0e}  val pseudo-R2 {v: .4f}  [{time.time()-t0:.0f}s]")
        if best is None or v > best[0]:
            best = (v, s, glm)
    v, s, glm = best
    test = float(glm.score(Z[part == "test"], y[part == "test"],
                           score_type="pseudo-r2-McFadden"))
    print(f"  {label:20s} -> strength {s:.0e}, TEST pseudo-R2 {test:.4f}")
    coef = np.asarray(glm.coef_) / sd            # coefficients on the raw features
    return dict(glm=glm, strength=s, test=test, coef=coef, mu=mu, sd=sd, Z=Z)


print("stimulus only:")
m_s = fit_select(X_stim, "stimulus only")
print("stimulus + history:")
m_sh = fit_select(np.hstack([X_stim, X_hist]), "stimulus+history")

# ---- reconstruct filters from the basis coefficients
_, kern = stim_basis.evaluate_on_grid(WIN_STIM)
lags_glm = np.arange(WIN_STIM) * al.BIN
strf_glm = (kern @ stim_basis.split_by_feature(m_s["coef"], axis=0)["stimulus"].T).T
strf_glm_h = (kern @ stim_basis.split_by_feature(
    m_sh["coef"][: X_stim.shape[1]], axis=0)["stimulus"].T).T
_, khist = hist_basis.evaluate_on_grid(WIN_HIST)
hist_filter = khist @ m_sh["coef"][X_stim.shape[1]:]

# ---- ridge STRF fitted on the same training blocks, scored on the same test blocks
X_ridge, lags_ridge = al.lag_design(coch)
psth = counts.mean(0) / al.BIN
Xv, psthv, sp = X_ridge[valid], psth[valid], split
w, b = al.ridge_fit(Xv[sp == "train"], psthv[sp == "train"], alpha=3.0)
strf_ridge = w.reshape(len(cfs), len(lags_ridge))
with np.errstate(all="ignore"):
    pred_ridge = Xv @ w + b
r_ridge = np.corrcoef(psthv[sp == "test"], pred_ridge[sp == "test"])[0, 1]

rate_glm = np.asarray(m_s["glm"].predict(m_s["Z"][:len(sp)])) / al.BIN
r_glm = np.corrcoef(psthv[sp == "test"], rate_glm[sp == "test"])[0, 1]
ceiling = al.split_half_ceiling(counts[:, valid][:, sp == "test"])
print(f"held-out PSTH correlation on test blocks: ridge {r_ridge:.3f}, GLM {r_glm:.3f}, "
      f"reliability ceiling {ceiling:.3f}")

np.savez("glm_results.npz", strf_glm=strf_glm, strf_glm_h=strf_glm_h,
         strf_ridge=strf_ridge, lags_glm=lags_glm, lags_ridge=lags_ridge, cfs=cfs,
         hist_filter=hist_filter, r2_s=m_s["test"], r2_sh=m_sh["test"],
         r_ridge=r_ridge, r_glm=r_glm, ceiling=ceiling)

# ------------------------------------------------------------------- figure 6
fig = plt.figure(figsize=(14, 6.8))
gs = fig.add_gridspec(2, 3, hspace=0.5, wspace=0.42)
panels = [(strf_ridge, lags_ridge,
           f"ridge reverse correlation\n{len(cfs)*len(lags_ridge)} free weights "
           "(train blocks)"),
          (strf_glm, lags_glm,
           f"Poisson GLM, raised-cosine basis\n{len(cfs)*N_BASIS_STIM} free weights"),
          (strf_glm_h, lags_glm, "GLM + spike history\n(stimulus filter)")]
for k, (S, lg, ttl) in enumerate(panels):
    ax = fig.add_subplot(gs[0, k])
    v = np.abs(S).max()
    im = ax.pcolormesh(lg * 1e3, cfs, S, cmap="RdBu_r", vmin=-v, vmax=v, shading="nearest")
    ax.set_yscale("log")
    ax.axhline(fb["info"]["bf_hz"], color="k", ls="--", lw=0.8)
    ax.set_xlabel("lag (ms)")
    if k == 0:
        ax.set_ylabel("frequency (Hz)")
    ax.set_title(ttl, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)

ax = fig.add_subplot(gs[1, 0])
bi = int(np.argmax(strf_ridge.max(1)))
bj = int(np.argmax(strf_glm.max(1)))
ax.plot(lags_ridge * 1e3, strf_ridge[bi] / np.abs(strf_ridge[bi]).max(), color="0.45",
        label=f"ridge ({cfs[bi]:.0f} Hz)")
ax.plot(lags_glm * 1e3, strf_glm[bj] / np.abs(strf_glm[bj]).max(), color="C0", lw=1.8,
        label=f"GLM ({cfs[bj]:.0f} Hz)")
ax.axhline(0, color="k", lw=0.6)
ax.axvline(0, color="k", lw=0.6)
ax.set_xlabel("lag (ms)")
ax.set_ylabel("normalized weight")
ax.set_title("best-frequency temporal profile", fontsize=9)
ax.legend(fontsize=8, frameon=False)

ax = fig.add_subplot(gs[1, 1])
ax.plot(np.arange(WIN_HIST) * al.BIN * 1e3, np.exp(hist_filter), color="C3", lw=1.6)
ax.axhline(1, color="k", lw=0.6)
ax.set_xlabel("time since spike (ms)")
ax.set_ylabel("gain (x baseline)")
ax.set_title("spike-history filter:\nrefractoriness then recovery", fontsize=9)

ax = fig.add_subplot(gs[1, 2])
ax.bar([0, 1], [m_s["test"], m_sh["test"]], color=["C0", "C3"], width=0.6)
ax.set_xticks([0, 1])
ax.set_xticklabels(["stimulus\nonly", "stimulus\n+ history"], fontsize=9)
ax.set_ylabel("test pseudo-$R^2$")
ax.set_title("single-trial spike prediction", fontsize=9)
for x, v in zip([0, 1], [m_s["test"], m_sh["test"]]):
    ax.text(x, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
fig.savefig("fig06_glm_strf.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote fig06_glm_strf.png")
