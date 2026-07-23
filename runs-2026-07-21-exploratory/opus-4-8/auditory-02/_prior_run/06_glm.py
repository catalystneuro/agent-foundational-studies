"""Poisson GLM (NeMoS) with frequency-specific tone kernels + spike history.

Confirms that the frequency dependence of the response is not an artefact of
rate differences or spike-train autocorrelation: each tone frequency gets its
own temporal kernel, fitted jointly with the unit's own spike history.
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pynapple as nap
import nemos as nmo
from scipy import stats
from tqdm import tqdm

from loader import list_assets, load_session
from analysis import bh_fdr

plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 130})

BIN = 0.005          # s
WIN_BINS = 40        # 200 ms stimulus kernel
HIST_BINS = 20       # 100 ms spike-history filter
N_FIT = 40           # units to fit

R = pickle.load(open("results_all_sessions.pkl", "rb"))
s0 = R[0]
FREQS = s0["freqs"]
FKHZ = [f"{f/1000:g}" for f in FREQS]

assets = list_assets()
nwb, raw = load_session(assets[0])
units = nwb["units"]
trials = raw.trials.to_dataframe()
onsets = trials.start_time.values
freq = trials.stim_frequency.values

# tone-block epoch: from first to last trial of each contiguous block
ep = nap.IntervalSet(start=trials.start_time.values, end=trials.stop_time.values)
ep = ep.merge_close_intervals(1.0)
print("tone-block epoch:", ep)

# binary event trains, one per tone frequency, on the GLM time base
counts_template = units[list(units.keys())[0]].count(BIN, ep=ep)
tt = counts_template.t
edges = np.append(tt - BIN / 2, tt[-1] + BIN / 2)
stim = np.zeros((len(tt), len(FREQS)), dtype=float)
for i, f in enumerate(FREQS):
    idx = np.searchsorted(edges, onsets[freq == f], side="right") - 1
    idx = idx[(idx >= 0) & (idx < len(tt))]
    np.add.at(stim[:, i], idx, 1.0)
print("tones placed per frequency:", stim.sum(0))
stim = nap.TsdFrame(t=tt, d=stim, time_support=counts_template.time_support)

stim_basis = nmo.basis.RaisedCosineLogConv(n_basis_funcs=6, window_size=WIN_BINS,
                                           label="tone")
hist_basis = nmo.basis.RaisedCosineLogConv(n_basis_funcs=5, window_size=HIST_BINS,
                                           label="history")
_, stim_kernels = stim_basis.evaluate_on_grid(WIN_BINS)

# choose units: strongest tuned units of this session, spanning best frequencies
q = bh_fdr(s0["p_tune"])
cand = np.flatnonzero((q < 0.01) & (bh_fdr(s0["p_resp"]) < 0.01))
cand = cand[np.argsort(-s0["depth"][cand] * np.clip(s0["reliability"][cand], 0, 1))]
fit_units = cand[:N_FIT]

glm_tc = np.zeros((len(fit_units), len(FREQS)))
kernels = np.zeros((len(fit_units), len(FREQS), WIN_BINS))
scores = np.zeros(len(fit_units))
for n, u in enumerate(tqdm(fit_units, desc="GLM fits")):
    y = units[s0["unit_keys"][u]].count(BIN, ep=ep)
    X_stim = stim_basis.compute_features(stim)          # 5 freq x 6 basis
    X_hist = hist_basis.compute_features(y)
    X = np.column_stack([np.asarray(X_stim), np.asarray(X_hist)])
    yv = np.asarray(y).squeeze()
    ok = ~np.isnan(X).any(1)
    model = nmo.glm.GLM(solver_name="LBFGS",
                        regularizer="Ridge", regularizer_strength=1e-4)
    model.fit(X[ok], yv[ok])
    scores[n] = model.score(X[ok], yv[ok], score_type="pseudo-r2-McFadden")
    w = np.asarray(model.coef_)[:len(FREQS) * 6].reshape(len(FREQS), 6)
    k = w @ stim_kernels.T                              # (n_freq, WIN_BINS)
    kernels[n] = k
    glm_tc[n] = k.max(1)                                # peak gain per frequency

glm_bf = np.argmax(glm_tc, 1)
emp_bf = s0["bf_idx"][fit_units]
agree = (glm_bf == emp_bf).mean()
rho = stats.spearmanr(glm_tc.ravel(), s0["delta"][fit_units].ravel()).statistic
print(f"GLM vs empirical BF agreement: {agree:.1%}; "
      f"tuning-curve Spearman rho = {rho:.2f}; "
      f"median pseudo-R2 = {np.median(scores):.3f}")

# ------------------------------------------------------------------ figure
tk = np.arange(WIN_BINS) * BIN
show = []
seen = set()
for i in np.argsort(-glm_tc.max(1)):
    if emp_bf[i] not in seen:
        show.append(i)
        seen.add(emp_bf[i])
    if len(show) == 4:
        break

fig = plt.figure(figsize=(12, 5.4))
gs = fig.add_gridspec(2, 4, height_ratios=[1, 1], hspace=0.55, wspace=0.35)
cmap = plt.cm.viridis(np.linspace(0, 0.92, len(FREQS)))
for j, i in enumerate(show):
    ax = fig.add_subplot(gs[0, j])
    for f in range(len(FREQS)):
        ax.plot(tk, kernels[i, f], color=cmap[f], label=f"{FKHZ[f]} kHz")
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xlabel("time from tone onset (s)")
    if j == 0:
        ax.set_ylabel("GLM tone kernel\n(log gain)")
        ax.legend(fontsize=6, frameon=False)
    ax.set_title(f"unit {s0['unit_keys'][fit_units[i]]}", fontsize=9)

ax = fig.add_subplot(gs[1, 0])
im = ax.imshow(glm_tc[np.argsort(glm_bf)] /
               np.abs(glm_tc[np.argsort(glm_bf)]).max(1, keepdims=True),
               aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(FREQS)))
ax.set_xticklabels(FKHZ)
ax.set_xlabel("tone frequency (kHz)")
ax.set_ylabel("unit (sorted by GLM BF)")
ax.set_title("GLM peak gain\n(normalised)", loc="left")
plt.colorbar(im, ax=ax)

ax = fig.add_subplot(gs[1, 1])
jit = np.random.default_rng(0).normal(0, 0.07, len(fit_units))
ax.scatter(emp_bf + jit, glm_bf + jit, s=14, color="crimson", alpha=0.8)
ax.plot([-0.5, 4.5], [-0.5, 4.5], "k--", lw=1)
ax.set_xticks(range(len(FREQS)))
ax.set_xticklabels(FKHZ)
ax.set_yticks(range(len(FREQS)))
ax.set_yticklabels(FKHZ)
ax.set_xlabel("empirical BF (kHz)")
ax.set_ylabel("GLM BF (kHz)")
ax.set_title(f"BF agreement = {agree:.0%}", loc="left")

ax = fig.add_subplot(gs[1, 2])
ax.scatter(s0["delta"][fit_units].ravel(), glm_tc.ravel(), s=8,
           color="steelblue", alpha=0.6)
ax.set_xlabel("empirical evoked rate\n(baseline-subtracted, Hz)")
ax.set_ylabel("GLM peak tone gain")
ax.set_title(f"Spearman $\\rho$ = {rho:.2f}", loc="left")

ax = fig.add_subplot(gs[1, 3])
ax.hist(scores, bins=15, color="0.5")
ax.set_xlabel("pseudo-$R^2$ (McFadden)")
ax.set_ylabel("units")
ax.set_title("GLM goodness of fit", loc="left")

fig.suptitle("Poisson GLM with frequency-specific tone kernels and spike history "
             f"({raw.subject.subject_id} session {raw.session_id}, n={N_FIT} units)",
             y=0.99)
fig.savefig("fig07_glm.png", bbox_inches="tight")
np.savez("glm_results.npz", glm_tc=glm_tc, kernels=kernels, scores=scores,
         fit_units=fit_units, glm_bf=glm_bf, emp_bf=emp_bf, tk=tk)
print("fig07_glm.png written")
