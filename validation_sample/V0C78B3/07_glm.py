"""Poisson GLM (NeMoS): frequency-specific temporal kernels for tone responses.

The design matrix contains one event train per tone frequency, each convolved with
a log-spaced raised-cosine basis. The fitted kernels give a model-based estimate of
frequency tuning (the gain of each frequency's kernel) that is independent of the
window choices used for the descriptive analysis.
"""
import pickle

import matplotlib.pyplot as plt
import nemos as nmo
import numpy as np
import pynapple as nap

import analysis
import common

SESSION = "sub-LA11/sub-LA11_ses-1_behavior.nwb"
BIN = 0.01              # s
WINDOW = 0.3            # s of tone-response kernel
N_BASIS = 8
N_TRIALS = 1500         # subset of trials keeps the fit tractable
N_UNITS = 24            # most strongly driven units

s = common.load_session(SESSION)
onsets = s["trials"]["start_time"].values[:N_TRIALS]
freq = s["frequency"][:N_TRIALS]
freqs = np.unique(freq)
ep = nap.IntervalSet(onsets[0] - 1.0, onsets[-1] + 1.0)

counts = s["units"].count(BIN, ep)
stim = np.zeros((counts.shape[0], len(freqs)))
idx = np.searchsorted(counts.t - BIN / 2, onsets, side="right") - 1
for j, f in enumerate(freqs):
    stim[idx[freq == f], j] = 1.0
assert stim.sum() == len(onsets)
stim = nap.TsdFrame(t=counts.t, d=stim, time_support=ep)

basis = nmo.basis.RaisedCosineLogConv(n_basis_funcs=N_BASIS,
                                      window_size=int(WINDOW / BIN), label="tone")
basis.set_input_shape(stim)
X = basis.compute_features(stim)

# pick the most strongly driven units so the kernels are estimable
tuning_emp, _, _, st, _ = analysis.session_tuning(s["units"], onsets, freq)
units = st.sort_values("peak_evoked", ascending=False).index[:N_UNITS].tolist()
y = np.asarray(counts[:, units].d, dtype=float)

model = nmo.glm.PopulationGLM(
    solver_name="LBFGS", regularizer="Ridge", regularizer_strength=1e-6,
    solver_kwargs=dict(maxiter=500)).fit(X, y)
print("fitted", model.coef_.shape)

# reconstruct one kernel per (frequency, unit)
_, kern = basis.evaluate_on_grid(int(WINDOW / BIN))          # (win, n_basis)
coef = model.coef_.reshape(len(freqs), N_BASIS, len(units))  # (freq, basis, unit)
filters = np.einsum("wb,fbu->fuw", kern, coef)               # (freq, unit, win)
b0 = np.asarray(model.intercept_)                            # (unit,)
# model-based tuning: extra spikes per tone predicted by each frequency's kernel
gain = (np.exp(b0[None, :, None] + filters) - np.exp(b0[None, :, None])).sum(-1)

pred = np.asarray(model.predict(X)) / BIN                    # spikes/s
pred = nap.TsdFrame(t=counts.t, d=pred, time_support=ep)

with open("glm_results.pkl", "wb") as fh:
    pickle.dump(dict(filters=filters, gain=gain, units=units, freqs=freqs,
                     bin=BIN, window=WINDOW, emp_tuning=tuning_emp.loc[units],
                     loglike=float(model.score(X, y))), fh)

# ------------------------------------------------------------------ figure ----
colors = dict(zip(freqs, plt.get_cmap("viridis")(np.linspace(0, 0.92, len(freqs)))))
show = units[:4]
fig, axes = plt.subplots(3, len(show), figsize=(3.4 * len(show), 8.4),
                         gridspec_kw=dict(hspace=0.45, wspace=0.3))
tt = np.arange(int(WINDOW / BIN)) * BIN
for c, u in enumerate(show):
    k = units.index(u)
    for j, f in enumerate(freqs):
        axes[0, c].plot(tt, filters[j, k], color=colors[f], lw=1.5,
                        label=f"{int(f/1000)} kHz")
    axes[0, c].axhline(0, color="0.6", lw=0.8, ls="--")
    axes[0, c].set_title(f"unit {u}", fontsize=10)
    axes[0, c].set_xlabel("time from tone onset (s)")
    if c == 0:
        axes[0, c].set_ylabel("GLM kernel (log gain)")
        axes[0, c].legend(fontsize=7, frameon=False)

    # model vs empirical PSTH at the unit's best frequency
    bf = freqs[np.argmax(gain[:, k])]
    ons_bf = onsets[freq == bf]
    t_emp, r_emp = analysis.psth(s["units"][u], ons_bf, window=(-0.1, 0.3),
                                 bin_size=BIN, sigma_bins=0)
    peth_pred = nap.compute_perievent(pred[:, k], nap.Ts(ons_bf), (-0.1, 0.3))
    axes[1, c].plot(t_emp, r_emp, color="k", lw=1.4, label="data")
    axes[1, c].plot(np.asarray(peth_pred.t), np.asarray(peth_pred.d).mean(1),
                    color="tab:red", lw=1.4, label="GLM")
    axes[1, c].axvspan(0, s["duration"], color="0.88", zorder=0)
    axes[1, c].set_xlabel("time from tone onset (s)")
    axes[1, c].set_title(f"PSTH at BF = {int(bf/1000)} kHz", fontsize=9)
    if c == 0:
        axes[1, c].set_ylabel("firing rate (spikes/s)")
        axes[1, c].legend(fontsize=7, frameon=False)

    axes[2, c].plot(freqs / 1000, gain[:, k], "o-", color="tab:red")
    axes[2, c].set_xscale("log", base=2)
    axes[2, c].set_xticks(freqs / 1000)
    axes[2, c].set_xticklabels([f"{int(f/1000)}" for f in freqs])
    axes[2, c].set_xlabel("tone frequency (kHz)")
    if c == 0:
        axes[2, c].set_ylabel("GLM: extra spikes per tone")
fig.suptitle("Poisson GLM with frequency-specific tone kernels "
             f"({s['subject']} session {s['session_id']})", y=0.965)
fig.savefig("fig05_glm_kernels.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ------------------------------------- GLM tuning vs descriptive tuning curve --
def center_norm(a):
    """Centre each unit's tuning curve and scale it to unit peak deviation."""
    c = a - a.mean(1, keepdims=True)
    return c / np.abs(c).max(1, keepdims=True)


emp = tuning_emp.loc[units].values
emp_n = center_norm(emp)
glm_n = center_norm(gain.T)
r_unit = np.array([np.corrcoef(emp_n[i], glm_n[i])[0, 1] for i in range(len(units))])
match = (np.argmax(emp, 1) == np.argmax(gain, 0)).mean()

fig, ax = plt.subplots(1, 2, figsize=(9.5, 4))
ax[0].scatter(emp_n.ravel(), glm_n.ravel(), s=14, color="tab:red", alpha=0.7)
ax[0].set_xlabel("descriptive tuning (normalized evoked rate)")
ax[0].set_ylabel("GLM tuning (normalized kernel gain)")
ax[0].set_title(f"r = {np.corrcoef(emp_n.ravel(), glm_n.ravel())[0, 1]:.2f} over "
                f"{emp_n.size} unit-frequency pairs", fontsize=10)
ax[1].hist(r_unit, np.linspace(-1, 1, 41), color="tab:red", alpha=0.8)
ax[1].axvline(np.median(r_unit), color="k", ls="--", lw=1,
              label=f"median r = {np.median(r_unit):.2f}")
ax[1].legend(fontsize=8, frameon=False, loc="upper left")
ax[1].set_xlabel("per-unit correlation between the two tuning estimates")
ax[1].set_ylabel("units")
ax[1].set_title(f"preferred frequency agrees in {100 * match:.0f}% of units", fontsize=10)
fig.tight_layout()
fig.savefig("fig06_glm_vs_empirical.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"median per-unit r = {np.median(r_unit):.2f}; BF agreement {100 * match:.0f}%")
print("log-likelihood per bin:", float(model.score(X, y)))
print("saved fig05_glm_kernels.png and fig06_glm_vs_empirical.png")
