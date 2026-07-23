"""A Poisson GLM view of frequency tuning, fitted with NeMoS.

The trial-averaged tuning curves above are model-free.  Here the same data are
described by an encoding model: spike counts in 5 ms bins are predicted from
five frequency-specific event trains, each convolved with a raised-cosine basis.
The fitted coefficients give a temporal response kernel per frequency, so tuning
and response latency fall out of the same fit, and the model can be scored on
held-out time.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nemos as nmo
import numpy as np
import pynapple as nap

import aft_analysis as aa
import aft_io

PROTOTYPE = "sub-LA11/sub-LA11_ses-1_behavior.nwb"
BIN = 0.005            # s
WINDOW_BINS = 40       # 200 ms stimulus kernel
N_BASIS = 8
N_TRIALS = 2500        # trials used for the GLM (keeps the design matrix small)
TRAIN_FRAC = 0.75

manifest = aft_io.list_assets()
nwbfile, nwb, h5 = aft_io.open_session(manifest[PROTOTYPE])
units = aft_io.load_units(nwbfile)
trials = aft_io.load_trials(nwbfile).iloc[:N_TRIALS]
res = aa.analyse_session(units, trials, n_perm=200)

freqs = res["freqs"]
n_freq = len(freqs)
driven = np.nonzero(res["driven"])[0]
unit_ids = res["unit_ids"][driven]
print(f"fitting {len(unit_ids)} tone-driven units on {len(trials)} trials")

# The tone block is broken by long spontaneous gaps; model only the contiguous
# stretches of tone presentation.
gaps = np.nonzero(np.diff(trials.start_time.values) > 2.0)[0]
starts = np.concatenate([[0], gaps + 1])
stops = np.concatenate([gaps, [len(trials) - 1]])
epoch = nap.IntervalSet(
    start=trials.start_time.values[starts] - 0.2,
    end=trials.start_time.values[stops] + 0.5,
)
print("modelled epochs:", np.round(epoch.values, 1).tolist())

counts = units[list(unit_ids)].count(BIN, epoch)
t = counts.times()
y = np.asarray(counts, dtype=np.float32)
print("design bins:", y.shape)

# One event train per frequency: a 1 in the bin containing the tone onset.
stim = np.zeros((len(t), n_freq), dtype=np.float32)
onset_bin = np.searchsorted(t, trials.start_time.values)
onset_bin = np.clip(onset_bin, 0, len(t) - 1)
fidx = np.searchsorted(freqs, trials.stim_frequency.values)
stim[onset_bin, fidx] = 1.0
print("onsets placed:", int(stim.sum()), "of", len(trials))

basis = nmo.basis.RaisedCosineLogConv(n_basis_funcs=N_BASIS,
                                      window_size=WINDOW_BINS, label="tone")
basis.set_input_shape(stim)
X = np.asarray(basis.compute_features(stim), dtype=np.float32)

valid = ~np.isnan(X).any(axis=1)
split = int(TRAIN_FRAC * len(t))
train = valid.copy(); train[split:] = False
test = valid.copy(); test[:split] = False
print(f"train bins {train.sum()}, test bins {test.sum()}")

model = nmo.glm.PopulationGLM(regularizer="Ridge", regularizer_strength=1e-6,
                              solver_name="LBFGS")
model.fit(X[train], y[train])

rate_test = np.asarray(model.predict(X[test]))


def poisson_ll(counts_, rate_):
    """Poisson log-likelihood per unit, dropping the count! constant."""
    from scipy.special import gammaln
    rate_ = np.maximum(rate_, 1e-10)
    return (counts_ * np.log(rate_) - rate_ - gammaln(counts_ + 1)).sum(axis=0)


ll_model = poisson_ll(y[test], rate_test)
ll_null = poisson_ll(y[test], np.tile(y[train].mean(axis=0), (test.sum(), 1)))
ll_sat = poisson_ll(y[test], y[test])
pseudo_r2 = (ll_model - ll_null) / (ll_sat - ll_null)
print(f"held-out pseudo-R2: median {np.median(pseudo_r2):.4f}, "
      f"{(pseudo_r2 > 0).mean() * 100:.0f}% of units above the mean-rate model")

# --- reconstruct the per-frequency temporal kernels ------------------------
coef = basis.split_by_feature(model.coef_, axis=0)["tone"]  # (n_freq, n_basis, n_units)
_, kernels = basis.evaluate_on_grid(WINDOW_BINS)            # (window, n_basis)
filt = np.einsum("tb,fbu->tfu", np.asarray(kernels), np.asarray(coef))
lag = np.arange(WINDOW_BINS) * BIN * 1000

# GLM tuning: peak gain of the kernel, in multiplicative units on the rate.
glm_tuning = np.exp(filt.max(axis=0)).T - 1.0            # (n_units, n_freq)
glm_bf = freqs[np.argmax(filt.max(axis=0), axis=0)]
emp_bf = res["best_freq"][driven]
agree = (glm_bf == emp_bf).mean()
print(f"GLM and empirical best frequency agree for {agree * 100:.1f}% of units")

# --- a single-unit model with spike history, for comparison ----------------
hist_basis = nmo.basis.RaisedCosineLogConv(n_basis_funcs=5, window_size=20,
                                           label="history")
example_ids = []
for k in range(n_freq):
    cand = np.nonzero(res["freq_tuned"][driven] & (res["bf_idx"][driven] == k))[0]
    if len(cand):
        example_ids.append(cand[np.argmax(res["depth_z"][driven][cand])])

hist_gain = {}
for j in example_ids:
    own = y[:, [j]]
    Xh = np.asarray(hist_basis.compute_features(own), dtype=np.float32)
    Xfull = np.concatenate([X, Xh], axis=1)
    ok = ~np.isnan(Xfull).any(axis=1)
    tr = ok & train
    te = ok & test
    m = nmo.glm.GLM(regularizer="Ridge", regularizer_strength=1e-6,
                    solver_name="LBFGS").fit(Xfull[tr], y[tr, j])
    r = np.asarray(m.predict(Xfull[te]))
    ll_m = poisson_ll(y[te, j][:, None], r[:, None])[0]
    ll_0 = poisson_ll(y[te, j][:, None],
                      np.full((te.sum(), 1), y[tr, j].mean()))[0]
    ll_s = poisson_ll(y[te, j][:, None], y[te, j][:, None])[0]
    hist_gain[j] = ((ll_m - ll_0) / (ll_s - ll_0), pseudo_r2[j])
    print(f"  unit {unit_ids[j]}: pseudo-R2 stim-only {pseudo_r2[j]:.4f} -> "
          f"with spike history {hist_gain[j][0]:.4f}")

# ------------------------------------------------------------------ figure 7
cmap = plt.get_cmap("viridis")
fcol = [cmap(i / (n_freq - 1)) for i in range(n_freq)]
n_ex = len(example_ids)
fig = plt.figure(figsize=(3.0 * n_ex, 7.4))
gs = fig.add_gridspec(3, n_ex, height_ratios=[1, 1, 1.25], hspace=0.55, wspace=0.36)

for c, j in enumerate(example_ids):
    ax = fig.add_subplot(gs[0, c])
    for k in range(n_freq):
        ax.plot(lag, filt[:, k, j], color=fcol[k], lw=1.4,
                label=f"{freqs[k] / 1000:g} kHz")
    ax.axhline(0, color="0.6", lw=0.8, ls=":")
    ax.set_xlabel("lag from tone onset (ms)")
    ax.set_title(f"unit {unit_ids[j]}", pad=6)
    if c == 0:
        ax.set_ylabel("GLM stimulus filter\n(log gain)")
        ax.legend(fontsize=6.5, frameon=False, loc="upper right")

    ax = fig.add_subplot(gs[1, c])
    ax.plot(freqs / 1000, glm_tuning[j], "o-", color="tab:purple")
    ax.set_xscale("log", base=2)
    ax.set_xticks(freqs / 1000)
    ax.set_xticklabels([f"{f / 1000:g}" for f in freqs])
    ax.set_xlabel("frequency (kHz)")
    if c == 0:
        ax.set_ylabel("GLM peak gain\n(fold change in rate)")

ax = fig.add_subplot(gs[2, :2])
emp = res["tuning_evoked"][driven]
empn = emp / np.maximum(np.abs(emp).max(axis=1, keepdims=True), 1e-9)
glmn = glm_tuning / np.maximum(np.abs(glm_tuning).max(axis=1, keepdims=True), 1e-9)
r_per_unit = np.array([np.corrcoef(empn[i], glmn[i])[0, 1] for i in range(len(driven))])
ax.hist(r_per_unit, bins=np.linspace(-1, 1, 41), color="tab:purple")
ax.axvline(np.nanmedian(r_per_unit), color="k", ls="--",
           label=f"median r = {np.nanmedian(r_per_unit):.2f}")
ax.set_xlabel("correlation between GLM and trial-averaged tuning curve")
ax.set_ylabel("units")
ax.set_title(f"GLM reproduces the measured tuning ({len(driven)} tone-driven units); "
             f"best frequency matches for {agree * 100:.0f}%")
ax.legend(frameon=False, fontsize=8)

ax = fig.add_subplot(gs[2, 2:])
ax.hist(pseudo_r2, bins=30, color="0.5")
ax.axvline(0, color="k", lw=1)
ax.set_xlabel("held-out pseudo-$R^2$ (McFadden)")
ax.set_ylabel("units")
ax.set_title("Held-out goodness of fit,\nstimulus-only Poisson GLM")

fig.suptitle("NeMoS Poisson GLM: frequency-specific response kernels "
             f"({aft_io.session_label(PROTOTYPE)})", y=0.985)
fig.savefig("fig07_nemos_glm.png", dpi=150, bbox_inches="tight")
plt.close(fig)

np.savez("glm_results.npz", filt=filt, lag=lag, glm_tuning=glm_tuning,
         pseudo_r2=pseudo_r2, r_per_unit=r_per_unit, unit_ids=unit_ids,
         freqs=freqs, glm_bf=glm_bf, emp_bf=emp_bf)
print("wrote fig07_nemos_glm.png, glm_results.npz")
