"""Poisson GLM of tone-evoked firing, fit with NeMoS.

Two nested encoding models are fit to the same binned spike trains:

  reduced : one temporal kernel driven by every tone onset, whatever its
            frequency (the neuron hears a sound but cannot tell them apart)
  full    : five temporal kernels, one per frequency

Comparing held-out log-likelihood between the two turns "is this neuron
frequency tuned?" into a model-comparison question, and the fitted kernels give
a smooth, regularized estimate of the response time course at each frequency.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nemos as nmo
import numpy as np
import pynapple as nap

import dandi_io
import tuning

ASSET = "aacd1c8a-73f7-469e-bf08-0afd5c1052f9"
BIN = 0.005                 # s
WINDOW = int(0.15 / BIN)    # 150 ms of stimulus history
N_BASIS = 7

sess = dandi_io.load_tone_session(ASSET)
spikes, onsets, freq = sess["spikes"], sess["trials"]["start_time"].values, sess["frequency"]
freqs = sess["frequencies"]

res = tuning.evoked_rates(spikes, onsets, freq)
stat = tuning.tuning_statistics(res, freq)
sel = np.where(stat["responsive"] & stat["tuned"] & stat["enhanced"])[0]
print("fitting %d units" % len(sel))

# ------------------------------------------------------------ design matrix
block = nap.IntervalSet(start=onsets.min() - 0.5, end=sess["trials"]["stop_time"].max() + 0.5)
counts = spikes[list(sel)].count(BIN, ep=block)
t = counts.times()
print("bins:", counts.shape)

# one binary event train per frequency, aligned to the spike-count bins
edges = np.append(t - BIN / 2, t[-1] + BIN / 2)
events = np.stack([np.histogram(onsets[freq == f], bins=edges)[0] for f in freqs], axis=1)
events_any = events.sum(axis=1, keepdims=True)
print("events per frequency:", events.sum(axis=0), "(total %d)" % events.sum())

basis = nmo.basis.RaisedCosineLogConv(n_basis_funcs=N_BASIS, window_size=WINDOW, label="tone")
X_full = np.concatenate([np.asarray(basis.compute_features(events[:, k]))
                         for k in range(len(freqs))], axis=1)
X_red = np.asarray(basis.compute_features(events_any[:, 0]))
y = np.asarray(counts)
print("X_full", X_full.shape, " X_red", X_red.shape)

# alternate 100 s blocks between train and test so slow drift affects both
blk = (t // 100).astype(int)
train = blk % 2 == 0
test = ~train
valid = np.isfinite(X_full).all(axis=1)
train &= valid
test &= valid
print("train bins %d, test bins %d" % (train.sum(), test.sum()))


def fit(X, y, mask):
    model = nmo.glm.PopulationGLM(observation_model="Poisson", regularizer="Ridge",
                                  regularizer_strength=1e-6, solver_name="LBFGS",
                                  solver_kwargs={"tol": 1e-10, "maxiter": 500})
    model.fit(X[mask], y[mask])
    return model


m_full = fit(X_full, y, train)
m_red = fit(X_red, y, train)


def per_neuron_pseudo_r2(model, X, y, mask):
    """McFadden pseudo-R^2 per neuron against a constant-rate null."""
    rate = np.asarray(model.predict(X[mask]))
    obs = y[mask]
    eps = 1e-12
    ll = (obs * np.log(rate + eps) - rate).sum(axis=0)
    mu = obs.mean(axis=0, keepdims=True)
    ll0 = (obs * np.log(mu + eps) - mu).sum(axis=0)
    llsat = (obs * np.log(obs + eps) - obs).sum(axis=0)
    return (ll - ll0) / (llsat - ll0)


r2_full = per_neuron_pseudo_r2(m_full, X_full, y, test)
r2_red = per_neuron_pseudo_r2(m_red, X_red, y, test)
gain = r2_full - r2_red
print("held-out pseudo-R2: full %.4f, reduced %.4f (median over units)"
      % (np.median(r2_full), np.median(r2_red)))
print("frequency-specific kernels improve held-out fit in %d/%d units"
      % ((gain > 0).sum(), len(sel)))

# ------------------------------------------------- reconstruct tone kernels
_, kern = basis.evaluate_on_grid(WINDOW)               # (WINDOW, N_BASIS)
coef = np.asarray(m_full.coef_)                        # (n_freq*N_BASIS, n_units)
coef = coef.reshape(len(freqs), N_BASIS, len(sel))
filters = np.einsum("bn,fnu->ufb", kern, coef)         # (n_units, n_freq, WINDOW)
lag = np.arange(WINDOW) * BIN

# GLM tuning curve: peak gain of each frequency kernel, in units of rate change
glm_tuning = filters.max(axis=2)
emp_tuning = res["evoked"][sel]
r_units = np.array([np.corrcoef(glm_tuning[i], emp_tuning[i])[0, 1] for i in range(len(sel))])
bf_agree = (np.argmax(glm_tuning, axis=1) == np.argmax(emp_tuning, axis=1)).mean()
print("GLM vs empirical tuning: median r = %.3f, same BF in %.0f%% of units"
      % (np.nanmedian(r_units), 100 * bf_agree))

# ------------------------------------------------------------------- figure
colors = plt.cm.viridis(np.linspace(0, 0.92, len(freqs)))
show = np.argsort(gain)[::-1][:3]

fig, axes = plt.subplots(2, 3, figsize=(14, 7.2))
for c, i in enumerate(show):
    ax = axes[0, c]
    for k, f in enumerate(freqs):
        ax.plot(lag * 1000, filters[i, k], color=colors[k], lw=1.6, label=f"{f/1000:g} kHz")
    ax.axhline(0, color="0.7", lw=0.8, ls=":")
    ax.set_title("unit %d   $\\Delta$pseudo-$R^2$ = %.3f" % (sess["spikes"].index[sel[i]], gain[i]),
                 fontsize=10)
    ax.set_xlabel("lag from tone onset (ms)")
    if c == 0:
        ax.set_ylabel("GLM tone kernel (log rate)")
axes[0, 2].legend(fontsize=8, frameon=False)

ax = axes[1, 0]
ax.scatter(r2_red, r2_full, s=14, color="tab:blue", alpha=0.7, edgecolor="none")
lim = [0, max(r2_full.max(), r2_red.max()) * 1.05]
ax.plot(lim, lim, "k--", lw=1)
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_xlabel("reduced model (frequency-blind)")
ax.set_ylabel("full model (frequency-specific)")
ax.set_title("held-out pseudo-$R^2$", fontsize=11)

ax = axes[1, 1]
ax.hist(gain, bins=30, color="0.5")
ax.axvline(0, color="k", lw=1)
ax.set_xlabel("$\\Delta$ held-out pseudo-$R^2$ (full $-$ reduced)")
ax.set_ylabel("# units")
ax.set_title("gain from knowing the frequency", fontsize=11)

ax = axes[1, 2]
ax.hist(r_units, bins=np.linspace(-1, 1, 25), color="0.5")
ax.axvline(np.nanmedian(r_units), color="crimson", lw=2)
ax.set_xlabel("correlation, GLM kernel peak vs measured tuning")
ax.set_ylabel("# units")
ax.set_title("GLM recovers the measured tuning\n(median r = %.2f)" % np.nanmedian(r_units),
             fontsize=11)

fig.suptitle("NeMoS Poisson GLM: frequency-specific tone kernels in auditory cortex", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("fig08_glm.png", dpi=150)
plt.close(fig)

np.savez("glm_LA11_ses1.npz", filters=filters, lag=lag, freqs=freqs, r2_full=r2_full,
         r2_red=r2_red, glm_tuning=glm_tuning, emp_tuning=emp_tuning, r_units=r_units)
