"""Model-based confirmation of frequency tuning.

Two complementary analyses, both with held-out data:

1. Encoding: a NeMoS Poisson population GLM predicts each unit's spike count in 5 ms bins
   around tone onset. The full model uses a (time-since-onset basis x tone frequency)
   design; the reduced model uses time-since-onset only. The gain in held-out
   log-likelihood is the evidence that frequency matters for that unit.

2. Decoding: a NeMoS multinomial classifier GLM recovers which of the 5 tones was
   played from the single-trial population spike counts.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nemos as nmo
import numpy as np
import pynapple as nap
from sklearn.model_selection import StratifiedKFold

import dandi_auditory as da

BIN = 0.005
# The window starts at tone onset so that the log-spaced basis puts its finest
# resolution on the first tens of milliseconds, where the transient response lives.
GLM_WINDOW = (0.0, 0.150)
N_BASIS = 10
# The encoding GLM uses a random subset of trials; 600 repeats per frequency already
# pin down 50 parameters per unit, and the full 7,447 trials only make the fit slower.
GLM_TRIALS = 3000
rng = np.random.default_rng(0)

assets = da.list_assets()
asset_id = assets.query("subject == 'LA11' and session == '1'")["asset_id"].iloc[0]
s = da.load_session(asset_id)
units, freqs, freq = s["units"], s["freqs"], s["frequency"]
onsets = s["trials"].start
n_units, n_trials, n_freq = len(units), len(onsets), len(freqs)
print(f"sub-{s['subject']} ses-{s['session_id']}: {n_units} units, {n_trials} trials")

# ------------------------------------------------------------------ 1. encoding
edges = np.arange(GLM_WINDOW[0], GLM_WINDOW[1] + BIN / 2, BIN)
n_bins = len(edges) - 1
t_since_onset = (edges[:-1] + edges[1:]) / 2

glm_trials = np.sort(rng.permutation(n_trials)[:GLM_TRIALS])
glm_onsets, glm_freq = onsets[glm_trials], freq[glm_trials]

# counts: (GLM_TRIALS, n_bins, n_units), built with pynapple interval counting
counts = np.zeros((len(glm_trials), n_bins, n_units), dtype=np.float32)
for b in range(n_bins):
    ep = nap.IntervalSet(start=glm_onsets + edges[b], end=glm_onsets + edges[b + 1])
    counts[:, b, :] = np.asarray(units.count(ep=ep).values)
print("counts", counts.shape, "mean rate %.2f Hz" % (counts.mean() / BIN))

# log-spaced raised-cosine basis over time since onset
basis = nmo.basis.RaisedCosineLogEval(n_basis_funcs=N_BASIS)
B = basis.compute_features(t_since_onset)      # (n_bins, N_BASIS)

onehot = (glm_freq[:, None] == freqs[None, :]).astype(np.float32)
# full design: outer product of frequency identity and time basis
X_full = np.einsum("tf,bk->tbfk", onehot, B).reshape(len(glm_trials) * n_bins, n_freq * N_BASIS)
X_red = np.tile(B, (len(glm_trials), 1))
y = counts.reshape(len(glm_trials) * n_bins, n_units)
print("design", X_full.shape, X_red.shape)

is_train = np.zeros(len(glm_trials), bool)
is_train[rng.permutation(len(glm_trials))[: int(0.8 * len(glm_trials))]] = True
bin_train = np.repeat(is_train, n_bins)


def fit_score(X):
    model = nmo.glm.PopulationGLM(
        regularizer="Ridge", regularizer_strength=1e-3, solver_name="LBFGS",
        solver_kwargs={"maxiter": 150, "tol": 1e-7},
    ).fit(X[bin_train], y[bin_train])
    # per-unit held-out Poisson log-likelihood
    rate = np.asarray(model.predict(X[~bin_train]))
    yt = y[~bin_train]
    from scipy.special import gammaln
    ll = (yt * np.log(rate + 1e-12) - rate - gammaln(yt + 1)).mean(axis=0)
    return model, ll


model_full, ll_full = fit_score(X_full)
model_red, ll_red = fit_score(X_red)

# null model: constant rate per unit estimated on training data
mu = y[bin_train].mean(axis=0)
from scipy.special import gammaln
ll_null = (y[~bin_train] * np.log(mu + 1e-12) - mu - gammaln(y[~bin_train] + 1)).mean(axis=0)

# McFadden pseudo-R^2 against the constant-rate null, evaluated on held-out trials
pr2_full = 1 - ll_full / ll_null
pr2_red = 1 - ll_red / ll_null
d_ll = ll_full - ll_red  # nats/bin gained by knowing the tone frequency
print("units with positive frequency gain: %d/%d" % (np.sum(d_ll > 0), n_units))

# reconstruct the model's frequency-specific response kernels
coef = np.asarray(model_full.coef_).reshape(n_freq, N_BASIS, n_units)
kernels = np.einsum("fku,bk->fbu", coef, B)
pred_rate = np.exp(kernels + np.asarray(model_full.intercept_)[None, None, :]) / BIN

# ------------------------------------------------------------------ 2. decoding
evok = da.trial_spike_counts(units, onsets, da.EVOKED_WINDOW).astype(np.float64)
labels = np.searchsorted(freqs, freq)

Xd = (evok - evok.mean(0)) / (evok.std(0) + 1e-9)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
conf = np.zeros((n_freq, n_freq))
acc = []
for tr, te in cv.split(Xd, labels):
    clf = nmo.glm.ClassifierGLM(
        n_classes=n_freq, regularizer="Ridge", regularizer_strength=0.01,
        solver_name="LBFGS",
    ).fit(Xd[tr], labels[tr])
    pred = np.asarray(clf.predict(Xd[te]))
    acc.append(np.mean(pred == labels[te]))
    for a, b in zip(labels[te], pred):
        conf[a, b] += 1
conf = conf / conf.sum(axis=1, keepdims=True)
print("decoding accuracy %.3f +/- %.3f (chance %.2f)" % (np.mean(acc), np.std(acc), 1 / n_freq))

# accuracy as a function of population size
sizes = [1, 2, 5, 10, 20, 50, 100, min(200, n_units), n_units]
sizes = sorted(set(int(x) for x in sizes if x <= n_units))
curve = np.zeros((len(sizes), 5))
for i, k in enumerate(sizes):
    for rep in range(5):
        sel = rng.choice(n_units, k, replace=False)
        tr, te = next(iter(cv.split(Xd, labels)))
        clf = nmo.glm.ClassifierGLM(
            n_classes=n_freq, regularizer="Ridge", regularizer_strength=0.01,
            solver_name="LBFGS",
        ).fit(Xd[tr][:, sel], labels[tr])
        curve[i, rep] = np.mean(np.asarray(clf.predict(Xd[te][:, sel])) == labels[te])
    print("  %d units: %.3f" % (k, curve[i].mean()))

# ------------------------------------------------------------------ figure
fig = plt.figure(figsize=(14, 7.8))
gs = fig.add_gridspec(2, 3, hspace=0.5, wspace=0.42)

# GLM-predicted frequency kernels for the best-fit unit
best = int(np.argmax(d_ll))
ax = fig.add_subplot(gs[0, 0])
obs = counts[:, :, best]
for fi, f in enumerate(freqs):
    ax.plot(t_since_onset, obs[glm_freq == f].mean(0) / BIN, color=da.FREQ_COLORS[fi],
            lw=1, alpha=0.45)
    ax.plot(t_since_onset, pred_rate[fi, :, best], color=da.FREQ_COLORS[fi], lw=2,
            label=f"{f/1000:g} kHz")
ax.axvspan(0, s["tone_duration"], color="0.9", zorder=0)
ax.set_xlabel("time from tone onset (s)")
ax.set_ylabel("firing rate (Hz)")
ax.set_title(f"GLM fit, unit {list(units.keys())[best]}\n(thin = data, thick = model)",
             fontsize=10)
ax.legend(fontsize=7, frameon=False)

ax = fig.add_subplot(gs[0, 1])
ax.hist(d_ll * 1000, bins=50, color="#4477AA")
ax.axvline(0, color="k", lw=1, ls="--")
ax.set_xlabel("held-out log-likelihood gain\nfrom tone frequency (millinats/bin)")
ax.set_ylabel("# units")
ax.set_title("Frequency improves prediction\nin %d/%d units" % (np.sum(d_ll > 0), n_units),
             fontsize=10)

ax = fig.add_subplot(gs[0, 2])
ax.scatter(pr2_red, pr2_full, s=8, color="#CC6677", alpha=0.7)
lim = [min(pr2_red.min(), pr2_full.min()), max(pr2_red.max(), pr2_full.max())]
ax.plot(lim, lim, "k--", lw=1)
ax.set_xlabel("pseudo-$R^2$, time-only model")
ax.set_ylabel("pseudo-$R^2$, time $\\times$ frequency")
ax.set_title("Held-out goodness of fit", fontsize=10)

ax = fig.add_subplot(gs[1, 0])
im = ax.imshow(conf, cmap="magma", vmin=0, vmax=conf.max())
ax.set_xticks(range(n_freq)); ax.set_xticklabels([f"{f/1000:g}" for f in freqs])
ax.set_yticks(range(n_freq)); ax.set_yticklabels([f"{f/1000:g}" for f in freqs])
ax.set_xlabel("decoded frequency (kHz)")
ax.set_ylabel("true frequency (kHz)")
ax.set_title("Single-trial decoding\naccuracy %.1f%% (chance %.0f%%)"
             % (100 * np.mean(acc), 100 / n_freq), fontsize=10)
fig.colorbar(im, ax=ax, fraction=0.046).set_label("P(decoded | true)", fontsize=8)

ax = fig.add_subplot(gs[1, 1])
ax.errorbar(sizes, curve.mean(1) * 100, yerr=curve.std(1) * 100, marker="o", color="k",
            capsize=3)
ax.axhline(100 / n_freq, color="0.5", ls="--", lw=1, label="chance")
ax.set_xscale("log")
ax.set_xlabel("# units in decoder")
ax.set_ylabel("accuracy (%)")
ax.set_title("Decoding scales with population size", fontsize=10)
ax.legend(fontsize=8, frameon=False)

ax = fig.add_subplot(gs[1, 2])
order = np.argsort(d_ll)[::-1][:30]
ax.barh(np.arange(len(order)), d_ll[order] * 1000, color="#117733")
ax.set_yticks([])
ax.invert_yaxis()
ax.set_xlabel("log-likelihood gain (millinats/bin)")
ax.set_title("30 most frequency-dependent units", fontsize=10)

fig.suptitle(f"GLM encoding and decoding of tone frequency, sub-{s['subject']} ses-{s['session_id']}",
             y=0.98)
fig.savefig("fig06_glm_decoding.png", dpi=150, bbox_inches="tight")
plt.close(fig)

np.savez("glm_decoding.npz", d_ll=d_ll, pr2_full=pr2_full, pr2_red=pr2_red, conf=conf,
         acc=np.array(acc), sizes=np.array(sizes), curve=curve, freqs=freqs)
print("wrote fig06_glm_decoding.png")
