"""Population decoding of tone frequency + a NeMoS Poisson GLM encoding model."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import nemos as nmo
import pynapple as nap
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import aud_common as ac

RNG = np.random.default_rng(1)
SESSION = "LA11_ses1"

assets = ac.list_assets()
row = assets[assets.session_name == SESSION].iloc[0]
nwb, nwbfile = ac.load_session(row.asset_id)
units = nwb["units"]
tr = ac.trial_table(nwbfile)
onsets = tr.start_time.values
freqs = tr.stim_frequency.values
ufreq = np.unique(freqs)
labels = np.searchsorted(ufreq, freqs)

evoked = ac.counts_in_windows(units, onsets, ac.EVOKED_WIN)
print("design:", evoked.shape)

# --- decoding ---------------------------------------------------------------
def cv_decode(X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    yp = np.empty_like(y)
    for tr_i, te_i in skf.split(X, y):
        sc = StandardScaler().fit(X[tr_i])
        clf = LogisticRegression(max_iter=2000, C=0.1)
        clf.fit(sc.transform(X[tr_i]), y[tr_i])
        yp[te_i] = clf.predict(sc.transform(X[te_i]))
    return yp


y_pred = cv_decode(evoked, labels)
acc = (y_pred == labels).mean()
y_shuf = cv_decode(evoked, RNG.permutation(labels))
acc_shuf = (y_shuf == RNG.permutation(labels)).mean()
print("decoding accuracy %.3f (chance %.3f, label-shuffled %.3f)"
      % (acc, 1 / len(ufreq), acc_shuf))
cm = confusion_matrix(labels, y_pred, normalize="true")

# accuracy as a function of population size
sizes = [1, 2, 5, 10, 20, 50, 100, 235]
sizes = [s for s in sizes if s <= evoked.shape[1]]
curve = []
for s in tqdm(sizes, desc="pop size"):
    reps = []
    for _ in range(5):
        idx = RNG.choice(evoked.shape[1], s, replace=False)
        reps.append((cv_decode(evoked[:, idx], labels) == labels).mean())
    curve.append(reps)
curve = np.array(curve)

# --- NeMoS Poisson GLM encoding model --------------------------------------
# Predict 5 ms binned spike counts from frequency-specific tone events convolved
# with a raised-cosine basis.  The fitted kernels are the model's estimate of
# each unit's response to each frequency.
BIN = 0.005
N_BASIS = 8
WINDOW = int(0.200 / BIN)          # 200 ms kernel
N_TRIALS_GLM = 2500

sub_on = onsets[:N_TRIALS_GLM]
sub_lab = labels[:N_TRIALS_GLM]
ep = nap.IntervalSet(start=sub_on[0] - 0.5, end=sub_on[-1] + 0.5)
counts = units.count(BIN, ep=ep)            # TsdFrame [n_bins, n_units]
t = counts.t
print("GLM bins:", counts.shape)

stim = np.zeros((counts.shape[0], len(ufreq)), dtype=np.float32)
bin_of_onset = np.searchsorted(t, sub_on) - 1
stim[bin_of_onset, sub_lab] = 1.0
print("stim events placed:", stim.sum(axis=0))

basis = nmo.basis.RaisedCosineLogConv(n_basis_funcs=N_BASIS, window_size=WINDOW)
X = np.concatenate([basis.compute_features(stim[:, i]) for i in range(len(ufreq))],
                   axis=1).astype(np.float32)
valid = ~np.any(np.isnan(X), axis=1)
X, Y = X[valid], np.asarray(counts)[valid].astype(np.float32)
print("GLM design:", X.shape, "targets:", Y.shape)

glm = nmo.glm.PopulationGLM(
    regularizer="Ridge", regularizer_strength=1e-4, solver_name="LBFGS",
    solver_kwargs={"tol": 1e-8},
)
glm.fit(X, Y)
print("GLM fitted, coef shape", glm.coef_.shape)

kernel_time = np.arange(WINDOW) * BIN * 1000
_, basis_kernels = basis.evaluate_on_grid(WINDOW)
W = np.asarray(glm.coef_).reshape(len(ufreq), N_BASIS, Y.shape[1])
kernels = np.einsum("tb,fbn->fnt", basis_kernels, W)   # [n_freq, n_units, n_time]
glm_gain = kernels.max(axis=2)                          # [n_freq, n_units]

emp = ac.tuning_from_counts(evoked, freqs, ac.EVOKED_WIN)[1]
emp_d = emp - ac.tuning_from_counts(
    ac.counts_in_windows(units, onsets, ac.BASELINE_WIN), freqs, ac.BASELINE_WIN)[1]
agree = np.mean(np.argmax(glm_gain, axis=0) == np.argmax(emp_d, axis=0))
print("GLM vs empirical best-frequency agreement: %.2f" % agree)

np.savez("decoding_glm.npz", cm=cm, acc=acc, acc_shuf=acc_shuf, sizes=sizes,
         curve=curve, kernels=kernels, glm_gain=glm_gain, emp_d=emp_d,
         ufreq=ufreq, kernel_time=kernel_time, agree=agree)

# --- Figure 7: decoding -----------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))
im = axes[0].imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
axes[0].set_xticks(range(5), ["%g" % (f / 1000) for f in ufreq])
axes[0].set_yticks(range(5), ["%g" % (f / 1000) for f in ufreq])
axes[0].set_xlabel("decoded frequency (kHz)")
axes[0].set_ylabel("presented frequency (kHz)")
axes[0].set_title("Single-trial decoding\n%.0f%% correct (chance 20%%)" % (100 * acc))
for i in range(5):
    for j in range(5):
        axes[0].text(j, i, "%.2f" % cm[i, j], ha="center", va="center",
                     color="w" if cm[i, j] > cm.max() / 2 else "k", fontsize=8)
plt.colorbar(im, ax=axes[0], label="P(decoded | presented)")

m = curve.mean(axis=1)
axes[1].plot(sizes, 100 * m, "o-", color="C0")
axes[1].fill_between(sizes, 100 * curve.min(axis=1), 100 * curve.max(axis=1),
                     alpha=0.25, color="C0")
axes[1].axhline(20, color="k", ls="--", lw=1, label="chance")
axes[1].set_xscale("log")
axes[1].set_xlabel("number of units")
axes[1].set_ylabel("decoding accuracy (%)")
axes[1].set_title("Accuracy vs population size")
axes[1].legend(frameon=False)

axes[2].scatter(emp_d.ravel(), glm_gain.ravel(), s=6, alpha=0.35, color="C2")
axes[2].set_xlabel("empirical evoked rate change (Hz)")
axes[2].set_ylabel("GLM kernel peak (log gain)")
axes[2].set_title("GLM vs empirical tuning\n(best-frequency agreement %.0f%%)"
                  % (100 * agree))
for a in axes[1:]:
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("fig07_decoding.png", dpi=140)
print("saved fig07")

# --- Figure 8: GLM kernels for example units -------------------------------
sel = np.argsort(-(glm_gain.max(axis=0) - glm_gain.min(axis=0)))[:6]
colors = plt.get_cmap("viridis")(np.linspace(0, 0.92, len(ufreq)))
fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
for c, j in enumerate(sel):
    ax = axes.ravel()[c]
    for i in range(len(ufreq)):
        ax.plot(kernel_time, kernels[i, j], color=colors[i],
                label="%g kHz" % (ufreq[i] / 1000))
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_title("unit %d" % j, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    if c >= 3:
        ax.set_xlabel("time from tone onset (ms)")
    if c % 3 == 0:
        ax.set_ylabel("GLM kernel (log rate)")
    if c == 0:
        ax.legend(fontsize=7, frameon=False)
fig.suptitle("NeMoS Poisson GLM: frequency-specific temporal response kernels (%s)"
             % SESSION)
fig.tight_layout()
fig.savefig("fig08_glm_kernels.png", dpi=140)
print("saved fig08")
