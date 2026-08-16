"""Prototype: decode upcoming choice from pre-stimulus population activity in one IBL session."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

rng = np.random.default_rng(0)

d = np.load("session_31f22c47.npz", allow_pickle=False)
spike_times = d["spike_times"]
spike_times_index = d["spike_times_index"]
n_units = len(spike_times_index)

trials = {k[len("trial__"):]: d[k] for k in d.files if k.startswith("trial__")}
print({k: v.shape for k, v in trials.items()})

choice = trials["choice"]            # +1 left, -1 right, 0 nogo
stimOn = trials["stimOn_times"]
contrastL = np.nan_to_num(trials["contrastLeft"], nan=0.0)
contrastR = np.nan_to_num(trials["contrastRight"], nan=0.0)
probL = trials["probabilityLeft"]
firstMov = trials["firstMovement_times"]
feedback = trials["feedback_times"]

print("choice values:", np.unique(choice, return_counts=True))
print("probLeft values:", np.unique(probL, return_counts=True))
print("contrasts L:", np.unique(contrastL), " R:", np.unique(contrastR))
print("stimOn nan:", np.isnan(stimOn).sum(), "firstMov nan:", np.isnan(firstMov).sum())
rt = firstMov - stimOn
print("firstMovement - stimOn: median", np.nanmedian(rt), "frac<0:", np.nanmean(rt < 0))

# ---- build unit spike-time list ----
bounds = np.concatenate([[0], spike_times_index])
unit_spikes = [spike_times[bounds[i]:bounds[i+1]] for i in range(n_units)]
print("n_units:", n_units, "median spikes/unit:", np.median(np.diff(bounds)))

def count_matrix(t0, t1):
    """(n_trials, n_units) spike counts in windows [t0[i], t1[i])."""
    X = np.empty((len(t0), n_units), dtype=np.float64)
    for u, st in enumerate(unit_spikes):
        X[:, u] = np.searchsorted(st, t1, side="left") - np.searchsorted(st, t0, side="left")
    return X

def decode(X, y, n_shuf=200):
    """5-fold CV balanced accuracy + label-shuffle null."""
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    def cv_acc(yy):
        accs = []
        for tr, te in skf.split(X, yy):
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))
            clf.fit(X[tr], yy[tr])
            p = clf.predict(X[te])
            # balanced accuracy: mean of per-class recall on the test fold
            accs.append(np.mean([np.mean(p[yy[te] == c] == c) for c in np.unique(yy)]))
        return np.mean(accs)
    real = cv_acc(y)
    null = np.array([cv_acc(rng.permutation(y)) for _ in range(n_shuf)])
    return real, null

# ---- behavior: psychometric by block prior ----
signed_contrast = contrastR - contrastL   # >0 means stimulus on right
chose_left = choice == 1
valid = (choice != 0) & ~np.isnan(stimOn)
fig, ax = plt.subplots(figsize=(6, 4))
for pl, color in zip([0.8, 0.5, 0.2], ["tab:blue", "k", "tab:red"]):
    m = valid & (probL == pl)
    sc = signed_contrast[m]
    cl = chose_left[m]
    bins = np.unique(sc)
    p = [cl[sc == b].mean() for b in bins]
    n = [np.sum(sc == b) for b in bins]
    ax.plot(bins, p, "o-", color=color, label=f"P(left)={pl}")
ax.axhline(0.5, color="gray", ls="--", lw=0.8)
ax.axvline(0, color="gray", ls="--", lw=0.8)
ax.set_xlabel("signed contrast (right - left)")
ax.set_ylabel("P(chose left)")
ax.set_title("Psychometric curves by block prior")
ax.legend()
fig.tight_layout()
fig.savefig("figs/proto_psychometric.png", dpi=150)

# ---- pre-stimulus decoding: 0% contrast trials (pure bias) ----
PRE = (-0.4, 0.0)
zero_contrast = valid & (contrastL == 0) & (contrastR == 0)
print("zero-contrast valid trials:", zero_contrast.sum())
t0 = stimOn[zero_contrast] + PRE[0]
t1 = stimOn[zero_contrast] + PRE[1]
y = chose_left[zero_contrast].astype(int)
print("class balance:", np.bincount(y))
X = count_matrix(t0, t1)
real, null = decode(X, y, n_shuf=100)
print(f"0%contrast pre-stim choice decoding: {real:.3f}  null {null.mean():.3f}+-{null.std():.3f}  p={(np.mean(null >= real)):.3f}")

# ---- pre-stimulus decoding: all valid trials ----
t0 = stimOn[valid] + PRE[0]; t1 = stimOn[valid] + PRE[1]
y_all = chose_left[valid].astype(int)
X_all = count_matrix(t0, t1)
real_all, null_all = decode(X_all, y_all, n_shuf=100)
print(f"all-trials pre-stim choice decoding: {real_all:.3f}  null {null_all.mean():.3f}+-{null_all.std():.3f}  p={(np.mean(null_all >= real_all)):.3f}")

# ---- block prior decoding (0.8 vs 0.2 blocks), pre-stimulus ----
block = valid & np.isin(probL, [0.2, 0.8])
t0 = stimOn[block] + PRE[0]; t1 = stimOn[block] + PRE[1]
y_block = (probL[block] == 0.8).astype(int)
X_block = count_matrix(t0, t1)
real_b, null_b = decode(X_block, y_block, n_shuf=100)
print(f"block-prior pre-stim decoding: {real_b:.3f}  null {null_b.mean():.3f}+-{null_b.std():.3f}  p={(np.mean(null_b >= real_b)):.3f}")

# ---- time-resolved decoding relative to stimOn (0% contrast) ----
edges = np.arange(-1.2, 0.8, 0.1)
centers = edges[:-1] + 0.05
accs, null_lo, null_hi = [], [], []
for a, b in zip(edges[:-1], edges[1:]):
    Xw = count_matrix(stimOn[zero_contrast] + a, stimOn[zero_contrast] + b)
    r, n_ = decode(Xw, y, n_shuf=20)
    accs.append(r); null_lo.append(np.percentile(n_, 5)); null_hi.append(np.percentile(n_, 95))
fig, ax = plt.subplots(figsize=(7, 4))
ax.fill_between(centers, null_lo, null_hi, color="gray", alpha=0.4, label="shuffle 5-95%")
ax.plot(centers, accs, "o-", color="tab:blue", label="real")
ax.axvline(0, color="k", ls="--", lw=1)
ax.axhline(0.5, color="gray", ls=":", lw=0.8)
ax.set_xlabel("time from stimulus onset (s), 100 ms window")
ax.set_ylabel("balanced accuracy")
ax.set_title("Time-resolved choice decoding (0% contrast trials)")
ax.legend()
fig.tight_layout()
fig.savefig("figs/proto_timecourse.png", dpi=150)
print("done")
