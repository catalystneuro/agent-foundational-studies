# GLM figure: example prediction, direction-vs-speed contributions, model comparison
import h5py
from pynwb import NWBHDF5IO
import remfile
import pynapple as nap
import numpy as np
import nemos as nmo
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

G = np.load("glm_results.npz")
pr2_full, pr2_dir, pr2_spd = G["pr2_full"], G["pr2_dir"], G["pr2_spd"]
dll_dir, dll_spd = G["dll_dir"], G["dll_spd"]

import pickle
with open("analysis_results.pkl", "rb") as f:
    R = pickle.load(f)
trials = R["trials"]
unit_ids = np.load("unit_ids.npy")

# rebuild design matrix to refit the best unit
s3 = "https://api.dandiarchive.org/api/assets/26e85f09-39b7-480f-b337-278a8f034007/download/"
rf = remfile.File(s3, disk_cache=remfile.DiskCache("/tmp/remfile_cache_mcmaze"))
f5 = h5py.File(rf, "r")
io = NWBHDF5IO(file=f5)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)
units = nwb["units"]
hv = nwb["hand_vel"]

onset_t = trials["move_onset_time"].values
BIN = 0.01
MOVE_EPOCH = nap.IntervalSet(start=onset_t, end=onset_t + 0.6)
counts = units.count(BIN, ep=MOVE_EPOCH)
hv_bin = hv.bin_average(BIN, ep=MOVE_EPOCH)
n_b = min(len(counts), len(hv_bin))
C = counts.values[:n_b]
V = hv_bin.values[:n_b]
speed = np.linalg.norm(V, axis=1)
vang = np.arctan2(V[:, 1], V[:, 0])
X_full = np.column_stack([np.cos(vang), np.sin(vang), speed / 500.0])
nbins_per_trial = int(round(0.6 / BIN))
trial_mask = np.repeat((trials["split"].values == "train"), nbins_per_trial)[:n_b]

# restrict to units with a reasonable firing rate (pseudo-R2 is unstable for near-silent units)
mean_rate = C.mean(axis=0) / BIN  # Hz over movement bins
rate_ok = mean_rate >= 2.0
print(f"units with rate >= 2 Hz: {rate_ok.sum()}/{len(rate_ok)}")
pr2_masked = np.where(rate_ok, pr2_full, -np.inf)
best = int(np.argmax(pr2_masked))
print("best unit:", best, "id:", unit_ids[best], "pseudo-R2:", pr2_full[best].round(3),
      "mean rate:", mean_rate[best].round(1), "Hz")
y = C[:, best].astype(float)
m = nmo.glm.GLM(solver_name="LBFGS")
init = (np.zeros(3), np.array([np.log(np.clip(y[trial_mask].mean(), 1e-6, None))]))
m.fit(X_full[trial_mask], y[trial_mask], init_params=init)
mu = np.asarray(m.predict(X_full[~trial_mask])) / BIN  # Hz
yte = y[~trial_mask] / BIN

fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
# (a) example prediction segment: most active 4 s chunk of held-out bins
win = 400
act = np.convolve(yte, np.ones(win), mode="valid")
s0 = int(np.argmax(act))
seg = slice(s0, s0 + win)
tt = np.arange(400) * BIN
axes[0].plot(tt, gaussian_filter1d(yte[seg], 5), color="k", lw=1.2, label="actual (smoothed)")
axes[0].plot(tt, mu[seg], color="crimson", lw=1.2, label="GLM prediction")
axes[0].set_xlabel("time within held-out trials (s)")
axes[0].set_ylabel("firing rate (Hz)")
axes[0].set_title(f"Example encoding: unit {unit_ids[best]} (held-out pseudo-R²={pr2_full[best]:.2f})", fontsize=10)
axes[0].legend(fontsize=8)

# (b) direction vs speed unique contributions
axes[1].scatter(dll_spd[rate_ok], dll_dir[rate_ok], s=12, alpha=0.6, color="slateblue", edgecolor="none")
lim = np.nanpercentile(np.abs(np.concatenate([dll_spd[rate_ok], dll_dir[rate_ok]])), 98)
axes[1].plot([-lim, lim], [-lim, lim], color="k", ls="--", lw=1)
axes[1].axhline(0, color="gray", lw=0.5); axes[1].axvline(0, color="gray", lw=0.5)
axes[1].set_xlabel("unique speed contribution (Δ log-likelihood)")
axes[1].set_ylabel("unique direction contribution (Δ log-likelihood)")
n_dir = (dll_dir[rate_ok] > dll_spd[rate_ok]).sum()
axes[1].set_title(f"Direction vs speed encoding per unit (rate ≥ 2 Hz)\n(n={n_dir}/{rate_ok.sum()} units: direction > speed)", fontsize=10)

# (c) model comparison
meds = [np.median(pr2_dir[rate_ok]), np.median(pr2_spd[rate_ok]), np.median(pr2_full[rate_ok])]
q25 = [np.percentile(pr2_dir[rate_ok], 25), np.percentile(pr2_spd[rate_ok], 25), np.percentile(pr2_full[rate_ok], 25)]
q75 = [np.percentile(pr2_dir[rate_ok], 75), np.percentile(pr2_spd[rate_ok], 75), np.percentile(pr2_full[rate_ok], 75)]
axes[2].bar(["direction\nonly", "speed\nonly", "direction\n+ speed"], meds,
            yerr=[np.array(meds) - np.array(q25), np.array(q75) - np.array(meds)],
            color=["steelblue", "teal", "crimson"], edgecolor="k", lw=0.5, capsize=4)
axes[2].set_ylabel("held-out pseudo-R² (median, IQR)")
axes[2].set_title("Poisson GLM encoding of kinematics\n(10 ms bins, movement epochs)", fontsize=10)

plt.tight_layout()
plt.savefig("fig7_glm_encoding.png", dpi=150)
print("saved fig7_glm_encoding.png")
