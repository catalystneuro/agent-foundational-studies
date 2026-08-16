# Figures: behavior overview + raw data segment
import h5py
from pynwb import NWBHDF5IO
import remfile
import pynapple as nap
import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("analysis_results.pkl", "rb") as f:
    R = pickle.load(f)
trials = R["trials"]

s3 = "https://api.dandiarchive.org/api/assets/26e85f09-39b7-480f-b337-278a8f034007/download/"
rf = remfile.File(s3, disk_cache=remfile.DiskCache("/tmp/remfile_cache_mcmaze"))
f5 = h5py.File(rf, "r")
io = NWBHDF5IO(file=f5)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)
units = nwb["units"]
hp = nwb["hand_pos"]
hv = nwb["hand_vel"]

unit_ids = np.load("unit_ids.npy")
onset_t = trials["move_onset_time"].values
reach_dir = trials["reach_dir"].values

# ---------------- fig1: behavior ----------------
fig = plt.figure(figsize=(14, 4.5))
gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.3, 0.9], wspace=0.3)

# (a) trajectories colored by reach direction
axa = fig.add_subplot(gs[0, 0])
rng = np.random.default_rng(3)
idx = rng.choice(len(trials), size=150, replace=False)
for i in idx:
    seg = hp.get(onset_t[i] - 0.05, onset_t[i] + 0.55)
    axa.plot(seg[:, 0].values, seg[:, 1].values,
             color=plt.cm.hsv((reach_dir[i] + np.pi) / (2 * np.pi)), alpha=0.45, lw=0.8)
axa.set_xlabel("x (mm)")
axa.set_ylabel("y (mm)")
axa.set_title("Reach trajectories (n=150)\ncolored by reach direction", fontsize=10)
axa.set_aspect("equal")

# (b) speed profiles aligned to move onset
axb = fig.add_subplot(gs[0, 1])
for i in idx[:40]:
    seg = hv.get(onset_t[i] - 0.3, onset_t[i] + 0.7)
    spd = np.linalg.norm(seg.values, axis=1)
    axb.plot(seg.t - onset_t[i], spd, color="steelblue", alpha=0.25, lw=0.8)
# mean speed
segs = []
for i in idx:
    seg = hv.get(onset_t[i] - 0.3, onset_t[i] + 0.7)
    segs.append(np.linalg.norm(seg.values, axis=1))
segs = np.array(segs)
t_spd = seg.t - onset_t[idx[-1]]
axb.plot(t_spd, segs.mean(axis=0), color="k", lw=2, label="mean")
axb.axvline(0, color="k", ls="--", lw=1)
axb.set_xlabel("time from move onset (s)")
axb.set_ylabel("hand speed (mm/s)")
axb.set_title("Hand speed around movement onset\n(40 example trials + mean)", fontsize=10)
axb.legend(fontsize=8)

# (c) reach direction distribution (polar)
axc = fig.add_subplot(gs[0, 2], projection="polar")
bins = np.linspace(-np.pi, np.pi, 25)
cnt, _ = np.histogram(reach_dir, bins=bins)
th = (bins[:-1] + bins[1:]) / 2
axc.bar(th, cnt, width=np.diff(bins), color="seagreen", edgecolor="k", lw=0.4)
axc.set_theta_zero_location("E"); axc.set_theta_direction(1)
axc.set_title(f"Reach direction distribution\n(n={len(trials)} trials)", fontsize=10, pad=22)
axc.tick_params(labelsize=8)
axc.set_rlabel_position(250)
axc.tick_params(axis="y", labelsize=7)

fig.suptitle("MC_Maze behavior: delayed center-out reaching with virtual barriers (monkey Jenkins)", fontsize=12, y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.90])
plt.savefig("fig1_behavior.png", dpi=150, bbox_inches="tight")
print("saved fig1_behavior.png")

# ---------------- fig2: raw data segment ----------------
# choose a window containing two consecutive movements
i0 = np.searchsorted(onset_t, 1000.0)
t0 = onset_t[i0] - 1.2
t1 = t0 + 4.8
seg_v = hv.get(t0, t1)
spd = np.linalg.norm(seg_v.values, axis=1)

fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                         gridspec_kw=dict(height_ratios=[2.2, 1], hspace=0.08))
ax_r, ax_k = axes
n_show = 60
show_units = np.linspace(0, len(units) - 1, n_show).astype(int)
for row, u in enumerate(show_units):
    st = np.asarray(units[unit_ids[u]].get(t0, t1).t)
    ax_r.scatter(st, np.full(st.shape, row, dtype=float), s=3, color="k", marker="|")
for ot in onset_t:
    if t0 <= ot <= t1:
        ax_r.axvline(ot, color="crimson", ls="--", lw=1)
        ax_k.axvline(ot, color="crimson", ls="--", lw=1)
ax_k.plot([], [], color="crimson", ls="--", label="move onset")  # legend proxy
ax_r.set_ylabel("unit")
ax_r.set_title("Simultaneously recorded M1/PMd units (raster) and hand kinematics", fontsize=11)
ax_r.set_ylim(-0.5, n_show - 0.5)
ax_r.set_yticks([0, 29, 59])
ax_r.set_yticklabels([1, 30, 60])

ax_k.plot(seg_v.t, seg_v.values[:, 0], color="steelblue", lw=1, label="velocity x")
ax_k.plot(seg_v.t, seg_v.values[:, 1], color="darkorange", lw=1, label="velocity y")
ax_k.plot(seg_v.t, spd, color="k", lw=1.4, label="speed")
ax_k.set_xlabel("time (s)")
ax_k.set_ylabel("velocity / speed (mm/s)")
ax_k.legend(fontsize=8, loc="upper left")
plt.tight_layout()
plt.savefig("fig2_raw_data.png", dpi=150)
print("saved fig2_raw_data.png")
