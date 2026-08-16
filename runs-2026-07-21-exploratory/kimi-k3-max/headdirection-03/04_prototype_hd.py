"""Prototype head-direction analysis on one session of DANDI 000056.

Steps:
1. Load session, compute head direction from dual LEDs.
2. Visualize raw tracking + HD angle.
3. Compute wake tuning curves for all units.
4. Identify HD cells via mean vector length + circular time-shift shuffle.
5. Visualize tuning curves and population stats.
"""
import h5py
from pynwb import NWBHDF5IO
import remfile
import pynapple as nap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ASSET_ID = "4cc64fe0-7b1e-404c-8b86-fb5659292830"  # sub-Mouse17_ses-Mouse17-130128
URL = f"https://api.dandiarchive.org/api/assets/{ASSET_ID}/download/"
FIGDIR = "figs_proto"

import os
os.makedirs(FIGDIR, exist_ok=True)

# ---------------------------------------------------------------- load
disk_cache = remfile.DiskCache("/tmp/remfile_cache_hd")
rem_file = remfile.File(URL, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)

units = nwb["units"]
states = nwb["states"]
red = nwb["SubjectPosition/RedLED"]
blue = nwb["SubjectPosition/BlueLED"]

# ------------------------------------------------- head direction signal
# Tracking failures are marked with -1 sentinel values.
valid = (red.values[:, 0] > 0) & (red.values[:, 1] > 0) & \
        (blue.values[:, 0] > 0) & (blue.values[:, 1] > 0)
print(f"valid tracking fraction: {valid.mean():.3f}")

dx = red.values[:, 0] - blue.values[:, 0]
dy = red.values[:, 1] - blue.values[:, 1]
hd_angle = np.arctan2(dy, dx) % (2 * np.pi)
hd_angle[~valid] = np.nan

hd = nap.Tsd(t=red.t, d=hd_angle)

# epochs
wake = states[states["label"] == "Awake"]
rem = states[states["label"] == "REM"]
nrem = states[states["label"] == "Non-REM"]
print(f"wake: {wake.tot_length():.0f}s in {len(wake)} epochs | "
      f"REM: {rem.tot_length():.0f}s in {len(rem)} epochs | "
      f"NREM: {nrem.tot_length():.0f}s in {len(nrem)} epochs")

# ---------------------------------------------------------------- figure 1: raw data
fig, axes = plt.subplots(3, 1, figsize=(12, 9), constrained_layout=True)

# LED trajectory over wake
ax = axes[0]
w0 = wake.loc[0, "start"] if hasattr(wake, "loc") else wake["start"][0]
ax.plot(red.values[valid, 0], red.values[valid, 1], ".", ms=0.5, color="0.7",
        label="Red LED")
ax.plot(blue.values[valid, 0], blue.values[valid, 1], ".", ms=0.5, color="0.3",
        label="Blue LED")
ax.set_xlabel("x (a.u.)")
ax.set_ylabel("y (a.u.)")
ax.set_title("LED positions over the session (valid samples)")
ax.legend(markerscale=8)
ax.set_aspect("equal")

# HD angle over a 60 s wake snippet
ax = axes[1]
t0, t1 = wake["start"][0], wake["start"][0] + 60
snippet = hd.get(t0, t1)
ax.plot(snippet.t, snippet.d, ".", ms=2)
ax.set_xlabel("time (s)")
ax.set_ylabel("head direction (rad)")
ax.set_title(f"Head direction, first 60 s of wake ({t0:.0f}-{t1:.0f} s)")
ax.set_ylim(-0.2, 2 * np.pi + 0.2)
ax.set_yticks([0, np.pi, 2 * np.pi], ["0", "$\\pi$", "$2\\pi$"])

# occupancy of HD during wake
ax = axes[2]
hd_wake = hd.restrict(wake)
occ, edges = np.histogram(hd_wake.values[~np.isnan(hd_wake.values)],
                          bins=60, range=(0, 2 * np.pi))
centers = (edges[:-1] + edges[1:]) / 2
ax.bar(centers, occ / 39.0625, width=2 * np.pi / 60)
ax.set_xlabel("head direction (rad)")
ax.set_ylabel("occupancy (s)")
ax.set_title("Head-direction occupancy during wake")
ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi],
              ["0", "$\\pi/2$", "$\\pi$", "$3\\pi/2$", "$2\\pi$"])

fig.savefig(f"{FIGDIR}/fig1_raw_head_direction.png", dpi=150)
plt.close(fig)
print("saved fig1")

# ---------------------------------------------------------------- tuning curves
# filter near-silent units
rates = units.metadata["rate"].values
keep = rates > 0.1
print(f"keeping {keep.sum()}/{len(keep)} units with rate > 0.1 Hz")
units_f = units[keep]

tuning = nap.compute_tuning_curves(units_f, hd, bins=60, range=(0, 2 * np.pi),
                                   epochs=wake, feature_names=["hd"])
print(tuning)

# ---------------------------------------------------------------- vector length + shuffle
def mean_vector_length(angles):
    angles = angles[~np.isnan(angles)]
    if len(angles) == 0:
        return np.nan, np.nan
    z = np.exp(1j * angles).mean()
    return np.abs(z), np.angle(z) % (2 * np.pi)

hd_w = hd.restrict(wake)
mvl = np.zeros(len(units_f))
pref = np.zeros(len(units_f))
for i, u in enumerate(units_f.keys()):
    ang = units_f[u].value_from(hd_w)
    mvl[i], pref[i] = mean_vector_length(np.asarray(ang))

# circular time-shift shuffle
rng = np.random.default_rng(42)
n_shuf = 200
w_start, w_end = wake["start"][0], wake["end"][-1]
span = w_end - w_start
shuf_mvl = np.zeros((len(units_f), n_shuf))
hd_full = hd  # use full-session HD so shifted spikes still land on valid samples
for s in range(n_shuf):
    shift = rng.uniform(20, span - 20)
    for i, u in enumerate(units_f.keys()):
        sp = units_f[u].restrict(wake)
        shifted = (sp.t - w_start + shift) % span + w_start
        ang = nap.Ts(shifted).value_from(hd_full)
        shuf_mvl[i, s], _ = mean_vector_length(np.asarray(ang))

thresh = np.percentile(shuf_mvl, 95, axis=1)
is_hd = mvl > thresh
print(f"HD cells: {is_hd.sum()}/{len(is_hd)} (per-unit shuffle p<0.05)")
print("MVL:", np.round(mvl, 3))
print("thresh:", np.round(thresh, 3))

np.savez(f"{FIGDIR}/proto_results.npz", mvl=mvl, pref=pref, thresh=thresh,
         is_hd=is_hd, rates=rates[keep])

# ---------------------------------------------------------------- figure 2: MVL distribution
fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
ax.scatter(mvl[~is_hd], thresh[~is_hd], c="0.6", s=20, label="not HD")
ax.scatter(mvl[is_hd], thresh[is_hd], c="crimson", s=20, label="HD (p<0.05)")
lim = [0, max(mvl.max(), thresh.max()) * 1.1]
ax.plot(lim, lim, "k--", lw=1)
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_xlabel("observed mean vector length")
ax.set_ylabel("95th percentile of shuffled MVL")
ax.set_title(f"HD cell identification, Mouse17-130128 ({is_hd.sum()}/{len(is_hd)} HD)")
ax.legend()
fig.savefig(f"{FIGDIR}/fig2_mvl_vs_shuffle.png", dpi=150)
plt.close(fig)
print("saved fig2")

# ---------------------------------------------------------------- figure 3: tuning curves
order = np.argsort(-mvl)
n_show = min(24, len(order))
ncol = 6
nrow = int(np.ceil(n_show / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(2.2 * ncol, 2.2 * nrow),
                         subplot_kw=dict(projection="polar"))
axes = np.atleast_1d(axes).ravel()
for k in range(n_show):
    u_idx = order[k]
    ax = axes[k]
    tc = tuning[u_idx].values
    theta = tuning.coords["hd"].values
    theta_c = np.concatenate([theta, [theta[0] + 2 * np.pi]])
    tc_c = np.concatenate([tc, [tc[0]]])
    ax.plot(theta_c, tc_c, color="crimson" if is_hd[u_idx] else "0.5")
    ax.fill(theta_c, tc_c, color="crimson" if is_hd[u_idx] else "0.5", alpha=0.3)
    ax.set_title(f"u{u_idx} MVL={mvl[u_idx]:.2f}", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
for k in range(n_show, len(axes)):
    axes[k].axis("off")
fig.suptitle("Wake HD tuning curves (sorted by vector length; red = significant)")
fig.savefig(f"{FIGDIR}/fig3_tuning_curves.png", dpi=150)
plt.close(fig)
print("saved fig3")
