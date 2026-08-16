"""Step 4: figures for raw data validation and place fields."""
import pickle
import numpy as np
import h5py
import remfile
import requests
from pynwb import NWBHDF5IO
import pynapple as nap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 120, "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False})

dl = "https://api.dandiarchive.org/api/assets/c0ac352b-9da5-44b0-b73d-41a9ee3c3b1d/download/"
s3_url = requests.get(dl, allow_redirects=False).headers["Location"]
h5py_file = h5py.File(remfile.File(s3_url, disk_cache=remfile.DiskCache("/tmp/remfile_cache")), "r")
nwb = nap.NWBFile(NWBHDF5IO(file=h5py_file).read())
units = nwb["units"]
lin = nwb["1.6mLinearMazeLinearizedTimeSeries"]
t_pos = lin.t
x = np.asarray(lin.values, dtype=float).ravel()

with open("place_fields.pkl", "rb") as f:
    pf = pickle.load(f)
results, place_keys = pf["results"], pf["place_keys"]
edges, centers = pf["edges"], pf["centers"]
run_bouts = pf["run_bouts"]

# ---------- fig 1: raw data overview ----------
fig, axes = plt.subplots(3, 1, figsize=(9, 6.5), height_ratios=[1.2, 1, 1])

ax = axes[0]
tt = t_pos - t_pos[0]
ax.plot(tt, x, lw=0.3, color="0.6", rasterized=True)
for s, e in run_bouts:
    ax.axvspan(s - t_pos[0], e - t_pos[0], color="tab:red", alpha=0.15, lw=0)
ax.set_xlim(0, tt[-1])
ax.set_xlabel("time in maze epoch (s)")
ax.set_ylabel("linearized position (m)")
ax.set_title("Linearized position on the 1.6 m track (red = run bouts used for place fields)")

ax = axes[1]
# raster of place cells during 3 example laps, sorted by field location
peak_pos = {k: centers[np.argmax(results[k]["rate_map"])] for k in place_keys}
sorted_keys = sorted(place_keys, key=lambda k: peak_pos[k])
s0, e0 = run_bouts[len(run_bouts) // 2]
for i, k in enumerate(sorted_keys):
    spk = units[k].t
    spk = spk[(spk >= s0) & (spk <= e0)]
    ax.plot(spk - s0, np.full_like(spk, i), "|", ms=2, color="k", rasterized=True)
ax.set_xlim(0, e0 - s0)
ax.set_xlabel("time in example run bout (s)")
ax.set_ylabel("place cell (sorted by field)")
ax.set_title("Spike raster of 87 place cells during one example run bout")

ax = axes[2]
# position vs spikes for that bout: scatter spike positions
i0, i1 = np.searchsorted(t_pos, s0), np.searchsorted(t_pos, e0)
ax.plot(t_pos[i0:i1] - s0, x[i0:i1], color="0.6", lw=1)
for i, k in enumerate(sorted_keys):
    spk = units[k].t
    spk = spk[(spk >= s0) & (spk <= e0)]
    if len(spk) == 0:
        continue
    idx = np.clip(np.searchsorted(t_pos, spk), 0, len(x) - 1)
    ax.plot(spk - s0, x[idx], ".", ms=1.5, rasterized=True)
ax.set_xlim(0, e0 - s0)
ax.set_xlabel("time in example run bout (s)")
ax.set_ylabel("position (m)")
ax.set_title("Trajectory with spikes (all place cells)")
fig.tight_layout()
fig.savefig("fig1_raw_data_overview.png", dpi=150)
plt.close(fig)
print("fig1 saved")

# ---------- fig 2: place fields ----------
fig = plt.figure(figsize=(10, 7))
gs = fig.add_gridspec(2, 2, height_ratios=[1.4, 1], hspace=0.35, wspace=0.25)

ax = fig.add_subplot(gs[0, :])
maps = np.array([results[k]["rate_map"] for k in sorted_keys])
maps_n = maps / maps.max(axis=1, keepdims=True)
ax.imshow(maps_n, aspect="auto", cmap="viridis",
          extent=[0, 1.6, 0, len(sorted_keys)], origin="lower")
ax.set_xlabel("position on track (m)")
ax.set_ylabel("place cell (sorted by peak)")
ax.set_title(f"Place fields of {len(place_keys)} place cells (direction-pooled, normalized)")

ax = fig.add_subplot(gs[1, 0])
si_all = [results[k]["si"] for k in results]
si_pl = [results[k]["si"] for k in place_keys]
ax.hist(si_all, bins=30, alpha=0.6, label="all excitatory", color="0.5")
ax.hist(si_pl, bins=30, alpha=0.8, label="place cells", color="tab:blue")
ax.set_xlabel("Skaggs spatial information (bits/spike)")
ax.set_ylabel("count")
ax.legend(frameon=False)
ax.set_title("Spatial information")

ax = fig.add_subplot(gs[1, 1])
# example SI null distributions for 3 cells
ex = sorted_keys[len(sorted_keys)//4::len(sorted_keys)//3][:3]
for color, k in zip(["tab:blue", "tab:orange", "tab:green"], ex):
    ax.hist(results[k]["si_null"], bins=40, histtype="step", density=True,
            label=f"cell {k} null", color=color)
    ax.axvline(results[k]["si"], color=color, ls="--", lw=1)
ax.set_xlabel("Skaggs SI (bits/spike)")
ax.set_ylabel("density")
ax.set_title("Example SI shuffle nulls (dashed = observed)")
ax.legend(frameon=False, fontsize=7)
fig.savefig("fig2_place_fields.png", dpi=150)
plt.close(fig)
print("fig2 saved")
