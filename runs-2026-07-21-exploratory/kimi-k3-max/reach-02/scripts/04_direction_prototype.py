"""Prototype: per-trial reach direction from kinematics + directional PSTHs."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import h5py
import remfile
from pynwb import NWBHDF5IO
import pynapple as nap
from tqdm import tqdm

S3_MCMAZE = "https://dandiarchive.s3.amazonaws.com/blobs/df3/e3f/df3e3f73-50ab-42b4-8827-82664ddd474a"

disk_cache = remfile.DiskCache("cache/remfile_cache")
rem_file = remfile.File(S3_MCMAZE, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)

hand_pos = nwb["hand_pos"]
hand_vel = nwb["hand_vel"]
units = nwb["units"]
df = nwbfile.intervals["trials"].to_dataframe()

# Use all trials; compute reach direction from hand displacement
# between move onset and the time of peak speed (initial feedforward phase).
move_on = df["move_onset_time"].values
go_cue = df["go_cue_time"].values
t_start = df["start_time"].values
t_stop = df["stop_time"].values

pos = hand_pos.d  # (T,2) loads into RAM from cache
vel = hand_vel.d
ts = hand_pos.t

def sample_at(t):
    """index of nearest timestamp"""
    return np.searchsorted(ts, t)

reach_dir = np.full(len(df), np.nan)
peak_speed = np.full(len(df), np.nan)
move_dur = np.full(len(df), np.nan)
for i in range(len(df)):
    i0 = sample_at(move_on[i])
    i1 = sample_at(t_stop[i])
    if i1 <= i0 + 10:
        continue
    sp = np.linalg.norm(vel[i0:i1], axis=1)
    ipk = i0 + int(np.argmax(sp))
    disp = pos[ipk] - pos[i0]
    if np.linalg.norm(disp) < 1.0:
        continue
    reach_dir[i] = np.arctan2(disp[1], disp[0])
    peak_speed[i] = sp.max()
    move_dur[i] = t_stop[i] - move_on[i]

df["reach_dir"] = reach_dir
df["peak_speed"] = peak_speed
df["move_dur"] = move_dur
print("trials with valid reach dir:", np.isfinite(reach_dir).sum(), "/", len(df))
print("move duration median (ms):", np.nanmedian(move_dur) * 1000)
print("peak speed median (mm/s):", np.nanmedian(peak_speed))

deg = np.degrees(reach_dir)
straight = df["trial_version"].values == 0
print("straight trials:", straight.sum())

# --- figure: trajectories colored by direction (straight trials) ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
ax = axes[0]
cmap = plt.cm.hsv
idx = np.where(straight & np.isfinite(reach_dir))[0]
for i in idx[::3]:
    i0, i1 = sample_at(move_on[i]), sample_at(t_stop[i])
    c = cmap((reach_dir[i] + np.pi) / (2 * np.pi))
    ax.plot(pos[i0:i1, 0], pos[i0:i1, 1], color=c, lw=0.5, alpha=0.5)
ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
ax.set_title("Straight-trial hand trajectories\ncolored by reach direction")
ax.set_aspect("equal")

ax = axes[1]
ax.hist(deg[straight & np.isfinite(reach_dir)], bins=np.arange(-180, 181, 10))
ax.set_xlabel("reach direction (deg)"); ax.set_ylabel("trials")
ax.set_title("Reach direction distribution (straight trials)")
fig.tight_layout()
fig.savefig("figures/02_trajectories_by_direction.png", dpi=150)
print("saved figures/02_trajectories_by_direction.png")

# --- directional PSTH for a few example units ---
# bin reach direction into 8 sectors
edges = np.linspace(-np.pi, np.pi, 9)
sector = np.digitize(reach_dir, edges) - 1  # 0..7
sector[sector == 8] = 0

# pick 4 example units with high rates
rates = units.get_info("rate")
good = rates[rates > 3].index.values
print("units with rate>3 Hz:", len(good))

win = (-0.3, 0.6)  # relative to move onset
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for col, u in enumerate(good[:4]):
    spk = units[u]
    # raster aligned to move onset, sorted by sector
    ax = axes[0, col]
    order = np.argsort(sector)
    y = 0
    yticks, ylabels = [], []
    for s in range(8):
        trs = np.where(sector == s)[0]
        for i in trs:
            ep = nap.IntervalSet(move_on[i] + win[0], move_on[i] + win[1])
            st = spk.restrict(ep).t - move_on[i]
            ax.plot(st, np.full_like(st, y), "|", color=plt.cm.hsv(s / 8), ms=1.5)
            y += 1
        yticks.append(y - len(trs) / 2)
        ylabels.append(f"{int(np.degrees((edges[s]+edges[s+1])/2))}")
    ax.axvline(0, color="k", ls="--", lw=0.8)
    ax.set_yticks(yticks); ax.set_yticklabels(ylabels, fontsize=6)
    ax.set_ylabel("direction (deg)")
    ax.set_title(f"unit {u}")
    if col == 0:
        ax.set_xlabel("time from move onset (s)")

    # PSTH per sector
    ax = axes[1, col]
    bins = np.arange(win[0], win[1] + 0.02, 0.02)
    for s in range(8):
        trs = np.where(sector == s)[0]
        pe = nap.compute_perievent(spk, nap.Ts(move_on[trs]), window=win)
        rel = np.concatenate([pe[i].t for i in range(len(trs))])
        counts, _ = np.histogram(rel, bins=bins)
        fr = counts / (len(trs) * 0.02)
        ax.plot(bins[:-1] + 0.01, fr, color=plt.cm.hsv(s / 8), lw=1)
    ax.axvline(0, color="k", ls="--", lw=0.8)
    ax.set_xlabel("time from move onset (s)")
    ax.set_ylabel("rate (Hz)")
fig.suptitle("Directional responses aligned to movement onset (4 example units)", y=1.0)
fig.tight_layout()
fig.savefig("figures/03_directional_psth_examples.png", dpi=150)
print("saved figures/03_directional_psth_examples.png")
