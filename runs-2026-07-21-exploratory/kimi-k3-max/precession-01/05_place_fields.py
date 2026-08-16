"""Step 5: run epochs (speed + on-track), per-direction tuning curves, place cell IDs."""
import numpy as np
import requests
import remfile
import h5py
from pynwb import NWBHDF5IO
import pynapple as nap
from scipy import signal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

ASSET_ID = "c0ac352b-9da5-44b0-b73d-41a9ee3c3b1d"
dl = f"https://api.dandiarchive.org/api/assets/{ASSET_ID}/download/"
r = requests.get(dl, allow_redirects=False)
s3_url = r.headers["Location"]

disk_cache = remfile.DiskCache("/tmp/remfile_cache_precession")
rem_file = remfile.File(s3_url, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)

units = nwb["units"]
pos2d = nwb["1.6mLinearMazeSpatialSeries"]
lin_ts = nwb["1.6mLinearMazeLinearizedTimeSeries"]

t_pos = pos2d.index.values
xy = np.asarray(pos2d.values)  # (n, 2) meters
lin = np.asarray(lin_ts.values)[:, 0]
dt = np.median(np.diff(t_pos))
print(f"position: {len(t_pos)} samples, dt={dt*1000:.1f} ms, "
      f"lin valid {np.mean(~np.isnan(lin))*100:.1f}%")

# Speed from 2D position (Gaussian-smoothed, sigma 0.25 s)
valid2d = ~np.isnan(xy[:, 0])
xyi = xy.copy()
for d in range(2):
    x1 = xy[:, d]
    x1[np.isnan(x1)] = np.interp(t_pos[np.isnan(x1)], t_pos[~np.isnan(x1)], x1[~np.isnan(x1)])
    xyi[:, d] = x1
kern = signal.windows.gaussian(int(0.5 / dt) // 2 * 2 + 1, std=int(0.25 / dt))
kern /= kern.sum()
vx = signal.convolve(np.gradient(xyi[:, 0], dt), kern, mode="same")
vy = signal.convolve(np.gradient(xyi[:, 1], dt), kern, mode="same")
speed = np.sqrt(vx**2 + vy**2)
direction = np.sign(vx)  # +1: lin increasing, -1: decreasing

# Run epochs: speed > 0.1 m/s AND linearized position valid
run_mask = (speed > 0.10) & (~np.isnan(lin))
# fill short gaps (<0.5 s), drop short bouts (<0.5 s)
def mask_to_epochs(mask, t, min_dur=0.5, max_gap=0.5):
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return np.zeros((0, 2))
    splits = np.where(np.diff(idx) > 1)[0]
    starts = np.concatenate([[idx[0]], idx[splits + 1]])
    ends = np.concatenate([idx[splits], [idx[-1]]])
    eps = np.stack([t[starts], t[ends]], axis=1)
    # merge gaps < max_gap
    merged = [eps[0]]
    for e in eps[1:]:
        if e[0] - merged[-1][1] < max_gap:
            merged[-1][1] = e[1]
        else:
            merged.append(e)
    eps = np.array(merged)
    eps = eps[(eps[:, 1] - eps[:, 0]) >= min_dur]
    return eps

run_eps_arr = mask_to_epochs(run_mask, t_pos)
run_epochs = nap.IntervalSet(run_eps_arr[:, 0], run_eps_arr[:, 1])
print(f"run epochs: {len(run_eps_arr)} bouts, total {np.sum(run_eps_arr[:,1]-run_eps_arr[:,0]):.1f} s")

# Direction-specific run epochs (>=80% of samples in one direction)
dir_epochs = {+1: [], -1: []}
for s, e in run_eps_arr:
    m = (t_pos >= s) & (t_pos <= e)
    frac_pos = np.mean(direction[m] > 0)
    if frac_pos >= 0.8:
        dir_epochs[+1].append([s, e])
    elif frac_pos <= 0.2:
        dir_epochs[-1].append([s, e])
for d in dir_epochs:
    arr = np.array(dir_epochs[d]) if dir_epochs[d] else np.zeros((0, 2))
    dir_epochs[d] = nap.IntervalSet(arr[:, 0], arr[:, 1]) if len(arr) else nap.IntervalSet([], [])
    print(f"direction {d:+d}: {len(arr)} bouts, "
          f"{np.sum(arr[:,1]-arr[:,0]) if len(arr) else 0:.1f} s")

# Tuning curves per direction (50 bins over 0-1.6 m)
NBINS = 50
exc_keys = [k for k in units.keys() if units.get_info("cell_type")[k] == "excitatory"]
print(f"{len(exc_keys)} excitatory units")
lin_tsd = nap.Tsd(t=t_pos, d=lin)

tuning = {}
bin_centers = None
for d, lab in [(+1, "pos"), (-1, "neg")]:
    ep = dir_epochs[d]
    tc = nap.compute_tuning_curves(units[exc_keys], lin_tsd, bins=NBINS,
                                   range=(0, 1.6), epochs=ep)
    # xarray DataArray, dims (unit, bin); bin centers in the second coordinate
    bin_centers = tc.coords[tc.dims[1]].values
    # smooth with 1.5-bin Gaussian along the position axis
    sm = signal.windows.gaussian(9, std=1.5)
    sm /= sm.sum()
    tc_smooth = np.apply_along_axis(lambda r: signal.convolve(r, sm, mode="same"), 1, tc.values)
    tuning[lab] = dict(bins=bin_centers, raw=tc.values, smooth=tc_smooth)

# Place cell criteria: peak rate > 1 Hz in at least one direction
place_info = {}
for i, k in enumerate(exc_keys):
    pk_pos = tuning["pos"]["smooth"][i, :].max()
    pk_neg = tuning["neg"]["smooth"][i, :].max()
    pref = "pos" if pk_pos >= pk_neg else "neg"
    peak = max(pk_pos, pk_neg)
    if peak >= 1.0:
        tc_pref = tuning[pref]["smooth"][i, :]
        peak_bin = int(np.argmax(tc_pref))
        thr = 0.2 * peak
        above = tc_pref >= thr
        # contiguous field around peak bin
        lo = peak_bin
        while lo > 0 and above[lo - 1]:
            lo -= 1
        hi = peak_bin
        while hi < NBINS - 1 and above[hi + 1]:
            hi += 1
        place_info[k] = dict(pref_dir=pref, peak_rate=peak, peak_pos=bin_centers[peak_bin],
                             field_lo=bin_centers[lo], field_hi=bin_centers[hi],
                             peak_bin=peak_bin, unit_index=i)
print(f"place cells (peak >= 1 Hz): {len(place_info)} / {len(exc_keys)}")

np.savez("place_fields.npz",
         bin_centers=bin_centers,
         tc_pos=tuning["pos"]["smooth"], tc_neg=tuning["neg"]["smooth"],
         exc_keys=np.array(exc_keys),
         run_eps=run_eps_arr,
         dir_pos=np.array(dir_epochs[+1].values if len(dir_epochs[+1]) else np.zeros((0,2))),
         dir_neg=np.array(dir_epochs[-1].values if len(dir_epochs[-1]) else np.zeros((0,2))),
         place_keys=np.array(list(place_info.keys())),
         place_pref=np.array([place_info[k]["pref_dir"] for k in place_info]),
         place_peak=np.array([place_info[k]["peak_pos"] for k in place_info]),
         place_lo=np.array([place_info[k]["field_lo"] for k in place_info]),
         place_hi=np.array([place_info[k]["field_hi"] for k in place_info]),
         place_rate=np.array([place_info[k]["peak_rate"] for k in place_info]))

# ---- figure: sorted population rate maps + position/speed example ----
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
pkeys = np.array(list(place_info.keys()))
for ax, lab, title in [(axes[0], "pos", "+ direction runs"), (axes[1], "neg", "- direction runs")]:
    peaks = [bin_centers[np.argmax(tuning[lab]["smooth"][exc_keys.index(k), :])] for k in pkeys]
    pkeys_sorted = pkeys[np.argsort(peaks)]
    mat = []
    for k in pkeys_sorted:
        i = exc_keys.index(k)
        r_ = tuning[lab]["smooth"][i, :]
        mat.append(r_ / r_.max() if r_.max() > 0 else r_ * 0)
    mat = np.array(mat)
    im = ax.imshow(mat, aspect="auto", cmap="viridis",
                   extent=[0, 1.6, len(pkeys), 0])
    ax.set_xlabel("linearized position (m)")
    ax.set_ylabel("place cell (sorted)")
    ax.set_title(f"Rate maps, {title}")
    fig.colorbar(im, ax=ax, label="norm. rate")

# position + speed snippet
snip = (t_pos >= 19000) & (t_pos <= 19060)
ax = axes[2]
ax.plot(t_pos[snip] - 19000, lin[snip], color="k", lw=1, label="linearized pos")
ax.set_xlabel("time (s)")
ax.set_ylabel("linearized position (m)")
ax2 = ax.twinx()
ax2.plot(t_pos[snip] - 19000, speed[snip] * 100, color="C3", lw=0.8, alpha=0.7, label="speed")
ax2.set_ylabel("speed (cm/s)", color="C3")
ax2.axhline(10, color="C3", ls=":", lw=0.8)
ax.set_title("Position and speed (60 s snippet)")
fig.tight_layout()
fig.savefig("fig_place_fields.png", dpi=150)
print("saved fig_place_fields.png")
