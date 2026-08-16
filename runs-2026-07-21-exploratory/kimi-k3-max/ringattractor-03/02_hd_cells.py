"""Prototype: HD angle, tuning curves, HD-cell identification on Mouse28-140310."""
import h5py
import remfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pynwb import NWBHDF5IO
import pynapple as nap
from tqdm import tqdm

ASSETS = {
    "Mouse17-130128": "4cc64fe0-7b1e-404c-8b86-fb5659292830",
    "Mouse20-130514": "748aa5de-c0de-4aa7-a7ef-2aad2f87a7eb",
    "Mouse24-131213": "ada02790-6eb6-48ee-902d-9ba017303586",
    "Mouse25-140123": "bdb30f7d-ba69-4d2e-8504-2efab69cd8d7",
    "Mouse28-140310": "656704ea-a4cd-40f4-8158-a6533ebf2eee",
}

def load_session(asset_id, cache_dir="/tmp/remfile_cache_hd"):
    url = f"https://api.dandiarchive.org/api/assets/{asset_id}/download/"
    disk_cache = remfile.DiskCache(cache_dir)
    rem_file = remfile.File(url, disk_cache=disk_cache)
    h5py_file = h5py.File(rem_file, "r")
    io = NWBHDF5IO(file=h5py_file)
    nwbfile = io.read()
    return nap.NWBFile(nwbfile)

def compute_hd(nwb):
    """Head direction from dual LEDs; -1 sentinels -> NaN."""
    red = nwb["SubjectPosition/RedLED"]
    blue = nwb["SubjectPosition/BlueLED"]
    rv = np.asarray(red.values)
    bv = np.asarray(blue.values)
    valid = (rv[:, 0] > 0) & (rv[:, 1] > 0) & (bv[:, 0] > 0) & (bv[:, 1] > 0)
    ang = np.full(rv.shape[0], np.nan)
    ang[valid] = np.arctan2(rv[valid, 1] - bv[valid, 1], rv[valid, 0] - bv[valid, 0]) % (2 * np.pi)
    return nap.Tsd(t=red.t, d=ang, time_support=red.time_support)

def identify_hd_cells(units, hd, wake, n_shuffle=1000, mvl_floor=0.3, alpha=0.05, seed=0):
    """MVL of spike angles + random-time resampling null."""
    rng = np.random.default_rng(seed)
    # valid wake HD sample times
    hd_w = hd.restrict(wake)
    t_valid = hd_w.t[~np.isnan(hd_w.d)]
    a_valid = hd_w.d[~np.isnan(hd_w.d)]
    mvl, pval = {}, {}
    for k in units.keys():
        spk = units[k].restrict(wake)
        ang = spk.value_from(hd)  # ts.value_from(tsd)
        ang = ang[~np.isnan(ang)]
        n = len(ang)
        if n < 100:
            continue
        if n > 100_000:  # memory guard
            ang = ang[rng.choice(n, 100_000, replace=False)]
            n = 100_000
        obs = np.abs(np.mean(np.exp(1j * ang)))
        # null: random times drawn from valid wake HD samples
        null = np.empty(n_shuffle)
        chunk = 100
        for c in range(0, n_shuffle, chunk):
            idx = rng.integers(0, len(t_valid), size=(chunk, n))
            null[c:c + chunk] = np.abs(np.mean(np.exp(1j * a_valid[idx]), axis=1))
        mvl[k] = obs
        pval[k] = (np.sum(null >= obs) + 1) / (n_shuffle + 1)
    hd_cells = [k for k in mvl if pval[k] < alpha and mvl[k] > mvl_floor]
    return mvl, pval, hd_cells

nwb = load_session(ASSETS["Mouse28-140310"])
units = nwb["units"]
states = nwb["states"]
wake = states[states["label"] == "Awake"]

hd = compute_hd(nwb)
print("HD valid fraction (wake):", np.mean(~np.isnan(hd.restrict(wake).d)))

# drop zero-rate units
good = [k for k in units.keys() if units[k].rate > 0]
units = units[good]
print(f"{len(units)} units with rate > 0")

mvl, pval, hd_cells = identify_hd_cells(units, hd, wake)
print(f"\nHD cells ({len(hd_cells)}):", hd_cells)
for k in hd_cells:
    print(f"  unit {k}: MVL={mvl[k]:.3f} p={pval[k]:.4f}")

# tuning curves
tc = nap.compute_tuning_curves(units, hd, bins=60, range=(0, 2 * np.pi),
                               epochs=wake, return_counts=True)
occ = tc.attrs["occupancy"] / tc.attrs["fs"]
tc_rate = tc / np.where(occ == 0, np.nan, occ)
bin_centers = tc.coords[tc.dims[1]].values

prefs = {}
for k in hd_cells:
    i = list(units.keys()).index(k)
    v = tc_rate.values[i]
    prefs[k] = bin_centers[np.nanargmax(v)]
print("\npreferred directions (rad):", {k: round(v, 2) for k, v in prefs.items()})

np.savez("/tmp/hd_proto_mouse28.npz", hd_cells=hd_cells, prefs=prefs,
         mvl=mvl, pval=pval, bin_centers=bin_centers, tc=tc_rate.values,
         unit_keys=list(units.keys()))

# ---- validation figure ----
fig = plt.figure(figsize=(14, 8))
gs = fig.add_gridspec(3, 6, hspace=0.6, wspace=0.8)
# HD trace snippet
ax = fig.add_subplot(gs[0, :4])
hdw = hd.restrict(wake)
m = (hdw.t > 17195) & (hdw.t < 17255)
ax.plot(hdw.t[m], hdw.d[m], lw=0.8)
ax.set_title("Head direction (60 s of wake)")
ax.set_ylabel("HD (rad)")
ax.set_yticks([0, np.pi, 2 * np.pi], ["0", "π", "2π"])
# occupancy
ax2 = fig.add_subplot(gs[0, 4:])
ax2.plot(bin_centers, occ, lw=1)
ax2.set_title("Wake HD occupancy (s)")
ax2.set_xticks([0, np.pi, 2 * np.pi], ["0", "π", "2π"])
# polar tuning curves for HD cells
for i, k in enumerate(hd_cells[:12]):
    axp = fig.add_subplot(gs[1 + i // 6, i % 6], projection="polar")
    ii = list(units.keys()).index(k)
    v = tc_rate.values[ii]
    axp.plot(np.append(bin_centers, bin_centers[0]), np.append(v, v[0]), lw=1.2)
    axp.set_title(f"u{k} MVL={mvl[k]:.2f}", fontsize=8)
    axp.set_xticks([])
    axp.set_yticks([])
fig.suptitle("Mouse28-140310: HD cells")
fig.savefig("fig_proto_hd_cells.png", dpi=150)
print("\nsaved fig_proto_hd_cells.png")
