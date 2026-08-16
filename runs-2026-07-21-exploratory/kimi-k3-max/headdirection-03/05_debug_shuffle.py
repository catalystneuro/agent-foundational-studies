"""Debug the shuffle: why are shuffled MVLs so high?"""
import h5py
from pynwb import NWBHDF5IO
import remfile
import pynapple as nap
import numpy as np

ASSET_ID = "4cc64fe0-7b1e-404c-8b86-fb5659292830"
URL = f"https://api.dandiarchive.org/api/assets/{ASSET_ID}/download/"

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

valid = (red.values[:, 0] > 0) & (red.values[:, 1] > 0) & \
        (blue.values[:, 0] > 0) & (blue.values[:, 1] > 0)
dx = red.values[:, 0] - blue.values[:, 0]
dy = red.values[:, 1] - blue.values[:, 1]
hd_angle = np.arctan2(dy, dx) % (2 * np.pi)
hd_angle[~valid] = np.nan
hd = nap.Tsd(t=red.t, d=hd_angle)

wake = states[states["label"] == "Awake"]
w_start, w_end = wake["start"][0], wake["end"][-1]
span = w_end - w_start
print("wake span:", w_start, w_end)

# one strong HD unit: original index 16 -> in filtered list it's unit key 17?
rates = units.metadata["rate"].values
keep = rates > 0.1
units_f = units[keep]
keys = list(units_f.keys())
print("keys:", keys)

u = keys[15]  # the one with observed MVL 0.669
sp = units_f[u].restrict(wake)
print("n spikes wake:", len(sp))

# observed
ang_obs = np.asarray(sp.value_from(hd.restrict(wake)))
print("observed MVL:", np.abs(np.exp(1j * ang_obs[~np.isnan(ang_obs)]).mean()))

# manual np.interp version
hd_t = hd.t
hd_d = hd.values
def mvl_from_times(times):
    a = np.interp(times, hd_t, hd_d, left=np.nan, right=np.nan)
    a = a[~np.isnan(a)]
    return np.abs(np.exp(1j * a).mean())

print("observed MVL (manual interp):", mvl_from_times(sp.t))

rng = np.random.default_rng(0)
for trial in range(5):
    shift = rng.uniform(20, span - 20)
    shifted = (sp.t - w_start + shift) % span + w_start
    # pynapple way
    a1 = np.asarray(nap.Ts(shifted).value_from(hd))
    m1 = np.abs(np.exp(1j * a1[~np.isnan(a1)]).mean())
    # manual way
    m2 = mvl_from_times(shifted)
    # sorted version
    m3 = mvl_from_times(np.sort(shifted))
    print(f"shift={shift:.0f} pynapple={m1:.3f} manual={m2:.3f} manual_sorted={m3:.3f}")

# check: is nap.Ts sorting?
shifted = (sp.t - w_start + 1000) % span + w_start
ts_shifted = nap.Ts(shifted)
print("Ts sorted?", np.all(np.diff(ts_shifted.t) >= 0), "| input sorted?", np.all(np.diff(shifted) >= 0))
