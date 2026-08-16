"""Verify wake-concatenated circular shift shuffle gives sensible null MVLs."""
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
hd_angle = np.arctan2(red.values[:, 0] - blue.values[:, 0],
                      red.values[:, 1] - blue.values[:, 1]) % (2 * np.pi)
hd_angle[~valid] = np.nan
hd = nap.Tsd(t=red.t, d=hd_angle)
wake = states[states["label"] == "Awake"]

# --- wake-concatenated timeline mapping ---
starts = np.asarray(wake["start"])
ends = np.asarray(wake["end"])
durs = ends - starts
cum = np.concatenate([[0], np.cumsum(durs)])  # length n_epochs+1
total_wake = cum[-1]

def to_wake_time(t):
    """Map absolute times inside wake epochs to concatenated wake time."""
    ep = np.searchsorted(ends, t, side="right")  # epoch index for each t
    ep = np.clip(ep, 0, len(durs) - 1)
    return cum[ep] + (t - starts[ep])

def from_wake_time(w):
    """Map concatenated wake time back to absolute times."""
    ep = np.searchsorted(cum[1:], w, side="right")
    ep = np.clip(ep, 0, len(durs) - 1)
    return starts[ep] + (w - cum[ep])

rates = units.metadata["rate"].values
units_f = units[rates > 0.1]
keys = list(units_f.keys())

hd_t, hd_d = hd.t, hd.values

def mvl_at(times):
    a = np.interp(times, hd_t, hd_d, left=np.nan, right=np.nan)
    a = a[~np.isnan(a)]
    return np.abs(np.exp(1j * a).mean()) if len(a) else np.nan

rng = np.random.default_rng(0)
for u in [keys[15], keys[16], keys[0], keys[3]]:
    sp = units_f[u].restrict(wake)
    w_sp = to_wake_time(sp.t)
    obs = mvl_at(sp.t)
    shufs = []
    for s in range(50):
        shift = rng.uniform(0, total_wake)
        t_sh = from_wake_time((w_sp + shift) % total_wake)
        shufs.append(mvl_at(t_sh))
    shufs = np.array(shufs)
    print(f"unit {u}: n={len(sp)} obs={obs:.3f} shuf95={np.percentile(shufs, 95):.3f} "
          f"shuf_mean={shufs.mean():.3f}")
