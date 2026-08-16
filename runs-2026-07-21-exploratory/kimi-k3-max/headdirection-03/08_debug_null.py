"""Verify random-time resampling null for HD significance."""
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
hd_angle = np.arctan2(red.values[:, 1] - blue.values[:, 1],
                      red.values[:, 0] - blue.values[:, 0]) % (2 * np.pi)
hd_angle[~valid] = np.nan
hd = nap.Tsd(t=red.t, d=hd_angle)
wake = states[states["label"] == "Awake"]

# valid wake HD samples only (drop NaN)
hd_w = hd.restrict(wake)
mask = ~np.isnan(hd_w.values)
hd_t_valid = hd_w.t[mask]
hd_d_valid = hd_w.values[mask]
print("valid wake HD samples:", len(hd_t_valid))

rates = units.metadata["rate"].values
units_f = units[rates > 0.1]
keys = list(units_f.keys())

rng = np.random.default_rng(0)
n_shuf = 500

def spike_angles(sp_t):
    return np.interp(sp_t, hd_t_valid, hd_d_valid)

print(f"{'unit':>5} {'n_spk':>7} {'obs':>6} {'null95':>7} {'null_mean':>9} {'HD?':>4}")
results = []
for u in keys:
    sp = units_f[u].restrict(wake)
    n = len(sp)
    if n < 100:
        continue
    obs_ang = spike_angles(sp.t)
    obs = np.abs(np.exp(1j * obs_ang).mean())
    # random-time null: sample n times uniformly from valid wake HD samples
    null = np.abs(np.exp(1j * rng.choice(hd_d_valid, size=(n_shuf, n))).mean(axis=1))
    p = (np.sum(null >= obs) + 1) / (n_shuf + 1)
    results.append((u, n, obs, np.percentile(null, 95), null.mean(), p < 0.05))
    print(f"{u:>5} {n:>7} {obs:>6.3f} {np.percentile(null,95):>7.3f} "
          f"{null.mean():>9.3f} {'YES' if p<0.05 else 'no':>4}")

print("\nHD cells:", sum(r[-1] for r in results), "/", len(results))
