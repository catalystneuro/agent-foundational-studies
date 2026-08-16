"""Examine HD signal structure during wake: occupancy concentration, static periods."""
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

hd_w = hd.restrict(wake)
v = hd_w.values
v = v[~np.isnan(v)]
print("wake HD samples:", len(v))
print("occupancy resultant |mean(exp(i*hd))|:",
      np.abs(np.exp(1j * v).mean()))

# occupancy histogram
occ, edges = np.histogram(v, bins=36, range=(0, 2 * np.pi))
print("occ min/max/mean:", occ.min(), occ.max(), occ.mean())
print("occ cv:", occ.std() / occ.mean())

# how static is HD? fraction of time with |angular velocity| < 5 deg/s
t = hd_w.t
d = hd_w.values
step = np.diff(d)
step = (step + np.pi) % (2 * np.pi) - np.pi
dt = np.diff(t)
angvel = np.abs(step) / dt
angvel = angvel[~np.isnan(angvel)]
print("median |ang vel| (deg/s):", np.median(np.degrees(angvel)))
print("frac time < 5 deg/s:", np.mean(angvel < np.radians(5)))

# LED separation (should be roughly constant)
sep = np.sqrt((red.values[:, 0] - blue.values[:, 0])**2 +
              (red.values[:, 1] - blue.values[:, 1])**2)
print("LED separation: median", np.nanmedian(sep[valid]),
      "p5", np.nanpercentile(sep[valid], 5),
      "p95", np.nanpercentile(sep[valid], 95))

# position coverage
print("red x range:", np.percentile(red.values[valid, 0], [1, 50, 99]))
print("red y range:", np.percentile(red.values[valid, 1], [1, 50, 99]))

# per-epoch occupancy resultant
for i in range(min(8, len(wake))):
    e = nap.IntervalSet(start=[wake["start"][i]], end=[wake["end"][i]])
    h = hd.restrict(e).values
    h = h[~np.isnan(h)]
    print(f"wake epoch {i}: dur={wake['end'][i]-wake['start'][i]:.0f}s "
          f"R_occ={np.abs(np.exp(1j*h).mean()):.3f}")
