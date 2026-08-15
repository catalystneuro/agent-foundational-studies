"""Check LED tracking quality and derive head direction for one session."""
import h5py, remfile, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pynwb import NWBHDF5IO

ASSET = "56d2e2d7-ba41-40a1-b017-b81871f3f0c3"
URL = f"https://api.dandiarchive.org/api/dandisets/000056/versions/draft/assets/{ASSET}/download/"
rem_file = remfile.File(URL, disk_cache=remfile.DiskCache("/tmp/remfile_cache"))
io = NWBHDF5IO(file=h5py.File(rem_file, "r"), load_namespaces=True)
nwbfile = io.read()

pos = nwbfile.processing["behavior"]["SubjectPosition"]
red = pos.spatial_series["RedLED"]
blue = pos.spatial_series["BlueLED"]
print("red rate", red.rate, "starting_time", red.starting_time, "conversion", red.conversion)
print("blue rate", blue.rate, "starting_time", blue.starting_time)
print("has timestamps:", red.timestamps is not None, blue.timestamps is not None)
if red.timestamps is not None:
    ts = red.timestamps[:]
    print("ts range", ts[0], ts[-1], "n", len(ts))
    d = np.diff(ts)
    print("dt median", np.median(d), "min", d.min(), "max", d.max(), "n_neg", (d <= 0).sum())

R = red.data[:]
B = blue.data[:]
print("R shape", R.shape, "nan frac", np.isnan(R).mean(), "range", np.nanmin(R, 0), np.nanmax(R, 0))
print("B nan frac", np.isnan(B).mean())
print("n zeros R", (R == 0).all(1).sum())

# also compare to acquisition copies
print("acq RedLED same object?", nwbfile.acquisition["RedLED"].data is red.data)
