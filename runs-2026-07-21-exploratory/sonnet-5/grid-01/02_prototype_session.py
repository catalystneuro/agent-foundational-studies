import h5py
from pynwb import NWBHDF5IO
import remfile
import pynapple as nap
import numpy as np
import matplotlib.pyplot as plt

from gridcell_utils import smooth_rate_map, spatial_autocorrelogram, gridness_score

url = "https://dandiarchive.s3.amazonaws.com/blobs/26a/22c/26a22c31-09bc-43a4-9187-edc7394ed12c"

disk_cache = remfile.DiskCache('cache/remfile_cache')
rem_file = remfile.File(url, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)

units = nwb["units"]
pos = nwb["SpatialSeriesLED1"]

n_bins = 40
tcs = nap.compute_tuning_curves(units, pos, bins=n_bins, range=[(-50, 50), (-50, 50)])
print(tcs.shape)
occupancy = tcs.attrs["occupancy"]
print("occupancy shape", occupancy.shape)
print("fs", tcs.attrs["fs"])

# mask bins with very low occupancy (< 100 ms) as unvisited -> nan
min_occ_samples = 1  # in samples at fs
fs = tcs.attrs["fs"]
min_occ_time = 0.1  # seconds
min_samples = min_occ_time * fs

fig, axes = plt.subplots(3, 8, figsize=(24, 9))
for i, unit_id in enumerate(tcs.coords["unit"].values):
    rate_map = tcs.sel(unit=unit_id).values.astype(float)
    rate_map = np.where(occupancy < min_samples, np.nan, rate_map)

    smoothed = smooth_rate_map(rate_map, sigma=1.0)
    autocorr = spatial_autocorrelogram(smoothed)
    g, r_in, r_out, peaks = gridness_score(autocorr)

    axes[0, i].imshow(rate_map, origin="lower", cmap="jet")
    axes[0, i].set_title(f"unit {unit_id}\nraw", fontsize=9)
    axes[0, i].axis("off")

    axes[1, i].imshow(smoothed, origin="lower", cmap="jet")
    axes[1, i].set_title("smoothed", fontsize=9)
    axes[1, i].axis("off")

    im = axes[2, i].imshow(autocorr, origin="lower", cmap="jet", vmin=-1, vmax=1)
    axes[2, i].set_title(f"autocorr\ng={g:.2f}", fontsize=9)
    axes[2, i].axis("off")

plt.tight_layout()
plt.savefig("figures/prototype_session_grid_cells.png", dpi=150)
print("saved figure")
