import h5py
from pynwb import NWBHDF5IO
import remfile
import pynapple as nap
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

s3_url = "https://dandiarchive.s3.amazonaws.com/blobs/275/de6/275de639-185d-433c-866b-edad9bfcdcc8"
disk_cache = remfile.DiskCache('remfile_cache')
rem_file = remfile.File(s3_url, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file, load_namespaces=True)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)

print(nwb)
print(nwbfile.session_description)
print(nwbfile.units.histology[:])
print("n units", len(nwbfile.units))

pos = nwb['SpatialSeriesLED1']
position = nap.TsdFrame(t=pos.t, d=pos.values, columns=['x', 'y']).dropna()
print("duration (s)", position.t[-1] - position.t[0])
print("n position samples", len(position))

units = nwb['units']
tc = nap.compute_tuning_curves(units, position, bins=30, epochs=position.time_support)
print(tc.coords['unit'].values)

n_units = tc.sizes['unit']
ncols = 4
nrows = int(np.ceil(n_units / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
axes = np.atleast_1d(axes).flatten()
for i, k in enumerate(tc.coords['unit'].values):
    ax = axes[i]
    rate_map = tc.sel(unit=k).values.T
    smoothed = gaussian_filter(np.nan_to_num(rate_map), sigma=1.0)
    im = ax.imshow(smoothed, origin='lower', cmap='jet')
    ax.set_title(f"unit {k}")
    plt.colorbar(im, ax=ax)
for j in range(i + 1, len(axes)):
    axes[j].axis('off')
plt.tight_layout()
plt.savefig('prototype_11265_ratemaps.png', dpi=120)
print("saved")
