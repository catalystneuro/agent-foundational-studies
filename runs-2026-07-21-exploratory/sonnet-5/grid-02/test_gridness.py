import h5py
from pynwb import NWBHDF5IO
import remfile
import pynapple as nap
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from grid_lib import spatial_autocorrelogram, gridness_score

s3_url = "https://dandiarchive.s3.amazonaws.com/blobs/275/de6/275de639-185d-433c-866b-edad9bfcdcc8"
disk_cache = remfile.DiskCache('remfile_cache')
rem_file = remfile.File(s3_url, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file, load_namespaces=True)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)

pos = nwb['SpatialSeriesLED1']
position = nap.TsdFrame(t=pos.t, d=pos.values, columns=['x', 'y']).dropna()
units = nwb['units']
tc = nap.compute_tuning_curves(units, position, bins=30, epochs=position.time_support)

scores = {}
for k in tc.coords['unit'].values:
    rate_map = tc.sel(unit=k).values.T
    smoothed = gaussian_filter(np.nan_to_num(rate_map), sigma=1.0)
    smoothed[np.isnan(rate_map)] = np.nan
    ac = spatial_autocorrelogram(smoothed)
    score = gridness_score(ac)
    scores[k] = score
    print(f"unit {k}: gridness = {score:.3f}")

print(sorted(scores.items(), key=lambda x: -x[1] if not np.isnan(x[1]) else 999))

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
show_units = [4, 11, 3, 0, 2, 6, 1, 5]
for i, k in enumerate(show_units):
    rate_map = tc.sel(unit=k).values.T
    smoothed = gaussian_filter(np.nan_to_num(rate_map), sigma=1.0)
    smoothed[np.isnan(rate_map)] = np.nan
    ac = spatial_autocorrelogram(smoothed)
    ax = axes.flat[i]
    im = ax.imshow(ac, origin='lower', cmap='jet', vmin=-1, vmax=1)
    ax.set_title(f"unit {k} g={scores[k]:.2f}")
    plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig('test_autocorr.png', dpi=120)
print('saved test_autocorr.png')
