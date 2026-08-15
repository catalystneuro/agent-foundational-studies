import h5py
from pynwb import NWBHDF5IO
import remfile
import pynapple as nap
import numpy as np
import matplotlib.pyplot as plt

s3_url = "https://dandiarchive.s3.amazonaws.com/blobs/d42/fc9/d42fc926-b773-471c-98ec-db22978d2286"
disk_cache = remfile.DiskCache('remfile_cache')
rem_file = remfile.File(s3_url, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file, load_namespaces=True)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)

print(nwb)

units = nwb['units']
pos = nwb['SpatialSeriesLED1']
print(pos)

# Build position TsdFrame with x,y
position = nap.TsdFrame(t=pos.t, d=pos.values, columns=['x', 'y'])
print(position)

# restrict to valid tracking (no nans)
position = position.dropna()
print("valid position samples", len(position))

# Compute 2D tuning curves (rate maps)
tc = nap.compute_tuning_curves(units, position, bins=20, epochs=position.time_support)
print(tc)
print(tc.coords)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for i, k in enumerate(tc.coords['unit'].values):
    ax = axes.flat[i]
    rate_map = tc.sel(unit=k).values
    im = ax.imshow(rate_map.T, origin='lower', cmap='jet')
    ax.set_title(f"unit {k}")
    plt.colorbar(im, ax=ax)
for j in range(i + 1, axes.flat.shape[0] if hasattr(axes.flat, 'shape') else len(axes.flat)):
    axes.flat[j].axis('off')
plt.tight_layout()
plt.savefig('prototype_ratemaps.png', dpi=120)
print("saved prototype_ratemaps.png")
