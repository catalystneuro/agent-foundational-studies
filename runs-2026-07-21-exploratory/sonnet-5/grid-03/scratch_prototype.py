import h5py
from pynwb import NWBHDF5IO
import remfile
import pynapple as nap
import numpy as np
import matplotlib.pyplot as plt

from grid_utils import smooth_ratemap, autocorrelogram, gridness_score

s3_url = "https://dandiarchive.s3.amazonaws.com/blobs/26a/22c/26a22c31-09bc-43a4-9187-edc7394ed12c"
disk_cache = remfile.DiskCache('/tmp/remfile_cache')
rem_file = remfile.File(s3_url, disk_cache=disk_cache)
h5f = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5f)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)

position = nwb['SpatialSeriesLED1']  # TsdFrame with columns x, y (cm)
units = nwb['units']
ep = position.time_support

print("Position time support:", ep)
print("N units:", len(units))

# Raw trajectory plot
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(position['x'].values, position['y'].values, lw=0.3, color='gray')
ax.set_xlabel('x (cm)')
ax.set_ylabel('y (cm)')
ax.set_title('Raw trajectory, sub-10073 ses-17010302')
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig('figures/scratch_trajectory.png', dpi=120)
plt.close()

bins = 30
range_ = [(-50, 50), (-50, 50)]

tc = nap.compute_tuning_curves(units, position, bins=bins, range=range_, epochs=ep)
print(tc.shape, tc.dims)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for i, unit_id in enumerate(tc.coords['unit'].values):
    ratemap = tc.sel(unit=unit_id).values
    ax = axes.flat[i]
    im = ax.imshow(ratemap.T, origin='lower', cmap='jet')
    ax.set_title(f"unit {unit_id}")
    plt.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
plt.savefig('figures/scratch_ratemaps.png', dpi=120)
plt.close()

# test autocorrelogram + gridness on one unit
unit_id = tc.coords['unit'].values[0]
ratemap = tc.sel(unit=unit_id).values
smooth = smooth_ratemap(ratemap, sigma=1.0)
corr = autocorrelogram(smooth)
score, rot_corrs, (inner, outer) = gridness_score(corr)
print("unit", unit_id, "gridness:", score, "annulus:", inner, outer, "rot corrs:", rot_corrs)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(ratemap.T, origin='lower', cmap='jet')
axes[0].set_title('raw ratemap')
axes[1].imshow(smooth.T, origin='lower', cmap='jet')
axes[1].set_title('smoothed ratemap')
im2 = axes[2].imshow(corr.T, origin='lower', cmap='jet')
axes[2].set_title(f'autocorrelogram, gridness={score:.2f}')
plt.colorbar(im2, ax=axes[2], fraction=0.046)
plt.tight_layout()
plt.savefig('figures/scratch_autocorr_test.png', dpi=120)
plt.close()

print("done")
