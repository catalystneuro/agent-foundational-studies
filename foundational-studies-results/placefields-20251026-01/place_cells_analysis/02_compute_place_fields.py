"""
Compute spatial firing rate maps (place fields) for hippocampal neurons.

This script computes 2D spatial firing rate maps for each neuron and identifies
place cells based on spatial information content.
"""

import h5py
import remfile
from pynwb import NWBHDF5IO
import pynapple as nap
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from tqdm import tqdm

# Configuration
ASSET_URL = "https://api.dandiarchive.org/api/assets/b048bd0a-48a4-4d19-ad96-fd879666bca5/download/"
CACHE_DIR = "/tmp/remfile_cache_dandi_000447"
BIN_SIZE = 3  # cm, spatial bin size
SMOOTHING_SIGMA = 1.5  # bins, for gaussian smoothing of rate maps

print("Loading NWB file from DANDI Archive...")
disk_cache = remfile.DiskCache(CACHE_DIR)
rem_file = remfile.File(ASSET_URL, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file, mode='r')
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)

# Get units and position data
units = nwb['units']
spatial_series = nwbfile.processing['behavior']['Position'].spatial_series['SpatialSeries']
pos_data = spatial_series.data[:]
pos_times = spatial_series.timestamps[:]

# Extract x, y position (3rd column appears to be z or another dimension)
x_pos = pos_data[:, 0]
y_pos = pos_data[:, 1]

print(f"\n=== Data Summary ===")
print(f"Number of units: {len(units)}")
print(f"Position samples: {len(x_pos)}")
print(f"Position range: x=[{x_pos.min():.2f}, {x_pos.max():.2f}], y=[{y_pos.min():.2f}, {y_pos.max():.2f}]")
print(f"Duration: {pos_times[-1] - pos_times[0]:.2f}s")

# Focus on first epoch (novel environment)
epochs = nwbfile.intervals['epoch intervals']
epoch_start = epochs['start_time'][0]
epoch_end = epochs['stop_time'][0]
print(f"\nAnalyzing Epoch 0 (Novel): {epoch_start:.2f}s - {epoch_end:.2f}s")

# Filter position data to epoch
epoch_mask = (pos_times >= epoch_start) & (pos_times <= epoch_end)
x_pos_epoch = x_pos[epoch_mask]
y_pos_epoch = y_pos[epoch_mask]
pos_times_epoch = pos_times[epoch_mask]

print(f"Epoch position samples: {len(x_pos_epoch)}")

# Create spatial bins
x_bins = np.arange(x_pos.min(), x_pos.max() + BIN_SIZE, BIN_SIZE)
y_bins = np.arange(y_pos.min(), y_pos.max() + BIN_SIZE, BIN_SIZE)
n_x_bins = len(x_bins) - 1
n_y_bins = len(y_bins) - 1

print(f"\n=== Spatial Binning ===")
print(f"Bin size: {BIN_SIZE} cm")
print(f"X bins: {n_x_bins}, Y bins: {n_y_bins}")
print(f"Total bins: {n_x_bins * n_y_bins}")

# Compute occupancy map
occupancy, _, _ = np.histogram2d(x_pos_epoch, y_pos_epoch, bins=[x_bins, y_bins])
# Convert to time (assuming ~30 Hz sampling)
sampling_rate = 1 / np.median(np.diff(pos_times_epoch))
occupancy_time = occupancy / sampling_rate  # seconds in each bin
print(f"Position sampling rate: {sampling_rate:.2f} Hz")
print(f"Total occupancy time: {occupancy_time.sum():.2f}s")

# Smooth occupancy map
occupancy_time_smooth = ndimage.gaussian_filter(occupancy_time, sigma=SMOOTHING_SIGMA)

# Compute firing rate maps for each unit
print("\n=== Computing Firing Rate Maps ===")
rate_maps = []
peak_rates = []
spatial_info = []
mean_rates = []

for unit_idx in tqdm(range(len(units)), desc="Computing place fields"):
    spikes = units[unit_idx]
    
    # Filter spikes to epoch
    spike_times = spikes.t[(spikes.t >= epoch_start) & (spikes.t <= epoch_end)]
    
    if len(spike_times) == 0:
        rate_maps.append(np.zeros((n_x_bins, n_y_bins)))
        peak_rates.append(0)
        spatial_info.append(0)
        mean_rates.append(0)
        continue
    
    # Interpolate position at spike times
    x_spike = np.interp(spike_times, pos_times_epoch, x_pos_epoch)
    y_spike = np.interp(spike_times, pos_times_epoch, y_pos_epoch)
    
    # Compute spike count map
    spike_counts, _, _ = np.histogram2d(x_spike, y_spike, bins=[x_bins, y_bins])
    
    # Smooth spike counts
    spike_counts_smooth = ndimage.gaussian_filter(spike_counts, sigma=SMOOTHING_SIGMA)
    
    # Compute firing rate map (Hz)
    rate_map = np.zeros_like(occupancy_time_smooth)
    valid_bins = occupancy_time_smooth > 0.1  # At least 0.1s occupancy
    rate_map[valid_bins] = spike_counts_smooth[valid_bins] / occupancy_time_smooth[valid_bins]
    
    rate_maps.append(rate_map)
    peak_rates.append(rate_map.max())
    mean_rate = len(spike_times) / (epoch_end - epoch_start)
    mean_rates.append(mean_rate)
    
    # Compute spatial information (bits/spike)
    # I = sum_i P(i) * (R(i) / R_mean) * log2(R(i) / R_mean)
    if mean_rate > 0:
        p_i = occupancy_time_smooth[valid_bins] / occupancy_time_smooth[valid_bins].sum()
        r_i = rate_map[valid_bins]
        ratio = r_i / mean_rate
        ratio[ratio <= 0] = 1e-10  # Avoid log(0)
        info = np.sum(p_i * ratio * np.log2(ratio))
        spatial_info.append(info)
    else:
        spatial_info.append(0)

rate_maps = np.array(rate_maps)
peak_rates = np.array(peak_rates)
spatial_info = np.array(spatial_info)
mean_rates = np.array(mean_rates)

print(f"\n=== Place Field Statistics ===")
print(f"Peak firing rate range: {peak_rates.min():.2f} - {peak_rates.max():.2f} Hz")
print(f"Mean firing rate: {mean_rates.mean():.2f} ± {mean_rates.std():.2f} Hz")
print(f"Spatial information: {spatial_info.mean():.3f} ± {spatial_info.std():.3f} bits/spike")

# Identify place cells (spatial info > 0.5 bits/spike and peak rate > 1 Hz)
place_cell_mask = (spatial_info > 0.5) & (peak_rates > 1.0)
n_place_cells = np.sum(place_cell_mask)
print(f"\nPlace cells identified: {n_place_cells} / {len(units)} ({100*n_place_cells/len(units):.1f}%)")

# Visualize example place fields
print("\n=== Creating Visualizations ===")

# Plot 1: Example place fields
fig, axes = plt.subplots(3, 5, figsize=(15, 9))
axes = axes.flatten()

# Sort by spatial information and plot top 15
sorted_indices = np.argsort(spatial_info)[::-1]

for plot_idx in range(15):
    ax = axes[plot_idx]
    unit_idx = sorted_indices[plot_idx]
    
    im = ax.imshow(rate_maps[unit_idx].T, origin='lower', aspect='auto', 
                   extent=[x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]],
                   cmap='hot', interpolation='gaussian')
    ax.set_title(f'Unit {unit_idx}\nInfo: {spatial_info[unit_idx]:.2f} b/s\nPeak: {peak_rates[unit_idx]:.1f} Hz',
                 fontsize=8)
    ax.set_xlabel('X (cm)', fontsize=8)
    ax.set_ylabel('Y (cm)', fontsize=8)
    plt.colorbar(im, ax=ax, label='Rate (Hz)', fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig('fig_02_place_fields.png', dpi=150, bbox_inches='tight')
print("Saved: fig_02_place_fields.png")

# Plot 2: Population statistics
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Spatial information distribution
ax = axes[0, 0]
ax.hist(spatial_info, bins=30, edgecolor='black', alpha=0.7)
ax.axvline(0.5, color='red', linestyle='--', label='Place cell threshold')
ax.set_xlabel('Spatial Information (bits/spike)')
ax.set_ylabel('Count')
ax.set_title('Spatial Information Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

# Peak rate distribution
ax = axes[0, 1]
ax.hist(peak_rates, bins=30, edgecolor='black', alpha=0.7)
ax.set_xlabel('Peak Firing Rate (Hz)')
ax.set_ylabel('Count')
ax.set_title('Peak Rate Distribution')
ax.grid(True, alpha=0.3)

# Spatial info vs peak rate
ax = axes[1, 0]
ax.scatter(peak_rates[~place_cell_mask], spatial_info[~place_cell_mask], 
           alpha=0.5, s=30, label='Non-place cells', color='gray')
ax.scatter(peak_rates[place_cell_mask], spatial_info[place_cell_mask], 
           alpha=0.7, s=50, label='Place cells', color='red')
ax.axhline(0.5, color='red', linestyle='--', alpha=0.5)
ax.axvline(1.0, color='red', linestyle='--', alpha=0.5)
ax.set_xlabel('Peak Firing Rate (Hz)')
ax.set_ylabel('Spatial Information (bits/spike)')
ax.set_title('Place Cell Identification')
ax.legend()
ax.grid(True, alpha=0.3)

# Occupancy map
ax = axes[1, 1]
im = ax.imshow(occupancy_time_smooth.T, origin='lower', aspect='auto',
               extent=[x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]],
               cmap='viridis')
ax.set_xlabel('X Position (cm)')
ax.set_ylabel('Y Position (cm)')
ax.set_title('Occupancy Map')
plt.colorbar(im, ax=ax, label='Time (s)', fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig('fig_03_place_cell_statistics.png', dpi=150, bbox_inches='tight')
print("Saved: fig_03_place_cell_statistics.png")

# Save results
np.savez('place_field_data.npz',
         rate_maps=rate_maps,
         peak_rates=peak_rates,
         spatial_info=spatial_info,
         mean_rates=mean_rates,
         place_cell_mask=place_cell_mask,
         x_bins=x_bins,
         y_bins=y_bins,
         occupancy=occupancy_time_smooth)
print("\nSaved: place_field_data.npz")

print("\n=== Place field computation complete ===")
