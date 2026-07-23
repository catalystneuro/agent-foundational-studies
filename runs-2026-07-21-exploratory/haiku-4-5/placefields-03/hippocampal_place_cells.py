# %% [markdown]
# # Hippocampal Place Cells from DANDI Archive
#
# This analysis demonstrates place cell properties in the hippocampus using data from DANDISET 000044.
# We load spike recordings from 68 hippocampal units recorded during navigation in a linear maze,
# extract spatial firing patterns (place fields), and identify place cells based on spatial information content.

# %% [markdown]
# ## Setup and Data Loading

# %%
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from scipy import stats
import h5py
from pynwb import NWBHDF5IO
import remfile
import pynapple as nap
from tqdm import tqdm

# Configure matplotlib for headless mode
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# S3 URL to DANDISET 000044 session (Buddy 06-27-2013)
s3_url = "https://dandiarchive.s3.amazonaws.com/blobs/98b/25c/98b25cb1-310c-45f7-97cc-669fce2057b7"

print("Loading NWB file from DANDI...")
disk_cache = remfile.DiskCache('/tmp/remfile_cache')
rem_file = remfile.File(s3_url, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")

io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()

# Create Pynapple NWBFile wrapper
nwb = nap.NWBFile(nwbfile)
print(nwb)

# %% [markdown]
# ## Data Inspection and Preprocessing

# %%
# Extract spike data
units_table = nwbfile.units
print(f"\nRecorded units: {len(units_table)}")
print(f"Unit metadata: {units_table.colnames}")

# Convert to Pynapple TsGroup (spike times for each unit)
spikes = nap.TsGroup({i: nap.Ts(units_table['spike_times'][i]) for i in range(len(units_table))})
print(f"Spike times object: {type(spikes)}")
print(f"Number of units in spike group: {len(spikes)}")

# Extract position data
# The behavior module contains position in the linear maze
position_container = nwbfile.processing['behavior']['1.6mLinearMazePosition']
spatial_series = position_container.get_spatial_series()

# Get position data (x,y coordinates)
position_xy = spatial_series.data[:]

# Reconstruct timestamps from rate and starting_time (timestamps can be None)
n_samples = position_xy.shape[0]
rate = spatial_series.rate
starting_time = spatial_series.starting_time
position_timestamps = starting_time + np.arange(n_samples) / rate

# Convert (x,y) to 1D position: use X coordinate (primary axis along maze)
# First, remove NaN rows
valid_idx = ~np.isnan(position_xy).any(axis=1)
position_x = position_xy[valid_idx, 0]
position_y = position_xy[valid_idx, 1]
position_timestamps_valid = position_timestamps[valid_idx]

# Normalize X position to approximate 0-1.6m maze coordinates
# The maze is approximately 1.6m long (given in dataset name)
position_x_norm = position_x - position_x.min()
position_x_norm = position_x_norm / position_x_norm.max() * 1.6

print(f"\nPosition data shape: valid={len(position_timestamps_valid)}, total={len(position_timestamps)}")
print(f"Position range: [{position_x_norm.min():.2f}, {position_x_norm.max():.2f}] meters")
print(f"Position sampling rate: {rate:.2f} Hz")
print(f"Recording segment duration: {position_timestamps_valid[-1] - position_timestamps_valid[0]:.1f} seconds")

# Create Pynapple Tsd (time series data) for position
position_tsd = nap.Tsd(t=position_timestamps_valid, d=position_x_norm)

# %% [markdown]
# ## Occupancy and Firing Rate Maps

# %%
# Define spatial bins (1cm resolution)
position_bins = np.arange(0, 1.6 + 0.01, 0.01)
n_bins = len(position_bins) - 1

# Compute occupancy map (time spent in each position bin)
# Restrict spikes and position to valid times
position_t = position_tsd.t
position_d = position_tsd.d

occupancy, bin_edges = np.histogram(position_d, bins=position_bins)
# Compute time in each bin from the actual position time series
bin_width = position_bins[1] - position_bins[0]
dt = np.median(np.diff(position_t))  # Median sampling interval
occupancy_time = occupancy * dt  # Time in each bin

recording_duration = position_tsd.end_time() - position_tsd.start_time()
print(f"Number of spatial bins: {n_bins}")
print(f"Total recording duration: {recording_duration:.1f} seconds")
print(f"Mean occupancy per bin: {occupancy_time.mean():.3f} seconds")

# Compute firing rate maps for each unit
firing_rate_maps = []
peak_firing_rates = []
spatial_info_values = []

print("\nComputing firing rate maps...")
for unit_idx in tqdm(range(len(spikes)), desc="Units"):
    spike_times = spikes[unit_idx].t  # Get spike times as array

    # Find which position samples correspond to spike times
    spike_positions = np.interp(spike_times, position_t, position_d, left=np.nan, right=np.nan)

    # Only use spikes that occurred during valid position tracking
    valid_spike_mask = ~np.isnan(spike_positions)
    spike_positions_valid = spike_positions[valid_spike_mask]

    if len(spike_positions_valid) == 0:
        firing_rate_maps.append(np.zeros(n_bins))
        peak_firing_rates.append(0)
        spatial_info_values.append(0)
        continue

    # Bin spikes by position
    spike_counts, _ = np.histogram(spike_positions_valid, bins=position_bins)

    # Firing rate: spikes per second
    firing_rate_map = spike_counts / occupancy_time
    firing_rate_map[occupancy_time < 0.05] = 0  # Exclude poorly sampled bins

    firing_rate_maps.append(firing_rate_map)
    peak_firing_rates.append(firing_rate_map.max())

    # Compute spatial information (bits per spike)
    # SI = sum(p_i * f_i * log2(f_i / f_avg))
    mean_rate = len(spike_positions_valid) / recording_duration

    if mean_rate > 0.05:  # Only for units with sufficient firing
        p_occupancy = occupancy_time / occupancy_time.sum()
        valid = (p_occupancy > 0) & (firing_rate_map > 0)

        if valid.sum() > 0:
            si = np.sum(p_occupancy[valid] * firing_rate_map[valid] *
                       np.log2(firing_rate_map[valid] / mean_rate))
            spatial_info_values.append(max(0, si))  # SI should be non-negative
        else:
            spatial_info_values.append(0)
    else:
        spatial_info_values.append(0)

firing_rate_maps = np.array(firing_rate_maps)
spatial_info_values = np.array(spatial_info_values)

print(f"\nSpatial information (mean ± std): {spatial_info_values.mean():.3f} ± {spatial_info_values.std():.3f} bits/spike")
print(f"Peak firing rates: {np.array(peak_firing_rates).mean():.2f} ± {np.array(peak_firing_rates).std():.2f} Hz")

# %% [markdown]
# ## Place Cell Classification

# %%
# Identify place cells using spatial information threshold
# Typical threshold: SI > 0.5 bits/spike (Skaggs et al., 1992)
si_threshold = 0.5
place_cell_mask = spatial_info_values > si_threshold
place_cell_indices = np.where(place_cell_mask)[0]

print(f"\nPlace cell identification:")
print(f"Spatial information threshold: {si_threshold} bits/spike")
print(f"Number of place cells: {len(place_cell_indices)} / {len(spikes)}")
print(f"Percentage of place cells: {100 * len(place_cell_indices) / len(spikes):.1f}%")

# Characterize place cells
place_cell_si = spatial_info_values[place_cell_mask]
place_cell_peak_rates = np.array(peak_firing_rates)[place_cell_mask]

print(f"\nPlace cell properties:")
print(f"SI (mean ± std): {place_cell_si.mean():.3f} ± {place_cell_si.std():.3f} bits/spike")
print(f"Peak firing rate (mean ± std): {place_cell_peak_rates.mean():.2f} ± {place_cell_peak_rates.std():.2f} Hz")

# %% [markdown]
# ## Visualization: Occupancy and Firing Rate Maps

# %%
fig, axes = plt.subplots(2, 1, figsize=(12, 6))

# Occupancy map
ax = axes[0]
position_bin_centers = (position_bins[:-1] + position_bins[1:]) / 2
ax.bar(position_bin_centers, occupancy_time, width=np.diff(position_bins).mean(),
       color='steelblue', alpha=0.7, edgecolor='none')
ax.set_xlabel('Position in maze (m)')
ax.set_ylabel('Time occupancy (s)')
ax.set_title('Occupancy Map - Linear Maze')
ax.grid(True, alpha=0.3)

# Peak firing rates distribution
ax = axes[1]
ax.hist(np.array(peak_firing_rates), bins=20, color='gray', alpha=0.6, label='All units', edgecolor='black')
ax.hist(place_cell_peak_rates, bins=15, color='coral', alpha=0.8, label=f'Place cells (n={len(place_cell_indices)})', edgecolor='black')
ax.axvline(place_cell_peak_rates.mean(), color='darkred', linestyle='--', linewidth=2, label='Place cell mean')
ax.set_xlabel('Peak firing rate (Hz)')
ax.set_ylabel('Number of units')
ax.set_title('Peak Firing Rate Distribution')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('01_occupancy_and_peak_rates.png', dpi=150, bbox_inches='tight')
print("Saved: 01_occupancy_and_peak_rates.png")
plt.close()

# %% [markdown]
# ## Visualization: Spatial Information vs Peak Firing Rate

# %%
fig, ax = plt.subplots(figsize=(10, 7))

# Scatter plot: SI vs peak firing rate
colors = np.where(place_cell_mask, 'coral', 'lightgray')
ax.scatter(np.array(peak_firing_rates), spatial_info_values,
          c=colors, s=80, alpha=0.7, edgecolors='black', linewidth=0.5)

# Threshold lines
ax.axhline(si_threshold, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'SI threshold ({si_threshold} bits/spike)')
ax.set_xlabel('Peak Firing Rate (Hz)', fontsize=11)
ax.set_ylabel('Spatial Information (bits/spike)', fontsize=11)
ax.set_title('Place Cell Identification: Spatial Information vs Peak Firing Rate', fontsize=12)
ax.grid(True, alpha=0.3)

# Legend
place_cell_patch = mpatches.Patch(color='coral', label=f'Place cells (n={len(place_cell_indices)})')
other_patch = mpatches.Patch(color='lightgray', label=f'Non-place cells (n={len(spikes)-len(place_cell_indices)})')
ax.legend(handles=[place_cell_patch, other_patch,
          mpatches.Patch(color='none', label=f'SI threshold ({si_threshold} bits/spike)')],
          loc='upper left', fontsize=10)

plt.tight_layout()
plt.savefig('02_place_cell_classification.png', dpi=150, bbox_inches='tight')
print("Saved: 02_place_cell_classification.png")
plt.close()

# %% [markdown]
# ## Visualization: Example Place Cell Firing Rate Maps

# %%
# Select top 6 place cells by spatial information
top_place_cells = np.argsort(place_cell_si)[-6:][::-1]

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

for plot_idx, unit_idx in enumerate(top_place_cells):
    ax = axes[plot_idx]

    # Get absolute unit index
    abs_unit_idx = place_cell_indices[unit_idx]
    firing_map = firing_rate_maps[abs_unit_idx]

    # Plot firing rate map as bar chart
    colors_map = plt.cm.YlOrRd(Normalize(vmin=0, vmax=firing_map.max())(firing_map))
    ax.bar(position_bin_centers, firing_map, width=np.diff(position_bins).mean(),
          color=colors_map, edgecolor='none')

    ax.set_xlabel('Position (m)', fontsize=9)
    ax.set_ylabel('Firing rate (Hz)', fontsize=9)
    si_val = spatial_info_values[abs_unit_idx]
    peak_rate = peak_firing_rates[abs_unit_idx]
    ax.set_title(f'Unit {abs_unit_idx}: SI={si_val:.2f} bits/spike, Peak={peak_rate:.1f} Hz',
                fontsize=10)
    ax.grid(True, alpha=0.2, axis='y')

plt.tight_layout()
plt.savefig('03_top_place_cells.png', dpi=150, bbox_inches='tight')
print("Saved: 03_top_place_cells.png")
plt.close()

# %% [markdown]
# ## Visualization: Spatial Information Distribution

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Histogram with threshold
ax = axes[0]
ax.hist(spatial_info_values, bins=25, color='steelblue', alpha=0.7, edgecolor='black')
ax.axvline(si_threshold, color='red', linestyle='--', linewidth=2.5, label=f'Threshold ({si_threshold})')
ax.axvline(spatial_info_values.mean(), color='darkblue', linestyle='-', linewidth=2, label=f'Mean ({spatial_info_values.mean():.2f})')
ax.set_xlabel('Spatial Information (bits/spike)', fontsize=11)
ax.set_ylabel('Number of units', fontsize=11)
ax.set_title('Spatial Information Distribution', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

# Cumulative distribution
ax = axes[1]
sorted_si = np.sort(spatial_info_values)
cumulative = np.arange(1, len(sorted_si) + 1) / len(sorted_si)
ax.plot(sorted_si, cumulative, linewidth=2.5, color='steelblue')
ax.axvline(si_threshold, color='red', linestyle='--', linewidth=2.5, label=f'Threshold ({si_threshold})')
ax.axhline(1 - len(place_cell_indices) / len(spikes), color='red', linestyle=':', linewidth=1.5, alpha=0.7)
ax.set_xlabel('Spatial Information (bits/spike)', fontsize=11)
ax.set_ylabel('Cumulative fraction of units', fontsize=11)
ax.set_title('Cumulative Distribution of Spatial Information', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('04_spatial_information_distribution.png', dpi=150, bbox_inches='tight')
print("Saved: 04_spatial_information_distribution.png")
plt.close()

# %% [markdown]
# ## Summary Statistics

# %%
print("\n" + "="*60)
print("HIPPOCAMPAL PLACE CELL ANALYSIS SUMMARY")
print("="*60)
print(f"\nDataset: DANDISET 000044 (Buddy, 2013-06-27)")
print(f"Brain region: Hippocampus CA1/CA3")
print(f"Recording type: Multi-electrode array (bilateral silicon probes)")
print(f"Behavioral task: Navigation in 1.6m linear maze")

print(f"\nRecording parameters:")
print(f"  Total units recorded: {len(spikes)}")
print(f"  Session duration: {recording_duration:.1f} seconds ({recording_duration/60:.1f} minutes)")
print(f"  Spatial coverage: 0.00 - 1.60 m")

print(f"\nPlace cell statistics:")
print(f"  Number of place cells: {len(place_cell_indices)}")
print(f"  Percentage of place cells: {100 * len(place_cell_indices) / len(spikes):.1f}%")
print(f"  Mean SI (place cells): {place_cell_si.mean():.3f} ± {place_cell_si.std():.3f} bits/spike")
print(f"  Mean peak firing rate (place cells): {place_cell_peak_rates.mean():.2f} ± {place_cell_peak_rates.std():.2f} Hz")

print(f"\nNon-place cell statistics:")
non_place_indices = np.where(~place_cell_mask)[0]
non_place_si = spatial_info_values[~place_cell_mask]
non_place_peaks = np.array(peak_firing_rates)[~place_cell_mask]
print(f"  Number of non-place cells: {len(non_place_indices)}")
print(f"  Mean SI (non-place cells): {non_place_si.mean():.3f} ± {non_place_si.std():.3f} bits/spike")
print(f"  Mean peak firing rate (non-place cells): {non_place_peaks.mean():.2f} ± {non_place_peaks.std():.2f} Hz")

print("\n" + "="*60)
print("Analysis complete. All figures saved.")
print("="*60)

io.close()
