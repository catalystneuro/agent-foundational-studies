"""
Load and explore hippocampal place cell data from DANDI Archive.

This script loads an NWB file from DANDI:000447 containing CA1 recordings
during W-track navigation and inspects the data structure.
"""

import h5py
import remfile
from pynwb import NWBHDF5IO
import pynapple as nap
import numpy as np
import matplotlib.pyplot as plt

# Configuration
ASSET_URL = "https://api.dandiarchive.org/api/assets/b048bd0a-48a4-4d19-ad96-fd879666bca5/download/"
CACHE_DIR = "/tmp/remfile_cache_dandi_000447"

print("Loading NWB file from DANDI Archive...")
print(f"URL: {ASSET_URL}")
print(f"Using cache directory: {CACHE_DIR}")

# Create disk cache and load file
disk_cache = remfile.DiskCache(CACHE_DIR)
rem_file = remfile.File(ASSET_URL, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file, mode='r')
nwbfile = io.read()

# Load with pynapple
nwb = nap.NWBFile(nwbfile)
print("\n=== NWB File Contents ===")
print(nwb)

# Session information
print("\n=== Session Description ===")
print(f"Session: {nwbfile.session_description}")
print(f"Subject: {nwbfile.subject.subject_id}")
print(f"Species: {nwbfile.subject.species}")
print(f"Session start: {nwbfile.session_start_time}")
print(f"Identifier: {nwbfile.identifier}")

# Units (spike data)
print("\n=== Units (Spike Data) ===")
units = nwb['units']
print(f"Number of units: {len(units)}")
print(f"Type: {type(units)}")

# Look at spike times for first few units
print("\nSpike times for first 3 units:")
for i in range(min(3, len(units))):
    spikes = units[i]
    print(f"  Unit {i}: {len(spikes)} spikes, range: {spikes.t[0]:.2f}s to {spikes.t[-1]:.2f}s")

# Position data
print("\n=== Position Data ===")
position_obj = nwbfile.processing['behavior']['Position']
print(f"Available spatial series: {list(position_obj.spatial_series.keys())}")

# Access position through pynapple - check what's available
print("\nChecking position data in pynapple object...")
for key in nwb.keys():
    if 'pos' in key.lower() or 'position' in key.lower():
        print(f"  Found: {key} - {type(nwb[key])}")

# Try to access position data directly from pynwb
spatial_series = position_obj.spatial_series['SpatialSeries']
print(f"\nSpatialSeries data shape: {spatial_series.data.shape}")
print(f"SpatialSeries timestamps shape: {spatial_series.timestamps.shape}")

# Get position as numpy arrays
pos_data = spatial_series.data[:]
pos_times = spatial_series.timestamps[:]
print(f"Position data range: x=[{pos_data[:, 0].min():.2f}, {pos_data[:, 0].max():.2f}], y=[{pos_data[:, 1].min():.2f}, {pos_data[:, 1].max():.2f}]")
print(f"Time range: {pos_times[0]:.2f}s to {pos_times[-1]:.2f}s")
print(f"Duration: {pos_times[-1] - pos_times[0]:.2f}s")

# Epoch information
print("\n=== Epoch Information ===")
epochs = nwbfile.intervals['epoch intervals']
print(f"Number of epochs: {len(epochs)}")
for i in range(len(epochs)):
    print(f"  Epoch {i}: {epochs['start_time'][i]:.2f}s - {epochs['stop_time'][i]:.2f}s ({epochs['stop_time'][i] - epochs['start_time'][i]:.2f}s)")

# Trial information
print("\n=== Trial Information ===")
trials = nwbfile.intervals['trials']
print(f"Number of trials: {len(trials)}")
print(f"Trial columns: {trials.colnames}")
print(f"Example trial data:")
print(f"  Trial 0: start={trials['start_time'][0]:.2f}s, stop={trials['stop_time'][0]:.2f}s")
print(f"  Trial 0: start_well={trials['start_well'][0]}, end_well={trials['end_well'][0]}, correct={trials['correct'][0]}")
print(f"  Trial 0: trajectory_type={trials['trajectory_type'][0]} (0=Outbound, 1=Inbound)")

# Electrode information
print("\n=== Electrode Information ===")
electrodes = nwbfile.electrodes
print(f"Number of electrodes: {len(electrodes)}")
print(f"Electrode columns: {electrodes.colnames}")
# Get unique locations
locations = electrodes['location'][:]
unique_locations = np.unique(locations)
print(f"Brain regions: {unique_locations}")
for loc in unique_locations:
    count = np.sum(locations == loc)
    print(f"  {loc}: {count} electrodes")

# Create a simple visualization
print("\n=== Creating Visualization ===")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Position trajectory
ax = axes[0, 0]
ax.plot(pos_data[:, 0], pos_data[:, 1], 'k-', alpha=0.3, linewidth=0.5)
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_title('W-Track Trajectory')
ax.axis('equal')
ax.grid(True, alpha=0.3)

# Plot 2: Position over time
ax = axes[0, 1]
ax.plot(pos_times, pos_data[:, 0], label='X', alpha=0.7)
ax.plot(pos_times, pos_data[:, 1], label='Y', alpha=0.7)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Position')
ax.set_title('Position Over Time')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Spike raster for first 10 units
ax = axes[1, 0]
for i in range(min(10, len(units))):
    spikes = units[i]
    # Only plot first 100 seconds for clarity
    mask = spikes.t < 100
    ax.plot(spikes.t[mask], np.ones(np.sum(mask)) * i, '|', markersize=2, color='black')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Unit #')
ax.set_title('Spike Raster (first 10 units, first 100s)')
ax.set_ylim(-0.5, 9.5)
ax.grid(True, alpha=0.3)

# Plot 4: Firing rate distribution
ax = axes[1, 1]
firing_rates = []
for i in range(len(units)):
    spikes = units[i]
    duration = spikes.t[-1] - spikes.t[0]
    firing_rates.append(len(spikes) / duration)
ax.hist(firing_rates, bins=20, edgecolor='black')
ax.set_xlabel('Firing Rate (Hz)')
ax.set_ylabel('Count')
ax.set_title('Firing Rate Distribution')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fig_01_data_exploration.png', dpi=150, bbox_inches='tight')
print("Saved: fig_01_data_exploration.png")

print("\n=== Exploration complete ===")
