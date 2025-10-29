"""
Script to inspect the NWB file structure and data from DANDI:000940
This helps understand what data is available before performing the analysis.
"""

import h5py
from pynwb import NWBHDF5IO
import lindi
import numpy as np
import matplotlib.pyplot as plt

# Load the NWB file using LINDI with local caching
print("Loading NWB file with LINDI...")
local_cache = lindi.LocalCache()
f = lindi.LindiH5pyFile.from_lindi_file(
    "https://lindi.neurosift.org/dandi/dandisets/000940/assets/612c80ff-c103-4725-88f5-12215598dcbe/nwb.lindi.json",
    local_cache=local_cache
)

io = NWBHDF5IO(file=f, mode='r')
nwbfile = io.read()

print("\n" + "="*60)
print("NWB FILE METADATA")
print("="*60)
print(f"Session description: {nwbfile.session_description}")
print(f"Identifier: {nwbfile.identifier}")
print(f"Session start time: {nwbfile.session_start_time}")
print(f"Experiment description: {nwbfile.experiment_description}")
print(f"Institution: {nwbfile.institution}")
print(f"Lab: {nwbfile.lab}")
print(f"\nSubject: {nwbfile.subject.subject_id}")
print(f"Species: {nwbfile.subject.species}")
print(f"Age: {nwbfile.subject.age}")
print(f"Sex: {nwbfile.subject.sex}")

# Inspect LFP data
print("\n" + "="*60)
print("LFP DATA")
print("="*60)
lfps = nwbfile.acquisition['LFPs']
print(f"LFP sampling rate: {lfps.rate} Hz")
print(f"LFP data shape: {lfps.data.shape}")
print(f"Number of electrodes: {len(lfps.electrodes)}")
print(f"Recording duration: {lfps.data.shape[0] / lfps.rate:.2f} seconds")
print(f"Starting time: {lfps.starting_time} seconds")

# Get electrode locations
electrodes = nwbfile.electrodes.to_dataframe()
print(f"\nElectrode locations:")
for idx, row in electrodes.iterrows():
    print(f"  Electrode {idx}: {row['location']}")

# Inspect unit (spike) data
print("\n" + "="*60)
print("UNITS (NEURONS)")
print("="*60)
units = nwbfile.units.to_dataframe()
print(f"Number of units: {len(units)}")
print(f"Unit columns: {units.columns.tolist()}")

# Show spike statistics for each unit
print(f"\nSpike statistics per unit:")
for idx, row in units.iterrows():
    spike_times = row['spike_times']
    print(f"  Unit {idx}: {len(spike_times)} spikes, "
          f"rate = {len(spike_times) / (lfps.data.shape[0] / lfps.rate):.2f} Hz")

# Inspect trial structure
print("\n" + "="*60)
print("TRIAL STRUCTURE")
print("="*60)
print("\nEncoding trials:")
encoding = nwbfile.intervals['encoding_table'].to_dataframe()
print(f"Number of encoding trials: {len(encoding)}")
print(f"Columns: {encoding.columns.tolist()}")
print(f"\nStimulus categories:")
print(encoding['stimCategory'].value_counts())

# Create visualization of data structure
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# Plot 1: LFP trace from first electrode (10 seconds)
ax = axes[0]
lfp_data = lfps.data[:2500, 0]  # First 10 seconds, first electrode
time = np.arange(len(lfp_data)) / lfps.rate + lfps.starting_time
ax.plot(time, lfp_data, 'b-', linewidth=0.5)
ax.set_xlabel('Time (s)')
ax.set_ylabel('LFP (μV)')
ax.set_title('Sample LFP Recording (First 10 seconds, Electrode 0)')
ax.grid(True, alpha=0.3)

# Plot 2: Spike raster for first 5 units (first 100 seconds)
ax = axes[1]
time_window = 100
for unit_idx in range(min(5, len(units))):
    spike_times = units.iloc[unit_idx]['spike_times']
    # Convert spike times to absolute times
    spike_times_abs = spike_times + lfps.starting_time
    # Filter to time window
    spikes_in_window = spike_times_abs[spike_times_abs < (lfps.starting_time + time_window)]
    ax.plot(spikes_in_window, np.ones_like(spikes_in_window) * unit_idx, 'k|', markersize=2)

ax.set_xlabel('Time (s)')
ax.set_ylabel('Unit ID')
ax.set_title(f'Spike Raster (First {min(5, len(units))} units, First {time_window} seconds)')
ax.set_ylim(-0.5, min(5, len(units)) - 0.5)
ax.grid(True, alpha=0.3)

# Plot 3: Trial timeline
ax = axes[2]
for idx, row in encoding.iterrows():
    start = row['start_time']
    stop = row['stop_time']
    category = row['stimCategory']
    color = ['blue', 'green', 'red'][int(category) - 1]
    ax.barh(0, stop - start, left=start, height=0.8, 
            color=color, alpha=0.6, edgecolor='black')

ax.set_xlabel('Time (s)')
ax.set_ylabel('Trials')
ax.set_title('Trial Timeline (Blue=NB, Green=SB, Red=HB)')
ax.set_ylim(-0.5, 0.5)
ax.set_yticks([])

plt.tight_layout()
plt.savefig('data_overview.png', dpi=150, bbox_inches='tight')
print("\nFigure saved: data_overview.png")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"This dataset contains {len(units)} hippocampal neurons")
print(f"recorded during {len(encoding)} encoding trials.")
print(f"LFP data is sampled at {lfps.rate} Hz from {len(lfps.electrodes)} electrodes.")
print(f"Total recording duration: {lfps.data.shape[0] / lfps.rate:.2f} seconds")
print("\nData is suitable for theta phase entrainment and precession analysis.")
print("="*60)

# Close the file
io.close()
