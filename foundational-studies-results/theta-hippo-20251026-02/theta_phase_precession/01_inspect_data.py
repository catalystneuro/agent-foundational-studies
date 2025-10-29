# %% [markdown]
# # Data Inspection: Hippocampal Place Cell Recording
#
# This script explores the NWB file structure from DANDI:000059 to understand:
# - Position tracking data
# - Unit (neuron) recordings
# - Trial structure
# - Available LFP data for theta extraction

# %%
import lindi
import pynwb
import numpy as np
import matplotlib.pyplot as plt
from pynwb import NWBHDF5IO

# %%
print("Loading NWB file with LINDI...")

# URL for the processed behavior+ecephys file
lindi_url = "https://lindi.neurosift.org/dandi/dandisets/000059/assets/931b05e6-9123-4c66-a9cf-49de6cec8086/nwb.lindi.json"

# Create a local cache for efficient data access
local_cache = lindi.LocalCache()

# Load the file
f = lindi.LindiH5pyFile.from_lindi_file(lindi_url, local_cache=local_cache)
io = NWBHDF5IO(file=f, mode='r')
nwb = io.read()

# %%
print("\n" + "="*60)
print("NWB FILE METADATA")
print("="*60)
print(f"Session description: {nwb.session_description}")
print(f"Identifier: {nwb.identifier}")
print(f"Session start time: {nwb.session_start_time}")
print(f"Institution: {nwb.institution}")
print(f"Lab: {nwb.lab}")
print(f"\nSubject: {nwb.subject.subject_id}")
print(f"Species: {nwb.subject.species}")
print(f"Sex: {nwb.subject.sex}")
print(f"Age: {nwb.subject.age}")

# %%
print("\n" + "="*60)
print("POSITION DATA")
print("="*60)

# Access position data
position = nwb.processing["behavior"]["SubjectPosition"]

# Check what spatial series are available
print(f"Available spatial series: {list(position.spatial_series.keys())}")

# Get the first available spatial series
spatial_series_name = list(position.spatial_series.keys())[0]
spatial_series = position.spatial_series[spatial_series_name]

pos_data = spatial_series.data[:]
pos_timestamps = spatial_series.timestamps[:]

print(f"Position data shape: {pos_data.shape}")
print(f"Number of time points: {len(pos_timestamps)}")
print(f"Position sampling rate: ~{1/np.mean(np.diff(pos_timestamps)):.2f} Hz")
print(f"Recording duration: {pos_timestamps[-1] - pos_timestamps[0]:.2f} seconds")

# Filter out NaN values for range calculation
valid_mask = ~np.isnan(pos_data[:, 0]) & ~np.isnan(pos_data[:, 1])
valid_pos = pos_data[valid_mask]
print(f"Valid position samples: {np.sum(valid_mask)} / {len(pos_data)}")
print(f"\nPosition range (valid samples):")
print(f"  X: {valid_pos[:, 0].min():.2f} to {valid_pos[:, 0].max():.2f} cm")
print(f"  Y: {valid_pos[:, 1].min():.2f} to {valid_pos[:, 1].max():.2f} cm")

# %%
print("\n" + "="*60)
print("SPEED DATA")
print("="*60)

speed = nwb.processing["behavior"]["SubjectSpeed"]
speed_data = speed.data[:]
speed_timestamps = speed.timestamps[:]

print(f"Speed data points: {len(speed_data)}")
print(f"Mean speed: {np.mean(speed_data):.2f} cm/s")
print(f"Speed range: {speed_data.min():.2f} to {speed_data.max():.2f} cm/s")

# %%
print("\n" + "="*60)
print("TRIAL INFORMATION")
print("="*60)

trials = nwb.intervals["trials"]
print(f"Number of trials: {len(trials)}")
print(f"Trial columns: {trials.colnames}")

# Extract trial conditions
conditions = [str(c) for c in trials["condition"][:]]
cooling_states = [str(c) for c in trials["cooling state"][:]]
errors = trials["error"][:]

print(f"\nTrial conditions:")
for cond in set(conditions):
    count = conditions.count(cond)
    print(f"  {cond}: {count} trials")

print(f"\nCooling states:")
for state in set(cooling_states):
    count = cooling_states.count(state)
    print(f"  {state}: {count} trials")

print(f"\nError trials: {np.sum(errors)} / {len(errors)}")

# %%
print("\n" + "="*60)
print("UNITS (NEURONS)")
print("="*60)

units = nwb.units
print(f"Number of units: {len(units)}")
print(f"Unit columns: {units.colnames}")

# Get quality labels
qualities = [str(q) for q in units["quality"][:]]
quality_counts = {}
for q in set(qualities):
    quality_counts[q] = qualities.count(q)

print(f"\nUnit quality distribution:")
for q, count in quality_counts.items():
    print(f"  {q}: {count} units")

# Look at spike counts
spike_counts = []
for i in range(len(units)):
    spike_times = units["spike_times"][i]
    spike_counts.append(len(spike_times))

print(f"\nSpike statistics:")
print(f"  Mean spikes per unit: {np.mean(spike_counts):.0f}")
print(f"  Median spikes per unit: {np.median(spike_counts):.0f}")
print(f"  Range: {np.min(spike_counts)} to {np.max(spike_counts)} spikes")

# %%
print("\n" + "="*60)
print("ELECTRODES")
print("="*60)

electrodes = nwb.electrodes
print(f"Number of electrodes: {len(electrodes)}")
print(f"Electrode columns: {electrodes.colnames}")

# Get electrode locations
locations = [str(loc) for loc in electrodes["location"][:]]
location_counts = {}
for loc in set(locations):
    location_counts[loc] = locations.count(loc)

print(f"\nElectrode locations:")
for loc, count in sorted(location_counts.items()):
    print(f"  {loc}: {count} electrodes")

# Check for theta reference
theta_refs = electrodes["theta_reference"][:]
print(f"\nTheta reference electrodes: {np.sum(theta_refs)}")

# %% [markdown]
# ## Visualizations

# %%
# Create visualization figure
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Position trajectory
ax = axes[0, 0]
# Plot only valid positions
ax.plot(valid_pos[::10, 0], valid_pos[::10, 1], 'b-', alpha=0.3, linewidth=0.5)
ax.set_xlabel('X position (cm)')
ax.set_ylabel('Y position (cm)')
ax.set_title('Animal Trajectory')
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

# Plot 2: Speed over time
ax = axes[0, 1]
time_subsample = slice(0, len(speed_data), 100)
ax.plot(speed_timestamps[time_subsample], speed_data[time_subsample], 'g-', linewidth=0.5)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Speed (cm/s)')
ax.set_title('Speed Over Time')
ax.grid(True, alpha=0.3)

# Plot 3: Spike count distribution
ax = axes[1, 0]
ax.hist(spike_counts, bins=50, edgecolor='black', alpha=0.7)
ax.set_xlabel('Number of spikes')
ax.set_ylabel('Number of units')
ax.set_title('Spike Count Distribution')
ax.grid(True, alpha=0.3)

# Plot 4: Trial structure
ax = axes[1, 1]
trial_starts = trials["start_time"][:]
trial_stops = trials["stop_time"][:]
trial_durations = trial_stops - trial_starts

ax.bar(range(len(trial_durations)), trial_durations, 
       color=['red' if e else 'blue' for e in errors],
       alpha=0.7)
ax.set_xlabel('Trial number')
ax.set_ylabel('Trial duration (s)')
ax.set_title('Trial Durations (Red = Error)')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('data_overview.png', dpi=150, bbox_inches='tight')
print("\nFigure saved: data_overview.png")

# %%
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"This dataset contains {len(units)} hippocampal neurons")
print(f"recorded during {len(trials)} trials of spatial navigation.")
print(f"Position is tracked at ~{1/np.mean(np.diff(pos_timestamps)):.1f} Hz")
print(f"Total recording duration: {(pos_timestamps[-1] - pos_timestamps[0]):.1f} seconds")
print(f"\nData is suitable for theta phase precession analysis.")
print("="*60)
