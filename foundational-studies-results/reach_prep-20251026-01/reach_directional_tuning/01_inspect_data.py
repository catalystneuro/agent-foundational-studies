# %% [markdown]
# # Data Inspection: Reach Planning Dataset
# 
# This script loads and inspects the NWB file containing neural recordings
# during a delayed reaching task from DANDI:000140.

# %%
import numpy as np
import matplotlib.pyplot as plt
import h5py
from pynwb import NWBHDF5IO
import lindi

print("Loading NWB file with LINDI...")

# Create a local cache for efficient data access
local_cache = lindi.LocalCache()

# Load the LINDI file (JSON reference to the NWB file)
lindi_url = "https://lindi.neurosift.org/dandi/dandisets/000140/assets/7821971e-c6a4-4568-8773-1bfa205c13f8/nwb.lindi.json"
f = lindi.LindiH5pyFile.from_lindi_file(lindi_url, local_cache=local_cache)

# Open as NWB file
io = NWBHDF5IO(file=f, mode='r')
nwbfile = io.read()

# %% [markdown]
# ## Dataset Metadata

# %%
print("=" * 60)
print("NWB FILE METADATA")
print("=" * 60)
print(f"Session description: {nwbfile.session_description}")
print(f"Identifier: {nwbfile.identifier}")
print(f"Session start time: {nwbfile.session_start_time}")
print(f"Experiment description: {nwbfile.experiment_description}")
print(f"Institution: {nwbfile.institution}")
print(f"Lab: {nwbfile.lab}")
print(f"\nSubject: {nwbfile.subject.subject_id}")
print(f"Species: {nwbfile.subject.species}")
print(f"Sex: {nwbfile.subject.sex}")

# %% [markdown]
# ## Trial Information

# %%
trials = nwbfile.intervals['trials']
print("\n" + "=" * 60)
print("TRIAL INFORMATION")
print("=" * 60)
print(f"Number of trials: {len(trials)}")
print(f"Trial columns: {trials.colnames}")

# Extract trial data  
# Some arrays are ragged, so keep as lists initially
target_pos = trials['target_pos'][:]
target_pos_index = trials['target_pos_index'][:]
active_target = trials['active_target'][:]
go_cue_times = np.array(trials['go_cue_time'][:])
target_on_times = np.array(trials['target_on_time'][:])
move_onset_times = np.array(trials['move_onset_time'][:])

# Debug: check the structure
print(f"\nDEBUG: target_pos type: {type(target_pos)}, len: {len(target_pos)}")
print(f"DEBUG: target_pos[0] shape: {np.array(target_pos[0]).shape}")
print(f"DEBUG: target_pos[0]: {target_pos[0]}")
print(f"DEBUG: target_pos_index[0]: {target_pos_index[0]}")
print(f"DEBUG: active_target[0]: {active_target[0]}")

# Calculate reach directions for each trial
# For this dataset, each trial has its own set of target positions
# and active_target tells us which one was the actual target
reach_directions = []
target_positions = []  # Store actual x,y coordinates

for i in range(len(trials)):
    # Get the array of target positions for this trial
    trial_targets = np.array(target_pos[i])
    # Get which target was active (0-indexed)
    active_idx = int(active_target[i])
    # Get the actual target coordinates
    target_xy = trial_targets[active_idx]
    target_positions.append(target_xy)
    # Calculate angle from center (assuming center is at origin)
    angle = np.arctan2(target_xy[1], target_xy[0])
    reach_directions.append(np.degrees(angle))

reach_directions = np.array(reach_directions)
target_positions = np.array(target_positions)

# Count unique target positions
unique_targets = []
for pos in target_positions:
    is_unique = True
    for upos in unique_targets:
        if np.allclose(pos, upos):
            is_unique = False
            break
    if is_unique:
        unique_targets.append(pos)

print(f"\nUnique target positions: {len(unique_targets)}")
print(f"Reach direction range: {reach_directions.min():.1f}° to {reach_directions.max():.1f}°")
print(f"Unique reach directions: {len(np.unique(np.round(reach_directions, 1)))}")

# Calculate delay periods
delay_durations = go_cue_times - target_on_times
print(f"\nDelay period duration:")
print(f"  Mean: {np.mean(delay_durations):.3f} seconds")
print(f"  Range: {np.min(delay_durations):.3f} to {np.max(delay_durations):.3f} seconds")

# %% [markdown]
# ## Neural Units

# %%
units = nwbfile.units
print("\n" + "=" * 60)
print("UNITS (NEURONS)")
print("=" * 60)
print(f"Number of units: {len(units)}")
print(f"Unit columns: {units.colnames}")

# Check if units have location information
if 'electrodes' in units.colnames:
    print("\nUnits have electrode location information")
    
# Look at first unit's spike times
unit_0_spikes = units['spike_times'][0]
print(f"\nFirst unit spike times (first 10): {unit_0_spikes[:10]}")
print(f"Total spikes in first unit: {len(unit_0_spikes)}")

# %% [markdown]
# ## Visualize Trial Distribution

# %%
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Target positions
ax = axes[0, 0]
# Plot all unique target positions
for i, upos in enumerate(unique_targets):
    ax.scatter(upos[0], upos[1], s=200, alpha=0.7, label=f'Target {i+1}')
ax.scatter(0, 0, s=300, marker='x', c='red', linewidths=3, label='Center')
ax.set_xlabel('X Position (mm)')
ax.set_ylabel('Y Position (mm)')
ax.set_title('Target Positions')
ax.grid(True, alpha=0.3)
ax.axis('equal')
if len(unique_targets) <= 10:
    ax.legend()

# Plot 2: Reach direction histogram
ax = axes[0, 1]
ax.hist(reach_directions, bins=20, edgecolor='black', alpha=0.7)
ax.set_xlabel('Reach Direction (degrees)')
ax.set_ylabel('Number of Trials')
ax.set_title('Distribution of Reach Directions')
ax.grid(True, alpha=0.3)

# Plot 3: Delay duration histogram
ax = axes[1, 0]
ax.hist(delay_durations, bins=20, edgecolor='black', alpha=0.7, color='orange')
ax.set_xlabel('Delay Duration (seconds)')
ax.set_ylabel('Number of Trials')
ax.set_title('Distribution of Delay Periods')
ax.grid(True, alpha=0.3)

# Plot 4: Polar plot of reach directions
ax = axes[1, 1]
ax = plt.subplot(2, 2, 4, projection='polar')
angles_rad = np.radians(reach_directions)
ax.hist(angles_rad, bins=20, alpha=0.7, color='green')
ax.set_title('Reach Directions (Polar)')
ax.set_theta_zero_location('E')
ax.set_theta_direction(1)

plt.tight_layout()
plt.savefig('trial_distribution.png', dpi=150, bbox_inches='tight')
print("\nFigure saved: trial_distribution.png")
plt.close()

# %% [markdown]
# ## Summary

# %%
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"This dataset contains {len(units)} neurons recorded from M1 and PMd")
print(f"during {len(trials)} trials of delayed reaching.")
print(f"Reaches span {len(np.unique(np.round(reach_directions, 1)))} different directions.")
print(f"The delay period (planning phase) averages {np.mean(delay_durations):.3f} seconds.")
print("Data is suitable for directional tuning analysis during reach planning.")
print("=" * 60)
