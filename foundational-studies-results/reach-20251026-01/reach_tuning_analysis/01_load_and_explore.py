# %% [markdown]
# # Load and Explore NWB Data for Reach Tuning Analysis
# This notebook loads the NWB file and explores its structure

# %%
import h5py
from pynwb import NWBHDF5IO
import remfile
import pynapple as nap
import numpy as np
import matplotlib.pyplot as plt

# %%
# Setup remote file access with caching
s3_url = "https://api.dandiarchive.org/api/assets/f81f938d-cb68-4a6b-bc86-36a918423362/download/"
cache_dirname = '/tmp/remfile_cache_dandi_000688'
disk_cache = remfile.DiskCache(cache_dirname)

print(f"Loading NWB file from: {s3_url}")
print(f"Using cache directory: {cache_dirname}")

# %%
# Open the file from S3
rem_file = remfile.File(s3_url, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)

# %%
print("\n=== NWB File Contents ===")
print(nwb)

# %%
print("\n=== Session Description ===")
print(f"Session: {nwbfile.session_description}")
print(f"Identifier: {nwbfile.identifier}")
print(f"Session start time: {nwbfile.session_start_time}")

# %%
print("\n=== Units (Spike Data) ===")
print(f"Number of units: {len(nwb['units'])}")
print(f"Available columns: {list(nwbfile.units.colnames)}")

# %%
print("\n=== Trials ===")
trials = nwb['trials']
print(f"Number of trials: {len(trials)}")
print(f"Trial intervals:\n{trials}")

# %%
print("\n=== Cursor Position ===")
cursor_pos = nwb['cursor_pos']
print(f"Shape: {cursor_pos.shape}")
print(f"Time range: {cursor_pos.index[0]:.2f}s to {cursor_pos.index[-1]:.2f}s")
print(f"Columns: {cursor_pos.columns}")

# %%
print("\n=== Cursor Velocity ===")
cursor_vel = nwb['cursor_vel']
print(f"Shape: {cursor_vel.shape}")
print(f"Time range: {cursor_vel.index[0]:.2f}s to {cursor_vel.index[-1]:.2f}s")
print(f"Columns: {cursor_vel.columns}")

# %%
# Plot cursor trajectories
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot first 10 trials
n_trials_to_plot = min(10, len(trials))
colors = plt.cm.viridis(np.linspace(0, 1, n_trials_to_plot))

for i in range(n_trials_to_plot):
    trial_start = trials.start[i]
    trial_end = trials.end[i]
    trial_pos = cursor_pos.get(trial_start, trial_end)
    axes[0].plot(trial_pos[:, 0], trial_pos[:, 1], color=colors[i], alpha=0.7, label=f'Trial {i+1}')

axes[0].set_xlabel('X Position')
axes[0].set_ylabel('Y Position')
axes[0].set_title(f'Cursor Trajectories (First {n_trials_to_plot} Trials)')
axes[0].axis('equal')
axes[0].grid(True, alpha=0.3)

# Plot speed over time for first trial
trial_start = trials.start[0]
trial_end = trials.end[0]
trial_pos = cursor_pos.get(trial_start, trial_end)
trial_vel = cursor_vel.get(trial_start, trial_end)
speed = np.sqrt(trial_vel.values[:, 0]**2 + trial_vel.values[:, 1]**2)

axes[1].plot(trial_vel.index - trial_start, speed)
axes[1].set_xlabel('Time from trial start (s)')
axes[1].set_ylabel('Speed (arbitrary units)')
axes[1].set_title('Movement Speed - Trial 1')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fig_01_cursor_trajectories.png', dpi=150, bbox_inches='tight')
print("\nFigure saved: fig_01_cursor_trajectories.png")
plt.show()

# %%
# Plot neural activity for a few units
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Select 3 example units
units = nwb['units']
unit_indices = [0, 10, 20]

for idx, unit_idx in enumerate(unit_indices):
    unit_spikes = units[unit_idx]
    
    # Plot raster for first 10 trials
    for trial_num in range(n_trials_to_plot):
        trial_start = trials.start[trial_num]
        trial_end = trials.end[trial_num]
        trial_spikes = unit_spikes.get(trial_start, trial_end)
        spike_times = trial_spikes.index - trial_start
        axes[idx].scatter(spike_times, np.ones_like(spike_times) * trial_num, 
                         marker='|', s=100, color='black', alpha=0.7)
    
    axes[idx].set_ylabel('Trial Number')
    axes[idx].set_title(f'Unit {unit_idx} Spike Raster (First {n_trials_to_plot} Trials)')
    axes[idx].set_ylim(-0.5, n_trials_to_plot - 0.5)
    axes[idx].grid(True, alpha=0.3, axis='x')

axes[-1].set_xlabel('Time from trial start (s)')
plt.tight_layout()
plt.savefig('fig_02_spike_rasters.png', dpi=150, bbox_inches='tight')
print("Figure saved: fig_02_spike_rasters.png")
plt.show()

print("\n=== Data exploration complete ===")
