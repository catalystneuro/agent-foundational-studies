# %% [markdown]
# # Reach Direction Tuning Analysis
# Demonstrate how motor cortex neurons are tuned to reach direction

# %%
import h5py
from pynwb import NWBHDF5IO
import remfile
import pynapple as nap
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import circmean

# %%
# Setup remote file access with caching
s3_url = "https://api.dandiarchive.org/api/assets/f81f938d-cb68-4a6b-bc86-36a918423362/download/"
cache_dirname = '/tmp/remfile_cache_dandi_000688'
disk_cache = remfile.DiskCache(cache_dirname)

print(f"Loading NWB file from: {s3_url}")

# %%
# Open the file from S3
rem_file = remfile.File(s3_url, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)

# %%
# Extract data
trials = nwb['trials']
units = nwb['units']
cursor_pos = nwb['cursor_pos']
cursor_vel = nwb['cursor_vel']

print(f"Number of units: {len(units)}")
print(f"Number of trials: {len(trials)}")

# %%
# Get trial metadata - check if target_dir is available
print("\nAvailable trial metadata columns:")
print(trials.columns)

# Extract target direction from metadata
target_dirs = trials['target_dir'].values

# Filter out NaN trials (aborted trials)
valid_trials = ~np.isnan(target_dirs)
print(f"\nValid trials with direction: {np.sum(valid_trials)} / {len(trials)}")

# %%
# Compute firing rates for each unit during movement period
# Define movement period as from go cue to end of trial

def compute_firing_rates_by_direction(units, trials, valid_trials, target_dirs):
    """Compute mean firing rate for each unit for each reach direction"""
    
    # Get unique directions
    unique_dirs = np.unique(target_dirs[valid_trials])
    n_units = len(units)
    n_dirs = len(unique_dirs)
    
    # Initialize firing rate matrix (units x directions)
    firing_rates = np.zeros((n_units, n_dirs))
    
    for unit_idx in range(n_units):
        unit_spikes = units[unit_idx]
        
        for dir_idx, direction in enumerate(unique_dirs):
            # Get all trials for this direction
            dir_trials = valid_trials & (target_dirs == direction)
            trial_indices = np.where(dir_trials)[0]
            
            # Compute firing rate for each trial of this direction
            rates = []
            for trial_idx in trial_indices:
                trial_start = trials.start[trial_idx]
                trial_end = trials.end[trial_idx]
                
                # Get go cue time (start of movement)
                go_cue = trials['go_cue_time'].values[trial_idx]
                if not np.isnan(go_cue):
                    movement_start = go_cue
                else:
                    movement_start = trial_start
                
                # Count spikes during movement period
                trial_spikes = unit_spikes.get(movement_start, trial_end)
                duration = trial_end - movement_start
                
                if duration > 0:
                    rate = len(trial_spikes) / duration
                    rates.append(rate)
            
            # Average across trials for this direction
            if len(rates) > 0:
                firing_rates[unit_idx, dir_idx] = np.mean(rates)
    
    return firing_rates, unique_dirs

print("\nComputing firing rates by direction...")
firing_rates, unique_dirs = compute_firing_rates_by_direction(units, trials, valid_trials, target_dirs)

print(f"Firing rate matrix shape: {firing_rates.shape}")
print(f"Unique directions (radians): {unique_dirs}")
print(f"Unique directions (degrees): {np.degrees(unique_dirs)}")

# %%
# Plot tuning curves for example neurons
fig, axes = plt.subplots(2, 3, figsize=(15, 10), subplot_kw=dict(projection='polar'))
axes = axes.flatten()

# Select neurons with good direction selectivity
# Compute direction selectivity index for each neuron
selectivity = np.std(firing_rates, axis=1) / (np.mean(firing_rates, axis=1) + 1e-10)
top_units = np.argsort(selectivity)[::-1][:6]

for idx, unit_idx in enumerate(top_units):
    ax = axes[idx]
    
    # Sort directions for smooth plotting
    sorted_idx = np.argsort(unique_dirs)
    dirs_sorted = unique_dirs[sorted_idx]
    rates_sorted = firing_rates[unit_idx, sorted_idx]
    
    # Close the polar plot by appending first point
    dirs_plot = np.append(dirs_sorted, dirs_sorted[0])
    rates_plot = np.append(rates_sorted, rates_sorted[0])
    
    # Plot tuning curve
    ax.plot(dirs_plot, rates_plot, 'o-', linewidth=2, markersize=8)
    ax.fill(dirs_plot, rates_plot, alpha=0.25)
    
    # Compute preferred direction
    preferred_dir = dirs_sorted[np.argmax(rates_sorted)]
    preferred_rate = np.max(rates_sorted)
    
    ax.plot([preferred_dir], [preferred_rate], 'r*', markersize=15)
    ax.set_title(f'Unit {unit_idx}\\nPref. Dir: {np.degrees(preferred_dir):.0f}°', fontsize=10)
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(1)
    
plt.suptitle('Direction Tuning Curves - Top 6 Selective Neurons', fontsize=14, y=0.98)
plt.tight_layout()
plt.savefig('fig_03_direction_tuning_curves.png', dpi=150, bbox_inches='tight')
print("\nFigure saved: fig_03_direction_tuning_curves.png")
plt.show()

# %%
# Compute population-level statistics
print("\n=== Direction Tuning Statistics ===")

# Preferred direction for each neuron
preferred_dirs = unique_dirs[np.argmax(firing_rates, axis=1)]

# Modulation depth (max - min) / (max + min)
max_rates = np.max(firing_rates, axis=1)
min_rates = np.min(firing_rates, axis=1)
modulation_depth = (max_rates - min_rates) / (max_rates + min_rates + 1e-10)

print(f"Mean modulation depth: {np.mean(modulation_depth):.3f} ± {np.std(modulation_depth):.3f}")
print(f"Neurons with modulation > 0.3: {np.sum(modulation_depth > 0.3)} / {len(units)}")

# %%
# Population visualization - preferred direction distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Polar histogram of preferred directions
ax = plt.subplot(121, projection='polar')
n_bins = 16
hist, bin_edges = np.histogram(preferred_dirs, bins=n_bins, range=(0, 2*np.pi))
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
width = 2 * np.pi / n_bins

bars = ax.bar(bin_centers, hist, width=width, alpha=0.7, edgecolor='black')
ax.set_theta_zero_location('E')
ax.set_theta_direction(1)
ax.set_title('Distribution of Preferred Directions', fontsize=12)

# Modulation depth distribution
ax = axes[1]
ax.hist(modulation_depth, bins=20, alpha=0.7, edgecolor='black')
ax.axvline(0.3, color='r', linestyle='--', linewidth=2, label='Threshold (0.3)')
ax.set_xlabel('Modulation Depth', fontsize=12)
ax.set_ylabel('Number of Neurons', fontsize=12)
ax.set_title('Distribution of Direction Selectivity', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fig_04_population_direction_tuning.png', dpi=150, bbox_inches='tight')
print("Figure saved: fig_04_population_direction_tuning.png")
plt.show()

print("\n=== Direction tuning analysis complete ===")
