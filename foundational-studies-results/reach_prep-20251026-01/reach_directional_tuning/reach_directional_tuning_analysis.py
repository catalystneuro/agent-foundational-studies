# %% [markdown]
# # Directional Tuning During Reach Planning
#
# ## Overview
# This analysis demonstrates directional tuning in motor and premotor cortical neurons 
# during the planning phase of reaching movements using data from DANDI:000140.
#
# **Dataset**: MC_Maze_Small - macaque primary motor and dorsal premotor cortex 
# spiking activity during delayed reaching
#
# **Key Question**: Do neurons in M1 and PMd show directional tuning during the 
# delay period (planning phase) before movement execution?
#
# **Analysis Window**: From 100ms after target onset to 100ms before go cue

# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import h5py
from pynwb import NWBHDF5IO
import lindi
from scipy import stats
from tqdm import tqdm

# %% [markdown]
# ## Load NWB File
#
# We use LINDI to efficiently stream data from the DANDI Archive

# %%
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
print(f"Experiment description: {nwbfile.experiment_description}")
print(f"Institution: {nwbfile.institution}")
print(f"Lab: {nwbfile.lab}")
print(f"\nSubject: {nwbfile.subject.subject_id}")
print(f"Species: {nwbfile.subject.species}")
print(f"Sex: {nwbfile.subject.sex}")

# %% [markdown]
# ## Extract Trial Information
#
# Each trial contains:
# - Target position (where to reach)
# - Target onset time (when target appears)
# - Go cue time (when to start moving)
# - Movement onset time (when movement actually begins)

# %%
print("\nExtracting trial and neural data...")

# Get trial information
trials = nwbfile.intervals['trials']
target_pos = trials['target_pos'][:]
active_target = trials['active_target'][:]
go_cue_times = np.array(trials['go_cue_time'][:])
target_on_times = np.array(trials['target_on_time'][:])
move_onset_times = np.array(trials['move_onset_time'][:])

# Calculate reach directions for each trial
reach_directions = []
target_positions = []

for i in range(len(trials)):
    trial_targets = np.array(target_pos[i])
    active_idx = int(active_target[i])
    target_xy = trial_targets[active_idx]
    target_positions.append(target_xy)
    # Calculate angle from center (0,0)
    angle = np.arctan2(target_xy[1], target_xy[0])
    reach_directions.append(np.degrees(angle))

reach_directions = np.array(reach_directions)
target_positions = np.array(target_positions)

# Get unique directions
unique_dirs = np.unique(np.round(reach_directions, 1))
n_directions = len(unique_dirs)

# Calculate delay periods
delay_durations = go_cue_times - target_on_times

print(f"Number of trials: {len(trials)}")
print(f"Number of unique directions: {n_directions}")
print(f"Directions (degrees): {unique_dirs}")
print(f"\nDelay period duration:")
print(f"  Mean: {np.mean(delay_durations):.3f} seconds")
print(f"  Range: {np.min(delay_durations):.3f} to {np.max(delay_durations):.3f} seconds")

# Get neural units
units = nwbfile.units
n_units = len(units)
print(f"\nNumber of units: {n_units}")

# %% [markdown]
# ## Visualize Trial Distribution

# %%
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

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Target positions
ax = axes[0, 0]
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
ax = plt.subplot(2, 2, 4, projection='polar')
angles_rad = np.radians(reach_directions)
ax.hist(angles_rad, bins=20, alpha=0.7, color='green')
ax.set_title('Reach Directions (Polar)')
ax.set_theta_zero_location('E')
ax.set_theta_direction(1)

plt.tight_layout()
plt.savefig('trial_distribution.png', dpi=150, bbox_inches='tight')
print("\nFigure saved: trial_distribution.png")
plt.show()

# %% [markdown]
# ## Calculate Firing Rates During Delay Period
#
# For each neuron and each reach direction, we calculate the average firing rate
# during the delay period (from 100ms after target onset to 100ms before go cue).
# This is the planning window where we expect to see directional tuning.

# %%
print("\nCalculating firing rates during delay period...")

# Define the delay period window
delay_start_offset = 0.1  # seconds after target on
delay_end_offset = -0.1   # seconds before go cue

# Calculate firing rates for each unit
unit_firing_rates = np.zeros((n_units, n_directions))

for unit_idx in tqdm(range(n_units), desc="Processing units"):
    spike_times = units['spike_times'][unit_idx]
    
    for dir_idx, direction in enumerate(unique_dirs):
        # Find trials with this direction
        dir_trials = np.where(np.abs(reach_directions - direction) < 0.5)[0]
        
        rates_for_direction = []
        for trial_idx in dir_trials:
            # Define delay period for this trial
            delay_start = target_on_times[trial_idx] + delay_start_offset
            delay_end = go_cue_times[trial_idx] + delay_end_offset
            
            # Count spikes in delay period
            spikes_in_delay = np.sum((spike_times >= delay_start) & (spike_times < delay_end))
            
            # Calculate firing rate (spikes/second)
            delay_duration = delay_end - delay_start
            if delay_duration > 0:
                firing_rate = spikes_in_delay / delay_duration
                rates_for_direction.append(firing_rate)
        
        # Store mean firing rate for this unit and direction
        if len(rates_for_direction) > 0:
            unit_firing_rates[unit_idx, dir_idx] = np.mean(rates_for_direction)

print("Firing rate calculation complete!")

# %% [markdown]
# ## Identify Well-Tuned Neurons
#
# We identify neurons with significant directional tuning using:
# 1. **Statistical significance**: One-way ANOVA across directions (p < 0.01)
# 2. **Tuning strength**: Ratio of maximum to minimum firing rate (> 1.5)

# %%
print("\nIdentifying well-tuned neurons...")

# Calculate tuning metrics for each neuron
tuning_strength = np.zeros(n_units)
preferred_direction = np.zeros(n_units)
tuning_significance = np.zeros(n_units)

for unit_idx in range(n_units):
    rates = unit_firing_rates[unit_idx, :]
    
    # Preferred direction (direction with highest firing rate)
    preferred_direction[unit_idx] = unique_dirs[np.argmax(rates)]
    
    # Tuning strength (ratio of max to min firing rate)
    if np.min(rates) > 0:
        tuning_strength[unit_idx] = np.max(rates) / np.min(rates)
    else:
        tuning_strength[unit_idx] = np.nan
    
    # Statistical significance (one-way ANOVA across directions)
    all_rates_per_dir = []
    for dir_idx, direction in enumerate(unique_dirs):
        dir_trials = np.where(np.abs(reach_directions - direction) < 0.5)[0]
        rates_this_dir = []
        spike_times = units['spike_times'][unit_idx]
        
        for trial_idx in dir_trials:
            delay_start = target_on_times[trial_idx] + delay_start_offset
            delay_end = go_cue_times[trial_idx] + delay_end_offset
            delay_duration = delay_end - delay_start
            if delay_duration > 0:
                spikes_in_delay = np.sum((spike_times >= delay_start) & (spike_times < delay_end))
                firing_rate = spikes_in_delay / delay_duration
                rates_this_dir.append(firing_rate)
        all_rates_per_dir.append(rates_this_dir)
    
    # Perform ANOVA
    if all(len(r) > 0 for r in all_rates_per_dir):
        f_stat, p_val = stats.f_oneway(*all_rates_per_dir)
        tuning_significance[unit_idx] = p_val

# Identify well-tuned neurons
well_tuned_mask = (tuning_significance < 0.01) & (tuning_strength > 1.5) & ~np.isnan(tuning_strength)
well_tuned_indices = np.where(well_tuned_mask)[0]

print(f"Well-tuned units: {len(well_tuned_indices)} / {n_units} ({100*len(well_tuned_indices)/n_units:.1f}%)")
print(f"Tuning strength range: {np.nanmin(tuning_strength[well_tuned_mask]):.2f} - {np.nanmax(tuning_strength[well_tuned_mask]):.2f}")

# %% [markdown]
# ## Visualize Tuning Curves
#
# Tuning curves show how each neuron's firing rate varies with reach direction.
# Neurons with directional tuning show higher firing rates for their "preferred direction".

# %%
print("\nGenerating tuning curve plots...")

# Select 6 well-tuned neurons
n_examples = min(6, len(well_tuned_indices))
example_indices = well_tuned_indices[np.argsort(tuning_strength[well_tuned_indices])[-n_examples:]]

fig, axes = plt.subplots(2, 3, figsize=(15, 10), subplot_kw={'projection': 'polar'})
axes = axes.flatten()

for i, unit_idx in enumerate(example_indices):
    ax = axes[i]
    
    # Get firing rates for this unit
    rates = unit_firing_rates[unit_idx, :]
    
    # Convert directions to radians for polar plot
    dirs_rad = np.radians(unique_dirs)
    
    # Close the curve by appending the first point
    dirs_rad_closed = np.append(dirs_rad, dirs_rad[0])
    rates_closed = np.append(rates, rates[0])
    
    # Plot tuning curve
    ax.plot(dirs_rad_closed, rates_closed, 'o-', linewidth=2, markersize=8)
    ax.fill(dirs_rad_closed, rates_closed, alpha=0.3)
    
    # Mark preferred direction
    pref_idx = np.argmax(rates)
    ax.plot(dirs_rad[pref_idx], rates[pref_idx], 'r*', markersize=15)
    
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(1)
    ax.set_title(f'Unit {unit_idx}\nTuning strength: {tuning_strength[unit_idx]:.2f}\np < {tuning_significance[unit_idx]:.4f}', 
                 fontsize=10, pad=20)
    ax.set_ylabel('Firing Rate (Hz)', labelpad=30)

plt.tight_layout()
plt.savefig('tuning_curves_examples.png', dpi=150, bbox_inches='tight')
print("Figure saved: tuning_curves_examples.png")
plt.show()

# %% [markdown]
# ## Raster Plots
#
# Raster plots show individual spike times across trials for different reach directions.
# Each row is a trial, each tick mark is a spike. This visualization shows the temporal
# dynamics of neural activity during the task.

# %%
print("\nGenerating raster plots...")

# Select best tuned neuron
best_unit_idx = well_tuned_indices[np.argmax(tuning_strength[well_tuned_indices])]

fig = plt.figure(figsize=(15, 10))
gs = GridSpec(n_directions, 1, figure=fig, hspace=0.3)

spike_times = units['spike_times'][best_unit_idx]

for dir_idx, direction in enumerate(unique_dirs):
    ax = fig.add_subplot(gs[dir_idx, 0])
    
    # Find trials with this direction
    dir_trials = np.where(np.abs(reach_directions - direction) < 0.5)[0]
    
    # Plot raster for each trial
    for plot_idx, trial_idx in enumerate(dir_trials):
        target_on = target_on_times[trial_idx]
        go_cue = go_cue_times[trial_idx]
        move_onset = move_onset_times[trial_idx]
        
        # Get spikes relative to target onset
        trial_spikes = spike_times[(spike_times >= target_on - 0.2) & 
                                   (spike_times <= move_onset + 0.5)] - target_on
        
        # Plot spikes
        ax.vlines(trial_spikes, plot_idx - 0.4, plot_idx + 0.4, colors='black', linewidth=1)
    
    # Mark important events
    ax.axvline(0, color='green', linestyle='--', linewidth=2, label='Target On', alpha=0.7)
    avg_go_cue = np.mean(go_cue_times[dir_trials] - target_on_times[dir_trials])
    ax.axvline(avg_go_cue, color='blue', linestyle='--', linewidth=2, label='Go Cue', alpha=0.7)
    avg_move = np.mean(move_onset_times[dir_trials] - target_on_times[dir_trials])
    ax.axvline(avg_move, color='red', linestyle='--', linewidth=2, label='Move Onset', alpha=0.7)
    
    # Shade delay period (planning window)
    ax.axvspan(delay_start_offset, avg_go_cue + delay_end_offset, alpha=0.2, color='yellow', label='Analysis Window')
    
    ax.set_ylabel(f'{direction:.0f}°\n({len(dir_trials)} trials)', fontsize=10)
    ax.set_xlim(-0.2, avg_move + 0.5)
    ax.set_ylim(-0.5, len(dir_trials) - 0.5)
    
    if dir_idx == 0:
        ax.legend(loc='upper right', fontsize=8)
    if dir_idx == n_directions - 1:
        ax.set_xlabel('Time from Target Onset (s)', fontsize=12)
    else:
        ax.set_xticklabels([])

fig.suptitle(f'Raster Plot for Unit {best_unit_idx} (Best Tuned Neuron)\nFiring Rate vs. Reach Direction', 
             fontsize=14, y=0.995)
plt.savefig('raster_plots.png', dpi=150, bbox_inches='tight')
print("Figure saved: raster_plots.png")
plt.show()

# %% [markdown]
# ## Population Analysis
#
# Analyzing the population of well-tuned neurons reveals:
# 1. Distribution of preferred directions across the population
# 2. Distribution of tuning strength
# 3. Average population tuning curve

# %%
print("\nGenerating population analysis plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Distribution of preferred directions
ax = axes[0, 0]
pref_dirs_tuned = preferred_direction[well_tuned_mask]
ax.hist(pref_dirs_tuned, bins=20, edgecolor='black', alpha=0.7)
ax.set_xlabel('Preferred Direction (degrees)')
ax.set_ylabel('Number of Neurons')
ax.set_title(f'Distribution of Preferred Directions\n({len(well_tuned_indices)} well-tuned neurons)')
ax.grid(True, alpha=0.3)

# Plot 2: Tuning strength distribution
ax = axes[0, 1]
ax.hist(tuning_strength[well_tuned_mask], bins=20, edgecolor='black', alpha=0.7, color='orange')
ax.set_xlabel('Tuning Strength (max/min firing rate)')
ax.set_ylabel('Number of Neurons')
ax.set_title('Distribution of Tuning Strength')
ax.grid(True, alpha=0.3)

# Plot 3: Polar histogram of preferred directions
ax = plt.subplot(2, 2, 3, projection='polar')
angles_rad = np.radians(pref_dirs_tuned)
ax.hist(angles_rad, bins=20, alpha=0.7, color='green')
ax.set_title('Preferred Directions (Polar)', pad=20)
ax.set_theta_zero_location('E')
ax.set_theta_direction(1)

# Plot 4: Mean population tuning curve
ax = axes[1, 1]
# Normalize each neuron's tuning curve and average
normalized_tuning = np.zeros((len(well_tuned_indices), n_directions))
for i, unit_idx in enumerate(well_tuned_indices):
    rates = unit_firing_rates[unit_idx, :]
    if np.max(rates) > 0:
        normalized_tuning[i, :] = rates / np.max(rates)

mean_tuning = np.mean(normalized_tuning, axis=0)
sem_tuning = stats.sem(normalized_tuning, axis=0)

ax.plot(unique_dirs, mean_tuning, 'o-', linewidth=2, markersize=8, color='purple')
ax.fill_between(unique_dirs, mean_tuning - sem_tuning, mean_tuning + sem_tuning, alpha=0.3, color='purple')
ax.set_xlabel('Reach Direction (degrees)')
ax.set_ylabel('Normalized Firing Rate')
ax.set_title('Average Population Tuning Curve')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('population_analysis.png', dpi=150, bbox_inches='tight')
print("Figure saved: population_analysis.png")
plt.show()

# %% [markdown]
# ## Summary and Conclusions

# %%
print("\n" + "=" * 60)
print("DIRECTIONAL TUNING ANALYSIS SUMMARY")
print("=" * 60)
print(f"Dataset: DANDI:000140")
print(f"Subject: {nwbfile.subject.subject_id} ({nwbfile.subject.species})")
print(f"Task: Delayed reaching with {n_directions} target directions")
print(f"")
print(f"Total trials: {len(trials)}")
print(f"Average delay period: {np.mean(delay_durations):.3f} seconds")
print(f"")
print(f"Total units analyzed: {n_units}")
print(f"Well-tuned units: {len(well_tuned_indices)} ({100*len(well_tuned_indices)/n_units:.1f}%)")
print(f"")
print(f"Tuning strength (well-tuned units):")
print(f"  Mean: {np.mean(tuning_strength[well_tuned_mask]):.2f}")
print(f"  Median: {np.median(tuning_strength[well_tuned_mask]):.2f}")
print(f"  Range: {np.min(tuning_strength[well_tuned_mask]):.2f} - {np.max(tuning_strength[well_tuned_mask]):.2f}")
print(f"")
print(f"Preferred direction distribution:")
print(f"  Mean: {np.mean(pref_dirs_tuned):.1f}°")
print(f"  Std: {np.std(pref_dirs_tuned):.1f}°")
print(f"")
print("CONCLUSIONS:")
print("- Motor and premotor cortex neurons show clear directional tuning")
print("  during the delay period (planning phase) of reaching")
print("- Tuning is present before movement onset, indicating that")
print("  these neurons encode the planned movement direction")
print("- Different neurons prefer different directions, providing a")
print("  population code for representing reach direction")
print("- This demonstrates that movement planning involves direction-")
print("  selective neural activity in motor cortical areas")
print("=" * 60)
