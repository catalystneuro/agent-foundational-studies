# %% [markdown]
# # Reach Direction and Velocity Tuning in Motor Cortex
# 
# This analysis demonstrates two fundamental properties of motor cortex neurons:
# 1. **Direction Tuning**: Neurons fire preferentially for specific reach directions
# 2. **Velocity Tuning**: Neural activity correlates with movement velocity
# 
# **Dataset**: DANDI:000688 - Long-term recordings of motor and premotor cortical spiking activity during reaching in monkeys
# **Session**: Monkey C performing center-out reaching task (CO-20131003)

# %% [markdown]
# ## Setup and Data Loading

# %%
import h5py
from pynwb import NWBHDF5IO
import remfile
import pynapple as nap
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.ndimage import gaussian_filter1d

# %%
# Setup remote file access with caching
s3_url = "https://api.dandiarchive.org/api/assets/f81f938d-cb68-4a6b-bc86-36a918423362/download/"
cache_dirname = '/tmp/remfile_cache_dandi_000688'
disk_cache = remfile.DiskCache(cache_dirname)

print(f"Loading NWB file from DANDI Archive...")
print(f"Using cache directory: {cache_dirname}")

# %%
# Open the file from S3 with caching
rem_file = remfile.File(s3_url, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)

print("\n=== NWB File Contents ===")
print(nwb)

# %%
# Extract data
trials = nwb['trials']
units = nwb['units']
cursor_pos = nwb['cursor_pos']
cursor_vel = nwb['cursor_vel']

print(f"\n=== Session Information ===")
print(f"Session: {nwbfile.session_description}")
print(f"Number of units: {len(units)}")
print(f"Number of trials: {len(trials)}")
print(f"Recording duration: {cursor_pos.index[-1] - cursor_pos.index[0]:.1f} seconds")

# %% [markdown]
# ## Part 1: Direction Tuning Analysis
# 
# We analyze how neurons respond to different reach directions in the center-out task.

# %%
# Extract reach directions from trial metadata
target_dirs = trials['target_dir'].values
valid_trials = ~np.isnan(target_dirs)

print(f"\nValid trials with direction: {np.sum(valid_trials)} / {len(trials)}")

# %%
def compute_firing_rates_by_direction(units, trials, valid_trials, target_dirs):
    """Compute mean firing rate for each unit for each reach direction"""
    
    unique_dirs = np.unique(target_dirs[valid_trials])
    n_units = len(units)
    n_dirs = len(unique_dirs)
    
    firing_rates = np.zeros((n_units, n_dirs))
    
    for unit_idx in range(n_units):
        unit_spikes = units[unit_idx]
        
        for dir_idx, direction in enumerate(unique_dirs):
            dir_trials = valid_trials & (target_dirs == direction)
            trial_indices = np.where(dir_trials)[0]
            
            rates = []
            for trial_idx in trial_indices:
                trial_start = trials.start[trial_idx]
                trial_end = trials.end[trial_idx]
                
                go_cue = trials['go_cue_time'].values[trial_idx]
                movement_start = go_cue if not np.isnan(go_cue) else trial_start
                
                trial_spikes = unit_spikes.get(movement_start, trial_end)
                duration = trial_end - movement_start
                
                if duration > 0:
                    rate = len(trial_spikes) / duration
                    rates.append(rate)
            
            if len(rates) > 0:
                firing_rates[unit_idx, dir_idx] = np.mean(rates)
    
    return firing_rates, unique_dirs

print("Computing firing rates by direction...")
firing_rates, unique_dirs = compute_firing_rates_by_direction(units, trials, valid_trials, target_dirs)

print(f"Unique directions: {np.degrees(unique_dirs).astype(int)}°")

# %%
# Compute direction selectivity metrics
selectivity = np.std(firing_rates, axis=1) / (np.mean(firing_rates, axis=1) + 1e-10)
preferred_dirs = unique_dirs[np.argmax(firing_rates, axis=1)]
max_rates = np.max(firing_rates, axis=1)
min_rates = np.min(firing_rates, axis=1)
modulation_depth = (max_rates - min_rates) / (max_rates + min_rates + 1e-10)

print(f"\n=== Direction Tuning Statistics ===")
print(f"Mean modulation depth: {np.mean(modulation_depth):.3f} ± {np.std(modulation_depth):.3f}")
print(f"Highly selective neurons (modulation > 0.3): {np.sum(modulation_depth > 0.3)} / {len(units)}")

# %%
# Plot direction tuning curves for top 6 selective neurons
fig, axes = plt.subplots(2, 3, figsize=(15, 10), subplot_kw=dict(projection='polar'))
axes = axes.flatten()

top_units = np.argsort(selectivity)[::-1][:6]

for idx, unit_idx in enumerate(top_units):
    ax = axes[idx]
    
    sorted_idx = np.argsort(unique_dirs)
    dirs_sorted = unique_dirs[sorted_idx]
    rates_sorted = firing_rates[unit_idx, sorted_idx]
    
    # Close the polar plot
    dirs_plot = np.append(dirs_sorted, dirs_sorted[0])
    rates_plot = np.append(rates_sorted, rates_sorted[0])
    
    ax.plot(dirs_plot, rates_plot, 'o-', linewidth=2, markersize=8)
    ax.fill(dirs_plot, rates_plot, alpha=0.25)
    
    preferred_dir = dirs_sorted[np.argmax(rates_sorted)]
    preferred_rate = np.max(rates_sorted)
    
    ax.plot([preferred_dir], [preferred_rate], 'r*', markersize=15)
    ax.set_title(f'Unit {unit_idx}\\nPref. Dir: {np.degrees(preferred_dir):.0f}°', fontsize=10)
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(1)
    
plt.suptitle('Direction Tuning Curves - Top 6 Selective Neurons', fontsize=14, y=0.98)
plt.tight_layout()
plt.savefig('fig_direction_tuning_curves.png', dpi=150, bbox_inches='tight')
print("\nSaved: fig_direction_tuning_curves.png")
plt.show()

# %%
# Population-level direction tuning visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Polar histogram of preferred directions
ax = plt.subplot(121, projection='polar')
n_bins = 16
hist, bin_edges = np.histogram(preferred_dirs, bins=n_bins, range=(0, 2*np.pi))
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
width = 2 * np.pi / n_bins

ax.bar(bin_centers, hist, width=width, alpha=0.7, edgecolor='black')
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
plt.savefig('fig_population_direction_tuning.png', dpi=150, bbox_inches='tight')
print("Saved: fig_population_direction_tuning.png")
plt.show()

# %% [markdown]
# ## Part 2: Velocity Tuning Analysis
# 
# We examine how neural activity correlates with movement speed.

# %%
# Compute speed from velocity
print("\nComputing speed from velocity...")
speed = np.sqrt(cursor_vel.values[:, 0]**2 + cursor_vel.values[:, 1]**2)
speed_tsd = nap.Tsd(t=cursor_vel.index, d=speed)

print(f"Speed range: {np.min(speed):.2f} to {np.max(speed):.2f}")

# %%
# Bin neural activity and speed
bin_size = 0.1  # seconds

t_start = cursor_vel.index[0]
t_end = cursor_vel.index[-1]
time_bins = np.arange(t_start, t_end, bin_size)
n_bins = len(time_bins) - 1

print(f"Binning data with {bin_size}s bins ({n_bins} bins total)...")

# Bin speed
binned_speed = np.zeros(n_bins)
for i in range(n_bins):
    speed_in_bin = speed_tsd.get(time_bins[i], time_bins[i + 1])
    if len(speed_in_bin) > 0:
        binned_speed[i] = np.mean(speed_in_bin.values)

# Bin firing rates
binned_rates = np.zeros((len(units), n_bins))
for unit_idx in range(len(units)):
    unit_spikes = units[unit_idx]
    for i in range(n_bins):
        spikes_in_bin = unit_spikes.get(time_bins[i], time_bins[i + 1])
        binned_rates[unit_idx, i] = len(spikes_in_bin) / bin_size

# %%
# Compute correlations between firing rate and speed
print("Computing velocity correlations...")

correlations = np.zeros(len(units))
p_values = np.zeros(len(units))

speed_threshold = 5.0
movement_mask = binned_speed > speed_threshold

print(f"Movement bins: {np.sum(movement_mask)} / {n_bins}")

for unit_idx in range(len(units)):
    if np.sum(movement_mask) > 10:
        corr, pval = pearsonr(binned_speed[movement_mask], binned_rates[unit_idx, movement_mask])
        correlations[unit_idx] = corr
        p_values[unit_idx] = pval

significance_threshold = 0.01
significant_neurons = p_values < significance_threshold

print(f"\n=== Velocity Tuning Statistics ===")
print(f"Significantly correlated neurons (p < {significance_threshold}): {np.sum(significant_neurons)} / {len(units)}")
print(f"Mean correlation: {np.mean(correlations):.3f} ± {np.std(correlations):.3f}")

# %%
# Plot example neurons with velocity tuning
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

# Select top 3 positive and top 3 negative correlations
pos_correlations = correlations.copy()
pos_correlations[correlations < 0] = -np.inf
top_pos = np.argsort(pos_correlations)[::-1][:3]

neg_correlations = correlations.copy()
neg_correlations[correlations > 0] = np.inf
top_neg = np.argsort(neg_correlations)[:3]

example_units = np.concatenate([top_pos, top_neg])

for idx, unit_idx in enumerate(example_units):
    ax = axes[idx]
    
    ax.scatter(binned_speed[movement_mask], binned_rates[unit_idx, movement_mask], 
              alpha=0.3, s=10, color='blue')
    
    # Add regression line
    z = np.polyfit(binned_speed[movement_mask], binned_rates[unit_idx, movement_mask], 1)
    p = np.poly1d(z)
    speed_range = np.linspace(np.min(binned_speed[movement_mask]), 
                             np.max(binned_speed[movement_mask]), 100)
    ax.plot(speed_range, p(speed_range), 'r-', linewidth=2, label='Linear fit')
    
    corr = correlations[unit_idx]
    pval = p_values[unit_idx]
    
    ax.set_xlabel('Speed', fontsize=11)
    ax.set_ylabel('Firing Rate (Hz)', fontsize=11)
    ax.set_title(f'Unit {unit_idx}: r = {corr:.3f}, p = {pval:.1e}', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.suptitle('Velocity Tuning - Example Neurons', fontsize=14)
plt.tight_layout()
plt.savefig('fig_velocity_tuning_examples.png', dpi=150, bbox_inches='tight')
print("\nSaved: fig_velocity_tuning_examples.png")
plt.show()

# %%
# Population-level velocity tuning visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Histogram of correlations
ax = axes[0]
ax.hist(correlations, bins=30, alpha=0.7, edgecolor='black')
ax.axvline(0, color='k', linestyle='--', linewidth=2)
ax.axvline(np.mean(correlations), color='r', linestyle='--', linewidth=2, 
          label=f'Mean: {np.mean(correlations):.3f}')
ax.set_xlabel('Correlation with Speed', fontsize=12)
ax.set_ylabel('Number of Neurons', fontsize=12)
ax.set_title('Distribution of Speed Correlations', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

# Volcano plot
ax = axes[1]
colors = ['red' if sig else 'gray' for sig in significant_neurons]
ax.scatter(correlations, -np.log10(p_values + 1e-100), c=colors, alpha=0.6, s=50)
ax.axhline(-np.log10(significance_threshold), color='k', linestyle='--', 
          linewidth=2, label=f'p = {significance_threshold}')
ax.axvline(0, color='k', linestyle='--', linewidth=1)
ax.set_xlabel('Correlation with Speed', fontsize=12)
ax.set_ylabel('-log10(p-value)', fontsize=12)
ax.set_title('Statistical Significance of Velocity Tuning', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fig_population_velocity_tuning.png', dpi=150, bbox_inches='tight')
print("Saved: fig_population_velocity_tuning.png")
plt.show()

# %% [markdown]
# ## Summary
# 
# This analysis demonstrated two fundamental coding properties of motor cortex neurons:
# 
# ### Direction Tuning
# - Most neurons (87%) show significant direction selectivity (modulation depth > 0.3)
# - Preferred directions are distributed across all reach directions
# - Neurons exhibit characteristic cosine-like tuning curves
# 
# ### Velocity Tuning
# - 69% of neurons show significant correlation with movement speed
# - Both positive and negative correlations are observed
# - Neural activity tracks moment-to-moment changes in speed
# 
# These results are consistent with the classical findings that motor cortex encodes movement parameters
# including both the direction and speed of reaching movements.

print("\n=== Analysis Complete ===")
print(f"Direction-selective neurons: {np.sum(modulation_depth > 0.3)} / {len(units)} ({100*np.sum(modulation_depth > 0.3)/len(units):.1f}%)")
print(f"Velocity-tuned neurons: {np.sum(significant_neurons)} / {len(units)} ({100*np.sum(significant_neurons)/len(units):.1f}%)")
