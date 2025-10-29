# %% [markdown]
# # Velocity Tuning Analysis
# Demonstrate how motor cortex neurons correlate with movement velocity

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
# Compute speed from velocity
print("\nComputing speed from velocity...")
speed = np.sqrt(cursor_vel.values[:, 0]**2 + cursor_vel.values[:, 1]**2)
speed_tsd = nap.Tsd(t=cursor_vel.index, d=speed)

print(f"Speed range: {np.min(speed):.2f} to {np.max(speed):.2f}")
print(f"Mean speed: {np.mean(speed):.2f} ± {np.std(speed):.2f}")

# %%
# Bin neural activity and speed
# Use 100ms bins for analysis
bin_size = 0.1  # seconds

print(f"\nBinning neural activity with {bin_size}s bins...")

# Create time bins covering the entire session
t_start = cursor_vel.index[0]
t_end = cursor_vel.index[-1]
time_bins = np.arange(t_start, t_end, bin_size)
n_bins = len(time_bins) - 1

# Bin speed
binned_speed = np.zeros(n_bins)
for i in range(n_bins):
    bin_start = time_bins[i]
    bin_end = time_bins[i + 1]
    speed_in_bin = speed_tsd.get(bin_start, bin_end)
    if len(speed_in_bin) > 0:
        binned_speed[i] = np.mean(speed_in_bin.values)

# Bin firing rates for each unit
binned_rates = np.zeros((len(units), n_bins))
for unit_idx in range(len(units)):
    unit_spikes = units[unit_idx]
    
    for i in range(n_bins):
        bin_start = time_bins[i]
        bin_end = time_bins[i + 1]
        spikes_in_bin = unit_spikes.get(bin_start, bin_end)
        binned_rates[unit_idx, i] = len(spikes_in_bin) / bin_size

print(f"Binned data shape: {binned_rates.shape}")

# %%
# Compute correlation between firing rate and speed for each neuron
print("\nComputing velocity correlations...")

correlations = np.zeros(len(units))
p_values = np.zeros(len(units))

# Only use bins where speed > threshold to focus on movement periods
speed_threshold = 5.0
movement_mask = binned_speed > speed_threshold

print(f"Movement bins: {np.sum(movement_mask)} / {n_bins}")

for unit_idx in range(len(units)):
    if np.sum(movement_mask) > 10:  # Need enough data points
        corr, pval = pearsonr(binned_speed[movement_mask], binned_rates[unit_idx, movement_mask])
        correlations[unit_idx] = corr
        p_values[unit_idx] = pval

# Identify significantly correlated neurons
significance_threshold = 0.01
significant_neurons = p_values < significance_threshold

print(f"\nSignificantly correlated neurons (p < {significance_threshold}): {np.sum(significant_neurons)} / {len(units)}")
print(f"Mean correlation: {np.mean(correlations):.3f} ± {np.std(correlations):.3f}")
print(f"Mean |correlation|: {np.mean(np.abs(correlations)):.3f}")

# %%
# Plot example neurons with positive and negative correlations
fig, axes = plt.subplots(3, 2, figsize=(15, 12))

# Select top 3 positive and top 3 negative correlations
pos_correlations = correlations.copy()
pos_correlations[correlations < 0] = -np.inf
top_pos = np.argsort(pos_correlations)[::-1][:3]

neg_correlations = correlations.copy()
neg_correlations[correlations > 0] = np.inf
top_neg = np.argsort(neg_correlations)[:3]

example_units = np.concatenate([top_pos, top_neg])

for idx, unit_idx in enumerate(example_units):
    ax = axes[idx // 2, idx % 2]
    
    # Smooth for visualization
    smoothed_rate = gaussian_filter1d(binned_rates[unit_idx, movement_mask], sigma=2)
    smoothed_speed = gaussian_filter1d(binned_speed[movement_mask], sigma=2)
    
    # Scatter plot
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
plt.savefig('fig_05_velocity_tuning_examples.png', dpi=150, bbox_inches='tight')
print("\nFigure saved: fig_05_velocity_tuning_examples.png")
plt.show()

# %%
# Population-level visualization
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

# Scatter: correlation vs significance
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
plt.savefig('fig_06_population_velocity_tuning.png', dpi=150, bbox_inches='tight')
print("Figure saved: fig_06_population_velocity_tuning.png")
plt.show()

# %%
# Time series example showing firing rate and speed relationship
print("\nCreating time series visualization...")

# Select one trial to show temporal relationship
trial_idx = 10
trial_start = trials.start[trial_idx]
trial_end = trials.end[trial_idx]

# Select a highly correlated neuron
best_unit = top_pos[0]
unit_spikes = units[best_unit]

# Get speed and spikes for this trial
trial_speed = speed_tsd.get(trial_start, trial_end)
trial_spikes = unit_spikes.get(trial_start, trial_end)

# Bin at finer resolution for visualization
fine_bin_size = 0.05
trial_duration = trial_end - trial_start
fine_bins = np.arange(0, trial_duration, fine_bin_size)
n_fine_bins = len(fine_bins) - 1

fine_speed = np.zeros(n_fine_bins)
fine_rate = np.zeros(n_fine_bins)

for i in range(n_fine_bins):
    bin_start = trial_start + fine_bins[i]
    bin_end = trial_start + fine_bins[i + 1]
    
    speed_in_bin = speed_tsd.get(bin_start, bin_end)
    if len(speed_in_bin) > 0:
        fine_speed[i] = np.mean(speed_in_bin.values)
    
    spikes_in_bin = unit_spikes.get(bin_start, bin_end)
    fine_rate[i] = len(spikes_in_bin) / fine_bin_size

# Plot
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Speed
ax = axes[0]
ax.plot(fine_bins[:-1], fine_speed, 'b-', linewidth=2)
ax.set_ylabel('Speed', fontsize=12)
ax.set_title(f'Trial {trial_idx}: Speed and Neural Activity', fontsize=13)
ax.grid(True, alpha=0.3)

# Firing rate
ax = axes[1]
ax.plot(fine_bins[:-1], fine_rate, 'r-', linewidth=2)
spike_times = trial_spikes.index - trial_start
ax.scatter(spike_times, np.ones_like(spike_times) * np.max(fine_rate) * 1.1,
          marker='|', s=100, color='black', alpha=0.5, label='Spikes')
ax.set_ylabel('Firing Rate (Hz)', fontsize=12)
ax.set_xlabel('Time from trial start (s)', fontsize=12)
ax.set_title(f'Unit {best_unit} (r = {correlations[best_unit]:.3f})', fontsize=13)
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig('fig_07_single_trial_velocity_tuning.png', dpi=150, bbox_inches='tight')
print("Figure saved: fig_07_single_trial_velocity_tuning.png")
plt.show()

print("\n=== Velocity tuning analysis complete ===")
