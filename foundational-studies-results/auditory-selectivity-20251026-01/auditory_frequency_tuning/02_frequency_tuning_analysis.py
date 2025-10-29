# %% [markdown]
# # Auditory Frequency Tuning Analysis
# 
# This script analyzes frequency tuning properties of neurons in mouse auditory cortex.
# We calculate firing rates in response to different tone frequencies and generate
# tuning curves showing frequency selectivity.

# %%
import pynwb
import lindi
import pynapple as nap
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# %% [markdown]
# ## Load NWB File

# %%
print("Loading NWB file...")
local_cache = lindi.LocalCache()
lindi_url = "https://lindi.neurosift.org/dandi/dandisets/001419/assets/b651a329-009d-4b85-b08c-ce99de014c18/nwb.lindi.json"
f = lindi.LindiH5pyFile.from_lindi_file(lindi_url, local_cache=local_cache)
io = pynwb.NWBHDF5IO(file=f, mode='r')
nwbfile = io.read()

# %% [markdown]
# ## Extract Trial Information and Neural Data

# %%
# Extract trial information
trials = nwbfile.intervals['trials']
trial_freqs = trials['frequency'][:]
trial_start_times = trials['start_time'][:]
trial_stop_times = trials['stop_time'][:]

# Get unique frequencies and sort them
unique_freqs = np.sort(np.unique(trial_freqs))
print(f"Analyzing responses to {len(unique_freqs)} frequencies")
print(f"Frequency range: {unique_freqs[0]:.0f} - {unique_freqs[-1]:.0f} Hz")

# Extract unit information
units = nwbfile.units
n_units = len(units)
unit_quality = units['quality'][:]
print(f"\nTotal units: {n_units}")
print(f"Good quality units: {np.sum(unit_quality == 'good')}")
print(f"Multi-unit activity: {np.sum(unit_quality == 'mua')}")

# %% [markdown]
# ## Calculate Firing Rates for Each Unit and Frequency
# 
# For each neuron, we calculate the mean firing rate in response to each frequency.
# We use a time window from 0 to 100 ms after stimulus onset.

# %%
# Parameters for analysis
response_window = [0.0, 0.1]  # 0-100 ms after stimulus onset
baseline_window = [-0.1, 0.0]  # 100 ms before stimulus onset

# Initialize arrays to store results
tuning_curves = np.zeros((n_units, len(unique_freqs)))
baseline_rates = np.zeros(n_units)
response_rates = np.zeros((n_units, len(unique_freqs)))

print("\nCalculating firing rates for each unit and frequency...")

for unit_idx in tqdm(range(n_units)):
    # Get spike times for this unit
    spike_times = units['spike_times'][unit_idx]
    
    # Calculate baseline firing rate
    baseline_spikes = []
    for trial_idx in range(len(trial_start_times)):
        trial_start = trial_start_times[trial_idx]
        baseline_start = trial_start + baseline_window[0]
        baseline_end = trial_start + baseline_window[1]
        
        # Count spikes in baseline window
        spikes_in_baseline = np.sum((spike_times >= baseline_start) & (spike_times < baseline_end))
        baseline_spikes.append(spikes_in_baseline)
    
    baseline_rates[unit_idx] = np.mean(baseline_spikes) / np.diff(baseline_window)[0]
    
    # Calculate response for each frequency
    for freq_idx, freq in enumerate(unique_freqs):
        # Find trials with this frequency
        freq_trials = np.where(trial_freqs == freq)[0]
        
        # Calculate firing rate for each trial
        trial_rates = []
        for trial_idx in freq_trials:
            trial_start = trial_start_times[trial_idx]
            response_start = trial_start + response_window[0]
            response_end = trial_start + response_window[1]
            
            # Count spikes in response window
            spikes_in_window = np.sum((spike_times >= response_start) & (spike_times < response_end))
            trial_rates.append(spikes_in_window)
        
        # Mean firing rate across trials (spikes per second)
        mean_rate = np.mean(trial_rates) / np.diff(response_window)[0]
        tuning_curves[unit_idx, freq_idx] = mean_rate
        response_rates[unit_idx, freq_idx] = mean_rate

print("Firing rate calculation complete!")

# %% [markdown]
# ## Identify Units with Strong Frequency Selectivity
# 
# We identify well-tuned units based on:
# 1. Significant response above baseline
# 2. Clear peak in tuning curve

# %%
# Calculate tuning metrics
max_response = np.max(tuning_curves, axis=1)
baseline_threshold = baseline_rates + 5  # 5 Hz above baseline
is_responsive = max_response > baseline_threshold

# Calculate tuning width (full width at half maximum)
best_freq_idx = np.argmax(tuning_curves, axis=1)
tuning_selectivity = np.zeros(n_units)

for unit_idx in range(n_units):
    max_rate = tuning_curves[unit_idx, best_freq_idx[unit_idx]]
    half_max = max_rate / 2
    
    # Calculate selectivity as ratio of peak to mean
    if np.mean(tuning_curves[unit_idx, :]) > 0:
        tuning_selectivity[unit_idx] = max_rate / np.mean(tuning_curves[unit_idx, :])

# Select well-tuned units
well_tuned_units = np.where((is_responsive) & (tuning_selectivity > 2.0))[0]
print(f"\nWell-tuned units: {len(well_tuned_units)} / {n_units}")

# Find best frequencies for well-tuned units
best_freqs = unique_freqs[best_freq_idx[well_tuned_units]]
print(f"Best frequencies range: {np.min(best_freqs):.0f} - {np.max(best_freqs):.0f} Hz")

# %% [markdown]
# ## Visualize Tuning Curves for Example Neurons

# %%
# Select top 9 well-tuned units based on selectivity
n_examples = min(9, len(well_tuned_units))
top_units = well_tuned_units[np.argsort(tuning_selectivity[well_tuned_units])[-n_examples:]]

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()

for plot_idx, unit_idx in enumerate(top_units):
    ax = axes[plot_idx]
    
    # Plot tuning curve
    ax.plot(unique_freqs / 1000, tuning_curves[unit_idx, :], 'o-', linewidth=2, markersize=8)
    ax.axhline(baseline_rates[unit_idx], color='gray', linestyle='--', label='Baseline')
    
    # Mark best frequency
    best_freq = unique_freqs[best_freq_idx[unit_idx]]
    ax.axvline(best_freq / 1000, color='red', linestyle=':', alpha=0.5, label=f'Best: {best_freq/1000:.1f} kHz')
    
    ax.set_xlabel('Frequency (kHz)')
    ax.set_ylabel('Firing Rate (Hz)')
    ax.set_title(f'Unit {unit_idx} ({unit_quality[unit_idx]})\nSelectivity: {tuning_selectivity[unit_idx]:.2f}')
    ax.set_xscale('log')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('tuning_curves_examples.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nFigure saved: tuning_curves_examples.png")

# %% [markdown]
# ## Generate Raster Plots for Select Units

# %%
# Select 3 units with different best frequencies
selected_units = []
target_freqs = [unique_freqs[2], unique_freqs[5], unique_freqs[8]]  # Low, mid, high

for target_freq in target_freqs:
    # Find unit with best frequency closest to target
    units_at_freq = well_tuned_units[np.argmin(np.abs(unique_freqs[best_freq_idx[well_tuned_units]] - target_freq))]
    selected_units.append(units_at_freq)

fig, axes = plt.subplots(len(selected_units), 1, figsize=(14, 10))

for plot_idx, unit_idx in enumerate(selected_units):
    ax = axes[plot_idx]
    
    best_freq = unique_freqs[best_freq_idx[unit_idx]]
    spike_times = units['spike_times'][unit_idx]
    
    # Plot rasters for each frequency
    for freq_idx, freq in enumerate(unique_freqs):
        freq_trials = np.where(trial_freqs == freq)[0]
        
        for trial_offset, trial_idx in enumerate(freq_trials):
            trial_start = trial_start_times[trial_idx]
            
            # Get spikes in window around stimulus
            window = [-0.1, 0.3]  # -100 to 300 ms
            spikes_in_trial = spike_times[(spike_times >= trial_start + window[0]) & 
                                          (spike_times < trial_start + window[1])]
            
            # Convert to relative time
            relative_times = (spikes_in_trial - trial_start) * 1000  # Convert to ms
            
            # Plot raster
            y_position = freq_idx * len(freq_trials) + trial_offset
            ax.scatter(relative_times, [y_position] * len(relative_times), 
                      s=1, c='black', alpha=0.5)
    
    # Add stimulus onset line
    ax.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Stimulus Onset')
    
    # Add response window
    ax.axvspan(0, 100, alpha=0.1, color='blue', label='Response Window')
    
    ax.set_xlabel('Time from Stimulus Onset (ms)')
    ax.set_ylabel('Trial Number')
    ax.set_title(f'Unit {unit_idx} - Raster Plot Across All Frequencies\nBest Frequency: {best_freq/1000:.1f} kHz')
    ax.set_xlim(-100, 300)
    if plot_idx == 0:
        ax.legend(loc='upper right')
    
plt.tight_layout()
plt.savefig('raster_plots.png', dpi=150, bbox_inches='tight')
plt.show()

print("Figure saved: raster_plots.png")

# %% [markdown]
# ## Population Analysis: Distribution of Best Frequencies

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Histogram of best frequencies
axes[0].hist(best_freqs / 1000, bins=10, edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Best Frequency (kHz)')
axes[0].set_ylabel('Number of Units')
axes[0].set_title('Distribution of Best Frequencies\nAcross Well-Tuned Units')
axes[0].set_xscale('log')
axes[0].grid(alpha=0.3)

# Plot 2: Tuning selectivity vs best frequency
axes[1].scatter(best_freqs / 1000, tuning_selectivity[well_tuned_units], 
                alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
axes[1].set_xlabel('Best Frequency (kHz)')
axes[1].set_ylabel('Tuning Selectivity (Peak/Mean)')
axes[1].set_title('Frequency Tuning Selectivity')
axes[1].set_xscale('log')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('population_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print("Figure saved: population_analysis.png")

# %% [markdown]
# ## Summary Statistics

# %%
print("\n" + "=" * 60)
print("FREQUENCY TUNING ANALYSIS SUMMARY")
print("=" * 60)
print(f"Total units analyzed: {n_units}")
print(f"Responsive units: {np.sum(is_responsive)} ({100*np.sum(is_responsive)/n_units:.1f}%)")
print(f"Well-tuned units: {len(well_tuned_units)} ({100*len(well_tuned_units)/n_units:.1f}%)")
print(f"\nBest frequency distribution:")
print(f"  Median: {np.median(best_freqs)/1000:.1f} kHz")
print(f"  Range: {np.min(best_freqs)/1000:.1f} - {np.max(best_freqs)/1000:.1f} kHz")
print(f"\nTuning selectivity:")
print(f"  Mean: {np.mean(tuning_selectivity[well_tuned_units]):.2f}")
print(f"  Median: {np.median(tuning_selectivity[well_tuned_units]):.2f}")
print(f"  Range: {np.min(tuning_selectivity[well_tuned_units]):.2f} - {np.max(tuning_selectivity[well_tuned_units]):.2f}")
print("=" * 60)
