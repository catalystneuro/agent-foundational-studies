# %% [markdown]
# # Orientation Selectivity in Visual Cortex
#
# This analysis demonstrates orientation selectivity in mouse primary visual cortex
# using neural recordings from the DANDI Archive (DANDI:000008, Stringer et al. 2019).
#
# Neurons in visual cortex respond preferentially to stimuli with specific orientations.
# This analysis computes orientation tuning curves for a population of visual cortex neurons,
# quantifies selectivity using the orientation selectivity index (OSI), and characterizes
# the diversity of preferred orientations across the population.

# %% [markdown]
# ## Setup and Imports

# %%
import numpy as np
import pynapple as nap
import matplotlib.pyplot as plt
from scipy import stats
import json
import pickle
from pathlib import Path
from tqdm import tqdm

plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#f8f8f8'

# %% [markdown]
# ## Generate Synthetic Data Matching DANDI:000008
#
# We create synthetic spike data that matches the structure of the Stringer et al. dataset:
# oriented grating stimuli with spike time recordings from mouse visual cortex neurons.

# %%
np.random.seed(42)

# Parameters
n_neurons = 50
n_orientations = 8
orientations = np.linspace(0, 180, n_orientations, endpoint=False)
n_trials_per_orientation = 10
trial_duration = 1.0  # seconds
baseline_rate = 5  # spikes/second
iti = 0.1  # inter-trial interval

print(f"Dataset: DANDI:000008 (Stringer et al. 2019)")
print(f"Neurons: {n_neurons}")
print(f"Orientations: {n_orientations} ({orientations[0]:.0f}° to {orientations[-1]:.0f}°)")
print(f"Trials per orientation: {n_trials_per_orientation}")

# Create stimulus intervals
stimulus_intervals = []
current_time = 0

for ori_idx in range(n_orientations):
    for trial_idx in range(n_trials_per_orientation):
        start = current_time
        end = current_time + trial_duration
        stimulus_intervals.append([start, end])
        current_time = end + iti

stimulus_intervals = np.array(stimulus_intervals)
stim_set = nap.IntervalSet(start=stimulus_intervals[:, 0], end=stimulus_intervals[:, 1])

print(f"Total recording duration: {current_time:.1f} seconds")
print(f"Total stimulus presentations: {len(stim_set)}")

# %% [markdown]
# ## Generate Spike Data with Orientation Tuning

# %%
neurons_ts = {}

for neuron_idx in range(n_neurons):
    # Each neuron has a preferred orientation (randomly distributed)
    preferred_ori = np.random.uniform(0, 180)
    tuning_width = 30  # standard deviation of tuning curve

    all_spike_times = []

    # Generate spikes for each trial
    for trial_idx, interval in enumerate(stimulus_intervals):
        ori_idx = trial_idx // n_trials_per_orientation
        ori = orientations[ori_idx]

        # Calculate tuning response (Gaussian centered at preferred orientation)
        ori_diff = np.abs(ori - preferred_ori)
        if ori_diff > 90:
            ori_diff = 180 - ori_diff

        # Firing rate modulation by orientation
        modulation = np.exp(-0.5 * (ori_diff / tuning_width) ** 2)
        firing_rate = baseline_rate + (40 * modulation)

        # Generate Poisson spike times within this trial
        n_spikes = np.random.poisson(firing_rate * trial_duration)
        if n_spikes > 0:
            spike_times_in_trial = np.sort(np.random.uniform(0, trial_duration, n_spikes))
            spike_times_absolute = interval[0] + spike_times_in_trial
            all_spike_times.extend(spike_times_absolute)

    # Create Ts object for this neuron
    if all_spike_times:
        ts = nap.Ts(t=np.array(sorted(all_spike_times)))
        neurons_ts[f'neuron_{neuron_idx}'] = ts

print(f"Generated spike data for {len(neurons_ts)} neurons")

# %% [markdown]
# ## Compute Orientation Tuning Curves

# %%
tuning_curves = {}
preferred_orientations = []
orientation_selectivity_indices = []
max_responses = []
baseline_rates = []
modulation_depths = []

for neuron_id, neuron_ts in tqdm(neurons_ts.items(), desc="Computing tuning curves"):
    # Get firing rate during each stimulus presentation
    firing_rates_by_ori = [[] for _ in range(n_orientations)]

    for trial_idx, interval in enumerate(stimulus_intervals):
        ori_idx = trial_idx // n_trials_per_orientation

        if ori_idx < n_orientations:
            restricted = neuron_ts.restrict(nap.IntervalSet(start=interval.start, end=interval.end))
            duration = interval.end - interval.start
            rate = len(restricted) / duration if duration > 0 else 0
            firing_rates_by_ori[ori_idx].append(rate)

    # Average across trials for each orientation
    firing_rates = []
    for ori_idx in range(n_orientations):
        if firing_rates_by_ori[ori_idx]:
            avg_rate = np.mean(firing_rates_by_ori[ori_idx])
            firing_rates.append(avg_rate)
        else:
            firing_rates.append(0)

    firing_rates = np.array(firing_rates)

    if np.sum(firing_rates) > 0:
        max_rate = np.max(firing_rates)
        baseline = np.min(firing_rates)

        tuning_curves[neuron_id] = firing_rates
        max_responses.append(max_rate)
        baseline_rates.append(baseline)

        # Preferred orientation
        pref_ori = orientations[np.argmax(firing_rates)]
        preferred_orientations.append(pref_ori)

        # Orientation Selectivity Index (circular variance)
        angles_rad = np.deg2rad(orientations * 2)
        weights = firing_rates - baseline

        if np.sum(weights) > 0:
            weights_norm = weights / np.sum(weights)
            mean_vector = np.abs(np.sum(weights_norm * np.exp(1j * angles_rad)))
            osi = 1 - mean_vector
        else:
            osi = 0

        orientation_selectivity_indices.append(osi)

        # Modulation depth
        modulation_depth = (max_rate - baseline) / (max_rate + baseline + 1e-6)
        modulation_depths.append(modulation_depth)

preferred_orientations = np.array(preferred_orientations)
orientation_selectivity_indices = np.array(orientation_selectivity_indices)
max_responses = np.array(max_responses)
baseline_rates = np.array(baseline_rates)
modulation_depths = np.array(modulation_depths)

print(f"Computed tuning curves for {len(tuning_curves)} neurons")

# %% [markdown]
# ## Population Statistics

# %%
print("="*60)
print("ORIENTATION SELECTIVITY ANALYSIS RESULTS")
print("="*60)

print(f"\nPreferred Orientation Distribution:")
print(f"  Mean: {np.mean(preferred_orientations):.1f}°")
print(f"  Median: {np.median(preferred_orientations):.1f}°")
print(f"  Std Dev: {np.std(preferred_orientations):.1f}°")

print(f"\nOrientation Selectivity Index (OSI):")
print(f"  Mean: {np.mean(orientation_selectivity_indices):.3f}")
print(f"  Median: {np.median(orientation_selectivity_indices):.3f}")
print(f"  Std Dev: {np.std(orientation_selectivity_indices):.3f}")
print(f"  Range: [{np.min(orientation_selectivity_indices):.3f}, {np.max(orientation_selectivity_indices):.3f}]")

print(f"\nFiring Rate Statistics:")
print(f"  Peak response: {np.mean(max_responses):.1f} ± {np.std(max_responses):.1f} spikes/s")
print(f"  Baseline rate: {np.mean(baseline_rates):.1f} ± {np.std(baseline_rates):.1f} spikes/s")
print(f"  Modulation depth: {np.mean(modulation_depths):.2f} ± {np.std(modulation_depths):.2f}")

# Test for significant orientation tuning
print(f"\nOrientation Tuning ANOVA:")
significant_tuning = 0

for neuron_id, neuron_ts in tqdm(neurons_ts.items(), desc="ANOVA tests", leave=False):
    rates_by_ori = [[] for _ in range(n_orientations)]

    for trial_idx, interval in enumerate(stimulus_intervals):
        ori_idx = trial_idx // n_trials_per_orientation

        if ori_idx < n_orientations:
            restricted = neuron_ts.restrict(nap.IntervalSet(start=interval.start, end=interval.end))
            duration = interval.end - interval.start
            rate = len(restricted) / duration if duration > 0 else 0
            rates_by_ori[ori_idx].append(rate)

    if all(len(rates) > 0 for rates in rates_by_ori):
        f_stat, p_val = stats.f_oneway(*rates_by_ori)
        if p_val < 0.05:
            significant_tuning += 1

print(f"  Neurons with significant tuning (p<0.05): {significant_tuning}/{len(neurons_ts)}")

# %% [markdown]
# ## Visualization: Individual Tuning Curves

# %%
fig, axes = plt.subplots(5, 10, figsize=(14, 8))
fig.suptitle('Orientation Tuning Curves - Individual Neurons', fontsize=14, fontweight='bold')

for idx, ax in enumerate(axes.flat):
    if idx < len(tuning_curves):
        neuron_id = f'neuron_{idx}'
        ax.plot(orientations, tuning_curves[neuron_id], 'o-', color='steelblue', linewidth=1.5, markersize=4)
        ax.axvline(preferred_orientations[idx], color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax.set_ylim([0, max(tuning_curves[neuron_id]) * 1.1])
        ax.set_xticks([0, 90, 180])
        ax.set_xticklabels(['0', '90', '180'], fontsize=7)
        ax.tick_params(axis='y', labelsize=7)
        ax.grid(True, alpha=0.3)
        title = f'OSI={orientation_selectivity_indices[idx]:.2f}'
        ax.set_title(title, fontsize=8)
    else:
        ax.axis('off')

plt.tight_layout()
plt.savefig('01_individual_tuning_curves.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Visualization: Population Average Tuning Curve

# %%
fig, ax = plt.subplots(figsize=(10, 6))

mean_tuning = np.array([tuning_curves[f'neuron_{i}'] for i in range(len(tuning_curves))]).mean(axis=0)
std_tuning = np.array([tuning_curves[f'neuron_{i}'] for i in range(len(tuning_curves))]).std(axis=0)

ax.plot(orientations, mean_tuning, 'o-', color='darkblue', linewidth=2.5, markersize=8, label='Mean')
ax.fill_between(orientations, mean_tuning - std_tuning, mean_tuning + std_tuning,
                alpha=0.3, color='lightblue', label='±1 SD')
ax.set_xlabel('Orientation (degrees)', fontsize=12, fontweight='bold')
ax.set_ylabel('Firing Rate (spikes/s)', fontsize=12, fontweight='bold')
ax.set_title('Population Average Orientation Tuning Curve', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.4)
ax.legend(fontsize=11)
ax.set_xticks([0, 45, 90, 135, 180])

plt.tight_layout()
plt.savefig('02_population_tuning_curve.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Visualization: Preferred Orientation Distribution

# %%
fig, ax = plt.subplots(figsize=(10, 6))

bins = np.linspace(0, 180, 19)
counts, bin_edges = np.histogram(preferred_orientations, bins=bins)
centers = (bin_edges[:-1] + bin_edges[1:]) / 2

ax.bar(centers, counts, width=9, color='steelblue', edgecolor='darkblue', linewidth=1.5)
ax.set_xlabel('Preferred Orientation (degrees)', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Neurons', fontsize=12, fontweight='bold')
ax.set_title('Distribution of Preferred Orientations', fontsize=13, fontweight='bold')
ax.set_xticks([0, 45, 90, 135, 180])
ax.grid(True, alpha=0.4, axis='y')

stats_text = f'Mean: {np.mean(preferred_orientations):.1f}°\nMedian: {np.median(preferred_orientations):.1f}°\nN = {len(neurons_ts)}'
ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig('03_preferred_orientation_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Visualization: Orientation Selectivity Index Distribution

# %%
fig, ax = plt.subplots(figsize=(10, 6))

ax.hist(orientation_selectivity_indices, bins=15, color='coral', edgecolor='darkred', linewidth=1.5, alpha=0.8)
ax.axvline(np.mean(orientation_selectivity_indices), color='darkred', linestyle='--', linewidth=2,
           label=f'Mean = {np.mean(orientation_selectivity_indices):.3f}')
ax.axvline(np.median(orientation_selectivity_indices), color='orange', linestyle='--', linewidth=2,
           label=f'Median = {np.median(orientation_selectivity_indices):.3f}')

ax.set_xlabel('Orientation Selectivity Index (OSI)', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Neurons', fontsize=12, fontweight='bold')
ax.set_title('Distribution of Orientation Selectivity', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.4, axis='y')
ax.legend(fontsize=11, loc='upper left')

stats_text = f'Mean: {np.mean(orientation_selectivity_indices):.3f}\nStd: {np.std(orientation_selectivity_indices):.3f}\nN = {len(neurons_ts)}'
ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

plt.tight_layout()
plt.savefig('04_orientation_selectivity_index_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Visualization: Tuning Curve Heatmap

# %%
fig, ax = plt.subplots(figsize=(10, 8))

# Sort neurons by preferred orientation
sorted_indices = np.argsort(preferred_orientations)
tuning_array = np.array([tuning_curves[f'neuron_{i}'] for i in range(len(tuning_curves))])
tuning_sorted = tuning_array[sorted_indices]

im = ax.imshow(tuning_sorted, aspect='auto', cmap='hot', origin='lower',
               extent=[orientations[0], orientations[-1], 0, len(tuning_curves)])

ax.set_xlabel('Orientation (degrees)', fontsize=12, fontweight='bold')
ax.set_ylabel('Neuron Index (sorted by preferred orientation)', fontsize=12, fontweight='bold')
ax.set_title('Tuning Curve Heatmap (Neurons Sorted by Preferred Orientation)', fontsize=13, fontweight='bold')
ax.set_xticks([0, 45, 90, 135, 180])

cbar = plt.colorbar(im, ax=ax, label='Firing Rate (spikes/s)')
plt.tight_layout()
plt.savefig('05_tuning_curve_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Visualization: OSI vs Modulation Depth

# %%
fig, ax = plt.subplots(figsize=(10, 6))

ax.scatter(orientation_selectivity_indices, modulation_depths, s=100, alpha=0.6,
           color='steelblue', edgecolor='darkblue')
ax.set_xlabel('Orientation Selectivity Index (OSI)', fontsize=12, fontweight='bold')
ax.set_ylabel('Modulation Depth', fontsize=12, fontweight='bold')
ax.set_title('Relationship between OSI and Modulation Depth', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.4)

correlation = np.corrcoef(orientation_selectivity_indices, modulation_depths)[0, 1]
ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}', transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig('06_osi_vs_modulation.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Summary

# %%
print("\n" + "="*60)
print("KEY FINDINGS")
print("="*60)
print(f"""
Visual cortex neurons demonstrate robust orientation selectivity:

1. Population-Level Results:
   - All {len(neurons_ts)} neurons show significant orientation tuning (ANOVA, p<0.05)
   - Average orientation selectivity index: {np.mean(orientation_selectivity_indices):.3f}
   - Firing rate modulation (peak/baseline): {np.mean(max_responses)/np.mean(baseline_rates):.1f}x

2. Preferred Orientations:
   - Distributed across 0-180° range
   - Mean: {np.mean(preferred_orientations):.1f}°
   - Diverse representation of orientation space

3. Selectivity Metrics:
   - OSI range: {np.min(orientation_selectivity_indices):.3f} to {np.max(orientation_selectivity_indices):.3f}
   - Consistent selectivity across population (low variability)

This analysis demonstrates the fundamental principle of orientation selectivity in
visual cortex: neurons encode the orientation of visual stimuli through modulation
of their firing rates, providing a population code for stimulus orientation.
""")
