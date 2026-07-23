"""
Analyze orientation selectivity in visual cortex.
Computes tuning curves, identifies selective neurons, and tests significance.
"""

import numpy as np
import pynapple as nap
import pickle
import json
from pathlib import Path
from scipy import stats
from tqdm import tqdm

print("="*60)
print("ORIENTATION TUNING ANALYSIS")
print("="*60)

# Load data
with open('pynapple_data.pkl', 'rb') as f:
    data = pickle.load(f)

neurons = data['neurons']
stimulus_intervals = data['stimulus_intervals']
orientations = data['orientations']
n_trials_per_orientation = data['n_trials_per_orientation']

print(f"\nLoaded data for {len(neurons)} neurons")
print(f"Stimulus intervals: {len(stimulus_intervals)}")
print(f"Orientations: {orientations}")

# Get the stimulus-orientation mapping
n_orientations = len(orientations)

print(f"Trials per orientation: {n_trials_per_orientation}")

# Create orientation label for each interval
orientation_labels = []
for ori_idx in range(n_orientations):
    for trial_idx in range(n_trials_per_orientation):
        orientation_labels.append(orientations[ori_idx])

orientation_labels = np.array(orientation_labels)

# Initialize results storage
results = {
    'neurons': {},
    'population_stats': {}
}

# Compute tuning curves for each neuron
print("\nComputing orientation tuning curves...")

tuning_curves = {}
preferred_orientations = []
orientation_selectivity_indices = []
max_responses = []
baseline_rates = []

for neuron_idx, (neuron_id, neuron_ts) in enumerate(tqdm(neurons.items(), desc="Neurons")):
    # Get firing rate during each stimulus presentation
    firing_rates_by_ori = [[] for _ in range(n_orientations)]

    for trial_idx, interval in enumerate(stimulus_intervals):
        # Get orientation for this trial
        ori_idx = trial_idx // n_trials_per_orientation

        if ori_idx < n_orientations:
            restricted = neuron_ts.restrict(nap.IntervalSet(start=interval.start, end=interval.end))

            # Firing rate = spike count / interval duration
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

    # Only analyze if we have data for all orientations
    if len(firing_rates) == n_orientations and np.sum(firing_rates) > 0:
        max_rate = np.max(firing_rates)
        baseline = np.min(firing_rates)

        # Compute metrics
        tuning_curves[neuron_id] = firing_rates
        max_responses.append(max_rate)
        baseline_rates.append(baseline)

        # Preferred orientation (orientation with maximum firing)
        pref_ori = orientations[np.argmax(firing_rates)]
        preferred_orientations.append(pref_ori)

        # Circular variance as orientation selectivity index
        # Convert orientations to complex exponentials (double angles for 0-180 range)
        angles_rad = np.deg2rad(orientations * 2)
        weights = firing_rates - baseline

        if np.sum(weights) > 0:
            weights_norm = weights / np.sum(weights)
            mean_vector = np.abs(np.sum(weights_norm * np.exp(1j * angles_rad)))
            osi = 1 - mean_vector  # Orientation Selectivity Index
        else:
            osi = 0

        orientation_selectivity_indices.append(osi)

        # Store neuron results
        results['neurons'][neuron_id] = {
            'firing_rates': firing_rates.tolist(),
            'preferred_orientation': float(pref_ori),
            'orientation_selectivity_index': float(osi),
            'max_response': float(max_rate),
            'baseline_rate': float(baseline),
            'modulation_depth': float((max_rate - baseline) / (max_rate + baseline + 1e-6))
        }

print(f"Computed tuning curves for {len(tuning_curves)} neurons")

if len(preferred_orientations) == 0:
    print("ERROR: No neurons with valid tuning curves!")
    exit(1)

# Population statistics
preferred_orientations = np.array(preferred_orientations)
orientation_selectivity_indices = np.array(orientation_selectivity_indices)

print("\n" + "="*60)
print("POPULATION STATISTICS")
print("="*60)

# Preferred orientation distribution
print(f"\nPreferred Orientation Distribution:")
print(f"  Mean: {np.mean(preferred_orientations):.1f}°")
print(f"  Median: {np.median(preferred_orientations):.1f}°")
print(f"  Std Dev: {np.std(preferred_orientations):.1f}°")

# Orientation selectivity index
print(f"\nOrientation Selectivity Index (OSI):")
print(f"  Mean: {np.mean(orientation_selectivity_indices):.3f}")
print(f"  Median: {np.median(orientation_selectivity_indices):.3f}")
print(f"  Std Dev: {np.std(orientation_selectivity_indices):.3f}")
print(f"  Range: [{np.min(orientation_selectivity_indices):.3f}, {np.max(orientation_selectivity_indices):.3f}]")

# Identify significantly orientation-selective neurons
if np.std(orientation_selectivity_indices) > 0:
    osi_threshold = np.median(orientation_selectivity_indices) + np.std(orientation_selectivity_indices)
else:
    osi_threshold = np.median(orientation_selectivity_indices)

selective_neurons = np.where(orientation_selectivity_indices > osi_threshold)[0]

print(f"\nOrientation-Selective Neurons (OSI > {osi_threshold:.3f}):")
print(f"  Count: {len(selective_neurons)} / {len(neurons)} ({100*len(selective_neurons)/len(neurons):.1f}%)")

# Firing rate statistics
print(f"\nFiring Rate Statistics:")
print(f"  Max response: {np.mean(max_responses):.1f} ± {np.std(max_responses):.1f} spikes/s")
print(f"  Baseline rate: {np.mean(baseline_rates):.1f} ± {np.std(baseline_rates):.1f} spikes/s")

modulation_depths = [(m - b) / (m + b + 1e-6) for m, b in zip(max_responses, baseline_rates)]
print(f"  Modulation depth: {np.mean(modulation_depths):.2f} ± {np.std(modulation_depths):.2f}")

# Test for significant orientation tuning using ANOVA
print("\nOrientation Tuning ANOVA:")
significant_tuning = 0

for neuron_id, neuron_ts in tqdm(neurons.items(), desc="ANOVA tests", leave=False):
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

print(f"  Neurons with significant orientation tuning (p < 0.05): {significant_tuning}/{len(neurons)}")

results['population_stats'] = {
    'total_neurons': len(neurons),
    'neurons_with_tuning': len(tuning_curves),
    'mean_preferred_orientation': float(np.mean(preferred_orientations)),
    'mean_osi': float(np.mean(orientation_selectivity_indices)),
    'median_osi': float(np.median(orientation_selectivity_indices)),
    'std_osi': float(np.std(orientation_selectivity_indices)),
    'selective_neurons_count': int(len(selective_neurons)),
    'selective_neurons_percent': float(100 * len(selective_neurons) / len(neurons)),
    'neurons_with_significant_tuning': int(significant_tuning),
    'mean_max_response': float(np.mean(max_responses)),
    'mean_baseline_rate': float(np.mean(baseline_rates)),
    'mean_modulation_depth': float(np.mean(modulation_depths))
}

# Save results
with open('orientation_tuning_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
print(f"Results saved to: orientation_tuning_results.json")

# Save tuning curves for visualization
neuron_ids = sorted(tuning_curves.keys())
tuning_array = np.array([tuning_curves[nid] for nid in neuron_ids])

np.save('tuning_curves.npy', tuning_array)
np.save('preferred_orientations.npy', preferred_orientations)
np.save('orientation_selectivity_indices.npy', orientation_selectivity_indices)
np.save('orientations.npy', orientations)
np.save('neuron_ids.npy', np.array(neuron_ids, dtype=object))

print("Tuning curve data saved for visualization")
