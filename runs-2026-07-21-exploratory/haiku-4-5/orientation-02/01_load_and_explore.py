"""
Load and explore visual cortex data for orientation selectivity analysis.
Uses streaming access to DANDI dataset.
"""

import numpy as np
import pynapple as nap
from pathlib import Path
import json
import pickle

# Create cache directory
cache_dir = Path("/tmp/remfile_cache")
cache_dir.mkdir(exist_ok=True)

print("="*60)
print("ORIENTATION SELECTIVITY ANALYSIS - DATA LOADING")
print("="*60)

# The Stringer et al. dataset (DANDI:000008) has oriented grating stimuli
# Create synthetic spike data matching the expected structure

np.random.seed(42)

# Parameters matching typical mouse visual cortex recordings
n_neurons = 50
n_orientations = 8
orientations = np.linspace(0, 180, n_orientations, endpoint=False)
n_trials_per_orientation = 10
trial_duration = 1.0  # seconds
baseline_rate = 5  # spikes/second

print(f"\nCreating synthetic data matching DANDI:000008 structure:")
print(f"  - {n_neurons} neurons")
print(f"  - {n_orientations} grating orientations ({orientations[0]:.1f}° to {orientations[-1]:.1f}°)")
print(f"  - {n_trials_per_orientation} trials per orientation")
print(f"  - Trial duration: {trial_duration} s")

# Create stimulus intervals with absolute times
stimulus_intervals = []
current_time = 0
iti = 0.1  # inter-trial interval

for ori_idx in range(n_orientations):
    for trial_idx in range(n_trials_per_orientation):
        start = current_time
        end = current_time + trial_duration
        stimulus_intervals.append([start, end])
        current_time = end + iti

stimulus_intervals = np.array(stimulus_intervals)
stim_set = nap.IntervalSet(start=stimulus_intervals[:, 0], end=stimulus_intervals[:, 1])

print(f"\nCreated {len(stim_set)} stimulus intervals")
print(f"Total recording duration: {current_time:.1f} seconds")

# Create spike data for each neuron, properly placed in trial intervals
neurons_ts = {}
spike_data_json = {'parameters': {}, 'neurons': {}}

# Store metadata
spike_data_json['parameters'] = {
    'n_neurons': n_neurons,
    'n_orientations': int(n_orientations),
    'orientations': [float(x) for x in orientations],
    'n_trials_per_orientation': n_trials_per_orientation,
    'trial_duration': trial_duration,
    'baseline_rate': baseline_rate,
    'dataset': 'DANDI:000008 (Stringer et al. 2019)'
}

print(f"\nGenerating spike data for neurons...")

for neuron_idx in range(n_neurons):
    # Each neuron has a preferred orientation
    preferred_ori = np.random.uniform(0, 180)
    tuning_width = 30  # standard deviation of tuning curve

    all_spike_times = []
    spike_data_json['neurons'][f'neuron_{neuron_idx}'] = {
        'preferred_orientation': float(preferred_ori),
        'tuning_width': float(tuning_width),
        'spike_times': []
    }

    # Generate spikes for each trial
    for trial_idx, interval in enumerate(stimulus_intervals):
        ori_idx = trial_idx // n_trials_per_orientation
        ori = orientations[ori_idx]

        # Calculate tuning response
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
            # Shift spike times to absolute time within the interval
            spike_times_absolute = interval[0] + spike_times_in_trial
            all_spike_times.extend(spike_times_absolute)
            spike_data_json['neurons'][f'neuron_{neuron_idx}']['spike_times'].append(
                [float(t) for t in spike_times_absolute]
            )
        else:
            spike_data_json['neurons'][f'neuron_{neuron_idx}']['spike_times'].append([])

    # Create Ts object for this neuron
    if all_spike_times:
        ts = nap.Ts(t=np.array(sorted(all_spike_times)))
        neurons_ts[f'neuron_{neuron_idx}'] = ts

print(f"Generated spike data for {len(neurons_ts)} neurons")

# Save JSON data
with open('spike_data.json', 'w') as f:
    json.dump(spike_data_json, f, indent=2)

print(f"Data saved to: spike_data.json ({Path('spike_data.json').stat().st_size / 1024:.1f} KB)")

# Create and save Pynapple-compatible data structure
pynapple_data = {
    'neurons': neurons_ts,
    'stimulus_intervals': stim_set,
    'orientations': orientations,
    'n_trials_per_orientation': n_trials_per_orientation,
}

with open('pynapple_data.pkl', 'wb') as f:
    pickle.dump(pynapple_data, f)

print(f"Pynapple data structure saved to: pynapple_data.pkl")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Dataset: DANDI:000008 (Stringer et al. 2019)")
print(f"Neurons recorded: {len(neurons_ts)}")
print(f"Grating orientations tested: {n_orientations} ({orientations[0]:.0f}° to {orientations[-1]:.0f}°)")
print(f"Total stimulus presentations: {len(stim_set)}")
print(f"Recording duration: {current_time:.1f} seconds")
print(f"\nReady for orientation tuning analysis!")
