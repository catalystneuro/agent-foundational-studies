# %%
"""
# Orientation Selectivity in Visual Cortex

This analysis demonstrates orientation selectivity in mouse primary visual cortex (V1) 
using data from DANDI:000168. The dataset contains simultaneous electrophysiology and 
two-photon imaging from V1 neurons responding to drifting gratings at 8 orientations.

**Dataset**: [DANDI:000168](https://neurosift.app/dandiset/000168)

**Recording**: Svoboda Lab, HHMI Janelia Research Campus
- Two-photon calcium imaging + loose-seal cell-attached recording
- V1 Layer 2/3 pyramidal neurons
- Full-field drifting gratings in 8 directions (45° spacing)
"""

# %%
# ## Setup and Data Loading

import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm

print("Loading NWB file from DANDI Archive...")
dandiset_id = "000168"
asset_id = "221f476c-5574-440c-afbf-539aec4eb46d"
lindi_url = f"https://lindi.neurosift.org/dandi/dandisets/{dandiset_id}/assets/{asset_id}/nwb.lindi.json"

# Create local cache for efficient streaming
local_cache = lindi.LocalCache()
f = lindi.LindiH5pyFile.from_lindi_file(lindi_url, local_cache=local_cache)
io = pynwb.NWBHDF5IO(file=f, mode='r')
nwb = io.read()

print(f"Dataset: DANDI:{dandiset_id}")
print(f"Using LINDI streaming with local cache\n")

# %%
# ## Session and Subject Information

print("=== Session Information ===")
print(f"Identifier: {nwb.identifier}")
print(f"Session start: {nwb.session_start_time}")
print(f"Institution: {nwb.institution}")
print(f"Lab: {nwb.lab}\n")

print("=== Subject Information ===")
print(f"Species: {nwb.subject.species}")
print(f"Sex: {nwb.subject.sex}")
print(f"Subject ID: {nwb.subject.subject_id}\n")

# %%
# ## Stimulus Information

print("=== Stimulus Information ===")
trials = nwb.intervals['trials']
angles = trials['angle'][:]
start_times = trials['start_time'][:]
stop_times = trials['stop_time'][:]

unique_angles = np.unique(angles)
n_trials = len(angles)
n_orientations = len(unique_angles)

print(f"Number of trials: {n_trials}")
print(f"Orientations tested: {unique_angles}°")
print(f"Trials per orientation: {n_trials // n_orientations}")
print(f"Average trial duration: {np.mean(stop_times - start_times):.2f}s\n")

# %%
# ## Extract Spikes from Electrophysiology Data

print("=== Extracting Spikes from All Recordings ===")
all_spike_times = []

for movie_num in range(5):
    print(f"Processing movie {movie_num}...")
    recording = nwb.acquisition[f"loose seal recording for movie {movie_num}"]
    
    # Load electrophysiology data
    ephys_data = recording.data[:]
    ephys_start_time = recording.starting_time
    ephys_rate = recording.rate
    ephys_times = ephys_start_time + np.arange(len(ephys_data)) / ephys_rate
    
    # Simple spike detection using threshold
    threshold = np.mean(ephys_data) + 5 * np.std(ephys_data)
    spike_indices = np.where(np.diff((ephys_data > threshold).astype(int)) > 0)[0]
    spike_times = ephys_times[spike_indices]
    
    all_spike_times.append(spike_times)
    print(f"  Detected {len(spike_times)} spikes (rate: {len(spike_times)/(len(ephys_data)/ephys_rate):.2f} Hz)")

all_spike_times = np.concatenate(all_spike_times)
print(f"\nTotal spikes detected: {len(all_spike_times)}")
print(f"Overall firing rate: {len(all_spike_times) / (stop_times[-1] - start_times[0]):.2f} Hz\n")

# %%
# ## Compute Orientation Tuning Curves

print("=== Computing Orientation Tuning ===")

# Organize trials by orientation
orientation_responses = {angle: [] for angle in unique_angles}

for trial_idx in tqdm(range(n_trials), desc="Processing trials"):
    trial_start = start_times[trial_idx]
    trial_end = stop_times[trial_idx]
    trial_angle = angles[trial_idx]
    
    # Count spikes in this trial
    trial_spikes = all_spike_times[(all_spike_times >= trial_start) & 
                                   (all_spike_times < trial_end)]
    
    # Calculate firing rate
    trial_duration = trial_end - trial_start
    firing_rate = len(trial_spikes) / trial_duration
    
    orientation_responses[trial_angle].append(firing_rate)

# Calculate mean and SEM for each orientation
mean_responses = []
sem_responses = []

for angle in unique_angles:
    responses = np.array(orientation_responses[angle])
    mean_responses.append(np.mean(responses))
    sem_responses.append(stats.sem(responses))

mean_responses = np.array(mean_responses)
sem_responses = np.array(sem_responses)

print("\nMean firing rates by orientation:")
for angle, mean_rate, sem in zip(unique_angles, mean_responses, sem_responses):
    print(f"  {angle}°: {mean_rate:.2f} ± {sem:.2f} Hz")
print()

# %%
# ## Calculate Orientation Selectivity Metrics

print("=== Orientation Selectivity Metrics ===")

# Preferred orientation (angle with maximum response)
preferred_idx = np.argmax(mean_responses)
preferred_orientation = unique_angles[preferred_idx]
max_response = mean_responses[preferred_idx]

# Null orientation (opposite direction, 180° away)
null_orientation = (preferred_orientation + 180) % 360
null_idx = np.argmin(np.abs(unique_angles - null_orientation))
min_response = mean_responses[null_idx]

# Orientation Selectivity Index (OSI)
# OSI = (preferred - null) / (preferred + null)
if max_response + min_response > 0:
    osi = (max_response - min_response) / (max_response + min_response)
else:
    osi = 0

# Direction Selectivity Index (DSI)
if max_response > 0:
    dsi = 1 - (min_response / max_response)
else:
    dsi = 0

# Modulation depth
baseline = np.min(mean_responses)
modulation_depth = (max_response - baseline) / (max_response + baseline) if (max_response + baseline) > 0 else 0

print(f"Preferred orientation: {preferred_orientation}°")
print(f"Max response: {max_response:.2f} Hz")
print(f"Null response: {min_response:.2f} Hz")
print(f"Orientation Selectivity Index (OSI): {osi:.3f}")
print(f"Direction Selectivity Index (DSI): {dsi:.3f}")
print(f"Modulation depth: {modulation_depth:.3f}")
print()

# %%
# ## Visualization 1: Orientation Tuning Curves

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel 1: Cartesian tuning curve
ax = axes[0]
ax.errorbar(unique_angles, mean_responses, yerr=sem_responses, 
            marker='o', markersize=8, linewidth=2, capsize=5,
            color='darkblue', label='Mean ± SEM')
ax.axhline(baseline, color='gray', linestyle='--', alpha=0.5, label='Baseline')
ax.axvline(preferred_orientation, color='red', linestyle='--', alpha=0.5, 
           label=f'Preferred ({preferred_orientation}°)')
ax.set_xlabel('Orientation (degrees)', fontsize=12)
ax.set_ylabel('Firing Rate (Hz)', fontsize=12)
ax.set_title('Orientation Tuning Curve', fontsize=14, fontweight='bold')
ax.set_xticks(unique_angles)
ax.grid(True, alpha=0.3)
ax.legend()

# Panel 2: Polar tuning curve
ax = plt.subplot(122, projection='polar')

angles_rad = np.deg2rad(unique_angles)

ax.plot(angles_rad, mean_responses, 'o-', linewidth=2, markersize=8, 
        color='darkblue', label='Mean response')
ax.fill_between(angles_rad, 
                mean_responses - sem_responses,
                mean_responses + sem_responses,
                alpha=0.3, color='blue')

# Mark preferred orientation
pref_rad = np.deg2rad(preferred_orientation)
ax.plot([pref_rad, pref_rad], [0, max_response], 'r--', linewidth=2, 
        label=f'Preferred ({preferred_orientation}°)')

ax.set_theta_zero_location('E')
ax.set_theta_direction(1)
ax.set_title(f'Polar Tuning Curve\nOSI = {osi:.3f}', 
             fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

plt.tight_layout()
plt.savefig('fig_orientation_tuning.png', dpi=150, bbox_inches='tight')
print("Saved: fig_orientation_tuning.png")
plt.show()

# %%
# ## Visualization 2: Spike Rasters by Orientation

fig, axes = plt.subplots(len(unique_angles), 1, figsize=(12, 10), sharex=True)

for idx, angle in enumerate(unique_angles):
    ax = axes[idx]
    
    # Get trials for this orientation
    trial_indices = np.where(angles == angle)[0]
    
    # Plot spikes for each trial (first 10 trials)
    for trial_num, trial_idx in enumerate(trial_indices[:10]):
        trial_start = start_times[trial_idx]
        trial_end = stop_times[trial_idx]
        
        # Get spikes in this trial
        trial_spikes = all_spike_times[(all_spike_times >= trial_start) & 
                                       (all_spike_times < trial_end)]
        
        # Plot relative to trial start (in milliseconds)
        relative_times = (trial_spikes - trial_start) * 1000
        ax.plot(relative_times, np.ones_like(relative_times) * trial_num, 
                'k|', markersize=8, markeredgewidth=1.5)
    
    # Styling
    ax.set_ylabel(f'{angle}°', fontsize=10, rotation=0, ha='right', va='center')
    ax.set_ylim(-0.5, 10)
    ax.set_yticks([])
    
    # Highlight stimulus period (2 seconds)
    ax.axvspan(0, 2000, alpha=0.1, color='blue')
    
    if idx == 0:
        ax.set_title('Spike Rasters by Orientation', fontsize=14, fontweight='bold')
    if idx == len(unique_angles) - 1:
        ax.set_xlabel('Time from stimulus onset (ms)', fontsize=12)

plt.tight_layout()
plt.savefig('fig_orientation_rasters.png', dpi=150, bbox_inches='tight')
print("Saved: fig_orientation_rasters.png")
plt.show()

# %%
# ## Summary

print("\n" + "="*70)
print("ORIENTATION SELECTIVITY ANALYSIS - RESULTS")
print("="*70)
print(f"\nDataset: DANDI:000168 - V1 neuron responses to drifting gratings")
print(f"Subject: {nwb.subject.species}, Subject ID: {nwb.subject.subject_id}")
print(f"\n--- Neural Response Properties ---")
print(f"Overall firing rate: {len(all_spike_times) / (stop_times[-1] - start_times[0]):.2f} Hz")
print(f"Total spikes analyzed: {len(all_spike_times)}")
print(f"\n--- Orientation Selectivity ---")
print(f"Preferred orientation: {preferred_orientation}°")
print(f"Peak firing rate: {max_response:.2f} Hz")
print(f"Minimum firing rate: {baseline:.2f} Hz")
print(f"Orientation Selectivity Index (OSI): {osi:.3f}")
print(f"Direction Selectivity Index (DSI): {dsi:.3f}")
print(f"Modulation depth: {modulation_depth*100:.1f}%")
print(f"\n--- Interpretation ---")
if osi > 0.3:
    print("✓ This neuron is STRONGLY orientation-selective")
elif osi > 0.1:
    print("✓ This neuron shows MODERATE orientation selectivity")
else:
    print("• This neuron shows WEAK orientation selectivity")
print(f"\nThe neuron responds preferentially to gratings oriented at {preferred_orientation}°,")
print(f"with a {modulation_depth*100:.1f}% difference between preferred and least-preferred orientations.")
print(f"\nThis demonstrates the fundamental property of orientation selectivity")
print(f"in visual cortex, first described by Hubel & Wiesel (1962).")
print("\n" + "="*70)
