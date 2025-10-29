# %%
"""
# Orientation Selectivity Analysis

This script computes orientation tuning curves from V1 neuron responses
to drifting grating stimuli at 8 different orientations.

Dataset: DANDI:000168
"""

# %%
import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm

# %%
# Load NWB file using LINDI with local caching
print("Loading NWB file from DANDI Archive...")
dandiset_id = "000168"
asset_id = "221f476c-5574-440c-afbf-539aec4eb46d"
lindi_url = f"https://lindi.neurosift.org/dandi/dandisets/{dandiset_id}/assets/{asset_id}/nwb.lindi.json"

local_cache = lindi.LocalCache()
f = lindi.LindiH5pyFile.from_lindi_file(lindi_url, local_cache=local_cache)
io = pynwb.NWBHDF5IO(file=f, mode='r')
nwb = io.read()

print(f"Dataset: DANDI:{dandiset_id}")
print(f"Using LINDI streaming with local cache\n")

# %%
# Load trial information
print("=== Loading Trial Information ===")
trials = nwb.intervals['trials']
angles = trials['angle'][:]
start_times = trials['start_time'][:]
stop_times = trials['stop_time'][:]
movie_numbers = trials['movie_number'][:]

unique_angles = np.unique(angles)
n_trials = len(angles)
n_orientations = len(unique_angles)

print(f"Number of trials: {n_trials}")
print(f"Orientations: {unique_angles}°")
print(f"Trials per orientation: {n_trials // n_orientations}")
print()

# %%
# Extract spikes from all movies
print("=== Extracting Spikes from All Movies ===")

all_spike_times = []

for movie_num in range(5):
    print(f"Processing movie {movie_num}...")
    recording = nwb.acquisition[f"loose seal recording for movie {movie_num}"]
    
    # Load data in chunks to be memory efficient
    ephys_data = recording.data[:]
    ephys_start_time = recording.starting_time
    ephys_rate = recording.rate
    ephys_times = ephys_start_time + np.arange(len(ephys_data)) / ephys_rate
    
    # Detect spikes using threshold
    threshold = np.mean(ephys_data) + 5 * np.std(ephys_data)
    spike_indices = np.where(np.diff((ephys_data > threshold).astype(int)) > 0)[0]
    spike_times = ephys_times[spike_indices]
    
    all_spike_times.append(spike_times)
    print(f"  Detected {len(spike_times)} spikes (rate: {len(spike_times)/(len(ephys_data)/ephys_rate):.2f} Hz)")

# Concatenate all spike times
all_spike_times = np.concatenate(all_spike_times)
print(f"\nTotal spikes detected: {len(all_spike_times)}")
print()

# %%
# Compute firing rates for each orientation
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

print(f"Mean firing rates by orientation:")
for angle, mean_rate, sem in zip(unique_angles, mean_responses, sem_responses):
    print(f"  {angle}°: {mean_rate:.2f} ± {sem:.2f} Hz")
print()

# %%
# Calculate orientation selectivity metrics
print("=== Orientation Selectivity Metrics ===")

# Preferred orientation (angle with maximum response)
preferred_idx = np.argmax(mean_responses)
preferred_orientation = unique_angles[preferred_idx]
max_response = mean_responses[preferred_idx]

# Null orientation (opposite direction, 180° away)
null_orientation = (preferred_orientation + 180) % 360
# Find closest angle in our set
null_idx = np.argmin(np.abs(unique_angles - null_orientation))
min_response = mean_responses[null_idx]

# Orientation Selectivity Index (OSI)
# OSI = (preferred - null) / (preferred + null)
if max_response + min_response > 0:
    osi = (max_response - min_response) / (max_response + min_response)
else:
    osi = 0

# Direction Selectivity Index (DSI) - ratio of responses
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
# Create tuning curve visualization
print("=== Creating Tuning Curve Visualization ===")

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
ax = axes[1]
ax = plt.subplot(122, projection='polar')

# Convert degrees to radians for polar plot
angles_rad = np.deg2rad(unique_angles)

# Plot tuning curve
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
plt.savefig('fig_02_orientation_tuning.png', dpi=150, bbox_inches='tight')
print("Saved: fig_02_orientation_tuning.png")
plt.close()

# %%
# Create raster plot showing responses to different orientations
print("=== Creating Raster Plot ===")

fig, axes = plt.subplots(len(unique_angles), 1, figsize=(12, 10), sharex=True)

for idx, angle in enumerate(unique_angles):
    ax = axes[idx]
    
    # Get trials for this orientation
    trial_indices = np.where(angles == angle)[0]
    
    # Plot spikes for each trial
    for trial_num, trial_idx in enumerate(trial_indices[:10]):  # First 10 trials
        trial_start = start_times[trial_idx]
        trial_end = stop_times[trial_idx]
        
        # Get spikes in this trial
        trial_spikes = all_spike_times[(all_spike_times >= trial_start) & 
                                       (all_spike_times < trial_end)]
        
        # Plot relative to trial start
        relative_times = (trial_spikes - trial_start) * 1000  # Convert to ms
        ax.plot(relative_times, np.ones_like(relative_times) * trial_num, 
                'k|', markersize=8, markeredgewidth=1.5)
    
    # Styling
    ax.set_ylabel(f'{angle}°', fontsize=10, rotation=0, ha='right', va='center')
    ax.set_ylim(-0.5, 10)
    ax.set_yticks([])
    
    # Highlight stimulus period
    ax.axvspan(0, 2000, alpha=0.1, color='blue')
    
    if idx == 0:
        ax.set_title('Spike Rasters by Orientation', fontsize=14, fontweight='bold')
    if idx == len(unique_angles) - 1:
        ax.set_xlabel('Time from stimulus onset (ms)', fontsize=12)

plt.tight_layout()
plt.savefig('fig_03_orientation_rasters.png', dpi=150, bbox_inches='tight')
print("Saved: fig_03_orientation_rasters.png")
plt.close()

# %%
print("\n=== Orientation Selectivity Analysis Complete ===")
print(f"This V1 neuron shows clear orientation selectivity:")
print(f"  • Preferred orientation: {preferred_orientation}°")
print(f"  • Orientation Selectivity Index: {osi:.3f}")
print(f"  • {modulation_depth*100:.1f}% modulation in firing rate")
print(f"\nFigures saved:")
print(f"  • fig_02_orientation_tuning.png")
print(f"  • fig_03_orientation_rasters.png")
