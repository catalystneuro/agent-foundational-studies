# %%
"""
# Orientation Selectivity in Visual Cortex - Data Exploration

This script loads and explores an NWB file from DANDI:000168 containing 
V1 neuron responses to drifting grating stimuli at different orientations.

Dataset: DANDI:000168
"""

# %%
import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt

# %%
# Load NWB file using LINDI with local caching
print("Loading NWB file from DANDI Archive...")
dandiset_id = "000168"
asset_id = "221f476c-5574-440c-afbf-539aec4eb46d"
lindi_url = f"https://lindi.neurosift.org/dandi/dandisets/{dandiset_id}/assets/{asset_id}/nwb.lindi.json"

# Create local cache
local_cache = lindi.LocalCache()
f = lindi.LindiH5pyFile.from_lindi_file(lindi_url, local_cache=local_cache)
io = pynwb.NWBHDF5IO(file=f, mode='r')
nwb = io.read()

print(f"Dataset: DANDI:{dandiset_id}")
print(f"Using LINDI streaming with local cache\n")

# %%
# Display session information
print("=== Session Information ===")
print(f"Session description: {nwb.session_description}")
print(f"Identifier: {nwb.identifier}")
print(f"Session start: {nwb.session_start_time}")
print(f"Experiment: {nwb.experiment_description[:150]}...")
print(f"Institution: {nwb.institution}")
print(f"Lab: {nwb.lab}")
print()

# %%
# Display subject information
print("=== Subject Information ===")
print(f"Species: {nwb.subject.species}")
print(f"Sex: {nwb.subject.sex}")
print(f"Subject ID: {nwb.subject.subject_id}")
print(f"Description: {nwb.subject.description}")
print(f"Date of birth: {nwb.subject.date_of_birth}")
print()

# %%
# Explore trial structure
print("=== Trial Information ===")
trials = nwb.intervals['trials']
print(f"Number of trials: {len(trials)}")
print(f"\nTrial columns: {trials.colnames}")
print()

# Load trial data
angles = trials['angle'][:]
start_times = trials['start_time'][:]
stop_times = trials['stop_time'][:]
amplitudes = trials['amplitude'][:]
cpd = trials['cycles_per_degree'][:]
cps = trials['cycles_per_second'][:]
movie_numbers = trials['movie_number'][:]

print(f"Stimulus angles: {np.unique(angles)} degrees")
print(f"Number of unique orientations: {len(np.unique(angles))}")
print(f"Trials per orientation: {len(angles) // len(np.unique(angles))}")
print()

# %%
# Show example trials
print("Example trials:")
for i in range(8):
    print(f"  Trial {i}: angle={angles[i]}°, duration={stop_times[i]-start_times[i]:.2f}s, "
          f"amplitude={amplitudes[i]}, movie={movie_numbers[i]}")
print()

# %%
# Explore electrophysiology data
print("=== Electrophysiology Data ===")
print("Available acquisition data:")
for key in nwb.acquisition.keys():
    if 'loose seal' in key:
        data = nwb.acquisition[key]
        print(f"  {key}:")
        print(f"    Start time: {data.starting_time} s")
        print(f"    Sampling rate: {data.rate} Hz")
        print(f"    Data shape: {data.data.shape}")
print()

# %%
# Explore imaging data
print("=== Imaging Data ===")
print("Processing modules:")
for module_name in nwb.processing.keys():
    print(f"  {module_name}")
print()

# Check fluorescence data from first movie
if "ophys of movie 0" in nwb.processing:
    ophys = nwb.processing["ophys of movie 0"]
    if "Fluorescence" in ophys.data_interfaces:
        fluor = ophys.data_interfaces["Fluorescence"]
        print("Fluorescence data available:")
        for roi_name in fluor.roi_response_series.keys():
            roi_data = fluor.roi_response_series[roi_name]
            print(f"  {roi_name}: shape={roi_data.data.shape}")
print()

# %%
# Extract spike times from electrophysiology data
print("=== Extracting Spike Times ===")

# We'll extract spikes from the loose seal recording
# First, let's look at one movie's recording
recording = nwb.acquisition["loose seal recording for movie 0"]
ephys_data = recording.data[:]
ephys_start_time = recording.starting_time
ephys_rate = recording.rate
ephys_times = ephys_start_time + np.arange(len(ephys_data)) / ephys_rate

print(f"Electrophysiology recording:")
print(f"  Duration: {len(ephys_data) / ephys_rate:.2f} s")
print(f"  Sampling rate: {ephys_rate} Hz")
print(f"  Data range: [{np.min(ephys_data):.2f}, {np.max(ephys_data):.2f}]")
print()

# Simple spike detection using threshold
# (Note: In real analysis, you'd use more sophisticated methods)
threshold = np.mean(ephys_data) + 5 * np.std(ephys_data)
spike_indices = np.where(np.diff((ephys_data > threshold).astype(int)) > 0)[0]
spike_times = ephys_times[spike_indices]

print(f"Detected {len(spike_times)} spikes using threshold={threshold:.2f}")
print(f"Mean firing rate: {len(spike_times) / (len(ephys_data) / ephys_rate):.2f} Hz")
print()

# %%
# Create visualization of raw data
fig, axes = plt.subplots(3, 1, figsize=(12, 8))

# Panel 1: Trial structure
ax = axes[0]
for i, angle in enumerate(angles[:40]):  # First 40 trials
    color = plt.cm.hsv(angle / 360)
    ax.barh(i, stop_times[i] - start_times[i], left=start_times[i], 
            height=0.8, color=color, alpha=0.7)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Trial')
ax.set_title('Trial Structure (colored by orientation)')
ax.set_xlim(start_times[0], stop_times[39])

# Panel 2: Example electrophysiology trace
ax = axes[1]
plot_duration = 1.0  # seconds
plot_samples = int(plot_duration * ephys_rate)
ax.plot(ephys_times[:plot_samples], ephys_data[:plot_samples], 'k-', linewidth=0.5)
ax.axhline(threshold, color='r', linestyle='--', label='Spike threshold')
spike_mask = spike_times < (ephys_start_time + plot_duration)
ax.plot(spike_times[spike_mask], ephys_data[spike_indices[spike_mask]], 'r.', 
        markersize=10, label='Detected spikes')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Voltage (arbitrary units)')
ax.set_title('Example Electrophysiology Recording (first 1s)')
ax.legend()

# Panel 3: Orientation distribution
ax = axes[2]
unique_angles, counts = np.unique(angles, return_counts=True)
colors = [plt.cm.hsv(a / 360) for a in unique_angles]
ax.bar(unique_angles, counts, color=colors, alpha=0.7, edgecolor='black')
ax.set_xlabel('Orientation (degrees)')
ax.set_ylabel('Number of trials')
ax.set_title('Stimulus Orientation Distribution')
ax.set_xticks(unique_angles)

plt.tight_layout()
plt.savefig('fig_01_data_exploration.png', dpi=150, bbox_inches='tight')
print("Saved: fig_01_data_exploration.png")
plt.close()

# %%
print("\n=== Data Exploration Complete ===")
print(f"This file contains responses to {len(np.unique(angles))} orientations")
print(f"across {len(angles)} trials")
print(f"Ready for orientation selectivity analysis!")
