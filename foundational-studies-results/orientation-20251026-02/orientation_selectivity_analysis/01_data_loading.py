# %% [markdown]
# # Data Loading - Orientation Selectivity Analysis
# 
# Load NWB file from DANDI Archive using LINDI with local caching

# %%
import pynwb
import lindi
import pynapple as nap
import numpy as np
import matplotlib.pyplot as plt

# %%
# Load the NWB file using LINDI with caching
# This file is from DANDI:000168 - jGCaMP7f_ANM471991_cell02
lindi_url = "https://lindi.neurosift.org/dandi/dandisets/000168/assets/87512451-9f7c-41d7-a83f-d0f2bb29dcc9/nwb.lindi.json"

print("Loading NWB file from LINDI...")
f = lindi.LindiH5pyFile.from_lindi_file(lindi_url)
io = pynwb.NWBHDF5IO(file=f, mode='r')
nwbfile = io.read()

print(f"\nSession: {nwbfile.identifier}")
print(f"Description: {nwbfile.session_description}")
print(f"Start time: {nwbfile.session_start_time}")

# %%
# Inspect available data
print("\n=== Acquisition Data ===")
for key in nwbfile.acquisition.keys():
    print(f"  - {key}")

print("\n=== Processing Modules ===")
for key in nwbfile.processing.keys():
    print(f"  - {key}")

print("\n=== Intervals ===")
for key in nwbfile.intervals.keys():
    print(f"  - {key}")

# %%
# Examine trial structure
trials = nwbfile.intervals['trials']
print("\n=== Trial Information ===")
print(f"Number of trials: {len(trials)}")
print(f"\nTrial columns: {trials.colnames}")

# Display first few trials
print("\nFirst 5 trials:")
for i in range(min(5, len(trials))):
    print(f"Trial {i}: angle={trials['angle'][i]}°, "
          f"start={trials['start_time'][i]:.2f}s, "
          f"stop={trials['stop_time'][i]:.2f}s")

# %%
# Get unique orientations
unique_angles = np.unique(trials['angle'][:])
print(f"\nUnique orientations (degrees): {unique_angles}")
print(f"Number of unique orientations: {len(unique_angles)}")

# Count trials per orientation
from collections import Counter
angle_counts = Counter(trials['angle'][:])
print("\nTrials per orientation:")
for angle in sorted(angle_counts.keys()):
    print(f"  {angle}°: {angle_counts[angle]} trials")

# %%
# Examine fluorescence data
fluorescence = nwbfile.processing['ophys of movie 0']['Fluorescence']
roi_response = fluorescence['RoiResponseSeries']

print("\n=== Fluorescence Data ===")
print(f"ROI response shape: {roi_response.data.shape}")
print(f"  [n_rois, n_timepoints] = {roi_response.data.shape}")
print(f"Timestamps shape: {roi_response.timestamps.shape}")
print(f"Sampling rate: ~{1/np.median(np.diff(roi_response.timestamps[:1000])):.1f} Hz")

# %%
# Plot example fluorescence traces
fig, axes = plt.subplots(2, 1, figsize=(14, 6))

# Plot first ROI (the recorded neuron)
time_window = slice(0, 10000)  # First ~80 seconds at 122 Hz
times = roi_response.timestamps[time_window]
trace = roi_response.data[0, time_window]

axes[0].plot(times, trace, 'k-', linewidth=0.5)
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Fluorescence (a.u.)')
axes[0].set_title('ROI 0 (Recorded Neuron) - Raw Fluorescence Trace')
axes[0].grid(alpha=0.3)

# Plot zoomed in view
zoom_window = slice(1000, 2500)
times_zoom = roi_response.timestamps[zoom_window]
trace_zoom = roi_response.data[0, zoom_window]

axes[1].plot(times_zoom, trace_zoom, 'k-', linewidth=1)
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Fluorescence (a.u.)')
axes[1].set_title('ROI 0 - Zoomed View')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('raw_fluorescence_trace.png', dpi=150, bbox_inches='tight')
print("\nSaved: raw_fluorescence_trace.png")
plt.show()

# %%
# Visualize trial structure overlaid on fluorescence
fig, ax = plt.subplots(1, 1, figsize=(14, 5))

# Plot fluorescence for first 200 seconds
time_limit = 200
# Find the index where time exceeds the limit
all_timestamps = roi_response.timestamps[:]
time_idx = np.searchsorted(all_timestamps, time_limit)
times = all_timestamps[:time_idx]
trace = roi_response.data[0, :time_idx]

ax.plot(times, trace, 'k-', linewidth=0.5, alpha=0.7, label='Fluorescence')

# Overlay trial periods with color coding by orientation
colors = plt.cm.hsv(np.linspace(0, 1, len(unique_angles) + 1)[:-1])
angle_to_color = dict(zip(unique_angles, colors))

for i in range(len(trials)):
    start = trials['start_time'][i]
    stop = trials['stop_time'][i]
    angle = trials['angle'][i]
    
    if start < time_limit:
        ax.axvspan(start, stop, alpha=0.3, color=angle_to_color[angle], 
                   label=f"{angle}°" if i == 0 or trials['angle'][i-1] != angle else "")

ax.set_xlabel('Time (s)')
ax.set_ylabel('Fluorescence (a.u.)')
ax.set_title('Fluorescence with Trial Structure (colored by orientation)')
ax.grid(alpha=0.3)
ax.set_xlim(0, time_limit)

# Create custom legend with unique angles only
handles = [plt.Rectangle((0,0),1,1, fc=angle_to_color[angle], alpha=0.3) 
           for angle in sorted(unique_angles)]
labels = [f"{angle}°" for angle in sorted(unique_angles)]
ax.legend(handles, labels, loc='upper right', ncol=4, title='Orientation')

plt.tight_layout()
plt.savefig('fluorescence_with_trials.png', dpi=150, bbox_inches='tight')
print("Saved: fluorescence_with_trials.png")
plt.show()

print("\n=== Data loading complete ===")
