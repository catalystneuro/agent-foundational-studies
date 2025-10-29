# %% [markdown]
# # Orientation Selectivity Analysis
# 
# Compute orientation tuning curves and analyze selectivity properties

# %%
import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

# %%
# Load the NWB file using LINDI
lindi_url = "https://lindi.neurosift.org/dandi/dandisets/000168/assets/87512451-9f7c-41d7-a83f-d0f2bb29dcc9/nwb.lindi.json"

print("Loading NWB file from LINDI...")
f = lindi.LindiH5pyFile.from_lindi_file(lindi_url)
io = pynwb.NWBHDF5IO(file=f, mode='r')
nwbfile = io.read()

# Extract data
trials = nwbfile.intervals['trials']
fluorescence = nwbfile.processing['ophys of movie 0']['Fluorescence']
roi_response = fluorescence['RoiResponseSeries']

print(f"Loaded session: {nwbfile.identifier}")

# %%
# Extract fluorescence traces and timestamps
print("\nExtracting fluorescence data...")
timestamps = roi_response.timestamps[:]
fluorescence_data = roi_response.data[0, :]  # Focus on ROI 0 (the recorded neuron)
sampling_rate = 1 / np.median(np.diff(timestamps[:1000]))

print(f"Sampling rate: {sampling_rate:.1f} Hz")
print(f"Recording duration: {timestamps[-1]:.1f} seconds")

# %%
# Compute dF/F (normalized fluorescence)
print("\nComputing dF/F...")
# Use baseline fluorescence (10th percentile)
baseline_f = np.percentile(fluorescence_data, 10)
df_f = (fluorescence_data - baseline_f) / baseline_f

# Apply mild smoothing
df_f_smooth = gaussian_filter1d(df_f, sigma=2)

# %%
# Extract trial information
print("\nOrganizing trials by orientation...")
trial_data = []

for i in range(len(trials)):
    trial_info = {
        'angle': trials['angle'][i],
        'start_time': trials['start_time'][i],
        'stop_time': trials['stop_time'][i],
        'trial_idx': i
    }
    trial_data.append(trial_info)

# Get unique orientations
unique_angles = np.unique([t['angle'] for t in trial_data])
print(f"Unique orientations: {unique_angles}")

# %%
# Extract trial-aligned responses
print("\nExtracting trial-aligned responses...")

# Define time window relative to trial start
pre_time = 1.0   # seconds before trial start
post_time = 3.0  # seconds after trial start
time_window = np.arange(-pre_time, post_time, 1/sampling_rate)

# Store responses for each trial
trial_responses = {angle: [] for angle in unique_angles}
trial_times = []

for trial in tqdm(trial_data):
    angle = trial['angle']
    start_time = trial['start_time']
    
    # Find indices for this trial window
    start_idx = np.searchsorted(timestamps, start_time - pre_time)
    end_idx = np.searchsorted(timestamps, start_time + post_time)
    
    # Extract response
    response = df_f_smooth[start_idx:end_idx]
    
    # Ensure consistent length (sometimes edge effects cause length differences)
    if len(response) > len(time_window):
        response = response[:len(time_window)]
    elif len(response) < len(time_window):
        # Pad with NaN if needed
        response = np.pad(response, (0, len(time_window) - len(response)), 
                         constant_values=np.nan)
    
    trial_responses[angle].append(response)

# Convert to arrays
for angle in unique_angles:
    trial_responses[angle] = np.array(trial_responses[angle])

print(f"Extracted responses for {len(trial_data)} trials")

# %%
# Plot raster plot for each orientation
print("\nCreating raster plots...")

fig, axes = plt.subplots(4, 2, figsize=(14, 12))
axes = axes.flatten()

for idx, angle in enumerate(sorted(unique_angles)):
    ax = axes[idx]
    responses = trial_responses[angle]
    
    # Plot each trial as a row in the raster
    for trial_idx in range(len(responses)):
        ax.plot(time_window, responses[trial_idx, :] + trial_idx * 0.5, 
               'k-', linewidth=0.8, alpha=0.7)
    
    # Highlight stimulus period
    ax.axvspan(0, 2, alpha=0.2, color='red', label='Stimulus')
    ax.axvline(0, color='red', linestyle='--', linewidth=1.5)
    
    ax.set_xlabel('Time from stimulus onset (s)')
    ax.set_ylabel('Trial # + dF/F offset')
    ax.set_title(f'Orientation: {angle}°')
    ax.grid(alpha=0.3)
    if idx == 0:
        ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig('orientation_raster_plots.png', dpi=150, bbox_inches='tight')
print("Saved: orientation_raster_plots.png")
plt.show()

# %%
# Compute mean responses for each orientation
print("\nComputing trial-averaged responses...")

mean_responses = {}
sem_responses = {}

for angle in unique_angles:
    responses = trial_responses[angle]
    mean_responses[angle] = np.nanmean(responses, axis=0)
    sem_responses[angle] = stats.sem(responses, axis=0, nan_policy='omit')

# %%
# Plot trial-averaged responses
fig, axes = plt.subplots(4, 2, figsize=(14, 12))
axes = axes.flatten()

colors = plt.cm.hsv(np.linspace(0, 1, len(unique_angles) + 1)[:-1])

for idx, angle in enumerate(sorted(unique_angles)):
    ax = axes[idx]
    
    mean_resp = mean_responses[angle]
    sem_resp = sem_responses[angle]
    
    # Plot mean ± SEM
    ax.plot(time_window, mean_resp, color=colors[idx], linewidth=2, 
           label=f'{angle}°')
    ax.fill_between(time_window, mean_resp - sem_resp, mean_resp + sem_resp,
                    color=colors[idx], alpha=0.3)
    
    # Highlight stimulus period
    ax.axvspan(0, 2, alpha=0.15, color='gray')
    ax.axvline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
    
    ax.set_xlabel('Time from stimulus onset (s)')
    ax.set_ylabel('dF/F')
    ax.set_title(f'Orientation: {angle}°')
    ax.grid(alpha=0.3)
    ax.legend()

plt.tight_layout()
plt.savefig('orientation_averaged_responses.png', dpi=150, bbox_inches='tight')
print("Saved: orientation_averaged_responses.png")
plt.show()

# %%
# Compute orientation tuning curve
print("\nComputing orientation tuning curve...")

# Define response window during stimulus presentation
stim_start_idx = np.searchsorted(time_window, 0)
stim_end_idx = np.searchsorted(time_window, 2)

# Compute mean response during stimulus for each orientation
tuning_responses = []
tuning_errors = []

for angle in sorted(unique_angles):
    responses = trial_responses[angle]
    # Mean response during stimulus period for each trial
    trial_means = np.nanmean(responses[:, stim_start_idx:stim_end_idx], axis=1)
    tuning_responses.append(np.mean(trial_means))
    tuning_errors.append(stats.sem(trial_means))

tuning_responses = np.array(tuning_responses)
tuning_errors = np.array(tuning_errors)

# %%
# Plot orientation tuning curve
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Linear plot
angles_sorted = sorted(unique_angles)
ax1.errorbar(angles_sorted, tuning_responses, yerr=tuning_errors, 
            marker='o', markersize=8, linewidth=2, capsize=5, 
            color='darkblue', label='Mean response')
ax1.set_xlabel('Orientation (degrees)')
ax1.set_ylabel('Mean dF/F during stimulus')
ax1.set_title('Orientation Tuning Curve')
ax1.grid(alpha=0.3)
ax1.set_xticks(angles_sorted)
ax1.legend()

# Polar plot
# Convert angles to radians (and handle 360° = 0°)
angles_rad = np.radians(angles_sorted)
# Duplicate first point to close the curve
angles_rad_closed = np.append(angles_rad, angles_rad[0])
tuning_closed = np.append(tuning_responses, tuning_responses[0])

ax2 = plt.subplot(122, projection='polar')
ax2.plot(angles_rad_closed, tuning_closed, 'o-', linewidth=2, 
        markersize=8, color='darkblue')
ax2.fill(angles_rad_closed, tuning_closed, alpha=0.25, color='darkblue')
ax2.set_theta_zero_location('E')  # 0° on the right
ax2.set_theta_direction(-1)  # Clockwise
ax2.set_title('Polar Tuning Curve', pad=20)

plt.tight_layout()
plt.savefig('orientation_tuning_curve.png', dpi=150, bbox_inches='tight')
print("Saved: orientation_tuning_curve.png")
plt.show()

# %%
# Compute selectivity metrics
print("\n=== Orientation Selectivity Metrics ===")

# Preferred orientation
preferred_idx = np.argmax(tuning_responses)
preferred_orientation = angles_sorted[preferred_idx]
max_response = tuning_responses[preferred_idx]

print(f"Preferred orientation: {preferred_orientation}°")
print(f"Maximum response: {max_response:.3f} dF/F")

# Orientation selectivity index (OSI)
# OSI = (R_pref - R_ortho) / (R_pref + R_ortho)
# Find orthogonal orientation (90° away)
ortho_angle = (preferred_orientation + 90) % 360
if ortho_angle == 0:
    ortho_angle = 360
ortho_idx = angles_sorted.index(ortho_angle)
ortho_response = tuning_responses[ortho_idx]

osi = (max_response - ortho_response) / (max_response + ortho_response)
print(f"Orthogonal orientation: {ortho_angle}°")
print(f"Orthogonal response: {ortho_response:.3f} dF/F")
print(f"Orientation Selectivity Index (OSI): {osi:.3f}")

# Direction selectivity index (DSI) 
# DSI = (R_pref - R_opposite) / (R_pref + R_opposite)
opposite_angle = (preferred_orientation + 180) % 360
if opposite_angle == 0:
    opposite_angle = 360
opposite_idx = angles_sorted.index(opposite_angle)
opposite_response = tuning_responses[opposite_idx]

dsi = (max_response - opposite_response) / (max_response + opposite_response)
print(f"Opposite orientation: {opposite_angle}°")
print(f"Opposite response: {opposite_response:.3f} dF/F")
print(f"Direction Selectivity Index (DSI): {dsi:.3f}")

# Tuning width (full width at half maximum)
half_max = max_response / 2
# Simple approximation
print(f"\nHalf-maximum response: {half_max:.3f} dF/F")

# Circular variance (measure of tuning width, 0 = narrow, 1 = broad)
angles_rad_all = np.radians(angles_sorted)
resultant_vector_x = np.sum(tuning_responses * np.cos(2 * angles_rad_all))
resultant_vector_y = np.sum(tuning_responses * np.sin(2 * angles_rad_all))
resultant_length = np.sqrt(resultant_vector_x**2 + resultant_vector_y**2) / np.sum(tuning_responses)
circular_variance = 1 - resultant_length

print(f"Circular variance: {circular_variance:.3f} (0=narrow tuning, 1=broad)")

# %%
# Summary visualization
print("\nCreating summary figure...")

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Example trial responses for preferred orientation
ax1 = fig.add_subplot(gs[0, :2])
pref_responses = trial_responses[preferred_orientation]
for trial_idx in range(len(pref_responses)):
    ax1.plot(time_window, pref_responses[trial_idx, :], 
            alpha=0.5, linewidth=1, color='gray')
mean_pref = np.nanmean(pref_responses, axis=0)
ax1.plot(time_window, mean_pref, linewidth=3, color='darkblue', 
        label='Mean response')
ax1.axvspan(0, 2, alpha=0.2, color='red', label='Stimulus')
ax1.axvline(0, color='red', linestyle='--', linewidth=1.5)
ax1.set_xlabel('Time from stimulus onset (s)')
ax1.set_ylabel('dF/F')
ax1.set_title(f'Responses to Preferred Orientation ({preferred_orientation}°)')
ax1.grid(alpha=0.3)
ax1.legend()

# 2. Orientation tuning curve (linear)
ax2 = fig.add_subplot(gs[0, 2])
ax2.errorbar(angles_sorted, tuning_responses, yerr=tuning_errors, 
            marker='o', markersize=8, linewidth=2, capsize=5, color='darkblue')
ax2.axhline(half_max, color='red', linestyle='--', alpha=0.5, label='Half-max')
ax2.set_xlabel('Orientation (°)')
ax2.set_ylabel('Mean dF/F')
ax2.set_title('Tuning Curve')
ax2.grid(alpha=0.3)
ax2.set_xticks([45, 90, 135, 180, 225, 270, 315, 360])
ax2.set_xticklabels(['45', '90', '135', '180', '225', '270', '315', '0/360'], rotation=45)
ax2.legend(fontsize=8)

# 3. All mean responses overlaid
ax3 = fig.add_subplot(gs[1, :])
for idx, angle in enumerate(sorted(unique_angles)):
    mean_resp = mean_responses[angle]
    ax3.plot(time_window, mean_resp, linewidth=2, label=f'{angle}°', 
            color=colors[idx])
ax3.axvspan(0, 2, alpha=0.15, color='gray', label='Stimulus')
ax3.axvline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
ax3.axhline(0, color='black', linestyle='-', linewidth=0.5)
ax3.set_xlabel('Time from stimulus onset (s)')
ax3.set_ylabel('dF/F')
ax3.set_title('Mean Responses to All Orientations')
ax3.grid(alpha=0.3)
ax3.legend(ncol=4, fontsize=9, loc='upper right')

# 4. Polar tuning curve
ax4 = fig.add_subplot(gs[2, 0], projection='polar')
ax4.plot(angles_rad_closed, tuning_closed, 'o-', linewidth=2, 
        markersize=8, color='darkblue')
ax4.fill(angles_rad_closed, tuning_closed, alpha=0.25, color='darkblue')
ax4.set_theta_zero_location('E')
ax4.set_theta_direction(-1)
ax4.set_title('Polar Tuning', pad=20)

# 5. Selectivity metrics
ax5 = fig.add_subplot(gs[2, 1:])
ax5.axis('off')
metrics_text = f"""
ORIENTATION SELECTIVITY METRICS

Preferred Orientation: {preferred_orientation}°
Maximum Response: {max_response:.3f} dF/F

Orientation Selectivity Index (OSI): {osi:.3f}
  - Compares response to preferred vs. orthogonal orientation
  - Range: 0 (unselective) to 1 (highly selective)

Direction Selectivity Index (DSI): {dsi:.3f}
  - Compares response to preferred vs. opposite direction
  - Range: 0 (direction-independent) to 1 (direction-selective)

Circular Variance: {circular_variance:.3f}
  - Measure of tuning width
  - Range: 0 (narrow tuning) to 1 (broad tuning)

Interpretation:
This neuron shows {'strong' if osi > 0.5 else 'moderate' if osi > 0.3 else 'weak'} orientation selectivity,
preferring {preferred_orientation}° oriented gratings. The neuron
{'is' if dsi > 0.5 else 'is not'} strongly direction-selective.
"""
ax5.text(0.1, 0.5, metrics_text, fontsize=11, family='monospace',
        verticalalignment='center', bbox=dict(boxstyle='round', 
        facecolor='wheat', alpha=0.3))

plt.suptitle(f'Orientation Selectivity Analysis - {nwbfile.identifier}', 
            fontsize=14, fontweight='bold')

plt.savefig('orientation_selectivity_summary.png', dpi=150, bbox_inches='tight')
print("Saved: orientation_selectivity_summary.png")
plt.show()

print("\n=== Analysis complete ===")
