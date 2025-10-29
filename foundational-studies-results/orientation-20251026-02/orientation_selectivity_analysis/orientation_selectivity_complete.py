# %% [markdown]
# # Orientation Selectivity in Mouse Visual Cortex
# 
# **Analysis of real experimental data from DANDI Archive**
# 
# This analysis demonstrates orientation selectivity - a fundamental property of neurons in primary 
# visual cortex (V1) where individual neurons respond preferentially to visual stimuli oriented at 
# specific angles. This property is crucial for edge detection and feature extraction in the visual system.
# 
# ## Dataset
# - **DANDI Archive**: [DANDI:000168](https://neurosift.app/dandiset/000168)
# - **Title**: Simultaneous loose seal cell-attached recordings and two-photon imaging of GCaMP8 expressing 
#   mouse V1 neurons with drifting gratings visual stimuli
# - **Authors**: Rozsa, Marton; Liang, Yajie; Zhang, Yan; Hasseman, Jeremy; Kolb, Ilya; Looger, Loren; Svoboda, Karel
# - **Recording**: Two-photon calcium imaging (122 Hz) of L2/3 pyramidal neurons
# - **Stimulus**: Full-field drifting gratings presented at 8 different orientations (45°, 90°, 135°, 180°, 225°, 270°, 315°, 360°)

# %% [markdown]
# ## 1. Setup and Data Loading

# %%
import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

print("Libraries loaded successfully")

# %%
# Load NWB file using LINDI with streaming access
lindi_url = "https://lindi.neurosift.org/dandi/dandisets/000168/assets/87512451-9f7c-41d7-a83f-d0f2bb29dcc9/nwb.lindi.json"

print("Loading NWB file from DANDI Archive via LINDI...")
print("This uses streaming access with local caching for efficiency")
f = lindi.LindiH5pyFile.from_lindi_file(lindi_url)
io = pynwb.NWBHDF5IO(file=f, mode='r')
nwbfile = io.read()

print(f"\nSession: {nwbfile.identifier}")
print(f"Start time: {nwbfile.session_start_time}")
print(f"Subject: Mouse {nwbfile.subject.subject_id}")

# %% [markdown]
# ## 2. Data Exploration

# %%
# Extract trial information
trials = nwbfile.intervals['trials']
print(f"Number of trials: {len(trials)}")
print(f"Trial metadata: {trials.colnames}")

# Get unique orientations
unique_angles = np.unique(trials['angle'][:])
print(f"\nStimulus orientations: {unique_angles}°")
print(f"Trials per orientation: {len(trials) // len(unique_angles)}")

# Display trial structure
print("\nExample trials:")
for i in range(5):
    print(f"  Trial {i}: {trials['angle'][i]}° at {trials['start_time'][i]:.1f}s")

# %%
# Extract fluorescence data
fluorescence = nwbfile.processing['ophys of movie 0']['Fluorescence']
roi_response = fluorescence['RoiResponseSeries']

timestamps = roi_response.timestamps[:]
fluorescence_data = roi_response.data[0, :]  # ROI 0 is the recorded neuron
sampling_rate = 1 / np.median(np.diff(timestamps[:1000]))

print(f"\nFluorescence data:")
print(f"  Number of ROIs: {roi_response.data.shape[0]}")
print(f"  Recording duration: {timestamps[-1]:.1f} seconds")
print(f"  Sampling rate: {sampling_rate:.1f} Hz")
print(f"  Total timepoints: {len(timestamps)}")

# %% [markdown]
# ## 3. Preprocessing

# %%
# Compute ΔF/F (normalized fluorescence)
print("Computing ΔF/F normalization...")

# Baseline: 10th percentile of fluorescence
baseline_f = np.percentile(fluorescence_data, 10)
df_f = (fluorescence_data - baseline_f) / baseline_f

# Apply mild Gaussian smoothing (sigma=2 frames at 122 Hz ≈ 16 ms)
df_f_smooth = gaussian_filter1d(df_f, sigma=2)

print(f"Baseline fluorescence: {baseline_f:.1f}")
print(f"ΔF/F range: [{np.min(df_f_smooth):.3f}, {np.max(df_f_smooth):.3f}]")

# %%
# Visualize raw fluorescence trace
fig, ax = plt.subplots(1, 1, figsize=(14, 4))

# Plot first 200 seconds
time_limit = 200
time_idx = np.searchsorted(timestamps, time_limit)
times = timestamps[:time_idx]
trace = df_f_smooth[:time_idx]

ax.plot(times, trace, 'k-', linewidth=0.5)

# Overlay trial periods colored by orientation
colors = plt.cm.hsv(np.linspace(0, 1, len(unique_angles) + 1)[:-1])
angle_to_color = dict(zip(unique_angles, colors))

for i in range(len(trials)):
    start = trials['start_time'][i]
    stop = trials['stop_time'][i]
    angle = trials['angle'][i]
    if start < time_limit:
        ax.axvspan(start, stop, alpha=0.3, color=angle_to_color[angle])

ax.set_xlabel('Time (s)', fontsize=12)
ax.set_ylabel('ΔF/F', fontsize=12)
ax.set_title('Fluorescence Trace with Trial Structure (colored by orientation)', fontsize=13)
ax.grid(alpha=0.3)
ax.set_xlim(0, time_limit)

# Legend
handles = [Rectangle((0,0),1,1, fc=angle_to_color[angle], alpha=0.3) 
           for angle in sorted(unique_angles)]
labels = [f"{angle}°" for angle in sorted(unique_angles)]
ax.legend(handles, labels, loc='upper right', ncol=4, title='Orientation', fontsize=9)

plt.tight_layout()
plt.savefig('01_fluorescence_trace.png', dpi=150, bbox_inches='tight')
print("Saved: 01_fluorescence_trace.png")
plt.show()

# %% [markdown]
# ## 4. Trial-Aligned Analysis

# %%
# Extract trial-aligned responses
print("\nExtracting trial-aligned responses...")

# Time window: 1s before to 3s after stimulus onset
pre_time = 1.0
post_time = 3.0
time_window = np.arange(-pre_time, post_time, 1/sampling_rate)

# Store responses organized by orientation
trial_responses = {angle: [] for angle in unique_angles}

for i in range(len(trials)):
    angle = trials['angle'][i]
    start_time = trials['start_time'][i]
    
    # Find time indices
    start_idx = np.searchsorted(timestamps, start_time - pre_time)
    end_idx = np.searchsorted(timestamps, start_time + post_time)
    
    # Extract response
    response = df_f_smooth[start_idx:end_idx]
    
    # Ensure consistent length
    if len(response) > len(time_window):
        response = response[:len(time_window)]
    elif len(response) < len(time_window):
        response = np.pad(response, (0, len(time_window) - len(response)), 
                         constant_values=np.nan)
    
    trial_responses[angle].append(response)

# Convert to arrays
for angle in unique_angles:
    trial_responses[angle] = np.array(trial_responses[angle])

print(f"Extracted {len(trials)} trial-aligned responses")
print(f"Time window: {pre_time}s before to {post_time}s after stimulus onset")

# %% [markdown]
# ## 5. Raster Plots

# %%
# Create raster plots showing individual trial responses
fig, axes = plt.subplots(4, 2, figsize=(14, 12))
axes = axes.flatten()

for idx, angle in enumerate(sorted(unique_angles)):
    ax = axes[idx]
    responses = trial_responses[angle]
    
    # Plot each trial with vertical offset
    for trial_idx in range(len(responses)):
        ax.plot(time_window, responses[trial_idx, :] + trial_idx * 0.5, 
               'k-', linewidth=0.8, alpha=0.7)
    
    # Highlight stimulus period
    ax.axvspan(0, 2, alpha=0.2, color='red', label='Stimulus' if idx == 0 else '')
    ax.axvline(0, color='red', linestyle='--', linewidth=1.5)
    
    ax.set_xlabel('Time from stimulus onset (s)')
    ax.set_ylabel('Trial # + ΔF/F offset')
    ax.set_title(f'Orientation: {angle}°', fontweight='bold')
    ax.grid(alpha=0.3)
    if idx == 0:
        ax.legend(loc='upper right')

plt.suptitle('Trial Raster Plots for Each Orientation', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('02_raster_plots.png', dpi=150, bbox_inches='tight')
print("Saved: 02_raster_plots.png")
plt.show()

# %% [markdown]
# ## 6. Trial-Averaged Responses

# %%
# Compute mean and SEM for each orientation
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
    ax.plot(time_window, mean_resp, color=colors[idx], linewidth=2.5, label=f'{angle}°')
    ax.fill_between(time_window, mean_resp - sem_resp, mean_resp + sem_resp,
                    color=colors[idx], alpha=0.3)
    
    # Mark stimulus period
    ax.axvspan(0, 2, alpha=0.15, color='gray')
    ax.axvline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
    
    ax.set_xlabel('Time from stimulus onset (s)')
    ax.set_ylabel('ΔF/F')
    ax.set_title(f'Orientation: {angle}°', fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend(loc='best')

plt.suptitle('Trial-Averaged Responses (Mean ± SEM)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('03_averaged_responses.png', dpi=150, bbox_inches='tight')
print("Saved: 03_averaged_responses.png")
plt.show()

# %% [markdown]
# ## 7. Orientation Tuning Curve

# %%
# Compute tuning curve: mean response during stimulus period for each orientation
stim_start_idx = np.searchsorted(time_window, 0)
stim_end_idx = np.searchsorted(time_window, 2)

tuning_responses = []
tuning_errors = []

for angle in sorted(unique_angles):
    responses = trial_responses[angle]
    # Mean response during stimulus for each trial
    trial_means = np.nanmean(responses[:, stim_start_idx:stim_end_idx], axis=1)
    tuning_responses.append(np.mean(trial_means))
    tuning_errors.append(stats.sem(trial_means))

tuning_responses = np.array(tuning_responses)
tuning_errors = np.array(tuning_errors)
angles_sorted = sorted(unique_angles)

print("\nOrientation tuning:")
for angle, resp in zip(angles_sorted, tuning_responses):
    print(f"  {angle}°: {resp:.3f} ΔF/F")

# %%
# Plot tuning curve
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Linear plot
ax1.errorbar(angles_sorted, tuning_responses, yerr=tuning_errors, 
            marker='o', markersize=10, linewidth=2.5, capsize=6, 
            color='darkblue', elinewidth=2, label='Mean ± SEM')
ax1.set_xlabel('Orientation (degrees)', fontsize=12)
ax1.set_ylabel('Mean ΔF/F during stimulus', fontsize=12)
ax1.set_title('Orientation Tuning Curve', fontsize=13, fontweight='bold')
ax1.grid(alpha=0.3)
ax1.set_xticks(angles_sorted)
ax1.legend(fontsize=11)

# Polar plot
angles_rad = np.radians(angles_sorted)
angles_rad_closed = np.append(angles_rad, angles_rad[0])
tuning_closed = np.append(tuning_responses, tuning_responses[0])

ax2 = plt.subplot(122, projection='polar')
ax2.plot(angles_rad_closed, tuning_closed, 'o-', linewidth=2.5, 
        markersize=10, color='darkblue', markerfacecolor='lightblue', 
        markeredgewidth=2)
ax2.fill(angles_rad_closed, tuning_closed, alpha=0.25, color='darkblue')
ax2.set_theta_zero_location('E')
ax2.set_theta_direction(-1)
ax2.set_title('Polar Tuning Curve', pad=20, fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('04_tuning_curve.png', dpi=150, bbox_inches='tight')
print("Saved: 04_tuning_curve.png")
plt.show()

# %% [markdown]
# ## 8. Selectivity Metrics

# %%
print("\n" + "="*60)
print("ORIENTATION SELECTIVITY METRICS")
print("="*60)

# Preferred orientation
preferred_idx = np.argmax(tuning_responses)
preferred_orientation = angles_sorted[preferred_idx]
max_response = tuning_responses[preferred_idx]

print(f"\nPreferred Orientation: {preferred_orientation}°")
print(f"Maximum Response: {max_response:.3f} ΔF/F")

# Orientation Selectivity Index (OSI)
# OSI = (R_pref - R_ortho) / (R_pref + R_ortho)
ortho_angle = (preferred_orientation + 90) % 360
if ortho_angle == 0:
    ortho_angle = 360
ortho_idx = angles_sorted.index(ortho_angle)
ortho_response = tuning_responses[ortho_idx]

osi = (max_response - ortho_response) / (max_response + ortho_response)
print(f"\nOrientation Selectivity Index (OSI): {osi:.3f}")
print(f"  - Preferred ({preferred_orientation}°): {max_response:.3f}")
print(f"  - Orthogonal ({ortho_angle}°): {ortho_response:.3f}")
print(f"  - Interpretation: {'Strong' if osi > 0.5 else 'Moderate' if osi > 0.3 else 'Weak'} selectivity")

# Direction Selectivity Index (DSI)
# DSI = (R_pref - R_opposite) / (R_pref + R_opposite)
opposite_angle = (preferred_orientation + 180) % 360
if opposite_angle == 0:
    opposite_angle = 360
opposite_idx = angles_sorted.index(opposite_angle)
opposite_response = tuning_responses[opposite_idx]

dsi = (max_response - opposite_response) / (max_response + opposite_response)
print(f"\nDirection Selectivity Index (DSI): {dsi:.3f}")
print(f"  - Preferred ({preferred_orientation}°): {max_response:.3f}")
print(f"  - Opposite ({opposite_angle}°): {opposite_response:.3f}")
print(f"  - Interpretation: {'Yes' if dsi > 0.5 else 'No'}, neuron {'is' if dsi > 0.5 else 'is not'} strongly direction-selective")

# Circular variance (tuning width)
angles_rad_all = np.radians(angles_sorted)
resultant_x = np.sum(tuning_responses * np.cos(2 * angles_rad_all))
resultant_y = np.sum(tuning_responses * np.sin(2 * angles_rad_all))
resultant_length = np.sqrt(resultant_x**2 + resultant_y**2) / np.sum(tuning_responses)
circular_variance = 1 - resultant_length

print(f"\nCircular Variance: {circular_variance:.3f}")
print(f"  - Interpretation: {'Narrow' if circular_variance < 0.3 else 'Moderate' if circular_variance < 0.7 else 'Broad'} tuning")

print("\n" + "="*60)

# %% [markdown]
# ## 9. Summary Visualization

# %%
# Create comprehensive summary figure
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)

# 1. Responses to preferred orientation
ax1 = fig.add_subplot(gs[0, :2])
pref_responses = trial_responses[preferred_orientation]
for trial_idx in range(len(pref_responses)):
    ax1.plot(time_window, pref_responses[trial_idx, :], 
            alpha=0.4, linewidth=1, color='gray')
mean_pref = np.nanmean(pref_responses, axis=0)
ax1.plot(time_window, mean_pref, linewidth=3.5, color='darkblue', 
        label='Mean response', zorder=10)
ax1.axvspan(0, 2, alpha=0.2, color='red', label='Stimulus')
ax1.axvline(0, color='red', linestyle='--', linewidth=2)
ax1.set_xlabel('Time from stimulus onset (s)', fontsize=11)
ax1.set_ylabel('ΔF/F', fontsize=11)
ax1.set_title(f'Responses to Preferred Orientation ({preferred_orientation}°)', 
             fontsize=12, fontweight='bold')
ax1.grid(alpha=0.3)
ax1.legend(fontsize=10)

# 2. Tuning curve
ax2 = fig.add_subplot(gs[0, 2])
ax2.errorbar(angles_sorted, tuning_responses, yerr=tuning_errors, 
            marker='o', markersize=8, linewidth=2, capsize=5, color='darkblue')
half_max = max_response / 2
ax2.axhline(half_max, color='red', linestyle='--', alpha=0.5, 
           label=f'Half-max ({half_max:.3f})')
ax2.set_xlabel('Orientation (°)', fontsize=11)
ax2.set_ylabel('Mean ΔF/F', fontsize=11)
ax2.set_title('Tuning Curve', fontsize=12, fontweight='bold')
ax2.grid(alpha=0.3)
ax2.set_xticks([45, 90, 135, 180, 225, 270, 315, 360])
ax2.set_xticklabels(['45', '90', '135', '180', '225', '270', '315', '0'], 
                     rotation=45, fontsize=9)
ax2.legend(fontsize=9)

# 3. All orientations overlaid
ax3 = fig.add_subplot(gs[1, :])\nfor idx, angle in enumerate(sorted(unique_angles)):
    mean_resp = mean_responses[angle]
    ax3.plot(time_window, mean_resp, linewidth=2.5, label=f'{angle}°', 
            color=colors[idx])
ax3.axvspan(0, 2, alpha=0.15, color='gray')
ax3.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.5)
ax3.axhline(0, color='black', linestyle='-', linewidth=0.5)
ax3.set_xlabel('Time from stimulus onset (s)', fontsize=11)
ax3.set_ylabel('ΔF/F', fontsize=11)
ax3.set_title('Mean Responses to All Orientations', fontsize=12, fontweight='bold')
ax3.grid(alpha=0.3)
ax3.legend(ncol=4, fontsize=9, loc='upper right')

# 4. Polar plot
ax4 = fig.add_subplot(gs[2, 0], projection='polar')
ax4.plot(angles_rad_closed, tuning_closed, 'o-', linewidth=2.5, 
        markersize=10, color='darkblue', markerfacecolor='lightblue',
        markeredgewidth=2)
ax4.fill(angles_rad_closed, tuning_closed, alpha=0.25, color='darkblue')
ax4.set_theta_zero_location('E')
ax4.set_theta_direction(-1)
ax4.set_title('Polar Tuning', pad=20, fontsize=12, fontweight='bold')

# 5. Metrics text
ax5 = fig.add_subplot(gs[2, 1:])
ax5.axis('off')
metrics_text = f\"\"\"
ORIENTATION SELECTIVITY METRICS

Preferred Orientation: {preferred_orientation}°
Maximum Response: {max_response:.3f} ΔF/F

Orientation Selectivity Index (OSI): {osi:.3f}
  • Compares preferred vs. orthogonal orientation
  • Range: 0 (unselective) to 1 (highly selective)
  • This neuron: {'Strong' if osi > 0.5 else 'Moderate' if osi > 0.3 else 'Weak'} selectivity

Direction Selectivity Index (DSI): {dsi:.3f}
  • Compares preferred vs. opposite direction
  • Range: 0 (direction-independent) to 1 (direction-selective)
  • This neuron: {'IS' if dsi > 0.5 else 'is NOT'} strongly direction-selective

Circular Variance: {circular_variance:.3f}
  • Measure of tuning width
  • Range: 0 (narrow) to 1 (broad tuning)
  • This neuron: {'Narrow' if circular_variance < 0.3 else 'Moderate' if circular_variance < 0.7 else 'Broad'} tuning

INTERPRETATION:
This V1 neuron demonstrates {'strong' if osi > 0.5 else 'moderate' if osi > 0.3 else 'weak'} orientation selectivity,
responding most strongly to {preferred_orientation}° oriented gratings. This selective
response to specific orientations is a hallmark of V1 neurons and
is fundamental to visual edge detection and feature processing.
\"\"\"
ax5.text(0.05, 0.5, metrics_text, fontsize=10, family='monospace',
        verticalalignment='center', 
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.4, pad=1))

plt.suptitle(f'Orientation Selectivity in Mouse V1 - {nwbfile.identifier}', 
            fontsize=15, fontweight='bold', y=0.995)

plt.savefig('05_summary_figure.png', dpi=150, bbox_inches='tight')
print("\nSaved: 05_summary_figure.png")
plt.show()

# %% [markdown]
# ## Conclusion
# 
# This analysis successfully demonstrates **orientation selectivity** in a layer 2/3 pyramidal neuron 
# from mouse primary visual cortex using real experimental data from the DANDI Archive.
# 
# ### Key Findings:
# 1. **Preferred Orientation**: The neuron responds most strongly to gratings oriented at a specific angle
# 2. **Selectivity Metrics**: Quantitative measures (OSI, DSI) characterize the strength of orientation tuning
# 3. **Tuning Curve**: Clear variation in response amplitude across different orientations
# 4. **Biological Significance**: Orientation selectivity is crucial for edge detection and visual processing
# 
# ### Methods Highlights:
# - Streaming data access using LINDI (no full download required)
# - ΔF/F normalization for calcium imaging analysis
# - Trial-averaged responses with statistical measures (SEM)
# - Multiple visualization approaches (raster, tuning curves, polar plots)
# - Standard selectivity metrics from visual neuroscience literature
# 
# **Dataset**: [DANDI:000168](https://neurosift.app/dandiset/000168) - Rozsa et al. (2022)

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
print(f"\nGenerated figures:")
print("  - 01_fluorescence_trace.png")
print("  - 02_raster_plots.png")
print("  - 03_averaged_responses.png")
print("  - 04_tuning_curve.png")
print("  - 05_summary_figure.png")
print(f"\nDataset: DANDI:000168")
print(f"Session: {nwbfile.identifier}")
print(f"\nThis analysis demonstrates orientation selectivity,")
print(f"a fundamental property of visual cortex neurons.")
print("="*60)
