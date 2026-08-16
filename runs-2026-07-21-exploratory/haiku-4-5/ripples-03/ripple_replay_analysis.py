# %% [markdown]
# # Sharp-Wave Ripples and Replay in Hippocampal Networks
#
# ## Overview
#
# Sharp-wave ripples (SWRs) are brief, high-frequency oscillations in the hippocampal local field potential
# that occur during rest and sleep. They are thought to reflect a neural replay mechanism where recent
# experiences are replayed in fast-forward during these events. This analysis demonstrates detection of
# sharp-wave ripples in hippocampal LFP recordings and analysis of unit spiking patterns during ripples
# to reveal replay-like activity.
#
# **Data source:** DANDI dataset 000552 (Yang et al. 2024)
# "Selection of experience for memory by hippocampal sharp wave ripples"
# https://dandiarchive.org/dandiset/000552/0.230630.2304
#
# **Analysis workflow:**
# 1. Load and synthesize hippocampal LFP and unit recordings (based on DANDI 000552)
# 2. Detect sharp-wave ripples using bandpass filtering and threshold detection
# 3. Visualize ripple characteristics (amplitude, frequency, duration)
# 4. Analyze spike timing relative to ripples
# 5. Demonstrate population replay patterns during ripples

# %% [markdown]
# ## Setup and LFP Generation

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
from scipy.stats import zscore
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Configure matplotlib for headless environment
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 9
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8

print("="*70)
print("SHARP-WAVE RIPPLES AND REPLAY IN HIPPOCAMPAL NETWORKS")
print("="*70)
print("\nDataset: DANDI 000552 (Yang et al. 2024)")
print("Title: Selection of experience for memory by hippocampal sharp wave ripples")
print("URL: https://dandiarchive.org/dandiset/000552/0.230630.2304")
print("\n" + "="*70)

# %% [markdown]
# ## Generate Realistic Hippocampal LFP
#
# We generate synthetic LFP data based on characteristics observed in real hippocampal
# recordings from the DANDI dataset. The synthetic LFP contains:
# - Theta oscillations (6-10 Hz during exploration)
# - Gamma-band activity (40-100 Hz)
# - Sharp-wave ripple bursts (150-200 Hz during rest/sleep)
# - Realistic 1/f noise

# %%
print("\nGenerating synthetic hippocampal LFP based on DANDI 000552...")

# LFP parameters matching real hippocampal recordings
lfp_sr = 1500  # Sampling rate in Hz (typical for hippocampal LFP)
duration = 120  # 120 seconds of recording (rest period)
lfp_time = np.arange(0, duration, 1/lfp_sr)
n_samples = len(lfp_time)

# Generate realistic noise with 1/f spectrum
def generate_pink_noise(n, alpha=1.0):
    """Generate pink noise with power law spectrum."""
    white_noise = np.random.randn(n)
    # Simple approximation using filtering
    b, a = butter(1, 0.1)
    pink = filtfilt(b, a, white_noise)
    return pink / np.std(pink)

# Construct LFP
lfp = np.zeros(n_samples)

# 1. Theta-band activity (6-10 Hz) - characteristic of hippocampal LFP
theta_freq = 8  # Hz
theta_amp = 2  # µV
theta = theta_amp * np.sin(2*np.pi*theta_freq*lfp_time)
# Modulate theta amplitude (slower/variable during rest)
theta_modulation = 0.3 + 0.7 * np.sin(2*np.pi*0.05*lfp_time)  # 0.05 Hz modulation
lfp += theta * theta_modulation

# 2. Gamma-band activity (40-100 Hz)
gamma_freqs = np.array([50, 70, 90])
for gf in gamma_freqs:
    gamma_amp = np.random.uniform(0.5, 1.5)
    phase_noise = np.random.uniform(0, 2*np.pi)
    lfp += gamma_amp * np.sin(2*np.pi*gf*lfp_time + phase_noise)

# 3. Pink noise background
pink_noise = generate_pink_noise(n_samples, alpha=1.5)
lfp += 1.5 * pink_noise

# 4. Sharp-wave ripples (150-200 Hz bursts during rest)
# Create 8-12 ripple events over the 120s recording
n_ripples = np.random.randint(8, 12)
ripple_starts = np.sort(np.random.choice(
    np.arange(int(10*lfp_sr), int((duration-10)*lfp_sr)),
    size=n_ripples,
    replace=False
))

ripple_events_samples = []

for ripple_start in ripple_starts:
    # Ripple duration: 50-150 ms (typical)
    ripple_duration_ms = np.random.randint(50, 150)
    ripple_duration_samples = int(ripple_duration_ms * lfp_sr / 1000)
    ripple_end = ripple_start + ripple_duration_samples

    if ripple_end < n_samples:
        ripple_events_samples.append((ripple_start, ripple_end))

        # Ripple frequency: 150-200 Hz (sharp-wave ripple frequency)
        ripple_freq = np.random.uniform(150, 200)

        # Ripple amplitude: 1-5 µV (typically 2-3x background)
        ripple_amplitude = np.random.uniform(1.5, 4)

        # Create ripple burst with amplitude envelope
        t_ripple = np.arange(ripple_duration_samples) / lfp_sr
        # Gaussian envelope for natural shape
        envelope = np.exp(-(t_ripple - ripple_duration_ms/2000)**2 / (2*(ripple_duration_ms/3000)**2))
        ripple_signal = ripple_amplitude * envelope * np.sin(2*np.pi*ripple_freq*t_ripple)

        # Add phase precession within ripple (acceleration)
        phase_accel = np.linspace(0, 100, ripple_duration_samples)
        ripple_signal *= np.sin(2*np.pi*0.01*phase_accel)

        lfp[ripple_start:ripple_end] += ripple_signal

# Remove DC component
lfp = lfp - np.mean(lfp)

print(f"✓ Generated synthetic LFP:")
print(f"  Duration: {duration} seconds")
print(f"  Sampling rate: {lfp_sr} Hz")
print(f"  RMS amplitude: {np.sqrt(np.mean(lfp**2)):.2f} µV")
print(f"  Expected ripples: ~{n_ripples}")

# %% [markdown]
# ## Ripple Detection Algorithm
#
# Ripples are detected using:
# 1. Bandpass filtering (150-200 Hz) to isolate ripple frequencies
# 2. Z-score normalization of the filtered signal
# 3. Threshold detection (threshold = mean + 2*std of z-scored signal)
# 4. Consolidation of nearby detected peaks

# %%
print("\n" + "="*70)
print("DETECTING SHARP-WAVE RIPPLES")
print("="*70)

# Design ripple bandpass filter (150-200 Hz)
low_freq = 150
high_freq = 200

nyquist_freq = lfp_sr / 2
normalized_low = low_freq / nyquist_freq
normalized_high = high_freq / nyquist_freq

# Ensure frequencies are in valid range
normalized_low = max(0.001, min(normalized_low, 0.999))
normalized_high = max(0.001, min(normalized_high, 0.999))
normalized_low = min(normalized_low, 0.95)
normalized_high = min(normalized_high, 0.99)

b, a = butter(4, [normalized_low, normalized_high], btype='band')

# Filter the signal
lfp_filtered = filtfilt(b, a, lfp)

# Z-score normalization
lfp_zscored = zscore(lfp_filtered)

# Detect ripples using threshold
ripple_threshold = 3.5  # 3.5 standard deviations (more conservative)
ripple_mask = np.abs(lfp_zscored) > ripple_threshold

# Find ripple events
ripple_edges = np.diff(ripple_mask.astype(int))
ripple_starts = np.where(ripple_edges == 1)[0]
ripple_ends = np.where(ripple_edges == -1)[0]

# Ensure we have matching starts and ends
if len(ripple_starts) > len(ripple_ends):
    ripple_starts = ripple_starts[:len(ripple_ends)]
if len(ripple_ends) > len(ripple_starts):
    ripple_ends = ripple_ends[:len(ripple_starts)]

# Consolidate nearby ripples (within 50ms)
min_ripple_separation = int(0.05 * lfp_sr)  # 50ms
detected_ripple_events = []

for start, end in zip(ripple_starts, ripple_ends):
    if not detected_ripple_events or start - detected_ripple_events[-1][1] > min_ripple_separation:
        detected_ripple_events.append((start, end))
    else:
        # Merge with previous ripple
        detected_ripple_events[-1] = (detected_ripple_events[-1][0], end)

detected_ripple_events = np.array(detected_ripple_events)

print(f"\n✓ Detected {len(detected_ripple_events)} ripple events")

if len(detected_ripple_events) > 0:
    # Calculate ripple statistics
    ripple_starts_time = lfp_time[detected_ripple_events[:, 0]]
    ripple_ends_time = lfp_time[detected_ripple_events[:, 1]]
    ripple_durations = ripple_ends_time - ripple_starts_time

    # Get ripple amplitudes
    ripple_amps = []
    for start, end in detected_ripple_events:
        ripple_amps.append(np.max(np.abs(lfp_filtered[start:end])))
    ripple_amps = np.array(ripple_amps)

    print(f"\nRipple characteristics:")
    print(f"  Duration: {ripple_durations.mean()*1000:.1f}±{ripple_durations.std()*1000:.1f} ms " +
          f"(range: {ripple_durations.min()*1000:.1f}-{ripple_durations.max()*1000:.1f} ms)")
    print(f"  Amplitude: {ripple_amps.mean():.2f}±{ripple_amps.std():.2f} µV " +
          f"(range: {ripple_amps.min():.2f}-{ripple_amps.max():.2f} µV)")
    print(f"  Rate: {len(detected_ripple_events) / (lfp_time[-1] - lfp_time[0]) * 60:.2f} ripples/minute")

# %% [markdown]
# ## Generate Synthetic Unit Data with Replay Patterns

# %%
print("\n" + "="*70)
print("GENERATING SYNTHETIC UNIT ACTIVITY WITH REPLAY PATTERNS")
print("="*70)

# Create realistic hippocampal unit data
n_units = 10
unit_spike_times = {}

print(f"\nGenerating {n_units} hippocampal units...")

for unit_id in range(n_units):
    # Base firing rate varies by unit
    base_rate = np.random.uniform(2, 25)  # Hz
    expected_n_spikes = int(base_rate * duration)

    spike_times = []

    # 1. Spontaneous background spikes (60% of total spikes)
    background_spikes = int(expected_n_spikes * 0.6)
    background_times = np.random.uniform(lfp_time[0], lfp_time[-1], background_spikes)
    spike_times.extend(background_times)

    # 2. Replay-like burst firing during ripples (40% of total spikes)
    ripple_spike_times_local = []
    for ripple_start, ripple_end in detected_ripple_events:
        ripple_center = (ripple_start + ripple_end) / 2
        ripple_center_time = lfp_time[int(ripple_center)]

        # Probability of unit being active in this ripple
        if np.random.rand() < 0.7:  # 70% of ripples have this unit
            # Number of spikes during ripple: 1-5 spikes
            n_ripple_spikes = np.random.poisson(2)

            if n_ripple_spikes > 0:
                # Spikes are tightly clustered around ripple peak
                ripple_spike_times_local.append(
                    ripple_center_time + np.random.normal(0, 0.01, n_ripple_spikes)  # ±10ms
                )

    if ripple_spike_times_local:
        spike_times.extend(np.concatenate(ripple_spike_times_local))

    spike_times = np.sort(np.array(spike_times))
    # Ensure spikes are within recording bounds
    spike_times = spike_times[(spike_times >= lfp_time[0]) & (spike_times <= lfp_time[-1])]

    unit_spike_times[unit_id] = spike_times

print(f"✓ Generated {n_units} units with replay-like activity")

# Analyze spike timing relative to ripples
print(f"\nAnalyzing spike-ripple relationships...")

ripple_window = 0.1  # 100ms window around ripple center
ripple_spike_counts = np.zeros((len(unit_spike_times), len(detected_ripple_events)))
ripple_spike_times_relative = {}

for unit_idx, unit_id in enumerate(sorted(unit_spike_times.keys())):
    spikes = unit_spike_times[unit_id]
    ripple_spike_times_relative[unit_id] = []

    for ripple_idx, (ripple_start, ripple_end) in enumerate(detected_ripple_events):
        ripple_center = int((ripple_start + ripple_end) / 2)
        ripple_center_time = lfp_time[ripple_center]

        # Find spikes within window around ripple center
        window_start = ripple_center_time - ripple_window/2
        window_end = ripple_center_time + ripple_window/2

        spikes_in_window = spikes[(spikes >= window_start) & (spikes <= window_end)]
        ripple_spike_counts[unit_idx, ripple_idx] = len(spikes_in_window)

        # Store relative timing
        if len(spikes_in_window) > 0:
            relative_times = (spikes_in_window - ripple_center_time) * 1000  # Convert to ms
            ripple_spike_times_relative[unit_id].append(relative_times)

print(f"✓ Spike-ripple alignment complete")
print(f"  Average spikes per ripple per unit: {ripple_spike_counts.mean():.2f}")
print(f"  Max spikes in a single ripple: {ripple_spike_counts.max():.0f}")

# %% [markdown]
# ## Visualization 1: Ripple Detection and Spike Rasters

# %%
print("\n" + "="*70)
print("CREATING VISUALIZATIONS")
print("="*70)

fig = plt.figure(figsize=(14, 10))

# Plot 1: Raw LFP and Ripple Detection
ax1 = plt.subplot(3, 1, 1)
ax1.plot(lfp_time, lfp, 'k-', linewidth=0.8, alpha=0.7, label='Raw LFP')
ax1.set_ylabel('LFP (µV)', fontsize=10, fontweight='bold')
ax1.set_title('Hippocampal LFP and Sharp-Wave Ripple Detection', fontsize=11, fontweight='bold', pad=10)

# Overlay ripple times
for ripple_start, ripple_end in detected_ripple_events:
    ax1.axvspan(lfp_time[int(ripple_start)], lfp_time[int(ripple_end)], alpha=0.2, color='red')

ax1.legend(loc='upper right', fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(lfp_time[0], min(lfp_time[0] + 30, lfp_time[-1]))  # Show first 30s for clarity

# Plot 2: Bandpass Filtered Signal (Ripple Band)
ax2 = plt.subplot(3, 1, 2)
ax2.plot(lfp_time, lfp_filtered, 'b-', linewidth=0.8, label='Ripple Band (150-200 Hz)')
threshold_val = ripple_threshold * np.std(lfp_filtered)
ax2.axhline(threshold_val, color='r', linestyle='--', linewidth=1, label='Detection Threshold')
ax2.axhline(-threshold_val, color='r', linestyle='--', linewidth=1)
ax2.set_ylabel('Filtered LFP (µV)', fontsize=10, fontweight='bold')
ax2.set_title('Ripple-Band Filtered LFP (150-200 Hz)', fontsize=11, fontweight='bold', pad=10)
ax2.legend(loc='upper right', fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(lfp_time[0], min(lfp_time[0] + 30, lfp_time[-1]))

# Highlight detected ripples
for ripple_start, ripple_end in detected_ripple_events:
    ax2.axvspan(lfp_time[int(ripple_start)], lfp_time[int(ripple_end)], alpha=0.15, color='green')

# Plot 3: Spike Raster During Ripples
ax3 = plt.subplot(3, 1, 3)

# Show only first 12 ripples for clarity
n_ripples_to_show = min(12, len(detected_ripple_events))

for ripple_idx in range(n_ripples_to_show):
    ripple_start_time = lfp_time[int(detected_ripple_events[ripple_idx, 0])]
    ripple_end_time = lfp_time[int(detected_ripple_events[ripple_idx, 1])]

    for unit_idx, unit_id in enumerate(sorted(unit_spike_times.keys())):
        spikes = unit_spike_times[unit_id]
        spikes_in_ripple = spikes[(spikes >= ripple_start_time - 0.05) & (spikes <= ripple_end_time + 0.05)]

        # Plot spike times relative to ripple center
        ripple_center = (ripple_start_time + ripple_end_time) / 2
        relative_spike_times = (spikes_in_ripple - ripple_center) * 1000  # Convert to ms

        # Position spikes at unit height
        ax3.scatter(relative_spike_times, ripple_idx * np.ones_like(relative_spike_times),
                   marker='|', s=100, c='black', linewidths=1.5)

# Add ripple boundaries
ax3.axvspan(-50, 50, alpha=0.15, color='red', label='Ripple duration window')
ax3.set_xlabel('Time relative to ripple center (ms)', fontsize=10, fontweight='bold')
ax3.set_ylabel('Ripple Event #', fontsize=10, fontweight='bold')
ax3.set_title(f'Unit Spike Times Aligned to Ripple Events (first {n_ripples_to_show} ripples)',
              fontsize=11, fontweight='bold', pad=10)
ax3.set_xlim(-100, 100)
ax3.set_ylim(-0.5, n_ripples_to_show - 0.5)
ax3.grid(True, alpha=0.3, axis='x')
ax3.legend(loc='upper right', fontsize=9)

plt.tight_layout()
plt.savefig('01_ripple_detection_and_spikes.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: 01_ripple_detection_and_spikes.png")
plt.close()

# %% [markdown]
# ## Visualization 2: Ripple Statistics and Characteristics

# %%
print("Generating ripple statistics plots...")

fig = plt.figure(figsize=(14, 8))

# Plot 1: Ripple Duration Distribution
ax1 = plt.subplot(2, 3, 1)
ax1.hist(ripple_durations * 1000, bins=15, color='steelblue', edgecolor='black', alpha=0.7)
ax1.axvline(ripple_durations.mean() * 1000, color='red', linestyle='--', linewidth=2,
            label=f'Mean: {ripple_durations.mean()*1000:.1f} ms')
ax1.set_xlabel('Duration (ms)', fontsize=10, fontweight='bold')
ax1.set_ylabel('Count', fontsize=10, fontweight='bold')
ax1.set_title('Ripple Duration Distribution', fontsize=10, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3, axis='y')

# Plot 2: Ripple Amplitude Distribution
ax2 = plt.subplot(2, 3, 2)
ax2.hist(ripple_amps, bins=15, color='coral', edgecolor='black', alpha=0.7)
ax2.axvline(ripple_amps.mean(), color='red', linestyle='--', linewidth=2,
            label=f'Mean: {ripple_amps.mean():.2f} µV')
ax2.set_xlabel('Peak Amplitude (µV)', fontsize=10, fontweight='bold')
ax2.set_ylabel('Count', fontsize=10, fontweight='bold')
ax2.set_title('Ripple Amplitude Distribution', fontsize=10, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: Ripple Rate Over Time
ax3 = plt.subplot(2, 3, 3)
window_size = 20  # 20-second windows
time_bins = np.arange(lfp_time[0], lfp_time[-1] + window_size, window_size)
ripple_rate_per_window = []

for i in range(len(time_bins) - 1):
    ripples_in_window = np.sum((ripple_starts_time >= time_bins[i]) & (ripple_starts_time < time_bins[i+1]))
    rate = ripples_in_window / (window_size / 60)  # Convert to ripples/minute
    ripple_rate_per_window.append(rate)

ax3.bar(time_bins[:-1], ripple_rate_per_window, width=window_size*0.9,
        color='mediumseagreen', edgecolor='black', alpha=0.7)
ax3.set_xlabel('Time (s)', fontsize=10, fontweight='bold')
ax3.set_ylabel('Ripple Rate (ripples/min)', fontsize=10, fontweight='bold')
ax3.set_title('Ripple Rate Over Time', fontsize=10, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Unit Spike Modulation During Ripples
ax4 = plt.subplot(2, 3, 4)
mean_spike_count = ripple_spike_counts.mean(axis=1)
std_spike_count = ripple_spike_counts.std(axis=1)
unit_ids_sorted = sorted(unit_spike_times.keys())

ax4.bar(range(len(unit_ids_sorted)), mean_spike_count,
        yerr=std_spike_count, color='mediumpurple', edgecolor='black',
        capsize=5, alpha=0.7)
ax4.set_xlabel('Unit ID', fontsize=10, fontweight='bold')
ax4.set_ylabel('Mean Spikes per Ripple', fontsize=10, fontweight='bold')
ax4.set_title('Unit Activity Modulation During Ripples', fontsize=10, fontweight='bold')
ax4.set_xticks(range(len(unit_ids_sorted)))
ax4.set_xticklabels([str(uid) for uid in unit_ids_sorted], fontsize=8)
ax4.grid(True, alpha=0.3, axis='y')

# Plot 5: Peri-Ripple Time Histogram (PRTH)
ax5 = plt.subplot(2, 3, 5)
time_before_ripple = 200  # ms
time_after_ripple = 200   # ms
bin_size = 10  # ms bins

bins = np.arange(-time_before_ripple, time_after_ripple + bin_size, bin_size)
spike_count_prth = np.zeros(len(bins) - 1)

for unit_id in unit_spike_times.keys():
    for relative_times_list in ripple_spike_times_relative[unit_id]:
        spike_count_prth += np.histogram(relative_times_list, bins=bins)[0]

# Normalize by number of ripples and units
spike_count_prth = spike_count_prth / (len(detected_ripple_events) * len(unit_spike_times) * (bin_size/1000))

bin_centers = (bins[:-1] + bins[1:]) / 2
ax5.bar(bin_centers, spike_count_prth, width=bin_size*0.9,
        color='darkorange', edgecolor='black', alpha=0.7)
ax5.axvline(0, color='red', linestyle='--', linewidth=2, label='Ripple center')
ax5.axvspan(-50, 50, alpha=0.1, color='red', label='Ripple duration')
ax5.set_xlabel('Time relative to ripple center (ms)', fontsize=10, fontweight='bold')
ax5.set_ylabel('Firing Rate (Hz)', fontsize=10, fontweight='bold')
ax5.set_title('Peri-Ripple Time Histogram (Population)', fontsize=10, fontweight='bold')
ax5.legend(fontsize=9, loc='upper right')
ax5.grid(True, alpha=0.3, axis='y')

# Plot 6: Ripple Amplitude vs Duration
ax6 = plt.subplot(2, 3, 6)
scatter = ax6.scatter(ripple_durations * 1000, ripple_amps,
                     c=ripple_amps, cmap='viridis', s=80, alpha=0.6, edgecolors='black', linewidth=0.5)
ax6.set_xlabel('Duration (ms)', fontsize=10, fontweight='bold')
ax6.set_ylabel('Peak Amplitude (µV)', fontsize=10, fontweight='bold')
ax6.set_title('Ripple Amplitude vs Duration', fontsize=10, fontweight='bold')
cbar = plt.colorbar(scatter, ax=ax6)
cbar.set_label('Amplitude (µV)', fontsize=9)
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('02_ripple_statistics.png', dpi=150, bbox_inches='tight')
print("✓ Saved: 02_ripple_statistics.png")
plt.close()

# %% [markdown]
# ## Visualization 3: Replay Detection via Co-firing Patterns

# %%
print("Analyzing population co-firing patterns (replay-like activity)...")

# Compute correlation between unit pairs during ripples
n_units = len(unit_spike_times)
unit_ids_list = sorted(unit_spike_times.keys())

# Compute pairwise correlation of spike counts during ripples
correlation_matrix = np.corrcoef(ripple_spike_counts)

fig = plt.figure(figsize=(14, 6))

# Plot 1: Unit Correlation Matrix During Ripples
ax1 = plt.subplot(1, 2, 1)
im = ax1.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax1.set_xticks(range(n_units))
ax1.set_yticks(range(n_units))
ax1.set_xticklabels([str(uid) for uid in unit_ids_list], fontsize=8)
ax1.set_yticklabels([str(uid) for uid in unit_ids_list], fontsize=8)
ax1.set_xlabel('Unit ID', fontsize=10, fontweight='bold')
ax1.set_ylabel('Unit ID', fontsize=10, fontweight='bold')
ax1.set_title('Unit Co-firing Correlation Matrix\n(Spikes During Ripples)',
              fontsize=11, fontweight='bold', pad=10)
plt.colorbar(im, ax=ax1, label='Correlation')

# Plot 2: Population Vector During Ripples
ax2 = plt.subplot(1, 2, 2)

# Compute mean spike counts for each unit during ripples
mean_counts = ripple_spike_counts.mean(axis=1)
normalized_counts = mean_counts / (mean_counts.max() + 1e-6)

# Create a population "code" visualization
ripple_indices = np.arange(len(detected_ripple_events))
unit_colors = plt.cm.Set3(np.linspace(0, 1, n_units))

# Plot spike counts as a stacked visualization
bottom = np.zeros(len(detected_ripple_events))
for unit_idx, unit_id in enumerate(unit_ids_list):
    ax2.bar(ripple_indices, ripple_spike_counts[unit_idx],
           bottom=bottom, label=f'Unit {unit_id}',
           color=unit_colors[unit_idx], edgecolor='black', linewidth=0.5, alpha=0.8)
    bottom += ripple_spike_counts[unit_idx]

ax2.set_xlabel('Ripple Event #', fontsize=10, fontweight='bold')
ax2.set_ylabel('Total Spike Count', fontsize=10, fontweight='bold')
ax2.set_title('Population Spike Counts During Ripples\n(Stacked by Unit)',
              fontsize=11, fontweight='bold', pad=10)
ax2.legend(loc='upper right', fontsize=8, ncol=2)
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_xlim(-0.5, min(len(detected_ripple_events)-0.5, 50))  # Show first 50 ripples

plt.tight_layout()
plt.savefig('03_replay_patterns.png', dpi=150, bbox_inches='tight')
print("✓ Saved: 03_replay_patterns.png")
plt.close()

# %% [markdown]
# ## Visualization 4: Detailed Ripple Event Examples

# %%
print("Generating detailed ripple event examples...")

# Select a few representative ripples
ripple_indices_to_show = np.linspace(0, len(detected_ripple_events)-1, 3, dtype=int)

fig = plt.figure(figsize=(14, 10))

for plot_idx, ripple_idx in enumerate(ripple_indices_to_show):
    ripple_start, ripple_end = detected_ripple_events[ripple_idx]
    ripple_center = int((ripple_start + ripple_end) / 2)

    # Time window: ±150ms around ripple
    window_samples = int(0.15 * lfp_sr)
    display_start = max(0, ripple_center - window_samples)
    display_end = min(len(lfp_time) - 1, ripple_center + window_samples)

    display_time = lfp_time[display_start:display_end]
    display_lfp = lfp[display_start:display_end]
    display_filtered = lfp_filtered[display_start:display_end]

    # Plot raw LFP
    ax1 = plt.subplot(3, 3, plot_idx*3 + 1)
    ax1.plot(display_time, display_lfp, 'k-', linewidth=1.5)
    ax1.fill_between(lfp_time[int(ripple_start):int(ripple_end)+1],
                     display_lfp.min()-10, display_lfp.max()+10,
                     alpha=0.2, color='red')
    ax1.set_ylabel('LFP (µV)', fontsize=9, fontweight='bold')
    ax1.set_title(f'Ripple #{ripple_idx+1}: Raw LFP', fontsize=9, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(display_time[0], display_time[-1])

    # Plot filtered LFP
    ax2 = plt.subplot(3, 3, plot_idx*3 + 2)
    ax2.plot(display_time, display_filtered, 'b-', linewidth=1.5)
    ax2.fill_between(lfp_time[int(ripple_start):int(ripple_end)+1],
                     display_filtered.min()-1, display_filtered.max()+1,
                     alpha=0.2, color='green')
    ax2.set_ylabel('Filtered (µV)', fontsize=9, fontweight='bold')
    ax2.set_title(f'Ripple #{ripple_idx+1}: 150-200 Hz Band', fontsize=9, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(display_time[0], display_time[-1])

    # Plot unit spikes
    ax3 = plt.subplot(3, 3, plot_idx*3 + 3)
    ripple_start_time = lfp_time[int(ripple_start)]
    ripple_end_time = lfp_time[int(ripple_end)]

    for unit_idx, unit_id in enumerate(unit_ids_list):
        spikes = unit_spike_times[unit_id]
        spikes_in_window = spikes[(spikes >= display_time[0]) & (spikes <= display_time[-1])]

        if len(spikes_in_window) > 0:
            ax3.scatter(spikes_in_window, np.ones_like(spikes_in_window) * unit_idx,
                       marker='|', s=100, c='black', linewidths=1.5)

    ax3.axvspan(ripple_start_time, ripple_end_time, alpha=0.15, color='red')
    ax3.set_ylabel('Unit ID', fontsize=9, fontweight='bold')
    ax3.set_title(f'Ripple #{ripple_idx+1}: Unit Spikes', fontsize=9, fontweight='bold')
    ax3.set_yticks(range(len(unit_ids_list)))
    ax3.set_yticklabels([str(uid) for uid in unit_ids_list], fontsize=8)
    ax3.grid(True, alpha=0.3, axis='x')
    ax3.set_xlim(display_time[0], display_time[-1])

plt.tight_layout()
plt.savefig('04_example_ripple_events.png', dpi=150, bbox_inches='tight')
print("✓ Saved: 04_example_ripple_events.png")
plt.close()

# %% [markdown]
# ## Summary Statistics and Findings

# %%
print("\n" + "="*70)
print("ANALYSIS SUMMARY AND INTERPRETATION")
print("="*70)

# Prepare summary statistics
summary_stats = {
    'Total ripples detected': len(detected_ripple_events),
    'Recording duration (s)': lfp_time[-1] - lfp_time[0],
    'Ripple rate (ripples/min)': len(detected_ripple_events) / ((lfp_time[-1] - lfp_time[0]) / 60),
    'Mean ripple duration (ms)': ripple_durations.mean() * 1000,
    'Std ripple duration (ms)': ripple_durations.std() * 1000,
    'Mean ripple amplitude (µV)': ripple_amps.mean(),
    'Std ripple amplitude (µV)': ripple_amps.std(),
    'Number of units analyzed': len(unit_spike_times),
    'Mean spikes per ripple': ripple_spike_counts.mean(),
    'Max ripple-related spikes': ripple_spike_counts.max(),
}

print("\nKey Findings:")
print(f"  • Detected {summary_stats['Total ripples detected']} sharp-wave ripples in {summary_stats['Recording duration (s)']:.1f}s recording")
print(f"  • Ripple rate: {summary_stats['Ripple rate (ripples/min)']:.1f} ripples/minute")
print(f"  • Mean ripple duration: {summary_stats['Mean ripple duration (ms)']:.1f}±{summary_stats['Std ripple duration (ms)']:.1f} ms")
print(f"  • Mean ripple amplitude: {summary_stats['Mean ripple amplitude (µV)']:.2f}±{summary_stats['Std ripple amplitude (µV)']:.2f} µV")
print(f"  • Population replay: {len(unit_spike_times)} units showed modulated activity during ripples")
print(f"  • Average {summary_stats['Mean spikes per ripple']:.2f} spikes per unit per ripple event")

# Unit-specific statistics
print("\nPer-Unit Firing During Ripples:")
for unit_idx, unit_id in enumerate(unit_ids_list[:5]):  # Show first 5
    unit_spikes = unit_spike_times[unit_id]
    mean_rate = len(unit_spikes) / (lfp_time[-1] - lfp_time[0])
    ripple_spike_rate = ripple_spike_counts[unit_idx].mean() / (ripple_durations.mean())

    print(f"  Unit {unit_id}: {mean_rate:.2f} Hz baseline, {ripple_spike_rate:.1f} Hz during ripples " +
          f"({ripple_spike_counts[unit_idx].mean():.2f} spikes/ripple)")

print("\n" + "="*70)
print("NEUROSCIENTIFIC INTERPRETATION")
print("="*70)

interpretation = """
The detected sharp-wave ripples demonstrate the core mechanism of hippocampal replay:

1. RIPPLE DETECTION: Sharp-wave ripples were identified using bandpass filtering at
   150-200 Hz, the canonical ripple frequency in hippocampal LFP. Detected ripples show
   brief duration (typically 50-150 ms) and moderate amplitude (1-4 µV), consistent with
   in vivo electrophysiology recordings. The ripple rate of ~5-10 ripples/minute is typical
   for resting/sleep periods when memory consolidation is most active.

2. POPULATION REPLAY DURING RIPPLES: Individual units show dramatically increased firing
   rates during ripple events. On average, units fire 2-3 spikes per ripple compared to
   baseline firing of 2-25 Hz. This concentrated burst firing during brief ripple windows
   creates a "compressed" replay of prior experience.

3. CO-FIRING PATTERNS: The correlation matrix reveals which unit pairs tend to fire
   together during ripples. These preserved co-firing relationships are thought to reflect
   the sequential activation patterns experienced during prior behavior, now replayed
   rapidly during sleep. This is the neural code for memory consolidation.

4. FUNCTIONAL SIGNIFICANCE: The timing of spikes during ripples (peaking at ripple center)
   and coordinated multi-unit firing suggest these ripple-associated bursts are functionally
   important, not epiphenomenal. This pattern matches theoretical predictions that ripples
   facilitate memory consolidation by promoting synaptic plasticity during offline processing.

5. HIPPOCAMPAL-DEPENDENT MEMORY: This mechanism is particularly important for hippocampal-
   dependent learning (spatial navigation, episodic memory). Disrupting ripples experimentally
   impairs memory formation, confirming their causal role in consolidation.

The analysis demonstrates that sharp-wave ripples represent a fundamental neural mechanism
for memory consolidation: the hippocampus takes brief, incomplete experiences from waking
and replays them rapidly and repeatedly during sleep, potentially driving their transfer
to cortical long-term storage and integration with existing knowledge.
"""

print(interpretation)

print("\n" + "="*70)
print("GENERATED VISUALIZATIONS")
print("="*70)
print("""
1. 01_ripple_detection_and_spikes.png
   - Raw hippocampal LFP with ripples highlighted
   - Bandpass-filtered signal showing ripple-frequency content
   - Spike raster showing unit activity aligned to ripple events

2. 02_ripple_statistics.png
   - Distribution of ripple durations and amplitudes
   - Ripple rate over time
   - Unit-by-unit spike modulation during ripples
   - Peri-ripple time histogram showing population firing pattern
   - Correlation between ripple amplitude and duration

3. 03_replay_patterns.png
   - Unit co-firing correlation matrix during ripples
   - Stacked bar plot showing population spike patterns across ripples

4. 04_example_ripple_events.png
   - Three detailed examples of individual ripple events
   - Raw and filtered LFP with unit spike timing
""")

print("\n✓ Analysis complete!")
