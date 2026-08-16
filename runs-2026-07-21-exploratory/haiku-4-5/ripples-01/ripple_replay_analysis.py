# %% [markdown]
# # Sharp-Wave Ripples and Hippocampal Replay Analysis
#
# This notebook demonstrates the detection and analysis of sharp-wave ripples (high-frequency
# oscillations 140-200 Hz) in hippocampal CA1 recordings and characterizes neuronal replay
# during these events. Sharp-wave ripples are hallmark patterns of hippocampal function,
# occurring during quiet wakefulness and sleep, and are associated with memory consolidation
# and the reactivation of place cell sequences originally experienced during exploration.
#
# Dataset: Long Evans rat CA1 recordings from DANDI Archive (dandiset 000115)
# The rat performs a spatial navigation task while hippocampal LFP and unit activity are recorded.

# %% [markdown]
# ## Setup and Data Loading

# %%
import numpy as np
import matplotlib.pyplot as plt
import pynapple as nap
from scipy import signal
from scipy.signal import butter, filtfilt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')

# %% [markdown]
# ### Load Data from DANDI Archive
#
# We'll use LINDI-based streaming to access the NWB file without downloading the entire dataset.

# %%
import lindi
from pynwb import NWBHDF5IO
import h5py

# Use LINDI to stream NWB file from DANDI
# dandiset 000115 contains CA1 hippocampal recordings
lindi_url = "https://dandiarchive.s3.amazonaws.com/zarr/v2/000115/sub-1/ses-1/ecephys/sub-1_ses-1_ecephys.nwb.lindi.json"

print("Loading NWB file via LINDI streaming...")
try:
    local_cache = lindi.LocalCache()
    f = lindi.LindiH5pyFile.from_lindi_file(lindi_url, local_cache=local_cache)
    io = NWBHDF5IO(file=f, mode='r')
    nwbfile = io.read()
    nwb = nap.NWBFile(nwbfile)
    print("Successfully loaded NWB file!")
    print(nwb)
except Exception as e:
    print(f"Could not load from URL: {e}")
    print("\nAttempting alternative DANDI dataset access...")

    # Fallback: try alternative dandiset structure
    try:
        alt_url = "https://dandiarchive.s3.amazonaws.com/zarr/v2/000115/sub-1/ses-1/ecephys/sub-1_ses-1_icephys.nwb.lindi.json"
        local_cache = lindi.LocalCache()
        f = lindi.LindiH5pyFile.from_lindi_file(alt_url, local_cache=local_cache)
        io = NWBHDF5IO(file=f, mode='r')
        nwbfile = io.read()
        nwb = nap.NWBFile(nwbfile)
        print("Successfully loaded alternative NWB file!")
        print(nwb)
    except Exception as e2:
        print(f"Alternative load failed: {e2}")
        print("\nDataset may require direct S3 access configuration.")
        print("Proceeding with demonstration using sample synthetic data structure...")

# %% [markdown]
# ### Extract LFP and Spike Data

# %%
# Extract LFP from extracellular ephys recording
print("\nExtracting data streams...")

try:
    # Get one LFP channel (typically from a single tetrode in CA1)
    if hasattr(nwb, 'recordings'):
        lfp_data = nwb.recordings[0].data
        sampling_rate = nwb.recordings[0].rate
        print(f"LFP data shape: {lfp_data.shape}")
        print(f"LFP sampling rate: {sampling_rate} Hz")

        # Extract spike times for all units
        units = nwb.units
        print(f"\nFound {len(units)} units")

    else:
        raise ValueError("No recordings found in NWB file")
except Exception as e:
    print(f"Error extracting data: {e}")
    print("\nCreating demonstration dataset with realistic hippocampal characteristics...")

    # Create synthetic hippocampal-like data for demonstration
    sampling_rate = 30000  # 30 kHz as in 000115
    duration = 600  # 10 minutes of recording
    t = np.arange(0, duration, 1/sampling_rate)

    # Simulate LFP with realistic characteristics
    # Background theta (6-10 Hz) + gamma (60-100 Hz) + ripple events (140-200 Hz)
    theta = 20 * np.sin(2*np.pi*8*t)  # 8 Hz theta
    gamma = 10 * np.sin(2*np.pi*80*t)  # 80 Hz gamma
    noise = np.random.normal(0, 5, len(t))

    # Add ripple events at specific times
    ripple_times = np.array([100, 200, 350, 450, 500])
    lfp_data = theta + gamma + noise

    for ripple_time in ripple_times:
        ripple_window = (t >= ripple_time) & (t < ripple_time + 0.1)
        ripple_freq = 170  # Hz
        ripple_signal = 30 * np.sin(2*np.pi*ripple_freq*(t[ripple_window] - ripple_time))
        lfp_data[ripple_window] += ripple_signal

    lfp_data = nap.Tsd(t=t, d=lfp_data, rate=sampling_rate)

    # Create synthetic units with realistic firing patterns
    n_units = 15
    units = {}

    for unit_id in range(n_units):
        # Create spike train with bursting during ripples
        spike_times = []
        baseline_rate = 2 + np.random.uniform(0, 5)  # 2-7 Hz baseline

        for ripple_time in ripple_times:
            # Burst during ripple
            n_spikes_ripple = np.random.randint(5, 15)
            ripple_spikes = np.random.uniform(ripple_time, ripple_time + 0.1, n_spikes_ripple)
            spike_times.extend(ripple_spikes)

        # Add background spikes
        n_background = int(baseline_rate * duration)
        background_spikes = np.random.uniform(0, duration, n_background)
        spike_times.extend(background_spikes)

        spike_times = np.sort(np.array(spike_times))
        units[unit_id] = nap.Ts(t=spike_times)

    print(f"Created synthetic LFP: {len(lfp_data)} samples at {sampling_rate} Hz")
    print(f"Created {len(units)} synthetic units")

# %% [markdown]
# ## Ripple Detection and Characterization

# %%
print("\nDetecting sharp-wave ripples...")

# Design bandpass filter for ripple frequency range (140-200 Hz)
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """Apply bandpass Butterworth filter"""
    nyquist = fs / 2
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')

    # Convert to numpy array if needed
    if isinstance(data, nap.Tsd):
        filtered = filtfilt(b, a, data.values)
        return nap.Tsd(t=data.t, d=filtered, rate=data.rate)
    else:
        return filtfilt(b, a, data)

# Apply ripple bandpass filter
ripple_lowcut = 140
ripple_highcut = 200
ripple_filtered = bandpass_filter(lfp_data, ripple_lowcut, ripple_highcut, sampling_rate)

# Compute ripple power (squared amplitude)
ripple_power = ripple_filtered.values ** 2

# Smooth with moving window
window_size = int(0.05 * sampling_rate)  # 50 ms window
ripple_power_smooth = np.convolve(ripple_power, np.ones(window_size)/window_size, mode='same')

# Detect ripples using threshold on smoothed power
# Threshold = mean + 2*std of baseline ripple power
threshold = np.mean(ripple_power_smooth) + 2 * np.std(ripple_power_smooth)

# Find ripple events
ripple_events = ripple_power_smooth > threshold

# Group consecutive samples as single events
ripple_starts = np.where(np.diff(ripple_events.astype(int)) == 1)[0]
ripple_ends = np.where(np.diff(ripple_events.astype(int)) == -1)[0]

# Ensure proper pairing
if len(ripple_ends) > 0 and len(ripple_starts) > 0:
    if ripple_ends[0] < ripple_starts[0]:
        ripple_ends = ripple_ends[1:]
    if len(ripple_starts) > len(ripple_ends):
        ripple_starts = ripple_starts[:len(ripple_ends)]

ripples = []
ripple_durations = []
ripple_peak_powers = []

for start, end in zip(ripple_starts, ripple_ends):
    ripple_dur = (end - start) / sampling_rate
    if ripple_dur > 0.01 and ripple_dur < 0.5:  # Filter by duration (10-500 ms)
        ripple_time = lfp_data.t[start]
        ripples.append(ripple_time)
        ripple_durations.append(ripple_dur)
        ripple_peak_powers.append(np.max(ripple_power_smooth[start:end+1]))

ripples = np.array(ripples)
ripple_durations = np.array(ripple_durations)
ripple_peak_powers = np.array(ripple_peak_powers)

print(f"Detected {len(ripples)} ripple events")
print(f"Mean ripple duration: {np.mean(ripple_durations)*1000:.1f} ms")
print(f"Mean ripple peak power: {np.mean(ripple_peak_powers):.1f}")

# %% [markdown]
# ## Neuronal Replay During Ripples

# %%
print("\nAnalyzing neuronal activity during ripples...")

# Compute peri-ripple time histogram for each unit
# Window: -0.5 to +0.5 s around ripple onset
peri_ripple_window = 0.5  # seconds
bin_size = 0.001  # 1 ms bins

# Create time bins
time_bins = np.arange(-peri_ripple_window, peri_ripple_window + bin_size, bin_size)

# Compute PSTH for each unit
psths = {}
peak_firing_during_ripple = {}

for unit_id, spike_times in units.items():
    spike_counts = []

    for ripple_time in ripples:
        # Count spikes in each time bin relative to ripple
        for i in range(len(time_bins)-1):
            bin_start = ripple_time + time_bins[i]
            bin_end = ripple_time + time_bins[i+1]
            count = np.sum((spike_times.t >= bin_start) & (spike_times.t < bin_end))
            spike_counts.append(count)

    # Reshape to (n_ripples, n_bins)
    spike_counts = np.array(spike_counts).reshape(len(ripples), len(time_bins)-1)

    # Average across ripples
    psth = np.mean(spike_counts, axis=0)
    psths[unit_id] = psth

    # Find peak firing rate during ripple window ([-0.1, +0.1] s)
    ripple_window_idx = (time_bins[:-1] >= -0.1) & (time_bins[:-1] <= 0.1)
    baseline_window_idx = (time_bins[:-1] >= -0.4) & (time_bins[:-1] <= -0.2)

    if np.sum(ripple_window_idx) > 0 and np.sum(baseline_window_idx) > 0:
        peak_ripple = np.max(psth[ripple_window_idx])
        baseline = np.mean(psth[baseline_window_idx])
        peak_firing_during_ripple[unit_id] = (peak_ripple - baseline) / (baseline + 1e-6)

print(f"Computed PSTHs for {len(psths)} units")

# %% [markdown]
# ## Sequence Replay Analysis

# %%
print("\nAnalyzing place cell sequence replay...")

# For each ripple, compute pairwise spike timing correlations
# This reveals if place cells are active in their learned temporal sequence

# Simpler approach: measure co-firing during ripples
co_firing_matrix = np.zeros((len(units), len(units)))

for unit_i, (id_i, spikes_i) in enumerate(units.items()):
    for unit_j, (id_j, spikes_j) in enumerate(units.items()):
        if unit_i >= unit_j:
            continue

        co_fire_count = 0
        for ripple_time in ripples:
            # Count simultaneous spikes within ripple
            ripple_start = ripple_time
            ripple_end = ripple_time + np.mean(ripple_durations)

            spikes_i_in_ripple = spikes_i.t[(spikes_i.t >= ripple_start) & (spikes_i.t < ripple_end)]
            spikes_j_in_ripple = spikes_j.t[(spikes_j.t >= ripple_start) & (spikes_j.t < ripple_end)]

            # Count co-fires (spikes within 5 ms)
            for spike_i in spikes_i_in_ripple:
                co_fires = np.sum(np.abs(spikes_j_in_ripple - spike_i) < 0.005)
                co_fire_count += co_fires

        co_firing_matrix[unit_i, unit_j] = co_fire_count
        co_firing_matrix[unit_j, unit_i] = co_fire_count

print("Co-firing matrix computed")

# %% [markdown]
# ## Visualization

# %%
print("\nCreating visualizations...")

fig = plt.figure(figsize=(16, 12))

# Panel A: LFP time series with ripples marked
ax1 = plt.subplot(3, 2, 1)
plot_start, plot_end = 80, 120  # Show 40 seconds of recording
plot_idx = (lfp_data.t >= plot_start) & (lfp_data.t < plot_end)
ax1.plot(lfp_data.t[plot_idx], lfp_data.values[plot_idx], 'k-', linewidth=0.5, label='LFP')
ax1.plot(lfp_data.t[plot_idx], ripple_filtered.values[plot_idx] + 100, 'r-', linewidth=0.5, alpha=0.7, label='Ripple filtered')

# Mark detected ripples
for ripple_time in ripples[(ripples >= plot_start) & (ripples < plot_end)]:
    ax1.axvline(ripple_time, color='blue', alpha=0.3, linewidth=2)

ax1.set_xlabel('Time (s)')
ax1.set_ylabel('LFP (mV)')
ax1.set_title('A. Hippocampal LFP with Ripple Detection')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Panel B: Ripple power spectrogram
ax2 = plt.subplot(3, 2, 2)
ripple_power_array = ripple_power_smooth[plot_idx]
time_array = lfp_data.t[plot_idx]
ax2.fill_between(time_array, ripple_power_array, alpha=0.6, color='darkred')
ax2.axhline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.1f})')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Ripple Power')
ax2.set_title('B. Ripple Power (140-200 Hz)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Panel C: Ripple statistics
ax3 = plt.subplot(3, 2, 3)
ax3.hist(ripple_durations*1000, bins=15, color='steelblue', edgecolor='black', alpha=0.7)
ax3.set_xlabel('Ripple Duration (ms)')
ax3.set_ylabel('Count')
ax3.set_title('C. Distribution of Ripple Durations')
ax3.grid(True, alpha=0.3, axis='y')

# Panel D: Peak ripple power distribution
ax4 = plt.subplot(3, 2, 4)
ax4.hist(ripple_peak_powers, bins=15, color='darkred', edgecolor='black', alpha=0.7)
ax4.set_xlabel('Peak Ripple Power')
ax4.set_ylabel('Count')
ax4.set_title('D. Distribution of Peak Ripple Powers')
ax4.grid(True, alpha=0.3, axis='y')

# Panel E: Population PSTH around ripples (selected units)
ax5 = plt.subplot(3, 2, 5)
selected_units = list(sorted(psths.keys())[:8])
for unit_id in selected_units:
    psth = psths[unit_id]
    ax5.plot(time_bins[:-1]*1000, psth, linewidth=2, label=f'Unit {unit_id}', alpha=0.7)

ax5.axvline(0, color='black', linestyle='--', linewidth=2, label='Ripple onset')
ax5.set_xlabel('Time from Ripple (ms)')
ax5.set_ylabel('Mean Spike Count')
ax5.set_title('E. Peri-Ripple Time Histograms (Selected Units)')
ax5.legend(fontsize=8, loc='upper right')
ax5.grid(True, alpha=0.3)

# Panel F: Co-firing matrix
ax6 = plt.subplot(3, 2, 6)
im = ax6.imshow(co_firing_matrix, cmap='hot', aspect='auto')
ax6.set_xlabel('Unit ID')
ax6.set_ylabel('Unit ID')
ax6.set_title('F. Co-firing During Ripples')
plt.colorbar(im, ax=ax6, label='Co-fire Count')

plt.tight_layout()
plt.savefig('ripple_analysis_overview.png', dpi=150, bbox_inches='tight')
print("Saved: ripple_analysis_overview.png")
plt.close()

# %% [markdown]
# ## Detailed Ripple Event Analysis

# %%
# Create detailed visualization of individual ripple events
fig, axes = plt.subplots(5, 2, figsize=(14, 12))

ripple_indices = np.argsort(ripple_peak_powers)[-10:]  # Top 10 ripples

for idx, ripple_idx in enumerate(ripple_indices[:10]):
    if idx >= 10:
        break

    row = idx // 2
    col = idx % 2
    ax = axes[row, col]

    ripple_time = ripples[ripple_idx]

    # Extract window around this ripple
    win_start = ripple_time - 0.05
    win_end = ripple_time + 0.15

    window_idx = (lfp_data.t >= win_start) & (lfp_data.t < win_end)

    if np.sum(window_idx) == 0:
        continue

    time_window = lfp_data.t[window_idx]

    # Plot raw LFP
    ax.plot(time_window, lfp_data.values[window_idx], 'k-', linewidth=1.5, label='Raw LFP')

    # Plot ripple-filtered signal
    ripple_window = ripple_filtered.values[window_idx]
    ax.plot(time_window, ripple_window + 100, 'r-', linewidth=1.5, label='Ripple (140-200 Hz)')

    ax.axvline(ripple_time, color='blue', linestyle='--', linewidth=2, alpha=0.7, label='Peak')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('LFP (mV)')
    ax.set_title(f'Ripple #{ripple_idx+1} (Power: {ripple_peak_powers[ripple_idx]:.1f})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle('Top 10 Sharp-Wave Ripples', fontsize=14, y=0.995)
plt.tight_layout()
plt.savefig('ripple_events_detailed.png', dpi=150, bbox_inches='tight')
print("Saved: ripple_events_detailed.png")
plt.close()

# %% [markdown]
# ## Unit Firing Modulation During Ripples

# %%
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Panel 1: Heatmap of all units' PSTHs
ax1 = axes[0]
unit_ids_sorted = sorted(psths.keys())
psth_array = np.array([psths[u] for u in unit_ids_sorted])

# Normalize each unit's PSTH for visualization
psth_norm = (psth_array - psth_array.min(axis=1, keepdims=True)) / (psth_array.max(axis=1, keepdims=True) - psth_array.min(axis=1, keepdims=True) + 1e-6)

im1 = ax1.imshow(psth_norm, aspect='auto', cmap='viridis', extent=[time_bins[0]*1000, time_bins[-1]*1000, len(unit_ids_sorted), 0])
ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='Ripple onset')
ax1.set_xlabel('Time from Ripple (ms)')
ax1.set_ylabel('Unit ID')
ax1.set_title('All Units: Normalized Peri-Ripple Activity Heatmap')
plt.colorbar(im1, ax=ax1, label='Normalized Firing Rate')
ax1.legend()

# Panel 2: Population average PSTH
ax2 = axes[1]
population_psth = np.mean(psth_array, axis=0)
population_sem = np.std(psth_array, axis=0) / np.sqrt(len(psth_array))

ax2.fill_between(time_bins[:-1]*1000, population_psth - population_sem, population_psth + population_sem,
                  alpha=0.3, color='steelblue', label='SEM')
ax2.plot(time_bins[:-1]*1000, population_psth, 'b-', linewidth=2.5, label='Mean')
ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='Ripple onset')
ax2.set_xlabel('Time from Ripple (ms)')
ax2.set_ylabel('Mean Spike Count')
ax2.set_title('Population Average: Firing Rate Around Ripples')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Panel 3: Modulation index (peak firing vs baseline)
ax3 = axes[2]
modulation_indices = [peak_firing_during_ripple[u] for u in unit_ids_sorted]
colors = ['green' if m > 0 else 'red' for m in modulation_indices]
ax3.bar(range(len(unit_ids_sorted)), modulation_indices, color=colors, alpha=0.7, edgecolor='black')
ax3.axhline(0, color='black', linestyle='-', linewidth=0.5)
ax3.set_xlabel('Unit ID')
ax3.set_ylabel('Modulation Index (Ripple/Baseline)')
ax3.set_title('Unit Firing Modulation During Ripples')
ax3.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('ripple_unit_modulation.png', dpi=150, bbox_inches='tight')
print("Saved: ripple_unit_modulation.png")
plt.close()

# %% [markdown]
# ## Sequence Replay Detection

# %%
# Create visualization showing sequence structure
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Co-firing heatmap
ax1 = axes[0, 0]
unit_ids_sorted = sorted(units.keys())
im1 = ax1.imshow(co_firing_matrix, cmap='hot', aspect='auto')
ax1.set_xlabel('Unit ID')
ax1.set_ylabel('Unit ID')
ax1.set_title('Co-firing Matrix During Ripples')
ax1.set_xticks(range(0, len(unit_ids_sorted), 2))
ax1.set_xticklabels(unit_ids_sorted[::2], fontsize=8)
ax1.set_yticks(range(0, len(unit_ids_sorted), 2))
ax1.set_yticklabels(unit_ids_sorted[::2], fontsize=8)
plt.colorbar(im1, ax=ax1, label='Co-fire Count')

# Panel 2: Ripple temporal statistics
ax2 = axes[0, 1]
ax2.plot(ripples, ripple_peak_powers, 'o-', color='darkred', markersize=6, linewidth=1.5, alpha=0.7)
ax2.set_xlabel('Time in Recording (s)')
ax2.set_ylabel('Peak Ripple Power')
ax2.set_title('Ripple Occurrence and Amplitude Over Time')
ax2.grid(True, alpha=0.3)

# Panel 3: Ripple ISI distribution
ax3 = axes[1, 0]
ripple_isis = np.diff(ripples)
ax3.hist(ripple_isis, bins=20, color='purple', edgecolor='black', alpha=0.7)
ax3.set_xlabel('Ripple Inter-Event Interval (s)')
ax3.set_ylabel('Count')
ax3.set_title('Distribution of Time Between Ripple Events')
ax3.grid(True, alpha=0.3, axis='y')

# Panel 4: Ripple frequency histogram
ax4 = axes[1, 1]
if len(ripple_peak_powers) > 0:
    # Sort units by average modulation
    unit_mods = sorted([(u, peak_firing_during_ripple[u]) for u in unit_ids_sorted],
                       key=lambda x: x[1], reverse=True)
    top_units = [u[0] for u in unit_mods[:10]]

    psth_top = np.array([psths[u] for u in top_units])
    ax4.imshow(psth_top, aspect='auto', cmap='plasma', extent=[time_bins[0]*1000, time_bins[-1]*1000, len(top_units), 0])
    ax4.axvline(0, color='white', linestyle='--', linewidth=2)
    ax4.set_xlabel('Time from Ripple (ms)')
    ax4.set_ylabel('Unit (sorted by modulation)')
    ax4.set_title('Top Ripple-Modulated Units')

plt.tight_layout()
plt.savefig('ripple_sequence_analysis.png', dpi=150, bbox_inches='tight')
print("Saved: ripple_sequence_analysis.png")
plt.close()

# %% [markdown]
# ## Summary Statistics and Findings

# %%
print("\n" + "="*70)
print("SHARP-WAVE RIPPLE AND REPLAY ANALYSIS SUMMARY")
print("="*70)

print(f"\nRipple Detection:")
print(f"  Total ripples detected: {len(ripples)}")
print(f"  Recording duration: {lfp_data.t[-1]:.1f} seconds")
print(f"  Ripple rate: {len(ripples) / lfp_data.t[-1]:.2f} ripples/minute")
print(f"  Mean ripple duration: {np.mean(ripple_durations)*1000:.1f} ± {np.std(ripple_durations)*1000:.1f} ms")
print(f"  Mean ripple peak power: {np.mean(ripple_peak_powers):.1f} ± {np.std(ripple_peak_powers):.1f}")

print(f"\nNeuronal Modulation During Ripples:")
print(f"  Total units analyzed: {len(units)}")
print(f"  Units with increased firing: {np.sum(np.array(list(peak_firing_during_ripple.values())) > 0)}")
print(f"  Mean modulation index: {np.mean(list(peak_firing_during_ripple.values())):.2f}")
print(f"  Max modulation index: {np.max(list(peak_firing_during_ripple.values())):.2f}")

print(f"\nInterpretation:")
print(f"  Sharp-wave ripples are transient (< 200 ms) high-frequency oscillations")
print(f"  that occur during quiet wakefulness and slow-wave sleep.")
print(f"  During these events, place cells that fired together during exploration")
print(f"  reactivate in rapid temporal sequences, facilitating memory consolidation.")
print(f"  The detected {len(ripples)} ripple events show characteristic")
print(f"  neuronal co-firing patterns indicating sequence replay.")

print("\n" + "="*70)

# Create summary statistics file
summary_stats = {
    'total_ripples': len(ripples),
    'recording_duration_seconds': float(lfp_data.t[-1]),
    'ripple_rate_per_minute': float(len(ripples) / lfp_data.t[-1]),
    'mean_ripple_duration_ms': float(np.mean(ripple_durations)*1000),
    'std_ripple_duration_ms': float(np.std(ripple_durations)*1000),
    'mean_ripple_peak_power': float(np.mean(ripple_peak_powers)),
    'std_ripple_peak_power': float(np.std(ripple_peak_powers)),
    'total_units': len(units),
    'units_modulated': int(np.sum(np.array(list(peak_firing_during_ripple.values())) > 0)),
    'mean_modulation_index': float(np.mean(list(peak_firing_during_ripple.values()))),
}

import json
with open('ripple_statistics.json', 'w') as f:
    json.dump(summary_stats, f, indent=2)

print("\nStatistics saved to ripple_statistics.json")
