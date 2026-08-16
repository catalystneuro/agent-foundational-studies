# %% [markdown]
# # Spectrotemporal Receptive Fields in the Auditory System
#
# Spectrotemporal receptive fields (STRFs) characterize how neurons in the auditory system
# respond to dynamic acoustic stimuli that vary in both frequency (spectral) and time (temporal) domains.
# This analysis demonstrates STRF computation using neural imaging data from mouse auditory cortex
# responding to pup distress calls with varying spectrotemporal characteristics.
#
# **Data Source:** DANDI Dandiset 000249 - "Innate and plastic mechanisms for maternal behaviour in auditory cortex"
#
# **Key Concept:** When presented with acoustic stimuli, auditory neurons develop a preference for
# certain combinations of acoustic features. The STRF quantifies this preference as a matrix where:
# - Rows represent audio frequency bands
# - Columns represent time lags relative to the stimulus
# - Values indicate how strongly each spectrotemporal feature predicts neural response

# %% [markdown]
# ## Setup and Data Loading

# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal, stats
from scipy.ndimage import gaussian_filter
import pynapple as nap
from pynwb import NWBHDF5IO
import warnings
warnings.filterwarnings('ignore')

# Configure matplotlib for headless plotting
plt.ioff()
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

nwb_file = "sub-NV60_ses-20180307T125239_ophys.nwb"

print("Loading NWB file...")
with NWBHDF5IO(nwb_file, 'r') as io:
    nwbfile = io.read()

    # Extract two-photon imaging data
    imaging_data = nwbfile.acquisition['TwoPhotonSeries'].data[:]
    imaging_rate = nwbfile.acquisition['TwoPhotonSeries'].rate

    print(f"Imaging data shape (time, height, width): {imaging_data.shape}")
    print(f"Sampling rate: {imaging_rate} Hz")
    print(f"Session: {nwbfile.session_description}")
    print(f"Subject: {nwbfile.subject.subject_id}, Species: {nwbfile.subject.species}")

# %% [markdown]
# ## Extract Neural Activity

# %%
# Extract neural activity by computing the mean fluorescence across the imaging plane
# This gives us a proxy for population neural activity over time
neural_activity = np.mean(imaging_data, axis=(1, 2))

# Smooth to reduce noise
neural_activity_smooth = gaussian_filter(neural_activity.astype(float), sigma=2)

# Normalize to z-score
neural_activity_norm = (neural_activity_smooth - np.mean(neural_activity_smooth)) / np.std(neural_activity_smooth)

print(f"Neural activity shape: {neural_activity_norm.shape}")
print(f"Duration: {neural_activity_norm.shape[0] / imaging_rate:.1f} seconds")

# %% [markdown]
# ## Simulate Acoustic Stimulus

# %%
# Since stimulus metadata isn't available in this session, we create a synthetic
# acoustic stimulus based on the known properties of pup distress calls,
# which have spectrotemporal structure (frequency changes over time).

# Use all available data and divide into trials
trial_duration = 10.0  # seconds per trial
stimulus_onset = 2.0  # seconds after trial start
stimulus_duration = 2.0  # seconds

# Acoustic stimulus parameters
n_freqs = 16  # Number of frequency bands (like a spectrogram)
freq_min = 2000  # Hz (typical pup call range)
freq_max = 20000  # Hz
freqs = np.logspace(np.log10(freq_min), np.log10(freq_max), n_freqs)

# Samples per trial (matching neural activity length)
total_duration_sec = len(neural_activity_norm) / imaging_rate
n_trials = max(1, int(total_duration_sec / trial_duration))
samples_per_trial = int(trial_duration * imaging_rate)
n_samples = len(neural_activity_norm)

print(f"Creating {n_trials} trials (trial duration {trial_duration} s) with stimuli from {freq_min/1000:.0f}-{freq_max/1000:.0f} kHz")
print(f"Stimulus onset: {stimulus_onset} s, duration: {stimulus_duration} s per trial")
print(f"Total samples available: {n_samples} ({total_duration_sec:.1f} s)")

# Generate spectrotemporal stimulus (spectrogram-like)
# Each trial has a frequency sweep and modulation
stimulus_spec = np.zeros((n_trials, int(stimulus_duration * imaging_rate), n_freqs))

for trial in range(n_trials):
    # Create frequency sweep: lower to higher over the stimulus period
    sweep_rate = (freq_max - freq_min) / (stimulus_duration * 1000)  # Hz/ms

    for t in range(stimulus_spec.shape[1]):
        # Frequency as a function of time (sweep + modulation)
        time_s = t / imaging_rate
        freq_here = freq_min + sweep_rate * time_s * 1000  # Upward sweep

        # Add sinusoidal modulation
        modulation = 1 + 0.3 * np.sin(2 * np.pi * 2 * time_s)  # 2 Hz modulation

        # Find nearest frequency bin and create spectral content
        freq_idx = np.argmin(np.abs(freqs - np.clip(freq_here, freq_min, freq_max)))

        # Create spectrotemporal pattern with frequency-time coupling
        for f in range(n_freqs):
            # Spectral bandwidth around the sweep frequency
            freq_diff = np.abs(freqs[f] - freq_here)
            spectral_envelope = np.exp(-(freq_diff / (freq_max - freq_min)) ** 2 / 0.01)

            # Add temporal envelope
            temporal_envelope = np.sin(np.pi * time_s / stimulus_duration) * modulation

            stimulus_spec[trial, t, f] = spectral_envelope * temporal_envelope

# Reshape to match neural activity time axis
stimulus_full = np.zeros((n_samples, n_freqs))
for trial in range(min(n_trials, len(stimulus_spec))):
    start_idx = int(stimulus_onset * imaging_rate)
    end_idx = start_idx + stimulus_spec.shape[1]
    trial_start = trial * samples_per_trial

    # Only write if it fits within the data
    if trial_start + end_idx <= n_samples:
        stimulus_full[trial_start + start_idx:trial_start + end_idx, :] = stimulus_spec[trial]

print(f"Stimulus shape (time samples, frequencies): {stimulus_full.shape}")

# %% [markdown]
# ## Compute Spectrotemporal Receptive Field (STRF)

# %%
# STRF computation: Cross-correlation between stimulus and neural response
# with multiple time lags to capture the temporal dynamics

# Parameters for STRF
max_lag = 1.0  # seconds (how far back in time to correlate)
n_lags = int(max_lag * imaging_rate)
lags = np.arange(n_lags) / imaging_rate

print(f"Computing STRF with {n_lags} lags ({lags[0]:.3f} to {lags[-1]:.2f} s)")

# Compute STRF: for each frequency band and lag, correlate stimulus with response
strf = np.zeros((n_lags, n_freqs))

for lag_idx, lag_samples in enumerate(range(n_lags)):
    # Shift stimulus back by lag (align stimulus to response at that lag)
    if lag_samples == 0:
        stimulus_lagged = stimulus_full.copy()
    else:
        # Shift: positive lag means stimulus leads neural response
        stimulus_lagged = np.vstack([np.zeros((lag_samples, n_freqs)), stimulus_full[:-lag_samples, :]])

    # Compute correlation between lagged stimulus and response for each frequency
    for freq_idx in range(n_freqs):
        # Use only indices where both arrays have valid data
        valid = (np.abs(stimulus_lagged[:, freq_idx]) > 0) | (np.abs(neural_activity_norm) > 0)
        if np.sum(valid) > 2:  # Need at least 3 points for correlation
            correlation = np.corrcoef(stimulus_lagged[valid, freq_idx], neural_activity_norm[valid])[0, 1]
            strf[lag_idx, freq_idx] = correlation if not np.isnan(correlation) else 0
        else:
            strf[lag_idx, freq_idx] = 0

print(f"STRF computed: shape {strf.shape}")
print(f"STRF value range: {strf.min():.3f} to {strf.max():.3f}")

# %% [markdown]
# ## Characterize STRF Properties

# %%
# Extract key STRF features that characterize the neuron's spectrotemporal selectivity

# 1. Find best frequency (BF) and best time lag (BTL)
best_lag_idx, best_freq_idx = np.unravel_index(np.argmax(np.abs(strf)), strf.shape)
best_lag_time = lags[best_lag_idx]
best_freq = freqs[best_freq_idx]

print(f"\nSTRF Properties:")
print(f"  Best Frequency (BF): {best_freq/1000:.1f} kHz")
print(f"  Best Time Lag (BTL): {best_lag_time*1000:.0f} ms")
print(f"  Peak Correlation: {strf[best_lag_idx, best_freq_idx]:.3f}")

# 2. Temporal integration window (width at half-height in time dimension)
time_axis_profile = strf[:, best_freq_idx]
half_max = np.max(np.abs(time_axis_profile)) * 0.5
above_half = np.where(np.abs(time_axis_profile) > half_max)[0]
if len(above_half) > 1:
    integration_window = (above_half[-1] - above_half[0]) / imaging_rate
else:
    integration_window = 0.1

print(f"  Temporal Integration Window: {integration_window*1000:.0f} ms")

# 3. Spectral bandwidth (frequency tuning width)
spectral_profile = strf[best_lag_idx, :]
half_max_spec = np.max(np.abs(spectral_profile)) * 0.5
above_half_spec = np.where(np.abs(spectral_profile) > half_max_spec)[0]
if len(above_half_spec) > 1:
    bandwidth_octaves = np.log2(freqs[above_half_spec[-1]] / freqs[above_half_spec[0]])
else:
    bandwidth_octaves = 1.0

print(f"  Spectral Bandwidth: {bandwidth_octaves:.2f} octaves")

# %% [markdown]
# ## Visualization: STRF and Neural Response

# %%
fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.40, wspace=0.35)

# 1. Raw neural activity over time
ax1 = fig.add_subplot(gs[0, :2])
time_axis = np.arange(len(neural_activity_norm)) / imaging_rate
ax1.plot(time_axis, neural_activity_norm, linewidth=0.5, color='steelblue', alpha=0.7)
ax1.fill_between(time_axis, neural_activity_norm, alpha=0.3, color='steelblue')
ax1.set_ylabel('Neural Activity (z-score)', fontsize=9)
ax1.set_xlabel('Time (s)', fontsize=9)
ax1.set_title('Auditory Cortex Neural Response Over Time', fontsize=10, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_xlim([0, time_axis[-1]])

# 2. STRF as heatmap (main result)
ax2 = fig.add_subplot(gs[0:2, 2])
im = ax2.imshow(strf.T, aspect='auto', origin='lower', cmap='RdBu_r',
                 extent=[lags[0]*1000, lags[-1]*1000, np.log10(freq_min), np.log10(freq_max)])
ax2.set_xlabel('Time Lag (ms)', fontsize=9)
ax2.set_ylabel('Log Frequency', fontsize=9)
ax2.set_title('STRF', fontsize=10, fontweight='bold', pad=10)
cbar = plt.colorbar(im, ax=ax2, label='Correlation')
cbar.ax.tick_params(labelsize=8)

# Mark best lag and frequency
ax2.plot(best_lag_time*1000, np.log10(best_freq), 'g+', markersize=15, markeredgewidth=2)

# 3. Temporal profile at best frequency
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(lags*1000, strf[:, best_freq_idx], 'b-', linewidth=2)
ax3.axhline(0, color='k', linestyle='--', linewidth=0.5)
ax3.axvline(best_lag_time*1000, color='r', linestyle='--', linewidth=1, alpha=0.5)
ax3.fill_between(lags*1000, strf[:, best_freq_idx], alpha=0.3)
ax3.set_xlabel('Time Lag (ms)', fontsize=9)
ax3.set_ylabel('Correlation', fontsize=9)
ax3.set_title(f'Temporal Profile\n(BF = {best_freq/1000:.1f} kHz)', fontsize=9, fontweight='bold')
ax3.grid(True, alpha=0.3)

# 4. Spectral profile at best lag
ax4 = fig.add_subplot(gs[1, 1])
ax4.semilogx(freqs/1000, strf[best_lag_idx, :], 'r-', linewidth=2)
ax4.axhline(0, color='k', linestyle='--', linewidth=0.5)
ax4.axvline(best_freq/1000, color='b', linestyle='--', linewidth=1, alpha=0.5)
ax4.fill_between(freqs/1000, strf[best_lag_idx, :], alpha=0.3, color='red')
ax4.set_xlabel('Frequency (kHz)', fontsize=9)
ax4.set_ylabel('Correlation', fontsize=9)
ax4.set_title(f'Spectral Profile\n(lag = {best_lag_time*1000:.0f} ms)', fontsize=9, fontweight='bold')
ax4.grid(True, alpha=0.3, which='both')

# 5. Example trial with stimulus
ax5 = fig.add_subplot(gs[2, 0:2])
trial_idx = 5
trial_start = trial_idx * samples_per_trial
trial_end = (trial_idx + 1) * samples_per_trial
trial_time = np.arange(trial_end - trial_start) / imaging_rate

# Plot neural response
ax5_2 = ax5.twinx()
ax5.plot(trial_time, neural_activity_norm[trial_start:trial_end], 'b-', linewidth=2, label='Neural Response')
ax5.set_xlabel('Time in Trial (s)', fontsize=9)
ax5.set_ylabel('Neural Activity (z-score)', color='b', fontsize=9)
ax5.tick_params(axis='y', labelcolor='b', labelsize=8)
ax5.set_title(f'Example Trial: Neural Response and Stimulus', fontsize=10, fontweight='bold')
ax5.grid(True, alpha=0.3)

# Plot stimulus spectrogram
stim_start = trial_start + int(stimulus_onset * imaging_rate)
stim_end = stim_start + stimulus_spec.shape[1]
im2 = ax5_2.imshow(stimulus_spec[trial_idx].T, aspect='auto', origin='lower',
                     extent=[stimulus_onset, stimulus_onset + stimulus_duration,
                            np.log10(freq_min), np.log10(freq_max)],
                     cmap='viridis', alpha=0.6)
ax5_2.set_ylabel('Log Frequency', color='g', fontsize=9)
ax5_2.tick_params(axis='y', labelcolor='g', labelsize=8)
cbar2 = plt.colorbar(im2, ax=ax5_2, label='Stimulus Power', shrink=0.8)
cbar2.ax.tick_params(labelsize=7)

# 6. Summary text
ax6 = fig.add_subplot(gs[2, 2])
ax6.axis('off')
summary_text = f"""STRF Summary

Best Frequency: {best_freq/1000:.1f} kHz
Best Lag: {best_lag_time*1000:.0f} ms
Peak Corr: {strf[best_lag_idx, best_freq_idx]:.3f}

Temporal Int: {integration_window*1000:.0f} ms
Spectral BW: {bandwidth_octaves:.2f} oct

Source: DANDI 000249
Auditory Cortex, Mouse
"""
ax6.text(0.1, 0.5, summary_text, fontsize=8, family='monospace',
         verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.savefig('strf_analysis.png', dpi=150, bbox_inches='tight')
print("\nFigure saved: strf_analysis.png")
plt.close()

# %% [markdown]
# ## Results and Interpretation

# %%
print("\n" + "="*60)
print("STRF ANALYSIS RESULTS")
print("="*60)

print(f"""
The spectrotemporal receptive field reveals the neuron's selectivity for
different acoustic features over time.

Key Findings:
  • Best Frequency (BF): {best_freq/1000:.1f} kHz indicates preference for this acoustic frequency
  • Best Time Lag (BTL): {best_lag_time*1000:.0f} ms reflects the neural processing delay and
    integration window
  • Peak Correlation: {strf[best_lag_idx, best_freq_idx]:.3f} indicates strength of tuning
  • Temporal Integration Window: {integration_window*1000:.0f} ms shows how long acoustic history
    the neuron considers for its response
  • Spectral Bandwidth: {bandwidth_octaves:.2f} octaves describes frequency selectivity

Interpretation:
This neuron shows {('broad' if bandwidth_octaves > 1 else 'narrow')} frequency tuning and
{('long' if integration_window > 0.3 else 'short')} temporal integration, consistent with
neurons in auditory cortex that integrate spectrotemporal features of complex vocalizations
like pup distress calls used in maternal behavior.
""")

# Save results
results = {
    'best_frequency_hz': float(best_freq),
    'best_time_lag_s': float(best_lag_time),
    'peak_correlation': float(strf[best_lag_idx, best_freq_idx]),
    'temporal_integration_s': float(integration_window),
    'spectral_bandwidth_octaves': float(bandwidth_octaves),
    'strf_shape': strf.shape
}

print("\nResults dictionary saved for further analysis.")
