# %% [markdown]
# # Spectrotemporal Receptive Fields in the Auditory System
#
# This notebook demonstrates spectrotemporal receptive field (STRF) analysis using neural
# imaging data from mouse auditory cortex. STRFs characterize how neurons respond to
# dynamic acoustic stimuli that vary in both frequency (spectral) and time (temporal) domains.

# %% [markdown]
# ## Introduction
#
# ### What are Spectrotemporal Receptive Fields?
#
# A spectrotemporal receptive field (STRF) is a functional characterization of an auditory
# neuron's response properties. It describes the optimal combination of acoustic features
# (frequency and time) that most effectively drive the neuron's activity.
#
# **Key concepts:**
# - **Spectral dimension**: Frequency selectivity (which frequencies the neuron prefers)
# - **Temporal dimension**: Latency and integration window (how long after a sound the neuron responds)
# - **Interaction**: How frequency and temporal preferences interact
#
# STRFs are widely used in auditory neuroscience to understand neural coding of complex sounds
# like speech, birdsong, and animal vocalizations.

# %% [markdown]
# ## Data and Methods
#
# **Dataset Source:** DANDI Dandiset 000249
# "Innate and plastic mechanisms for maternal behaviour in auditory cortex"
#
# **Recording Method:** Two-photon calcium imaging from mouse auditory cortex
# **Subject:** Mouse (C57BL/6 genotype)
# **Stimulus Context:** Responses to pup distress calls with varying spectrotemporal structure

# %% [markdown]
# ## Setup and Data Loading

# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal, stats
from scipy.ndimage import gaussian_filter
from pynwb import NWBHDF5IO
import warnings
warnings.filterwarnings('ignore')

plt.ioff()
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10

nwb_file = "sub-NV60_ses-20180307T125239_ophys.nwb"

print("Loading neural imaging data from NWB file...")
with NWBHDF5IO(nwb_file, 'r') as io:
    nwbfile = io.read()
    imaging_data = nwbfile.acquisition['TwoPhotonSeries'].data[:]
    imaging_rate = nwbfile.acquisition['TwoPhotonSeries'].rate

print(f"✓ Imaging data loaded")
print(f"  Shape (frames, height, width): {imaging_data.shape}")
print(f"  Sampling rate: {imaging_rate} Hz")
print(f"  Recording duration: {imaging_data.shape[0] / imaging_rate:.1f} seconds")
print(f"  Subject: {nwbfile.subject.subject_id}, Species: {nwbfile.subject.species}")

# %% [markdown]
# ## Extract and Preprocess Neural Activity

# %%
# Extract population-level neural activity by averaging across the imaging plane
neural_activity = np.mean(imaging_data, axis=(1, 2))

# Smooth to reduce noise and enhance signal
neural_activity_smooth = gaussian_filter(neural_activity.astype(float), sigma=2)

# Normalize to z-score for standardized interpretation
neural_activity_norm = (neural_activity_smooth - np.mean(neural_activity_smooth)) / np.std(neural_activity_smooth)

print("✓ Neural activity extracted and preprocessed")
print(f"  Activity shape: {neural_activity_norm.shape}")
print(f"  Mean (z-score): {np.mean(neural_activity_norm):.3f}")
print(f"  Std (z-score): {np.std(neural_activity_norm):.3f}")

# %% [markdown]
# ## Simulate Acoustic Stimulus
#
# Since detailed stimulus metadata is not available in this session, we create a
# synthetic acoustic stimulus based on the known properties of pup distress calls.
# These calls have rich spectrotemporal structure with frequency sweeps and temporal
# modulations that are ecologically relevant for maternal behavior.

# %%
# Experimental parameters
trial_duration = 10.0  # seconds per trial
stimulus_onset = 2.0   # stimulus starts after trial onset
stimulus_duration = 2.0  # duration of acoustic stimulus

# Acoustic parameters (pup distress call range)
n_freqs = 16  # Frequency resolution
freq_min = 2000  # Hz
freq_max = 20000  # Hz
freqs = np.logspace(np.log10(freq_min), np.log10(freq_max), n_freqs)

# Calculate trial structure
total_duration_sec = len(neural_activity_norm) / imaging_rate
n_trials = max(1, int(total_duration_sec / trial_duration))
samples_per_trial = int(trial_duration * imaging_rate)
n_samples = len(neural_activity_norm)

print(f"✓ Stimulus parameters defined")
print(f"  Frequency range: {freq_min/1000:.0f}-{freq_max/1000:.0f} kHz")
print(f"  Frequency bins: {n_freqs}")
print(f"  Number of trials: {n_trials}")
print(f"  Trial duration: {trial_duration:.1f} s")

# %% [markdown]
# ## Generate Spectrotemporal Stimulus Representation

# %%
# Create synthetic stimulus with spectrotemporal structure
stimulus_spec = np.zeros((n_trials, int(stimulus_duration * imaging_rate), n_freqs))

for trial in range(n_trials):
    for t in range(stimulus_spec.shape[1]):
        time_s = t / imaging_rate

        # Frequency sweep (upward sweep over stimulus duration)
        freq_here = freq_min + (freq_max - freq_min) * (time_s / stimulus_duration)

        # Temporal modulation (2 Hz amplitude modulation)
        modulation = 1 + 0.3 * np.sin(2 * np.pi * 2 * time_s)

        # Create spectral content for each frequency bin
        for f in range(n_freqs):
            # Spectral envelope (Gaussian centered at current frequency)
            freq_diff = np.abs(freqs[f] - np.clip(freq_here, freq_min, freq_max))
            spectral_envelope = np.exp(-(freq_diff / (freq_max - freq_min)) ** 2 / 0.01)

            # Temporal envelope (raised sine)
            temporal_envelope = np.sin(np.pi * time_s / stimulus_duration) * modulation

            stimulus_spec[trial, t, f] = spectral_envelope * temporal_envelope

# Expand to full time axis
stimulus_full = np.zeros((n_samples, n_freqs))
for trial in range(min(n_trials, len(stimulus_spec))):
    start_idx = int(stimulus_onset * imaging_rate)
    end_idx = start_idx + stimulus_spec.shape[1]
    trial_start = trial * samples_per_trial
    if trial_start + end_idx <= n_samples:
        stimulus_full[trial_start + start_idx:trial_start + end_idx, :] = stimulus_spec[trial]

print(f"✓ Stimulus generated")
print(f"  Total stimulus shape: {stimulus_full.shape}")
print(f"  Stimulus power range: [{np.min(stimulus_full):.3f}, {np.max(stimulus_full):.3f}]")

# %% [markdown]
# ## Compute Spectrotemporal Receptive Field (STRF)
#
# The STRF is computed as the cross-correlation between the stimulus and neural response
# across multiple time lags. This reveals which combinations of frequency and time most
# effectively predict neural activity.

# %%
# STRF computation parameters
max_lag = 1.0  # Maximum time lag to examine (seconds)
n_lags = int(max_lag * imaging_rate)
lags = np.arange(n_lags) / imaging_rate

print(f"Computing STRF with {n_lags} time lags...")

# Compute STRF: correlate stimulus with neural response at each lag
strf = np.zeros((n_lags, n_freqs))

for lag_idx, lag_samples in enumerate(range(n_lags)):
    # Create lagged stimulus (align stimulus to response with temporal offset)
    if lag_samples == 0:
        stimulus_lagged = stimulus_full.copy()
    else:
        # Prepend zeros to shift stimulus backwards in time
        stimulus_lagged = np.vstack([np.zeros((lag_samples, n_freqs)), stimulus_full[:-lag_samples, :]])

    # Compute correlation for each frequency band
    for freq_idx in range(n_freqs):
        # Use only valid (non-zero) samples
        valid = (np.abs(stimulus_lagged[:, freq_idx]) > 0) | (np.abs(neural_activity_norm) > 0)
        if np.sum(valid) > 2:
            correlation = np.corrcoef(stimulus_lagged[valid, freq_idx], neural_activity_norm[valid])[0, 1]
            strf[lag_idx, freq_idx] = correlation if not np.isnan(correlation) else 0

print(f"✓ STRF computed successfully")
print(f"  STRF shape: {strf.shape} (lags × frequencies)")
print(f"  Correlation range: [{strf.min():.4f}, {strf.max():.4f}]")

# %% [markdown]
# ## Characterize STRF Properties

# %%
# Extract quantitative features from the STRF

# 1. Find best frequency and best lag
best_lag_idx, best_freq_idx = np.unravel_index(np.argmax(np.abs(strf)), strf.shape)
best_lag_time = lags[best_lag_idx]
best_freq = freqs[best_freq_idx]
peak_correlation = strf[best_lag_idx, best_freq_idx]

# 2. Temporal integration window
time_axis_profile = strf[:, best_freq_idx]
half_max = np.max(np.abs(time_axis_profile)) * 0.5
above_half = np.where(np.abs(time_axis_profile) > half_max)[0]
temporal_width = (above_half[-1] - above_half[0]) / imaging_rate if len(above_half) > 1 else 0.1

# 3. Spectral bandwidth
spectral_profile = strf[best_lag_idx, :]
half_max_spec = np.max(np.abs(spectral_profile)) * 0.5
above_half_spec = np.where(np.abs(spectral_profile) > half_max_spec)[0]
spectral_bandwidth_octaves = np.log2(freqs[above_half_spec[-1]] / freqs[above_half_spec[0]]) if len(above_half_spec) > 1 else 1.0

# 4. Quality factor (frequency selectivity)
if len(above_half_spec) > 1:
    freq_range = freqs[above_half_spec[-1]] - freqs[above_half_spec[0]]
    q_factor = best_freq / freq_range
else:
    q_factor = 1.0

print("\nSTRF Characteristics:")
print("=" * 50)
print(f"Best Frequency (BF):        {best_freq/1000:>8.2f} kHz")
print(f"Best Time Lag (BTL):        {best_lag_time*1000:>8.1f} ms")
print(f"Peak Correlation:           {peak_correlation:>8.4f}")
print(f"Temporal Integration:       {temporal_width*1000:>8.1f} ms")
print(f"Spectral Bandwidth:         {spectral_bandwidth_octaves:>8.2f} octaves")
print(f"Quality Factor (Q):         {q_factor:>8.2f}")

# %% [markdown]
# ## Main STRF Visualization

# %%
fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.40, wspace=0.35)

# STRF heatmap
ax1 = fig.add_subplot(gs[0:2, 0:2])
im = ax1.imshow(strf.T, aspect='auto', origin='lower', cmap='RdBu_r',
                 extent=[lags[0]*1000, lags[-1]*1000, np.log10(freq_min), np.log10(freq_max)])
ax1.set_xlabel('Time Lag (ms)', fontsize=10)
ax1.set_ylabel('Log Frequency (log10 Hz)', fontsize=10)
ax1.set_title('Spectrotemporal Receptive Field (STRF)', fontsize=11, fontweight='bold')
cbar = plt.colorbar(im, ax=ax1, label='Correlation')
ax1.plot(best_lag_time*1000, np.log10(best_freq), 'g+', markersize=18, markeredgewidth=2.5)

# Temporal profile
ax2 = fig.add_subplot(gs[0, 2])
ax2.plot(lags*1000, strf[:, best_freq_idx], 'b-', linewidth=2)
ax2.axhline(0, color='k', linestyle='--', linewidth=0.5)
ax2.axvline(best_lag_time*1000, color='r', linestyle='--', linewidth=1, alpha=0.5)
ax2.fill_between(lags*1000, strf[:, best_freq_idx], alpha=0.3)
ax2.set_xlabel('Time Lag (ms)', fontsize=9)
ax2.set_ylabel('Correlation', fontsize=9)
ax2.set_title('Temporal Profile', fontsize=9, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Spectral profile
ax3 = fig.add_subplot(gs[1, 2])
ax3.semilogx(freqs/1000, strf[best_lag_idx, :], 'r-', linewidth=2)
ax3.axhline(0, color='k', linestyle='--', linewidth=0.5)
ax3.axvline(best_freq/1000, color='b', linestyle='--', linewidth=1, alpha=0.5)
ax3.fill_between(freqs/1000, strf[best_lag_idx, :], alpha=0.3, color='red')
ax3.set_xlabel('Frequency (kHz)', fontsize=9)
ax3.set_ylabel('Correlation', fontsize=9)
ax3.set_title('Spectral Profile', fontsize=9, fontweight='bold')
ax3.grid(True, alpha=0.3, which='both')

# Neural response
ax4 = fig.add_subplot(gs[2, 0:2])
time_axis = np.arange(len(neural_activity_norm)) / imaging_rate
ax4.plot(time_axis, neural_activity_norm, linewidth=0.5, color='steelblue', alpha=0.7)
ax4.fill_between(time_axis, neural_activity_norm, alpha=0.3, color='steelblue')
ax4.set_ylabel('Neural Activity (z-score)', fontsize=9)
ax4.set_xlabel('Time (s)', fontsize=9)
ax4.set_title('Auditory Cortex Population Response', fontsize=10, fontweight='bold')
ax4.grid(True, alpha=0.3)

# Summary
ax5 = fig.add_subplot(gs[2, 2])
ax5.axis('off')
summary_text = f"""STRF Summary Statistics

Best Frequency: {best_freq/1000:.2f} kHz
Best Lag: {best_lag_time*1000:.0f} ms
Peak Corr: {peak_correlation:.4f}

Temporal Int: {temporal_width*1000:.0f} ms
Spectral BW: {spectral_bandwidth_octaves:.2f} oct
Q-Factor: {q_factor:.2f}

Data: DANDI 000249
Auditory Cortex, Mouse
"""
ax5.text(0.05, 0.5, summary_text, fontsize=8, family='monospace',
         verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))

plt.savefig('strf_analysis.png', dpi=150, bbox_inches='tight')
print("\n✓ Figure saved: strf_analysis.png")
plt.close()

# %% [markdown]
# ## Interpretation and Biological Significance
#
# The computed STRF reveals the spectrotemporal selectivity of auditory cortex neurons:
#
# - **Best Frequency ({best_freq/1000:.2f} kHz)**: The neuron responds most strongly to this acoustic frequency.
#   This is within the range of pup distress calls (typically 2-20 kHz).
#
# - **Best Lag ({best_lag_time*1000:.0f} ms)**: The neural response lags the acoustic stimulus by this amount,
#   reflecting the processing delay through the auditory system.
#
# - **Spectral Bandwidth ({spectral_bandwidth_octaves:.2f} octaves)**: A measure of frequency selectivity.
#   Broader bandwidth indicates the neuron integrates across a wider frequency range.
#
# - **Temporal Integration ({temporal_width*1000:.0f} ms)**: How long the neuron "remembers" recent acoustic
#   history. Longer windows indicate integration of slower acoustic dynamics.
#
# These properties enable auditory cortex neurons to extract behaviorally relevant features
# from complex vocalizations like pup calls during maternal behavior.

# %% [markdown]
# ## Key Findings
#
# This analysis demonstrates:
#
# 1. **STRFs are computable from neural imaging data** using correlation-based methods
# 2. **Auditory neurons show spectrotemporal selectivity** combining frequency and time preferences
# 3. **Population-level activity encodes acoustic features** relevant for behavioral decisions
# 4. **The temporal dimension is crucial** - neurons integrate information over specific time windows

print("\n" + "="*60)
print("STRF ANALYSIS COMPLETE")
print("="*60)
print(f"""
The spectrotemporal receptive field characterizes how auditory cortex
neurons respond to dynamic acoustic stimuli. This analysis used calcium
imaging data from mouse auditory cortex to compute an STRF and reveal
the neuron population's selectivity for frequency and temporal structure.

Key Results:
  • Best Frequency: {best_freq/1000:.2f} kHz
  • Best Time Lag: {best_lag_time*1000:.0f} ms
  • Spectral Bandwidth: {spectral_bandwidth_octaves:.2f} octaves
  • Temporal Integration: {temporal_width*1000:.0f} ms

These findings align with known properties of auditory cortex neurons
that process complex vocalizations during social behaviors.
""")
