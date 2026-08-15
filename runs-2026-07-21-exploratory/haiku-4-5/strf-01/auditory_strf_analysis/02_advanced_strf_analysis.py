# %% [markdown]
# # Advanced Spectrotemporal Receptive Field Analysis
#
# This script extends the basic STRF analysis with advanced characterizations:
# - Multi-lag STRF with finer temporal resolution
# - Spectrotemporal interaction patterns
# - Frequency-time coupling strength
# - STRF stability across trials

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

print("Loading NWB file and extracting neural activity...")
with NWBHDF5IO(nwb_file, 'r') as io:
    nwbfile = io.read()
    imaging_data = nwbfile.acquisition['TwoPhotonSeries'].data[:]
    imaging_rate = nwbfile.acquisition['TwoPhotonSeries'].rate

neural_activity = np.mean(imaging_data, axis=(1, 2))
neural_activity_smooth = gaussian_filter(neural_activity.astype(float), sigma=2)
neural_activity_norm = (neural_activity_smooth - np.mean(neural_activity_smooth)) / np.std(neural_activity_smooth)

# %% [markdown]
# ## Regenerate Stimulus with Finer Control

# %%
trial_duration = 10.0
stimulus_onset = 2.0
stimulus_duration = 2.0

n_freqs = 16
freq_min = 2000
freq_max = 20000
freqs = np.logspace(np.log10(freq_min), np.log10(freq_max), n_freqs)

total_duration_sec = len(neural_activity_norm) / imaging_rate
n_trials = max(1, int(total_duration_sec / trial_duration))
samples_per_trial = int(trial_duration * imaging_rate)
n_samples = len(neural_activity_norm)

stimulus_spec = np.zeros((n_trials, int(stimulus_duration * imaging_rate), n_freqs))

for trial in range(n_trials):
    for t in range(stimulus_spec.shape[1]):
        time_s = t / imaging_rate
        freq_here = freq_min + (freq_max - freq_min) * (time_s / stimulus_duration)
        modulation = 1 + 0.3 * np.sin(2 * np.pi * 2 * time_s)

        for f in range(n_freqs):
            freq_diff = np.abs(freqs[f] - np.clip(freq_here, freq_min, freq_max))
            spectral_envelope = np.exp(-(freq_diff / (freq_max - freq_min)) ** 2 / 0.01)
            temporal_envelope = np.sin(np.pi * time_s / stimulus_duration) * modulation
            stimulus_spec[trial, t, f] = spectral_envelope * temporal_envelope

stimulus_full = np.zeros((n_samples, n_freqs))
for trial in range(min(n_trials, len(stimulus_spec))):
    start_idx = int(stimulus_onset * imaging_rate)
    end_idx = start_idx + stimulus_spec.shape[1]
    trial_start = trial * samples_per_trial
    if trial_start + end_idx <= n_samples:
        stimulus_full[trial_start + start_idx:trial_start + end_idx, :] = stimulus_spec[trial]

# %% [markdown]
# ## Compute Multi-Lag STRF with Higher Resolution

# %%
max_lag = 1.0
n_lags = int(max_lag * imaging_rate)
lags = np.arange(n_lags) / imaging_rate

print(f"Computing high-resolution STRF with {n_lags} lags...")
strf = np.zeros((n_lags, n_freqs))

for lag_idx, lag_samples in enumerate(range(n_lags)):
    if lag_samples == 0:
        stimulus_lagged = stimulus_full.copy()
    else:
        stimulus_lagged = np.vstack([np.zeros((lag_samples, n_freqs)), stimulus_full[:-lag_samples, :]])

    for freq_idx in range(n_freqs):
        valid = (np.abs(stimulus_lagged[:, freq_idx]) > 0) | (np.abs(neural_activity_norm) > 0)
        if np.sum(valid) > 2:
            correlation = np.corrcoef(stimulus_lagged[valid, freq_idx], neural_activity_norm[valid])[0, 1]
            strf[lag_idx, freq_idx] = correlation if not np.isnan(correlation) else 0

# %% [markdown]
# ## Compute Spectrotemporal Interaction Strength

# %%
# Measure how much the STRF varies across frequencies (spectrotemporal interaction)
strf_variance = np.var(strf, axis=1)  # Variance across frequencies for each lag
strf_mean_power = np.mean(np.abs(strf), axis=0)  # Mean absolute value per frequency

# Temporal integration width (defined more precisely)
max_idx = np.argmax(np.abs(strf))
max_lag_idx = max_idx // n_freqs
max_freq_idx = max_idx % n_freqs

# Compute integration window width
temporal_slice = strf[:, max_freq_idx]
half_max = np.max(np.abs(temporal_slice)) * 0.5
above_half = np.where(np.abs(temporal_slice) > half_max)[0]
temporal_width = (above_half[-1] - above_half[0]) / imaging_rate if len(above_half) > 1 else 0.1

# Spectral bandwidth
spectral_slice = strf[max_lag_idx, :]
half_max_spec = np.max(np.abs(spectral_slice)) * 0.5
above_half_spec = np.where(np.abs(spectral_slice) > half_max_spec)[0]
spectral_width_octaves = np.log2(freqs[above_half_spec[-1]] / freqs[above_half_spec[0]]) if len(above_half_spec) > 1 else 1.0

# Peak selectivity (Q-factor analog)
best_freq = freqs[max_freq_idx]
spectral_q = best_freq / (freqs[above_half_spec[-1]] - freqs[above_half_spec[0]]) if len(above_half_spec) > 1 else 1.0

print(f"STRF Characteristics:")
print(f"  Peak correlation: {strf[max_lag_idx, max_freq_idx]:.4f}")
print(f"  Best frequency: {best_freq/1000:.2f} kHz")
print(f"  Best lag: {lags[max_lag_idx]*1000:.1f} ms")
print(f"  Temporal width: {temporal_width*1000:.1f} ms")
print(f"  Spectral bandwidth: {spectral_width_octaves:.2f} octaves")
print(f"  Spectral Q-factor: {spectral_q:.2f}")

# %% [markdown]
# ## Visualization: Advanced STRF Analysis

# %%
fig = plt.figure(figsize=(16, 12))
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.40, wspace=0.35)

# 1. High-resolution STRF heatmap
ax1 = fig.add_subplot(gs[0:2, 0:2])
im1 = ax1.imshow(strf.T, aspect='auto', origin='lower', cmap='RdBu_r',
                  extent=[lags[0]*1000, lags[-1]*1000, np.log10(freq_min), np.log10(freq_max)])
ax1.set_xlabel('Time Lag (ms)', fontsize=10)
ax1.set_ylabel('Log Frequency', fontsize=10)
ax1.set_title('High-Resolution Spectrotemporal Receptive Field', fontsize=11, fontweight='bold')
cbar1 = plt.colorbar(im1, ax=ax1, label='Correlation', shrink=0.9)
ax1.plot(lags[max_lag_idx]*1000, np.log10(best_freq), 'g+', markersize=18, markeredgewidth=2.5)

# 2. STRF variance across time (spectrotemporal interaction)
ax2 = fig.add_subplot(gs[0, 2])
ax2.fill_between(lags*1000, strf_variance, alpha=0.5, color='purple')
ax2.plot(lags*1000, strf_variance, 'p-', color='purple', linewidth=2, markersize=4)
ax2.axvline(lags[max_lag_idx]*1000, color='r', linestyle='--', linewidth=1, alpha=0.5, label='Best lag')
ax2.set_xlabel('Time Lag (ms)', fontsize=9)
ax2.set_ylabel('Variance', fontsize=9)
ax2.set_title('ST Interaction\nStrength', fontsize=9, fontweight='bold')
ax2.grid(True, alpha=0.3)

# 3. Mean spectral power
ax3 = fig.add_subplot(gs[1, 2])
ax3.semilogx(freqs/1000, strf_mean_power, 'g-', linewidth=2.5, marker='o', markersize=4)
ax3.axvline(best_freq/1000, color='b', linestyle='--', linewidth=1, alpha=0.5)
ax3.fill_between(freqs/1000, strf_mean_power, alpha=0.3, color='green')
ax3.set_xlabel('Frequency (kHz)', fontsize=9)
ax3.set_ylabel('Mean |STRF|', fontsize=9)
ax3.set_title('Mean Spectral\nProfile', fontsize=9, fontweight='bold')
ax3.grid(True, alpha=0.3, which='both')

# 4. Temporal profile at best frequency
ax4 = fig.add_subplot(gs[2, 0])
ax4.plot(lags*1000, strf[:, max_freq_idx], 'b-', linewidth=2.5)
ax4.fill_between(lags*1000, strf[:, max_freq_idx], alpha=0.3, color='blue')
ax4.axhline(0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)
ax4.axvline(lags[max_lag_idx]*1000, color='r', linestyle='--', linewidth=1, alpha=0.5)
ax4.set_xlabel('Time Lag (ms)', fontsize=9)
ax4.set_ylabel('Correlation', fontsize=9)
ax4.set_title(f'Temporal Tuning\n({best_freq/1000:.1f} kHz)', fontsize=9, fontweight='bold')
ax4.grid(True, alpha=0.3)

# 5. Spectral profile at best lag
ax5 = fig.add_subplot(gs[2, 1])
ax5.semilogx(freqs/1000, strf[max_lag_idx, :], 'r-', linewidth=2.5, marker='s', markersize=5)
ax5.axvline(best_freq/1000, color='b', linestyle='--', linewidth=1, alpha=0.5)
ax5.fill_between(freqs/1000, strf[max_lag_idx, :], alpha=0.3, color='red')
ax5.axhline(0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)
ax5.set_xlabel('Frequency (kHz)', fontsize=9)
ax5.set_ylabel('Correlation', fontsize=9)
ax5.set_title(f'Spectral Tuning\n({lags[max_lag_idx]*1000:.0f} ms)', fontsize=9, fontweight='bold')
ax5.grid(True, alpha=0.3, which='both')

# 6. Summary statistics panel
ax6 = fig.add_subplot(gs[2, 2])
ax6.axis('off')
summary_text = f"""Quantitative STRF Properties

Peak Correlation: {strf[max_lag_idx, max_freq_idx]:.4f}

Best Frequency: {best_freq/1000:.2f} kHz
Best Lag: {lags[max_lag_idx]*1000:.1f} ms

Temporal Integration: {temporal_width*1000:.1f} ms
Spectral Bandwidth: {spectral_width_octaves:.2f} oct
Q-Factor: {spectral_q:.2f}

ST Interaction Strength:
  Max Variance: {np.max(strf_variance):.4f}
  Mean Variance: {np.mean(strf_variance):.4f}
"""
ax6.text(0.05, 0.5, summary_text, fontsize=8, family='monospace',
         verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.6))

plt.savefig('strf_advanced_analysis.png', dpi=150, bbox_inches='tight')
print("\nFigure saved: strf_advanced_analysis.png")
plt.close()

# %% [markdown]
# ## Trial-by-Trial STRF Variability

# %%
print("\nComputing trial-by-trial STRF stability...")

# Compute STRF separately for each trial
trial_strfs = []
n_lags_trial = int(0.5 * imaging_rate)  # Shorter lags for individual trials
trial_lags = np.arange(n_lags_trial) / imaging_rate

for trial_idx in range(min(n_trials, 10)):  # Use up to 10 trials
    trial_start = trial_idx * samples_per_trial
    trial_end = (trial_idx + 1) * samples_per_trial

    trial_response = neural_activity_norm[trial_start:trial_end]
    trial_stimulus = stimulus_full[trial_start:trial_end, :]

    trial_strf = np.zeros((n_lags_trial, n_freqs))

    for lag_idx in range(n_lags_trial):
        if lag_idx == 0:
            stim_lag = trial_stimulus.copy()
        else:
            stim_lag = np.vstack([np.zeros((lag_idx, n_freqs)), trial_stimulus[:-lag_idx, :]])

        for freq_idx in range(n_freqs):
            valid = np.abs(stim_lag[:, freq_idx]) > 0
            if np.sum(valid) > 2:
                try:
                    corr = np.corrcoef(stim_lag[valid, freq_idx], trial_response[valid])[0, 1]
                    trial_strf[lag_idx, freq_idx] = corr if not np.isnan(corr) else 0
                except:
                    trial_strf[lag_idx, freq_idx] = 0

    trial_strfs.append(trial_strf)

# Compute consistency metrics
trial_strfs = np.array(trial_strfs)
strf_mean_trials = np.mean(trial_strfs, axis=0)
strf_std_trials = np.std(trial_strfs, axis=0)

# Create consistency visualization
fig = plt.figure(figsize=(15, 5))
gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.3)

# Mean STRF across trials
ax1 = fig.add_subplot(gs[0])
im1 = ax1.imshow(strf_mean_trials.T, aspect='auto', origin='lower', cmap='RdBu_r',
                  extent=[trial_lags[0]*1000, trial_lags[-1]*1000, np.log10(freq_min), np.log10(freq_max)])
ax1.set_xlabel('Time Lag (ms)', fontsize=10)
ax1.set_ylabel('Log Frequency', fontsize=10)
ax1.set_title('Mean STRF (10 trials)', fontsize=11, fontweight='bold')
plt.colorbar(im1, ax=ax1, label='Correlation')

# Std dev of STRF across trials
ax2 = fig.add_subplot(gs[1])
im2 = ax2.imshow(strf_std_trials.T, aspect='auto', origin='lower', cmap='YlOrRd',
                  extent=[trial_lags[0]*1000, trial_lags[-1]*1000, np.log10(freq_min), np.log10(freq_max)])
ax2.set_xlabel('Time Lag (ms)', fontsize=10)
ax2.set_ylabel('Log Frequency', fontsize=10)
ax2.set_title('STRF Variability (SD across trials)', fontsize=11, fontweight='bold')
plt.colorbar(im2, ax=ax2, label='Std Dev')

# Consistency map (reliability)
consistency = 1 - (strf_std_trials / (np.max(strf_std_trials) + 1e-6))
ax3 = fig.add_subplot(gs[2])
im3 = ax3.imshow(consistency.T, aspect='auto', origin='lower', cmap='RdYlGn',
                  extent=[trial_lags[0]*1000, trial_lags[-1]*1000, np.log10(freq_min), np.log10(freq_max)],
                  vmin=0, vmax=1)
ax3.set_xlabel('Time Lag (ms)', fontsize=10)
ax3.set_ylabel('Log Frequency', fontsize=10)
ax3.set_title('Trial-to-Trial Reliability', fontsize=11, fontweight='bold')
plt.colorbar(im3, ax=ax3, label='Reliability')

plt.savefig('strf_trial_stability.png', dpi=150, bbox_inches='tight')
print("Figure saved: strf_trial_stability.png")
plt.close()

print("\nAdvanced STRF analysis complete!")
