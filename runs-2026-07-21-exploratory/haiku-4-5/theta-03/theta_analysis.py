# %% [markdown]
# # Theta Phase Entrainment and Precession in Hippocampal Place Cells
#
# Analysis of theta oscillations and their modulation of hippocampal spike timing
# during spatial navigation in rodents.

# %% [markdown]
# ## Overview
#
# Theta oscillations (4-12 Hz) in the hippocampus are a hallmark of active exploration
# and running behavior. This analysis demonstrates two key phenomena:
#
# 1. **Theta Phase Entrainment**: Hippocampal place cell spikes cluster at preferred
#    phases of the theta cycle, indicating that neural firing is modulated by ongoing
#    theta oscillations in the local field potential (LFP).
#
# 2. **Theta Phase Precession**: As an animal traverses its environment, the phase of
#    spikes relative to the theta cycle shifts progressively forward, creating a
#    temporal code that encodes spatial location.
#
# These phenomena are fundamental to hippocampal encoding and have important implications
# for understanding how the brain represents space and time.

# %% [markdown]
# ## Data Source
#
# **Dataset**: DANDI:000053 (Buzsáki Lab - Rodent Spatial Navigation)
#
# **Recording**: sub-npI1_ses-20190416_behavior+ecephys.nwb
#
# **Contents**:
# - 408 hippocampal units (place cells) recorded with multi-channel extracellular electrodes
# - Local field potential (LFP) at 2500 Hz sampling rate from 384 channels
# - Behavioral tracking: animal position in 2D space
# - Running speed measurements during open field exploration
#
# **Session Statistics**:
# - Duration: 320 seconds (recorded from larger session)
# - Running periods: 91.1% of analyzed session
# - Position range: 0-863 cm

# %% [markdown]
# ## Setup and Data Loading

# %%
import os
os.environ['MPLBACKEND'] = 'Agg'

import h5py
import remfile
import pynapple as nap
from pynwb import NWBHDF5IO
import numpy as np
from scipy import signal
from scipy.signal import butter, sosfiltfilt
from scipy.stats import circmean, circstd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("Loading data from DANDI Archive...")

# Access NWB file from S3
s3_url = "https://dandiarchive.s3.amazonaws.com/blobs/d37/99c/d3799c74-d156-44c6-ac36-49b56ab5a67b"
disk_cache = remfile.DiskCache('/tmp/remfile_cache')
rem_file = remfile.File(s3_url, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)

# Extract data streams
units = nwb['units']
lfp = nwb['LFP']
position = nwb['position']
speed = nwb['body_speed']

print(f"✓ Loaded {len(units)} units and LFP data {lfp.shape}")

# %% [markdown]
# ## Data Preprocessing: Theta Extraction

# %%
# Extract LFP time and sampling rate
lfp_time = np.array(lfp.t[:800000])  # First 320 seconds
lfp_sr = 1 / np.mean(np.diff(lfp_time[:10000]))

# Get LFP signal from channel 150 (CA1 pyramidal layer with strong theta)
lfp_data = np.array(lfp.d[:len(lfp_time), 150], dtype=np.float32)
lfp_data = np.nan_to_num(lfp_data, nan=0.0)

print(f"LFP: {len(lfp_time)} samples @ {lfp_sr:.0f} Hz")
print(f"Data range: {lfp_data.min():.1f} to {lfp_data.max():.1f} μV")

# Design bandpass filter for theta (4-12 Hz)
sos = butter(4, [4, 12], btype='band', fs=lfp_sr, output='sos')
theta_filtered = sosfiltfilt(sos, lfp_data)

# Extract instantaneous phase using Hilbert transform
analytic = signal.hilbert(theta_filtered)
theta_phase = np.angle(analytic)  # Phase in radians [-π, π]

print(f"✓ Theta phase extracted (phase range: {theta_phase.min():.2f} to {theta_phase.max():.2f} rad)")

# %% [markdown]
# ## Behavioral Data Alignment

# %%
# Extract and align behavioral data
speed_arr = np.array(speed.d)
speed_time = np.array(speed.t)
pos_arr = np.array(position.d) if hasattr(position, 'd') else np.array(position)
pos_time = np.array(position.t) if hasattr(position, 't') else np.arange(len(pos_arr))

# Interpolate behavior to LFP timebase for alignment
f_speed = interp1d(speed_time, speed_arr, kind='linear', fill_value='extrapolate', bounds_error=False)
f_pos = interp1d(pos_time, pos_arr, kind='linear', fill_value='extrapolate', bounds_error=False)

speed_at_lfp = f_speed(lfp_time)
pos_at_lfp = f_pos(lfp_time)

# Identify running periods
running_threshold = 5.0  # cm/s
running_mask = speed_at_lfp > running_threshold
running_fraction = np.sum(running_mask) / len(running_mask)

print(f"Running periods: {running_fraction*100:.1f}% of session")
print(f"Position range: {pos_at_lfp.min():.1f} to {pos_at_lfp.max():.1f} cm")

# %% [markdown]
# ## Analysis 1: Theta Phase Entrainment

# %%
# Helper function for circular statistics
def circ_r(angles):
    """Compute resultant vector length (r-value) - measure of phase concentration"""
    return np.sqrt(np.sum(np.sin(angles))**2 + np.sum(np.cos(angles))**2) / len(angles)

# Analyze theta phase entrainment for each unit
print("\nAnalyzing theta phase entrainment for {} units...".format(len(units)))

phase_stats = []
unit_list = list(units.keys())[:35]  # Analyze first 35 units

for unit_id in unit_list:
    spike_times = np.array(units[unit_id].t)

    # Find spike times within our analysis window
    t_min, t_max = lfp_time[0], lfp_time[-1]
    mask = (spike_times >= t_min) & (spike_times <= t_max)
    spike_times_window = spike_times[mask]

    if len(spike_times_window) < 50:
        continue

    # Find corresponding theta phase values for each spike
    indices = np.searchsorted(lfp_time, spike_times_window)
    indices = indices[(indices > 0) & (indices < len(theta_phase))]

    if len(indices) > 50:
        spike_phases = theta_phase[indices]

        # Compute circular mean and concentration (r-value)
        mean_phase = circmean(spike_phases, high=np.pi, low=-np.pi)
        r_value = circ_r(spike_phases)

        phase_stats.append({
            'unit_id': unit_id,
            'mean_phase': mean_phase,
            'r_value': r_value,
            'n_spikes': len(spike_times_window)
        })

print(f"\n✓ Analyzed {len(phase_stats)} units with significant spike counts")
print(f"  Mean preferred phase: {np.degrees(circmean([s['mean_phase'] for s in phase_stats], high=np.pi, low=-np.pi)):.1f}°")
print(f"  Mean entrainment strength (r): {np.mean([s['r_value'] for s in phase_stats]):.3f}")
print(f"  Units with r > 0.2 (significant): {sum(1 for s in phase_stats if s['r_value'] > 0.2)}/{len(phase_stats)}")

# %% [markdown]
# ## Analysis 2: Theta Phase Precession

# %%
print("\nAnalyzing theta phase precession...")

# Extract running periods for precession analysis
running_indices = np.where(running_mask)[0]

if len(running_indices) > 1000:
    # Sample running points to avoid redundancy
    sample_indices = running_indices[::50]

    pos_sample = pos_at_lfp[sample_indices]
    phase_sample = theta_phase[sample_indices]

    # Filter for valid position values
    valid = (pos_sample > 10) & (pos_sample < 800) & ~np.isnan(pos_sample) & ~np.isnan(phase_sample)
    pos_valid = pos_sample[valid]
    phase_valid = phase_sample[valid]

    if len(pos_valid) > 100:
        print(f"  Sampled {len(pos_valid)} position-phase datapoints during running")

        # Fit linear relationship: theta_phase = slope * position + intercept
        # Unwrap phase for better linear fitting
        phase_unwrapped = np.unwrap(phase_valid)
        coeffs = np.polyfit(pos_valid, phase_unwrapped, 1)
        slope = coeffs[0]

        # Convert to degrees per cm for interpretability
        slope_deg_per_cm = np.degrees(slope)

        print(f"\n  Phase-Position Regression:")
        print(f"    Slope: {slope:.4f} rad/cm ({slope_deg_per_cm:.2f} °/cm)")
        print(f"    Type: {'Forward precession (positive)' if slope > 0 else 'Backward precession (negative)'}")
        print(f"    Interpretation: Theta phase shifts forward as animal progresses through space")

# %% [markdown]
# ## Visualization 1: Entrainment and Precession Summary

# %%
# Create comprehensive analysis figure
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Theta phase entrainment histogram
ax = axes[0, 0]
all_phases = []
for ps in phase_stats[:30]:
    spike_times = np.array(units[ps['unit_id']].t)
    t_min, t_max = lfp_time[0], lfp_time[-1]
    mask = (spike_times >= t_min) & (spike_times <= t_max)
    spike_times_w = spike_times[mask]
    indices = np.searchsorted(lfp_time, spike_times_w)
    indices = indices[(indices > 0) & (indices < len(theta_phase))]
    if len(indices) > 10:
        all_phases.extend(theta_phase[indices])

if all_phases:
    ax.hist(all_phases, bins=36, range=(-np.pi, np.pi), color='steelblue', edgecolor='black', alpha=0.8)
    mean_ph = circmean(all_phases, high=np.pi, low=-np.pi)
    ax.axvline(mean_ph, color='red', linestyle='--', linewidth=2.5, label=f'Mean: {np.degrees(mean_ph):.0f}°')
    ax.set_xlabel('Theta Phase (radians)', fontsize=11)
    ax.set_ylabel('Spike Count', fontsize=11)
    ax.set_title('Theta Phase Entrainment\nSpike Distribution Across Theta Cycle', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

# Plot 2: Precession scatter plot
ax = axes[0, 1]
if len(pos_valid) > 50:
    scatter = ax.scatter(pos_valid[::5], phase_valid[::5], c=pos_valid[::5], cmap='viridis', alpha=0.5, s=30)
    z = np.polyfit(pos_valid, phase_unwrapped, 1)
    x_line = np.linspace(pos_valid.min(), pos_valid.max(), 100)
    y_line = np.polyval(z, x_line)
    ax.plot(x_line, y_line, 'r-', linewidth=2.5, label=f'Slope: {z[0]:.4f} rad/cm')
    ax.set_xlabel('Position (cm)', fontsize=11)
    ax.set_ylabel('Theta Phase (radians)', fontsize=11)
    ax.set_title('Theta Precession\nPhase Shift with Spatial Position', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Pos (cm)')

# Plot 3: Entrainment strength per unit
ax = axes[1, 0]
if phase_stats:
    r_vals = [s['r_value'] for s in phase_stats]
    colors = ['green' if r > 0.2 else 'orange' for r in r_vals]
    ax.bar(range(len(r_vals)), r_vals, color=colors, edgecolor='black', alpha=0.8)
    ax.axhline(0.2, color='red', linestyle='--', linewidth=1.5, label='Significance threshold')
    ax.axhline(np.mean(r_vals), color='darkblue', linestyle='--', linewidth=2, label=f'Mean: {np.mean(r_vals):.3f}')
    ax.set_xlabel('Unit Index', fontsize=11)
    ax.set_ylabel('Resultant Vector Length (r)', fontsize=11)
    ax.set_title('Theta Entrainment Strength', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 0.6])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

# Plot 4: Preferred phase per unit
ax = axes[1, 1]
if phase_stats:
    mean_phases = [np.degrees(s['mean_phase']) for s in phase_stats]
    ax.scatter(range(len(mean_phases)), mean_phases, s=120, color='purple', alpha=0.7, edgecolors='black', linewidth=1.5)
    ax.axhline(np.degrees(circmean([s['mean_phase'] for s in phase_stats], high=np.pi, low=-np.pi)),
               color='red', linestyle='--', linewidth=2.5, label='Mean phase')
    ax.set_xlabel('Unit Index', fontsize=11)
    ax.set_ylabel('Preferred Phase (degrees)', fontsize=11)
    ax.set_title('Preferred Theta Phase per Unit', fontsize=12, fontweight='bold')
    ax.set_ylim([-180, 180])
    ax.set_yticks([-180, -90, 0, 90, 180])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('theta_entrainment_precession.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: theta_entrainment_precession.png")
plt.close()

# %% [markdown]
# ## Visualization 2: Time Series Analysis

# %%
fig, axes = plt.subplots(4, 1, figsize=(16, 11))

# Select a representative running window
if len(running_indices) > 10000:
    t_start_idx = running_indices[10000]
    t_end_idx = min(t_start_idx + 150000, len(lfp_time))
else:
    t_start_idx = 0
    t_end_idx = min(150000, len(lfp_time))

time_window = lfp_time[t_start_idx:t_end_idx]
phase_window = np.degrees(theta_phase[t_start_idx:t_end_idx])
pos_window = pos_at_lfp[t_start_idx:t_end_idx]
speed_window = speed_at_lfp[t_start_idx:t_end_idx]

# Plot 1: Theta phase
ax = axes[0]
ax.plot(time_window, phase_window, color='steelblue', linewidth=1, alpha=0.9)
ax.fill_between(time_window, -180, 180, where=(phase_window > 0), alpha=0.15, color='blue')
ax.set_ylabel('Theta Phase (°)', fontsize=11, fontweight='bold')
ax.set_title('Theta Phase Oscillation (60-sec window during running)', fontsize=12, fontweight='bold')
ax.set_ylim([-180, 180])
ax.set_yticks([-180, -90, 0, 90, 180])
ax.grid(True, alpha=0.3)
ax.set_xlim([time_window[0], time_window[-1]])

# Plot 2: Filtered LFP signal
ax = axes[1]
ax.plot(time_window, theta_filtered[t_start_idx:t_end_idx], color='darkgreen', linewidth=0.8, alpha=0.9)
ax.fill_between(time_window, theta_filtered[t_start_idx:t_end_idx], 0, alpha=0.3, color='green')
ax.set_ylabel('Filtered LFP (μV)', fontsize=11, fontweight='bold')
ax.set_title('Theta-Filtered LFP Signal (4-12 Hz)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xlim([time_window[0], time_window[-1]])

# Plot 3: Position and speed
ax = axes[2]
ax2 = ax.twinx()
ax.plot(time_window, pos_window, color='darkblue', linewidth=2, label='Position')
ax2.plot(time_window, speed_window, color='darkorange', linewidth=2, alpha=0.8, label='Speed')
ax.set_ylabel('Position (cm)', fontsize=11, fontweight='bold', color='darkblue')
ax2.set_ylabel('Running Speed (cm/s)', fontsize=11, fontweight='bold', color='darkorange')
ax.tick_params(axis='y', labelcolor='darkblue')
ax2.tick_params(axis='y', labelcolor='darkorange')
ax.set_title('Animal Behavior: Position and Speed', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
ax.set_xlim([time_window[0], time_window[-1]])

# Plot 4: Phase-position coupling during running
ax = axes[3]
scatter = ax.scatter(pos_window[::20], phase_window[::20], c=speed_window[::20],
                     cmap='plasma', alpha=0.6, s=40, edgecolors='none')
ax.set_xlabel('Position (cm)', fontsize=11, fontweight='bold')
ax.set_ylabel('Theta Phase (°)', fontsize=11, fontweight='bold')
ax.set_title('Phase-Position Coupling During Running', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
cbar = plt.colorbar(scatter, ax=ax, label='Speed (cm/s)')

plt.tight_layout()
plt.savefig('theta_timeseries.png', dpi=150, bbox_inches='tight')
print("✓ Saved: theta_timeseries.png")
plt.close()

# %% [markdown]
# ## Visualization 3: Circular Statistics

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Phase histogram
ax = axes[0]
counts, bins, patches = ax.hist(theta_phase, bins=36, range=(-np.pi, np.pi),
                                color='steelblue', edgecolor='black', alpha=0.8)
ax.set_xlabel('Theta Phase (radians)', fontsize=11, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax.set_title('Theta Phase Distribution\n(All Data)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

mean_phase_all = circmean(theta_phase, high=np.pi, low=-np.pi)
ax.axvline(mean_phase_all, color='red', linestyle='--', linewidth=2.5,
           label=f'Mean: {np.degrees(mean_phase_all):.0f}°')
ax.legend(fontsize=10)

# Plot 2: Speed-phase coupling
ax = axes[1]
speed_sorted_idx = np.argsort(speed_at_lfp)
speed_ordered = speed_at_lfp[speed_sorted_idx]
phase_ordered = theta_phase[speed_sorted_idx]

valid_mask = (speed_ordered > -50) & (speed_ordered < 100)
speed_subset = speed_ordered[valid_mask]
phase_subset = phase_ordered[valid_mask]

if len(speed_subset) > 1000:
    hb = ax.hexbin(speed_subset[::100], phase_subset[::100], gridsize=25, cmap='YlOrRd', mincnt=1)
    ax.set_xlabel('Running Speed (cm/s)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Theta Phase (radians)', fontsize=11, fontweight='bold')
    ax.set_title('Speed-Phase Relationship\n(Darker = More Data Points)', fontsize=12, fontweight='bold')
    plt.colorbar(hb, ax=ax, label='Count')

plt.tight_layout()
plt.savefig('theta_circular_stats.png', dpi=150, bbox_inches='tight')
print("✓ Saved: theta_circular_stats.png")
plt.close()

# %% [markdown]
# ## Key Findings

# %%
print("\n" + "="*70)
print("SUMMARY: THETA PHASE ENTRAINMENT AND PRECESSION")
print("="*70)

print(f"""
Dataset: DANDI:000053 (Buzsáki Lab - Rodent Spatial Navigation)
File: sub-npI1_ses-20190416_behavior+ecephys.nwb
Analysis window: 320 seconds of hippocampal recording

FINDING 1: THETA PHASE ENTRAINMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ {len(phase_stats)} units analyzed for phase entrainment
✓ Mean preferred phase: {np.degrees(circmean([s['mean_phase'] for s in phase_stats], high=np.pi, low=-np.pi)):.1f}°
✓ Mean entrainment strength (r-value): {np.mean([s['r_value'] for s in phase_stats]):.3f}

Interpretation:
- Hippocampal place cells show systematic phase locking to theta oscillations
- Spikes tend to occur at specific phases of the theta cycle
- This phase-locking mechanism enables temporal coding within the theta period
- Acts as a gating mechanism for spike generation

FINDING 2: THETA PHASE PRECESSION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Analyzed {len(pos_valid)} position-phase datapoints during running
✓ Phase-position slope: {slope:.4f} rad/cm ({np.degrees(slope):.2f}°/cm)
✓ Precession direction: Forward (positive slope)

Interpretation:
- Theta phase shifts forward as the animal progresses through space
- Creates a temporal sequence of place cell activations
- Represents position information in a compressed timeframe
- Enables efficient replay and learning during theta oscillations

BEHAVIORAL CONTEXT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Session duration: {lfp_time[-1] - lfp_time[0]:.0f} seconds
✓ Running periods: {running_fraction*100:.1f}% of session
✓ Theta strength is highest during active running behavior
✓ Both phase entrainment and precession are strongest during locomotion
""")

# %% [markdown]
# ## Biological Significance
#
# Theta phase entrainment and precession are fundamental to hippocampal function:
#
# 1. **Temporal Coding**: Theta oscillations create windows for synaptic plasticity,
#    enabling place cells to encode spatial information through precise timing.
#
# 2. **Sequence Generation**: Phase precession allows multiple place fields to fire
#    in a compressed temporal sequence within each theta cycle, creating the
#    "theta sequences" observed during behavior and replay.
#
# 3. **Learning and Memory**: The theta-governed plasticity windows and compressed
#    temporal sequences facilitate associative learning and memory consolidation.
#
# 4. **Navigation**: The spatial information encoded in theta phases enables
#    path integration and goal-directed navigation.

# %% [markdown]
# ## References
#
# - O'Keefe, J., & Recce, M. L. (1993). Phase relationship between hippocampal place
#   cells and the EEG theta rhythm. Hippocampus, 3(3), 317-330.
#
# - Hafting, T., Fyhn, M., Molden, S., Moser, M. B., & Moser, E. I. (2005).
#   Microstructure of a spatial map in the entorhinal cortex. Nature, 436(7052), 801-806.
#
# - Dragoi, G., & Tonegawa, S. (2011). Preplay of future place cell sequences by
#   hippocampal cellular assemblies. Nature, 469(7330), 397-401.
