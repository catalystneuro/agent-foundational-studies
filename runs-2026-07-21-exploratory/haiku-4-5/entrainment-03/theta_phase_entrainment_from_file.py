# %% [markdown]
# # Theta Phase Entrainment of Hippocampal Neurons
#
# This analysis demonstrates phase entrainment of hippocampal principal cells to the theta
# oscillation (4-12 Hz), a dominant rhythm during spatial navigation. We quantify how
# individual neuron spike times phase-lock to the ongoing theta oscillation by computing
# the mean resultant vector length (MVRL) and plotting spike phase distributions.
#
# Data source: DANDI 000003 - Hippocampal recordings from Senzai & Buzsaki (Neuron 2017)
# during theta maze exploration. The dataset contains extracellular electrophysiology
# from hippocampal CA1/CA3 and dentate gyrus.

# %% [markdown]
# ## Setup and Dependencies

# %%
import numpy as np
import h5py
from pynwb import NWBHDF5IO
import pynapple as nap
from scipy import signal, stats
from scipy.signal import hilbert, filtfilt, butter
from scipy.stats import circmean, circstd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import warnings
import os
import glob
warnings.filterwarnings('ignore')

# Configure plotting backend for headless operation
import matplotlib
matplotlib.use('Agg')

print("Dependencies loaded successfully")

# %% [markdown]
# ## Find Downloaded NWB File

# %%
# Look for downloaded NWB files
possible_paths = [
    '/var/folders/67/qdwczmzx315gj1xp7hp1f11r0000gn/T/dandi_nwb_*/session.nwb',
    './session.nwb',
    '/tmp/dandi_nwb_*/session.nwb',
]

nwb_file = None
for pattern in possible_paths:
    matches = glob.glob(pattern)
    if matches:
        nwb_file = matches[0]
        break

if nwb_file is None:
    # Try a broader search
    import tempfile
    temp_root = tempfile.gettempdir()
    for root, dirs, files in os.walk(temp_root):
        for file in files:
            if file.endswith('.nwb') and 'dandi' in root.lower():
                nwb_file = os.path.join(root, file)
                break
        if nwb_file:
            break

if nwb_file:
    print(f"Found NWB file: {nwb_file}")
    print(f"File size: {os.path.getsize(nwb_file) / 1e9:.2f} GB")
else:
    print("ERROR: No NWB file found. Download may still be in progress.")
    print("Please wait and try again.")
    import sys
    sys.exit(1)

# %% [markdown]
# ## Load NWB File

# %%
print("Loading NWB file (this may take 1-2 minutes)...")
print(f"File path: {nwb_file}")
print(f"File size: {os.path.getsize(nwb_file) / 1e9:.2f} GB")

try:
    io = NWBHDF5IO(nwb_file, 'r', load_namespaces=False)
    nwbfile = io.read()
except Exception as e:
    print(f"Error loading file: {e}")
    print("File may still be downloading. Waiting...")
    import time
    time.sleep(30)
    io = NWBHDF5IO(nwb_file, 'r', load_namespaces=False)
    nwbfile = io.read()

# Wrap with Pynapple
nwb = nap.NWBFile(nwbfile)
print(f"NWB file loaded successfully")
print(f"Session ID: {nwb.identifier}")

# %% [markdown]
# ## Inspect Data Contents

# %%
print("\n" + "="*60)
print("NWB FILE CONTENTS")
print("="*60)

# Units (neural spikes)
print(f"\nUnits: {len(nwb.units)} neurons recorded")
if len(nwb.units) > 0:
    print(f"  Spike times available: {nwb.units['spike_times'].values is not None}")
    print(f"  Sample unit: {list(nwb.units.index)[0]}")

# LFP data
print(f"\nLFP data streams:")
for key in nwb.lfp.keys():
    lfp_series = nwb.lfp[key]
    print(f"  {key}:")
    print(f"    Shape: {lfp_series.values.shape}")
    print(f"    Sampling rate: {lfp_series.rate:.1f} Hz")
    print(f"    Duration: {lfp_series.values.shape[0] / lfp_series.rate:.1f} s")

# Get LFP reference channel
lfp_keys = list(nwb.lfp.keys())
selected_lfp_key = lfp_keys[0]
lfp_series = nwb.lfp[selected_lfp_key]
print(f"\nUsing LFP: {selected_lfp_key}")

# %% [markdown]
# ## Extract Theta Oscillation
#
# We filter the LFP to the theta band (4-12 Hz) and compute the analytic signal
# using the Hilbert transform to extract instantaneous phase.

# %%
# LFP parameters
lfp_data = lfp_series.values.astype(np.float32)
lfp_timestamps = lfp_series.timestamps[:]
lfp_sr = lfp_series.rate

print(f"LFP shape: {lfp_data.shape}")
print(f"LFP sampling rate: {lfp_sr} Hz")
print(f"LFP duration: {(lfp_timestamps[-1] - lfp_timestamps[0]):.1f} s")

# If LFP is multi-channel, use first channel or average
if lfp_data.ndim > 1:
    print(f"LFP has {lfp_data.shape[1]} channels, using first channel...")
    lfp_data = lfp_data[:, 0]

print(f"LFP signal shape after preprocessing: {lfp_data.shape}")

# Design theta band filter (4-12 Hz)
theta_band = (4, 12)
sos = butter(4, theta_band, btype='band', fs=lfp_sr, output='sos')

# Apply filter with padding to avoid edge artifacts
pad_samples = int(lfp_sr * 5)  # 5 second padding
lfp_padded = np.pad(lfp_data, pad_samples, mode='reflect')
theta_filtered = filtfilt(sos, lfp_padded, axis=0)[pad_samples:-pad_samples]

# Extract instantaneous phase using Hilbert transform
print("Computing theta phase using Hilbert transform...")
theta_analytic = hilbert(theta_filtered)
theta_phase = np.angle(theta_analytic)
theta_power = np.abs(theta_analytic)**2

print(f"Theta phase shape: {theta_phase.shape}")
print(f"Theta phase range: [{theta_phase.min():.2f}, {theta_phase.max():.2f}]")

# %% [markdown]
# ## Extract Unit Spike Times and Assign Theta Phase

# %%
# Get spike times for each unit
units = nwb.units
print(f"\nExtracting spike times from {len(units)} units...")

unit_spike_phases = {}
unit_names = []

for unit_id in tqdm(units.index, desc="Processing units"):
    spike_times = units[unit_id]['spike_times'].values

    if len(spike_times) < 10:  # Skip units with too few spikes
        continue

    # Find the closest LFP timestamp for each spike
    spike_phase_list = []
    for spike_t in spike_times:
        # Find nearest LFP sample
        idx = np.searchsorted(lfp_timestamps, spike_t)
        if 0 <= idx < len(theta_phase):
            spike_phase_list.append(theta_phase[idx])

    if len(spike_phase_list) > 10:  # Keep units with enough phase values
        unit_spike_phases[unit_id] = np.array(spike_phase_list)
        unit_names.append(unit_id)

print(f"Analyzed {len(unit_spike_phases)} units with sufficient spikes")

# %% [markdown]
# ## Compute Phase Entrainment Metrics
#
# For each neuron, we compute the mean resultant vector length (MVRL), which
# quantifies the degree of phase locking. MVRL ranges from 0 (uniform distribution)
# to 1 (perfect phase locking).

# %%
def compute_mvrl(phases):
    """
    Compute mean resultant vector length - a measure of phase concentration.
    MVRL = 1 indicates perfect phase locking; MVRL = 0 indicates random distribution.
    """
    sin_sum = np.sum(np.sin(phases))
    cos_sum = np.sum(np.cos(phases))
    mvrl = np.sqrt(sin_sum**2 + cos_sum**2) / len(phases)
    return mvrl

def compute_preferred_phase(phases):
    """Compute the preferred phase and its concentration."""
    sin_sum = np.sum(np.sin(phases))
    cos_sum = np.sum(np.cos(phases))
    preferred_phase = np.arctan2(sin_sum, cos_sum)
    return preferred_phase

# Compute entrainment metrics for each unit
mvrl_values = []
preferred_phases = []
spike_counts = []

for unit_id in tqdm(unit_names, desc="Computing entrainment metrics"):
    phases = unit_spike_phases[unit_id]
    mvrl = compute_mvrl(phases)
    pref_phase = compute_preferred_phase(phases)

    mvrl_values.append(mvrl)
    preferred_phases.append(pref_phase)
    spike_counts.append(len(phases))

mvrl_values = np.array(mvrl_values)
preferred_phases = np.array(preferred_phases)
spike_counts = np.array(spike_counts)

print(f"\nPhase Entrainment Summary:")
print(f"  Mean MVRL: {mvrl_values.mean():.3f} ± {mvrl_values.std():.3f}")
print(f"  MVRL range: [{mvrl_values.min():.3f}, {mvrl_values.max():.3f}]")
print(f"  Mean spike count per unit: {spike_counts.mean():.0f} ± {spike_counts.std():.0f}")

# Test significance: is MVRL significantly different from 0?
# Using Rayleigh test for circular uniformity
p_values = []
for phases in tqdm(unit_spike_phases.values(), desc="Computing significance"):
    # Rayleigh test statistic
    r = compute_mvrl(phases)
    n = len(phases)
    z = n * r**2
    # Approximation for p-value (Rayleigh test)
    p = np.exp(-z) * (1 + (2*z - z**2) / (4*n) - (24*z - 132*z**2 + 76*z**3 - 9*z**4) / (288*n**2))
    p_values.append(p)

p_values = np.array(p_values)
significant_entrained = (p_values < 0.05).sum()

print(f"\nStatistical Significance:")
print(f"  Units with significant entrainment (p<0.05): {significant_entrained}/{len(p_values)} ({100*significant_entrained/len(p_values):.1f}%)")
print(f"  Mean p-value: {p_values.mean():.4f}")

# %% [markdown]
# ## Visualizations

# %%
fig = plt.figure(figsize=(14, 10))

# 1. Theta oscillation and power over time (5 minute window)
ax1 = plt.subplot(3, 3, 1)
time_window = 300  # 5 minutes
window_samples = int(min(time_window * lfp_sr, len(theta_filtered)))
time_axis = lfp_timestamps[:window_samples]

ax1.plot(time_axis, theta_filtered[:window_samples], 'b-', linewidth=0.5, alpha=0.7)
ax1.set_xlabel('Time (s)', fontsize=10)
ax1.set_ylabel('Theta LFP (μV)', fontsize=10)
ax1.set_title('Theta-Filtered LFP (4-12 Hz)', fontsize=11, fontweight='bold')
ax1.grid(True, alpha=0.3)

# 2. Theta power over time
ax2 = plt.subplot(3, 3, 2)
power_smooth = np.convolve(theta_power[:window_samples], np.ones(int(lfp_sr))/lfp_sr, mode='same')
ax2.plot(time_axis, power_smooth, 'r-', linewidth=1)
ax2.set_xlabel('Time (s)', fontsize=10)
ax2.set_ylabel('Theta Power (μV²)', fontsize=10)
ax2.set_title('Theta Power Over Time', fontsize=11, fontweight='bold')
ax2.grid(True, alpha=0.3)

# 3. Phase distribution histogram (all units)
ax3 = plt.subplot(3, 3, 3)
all_phases = np.concatenate(list(unit_spike_phases.values()))
ax3.hist(all_phases, bins=36, range=(-np.pi, np.pi), alpha=0.7, color='blue', edgecolor='black')
ax3.set_xlabel('Theta Phase (rad)', fontsize=10)
ax3.set_ylabel('Spike Count', fontsize=10)
ax3.set_title('Population Spike Phase Distribution', fontsize=11, fontweight='bold')
ax3.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
ax3.set_xticklabels(['-π', '-π/2', '0', 'π/2', 'π'], fontsize=9)
ax3.grid(True, alpha=0.3, axis='y')

# 4. MVRL distribution
ax4 = plt.subplot(3, 3, 4)
ax4.hist(mvrl_values, bins=20, alpha=0.7, color='green', edgecolor='black')
ax4.axvline(mvrl_values.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {mvrl_values.mean():.3f}')
ax4.set_xlabel('Mean Resultant Vector Length', fontsize=10)
ax4.set_ylabel('Number of Units', fontsize=10)
ax4.set_title('Phase Entrainment Strength Distribution', fontsize=11, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3, axis='y')

# 5. MVRL vs spike count
ax5 = plt.subplot(3, 3, 5)
scatter = ax5.scatter(spike_counts, mvrl_values, alpha=0.6, c=p_values, cmap='RdYlGn_r', s=60)
ax5.set_xlabel('Spike Count', fontsize=10)
ax5.set_ylabel('MVRL (Phase Locking)', fontsize=10)
ax5.set_title('Phase Locking vs Spike Count', fontsize=11, fontweight='bold')
ax5.grid(True, alpha=0.3)
cbar = plt.colorbar(scatter, ax=ax5)
cbar.set_label('p-value', fontsize=9)

# 6. Polar plot of preferred phases
ax6 = plt.subplot(3, 3, 6, projection='polar')
# Weighted by MVRL
weights = mvrl_values / mvrl_values.max()
for phase, weight in zip(preferred_phases, weights):
    ax6.scatter(phase, 1, s=100*weight, alpha=0.6, c='blue')
ax6.set_theta_zero_location('E')
ax6.set_theta_direction(1)
ax6.set_title('Preferred Phases (size ∝ MVRL)', fontsize=11, fontweight='bold', pad=15)

# 7-9. Example circular phase histograms for 3 units with highest MVRL
top_units_idx = np.argsort(mvrl_values)[-3:]
for plot_idx, unit_idx in enumerate(top_units_idx):
    ax = plt.subplot(3, 3, 7 + plot_idx, projection='polar')
    phases = unit_spike_phases[unit_names[unit_idx]]

    # Create circular histogram
    bin_edges = np.linspace(-np.pi, np.pi, 37)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    hist, _ = np.histogram(phases, bins=bin_edges)

    # Plot as bars in polar coordinates
    bars = ax.bar(bin_centers, hist, width=np.pi/18, alpha=0.7, color='steelblue', edgecolor='black')

    # Overlay mean vector
    mean_vector_len = mvrl_values[unit_idx] * hist.max()
    mean_phase = preferred_phases[unit_idx]
    ax.arrow(mean_phase, 0, 0, mean_vector_len, head_width=0.2, head_length=5,
             fc='red', ec='red', linewidth=2)

    ax.set_theta_zero_location('E')
    ax.set_theta_direction(1)
    ax.set_ylim(0, hist.max() * 1.1)
    ax.set_title(f'Unit {unit_names[unit_idx]}\nMVRL={mvrl_values[unit_idx]:.3f}',
                fontsize=10, fontweight='bold', pad=10)

plt.tight_layout()
plt.savefig('theta_entrainment_analysis.png', dpi=150, bbox_inches='tight')
print("\nSaved: theta_entrainment_analysis.png")
plt.close()

# %% [markdown]
# ## Extended Population Analysis

# %%
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Cumulative distribution of MVRL
ax = axes[0, 0]
sorted_mvrl = np.sort(mvrl_values)
cumsum = np.arange(1, len(sorted_mvrl) + 1) / len(sorted_mvrl)
ax.plot(sorted_mvrl, cumsum, 'b-', linewidth=2)
ax.set_xlabel('Mean Resultant Vector Length', fontsize=11)
ax.set_ylabel('Cumulative Proportion of Units', fontsize=11)
ax.set_title('Cumulative Distribution of Phase Entrainment', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# 2. Spike phase scatter with density
ax = axes[0, 1]
all_phases_wrapped = np.mod(all_phases + np.pi, 2*np.pi) - np.pi
phase_bins = np.linspace(-np.pi, np.pi, 50)
phase_hist, _ = np.histogram(all_phases_wrapped, bins=phase_bins)
phase_centers = (phase_bins[:-1] + phase_bins[1:]) / 2

colors = plt.cm.YlOrRd(phase_hist / phase_hist.max())
for i, (center, count) in enumerate(zip(phase_centers, phase_hist)):
    ax.bar(center, count, width=np.pi/25, color=colors[i], edgecolor='black', linewidth=0.5)

ax.set_xlabel('Theta Phase (rad)', fontsize=11)
ax.set_ylabel('Spike Density', fontsize=11)
ax.set_title('Population Spike Phase Density', fontsize=12, fontweight='bold')
ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
ax.set_xticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
ax.grid(True, alpha=0.3, axis='y')

# 3. p-value distribution
ax = axes[1, 0]
ax.hist(-np.log10(p_values), bins=20, alpha=0.7, color='purple', edgecolor='black')
ax.axvline(-np.log10(0.05), color='red', linestyle='--', linewidth=2, label='p=0.05 threshold')
ax.set_xlabel('-log10(p-value)', fontsize=11)
ax.set_ylabel('Number of Units', fontsize=11)
ax.set_title('Entrainment Significance (Rayleigh Test)', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

# 4. Relationship between theta power and entrainment strength
ax = axes[1, 1]
# Correlate MVRL with local theta power at spike times
theta_power_at_spikes = []
for unit_id in unit_names:
    spike_times = units[unit_id]['spike_times'].values
    power_list = []
    for spike_t in spike_times:
        idx = np.searchsorted(lfp_timestamps, spike_t)
        if 0 <= idx < len(theta_power):
            power_list.append(theta_power[idx])
    if power_list:
        theta_power_at_spikes.append(np.mean(power_list))
    else:
        theta_power_at_spikes.append(0)

theta_power_at_spikes = np.array(theta_power_at_spikes)
scatter = ax.scatter(theta_power_at_spikes, mvrl_values, alpha=0.6, s=80, c=spike_counts, cmap='viridis')
ax.set_xlabel('Mean Theta Power at Spike Times (μV²)', fontsize=11)
ax.set_ylabel('MVRL (Phase Locking Strength)', fontsize=11)
ax.set_title('Phase Locking vs Local Theta Power', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Spike Count', fontsize=10)

plt.tight_layout()
plt.savefig('theta_entrainment_population_analysis.png', dpi=150, bbox_inches='tight')
print("Saved: theta_entrainment_population_analysis.png")
plt.close()

# %% [markdown]
# ## Summary Statistics

# %%
print("\n" + "="*70)
print("THETA PHASE ENTRAINMENT ANALYSIS - SUMMARY")
print("="*70)
print(f"\nDataset: DANDI 000003 - Senzai & Buzsaki, Neuron 2017")
print(f"Neuroscience phenomenon: Theta phase entrainment")
print(f"Recording Duration: {(lfp_timestamps[-1] - lfp_timestamps[0])/60:.1f} minutes")
print(f"Sampling Rate: {lfp_sr:.0f} Hz")

print(f"\nNeural Population:")
print(f"  Units analyzed: {len(unit_spike_phases)}")
print(f"  Mean spikes per unit: {spike_counts.mean():.0f}")
print(f"  Median spikes per unit: {np.median(spike_counts):.0f}")
print(f"  Total spikes analyzed: {spike_counts.sum():.0f}")

print(f"\nTheta Oscillation (4-12 Hz):")
print(f"  Mean LFP amplitude: {np.mean(np.abs(theta_filtered)):.2f} μV")
print(f"  Peak LFP amplitude: {np.max(np.abs(theta_filtered)):.2f} μV")
print(f"  Mean theta power: {np.mean(theta_power):.2f} μV²")

print(f"\nPhase Entrainment Metrics:")
print(f"  Mean MVRL (population): {mvrl_values.mean():.4f}")
print(f"  Std MVRL: {mvrl_values.std():.4f}")
print(f"  MVRL range: [{mvrl_values.min():.4f}, {mvrl_values.max():.4f}]")
print(f"  Units with MVRL > 0.1: {(mvrl_values > 0.1).sum()} ({100*(mvrl_values > 0.1).sum()/len(mvrl_values):.1f}%)")
print(f"  Units with significant entrainment (p<0.05): {significant_entrained} ({100*significant_entrained/len(p_values):.1f}%)")

print(f"\nPhase Distribution:")
preferred_phase_deg = np.degrees(preferred_phases)
print(f"  Mean preferred phase: {np.degrees(circmean(preferred_phases)):.1f}° ± {np.degrees(circstd(preferred_phases)):.1f}°")
print(f"  Preferred phase range: [{preferred_phase_deg.min():.1f}°, {preferred_phase_deg.max():.1f}°]")

print("\n" + "="*70)
print("Analysis complete. Figures saved:")
print("  - theta_entrainment_analysis.png")
print("  - theta_entrainment_population_analysis.png")
print("="*70)

# Clean up
io.close()
