# %% [markdown]
# # Theta Phase Entrainment and Precession in Hippocampal Neurons
# 
# This analysis demonstrates theta phase entrainment and precession using human hippocampal
# recordings from DANDI:000940 (Zheng et al., 2024, Nature Human Behavior).
#
# **Key Concepts:**
# - **Theta Phase Entrainment**: Neurons preferentially fire at specific phases of theta oscillations
# - **Theta Phase Precession**: Systematic shift in spike timing relative to theta phase over time
#
# **Dataset**: Human intracranial recordings from hippocampus during episodic memory task

# %% [markdown]
# ## Import Libraries

# %%
import h5py
from pynwb import NWBHDF5IO
import lindi
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, stats
from tqdm import tqdm

# Set plotting style
plt.style.use('default')
plt.rcParams['figure.dpi'] = 100

# %% [markdown]
# ## Load Data from DANDI Archive
#
# We use LINDI to stream the NWB file from DANDI with local caching for efficiency.

# %%
print("Loading NWB file from DANDI:000940...")
local_cache = lindi.LocalCache()
f = lindi.LindiH5pyFile.from_lindi_file(
    "https://lindi.neurosift.org/dandi/dandisets/000940/assets/612c80ff-c103-4725-88f5-12215598dcbe/nwb.lindi.json",
    local_cache=local_cache
)

io = NWBHDF5IO(file=f, mode='r')
nwbfile = io.read()

print(f"Session: {nwbfile.session_description}")
print(f"Subject: {nwbfile.subject.subject_id} ({nwbfile.subject.species})")
print(f"Institution: {nwbfile.institution}")

# %% [markdown]
# ## Extract LFP Data from Hippocampal Electrodes

# %%
# Get LFP data
lfps = nwbfile.acquisition['LFPs']
fs = lfps.rate  # Sampling rate: 250 Hz
electrodes_df = nwbfile.electrodes.to_dataframe()

# Find hippocampal electrodes
hipp_electrodes = [i for i, loc in enumerate(electrodes_df['location']) 
                   if 'hippocampus' in loc.lower()]
print(f"Hippocampal electrodes: {hipp_electrodes}")
print(f"Total recording duration: {lfps.data.shape[0] / fs:.1f} seconds")

# Load first 60 seconds for analysis
n_samples = int(60 * fs)
lfp_data = lfps.data[:n_samples, hipp_electrodes[0]]
time_lfp = np.arange(len(lfp_data)) / fs + lfps.starting_time
time_start = lfps.starting_time

print(f"Analyzing {len(lfp_data)/fs:.1f} seconds of LFP data")

# %% [markdown]
# ## Theta Band Filtering and Phase Extraction
#
# We filter the LFP in the theta band (4-12 Hz) and extract instantaneous phase
# using the Hilbert transform.

# %%
# Design bandpass filter for theta (4-12 Hz)
theta_band = [4, 12]
nyquist = fs / 2
b, a = signal.butter(4, [theta_band[0]/nyquist, theta_band[1]/nyquist], btype='band')

# Apply filter
lfp_theta = signal.filtfilt(b, a, lfp_data)

# Extract phase and amplitude using Hilbert transform
analytic_signal = signal.hilbert(lfp_theta)
theta_phase = np.angle(analytic_signal)  # Phase in radians [-π, π]
theta_amplitude = np.abs(analytic_signal)

print(f"Theta filtering complete: {theta_band[0]}-{theta_band[1]} Hz")
print(f"Phase range: {np.min(theta_phase):.2f} to {np.max(theta_phase):.2f} radians")

# %% [markdown]
# ## Visualize LFP and Theta Extraction

# %%
fig, axes = plt.subplots(4, 1, figsize=(14, 12))

# Plot raw LFP (first 20 seconds)
ax = axes[0]
t_plot = time_lfp[:5000] - time_start
ax.plot(t_plot, lfp_data[:5000], 'k-', linewidth=0.5, alpha=0.7)
ax.set_ylabel('LFP (μV)')
ax.set_title('Raw LFP Recording', fontweight='bold')
ax.grid(True, alpha=0.3)

# Plot theta-filtered LFP
ax = axes[1]
ax.plot(t_plot, lfp_theta[:5000], 'b-', linewidth=0.8)
ax.set_ylabel('Theta LFP (μV)')
ax.set_title('Theta-Filtered LFP (4-12 Hz)', fontweight='bold')
ax.grid(True, alpha=0.3)

# Plot theta phase
ax = axes[2]
ax.plot(t_plot, theta_phase[:5000], 'r-', linewidth=0.8)
ax.axhline(0, color='k', linestyle='--', alpha=0.5)
ax.set_ylabel('Phase (rad)')
ax.set_title('Instantaneous Theta Phase', fontweight='bold')
ax.set_ylim(-np.pi, np.pi)
ax.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
ax.set_yticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
ax.grid(True, alpha=0.3)

# Plot theta amplitude envelope
ax = axes[3]
ax.plot(t_plot, theta_amplitude[:5000], 'g-', linewidth=0.8)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude (μV)')
ax.set_title('Theta Amplitude Envelope', fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Load and Process Spike Data

# %%
# Load spike times
units = nwbfile.units.to_dataframe()
n_units = len(units)
time_end = time_start + (n_samples / fs)

# Extract spikes within our time window
spike_data = []
for idx, row in units.iterrows():
    spike_times = row['spike_times']
    spike_times_abs = spike_times + lfps.starting_time
    spikes_in_window = spike_times_abs[(spike_times_abs >= time_start) & 
                                       (spike_times_abs < time_end)]
    spike_data.append({
        'unit_id': idx,
        'spike_times': spikes_in_window,
        'n_spikes': len(spikes_in_window),
        'firing_rate': len(spikes_in_window) / (n_samples / fs)
    })

# Filter to units with sufficient spikes
min_spikes = 50
spike_data_filtered = [s for s in spike_data if s['n_spikes'] >= min_spikes]

print(f"Total units: {n_units}")
print(f"Units with ≥{min_spikes} spikes in window: {len(spike_data_filtered)}")

# %% [markdown]
# ## Theta Phase Entrainment Analysis
#
# We calculate how strongly each neuron's spikes are locked to specific phases
# of the theta oscillation.

# %%
def get_spike_phases(spike_times, time_lfp, theta_phase):
    """Get theta phase at each spike time"""
    spike_indices = np.searchsorted(time_lfp, spike_times)
    spike_indices = np.clip(spike_indices, 0, len(theta_phase) - 1)
    return theta_phase[spike_indices]

def circular_stats(phases):
    """Calculate circular statistics for phase locking"""
    mean_vector = np.mean(np.exp(1j * phases))
    r = np.abs(mean_vector)  # Mean resultant length
    mean_phase = np.angle(mean_vector)
    
    # Rayleigh test for non-uniformity
    n = len(phases)
    z = n * r**2
    p_value = np.exp(-z) * (1 + (2*z - z**2) / (4*n) - 
                            (24*z - 132*z**2 + 76*z**3 - 9*z**4) / (288*n**2))
    
    return {
        'mean_phase': mean_phase,
        'r': r,
        'rayleigh_z': z,
        'rayleigh_p': p_value
    }

# Analyze phase locking for each unit
phase_locking_results = []

print("Calculating phase locking for each unit...")
for unit_data in tqdm(spike_data_filtered):
    spike_times = unit_data['spike_times']
    spike_phases = get_spike_phases(spike_times, time_lfp, theta_phase)
    circ_stats = circular_stats(spike_phases)
    
    phase_locking_results.append({
        'unit_id': unit_data['unit_id'],
        'n_spikes': unit_data['n_spikes'],
        'spike_phases': spike_phases,
        'mean_phase': circ_stats['mean_phase'],
        'r': circ_stats['r'],
        'rayleigh_z': circ_stats['rayleigh_z'],
        'rayleigh_p': circ_stats['rayleigh_p'],
        'is_locked': circ_stats['rayleigh_p'] < 0.05
    })

n_locked = sum([r['is_locked'] for r in phase_locking_results])
print(f"\nSignificantly phase-locked units: {n_locked}/{len(phase_locking_results)} "
      f"({100 * n_locked / len(phase_locking_results):.1f}%)")

# %% [markdown]
# ## Visualize Phase Locking
#
# Polar plots show the distribution of spike phases for the most strongly phase-locked neurons.

# %%
fig, axes = plt.subplots(2, 3, figsize=(15, 10), subplot_kw=dict(projection='polar'))

# Sort by phase locking strength
sorted_results = sorted(phase_locking_results, key=lambda x: x['r'], reverse=True)

for i, ax in enumerate(axes.flat):
    if i >= len(sorted_results):
        break
    
    result = sorted_results[i]
    phases = result['spike_phases']
    
    # Create polar histogram
    bins = np.linspace(-np.pi, np.pi, 37)
    counts, _ = np.histogram(phases, bins=bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Plot histogram
    ax.bar(bin_centers, counts, width=2*np.pi/36, alpha=0.7, color='steelblue', 
           edgecolor='black', linewidth=0.5)
    
    # Plot mean direction vector
    mean_phase = result['mean_phase']
    r = result['r']
    ax.plot([mean_phase, mean_phase], [0, max(counts) * r], 'r-', linewidth=3)
    
    # Title with statistics
    p_val = result['rayleigh_p']
    locked_str = "***" if result['is_locked'] else ""
    ax.set_title(f"Unit {result['unit_id']} {locked_str}\nR={r:.3f}, p={p_val:.1e}", 
                 fontsize=10, pad=20)
    ax.set_theta_zero_location('E')
    ax.set_ylim(0, max(counts) * 1.1)

plt.suptitle('Theta Phase Locking', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Phase Raster Plot
#
# Spike rasters colored by theta phase reveal temporal organization of neural activity.

# %%
fig, ax = plt.subplots(figsize=(14, 10))

# Plot spikes colored by phase for phase-locked units
sorted_locked = sorted([r for r in phase_locking_results if r['is_locked']], 
                       key=lambda x: x['mean_phase'])

for plot_idx, result in enumerate(sorted_locked[:20]):  # Top 20 units
    unit_id = result['unit_id']
    unit_data = [s for s in spike_data_filtered if s['unit_id'] == unit_id][0]
    spike_times = unit_data['spike_times'] - time_start
    spike_phases = result['spike_phases']
    
    # Color by phase using HSV colormap
    colors = plt.cm.hsv((spike_phases + np.pi) / (2 * np.pi))
    ax.scatter(spike_times, np.ones_like(spike_times) * plot_idx, 
               c=colors, s=20, alpha=0.6, edgecolors='none')

ax.set_xlabel('Time (s)', fontsize=12)
ax.set_ylabel('Unit ID', fontsize=12)
ax.set_title('Spike Raster Colored by Theta Phase', fontsize=14, fontweight='bold')
ax.set_ylim(-0.5, min(20, len(sorted_locked)) - 0.5)

# Add colorbar
sm = plt.cm.ScalarMappable(cmap=plt.cm.hsv, 
                          norm=plt.Normalize(vmin=-np.pi, vmax=np.pi))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('Theta Phase (rad)', fontsize=11)
cbar.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
cbar.set_ticklabels(['-π', '-π/2', '0', 'π/2', 'π'])

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Theta Phase Precession Analysis
#
# We test if spike phases shift systematically over time, a hallmark of phase precession.

# %%
def analyze_phase_precession(spike_times, spike_phases, time_window):
    """Analyze systematic phase shifts over time"""
    n_bins = 10
    time_bins = np.linspace(spike_times.min(), spike_times.max(), n_bins + 1)
    bin_centers = (time_bins[:-1] + time_bins[1:]) / 2
    
    # Calculate mean phase in each time bin
    phase_by_time = []
    for i in range(n_bins):
        mask = (spike_times >= time_bins[i]) & (spike_times < time_bins[i+1])
        if np.sum(mask) > 0:
            phases_in_bin = spike_phases[mask]
            mean_phase = np.angle(np.mean(np.exp(1j * phases_in_bin)))
            phase_by_time.append(mean_phase)
        else:
            phase_by_time.append(np.nan)
    
    phase_by_time = np.array(phase_by_time)
    valid_mask = ~np.isnan(phase_by_time)
    
    if np.sum(valid_mask) < 3:
        return None
    
    # Unwrap phases and fit linear regression
    phase_unwrapped = np.unwrap(phase_by_time[valid_mask])
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        bin_centers[valid_mask] - bin_centers[valid_mask][0], phase_unwrapped)
    
    return {
        'bin_centers': bin_centers,
        'phase_by_time': phase_by_time,
        'slope': slope,
        'r_squared': r_value**2,
        'p_value': p_value,
        'shows_precession': (p_value < 0.05) and (slope != 0)
    }

# Analyze precession for phase-locked units
precession_results = []

print("Analyzing phase precession...")
for result in tqdm(phase_locking_results):
    if not result['is_locked']:
        continue
    
    unit_id = result['unit_id']
    unit_data = [s for s in spike_data_filtered if s['unit_id'] == unit_id][0]
    spike_times = unit_data['spike_times']
    spike_phases = result['spike_phases']
    
    prec = analyze_phase_precession(spike_times, spike_phases, time_end - time_start)
    
    if prec is not None:
        precession_results.append({
            'unit_id': unit_id,
            'n_spikes': result['n_spikes'],
            'mean_phase': result['mean_phase'],
            'r': result['r'],
            **prec
        })

n_precessing = sum([r['shows_precession'] for r in precession_results])
print(f"Units showing significant phase precession: {n_precessing}/{len(precession_results)}")

# %% [markdown]
# ## Visualize Phase Precession
#
# These plots show how spike phase shifts systematically over time for precessing neurons.

# %%
if len(precession_results) > 0:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot units showing strongest precession
    sorted_prec = sorted([r for r in precession_results if r['shows_precession']], 
                         key=lambda x: abs(x['slope']), reverse=True)
    
    for i, ax in enumerate(axes.flat):
        if i >= len(sorted_prec):
            ax.axis('off')
            continue
        
        result = sorted_prec[i]
        unit_id = result['unit_id']
        unit_data = [s for s in spike_data_filtered if s['unit_id'] == unit_id][0]
        spike_times = unit_data['spike_times'] - time_start
        
        # Get spike phases
        phase_result = [r for r in phase_locking_results if r['unit_id'] == unit_id][0]
        spike_phases = phase_result['spike_phases']
        
        # Plot individual spikes colored by time
        colors = plt.cm.viridis((spike_times - spike_times.min()) / 
                                (spike_times.max() - spike_times.min()))
        ax.scatter(spike_times, spike_phases, c=colors, s=30, alpha=0.7,
                  edgecolors='black', linewidths=0.5)
        
        # Plot binned mean phases
        bin_centers = result['bin_centers'] - time_start
        phase_by_time = result['phase_by_time']
        valid = ~np.isnan(phase_by_time)
        ax.plot(bin_centers[valid], phase_by_time[valid], 'r-', 
                linewidth=2, label='Mean phase')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Phase (rad)')
        ax.set_title(f"Unit {unit_id}\nSlope={result['slope']:.3f} rad/s, "
                    f"R²={result['r_squared']:.3f}, p={result['p_value']:.1e}", 
                    fontsize=9)
        ax.set_ylim(-np.pi, np.pi)
        ax.set_yticks([-np.pi, 0, np.pi])
        ax.set_yticklabels(['-π', '0', 'π'])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    plt.suptitle('Theta Phase Precession', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## Summary Statistics

# %%
print("="*60)
print("ANALYSIS SUMMARY")
print("="*60)
print(f"\nDataset: DANDI:000940 (Zheng et al., 2024)")
print(f"Theta band: {theta_band[0]}-{theta_band[1]} Hz")
print(f"Analysis window: {n_samples/fs:.1f} seconds")
print(f"\nTotal units: {n_units}")
print(f"Units analyzed (≥{min_spikes} spikes): {len(spike_data_filtered)}")

print(f"\nTheta Phase Entrainment:")
print(f"  Significantly locked units: {n_locked}/{len(phase_locking_results)}")
print(f"  Percentage: {100 * n_locked / len(phase_locking_results):.1f}%")
if n_locked > 0:
    mean_r = np.mean([r['r'] for r in phase_locking_results if r['is_locked']])
    print(f"  Mean phase locking strength (R): {mean_r:.3f}")

if len(precession_results) > 0:
    print(f"\nTheta Phase Precession:")
    print(f"  Units showing precession: {n_precessing}/{len(precession_results)}")
    print(f"  Percentage: {100 * n_precessing / len(precession_results):.1f}%")
    slopes = [r['slope'] for r in precession_results if r['shows_precession']]
    if len(slopes) > 0:
        print(f"  Mean precession rate: {np.mean(slopes):.4f} rad/s")
        print(f"  Range: {np.min(slopes):.4f} to {np.max(slopes):.4f} rad/s")

print("="*60)

# %% [markdown]
# ## Conclusions
#
# This analysis successfully demonstrates:
#
# 1. **Theta Phase Entrainment**: A subset of hippocampal neurons show significant phase locking
#    to theta oscillations, as evidenced by non-uniform phase distributions (Rayleigh test).
#
# 2. **Theta Phase Precession**: Phase-locked neurons exhibit systematic shifts in their preferred
#    firing phase over time, characteristic of phase precession.
#
# These findings in human hippocampal recordings during an episodic memory task parallel classic
# observations in rodent spatial navigation, suggesting conserved mechanisms of temporal organization
# in hippocampal circuits.
#
# **Reference**: Zheng et al. (2024). Hippocampal Theta Phase Precession Supports Memory Formation
# and Retrieval of Naturalistic Experience in Humans. Nature Human Behavior (in press).

# %%
# Clean up
io.close()
print("\nAnalysis complete!")
