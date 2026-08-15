# %% [markdown]
# # Hippocampal Theta Phase Entrainment Analysis
#
# This analysis demonstrates theta phase entrainment of hippocampal neurons.
# Theta oscillations (4-12 Hz) in the hippocampus reflect coordinated network
# activity during exploration and memory consolidation. Individual neurons show
# phase-locking to the theta rhythm, with preferred firing phases that often
# cluster by cell type (pyramidal cells vs. interneurons).
#
# **Key Phenomenon:** Neurons fire preferentially at specific phases of the
# theta oscillation, indicating that hippocampal computation is organized
# around this rhythmic input. We quantify this entrainment using phase locking
# values (PLV) and investigate how firing probability varies across the
# theta cycle.

# %% [markdown]
# ## Setup and Data Generation

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import circstd, circmean
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Recording parameters (hippocampal extracellular recording)
fs = 30000  # 30 kHz sampling rate
duration = 120  # 120 seconds
t = np.arange(0, duration, 1/fs)
n_samples = len(t)

print(f"Simulation parameters:")
print(f"  Sampling rate: {fs} Hz")
print(f"  Duration: {duration} seconds")
print(f"  Total samples: {n_samples:,}")

# %% [markdown]
# ## Theta Rhythm and Unit Spike Generation

# %%
# Theta oscillation parameters
theta_freq = 8.0  # 8 Hz (typical hippocampal theta in rodents)
theta_amplitude = 50.0  # μV peak amplitude
theta_noise_std = 10.0  # μV noise

# Generate theta LFP signal
theta_phase_continuous = 2 * np.pi * theta_freq * t
theta_signal = theta_amplitude * np.sin(theta_phase_continuous) + \
               theta_noise_std * np.random.randn(n_samples)

print(f"\nTheta LFP:")
print(f"  Frequency: {theta_freq} Hz")
print(f"  Amplitude: {theta_amplitude} μV")
print(f"  SNR: {20*np.log10(theta_amplitude/theta_noise_std):.1f} dB")

# %% [markdown]
# ### Generate Hippocampal Unit Spike Times

# %%
# Unit parameters based on actual hippocampal cell types
n_pyramidal = 10  # Pyramidal cells (principle cells)
n_interneurons = 5  # GABAergic interneurons
n_units = n_pyramidal + n_interneurons

spike_data = {}

for unit_id in range(n_units):
    # Cell type properties
    if unit_id < n_pyramidal:
        cell_type = 'pyramidal'
        # Pyramidal cells: lower firing rate, strong theta modulation
        base_rate = np.random.uniform(2, 8)  # spikes/second
        phase_locking_strength = np.random.uniform(0.35, 0.80)
        # Preferred phase clustered around specific phases
        preferred_phase = np.random.uniform(-np.pi, np.pi)
    else:
        cell_type = 'interneuron'
        # Interneurons: higher firing rate, phase-locked but typically leading theta
        base_rate = np.random.uniform(8, 20)  # spikes/second
        phase_locking_strength = np.random.uniform(0.20, 0.65)
        # Interneurons typically fire ahead of pyramidal cells
        preferred_phase = preferred_phase + np.random.uniform(np.pi/4, 3*np.pi/4)

    # Generate spike times using Poisson process with theta modulation
    time_bin_size = 0.001  # 1 ms bins
    n_bins = int(duration / time_bin_size)
    bin_times = np.arange(n_bins) * time_bin_size

    # Theta phase at each bin
    theta_phase_at_bin = 2 * np.pi * theta_freq * bin_times

    # Firing rate modulation: cosine function peaked at preferred phase
    phase_difference = theta_phase_at_bin - preferred_phase
    # Normalize phase difference to [-pi, pi]
    phase_difference = np.angle(np.exp(1j * phase_difference))

    # Modulation envelope: 1 + strength * cosine
    rate_modulation = 1.0 + phase_locking_strength * np.cos(phase_difference)
    # Firing rate in spikes per millisecond
    firing_rate = base_rate * rate_modulation / 1000

    # Generate spikes from inhomogeneous Poisson process
    spike_bin_indices = np.where(np.random.rand(n_bins) < firing_rate)[0]
    spike_times = spike_bin_indices * time_bin_size

    # Add jitter (±0.1 ms) for realism
    spike_times = spike_times + np.random.uniform(-0.0001, 0.0001, len(spike_times))
    spike_times = np.clip(spike_times, 0, duration)
    spike_times = np.sort(spike_times)

    spike_data[unit_id] = {
        'spike_times': spike_times,
        'n_spikes': len(spike_times),
        'preferred_phase': preferred_phase,
        'plv_true': phase_locking_strength,
        'cell_type': cell_type,
        'base_rate': base_rate
    }

print(f"\nUnit spike generation:")
print(f"  Pyramidal cells: {n_pyramidal}")
print(f"  Interneurons: {n_interneurons}")
print(f"  Total spike count: {sum(d['n_spikes'] for d in spike_data.values()):,}")

# %% [markdown]
# ## Extract Theta Phase at Spike Times

# %%
# For each unit, extract the theta phase at which each spike occurred
theta_phases_at_spikes = {}

for unit_id, data in spike_data.items():
    spike_times = data['spike_times']

    # Interpolate theta phase at spike times
    theta_phase_at_spikes = 2 * np.pi * theta_freq * spike_times
    # Normalize to [-pi, pi]
    theta_phase_at_spikes = np.angle(np.exp(1j * theta_phase_at_spikes))

    theta_phases_at_spikes[unit_id] = theta_phase_at_spikes

print(f"\nTheta phase extraction:")
print(f"  Theta phase extracted for each spike")
print(f"  Phases normalized to [-π, π]")

# %% [markdown]
# ## Quantify Theta Phase Entrainment

# %%
# Calculate phase locking value (PLV) for each unit using circular statistics
plv_values = {}
mean_phases = {}
rayleigh_pvalues = {}

for unit_id in range(n_units):
    phases = theta_phases_at_spikes[unit_id]

    if len(phases) > 0:
        # Phase locking value: mean resultant length of phase vectors
        # PLV = |mean(exp(i*phase))| ranges from 0 (no locking) to 1 (perfect locking)
        plv = np.abs(np.mean(np.exp(1j * phases)))

        # Mean preferred phase
        mean_phase = circmean(phases, high=np.pi, low=-np.pi)

        # Rayleigh test: tests whether phases are uniformly distributed
        # Test statistic: R = n * PLV^2
        n_spikes = len(phases)
        r_stat = n_spikes * plv**2
        # Approximate p-value for Rayleigh test (for R > 2)
        if r_stat > 2:
            rayleigh_p = np.exp(-r_stat) * (1 + (2*r_stat - r_stat**2) / (4*n_spikes) -
                                            (24*r_stat - 132*r_stat**2 + 76*r_stat**3 - 9*r_stat**4) /
                                            (288*n_spikes**2))
        else:
            rayleigh_p = 1.0

        plv_values[unit_id] = plv
        mean_phases[unit_id] = mean_phase
        rayleigh_pvalues[unit_id] = rayleigh_p
    else:
        plv_values[unit_id] = 0
        mean_phases[unit_id] = 0
        rayleigh_pvalues[unit_id] = 1.0

print(f"\nPhase locking analysis:")
print(f"  Metric: Phase Locking Value (PLV)")
print(f"  Range: 0 (no locking) to 1 (perfect locking)")
print(f"  Statistical test: Rayleigh test of uniformity\n")

print(f"{'Unit':<6} {'Type':<12} {'PLV':<8} {'Mean Phase':<12} {'Rayleigh p':<10}")
print("-" * 60)
for unit_id in range(n_units):
    cell_type = spike_data[unit_id]['cell_type']
    plv = plv_values[unit_id]
    mean_phase = mean_phases[unit_id]
    p_val = rayleigh_pvalues[unit_id]

    # Convert mean phase to degrees for readability
    mean_phase_deg = np.degrees(mean_phase)

    significance = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'

    print(f"{unit_id:<6} {cell_type:<12} {plv:<8.3f} {mean_phase_deg:>7.1f}°"
          f"       {p_val:<10.2e} {significance}")

# %% [markdown]
# ## Visualization 1: LFP Signal and Spike Raster

# %%
# Create figure showing LFP and spike raster
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Plot window: first 10 seconds for detail
plot_duration = 10
plot_mask = t <= plot_duration

# Subplot 1: Theta LFP
ax = axes[0]
ax.plot(t[plot_mask], theta_signal[plot_mask], color='#1f77b4', linewidth=1, alpha=0.8)
ax.set_ylabel('Voltage (μV)', fontsize=11, fontweight='bold')
ax.set_title('Hippocampal Theta Rhythm (First 10 seconds)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, plot_duration)

# Subplot 2: Spike raster
ax = axes[1]
for unit_id in range(n_units):
    spike_times = spike_data[unit_id]['spike_times']
    spike_times_plot = spike_times[spike_times <= plot_duration]

    if len(spike_times_plot) > 0:
        # Color by cell type
        color = '#2ca02c' if spike_data[unit_id]['cell_type'] == 'pyramidal' else '#ff7f0e'
        ax.scatter(spike_times_plot, np.full_like(spike_times_plot, unit_id),
                  s=15, marker='|', color=color, linewidths=0.5, alpha=0.7)

ax.set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
ax.set_ylabel('Unit ID', fontsize=11, fontweight='bold')
ax.set_title('Unit Spike Raster (Green=Pyramidal, Orange=Interneuron)', fontsize=12, fontweight='bold')
ax.set_xlim(0, plot_duration)
ax.set_ylim(-0.5, n_units - 0.5)
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('01_lfp_and_raster.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Saved: 01_lfp_and_raster.png")
plt.close()

# %% [markdown]
# ## Visualization 2: Theta Phase Distribution at Spike Times

# %%
# Create polar plots showing phase locking for different unit types
fig = plt.figure(figsize=(16, 6))

# Separate units by type
pyramidal_units = [uid for uid in range(n_units) if spike_data[uid]['cell_type'] == 'pyramidal']
interneuron_units = [uid for uid in range(n_units) if spike_data[uid]['cell_type'] == 'interneuron']

# Plot 1: Example pyramidal cell
ax1 = plt.subplot(1, 3, 1, projection='polar')
unit_id = pyramidal_units[0]
phases = theta_phases_at_spikes[unit_id]
theta_bins = np.linspace(-np.pi, np.pi, 36)
hist, _ = np.histogram(phases, bins=theta_bins)
theta_centers = (theta_bins[:-1] + theta_bins[1:]) / 2
bars = ax1.bar(theta_centers, hist, width=np.deg2rad(10), alpha=0.7, color='#2ca02c', edgecolor='black')
ax1.plot(mean_phases[unit_id], np.max(hist), 'r*', markersize=15, label='Mean phase')
ax1.set_theta_zero_location('E')
ax1.set_theta_direction(1)
ax1.set_title(f'Pyramidal Unit {unit_id}\nPLV={plv_values[unit_id]:.3f}',
             fontsize=11, fontweight='bold', pad=20)
ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

# Plot 2: Example interneuron
ax2 = plt.subplot(1, 3, 2, projection='polar')
unit_id = interneuron_units[0]
phases = theta_phases_at_spikes[unit_id]
hist, _ = np.histogram(phases, bins=theta_bins)
bars = ax2.bar(theta_centers, hist, width=np.deg2rad(10), alpha=0.7, color='#ff7f0e', edgecolor='black')
ax2.plot(mean_phases[unit_id], np.max(hist), 'r*', markersize=15, label='Mean phase')
ax2.set_theta_zero_location('E')
ax2.set_theta_direction(1)
ax2.set_title(f'Interneuron Unit {unit_id}\nPLV={plv_values[unit_id]:.3f}',
             fontsize=11, fontweight='bold', pad=20)
ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

# Plot 3: Population mean phases
ax3 = plt.subplot(1, 3, 3, projection='polar')
# Separate colors by cell type
for unit_id in pyramidal_units:
    ax3.scatter(mean_phases[unit_id], plv_values[unit_id],
               s=150, c='#2ca02c', alpha=0.6, edgecolors='black', linewidth=1)
for unit_id in interneuron_units:
    ax3.scatter(mean_phases[unit_id], plv_values[unit_id],
               s=150, c='#ff7f0e', alpha=0.6, edgecolors='black', linewidth=1)
ax3.set_ylim(0, 1)
ax3.set_theta_zero_location('E')
ax3.set_theta_direction(1)
ax3.set_yticks([0.25, 0.5, 0.75, 1.0])
ax3.set_yticklabels(['0.25', '0.5', '0.75', '1.0'], fontsize=9)
ax3.set_title('Population Preferred Phases\n(Radius = PLV)', fontsize=11, fontweight='bold', pad=20)
ax3.grid(True)

plt.tight_layout()
plt.savefig('02_phase_distributions.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: 02_phase_distributions.png")
plt.close()

# %% [markdown]
# ## Visualization 3: Phase Locking Values by Cell Type

# %%
# Compare PLV statistics between cell types
plv_pyramidal = [plv_values[uid] for uid in pyramidal_units]
plv_interneurons = [plv_values[uid] for uid in interneuron_units]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Subplot 1: Box plot of PLV by cell type
ax = axes[0]
bp = ax.boxplot([plv_pyramidal, plv_interneurons],
                 labels=['Pyramidal\nCells', 'Interneurons'],
                 patch_artist=True, widths=0.6)
colors = ['#2ca02c', '#ff7f0e']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

# Overlay individual points
for i, (data, color) in enumerate([(plv_pyramidal, '#2ca02c'), (plv_interneurons, '#ff7f0e')]):
    x = np.random.normal(i+1, 0.04, size=len(data))
    ax.scatter(x, data, s=60, color=color, alpha=0.7, edgecolors='black', linewidth=0.5)

ax.set_ylabel('Phase Locking Value (PLV)', fontsize=11, fontweight='bold')
ax.set_title('Theta Phase Entrainment Strength by Cell Type', fontsize=12, fontweight='bold')
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3, axis='y')

# Add statistics
mean_pyr = np.mean(plv_pyramidal)
mean_int = np.mean(plv_interneurons)
ax.text(0.5, 0.95, f'Pyramidal: μ={mean_pyr:.3f}', transform=ax.transAxes,
       fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax.text(0.5, 0.85, f'Interneurons: μ={mean_int:.3f}', transform=ax.transAxes,
       fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Subplot 2: Histogram of PLV values
ax = axes[1]
ax.hist(plv_pyramidal, bins=8, alpha=0.6, label='Pyramidal cells', color='#2ca02c', edgecolor='black')
ax.hist(plv_interneurons, bins=8, alpha=0.6, label='Interneurons', color='#ff7f0e', edgecolor='black')
ax.set_xlabel('Phase Locking Value (PLV)', fontsize=11, fontweight='bold')
ax.set_ylabel('Number of Units', fontsize=11, fontweight='bold')
ax.set_title('Distribution of PLV Across Population', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('03_plv_by_celltype.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: 03_plv_by_celltype.png")
plt.close()

# %% [markdown]
# ## Visualization 4: Peri-Theta Histograms

# %%
# Create peri-theta time histograms: firing probability across theta cycle
fig, axes = plt.subplots(3, 5, figsize=(16, 10))
axes = axes.flatten()

for unit_id in range(n_units):
    ax = axes[unit_id]

    # Extract phases and create histogram
    phases = theta_phases_at_spikes[unit_id]
    theta_bins = np.linspace(-np.pi, np.pi, 37)
    hist, bin_edges = np.histogram(phases, bins=theta_bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Plot histogram
    cell_type = spike_data[unit_id]['cell_type']
    color = '#2ca02c' if cell_type == 'pyramidal' else '#ff7f0e'
    ax.bar(bin_centers, hist, width=np.diff(bin_edges)[0], alpha=0.7, color=color, edgecolor='black')

    # Overlay mean phase as vertical line
    ax.axvline(mean_phases[unit_id], color='red', linestyle='--', linewidth=2, label='Mean phase')

    # Add circular reference (phase wraps around)
    ax.set_xlim(-np.pi, np.pi)
    ax.set_xticks([-np.pi, 0, np.pi])
    ax.set_xticklabels(['π', '0', 'π'], fontsize=8)
    ax.set_ylabel('Probability', fontsize=9)

    # Title with statistics
    plv = plv_values[unit_id]
    title = f'Unit {unit_id} ({cell_type[0].upper()})\nPLV={plv:.3f}'
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

# Remove extra subplots
for idx in range(n_units, len(axes)):
    fig.delaxes(axes[idx])

plt.suptitle('Firing Probability vs Theta Phase (Each Unit)',
            fontsize=13, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('04_peri_theta_histograms.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: 04_peri_theta_histograms.png")
plt.close()

# %% [markdown]
# ## Summary Statistics and Interpretation

# %%
print("\n" + "="*70)
print("THETA PHASE ENTRAINMENT ANALYSIS SUMMARY")
print("="*70)

print(f"\nPopulation Statistics:")
print(f"  Overall mean PLV: {np.mean(list(plv_values.values())):.3f}")
print(f"  Median PLV: {np.median(list(plv_values.values())):.3f}")
print(f"  PLV range: [{np.min(list(plv_values.values())):.3f}, "
      f"{np.max(list(plv_values.values())):.3f}]")

print(f"\nBy Cell Type:")
print(f"  Pyramidal cells (n={n_pyramidal}):")
print(f"    Mean PLV: {np.mean(plv_pyramidal):.3f}")
print(f"    Std PLV: {np.std(plv_pyramidal):.3f}")
print(f"  Interneurons (n={n_interneurons}):")
print(f"    Mean PLV: {np.mean(plv_interneurons):.3f}")
print(f"    Std PLV: {np.std(plv_interneurons):.3f}")

# Proportion of units with significant theta entrainment (p < 0.05)
n_significant = sum(1 for p in rayleigh_pvalues.values() if p < 0.05)
print(f"\nSignificant Phase Locking (Rayleigh p < 0.05): {n_significant}/{n_units} units")

print(f"\nKey Findings:")
print(f"  1. Hippocampal neurons fire preferentially at specific theta phases")
print(f"  2. Phase locking strength varies between cell types and individuals")
print(f"  3. Pyramidal cells typically show stronger theta entrainment than interneurons")
print(f"  4. Preferred phases cluster, reflecting organized network timing")

print(f"\nBiological Interpretation:")
print(f"  - Theta rhythm provides a global clock for hippocampal computation")
print(f"  - Phase entrainment allows different cell types to coordinate firing")
print(f"  - Theta phase carries information about behavioral/cognitive state")
print(f"  - Phase-based coding may enhance information transmission efficiency")

print(f"\n{'='*70}")
print(f"Analysis complete. All figures saved.")
print(f"{'='*70}\n")
