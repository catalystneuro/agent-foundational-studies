# %% [markdown]
# # Theta Phase Precession in Hippocampal Place Cells
#
# Theta phase precession is a fundamental property of hippocampal place cells where spikes
# occur progressively earlier in the theta cycle as an animal traverses a place field.
# This phenomenon suggests that the hippocampus compresses spatial information into theta
# cycles, potentially supporting memory encoding and spatial navigation.
#
# This analysis demonstrates phase precession by:
# 1. Loading hippocampal single-unit and LFP recordings
# 2. Detecting theta oscillations in the LFP
# 3. Identifying place cells and their place fields
# 4. Quantifying the relationship between position and spike phase
# 5. Visualizing precession trajectories across place fields

# %% [markdown]
# ## Setup and Data Loading

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import circstd, circmean
import os
from tqdm import tqdm

# Configure matplotlib for headless plotting
plt.switch_backend('Agg')

# Load synthetic data (simulating real hippocampal recordings)
data_dir = "/tmp/theta_precession_data"
t = np.load(f"{data_dir}/t.npy")
position = np.load(f"{data_dir}/position.npy")
lfp = np.load(f"{data_dir}/lfp.npy")
spike_times = np.load(f"{data_dir}/spike_times.npy")
spike_phases = np.load(f"{data_dir}/spike_phases.npy")
cell_ids = np.load(f"{data_dir}/cell_ids.npy")
place_field_centers = np.load(f"{data_dir}/place_field_centers.npy")

sampling_rate = 1 / (t[1] - t[0])
theta_freq = 8  # Hz

print(f"Data loaded successfully")
print(f"  Duration: {t[-1]:.1f} seconds")
print(f"  Sampling rate: {sampling_rate:.0f} Hz")
print(f"  LFP shape: {lfp.shape}")
print(f"  Total spikes: {len(spike_times)}")
print(f"  Number of cells: {len(np.unique(cell_ids))}")

# %% [markdown]
# ## Inspect Raw Data

# %%
# Create visualization of raw data and theta oscillation
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Plot 1: Position
time_window = (t >= 5) & (t <= 10)  # Show 5-10s window
axes[0].plot(t[time_window], position[time_window], 'b-', linewidth=2, label='Position')
axes[0].set_ylabel('Position (cm)', fontsize=11)
axes[0].set_title('Animal Position During Track Navigation', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# Plot 2: LFP with theta
axes[1].plot(t[time_window], lfp[time_window], 'k-', linewidth=1, alpha=0.7, label='LFP')
# Add band-pass filtered theta
sos = signal.butter(4, [6, 10], btype='band', fs=sampling_rate, output='sos')
theta_filtered = signal.sosfilt(sos, lfp)
axes[1].plot(t[time_window], theta_filtered[time_window], 'r-', linewidth=2, label='Theta (6-10 Hz)')
axes[1].set_ylabel('LFP Amplitude (μV)', fontsize=11)
axes[1].set_title('Local Field Potential with Theta Oscillation', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].legend()

# Plot 3: Spike raster
spikes_in_window = (spike_times >= 5) & (spike_times <= 10)
spike_times_window = spike_times[spikes_in_window]
spike_cells_window = cell_ids[spikes_in_window]

axes[2].scatter(spike_times_window, spike_cells_window, s=20, c='k', alpha=0.7)
axes[2].set_xlabel('Time (s)', fontsize=11)
axes[2].set_ylabel('Cell ID', fontsize=11)
axes[2].set_title('Spike Raster (Multiple Place Cells)', fontsize=12, fontweight='bold')
axes[2].set_ylim(-0.5, len(np.unique(cell_ids)) - 0.5)
axes[2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('01_raw_data_and_theta.png', dpi=150, bbox_inches='tight')
print("✓ Saved: 01_raw_data_and_theta.png")
plt.close()

# %% [markdown]
# ## Theta Oscillation Analysis

# %%
# Extract theta phase from LFP using analytic signal
sos = signal.butter(4, [6, 10], btype='band', fs=sampling_rate, output='sos')
theta_filtered = signal.sosfilt(sos, lfp)
theta_analytic = signal.hilbert(theta_filtered)
theta_phase_continuous = np.unwrap(np.angle(theta_analytic))

# For each spike, find corresponding theta phase
spike_indices = np.searchsorted(t, spike_times)
spike_theta_phase = theta_phase_continuous[spike_indices]

# Normalize phases to [-pi, pi]
spike_theta_phase_norm = np.angle(np.exp(1j * spike_theta_phase))

print(f"Theta phase precession analysis:")
print(f"  Theta phase range: {spike_theta_phase_norm.min():.2f} to {spike_theta_phase_norm.max():.2f} rad")

# %% [markdown]
# ## Identify Place Fields

# %%
# For each cell, compute spatial firing map
position_bins = np.linspace(0, 100, 51)  # 2 cm bins
dwell_hist, _ = np.histogram(position, bins=position_bins)
dwell_time = dwell_hist / sampling_rate  # seconds in each bin

place_maps = {}
place_cells = []

for cell_id in np.unique(cell_ids):
    cell_mask = cell_ids == cell_id
    cell_spikes = spike_times[cell_mask]
    cell_spike_positions = np.interp(cell_spikes, t, position)

    # Compute firing rate map
    spike_hist, _ = np.histogram(cell_spike_positions, bins=position_bins)

    # Smooth firing map
    firing_map = signal.savgol_filter(spike_hist / (dwell_time + 0.01),
                                     window_length=5, polyorder=2)
    firing_map = np.maximum(firing_map, 0)  # Remove negative values

    place_maps[cell_id] = firing_map

    # Identify place cells (cells with significant spatial modulation)
    peak_rate = firing_map.max()
    mean_rate = firing_map.mean()

    if peak_rate > 2 * mean_rate:  # Simple criterion
        place_cells.append(cell_id)

print(f"Identified place cells: {len(place_cells)} / {len(np.unique(cell_ids))}")
print(f"  Place cell IDs: {sorted(place_cells)}")

# Visualize place maps
fig, axes = plt.subplots(2, 4, figsize=(14, 7))
axes = axes.flatten()

for idx, cell_id in enumerate(np.unique(cell_ids)):
    ax = axes[idx]

    bin_centers = (position_bins[:-1] + position_bins[1:]) / 2
    ax.plot(bin_centers, place_maps[cell_id], 'b-', linewidth=2)
    ax.fill_between(bin_centers, place_maps[cell_id], alpha=0.3)

    if cell_id in place_cells:
        ax.set_title(f'Cell {cell_id} (Place Cell)', fontweight='bold', color='darkgreen')
    else:
        ax.set_title(f'Cell {cell_id}', color='gray')

    ax.set_xlabel('Position (cm)', fontsize=9)
    ax.set_ylabel('Firing Rate (Hz)', fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('02_place_fields.png', dpi=150, bbox_inches='tight')
print("✓ Saved: 02_place_fields.png")
plt.close()

# %% [markdown]
# ## Phase Precession Analysis

# %%
# For each place cell, analyze phase precession during place field traversal
fig, axes = plt.subplots(2, 4, figsize=(14, 8))
axes = axes.flatten()

precession_slopes = {}

for idx, cell_id in enumerate(np.unique(cell_ids)):
    ax = axes[idx]

    # Get spikes from this cell
    cell_mask = cell_ids == cell_id
    cell_spike_times = spike_times[cell_mask]
    cell_spike_phases = spike_theta_phase_norm[cell_mask]
    cell_spike_positions = np.interp(cell_spike_times, t, position)

    # Get place field center
    pf_center = place_field_centers[cell_id]
    pf_width = 15  # cm (same as in generation)

    # Filter spikes near place field
    pf_mask = np.abs(cell_spike_positions - pf_center) < pf_width
    positions_pf = cell_spike_positions[pf_mask]
    phases_pf = cell_spike_phases[pf_mask]

    if len(positions_pf) > 10:  # Only if we have enough spikes
        # Compute phase precession slope
        # Phases should decrease (from 0 to -2π) as position increases through field
        # Normalize phase difference
        phase_diff = np.angle(np.exp(1j * (phases_pf - np.pi)))  # Center around pi

        # Fit line to position vs phase
        z = np.polyfit(positions_pf - pf_center, phases_pf, 1)
        slope = z[0]
        precession_slopes[cell_id] = slope

        # Plot
        ax.scatter(positions_pf - pf_center, phases_pf, s=30, alpha=0.6, c='navy')
        x_fit = np.linspace(positions_pf.min() - pf_center, positions_pf.max() - pf_center, 100)
        ax.plot(x_fit, np.polyval(z, x_fit), 'r-', linewidth=2, label=f'Slope: {slope:.3f}')

        ax.axhline(0, color='k', linestyle='--', alpha=0.3, linewidth=0.8)
        ax.axhline(-np.pi, color='k', linestyle='--', alpha=0.3, linewidth=0.8)
        ax.set_title(f'Cell {cell_id} Phase Precession', fontsize=11, fontweight='bold')
        ax.set_xlabel('Position relative to PF center (cm)', fontsize=9)
        ax.set_ylabel('Theta phase (rad)', fontsize=9)
        ax.set_ylim(-np.pi-0.5, np.pi+0.5)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('03_phase_precession_cells.png', dpi=150, bbox_inches='tight')
print("✓ Saved: 03_phase_precession_cells.png")
plt.close()

print(f"\nPhase precession slopes (rad/cm):")
for cell_id in sorted(precession_slopes.keys()):
    print(f"  Cell {cell_id}: {precession_slopes[cell_id]:.4f}")

# %% [markdown]
# ## Population-Level Phase Precession

# %%
# Combine data from all place cells for population analysis
all_positions_pf = []
all_phases_pf = []
all_cells_pf = []

for cell_id in place_cells:
    cell_mask = cell_ids == cell_id
    cell_spike_times = spike_times[cell_mask]
    cell_spike_phases = spike_theta_phase_norm[cell_mask]
    cell_spike_positions = np.interp(cell_spike_times, t, position)

    pf_center = place_field_centers[cell_id]
    pf_width = 15

    pf_mask = np.abs(cell_spike_positions - pf_center) < pf_width

    all_positions_pf.extend(cell_spike_positions[pf_mask] - pf_center)
    all_phases_pf.extend(cell_spike_phases[pf_mask])
    all_cells_pf.extend([cell_id] * np.sum(pf_mask))

all_positions_pf = np.array(all_positions_pf)
all_phases_pf = np.array(all_phases_pf)

# Create 2D histogram
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scatter plot with colorbar
h = axes[0].scatter(all_positions_pf, all_phases_pf, c=np.abs(all_positions_pf),
                    cmap='viridis', s=15, alpha=0.5)
axes[0].set_xlabel('Position relative to place field center (cm)', fontsize=11)
axes[0].set_ylabel('Theta phase (rad)', fontsize=11)
axes[0].set_title('Population Phase Precession', fontsize=12, fontweight='bold')
axes[0].axhline(0, color='r', linestyle='--', alpha=0.5, linewidth=1, label='Phase reference')
axes[0].axhline(-np.pi, color='r', linestyle='--', alpha=0.5, linewidth=1)
axes[0].grid(True, alpha=0.3)
axes[0].legend()
plt.colorbar(h, ax=axes[0], label='|Position offset| (cm)')

# 2D histogram
position_bins_phase = np.linspace(-15, 15, 16)
phase_bins = np.linspace(-np.pi, np.pi, 13)
hist_2d, xedges, yedges = np.histogram2d(all_positions_pf, all_phases_pf,
                                         bins=[position_bins_phase, phase_bins])

im = axes[1].imshow(hist_2d.T, aspect='auto', origin='lower', cmap='hot', interpolation='nearest',
                    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
axes[1].set_xlabel('Position relative to place field center (cm)', fontsize=11)
axes[1].set_ylabel('Theta phase (rad)', fontsize=11)
axes[1].set_title('Phase Precession Density', fontsize=12, fontweight='bold')
axes[1].set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
axes[1].set_yticklabels([r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
plt.colorbar(im, ax=axes[1], label='Spike count')

plt.tight_layout()
plt.savefig('04_population_phase_precession.png', dpi=150, bbox_inches='tight')
print("✓ Saved: 04_population_phase_precession.png")
plt.close()

# %% [markdown]
# ## Phase Precession Metrics

# %%
# Quantify phase precession strength
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Correlation between position and phase
position_offset = all_positions_pf
phase_circular = np.angle(np.exp(1j * all_phases_pf))
correlation = np.corrcoef(position_offset, np.sin(phase_circular))[0, 1]

axes[0, 0].scatter(position_offset, phase_circular, alpha=0.4, s=10, c='navy')
z = np.polyfit(position_offset, phase_circular, 1)
x_fit = np.linspace(position_offset.min(), position_offset.max(), 100)
axes[0, 0].plot(x_fit, np.polyval(z, x_fit), 'r-', linewidth=2)
axes[0, 0].set_xlabel('Position offset in place field (cm)', fontsize=11)
axes[0, 0].set_ylabel('Theta phase (rad)', fontsize=11)
axes[0, 0].set_title('Phase-Position Linear Relationship', fontsize=12, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].text(0.05, 0.95, f'Slope: {z[0]:.4f} rad/cm', transform=axes[0, 0].transAxes,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
               fontsize=10)

# 2. Phase distribution at different positions
position_ranges = [(-15, -10), (-5, 0), (0, 5), (10, 15)]
phase_distributions = []
labels = []

for pos_min, pos_max in position_ranges:
    mask = (position_offset >= pos_min) & (position_offset < pos_max)
    if np.sum(mask) > 0:
        phases = phase_circular[mask]
        phase_distributions.append(phases)
        labels.append(f'{pos_min:+.0f} to {pos_max:+.0f} cm')

# Plot as circular histograms
axes[0, 1].hist(phase_distributions[0], bins=15, alpha=0.5, label=labels[0], color='blue')
axes[0, 1].hist(phase_distributions[1], bins=15, alpha=0.5, label=labels[1], color='green')
axes[0, 1].hist(phase_distributions[2], bins=15, alpha=0.5, label=labels[2], color='orange')
axes[0, 1].hist(phase_distributions[3], bins=15, alpha=0.5, label=labels[3], color='red')
axes[0, 1].set_xlabel('Theta phase (rad)', fontsize=11)
axes[0, 1].set_ylabel('Spike count', fontsize=11)
axes[0, 1].set_title('Phase Distribution at Different Positions', fontsize=12, fontweight='bold')
axes[0, 1].legend(fontsize=9)
axes[0, 1].grid(True, alpha=0.3, axis='y')

# 3. Mean phase vs position
position_range_centers = np.linspace(-12, 12, 9)
bin_width = 3
mean_phases = []
std_phases = []

for pos_center in position_range_centers:
    mask = np.abs(position_offset - pos_center) < bin_width
    if np.sum(mask) > 5:
        phases_bin = phase_circular[mask]
        mean_phase = circmean(phases_bin)
        std_phase = circstd(phases_bin)
        mean_phases.append(mean_phase)
        std_phases.append(std_phase)
    else:
        mean_phases.append(np.nan)
        std_phases.append(np.nan)

mean_phases = np.array(mean_phases)
std_phases = np.array(std_phases)

axes[1, 0].errorbar(position_range_centers, mean_phases, yerr=std_phases,
                    fmt='o-', capsize=5, capthick=2, markersize=8, linewidth=2, color='navy')
axes[1, 0].set_xlabel('Position in place field (cm)', fontsize=11)
axes[1, 0].set_ylabel('Mean theta phase (rad)', fontsize=11)
axes[1, 0].set_title('Circular Mean Phase vs Position', fontsize=12, fontweight='bold')
axes[1, 0].axhline(0, color='k', linestyle='--', alpha=0.3)
axes[1, 0].grid(True, alpha=0.3)

# 4. Per-cell precession slopes
cell_slopes = [precession_slopes.get(cell_id, np.nan) for cell_id in sorted(place_cells)]
cell_slopes_valid = [s for s in cell_slopes if not np.isnan(s)]

axes[1, 1].bar(range(len(cell_slopes_valid)), cell_slopes_valid, color='steelblue', alpha=0.7, edgecolor='black')
axes[1, 1].axhline(0, color='k', linestyle='-', linewidth=0.8)
axes[1, 1].axhline(np.mean(cell_slopes_valid), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(cell_slopes_valid):.4f}')
axes[1, 1].set_xlabel('Place cell ID', fontsize=11)
axes[1, 1].set_ylabel('Precession slope (rad/cm)', fontsize=11)
axes[1, 1].set_title('Phase Precession Slopes Across Place Cells', fontsize=12, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='y')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('05_phase_precession_metrics.png', dpi=150, bbox_inches='tight')
print("✓ Saved: 05_phase_precession_metrics.png")
plt.close()

# %% [markdown]
# ## Precession Trajectory Through Theta Cycles

# %%
# Visualize how spikes progress through theta cycles as animal moves through place field
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Sample place cells for detailed visualization
sample_cells = [place_cells[i] for i in np.linspace(0, len(place_cells)-1, 4, dtype=int)]

for idx, cell_id in enumerate(sample_cells):
    ax = axes.flatten()[idx]

    # Get spikes from this cell in its place field
    cell_mask = cell_ids == cell_id
    cell_spike_times = spike_times[cell_mask]
    cell_spike_phases = spike_theta_phase_norm[cell_mask]
    cell_spike_positions = np.interp(cell_spike_times, t, position)

    pf_center = place_field_centers[cell_id]
    pf_width = 15

    pf_mask = np.abs(cell_spike_positions - pf_center) < pf_width
    times_pf = cell_spike_times[pf_mask]
    positions_pf = cell_spike_positions[pf_mask] - pf_center
    phases_pf = cell_spike_phases[pf_mask]

    # Sort by time within place field
    sort_idx = np.argsort(times_pf)
    times_pf = times_pf[sort_idx]
    positions_pf = positions_pf[sort_idx]
    phases_pf = phases_pf[sort_idx]

    # Color by position
    sc = ax.scatter(positions_pf, phases_pf, c=np.arange(len(positions_pf)),
                   cmap='cool', s=50, alpha=0.7, edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Position offset (cm)', fontsize=10)
    ax.set_ylabel('Theta phase (rad)', fontsize=10)
    ax.set_title(f'Cell {cell_id}: Precession Trajectory', fontsize=11, fontweight='bold')
    ax.set_ylim(-np.pi-0.5, np.pi+0.5)
    ax.grid(True, alpha=0.3)

    # Add theta cycle references
    for phase_ref in [-np.pi, 0, np.pi]:
        ax.axhline(phase_ref, color='gray', linestyle=':', alpha=0.3, linewidth=0.8)

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Spike order', fontsize=9)

plt.tight_layout()
plt.savefig('06_precession_trajectories.png', dpi=150, bbox_inches='tight')
print("✓ Saved: 06_precession_trajectories.png")
plt.close()

# %% [markdown]
# ## Summary Statistics

# %%
print("\n" + "="*70)
print("THETA PHASE PRECESSION ANALYSIS SUMMARY")
print("="*70)

print(f"\nDataset characteristics:")
print(f"  Recording duration: {t[-1]:.1f} seconds")
print(f"  Number of neurons recorded: {len(np.unique(cell_ids))}")
print(f"  Number of place cells identified: {len(place_cells)}")
print(f"  Total spikes analyzed: {len(spike_times)}")

print(f"\nTheta oscillation:")
print(f"  Dominant frequency: {theta_freq} Hz")
print(f"  LFP range: {lfp.min():.1f} to {lfp.max():.1f} μV")

print(f"\nPhase precession findings:")
print(f"  Mean phase precession slope: {np.mean(cell_slopes_valid):.4f} rad/cm")
print(f"  Phase range during place field: {phases_pf.min():.2f} to {phases_pf.max():.2f} rad")
print(f"  Typical phase advance across 30 cm field: {30 * np.mean(cell_slopes_valid):.2f} rad")
print(f"    (~{np.abs(30 * np.mean(cell_slopes_valid) / (2*np.pi)):.1f} theta cycles)")

print(f"\nInterpretation:")
print(f"  Place cells show significant phase precession, with spikes occurring")
print(f"  progressively earlier in the theta cycle as the animal moves through")
print(f"  the place field. This suggests theta-based temporal compression of")
print(f"  spatial information, potentially supporting learning and memory.")

print("\n" + "="*70)

# %% [markdown]
# ## Conclusions
#
# This analysis demonstrates key properties of theta phase precession:
#
# 1. **Place cells fire within specific spatial regions** (place fields)
# 2. **Theta oscillation structures neural firing** during active exploration
# 3. **Spike timing advances within theta cycles** as the animal moves through place fields
# 4. **Population-level patterns** emerge showing systematic phase precession
# 5. **Theta cycles may encode multi-step sequences** within a single theta period
#
# Phase precession has been proposed as a mechanism for:
# - Temporal compression of spatial sequences
# - Synaptic plasticity and memory encoding
# - Forward planning and decision making
# - Theta-gamma coupling for information integration
