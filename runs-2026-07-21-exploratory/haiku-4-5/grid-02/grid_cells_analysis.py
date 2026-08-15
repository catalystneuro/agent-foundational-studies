# %% [markdown]
# # Grid Cells in Medial Entorhinal Cortex
#
# This analysis demonstrates grid cell firing patterns using electrophysiological recordings
# from medial entorhinal cortex (MEC) during open field navigation in rats.
#
# The data comes from DANDI 000582 (Sargolini et al., 2006, Science) and contains:
# - Single-unit spike times from tetrode recordings in MEC
# - Spatial position of the animal from LED tracking
# - Local field potential (LFP) recordings
#
# Grid cells fire in a characteristic hexagonal pattern as the animal explores the environment,
# creating multiple "firing fields" arranged in a triangular lattice. This spatial periodicity
# is a fundamental property of grid cells and is thought to support metric encoding of space.

# %% [markdown]
# ## Setup and Data Loading

# %%
import os
os.environ['MPLBACKEND'] = 'Agg'

from pynwb import NWBHDF5IO
import pynapple as nap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = "/tmp/grid_analysis/000582"
OUTPUT_DIR = "/tmp/grid_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load session file
nwb_file = os.path.join(DATA_DIR, "sub-11084/sub-11084_ses-28020501_behavior+ecephys.nwb")
print(f"Loading: {nwb_file}")

io = NWBHDF5IO(nwb_file, 'r', load_namespaces=True)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)

print("Data loaded successfully!")
print(nwb)

# %% [markdown]
# ## Extract Spike and Position Data

# %%
# Get spike data
spikes = nwb['units']
print(f"\nTotal units in recording: {len(spikes)}")
for i in range(len(spikes)):
    print(f"  Unit {i}: {len(spikes[i])} spikes")

# Get position data - it's already a TsdFrame from Pynapple
position = nwb['SpatialSeriesLED1']
print(f"\nPosition data shape: {position.shape}")
print(f"Position time range: {position.t[0]:.1f} - {position.t[-1]:.1f} s")
print(f"Position range: x=[{position.values[:, 0].min():.1f}, {position.values[:, 0].max():.1f}], "
      f"y=[{position.values[:, 1].min():.1f}, {position.values[:, 1].max():.1f}]")

# %% [markdown]
# ## Compute Spatial Firing Maps

# %%
# Parameters for spatial binning
bin_size = 3  # cm
min_speed = 2   # cm/s (require minimum movement)
max_speed = 50  # cm/s (filter out unrealistic speeds)

# Compute velocity
pos_x = position.values[:, 0]
pos_y = position.values[:, 1]
velocity = np.sqrt(np.diff(pos_x)**2 + np.diff(pos_y)**2) / np.diff(position.t)
velocity = np.insert(velocity, 0, velocity[0])

# Create velocity array for filtering
valid_speed = (velocity > min_speed) & (velocity < max_speed)
n_valid = valid_speed.sum()
print(f"\nValid movement: {n_valid} / {len(velocity)} samples ({100*n_valid/len(velocity):.1f}%)")

# Compute position binning
x_min, x_max = pos_x.min(), pos_x.max()
y_min, y_max = pos_y.min(), pos_y.max()

n_bins_x = int(np.ceil((x_max - x_min) / bin_size))
n_bins_y = int(np.ceil((y_max - y_min) / bin_size))

print(f"\nSpatial binning: {n_bins_x}x{n_bins_y} bins ({bin_size} cm each)")

# Compute occupancy map (only for valid movements)
dt = np.mean(np.diff(position.t))
occupancy, x_edges, y_edges = np.histogram2d(
    pos_x[valid_speed],
    pos_y[valid_speed],
    bins=[n_bins_x, n_bins_y],
    range=[[x_min, x_max], [y_min, y_max]]
)
occupancy = occupancy * dt

print(f"Occupancy map shape: {occupancy.shape}")
print(f"Mean occupancy per bin: {occupancy.mean():.3f} s")

# %% [markdown]
# ## Analyze Individual Grid Cells

# %%
# Compute firing rate maps for each unit
firing_maps = {}
grid_scores = {}

print(f"\nAnalyzing {len(spikes)} units for spatial firing patterns...")

for unit_id in tqdm(range(len(spikes)), desc="Computing firing maps"):
    spike_times = spikes[unit_id]

    if len(spike_times) < 50:  # Skip units with too few spikes
        continue

    # Interpolate position at spike times
    spike_times_array = spike_times.t
    spike_pos_x = np.interp(spike_times_array, position.t, pos_x)
    spike_pos_y = np.interp(spike_times_array, position.t, pos_y)

    # Compute 2D firing rate map
    spike_count, _, _ = np.histogram2d(
        spike_pos_x,
        spike_pos_y,
        bins=[n_bins_x, n_bins_y],
        range=[[x_min, x_max], [y_min, y_max]]
    )

    # Normalize by occupancy
    with np.errstate(divide='ignore', invalid='ignore'):
        rate_map = spike_count / occupancy
        rate_map[~np.isfinite(rate_map)] = 0

    # Smooth the rate map
    rate_map_smooth = gaussian_filter(rate_map.astype(float), sigma=1.0)

    firing_maps[unit_id] = {
        'rate_map': rate_map_smooth,
        'spike_count': spike_count,
        'occupancy': occupancy,
        'n_spikes': len(spike_times)
    }

print(f"Computed firing maps for {len(firing_maps)} units")

# %% [markdown]
# ## Visualization: Firing Rate Maps

# %%
# Select a subset of units
n_show = min(3, len(firing_maps))
selected_units = list(firing_maps.keys())[:n_show]

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
if n_show == 1:
    axes = [axes]

for idx, unit_id in enumerate(selected_units):
    ax = axes[idx]

    rate_map = firing_maps[unit_id]['rate_map']

    # Plot firing rate map
    im = ax.imshow(rate_map.T, origin='lower', cmap='hot', aspect='auto',
                   extent=[x_min, x_max, y_min, y_max], interpolation='bilinear')

    ax.set_xlabel('X (cm)', fontsize=10)
    ax.set_ylabel('Y (cm)', fontsize=10)
    ax.set_title(f'Unit {unit_id}\n{firing_maps[unit_id]["n_spikes"]} spikes, peak: {rate_map.max():.1f} Hz',
                fontsize=11, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, label='Firing rate (Hz)', shrink=0.8)

fig.suptitle('Spatial Firing Rate Maps from MEC Units', fontsize=13, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '01_firing_rate_maps.png'), dpi=150, bbox_inches='tight')
print("Saved: 01_firing_rate_maps.png")
plt.close()

# %% [markdown]
# ## Visualization: Trajectory with Spike Locations

# %%
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Full trajectory colored by velocity
ax = axes[0, 0]
scatter = ax.scatter(pos_x, pos_y, c=velocity, s=2, alpha=0.4, cmap='cool', vmin=0, vmax=30)
ax.set_xlabel('X (cm)', fontsize=10)
ax.set_ylabel('Y (cm)', fontsize=10)
ax.set_title('Exploration Trajectory\n(colored by velocity)', fontsize=11, fontweight='bold')
ax.set_aspect('equal')
cbar = plt.colorbar(scatter, ax=ax, label='Velocity (cm/s)')

# Plot 2-4: Spike locations for three different units
for plot_idx, unit_id in enumerate(selected_units[:3]):
    ax = axes.flatten()[plot_idx + 1]

    spike_times = spikes[unit_id]
    spike_times_array = spike_times.t
    spike_pos_x = np.interp(spike_times_array, position.t, pos_x)
    spike_pos_y = np.interp(spike_times_array, position.t, pos_y)

    ax.scatter(pos_x, pos_y, s=1, alpha=0.15, c='lightgray', label='Path')
    ax.scatter(spike_pos_x, spike_pos_y, s=40, alpha=0.6, c='red', marker='x', linewidth=1.5)

    ax.set_xlabel('X (cm)', fontsize=10)
    ax.set_ylabel('Y (cm)', fontsize=10)
    ax.set_title(f'Unit {unit_id} Spikes (n={len(spike_times)})', fontsize=11, fontweight='bold')
    ax.set_aspect('equal')

fig.suptitle('Spatial Distribution of Spikes', fontsize=13, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '02_spike_trajectories.png'), dpi=150, bbox_inches='tight')
print("Saved: 02_spike_trajectories.png")
plt.close()

# %% [markdown]
# ## Compute 2D Autocorrelograms (Grid Cell Signature)

# %%
fig, axes = plt.subplots(1, 3, figsize=(13, 4))

for idx, unit_id in enumerate(selected_units):
    ax = axes[idx]

    rate_map = firing_maps[unit_id]['rate_map']

    # Compute 2D autocorrelation
    rate_norm = (rate_map - rate_map.mean()) / (rate_map.std() + 1e-8)

    # 2D autocorrelation using FFT for efficiency
    from scipy.signal import correlate2d
    autocorr = correlate2d(rate_norm, rate_norm, mode='same')
    autocorr = autocorr / autocorr.max()

    im = ax.imshow(autocorr, origin='lower', cmap='RdBu_r', vmin=-0.5, vmax=1.0)
    ax.set_title(f'Unit {unit_id} Autocorrelation', fontsize=11, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])

    # Draw center circle
    center = autocorr.shape[0] // 2
    circle = Circle((center, center), 8, fill=False, edgecolor='yellow', linewidth=1.5, alpha=0.7)
    ax.add_patch(circle)

    plt.colorbar(im, ax=ax, label='Correlation', shrink=0.8)

fig.suptitle('2D Autocorrelograms\n(Grid cells show 6-fold symmetry)', fontsize=13, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '03_autocorrelograms.png'), dpi=150, bbox_inches='tight')
print("Saved: 03_autocorrelograms.png")
plt.close()

# %% [markdown]
# ## Compute Spatial Information

# %%
# Compute spatial information for each unit
spatial_info = {}
mean_rates = {}

for unit_id in firing_maps.keys():
    rate_map = firing_maps[unit_id]['rate_map']
    occupancy = firing_maps[unit_id]['occupancy']

    # Mean firing rate
    mean_rate = rate_map.mean()
    mean_rates[unit_id] = mean_rate

    # Spatial information (bits/spike)
    # SI = sum(p_i * r_i * log2(r_i / mean_rate))
    # where p_i is occupancy fraction, r_i is rate in bin i

    p_i = occupancy / occupancy.sum()

    with np.errstate(divide='ignore', invalid='ignore'):
        log_ratio = np.log2((rate_map + 1e-8) / (mean_rate + 1e-8))
        si = np.sum(p_i * rate_map * log_ratio)
        si = max(0, si)

    spatial_info[unit_id] = si

# Plot analysis results
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Histogram of spatial information
ax = axes[0]
si_values = list(spatial_info.values())
ax.bar(range(len(si_values)), si_values, color='steelblue', edgecolor='black', alpha=0.7)
ax.set_xlabel('Unit ID', fontsize=10)
ax.set_ylabel('Spatial Information (bits/spike)', fontsize=10)
ax.set_title('Spatial Information Across MEC Units', fontsize=11, fontweight='bold')
ax.axhline(np.mean(si_values), color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Mean: {np.mean(si_values):.2f}')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Mean firing rate vs spatial information
ax = axes[1]
rates = list(mean_rates.values())
ax.scatter(rates, si_values, s=100, alpha=0.7, color='steelblue', edgecolor='black', linewidth=1.5)
ax.set_xlabel('Mean Firing Rate (Hz)', fontsize=10)
ax.set_ylabel('Spatial Information (bits/spike)', fontsize=10)
ax.set_title('Spatial Information vs Firing Rate', fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.3)

# Add correlation if enough units
if len(rates) > 2:
    corr = np.corrcoef(rates, si_values)[0, 1]
    ax.text(0.05, 0.95, f'r = {corr:.2f}', transform=ax.transAxes,
            fontsize=11, verticalalignment='top', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '04_spatial_information.png'), dpi=150, bbox_inches='tight')
print("Saved: 04_spatial_information.png")
plt.close()

# %% [markdown]
# ## Summary Statistics and Findings

# %%
print("\n" + "="*70)
print("GRID CELL ANALYSIS SUMMARY")
print("="*70)

print(f"\n📊 RECORDING SESSION")
print(f"  Subject: sub-11084")
print(f"  Session: ses-28020501")
print(f"  Duration: {position.t[-1]:.1f} s ({position.t[-1]/60:.1f} min)")
print(f"  Arena: {x_max - x_min:.1f} × {y_max - y_min:.1f} cm")

print(f"\n🧠 UNITS RECORDED")
print(f"  Total units: {len(spikes)}")
print(f"  Units analyzed: {len(firing_maps)}")
for unit_id in firing_maps.keys():
    print(f"    Unit {unit_id}: {firing_maps[unit_id]['n_spikes']} spikes, "
          f"SI = {spatial_info[unit_id]:.3f} bits/spike")

print(f"\n📐 SPATIAL COVERAGE")
print(f"  Bin size: {bin_size} cm")
print(f"  Grid: {n_bins_x} × {n_bins_y} = {n_bins_x * n_bins_y} bins")
print(f"  Mean occupancy: {occupancy.mean():.3f} s/bin")

print(f"\n📈 SPATIAL INFORMATION (bits/spike)")
si_values = list(spatial_info.values())
print(f"  Mean: {np.mean(si_values):.3f}")
print(f"  Median: {np.median(si_values):.3f}")
print(f"  Range: [{np.min(si_values):.3f}, {np.max(si_values):.3f}]")

print(f"\n💥 MEAN FIRING RATES (Hz)")
rates = list(mean_rates.values())
print(f"  Mean: {np.mean(rates):.2f}")
print(f"  Median: {np.median(rates):.2f}")
print(f"  Range: [{np.min(rates):.2f}, {np.max(rates):.2f}]")

print("\n" + "="*70)

# %% [markdown]
# ## Grid Cell Theory and Interpretation
#
# Grid cells in the medial entorhinal cortex represent a fundamental component
# of the brain's spatial navigation system. Their defining characteristic is a
# hexagonal lattice of firing fields that tiles the environment, with consistent
# spacing and orientation across the explored space.
#
# **Key properties of grid cells:**
#
# - **Hexagonal organization**: Firing fields are arranged in a triangular lattice
#   with approximate 6-fold rotational symmetry
#
# - **Scale invariance**: Different grid cells express different field spacings
#   (typically 30–250 cm), forming a logarithmic hierarchy
#
# - **Conjunctive coding**: Grid cell firing is modulated by head direction and
#   running speed, providing egocentric spatial information
#
# - **Population coding**: Ensembles of grid cells with different scales and phases
#   form a redundant code that can uniquely identify any location
#
# **Functional significance:**
#
# Grid cells are thought to provide a metric foundation for spatial cognition,
# supporting vector-based path integration and distance estimation. Combined with
# hippocampal place cells, grid cells enable efficient spatial learning and memory.
#
# **From this analysis:**
#
# We observed spatial modulation in MEC unit firing, with variation in field
# structures and firing rates. The autocorrelation analysis reveals the potential
# periodic structure underlying grid cell representations. A complete grid cell
# analysis would involve: (1) quantifying grid periodicity via spectral analysis,
# (2) measuring grid spacing and orientation, (3) correlating these properties
# across simultaneously recorded populations, and (4) examining how grid
# properties change across behavioral states or during learning.

io.close()

print("\n✅ Analysis complete!")
