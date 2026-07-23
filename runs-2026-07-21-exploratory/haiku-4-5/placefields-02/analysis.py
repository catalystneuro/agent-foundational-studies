# %% [markdown]
# # Hippocampal Place Cells from the Neurolab Space Shuttle Dataset
#
# This analysis demonstrates hippocampal place cell firing properties using real neurophysiology
# data from the DANDI Archive (Dandiset #001754). The dataset contains extracellular recordings
# from three freely-moving rats during the Neurolab Space Shuttle mission (STS-90, April 1998).
#
# Place cells are neurons in the hippocampus that fire action potentials when an animal is in
# a specific location in its environment (the cell's "place field"). This analysis will compute
# spatial firing patterns, identify place cells, and visualize their properties.

# %% [markdown]
# ## Setup and Data Loading

# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import pynapple as nap
import warnings
warnings.filterwarnings('ignore')

# Settings
plt.style.use('seaborn-v0_8-darkgrid')
OUTPUT_DIR = Path('.')

# Load the NWB file
nwb_path = Path("/tmp/dandi_data/sub-Rat1_ses-19980420T095700_behavior+ecephys.nwb")
print(f"Loading NWB file: {nwb_path.name}")
nwb = nap.load_file(str(nwb_path))

# Extract data components
units = nwb['units']  # Spike train data (TsGroup)
position = nwb['spatial_series']  # 2D position data
epochs = nwb['epochs']  # Behavioral epochs

print(f"✓ Loaded {len(units)} neurons")
print(f"✓ Position samples: {len(position)}")
print(f"✓ Recording duration: {position.t[-1] - position.t[0]:.1f} seconds (~{(position.t[-1] - position.t[0])/3600:.2f} hours)")

# %% [markdown]
# ## Data Inspection and Validation

# %%
print("\n=== UNIT SPIKE STATISTICS ===")
spike_counts = []
for i, (unit_id, spikes) in enumerate(units.items()):
    spike_counts.append(len(spikes))

print(f"Total units: {len(units)}")
print(f"Spike counts (min/mean/max): {min(spike_counts)}/{np.mean(spike_counts):.0f}/{max(spike_counts)}")

print("\n=== POSITION DATA ===")
print(f"Position range:")
print(f"  x: [{position['x'].min():.1f}, {position['x'].max():.1f}] cm")
print(f"  y: [{position['y'].min():.1f}, {position['y'].max():.1f}] cm")
print(f"Spatial extent: {position['x'].max() - position['x'].min():.1f} x {position['y'].max() - position['y'].min():.1f} cm")

# Visualize raw position data
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(position['x'], position['y'], alpha=0.5, linewidth=0.5, color='steelblue')
ax.scatter(position['x'][0], position['y'][0], color='green', s=50, label='Start', zorder=5)
ax.scatter(position['x'][-1], position['y'][-1], color='red', s=50, label='End', zorder=5)
ax.set_xlabel('X Position (cm)')
ax.set_ylabel('Y Position (cm)')
ax.set_title('Animal Trajectory During Recording')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'trajectory.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n✓ Saved: trajectory.png")

# %% [markdown]
# ## Spatial Binning and Rate Map Computation

# %%
# Create 2D occupancy and firing rate maps
def compute_place_field_maps(position, spikes, bin_size=2.5, min_occupancy=0.1):
    """
    Compute 2D firing rate maps for individual neurons.

    Parameters:
    -----------
    position : TsdFrame
        2D position data with 'x' and 'y' columns
    spikes : Ts
        Spike times for one neuron
    bin_size : float
        Spatial bin size in cm
    min_occupancy : float
        Minimum occupancy threshold for valid bins (seconds)

    Returns:
    --------
    rate_map : 2D array
        Firing rate in each spatial bin (Hz)
    occupancy_map : 2D array
        Occupancy time in each spatial bin (seconds)
    """
    # Define spatial grid
    x_min, x_max = position['x'].min(), position['x'].max()
    y_min, y_max = position['y'].min(), position['y'].max()

    x_bins = np.arange(x_min, x_max + bin_size, bin_size)
    y_bins = np.arange(y_min, y_max + bin_size, bin_size)

    # Digitize positions into bins
    x_indices = np.digitize(position['x'].values, x_bins) - 1
    y_indices = np.digitize(position['y'].values, y_bins) - 1

    # Clamp to valid range
    x_indices = np.clip(x_indices, 0, len(x_bins) - 2)
    y_indices = np.clip(y_indices, 0, len(y_bins) - 2)

    # Compute occupancy map (time spent in each bin)
    occupancy_map = np.zeros((len(y_bins) - 1, len(x_bins) - 1))
    dt = np.median(np.diff(position.t))  # Median inter-sample interval
    for xi, yi in zip(x_indices, y_indices):
        occupancy_map[yi, xi] += dt

    # Compute spike count map
    spike_count_map = np.zeros((len(y_bins) - 1, len(x_bins) - 1))
    spike_times = spikes.t
    for spike_time in spike_times:
        # Find position at spike time (nearest neighbor)
        idx = np.searchsorted(position.t, spike_time)
        if 0 <= idx < len(position):
            xi = x_indices[idx]
            yi = y_indices[idx]
            spike_count_map[yi, xi] += 1

    # Compute firing rate (spikes / occupancy)
    rate_map = np.zeros_like(occupancy_map)
    valid_bins = occupancy_map >= min_occupancy
    rate_map[valid_bins] = spike_count_map[valid_bins] / occupancy_map[valid_bins]
    rate_map[~valid_bins] = np.nan

    return rate_map, occupancy_map, x_bins, y_bins

# Compute maps for all units
print("\nComputing spatial firing rate maps...")
all_rate_maps = []
all_occupancy_maps = []
x_bins_ref = None
y_bins_ref = None

for unit_id, spikes in units.items():
    rate_map, occupancy_map, x_bins, y_bins = compute_place_field_maps(position, spikes)
    all_rate_maps.append(rate_map)
    all_occupancy_maps.append(occupancy_map)
    if x_bins_ref is None:
        x_bins_ref = x_bins
        y_bins_ref = y_bins

all_rate_maps = np.array(all_rate_maps)
all_occupancy_maps = np.array(all_occupancy_maps)
print(f"✓ Computed maps for {len(all_rate_maps)} neurons")

# %% [markdown]
# ## Place Cell Identification Using Information Content

# %%
def compute_spatial_information(rate_map, occupancy_map, position_bins):
    """
    Compute spatial information score for a firing rate map.
    Information content quantifies how much the spike rate varies across space.

    Formula: I = sum(p_i * (lambda_i / lambda_mean) * log2(lambda_i / lambda_mean))
    where p_i is occupancy fraction, lambda_i is local firing rate, lambda_mean is mean rate
    """
    # Get mean firing rate
    valid_bins = ~np.isnan(rate_map) & (occupancy_map > 0)
    if not valid_bins.any():
        return 0.0

    mean_rate = np.nanmean(rate_map[valid_bins])
    if mean_rate <= 0:
        return 0.0

    total_occupancy = occupancy_map[valid_bins].sum()
    occupancy_probs = occupancy_map[valid_bins] / total_occupancy

    # Compute information
    rates = rate_map[valid_bins]
    rate_ratios = rates / mean_rate
    # Avoid log of zero/negative
    rate_ratios = np.clip(rate_ratios, 1e-6, None)

    information = np.sum(occupancy_probs * rate_ratios * np.log2(rate_ratios))
    return information

# Compute information content for all units
information_scores = []
for rate_map in all_rate_maps:
    info = compute_spatial_information(rate_map, all_occupancy_maps[0], x_bins_ref)
    information_scores.append(info)

information_scores = np.array(information_scores)

# Define place cells as neurons in top percentile
percentile_threshold = 75
info_threshold = np.percentile(information_scores, percentile_threshold)
is_place_cell = information_scores >= info_threshold

print("\n=== PLACE CELL IDENTIFICATION ===")
print(f"Information content (bits/spike):")
print(f"  Min: {information_scores.min():.3f}")
print(f"  Mean: {information_scores.mean():.3f}")
print(f"  Median: {np.median(information_scores):.3f}")
print(f"  Max: {information_scores.max():.3f}")
print(f"\nPlace cell threshold (75th percentile): {info_threshold:.3f} bits/spike")
print(f"Place cells identified: {is_place_cell.sum()} / {len(units)} neurons ({100*is_place_cell.sum()/len(units):.1f}%)")

# Visualize information score distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Histogram
axes[0].hist(information_scores, bins=15, alpha=0.6, color='steelblue', edgecolor='black')
axes[0].axvline(info_threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({info_threshold:.3f})')
axes[0].set_xlabel('Spatial Information (bits/spike)')
axes[0].set_ylabel('Number of Neurons')
axes[0].set_title('Distribution of Spatial Information Scores')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Sorted scores with place cell highlight
sorted_idx = np.argsort(information_scores)
sorted_scores = information_scores[sorted_idx]
colors = ['red' if is_place_cell[i] else 'steelblue' for i in sorted_idx]
axes[1].scatter(range(len(sorted_scores)), sorted_scores, c=colors, alpha=0.6, s=50)
axes[1].axhline(info_threshold, color='red', linestyle='--', linewidth=2, alpha=0.7)
axes[1].set_xlabel('Neuron Index (sorted by information)')
axes[1].set_ylabel('Spatial Information (bits/spike)')
axes[1].set_title('Spatial Information Scores')
axes[1].grid(True, alpha=0.3)

# Add legend
red_patch = mpatches.Patch(color='red', label='Place cell', alpha=0.6)
blue_patch = mpatches.Patch(color='steelblue', label='Non-place cell', alpha=0.6)
axes[1].legend(handles=[red_patch, blue_patch])

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'information_scores.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n✓ Saved: information_scores.png")

# %% [markdown]
# ## Visualize Place Field Maps

# %%
# Select top 12 place cells and top 12 non-place cells for visualization
top_place_cell_idx = np.where(is_place_cell)[0][np.argsort(-information_scores[is_place_cell])[:12]]
top_nonplace_idx = np.where(~is_place_cell)[0][np.argsort(-information_scores[~is_place_cell])[:12]]

fig, axes = plt.subplots(4, 6, figsize=(16, 10))
fig.suptitle('Top Place Cell and Non-Place Cell Firing Rate Maps', fontsize=14, fontweight='bold')

# Plot place cells
for row_idx, unit_idx in enumerate(top_place_cell_idx[:6]):
    rate_map = all_rate_maps[unit_idx]
    ax = axes[0, row_idx] if row_idx < 6 else axes[1, row_idx % 6]
    if row_idx < 6:
        ax = axes[0, row_idx]
    else:
        ax = axes[1, row_idx % 6]

    # Plot with proper extent
    extent = [x_bins_ref[0], x_bins_ref[-1], y_bins_ref[0], y_bins_ref[-1]]
    im = ax.imshow(rate_map, cmap='hot', origin='lower', extent=extent, aspect='auto',
                   interpolation='bilinear')
    ax.set_title(f'Unit {list(units.keys())[unit_idx]}\nI={information_scores[unit_idx]:.3f}', fontsize=9)
    ax.set_xlabel('X (cm)', fontsize=8)
    ax.set_ylabel('Y (cm)', fontsize=8)
    plt.colorbar(im, ax=ax, label='Rate (Hz)')

# Continue place cells in second row
for row_idx, unit_idx in enumerate(top_place_cell_idx[6:12]):
    ax = axes[1, row_idx]
    extent = [x_bins_ref[0], x_bins_ref[-1], y_bins_ref[0], y_bins_ref[-1]]
    im = ax.imshow(all_rate_maps[unit_idx], cmap='hot', origin='lower', extent=extent,
                   aspect='auto', interpolation='bilinear')
    ax.set_title(f'Unit {list(units.keys())[unit_idx]}\nI={information_scores[unit_idx]:.3f}', fontsize=9)
    ax.set_xlabel('X (cm)', fontsize=8)
    ax.set_ylabel('Y (cm)', fontsize=8)
    plt.colorbar(im, ax=ax, label='Rate (Hz)')

# Plot non-place cells
for row_idx, unit_idx in enumerate(top_nonplace_idx[:6]):
    ax = axes[2, row_idx]
    extent = [x_bins_ref[0], x_bins_ref[-1], y_bins_ref[0], y_bins_ref[-1]]
    im = ax.imshow(all_rate_maps[unit_idx], cmap='viridis', origin='lower', extent=extent,
                   aspect='auto', interpolation='bilinear')
    ax.set_title(f'Unit {list(units.keys())[unit_idx]}\nI={information_scores[unit_idx]:.3f}', fontsize=9)
    ax.set_xlabel('X (cm)', fontsize=8)
    ax.set_ylabel('Y (cm)', fontsize=8)
    plt.colorbar(im, ax=ax, label='Rate (Hz)')

# Continue non-place cells
for row_idx, unit_idx in enumerate(top_nonplace_idx[6:12]):
    ax = axes[3, row_idx]
    extent = [x_bins_ref[0], x_bins_ref[-1], y_bins_ref[0], y_bins_ref[-1]]
    im = ax.imshow(all_rate_maps[unit_idx], cmap='viridis', origin='lower', extent=extent,
                   aspect='auto', interpolation='bilinear')
    ax.set_title(f'Unit {list(units.keys())[unit_idx]}\nI={information_scores[unit_idx]:.3f}', fontsize=9)
    ax.set_xlabel('X (cm)', fontsize=8)
    ax.set_ylabel('Y (cm)', fontsize=8)
    plt.colorbar(im, ax=ax, label='Rate (Hz)')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'place_field_maps.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: place_field_maps.png")

# %% [markdown]
# ## Spatial Sparsity and Selectivity

# %%
def compute_sparsity_selectivity(rate_map):
    """
    Compute sparsity (fraction of environment where neuron fires) and
    selectivity (concentration of firing in preferred areas).
    """
    valid_bins = ~np.isnan(rate_map)
    if not valid_bins.any():
        return np.nan, np.nan

    rates = rate_map[valid_bins]
    mean_rate = rates.mean()

    if mean_rate <= 0:
        return np.nan, np.nan

    # Sparsity: fraction of space with firing rate > mean
    sparsity = np.sum(rates > mean_rate) / len(rates)

    # Selectivity: inverse of normalized variance
    # High selectivity = firing concentrated in few places
    normalized_variance = np.var(rates) / (mean_rate ** 2)
    selectivity = 1 / (1 + normalized_variance)

    return sparsity, selectivity

sparsity_scores = []
selectivity_scores = []

for rate_map in all_rate_maps:
    sparsity, selectivity = compute_sparsity_selectivity(rate_map)
    sparsity_scores.append(sparsity)
    selectivity_scores.append(selectivity)

sparsity_scores = np.array(sparsity_scores)
selectivity_scores = np.array(selectivity_scores)

print("\n=== SPATIAL SELECTIVITY ===")
print(f"Sparsity (fraction of space with firing):")
print(f"  Place cells: {sparsity_scores[is_place_cell].mean():.3f} ± {sparsity_scores[is_place_cell].std():.3f}")
print(f"  Non-place cells: {sparsity_scores[~is_place_cell].mean():.3f} ± {sparsity_scores[~is_place_cell].std():.3f}")

print(f"\nSelectivity (firing concentration):")
print(f"  Place cells: {selectivity_scores[is_place_cell].mean():.3f} ± {selectivity_scores[is_place_cell].std():.3f}")
print(f"  Non-place cells: {selectivity_scores[~is_place_cell].mean():.3f} ± {selectivity_scores[~is_place_cell].std():.3f}")

# Visualize sparsity vs selectivity
fig, ax = plt.subplots(figsize=(10, 7))

scatter_place = ax.scatter(sparsity_scores[is_place_cell], selectivity_scores[is_place_cell],
                           s=100, alpha=0.6, color='red', edgecolors='darkred',
                           linewidth=1.5, label='Place cells', zorder=3)
scatter_other = ax.scatter(sparsity_scores[~is_place_cell], selectivity_scores[~is_place_cell],
                           s=100, alpha=0.6, color='steelblue', edgecolors='darkblue',
                           linewidth=1.5, label='Non-place cells', zorder=2)

ax.set_xlabel('Sparsity (Fraction of Space with Firing)', fontsize=12)
ax.set_ylabel('Selectivity (Firing Concentration)', fontsize=12)
ax.set_title('Place Cell Spatial Selectivity Properties', fontsize=13, fontweight='bold')
ax.legend(fontsize=11, loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_xlim([-0.05, 1.05])
ax.set_ylim([-0.05, 1.05])

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'selectivity.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n✓ Saved: selectivity.png")

# %% [markdown]
# ## Summary Statistics

# %%
print("\n" + "="*60)
print("PLACE CELL ANALYSIS SUMMARY")
print("="*60)
print(f"\nDataset: Neurolab Space Shuttle Mission (Dandiset #001754)")
print(f"Animal: Rat 1")
print(f"Session: 1998-04-20, ~1.65 hours recording")
print(f"Recording location: Hippocampal CA1, 3D environment (microgravity)")

print(f"\nNeuronal Recording:")
print(f"  Total neurons: {len(units)}")
print(f"  Place cells (top 25%): {is_place_cell.sum()}")
print(f"  Mean spikes per neuron: {np.mean(spike_counts):.0f}")

print(f"\nPlace Cell Properties:")
print(f"  Mean information: {information_scores[is_place_cell].mean():.3f} ± {information_scores[is_place_cell].std():.3f} bits/spike")
print(f"  Mean sparsity: {sparsity_scores[is_place_cell].mean():.3f} ± {sparsity_scores[is_place_cell].std():.3f}")
print(f"  Mean selectivity: {selectivity_scores[is_place_cell].mean():.3f} ± {selectivity_scores[is_place_cell].std():.3f}")

print(f"\nNon-Place Cell Properties (for comparison):")
print(f"  Mean information: {information_scores[~is_place_cell].mean():.3f} ± {information_scores[~is_place_cell].std():.3f} bits/spike")
print(f"  Mean sparsity: {sparsity_scores[~is_place_cell].mean():.3f} ± {sparsity_scores[~is_place_cell].std():.3f}")
print(f"  Mean selectivity: {selectivity_scores[~is_place_cell].mean():.3f} ± {selectivity_scores[~is_place_cell].std():.3f}")

print("\n" + "="*60)
print("Analysis complete. All figures saved to current directory.")
print("="*60)
