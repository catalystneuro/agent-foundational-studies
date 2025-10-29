# %% [markdown]
# # Place Cell Identification
#
# This script identifies neurons with clear spatial receptive fields (place cells).
# We will:
# 1. Compute spatial firing rate maps for each neuron
# 2. Calculate place field properties (peak rate, spatial information, etc.)
# 3. Identify well-tuned place cells for phase precession analysis

# %%
import lindi
import pynwb
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, stats
from pynwb import NWBHDF5IO
from tqdm import tqdm

# %%
print("Loading NWB file...")
lindi_url = "https://lindi.neurosift.org/dandi/dandisets/000059/assets/931b05e6-9123-4c66-a9cf-49de6cec8086/nwb.lindi.json"
local_cache = lindi.LocalCache()
f = lindi.LindiH5pyFile.from_lindi_file(lindi_url, local_cache=local_cache)
io = NWBHDF5IO(file=f, mode='r')
nwb = io.read()

# %%
print("Loading position and speed data...")

# Get position data
position = nwb.processing["behavior"]["SubjectPosition"]
spatial_series = position.spatial_series["SpatialSeries"]
pos_data = spatial_series.data[:]
pos_timestamps = spatial_series.timestamps[:]

# Get speed data
speed = nwb.processing["behavior"]["SubjectSpeed"]
speed_data = speed.data[:]
speed_timestamps = speed.timestamps[:]

# Filter valid positions
valid_mask = ~np.isnan(pos_data[:, 0]) & ~np.isnan(pos_data[:, 1])
pos_data = pos_data[valid_mask]
pos_timestamps = pos_timestamps[valid_mask]
speed_data = speed_data[valid_mask]

print(f"Position data: {len(pos_data)} valid samples at ~40 Hz")
print(f"Position range: X=[{pos_data[:, 0].min():.1f}, {pos_data[:, 0].max():.1f}] cm")
print(f"                Y=[{pos_data[:, 1].min():.1f}, {pos_data[:, 1].max():.1f}] cm")

# %% [markdown]
# ## Define helper functions for spatial analysis

# %%
def compute_occupancy_map(pos_x, pos_y, bin_size=2.5, sigma=1.5):
    """
    Compute occupancy map (time spent in each spatial bin).
    
    Parameters:
    -----------
    pos_x, pos_y : array
        Position coordinates in cm
    bin_size : float
        Spatial bin size in cm
    sigma : float
        Gaussian smoothing sigma in bins
        
    Returns:
    --------
    occupancy : 2D array
        Time spent in each bin (seconds)
    x_edges, y_edges : arrays
        Bin edges
    """
    # Create 2D histogram
    x_edges = np.arange(pos_x.min(), pos_x.max() + bin_size, bin_size)
    y_edges = np.arange(pos_y.min(), pos_y.max() + bin_size, bin_size)
    
    occupancy, _, _ = np.histogram2d(pos_x, pos_y, bins=[x_edges, y_edges])
    
    # Convert to seconds (assuming 40 Hz sampling)
    occupancy = occupancy / 40.0
    
    # Smooth
    occupancy = ndimage.gaussian_filter(occupancy, sigma=sigma)
    
    return occupancy, x_edges, y_edges


def compute_firing_rate_map(spike_times, pos_timestamps, pos_x, pos_y, 
                            bin_size=2.5, sigma=1.5, speed_threshold=2.0):
    """
    Compute spatial firing rate map for a neuron.
    
    Parameters:
    -----------
    spike_times : array
        Spike timestamps
    pos_timestamps : array
        Position timestamps
    pos_x, pos_y : arrays
        Position coordinates
    bin_size : float
        Spatial bin size in cm
    sigma : float
        Gaussian smoothing sigma
    speed_threshold : float
        Minimum speed in cm/s (to exclude stationary periods)
        
    Returns:
    --------
    rate_map : 2D array
        Firing rate in each spatial bin (Hz)
    """
    # Filter for movement periods
    moving_mask = speed_data > speed_threshold
    pos_x_moving = pos_x[moving_mask]
    pos_y_moving = pos_y[moving_mask]
    pos_t_moving = pos_timestamps[moving_mask]
    
    if len(pos_x_moving) < 100:
        return None, None, None
    
    # Compute occupancy
    occupancy, x_edges, y_edges = compute_occupancy_map(pos_x_moving, pos_y_moving, 
                                                         bin_size, sigma)
    
    # Find spike positions
    spike_pos_x = np.interp(spike_times, pos_t_moving, pos_x_moving)
    spike_pos_y = np.interp(spike_times, pos_t_moving, pos_y_moving)
    
    # Create spike count histogram
    spike_counts, _, _ = np.histogram2d(spike_pos_x, spike_pos_y, 
                                       bins=[x_edges, y_edges])
    
    # Smooth spike counts
    spike_counts = ndimage.gaussian_filter(spike_counts, sigma=sigma)
    
    # Compute firing rate (avoiding division by zero)
    rate_map = np.zeros_like(occupancy)
    valid_occ = occupancy > 0.1  # At least 0.1 seconds in bin
    rate_map[valid_occ] = spike_counts[valid_occ] / occupancy[valid_occ]
    
    return rate_map, x_edges, y_edges


def compute_spatial_information(rate_map, occupancy):
    """
    Compute spatial information content in bits/spike.
    
    This measures how much information about position is conveyed
    by a single spike.
    """
    # Normalize occupancy to probability
    total_time = np.sum(occupancy)
    if total_time == 0:
        return 0
        
    p_x = occupancy / total_time
    
    # Mean firing rate
    mean_rate = np.sum(rate_map * p_x)
    
    if mean_rate == 0:
        return 0
    
    # Spatial information
    spatial_info = 0
    for i in range(rate_map.shape[0]):
        for j in range(rate_map.shape[1]):
            if p_x[i, j] > 0 and rate_map[i, j] > 0:
                ratio = rate_map[i, j] / mean_rate
                spatial_info += p_x[i, j] * rate_map[i, j] * np.log2(ratio)
    
    spatial_info = spatial_info / mean_rate
    return spatial_info


def analyze_place_field(rate_map):
    """
    Analyze place field properties.
    
    Returns peak rate, field size, and coherence.
    """
    peak_rate = np.max(rate_map)
    
    # Field size: number of bins above 20% of peak
    threshold = 0.2 * peak_rate
    field_size = np.sum(rate_map > threshold)
    
    # Spatial coherence: correlation with smoothed version
    smoothed = ndimage.gaussian_filter(rate_map, sigma=1.0)
    flat_rate = rate_map.flatten()
    flat_smooth = smoothed.flatten()
    
    if np.std(flat_rate) > 0 and np.std(flat_smooth) > 0:
        coherence = np.corrcoef(flat_rate, flat_smooth)[0, 1]
    else:
        coherence = 0
    
    return peak_rate, field_size, coherence

# %% [markdown]
# ## Analyze all units to identify place cells

# %%
print("\nAnalyzing spatial tuning for all units...")

units = nwb.units
n_units = len(units)

# Storage for results
place_cell_metrics = {
    'unit_id': [],
    'peak_rate': [],
    'spatial_info': [],
    'field_size': [],
    'coherence': [],
    'n_spikes': [],
    'mean_rate': []
}

# Analyze each unit
for unit_idx in tqdm(range(n_units), desc="Processing units"):
    spike_times = units["spike_times"][unit_idx]
    
    # Filter spikes to recording period
    valid_spikes = spike_times[(spike_times >= pos_timestamps[0]) & 
                               (spike_times <= pos_timestamps[-1])]
    
    if len(valid_spikes) < 50:  # Skip units with too few spikes
        continue
    
    # Compute firing rate map
    rate_map, x_edges, y_edges = compute_firing_rate_map(
        valid_spikes, pos_timestamps, pos_data[:, 0], pos_data[:, 1]
    )
    
    if rate_map is None:
        continue
    
    # Compute occupancy for information calculation
    moving_mask = speed_data > 2.0
    occupancy, _, _ = compute_occupancy_map(
        pos_data[moving_mask, 0], pos_data[moving_mask, 1]
    )
    
    # Calculate metrics
    spatial_info = compute_spatial_information(rate_map, occupancy)
    peak_rate, field_size, coherence = analyze_place_field(rate_map)
    mean_rate = len(valid_spikes) / (pos_timestamps[-1] - pos_timestamps[0])
    
    # Store results
    place_cell_metrics['unit_id'].append(unit_idx)
    place_cell_metrics['peak_rate'].append(peak_rate)
    place_cell_metrics['spatial_info'].append(spatial_info)
    place_cell_metrics['field_size'].append(field_size)
    place_cell_metrics['coherence'].append(coherence)
    place_cell_metrics['n_spikes'].append(len(valid_spikes))
    place_cell_metrics['mean_rate'].append(mean_rate)

print(f"\nAnalyzed {len(place_cell_metrics['unit_id'])} / {n_units} units")

# %% [markdown]
# ## Identify well-tuned place cells

# %%
# Convert to arrays for easier manipulation
for key in place_cell_metrics:
    place_cell_metrics[key] = np.array(place_cell_metrics[key])

# Criteria for place cells:
# 1. Peak rate > 1 Hz
# 2. Spatial information > 0.5 bits/spike
# 3. Coherence > 0.6

place_cell_mask = (
    (place_cell_metrics['peak_rate'] > 1.0) &
    (place_cell_metrics['spatial_info'] > 0.5) &
    (place_cell_metrics['coherence'] > 0.6)
)

n_place_cells = np.sum(place_cell_mask)
place_cell_indices = place_cell_metrics['unit_id'][place_cell_mask]

print(f"\nIdentified {n_place_cells} place cells out of {len(place_cell_metrics['unit_id'])} analyzed units")
print(f"Percentage: {100 * n_place_cells / len(place_cell_metrics['unit_id']):.1f}%")

print(f"\nPlace cell properties:")
print(f"  Peak rate: {place_cell_metrics['peak_rate'][place_cell_mask].mean():.2f} ± {place_cell_metrics['peak_rate'][place_cell_mask].std():.2f} Hz")
print(f"  Spatial info: {place_cell_metrics['spatial_info'][place_cell_mask].mean():.2f} ± {place_cell_metrics['spatial_info'][place_cell_mask].std():.2f} bits/spike")
print(f"  Coherence: {place_cell_metrics['coherence'][place_cell_mask].mean():.2f} ± {place_cell_metrics['coherence'][place_cell_mask].std():.2f}")

# %% [markdown]
# ## Visualize example place cells

# %%
# Select top 6 place cells by spatial information
top_indices = np.argsort(place_cell_metrics['spatial_info'][place_cell_mask])[-6:]
example_cells = place_cell_indices[top_indices]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for plot_idx, unit_idx in enumerate(example_cells):
    spike_times = units["spike_times"][int(unit_idx)]
    valid_spikes = spike_times[(spike_times >= pos_timestamps[0]) & 
                               (spike_times <= pos_timestamps[-1])]
    
    rate_map, x_edges, y_edges = compute_firing_rate_map(
        valid_spikes, pos_timestamps, pos_data[:, 0], pos_data[:, 1]
    )
    
    ax = axes[plot_idx]
    im = ax.imshow(rate_map.T, origin='lower', aspect='auto', cmap='jet',
                   extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])
    
    # Get metrics for this cell
    metrics_idx = np.where(place_cell_metrics['unit_id'] == unit_idx)[0][0]
    peak = place_cell_metrics['peak_rate'][metrics_idx]
    info = place_cell_metrics['spatial_info'][metrics_idx]
    
    ax.set_title(f'Unit {unit_idx}\nPeak: {peak:.1f} Hz, Info: {info:.2f} bits/sp')
    ax.set_xlabel('X position (cm)')
    ax.set_ylabel('Y position (cm)')
    plt.colorbar(im, ax=ax, label='Firing rate (Hz)')

plt.tight_layout()
plt.savefig('place_cell_examples.png', dpi=150, bbox_inches='tight')
print("\nFigure saved: place_cell_examples.png")

# %% [markdown]
# ## Summary statistics

# %%
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Spatial information distribution
ax = axes[0, 0]
ax.hist(place_cell_metrics['spatial_info'], bins=30, alpha=0.5, label='All')
ax.hist(place_cell_metrics['spatial_info'][place_cell_mask], bins=30, alpha=0.5, label='Place cells')
ax.axvline(0.5, color='r', linestyle='--', label='Threshold')
ax.set_xlabel('Spatial Information (bits/spike)')
ax.set_ylabel('Count')
ax.set_title('Spatial Information Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Peak rate vs spatial info
ax = axes[0, 1]
ax.scatter(place_cell_metrics['peak_rate'], place_cell_metrics['spatial_info'], 
          alpha=0.3, s=10, label='Non-place cells')
ax.scatter(place_cell_metrics['peak_rate'][place_cell_mask], 
          place_cell_metrics['spatial_info'][place_cell_mask],
          alpha=0.7, s=20, color='red', label='Place cells')
ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
ax.axvline(1.0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Peak Firing Rate (Hz)')
ax.set_ylabel('Spatial Information (bits/spike)')
ax.set_title('Peak Rate vs Spatial Information')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Coherence distribution
ax = axes[1, 0]
ax.hist(place_cell_metrics['coherence'], bins=30, alpha=0.5, label='All')
ax.hist(place_cell_metrics['coherence'][place_cell_mask], bins=30, alpha=0.5, label='Place cells')
ax.axvline(0.6, color='r', linestyle='--', label='Threshold')
ax.set_xlabel('Spatial Coherence')
ax.set_ylabel('Count')
ax.set_title('Spatial Coherence Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Field size distribution
ax = axes[1, 1]
ax.hist(place_cell_metrics['field_size'], bins=30, alpha=0.5, label='All')
ax.hist(place_cell_metrics['field_size'][place_cell_mask], bins=30, alpha=0.5, label='Place cells')
ax.set_xlabel('Place Field Size (bins)')
ax.set_ylabel('Count')
ax.set_title('Place Field Size Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('place_cell_analysis.png', dpi=150, bbox_inches='tight')
print("Figure saved: place_cell_analysis.png")

# %%
# Save place cell indices for next analysis
np.save('place_cell_indices.npy', place_cell_indices)
np.save('place_cell_metrics.npy', place_cell_metrics)
print(f"\nSaved {len(place_cell_indices)} place cell indices for phase precession analysis")

print("\n" + "="*60)
print("PLACE CELL IDENTIFICATION COMPLETE")
print("="*60)
print(f"Total units analyzed: {len(place_cell_metrics['unit_id'])}")
print(f"Place cells identified: {n_place_cells}")
print(f"Best place cells selected for phase precession analysis")
print("="*60)
