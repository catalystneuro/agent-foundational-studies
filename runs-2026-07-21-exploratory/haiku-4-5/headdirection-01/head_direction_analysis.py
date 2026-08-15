# %% [markdown]
# # Head Direction Cells Analysis
#
# This notebook demonstrates head direction cells using extracellular recordings from freely moving mice
# performing open-field exploration and theta maze navigation. Head direction cells are neurons that fire
# maximally when the animal's head points in a specific direction, regardless of the animal's location.
#
# **Dataset**: DANDI:000003 - Yuta Buzsáki Lab rodent navigation recordings
# **Data Type**: Extracellular recordings + position tracking from mice in 2D environments

# %% [markdown]
# ## Setup and Data Loading

# %%
import h5py
from pynwb import NWBHDF5IO
import remfile
import pynapple as nap
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm
import os

# Configure matplotlib for headless plotting
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
os.environ['MPLBACKEND'] = 'Agg'

# Setup cache for remote file access
cache_dir = "/tmp/nwb_cache_dandi"
os.makedirs(cache_dir, exist_ok=True)

print("Initializing head direction cell analysis...")

# %% [markdown]
# ### Loading NWB Data from DANDI:000003
#
# We use remfile to stream the NWB file from AWS S3, avoiding full download of the 8GB file.

# %%
s3_url = "https://dandiarchive.s3.amazonaws.com/blobs/58c/537/58c53789-eec4-4080-ad3b-207cf2a1cac9"

print(f"Loading NWB file from DANDI:000003 (streaming)...")

disk_cache = remfile.DiskCache(cache_dir)
rem_file = remfile.File(s3_url, disk_cache=disk_cache)
h5_file = h5py.File(rem_file, "r")

io = NWBHDF5IO(file=h5_file)
nwbfile = io.read()

print(f"✓ NWB file loaded!")
print(f"  Identifier: {nwbfile.identifier}")
print(f"  Date: {nwbfile.session_start_time}")
print(f"  Description: {nwbfile.session_description}")
print(f"  Number of neurons: {len(nwbfile.units)}\n")

# %% [markdown]
# ## Data Extraction

# %%
print(f"Extracting spike times and position data...")

spikes = {}
for unit_id in range(len(nwbfile.units)):
    spike_times = nwbfile.units[unit_id]['spike_times'][:]
    if len(spike_times) > 0:
        spikes[unit_id] = nap.Ts(spike_times)

print(f"✓ Extracted {len(spikes)} neurons with spike data")

# Get position data - use subsampling for efficiency
pos0 = nwbfile.acquisition['position_sensor0']
times = nap.Ts(pos0.timestamps[:])
pos0_coords = pos0.data[:]
pos1_coords = nwbfile.acquisition['position_sensor1'].data[:]

print(f"✓ Position data: {pos0_coords.shape[0]} timepoints at {pos0.rate:.1f} Hz")

# %% [markdown]
# ## Head Direction Computation
#
# Head direction is computed from two sensors on the animal's head using atan2.

# %%
print(f"\nComputing head direction from dual sensors...")

head_vector = pos1_coords - pos0_coords
head_direction_rad = np.arctan2(head_vector[:, 1], head_vector[:, 0])
head_direction_rad = np.mod(head_direction_rad, 2 * np.pi)
head_direction = nap.Tsd(head_direction_rad, t=times)

print(f"✓ Head direction computed")
print(f"  Range: [{np.degrees(head_direction.values.min()):.1f}°, {np.degrees(head_direction.values.max()):.1f}°]")

# %% [markdown]
# ## Directional Tuning Analysis
#
# For each neuron, compute firing rate as a function of head direction.

# %%
print(f"\nComputing directional tuning curves...")

n_bins = 36
bin_edges = np.linspace(0, 2*np.pi, n_bins + 1)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

tuning_curves = {}
tuning_stats = {}

for unit_id in tqdm(spikes.keys(), desc="Processing neurons"):
    spike_times = spikes[unit_id]
    head_dir_at_spikes = head_direction.restrict(spike_times)

    spike_counts = np.zeros(n_bins)
    time_in_bin = np.zeros(n_bins)

    for i, (bin_start, bin_end) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        in_bin = (head_dir_at_spikes.values >= bin_start) & (head_dir_at_spikes.values < bin_end)
        spike_counts[i] = np.sum(in_bin)

        head_dir_in_bin = (head_direction.values >= bin_start) & (head_direction.values < bin_end)
        dt = 1.0 / pos0.rate
        time_in_bin[i] = np.sum(head_dir_in_bin) * dt

    firing_rates = np.divide(spike_counts, time_in_bin,
                             where=time_in_bin > 0,
                             out=np.zeros_like(spike_counts))

    tuning_curves[unit_id] = firing_rates

    # Compute metrics
    max_idx = np.argmax(firing_rates)
    preferred_dir = bin_centers[max_idx]
    max_firing_rate = firing_rates[max_idx]

    if max_firing_rate > 0:
        normalized_rates = firing_rates / (firing_rates.sum() + 1e-10)
        cos_sum = np.sum(normalized_rates * np.cos(bin_centers))
        sin_sum = np.sum(normalized_rates * np.sin(bin_centers))
        r = np.sqrt(cos_sum**2 + sin_sum**2)
    else:
        r = 0.0

    tuning_stats[unit_id] = {
        'preferred_direction': np.degrees(preferred_dir),
        'max_firing_rate': max_firing_rate,
        'directionality_index': r,
        'total_spikes': len(spike_times)
    }

print(f"✓ Tuning curves computed for {len(tuning_curves)} neurons")

# %% [markdown]
# ## Results Summary

# %%
print(f"\n{'='*80}")
print("HEAD DIRECTION CELL PROPERTIES")
print(f"{'='*80}\n")

pref_dirs = np.array([tuning_stats[uid]['preferred_direction'] for uid in tuning_curves.keys()])
tuning_strengths = np.array([tuning_stats[uid]['directionality_index'] for uid in tuning_curves.keys()])
max_rates = np.array([tuning_stats[uid]['max_firing_rate'] for uid in tuning_curves.keys()])

print(f"Unit | Preferred Dir (°) | Max Firing (Hz) | Tuning Strength")
print(f"-----|-------------------|-----------------|----------------")
for unit_id in sorted(spikes.keys()):
    s = tuning_stats[unit_id]
    print(f"{unit_id:4d} | {s['preferred_direction']:17.1f} | {s['max_firing_rate']:15.2f} | {s['directionality_index']:15.3f}")

print(f"\nPopulation Statistics:")
print(f"  Strong HD cells (DI > 0.3): {np.sum(tuning_strengths > 0.3)}/{len(tuning_strengths)}")
print(f"  Mean tuning strength: {np.mean(tuning_strengths):.3f}")
print(f"  Mean preferred direction: {np.mean(pref_dirs):.1f}°")
print(f"  Mean peak firing rate: {np.mean(max_rates):.2f} ± {np.std(max_rates):.2f} Hz")

# %% [markdown]
# ## Visualization 1: Polar Tuning Curves

# %%
print(f"\nGenerating visualizations...")

fig, axes = plt.subplots(3, 5, figsize=(16, 10), subplot_kw=dict(projection='polar'))
axes = axes.flatten()

for idx, (unit_id, firing_rates) in enumerate(tuning_curves.items()):
    ax = axes[idx]

    angles_plot = np.concatenate([bin_centers, [bin_centers[0]]])
    rates_plot = np.concatenate([firing_rates, [firing_rates[0]]])

    ax.plot(angles_plot, rates_plot, 'b-', linewidth=2)
    ax.fill(angles_plot, rates_plot, alpha=0.25)

    pref_dir = tuning_stats[unit_id]['preferred_direction']
    max_rate = tuning_stats[unit_id]['max_firing_rate']
    ax.plot(np.radians(pref_dir), max_rate, 'r*', markersize=15)

    ax.set_ylim(bottom=0)
    ax.set_title(f'Unit {unit_id}\n({tuning_stats[unit_id]["directionality_index"]:.2f})',
                 fontsize=10, pad=10)
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(-1)
    ax.grid(True, alpha=0.3)

for idx in range(len(tuning_curves), len(axes)):
    axes[idx].set_visible(False)

plt.suptitle('Head Direction Tuning Curves - Polar View (10° bins)',
             fontsize=14, y=0.98)
plt.tight_layout()
plt.savefig('tuning_curves_polar.png', dpi=150, bbox_inches='tight')
print("  ✓ tuning_curves_polar.png")
plt.close()

# %% [markdown]
# ## Visualization 2: Cartesian Tuning Curves

# %%
fig, axes = plt.subplots(3, 5, figsize=(16, 10))
axes = axes.flatten()

for idx, (unit_id, firing_rates) in enumerate(tuning_curves.items()):
    ax = axes[idx]

    angles_deg = np.degrees(bin_centers)
    ax.bar(angles_deg, firing_rates, width=8, alpha=0.7, color='steelblue', edgecolor='black')

    pref_dir = tuning_stats[unit_id]['preferred_direction']
    max_rate = tuning_stats[unit_id]['max_firing_rate']
    ax.axvline(pref_dir, color='red', linestyle='--', linewidth=2, alpha=0.7)

    ax.set_xlim(-10, 370)
    ax.set_xlabel('Head Direction (°)', fontsize=9)
    ax.set_ylabel('Firing Rate (Hz)', fontsize=9)
    ax.set_xticks([0, 90, 180, 270])
    ax.set_title(f'Unit {unit_id}\nDI: {tuning_stats[unit_id]["directionality_index"]:.2f}',
                 fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

for idx in range(len(tuning_curves), len(axes)):
    axes[idx].set_visible(False)

plt.suptitle('Head Direction Tuning Curves - Cartesian View', fontsize=14)
plt.tight_layout()
plt.savefig('tuning_curves_cartesian.png', dpi=150, bbox_inches='tight')
print("  ✓ tuning_curves_cartesian.png")
plt.close()

# %% [markdown]
# ## Visualization 3: Population Statistics

# %%
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

ax = axes[0, 0]
ax.hist(pref_dirs, bins=12, color='steelblue', edgecolor='black', alpha=0.7)
ax.set_xlabel('Preferred Direction (°)', fontsize=11)
ax.set_ylabel('Number of Neurons', fontsize=11)
ax.set_title('Preferred Direction Distribution', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

ax = axes[0, 1]
ax.hist(tuning_strengths, bins=8, color='coral', edgecolor='black', alpha=0.7)
ax.set_xlabel('Directionality Index', fontsize=11)
ax.set_ylabel('Number of Neurons', fontsize=11)
ax.set_title('Tuning Strength Distribution', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')
ax.axvline(0.3, color='red', linestyle='--', linewidth=2, alpha=0.7, label='HD threshold')
ax.legend()

ax = axes[1, 0]
ax.hist(max_rates, bins=8, color='lightgreen', edgecolor='black', alpha=0.7)
ax.set_xlabel('Maximum Firing Rate (Hz)', fontsize=11)
ax.set_ylabel('Number of Neurons', fontsize=11)
ax.set_title('Peak Firing Rate Distribution', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

ax = axes[1, 1]
scatter = ax.scatter(max_rates, tuning_strengths, s=100, alpha=0.6, c=pref_dirs,
                     cmap='hsv', edgecolor='black')
ax.set_xlabel('Maximum Firing Rate (Hz)', fontsize=11)
ax.set_ylabel('Directionality Index', fontsize=11)
ax.set_title('Firing Rate vs. Tuning Strength', fontsize=12)
ax.grid(True, alpha=0.3)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Preferred Direction (°)', fontsize=10)
ax.axhline(0.3, color='red', linestyle='--', linewidth=1, alpha=0.5)

plt.suptitle('Population-Level Statistics', fontsize=14)
plt.tight_layout()
plt.savefig('population_statistics.png', dpi=150, bbox_inches='tight')
print("  ✓ population_statistics.png")
plt.close()

# %% [markdown]
# ## Visualization 4: Behavioral Context

# %%
position = nap.TsdFrame(pos0_coords, t=times, columns=['x', 'y'])

fig = plt.figure(figsize=(15, 10))
gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

ax1 = fig.add_subplot(gs[0, :])
scatter = ax1.scatter(position.values[:, 0], position.values[:, 1],
                     c=np.degrees(head_direction.values), cmap='hsv',
                     s=2, alpha=0.5)
ax1.set_xlabel('X Position (cm)', fontsize=11)
ax1.set_ylabel('Y Position (cm)', fontsize=11)
ax1.set_title('Animal Trajectory Colored by Head Direction', fontsize=12)
ax1.set_aspect('equal')
cbar1 = plt.colorbar(scatter, ax=ax1)
cbar1.set_label('Head Direction (°)', fontsize=10)
ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(gs[1, 0])
skip = max(1, len(head_direction) // 1000)
ax2.plot(head_direction.t[::skip], np.degrees(head_direction.values[::skip]),
         linewidth=0.5, alpha=0.7)
ax2.set_xlabel('Time (s)', fontsize=11)
ax2.set_ylabel('Head Direction (°)', fontsize=11)
ax2.set_title('Head Direction Over Time', fontsize=12)
ax2.set_ylim(-20, 380)
ax2.grid(True, alpha=0.3)

ax3 = fig.add_subplot(gs[1, 1])
position_diff = np.diff(position.values, axis=0)
dt = np.diff(position.t.values)
speed = np.linalg.norm(position_diff, axis=1) / dt
speed_ts = nap.Tsd(speed, t=position.t[:-1])
skip = max(1, len(speed) // 1000)
ax3.plot(speed_ts.t[::skip], speed_ts.values[::skip], linewidth=0.5, alpha=0.7, color='green')
ax3.set_xlabel('Time (s)', fontsize=11)
ax3.set_ylabel('Speed (cm/s)', fontsize=11)
ax3.set_title('Movement Speed Over Time', fontsize=12)
ax3.grid(True, alpha=0.3)

ax4 = fig.add_subplot(gs[2, 0])
ax4.hist(np.degrees(head_direction.values), bins=36, color='purple', alpha=0.6, edgecolor='black')
ax4.set_xlabel('Head Direction (°)', fontsize=11)
ax4.set_ylabel('Frequency', fontsize=11)
ax4.set_title('Head Direction Distribution', fontsize=12)
ax4.grid(True, alpha=0.3, axis='y')

ax5 = fig.add_subplot(gs[2, 1])
sorted_pref = sorted(pref_dirs)
ax5.bar(range(len(sorted_pref)), sorted_pref, color='orange', alpha=0.7, edgecolor='black')
ax5.set_xlabel('Neuron (sorted by pref. dir.)', fontsize=11)
ax5.set_ylabel('Preferred Direction (°)', fontsize=11)
ax5.set_title('Neurons: Preferred Direction (sorted)', fontsize=12)
ax5.set_ylim(-20, 380)
ax5.grid(True, alpha=0.3, axis='y')

plt.suptitle('Behavioral Context and Head Direction Properties', fontsize=14, y=0.995)
plt.tight_layout()
plt.savefig('behavioral_context.png', dpi=150, bbox_inches='tight')
print("  ✓ behavioral_context.png")
plt.close()

# %% [markdown]
# ## Summary

# %%
print(f"\n{'='*80}")
print("ANALYSIS SUMMARY")
print(f"{'='*80}\n")

duration = position.t[-1] - position.t[0]
print(f"Recording: {duration:.1f} seconds ({duration/60:.1f} minutes)")
print(f"Neurons: {len(spikes)}")
print(f"Mean firing rate: {np.mean([len(s)/duration for s in spikes.values()]):.2f} Hz")
print(f"\nHead Direction Selectivity:")
print(f"  Mean directionality index: {np.mean(tuning_strengths):.3f}")
print(f"  Strong HD cells (DI > 0.3): {np.sum(tuning_strengths > 0.3)} neurons")
print(f"  Mean peak firing rate: {np.mean(max_rates):.2f} ± {np.std(max_rates):.2f} Hz")
print(f"\nBehavior:")
mean_speed = np.nanmean(speed)
print(f"  Mean speed: {mean_speed:.2f} cm/s")
total_dist = np.sum(np.linalg.norm(position_diff, axis=1))
print(f"  Total distance: {total_dist:.1f} cm")

print(f"\n✓ Analysis complete! All figures saved.")

# %%
io.close()
h5_file.close()
