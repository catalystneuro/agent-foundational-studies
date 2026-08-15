# %% [markdown]
# # Head Direction Cells Analysis - Fast Version
#
# Optimized analysis using cached data access

# %%
import h5py
from pynwb import NWBHDF5IO
import remfile
import pynapple as nap
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

plt.rcParams['figure.figsize'] = (12, 8)
os.environ['MPLBACKEND'] = 'Agg'

print("Loading NWB file...")

# Use existing file if present, otherwise stream
if os.path.exists('data.nwb'):
    print("Using local data.nwb")
    h5_file = h5py.File('data.nwb', 'r')
else:
    print("Streaming from S3...")
    cache_dir = "/tmp/nwb_cache"
    os.makedirs(cache_dir, exist_ok=True)
    disk_cache = remfile.DiskCache(cache_dir)
    rem_file = remfile.File("https://dandiarchive.s3.amazonaws.com/blobs/58c/537/58c53789-eec4-4080-ad3b-207cf2a1cac9",
                            disk_cache=disk_cache)
    h5_file = h5py.File(rem_file, "r")

io = NWBHDF5IO(file=h5_file)
nwbfile = io.read()

print(f"✓ Loaded: {nwbfile.identifier}")
print(f"  Date: {nwbfile.session_start_time}")
print(f"  Neurons: {len(nwbfile.units)}\n")

# %% Extract data
print("Extracting data...")

spikes = {}
for unit_id in range(len(nwbfile.units)):
    spike_times = nwbfile.units[unit_id]['spike_times'][:]
    if len(spike_times) > 0:
        spikes[unit_id] = nap.Ts(spike_times)

pos0 = nwbfile.acquisition['position_sensor0']
times = nap.Ts(pos0.timestamps[:])
pos0_coords = pos0.data[:]
pos1_coords = nwbfile.acquisition['position_sensor1'].data[:]

# Compute head direction
head_vector = pos1_coords - pos0_coords
head_direction_rad = np.arctan2(head_vector[:, 1], head_vector[:, 0])
head_direction_rad = np.mod(head_direction_rad, 2 * np.pi)
head_direction = nap.Tsd(head_direction_rad, t=times)

print(f"✓ Data ready: {len(spikes)} neurons\n")

# %% Compute tuning curves
print("Computing tuning curves...")

n_bins = 36
bin_edges = np.linspace(0, 2*np.pi, n_bins + 1)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

tuning_curves = {}
tuning_stats = {}

for unit_id in tqdm(spikes.keys()):
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

    firing_rates = np.divide(spike_counts, time_in_bin, where=time_in_bin > 0, out=np.zeros_like(spike_counts))
    tuning_curves[unit_id] = firing_rates

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

print(f"✓ Tuning curves computed\n")

# %% Print results
print(f"{'='*80}")
print("RESULTS")
print(f"{'='*80}\n")

for uid in sorted(spikes.keys()):
    s = tuning_stats[uid]
    print(f"Unit {uid:2d}: Dir {s['preferred_direction']:6.1f}° | Max {s['max_firing_rate']:6.2f} Hz | Strength {s['directionality_index']:.3f}")

# %% Visualization 1
print(f"\nGenerating figures...")

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
    ax.set_title(f'Unit {unit_id}\n({tuning_stats[unit_id]["directionality_index"]:.2f})', fontsize=10, pad=10)
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(-1)
    ax.grid(True, alpha=0.3)

for idx in range(len(tuning_curves), len(axes)):
    axes[idx].set_visible(False)

plt.suptitle('Head Direction Tuning Curves - Polar View', fontsize=14, y=0.98)
plt.tight_layout()
plt.savefig('tuning_curves_polar.png', dpi=150, bbox_inches='tight')
print("  ✓ tuning_curves_polar.png")
plt.close()

# %% Visualization 2
fig, axes = plt.subplots(3, 5, figsize=(16, 10))
axes = axes.flatten()

for idx, (unit_id, firing_rates) in enumerate(tuning_curves.items()):
    ax = axes[idx]
    angles_deg = np.degrees(bin_centers)
    ax.bar(angles_deg, firing_rates, width=8, alpha=0.7, color='steelblue', edgecolor='black')
    pref_dir = tuning_stats[unit_id]['preferred_direction']
    ax.axvline(pref_dir, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_xlim(-10, 370)
    ax.set_xlabel('Head Direction (°)', fontsize=9)
    ax.set_ylabel('Firing Rate (Hz)', fontsize=9)
    ax.set_xticks([0, 90, 180, 270])
    ax.set_title(f'Unit {unit_id}\nDI: {tuning_stats[unit_id]["directionality_index"]:.2f}', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

for idx in range(len(tuning_curves), len(axes)):
    axes[idx].set_visible(False)

plt.suptitle('Head Direction Tuning Curves - Cartesian View', fontsize=14)
plt.tight_layout()
plt.savefig('tuning_curves_cartesian.png', dpi=150, bbox_inches='tight')
print("  ✓ tuning_curves_cartesian.png")
plt.close()

# %% Visualization 3
pref_dirs = np.array([tuning_stats[uid]['preferred_direction'] for uid in tuning_curves.keys()])
tuning_strengths = np.array([tuning_stats[uid]['directionality_index'] for uid in tuning_curves.keys()])
max_rates = np.array([tuning_stats[uid]['max_firing_rate'] for uid in tuning_curves.keys()])

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].hist(pref_dirs, bins=12, color='steelblue', edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Preferred Direction (°)')
axes[0, 0].set_ylabel('Number of Neurons')
axes[0, 0].set_title('Preferred Direction Distribution')
axes[0, 0].grid(True, alpha=0.3, axis='y')

axes[0, 1].hist(tuning_strengths, bins=8, color='coral', edgecolor='black', alpha=0.7)
axes[0, 1].set_xlabel('Directionality Index')
axes[0, 1].set_ylabel('Number of Neurons')
axes[0, 1].set_title('Tuning Strength Distribution')
axes[0, 1].axvline(0.3, color='red', linestyle='--', linewidth=2, alpha=0.7, label='HD threshold')
axes[0, 1].grid(True, alpha=0.3, axis='y')
axes[0, 1].legend()

axes[1, 0].hist(max_rates, bins=8, color='lightgreen', edgecolor='black', alpha=0.7)
axes[1, 0].set_xlabel('Maximum Firing Rate (Hz)')
axes[1, 0].set_ylabel('Number of Neurons')
axes[1, 0].set_title('Peak Firing Rate Distribution')
axes[1, 0].grid(True, alpha=0.3, axis='y')

scatter = axes[1, 1].scatter(max_rates, tuning_strengths, s=100, alpha=0.6, c=pref_dirs, cmap='hsv', edgecolor='black')
axes[1, 1].set_xlabel('Maximum Firing Rate (Hz)')
axes[1, 1].set_ylabel('Directionality Index')
axes[1, 1].set_title('Firing Rate vs. Tuning Strength')
axes[1, 1].grid(True, alpha=0.3)
cbar = plt.colorbar(scatter, ax=axes[1, 1])
cbar.set_label('Preferred Direction (°)')
axes[1, 1].axhline(0.3, color='red', linestyle='--', linewidth=1, alpha=0.5)

plt.suptitle('Population-Level Statistics', fontsize=14)
plt.tight_layout()
plt.savefig('population_statistics.png', dpi=150, bbox_inches='tight')
print("  ✓ population_statistics.png")
plt.close()

# %% Visualization 4
position = nap.TsdFrame(pos0_coords, t=times, columns=['x', 'y'])

fig = plt.figure(figsize=(15, 10))
gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

ax1 = fig.add_subplot(gs[0, :])
scatter = ax1.scatter(position.values[:, 0], position.values[:, 1], c=np.degrees(head_direction.values), cmap='hsv', s=2, alpha=0.5)
ax1.set_xlabel('X Position (cm)')
ax1.set_ylabel('Y Position (cm)')
ax1.set_title('Animal Trajectory Colored by Head Direction')
ax1.set_aspect('equal')
cbar1 = plt.colorbar(scatter, ax=ax1)
cbar1.set_label('Head Direction (°)')
ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(gs[1, 0])
skip = max(1, len(head_direction) // 1000)
ax2.plot(head_direction.t[::skip], np.degrees(head_direction.values[::skip]), linewidth=0.5, alpha=0.7)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Head Direction (°)')
ax2.set_title('Head Direction Over Time')
ax2.set_ylim(-20, 380)
ax2.grid(True, alpha=0.3)

ax3 = fig.add_subplot(gs[1, 1])
position_diff = np.diff(position.values, axis=0)
dt = np.diff(position.t.values)
speed = np.linalg.norm(position_diff, axis=1) / dt
speed_ts = nap.Tsd(speed, t=position.t[:-1])
skip = max(1, len(speed) // 1000)
ax3.plot(speed_ts.t[::skip], speed_ts.values[::skip], linewidth=0.5, alpha=0.7, color='green')
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Speed (cm/s)')
ax3.set_title('Movement Speed Over Time')
ax3.grid(True, alpha=0.3)

ax4 = fig.add_subplot(gs[2, 0])
ax4.hist(np.degrees(head_direction.values), bins=36, color='purple', alpha=0.6, edgecolor='black')
ax4.set_xlabel('Head Direction (°)')
ax4.set_ylabel('Frequency')
ax4.set_title('Head Direction Distribution')
ax4.grid(True, alpha=0.3, axis='y')

ax5 = fig.add_subplot(gs[2, 1])
sorted_pref = sorted(pref_dirs)
ax5.bar(range(len(sorted_pref)), sorted_pref, color='orange', alpha=0.7, edgecolor='black')
ax5.set_xlabel('Neuron (sorted by pref. dir.)')
ax5.set_ylabel('Preferred Direction (°)')
ax5.set_title('Neurons: Preferred Direction (sorted)')
ax5.set_ylim(-20, 380)
ax5.grid(True, alpha=0.3, axis='y')

plt.suptitle('Behavioral Context and Head Direction Properties', fontsize=14, y=0.995)
plt.tight_layout()
plt.savefig('behavioral_context.png', dpi=150, bbox_inches='tight')
print("  ✓ behavioral_context.png")
plt.close()

# %% Summary
print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}\n")

duration = position.t[-1] - position.t[0]
print(f"Recording: {duration:.1f} sec ({duration/60:.1f} min)")
print(f"Neurons: {len(spikes)}")
print(f"Mean firing rate: {np.mean([len(s)/duration for s in spikes.values()]):.2f} Hz")
print(f"Mean tuning strength: {np.mean(tuning_strengths):.3f}")
print(f"Strong HD cells: {np.sum(tuning_strengths > 0.3)} neurons")
print(f"Mean peak rate: {np.mean(max_rates):.2f} ± {np.std(max_rates):.2f} Hz")
print(f"Mean speed: {np.nanmean(speed):.2f} cm/s")

print(f"\n✓ Analysis complete!")

io.close()
h5_file.close()
