# %% [markdown]
# # Hippocampal Replay: Decoding Spatial Trajectories During Sharp-Wave Ripples
# Fast analysis version

# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, sosfiltfilt
from tqdm import tqdm
import pickle
import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')
sns.set_palette("Set2")

print("Loading synthetic hippocampal data...")

# Load data
with open('synthetic_replay_data.pkl', 'rb') as f:
    data = pickle.load(f)

spike_times = data['spike_times']
position = data['position']
times = data['times']
lfp_signal = data['lfp_signal']
ripple_times = data['ripple_times']
place_cell_data = data['place_cell_data']
sampling_rate = data['sampling_rate']
env_width = data['env_width']
env_height = data['env_height']

print(f"Duration: {times[-1]:.1f}s, Neurons: {len(spike_times)}, Ripples: {len(ripple_times)}")

# %%
# Ripple detection in LFP
lfp_rate = sampling_rate
lfp_times = times

order = 4
ripple_freqs = [150, 250]
sos = butter(order, ripple_freqs, 'band', fs=lfp_rate, output='sos')
ripple_filtered = sosfiltfilt(sos, lfp_signal)

ripple_envelope = np.abs(signal.hilbert(ripple_filtered))
smoothing_window = int(lfp_rate * 0.01)
ripple_envelope_smooth = gaussian_filter1d(ripple_envelope, sigma=smoothing_window/4)

ripple_threshold = np.mean(ripple_envelope_smooth) + 2 * np.std(ripple_envelope_smooth)
ripple_detection = ripple_envelope_smooth > ripple_threshold

ripple_starts = np.where(np.diff(ripple_detection.astype(int)) == 1)[0]
ripple_ends = np.where(np.diff(ripple_detection.astype(int)) == -1)[0]

if len(ripple_starts) > len(ripple_ends):
    ripple_starts = ripple_starts[:len(ripple_ends)]

ripple_times_detected = np.column_stack([lfp_times[ripple_starts], lfp_times[ripple_ends]])
ripple_durations = ripple_times_detected[:, 1] - ripple_times_detected[:, 0]
valid_ripples = ripple_durations > 0.03
ripple_times_detected = ripple_times_detected[valid_ripples]

print(f"\nRipples detected: {len(ripple_times_detected)}")
if len(ripple_times_detected) > 0:
    ripple_durations_ms = (ripple_times_detected[:, 1] - ripple_times_detected[:, 0]) * 1000
    print(f"  Duration: {ripple_durations_ms.mean():.1f} ± {ripple_durations_ms.std():.1f} ms")

# %%
# FAST place cell identification using grid-based method
print("\nFast place cell identification...")

n_bins = 12
x_bins = np.linspace(0, env_width, n_bins + 1)
y_bins = np.linspace(0, env_height, n_bins + 1)

place_cell_info = {}
unit_ids_list = list(spike_times.keys())

for unit_id in unit_ids_list:
    spikes = spike_times[unit_id]
    if len(spikes) < 50:
        continue

    spike_x = np.interp(spikes, times, position[:, 0])
    spike_y = np.interp(spikes, times, position[:, 1])

    rate_map = np.zeros((n_bins, n_bins))

    for i in range(n_bins):
        for j in range(n_bins):
            in_bin = ((spike_x >= x_bins[i]) & (spike_x < x_bins[i+1]) &
                     (spike_y >= y_bins[j]) & (spike_y < y_bins[j+1]))
            occ = ((position[:, 0] >= x_bins[i]) & (position[:, 0] < x_bins[i+1]) &
                  (position[:, 1] >= y_bins[j]) & (position[:, 1] < y_bins[j+1]))
            occ_time = np.sum(occ) / sampling_rate
            if occ_time > 0:
                rate_map[i, j] = np.sum(in_bin) / occ_time

    # Simple spatial selectivity: peak firing / mean firing
    firing_rate = len(spikes) / (times[-1] - times[0])
    if firing_rate > 0:
        max_rate = rate_map.max()
        selectivity = max_rate / firing_rate if firing_rate > 0 else 0

        if selectivity > 2:  # Neurons that fire > 2x average in their peak location
            place_cell_info[unit_id] = {
                'rate_map': rate_map,
                'selectivity': selectivity,
                'peak_rate': max_rate,
                'mean_rate': firing_rate,
            }

print(f"Place cells identified: {len(place_cell_info)}")

# %%
# Build decoder
print("Building Bayesian decoder...")

if len(place_cell_info) > 5:
    place_cell_ids = list(place_cell_info.keys())

    decoder_model = {}
    for i in range(n_bins):
        for j in range(n_bins):
            decoder_model[(i, j)] = {}

            in_bin = ((position[:, 0] < x_bins[i+1]) & (position[:, 0] >= x_bins[i]) &
                     (position[:, 1] < y_bins[j+1]) & (position[:, 1] >= y_bins[j]))

            if np.sum(in_bin) > 0:
                for unit_id in place_cell_ids:
                    spikes = spike_times[unit_id]
                    spike_count = 0
                    for spike_time in spikes:
                        spike_idx = np.argmin(np.abs(times - spike_time))
                        if in_bin[spike_idx]:
                            spike_count += 1

                    occ_time = np.sum(in_bin) / sampling_rate
                    decoder_model[(i, j)][unit_id] = spike_count / occ_time if occ_time > 0 else 0.1

    print(f"Decoder ready with {len(place_cell_ids)} place cells")

    # Decode ripples
    print("Decoding trajectories...")
    ripple_trajectories = []

    n_ripples_to_decode = min(40, len(ripple_times_detected))
    ripple_sample_indices = np.linspace(0, len(ripple_times_detected) - 1, n_ripples_to_decode, dtype=int)

    for ripple_idx in tqdm(ripple_sample_indices, desc="Decoding ripples"):
        ripple_start, ripple_end = ripple_times_detected[ripple_idx]
        ripple_duration = ripple_end - ripple_start

        n_timepoints = max(4, int(ripple_duration * 500))  # 2ms resolution
        ripple_time_bins = np.linspace(ripple_start, ripple_end, n_timepoints)

        decoded_positions = []

        for t_start, t_end in zip(ripple_time_bins[:-1], ripple_time_bins[1:]):
            firing_rates = {}

            for unit_id in place_cell_ids:
                spikes = spike_times[unit_id]
                spikes_in_window = spikes[(spikes >= t_start) & (spikes < t_end)]
                dt = t_end - t_start
                firing_rates[unit_id] = len(spikes_in_window) / dt if dt > 0 else 0

            best_position = None
            best_likelihood = -np.inf

            for (i, j) in decoder_model.keys():
                log_likelihood = 0
                for unit_id in place_cell_ids:
                    expected_rate = decoder_model[(i, j)].get(unit_id, 0.1)
                    actual_rate = firing_rates.get(unit_id, 0)

                    if expected_rate > 0:
                        log_likelihood += actual_rate * np.log(expected_rate) - expected_rate

                if log_likelihood > best_likelihood:
                    best_likelihood = log_likelihood
                    best_position = (i, j)

            if best_position is not None:
                i, j = best_position
                x_pos = (x_bins[i] + x_bins[i+1]) / 2
                y_pos = (y_bins[j] + y_bins[j+1]) / 2
                decoded_positions.append([x_pos, y_pos])

        if len(decoded_positions) > 1:
            ripple_trajectories.append({
                'times': ripple_time_bins[:-1],
                'positions': np.array(decoded_positions),
                'ripple_start': ripple_start,
                'ripple_end': ripple_end
            })

    print(f"Successfully decoded {len(ripple_trajectories)} ripples")
else:
    ripple_trajectories = []
    place_cell_ids = []

# %%
# FIGURE 1: Ripple detection
print("\nGenerating figures...")

fig, axes = plt.subplots(3, 1, figsize=(14, 8))

time_window = [50, 70]
time_mask = (lfp_times >= time_window[0]) & (lfp_times < time_window[1])

ax = axes[0]
ax.plot(lfp_times[time_mask], lfp_signal[time_mask], 'k-', alpha=0.5, linewidth=0.5)
ax.set_ylabel('LFP (μV)', fontsize=11)
ax.set_title('Raw LFP Signal', fontsize=12, fontweight='bold')
ax.set_xlim(time_window)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(lfp_times[time_mask], ripple_filtered[time_mask], 'b-', alpha=0.7, linewidth=0.5)
ax.set_ylabel('Ripple Band (μV)', fontsize=11)
ax.set_title('Ripple-Filtered LFP (150-250 Hz)', fontsize=12, fontweight='bold')
ax.set_xlim(time_window)
ax.grid(True, alpha=0.3)

ax = axes[2]
ax.plot(lfp_times[time_mask], ripple_envelope_smooth[time_mask], 'g-', linewidth=1, label='Ripple Envelope')
ax.axhline(ripple_threshold, color='r', linestyle='--', linewidth=2, label='Threshold')
ax.fill_between(lfp_times[time_mask], 0, ripple_envelope_smooth[time_mask],
                 where=ripple_detection[time_mask], alpha=0.3, color='orange', label='Detected Ripples')

for r_start, r_end in ripple_times_detected:
    if r_end > time_window[0] and r_start < time_window[1]:
        ax.axvline(r_start, color='red', linestyle=':', alpha=0.5, linewidth=1)

ax.set_xlabel('Time (s)', fontsize=11)
ax.set_ylabel('Envelope', fontsize=11)
ax.set_title('Sharp-Wave Ripple Detection', fontsize=12, fontweight='bold')
ax.legend(loc='upper right', fontsize=10)
ax.set_xlim(time_window)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('01_ripple_detection.png', dpi=150, bbox_inches='tight')
print("✓ Saved: 01_ripple_detection.png")
plt.close()

# %%
# FIGURE 2: Place cell rate maps
if len(place_cell_info) > 0:
    sorted_cells = sorted(place_cell_info.items(),
                         key=lambda x: x[1]['selectivity'], reverse=True)

    n_show = min(12, len(sorted_cells))
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    axes = axes.flatten()

    for idx, (unit_id, info) in enumerate(sorted_cells[:n_show]):
        ax = axes[idx]
        rate_map = info['rate_map']
        rate_map_smooth = gaussian_filter1d(gaussian_filter1d(rate_map, sigma=1), sigma=1, axis=1)

        im = ax.imshow(rate_map_smooth, cmap='hot', origin='lower', aspect='auto')
        ax.set_title(f'Unit {unit_id}\nSel={info["selectivity"]:.1f}', fontsize=9)
        ax.set_xlabel('X (bins)', fontsize=8)
        ax.set_ylabel('Y (bins)', fontsize=8)
        ax.tick_params(labelsize=8)
        cb = plt.colorbar(im, ax=ax, label='Hz', fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=8)

    for idx in range(n_show, len(axes)):
        axes[idx].axis('off')

    fig.suptitle('Place Cell Rate Maps (Sorted by Spatial Selectivity)',
                 fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('02_place_cell_maps.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: 02_place_cell_maps.png")
    plt.close()

# %%
# FIGURE 3: Decoded trajectories
if len(ripple_trajectories) > 0:
    n_examples = min(9, len(ripple_trajectories))
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    axes = axes.flatten()

    for idx in range(n_examples):
        ax = axes[idx]
        trajectory = ripple_trajectories[idx]
        positions = trajectory['positions']

        ax.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2.5, alpha=0.7, marker='o',
                markersize=5, markerfacecolor='lightblue', markeredgecolor='blue')
        ax.plot(positions[0, 0], positions[0, 1], 'go', markersize=10, label='Start', zorder=5)
        ax.plot(positions[-1, 0], positions[-1, 1], 'r^', markersize=10, label='End', zorder=5)

        ax.set_xlim([0, env_width])
        ax.set_ylim([0, env_height])

        duration = trajectory['ripple_end'] - trajectory['ripple_start']
        ax.set_title(f'Ripple {idx+1} ({duration*1000:.0f}ms)', fontsize=10, fontweight='bold')
        ax.set_aspect('equal')
        ax.set_xlabel('X (cm)', fontsize=9)
        ax.set_ylabel('Y (cm)', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=9)

        if idx == 0:
            ax.legend(fontsize=9, loc='upper right')

    for idx in range(n_examples, len(axes)):
        axes[idx].axis('off')

    fig.suptitle('Decoded Spatial Trajectories During Ripples\n(Bayesian Decoding from Place Cells)',
                 fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('03_decoded_trajectories.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: 03_decoded_trajectories.png")
    plt.close()

# %%
# FIGURE 4: Summary statistics
fig = plt.figure(figsize=(12, 9))
gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

# Spatial selectivity
if len(place_cell_info) > 0:
    ax = fig.add_subplot(gs[0, 0])
    selectivities = [info['selectivity'] for info in place_cell_info.values()]
    ax.hist(selectivities, bins=15, color='steelblue', alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.axvline(2.0, color='red', linestyle='--', linewidth=2.5, label='Selection threshold')
    ax.set_xlabel('Spatial Selectivity', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Spatial Selectivity of Identified Units', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(labelsize=10)

# Ripple durations
ax = fig.add_subplot(gs[0, 1])
if len(ripple_times_detected) > 0:
    ripple_durations_ms = (ripple_times_detected[:, 1] - ripple_times_detected[:, 0]) * 1000
    ax.hist(ripple_durations_ms, bins=25, color='coral', alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Ripple Duration (ms)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title(f'Ripple Event Duration (n={len(ripple_times_detected)})', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(labelsize=10)

    stats_text = f"Mean: {ripple_durations_ms.mean():.1f} ms\nMedian: {np.median(ripple_durations_ms):.1f} ms"
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=10)

# Ripple rate
ax = fig.add_subplot(gs[1, 0])
if len(ripple_times_detected) > 0:
    session_duration = times[-1]
    time_bins = np.arange(0, session_duration, 60)
    ripple_counts, _ = np.histogram(ripple_times_detected[:, 0], bins=time_bins)
    bin_centers = (time_bins[:-1] + time_bins[1:]) / 2

    ax.bar(bin_centers / 60, ripple_counts, width=0.8, color='lightgreen', alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Time (minutes)', fontsize=11)
    ax.set_ylabel('Ripples per Minute', fontsize=11)
    ax.set_title('Ripple Rate Over Recording Session', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(labelsize=10)

# Trajectory speeds
ax = fig.add_subplot(gs[1, 1])
if len(ripple_trajectories) > 0:
    trajectory_speeds = []
    for traj in ripple_trajectories:
        positions = traj['positions']
        if len(positions) > 1:
            distances = np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))
            times_traj = np.diff(traj['times'])
            speeds = distances / times_traj
            trajectory_speeds.extend(speeds)

    if trajectory_speeds:
        ax.hist(trajectory_speeds, bins=20, color='mediumpurple', alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.set_xlabel('Trajectory Speed (cm/s)', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title('Speed Distribution of Decoded Trajectories', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(labelsize=10)

        stats_text = f"Mean: {np.mean(trajectory_speeds):.1f} cm/s\nMedian: {np.median(trajectory_speeds):.1f} cm/s"
        ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=10)

plt.savefig('04_summary_statistics.png', dpi=150, bbox_inches='tight')
print("✓ Saved: 04_summary_statistics.png")
plt.close()

# %%
# Print summary
print("\n" + "="*70)
print("HIPPOCAMPAL REPLAY ANALYSIS COMPLETE")
print("="*70)

print(f"\nRecording: {times[-1]:.1f}s ({times[-1]/60:.2f} minutes)")
print(f"Neurons: {len(unit_ids_list)} total, {len(place_cell_info)} place cells")

if len(ripple_times_detected) > 0:
    print(f"\nRipples: {len(ripple_times_detected)} detected")
    print(f"  Duration: {ripple_durations_ms.mean():.1f} ± {ripple_durations_ms.std():.1f} ms")
    print(f"  Rate: {len(ripple_times_detected) / (times[-1]/60):.1f} ripples/minute")

if len(ripple_trajectories) > 0:
    print(f"\nDecoded Trajectories: {len(ripple_trajectories)} ripples")
    trajectory_speeds = []
    for traj in ripple_trajectories:
        positions = traj['positions']
        if len(positions) > 1:
            distances = np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))
            times_traj = np.diff(traj['times'])
            speeds = distances / times_traj
            trajectory_speeds.extend(speeds)

    if trajectory_speeds:
        print(f"  Speed: {np.mean(trajectory_speeds):.1f} ± {np.std(trajectory_speeds):.1f} cm/s")

print("\n" + "="*70)
print("Analysis demonstrates hippocampal replay through rapid sequential")
print("reactivation of place cells during sharp-wave ripples.")
print("="*70)
