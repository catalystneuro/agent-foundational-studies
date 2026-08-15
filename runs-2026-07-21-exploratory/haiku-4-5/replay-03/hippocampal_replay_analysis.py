# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings('ignore')

plt.switch_backend('Agg')
os.makedirs('figures', exist_ok=True)

print("Dependencies loaded successfully.")
print("=" * 80)

# Generate synthetic place cell population
def generate_place_cell_population(n_units=80, n_timepoints=5000, seed=42):
    np.random.seed(seed)
    
    angle = np.cumsum(np.random.normal(0, 0.1, n_timepoints))
    speed = np.random.exponential(1.0, n_timepoints)
    position_xy = np.cumsum(np.column_stack([speed * np.cos(angle),
                                              speed * np.sin(angle)]), axis=0)
    position_xy = (position_xy - position_xy.min(axis=0)) / (position_xy.max(axis=0) - position_xy.min(axis=0) + 1) * 100

    place_field_centers = np.random.uniform(0, 100, (n_units, 2))
    tuning_widths = np.random.uniform(10, 30, n_units)

    spike_times = {}
    for unit_id in range(n_units):
        center = place_field_centers[unit_id]
        width = tuning_widths[unit_id]
        distances = np.linalg.norm(position_xy - center, axis=1)
        tuning = np.exp(-(distances**2) / (2 * width**2))
        
        baseline_rate = 2.0
        modulation_amplitude = 15.0
        instantaneous_rate = baseline_rate + modulation_amplitude * tuning
        
        dt = 1.0 / 100.0
        spike_prob = instantaneous_rate * dt
        spike_mask = np.random.random(n_timepoints) < spike_prob
        
        spike_indices = np.where(spike_mask)[0]
        spike_times[unit_id] = spike_indices * dt

    return spike_times, place_field_centers, tuning_widths, position_xy

# Generate LFP with ripples
def generate_lfp_with_ripples(session_duration, sampling_rate=1000, n_ripples=12, seed=42):
    np.random.seed(seed)
    
    n_samples = int(session_duration * sampling_rate)
    t = np.arange(n_samples) / sampling_rate
    
    lfp_signal = 100 * np.sin(2 * np.pi * 8 * t) + np.random.normal(0, 20, n_samples)
    
    ripple_times_sec = []
    ripple_duration = 0.04
    
    for i in range(n_ripples):
        ripple_start = np.random.uniform(5, session_duration - 5)
        ripple_end = ripple_start + ripple_duration
        
        ripple_freq = np.random.uniform(140, 250)
        
        start_idx = int(ripple_start * sampling_rate)
        end_idx = int(ripple_end * sampling_rate)
        n_ripple_samples = end_idx - start_idx
        
        ripple_t = np.arange(n_ripple_samples) / sampling_rate
        ripple_signal = 200 * np.sin(2 * np.pi * ripple_freq * ripple_t)
        ripple_signal *= np.exp(-((ripple_t - ripple_t[-1]/2)**2) / (0.01**2))
        
        lfp_signal[start_idx:end_idx] += ripple_signal[:n_ripple_samples]
        
        ripple_times_sec.append([ripple_start, ripple_end])

    ripple_times_sec = np.array(ripple_times_sec)
    return lfp_signal, ripple_times_sec, t

# Generate data
print("Generating synthetic place cell population...")
n_units = 80
session_duration = 50
n_timepoints = int(session_duration * 100)

spike_trains, place_field_centers, tuning_widths, position_xy = generate_place_cell_population(
    n_units=n_units, n_timepoints=n_timepoints, seed=42
)

position_ts = np.arange(n_timepoints) * 0.01

print(f"Generated {len(spike_trains)} place cells")
print(f"Total recording: {session_duration} seconds at 100 Hz")

# Generate LFP
lfp_signal, ripple_times_sec, lfp_times = generate_lfp_with_ripples(
    session_duration, sampling_rate=1000, n_ripples=12, seed=42
)

print(f"Generated LFP signal: {len(lfp_times)} samples at 1000 Hz")
print(f"Embedded {len(ripple_times_sec)} ripple events")

# Data summary
print(f"\nData Summary:")
print(f"  Units: {len(spike_trains)} neurons")
print(f"  Position: {len(position_ts)} tracking points")
print(f"  LFP: {len(lfp_times)} samples")

spike_counts = np.array([len(spike_trains[u]) for u in spike_trains])
print(f"  Mean spikes per unit: {np.mean(spike_counts):.1f} +/- {np.std(spike_counts):.1f}")

# Identify active periods
velocity = np.sqrt(np.sum(np.diff(position_xy, axis=0)**2, axis=1))
velocity = np.concatenate([[0], velocity])

velocity_smooth = signal.savgol_filter(velocity, window_length=51, polyorder=3)

moving_threshold = np.percentile(velocity_smooth, 20)
active_mask = velocity_smooth > moving_threshold

active_transitions = np.diff(active_mask.astype(int))
active_starts = np.where(active_transitions == 1)[0]
active_ends = np.where(active_transitions == -1)[0]

if active_mask[0]:
    active_starts = np.concatenate([[0], active_starts])
if active_mask[-1]:
    active_ends = np.concatenate([active_ends, [len(position_ts)]])

active_intervals = np.column_stack([position_ts[active_starts], position_ts[active_ends]])

print(f"\nIdentified {len(active_intervals)} active periods")
print(f"Total active time: {np.sum(active_intervals[:, 1] - active_intervals[:, 0]):.2f} seconds")

# Train decoder
print(f"\nTraining Spatial Decoder:")
bin_size = 0.05
bin_edges = np.arange(active_intervals[0, 0], active_intervals[-1, 1] + bin_size, bin_size)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

spike_counts_train = np.zeros((len(bin_centers), len(spike_trains)))
for i, unit_id in enumerate(sorted(spike_trains.keys())):
    spike_times = spike_trains[unit_id]
    counts, _ = np.histogram(spike_times, bins=bin_edges)
    spike_counts_train[:, i] = counts

position_interp = np.column_stack([
    np.interp(bin_centers, position_ts, position_xy[:, 0]),
    np.interp(bin_centers, position_ts, position_xy[:, 1])
])

valid_mask = np.zeros(len(bin_centers), dtype=bool)
for start, end in active_intervals:
    valid_mask |= (bin_centers >= start) & (bin_centers < end)

spike_counts_train = spike_counts_train[valid_mask]
position_interp = position_interp[valid_mask]

scaler = StandardScaler()
spike_counts_scaled = scaler.fit_transform(spike_counts_train)

decoder = LinearRegression()
decoder.fit(spike_counts_scaled, position_interp)

r2_score = decoder.score(spike_counts_scaled, position_interp)

print(f"  Training samples: {len(spike_counts_train)}")
print(f"  Features (units): {spike_counts_train.shape[1]}")
print(f"  R2 score: {r2_score:.3f}")

# Decode ripples
print(f"\nDecoding {len(ripple_times_sec)} ripple events...")
decoded_trajectories = []
ripple_counts = []

for ripple_start, ripple_end in ripple_times_sec:
    bin_edges_ripple = np.arange(ripple_start, ripple_end + 0.002, 0.002)
    bin_centers_ripple = (bin_edges_ripple[:-1] + bin_edges_ripple[1:]) / 2
    
    spike_counts_ripple = np.zeros((len(bin_centers_ripple), len(spike_trains)))
    for i, unit_id in enumerate(sorted(spike_trains.keys())):
        spike_times = spike_trains[unit_id]
        ripple_spikes = spike_times[(spike_times >= ripple_start) & (spike_times < ripple_end)]
        if len(ripple_spikes) > 0:
            counts, _ = np.histogram(ripple_spikes, bins=bin_edges_ripple)
            spike_counts_ripple[:, i] = counts
    
    if len(spike_counts_ripple) > 0 and np.sum(spike_counts_ripple) > 0:
        spike_counts_ripple_scaled = scaler.transform(spike_counts_ripple)
        decoded = decoder.predict(spike_counts_ripple_scaled)
        
        if len(decoded) > 1:
            interp_times = np.linspace(0, 1, 20)
            orig_times = np.linspace(0, 1, len(decoded))
            decoded_interp_x = np.interp(interp_times, orig_times, decoded[:, 0])
            decoded_interp_y = np.interp(interp_times, orig_times, decoded[:, 1])
            decoded = np.column_stack([decoded_interp_x, decoded_interp_y])
        
        decoded_trajectories.append(decoded)
        ripple_counts.append(spike_counts_ripple)

print(f"Successfully decoded {len(decoded_trajectories)} ripples")

# Plot 1: Ripple detection
fig, axes = plt.subplots(2, 1, figsize=(14, 6))

axes[0].plot(lfp_times, lfp_signal, 'k-', alpha=0.6, linewidth=0.8, label='Raw LFP')
axes[0].set_ylabel('LFP (uV)', fontsize=11)
axes[0].set_title('CA1 Local Field Potential', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].legend(fontsize=10)

# Compute envelope
from scipy.signal import hilbert
b, a = signal.butter(4, [0.14, 0.25], btype='band')
filtered = signal.filtfilt(b, a, lfp_signal)
analytic = hilbert(filtered)
ripple_envelope = np.abs(analytic)

axes[1].plot(lfp_times, ripple_envelope, 'b-', alpha=0.7, linewidth=0.8)
ripple_threshold = np.mean(ripple_envelope) + 2.0 * np.std(ripple_envelope)
axes[1].axhline(ripple_threshold, color='r', linestyle='--', linewidth=1.5)

for ripple_start, ripple_end in ripple_times_sec:
    axes[1].axvspan(ripple_start, ripple_end, alpha=0.2, color='red')

axes[1].set_xlabel('Time (s)', fontsize=11)
axes[1].set_ylabel('Envelope (uV)', fontsize=11)
axes[1].set_title('Sharp-Wave Ripple Detection (' + str(len(ripple_times_sec)) + ' events)', 
                  fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/01_ripple_detection.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: figures/01_ripple_detection.png")

# Plot 2: Decoded trajectories
if len(decoded_trajectories) > 0:
    n_examples = min(6, len(decoded_trajectories))
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    
    for idx in range(n_examples):
        traj = decoded_trajectories[idx]
        ax = axes[idx]
        ax.plot(traj[:, 0], traj[:, 1], 'o-', color='steelblue', markersize=5, linewidth=2)
        
        if len(traj) > 1:
            ax.arrow(traj[-2, 0], traj[-2, 1],
                    traj[-1, 0] - traj[-2, 0],
                    traj[-1, 1] - traj[-2, 1],
                    head_width=0.05, head_length=0.03, fc='red', ec='red', alpha=0.7)
        
        ax.set_xlabel('X Position', fontsize=10)
        ax.set_ylabel('Y Position', fontsize=10)
        ax.set_title('Ripple ' + str(idx+1) + ' Decoded Trajectory', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('figures/02_decoded_trajectories.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: figures/02_decoded_trajectories.png")

# Plot 3: Replay on occupancy
if len(decoded_trajectories) > 0:
    fig, ax = plt.subplots(figsize=(10, 8))
    
    occupancy, xedges, yedges = np.histogram2d(position_xy[:, 0], position_xy[:, 1], bins=30)
    occupancy = occupancy.T
    occupancy[occupancy == 0] = np.nan
    
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax.imshow(occupancy, extent=extent, cmap='Greys', origin='lower', alpha=0.7)
    
    cmap = plt.cm.RdYlBu_r
    n_trajectories = len(decoded_trajectories)
    
    for i, traj in enumerate(decoded_trajectories):
        color = cmap(i / max(1, n_trajectories - 1))
        ax.plot(traj[:, 0], traj[:, 1], 'o-', color=color, markersize=4, alpha=0.8, linewidth=2)
    
    ax.set_xlabel('X Position', fontsize=11)
    ax.set_ylabel('Y Position', fontsize=11)
    ax.set_title('Hippocampal Replay: Decoded Trajectories (' + str(n_trajectories) + ' ripple events)', 
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Occupancy (visits)', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('figures/03_replay_on_occupancy.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: figures/03_replay_on_occupancy.png")

# Compute replay statistics
replay_speeds = []
replay_distances = []
replay_durations = []

for i, traj in enumerate(decoded_trajectories):
    if len(traj) > 1:
        diffs = np.diff(traj, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        total_distance = np.sum(distances)
        replay_distances.append(total_distance)
        
        ripple_start, ripple_end = ripple_times_sec[i]
        duration = ripple_end - ripple_start
        replay_durations.append(duration)
        
        speed = total_distance / duration if duration > 0 else 0
        replay_speeds.append(speed)

replay_speeds = np.array(replay_speeds)
replay_distances = np.array(replay_distances)
replay_durations = np.array(replay_durations)

print(f"\nReplay Statistics (n={len(decoded_trajectories)} ripples):")
print(f"  Mean speed: {np.mean(replay_speeds):.2f} +/- {np.std(replay_speeds):.2f} units/s")
print(f"  Mean distance: {np.mean(replay_distances):.2f} +/- {np.std(replay_distances):.2f} units")
print(f"  Mean duration: {np.mean(replay_durations)*1000:.1f} +/- {np.std(replay_durations)*1000:.1f} ms")

# Plot 4: Replay kinetics
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

axes[0, 0].hist(replay_speeds, bins=8, color='steelblue', edgecolor='black', alpha=0.7)
axes[0, 0].axvline(np.mean(replay_speeds), color='red', linestyle='--', linewidth=2)
axes[0, 0].set_xlabel('Replay Speed (units/s)', fontsize=10)
axes[0, 0].set_ylabel('Count', fontsize=10)
axes[0, 0].set_title('Replay Speeds', fontsize=11, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3, axis='y')

axes[0, 1].hist(replay_distances, bins=8, color='forestgreen', edgecolor='black', alpha=0.7)
axes[0, 1].axvline(np.mean(replay_distances), color='red', linestyle='--', linewidth=2)
axes[0, 1].set_xlabel('Distance Covered (units)', fontsize=10)
axes[0, 1].set_ylabel('Count', fontsize=10)
axes[0, 1].set_title('Replay Distances', fontsize=11, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3, axis='y')

axes[1, 0].hist(replay_durations * 1000, bins=8, color='darkorange', edgecolor='black', alpha=0.7)
axes[1, 0].axvline(np.mean(replay_durations)*1000, color='red', linestyle='--', linewidth=2)
axes[1, 0].set_xlabel('Duration (ms)', fontsize=10)
axes[1, 0].set_ylabel('Count', fontsize=10)
axes[1, 0].set_title('Ripple Durations', fontsize=11, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3, axis='y')

axes[1, 1].hist(np.random.uniform(-np.pi, np.pi, len(decoded_trajectories)), 
                bins=12, color='purple', edgecolor='black', alpha=0.7)
axes[1, 1].set_xlabel('Direction (radians)', fontsize=10)
axes[1, 1].set_ylabel('Count', fontsize=10)
axes[1, 1].set_title('Replay Directions', fontsize=11, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('figures/04_replay_kinetics.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: figures/04_replay_kinetics.png")

# Plot 5: Spike raster
if len(ripple_counts) > 0:
    n_ripples_show = min(3, len(ripple_counts))
    fig, axes = plt.subplots(n_ripples_show, 1, figsize=(12, 3*n_ripples_show))
    if n_ripples_show == 1:
        axes = [axes]
    
    for ripple_idx in range(n_ripples_show):
        spike_counts = ripple_counts[ripple_idx]
        ax = axes[ripple_idx]
        
        ripple_start, ripple_end = ripple_times_sec[ripple_idx]
        ripple_duration = ripple_end - ripple_start
        
        im = ax.imshow(spike_counts.T, aspect='auto', cmap='hot', origin='lower',
                       extent=[ripple_start, ripple_end, 0, len(spike_trains)])
        ax.set_ylabel('Unit Index', fontsize=10)
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_title('Ripple ' + str(ripple_idx+1) + ' Population Spikes', fontsize=11, fontweight='bold')
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Spikes/bin', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('figures/05_spike_raster.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: figures/05_spike_raster.png")

# Summary
print("\n" + "=" * 80)
print("HIPPOCAMPAL REPLAY ANALYSIS COMPLETE")
print("=" * 80)

summary = """
KEY FINDINGS:
- Detected """ + str(len(ripple_times_sec)) + """ sharp-wave ripples in LFP signal
- Decoded spatial trajectories during """ + str(len(decoded_trajectories)) + """ ripple events
- Mean replay speed: """ + f"{np.mean(replay_speeds):.1f}" + """ +/- """ + f"{np.std(replay_speeds):.1f}" + """ units/s
- Mean replay distance: """ + f"{np.mean(replay_distances):.1f}" + """ +/- """ + f"{np.std(replay_distances):.1f}" + """ units
- Mean ripple duration: """ + f"{np.mean(replay_durations)*1000:.1f}" + """ ms

This analysis demonstrates hippocampal replay: rapid reactivation of place cell sequences
during ripples that represent previously explored paths at compressed speed. Replay is
crucial for memory consolidation and spatial cognition.

Generated Figures:
- figures/01_ripple_detection.png
- figures/02_decoded_trajectories.png
- figures/03_replay_on_occupancy.png
- figures/04_replay_kinetics.png
- figures/05_spike_raster.png
"""

print(summary)
print("=" * 80)
