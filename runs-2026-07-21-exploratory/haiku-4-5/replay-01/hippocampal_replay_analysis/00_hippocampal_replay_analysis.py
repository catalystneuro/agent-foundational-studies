# %% [markdown]
# # Hippocampal Replay: Decoding Spatial Trajectories During Sharp-Wave Ripples
# 
# This analysis demonstrates hippocampal replay by decoding spatial trajectories 
# represented in place cell activity during sharp-wave ripple (SWR) events. We use 
# real data from the Senzai et al. dataset (DANDI 000003), which contains high-quality
# extracellular recordings from rat CA1 during open-field exploration.
#
# ## Background
#
# Sharp-wave ripples are high-frequency LFP oscillations (150-200 Hz) that occur 
# during immobility or sleep and are associated with the replay of previously 
# experienced trajectories. Place cells, neurons that fire at specific spatial locations,
# represent compressed versions of real trajectories during these events.
#
# We will:
# 1. Load multichannel LFP and spike data from a hippocampal recording
# 2. Detect sharp-wave ripple events in the LFP
# 3. Extract place cell activity during ripples
# 4. Train a Bayesian decoder on awake place cell activity
# 5. Decode spatial position during ripples
# 6. Visualize decoded trajectories to demonstrate replay

# %% [markdown]
# ## Setup and Dependencies

# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from scipy import signal, stats
from scipy.ndimage import gaussian_filter1d
import pynapple as nap
from tqdm import tqdm
import h5py
import os
from pathlib import Path
import remfile
from pynwb import NWBHDF5IO
import json

plt.style.use('default')
sns.set_palette("husl")

# %% [markdown]
# ## Data Loading and Exploration

# %%
# Dataset: Senzai et al. (DANDI 000003)
# These are high-quality rat CA1 recordings during open field exploration

# Check if file is available locally (downloaded via dandi)
local_file = Path('/tmp/dandi_test/sub-YutaMouse20/sub-YutaMouse20_ses-YutaMouse20-140327_behavior+ecephys.nwb')

if local_file.exists():
    nwb_path = str(local_file)
    print(f"Loading NWB file from local cache: {nwb_path}")
else:
    print("Waiting for file download... checking /tmp/dandi_test")
    import time
    for i in range(120):  # Wait up to 2 minutes
        if local_file.exists():
            nwb_path = str(local_file)
            print(f"File downloaded! Loading from: {nwb_path}")
            break
        time.sleep(1)
        if i % 10 == 0:
            print(f"  Still downloading ({i}s elapsed)...")
    else:
        raise FileNotFoundError(f"NWB file not found at {local_file}")

print("Loading NWB file...")
io = NWBHDF5IO(nwb_path, load_namespaces=True)
nwbfile = io.read()

# Create Pynapple wrapper for easy access
nwb = nap.NWBFile(nwbfile)
print("\nNWB File loaded successfully!")
print(f"Available data: {nwb}")

# %% [markdown]
# ## Extract Spike and LFP Data

# %%
# Get units (neurons) and their spike data
units = nwb.units
print(f"\nTotal units: {len(units)}")
print(f"Unit IDs: {list(units.keys())[:10]}...")  # Show first 10

# Extract spike times for each unit
spike_times = {}
for unit_id in units.keys():
    spike_times[unit_id] = units[unit_id].t
    
print(f"\nExtracted spike times for {len(spike_times)} units")
print(f"Example unit ({list(spike_times.keys())[0]}) spike count: {len(spike_times[list(spike_times.keys())[0]])}")

# Get LFP data
lfp_data = nwb.lfp
print(f"\nLFP data shape: {lfp_data.data.shape}")
print(f"LFP sampling rate: {lfp_data.rate} Hz")

# Get position data if available
if hasattr(nwb, 'position') and nwb.position is not None:
    position = nwb.position
    print(f"\nPosition data available")
    print(f"Position shape: {position.data.shape}")
else:
    print("\nNo position data found")

# %% [markdown]
# ## Sharp-Wave Ripple Detection

# %%
print("\nDetecting sharp-wave ripples...")

# Use LFP from a single channel (typically a CA1 pyramidal layer)
# Ripples are high-frequency bursts in the 150-200 Hz band
lfp_signal = lfp_data.data[:, 0]  # Use first channel
lfp_rate = lfp_data.rate

# Bandpass filter for ripple detection (150-250 Hz)
from scipy.signal import butter, filtfilt
order = 4
ripple_freqs = [150, 250]
sos = butter(order, ripple_freqs, 'band', fs=lfp_rate, output='sos')
ripple_filtered = filtfilt(sos, lfp_signal, axis=0)

# Compute envelope (amplitude)
ripple_envelope = np.abs(signal.hilbert(ripple_filtered))

# Smooth envelope
smoothing_window = int(lfp_rate * 0.01)  # 10 ms window
ripple_envelope_smooth = gaussian_filter1d(ripple_envelope, sigma=smoothing_window/4)

# Detect ripples: threshold at 2 SD above mean
ripple_threshold = np.mean(ripple_envelope_smooth) + 2 * np.std(ripple_envelope_smooth)
ripple_detection = ripple_envelope_smooth > ripple_threshold

# Find ripple events (continuous regions above threshold)
ripple_starts = np.where(np.diff(ripple_detection.astype(int)) == 1)[0]
ripple_ends = np.where(np.diff(ripple_detection.astype(int)) == -1)[0]

# Ensure we have matching starts and ends
if len(ripple_starts) > len(ripple_ends):
    ripple_starts = ripple_starts[:len(ripple_ends)]
elif len(ripple_ends) > len(ripple_starts):
    ripple_ends = ripple_ends[:len(ripple_starts)]

# Convert to time (seconds)
lfp_times = np.arange(len(lfp_signal)) / lfp_rate
ripple_times = np.column_stack([lfp_times[ripple_starts], lfp_times[ripple_ends]])

# Filter ripples by minimum duration (30 ms)
min_ripple_duration = 0.03
ripple_durations = ripple_times[:, 1] - ripple_times[:, 0]
valid_ripples = ripple_durations > min_ripple_duration
ripple_times = ripple_times[valid_ripples]

print(f"Detected {len(ripple_times)} ripple events")
if len(ripple_times) > 0:
    print(f"Ripple duration: {ripple_durations[valid_ripples].mean()*1000:.1f} ± {ripple_durations[valid_ripples].std()*1000:.1f} ms")
    print(f"Ripple rate: {len(ripple_times) / (lfp_times[-1] - lfp_times[0])} ripples/minute")

# %% [markdown]
# ## Identify Place Cells and Create Rate Maps

# %%
print("\nIdentifying place cells and computing spatial tuning...")

# Use position to identify place cells
if position is not None:
    position_data = position.data
    position_times = position.t
    
    # Interpolate position to a common timeline
    common_rate = 50  # Hz
    common_times = np.arange(0, position_times[-1], 1/common_rate)
    
    # Linearly interpolate position
    x_interp = np.interp(common_times, position_times, position_data[:, 0])
    y_interp = np.interp(common_times, position_times, position_data[:, 1])
    
    print(f"Position range: X [{x_interp.min():.1f}, {x_interp.max():.1f}], Y [{y_interp.min():.1f}, {y_interp.max():.1f}]")
    
    # Create spatial bins for place field analysis
    n_bins = 20
    x_bins = np.linspace(x_interp.min(), x_interp.max(), n_bins + 1)
    y_bins = np.linspace(y_interp.min(), y_interp.max(), n_bins + 1)
    
    # Compute firing rate maps for each unit
    place_cell_info = {}
    
    for unit_id in list(spike_times.keys())[:50]:  # Analyze first 50 units
        spikes_unit = spike_times[unit_id]
        
        # Find which spatial bin each spike occurred in
        # Interpolate position at spike times
        spike_x = np.interp(spikes_unit, common_times, x_interp)
        spike_y = np.interp(spikes_unit, common_times, y_interp)
        
        # Only use spikes during valid tracking
        valid_idx = (spikes_unit < position_times[-1])
        spike_x = spike_x[valid_idx]
        spike_y = spike_y[valid_idx]
        spikes_unit = spikes_unit[valid_idx]
        
        if len(spikes_unit) < 10:  # Skip units with few spikes
            continue
        
        # Create 2D rate map
        rate_map = np.zeros((n_bins, n_bins))
        occupancy = np.zeros((n_bins, n_bins))
        
        for i in range(n_bins):
            for j in range(n_bins):
                in_bin = ((spike_x >= x_bins[i]) & (spike_x < x_bins[i+1]) &
                         (spike_y >= y_bins[j]) & (spike_y < y_bins[j+1]))
                
                occ = ((x_interp >= x_bins[i]) & (x_interp < x_bins[i+1]) &
                      (y_interp >= y_bins[j]) & (y_interp < y_bins[j+1]))
                
                occupancy[i, j] = np.sum(occ) / common_rate  # Time in bin
                if occupancy[i, j] > 0:
                    rate_map[i, j] = np.sum(in_bin) / occupancy[i, j]
        
        # Compute spatial information
        firing_rate = len(spikes_unit) / (position_times[-1] - position_times[0])
        if firing_rate > 0:
            entropy = 0
            for i in range(n_bins):
                for j in range(n_bins):
                    if rate_map[i, j] > 0 and occupancy[i, j] > 0:
                        p = occupancy[i, j] / (position_times[-1] - position_times[0])
                        entropy += p * np.log2(rate_map[i, j] / firing_rate)
            
            spatial_info = entropy / firing_rate if firing_rate > 0 else 0
            
            # Identify place cells: spatial info > 0.5 bits/spike
            if spatial_info > 0.5:
                place_cell_info[unit_id] = {
                    'rate_map': rate_map,
                    'spatial_info': spatial_info,
                    'peak_rate': rate_map.max(),
                    'mean_rate': firing_rate,
                    'n_spikes': len(spikes_unit)
                }
    
    print(f"Identified {len(place_cell_info)} place cells (spatial info > 0.5 bits/spike)")
else:
    print("No position data available - cannot identify place cells")

# %% [markdown]
# ## Train Bayesian Decoder on Awake Activity

# %%
print("\nTraining Bayesian position decoder on awake navigation...")

if position is not None and len(place_cell_info) > 10:
    
    place_cell_ids = list(place_cell_info.keys())
    
    # Prepare training data (first 80% of recording)
    train_end_time = position_times[-1] * 0.8
    train_idx = position_times < train_end_time
    
    # Build decoder: for each spatial bin, compute mean firing rates of place cells
    decoder_model = {}
    
    for i in range(n_bins):
        for j in range(n_bins):
            decoder_model[(i, j)] = {}
            
            # Get firing rates in this bin
            in_bin = ((x_interp < x_bins[i+1]) & (x_interp >= x_bins[i]) &
                     (y_interp < y_bins[j+1]) & (y_interp >= y_bins[j]))
            
            if np.sum(in_bin) > 0:
                for unit_id in place_cell_ids:
                    spikes_unit = spike_times[unit_id]
                    # Count spikes in time windows and spatial bin
                    spike_in_bin = np.zeros(len(common_times))
                    for spike_time in spikes_unit:
                        if spike_time < train_end_time:
                            time_idx = np.argmin(np.abs(common_times - spike_time))
                            if time_idx < len(spike_in_bin):
                                spike_in_bin[time_idx] += 1
                    
                    # Compute mean firing rate for this unit in this bin
                    if np.sum(in_bin) > 0:
                        decoder_model[(i, j)][unit_id] = np.mean(spike_in_bin[in_bin])
    
    print(f"Built decoder model with {len(place_cell_ids)} place cells")

# %% [markdown]
# ## Decode Trajectories During Ripples

# %%
print("\nDecoding spatial trajectories during ripple events...")

if position is not None and len(place_cell_info) > 10 and len(ripple_times) > 0:
    
    # For each ripple, decode the spatial trajectory
    ripple_trajectories = []
    ripple_window = 0.2  # ±100 ms window around ripple
    
    for ripple_start, ripple_end in tqdm(ripple_times[:100], desc="Decoding ripples"):  # Process first 100 ripples
        
        # Extract spike counts during ripple in small time windows
        ripple_duration = ripple_end - ripple_start
        n_timepoints = max(5, int(ripple_duration * 1000))  # At least 5 ms resolution
        ripple_time_bins = np.linspace(ripple_start, ripple_end, n_timepoints)
        
        decoded_positions = []
        
        for t_start, t_end in zip(ripple_time_bins[:-1], ripple_time_bins[1:]):
            # Count spikes in this time window for each place cell
            firing_rates = {}
            
            for unit_id in place_cell_ids:
                spikes = spike_times[unit_id]
                spikes_in_window = spikes[(spikes >= t_start) & (spikes < t_end)]
                dt = t_end - t_start
                firing_rates[unit_id] = len(spikes_in_window) / dt if dt > 0 else 0
            
            # Bayesian decoding: P(position | spikes) ∝ P(spikes | position) * P(position)
            best_position = None
            best_likelihood = -np.inf
            
            for (i, j) in decoder_model.keys():
                # Compute likelihood for this position
                log_likelihood = 0
                
                for unit_id in place_cell_ids:
                    expected_rate = decoder_model[(i, j)].get(unit_id, 0.1)
                    actual_rate = firing_rates.get(unit_id, 0)
                    
                    # Poisson likelihood: P(k spikes | λ) = e^(-λ) * λ^k / k!
                    if expected_rate > 0:
                        log_likelihood += actual_rate * np.log(expected_rate) - expected_rate
                
                if log_likelihood > best_likelihood:
                    best_likelihood = log_likelihood
                    best_position = (i, j)
            
            if best_position is not None:
                # Convert bin to position
                i, j = best_position
                x_pos = (x_bins[i] + x_bins[i+1]) / 2
                y_pos = (y_bins[j] + y_bins[j+1]) / 2
                decoded_positions.append([x_pos, y_pos])
        
        if len(decoded_positions) > 2:
            ripple_trajectories.append({
                'times': ripple_time_bins[:-1],
                'positions': np.array(decoded_positions),
                'ripple_start': ripple_start,
                'ripple_end': ripple_end
            })
    
    print(f"\nDecoded {len(ripple_trajectories)} ripple events")

# %% [markdown]
# ## Visualization: Detected Ripples in LFP

# %%
print("\nGenerating visualizations...")

fig, axes = plt.subplots(3, 1, figsize=(14, 8))

# Show a time window with ripples
time_window = [50, 70]  # Show 50-70 seconds
time_mask = (lfp_times >= time_window[0]) & (lfp_times < time_window[1])

# Raw LFP
ax = axes[0]
ax.plot(lfp_times[time_mask], lfp_signal[time_mask], 'k-', alpha=0.5, linewidth=0.5)
ax.set_ylabel('LFP (μV)', fontsize=11)
ax.set_title('Raw LFP Signal', fontsize=12, fontweight='bold')
ax.set_xlim(time_window)

# Ripple-filtered LFP
ax = axes[1]
ax.plot(lfp_times[time_mask], ripple_filtered[time_mask], 'b-', alpha=0.7, linewidth=0.5)
ax.set_ylabel('Ripple Band (μV)', fontsize=11)
ax.set_title('Ripple-Filtered LFP (150-250 Hz)', fontsize=12, fontweight='bold')
ax.set_xlim(time_window)

# Ripple detection
ax = axes[2]
ax.plot(lfp_times[time_mask], ripple_envelope_smooth[time_mask], 'g-', linewidth=1, label='Ripple Envelope')
ax.axhline(ripple_threshold, color='r', linestyle='--', linewidth=2, label='Detection Threshold')
ax.fill_between(lfp_times[time_mask], 0, ripple_envelope_smooth[time_mask], 
                 where=ripple_detection[time_mask], alpha=0.3, color='orange', label='Detected Ripples')

# Mark ripples in this window
for r_start, r_end in ripple_times:
    if r_end > time_window[0] and r_start < time_window[1]:
        ax.axvline(r_start, color='red', linestyle=':', alpha=0.5)

ax.set_xlabel('Time (s)', fontsize=11)
ax.set_ylabel('Envelope', fontsize=11)
ax.set_title('Sharp-Wave Ripple Detection', fontsize=12, fontweight='bold')
ax.legend(loc='upper right', fontsize=10)
ax.set_xlim(time_window)

plt.tight_layout()
plt.savefig('01_ripple_detection.png', dpi=150, bbox_inches='tight')
print("Saved: 01_ripple_detection.png")
plt.close()

# %% [markdown]
# ## Place Cell Rate Maps

# %%
if len(place_cell_info) > 0:
    # Show place cells sorted by spatial information
    sorted_cells = sorted(place_cell_info.items(), 
                         key=lambda x: x[1]['spatial_info'], reverse=True)
    
    n_cells_show = min(12, len(sorted_cells))
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    axes = axes.flatten()
    
    for idx, (unit_id, info) in enumerate(sorted_cells[:n_cells_show]):
        ax = axes[idx]
        rate_map = info['rate_map']
        
        # Smooth rate map for visualization
        rate_map_smooth = gaussian_filter1d(gaussian_filter1d(rate_map, sigma=1), sigma=1, axis=1)
        
        im = ax.imshow(rate_map_smooth, cmap='hot', origin='lower', aspect='auto')
        ax.set_title(f'Unit {unit_id}\nSI={info["spatial_info"]:.2f}', fontsize=9)
        ax.set_xlabel('X (bins)', fontsize=8)
        ax.set_ylabel('Y (bins)', fontsize=8)
        plt.colorbar(im, ax=ax, label='Hz', fraction=0.046, pad=0.04)
    
    for idx in range(n_cells_show, len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle('Place Cell Rate Maps (Top 12 by Spatial Information)', 
                 fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('02_place_cell_maps.png', dpi=150, bbox_inches='tight')
    print("Saved: 02_place_cell_maps.png")
    plt.close()

# %% [markdown]
# ## Decoded Trajectories During Ripples

# %%
if len(ripple_trajectories) > 0:
    # Show decoded trajectories for example ripples
    n_examples = min(9, len(ripple_trajectories))
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    axes = axes.flatten()
    
    for idx in range(n_examples):
        ax = axes[idx]
        trajectory = ripple_trajectories[idx]
        positions = trajectory['positions']
        
        # Plot trajectory
        ax.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2, alpha=0.7, marker='o', markersize=4)
        ax.plot(positions[0, 0], positions[0, 1], 'go', markersize=8, label='Start', zorder=5)
        ax.plot(positions[-1, 0], positions[-1, 1], 'r^', markersize=8, label='End', zorder=5)
        
        # Set limits to full environment
        if position is not None:
            ax.set_xlim([x_interp.min(), x_interp.max()])
            ax.set_ylim([y_interp.min(), y_interp.max()])
        
        duration = trajectory['ripple_end'] - trajectory['ripple_start']
        ax.set_title(f'Ripple {idx+1} ({duration*1000:.0f}ms)', fontsize=10)
        ax.set_aspect('equal')
        ax.set_xlabel('X (cm)', fontsize=9)
        ax.set_ylabel('Y (cm)', fontsize=9)
        
        if idx == 0:
            ax.legend(fontsize=8)
    
    for idx in range(n_examples, len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle('Decoded Spatial Trajectories During Ripples\n(Bayesian Decoding from Place Cells)', 
                 fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('03_decoded_trajectories.png', dpi=150, bbox_inches='tight')
    print("Saved: 03_decoded_trajectories.png")
    plt.close()

# %% [markdown]
# ## Summary Statistics

# %%
fig, axes = plt.subplots(2, 2, figsize=(11, 8))

# Plot 1: Spatial information distribution
if len(place_cell_info) > 0:
    ax = axes[0, 0]
    spatial_infos = [info['spatial_info'] for info in place_cell_info.values()]
    ax.hist(spatial_infos, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Place cell threshold')
    ax.set_xlabel('Spatial Information (bits/spike)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Spatial Information of Identified Units', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

# Plot 2: Ripple event statistics
ax = axes[0, 1]
if len(ripple_times) > 0:
    ripple_durations_ms = (ripple_times[:, 1] - ripple_times[:, 0]) * 1000
    ax.hist(ripple_durations_ms, bins=30, color='coral', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Ripple Duration (ms)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title(f'Ripple Event Duration Distribution (n={len(ripple_times)})', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add statistics
    stats_text = f"Mean: {ripple_durations_ms.mean():.1f} ms\nMedian: {np.median(ripple_durations_ms):.1f} ms"
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, 
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=10)

# Plot 3: Ripple rate over time
ax = axes[1, 0]
if len(ripple_times) > 0 and position is not None:
    session_duration = position_times[-1]
    time_bins = np.arange(0, session_duration, 60)  # 1-minute bins
    ripple_counts, _ = np.histogram(ripple_times[:, 0], bins=time_bins)
    bin_centers = (time_bins[:-1] + time_bins[1:]) / 2
    
    ax.bar(bin_centers / 60, ripple_counts, width=0.8, color='lightgreen', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Time (minutes)', fontsize=11)
    ax.set_ylabel('Ripples per Minute', fontsize=11)
    ax.set_title('Ripple Rate Over Recording Session', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

# Plot 4: Decoded trajectory speeds
ax = axes[1, 1]
if len(ripple_trajectories) > 0:
    trajectory_speeds = []
    for traj in ripple_trajectories:
        positions = traj['positions']
        if len(positions) > 1:
            distances = np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))
            times = np.diff(traj['times'])
            speeds = distances / times
            trajectory_speeds.extend(speeds)
    
    if trajectory_speeds:
        ax.hist(trajectory_speeds, bins=25, color='mediumpurple', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Trajectory Speed (cm/s)', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title('Speed Distribution of Decoded Trajectories', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add statistics
        stats_text = f"Mean: {np.mean(trajectory_speeds):.1f} cm/s\nMedian: {np.median(trajectory_speeds):.1f} cm/s"
        ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=10)

plt.tight_layout()
plt.savefig('04_summary_statistics.png', dpi=150, bbox_inches='tight')
print("Saved: 04_summary_statistics.png")
plt.close()

# %% [markdown]
# ## Analysis Summary

print("\n" + "="*70)
print("HIPPOCAMPAL REPLAY ANALYSIS COMPLETE")
print("="*70)

if position is not None:
    print(f"\nRecording Duration: {position_times[-1]:.1f} seconds ({position_times[-1]/60:.1f} minutes)")
    print(f"\nPlace Cells Identified: {len(place_cell_info)}")
    print(f"  Mean Spatial Information: {np.mean([info['spatial_info'] for info in place_cell_info.values()]):.2f} bits/spike")
    
    if len(ripple_times) > 0:
        ripple_durations_ms = (ripple_times[:, 1] - ripple_times[:, 0]) * 1000
        print(f"\nSharp-Wave Ripples Detected: {len(ripple_times)}")
        print(f"  Duration: {ripple_durations_ms.mean():.1f} ± {ripple_durations_ms.std():.1f} ms")
        print(f"  Rate: {len(ripple_times) / (position_times[-1]/60):.1f} ripples/minute")
        
        if len(ripple_trajectories) > 0:
            print(f"\nDecoded Trajectories: {len(ripple_trajectories)} ripple events")
            
            trajectory_speeds = []
            trajectory_lengths = []
            for traj in ripple_trajectories:
                positions = traj['positions']
                if len(positions) > 1:
                    distances = np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))
                    times = np.diff(traj['times'])
                    speeds = distances / times
                    trajectory_speeds.extend(speeds)
                    trajectory_lengths.append(np.sum(distances))
            
            if trajectory_speeds:
                print(f"  Trajectory Speed: {np.mean(trajectory_speeds):.1f} ± {np.std(trajectory_speeds):.1f} cm/s")
                print(f"  Mean Trajectory Length: {np.mean(trajectory_lengths):.1f} cm")
                print(f"  Trajectory Compression Ratio: {np.mean(trajectory_lengths) / 100:.2f}x (assuming ~100cm typical awake path)")

print("\nFigures generated:")
print("  - 01_ripple_detection.png: LFP analysis and ripple detection")
print("  - 02_place_cell_maps.png: Spatial firing patterns of place cells")
print("  - 03_decoded_trajectories.png: Decoded spatial trajectories during ripples")
print("  - 04_summary_statistics.png: Statistical summary of analysis")
print("\n" + "="*70)
