# %% [markdown]
# # Sharp-Wave Ripples and Replay in the Hippocampus
#
# This analysis demonstrates the detection and characterization of sharp-wave ripples
# (100-250 Hz transient network events) in hippocampal local field potentials (LFP)
# and the identification of neuronal replay during these events. Sharp-wave ripples
# occur predominantly during sleep and quiet wakefulness and are thought to mediate
# the consolidation of spatial memories through the reactivation of place cell
# ensembles that were active during exploration.

# %% [markdown]
# ## Setup and Data Loading

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 10

print("Loading hippocampal recording data...")

# Load the realistic dataset
data_file = Path('/tmp/hippocampus_ripple_data.pkl')

with open(data_file, 'rb') as f:
    data = pickle.load(f)

spike_times_dict = data['spike_times']
unit_ids = data['unit_ids']
lfp = data['lfp']
t_lfp = data['t_lfp']
lfp_sr = data['lfp_sampling_rate']
duration = data['duration']
ripple_event_times_ground_truth = data['ripple_times']
states = data['states']
state_duration = data['state_duration']

print(f"✓ Loaded: {len(unit_ids)} neurons, {len(lfp)} LFP samples, {len(ripple_event_times_ground_truth)} ripples")

# Create behavioral state masks
n_segments = len(states)
state_durations_s = np.array([state_duration] * n_segments)
state_starts = np.arange(n_segments) * state_duration
state_ends = state_starts + state_duration

exploration_mask = (states == 'exploration')
sleep_mask = (states == 'sleep')

exploration_times = t_lfp[(t_lfp >= state_starts[exploration_mask][0]) &
                          (t_lfp < state_ends[exploration_mask][-1])] if np.any(exploration_mask) else np.array([])
sleep_times = []
for i, is_sleep in enumerate(sleep_mask):
    if is_sleep:
        sleep_mask_time = (t_lfp >= state_starts[i]) & (t_lfp < state_ends[i])
        sleep_times.extend(t_lfp[sleep_mask_time])
sleep_times = np.array(sleep_times)

print(f"✓ Exploration: {len(exploration_times)} samples ({len(exploration_times)/len(t_lfp)*100:.1f}%)")
print(f"✓ Sleep: {len(sleep_times)} samples ({len(sleep_times)/len(t_lfp)*100:.1f}%)")

# %% [markdown]
# ## Ripple Detection from LFP

# %%
print("\n" + "="*60)
print("RIPPLE DETECTION")
print("="*60)

# Bandpass filter LFP to ripple frequency band (100-250 Hz)
print("\nFiltering LFP to ripple band (100-250 Hz)...")

order = 4
sos = signal.butter(order, [100, 250], btype='band', fs=lfp_sr, output='sos')
lfp_ripple = signal.sosfilt(sos, lfp)

# Compute envelope using Hilbert transform
print("Computing ripple envelope...")
analytic_signal = signal.hilbert(lfp_ripple)
ripple_envelope = np.abs(analytic_signal)

# Smooth envelope
window_size = int(0.05 * lfp_sr)
ripple_envelope_smooth = signal.savgol_filter(ripple_envelope, window_size | 1, 3)

# Threshold for ripple detection
if len(sleep_times) > 0:
    sleep_indices = np.searchsorted(t_lfp, sleep_times)
    sleep_indices = sleep_indices[sleep_indices < len(ripple_envelope_smooth)]
    sleep_envelope = ripple_envelope_smooth[sleep_indices]
    threshold_std = 2
    ripple_threshold = np.mean(sleep_envelope) + threshold_std * np.std(sleep_envelope)
else:
    ripple_threshold = np.mean(ripple_envelope_smooth) + 2 * np.std(ripple_envelope_smooth)

print(f"Ripple threshold: {ripple_threshold:.2f}")

# Detect ripples
ripple_detected = ripple_envelope_smooth > ripple_threshold
diff = np.diff(ripple_detected.astype(int))
ripple_starts = np.where(diff == 1)[0]
ripple_ends = np.where(diff == -1)[0]

if len(ripple_starts) > len(ripple_ends):
    ripple_starts = ripple_starts[:len(ripple_ends)]
elif len(ripple_ends) > len(ripple_starts):
    ripple_ends = ripple_ends[:len(ripple_starts)]

ripple_starts_s = t_lfp[ripple_starts]
ripple_ends_s = t_lfp[ripple_ends]
ripple_durations_detected = ripple_ends_s - ripple_starts_s

# Filter short ripples
min_ripple_duration = 0.03
valid_ripples = ripple_durations_detected > min_ripple_duration
ripple_starts_s = ripple_starts_s[valid_ripples]
ripple_ends_s = ripple_ends_s[valid_ripples]
ripple_durations_detected = ripple_durations_detected[valid_ripples]

print(f"\n✓ Detected {len(ripple_starts_s)} ripples")
print(f"  Mean duration: {np.mean(ripple_durations_detected)*1000:.1f} ms")
print(f"  Min: {np.min(ripple_durations_detected)*1000:.1f} ms, Max: {np.max(ripple_durations_detected)*1000:.1f} ms")

# %% [markdown]
# ## Ripple-Modulated Neurons

# %%
print("\n" + "="*60)
print("RIPPLE-MODULATED NEURONS")
print("="*60)

print("\nAnalyzing neuron firing during ripples...")

firing_rates_during_ripple = {}
firing_rates_outside_ripple = {}
ripple_mod_index = {}

for unit_id in tqdm(unit_ids, desc="Computing ripple modulation"):
    spikes = spike_times_dict[unit_id]

    # Count spikes during ripples
    in_ripple = np.zeros(len(spikes), dtype=bool)
    for start, end in zip(ripple_starts_s, ripple_ends_s):
        in_ripple |= (spikes >= start) & (spikes < end)
    n_spikes_ripple = np.sum(in_ripple)

    # Count spikes during sleep outside ripples
    in_sleep = np.zeros(len(spikes), dtype=bool)
    for i in np.where(sleep_mask)[0]:
        start = state_starts[i]
        end = state_ends[i]
        in_sleep |= (spikes >= start) & (spikes < end)

    in_sleep_outside_ripple = in_sleep & ~in_ripple
    n_spikes_outside = np.sum(in_sleep_outside_ripple)

    # Firing rates
    ripple_duration_total = np.sum(ripple_durations_detected)
    sleep_duration_outside_ripple = np.sum(state_durations_s[sleep_mask]) - ripple_duration_total

    rate_ripple = n_spikes_ripple / ripple_duration_total if ripple_duration_total > 0 else 0
    rate_outside = n_spikes_outside / sleep_duration_outside_ripple if sleep_duration_outside_ripple > 0 else 0

    firing_rates_during_ripple[unit_id] = rate_ripple
    firing_rates_outside_ripple[unit_id] = rate_outside

    # Modulation index
    if (rate_ripple + rate_outside) > 0:
        mod_index = (rate_ripple - rate_outside) / (rate_ripple + rate_outside)
    else:
        mod_index = 0

    ripple_mod_index[unit_id] = mod_index

# Identify ripple-modulated neurons
mod_indices = list(ripple_mod_index.values())
ripple_mod_threshold = np.percentile(mod_indices, 80)
ripple_modulated_units = [uid for uid, idx in ripple_mod_index.items() if idx >= ripple_mod_threshold]

print(f"✓ Identified {len(ripple_modulated_units)} ripple-modulated neurons (top 20%)")
print(f"  Mean modulation index: {np.mean([ripple_mod_index[uid] for uid in ripple_modulated_units]):.2f}")

# %% [markdown]
# ## Spike Timing During Ripples

# %%
print("\n" + "="*60)
print("SPIKE TIMING PRECISION")
print("="*60)

print("\nAnalyzing spike timing during ripples...")

spike_precision_data = []
ripple_peak_times = []

for ripple_start, ripple_end in tqdm(zip(ripple_starts_s, ripple_ends_s),
                                      total=len(ripple_starts_s), desc="Ripples"):
    idx_start = int(ripple_start * lfp_sr)
    idx_end = int(ripple_end * lfp_sr)

    if idx_end > idx_start + 10:
        ripple_peak_idx = idx_start + np.argmax(ripple_envelope[idx_start:idx_end])
        ripple_peak_time = t_lfp[ripple_peak_idx]
        ripple_peak_times.append(ripple_peak_time)

        # Spike timing relative to ripple
        for unit_id in ripple_modulated_units:
            spikes = spike_times_dict[unit_id]
            in_ripple = (spikes >= ripple_start) & (spikes < ripple_end)
            spikes_in_ripple = spikes[in_ripple]

            for spike_time in spikes_in_ripple:
                spike_times_relative = (spike_time - ripple_peak_time) * 1000
                spike_precision_data.append({
                    'time_to_peak_ms': spike_times_relative,
                    'ripple_idx': len(ripple_peak_times) - 1
                })

print(f"✓ Recorded {len(spike_precision_data)} spike-to-ripple relationships")

if len(spike_precision_data) > 0:
    spike_times_ms = [d['time_to_peak_ms'] for d in spike_precision_data]
    print(f"  Spike timing: mean={np.mean(spike_times_ms):.1f}ms, std={np.std(spike_times_ms):.1f}ms")

# %% [markdown]
# ## Neuronal Replay Detection

# %%
print("\n" + "="*60)
print("NEURONAL REPLAY ANALYSIS")
print("="*60)

print("\nDetecting replay sequences...")

# Build exploration reference
print("Building exploration reference...")
neuron_peak_times_explore = {}

for unit_id in ripple_modulated_units:
    spikes = spike_times_dict[unit_id]

    # Find spikes during exploration
    in_explore = np.zeros(len(spikes), dtype=bool)
    for i in np.where(exploration_mask)[0]:
        start = state_starts[i]
        end = state_ends[i]
        in_explore |= (spikes >= start) & (spikes < end)

    spikes_explore = spikes[in_explore]

    if len(spikes_explore) > 10:
        # Peak firing time during exploration
        hist, bin_edges = np.histogram(spikes_explore, bins=len(t_lfp))
        window_size = int(2 * lfp_sr)
        spike_counts = np.convolve(hist, np.ones(window_size), mode='same')

        if np.max(spike_counts) > 0:
            peak_idx = np.argmin(np.abs(bin_edges[:-1] - t_lfp[np.argmax(spike_counts)]))
            neuron_peak_times_explore[unit_id] = bin_edges[peak_idx]

print(f"✓ Built exploration reference for {len(neuron_peak_times_explore)} neurons")

# Detect replay
print("\nDetecting replay in ripples...")

replay_events = []
replay_quality_scores = []

for ripple_idx, (ripple_start, ripple_end) in enumerate(zip(ripple_starts_s, ripple_ends_s)):
    ripple_duration_ms = (ripple_end - ripple_start) * 1000

    spikes_in_ripple = {}
    for unit_id in ripple_modulated_units:
        if unit_id in neuron_peak_times_explore:
            spikes = spike_times_dict[unit_id]
            in_ripple = (spikes >= ripple_start) & (spikes < ripple_end)
            spikes_times = spikes[in_ripple]
            if len(spikes_times) > 0:
                spikes_in_ripple[unit_id] = spikes_times - ripple_start

    if len(spikes_in_ripple) >= 3:
        # Build spike order
        spikes_list = []
        neurons_list = []
        for unit_id, spike_times in spikes_in_ripple.items():
            for st in spike_times:
                spikes_list.append(st)
                neurons_list.append(unit_id)

        if len(spikes_list) > 1:
            spikes_array = np.array(spikes_list)
            order_indices = np.argsort(spikes_array)
            ripple_order = [neurons_list[i] for i in order_indices]

            # Compute replay quality (rank correlation)
            explore_order = sorted(set(ripple_order),
                                   key=lambda x: neuron_peak_times_explore.get(x, 0))

            unique_neurons = list(set(ripple_order))
            if len(unique_neurons) > 1:
                try:
                    ripple_ranks = [unique_neurons.index(n) for n in ripple_order]
                    explore_ranks = [explore_order.index(n) for n in ripple_order]
                    corr, _ = spearmanr(ripple_ranks, explore_ranks)
                    quality = max(0, corr)
                except:
                    quality = 0
            else:
                quality = 0

            replay_speed = 60 / ripple_duration_ms if ripple_duration_ms > 0 else 0

            replay_events.append({
                'ripple_start': ripple_start,
                'ripple_end': ripple_end,
                'n_neurons_active': len(spikes_in_ripple),
                'quality_score': quality,
                'replay_speed_compression': replay_speed,
                'n_spikes': len(spikes_list)
            })

            replay_quality_scores.append(quality)

print(f"✓ Detected {len(replay_events)} ripples with replay")

if len(replay_quality_scores) > 0:
    print(f"  Quality: mean={np.mean(replay_quality_scores):.3f}, median={np.median(replay_quality_scores):.3f}")

high_quality_threshold = np.percentile(replay_quality_scores, 75) if len(replay_quality_scores) > 0 else 0.5
high_quality_replays = [r for r in replay_events if r['quality_score'] >= high_quality_threshold]

print(f"✓ Identified {len(high_quality_replays)} high-quality replay events")

# %% [markdown]
# ## Visualization

# %%
print("\n" + "="*60)
print("CREATING VISUALIZATIONS")
print("="*60)

# Figure 1: LFP and ripple detection
print("\nFigure 1: LFP and ripple detection...")

fig, axes = plt.subplots(3, 1, figsize=(14, 8))

plot_window = slice(0, min(10 * lfp_sr, len(t_lfp)))

ax = axes[0]
ax.plot(t_lfp[plot_window], lfp[plot_window], 'k-', linewidth=0.5, label='Raw LFP')
ax.set_ylabel('LFP (mV)', fontsize=10)
ax.set_title('Hippocampal LFP with Detected Ripples', fontsize=12, fontweight='bold')
ax.legend(loc='upper right')
ax.set_xlim([t_lfp[plot_window.start], t_lfp[plot_window.stop-1]])

ax = axes[1]
ax.plot(t_lfp[plot_window], lfp_ripple[plot_window], 'b-', linewidth=0.8, label='Ripple band (100-250 Hz)')
ax.plot(t_lfp[plot_window], ripple_envelope_smooth[plot_window], 'r-', linewidth=1.5, label='Envelope')
ax.axhline(ripple_threshold, color='r', linestyle='--', linewidth=1, label='Threshold')
ax.set_ylabel('Amplitude', fontsize=10)
ax.legend(loc='upper right')
ax.set_xlim([t_lfp[plot_window.start], t_lfp[plot_window.stop-1]])

ax = axes[2]
ax.plot(t_lfp[plot_window], lfp[plot_window], 'k-', linewidth=0.5, alpha=0.5)

for ripple_start, ripple_end in zip(ripple_starts_s, ripple_ends_s):
    if ripple_start >= t_lfp[plot_window.start] and ripple_end <= t_lfp[plot_window.stop-1]:
        ax.axvspan(ripple_start, ripple_end, alpha=0.3, color='red')

ax.set_xlabel('Time (s)', fontsize=10)
ax.set_ylabel('LFP (mV)', fontsize=10)
ax.set_xlim([t_lfp[plot_window.start], t_lfp[plot_window.stop-1]])

plt.tight_layout()
plt.savefig('01_ripple_detection.png', dpi=150, bbox_inches='tight')
print("✓ Saved: 01_ripple_detection.png")
plt.close()

# Figure 2: Ripple modulation
print("\nFigure 2: Ripple modulation...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

mod_indices_sorted = sorted(ripple_mod_index.items(), key=lambda x: x[1], reverse=True)
units_sorted, indices_sorted = zip(*mod_indices_sorted)

ax = axes[0]
colors = ['red' if uid in ripple_modulated_units else 'gray' for uid in units_sorted]
ax.barh(range(len(indices_sorted)), indices_sorted, color=colors, alpha=0.7)
ax.set_xlabel('Ripple Modulation Index', fontsize=11)
ax.set_title('Ripple-Modulated Neurons', fontsize=12, fontweight='bold')
ax.axvline(ripple_mod_threshold, color='red', linestyle='--', linewidth=1)
ax.set_yticks([])

ax = axes[1]
ax.hist(mod_indices, bins=15, alpha=0.7, color='steelblue', edgecolor='black')
ax.axvline(np.mean(mod_indices), color='red', linestyle='--', linewidth=2,
           label=f'Mean: {np.mean(mod_indices):.2f}')
ax.set_xlabel('Modulation Index', fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.set_title('Distribution of Modulation Indices', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('02_ripple_modulation.png', dpi=150, bbox_inches='tight')
print("✓ Saved: 02_ripple_modulation.png")
plt.close()

# Figure 3: Firing rate increase
print("\nFigure 3: Firing rate increase...")

fig, ax = plt.subplots(figsize=(10, 6))

firing_increase = []
for uid in ripple_modulated_units:
    rate_during = firing_rates_during_ripple[uid]
    rate_outside = firing_rates_outside_ripple[uid]
    if rate_outside > 0:
        firing_increase.append(rate_during / rate_outside)
    else:
        firing_increase.append(1.0)

ax.hist(firing_increase, bins=15, alpha=0.7, color='forestgreen', edgecolor='black')
ax.axvline(np.median(firing_increase), color='red', linestyle='--', linewidth=2,
           label=f'Median: {np.median(firing_increase):.1f}x')
ax.set_xlabel('Firing Rate Increase During Ripples (fold)', fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.set_title('Ripple-Induced Firing Rate Increase', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('03_firing_rate_increase.png', dpi=150, bbox_inches='tight')
print("✓ Saved: 03_firing_rate_increase.png")
plt.close()

# Figure 4: Spike timing precision
print("\nFigure 4: Spike timing precision...")

fig, ax = plt.subplots(figsize=(10, 6))

if len(spike_precision_data) > 0:
    spike_times_ms = [d['time_to_peak_ms'] for d in spike_precision_data]
    ax.hist(spike_times_ms, bins=50, alpha=0.7, color='purple', edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Ripple peak')
    ax.axvline(np.mean(spike_times_ms), color='orange', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(spike_times_ms):.1f} ms')
    ax.set_xlabel('Spike Time Relative to Ripple Peak (ms)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Spike Timing Precision During Ripples', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('04_spike_timing_precision.png', dpi=150, bbox_inches='tight')
print("✓ Saved: 04_spike_timing_precision.png")
plt.close()

# Figure 5: Replay analysis
print("\nFigure 5: Neuronal replay...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

ax = axes[0, 0]
if len(replay_quality_scores) > 0:
    ax.hist(replay_quality_scores, bins=20, alpha=0.7, color='teal', edgecolor='black')
    ax.axvline(high_quality_threshold, color='red', linestyle='--', linewidth=2, label='High-quality')
    ax.set_xlabel('Replay Quality Score', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Replay Quality Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

ax = axes[0, 1]
if len(replay_events) > 0:
    n_neurons = [r['n_neurons_active'] for r in replay_events]
    ax.hist(n_neurons, bins=range(min(n_neurons), max(n_neurons)+2), alpha=0.7,
            color='darkorange', edgecolor='black')
    ax.set_xlabel('Number of Active Neurons', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Neuronal Participation in Ripples', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

ax = axes[1, 0]
if len(high_quality_replays) > 0:
    replay_speeds = [r['replay_speed_compression'] for r in high_quality_replays]
    ax.hist(replay_speeds, bins=15, alpha=0.7, color='crimson', edgecolor='black')
    ax.axvline(np.mean(replay_speeds), color='blue', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(replay_speeds):.0f}x')
    ax.set_xlabel('Temporal Compression (fold)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Replay Temporal Compression', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

ax = axes[1, 1]
if len(replay_events) > 0:
    quality = [r['quality_score'] for r in replay_events]
    n_neurons = [r['n_neurons_active'] for r in replay_events]
    colors = ['red' if q >= high_quality_threshold else 'gray' for q in quality]
    ax.scatter(n_neurons, quality, alpha=0.6, s=50, c=colors)
    ax.axhline(high_quality_threshold, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Number of Active Neurons', fontsize=11)
    ax.set_ylabel('Replay Quality Score', fontsize=11)
    ax.set_title('Quality vs Participation', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('05_replay_analysis.png', dpi=150, bbox_inches='tight')
print("✓ Saved: 05_replay_analysis.png")
plt.close()

# Figure 6: Example raster
print("\nFigure 6: Example replay raster...")

if len(high_quality_replays) > 0:
    best_ripple = max(high_quality_replays, key=lambda x: x['quality_score'])
    ripple_center = (best_ripple['ripple_start'] + best_ripple['ripple_end']) / 2
    plot_start = ripple_center - 1.0
    plot_end = ripple_center + 1.0

    fig, ax = plt.subplots(figsize=(12, 8))

    y_pos = 0
    for unit_id in sorted(ripple_modulated_units):
        spikes = spike_times_dict[unit_id]
        in_window = (spikes >= plot_start) & (spikes < plot_end)
        spikes_window = spikes[in_window]

        for spike_time in spikes_window:
            ax.plot([spike_time, spike_time], [y_pos - 0.4, y_pos + 0.4], 'k-', linewidth=0.5)

        y_pos += 1

    ax.axvspan(best_ripple['ripple_start'], best_ripple['ripple_end'],
               alpha=0.2, color='red', label='Ripple')

    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Neuron', fontsize=11)
    ax.set_title(f'Example High-Quality Replay\n(Quality: {best_ripple["quality_score"]:.3f}, ' +
                 f'{best_ripple["n_neurons_active"]} neurons, {best_ripple["replay_speed_compression"]:.0f}x)',
                 fontsize=12, fontweight='bold')
    ax.set_xlim([plot_start, plot_end])
    ax.set_ylim([-0.5, y_pos + 0.5])
    ax.legend()
    ax.grid(alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig('06_example_replay_raster.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: 06_example_replay_raster.png")
    plt.close()

# Figure 7: Summary
print("\nFigure 7: Summary statistics...")

fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)

# Ripples over time
ax = fig.add_subplot(gs[0, :])
time_bins = np.arange(0, duration, 60)
ripple_counts, _ = np.histogram(ripple_event_times_ground_truth, bins=time_bins)
ax.bar(time_bins[:-1]/60, ripple_counts, width=0.8, alpha=0.7, color='steelblue', edgecolor='black')
ax.set_xlabel('Time (min)', fontsize=11)
ax.set_ylabel('Ripples per Minute', fontsize=11)
ax.set_title('Ripple Occurrence Over Time', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# Duration distribution
ax = fig.add_subplot(gs[1, 0])
ax.hist(ripple_durations_detected*1000, bins=20, alpha=0.7, color='coral', edgecolor='black')
ax.set_xlabel('Duration (ms)', fontsize=10)
ax.set_ylabel('Count', fontsize=10)
ax.set_title('Ripple Duration', fontsize=11, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# State breakdown
ax = fig.add_subplot(gs[1, 1])
state_names = np.array(['Exploration', 'Sleep'])
state_mask_explore = exploration_mask
state_mask_sleep = sleep_mask
ripple_rate_explore = len(ripple_starts_s[ripple_starts_s < state_starts[state_mask_explore][-1] + state_duration]) / (np.sum(state_durations_s[state_mask_explore]) / 60) if np.sum(state_mask_explore) > 0 else 0
ripple_rate_sleep = len(ripple_starts_s) / (np.sum(state_durations_s[state_mask_sleep]) / 60) if np.sum(state_mask_sleep) > 0 else 0

ax.bar(['Exploration', 'Sleep'], [ripple_rate_explore, ripple_rate_sleep], alpha=0.7, color='lightseagreen', edgecolor='black')
ax.set_ylabel('Ripples per Minute', fontsize=10)
ax.set_title('Ripple Rate by State', fontsize=11, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# Participation
ax = fig.add_subplot(gs[1, 2])
sizes = [len(ripple_modulated_units), len(unit_ids)-len(ripple_modulated_units)]
colors_pie = ['#ff9999', '#cccccc']
labels = [f'Ripple-mod\n(n={len(ripple_modulated_units)})',
          f'Other\n(n={len(unit_ids)-len(ripple_modulated_units)})']
ax.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90)
ax.set_title('Neuron Participation', fontsize=11, fontweight='bold')

# Replay rate
ax = fig.add_subplot(gs[2, 0])
replay_rate = len(replay_events) / len(ripple_starts_s) * 100 if len(ripple_starts_s) > 0 else 0
hq_replay_rate = len(high_quality_replays) / len(ripple_starts_s) * 100 if len(ripple_starts_s) > 0 else 0
ax.bar(['Any Replay', 'High Quality'], [replay_rate, hq_replay_rate], alpha=0.7, color='gold', edgecolor='black')
ax.set_ylabel('Percent of Ripples', fontsize=10)
ax.set_title('Ripples with Replay', fontsize=11, fontweight='bold')
ax.set_ylim([0, 100])
ax.grid(axis='y', alpha=0.3)

# Summary text
ax = fig.add_subplot(gs[2, 1:])
ax.axis('off')

spike_timing_precision = np.std(spike_times_ms) if len(spike_precision_data) > 0 else 0
mean_compression = np.mean([r['replay_speed_compression'] for r in high_quality_replays]) if len(high_quality_replays) > 0 else 0

summary_text = f"""Key Findings:

• Total Ripples: {len(ripple_starts_s)}
  Mean duration: {np.mean(ripple_durations_detected)*1000:.1f} ms
  Rate during sleep: {len(ripple_starts_s) / (np.sum(state_durations_s[state_mask_sleep])/60):.2f} /min

• Ripple-Modulated Neurons: {len(ripple_modulated_units)}/{len(unit_ids)} ({len(ripple_modulated_units)/len(unit_ids)*100:.0f}%)
  Median firing increase: {np.median(firing_increase):.1f}x

• High-Quality Replay Events: {len(high_quality_replays)}/{len(ripple_starts_s)} ({hq_replay_rate:.0f}%)
  Mean temporal compression: {mean_compression:.0f}x
  Mean neurons per replay: {np.mean([r['n_neurons_active'] for r in high_quality_replays]):.1f}

• Spike Timing Precision: ±{spike_timing_precision:.1f} ms
"""

ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.savefig('07_summary_statistics.png', dpi=150, bbox_inches='tight')
print("✓ Saved: 07_summary_statistics.png")
plt.close()

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
print(f"\nGenerated 7 figures demonstrating:")
print(f"  • Sharp-wave ripple detection from LFP")
print(f"  • Ripple-modulated neuron identification")
print(f"  • Spike timing precision during ripples")
print(f"  • Neuronal replay during ripple events")
