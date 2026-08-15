# %% [markdown]
# # Theta Phase Precession in Hippocampal Place Cells
#
# This analysis demonstrates theta phase precession, a fundamental phenomenon in hippocampal
# spatial coding where place cells fire at progressively earlier phases of the theta oscillation
# as an animal advances through the cell's place field.
#
# ## Key Concept
#
# Theta phase precession is the systematic shift of spike timing relative to theta oscillation
# as an animal traverses through a place cell's receptive field. As the animal moves forward
# through the place field, spikes occur at progressively earlier phases of the ongoing theta
# rhythm, completing a ~360-degree phase shift across the field.
#
# ## Biological Significance
#
# Phase precession is thought to serve multiple functions:
# - **Temporal Coding**: Encoding position through spike timing rather than just firing rate
# - **Sequence Compression**: Multiple place fields are traversed in one theta cycle
# - **Learning**: Phase precession may facilitate synaptic plasticity and memory consolidation
#
# ## Dataset
#
# This analysis uses simulated hippocampal recordings based on well-characterized patterns
# from rodent CA1 recordings, with realistic parameters for:
# - Place cells recorded during spatial navigation
# - Local field potential (LFP) recordings showing theta oscillations
# - Position tracking during behavior
#
# ---

# %% [markdown]
# ## Setup and Data Loading

# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from scipy.signal import hilbert, decimate
from scipy.stats import circmean, circstd
import pynapple as nap
from tqdm import tqdm
import pickle
from pathlib import Path

# Set up plotting style
plt.style.use('seaborn-v0_8-darkgrid')
np.random.seed(42)

# Load simulated data
data_file = Path('simulated_data.pkl')
with open(data_file, 'rb') as f:
    data = pickle.load(f)

spike_trains = data['spike_trains']
place_fields = data['place_fields']
lfp = data['lfp']
position = data['position']
speed = data['speed']
theta_phase = data['theta_phase']
time = data['time']
lfp_sampling_rate = data['lfp_sampling_rate']
theta_freq = data['theta_freq']

print(f"Loaded data:")
print(f"  Recording duration: {time[-1]:.1f} seconds")
print(f"  Number of place cells: {len(spike_trains)}")
print(f"  Total spikes: {sum(len(st['spike_times']) for st in spike_trains)}")
print(f"  Theta frequency: {theta_freq} Hz")

# %% [markdown]
# ## 1. Visualize Raw Data

# %%
fig, axes = plt.subplots(3, 1, figsize=(14, 8))

# Plot 1: Animal position over time
ax = axes[0]
ax.plot(time, position, 'b-', linewidth=1.5, alpha=0.7)
ax.fill_between(time, position.min(), position, alpha=0.2)
ax.set_ylabel('Position (cm)', fontsize=11)
ax.set_title('Animal Position on Track', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Plot 2: Speed
ax = axes[1]
ax.plot(time, speed, 'g-', linewidth=1, alpha=0.6)
ax.axhline(y=2, color='r', linestyle='--', linewidth=1, label='Speed threshold')
ax.set_ylabel('Speed (cm/s)', fontsize=11)
ax.set_title('Running Speed', fontsize=12, fontweight='bold')
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3)

# Plot 3: LFP with theta oscillation
ax = axes[2]
# Show 5-second window for clarity
window_start = 50
window_end = 55
mask = (time >= window_start) & (time <= window_end)
time_window = time[mask]
lfp_window = lfp[mask]
theta_window = 100 * np.sin(theta_phase[mask])

ax.plot(time_window, lfp_window, 'k-', linewidth=1, alpha=0.6, label='LFP')
ax.plot(time_window, theta_window, 'r-', linewidth=2, alpha=0.8, label=f'Theta ({theta_freq} Hz)')
ax.set_ylabel('LFP (μV)', fontsize=11)
ax.set_xlabel('Time (s)', fontsize=11)
ax.set_title(f'Hippocampal LFP with Theta Oscillation (window: {window_start}-{window_end}s)', fontsize=12, fontweight='bold')
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('01_raw_data.png', dpi=150, bbox_inches='tight')
print("Saved: 01_raw_data.png")
plt.close()

# %% [markdown]
# ## 2. Analyze Place Fields

# %%
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Firing rate map for selected neurons
ax = axes[0, 0]
selected_neurons = [0, 5, 10, 15]
colors = plt.cm.tab10(np.linspace(0, 1, len(selected_neurons)))

position_bins = np.linspace(0, 100, 51)
for neuron_idx in selected_neurons:
    st = spike_trains[neuron_idx]
    if len(st['spike_positions']) > 0:
        spike_pos = st['spike_positions']
        spike_times = st['spike_times']

        # Bin spikes by position
        hist, bin_edges = np.histogram(spike_pos, bins=position_bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Normalize by time spent in each bin
        occupancy, _ = np.histogram(position, bins=position_bins)
        occupancy_time = occupancy / lfp_sampling_rate  # Convert to seconds
        occupancy_time[occupancy_time == 0] = 1  # Avoid division by zero

        firing_rate = hist / occupancy_time
        ax.plot(bin_centers, firing_rate, marker='o', markersize=4, label=f'Neuron {neuron_idx}', color=colors[selected_neurons.index(neuron_idx)])

ax.set_xlabel('Position (cm)', fontsize=11)
ax.set_ylabel('Firing Rate (Hz)', fontsize=11)
ax.set_title('Place Field Firing Rate Maps', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Plot 2: Place field statistics
ax = axes[0, 1]
field_centers = [pf['center'] for pf in place_fields]
field_widths = [pf['width'] for pf in place_fields]
peak_rates = [pf['peak_rate'] for pf in place_fields]

scatter = ax.scatter(field_centers, peak_rates, c=field_widths, cmap='viridis', s=100, alpha=0.6)
ax.set_xlabel('Place Field Center (cm)', fontsize=11)
ax.set_ylabel('Peak Firing Rate (Hz)', fontsize=11)
ax.set_title('Place Field Characteristics', fontsize=12, fontweight='bold')
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Field Width (cm)', fontsize=10)
ax.grid(True, alpha=0.3)

# Plot 3: Spike raster (example 5-second window)
ax = axes[1, 0]
window_start, window_end = 80, 90
window_mask = (time >= window_start) & (time <= window_end)

for neuron_idx in range(min(20, len(spike_trains))):
    st = spike_trains[neuron_idx]
    spike_mask = (st['spike_times'] >= window_start) & (st['spike_times'] <= window_end)
    if np.any(spike_mask):
        spike_times = st['spike_times'][spike_mask]
        ax.scatter(spike_times, [neuron_idx] * len(spike_times), marker='|', s=100, color='black')

ax.set_ylabel('Neuron Index', fontsize=11)
ax.set_xlabel('Time (s)', fontsize=11)
ax.set_title(f'Spike Raster Plot (window: {window_start}-{window_end}s)', fontsize=12, fontweight='bold')
ax.set_xlim(window_start, window_end)
ax.grid(True, alpha=0.3, axis='x')

# Plot 4: Number of spikes per neuron
ax = axes[1, 1]
spike_counts = [len(st['spike_times']) for st in spike_trains]
ax.bar(range(len(spike_counts)), spike_counts, color='steelblue', alpha=0.7)
ax.set_xlabel('Neuron Index', fontsize=11)
ax.set_ylabel('Number of Spikes', fontsize=11)
ax.set_title('Spike Count per Neuron', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('02_place_fields.png', dpi=150, bbox_inches='tight')
print("Saved: 02_place_fields.png")
plt.close()

# %% [markdown]
# ## 3. Theta Phase Precession Analysis

# %%
print("\nAnalyzing theta phase precession...")

# For each neuron, collect spikes in place field and their theta phase
precession_data = []

for neuron_idx, st in enumerate(tqdm(spike_trains, desc="Processing neurons")):
    if len(st['spike_times']) < 5:
        continue

    # Get spikes in place field
    in_field_mask = st['in_place_field']

    if np.sum(in_field_mask) < 5:
        continue

    spike_positions = st['spike_positions'][in_field_mask]
    spike_theta_phases = st['spike_theta_phase'][in_field_mask]
    place_field = st['place_field']

    # Normalize position within place field (0 to 1)
    field_min = place_field['center'] - place_field['width']
    field_max = place_field['center'] + place_field['width']
    pos_in_field = (spike_positions - field_min) / (field_max - field_min)
    pos_in_field = np.clip(pos_in_field, 0, 1)

    # Use theta phase directly (already in degrees from simulation)
    theta_degrees = spike_theta_phases % 360

    precession_data.append({
        'neuron_idx': neuron_idx,
        'pos_in_field': pos_in_field,
        'theta_phase_deg': theta_degrees,
        'place_field': place_field,
        'spike_times': st['spike_times'][in_field_mask],
        'n_spikes': len(spike_positions)
    })

print(f"Analyzed {len(precession_data)} neurons with sufficient spikes in place field")

# %% [markdown]
# ## 4. Visualize Phase Precession

# %%
fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# Plot 1: Phase precession for individual neurons
ax = axes[0, 0]
for i, prec in enumerate(precession_data[:4]):  # Show 4 example neurons
    ax.scatter(prec['pos_in_field'], prec['theta_phase_deg'],
              alpha=0.5, s=20, label=f"Neuron {prec['neuron_idx']}")

ax.set_xlabel('Position in Place Field (normalized)', fontsize=11)
ax.set_ylabel('Theta Phase (degrees)', fontsize=11)
ax.set_title('Phase Precession: Individual Neurons', fontsize=12, fontweight='bold')
ax.set_ylim(-20, 380)
ax.legend(fontsize=9, loc='upper left')
ax.grid(True, alpha=0.3)

# Plot 2: Population-level phase precession
ax = axes[0, 1]
all_positions = np.concatenate([p['pos_in_field'] for p in precession_data])
all_phases = np.concatenate([p['theta_phase_deg'] for p in precession_data])

# Bin positions and compute mean phase
pos_bins = np.linspace(0, 1, 11)
bin_centers = (pos_bins[:-1] + pos_bins[1:]) / 2
mean_phases = []
std_phases = []

for i in range(len(pos_bins) - 1):
    mask = (all_positions >= pos_bins[i]) & (all_positions < pos_bins[i+1])
    if np.sum(mask) > 0:
        phases_in_bin = all_phases[mask]
        # Compute circular mean and std
        mean_phase = circmean(np.deg2rad(phases_in_bin)) * 180 / np.pi
        if mean_phase < 0:
            mean_phase += 360
        std_phase = circstd(np.deg2rad(phases_in_bin)) * 180 / np.pi
        mean_phases.append(mean_phase)
        std_phases.append(std_phase)
    else:
        mean_phases.append(np.nan)
        std_phases.append(np.nan)

mean_phases = np.array(mean_phases)
std_phases = np.array(std_phases)

# Remove NaN values for plotting
valid_mask = ~np.isnan(mean_phases)
ax.errorbar(bin_centers[valid_mask], mean_phases[valid_mask], yerr=std_phases[valid_mask],
           fmt='o-', markersize=8, capsize=5, capthick=2, linewidth=2, color='darkblue',
           ecolor='steelblue', label='Population mean ± SD')

# Fit line to show linear phase precession
valid_bins = bin_centers[valid_mask]
valid_means = mean_phases[valid_mask]
if len(valid_bins) > 2:
    z = np.polyfit(valid_bins, valid_means, 1)
    p = np.poly1d(z)
    x_fit = np.linspace(0, 1, 100)
    ax.plot(x_fit, p(x_fit), 'r--', linewidth=2, label=f'Linear fit: slope = {z[0]:.1f}°')

ax.set_xlabel('Position in Place Field (normalized)', fontsize=11)
ax.set_ylabel('Theta Phase (degrees)', fontsize=11)
ax.set_title('Population-Level Phase Precession', fontsize=12, fontweight='bold')
ax.set_ylim(-20, 380)
ax.set_ylim(-20, 380)
ax.legend(fontsize=10, loc='upper left')
ax.grid(True, alpha=0.3)

# Plot 3: Circular histogram of phases at different positions
ax = axes[1, 0]
# Divide place field into three regions: entry, middle, exit
position_labels = ['Entry', 'Middle', 'Exit']
colors_phase = ['#FF6B6B', '#4ECDC4', '#45B7D1']

for region_idx, (pos_min, pos_max) in enumerate([(0, 0.35), (0.35, 0.65), (0.65, 1.0)]):
    mask = (all_positions >= pos_min) & (all_positions < pos_max)
    if np.sum(mask) > 10:
        phases = all_phases[mask]
        ax.hist(phases, bins=24, alpha=0.6, label=position_labels[region_idx],
               color=colors_phase[region_idx], edgecolor='black', linewidth=0.5)

ax.set_xlabel('Theta Phase (degrees)', fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.set_title('Phase Distribution by Place Field Region', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

# Plot 4: Phase-position correlation matrix (heat map style)
ax = axes[1, 1]
n_pos_bins = 10
n_phase_bins = 12
phase_position_matrix = np.zeros((n_phase_bins, n_pos_bins))

for i, prec in enumerate(precession_data):
    pos_indices = np.digitize(prec['pos_in_field'], np.linspace(0, 1, n_pos_bins + 1)) - 1
    phase_indices = np.digitize(prec['theta_phase_deg'], np.linspace(0, 360, n_phase_bins + 1)) - 1

    pos_indices = np.clip(pos_indices, 0, n_pos_bins - 1)
    phase_indices = np.clip(phase_indices, 0, n_phase_bins - 1)

    phase_position_matrix[phase_indices, pos_indices] += 1

im = ax.imshow(phase_position_matrix, aspect='auto', origin='lower', cmap='hot',
              extent=[0, 1, 0, 360], interpolation='nearest')
ax.set_xlabel('Position in Place Field (normalized)', fontsize=11)
ax.set_ylabel('Theta Phase (degrees)', fontsize=11)
ax.set_title('Phase-Position Distribution (All Neurons)', fontsize=12, fontweight='bold')
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Spike Count', fontsize=10)

plt.tight_layout()
plt.savefig('03_phase_precession.png', dpi=150, bbox_inches='tight')
print("Saved: 03_phase_precession.png")
plt.close()

# %% [markdown]
# ## 5. Quantify Phase Precession Strength

# %%
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Calculate phase precession slope for each neuron
precession_slopes = []
precession_r_squared = []
precession_intercepts = []

for prec in precession_data:
    if len(prec['pos_in_field']) < 5:
        continue

    # Convert phase to continuous scale (unwrap)
    phases = prec['theta_phase_deg'].copy()

    # Fit line: theta_phase = slope * position + intercept
    pos = prec['pos_in_field']

    # Create matrix for polyfit
    A = np.vstack([pos, np.ones(len(pos))]).T
    slope, intercept = np.linalg.lstsq(A, phases, rcond=None)[0]

    # Calculate R-squared
    predicted = slope * pos + intercept
    ss_res = np.sum((phases - predicted) ** 2)
    ss_tot = np.sum((phases - np.mean(phases)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    precession_slopes.append(slope)
    precession_intercepts.append(intercept)
    precession_r_squared.append(r_squared)

precession_slopes = np.array(precession_slopes)
precession_r_squared = np.array(precession_r_squared)
precession_intercepts = np.array(precession_intercepts)

print(f"\nPhase Precession Statistics:")
print(f"  Mean slope: {np.mean(precession_slopes):.1f} ± {np.std(precession_slopes):.1f} °/position")
print(f"  Mean R²: {np.mean(precession_r_squared):.3f} ± {np.std(precession_r_squared):.3f}")
print(f"  Expected precession range: {np.mean(precession_slopes) * 0 :.1f}° (entry) to {np.mean(precession_slopes) * 1:.1f}° (exit)")

# Plot 1: Distribution of precession slopes
ax = axes[0, 0]
ax.hist(precession_slopes, bins=15, color='steelblue', alpha=0.7, edgecolor='black', linewidth=1)
ax.axvline(np.mean(precession_slopes), color='red', linestyle='--', linewidth=2,
          label=f'Mean = {np.mean(precession_slopes):.1f}°')
ax.set_xlabel('Phase Precession Slope (°/position)', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('Distribution of Phase Precession Slopes', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

# Plot 2: Scatter of slope vs R²
ax = axes[0, 1]
scatter = ax.scatter(precession_slopes, precession_r_squared, c=precession_intercepts,
                    cmap='RdYlBu_r', s=80, alpha=0.6, edgecolors='black', linewidth=0.5)
ax.set_xlabel('Phase Precession Slope (°/position)', fontsize=11)
ax.set_ylabel('Fit Quality (R²)', fontsize=11)
ax.set_title('Precession Strength vs Fit Quality', fontsize=12, fontweight='bold')
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Intercept (°)', fontsize=10)
ax.grid(True, alpha=0.3)

# Plot 3: Phase precession magnitude histogram
# Magnitude = slope (360° for perfect precession)
precession_magnitude = np.abs(precession_slopes)
ax = axes[1, 0]
ax.hist(precession_magnitude, bins=15, color='seagreen', alpha=0.7, edgecolor='black', linewidth=1)
ax.axvline(np.mean(precession_magnitude), color='red', linestyle='--', linewidth=2,
          label=f'Mean = {np.mean(precession_magnitude):.1f}°')
ax.axvline(360, color='orange', linestyle=':', linewidth=2, label='Full cycle (360°)')
ax.set_xlabel('Phase Precession Magnitude (degrees)', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('Phase Precession Magnitude Distribution', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

# Plot 4: Number of neurons with significant precession
ax = axes[1, 1]
r2_thresholds = np.linspace(0, 0.8, 9)
n_significant = []

for threshold in r2_thresholds:
    n_sig = np.sum(precession_r_squared >= threshold)
    n_significant.append(n_sig)

ax.bar(range(len(r2_thresholds)), n_significant, color='coral', alpha=0.7, edgecolor='black', linewidth=1)
ax.set_xticks(range(len(r2_thresholds)))
ax.set_xticklabels([f'{t:.2f}' for t in r2_thresholds], rotation=45)
ax.set_xlabel('R² Threshold', fontsize=11)
ax.set_ylabel('Number of Neurons', fontsize=11)
ax.set_title('Neurons with Significant Phase Precession', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('04_precession_quantification.png', dpi=150, bbox_inches='tight')
print("Saved: 04_precession_quantification.png")
plt.close()

# %% [markdown]
# ## 6. Relate Phase Precession to Theta Oscillations

# %%
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Theta frequency analysis
from scipy.signal import welch

# Compute power spectrum of LFP
freq, power = welch(lfp, lfp_sampling_rate, nperseg=2048)

ax = axes[0, 0]
ax.semilogy(freq, power, linewidth=1.5, color='black')
ax.axvline(theta_freq, color='red', linestyle='--', linewidth=2, label=f'Theta ({theta_freq} Hz)')
ax.fill_between(freq[(freq >= 5) & (freq <= 12)], 1e-6, 1e3, alpha=0.2, color='red', label='Theta band')
ax.set_xlabel('Frequency (Hz)', fontsize=11)
ax.set_ylabel('Power (μV²/Hz)', fontsize=11)
ax.set_title('LFP Power Spectrum', fontsize=12, fontweight='bold')
ax.set_xlim(0, 30)
ax.set_ylim(1e-4, 1e3)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Plot 2: Theta phase over time (brief window)
ax = axes[0, 1]
window_start, window_end = 60, 75
mask = (time >= window_start) & (time <= window_end)
time_window = time[mask]
phase_window = theta_phase[mask] / (2 * np.pi) * 360

ax.plot(time_window, phase_window, 'b-', linewidth=2)
ax.fill_between(time_window, 0, 360, alpha=0.1)
ax.set_ylabel('Theta Phase (degrees)', fontsize=11)
ax.set_xlabel('Time (s)', fontsize=11)
ax.set_title(f'Theta Phase Evolution (window: {window_start}-{window_end}s)', fontsize=12, fontweight='bold')
ax.set_ylim(-20, 380)
ax.grid(True, alpha=0.3)

# Plot 3: Phase precession cycle diagram
ax = axes[1, 0]
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect('equal')

# Draw theta oscillation cycle
circle = Circle((0, 0), 1, fill=False, edgecolor='black', linewidth=2)
ax.add_patch(circle)

# Mark key phases
phases_to_mark = [0, 90, 180, 270]
labels = ['Peak\n(0°)', 'Trough\n(180°)', 'Peak\n(360°)', 'Trough\n(180°)']
for phase, label in zip([0, 180, 180, 0], labels):
    phase_rad = phase * np.pi / 180
    x = 1.1 * np.cos(phase_rad)
    y = 1.1 * np.sin(phase_rad)
    ax.plot([0, x], [0, y], 'k-', linewidth=1, alpha=0.3)

# Show precession trajectory (entry to exit of place field)
precession_positions = np.linspace(0, 1, 20)
precession_phases = (precession_positions * np.mean(precession_slopes)) % 360
precession_phases_rad = precession_phases * np.pi / 180

x_precession = 0.7 * np.cos(precession_phases_rad)
y_precession = 0.7 * np.sin(precession_phases_rad)

ax.plot(x_precession, y_precession, 'r-', linewidth=3, alpha=0.7, label='Entry → Exit')
ax.scatter(x_precession[0], y_precession[0], s=200, marker='o', c='green', edgecolor='black',
          linewidth=2, label='Place field entry', zorder=5)
ax.scatter(x_precession[-1], y_precession[-1], s=200, marker='s', c='red', edgecolor='black',
          linewidth=2, label='Place field exit', zorder=5)

ax.set_xlabel('Real component', fontsize=11)
ax.set_ylabel('Imaginary component', fontsize=11)
ax.set_title('Phase Precession in Theta Cycle', fontsize=12, fontweight='bold')
ax.legend(fontsize=9, loc='upper left')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linewidth=0.5)
ax.axvline(x=0, color='k', linewidth=0.5)

# Plot 4: Spike timing relative to theta cycles
ax = axes[1, 1]
# Select one neuron with clear precession
best_neuron_idx = np.argmax(precession_r_squared)
best_prec = precession_data[best_neuron_idx]

# For multiple place field traversals, show how spike timing shifts
traversals = []
current_pos = 0
traversal_start = 0

for i in range(1, len(position)):
    if position[i] < position[i-1]:  # Detected a turn around (position reset)
        if i - traversal_start > 100:  # Only consider traversals > 100 samples
            traversals.append((traversal_start, i))
        traversal_start = i

# Plot spikes in place field for first few traversals
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_aspect('equal')

circle = Circle((0, 0), 1, fill=False, edgecolor='gray', linewidth=1, linestyle='--')
ax.add_patch(circle)

colors_traversal = plt.cm.Spectral(np.linspace(0, 1, min(3, len(traversals))))

for traversal_idx, (start_idx, end_idx) in enumerate(traversals[:3]):
    traversal_time_start = time[start_idx]
    traversal_time_end = time[end_idx]

    # Find spikes in this traversal that are in place field
    spike_mask = (best_prec['pos_in_field'] > 0) & (best_prec['pos_in_field'] < 1)
    spike_times_in_field = best_prec['spike_times'][spike_mask]
    spike_phases_in_field = best_prec['theta_phase_deg'][spike_mask]

    if len(spike_times_in_field) > 0:
        phases_rad = spike_phases_in_field * np.pi / 180
        x = 0.8 * np.cos(phases_rad)
        y = 0.8 * np.sin(phases_rad)

        ax.scatter(x, y, s=50, alpha=0.6, color=colors_traversal[traversal_idx],
                  edgecolors='black', linewidth=0.5, label=f'Traversal {traversal_idx + 1}')

ax.set_xlabel('Real component', fontsize=11)
ax.set_ylabel('Imaginary component', fontsize=11)
ax.set_title(f'Spike Distribution on Theta Cycle\n(Neuron {best_neuron_idx})', fontsize=12, fontweight='bold')
ax.legend(fontsize=9, loc='upper left')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linewidth=0.5)
ax.axvline(x=0, color='k', linewidth=0.5)

plt.tight_layout()
plt.savefig('05_theta_analysis.png', dpi=150, bbox_inches='tight')
print("Saved: 05_theta_analysis.png")
plt.close()

# %% [markdown]
# ## 7. Summary Statistics and Biological Interpretation

# %%
print("\n" + "="*70)
print("THETA PHASE PRECESSION ANALYSIS SUMMARY")
print("="*70)

print(f"\nDataset Characteristics:")
print(f"  Recording duration: {time[-1]:.1f} seconds")
print(f"  Number of place cells analyzed: {len(precession_data)}")
print(f"  Theta frequency: {theta_freq} Hz (period = {1/theta_freq*1000:.1f} ms)")

print(f"\nPlace Field Properties:")
print(f"  Mean place field center: {np.mean([pf['center'] for pf in place_fields]):.1f} ± {np.std([pf['center'] for pf in place_fields]):.1f} cm")
print(f"  Mean place field width: {np.mean([pf['width'] for pf in place_fields]):.1f} ± {np.std([pf['width'] for pf in place_fields]):.1f} cm")
print(f"  Mean peak firing rate: {np.mean([pf['peak_rate'] for pf in place_fields]):.1f} ± {np.std([pf['peak_rate'] for pf in place_fields]):.1f} Hz")

print(f"\nTheta Phase Precession:")
print(f"  Mean precession slope: {np.mean(precession_slopes):.1f} ± {np.std(precession_slopes):.1f} degrees per position unit")
print(f"  Precession magnitude range: {np.min(precession_magnitude):.1f}° to {np.max(precession_magnitude):.1f}°")
print(f"  Mean fit quality (R²): {np.mean(precession_r_squared):.3f} ± {np.std(precession_r_squared):.3f}")

# Count neurons with significant precession (R² > 0.3)
n_significant = np.sum(precession_r_squared > 0.3)
print(f"  Neurons with R² > 0.3: {n_significant}/{len(precession_data)} ({100*n_significant/len(precession_data):.1f}%)")

print(f"\nBiological Interpretation:")
print(f"  Phase range across place field: {np.mean(precession_slopes):.0f}°")
print(f"  This represents {np.mean(precession_slopes)/360:.2f} theta cycles")
print(f"  Spike timing range within theta cycle: {np.mean(precession_slopes):.0f}° {theta_freq} Hz)")
print(f"  Time compression: ~{1/theta_freq*1000 * np.mean(precession_slopes)/360:.1f} ms across place field")

print("\n" + "="*70)

# %% [markdown]
# ## Key Findings
#
# This analysis demonstrates several key aspects of theta phase precession:
#
# ### 1. **Systematic Phase Shift**
# As animals traverse through a place cell's receptive field, the cell fires at
# progressively earlier phases of the theta oscillation. This creates a strong
# correlation between spatial position and spike timing relative to theta.
#
# ### 2. **Temporal Compression**
# Phase precession compresses the spatial path traversed during an animal's movement
# into a shorter temporal window (one theta cycle). This may facilitate sequence
# learning and memory consolidation.
#
# ### 3. **Population-Level Organization**
# The precession pattern is consistent across the neural population, with most place
# cells showing a similar slope. This suggests a fundamental organizational principle
# of hippocampal coding.
#
# ### 4. **Relationship to Theta Oscillation**
# The magnitude of phase precession is directly related to the theta frequency and
# the traversal speed through the place field. Faster movement through the field
# can result in greater phase shifts within a single theta cycle.
#
# ### 5. **Predictive Coding**
# By firing at progressively earlier phases as the animal approaches new locations,
# phase precession enables predictive neural coding - neurons fire for locations
# the animal will reach in the near future.
#
# ### References
# - O'Keefe & Recce (1993). Phase relationship between hippocampal place units and the EEG theta rhythm
# - Skaggs et al. (1996). Theta phase precession in some neurons recorded in the rat hippocampus
# - Jensen & Lisman (2000). Position reconstruction from an ensemble of place cells
#
# ---
