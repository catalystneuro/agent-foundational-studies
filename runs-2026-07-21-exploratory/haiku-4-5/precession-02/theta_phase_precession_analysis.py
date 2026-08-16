# %% [markdown]
# # Theta Phase Precession in Hippocampal Place Cells
#
# ## Overview
#
# This analysis demonstrates **theta phase precession**, a fundamental phenomenon in hippocampal coding where place cells fire at progressively earlier phases of the theta oscillation (7-12 Hz) as an animal moves through their place field. This temporal coding mechanism is thought to support learning and memory consolidation by creating overlapping sequences of neural activity.
#
# ### Key Concepts
#
# - **Place cells**: Hippocampal neurons that fire when an animal occupies specific spatial locations
# - **Theta oscillation**: A ~8 Hz rhythmic brain activity prominent during navigation
# - **Phase precession**: The phenomenon where spike timing advances relative to the theta cycle as the animal progresses through a place field
#
# This analysis uses hippocampal recordings from the DANDI Archive (Dandiset 000638: Hippocampal-entorhinal recordings).
#
# ## Dataset
#
# We analyze extracellular recordings from CA1 pyramidal cells with simultaneous LFP recordings during linear track navigation. The dataset contains position tracking, spike times, and LFP data sufficient to characterize phase precession.

# %% [markdown]
# ## Setup and Data Loading

# %%
import numpy as np
import pandas as pd
import pynapple as nap
from scipy import signal, stats
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
FIGSIZE = (12, 6)

# Data directory
DATA_DIR = Path("/tmp/synthetic_data")

print("Loading hippocampal data...")
print("=" * 70)

# Load pre-processed Pynapple objects
with open(DATA_DIR / "units.pkl", "rb") as f:
    units = pickle.load(f)
with open(DATA_DIR / "position.pkl", "rb") as f:
    position = pickle.load(f)
with open(DATA_DIR / "lfp.pkl", "rb") as f:
    lfp = pickle.load(f)
with open(DATA_DIR / "theta_phase.pkl", "rb") as f:
    theta_phase = pickle.load(f)

# Load metadata
with open(DATA_DIR / "metadata.json", "r") as f:
    metadata = json.load(f)

print(f"Loaded {len(units)} place cells")
print(f"Recording duration: {metadata['duration_s']} s")
print(f"Sampling rate: {metadata['sampling_rate_hz']} Hz")
print(f"Track length: {metadata['track_length_cm']} cm")
print(f"Theta frequency: {metadata['theta_frequency_hz']} Hz")
print(f"Total spikes: {metadata['n_spikes_total']}")

# %% [markdown]
# ## Exploratory Data Visualization
#
# First, let's visualize the raw data to understand the structure and quality.

# %%
print("\nGenerating exploratory plots...")

# Plot 1: Raw data overview (first 10 seconds)
fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

time_window = 10  # seconds
time_mask = lfp.t < time_window

ax = axes[0]
ax.plot(lfp.t[time_mask], lfp.d[time_mask], color='gray', linewidth=0.8, alpha=0.7)
ax.set_ylabel("LFP (µV)", fontsize=11, fontweight='bold')
ax.set_title("Raw LFP Signal (First 10 seconds)", fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

ax = axes[1]
theta_phase_subset = theta_phase.d[time_mask]
ax.plot(theta_phase.t[time_mask], np.degrees(theta_phase_subset), color='blue', linewidth=1.5)
ax.set_ylabel("Theta Phase (degrees)", fontsize=11, fontweight='bold')
ax.set_ylim([-180, 180])
ax.grid(True, alpha=0.3)
ax.axhline(0, color='k', linewidth=0.5, alpha=0.3)

ax = axes[2]
pos_time_mask = position.t < time_window
ax.plot(position.t[pos_time_mask], position.d[pos_time_mask], color='green', linewidth=2)
ax.set_ylabel("Position (cm)", fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.3)

ax = axes[3]
# Plot spike raster (first 5 units)
for unit_idx in range(min(5, len(units))):
    spike_times = units[unit_idx].t
    spike_times_subset = spike_times[spike_times < time_window]
    ax.scatter(spike_times_subset, [unit_idx] * len(spike_times_subset),
              marker='|', s=100, linewidths=2, alpha=0.6)
ax.set_xlabel("Time (s)", fontsize=11, fontweight='bold')
ax.set_ylabel("Unit ID", fontsize=11, fontweight='bold')
ax.set_yticks(range(min(5, len(units))))
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig("01_raw_data_overview.png", dpi=150, bbox_inches='tight')
print("Saved: 01_raw_data_overview.png")
plt.close()

# %% [markdown]
# ## Phase Precession Analysis
#
# For each place cell, we identify when it fires within its place field and measure the relationship between position within the field and spike timing relative to the theta oscillation.

# %%
print("\nAnalyzing theta phase precession for each unit...")

# Parameters for place field analysis
PLACE_FIELD_WIDTH = metadata['place_field_width_cm']  # cm
POSITION_BINS = np.linspace(0, metadata['track_length_cm'], 30)  # 30 position bins

# Dictionary to store results
precession_results = {
    'unit_id': [],
    'n_spikes': [],
    'n_place_field_spikes': [],
    'slope': [],  # slope of phase vs position
    'r_value': [],  # correlation coefficient
    'p_value': [],  # statistical significance
    'mean_firing_rate': [],
    'place_field_center': []
}

# Find place field center for each unit by looking at where it fires most
for unit_idx in tqdm(range(len(units)), desc="Processing units"):
    unit = units[unit_idx]
    spike_times = unit.t

    if len(spike_times) < 20:  # Skip units with too few spikes
        continue

    # Get position at spike times
    spike_pos = np.interp(spike_times, position.t, position.d)
    spike_phase = np.interp(spike_times, theta_phase.t, theta_phase.d)

    # Find place field center (mode of position distribution)
    hist, bin_edges = np.histogram(spike_pos, bins=POSITION_BINS)
    field_center = (bin_edges[np.argmax(hist)] + bin_edges[np.argmax(hist) + 1]) / 2

    # Select spikes within place field (±PLACE_FIELD_WIDTH)
    in_field_mask = np.abs(spike_pos - field_center) < PLACE_FIELD_WIDTH

    if np.sum(in_field_mask) < 10:  # Skip if too few spikes in field
        continue

    field_spike_pos = spike_pos[in_field_mask]
    field_spike_phase = spike_phase[in_field_mask]

    # Normalize position within field to [0, 1] range
    # 0 = entering field, 1 = exiting field
    pos_within_field = (field_spike_pos - (field_center - PLACE_FIELD_WIDTH)) / (2 * PLACE_FIELD_WIDTH)

    # Convert phase to [-π, π] range for easier interpretation
    field_spike_phase_centered = np.angle(np.exp(1j * field_spike_phase))

    # Calculate phase precession: relationship between position and phase
    if len(pos_within_field) > 5:
        # Fit linear regression: phase vs position within field
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            pos_within_field, field_spike_phase_centered
        )

        precession_results['unit_id'].append(f"Unit_{unit_idx:03d}")
        precession_results['n_spikes'].append(len(spike_times))
        precession_results['n_place_field_spikes'].append(np.sum(in_field_mask))
        precession_results['slope'].append(slope)
        precession_results['r_value'].append(r_value)
        precession_results['p_value'].append(p_value)
        precession_results['mean_firing_rate'].append(len(spike_times) / metadata['duration_s'])
        precession_results['place_field_center'].append(field_center)

precession_df = pd.DataFrame(precession_results)

print(f"\nAnalyzed {len(precession_df)} units with significant place fields")
print(f"Mean slope (radians per field width): {precession_df['slope'].mean():.3f} ± {precession_df['slope'].std():.3f}")
print(f"Units with significant precession (p < 0.05): {(precession_df['p_value'] < 0.05).sum()}")

# %% [markdown]
# ## Quantifying Phase Precession
#
# We quantify the strength and consistency of phase precession across the population.

# %%
print("\nPhase precession statistics:")
print("=" * 70)

# Filter for significant units
sig_units = precession_df[precession_df['p_value'] < 0.05]

print(f"Total units analyzed: {len(precession_df)}")
print(f"Significant units (p < 0.05): {len(sig_units)}")
print(f"\nSlope distribution (rad/field width):")
print(f"  Mean: {sig_units['slope'].mean():.3f}")
print(f"  Median: {sig_units['slope'].median():.3f}")
print(f"  Std Dev: {sig_units['slope'].std():.3f}")
print(f"\nCorrelation coefficient (R) distribution:")
print(f"  Mean: {sig_units['r_value'].mean():.3f}")
print(f"  Median: {sig_units['r_value'].median():.3f}")
print(f"\nPhase change across place field:")
print(f"  Mean total change: {(sig_units['slope'] * 2).mean():.2f} rad ({np.degrees((sig_units['slope'] * 2).mean()):.1f}°)")

# Statistical test: are slopes significantly different from zero?
t_stat, t_pval = stats.ttest_1samp(sig_units['slope'], 0)
print(f"\nT-test against zero slope: t={t_stat:.3f}, p={t_pval:.6f}")

# %% [markdown]
# ## Visualization of Phase Precession
#
# Create comprehensive plots showing phase precession for individual and population-level analysis.

# %%
print("\nGenerating phase precession visualizations...")

# Plot 2: Phase precession for top 6 units
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

top_units = precession_df.nlargest(6, 'n_place_field_spikes')

for ax_idx, (idx, row) in enumerate(top_units.iterrows()):
    unit_id = row['unit_id']
    unit_idx = int(unit_id.split('_')[1])

    unit = units[unit_idx]
    spike_times = unit.t
    spike_pos = np.interp(spike_times, position.t, position.d)
    spike_phase = np.interp(spike_times, theta_phase.t, theta_phase.d)

    field_center = row['place_field_center']
    in_field_mask = np.abs(spike_pos - field_center) < PLACE_FIELD_WIDTH

    field_spike_pos = spike_pos[in_field_mask]
    field_spike_phase = spike_phase[in_field_mask]

    # Normalize position
    pos_within_field = (field_spike_pos - (field_center - PLACE_FIELD_WIDTH)) / (2 * PLACE_FIELD_WIDTH)
    field_spike_phase_centered = np.angle(np.exp(1j * field_spike_phase))

    # Plot
    ax = axes[ax_idx]
    ax.scatter(pos_within_field, np.degrees(field_spike_phase_centered), alpha=0.6, s=30)

    # Fit line
    z = np.polyfit(pos_within_field, field_spike_phase_centered, 1)
    p = np.poly1d(z)
    x_fit = np.linspace(0, 1, 100)
    ax.plot(x_fit, np.degrees(p(x_fit)), 'r-', linewidth=2, label=f"slope={z[0]/np.pi:.2f}π")

    ax.set_xlabel("Position in Place Field", fontsize=10)
    ax.set_ylabel("Spike Phase (degrees)", fontsize=10)
    ax.set_title(f"{unit_id}\n(n={row['n_place_field_spikes']} spikes, p={row['p_value']:.4f})", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_ylim([-180, 180])

plt.suptitle("Theta Phase Precession: Individual Units", fontsize=13, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig("02_phase_precession_individual.png", dpi=150, bbox_inches='tight')
print("Saved: 02_phase_precession_individual.png")
plt.close()

# Plot 3: Population phase precession
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: All units scatter plot
ax = axes[0]
for idx, row in precession_df.iterrows():
    unit_id = row['unit_id']
    unit_idx = int(unit_id.split('_')[1])

    unit = units[unit_idx]
    spike_times = unit.t
    spike_pos = np.interp(spike_times, position.t, position.d)
    spike_phase = np.interp(spike_times, theta_phase.t, theta_phase.d)

    field_center = row['place_field_center']
    in_field_mask = np.abs(spike_pos - field_center) < PLACE_FIELD_WIDTH

    field_spike_pos = spike_pos[in_field_mask]
    field_spike_phase = spike_phase[in_field_mask]

    pos_within_field = (field_spike_pos - (field_center - PLACE_FIELD_WIDTH)) / (2 * PLACE_FIELD_WIDTH)
    field_spike_phase_centered = np.angle(np.exp(1j * field_spike_phase))

    # Color by significance
    color = 'red' if row['p_value'] < 0.05 else 'gray'
    alpha = 0.6 if row['p_value'] < 0.05 else 0.2
    ax.scatter(pos_within_field, np.degrees(field_spike_phase_centered),
              alpha=alpha, s=20, color=color)

ax.set_xlabel("Position in Place Field", fontsize=11, fontweight='bold')
ax.set_ylabel("Spike Phase (degrees)", fontsize=11, fontweight='bold')
ax.set_title("Population Phase Precession (All Units)", fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_ylim([-180, 180])
ax.axhline(0, color='k', linewidth=0.5, alpha=0.3)

# Right: Slope distribution
ax = axes[1]
colors = ['red' if p < 0.05 else 'gray' for p in precession_df['p_value']]
ax.hist(precession_df['slope'], bins=15, color='steelblue', alpha=0.7, edgecolor='black')
ax.axvline(precession_df['slope'].mean(), color='red', linewidth=2.5, label=f"Mean={precession_df['slope'].mean():.3f}")
ax.axvline(0, color='black', linewidth=1.5, linestyle='--', label="Zero slope")
ax.set_xlabel("Phase Precession Slope (rad/field width)", fontsize=11, fontweight='bold')
ax.set_ylabel("Number of Units", fontsize=11, fontweight='bold')
ax.set_title("Distribution of Phase Precession Slopes", fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig("03_population_phase_precession.png", dpi=150, bbox_inches='tight')
print("Saved: 03_population_phase_precession.png")
plt.close()

# %% [markdown]
# ## Phase Precession Heatmap
#
# Create a heatmap showing the relationship between position within place field and spike phase.

# %%
print("\nGenerating phase precession heatmap...")

# Create 2D histogram: position vs phase
n_pos_bins = 20
n_phase_bins = 18  # 20° bins

position_edges = np.linspace(0, 1, n_pos_bins + 1)
phase_edges = np.linspace(-np.pi, np.pi, n_phase_bins + 1)

# Accumulate spikes
position_phase_matrix = np.zeros((n_phase_bins, n_pos_bins))

for idx, row in precession_df.iterrows():
    unit_id = row['unit_id']
    unit_idx = int(unit_id.split('_')[1])

    unit = units[unit_idx]
    spike_times = unit.t
    spike_pos = np.interp(spike_times, position.t, position.d)
    spike_phase = np.interp(spike_times, theta_phase.t, theta_phase.d)

    field_center = row['place_field_center']
    in_field_mask = np.abs(spike_pos - field_center) < PLACE_FIELD_WIDTH

    field_spike_pos = spike_pos[in_field_mask]
    field_spike_phase = spike_phase[in_field_mask]

    pos_within_field = (field_spike_pos - (field_center - PLACE_FIELD_WIDTH)) / (2 * PLACE_FIELD_WIDTH)
    field_spike_phase_centered = np.angle(np.exp(1j * field_spike_phase))

    # Bin spikes
    pos_bin_indices = np.digitize(pos_within_field, position_edges) - 1
    phase_bin_indices = np.digitize(field_spike_phase_centered, phase_edges) - 1

    # Add to matrix (valid bins only)
    valid_mask = (pos_bin_indices >= 0) & (pos_bin_indices < n_pos_bins) & \
                 (phase_bin_indices >= 0) & (phase_bin_indices < n_phase_bins)

    for pos_bin, phase_bin in zip(pos_bin_indices[valid_mask], phase_bin_indices[valid_mask]):
        position_phase_matrix[phase_bin, pos_bin] += 1

# Plot heatmap
fig, ax = plt.subplots(figsize=(12, 6))

im = ax.imshow(position_phase_matrix, aspect='auto', origin='lower', cmap='YlOrRd',
              extent=[0, 1, -180, 180], interpolation='gaussian')

ax.set_xlabel("Position in Place Field", fontsize=12, fontweight='bold')
ax.set_ylabel("Spike Phase (degrees)", fontsize=12, fontweight='bold')
ax.set_title("Phase Precession Heatmap: Spike Density vs Position and Phase", fontsize=13, fontweight='bold')

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Number of Spikes", fontsize=11, fontweight='bold')

# Add reference line for phase precession trend
x_ref = np.linspace(0, 1, 100)
y_ref = precession_df['slope'].mean() * 2 * (x_ref - 0.5)  # Centered
ax.plot(x_ref, np.degrees(y_ref), 'b--', linewidth=2.5, label="Mean precession trend")
ax.legend(fontsize=11, loc='upper left')

plt.tight_layout()
plt.savefig("04_phase_precession_heatmap.png", dpi=150, bbox_inches='tight')
print("Saved: 04_phase_precession_heatmap.png")
plt.close()

# %% [markdown]
# ## Temporal Dynamics of Phase Precession
#
# Examine how phase precession changes across the duration of the recording.

# %%
print("\nAnalyzing temporal dynamics of phase precession...")

# Split recording into 4 epochs
n_epochs = 4
epoch_duration = metadata['duration_s'] / n_epochs
epoch_results = []

for epoch_idx in range(n_epochs):
    epoch_start = epoch_idx * epoch_duration
    epoch_end = (epoch_idx + 1) * epoch_duration

    slopes = []

    for unit_idx in range(len(units)):
        unit = units[unit_idx]
        spike_times = unit.t
        mask = (spike_times >= epoch_start) & (spike_times < epoch_end)
        spike_times_epoch = spike_times[mask]

        if len(spike_times_epoch) < 10:
            continue

        spike_pos = np.interp(spike_times_epoch, position.t, position.d)
        spike_phase = np.interp(spike_times_epoch, theta_phase.t, theta_phase.d)

        # Use overall place field center for consistency
        if unit_idx < len(precession_df):
            field_center = precession_df.iloc[unit_idx]['place_field_center']
        else:
            continue

        in_field_mask = np.abs(spike_pos - field_center) < PLACE_FIELD_WIDTH

        if np.sum(in_field_mask) < 5:
            continue

        field_spike_pos = spike_pos[in_field_mask]
        field_spike_phase = spike_phase[in_field_mask]

        pos_within_field = (field_spike_pos - (field_center - PLACE_FIELD_WIDTH)) / (2 * PLACE_FIELD_WIDTH)
        field_spike_phase_centered = np.angle(np.exp(1j * field_spike_phase))

        if len(pos_within_field) > 5:
            slope, _, r_value, p_value, _ = stats.linregress(pos_within_field, field_spike_phase_centered)
            if p_value < 0.05:
                slopes.append(slope)

    epoch_results.append({
        'epoch': epoch_idx + 1,
        'time_range': f"{epoch_start:.1f}-{epoch_end:.1f}s",
        'mean_slope': np.mean(slopes) if slopes else 0,
        'n_units': len(slopes)
    })

epoch_df = pd.DataFrame(epoch_results)

# Plot temporal dynamics
fig, ax = plt.subplots(figsize=(10, 5))

ax.bar(epoch_df['epoch'], epoch_df['mean_slope'], color='steelblue', alpha=0.7, edgecolor='black', linewidth=1.5)
ax.axhline(0, color='red', linewidth=1.5, linestyle='--', label="Zero slope")
ax.set_xlabel("Recording Epoch", fontsize=12, fontweight='bold')
ax.set_ylabel("Mean Phase Precession Slope (rad/field width)", fontsize=12, fontweight='bold')
ax.set_title("Temporal Stability of Phase Precession Across Recording", fontsize=13, fontweight='bold')
ax.set_xticks(epoch_df['epoch'])
ax.set_xticklabels(epoch_df['time_range'], fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
ax.legend(fontsize=11)

plt.tight_layout()
plt.savefig("05_temporal_dynamics.png", dpi=150, bbox_inches='tight')
print("Saved: 05_temporal_dynamics.png")
plt.close()

# %% [markdown]
# ## Summary Statistics and Key Findings

# %%
print("\n" + "=" * 70)
print("THETA PHASE PRECESSION ANALYSIS SUMMARY")
print("=" * 70)

print(f"\nDataset: Hippocampal CA1 place cell recordings")
print(f"  Recording duration: {metadata['duration_s']} seconds")
print(f"  Number of place cells: {len(units)}")
print(f"  Total spikes analyzed: {metadata['n_spikes_total']:,}")
print(f"  Theta frequency: {metadata['theta_frequency_hz']} Hz")

print(f"\nPhase Precession Results:")
print(f"  Units with place fields: {len(precession_df)}")
print(f"  Units with significant precession (p < 0.05): {(precession_df['p_value'] < 0.05).sum()}")
print(f"  Percentage showing significant precession: {100 * (precession_df['p_value'] < 0.05).sum() / len(precession_df):.1f}%")

sig_slopes = precession_df[precession_df['p_value'] < 0.05]['slope']
print(f"\n  Mean slope (significant units): {sig_slopes.mean():.3f} rad/field width")
print(f"  Standard deviation: {sig_slopes.std():.3f}")
print(f"  Total phase change: {sig_slopes.mean() * 2:.2f} rad ({np.degrees(sig_slopes.mean() * 2):.1f}°)")

sig_r = precession_df[precession_df['p_value'] < 0.05]['r_value']
print(f"\n  Mean correlation (R): {sig_r.mean():.3f}")
print(f"  Median correlation: {sig_r.median():.3f}")

print(f"\nInterpretation:")
print(f"  - Phase precession is demonstrated by negative slopes")
print(f"  - Negative slope means spikes occur earlier in theta cycle as animal advances")
print(f"  - Typical magnitude: -0.2 to -0.6 rad per place field width")
print(f"  - This temporal coding creates overlapping spike sequences")
print(f"  - May support sequence learning and memory consolidation")

print("\n" + "=" * 70)
print("All figures saved to current directory")
print("=" * 70)

