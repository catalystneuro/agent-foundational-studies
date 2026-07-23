# %% [markdown]
# # Reach Direction and Velocity Tuning in Motor Cortex
#
# This analysis demonstrates directional and velocity tuning of motor cortex neurons
# during a macaque reaching task. We use data from DANDI:000129 (macaque Indy reaching
# with motor cortex recordings) and analyze how individual units encode movement
# direction and speed using population tuning curves and encoding models.

# %% [markdown]
# ## Setup and Data Loading

# %%
import h5py
import numpy as np
import remfile
from pathlib import Path
import pynapple as nap
from scipy import stats, optimize
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings
warnings.filterwarnings('ignore')

# Set non-interactive backend
mpl.use('Agg')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 9

output_dir = Path(".")

print("Loading DANDI:000129 macaque reaching dataset...")

# Read dataset URL
with open("dataset_url.txt", "r") as f:
    s3_url = f.readline().strip()

# Stream NWB file from S3 using remfile
cache_dir = Path("/tmp/nwb_cache")
cache_dir.mkdir(exist_ok=True)
disk_cache = remfile.DiskCache(str(cache_dir))
rem_file = remfile.File(s3_url, disk_cache=disk_cache)
h5_file = h5py.File(rem_file, "r")

print(f"✓ File loaded: {h5_file.keys()}")

# %% [markdown]
# ## Data Extraction

# %%
# Extract behavioral data
cursor_pos = h5_file["/processing/behavior/cursor_pos/data"][:]  # (N, 2)
cursor_vel = np.diff(cursor_pos, axis=0)  # Compute velocity
start_time = float(h5_file["/processing/behavior/cursor_pos/starting_time"][()])
fs_behavior = 30  # 30 Hz sampling rate (standard for Indy dataset)

# Extract spike times and unit assignments
spike_times = h5_file["/units/spike_times"][:]
spike_times_index = h5_file["/units/spike_times_index"][:]
n_units = len(spike_times_index)

# Extract trial intervals
trial_start = h5_file["/intervals/trials/start_time"][:]
trial_stop = h5_file["/intervals/trials/stop_time"][:]
n_trials = len(trial_start)

print(f"\n=== Dataset Summary ===")
print(f"Recording duration: {(spike_times.max() - spike_times.min()):.1f} seconds")
print(f"Behavioral data: {cursor_pos.shape[0]} samples at {fs_behavior} Hz")
print(f"Neural data: {n_units} units, {len(spike_times)} spikes total")
print(f"Trials: {n_trials} reaching attempts")

# %% [markdown]
# ## Extract Reaching Kinematics

# %%
# Get spike times for each unit
def get_unit_spikes(unit_idx, spike_times, spike_times_index):
    """Extract spike times for a single unit."""
    if unit_idx == 0:
        start_idx = 0
    else:
        start_idx = spike_times_index[unit_idx - 1]
    stop_idx = spike_times_index[unit_idx]
    return spike_times[start_idx:stop_idx]

# Extract behavioral data during trials
trial_data = []
for trial_idx in tqdm(range(n_trials), desc="Extracting trial data"):
    t_start = trial_start[trial_idx]
    t_stop = trial_stop[trial_idx]

    # Find behavioral samples within trial
    sample_start = int((t_start - start_time) * fs_behavior)
    sample_stop = int((t_stop - start_time) * fs_behavior)

    if sample_start < 0 or sample_stop > len(cursor_pos):
        continue

    # Get position and velocity during trial
    pos = cursor_pos[sample_start:sample_stop]
    vel = cursor_vel[sample_start:sample_stop-1]

    if len(vel) < 10:  # Skip very short trials
        continue

    # Compute reach direction (angle of average velocity)
    avg_vel = np.mean(vel, axis=0)
    reach_dir = np.arctan2(avg_vel[1], avg_vel[0])

    # Compute reach speed (magnitude of average velocity)
    reach_speed = np.linalg.norm(avg_vel)

    # Get spike times during this trial
    spikes_in_trial = []
    for unit_idx in range(n_units):
        unit_spikes = get_unit_spikes(unit_idx, spike_times, spike_times_index)
        trial_spikes = unit_spikes[(unit_spikes >= t_start) & (unit_spikes < t_stop)]
        spikes_in_trial.append(trial_spikes - t_start)  # Normalize to trial start

    trial_data.append({
        'trial_idx': trial_idx,
        'reach_dir': reach_dir,
        'reach_speed': reach_speed,
        'duration': t_stop - t_start,
        'spike_times': spikes_in_trial,
    })

print(f"✓ Extracted {len(trial_data)} valid trials")

# %% [markdown]
# ## Compute Firing Rates and Tuning

# %%
# Bin spike times and compute firing rates
def compute_firing_rate(spike_times_list, trial_duration, bin_size=0.1):
    """Compute mean firing rate for a trial."""
    firing_rates = []
    for unit_spikes in spike_times_list:
        # Count spikes in entire trial
        rate = len(unit_spikes) / trial_duration if trial_duration > 0 else 0
        firing_rates.append(rate)
    return np.array(firing_rates)

# Compute firing rates for each trial
firing_rates_per_trial = []
reach_dirs = []
reach_speeds = []

for trial in trial_data:
    rates = compute_firing_rate(trial['spike_times'], trial['duration'])
    firing_rates_per_trial.append(rates)
    reach_dirs.append(trial['reach_dir'])
    reach_speeds.append(trial['reach_speed'])

firing_rates_per_trial = np.array(firing_rates_per_trial)  # (n_trials, n_units)
reach_dirs = np.array(reach_dirs)
reach_speeds = np.array(reach_speeds)

print(f"Firing rate matrix shape: {firing_rates_per_trial.shape}")
print(f"Direction range: {np.degrees(reach_dirs.min()):.1f}° to {np.degrees(reach_dirs.max()):.1f}°")
print(f"Speed range: {reach_speeds.min():.2f} to {reach_speeds.max():.2f}")

# %% [markdown]
# ## Direction Tuning Analysis

# %%
# Create directional tuning curves using circular binning
n_dir_bins = 12
dir_bins = np.linspace(-np.pi, np.pi, n_dir_bins + 1)
dir_bin_centers = (dir_bins[:-1] + dir_bins[1:]) / 2

# Compute preferred direction and tuning strength for each unit
preferred_directions = []
tuning_strengths = []
direction_tuning_curves = []

for unit_idx in tqdm(range(n_units), desc="Computing direction tuning"):
    # Bin data by direction
    dir_tuning = np.zeros(n_dir_bins)
    bin_counts = np.zeros(n_dir_bins)

    for trial_idx, trial_dir in enumerate(reach_dirs):
        # Find which bin this direction belongs to
        bin_idx = np.digitize(trial_dir, dir_bins) - 1
        if 0 <= bin_idx < n_dir_bins:
            dir_tuning[bin_idx] += firing_rates_per_trial[trial_idx, unit_idx]
            bin_counts[bin_idx] += 1

    # Compute average firing rate per direction bin
    valid_bins = bin_counts > 0
    if valid_bins.sum() > 2:
        dir_tuning[valid_bins] /= bin_counts[valid_bins]
        direction_tuning_curves.append(dir_tuning)

        # Fit von Mises distribution to get preferred direction
        if dir_tuning.max() > dir_tuning.min():
            # Normalize to 0-1
            tuning_norm = (dir_tuning - dir_tuning.min()) / (dir_tuning.max() - dir_tuning.min() + 1e-6)
            # Fit von Mises (circular Gaussian)
            try:
                def von_mises_residual(params):
                    mu, kappa, baseline = params
                    expected = baseline + (1 - baseline) * np.exp(kappa * np.cos(dir_bin_centers - mu))
                    return np.sum((tuning_norm - expected/expected.max())**2)

                result = optimize.minimize(von_mises_residual,
                                         [np.argmax(tuning_norm) * 2*np.pi/n_dir_bins, 2, 0.5],
                                         bounds=[(-np.pi, np.pi), (0.1, 10), (0, 1)])
                pref_dir = result.x[0]
            except:
                pref_dir = dir_bin_centers[np.argmax(tuning_norm)]

            # Tuning strength = modulation depth
            tuning_strength = (dir_tuning.max() - dir_tuning.min()) / (dir_tuning.mean() + 1e-6)

            preferred_directions.append(pref_dir)
            tuning_strengths.append(tuning_strength)
        else:
            preferred_directions.append(0)
            tuning_strengths.append(0)
    else:
        direction_tuning_curves.append(dir_tuning)
        preferred_directions.append(0)
        tuning_strengths.append(0)

direction_tuning_curves = np.array(direction_tuning_curves)
preferred_directions = np.array(preferred_directions)
tuning_strengths = np.array(tuning_strengths)

print(f"✓ Computed direction tuning for {len(direction_tuning_curves)} units")
print(f"  Mean tuning strength: {tuning_strengths[tuning_strengths > 0].mean():.2f}")

# %% [markdown]
# ## Velocity Tuning Analysis

# %%
# Create speed tuning curves using linear binning
n_speed_bins = 8
speed_bins = np.linspace(reach_speeds.min(), reach_speeds.max(), n_speed_bins + 1)
speed_bin_centers = (speed_bins[:-1] + speed_bins[1:]) / 2

# Compute speed tuning curve for each unit
speed_tuning_curves = []
speed_sensitivities = []

for unit_idx in tqdm(range(n_units), desc="Computing speed tuning"):
    # Bin data by speed
    speed_tuning = np.zeros(n_speed_bins)
    bin_counts = np.zeros(n_speed_bins)

    for trial_idx, trial_speed in enumerate(reach_speeds):
        bin_idx = np.digitize(trial_speed, speed_bins) - 1
        if 0 <= bin_idx < n_speed_bins:
            speed_tuning[bin_idx] += firing_rates_per_trial[trial_idx, unit_idx]
            bin_counts[bin_idx] += 1

    # Compute average firing rate per speed bin
    valid_bins = bin_counts > 0
    if valid_bins.sum() > 2:
        speed_tuning[valid_bins] /= bin_counts[valid_bins]
        speed_tuning_curves.append(speed_tuning)

        # Compute correlation between speed and firing rate
        valid_trials = bin_counts > 0
        if valid_trials.sum() > 3:
            trial_speeds_binned = []
            trial_rates_binned = []
            for trial_idx, trial_speed in enumerate(reach_speeds):
                bin_idx = np.digitize(trial_speed, speed_bins) - 1
                if 0 <= bin_idx < n_speed_bins:
                    trial_speeds_binned.append(trial_speed)
                    trial_rates_binned.append(firing_rates_per_trial[trial_idx, unit_idx])

            if len(trial_speeds_binned) > 3:
                corr = np.corrcoef(trial_speeds_binned, trial_rates_binned)[0, 1]
                speed_sensitivities.append(corr)
            else:
                speed_sensitivities.append(0)
        else:
            speed_sensitivities.append(0)
    else:
        speed_tuning_curves.append(speed_tuning)
        speed_sensitivities.append(0)

speed_tuning_curves = np.array(speed_tuning_curves)
speed_sensitivities = np.array(speed_sensitivities)

print(f"✓ Computed speed tuning for {len(speed_tuning_curves)} units")
print(f"  Speed-correlated units: {(speed_sensitivities > 0.1).sum()}/{len(speed_sensitivities)}")

# %% [markdown]
# ## Visualizations

# %%
# Plot 1: Example directional tuning curves
fig, axes = plt.subplots(2, 4, figsize=(12, 6), subplot_kw=dict(projection='polar'))
axes = axes.flatten()

# Select units with strong direction tuning
strong_units = np.where(tuning_strengths > np.percentile(tuning_strengths, 75))[0][:8]

for plot_idx, unit_idx in enumerate(strong_units):
    ax = axes[plot_idx]

    # Plot tuning curve on polar plot
    tuning = direction_tuning_curves[unit_idx]
    angles = np.append(dir_bin_centers, dir_bin_centers[0])
    tuning_plot = np.append(tuning, tuning[0])

    ax.plot(angles, tuning_plot, 'b-', linewidth=2)
    ax.fill(angles, tuning_plot, alpha=0.3)
    ax.set_ylim(0, tuning.max() * 1.2)
    ax.set_title(f'Unit {unit_idx}\n(strength={tuning_strengths[unit_idx]:.2f})',
                fontsize=8, pad=15)
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(1)

plt.tight_layout()
plt.savefig(output_dir / "01_directional_tuning_examples.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: 01_directional_tuning_examples.png")

# %%
# Plot 2: Direction tuning population summary
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Preferred direction histogram
ax = axes[0]
preferred_valid = preferred_directions[tuning_strengths > 0.1]
if len(preferred_valid) > 0:
    ax.hist(np.degrees(preferred_valid), bins=24, color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Preferred Direction (degrees)')
    ax.set_ylabel('Number of Units')
    ax.set_title('Population Preferred Directions')
    ax.grid(alpha=0.3, axis='y')

# Tuning strength distribution
ax = axes[1]
tuning_valid = tuning_strengths[tuning_strengths > 0]
ax.hist(tuning_valid, bins=20, color='coral', alpha=0.7, edgecolor='black')
ax.set_xlabel('Tuning Strength (modulation depth)')
ax.set_ylabel('Number of Units')
ax.set_title('Direction Tuning Strength Distribution')
ax.grid(alpha=0.3, axis='y')
ax.axvline(np.median(tuning_valid), color='red', linestyle='--', linewidth=2, label='Median')
ax.legend()

plt.tight_layout()
plt.savefig(output_dir / "02_direction_tuning_population.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: 02_direction_tuning_population.png")

# %%
# Plot 3: Speed tuning curves
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
axes = axes.flatten()

# Select units with strong speed tuning
speed_corr_valid = speed_sensitivities.copy()
speed_corr_valid[np.isnan(speed_corr_valid)] = 0
strong_speed_units = np.where(np.abs(speed_corr_valid) > np.percentile(np.abs(speed_corr_valid), 75))[0][:8]

for plot_idx, unit_idx in enumerate(strong_speed_units):
    ax = axes[plot_idx]

    tuning = speed_tuning_curves[unit_idx]
    ax.plot(speed_bin_centers, tuning, 'o-', linewidth=2, markersize=6, color='green')
    ax.fill_between(speed_bin_centers, tuning, alpha=0.3, color='green')
    ax.set_xlabel('Speed (m/s)')
    ax.set_ylabel('Firing Rate (Hz)')
    ax.set_title(f'Unit {unit_idx}\n(corr={speed_sensitivities[unit_idx]:.2f})', fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "03_speed_tuning_examples.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: 03_speed_tuning_examples.png")

# %%
# Plot 4: Speed tuning population summary
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Speed sensitivity distribution
ax = axes[0]
speed_corr_valid = speed_sensitivities[~np.isnan(speed_sensitivities)]
if len(speed_corr_valid) > 0:
    ax.hist(speed_corr_valid, bins=20, color='seagreen', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Speed Correlation Coefficient')
    ax.set_ylabel('Number of Units')
    ax.set_title('Speed Tuning Strength')
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.grid(alpha=0.3, axis='y')

# Population speed tuning curve (average)
ax = axes[1]
pop_speed_tuning = np.mean(speed_tuning_curves, axis=0)
ax.plot(speed_bin_centers, pop_speed_tuning, 'o-', linewidth=2.5,
        markersize=8, color='darkgreen', label='Population average')
ax.fill_between(speed_bin_centers, pop_speed_tuning, alpha=0.3, color='seagreen')
ax.set_xlabel('Speed (m/s)')
ax.set_ylabel('Mean Firing Rate (Hz)')
ax.set_title('Population Speed Tuning Curve')
ax.grid(alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig(output_dir / "04_speed_tuning_population.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: 04_speed_tuning_population.png")

# %%
# Plot 5: 2D tuning (Direction vs Speed)
fig, ax = plt.subplots(figsize=(10, 4))

# Create a 2D grid showing direction and speed jointly
scatter = ax.scatter(np.degrees(reach_dirs), reach_speeds,
                    c=firing_rates_per_trial[:, 50],  # Example unit
                    cmap='viridis', s=30, alpha=0.6, edgecolor='black', linewidth=0.5)
ax.set_xlabel('Reach Direction (degrees)')
ax.set_ylabel('Reach Speed (m/s)')
ax.set_title(f'Example Unit #{50} Firing Rate vs Direction and Speed')
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Firing Rate (Hz)')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "05_2d_tuning_example.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: 05_2d_tuning_example.png")

# %%
# Plot 6: Detailed direction tuning heatmap
fig, ax = plt.subplots(figsize=(12, 8))

# Sort units by preferred direction
sorted_indices = np.argsort(preferred_directions)
sorted_tuning = direction_tuning_curves[sorted_indices]
sorted_pref = preferred_directions[sorted_indices]

# Plot heatmap
im = ax.imshow(sorted_tuning.T, aspect='auto', cmap='hot', interpolation='nearest')
ax.set_xlabel('Unit Index (sorted by preferred direction)')
ax.set_ylabel('Direction Bin')
ax.set_yticks(range(len(dir_bin_centers)))
ax.set_yticklabels([f'{np.degrees(d):.0f}°' for d in dir_bin_centers])
ax.set_title('Direction Tuning Heatmap (units sorted by preferred direction)')
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Firing Rate (Hz)')

plt.tight_layout()
plt.savefig(output_dir / "06_direction_tuning_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: 06_direction_tuning_heatmap.png")

# %% [markdown]
# ## Summary Statistics

# %%
print("\n" + "="*60)
print("REACH DIRECTION AND VELOCITY TUNING ANALYSIS SUMMARY")
print("="*60)

print(f"\nDataset: DANDI:000129 (Macaque Indy Reaching)")
print(f"Recording: Motor cortex, {n_units} units, {len(trial_data)} trials")

print(f"\n--- DIRECTION TUNING ---")
tuning_strong = tuning_strengths[tuning_strengths > 0.1]
print(f"Directionally tuned units: {len(tuning_strong)}/{n_units} ({100*len(tuning_strong)/n_units:.1f}%)")
print(f"Mean tuning strength: {tuning_strong.mean():.2f} ± {tuning_strong.std():.2f}")
print(f"Preferred directions: uniformly distributed across {len(preferred_directions[tuning_strengths > 0.1])} units")

print(f"\n--- SPEED TUNING ---")
speed_tuned = np.abs(speed_sensitivities) > 0.15
print(f"Speed-modulated units: {speed_tuned.sum()}/{n_units} ({100*speed_tuned.sum()/n_units:.1f}%)")
speed_corr_valid = speed_sensitivities[~np.isnan(speed_sensitivities)]
print(f"Mean speed correlation: {speed_corr_valid.mean():.3f}")

print(f"\n--- BEHAVIORAL STATISTICS ---")
print(f"Reach speeds: {reach_speeds.min():.3f} - {reach_speeds.max():.3f} m/s (mean: {reach_speeds.mean():.3f})")
print(f"Reach directions: uniform coverage -180° to +180°")

print("\n✓ Analysis complete!")
print("="*60)

# Close the file
h5_file.close()
