# %% [markdown]
# # Reach Direction and Velocity Tuning Analysis
#
# Demonstration of direction and velocity selectivity in motor cortex during reaching movements.
# This analysis demonstrates how motor neurons encode both the direction of movement and the
# speed of the reach, fundamental properties of motor cortex coding.

# %% [markdown]
# ## Setup and Data Generation

# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import pandas as pd
from scipy import stats
from scipy.signal import convolve
import warnings
warnings.filterwarnings('ignore')

# Set up plotting
plt.style.use('default')
np.random.seed(42)

# %% [markdown]
# ### Generate Synthetic Reach Data
# Motor cortex neurons typically show cosine tuning for reach direction and
# modulation for reach velocity. We simulate 30 neurons recorded during 500 reaching trials.

# %%
def generate_motor_cortex_data(n_trials=500, n_neurons=30, n_directions=8, n_velocities=4):
    """
    Generate synthetic motor cortex spike data with direction and velocity tuning.
    """
    reach_directions = np.linspace(0, 2*np.pi, n_directions, endpoint=False)
    reach_velocities = np.array([10, 20, 30, 40])  # cm/s

    # Randomly assign directions and velocities to trials
    trial_directions = np.random.choice(reach_directions, n_trials)
    trial_velocities = np.random.choice(reach_velocities, n_trials)

    # Generate spike counts with tuning properties
    spike_counts = np.zeros((n_trials, n_neurons))
    baseline_rates = np.random.uniform(5, 20, n_neurons)

    # Assign preferred tuning properties to each neuron
    pref_directions = np.linspace(0, 2*np.pi, n_neurons, endpoint=False)
    pref_velocities = np.random.choice(reach_velocities, n_neurons)

    for trial in range(n_trials):
        direction = trial_directions[trial]
        velocity = trial_velocities[trial]

        for neuron in range(n_neurons):
            # Cosine direction tuning
            dir_tuning = np.cos(direction - pref_directions[neuron])
            # Gaussian velocity tuning
            vel_tuning = np.exp(-((velocity - pref_velocities[neuron])**2) / (2 * 100))

            # Combined modulation
            firing_rate = baseline_rates[neuron] * (1 + 0.6 * dir_tuning) * (1 + 0.8 * vel_tuning)
            firing_rate = np.maximum(firing_rate, 0.5)

            spike_counts[trial, neuron] = np.random.poisson(firing_rate)

    return {
        'spike_counts': spike_counts.astype(int),
        'trial_directions': trial_directions,
        'trial_velocities': trial_velocities,
        'reach_directions': reach_directions,
        'reach_velocities': reach_velocities,
        'baseline_rates': baseline_rates,
        'pref_directions': pref_directions,
        'pref_velocities': pref_velocities
    }

data = generate_motor_cortex_data()
print(f"Generated data for {data['spike_counts'].shape[1]} neurons over {data['spike_counts'].shape[0]} trials")
print(f"Direction tuning: {len(data['reach_directions'])} reach directions")
print(f"Velocity tuning: {len(data['reach_velocities'])} reach velocities")

# %% [markdown]
# ## Direction Tuning Analysis
#
# Motor cortex neurons show cosine-like tuning to reach direction. We compute
# tuning curves by averaging firing rate for each direction.

# %%
def compute_direction_tuning(spike_counts, trial_directions, reach_directions):
    """Compute mean firing rate for each direction."""
    n_neurons = spike_counts.shape[1]
    n_directions = len(reach_directions)
    direction_tuning = np.zeros((n_neurons, n_directions))

    for neuron in range(n_neurons):
        for dir_idx, direction in enumerate(reach_directions):
            mask = np.abs(trial_directions - direction) < 0.01
            if np.sum(mask) > 0:
                direction_tuning[neuron, dir_idx] = np.mean(spike_counts[mask, neuron])

    return direction_tuning

direction_tuning = compute_direction_tuning(
    data['spike_counts'],
    data['trial_directions'],
    data['reach_directions']
)

print(f"Direction tuning computed: shape {direction_tuning.shape}")
print(f"Mean firing rates range: {direction_tuning.min():.2f} - {direction_tuning.max():.2f} spikes/s")

# %% [markdown]
# ## Velocity Tuning Analysis
#
# Motor cortex also encodes movement velocity with neurons showing peak firing
# at their preferred velocity.

# %%
def compute_velocity_tuning(spike_counts, trial_velocities, reach_velocities):
    """Compute mean firing rate for each velocity."""
    n_neurons = spike_counts.shape[1]
    n_velocities = len(reach_velocities)
    velocity_tuning = np.zeros((n_neurons, n_velocities))

    for neuron in range(n_neurons):
        for vel_idx, velocity in enumerate(reach_velocities):
            mask = np.abs(trial_velocities - velocity) < 0.5
            if np.sum(mask) > 0:
                velocity_tuning[neuron, vel_idx] = np.mean(spike_counts[mask, neuron])

    return velocity_tuning

velocity_tuning = compute_velocity_tuning(
    data['spike_counts'],
    data['trial_velocities'],
    data['reach_velocities']
)

print(f"Velocity tuning computed: shape {velocity_tuning.shape}")
print(f"Mean firing rates range: {velocity_tuning.min():.2f} - {velocity_tuning.max():.2f} spikes/s")

# %% [markdown]
# ## Preferred Direction Analysis
#
# For each neuron, we identify its preferred direction by finding the direction
# with maximum mean firing rate.

# %%
def compute_preferred_direction(direction_tuning, reach_directions):
    """Find preferred direction for each neuron using circular mean."""
    n_neurons = direction_tuning.shape[0]
    pref_dirs = np.zeros(n_neurons)
    modulation_depth = np.zeros(n_neurons)

    for neuron in range(n_neurons):
        # Normalize firing rates
        rates = direction_tuning[neuron, :]
        if rates.max() > rates.min():
            norm_rates = (rates - rates.min()) / (rates.max() - rates.min())
        else:
            norm_rates = np.ones_like(rates) / len(rates)

        # Circular mean of preferred direction
        sin_sum = np.sum(norm_rates * np.sin(reach_directions))
        cos_sum = np.sum(norm_rates * np.cos(reach_directions))
        pref_dirs[neuron] = np.arctan2(sin_sum, cos_sum) % (2*np.pi)

        # Tuning depth (modulation index)
        modulation_depth[neuron] = (rates.max() - rates.min()) / (rates.max() + rates.min() + 1e-6)

    return pref_dirs, modulation_depth

pref_dirs, mod_depth = compute_preferred_direction(direction_tuning, data['reach_directions'])
print(f"Preferred directions range: 0 to 2π")
print(f"Mean modulation depth: {mod_depth.mean():.3f}")

# %% [markdown]
# ## Preferred Velocity Analysis

# %%
def compute_preferred_velocity(velocity_tuning, reach_velocities):
    """Find preferred velocity for each neuron."""
    n_neurons = velocity_tuning.shape[0]
    pref_vels = np.zeros(n_neurons)
    vel_mod_depth = np.zeros(n_neurons)

    for neuron in range(n_neurons):
        rates = velocity_tuning[neuron, :]
        pref_idx = np.argmax(rates)
        pref_vels[neuron] = reach_velocities[pref_idx]

        if rates.max() > rates.min():
            vel_mod_depth[neuron] = (rates.max() - rates.min()) / (rates.max() + rates.min() + 1e-6)
        else:
            vel_mod_depth[neuron] = 0

    return pref_vels, vel_mod_depth

pref_vels, vel_mod_depth = compute_preferred_velocity(velocity_tuning, data['reach_velocities'])
print(f"Preferred velocities: {np.unique(pref_vels)} cm/s")
print(f"Mean velocity modulation depth: {vel_mod_depth.mean():.3f}")

# %% [markdown]
# ## Visualization 1: Direction Tuning Curves (Polar Plots)
#
# Polar plots show how firing rate varies with reach direction for selected neurons.
# Each neuron exhibits a characteristic tuning curve with a preferred direction
# (maximum firing rate).

# %%
fig = plt.figure(figsize=(14, 10))
n_example_neurons = 12
neurons_to_plot = np.linspace(0, data['spike_counts'].shape[1]-1, n_example_neurons, dtype=int)

for plot_idx, neuron in enumerate(neurons_to_plot):
    ax = fig.add_subplot(3, 4, plot_idx+1, projection='polar')

    # Plot tuning curve
    angles = np.concatenate([data['reach_directions'], [data['reach_directions'][0]]])
    rates = np.concatenate([direction_tuning[neuron, :], [direction_tuning[neuron, 0]]])

    ax.plot(angles, rates, 'o-', linewidth=2, markersize=6, color='steelblue', label='Tuning curve')
    ax.fill(angles, rates, alpha=0.3, color='steelblue')

    # Mark preferred direction
    ax.plot([pref_dirs[neuron]], [direction_tuning[neuron, :].max()], 'r*', markersize=15, label='Pref. dir.')

    ax.set_ylim(0, direction_tuning[neuron, :].max() * 1.2)
    ax.set_title(f'Neuron {neuron+1}\nPref: {np.degrees(pref_dirs[neuron]):.0f}°', fontsize=10, pad=10)
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('01_direction_tuning_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: 01_direction_tuning_curves.png")

# %% [markdown]
# ## Visualization 2: Population Direction Tuning Map
#
# A heatmap showing how all neurons respond to different reach directions.
# Each row is a neuron, each column is a reach direction.

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))

# Direction tuning heatmap
im1 = ax1.imshow(direction_tuning, aspect='auto', cmap='viridis', interpolation='nearest')
ax1.set_xlabel('Reach Direction (ordinal)', fontsize=11)
ax1.set_ylabel('Neuron', fontsize=11)
ax1.set_title('Population Direction Tuning', fontsize=12, fontweight='bold')
ax1.set_xticks(range(len(data['reach_directions'])))
ax1.set_xticklabels([f'{int(np.degrees(d))}°' for d in data['reach_directions']], rotation=45)
cbar1 = plt.colorbar(im1, ax=ax1)
cbar1.set_label('Firing Rate (spikes/s)', fontsize=10)

# Velocity tuning heatmap
im2 = ax2.imshow(velocity_tuning, aspect='auto', cmap='plasma', interpolation='nearest')
ax2.set_xlabel('Reach Velocity (cm/s)', fontsize=11)
ax2.set_ylabel('Neuron', fontsize=11)
ax2.set_title('Population Velocity Tuning', fontsize=12, fontweight='bold')
ax2.set_xticks(range(len(data['reach_velocities'])))
ax2.set_xticklabels([f'{int(v)}' for v in data['reach_velocities']])
cbar2 = plt.colorbar(im2, ax=ax2)
cbar2.set_label('Firing Rate (spikes/s)', fontsize=10)

plt.tight_layout()
plt.savefig('02_population_tuning_heatmaps.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: 02_population_tuning_heatmaps.png")

# %% [markdown]
# ## Visualization 3: Preferred Direction Distribution
#
# A circular histogram showing how neurons distribute their preferred directions.
# Uniform distribution suggests the population covers all reach directions.

# %%
fig = plt.figure(figsize=(12, 5))

# Preferred direction histogram
ax1 = fig.add_subplot(1, 2, 1, projection='polar')
bins = np.linspace(0, 2*np.pi, 9)
hist, bin_edges = np.histogram(pref_dirs, bins=bins)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
width = bin_edges[1] - bin_edges[0]
ax1.bar(bin_centers, hist, width=width*0.9, alpha=0.7, color='steelblue', edgecolor='black')
ax1.set_ylim(0, max(hist) * 1.3)
ax1.set_title('Preferred Direction Distribution', fontsize=12, fontweight='bold', pad=20)
ax1.set_theta_offset(np.pi/2)
ax1.set_theta_direction(-1)
ax1.grid(True, alpha=0.3)

# Modulation depth vs direction
ax2 = fig.add_subplot(1, 2, 2)
scatter = ax2.scatter(np.degrees(pref_dirs), mod_depth, c=mod_depth, cmap='viridis', s=80, alpha=0.6, edgecolors='black')
ax2.set_xlabel('Preferred Direction (degrees)', fontsize=11)
ax2.set_ylabel('Modulation Depth', fontsize=11)
ax2.set_title('Direction Tuning Strength', fontsize=12, fontweight='bold')
ax2.set_xlim(-10, 370)
ax2.grid(True, alpha=0.3)
ax2.axhline(mod_depth.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {mod_depth.mean():.2f}')
ax2.legend()
cbar = plt.colorbar(scatter, ax=ax2)
cbar.set_label('Modulation Depth', fontsize=10)

plt.tight_layout()
plt.savefig('03_preferred_direction_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: 03_preferred_direction_distribution.png")

# %% [markdown]
# ## Visualization 4: Velocity Tuning Profiles
#
# Bar plots showing mean firing rate at each velocity for selected neurons,
# demonstrating diversity in velocity preferences.

# %%
fig = plt.figure(figsize=(14, 6))
n_velocity_examples = 8
neurons_to_plot = np.linspace(0, data['spike_counts'].shape[1]-1, n_velocity_examples, dtype=int)

for plot_idx, neuron in enumerate(neurons_to_plot):
    ax = fig.add_subplot(2, 4, plot_idx+1)

    rates = velocity_tuning[neuron, :]
    colors = ['steelblue' if v != pref_vels[neuron] else 'coral' for v in data['reach_velocities']]

    bars = ax.bar([f'{int(v)}' for v in data['reach_velocities']], rates, color=colors, edgecolor='black', alpha=0.7)
    ax.set_ylabel('Firing Rate (spikes/s)', fontsize=10)
    ax.set_xlabel('Velocity (cm/s)', fontsize=10)
    ax.set_title(f'Neuron {neuron+1}\nPref: {int(pref_vels[neuron])} cm/s', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, rates.max() * 1.3)

plt.suptitle('Velocity Tuning Profiles by Neuron', fontsize=13, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('04_velocity_tuning_profiles.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: 04_velocity_tuning_profiles.png")

# %% [markdown]
# ## Visualization 5: Joint Direction-Velocity Tuning
#
# A 2D heatmap showing how a neuron's firing rate depends on both direction
# and velocity. This demonstrates that motor cortex encodes both dimensions.

# %%
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
example_neurons = [2, 8, 15, 25]

for plot_idx, neuron in enumerate(example_neurons):
    ax = axes.flat[plot_idx]

    # Create joint tuning surface
    joint_tuning = np.zeros((len(data['reach_velocities']), len(data['reach_directions'])))

    for vel_idx, velocity in enumerate(data['reach_velocities']):
        for dir_idx, direction in enumerate(data['reach_directions']):
            mask = (np.abs(data['trial_velocities'] - velocity) < 0.5) & \
                   (np.abs(data['trial_directions'] - direction) < 0.01)
            if np.sum(mask) > 0:
                joint_tuning[vel_idx, dir_idx] = np.mean(data['spike_counts'][mask, neuron])

    im = ax.imshow(joint_tuning, aspect='auto', cmap='hot', interpolation='bilinear')
    ax.set_xlabel('Direction (ordinal)', fontsize=10)
    ax.set_ylabel('Velocity (cm/s)', fontsize=10)
    ax.set_title(f'Neuron {neuron+1}\nJoint Dir-Vel Tuning', fontsize=11, fontweight='bold')
    ax.set_xticks(range(len(data['reach_directions'])))
    ax.set_xticklabels([f'{int(np.degrees(d))}°' for d in data['reach_directions']], rotation=45, fontsize=9)
    ax.set_yticks(range(len(data['reach_velocities'])))
    ax.set_yticklabels([f'{int(v)}' for v in data['reach_velocities']], fontsize=9)
    plt.colorbar(im, ax=ax, label='Rate (sp/s)')

plt.suptitle('Joint Direction-Velocity Tuning Surface', fontsize=13, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('05_joint_direction_velocity_tuning.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: 05_joint_direction_velocity_tuning.png")

# %% [markdown]
# ## Statistical Analysis
#
# We perform ANOVA to test whether direction and velocity significantly modulate firing rates.

# %%
from scipy.stats import f_oneway

# Direction ANOVA
direction_f_stats = []
direction_p_values = []

for neuron in range(data['spike_counts'].shape[1]):
    groups = []
    for direction in data['reach_directions']:
        mask = np.abs(data['trial_directions'] - direction) < 0.01
        if np.sum(mask) > 0:
            groups.append(data['spike_counts'][mask, neuron])

    if len(groups) > 1:
        f_stat, p_val = f_oneway(*groups)
        direction_f_stats.append(f_stat)
        direction_p_values.append(p_val)

# Velocity ANOVA
velocity_f_stats = []
velocity_p_values = []

for neuron in range(data['spike_counts'].shape[1]):
    groups = []
    for velocity in data['reach_velocities']:
        mask = np.abs(data['trial_velocities'] - velocity) < 0.5
        if np.sum(mask) > 0:
            groups.append(data['spike_counts'][mask, neuron])

    if len(groups) > 1:
        f_stat, p_val = f_oneway(*groups)
        velocity_f_stats.append(f_stat)
        velocity_p_values.append(p_val)

# Count significantly tuned neurons
sig_direction = np.sum(np.array(direction_p_values) < 0.05)
sig_velocity = np.sum(np.array(velocity_p_values) < 0.05)

print(f"\nStatistical Analysis Results:")
print(f"Direction-tuned neurons (p<0.05): {sig_direction}/{len(direction_p_values)} ({100*sig_direction/len(direction_p_values):.1f}%)")
print(f"Velocity-tuned neurons (p<0.05): {sig_velocity}/{len(velocity_p_values)} ({100*sig_velocity/len(velocity_p_values):.1f}%)")
print(f"Mean F-statistic (direction): {np.mean(direction_f_stats):.2f}")
print(f"Mean F-statistic (velocity): {np.mean(velocity_f_stats):.2f}")

# %% [markdown]
# ## Visualization 6: Statistical Significance
#
# Volcano plots showing which neurons are significantly tuned to direction and velocity.

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Direction tuning significance
log_p_direction = -np.log10(np.array(direction_p_values) + 1e-10)
sig_mask_dir = np.array(direction_p_values) < 0.05

ax1.scatter(direction_f_stats, log_p_direction, c=sig_mask_dir, cmap='RdYlBu_r', s=100, alpha=0.6, edgecolors='black')
ax1.axhline(-np.log10(0.05), color='red', linestyle='--', linewidth=2, label='p = 0.05')
ax1.set_xlabel('F-statistic', fontsize=11)
ax1.set_ylabel('-log10(p-value)', fontsize=11)
ax1.set_title('Direction Tuning Significance', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Velocity tuning significance
log_p_velocity = -np.log10(np.array(velocity_p_values) + 1e-10)
sig_mask_vel = np.array(velocity_p_values) < 0.05

ax2.scatter(velocity_f_stats, log_p_velocity, c=sig_mask_vel, cmap='RdYlBu_r', s=100, alpha=0.6, edgecolors='black')
ax2.axhline(-np.log10(0.05), color='red', linestyle='--', linewidth=2, label='p = 0.05')
ax2.set_xlabel('F-statistic', fontsize=11)
ax2.set_ylabel('-log10(p-value)', fontsize=11)
ax2.set_title('Velocity Tuning Significance', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig('06_tuning_significance.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: 06_tuning_significance.png")

# %% [markdown]
# ## Summary and Key Findings

# %%
print("\n" + "="*60)
print("REACH DIRECTION AND VELOCITY TUNING ANALYSIS - SUMMARY")
print("="*60)

print(f"\n1. DIRECTION TUNING:")
print(f"   - {sig_direction} of {len(direction_p_values)} neurons significantly tuned")
print(f"   - Preferred directions uniformly distributed across 0-360°")
print(f"   - Mean modulation depth: {mod_depth.mean():.3f}")
print(f"   - Modulation range: {mod_depth.min():.3f} - {mod_depth.max():.3f}")

print(f"\n2. VELOCITY TUNING:")
print(f"   - {sig_velocity} of {len(velocity_p_values)} neurons significantly tuned")
print(f"   - Preferred velocities: {sorted(np.unique(pref_vels))}")
print(f"   - Mean velocity modulation depth: {vel_mod_depth.mean():.3f}")
print(f"   - Velocity modulation range: {vel_mod_depth.min():.3f} - {vel_mod_depth.max():.3f}")

print(f"\n3. POPULATION ENCODING:")
print(f"   - Motor cortex exhibits two-dimensional encoding:")
print(f"     * Direction-selective cells with cosine-like tuning")
print(f"     * Velocity-selective cells with preferred speed")
print(f"   - This dual encoding enables flexible motor control")
print(f"   - Population vector can decode both reach direction and speed")

print(f"\n4. FUNCTIONAL SIGNIFICANCE:")
print(f"   - Direction tuning: necessary for reaching in correct direction")
print(f"   - Velocity tuning: necessary for scaling movement speed")
print(f"   - Joint encoding: motor cortex simultaneously represents these")
print(f"     aspects of movement, enabling coordinated motor commands")

print("\n" + "="*60)
print(f"Analysis complete. Generated 6 figures in current directory.")
print("="*60)
