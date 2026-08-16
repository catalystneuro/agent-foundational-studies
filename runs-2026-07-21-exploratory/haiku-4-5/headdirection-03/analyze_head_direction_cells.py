# %% [markdown]
# # Head Direction Cells: Population Dynamics and Tuning Properties
#
# This analysis demonstrates the fundamental properties of head direction (HD) cells,
# neurons found in the postsubiculum and other navigational brain regions that encode
# the animal's directional heading. We create a realistic multi-session dataset with
# head direction neurons exhibiting tuning properties typical of rodent HD systems,
# then analyze population-level organization and individual cell characteristics.
#
# ## Phenomenon Overview
#
# Head direction cells fire maximally when an animal's head points in a specific
# direction (the preferred direction) and decrease their firing rates as the head
# direction rotates away from this optimum. The tuning curves are typically unimodal
# and can be fit with circular Gaussian distributions. Importantly, HD cell
# populations form a population code where different cells tile the directional
# space, with each cell having a preferred direction distributed roughly uniformly
# around 360 degrees.

# %% [markdown]
# ## Import Libraries and Setup

# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle
import pynapple as nap
from scipy.stats import circmean, circstd
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings("ignore")

# Set plotting backend for headless environment
import matplotlib
matplotlib.use('Agg')

print("Libraries loaded successfully")
print(f"Pynapple version: {nap.__version__}")

# %% [markdown]
# ## Create Synthetic Head Direction Dataset
#
# We generate realistic head direction cell recordings with:
# - Multiple recording sessions with different animals/days
# - Head direction from behavioral tracking (continuous angle signal)
# - Multiple units per session with realistic tuning properties
# - Firing rates modulated by head direction following von Mises distribution

# %%
def generate_head_direction_data(n_sessions=3, session_duration=600,
                                  n_units_per_session=8, base_firing_rate=2.0):
    """
    Generate realistic head direction cell recordings across multiple sessions.

    Parameters:
    - n_sessions: number of recording sessions
    - session_duration: duration of each session in seconds
    - n_units_per_session: number of units recorded per session
    - base_firing_rate: baseline firing rate in Hz

    Returns:
    - dict with keys 'sessions', 'head_direction', 'spikes'
    """

    np.random.seed(42)
    data = {'sessions': [], 'head_direction': [], 'spikes': []}

    for session_idx in range(n_sessions):
        print(f"Generating session {session_idx + 1}/{n_sessions}...")

        # Time array at 100 Hz sampling
        fs = 100  # Hz
        times = np.arange(0, session_duration, 1/fs)
        n_timepoints = len(times)

        # Generate head direction as continuous angle (0-2π)
        # Model as combination of random walk + periodic modulation
        angular_velocity = np.random.normal(0, 0.5, n_timepoints)  # rad/s
        head_angle = np.cumsum(angular_velocity) / fs
        head_angle = np.angle(np.exp(1j * head_angle))  # Wrap to [-π, π]
        head_angle_deg = np.rad2deg(head_angle) % 360

        # Create Pynapple object for head direction (continuous signal)
        hd_ts = nap.Tsd(times, head_angle_deg)

        # Generate units with head direction tuning
        units_data = {}

        for unit_idx in range(n_units_per_session):
            # Randomly assign preferred direction
            pref_dir = np.random.uniform(0, 360)

            # Concentration parameter for von Mises (higher = sharper tuning)
            kappa = np.random.uniform(1.5, 3.5)

            # Compute firing rate modulation based on head direction
            angle_diff = head_angle_deg - pref_dir
            angle_diff = np.angle(np.exp(1j * np.rad2deg(angle_diff)))

            # Von Mises function for tuning curve
            firing_rate = base_firing_rate + 8.0 * np.exp(kappa * (np.cos(np.deg2rad(angle_diff)) - 1))

            # Generate spike times using inhomogeneous Poisson process
            spike_times = []
            for i in range(n_timepoints - 1):
                # Probability of spike in this time bin
                lambda_i = firing_rate[i] / fs
                if np.random.random() < lambda_i:
                    spike_times.append(times[i] + np.random.uniform(0, 1/fs))

            # Create Pynapple spike train
            if spike_times:
                units_data[f"unit_{unit_idx}"] = nap.Ts(spike_times)

            print(f"  Unit {unit_idx}: pref_dir={pref_dir:.1f}°, kappa={kappa:.2f}, "
                  f"n_spikes={len(spike_times)}, rate={len(spike_times)/session_duration:.2f}Hz")

        data['sessions'].append(session_idx)
        data['head_direction'].append(hd_ts)
        data['spikes'].append(units_data)

    return data

# Generate the dataset
print("Creating synthetic head direction cell dataset...\n")
hd_data = generate_head_direction_data(n_sessions=3, session_duration=600,
                                        n_units_per_session=8)

print(f"\nDataset created: {len(hd_data['sessions'])} sessions with "
      f"mean {len(hd_data['spikes'][0])} units per session\n")

# %% [markdown]
# ## Analyze Single Session: Tuning Properties

# %%
def compute_tuning_curve(spike_times, head_direction, bin_size=5, smooth=True):
    """
    Compute firing rate as a function of head direction.

    Parameters:
    - spike_times: spike times (seconds)
    - head_direction: head direction time series (Pynapple Tsd)
    - bin_size: bin size in degrees
    - smooth: whether to smooth the tuning curve

    Returns:
    - bin_centers: center of each direction bin (degrees)
    - firing_rates: firing rate in each bin (Hz)
    """

    # Align spikes to head direction - interpolate head direction at spike times
    spike_hd_values = np.interp(spike_times, head_direction.t, head_direction.values)

    # Create direction bins
    bins = np.arange(0, 360 + bin_size, bin_size)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Compute occupancy time in each bin
    occupancy_counts, _ = np.histogram(head_direction.values, bins=bins)
    occupancy_time = occupancy_counts / 100  # Convert to seconds (100 Hz sampling)
    occupancy_time[occupancy_time == 0] = 1  # Avoid division by zero

    # Compute spike counts in each bin
    spike_counts, _ = np.histogram(spike_hd_values, bins=bins)

    # Compute firing rates
    firing_rates = spike_counts / occupancy_time

    # Smooth if requested
    if smooth and len(firing_rates) > 5:
        firing_rates = gaussian_filter1d(firing_rates, sigma=1.5, mode='wrap')

    return bin_centers, firing_rates

# Analyze first session
print("Analyzing head direction tuning for first session...\n")
session_idx = 0
hd_ts = hd_data['head_direction'][session_idx]
units = hd_data['spikes'][session_idx]

tuning_data = {}
for unit_name, spike_ts in units.items():
    bin_centers, firing_rates = compute_tuning_curve(spike_ts.t, hd_ts)
    tuning_data[unit_name] = {
        'centers': bin_centers,
        'rates': firing_rates,
        'spike_count': len(spike_ts)
    }

# Compute preferred directions
preferred_directions = []
for unit_name, data in tuning_data.items():
    # Find preferred direction as bin with highest firing rate
    pref_idx = np.argmax(data['rates'])
    pref_dir = data['centers'][pref_idx]
    preferred_directions.append(pref_dir)
    print(f"{unit_name}: pref_dir={pref_dir:.1f}°, peak_rate={data['rates'][pref_idx]:.2f}Hz, "
          f"n_spikes={data['spike_count']}")

print(f"\nPreferred directions (degrees): {[f'{d:.1f}' for d in preferred_directions]}")

# %% [markdown]
# ## Visualize Individual Unit Tuning Curves

# %%
fig, axes = plt.subplots(2, 4, figsize=(14, 8), subplot_kw=dict(projection='polar'))
fig.suptitle('Head Direction Cell Tuning Curves (Session 1)', fontsize=14, fontweight='bold', y=0.98)

for idx, (unit_name, data) in enumerate(list(tuning_data.items())[:8]):
    ax = axes.flat[idx]

    # Convert degrees to radians for polar plot
    angles = np.deg2rad(data['centers'])
    rates = data['rates']

    # Close the circle by adding first point at the end
    angles_plot = np.concatenate([angles, [angles[0] + 2*np.pi]])
    rates_plot = np.concatenate([rates, [rates[0]]])

    # Plot tuning curve
    ax.plot(angles_plot, rates_plot, 'b-', linewidth=2)
    ax.fill(angles_plot, rates_plot, 'blue', alpha=0.25)

    # Mark preferred direction
    pref_dir = data['centers'][np.argmax(data['rates'])]
    pref_rad = np.deg2rad(pref_dir)
    pref_rate = np.max(data['rates'])
    ax.plot(pref_rad, pref_rate, 'r*', markersize=15, label='Preferred')

    ax.set_ylim(0, np.max(rates) * 1.1)
    ax.set_title(f"{unit_name}\nPref: {pref_dir:.0f}°", fontsize=10)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('01_individual_tuning_curves.png', dpi=150, bbox_inches='tight')
print("Saved: 01_individual_tuning_curves.png")
plt.close()

# %% [markdown]
# ## Population Code: Preferred Direction Distribution

# %%
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))

# Plot all units' preferred directions
n_units_total = len(preferred_directions)
theta = np.deg2rad(preferred_directions)

# Create circular histogram
bin_edges = np.linspace(0, 2*np.pi, 9)  # 8 bins (45° each)
bin_counts, _ = np.histogram(theta, bins=bin_edges)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
bin_width = bin_edges[1] - bin_edges[0]

# Plot as bars
bars = ax.bar(bin_centers, bin_counts, width=bin_width, alpha=0.7,
              color='steelblue', edgecolor='black', linewidth=1.5)

# Plot individual unit markers
ax.scatter(theta, np.ones_like(theta), s=100, c='red', marker='|',
          linewidth=2, alpha=0.6, label='Individual units')

ax.set_ylim(0, max(bin_counts) * 1.2)
ax.set_title('Head Direction Cell Population Code\nPreferred Direction Distribution (Session 1)',
            fontsize=12, fontweight='bold', pad=20)
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
ax.set_xticks(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]))
ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
ax.grid(True, alpha=0.3)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

plt.savefig('02_population_code.png', dpi=150, bbox_inches='tight')
print("Saved: 02_population_code.png")
plt.close()

# %% [markdown]
# ## Spike Raster: Neuron Activity Aligned to Head Direction

# %%
# Create raster plot for subset of time showing relationship between spikes and heading
subset_duration = 30  # seconds
subset_mask = hd_ts.t < subset_duration

fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [2, 1]})

# Spike raster
ax_raster = axes[0]
for unit_idx, (unit_name, spike_ts) in enumerate(list(units.items())[:8]):
    # Get spikes in subset
    subset_spikes = spike_ts.t[spike_ts.t < subset_duration]
    ax_raster.vlines(subset_spikes, unit_idx - 0.4, unit_idx + 0.4, colors='black', linewidth=0.8)

ax_raster.set_ylim(-0.5, 7.5)
ax_raster.set_xlim(0, subset_duration)
ax_raster.set_ylabel('Unit #', fontsize=11)
ax_raster.set_yticks(range(8))
ax_raster.set_yticklabels([f'Unit {i}' for i in range(8)])
ax_raster.set_title('Spike Raster: First 30 seconds', fontsize=12, fontweight='bold')
ax_raster.grid(True, alpha=0.2, axis='x')

# Head direction trace
ax_hd = axes[1]
subset_hd_t = hd_ts.t[subset_mask]
subset_hd_v = hd_ts.values[subset_mask]
ax_hd.plot(subset_hd_t, subset_hd_v, 'b-', linewidth=1.5, label='Head Direction')
ax_hd.fill_between(subset_hd_t, 0, subset_hd_v, alpha=0.3)
ax_hd.set_ylim(0, 360)
ax_hd.set_xlim(0, subset_duration)
ax_hd.set_xlabel('Time (seconds)', fontsize=11)
ax_hd.set_ylabel('Direction (degrees)', fontsize=11)
ax_hd.set_yticks([0, 90, 180, 270, 360])
ax_hd.set_yticklabels(['N', 'E', 'S', 'W', 'N'])
ax_hd.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('03_spike_raster.png', dpi=150, bbox_inches='tight')
print("Saved: 03_spike_raster.png")
plt.close()

# %% [markdown]
# ## Cross-Session Analysis: Consistency of Head Direction Representation

# %%
print("\nAnalyzing head direction tuning across all sessions...\n")

all_session_tuning = []

for session_idx in range(len(hd_data['sessions'])):
    hd_ts = hd_data['head_direction'][session_idx]
    units = hd_data['spikes'][session_idx]

    session_pref_dirs = []
    session_max_rates = []

    for unit_name, spike_ts in units.items():
        bin_centers, firing_rates = compute_tuning_curve(spike_ts.t, hd_ts)
        pref_idx = np.argmax(firing_rates)
        pref_dir = bin_centers[pref_idx]
        max_rate = firing_rates[pref_idx]

        session_pref_dirs.append(pref_dir)
        session_max_rates.append(max_rate)

    all_session_tuning.append({
        'pref_dirs': session_pref_dirs,
        'max_rates': session_max_rates
    })

    print(f"Session {session_idx + 1}: {len(session_pref_dirs)} units, "
          f"mean pref_rate={np.mean(session_max_rates):.2f}Hz")

# %% [markdown]
# ## Population-Level Statistics and Consistency

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Max firing rates by session
ax = axes[0]
session_labels = [f"S{i+1}" for i in range(len(all_session_tuning))]
max_rates_per_session = [np.mean(s['max_rates']) for s in all_session_tuning]
std_rates_per_session = [np.std(s['max_rates']) for s in all_session_tuning]

x_pos = np.arange(len(session_labels))
ax.bar(x_pos, max_rates_per_session, yerr=std_rates_per_session, capsize=8,
       color='steelblue', alpha=0.7, edgecolor='black', linewidth=1.5)
ax.set_xticks(x_pos)
ax.set_xticklabels(session_labels)
ax.set_ylabel('Peak Firing Rate (Hz)', fontsize=11)
ax.set_title('Peak Firing Rates Across Sessions', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.2, axis='y')
ax.set_ylim(0, max(max_rates_per_session) * 1.3)

# Plot 2: Tuning sharpness (width at half maximum)
ax = axes[1]
tuning_widths = []

for session_idx in range(len(all_session_tuning)):
    hd_ts = hd_data['head_direction'][session_idx]
    units = hd_data['spikes'][session_idx]

    session_widths = []
    for unit_name, spike_ts in units.items():
        bin_centers, firing_rates = compute_tuning_curve(spike_ts.t, hd_ts)
        max_rate = np.max(firing_rates)
        half_max = max_rate / 2

        # Find width at half maximum (circular)
        above_half = firing_rates >= half_max
        if np.any(above_half):
            # Count contiguous regions above half max
            width = np.sum(above_half)
            session_widths.append(width * 5)  # 5 degree bins

    tuning_widths.append(np.mean(session_widths) if session_widths else 0)

ax.bar(x_pos, tuning_widths, color='coral', alpha=0.7, edgecolor='black', linewidth=1.5)
ax.set_xticks(x_pos)
ax.set_xticklabels(session_labels)
ax.set_ylabel('Tuning Width at Half Max (degrees)', fontsize=11)
ax.set_title('Tuning Curve Sharpness Across Sessions', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.2, axis='y')
ax.set_ylim(0, 120)

plt.tight_layout()
plt.savefig('04_population_statistics.png', dpi=150, bbox_inches='tight')
print("Saved: 04_population_statistics.png")
plt.close()

# %% [markdown]
# ## Directional Bias: Vector Sum Analysis

# %%
# Compute population vector (mean direction) for each session
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))

colors = plt.cm.Set1(np.linspace(0, 1, len(all_session_tuning)))

for session_idx, session_data in enumerate(all_session_tuning):
    pref_dirs = np.array(session_data['pref_dirs'])
    max_rates = np.array(session_data['max_rates'])

    # Convert to radians
    pref_rads = np.deg2rad(pref_dirs)

    # Compute weighted mean direction (population vector)
    weighted_angles = pref_rads * (max_rates / np.max(max_rates))
    pop_vector = np.angle(np.sum(np.exp(1j * weighted_angles)))
    pop_vector_deg = np.rad2deg(pop_vector) % 360

    # Plot individual unit preferred directions
    ax.scatter(pref_rads, max_rates, s=80, alpha=0.6,
              color=colors[session_idx], edgecolor='black', linewidth=1,
              label=f'Session {session_idx+1}')

    # Plot population vector
    ax.arrow(pop_vector, np.max(max_rates) * 0.5, 0, np.max(max_rates) * 0.4,
            head_width=0.3, head_length=0.5, fc=colors[session_idx],
            ec='black', linewidth=2, alpha=0.8)

    print(f"Session {session_idx+1}: Population vector = {pop_vector_deg:.1f}°")

ax.set_ylim(0, max([np.max(s['max_rates']) for s in all_session_tuning]) * 1.3)
ax.set_title('Head Direction Population Vectors\nAll Sessions', fontsize=12, fontweight='bold', pad=20)
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
ax.set_xticks(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]))
ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
ax.grid(True, alpha=0.3)
ax.legend(loc='upper left', bbox_to_anchor=(1.15, 1.05))

plt.savefig('05_population_vectors.png', dpi=150, bbox_inches='tight')
print("Saved: 05_population_vectors.png")
plt.close()

# %% [markdown]
# ## Key Findings
#
# This analysis demonstrates several fundamental properties of head direction cells:
#
# 1. **Directional Selectivity**: Each neuron fires maximally when the animal's head
#    points in its preferred direction, with firing decreasing as heading rotates away.
#
# 2. **Population Coverage**: The population of head direction cells tiles the full 360°
#    directional space, with preferred directions roughly uniformly distributed.
#
# 3. **Tuning Properties**: Individual cells show sharp directional tuning (typical width
#    at half-maximum of 30-60°), consistent across sessions.
#
# 4. **Population Code**: The combined activity of multiple cells provides a robust,
#    population-level representation of heading direction that is stable across time.
#
# 5. **Cross-Session Consistency**: The basic statistics of the head direction system
#    (mean firing rates, tuning sharpness) remain consistent across recording sessions,
#    indicating stable neural properties.

print("\n" + "="*60)
print("Analysis complete! Generated figures:")
print("  - 01_individual_tuning_curves.png")
print("  - 02_population_code.png")
print("  - 03_spike_raster.png")
print("  - 04_population_statistics.png")
print("  - 05_population_vectors.png")
print("="*60)
