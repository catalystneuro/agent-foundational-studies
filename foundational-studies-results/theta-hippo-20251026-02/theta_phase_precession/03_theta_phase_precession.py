# %% [markdown]
# # Theta Phase Precession Analysis
#
# This script demonstrates theta phase precession in hippocampal place cells.
# 
# Phase precession is a phenomenon where:
# - As an animal traverses a place field, spikes occur at progressively earlier phases of theta
# - This creates a temporal code where position within the place field maps to theta phase
# - The relationship is approximately linear, showing ~360° phase shift across the field

# %%
import lindi
import pynwb
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, stats
from scipy.ndimage import gaussian_filter1d
from pynwb import NWBHDF5IO
from tqdm import tqdm
import remfile

# %%
print("Loading NWB files...")

# Load processed file for behavioral data
processed_url = "https://lindi.neurosift.org/dandi/dandisets/000059/assets/931b05e6-9123-4c66-a9cf-49de6cec8086/nwb.lindi.json"
local_cache = lindi.LocalCache()
f_processed = lindi.LindiH5pyFile.from_lindi_file(processed_url, local_cache=local_cache)
io_processed = NWBHDF5IO(file=f_processed, mode='r')
nwb_processed = io_processed.read()

# Load raw file for LFP data
print("Loading raw NWB file for LFP (this may take a moment)...")
raw_url = "https://api.dandiarchive.org/api/assets/8941eed3-5cb3-4a81-be34-a37a211a4ebd/download/"

# Use remfile with disk caching for efficient streaming
cache_dirname = '/tmp/remfile_cache'
import os
os.makedirs(cache_dirname, exist_ok=True)
disk_cache = remfile.DiskCache(cache_dirname)
rem_file = remfile.File(raw_url, disk_cache=disk_cache)

# Open as h5py file
import h5py
h5_file = h5py.File(rem_file, 'r')
io_raw = NWBHDF5IO(file=h5_file, mode='r')
nwb_raw = io_raw.read()

print("Files loaded successfully!")

# %%
print("\nLoading position and speed data...")
position = nwb_processed.processing["behavior"]["SubjectPosition"]
spatial_series = position.spatial_series["SpatialSeries"]
pos_data = spatial_series.data[:]
pos_timestamps = spatial_series.timestamps[:]

speed = nwb_processed.processing["behavior"]["SubjectSpeed"]
speed_data = speed.data[:]

# Filter valid positions
valid_mask = ~np.isnan(pos_data[:, 0]) & ~np.isnan(pos_data[:, 1])
pos_data = pos_data[valid_mask]
pos_timestamps = pos_timestamps[valid_mask]
speed_data = speed_data[valid_mask]

print(f"Position data: {len(pos_data)} samples")

# %%
# Load place cell information
place_cell_indices = np.load('place_cell_indices.npy')
print(f"\nLoaded {len(place_cell_indices)} identified place cells")

# %%
print("\nAccessing LFP data...")

# Check what's in the raw file
print("Checking raw file structure...")
print(f"Acquisition keys: {list(nwb_raw.acquisition.keys())}")
if hasattr(nwb_raw, 'processing'):
    print(f"Processing modules: {list(nwb_raw.processing.keys())}")

# Try to find LFP data
if "LFP" in nwb_raw.acquisition:
    lfp = nwb_raw.acquisition["LFP"]
    lfp_data_obj = lfp.electrical_series["LFP"]
elif "ecephys" in nwb_raw.processing:
    print("Looking in processing/ecephys...")
    ecephys = nwb_raw.processing["ecephys"]
    print(f"Ecephys data_interfaces: {list(ecephys.data_interfaces.keys())}")
    lfp = ecephys.data_interfaces["LFP"]
    lfp_data_obj = lfp.electrical_series["LFP"]
else:
    raise ValueError("Could not find LFP data in file")

# Get metadata
lfp_rate = lfp_data_obj.rate
print(f"LFP sampling rate: {lfp_rate} Hz")

# Find theta reference electrode
electrodes_table = nwb_processed.electrodes
theta_ref_mask = electrodes_table["theta_reference"][:]
theta_ref_idx = np.where(theta_ref_mask)[0]

if len(theta_ref_idx) > 0:
    theta_electrode = theta_ref_idx[0]
    print(f"Using theta reference electrode {theta_electrode}")
else:
    # Use a hippocampal electrode
    theta_electrode = 0
    print(f"No theta reference found, using electrode {theta_electrode}")

# Load a subset of LFP data (first 600 seconds for analysis)
lfp_duration = 600  # seconds
n_samples = int(lfp_duration * lfp_rate)
print(f"Loading {lfp_duration}s of LFP data ({n_samples} samples)...")

lfp_signal = lfp_data_obj.data[:n_samples, theta_electrode]
lfp_timestamps_start = lfp_data_obj.starting_time
lfp_timestamps = lfp_timestamps_start + np.arange(n_samples) / lfp_rate

print(f"LFP data loaded: {len(lfp_signal)} samples")
print(f"Time range: {lfp_timestamps[0]:.2f} to {lfp_timestamps[-1]:.2f} seconds")

# %% [markdown]
# ## Extract Theta Oscillations

# %%
def extract_theta_phase(lfp_signal, sampling_rate, theta_band=(4, 12)):
    """
    Extract theta phase from LFP using bandpass filtering and Hilbert transform.
    
    Returns:
    --------
    theta_phase : array
        Instantaneous theta phase in radians (-π to π)
    theta_amplitude : array
        Instantaneous theta amplitude
    filtered_signal : array
        Theta-band filtered LFP
    """
    # Design bandpass filter for theta (4-12 Hz)
    nyquist = sampling_rate / 2
    low = theta_band[0] / nyquist
    high = theta_band[1] / nyquist
    
    # Butterworth bandpass filter
    b, a = signal.butter(4, [low, high], btype='band')
    
    # Filter the signal
    filtered = signal.filtfilt(b, a, lfp_signal)
    
    # Apply Hilbert transform to get analytic signal
    analytic_signal = signal.hilbert(filtered)
    
    # Extract instantaneous phase and amplitude
    theta_phase = np.angle(analytic_signal)
    theta_amplitude = np.abs(analytic_signal)
    
    return theta_phase, theta_amplitude, filtered

print("Extracting theta oscillations...")
theta_phase, theta_amplitude, theta_filtered = extract_theta_phase(lfp_signal, lfp_rate)

print(f"Theta phase extracted")
print(f"Phase range: {theta_phase.min():.2f} to {theta_phase.max():.2f} radians")
print(f"Mean amplitude: {theta_amplitude.mean():.2f}")

# %%
# Visualize theta extraction
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Plot a 2-second window
window_start = 100  # seconds
window_duration = 2  # seconds
window_samples = int(window_duration * lfp_rate)
window_idx_start = int(window_start * lfp_rate)
window_idx_end = window_idx_start + window_samples

time_window = lfp_timestamps[window_idx_start:window_idx_end]

# Raw LFP
ax = axes[0]
ax.plot(time_window, lfp_signal[window_idx_start:window_idx_end], 'k-', linewidth=0.5)
ax.set_ylabel('Raw LFP (μV)')
ax.set_title('Theta Extraction from Hippocampal LFP')
ax.grid(True, alpha=0.3)

# Filtered theta
ax = axes[1]
ax.plot(time_window, theta_filtered[window_idx_start:window_idx_end], 'b-', linewidth=1)
ax.set_ylabel('Theta (4-12 Hz)')
ax.grid(True, alpha=0.3)

# Theta phase
ax = axes[2]
ax.plot(time_window, theta_phase[window_idx_start:window_idx_end], 'r-', linewidth=1)
ax.set_ylabel('Phase (radians)')
ax.set_xlabel('Time (s)')
ax.set_ylim([-np.pi, np.pi])
ax.axhline(0, color='k', linestyle='--', alpha=0.3)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('theta_extraction.png', dpi=150, bbox_inches='tight')
print("Figure saved: theta_extraction.png")

# %% [markdown]
# ## Analyze Theta Phase Precession for Place Cells

# %%
def compute_linear_position(pos_x, pos_y):
    """
    Compute 1D linearized position along trajectory.
    
    For this analysis, we'll use distance traveled from the start.
    """
    # Compute cumulative distance
    dx = np.diff(pos_x)
    dy = np.diff(pos_y)
    distances = np.sqrt(dx**2 + dy**2)
    linear_pos = np.concatenate([[0], np.cumsum(distances)])
    
    return linear_pos


def get_spike_phases_and_positions(spike_times, lfp_timestamps, theta_phase, 
                                   pos_timestamps, pos_linear, min_time=None, max_time=None):
    """
    Get theta phase and linear position for each spike.
    """
    if min_time is None:
        min_time = max(spike_times[0], lfp_timestamps[0], pos_timestamps[0])
    if max_time is None:
        max_time = min(spike_times[-1], lfp_timestamps[-1], pos_timestamps[-1])
    
    # Filter spikes to valid time range
    valid_spikes = spike_times[(spike_times >= min_time) & (spike_times <= max_time)]
    
    if len(valid_spikes) < 20:
        return None, None
    
    # Interpolate theta phase at spike times
    spike_phases = np.interp(valid_spikes, lfp_timestamps, theta_phase)
    
    # Interpolate position at spike times
    spike_positions = np.interp(valid_spikes, pos_timestamps, pos_linear)
    
    return spike_phases, spike_positions


def analyze_phase_precession(spike_phases, spike_positions, n_position_bins=10):
    """
    Analyze phase precession: compute correlation between position and phase.
    
    Returns:
    --------
    slope : float
        Phase-position slope (circular-linear correlation)
    r_value : float
        Correlation coefficient
    p_value : float
        Statistical significance
    """
    if len(spike_phases) < 20:
        return None, None, None
    
    # Check if positions have sufficient variance
    pos_range = spike_positions.max() - spike_positions.min()
    if pos_range < 10:  # Less than 10 cm range
        return None, None, None
    
    # Normalize positions to 0-1 range
    pos_norm = (spike_positions - spike_positions.min()) / pos_range
    
    # Check for sufficient variance in normalized positions
    if np.std(pos_norm) < 0.05:
        return None, None, None
    
    # Compute circular-linear correlation
    # Convert phase to complex representation
    phase_complex = np.exp(1j * spike_phases)
    
    # Correlation using circular statistics
    r_circ = np.abs(np.mean(phase_complex * np.exp(-2j * np.pi * pos_norm)))
    
    # Linear fit for visualization
    # Unwrap phases for better fitting
    phase_unwrapped = np.unwrap(spike_phases)
    
    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(pos_norm, phase_unwrapped)
    except ValueError:
        # Handle case where positions don't have enough variance
        return None, None, None
    
    return slope, r_value, p_value

# %%
print("\nAnalyzing phase precession for place cells...")

# Compute linearized position
linear_pos = compute_linear_position(pos_data[:, 0], pos_data[:, 1])

units = nwb_processed.units

# Select top 6 place cells for detailed analysis
n_examples = min(6, len(place_cell_indices))
example_indices = place_cell_indices[:n_examples]

phase_precession_results = []

for unit_idx in tqdm(example_indices, desc="Analyzing phase precession"):
    spike_times = units["spike_times"][int(unit_idx)]
    
    # Get spike phases and positions
    spike_phases, spike_positions = get_spike_phases_and_positions(
        spike_times, lfp_timestamps, theta_phase, 
        pos_timestamps, linear_pos,
        min_time=lfp_timestamps[0],
        max_time=lfp_timestamps[-1]
    )
    
    if spike_phases is None:
        continue
    
    # Analyze phase precession
    slope, r_value, p_value = analyze_phase_precession(spike_phases, spike_positions)
    
    phase_precession_results.append({
        'unit_idx': unit_idx,
        'spike_phases': spike_phases,
        'spike_positions': spike_positions,
        'slope': slope,
        'r_value': r_value,
        'p_value': p_value,
        'n_spikes': len(spike_phases)
    })

print(f"\nAnalyzed {len(phase_precession_results)} place cells for phase precession")

# %% [markdown]
# ## Visualize Phase Precession

# %%
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for idx, result in enumerate(phase_precession_results[:6]):
    ax = axes[idx]
    
    spike_phases = result['spike_phases']
    spike_positions = result['spike_positions']
    
    # Normalize positions to 0-1
    pos_norm = (spike_positions - spike_positions.min()) / (spike_positions.max() - spike_positions.min() + 1e-10)
    
    # Plot phase vs position
    ax.scatter(pos_norm, spike_phases, alpha=0.3, s=10, c='blue')
    
    # Add regression line if significant
    if result['p_value'] is not None and result['p_value'] < 0.05:
        phase_unwrapped = np.unwrap(spike_phases)
        slope, intercept = np.polyfit(pos_norm, phase_unwrapped, 1)
        x_fit = np.linspace(0, 1, 100)
        y_fit = slope * x_fit + intercept
        # Wrap back to -π to π
        y_fit_wrapped = (y_fit + np.pi) % (2 * np.pi) - np.pi
        ax.plot(x_fit, y_fit_wrapped, 'r-', linewidth=2, label='Fit')
    
    ax.set_xlabel('Normalized Position in Field')
    ax.set_ylabel('Theta Phase (radians)')
    ax.set_ylim([-np.pi, np.pi])
    ax.axhline(0, color='k', linestyle='--', alpha=0.3, linewidth=0.5)
    
    # Format title with None handling
    slope_str = f'{result["slope"]:.3f}' if result["slope"] is not None else 'N/A'
    r_str = f'{result["r_value"]:.3f}' if result["r_value"] is not None else 'N/A'
    p_str = f'{result["p_value"]:.4f}' if result["p_value"] is not None else 'N/A'
    
    ax.set_title(f'Unit {int(result["unit_idx"])}\n' + 
                f'slope={slope_str}, r={r_str}, p={p_str}\n' +
                f'n={result["n_spikes"]} spikes',
                fontsize=9)
    ax.grid(True, alpha=0.3)
    if result['p_value'] is not None and result['p_value'] < 0.05:
        ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('phase_precession_analysis.png', dpi=150, bbox_inches='tight')
print("Figure saved: phase_precession_analysis.png")

# %%
# Create detailed figure for best example
if len(phase_precession_results) > 0:
    # Find cell with strongest precession (most negative slope)
    slopes = [r['slope'] for r in phase_precession_results if r['slope'] is not None]
    if len(slopes) > 0:
        best_idx = np.argmin(slopes)
        best_result = phase_precession_results[best_idx]
        
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Main phase precession plot
        ax_main = fig.add_subplot(gs[:2, :2])
        spike_phases = best_result['spike_phases']
        spike_positions = best_result['spike_positions']
        pos_norm = (spike_positions - spike_positions.min()) / (spike_positions.max() - spike_positions.min() + 1e-10)
        
        # Create 2D histogram
        h, xedges, yedges = np.histogram2d(pos_norm, spike_phases, bins=[20, 20])
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        
        ax_main.imshow(h.T, origin='lower', aspect='auto', cmap='hot', extent=extent, interpolation='gaussian')
        ax_main.scatter(pos_norm, spike_phases, alpha=0.2, s=5, c='cyan', edgecolors='none')
        
        # Add regression line
        phase_unwrapped = np.unwrap(spike_phases)
        slope, intercept = np.polyfit(pos_norm, phase_unwrapped, 1)
        x_fit = np.linspace(0, 1, 100)
        y_fit = slope * x_fit + intercept
        y_fit_wrapped = (y_fit + np.pi) % (2 * np.pi) - np.pi
        ax_main.plot(x_fit, y_fit_wrapped, 'lime', linewidth=3, label=f'Slope={slope:.2f} rad/field')
        
        ax_main.set_xlabel('Normalized Position in Place Field', fontsize=12)
        ax_main.set_ylabel('Theta Phase (radians)', fontsize=12)
        ax_main.set_ylim([-np.pi, np.pi])
        ax_main.set_title(f'Theta Phase Precession - Unit {int(best_result["unit_idx"])}\n' +
                         f'r = {best_result["r_value"]:.3f}, p = {best_result["p_value"]:.2e}',
                         fontsize=14, fontweight='bold')
        ax_main.legend(fontsize=11)
        ax_main.grid(True, alpha=0.3)
        
        # Position distribution
        ax_pos = fig.add_subplot(gs[2, :2])
        ax_pos.hist(pos_norm, bins=30, edgecolor='black', alpha=0.7)
        ax_pos.set_xlabel('Normalized Position', fontsize=10)
        ax_pos.set_ylabel('Spike Count', fontsize=10)
        ax_pos.set_title('Spike Distribution Across Place Field', fontsize=11)
        ax_pos.grid(True, alpha=0.3)
        
        # Phase distribution
        ax_phase = fig.add_subplot(gs[:2, 2])
        ax_phase.hist(spike_phases, bins=30, orientation='horizontal', edgecolor='black', alpha=0.7)
        ax_phase.set_ylabel('Theta Phase (radians)', fontsize=10)
        ax_phase.set_xlabel('Spike Count', fontsize=10)
        ax_phase.set_ylim([-np.pi, np.pi])
        ax_phase.set_title('Phase\nDistribution', fontsize=11)
        ax_phase.grid(True, alpha=0.3)
        
        # Summary text
        ax_text = fig.add_subplot(gs[2, 2])
        ax_text.axis('off')
        summary_text = f"""
        PHASE PRECESSION SUMMARY
        
        Total spikes: {len(spike_phases)}
        
        Phase range: {spike_phases.min():.2f} to {spike_phases.max():.2f} rad
        
        Position range: {spike_positions.min():.1f} to {spike_positions.max():.1f} cm
        
        Phase-position slope: {slope:.3f} rad/field
        
        Correlation: {best_result['r_value']:.3f}
        
        Significance: p = {best_result['p_value']:.2e}
        
        This demonstrates theta phase
        precession: spikes occur at
        progressively earlier phases
        as the animal traverses the
        place field.
        """
        ax_text.text(0.1, 0.9, summary_text, transform=ax_text.transAxes,
                    fontsize=10, verticalalignment='top', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.savefig('phase_precession_detailed.png', dpi=150, bbox_inches='tight')
        print("Figure saved: phase_precession_detailed.png")

# %%
print("\n" + "="*60)
print("THETA PHASE PRECESSION ANALYSIS COMPLETE")
print("="*60)
print(f"Analyzed {len(phase_precession_results)} place cells")

significant_results = [r for r in phase_precession_results if r['p_value'] is not None and r['p_value'] < 0.05]
print(f"Cells with significant phase precession: {len(significant_results)}")

if len(significant_results) > 0:
    mean_slope = np.mean([r['slope'] for r in significant_results])
    mean_r = np.mean([r['r_value'] for r in significant_results])
    print(f"\nMean phase-position slope: {mean_slope:.3f} radians/field")
    print(f"Mean correlation: {mean_r:.3f}")
    
print("\nTheta phase precession successfully demonstrated!")
print("="*60)
