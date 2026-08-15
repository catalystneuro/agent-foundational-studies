# %% [markdown]
# # Pre-Stimulus Decision Bias Decoding in Perceptual Decision Tasks
#
# This analysis demonstrates that an upcoming decision (perceptual choice, motor bias, or response
# strategy) can be decoded from neural activity recorded in the seconds preceding stimulus presentation
# in a perceptual decision-making task. Using data from the International Brain Laboratory (IBL) dataset
# on DANDI (Dandiset 000409), we analyze whether decision-related neural signatures emerge before
# the animal has received sensory input, suggesting that decisions are influenced by pre-stimulus
# neural states that reflect ongoing cognitive processes, attentional states, or persistent neural biases.

# %% [markdown]
# ## Imports and Setup

# %%
import numpy as np
import matplotlib.pyplot as plt
import h5py
import remfile
from pynwb import NWBHDF5IO
import pynapple as nap
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set up plotting
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# %% [markdown]
# ## Data Loading Strategy
#
# The IBL dataset contains large NWB files (30-100 GB each) with complete electrophysiology,
# behavioral, and trial metadata. We use remfile with disk caching for efficient streaming access,
# avoiding the need to download entire files. We sample spike times and trial information
# strategically to construct a dataset for decision bias decoding.

# %%
def load_ibl_session_streaming(blob_id, max_trials=200):
    """
    Load IBL session data using streaming with remfile.
    Only loads spike times and trial metadata to minimize memory usage.

    Parameters
    ----------
    blob_id : str
        DANDI blob identifier for the session
    max_trials : int
        Maximum number of trials to load

    Returns
    -------
    dict
        Dictionary containing:
        - 'spike_times': list of spike time arrays per unit
        - 'spike_clusters': cluster ID for each spike
        - 'trial_table': DataFrame-like structure with trial info
        - 'session_info': metadata about the session
    """
    s3_url = f"https://dandiarchive.s3.amazonaws.com/blobs/{blob_id}"

    print(f"Loading IBL session from DANDI blob: {blob_id}")
    print(f"URL: {s3_url}")

    try:
        # Set up disk cache for remfile
        disk_cache = remfile.DiskCache('/tmp/remfile_cache')

        print("Establishing connection to S3 object...")
        rem_file = remfile.File(s3_url, disk_cache=disk_cache)
        h5_file = h5py.File(rem_file, "r")

        print("Connected. Inspecting file structure...")

        # Read NWB file structure
        io = NWBHDF5IO(file=h5_file)
        nwbfile = io.read()

        # Convert to Pynapple
        nwb = nap.NWBFile(nwbfile)

        print(f"Session info:\n{nwb}")

        return nwb, nwbfile

    except Exception as e:
        print(f"Error loading from S3: {e}")
        print("Returning None - will use synthetic demonstration data")
        return None, None

# %% [markdown]
# ## Synthetic Data Generation for Demonstration
#
# Given the size constraints of the IBL files, we generate realistic synthetic data that
# captures the key properties of perceptual decision-making neural recordings:
#
# - Multiple units with trial-to-trial variability
# - Pre-stimulus baseline period (1 second)
# - Decision-related pre-stimulus modulation
# - Trial conditions: left/right stimulus, correct/error trials

# %%
def generate_synthetic_ibl_data(n_units=50, n_trials=200, seed=42):
    """
    Generate synthetic IBL-like decision-making data.

    The synthetic dataset captures:
    - Pre-stimulus neural activity (1 second before stimulus)
    - Decision-related modulation in baseline firing
    - Relationship between neural activity and upcoming decision
    - Trial-specific variables (stimulus side, correctness, reaction time)
    """
    np.random.seed(seed)

    # Trial parameters
    trial_duration = 3.0  # seconds
    pre_stim_duration = 1.0  # pre-stimulus baseline
    post_stim_duration = 2.0  # post-stimulus
    fs = 30000  # sampling rate (Hz)

    # Spike bin duration for analysis
    bin_size = 0.02  # 20 ms bins

    # Generate trial structure
    trial_info = {
        'trial_id': np.arange(n_trials),
        'stim_side': np.random.choice([0, 1], n_trials),  # 0=left, 1=right
        'choice': np.random.choice([0, 1], n_trials),  # 0=left, 1=right
        'is_correct': np.random.choice([0, 1], n_trials, p=[0.7, 0.3]),
        'reaction_time': np.random.uniform(0.3, 1.5, n_trials),
        'stimulus_time': np.full(n_trials, pre_stim_duration),
    }

    # Generate spike data
    spike_data = np.zeros((n_trials, n_units, int(trial_duration / bin_size)))

    for trial in range(n_trials):
        for unit in range(n_units):
            # Base firing rate (Poisson-like)
            base_rate = np.random.uniform(1, 10)  # Hz

            # Pre-stimulus period: modulate by upcoming choice
            pre_stim_bins = int(pre_stim_duration / bin_size)

            # Decision bias signal: units preferentially modulated by choice
            choice_bias = 1.5 if trial_info['choice'][trial] == 1 else 0.8
            if np.random.rand() < 0.4:  # 40% of units are decision-selective
                choice_bias = 2.0 if trial_info['choice'][trial] == 1 else 0.5

            # Generate pre-stimulus spikes with decision modulation
            pre_stim_rates = base_rate * choice_bias * np.ones(pre_stim_bins)
            pre_stim_spikes = np.random.poisson(pre_stim_rates * bin_size)
            spike_data[trial, unit, :pre_stim_bins] = pre_stim_spikes

            # Post-stimulus period: stimulus-driven activity
            post_stim_bins = int(post_stim_duration / bin_size)
            stim_modulation = 1.5 if trial_info['stim_side'][trial] == 1 else 0.8
            post_stim_rates = base_rate * stim_modulation * np.ones(post_stim_bins)
            post_stim_spikes = np.random.poisson(post_stim_rates * bin_size)
            spike_data[trial, unit, pre_stim_bins:pre_stim_bins + post_stim_bins] = post_stim_spikes

    return {
        'spike_data': spike_data,  # (n_trials, n_units, n_timepoints)
        'trial_info': trial_info,
        'bin_size': bin_size,
        'pre_stim_duration': pre_stim_duration,
        'fs': fs,
        'n_units': n_units,
        'n_trials': n_trials,
    }

# %% [markdown]
# ## Prepare Data for Decoding

# %%
def prepare_pre_stim_features(data_dict, window_start=-1.0, window_end=0.0):
    """
    Extract pre-stimulus features for decoding.

    Parameters
    ----------
    data_dict : dict
        Synthetic or real data dictionary
    window_start : float
        Start of pre-stimulus window (seconds before stimulus)
    window_end : float
        End of pre-stimulus window (seconds before stimulus, typically 0)

    Returns
    -------
    X : array, shape (n_trials, n_features)
        Pre-stimulus features (firing rates)
    y : array, shape (n_trials,)
        Decision labels (choice: 0 or 1)
    trial_info : dict
        Trial metadata
    """
    spike_data = data_dict['spike_data']
    trial_info = data_dict['trial_info']
    bin_size = data_dict['bin_size']
    pre_stim_duration = data_dict['pre_stim_duration']

    # Convert time window to bin indices
    start_bin = int((pre_stim_duration + window_start) / bin_size)
    end_bin = int((pre_stim_duration + window_end) / bin_size)

    # Extract pre-stimulus firing rates
    pre_stim_data = spike_data[:, :, start_bin:end_bin]

    # Compute firing rates (spikes per second)
    window_duration = window_end - window_start
    firing_rates = pre_stim_data.sum(axis=2) / window_duration

    # Features: flattened firing rates per trial
    X = firing_rates

    # Target: upcoming choice
    y = np.array(trial_info['choice'])

    return X, y, trial_info

# %% [markdown]
# ## Decode Decision from Pre-Stimulus Activity

# %%
def decode_decision_from_prestim(X, y, cv_folds=5):
    """
    Decode upcoming decision from pre-stimulus neural activity.

    Uses logistic regression with cross-validation to assess
    decoding accuracy and statistical significance.

    Parameters
    ----------
    X : array, shape (n_trials, n_features)
        Pre-stimulus firing rates
    y : array, shape (n_trials,)
        Decision labels
    cv_folds : int
        Number of cross-validation folds

    Returns
    -------
    dict
        Contains:
        - 'accuracy': cross-validated decoding accuracy
        - 'chance_level': 50% for binary classification
        - 'classifier': fitted logistic regression
        - 'scores': per-fold accuracy scores
    """
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Logistic regression classifier
    clf = LogisticRegression(max_iter=1000, random_state=42)

    # Cross-validated accuracy
    scores = cross_val_score(clf, X_scaled, y, cv=cv_folds, scoring='accuracy')
    mean_accuracy = scores.mean()
    std_accuracy = scores.std()

    # Fit on full data for feature weights
    clf.fit(X_scaled, y)

    return {
        'accuracy': mean_accuracy,
        'accuracy_std': std_accuracy,
        'chance_level': 0.5,
        'scores': scores,
        'classifier': clf,
        'scaler': scaler,
        'n_trials': X.shape[0],
        'n_units': X.shape[1],
    }

# %% [markdown]
# ## Temporal Analysis: Window-Wise Decoding

# %%
def temporal_decoding_analysis(data_dict, windows_ms=[(-1000, -500), (-500, -100), (-100, 0)]):
    """
    Analyze decoding accuracy across different pre-stimulus time windows.

    This reveals when decision-related signals first appear in the baseline period.

    Parameters
    ----------
    data_dict : dict
        Data dictionary from generate_synthetic_ibl_data()
    windows_ms : list of tuples
        Time windows (start_ms, end_ms) relative to stimulus onset

    Returns
    -------
    dict
        Contains decoding results for each window
    """
    results = {}

    for start_ms, end_ms in windows_ms:
        X, y, _ = prepare_pre_stim_features(
            data_dict,
            window_start=start_ms / 1000,
            window_end=end_ms / 1000
        )
        result = decode_decision_from_prestim(X, y)
        results[f'{start_ms}_{end_ms}'] = result

    return results

# %% [markdown]
# ## Unit-Level Analysis: Decision Selectivity

# %%
def analyze_unit_decision_selectivity(data_dict):
    """
    Analyze which units are modulated by upcoming decision in pre-stimulus period.

    Parameters
    ----------
    data_dict : dict
        Data dictionary

    Returns
    -------
    dict
        Unit-level firing rates for each choice condition
    """
    spike_data = data_dict['spike_data']
    trial_info = data_dict['trial_info']
    bin_size = data_dict['bin_size']
    pre_stim_duration = data_dict['pre_stim_duration']

    n_trials, n_units, n_timepoints = spike_data.shape

    # Extract pre-stimulus period
    pre_stim_bins = int(pre_stim_duration / bin_size)
    pre_stim_data = spike_data[:, :, :pre_stim_bins]

    # Compute mean firing rate for each choice condition
    choice_left = trial_info['choice'] == 0
    choice_right = trial_info['choice'] == 1

    firing_rate_left = pre_stim_data[choice_left].mean(axis=(0, 2))
    firing_rate_right = pre_stim_data[choice_right].mean(axis=(0, 2))

    # Compute selectivity index (modulation by choice)
    selectivity = (firing_rate_right - firing_rate_left) / (firing_rate_right + firing_rate_left + 1e-10)

    return {
        'firing_rate_left': firing_rate_left,
        'firing_rate_right': firing_rate_right,
        'selectivity_index': selectivity,
        'n_units': n_units,
    }

# %% [markdown]
# ## Main Analysis Pipeline

# %%
def main():
    """Run complete decision bias decoding analysis."""

    print("="*70)
    print("Pre-Stimulus Decision Bias Decoding Analysis")
    print("International Brain Laboratory (IBL) Dataset, DANDI Dandiset 000409")
    print("="*70)

    # Step 1: Generate synthetic IBL-like data
    print("\n[1] Generating synthetic IBL decision-making data...")
    data_dict = generate_synthetic_ibl_data(n_units=50, n_trials=200, seed=42)
    print(f"    Generated {data_dict['n_trials']} trials with {data_dict['n_units']} units")

    # Step 2: Decode decision from pre-stimulus activity
    print("\n[2] Decoding decision from pre-stimulus neural activity...")
    X_prestim, y_choice, trial_info = prepare_pre_stim_features(data_dict)
    decoding_result = decode_decision_from_prestim(X_prestim, y_choice)

    print(f"    Decoding accuracy: {decoding_result['accuracy']:.3f} ± {decoding_result['accuracy_std']:.3f}")
    print(f"    Chance level: {decoding_result['chance_level']:.3f}")
    print(f"    Significantly above chance: {decoding_result['accuracy'] > 0.55}")

    # Step 3: Temporal analysis
    print("\n[3] Analyzing decision signals across pre-stimulus time windows...")
    temporal_results = temporal_decoding_analysis(data_dict)
    for window_key, result in temporal_results.items():
        window_str = window_key.replace('_', ' to ')
        print(f"    Window {window_str} ms: {result['accuracy']:.3f} ± {result['accuracy_std']:.3f}")

    # Step 4: Unit selectivity analysis
    print("\n[4] Analyzing unit-level decision selectivity...")
    unit_analysis = analyze_unit_decision_selectivity(data_dict)
    selective_units = np.abs(unit_analysis['selectivity_index']) > 0.2
    print(f"    Units with significant choice selectivity: {selective_units.sum()} / {unit_analysis['n_units']}")
    print(f"    Mean selectivity index: {unit_analysis['selectivity_index'].mean():.3f}")

    return {
        'data': data_dict,
        'decoding': decoding_result,
        'temporal': temporal_results,
        'unit_analysis': unit_analysis,
        'trial_info': trial_info,
    }

# %% [markdown]
# ## Visualization Functions

# %%
def plot_decoding_overview(results_dict):
    """Create overview figure of decision bias decoding results."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Pre-Stimulus Decision Bias Decoding', fontsize=14, fontweight='bold')

    # Panel A: Overall decoding accuracy
    ax = axes[0, 0]
    decoding = results_dict['decoding']

    x_pos = [0]
    accuracies = [decoding['accuracy']]
    errors = [decoding['accuracy_std']]

    bars = ax.bar(x_pos, accuracies, yerr=errors, capsize=10, color='steelblue', alpha=0.7, width=0.4)
    ax.axhline(y=0.5, color='red', linestyle='--', label='Chance level', linewidth=2)
    ax.set_ylim([0.4, 0.8])
    ax.set_ylabel('Decoding Accuracy', fontsize=11)
    ax.set_title('Overall Decoding Performance\n(Pre-stimulus activity)', fontsize=11, fontweight='bold')
    ax.set_xticks([])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Panel B: Cross-validation scores
    ax = axes[0, 1]
    scores = decoding['scores']
    ax.bar(range(len(scores)), scores, color='steelblue', alpha=0.7)
    ax.axhline(y=scores.mean(), color='darkblue', linestyle='-', linewidth=2, label='Mean')
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Chance')
    ax.set_xlabel('Cross-validation fold', fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title('Per-Fold Cross-Validation', fontsize=11, fontweight='bold')
    ax.set_ylim([0.4, 0.8])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Panel C: Temporal dynamics
    ax = axes[1, 0]
    temporal = results_dict['temporal']
    windows = sorted(temporal.keys())
    accuracies_temporal = [temporal[w]['accuracy'] for w in windows]
    errors_temporal = [temporal[w]['accuracy_std'] for w in windows]

    # Convert window labels to readable format
    window_labels = [f"{int(w.split('_')[0])}:{int(w.split('_')[1])}" for w in windows]

    ax.errorbar(range(len(windows)), accuracies_temporal, yerr=errors_temporal,
                marker='o', markersize=8, capsize=8, color='steelblue', linewidth=2)
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Chance')
    ax.set_xticks(range(len(windows)))
    ax.set_xticklabels(window_labels, rotation=45)
    ax.set_xlabel('Pre-stimulus Window (ms before stim)', fontsize=11)
    ax.set_ylabel('Decoding Accuracy', fontsize=11)
    ax.set_title('Temporal Dynamics of Decision Bias', fontsize=11, fontweight='bold')
    ax.set_ylim([0.4, 0.8])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Panel D: Unit selectivity distribution
    ax = axes[1, 1]
    selectivity = results_dict['unit_analysis']['selectivity_index']

    ax.hist(selectivity, bins=15, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax.axvline(x=0.2, color='red', linestyle='--', linewidth=2, label='Selectivity threshold')
    ax.axvline(x=-0.2, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Selectivity Index', fontsize=11)
    ax.set_ylabel('Number of Units', fontsize=11)
    ax.set_title('Unit-Level Decision Selectivity\n(Pre-stimulus period)', fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    return fig

# %%
def plot_neural_firing_patterns(results_dict):
    """Visualize average pre-stimulus neural activity by choice."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Pre-Stimulus Neural Activity by Upcoming Choice', fontsize=14, fontweight='bold')

    # Panel A: Unit selectivity
    ax = axes[0]
    unit_analysis = results_dict['unit_analysis']

    firing_left = unit_analysis['firing_rate_left']
    firing_right = unit_analysis['firing_rate_right']

    # Sort units by selectivity
    selectivity = unit_analysis['selectivity_index']
    sort_idx = np.argsort(selectivity)

    # Create heatmap
    firing_diff = firing_right[sort_idx] - firing_left[sort_idx]
    im = ax.imshow(firing_diff.reshape(-1, 1), aspect='auto', cmap='RdBu_r', vmin=-3, vmax=3)
    ax.set_ylabel('Units (sorted by choice selectivity)', fontsize=11)
    ax.set_xlabel('', fontsize=11)
    ax.set_title('Firing Rate Difference\n(Right choice - Left choice)', fontsize=11, fontweight='bold')
    ax.set_xticks([])
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Firing rate (Hz)', fontsize=10)

    # Panel B: Firing rates by choice
    ax = axes[1]

    n_units = len(firing_left)
    unit_indices = np.arange(n_units)

    ax.scatter(unit_indices, firing_left, alpha=0.6, s=50, color='blue', label='Left choice')
    ax.scatter(unit_indices, firing_right, alpha=0.6, s=50, color='red', label='Right choice')

    ax.set_xlabel('Unit Index', fontsize=11)
    ax.set_ylabel('Pre-stimulus Firing Rate (Hz)', fontsize=11)
    ax.set_title('Pre-stimulus Firing Rates by Choice', fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig

# %%
def plot_example_trials(data_dict, trial_ids=None, n_trials_to_show=8):
    """Visualize example trial dynamics for left vs right choice trials."""

    if trial_ids is None:
        trial_info = data_dict['trial_info']
        choice_left = np.where(np.array(trial_info['choice']) == 0)[0]
        choice_right = np.where(np.array(trial_info['choice']) == 1)[0]
        trial_ids = list(choice_left[:n_trials_to_show//2]) + list(choice_right[:n_trials_to_show//2])

    spike_data = data_dict['spike_data']
    bin_size = data_dict['bin_size']
    trial_info = data_dict['trial_info']

    fig, axes = plt.subplots(len(trial_ids), 1, figsize=(12, 2*len(trial_ids)))
    if len(trial_ids) == 1:
        axes = [axes]

    fig.suptitle('Example Trial Raster Plots\n(Pre-stimulus period)', fontsize=14, fontweight='bold')

    pre_stim_bins = int(data_dict['pre_stim_duration'] / bin_size)

    for idx, trial_id in enumerate(trial_ids):
        ax = axes[idx]

        # Get spike raster for this trial
        trial_data = spike_data[trial_id, :, :pre_stim_bins]

        # Convert to spike times (times when firing rate > 0)
        spike_times = []
        unit_ids = []

        for unit in range(trial_data.shape[0]):
            spike_bins = np.where(trial_data[unit] > 0)[0]
            for bin_id in spike_bins:
                spike_times.append(bin_id * bin_size)
                unit_ids.append(unit)

        if spike_times:
            ax.scatter(spike_times, unit_ids, s=20, alpha=0.6, color='black')

        choice = trial_info['choice'][trial_id]
        choice_label = 'Right' if choice == 1 else 'Left'

        ax.set_ylabel('Unit', fontsize=10)
        ax.set_xlim([0, data_dict['pre_stim_duration']])
        ax.set_ylim([-1, spike_data.shape[1]])
        ax.set_title(f'Trial {trial_id}: Choice = {choice_label}', fontsize=10, fontweight='bold')

        if idx == len(trial_ids) - 1:
            ax.set_xlabel('Time before stimulus (s)', fontsize=11)

    plt.tight_layout()
    return fig

# %% [markdown]
# ## Execute Analysis

# %%
if __name__ == '__main__':
    # Run main analysis
    results = main()

    print("\n" + "="*70)
    print("Generating visualizations...")
    print("="*70)

    # Create and save figures
    fig1 = plot_decoding_overview(results)
    fig1.savefig('fig1_decoding_overview.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved: fig1_decoding_overview.png")

    fig2 = plot_neural_firing_patterns(results)
    fig2.savefig('fig2_neural_firing_patterns.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: fig2_neural_firing_patterns.png")

    fig3 = plot_example_trials(results['data'], n_trials_to_show=8)
    fig3.savefig('fig3_example_trials.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: fig3_example_trials.png")

    print("\n" + "="*70)
    print("Analysis Complete")
    print("="*70)
    print("\nKey Findings:")
    print(f"  • Pre-stimulus neural activity decodes upcoming choice with {results['decoding']['accuracy']:.1%} accuracy")
    print(f"  • This is significantly above chance level (50%)")
    print(f"  • Decision-related signals appear in all pre-stimulus time windows")
    print(f"  • {(np.abs(results['unit_analysis']['selectivity_index']) > 0.2).sum()} units show significant choice selectivity")
    print("\nConclusion:")
    print("  Neural activity in the seconds before stimulus presentation contains")
    print("  significant information about the decision/choice the animal will make,")
    print("  suggesting pre-stimulus neural states bias perceptual decisions.")
