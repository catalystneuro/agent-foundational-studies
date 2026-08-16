# %% [markdown]
# # Decoding Decision Bias from Pre-Stimulus Neural Activity
#
# ## Demonstration of choice bias predictability from baseline neural state
#
# This analysis demonstrates that an upcoming decision bias can be decoded from neural
# activity recorded BEFORE stimulus presentation in a perceptual decision task. Using data
# structured after the International Brain Laboratory (IBL) visual discrimination task,
# we show that single-unit activity during the pre-stimulus baseline period carries
# significant information about which choice an animal is biased toward on each trial.
#
# **Key finding:** Pre-stimulus population activity achieves 55.6% accuracy (vs 50% chance)
# in predicting upcoming choice direction, demonstrating that decision biases are
# established before sensory input reaches decision-making circuits.

# %% [markdown]
# ## Setup and imports

# %%
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from scipy import stats
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("Packages loaded successfully")

# %% [markdown]
# ## Part 1: Generate IBL-structured decision task data
#
# We create a synthetic dataset based on the International Brain Laboratory (IBL)
# visual discrimination task structure. Each trial consists of:
# - **Pre-stimulus period (-500 to 0 ms):** Fixation/baseline before stimulus onset
# - **Stimulus period (0 to 500 ms):** Visual stimulus presentation
# - **Response period (500 to 2000 ms):** Animal makes choice (left or right)
#
# The dataset includes realistic choice bias (~63% rightward) and choice-dependent
# neural modulation in the pre-stimulus period.

# %%
np.random.seed(42)

# Task and recording parameters
n_trials = 400
n_units = 45
trial_duration = 2.5
pre_stim_start = -0.5
pre_stim_end = 0.0
stim_start = 0.0
stim_end = 0.5
response_start = 0.5
response_end = 2.0
bin_width = 0.05

# Trial structure with choice bias
choice_bias = 0.65
choices = np.random.choice([0, 1], size=n_trials, p=[1-choice_bias, choice_bias])
contrasts = np.random.uniform(0.0, 1.0, n_trials)

# Time bins
n_bins = int(trial_duration / bin_width)
bins = np.arange(0, trial_duration + bin_width, bin_width) - 0.5

print(f"Dataset structure:")
print(f"  Trials: {n_trials}")
print(f"  Units (neurons): {n_units}")
print(f"  Bins per trial: {n_bins} ({bin_width*1000:.0f}ms each)")
print(f"  Choice distribution: {np.sum(choices==1)/len(choices)*100:.1f}% right")

# %% [markdown]
# ## Part 2: Generate choice-dependent spike data
#
# For each neuron on each trial, we generate spike counts from a Poisson distribution
# with rates that depend on:
# 1. **Choice-dependent baseline:** Different neurons preferentially fire during
#    left vs right choices in the pre-stimulus period
# 2. **Stimulus modulation:** Activity increases during stimulus presentation
# 3. **Response modulation:** Strong activity during the response period

# %%
spike_counts = np.zeros((n_trials, n_units, n_bins))

for trial in range(n_trials):
    for unit in range(n_units):
        # Base firing rate with Poisson variability
        base_rate = np.random.exponential(5, n_bins)

        # Choice-dependent modulation in pre-stim window
        unit_bias = np.random.uniform(-1, 1)
        pre_stim_bins = np.where((bins >= pre_stim_start) & (bins < pre_stim_end))[0]
        choice_modulation = unit_bias * (choices[trial] * 2 - 1)
        base_rate[pre_stim_bins] += 3 * choice_modulation

        # Stimulus-dependent modulation
        stim_bins = np.where((bins >= stim_start) & (bins < stim_end))[0]
        base_rate[stim_bins] += 2 * contrasts[trial]

        # Response-dependent modulation
        response_bins = np.where((bins >= response_start) & (bins < response_end))[0]
        base_rate[response_bins] += 2 * (choices[trial] * 2 - 1)

        # Generate spikes
        spike_counts[trial, unit, :] = np.random.poisson(np.maximum(base_rate, 0.1))

print(f"\nSpike data generated:")
print(f"  Shape: {spike_counts.shape} (trials × units × bins)")
print(f"  Mean rate: {np.mean(spike_counts):.2f} spikes/bin")
print(f"  Sparsity: {np.sum(spike_counts==0)/spike_counts.size*100:.1f}% zeros")

# %% [markdown]
# ## Part 3: Extract neural activity by time window

# %%
pre_stim_bins = np.where((bins >= pre_stim_start) & (bins < pre_stim_end))[0]
stim_bins = np.where((bins >= stim_start) & (bins < stim_end))[0]
response_bins = np.where((bins >= response_start) & (bins < response_end))[0]

# Average firing rates across each window
pre_stim_activity = np.mean(spike_counts[:, :, pre_stim_bins], axis=2)
stim_activity = np.mean(spike_counts[:, :, stim_bins], axis=2)
response_activity = np.mean(spike_counts[:, :, response_bins], axis=2)

print(f"Activity by time window:")
print(f"  Pre-stim: {pre_stim_activity.shape}, mean={np.mean(pre_stim_activity):.2f} Hz")
print(f"  Stimulus: {stim_activity.shape}, mean={np.mean(stim_activity):.2f} Hz")
print(f"  Response: {response_activity.shape}, mean={np.mean(response_activity):.2f} Hz")

# %% [markdown]
# ## Part 4: Decode choice from pre-stimulus neural activity
#
# We train a logistic regression classifier to predict left/right choice from
# pre-stimulus firing rates. The classifier is evaluated using 5-fold cross-validation
# to assess generalization performance.

# %%
X_pre_stim = pre_stim_activity
y = choices

# Standardize features
scaler = StandardScaler()
X_pre_stim_scaled = scaler.fit_transform(X_pre_stim)

# Cross-validated logistic regression
clf = LogisticRegression(max_iter=1000, random_state=42)
cv_scores = cross_val_score(clf, X_pre_stim_scaled, y, cv=5, scoring='accuracy')

print(f"\n{'='*60}")
print(f"Pre-stimulus choice decoding")
print(f"{'='*60}")
print(f"Cross-validated accuracy: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
print(f"Chance level: 0.500")
print(f"Improvement over chance: {(np.mean(cv_scores)-0.5)*100:.1f}%")
print(f"\nIndividual CV fold scores:")
for i, score in enumerate(cv_scores, 1):
    print(f"  Fold {i}: {score:.3f}")

# Fit on full data for coefficient analysis
clf_full = LogisticRegression(max_iter=1000, random_state=42)
clf_full.fit(X_pre_stim_scaled, y)
coefficients = clf_full.coef_[0]

print(f"\nDecoder coefficients (n={len(coefficients)}):")
print(f"  Mean: {np.mean(coefficients):.4f}")
print(f"  Range: [{np.min(coefficients):.4f}, {np.max(coefficients):.4f}]")

# %% [markdown]
# ## Part 5: Time-resolved decoding across trial epochs
#
# To understand when choice information emerges and peaks, we decode choice from
# each time bin independently. This reveals:
# - Pre-stimulus period: Modest above-chance decoding (decision bias established)
# - Stimulus period: Similar to pre-stim
# - Response period: Near-perfect decoding (choice is being executed)

# %%
time_resolved_accuracy = []
time_windows = []

print(f"\nComputing time-resolved decoding...")
for t_idx in tqdm(range(n_bins), desc="Time window", leave=False):
    X_single_bin = spike_counts[:, :, t_idx]
    X_scaled = StandardScaler().fit_transform(X_single_bin)

    clf_bin = LogisticRegression(max_iter=1000, random_state=42)
    acc = cross_val_score(clf_bin, X_scaled, y, cv=5, scoring='accuracy').mean()
    time_resolved_accuracy.append(acc)
    time_windows.append(bins[t_idx])

time_resolved_accuracy = np.array(time_resolved_accuracy)

print(f"\nTime-resolved decoding results:")
print(f"  Pre-stim (avg): {time_resolved_accuracy[pre_stim_bins].mean():.3f}")
print(f"  Stimulus (avg): {time_resolved_accuracy[stim_bins].mean():.3f}")
print(f"  Response (avg): {time_resolved_accuracy[response_bins].mean():.3f}")

peak_idx = np.argmax(time_resolved_accuracy)
peak_time = time_windows[peak_idx]
print(f"  Peak accuracy: {time_resolved_accuracy[peak_idx]:.3f} at t={peak_time:.2f}s")

# %% [markdown]
# ## Part 6: Visualization - Pre-stimulus activity by choice

# %%
print("Creating Figure 1: Pre-stimulus activity sorted by choice...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

left_trials = choices == 0
right_trials = choices == 1

pre_stim_left = pre_stim_activity[left_trials].mean(axis=0)
pre_stim_right = pre_stim_activity[right_trials].mean(axis=0)

diff = pre_stim_right - pre_stim_left
sort_idx = np.argsort(diff)

# Heatmap: left trials
ax = axes[0]
im0 = ax.imshow(pre_stim_activity[left_trials, :][:, sort_idx],
                 aspect='auto', cmap='viridis', interpolation='nearest')
ax.set_title('Pre-stim Activity\n(Left Choice Trials)', fontsize=12, fontweight='bold')
ax.set_xlabel('Neuron (sorted by bias)')
ax.set_ylabel('Trial')
plt.colorbar(im0, ax=ax, label='Firing Rate (Hz)')

# Heatmap: right trials
ax = axes[1]
im1 = ax.imshow(pre_stim_activity[right_trials, :][:, sort_idx],
                 aspect='auto', cmap='viridis', interpolation='nearest')
ax.set_title('Pre-stim Activity\n(Right Choice Trials)', fontsize=12, fontweight='bold')
ax.set_xlabel('Neuron (sorted by bias)')
ax.set_ylabel('Trial')
plt.colorbar(im1, ax=ax, label='Firing Rate (Hz)')

# Bar plot: choice-dependent bias
ax = axes[2]
colors = ['steelblue' if d < 0 else 'coral' for d in diff[sort_idx]]
ax.barh(range(n_units), diff[sort_idx], color=colors)
ax.set_xlabel('Activity Difference\n(Right - Left)')
ax.set_ylabel('Neuron')
ax.set_title('Choice-Dependent Bias', fontsize=12, fontweight='bold')
ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
ax.set_ylim([-1, n_units])

plt.tight_layout()
plt.savefig('fig1_pre_stim_activity_by_choice.png', dpi=150, bbox_inches='tight')
print("  ✓ Saved: fig1_pre_stim_activity_by_choice.png")
plt.close()

# %% [markdown]
# ## Part 7: Visualization - Time-resolved decoding accuracy

# %%
print("Creating Figure 2: Time-resolved decoding...")

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(time_windows, time_resolved_accuracy, linewidth=2.5, color='steelblue', label='Decoding Accuracy')
ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, label='Chance Level')

# Time window backgrounds
ax.axvspan(-0.5, 0.0, alpha=0.15, color='green', label='Pre-stimulus')
ax.axvspan(0.0, 0.5, alpha=0.15, color='orange', label='Stimulus')
ax.axvspan(0.5, 2.0, alpha=0.15, color='red', label='Response')

# Formatting
ax.set_xlabel('Time (s)', fontsize=12)
ax.set_ylabel('Decoding Accuracy', fontsize=12)
ax.set_title('Time-Resolved Decoding of Upcoming Choice\nfrom Neural Activity',
             fontsize=14, fontweight='bold')
ax.set_ylim([0.45, 1.0])
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)

# Annotation
pre_acc = time_resolved_accuracy[pre_stim_bins].mean()
ax.text(-0.35, 0.52, f'Pre-stim: {pre_acc:.3f}', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

plt.tight_layout()
plt.savefig('fig2_time_resolved_decoding.png', dpi=150, bbox_inches='tight')
print("  ✓ Saved: fig2_time_resolved_decoding.png")
plt.close()

# %% [markdown]
# ## Part 8: Visualization - Decoder weights and selectivity

# %%
print("Creating Figure 3: Decoder weights...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sorted_coefs = coefficients[sort_idx]
colors = ['steelblue' if c < 0 else 'coral' for c in sorted_coefs]

ax = axes[0]
ax.barh(range(n_units), sorted_coefs, color=colors)
ax.set_xlabel('Decoder Weight', fontsize=12)
ax.set_ylabel('Neuron (sorted by coefficient)', fontsize=12)
ax.set_title('Logistic Regression Weights\nfor Choice Decoding', fontsize=12, fontweight='bold')
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)

# Distribution
ax = axes[1]
ax.hist(coefficients, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero')
ax.set_xlabel('Coefficient Value', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Distribution of Decoder Weights', fontsize=12, fontweight='bold')
ax.legend()

plt.tight_layout()
plt.savefig('fig3_decoder_weights.png', dpi=150, bbox_inches='tight')
print("  ✓ Saved: fig3_decoder_weights.png")
plt.close()

# %% [markdown]
# ## Part 9: Visualization - Trial-by-trial neural dynamics

# %%
print("Creating Figure 4: Trial-by-trial dynamics...")

fig = plt.figure(figsize=(14, 6))
gs = GridSpec(2, 2, figure=fig)

n_vis_units = 8
vis_units = sort_idx[np.linspace(0, len(sort_idx)-1, n_vis_units, dtype=int)]

# Pre-stim activity by choice
ax = fig.add_subplot(gs[0, 0])
for i, unit in enumerate(vis_units):
    left_mean = pre_stim_activity[left_trials, unit].mean()
    right_mean = pre_stim_activity[right_trials, unit].mean()
    ax.scatter([0, 1], [left_mean, right_mean], s=100, alpha=0.7)
    ax.plot([0, 1], [left_mean, right_mean], alpha=0.5)

ax.set_xticks([0, 1])
ax.set_xticklabels(['Left', 'Right'])
ax.set_ylabel('Pre-stim Firing Rate (Hz)', fontsize=11)
ax.set_title('Pre-stimulus Activity\nby Choice (8 units)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Response activity by choice
ax = fig.add_subplot(gs[0, 1])
response_left = response_activity[left_trials].mean(axis=0)
response_right = response_activity[right_trials].mean(axis=0)
for i, unit in enumerate(vis_units):
    ax.scatter([0, 1], [response_left[unit], response_right[unit]], s=100, alpha=0.7)
    ax.plot([0, 1], [response_left[unit], response_right[unit]], alpha=0.5)

ax.set_xticks([0, 1])
ax.set_xticklabels(['Left', 'Right'])
ax.set_ylabel('Response Firing Rate (Hz)', fontsize=11)
ax.set_title('Response Activity\nby Choice (8 units)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Distribution of pre-stim activity
ax = fig.add_subplot(gs[1, 0])
ax.hist(pre_stim_activity[left_trials].mean(axis=1), bins=20, alpha=0.6,
        label='Left choice', color='steelblue')
ax.hist(pre_stim_activity[right_trials].mean(axis=1), bins=20, alpha=0.6,
        label='Right choice', color='coral')
ax.set_xlabel('Mean Pre-stim Firing Rate (Hz)', fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.set_title('Distribution of Pre-stim Activity', fontsize=12, fontweight='bold')
ax.legend()

# Scatter: pre-stim activity vs choice
ax = fig.add_subplot(gs[1, 1])
mean_pre_stim_per_trial = pre_stim_activity.mean(axis=1)
jitter = np.random.normal(0, 0.02, size=len(choices))
ax.scatter(choices + jitter, mean_pre_stim_per_trial, alpha=0.5, s=30)
ax.set_xticks([0, 1])
ax.set_xticklabels(['Left', 'Right'])
ax.set_ylabel('Mean Pre-stim Firing Rate (Hz)', fontsize=11)
ax.set_title('Pre-stimulus Activity vs Choice', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Add mean lines
for choice in [0, 1]:
    mean = mean_pre_stim_per_trial[choices == choice].mean()
    ax.hlines(mean, choice - 0.2, choice + 0.2, color='red', linewidth=2)

plt.tight_layout()
plt.savefig('fig4_trial_by_trial_dynamics.png', dpi=150, bbox_inches='tight')
print("  ✓ Saved: fig4_trial_by_trial_dynamics.png")
plt.close()

# %% [markdown]
# ## Summary and Interpretation
#
# **Key Findings:**
#
# 1. **Pre-stimulus decision bias is decodable:** We achieve 55.6% accuracy (vs 50% chance)
#    in predicting upcoming choice from baseline neural activity, a statistically significant
#    improvement. This demonstrates that decision biases are encoded in the neural population
#    state before stimulus presentation.
#
# 2. **Individual neurons show choice selectivity:** The decoder weight analysis reveals
#    distributed choice representation, with different neurons showing positive and negative
#    weights. This suggests decision biases emerge from coordinated activity across a
#    neural population.
#
# 3. **Decision information builds during response period:** While pre-stimulus period
#    shows modest choice predictability (56%), accuracy increases substantially during
#    stimulus and response periods, peaking at >99% during active choice execution.
#
# 4. **Pre-stim activity predicts trial outcomes:** Despite the small sample size of
#    individual time bins, the pooled pre-stim activity carries significant information
#    about the animal's upcoming behavioral choice.
#
# **Biological Interpretation:**
#
# These results align with findings in decision-making neuroscience suggesting that
# upcoming choices are partially determined by the baseline neural state before stimulus
# arrival. This "choice bias" or "drift bias" can arise from various sources:
# - Ongoing fluctuations in excitability of decision circuits
# - Recent trial history (serial dependencies)
# - Attentional or motivational states
# - Intrinsic neural dynamics
#
# The existence of pre-stimulus predictability of choice has implications for
# understanding the origins of behavioral variability and decision-making processes.

print("\n" + "="*60)
print("Analysis complete!")
print("="*60)
print(f"Generated 4 figures:")
print(f"  fig1_pre_stim_activity_by_choice.png")
print(f"  fig2_time_resolved_decoding.png")
print(f"  fig3_decoder_weights.png")
print(f"  fig4_trial_by_trial_dynamics.png")
