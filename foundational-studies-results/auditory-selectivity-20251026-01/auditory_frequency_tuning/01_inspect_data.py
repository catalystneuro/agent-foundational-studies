# %% [markdown]
# # Inspect Auditory Cortex NWB File
# 
# This script loads and inspects the structure of an NWB file containing auditory cortex
# recordings during pure tone presentations.

# %%
import pynwb
import lindi
import pynapple as nap
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# ## Load NWB File with LINDI
# Using LINDI with local caching for efficient streaming access

# %%
# Create local cache
local_cache = lindi.LocalCache()

# Load the NWB file
lindi_url = "https://lindi.neurosift.org/dandi/dandisets/001419/assets/b651a329-009d-4b85-b08c-ce99de014c18/nwb.lindi.json"
f = lindi.LindiH5pyFile.from_lindi_file(lindi_url, local_cache=local_cache)

io = pynwb.NWBHDF5IO(file=f, mode='r')
nwbfile = io.read()

# %% [markdown]
# ## Inspect NWB File Metadata

# %%
print("=" * 60)
print("NWB FILE METADATA")
print("=" * 60)
print(f"Session description: {nwbfile.session_description}")
print(f"Identifier: {nwbfile.identifier}")
print(f"Session start time: {nwbfile.session_start_time}")
print(f"Subject: {nwbfile.subject.subject_id}")
print(f"Species: {nwbfile.subject.species}")
print(f"Age: {nwbfile.subject.age}")
print(f"Sex: {nwbfile.subject.sex}")
print(f"Description: {nwbfile.subject.description}")

# %% [markdown]
# ## Inspect Trial Structure

# %%
print("\n" + "=" * 60)
print("TRIAL INFORMATION")
print("=" * 60)

trials = nwbfile.intervals['trials']
print(f"Number of trials: {len(trials)}")
print(f"Trial columns: {trials.colnames}")

# Extract trial data
trial_freqs = trials['frequency'][:]
trial_intensities = trials['intensity'][:]
trial_start_times = trials['start_time'][:]
trial_stop_times = trials['stop_time'][:]

print(f"\nUnique frequencies (Hz): {np.unique(trial_freqs)}")
print(f"Unique intensities (dB): {np.unique(trial_intensities)}")
print(f"Number of trials per frequency: {len(trial_freqs) // len(np.unique(trial_freqs))}")

# %% [markdown]
# ## Inspect Units (Neurons)

# %%
print("\n" + "=" * 60)
print("UNITS (NEURONS)")
print("=" * 60)

units = nwbfile.units
print(f"Number of units: {len(units)}")
print(f"Unit columns: {units.colnames}")

# Check unit quality
unit_quality = units['quality'][:]
unique_quality, counts = np.unique(unit_quality, return_counts=True)
print(f"\nUnit quality distribution:")
for qual, count in zip(unique_quality, counts):
    print(f"  {qual}: {count} units")

# Sample spike times from first unit
print(f"\nFirst unit spike times (first 10): {units['spike_times'][0][:10]}")
print(f"Total spikes in first unit: {len(units['spike_times'][0])}")

# %% [markdown]
# ## Visualize Frequency Distribution

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Plot frequency distribution
axes[0].hist(trial_freqs, bins=len(np.unique(trial_freqs)), edgecolor='black')
axes[0].set_xlabel('Frequency (Hz)')
axes[0].set_ylabel('Number of Trials')
axes[0].set_title('Distribution of Tone Frequencies')
axes[0].set_xscale('log')

# Plot intensity distribution
axes[1].hist(trial_intensities, bins=len(np.unique(trial_intensities)), edgecolor='black')
axes[1].set_xlabel('Intensity (dB)')
axes[1].set_ylabel('Number of Trials')
axes[1].set_title('Distribution of Tone Intensities')

plt.tight_layout()
plt.savefig('trial_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nFigure saved: trial_distribution.png")

# %% [markdown]
# ## Summary

# %%
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"This dataset contains {len(units)} neurons recorded from auditory cortex")
print(f"during {len(trials)} trials of pure tone stimuli.")
print(f"Tones span {len(np.unique(trial_freqs))} different frequencies")
print(f"from {np.min(trial_freqs):.0f} Hz to {np.max(trial_freqs):.0f} Hz.")
print(f"Data is suitable for frequency tuning analysis.")
