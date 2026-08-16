# %% [markdown]
# # Explore single session: DANDI 000986 (auditory cortex Neuropixels)
#
# Quick prototyping on one session before scaling to multiple sessions.

# %%
import h5py
from pynwb import NWBHDF5IO
import remfile
import pynapple as nap
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("figures", exist_ok=True)

S3_URL = "https://dandiarchive.s3.amazonaws.com/blobs/660/dee/660deeed-2c8a-4910-9bcd-cd9d87911e3f"

disk_cache = remfile.DiskCache('/tmp/remfile_cache')
rem_file = remfile.File(S3_URL, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()

# %%
# Pull stimulus structure
trials = nwbfile.trials
stim_freq = trials['stim_frequency'][:]
trial_start = trials['start_time'][:]
trial_stop = trials['stop_time'][:]

print(f"Sessions has {len(stim_freq)} trials, {len(np.unique(stim_freq))} unique frequencies.")
print(f"Frequencies: {np.unique(stim_freq)} Hz")

# %%
# Pull units via Pynapple
nwb = nap.NWBFile(nwbfile)
print(nwb)

# %%
units = nwb['units']
print(f"N units: {len(units)}")
print(f"Time support: {units.time_support}")

# Compute mean firing rates as quality check
mean_rates = np.array([len(units[u]) / (units.time_support.end[0] - units.time_support.start[0])
                       for u in units.index])
print(f"Mean firing rates: median={np.median(mean_rates):.2f} Hz, range=[{mean_rates.min():.2f}, {mean_rates.max():.2f}] Hz")

# %%
# Plot raw activity for first few units across one stim block
fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)

t0, t1 = trial_start[0] - 1, trial_start[20]  # First 20 trials
unit_ids = list(units.index)[:8]
for i, uid in enumerate(unit_ids):
    st = units[uid].restrict(nap.IntervalSet(t0, t1)).t
    axes[0].plot(st, np.full_like(st, i), '.', ms=2, color='C0')
axes[0].set_ylabel("Unit index")
axes[0].set_title(f"Spikes for first {len(unit_ids)} units (trials 0-20)")

# Stim onsets in this window
in_win = (trial_start >= t0) & (trial_start <= t1)
for ts, fs in zip(trial_start[in_win], stim_freq[in_win]):
    axes[1].axvline(ts, color='red', alpha=0.4)
    axes[1].text(ts, 0.5, f"{int(fs/1000)}", fontsize=7, ha='center')
axes[1].set_ylabel("Stim onset")
axes[1].set_yticks([])

# Population firing rate
all_spikes = np.concatenate([units[u].t for u in units.index])
hist, bin_edges = np.histogram(all_spikes, bins=np.arange(t0, t1, 0.01))
axes[2].plot((bin_edges[:-1] + bin_edges[1:]) / 2, hist / 0.01 / len(units))
axes[2].set_ylabel("Pop rate (Hz/unit)")
axes[2].set_xlabel("Time (s)")

plt.tight_layout()
plt.savefig("figures/01_raw_activity.png", dpi=120)
plt.close()
print("Saved figures/01_raw_activity.png")
