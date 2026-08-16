"""Raw data validation: raster + PSTH for example VISp units aligned to
drifting grating onset, split by direction. Produces fig1_raw_data.png."""
import h5py
from pynwb import NWBHDF5IO
import remfile
import pynapple as nap
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DANDISET = "000021"
VERSION = "0.251116.2246"
ASSET_ID = "58703c97-c0a9-4736-b684-73c85c1a444a"
URL = f"https://api.dandiarchive.org/api/dandisets/{DANDISET}/versions/{VERSION}/assets/{ASSET_ID}/download/"

disk_cache = remfile.DiskCache('/tmp/remfile_cache_orientation')
rem_file = remfile.File(URL, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)

dg = nwbfile.intervals['drifting_gratings_presentations'].to_dataframe()
inv = nwb['invalid_times'].values
overlap = np.zeros(len(dg), dtype=bool)
for a, b in inv:
    overlap |= (dg['start_time'].values < b) & (dg['stop_time'].values > a)
dg = dg[~overlap].reset_index(drop=True)
dg_dir = dg.dropna(subset=['orientation'])

units = nwb['units']
meta = units.metadata.copy()
elec = nwbfile.electrodes.to_dataframe()
meta['structure'] = meta['peak_channel_id'].map(elec['location'])
visp_good = meta[(meta['quality'] == 'good') & (meta['structure'] == 'VISp')]

# Pick 3 example units with decent firing rates
candidates = visp_good[visp_good['firing_rate'] > 3].index[:3]
print("Example VISp units:", list(candidates))

directions = np.sort(dg_dir['orientation'].unique())
colors = plt.cm.hsv(np.linspace(0, 1, len(directions), endpoint=False))

fig, axes = plt.subplots(3, 2, figsize=(13, 11))
t_pre, t_post = 0.5, 2.0
bin_size = 0.02

for row, uid in enumerate(candidates):
    ts = units[uid]
    # Raster
    ax = axes[row, 0]
    y = 0
    yticks, ylabels = [], []
    for di, d in enumerate(directions):
        onsets = dg_dir.loc[dg_dir['orientation'] == d, 'start_time'].values
        peri = nap.compute_perievent(ts, nap.Ts(t=onsets), window=(-t_pre, t_post))
        for trial_spikes in peri.values():
            st = trial_spikes.t
            ax.plot(st, np.full_like(st, y), '|', color=colors[di],
                    markersize=3, alpha=0.8)
            y += 1
        yticks.append(y - len(onsets) / 2)
        ylabels.append(f"{int(d)}°")
        ax.axhline(y, color='k', lw=0.3, alpha=0.3)
        y += 2  # gap between direction blocks
    ax.axvline(0, color='k', ls='--', lw=1)
    ax.axvspan(0, 2.0, color='gray', alpha=0.12)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=7)
    ax.set_ylabel('Direction (trials grouped)')
    ax.set_xlabel('Time from grating onset (s)')
    ax.set_title(f'Unit {uid} (VISp) — spike raster', fontsize=10)
    ax.set_xlim(-t_pre, t_post + 0.3)

    # PSTH per direction
    ax = axes[row, 1]
    for di, d in enumerate(directions):
        onsets = dg_dir.loc[dg_dir['orientation'] == d, 'start_time'].values
        peri = nap.compute_perievent(ts, nap.Ts(t=onsets), window=(-t_pre, t_post))
        all_spikes = np.concatenate([p.t for p in peri.values()]) \
            if len(peri) else np.array([])
        bins = np.arange(-t_pre, t_post + bin_size, bin_size)
        counts, edges = np.histogram(all_spikes, bins=bins)
        rate = counts / (len(onsets) * bin_size)
        centers = (edges[:-1] + edges[1:]) / 2
        ax.plot(centers, rate, color=colors[di], lw=1.2, label=f"{int(d)}°")
    ax.axvline(0, color='k', ls='--', lw=1)
    ax.axvspan(0, 2.0, color='gray', alpha=0.12)
    ax.set_xlabel('Time from grating onset (s)')
    ax.set_ylabel('Firing rate (Hz)')
    ax.set_title(f'Unit {uid} — PSTH by direction', fontsize=10)
    ax.legend(fontsize=7, ncol=2, frameon=False)
    ax.set_xlim(-t_pre, t_post + 0.3)

plt.tight_layout()
plt.savefig('fig1_raw_data.png', dpi=150)
print("Saved fig1_raw_data.png")
