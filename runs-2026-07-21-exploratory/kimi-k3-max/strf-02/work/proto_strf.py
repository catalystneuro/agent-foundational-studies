import numpy as np, lindi, pynapple as nap
from pynwb import NWBHDF5IO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

asset_id = "b8d3abca-0e78-4df1-9a51-d122a383be63"  # sub-LA11 ses-2
lindi_url = f"https://lindi.neurosift.org/dandi/dandisets/000986/assets/{asset_id}/nwb.lindi.json"
f = lindi.LindiH5pyFile.from_lindi_file(lindi_url, local_cache=lindi.LocalCache())
nwbfile = NWBHDF5IO(file=f).read()

# --- bulk-read spike times (memory gotcha: avoid ragged units['spike_times'][:])
sp = f['units/spike_times'][:]
idx = f['units/spike_times_index'][:]
starts = np.concatenate([[0], idx[:-1]])
unit_spikes = [sp[s:e] for s, e in zip(starts, idx)]
print('units:', len(unit_spikes), '| total spikes:', sum(len(u) for u in unit_spikes))

tr = nwbfile.trials.to_dataframe()
onsets = tr['start_time'].values
freqs = tr['stim_frequency'].values
freq_vals = np.array([2000., 4000., 8000., 16000., 32000.])

# --- STRF: per-unit, per-frequency rate in time bins relative to onset
bin_edges = np.arange(-0.05, 0.155, 0.005)  # -50..150 ms, 5 ms bins
bin_c = 0.5 * (bin_edges[:-1] + bin_edges[1:])
base_mask = bin_c < 0
resp_mask = bin_c >= 0

def unit_strf(spikes):
    # counts[ trial, timebin ]
    counts = np.zeros((len(onsets), len(bin_c)))
    for ti, t0 in enumerate(onsets):
        lo, hi = np.searchsorted(spikes, [t0 + bin_edges[0], t0 + bin_edges[-1]])
        c, _ = np.histogram(spikes[lo:hi] - t0, bins=bin_edges)
        counts[ti] = c
    strf = np.zeros((len(freq_vals), len(bin_c)))
    for fi, fv in enumerate(freq_vals):
        m = freqs == fv
        rate = counts[m].mean(axis=0) / 0.005      # Hz
        base = counts[m][:, base_mask].mean() / 0.005
        strf[fi] = rate - base                      # baseline-subtracted
    return strf

# pick a few units with strong responses for the validation figure
rates = np.array([len(u) / (onsets[-1] - onsets[0]) for u in unit_spikes])
print('rate range:', rates.min(), rates.max())
test_units = [int(np.argmax(rates)), 10, 50]
strfs = {}
for u in test_units:
    strfs[u] = unit_strf(unit_spikes[u])
    print('unit', u, 'peak |STRF|:', np.abs(strfs[u]).max())

fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
for ax, u in zip(axes, test_units):
    im = ax.imshow(strfs[u], aspect='auto', origin='lower', cmap='RdBu_r',
                   extent=[bin_c[0]*1000, bin_c[-1]*1000, -0.5, 4.5],
                   vmin=-np.abs(strfs[u]).max(), vmax=np.abs(strfs[u]).max())
    ax.set_yticks(range(5)); ax.set_yticklabels([f'{int(fv/1000)}' for fv in freq_vals])
    ax.axvline(0, color='k', lw=0.5); ax.axvline(25, color='k', lw=0.5, ls='--')
    ax.set_xlabel('time from onset (ms)'); ax.set_ylabel('frequency (kHz)')
    ax.set_title(f'unit {u}')
    fig.colorbar(im, ax=ax, label='rate - baseline (Hz)')
fig.savefig('proto_strf.png', dpi=150)
print('saved proto_strf.png')
