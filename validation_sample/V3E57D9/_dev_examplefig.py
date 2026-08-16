import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pynapple as nap
from _dev_tuning import load_session, FREQUENCIES, RESP_WIN, BASE_WIN

url = 'https://dandiarchive.s3.amazonaws.com/blobs/660/dee/660deeed-2c8a-4910-9bcd-cd9d87911e3f'
nwb, nwbfile = load_session(url)
units = nwb['units']
trials = nwb['trials'].as_dataframe()
trials['start'] = nwb['trials'].start

example_units = {2000.0: 74, 4000.0: 50, 8000.0: 96, 16000.0: 123, 32000.0: 98}

def psth(unit_id, freq, window=(-0.05, 0.15), bin_size=0.005):
    sub = trials[trials.stim_frequency == freq]
    events = nap.Ts(t=sub['start'].values)
    peri = nap.compute_perievent(units[unit_id], events, window=window)
    all_t = np.concatenate([np.asarray(peri[i].t) for i in peri.index]) if len(peri) > 0 else np.array([])
    bins = np.arange(window[0], window[1] + bin_size, bin_size)
    counts, edges = np.histogram(all_t, bins=bins)
    rate = counts / (len(peri) * bin_size)
    centers = (edges[:-1] + edges[1:]) / 2
    return centers, rate, peri

cmap = plt.get_cmap('viridis')
freq_colors = {f: cmap(i / (len(FREQUENCIES) - 1)) for i, f in enumerate(FREQUENCIES)}

fig, axes = plt.subplots(len(example_units), 2, figsize=(11, 3.0 * len(example_units)))

for row, (bf, uid) in enumerate(example_units.items()):
    ax_psth, ax_tuning = axes[row]
    tuning_vals = []
    for f in FREQUENCIES:
        centers, rate, peri = psth(uid, f)
        ax_psth.plot(centers * 1000, rate, color=freq_colors[f], label=f'{int(f/1000)} kHz', linewidth=1.5)
        # tuning value: mean evoked rate in response window minus baseline
        resp = peri.restrict(nap.IntervalSet(*RESP_WIN)).get_info('rate').values
        base = peri.restrict(nap.IntervalSet(*BASE_WIN)).get_info('rate').values
        tuning_vals.append(np.mean(resp) - np.mean(base))
    ax_psth.axvspan(0, 25, color='0.9', zorder=0, label='tone (25 ms)')
    ax_psth.set_xlabel('Time from tone onset (ms)')
    ax_psth.set_ylabel('Firing rate (Hz)')
    ax_psth.set_title(f'Unit {uid} (BF = {int(bf/1000)} kHz): PSTH')
    if row == 0:
        ax_psth.legend(fontsize=7, ncol=2, loc='upper right')

    xpos = np.arange(len(FREQUENCIES))
    ax_tuning.bar(xpos, tuning_vals, color=[freq_colors[f] for f in FREQUENCIES])
    ax_tuning.set_xticks(xpos)
    ax_tuning.set_xticklabels([f'{int(f/1000)}' for f in FREQUENCIES])
    ax_tuning.set_xlabel('Tone frequency (kHz)')
    ax_tuning.set_ylabel('Evoked rate (Hz)')
    ax_tuning.set_title(f'Unit {uid}: frequency tuning curve')

plt.tight_layout()
plt.savefig('figures/02_example_tuning_curves.png', dpi=150)
print('saved')
