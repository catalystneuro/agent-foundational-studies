"""Figure: rasters, PSTHs and tuning curves for example auditory-cortex units."""
import pickle
import numpy as np
import pynapple as nap
import matplotlib.pyplot as plt

import loaders, pipeline986

SESSION = 'sub-LA9/sub-LA9_ses-1_behavior.nwb'
res = pickle.load(open('results_000986.pkl', 'rb'))[SESSION]
assets = loaders.list_assets('000986')
nwbfile = loaders.open_nwb('000986', assets[SESSION])
nwb = nap.NWBFile(nwbfile)
units = nwb['units']
trials = nwbfile.trials.to_dataframe()
onsets, freqs = trials.start_time.values, trials.stim_frequency.values
ufreq = res['ufreq']
cmap = plt.get_cmap('viridis')
fcol = {f: cmap(i / (len(ufreq) - 1)) for i, f in enumerate(ufreq)}

stats = res['stats']
tuned = stats[stats.tuned]
# one example per best frequency, choosing the most strongly driven unit at each
examples = [tuned[tuned.best_frequency == f].peak_evoked_hz.idxmax()
            for f in ufreq if (tuned.best_frequency == f).any()]

n = len(examples)
fig, axes = plt.subplots(n, 3, figsize=(13, 2.6 * n),
                         gridspec_kw=dict(width_ratios=[1.25, 1.1, 0.9], hspace=0.55, wspace=0.28))
RAST_N = 40   # trials drawn per frequency
for r, uid in enumerate(examples):
    axR, axP, axT = axes[r]
    y = 0
    for f in ufreq:
        sel = onsets[freqs == f][:RAST_N]
        pe = nap.compute_perievent(units[uid], nap.Ts(sel), window=(-0.05, 0.2))
        for k in range(len(sel)):
            st = pe[k].t
            axR.plot(st, np.full_like(st, y), '|', color=fcol[f], ms=3.5, mew=0.8)
            y += 1
        axR.axhline(y - 0.5, color='0.85', lw=0.5)
    axR.axvspan(0, 0.025, color='0.85', zorder=0, lw=0)
    axR.set_ylim(0, y); axR.set_xlim(-0.05, 0.2)
    axR.set_ylabel('trial (grouped\nby frequency)', fontsize=9)
    axR.set_title(f'unit {uid} — raster ({RAST_N} trials/freq)', fontsize=10)

    t = res['psth_t']
    for i, f in enumerate(ufreq):
        axP.plot(t, res['psths'][list(res['unit_ids']).index(uid), i], color=fcol[f], lw=1.4,
                 label=f'{f/1000:g} kHz')
    axP.axvspan(0, 0.025, color='0.85', zorder=0, lw=0)
    axP.set_xlim(-0.1, 0.3); axP.set_xlabel('time from tone onset (s)', fontsize=9)
    axP.set_ylabel('firing rate (Hz)', fontsize=9)
    axP.set_title('PSTH by tone frequency', fontsize=10)
    if r == 0:
        axP.legend(fontsize=8, frameon=False, ncol=2)

    c = res['curves'].loc[uid].values
    e = res['sems'].loc[uid].values
    axT.errorbar(ufreq / 1000, c, yerr=e, marker='o', color='k', lw=1.5, capsize=3)
    axT.axhline(0, color='0.6', lw=0.8, ls=':')
    bf = stats.loc[uid, 'best_frequency']
    axT.plot(bf / 1000, c[list(ufreq).index(bf)], 'o', ms=12, mfc='none', mec='crimson', mew=2)
    axT.set_xscale('log', base=2)
    axT.set_xticks(ufreq / 1000); axT.set_xticklabels([f'{f/1000:g}' for f in ufreq])
    axT.set_xlabel('tone frequency (kHz)', fontsize=9)
    axT.set_ylabel('evoked rate (Hz)\n(baseline subtracted)', fontsize=9)
    axT.set_title(f'tuning curve — BF {bf/1000:g} kHz, p={stats.loc[uid,"p_tuned"]:.1e}',
                  fontsize=10)
axes[-1][0].set_xlabel('time from tone onset (s)', fontsize=9)
fig.suptitle(f'DANDI:000986 {SESSION} — single-unit frequency tuning in mouse auditory cortex\n'
             'grey bar = 25 ms tone, 60 dB SPL', y=0.995, fontsize=12)
fig.savefig('figures/fig03_example_units.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('examples:', examples)
