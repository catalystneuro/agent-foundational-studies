"""Does frequency tuning survive changes in arousal? Split trials by pre-tone pupil size.

DANDI:000986 was recorded specifically to relate auditory responses to arousal state, so
the pupil trace gives a natural robustness check on the tuning curves.
"""
import pickle
import numpy as np
import pandas as pd
import pynapple as nap
import matplotlib.pyplot as plt
from scipy import stats as sstats
from tqdm import tqdm

import analysis_core as ac
import loaders
from pipeline986 import WIN_EV, WIN_BL

res = pickle.load(open('results_000986.pkl', 'rb'))
assets = loaders.list_assets('000986')
# one session per mouse: the one with the most units
by_subj = {}
for s, r in res.items():
    subj = r['stats'].subject.iloc[0]
    if subj not in by_subj or len(r['stats']) > len(res[by_subj[subj]]['stats']):
        by_subj[subj] = s
sessions = [by_subj[k] for k in sorted(by_subj)]
print('sessions:', sessions)

rows, curves_lo, curves_hi = [], [], []
for s in tqdm(sessions, desc='arousal split'):
    r = res[s]
    nwbfile = loaders.open_nwb('000986', assets[s])
    nwb = nap.NWBFile(nwbfile)
    pupil = nwb['pupil_diameter']
    pt, pd_ = pupil.t, np.asarray(pupil)
    onsets, freqs, ufreq = r['onsets'], r['freqs'], r['ufreq']
    lo_i = np.searchsorted(pt, onsets - 0.5)
    hi_i = np.searchsorted(pt, onsets)
    pre_pupil = np.array([np.nanmean(pd_[a:b]) if b > a else np.nan
                          for a, b in zip(lo_i, hi_i)])
    ok = ~np.isnan(pre_pupil)
    med = np.nanmedian(pre_pupil)
    low, high = ok & (pre_pupil <= med), ok & (pre_pupil > med)

    d = r['counts_ev'] / (WIN_EV[1] - WIN_EV[0]) - r['counts_bl'] / (WIN_BL[1] - WIN_BL[0])
    keep = (r['stats'].tuned & r['stats'].responsive).values
    for mask, store in ((low, curves_lo), (high, curves_hi)):
        c = np.array([[d[mask & (freqs == f), j].mean() for f in ufreq]
                      for j in np.where(keep)[0]])
        store.append(c)
    base_lo = r['counts_bl'][low].mean(0) / (WIN_BL[1] - WIN_BL[0])
    base_hi = r['counts_bl'][high].mean(0) / (WIN_BL[1] - WIN_BL[0])
    rows.append(dict(session=s, subject=r['stats'].subject.iloc[0], n_units=int(keep.sum()),
                     pupil_median=float(med), n_low=int(low.sum()), n_high=int(high.sum()),
                     baseline_lo=float(base_lo[keep].mean()), baseline_hi=float(base_hi[keep].mean())))

Clo, Chi = np.concatenate(curves_lo), np.concatenate(curves_hi)
ufreq = res[sessions[0]]['ufreq']
# BF-aligned normalised curves in each arousal state, aligned and normalised by the
# all-trial (pooled) tuning curve so the comparison is not biased by picking the peak
# separately in each state.
Cpool = np.concatenate([res[s]['curves'].values[(res[s]['stats'].tuned &
                                                 res[s]['stats'].responsive).values]
                        for s in sessions])
bf_pool = Cpool.argmax(1)
big = Cpool.max(1) >= 1.0          # units with a pooled peak above 1 Hz
bf_lo, bf_hi = Clo.argmax(1), Chi.argmax(1)
agree = (bf_lo[big] == bf_hi[big]).mean()
peak_lo = Clo[np.arange(len(Clo)), bf_pool]
peak_hi = Chi[np.arange(len(Chi)), bf_pool]
offsets = np.arange(-(len(ufreq) - 1), len(ufreq))


def aligned(C):
    acc = {o: [] for o in offsets}
    for i in range(len(C)):
        if not big[i]:
            continue
        norm = C[i] / max(Cpool[i].max(), 1e-9)
        for j in range(len(ufreq)):
            acc[j - bf_pool[i]].append(norm[j])
    m = np.array([np.mean(acc[o]) if acc[o] else np.nan for o in offsets])
    se = np.array([np.std(acc[o]) / np.sqrt(len(acc[o])) if acc[o] else np.nan for o in offsets])
    return m, se


mlo, slo = aligned(Clo)
mhi, shi = aligned(Chi)
w = sstats.wilcoxon(peak_lo[big], peak_hi[big])

fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), gridspec_kw=dict(wspace=0.33))
ax = axes[0]
ax.errorbar(offsets, mlo, yerr=slo, marker='o', color='navy', lw=2, capsize=3, label='low arousal')
ax.errorbar(offsets, mhi, yerr=shi, marker='s', color='darkorange', lw=2, capsize=3,
            label='high arousal')
ax.axvline(0, color='0.6', ls=':'); ax.axhline(0, color='0.6', ls=':')
ax.set_xlabel('octaves from best frequency'); ax.set_ylabel('normalised evoked rate')
ax.set_title('Tuning shape is preserved across arousal')
ax.legend(frameon=False)

ax = axes[1]
lim = np.percentile(np.r_[peak_lo[big], peak_hi[big]], 99)
ax.plot(peak_lo[big], peak_hi[big], '.', ms=4, color='0.35', alpha=0.6)
ax.plot([0, lim], [0, lim], 'r--', lw=1)
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_xlabel('evoked rate at BF, low arousal (Hz)')
ax.set_ylabel('evoked rate at BF, high arousal (Hz)')
ax.set_title(f'Response gain (n={int(big.sum())} units)\nWilcoxon p={w.pvalue:.1e}')

ax = axes[2]
cm = np.zeros((len(ufreq), len(ufreq)))
for a, b in zip(bf_lo[big], bf_hi[big]):
    cm[a, b] += 1
cm = cm / cm.sum(1, keepdims=True)
im = ax.imshow(cm, cmap='viridis', vmin=0, vmax=1)
ax.set_xticks(range(5)); ax.set_xticklabels([f'{f/1000:g}' for f in ufreq])
ax.set_yticks(range(5)); ax.set_yticklabels([f'{f/1000:g}' for f in ufreq])
ax.set_xlabel('BF, high arousal (kHz)'); ax.set_ylabel('BF, low arousal (kHz)')
ax.set_title(f'Best frequency is stable\n{agree*100:.0f}% of units keep the same BF')
for i in range(5):
    for j in range(5):
        ax.text(j, i, f'{cm[i,j]:.2f}', ha='center', va='center', fontsize=8,
                color='w' if cm[i, j] < 0.6 else 'k')
plt.colorbar(im, ax=ax, pad=0.02, label='fraction')
fig.suptitle('Frequency tuning under low vs high arousal (median split on pre-tone pupil '
             f'diameter, {len(sessions)} sessions, one per mouse)', y=1.04, fontsize=12)
fig.savefig('figures/fig08_arousal.png', dpi=150, bbox_inches='tight')
plt.close(fig)

tab = pd.DataFrame(rows)
print(tab.to_string())
print(f'BF agreement across arousal states: {agree*100:.1f}%  (n={int(big.sum())} units used)')
print('median evoked rate at BF: low %.2f Hz, high %.2f Hz, Wilcoxon p=%.2e'
      % (np.median(peak_lo[big]), np.median(peak_hi[big]), w.pvalue))
pickle.dump(dict(agree=float(agree), p=float(w.pvalue), table=tab,
                 med_lo=float(np.median(peak_lo[big])), med_hi=float(np.median(peak_hi[big]))),
            open('summary_arousal.pkl', 'wb'))
