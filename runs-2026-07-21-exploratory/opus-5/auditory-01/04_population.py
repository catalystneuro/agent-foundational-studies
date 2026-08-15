"""Population-level frequency tuning across all 15 sessions of DANDI:000986."""
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as sstats

res = pickle.load(open('results_000986.pkl', 'rb'))
sessions = sorted(res)
ufreq = res[sessions[0]]['ufreq']
oct_axis = np.log2(ufreq / ufreq[0])

# ---------------------------------------------------------------- pool units
stats_all = pd.concat([res[s]['stats'] for s in sessions], keys=sessions, names=['file', 'unit'])
curves_all = pd.concat([res[s]['curves'] for s in sessions], keys=sessions, names=['file', 'unit'])
psth_t = res[sessions[0]]['psth_t']
psth_all = np.concatenate([res[s]['psths'] for s in sessions], axis=0)
print('pooled units:', len(stats_all), ' responsive:', int(stats_all.responsive.sum()),
      ' tuned:', int(stats_all.tuned.sum()))

tun_mask = stats_all.tuned.values & stats_all.responsive.values
C = curves_all.values[tun_mask]
S = stats_all[tun_mask]
P = psth_all[tun_mask]

# normalise each tuning curve to its peak. Units whose peak evoked rate is below
# 1 Hz are excluded from the normalised displays: dividing by a near-zero peak
# would blow the curve up without adding information.
MIN_PEAK_HZ = 1.0
pos = C.max(1) >= MIN_PEAK_HZ
Cn = C[pos] / C[pos].max(1, keepdims=True)
bf_idx = C[pos].argmax(1)
centroid = (Cn * np.arange(len(ufreq))).sum(1) / np.maximum(Cn.sum(1), 1e-9)
order = np.lexsort((centroid, bf_idx))

fig = plt.figure(figsize=(14, 9))
gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.32)

ax = fig.add_subplot(gs[0, 0])
im = ax.imshow(Cn[order], aspect='auto', origin='lower', cmap='RdBu_r', vmin=-1, vmax=1,
               extent=[-0.5, len(ufreq) - 0.5, 0, pos.sum()])
ax.set_xticks(range(len(ufreq))); ax.set_xticklabels([f'{f/1000:g}' for f in ufreq])
ax.set_xlabel('tone frequency (kHz)'); ax.set_ylabel('frequency-tuned unit (sorted by BF)')
ax.set_title(f'Peak-normalised tuning curves\n{pos.sum()} tuned units with peak > {MIN_PEAK_HZ:g} Hz')
plt.colorbar(im, ax=ax, pad=0.02, label='norm. rate')

# ------------------------------------------------- BF-aligned population curve
ax = fig.add_subplot(gs[0, 1])
offsets = np.arange(-(len(ufreq) - 1), len(ufreq))
acc = {o: [] for o in offsets}
for i in range(Cn.shape[0]):
    b = bf_idx[i]
    for j in range(len(ufreq)):
        acc[j - b].append(Cn[i, j])
m = np.array([np.mean(acc[o]) if len(acc[o]) else np.nan for o in offsets])
se = np.array([np.std(acc[o]) / np.sqrt(len(acc[o])) if len(acc[o]) else np.nan for o in offsets])
nrec = np.array([len(acc[o]) for o in offsets])
ax.errorbar(offsets, m, yerr=se, marker='o', color='crimson', lw=2, capsize=3)
ax.axvline(0, color='0.6', ls=':'); ax.axhline(0, color='0.6', ls=':')
ax.set_xlabel('octaves from best frequency'); ax.set_ylabel('normalised evoked rate')
ax.set_title('Population tuning aligned to each unit\'s BF')
for o, y, k in zip(offsets, m, nrec):
    if k < 0.1 * Cn.shape[0]:
        ax.annotate(f'n={k}', (o, y), fontsize=7, textcoords='offset points', xytext=(0, -16),
                    ha='center', color='0.4')

# ------------------------------------------------------------- PSTH at BF
ax = fig.add_subplot(gs[0, 2])
bf_psth = np.array([P[i, C[i].argmax()] for i in np.where(pos)[0]])
base = bf_psth[:, psth_t < 0].mean(1, keepdims=True)
sd = bf_psth[:, psth_t < 0].std(1, keepdims=True) + 1e-6
Z = (bf_psth - base) / sd
lat_order = np.argsort(np.argmax(Z[:, psth_t >= 0], axis=1))
im = ax.imshow(Z[lat_order], aspect='auto', origin='lower', cmap='magma', vmin=-2, vmax=10,
               extent=[psth_t[0], psth_t[-1], 0, len(Z)])
ax.axvline(0, color='w', lw=0.8); ax.axvline(0.025, color='w', lw=0.8, ls=':')
ax.set_xlabel('time from tone onset (s)'); ax.set_ylabel('tuned unit (sorted by peak time)')
ax.set_title('Response at each unit\'s best frequency')
plt.colorbar(im, ax=ax, pad=0.02, label='z-score vs pre-tone')

# ----------------------------------------- best-frequency distribution by mouse
ax = fig.add_subplot(gs[1, 0])
tab = (S.groupby(['subject', 'best_frequency']).size().unstack(fill_value=0))
tab = tab.div(tab.sum(1), axis=0)
x = np.arange(len(ufreq)); w = 0.15
for i, (subj, row) in enumerate(tab.iterrows()):
    ax.bar(x + (i - len(tab) / 2) * w, [row.get(f, 0) for f in ufreq], w, label=subj)
ax.set_xticks(x); ax.set_xticklabels([f'{f/1000:g}' for f in ufreq])
ax.set_xlabel('best frequency (kHz)'); ax.set_ylabel('fraction of tuned units')
ax.set_title('Best-frequency distribution per mouse')
ax.legend(fontsize=8, frameon=False, ncol=2, title='subject')

# ------------------------------------------------------ split-half reliability
rel, rel_shuf = [], []
rng = np.random.default_rng(0)
for s in sessions:
    r = res[s]
    ev, bl, fr = r['counts_ev'], r['counts_bl'], r['freqs']
    d = ev / 0.1 - bl / 0.1
    keep = r['stats'].tuned.values & r['stats'].responsive.values
    half = np.arange(len(fr)) % 2
    for shuffle in (False, True):
        f = rng.permutation(fr) if shuffle else fr
        a = np.array([[d[(half == 0) & (f == q), j].mean() for q in ufreq]
                      for j in np.where(keep)[0]])
        b = np.array([[d[(half == 1) & (f == q), j].mean() for q in ufreq]
                      for j in np.where(keep)[0]])
        rr = [sstats.pearsonr(a[i], b[i])[0] for i in range(len(a))]
        (rel_shuf if shuffle else rel).extend(rr)
rel, rel_shuf = np.array(rel), np.array(rel_shuf)
ax = fig.add_subplot(gs[1, 1])
bins = np.linspace(-1, 1, 41)
ax.hist(rel_shuf, bins, color='0.7', label=f'frequency labels shuffled\n(median {np.nanmedian(rel_shuf):.2f})')
ax.hist(rel, bins, color='crimson', alpha=0.75, label=f'observed\n(median {np.nanmedian(rel):.2f})')
ax.set_xlabel('split-half correlation of tuning curve (odd vs even trials)')
ax.set_ylabel('units'); ax.set_title('Tuning curves are reproducible within session')
ax.legend(fontsize=8, frameon=False, loc='upper left')

# ---------------------------------------------------- latency and selectivity
ax = fig.add_subplot(gs[1, 2])
lat = S.latency_s.dropna().values * 1000
ax.hist(lat, bins=np.arange(0, 105, 5), color='steelblue')
ax.set_xlabel('onset latency at BF (ms)'); ax.set_ylabel('units')
ax.set_title(f'Tone-onset latency\nmedian {np.median(lat):.0f} ms  (n={len(lat)})')
ax2 = ax.inset_axes([0.52, 0.45, 0.45, 0.5])
ax2.hist(S.sparseness.dropna(), bins=20, color='seagreen')
ax2.set_xlabel('lifetime sparseness', fontsize=8)
ax2.set_ylabel('units', fontsize=8); ax2.tick_params(labelsize=7)

fig.suptitle('Frequency tuning across the DANDI:000986 population '
             f'({len(stats_all)} units, {len(sessions)} sessions, '
             f'{stats_all.subject.nunique()} mice)', fontsize=13, y=0.96)
fig.savefig('figures/fig04_population.png', dpi=150, bbox_inches='tight')
plt.close(fig)

summary = dict(
    n_units=int(len(stats_all)), n_responsive=int(stats_all.responsive.sum()),
    n_tuned=int(tun_mask.sum()), median_latency_ms=float(np.median(lat)),
    median_splithalf_r=float(np.nanmedian(rel)), median_splithalf_r_shuffled=float(np.nanmedian(rel_shuf)),
    bf_counts={float(k): int(v) for k, v in S.best_frequency.value_counts().items()},
)
print(summary)
pickle.dump(summary, open('summary_population.pkl', 'wb'))
