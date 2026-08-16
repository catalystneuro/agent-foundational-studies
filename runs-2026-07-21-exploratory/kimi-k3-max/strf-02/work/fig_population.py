"""Population-level STRF figure across sessions."""
import numpy as np, glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from session_strf import FREQ_VALS, BIN_C, BASE_MASK

files = sorted(glob.glob('session_*.npz'))
print('sessions:', files)
S = []
for f in files:
    d = np.load(f)
    S.append({k: d[k] for k in ['strfs','glm_rate','pseudo_r2','p_resp','p_tune','tuned','bf','peak_lat','peak_hz','min_hz','separability']})

n_units = np.array([len(s['tuned']) for s in S])
n_tuned = np.array([s['tuned'].sum() for s in S])
print('units/session:', n_units, 'tuned:', n_tuned)

bf = np.concatenate([s['bf'][s['tuned']] for s in S])
lat = np.concatenate([s['peak_lat'][s['tuned']] for s in S]) * 1000
sep = np.concatenate([s['separability'][s['tuned']] for s in S])
pk = np.concatenate([s['peak_hz'][s['tuned']] for s in S])
mn = np.concatenate([s['min_hz'][s['tuned']] for s in S])
pr2 = np.concatenate([s['pseudo_r2'][s['tuned']] for s in S])
strfs = np.concatenate([s['strfs'][s['tuned']] for s in S])

fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
ax = axes[0, 0]
counts, edges = np.histogram(bf/1000, bins=np.array([1,3,6,12,24,40]))
ax.bar(range(5), counts, tick_label=['2-3','4-6','8-12','16-24','32-40'])
ax.set_xlabel('best frequency (kHz)'); ax.set_ylabel('# tuned units')
ax.set_title(f'BF distribution (n={len(bf)} tuned)')

ax = axes[0, 1]
ax.hist(lat, bins=np.arange(5, 105, 5), color='C0')
ax.axvline(np.median(lat), color='r', ls='--', label=f'median {np.median(lat):.0f} ms')
ax.set_xlabel('peak latency (ms)'); ax.set_ylabel('# units'); ax.legend(frameon=False)
ax.set_title('STRF peak latency')

ax = axes[0, 2]
ax.hist(sep, bins=np.linspace(0.2, 1.0, 33), color='C2')
ax.axvline(np.median(sep), color='r', ls='--', label=f'median {np.median(sep):.2f}')
ax.set_xlabel('separability index (rank-1 var. frac.)'); ax.set_ylabel('# units'); ax.legend(frameon=False)
ax.set_title('STRF separability (SVD)')

ax = axes[1, 0]
ax.scatter(bf/1000, lat, s=6, alpha=0.4)
ax.set_xscale('log'); ax.set_xticks([2,4,8,16,32]); ax.set_xticklabels([2,4,8,16,32])
ax.set_xlabel('best frequency (kHz)'); ax.set_ylabel('peak latency (ms)')
ax.set_title('latency vs BF')

ax = axes[1, 1]
supp = -mn / (pk - mn + 1e-9)   # suppression prominence 0..1
ax.hist(supp, bins=np.linspace(0, 1, 33), color='C3')
ax.axvline(np.median(supp), color='r', ls='--', label=f'median {np.median(supp):.2f}')
ax.set_xlabel('suppression index |min|/(max+|min|)'); ax.set_ylabel('# units'); ax.legend(frameon=False)
ax.set_title('suppressive subfield prominence')

ax = axes[1, 2]
ax.hist(pr2[np.isfinite(pr2)], bins=np.linspace(0, 0.35, 35), color='C4')
ax.axvline(np.nanmedian(pr2), color='r', ls='--', label=f'median {np.nanmedian(pr2):.3f}')
ax.set_xlabel('GLM test pseudo-$R^2$'); ax.set_ylabel('# units'); ax.legend(frameon=False)
ax.set_title('GLM STRF predictive performance')

fig.savefig('fig4_population.png', dpi=150)
print('fig4 saved')

# ---- Fig 5: separable vs inseparable example STRFs
order = np.argsort(sep)
# most separable strong units and least separable strong units
strong = pk > np.percentile(pk, 50)
idx_sep = [i for i in order[::-1] if strong[i]][:4]
idx_insep = [i for i in order if strong[i]][:4]
fig, axes = plt.subplots(2, 4, figsize=(14, 6), constrained_layout=True)
for r, idxs, ttl in [(0, idx_sep, 'separable'), (1, idx_insep, 'inseparable')]:
    for c, i in enumerate(idxs):
        ax = axes[r, c]
        vm = np.abs(strfs[i]).max()
        ax.imshow(strfs[i], aspect='auto', origin='lower', cmap='RdBu_r',
                  extent=[BIN_C[0]*1000, BIN_C[-1]*1000, -0.5, 4.5], vmin=-vm, vmax=vm)
        ax.set_yticks(range(5)); ax.set_yticklabels([str(int(fv/1000)) for fv in FREQ_VALS], fontsize=8)
        ax.axvline(0, color='k', lw=0.4); ax.axvline(25, color='k', lw=0.4, ls='--')
        ax.set_title(f'{ttl} | sep={sep[i]:.2f} | BF {int(bf[i]/1000)} kHz', fontsize=9)
        if c == 0: ax.set_ylabel('freq (kHz)')
        ax.set_xlabel('time (ms)', fontsize=8)
fig.suptitle('Most separable (top) vs least separable (bottom) STRFs among tuned units')
fig.savefig('fig5_separability.png', dpi=150)
print('fig5 saved')

# summary stats for README
print('n tuned total:', len(bf), 'of', n_units.sum())
print('median latency ms:', np.median(lat))
print('median separability:', np.median(sep))
print('frac sep>0.7:', np.mean(sep > 0.7))
print('median suppression index:', np.median(supp))
print('frac with supp>0.2:', np.mean(supp > 0.2))
print('median pseudoR2:', np.nanmedian(pr2), '| frac >0.05:', np.nanmean(pr2 > 0.05))
print('BF hist:', counts)
