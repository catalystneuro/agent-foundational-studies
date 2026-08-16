"""Pool the analysis across all cached sessions and produce population figures."""
import glob, pickle, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm
from analysis_lib import load, drifting_analysis, static_analysis

plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 130, 'savefig.dpi': 150, 'savefig.bbox': 'tight'})
REG_COLORS = {'VISp':'#1f77b4','VISl':'#2ca02c','VISrl':'#9467bd',
              'VISam':'#8c564b','VISpm':'#e377c2','LGd':'#d62728'}
ORDER = ['VISp','VISl','VISrl','VISam','VISpm','LGd']

files = sorted(glob.glob('cache_*.pkl'))
print(len(files), 'sessions')
rows = []
for fn in tqdm(files, desc='sessions'):
    d, tsg = load(fn)
    dg = drifting_analysis(tsg, d['stim']['drifting_gratings_presentations'], n_perm=500)
    sg = static_analysis(tsg, d['stim']['static_gratings_presentations'])
    M = dg['metrics']
    rows.append(dict(session=d['session_id'], region=np.array(d['meta']['region']),
                     unit_id=d['meta']['unit_id'],
                     waveform_duration=d['meta']['waveform_duration'],
                     **{k: M[k] for k in M},
                     sg_gOSI=sg['gOSI'], sg_pref_ori=sg['pref_ori'], sg_p=sg['anova_p'],
                     tc=dg['tc'], dirs=dg['dirs']))
pickle.dump(rows, open('pooled.pkl','wb'))

cat = lambda k: np.concatenate([r[k] for r in rows])
reg = cat('region'); gOSI = cat('gOSI'); gDSI = cat('gDSI'); DSI = cat('DSI')
pref_ori = cat('pref_ori'); pref_dir = cat('pref_dir')
sg_gOSI = cat('sg_gOSI'); sg_pref = cat('sg_pref_ori'); sg_p = cat('sg_p')
anova_p = cat('anova_p'); perm_p = cat('perm_p')
evoked = cat('evoked'); baseline = cat('baseline'); wdur = cat('waveform_duration')
sess = np.concatenate([[r['session']]*len(r['region']) for r in rows])
sig = (anova_p < 0.01) & (perm_p < 0.05)

print('pooled units per region:', {r: int((reg==r).sum()) for r in ORDER})

# ------------- Fig 8: pooled population summary -------------
fig, axes = plt.subplots(2, 2, figsize=(11, 7.6))
ax = axes[0,0]
data = [gOSI[(reg==r) & np.isfinite(gOSI)] for r in ORDER]
bp = ax.boxplot(data, tick_labels=ORDER, widths=0.62, patch_artist=True, showfliers=False)
for p, r in zip(bp['boxes'], ORDER): p.set_facecolor(REG_COLORS[r]); p.set_alpha(0.55)
for m in bp['medians']: m.set_color('k')
for i, r in enumerate(ORDER):   # per-session medians
    med = [np.nanmedian(x['gOSI'][x['region']==r]) for x in rows if (x['region']==r).sum() > 5]
    ax.plot(np.full(len(med), i+1), med, 'o', ms=4, mfc='w', mec='k', mew=0.9, zorder=5)
ax.set_ylabel('global OSI')
ax.set_title(f'Orientation selectivity across areas (N = {len(files)} sessions)\n'
             'white circles = per-session medians', fontsize=9)
U, p = stats.mannwhitneyu(data[0], data[-1])
ax.text(0.02, 0.97, f'VISp (n={len(data[0])}) vs LGd (n={len(data[-1])}): p = {p:.1e}',
        transform=ax.transAxes, va='top', fontsize=8)

ax = axes[0,1]
frac = [sig[reg==r].mean()*100 for r in ORDER]
n = [(reg==r).sum() for r in ORDER]
ax.bar(ORDER, frac, color=[REG_COLORS[r] for r in ORDER], alpha=0.85)
for i,(f_,nn) in enumerate(zip(frac,n)):
    ax.text(i, f_+1.2, f'{f_:.0f}%\n(n={nn})', ha='center', fontsize=7.5)
ax.set_ylabel('% units significantly orientation tuned'); ax.set_ylim(0, max(frac)*1.35)
ax.set_title('Direction ANOVA p<0.01 AND gOSI permutation p<0.05', fontsize=9)

ax = axes[1,0]
for r in ['VISp','LGd']:
    k = (reg==r) & np.isfinite(gOSI)
    x = np.sort(gOSI[k])
    ax.plot(x, np.arange(1,len(x)+1)/len(x), lw=2.2, color=REG_COLORS[r],
            label=f'{r} (n={k.sum()}, median {np.median(x):.2f})')
ax.set_xlabel('global OSI'); ax.set_ylabel('cumulative fraction of units')
ax.legend(frameon=False, fontsize=8); ax.set_title('Cortex vs thalamus', fontsize=9)

ax = axes[1,1]
k = np.isfinite(gOSI) & np.isfinite(sg_gOSI)
for r in ['VISp','LGd']:
    kk = k & (reg==r)
    ax.plot(gOSI[kk], sg_gOSI[kk], 'o', ms=3.5, alpha=0.55, mec='none',
            color=REG_COLORS[r], label=r)
rr = stats.spearmanr(gOSI[k & np.isin(reg, ORDER)], sg_gOSI[k & np.isin(reg, ORDER)])
lim = 1.0; ax.plot([0,lim],[0,lim],'k--',lw=0.8)
ax.set_xlabel('gOSI, drifting gratings'); ax.set_ylabel('gOSI, static gratings')
ax.legend(frameon=False, fontsize=8)
ax.set_title(f'Selectivity is consistent across stimulus types\nSpearman r = {rr.statistic:.2f}',
             fontsize=9)
fig.tight_layout()
fig.savefig('fig08_pooled_population.png'); plt.close(fig)
print('fig08 done')

# ------------- Fig 9: preferred orientation / direction structure -------------
fig = plt.figure(figsize=(12, 4.4))
ax = fig.add_subplot(1, 3, 1, projection='polar')
k = (reg=='VISp') & sig & np.isfinite(pref_ori)
th = np.deg2rad(pref_ori[k]*2)          # doubled angle: orientation space
h, edges = np.histogram(pref_ori[k], bins=np.arange(0, 181, 15))
ax.bar(np.deg2rad(edges[:-1]*2), h, width=np.deg2rad(30), align='edge',
       color=REG_COLORS['VISp'], alpha=0.85)
ax.set_xticks(np.deg2rad(np.arange(0, 360, 60)))
ax.set_xticklabels([f'{int(x/2)}°' for x in np.arange(0, 360, 60)])
z = np.abs(np.mean(np.exp(1j*th))); ray_p = np.exp(-len(th)*z**2)
ax.set_title(f'Preferred orientation, VISp (n={k.sum()})\n'
             f'plotted in doubled-angle space\nRayleigh p = {ray_p:.1e}', fontsize=8.5, pad=18)

ax = fig.add_subplot(1, 3, 2)
kk = (reg=='VISp') & sig & (sg_p<0.01) & np.isfinite(sg_pref) & np.isfinite(pref_ori)
diff = ((sg_pref[kk] - pref_ori[kk] + 90) % 180) - 90
ax.hist(diff, bins=np.arange(-90, 91, 15), color=REG_COLORS['VISp'], alpha=0.85)
thd = np.deg2rad(2*diff); zz = np.abs(np.mean(np.exp(1j*thd)))
pd_ = np.exp(-len(thd)*zz**2)
ax.set_xlabel('preferred orientation, static − drifting (deg)')
ax.set_ylabel('VISp units')
ax.set_title(f'Cross-stimulus agreement (n={kk.sum()})\nmedian |Δ| = {np.median(np.abs(diff)):.0f}°, '
             f'Rayleigh p = {pd_:.1e}', fontsize=8.5)

ax = fig.add_subplot(1, 3, 3)
bins = np.linspace(0, 1, 21)
for r in ['VISp','LGd']:
    k2 = (reg==r) & np.isfinite(DSI)
    ax.hist(DSI[k2], bins=bins, histtype='step', lw=2, density=True,
            color=REG_COLORS[r], label=f'{r} (median {np.nanmedian(DSI[k2]):.2f})')
ax.set_xlabel('direction selectivity index'); ax.set_ylabel('density')
ax.legend(frameon=False, fontsize=8)
ax.set_title('Direction selectivity', fontsize=9)
fig.tight_layout()
fig.savefig('fig09_preferred_orientation.png'); plt.close(fig)
print('fig09 done')

# ------------- Fig 10: heatmap of normalized tuning curves, aligned to preference -------------
fig, axes = plt.subplots(1, 3, figsize=(12, 4.2),
                         gridspec_kw={'width_ratios':[1,1,1.15]})
for ax, r in zip(axes[:2], ['VISp','LGd']):
    idx = np.where(reg==r)[0]
    tc = np.concatenate([x['tc'].T for x in rows])[idx]
    dirs = rows[0]['dirs']
    pk = np.argmax(tc, axis=1)
    rolled = np.array([np.roll(t, 4-p) for t, p in zip(tc, pk)])
    norm = rolled / np.clip(rolled.max(1, keepdims=True), 1e-9, None)
    o = np.argsort(-gOSI[idx])
    im = ax.imshow(norm[o], aspect='auto', cmap='viridis', vmin=0, vmax=1,
                   extent=[-180, 180, len(idx), 0])
    ax.set_xticks([-180,-90,0,90,180])
    ax.set_xlabel('direction relative to preferred (deg)')
    ax.set_ylabel('unit (sorted by gOSI)')
    ax.set_title(f'{r} (n={len(idx)})', fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, label='normalized rate')
ax = axes[2]
for r in ['VISp','LGd']:
    idx = np.where(reg==r)[0]
    tc = np.concatenate([x['tc'].T for x in rows])[idx]
    pk = np.argmax(tc, axis=1)
    rolled = np.array([np.roll(t, 4-p) for t, p in zip(tc, pk)])
    norm = rolled / np.clip(rolled.max(1, keepdims=True), 1e-9, None)
    x = np.arange(-180, 180, 45)
    m, s = norm.mean(0), norm.std(0)/np.sqrt(len(norm))
    ax.plot(x, m, '-o', color=REG_COLORS[r], label=r, ms=4)
    ax.fill_between(x, m-s, m+s, color=REG_COLORS[r], alpha=0.25)
ax.set_xticks(x); ax.set_xlabel('direction relative to preferred (deg)')
ax.set_ylabel('normalized firing rate'); ax.legend(frameon=False)
ax.set_title('Population-average tuning:\nV1 shows a clear second peak at 180°', fontsize=9)
fig.tight_layout()
fig.savefig('fig10_aligned_tuning.png'); plt.close(fig)
print('fig10 done')

with open('pooled_stats.txt','w') as fh:
    fh.write(f'sessions: {len(files)}\n')
    for r in ORDER:
        k = reg==r
        fh.write(f'{r:6s} n={k.sum():4d} tuned={sig[k].mean()*100:5.1f}% '
                 f'median gOSI={np.nanmedian(gOSI[k]):.3f} '
                 f'median DSI={np.nanmedian(DSI[k]):.3f} '
                 f'median gOSI(static)={np.nanmedian(sg_gOSI[k]):.3f}\n')
    U, p = stats.mannwhitneyu(gOSI[reg=='VISp'], gOSI[reg=='LGd'])
    fh.write(f'VISp vs LGd gOSI Mann-Whitney U={U:.0f} p={p:.3e}\n')
print(open('pooled_stats.txt').read())
