"""Population, cross-stimulus, GLM and decoding figures (prototype session)."""
import pickle, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from analysis_lib import load
import nemos as nmo
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 130, 'savefig.dpi': 150, 'savefig.bbox': 'tight'})
REG_COLORS = {'VISp':'#1f77b4','VISl':'#2ca02c','VISrl':'#9467bd',
              'VISam':'#8c564b','VISpm':'#e377c2','LGd':'#d62728'}
ORDER = ['VISp','VISl','VISrl','VISam','VISpm','LGd']

d, tsg = load('cache_715093703.pkl')
res = pickle.load(open('results.pkl','rb'))
dg, sg = res['dg'], res['sg']
reg = np.array(d['meta']['region']); uids = np.array(list(tsg.keys()))
M = dg['metrics']; dirs = dg['dirs']
sig = (M['anova_p'] < 0.01) & (M['perm_p'] < 0.05)

# ---------------- Fig 4: population selectivity summary ----------------
fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.4))
ax = axes[0,0]
data = [M['gOSI'][(reg==r) & np.isfinite(M['gOSI'])] for r in ORDER]
bp = ax.boxplot(data, labels=ORDER, widths=0.6, patch_artist=True, showfliers=False)
for p, r in zip(bp['boxes'], ORDER):
    p.set_facecolor(REG_COLORS[r]); p.set_alpha(0.55)
for m in bp['medians']: m.set_color('k')
for i, dd in enumerate(data):
    ax.plot(np.random.normal(i+1, 0.07, len(dd)), dd, '.', ms=2.5, color='0.25', alpha=0.6)
ax.set_ylabel('global OSI'); ax.set_title('Orientation selectivity by area')
u, p = stats.mannwhitneyu(data[0], data[-1])
ax.text(0.02, 0.97, f'VISp vs LGd: Mann-Whitney U p = {p:.1e}',
        transform=ax.transAxes, va='top', fontsize=8)

ax = axes[0,1]
frac = [sig[reg==r].mean()*100 for r in ORDER]
n = [(reg==r).sum() for r in ORDER]
ax.bar(ORDER, frac, color=[REG_COLORS[r] for r in ORDER], alpha=0.8)
for i,(f,nn) in enumerate(zip(frac,n)):
    ax.text(i, f+1.5, f'{f:.0f}%\n(n={nn})', ha='center', fontsize=7.5)
ax.set_ylabel('% units significantly tuned')
ax.set_ylim(0, max(frac)*1.35)
ax.set_title('ANOVA p<0.01 across directions\n& permutation p<0.05 on gOSI', fontsize=9)

ax = axes[1,0]
for r in ['VISp','LGd']:
    k = (reg==r) & np.isfinite(M['gOSI'])
    x = np.sort(M['gOSI'][k])
    ax.plot(x, np.arange(1, len(x)+1)/len(x), lw=2, color=REG_COLORS[r], label=f'{r} (n={k.sum()})')
ax.set_xlabel('global OSI'); ax.set_ylabel('cumulative fraction of units')
ax.legend(frameon=False); ax.set_title('Cortex is far more orientation-selective than thalamus')

ax = axes[1,1]
for r in ['VISp','LGd']:
    k = reg==r
    ax.plot(M['gOSI'][k], M['gDSI'][k], 'o', ms=4, alpha=0.7,
            color=REG_COLORS[r], label=r, mec='none')
lim = np.nanmax([M['gOSI'], M['gDSI']])*1.05
ax.plot([0,lim],[0,lim],'k--',lw=0.8)
ax.set_xlabel('global OSI'); ax.set_ylabel('global DSI'); ax.legend(frameon=False)
ax.set_title('Orientation vs direction selectivity')
fig.tight_layout()
fig.savefig('fig04_population_selectivity.png'); plt.close(fig)
print('fig04 done')

# ---------------- Fig 5: static gratings & cross-stimulus consistency ----------------
sg_sig = sg['anova_p'] < 0.01
fig = plt.figure(figsize=(11.5, 7.2))
gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.32)
vp_sg = np.where((reg=='VISp') & sg_sig)[0]
vp_sg = vp_sg[np.argsort(-sg['gOSI'][vp_sg])][:3]
for i, u in enumerate(vp_sg):
    ax = fig.add_subplot(gs[0, i])
    ax.errorbar(sg['oris'], sg['tc'][:, u], yerr=sg['tc_sem'][:, u],
                marker='o', color=REG_COLORS['VISp'], capsize=2, lw=1.5)
    ax.set_xticks(sg['oris']); ax.set_xlabel('orientation (deg)')
    if i == 0: ax.set_ylabel('firing rate (Hz)')
    ax.set_title(f"VISp {uids[u]}\nstatic gratings, SF={sg['pref_sf'][u]:g} cpd, gOSI={sg['gOSI'][u]:.2f}",
                 fontsize=8)

ax = fig.add_subplot(gs[1, 0])
for r in ['VISp','LGd']:
    k = (reg==r) & np.isfinite(sg['gOSI'])
    x = np.sort(sg['gOSI'][k])
    ax.plot(x, np.arange(1,len(x)+1)/len(x), lw=2, color=REG_COLORS[r], label=r)
ax.set_xlabel('global OSI (static gratings)'); ax.set_ylabel('cumulative fraction')
ax.legend(frameon=False); ax.set_title('Static gratings replicate the\ncortex/thalamus difference', fontsize=9)

ax = fig.add_subplot(gs[1, 1])
k = (reg=='VISp') & sig & sg_sig & np.isfinite(sg['pref_ori']) & np.isfinite(M['pref_ori'])
a = M['pref_ori'][k]; b = sg['pref_ori'][k]
ax.plot(a, b, 'o', color=REG_COLORS['VISp'], ms=5, mec='none', alpha=0.8)
ax.plot([0,180],[0,180],'k--',lw=0.8)
diff = ((b - a + 90) % 180) - 90
# circular correlation on doubled angles
aa, bb = np.deg2rad(2*a), np.deg2rad(2*b)
num = np.sum(np.sin(aa-stats.circmean(aa))*np.sin(bb-stats.circmean(bb)))
den = np.sqrt(np.sum(np.sin(aa-stats.circmean(aa))**2)*np.sum(np.sin(bb-stats.circmean(bb))**2))
rho = num/den
ax.set_xlabel('preferred orientation, drifting (deg)')
ax.set_ylabel('preferred orientation, static (deg)')
ax.set_title(f'Cross-stimulus consistency\ncirc. r = {rho:.2f}, n = {k.sum()}', fontsize=9)

ax = fig.add_subplot(gs[1, 2])
ax.hist(diff, bins=np.arange(-90, 91, 15), color=REG_COLORS['VISp'], alpha=0.85)
ax.set_xlabel('preferred orientation difference (deg)\nstatic − drifting')
ax.set_ylabel('units')
ax.set_title(f'median |Δ| = {np.median(np.abs(diff)):.0f}°', fontsize=9)
fig.suptitle('Orientation preference measured with static gratings agrees with drifting gratings', y=0.98)
fig.savefig('fig05_static_gratings.png'); plt.close(fig)
print('fig05 done')

