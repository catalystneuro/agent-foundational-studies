import numpy as np, pandas as pd, pickle, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
G = pickle.load(open('./cache/glm_results.pkl','rb'))
D = pickle.load(open('./cache/decoding.pkl','rb'))
M = pickle.load(open('./cache/binned.pkl','rb'))
tr = pd.read_pickle('./cache/trials.pkl')

fig = plt.figure(figsize=(16, 9.5))
gs = fig.add_gridspec(2, 3, hspace=.38, wspace=.30)

ax = fig.add_subplot(gs[0, 0])
names = list(G['res'].keys()); vals = [G['res'][n] for n in names]
bp = ax.boxplot(vals, showfliers=False, patch_artist=True, widths=.6, vert=False)
cols = ['0.7', 'tab:blue', 'tab:red', 'tab:green', 'tab:purple']
for p, c in zip(bp['boxes'], cols): p.set_facecolor(c)
for i, v in enumerate(vals):
    ax.plot(v, np.random.default_rng(i).normal(i+1, .07, len(v)), '.', color='k', ms=2.5, alpha=.4)
ax.axvline(0, color='k', lw=.8)
ax.set(yticks=range(1, len(names)+1), xlabel='held-out Poisson deviance explained',
       title='Poisson GLM encoding models (NeMoS)\n5-fold CV grouped by trial, 115 units')
ax.set_yticklabels(names, fontsize=9)

ax = fig.add_subplot(gs[0, 1])
ax.plot(G['res']['direction only'], G['res']['direction x speed'], 'o', ms=5, color='k', alpha=.6)
lim = [0, max(G['res']['direction x speed'].max(), G['res']['direction only'].max())*1.05]
ax.plot(lim, lim, 'r--'); ax.set(xlim=lim, ylim=lim, xlabel='direction only',
    ylabel='direction $\\times$ speed',
    title='Adding speed improves the fit for\n%d/%d units' %
          ((G['res']['direction x speed'] > G['res']['direction only']).sum(), len(G['res']['direction only'])))

ax = fig.add_subplot(gs[0, 2])
ax.plot(G['res']['position'], G['res']['direction x speed'], 'o', ms=5, color='k', alpha=.6)
lim = [0, max(G['res']['position'].max(), G['res']['direction x speed'].max())*1.05]
ax.plot(lim, lim, 'r--'); ax.set(xlim=lim, ylim=lim, xlabel='position model',
    ylabel='velocity model (direction $\\times$ speed)',
    title='Velocity vs. position as predictors\n(velocity better for %d/%d units)' %
          ((G['res']['direction x speed'] > G['res']['position']).sum(), len(G['res']['position'])))

# decoded velocity traces
peri = D['peri']; good = D['good']
t_all = M['t'][peri][good]; trial_all = M['trial'][peri][good]
sel = np.unique(trial_all)[40:46]
mask = np.isin(trial_all, sel)
ax = fig.add_subplot(gs[1, :2])
tt = np.arange(mask.sum())*0.02
ax.plot(tt, D['Y'][mask, 0], 'k', lw=1.6, label='actual $v_x$')
ax.plot(tt, D['pred'][mask, 0], 'tab:red', lw=1.4, label='decoded $v_x$')
ax.plot(tt, D['Y'][mask, 1]-1500, 'k', lw=1.6)
ax.plot(tt, D['pred'][mask, 1]-1500, 'tab:blue', lw=1.4, label='decoded $v_y$')
for b in np.flatnonzero(np.diff(trial_all[mask]) != 0): ax.axvline(tt[b], color='0.8', lw=1)
ax.text(0.4, 400, '$v_x$', fontsize=11); ax.text(0.4, -1100, '$v_y$', fontsize=11)
ax.set(xlabel='concatenated peri-movement time (s)', ylabel='velocity (mm/s), $v_y$ offset',
       title='Cross-validated linear decoding of hand velocity from 115 units '
             f'($R^2_{{v_x}}$={D["r2"][0]:.2f}, $R^2_{{v_y}}$={D["r2"][1]:.2f})')
ax.legend(ncol=3, fontsize=9, loc='lower left')

ax = fig.add_subplot(gs[1, 2])
ax.hist(D['ang_err'][D['fast']], bins=np.arange(0, 185, 5), color='0.4')
ax.axvline(np.median(D['ang_err'][D['fast']]), color='r', ls='--',
           label='median %.0f$\\degree$' % np.median(D['ang_err'][D['fast']]))
ax.set(xlabel='decoded - actual movement direction (deg)', ylabel='20 ms bins',
       title='Instantaneous direction decoding error\n(bins with speed > 100 mm/s)')
ax.legend(fontsize=9)
plt.savefig('fig05_glm_and_decoding.png', dpi=140, bbox_inches='tight')
for n in names: print(f'{n:26s} median dev.expl. {np.median(G["res"][n]):.4f}')
