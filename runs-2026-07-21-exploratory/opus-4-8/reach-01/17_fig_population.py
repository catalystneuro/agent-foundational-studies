import numpy as np, pandas as pd, pickle, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
allf = pd.read_pickle('./cache/dir_fits.pkl')
fs = pd.read_pickle('./cache/dir_fits_straight.pkl')
fc = pd.read_pickle('./cache/dir_fits_curved.pkl')
D = pickle.load(open('./cache/decoding.pkl','rb'))
tr = pd.read_pickle('./cache/trials.pkl')
V = pickle.load(open('./cache/veltuning.pkl','rb'))
ep_order = ['baseline', 'preparatory', 'perimovement', 'late_movement']
lbl = ['baseline\n(pre-target)', 'delay\n(-300-0 ms\nre go cue)', 'peri-move\n(-50-250 ms\nre onset)',
       'late move\n(250-550 ms\nre onset)']

fig = plt.figure(figsize=(16, 9))
gs = fig.add_gridspec(2, 3, hspace=.42, wspace=.30)

ax = fig.add_subplot(gs[0, 0])
frac = [ (allf[allf.epoch==e]['p'] < 0.01).mean() for e in ep_order ]
ax.bar(range(4), frac, color=['0.6', 'tab:blue', 'tab:red', 'tab:orange'])
ax.axhline(0.01, color='k', ls=':', label='chance (p<0.01)')
ax.set(xticks=range(4), ylabel='fraction of units cosine-tuned',
       title='Directional tuning by task epoch', ylim=(0, 1.05))
ax.set_xticklabels(lbl, fontsize=7.5); ax.legend(fontsize=8)
for i, f in enumerate(frac): ax.text(i, f + .02, f'{f:.2f}', ha='center', fontsize=9)

ax = fig.add_subplot(gs[0, 1])
data = [allf[allf.epoch == e]['mod_depth'].values for e in ep_order]
bp = ax.boxplot(data, showfliers=False, patch_artist=True, widths=.6)
for p, c in zip(bp['boxes'], ['0.6', 'tab:blue', 'tab:red', 'tab:orange']): p.set_facecolor(c)
for i, dd in enumerate(data):
    ax.plot(np.random.default_rng(i).normal(i+1, .07, len(dd)), dd, '.', color='k', ms=2, alpha=.4)
ax.set(xticks=range(1, 5), ylabel='cosine modulation depth (Hz)',
       title='Depth of direction tuning')
ax.set_xticklabels(lbl, fontsize=7.5)

ax = fig.add_subplot(gs[0, 2], projection='polar')
sig = allf[(allf.epoch == 'perimovement')]['p'].values < 0.01
ax.hist(allf[allf.epoch == 'perimovement']['pd'].values[sig],
        bins=np.linspace(-np.pi, np.pi, 25), color='tab:red', alpha=.8, label='peri-movement')
ax.hist(allf[allf.epoch == 'preparatory']['pd'].values[
            allf[allf.epoch == 'preparatory']['p'].values < 0.01],
        bins=np.linspace(-np.pi, np.pi, 25), histtype='step', color='tab:blue', lw=2,
        label='preparatory')
ax.set_title('Preferred directions', pad=26); ax.set_rlabel_position(250)
ax.legend(fontsize=8, loc='lower right', bbox_to_anchor=(1.25, -0.1))
ax.tick_params(labelsize=8)

ax = fig.add_subplot(gs[1, 0])
prep = allf[allf.epoch == 'preparatory']; move = allf[allf.epoch == 'perimovement']
both = ((prep['p'].values < 0.01) & (move['p'].values < 0.01)
        & (prep['mod_depth'].values > 0.5) & (move['mod_depth'].values > 1.0))
a1, a2 = prep['pd'].values[both], move['pd'].values[both]
m1, m2 = np.angle(np.exp(1j*a1).mean()), np.angle(np.exp(1j*a2).mean())
cc = (np.sum(np.sin(a1-m1)*np.sin(a2-m2))
      / np.sqrt(np.sum(np.sin(a1-m1)**2) * np.sum(np.sin(a2-m2)**2)))
ax.plot(np.degrees(prep['pd'].values[both]), np.degrees(move['pd'].values[both]), 'o', ms=4,
        color='k', alpha=.6)
ax.plot([-180, 180], [-180, 180], 'r--', lw=1)
dd = np.degrees(np.angle(np.exp(1j*(move['pd'].values[both] - prep['pd'].values[both]))))
ax.set(xlabel='preferred direction, delay (deg)', ylabel='preferred direction, movement (deg)',
       title=f'Delay vs. movement PD (well-tuned units)\ncirc. r = {cc:.2f}, median |shift| = '
             f'{np.median(np.abs(dd)):.0f}$\\degree$, n={both.sum()}',
       xlim=(-185, 185), ylim=(-185, 185))

ax = fig.add_subplot(gs[1, 1])
both2 = (fs['p'].values < 0.01) & (fc['p'].values < 0.01)
d2 = np.degrees(np.angle(np.exp(1j*(fc['pd'].values[both2] - fs['pd'].values[both2]))))
ax.hist(d2, bins=np.arange(-180, 190, 15), color='0.4')
ax.axvline(0, color='r', ls='--')
ax.set(xlabel='PD(curved reaches) - PD(straight reaches) (deg)', ylabel='units',
       title=f'Target direction is not the whole story:\nPD shifts when the path is curved '
             f'(median |shift| {np.median(np.abs(d2)):.0f}$\\degree$)')

ax = fig.add_subplot(gs[1, 2])
names = list(D['acc'].keys()); vals = [D['acc'][n] for n in names]
ax.bar(range(len(names)), vals, color=['0.6', 'tab:blue', 'tab:red'])
ax.axhline(1/len(np.unique(D['lab'])), color='k', ls=':', label='chance')
for i, v in enumerate(vals): ax.text(i, v + .02, f'{v:.2f}', ha='center')
ax.set(xticks=range(len(names)), ylabel='7-way decoding accuracy', ylim=(0, 1),
       title='Reach direction decoded from\npopulation activity')
ax.set_xticklabels(['baseline', 'delay', 'movement'], fontsize=9); ax.legend(fontsize=8)
fig.suptitle('Population summary: reach-direction tuning in 115 M1/PMd units', y=0.98)
plt.savefig('fig04_population_direction.png', dpi=140, bbox_inches='tight')
