import numpy as np, pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with open('data/direction_stats.pkl','rb') as f:
    D = pickle.load(f)
res, bm, bc, occ, grp = D['res'], D['bin_means'], D['bin_centers'], D['occupied'], D['group']

sig = res['anova_p_move'] < 0.01
fig = plt.figure(figsize=(14, 9))
gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

# (a) PD histogram, M1 vs PMd (sig units only)
ax = fig.add_subplot(gs[0, 0])
bins = np.arange(0, 361, 20)
for g, c in [('M1','tab:red'), ('PMd','tab:blue')]:
    m = sig & (grp == g)
    ax.hist(res['pd_move'][m], bins=bins, alpha=0.6, color=c, label=f'{g} (n={m.sum()})')
ax.set_xlabel('preferred direction (deg)'); ax.set_ylabel('units')
ax.legend(fontsize=9)
ax.set_title('A  Preferred direction distribution (movement)', loc='left', fontsize=11)

# (b) tuning depth distribution
ax = fig.add_subplot(gs[0, 1])
ax.hist(res['depth_move'][sig], bins=30, color='tab:blue', alpha=0.75, label=f'sig (n={sig.sum()})')
ax.hist(res['depth_move'][~sig], bins=30, color='gray', alpha=0.6, label=f'not sig (n={(~sig).sum()})')
ax.set_xlabel('modulation depth (sp/s)'); ax.set_ylabel('units')
ax.legend(fontsize=9)
ax.set_title('B  Direction modulation depth', loc='left', fontsize=11)

# (c) cosine fit R2 vs depth
ax = fig.add_subplot(gs[0, 2])
ax.scatter(res['depth_move'][sig], res['cos_r2_move'][sig], s=12, alpha=0.6, c='tab:blue')
ax.scatter(res['depth_move'][~sig], res['cos_r2_move'][~sig], s=12, alpha=0.6, c='gray')
ax.set_xlabel('modulation depth (sp/s)'); ax.set_ylabel('cosine fit R²')
ax.set_title('C  Cosine tuning fit quality', loc='left', fontsize=11)

# (d) delay vs move depth (planning vs execution)
ax = fig.add_subplot(gs[1, 0])
ok = ~np.isnan(res['depth_delay'])
ax.scatter(res['depth_delay'][ok], res['depth_move'][ok], s=12, alpha=0.5,
           c=['tab:red' if g=='M1' else 'tab:blue' for g in grp[ok]])
lim = np.nanmax(np.concatenate([res['depth_delay'], res['depth_move']]))
ax.plot([0,lim],[0,lim],'k--',lw=0.8)
ax.set_xlabel('delay-period depth (sp/s)'); ax.set_ylabel('movement-period depth (sp/s)')
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([],[],marker='o',ls='',color='tab:red',label='M1'),
                   Line2D([],[],marker='o',ls='',color='tab:blue',label='PMd')], fontsize=9)
ax.set_title('D  Planning vs execution modulation', loc='left', fontsize=11)

# (e) PD consistency delay vs move (circular diff) for units sig in both
ax = fig.add_subplot(gs[1, 1])
both = (res['anova_p_move']<0.01) & (res['anova_p_delay']<0.01)
d = (res['pd_move'][both] - res['pd_delay'][both] + 180) % 360 - 180
ax.hist(d, bins=np.arange(-180,181,20), color='tab:green', alpha=0.8)
ax.set_xlabel('PD(move) − PD(delay), wrapped (deg)'); ax.set_ylabel('units')
ax.set_title(f'E  PD stability across epochs (n={both.sum()})', loc='left', fontsize=11)

# (f) example cosine fit
ax = fig.add_subplot(gs[1, 2])
u = int(np.nanargmax(res['cos_r2_move'] * (res['depth_move']>10)))
th = np.deg2rad(bc[occ])
X = np.column_stack([np.ones_like(th), np.cos(th), np.sin(th)])
beta, *_ = np.linalg.lstsq(X, bm[u][occ], rcond=None)
thf = np.linspace(0, 2*np.pi, 200)
fitf = beta[0] + beta[1]*np.cos(thf) + beta[2]*np.sin(thf)
ax.plot(np.rad2deg(thf), fitf, 'k-', lw=1.5, label='cosine fit')
ax.plot(bc[occ], bm[u][occ], 'o', color='tab:blue', label='data')
ax.set_xlabel('reach direction (deg)'); ax.set_ylabel('firing rate (sp/s)')
ax.legend(fontsize=9)
ax.set_title(f'F  Cosine fit, unit {u} (R²={res["cos_r2_move"][u]:.2f})', loc='left', fontsize=11)

fig.suptitle('fig05  Population direction tuning: 182 units (86 M1, 96 PMd), MC_Maze sub-Jenkins', fontsize=13)
fig.savefig('figures/fig05_population_direction.png', dpi=150, bbox_inches='tight')
print("fig05 saved")
print("sig move:", sig.sum(), " sig delay:", (res['anova_p_delay']<0.01).sum(), " both:", both.sum())
print("median |PD diff| for both-sig:", round(np.nanmedian(np.abs(d)),1), "deg")
