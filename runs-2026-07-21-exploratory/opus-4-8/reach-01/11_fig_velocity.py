import numpy as np, pandas as pd, pickle, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from importlib.machinery import SourceFileLoader
vt = SourceFileLoader('vt','09_velocity_tuning.py').load_module()
bm = SourceFileLoader('bm','07_binned_matrix.py').load_module()
M = pickle.load(open('./cache/binned.pkl','rb'))
V = pickle.load(open('./cache/veltuning.pkl','rb'))
L = pickle.load(open('./cache/lagsweep.pkl','rb'))
fit = V['fit']; units = M['units']

LAG = V['LAG']
i_n, i_k = bm.shift_by_lag(M, int(round(LAG/M['bin_size'])))
mov = M['speed'][i_k] > 100
C, A, S = M['counts'][i_n][mov], M['vel_angle'][i_k][mov], M['speed'][i_k][mov]

# empirical 1-D direction tuning curves (18 bins)
nb = 18; edges = np.linspace(-np.pi, np.pi, nb+1); ctr = 0.5*(edges[:-1]+edges[1:])
ib = np.clip(np.digitize(A, edges)-1, 0, nb-1)
tc = np.stack([[C[ib==b, u].mean()/M['bin_size'] for b in range(nb)] for u in range(C.shape[1])])
tc_se = np.stack([[C[ib==b, u].std()/np.sqrt((ib==b).sum())/M['bin_size'] for b in range(nb)]
                  for u in range(C.shape[1])])

order = np.argsort(-fit['r2'].values)[:4]
fig = plt.figure(figsize=(16, 10.5))
gs = fig.add_gridspec(3, 4, hspace=.75, wspace=.42, height_ratios=[1, 1, 1])
for j, u in enumerate(order):
    ax = fig.add_subplot(gs[0, j], projection='polar')
    ax.errorbar(np.r_[ctr, ctr[0]], np.r_[tc[u], tc[u][0]], yerr=np.r_[tc_se[u], tc_se[u][0]],
                color='k', lw=1.5, marker='.', ms=4)
    g = np.linspace(-np.pi, np.pi, 200)
    ax.plot(g, np.clip(fit['b0'][u] + fit['mod_depth'][u]*np.cos(g-fit['pd'][u]), 0, None), 'r', lw=2)
    ax.set_title(f'unit {units[u]}\nPD={np.degrees(fit["pd"][u]):.0f}$\\degree$, '
                 f'MD={fit["mod_depth"][u]:.1f} Hz', pad=24, fontsize=10)
    ax.set_rlabel_position(np.degrees(fit['pd'][u])+150); ax.tick_params(labelsize=7)
fig.text(0.5, 0.975, 'Firing rate vs. instantaneous hand-velocity direction '
         f'(20 ms bins, speed > 100 mm/s, neural lead {LAG*1000:.0f} ms)',
         ha='center', fontsize=12)

# 2-D velocity maps for the same units
e = V['edges']; ext = [e[0], e[-1], e[0], e[-1]]
for j, u in enumerate(order):
    ax = fig.add_subplot(gs[1, j])
    im = ax.imshow(V['maps'][u], origin='lower', extent=ext, cmap='viridis')
    ax.arrow(0, 0, 380*np.cos(fit['pd'][u]), 380*np.sin(fit['pd'][u]), color='w',
             width=14, head_width=48, length_includes_head=True)
    ax.set(xlabel='$v_x$ (mm/s)', ylabel='$v_y$ (mm/s)', title=f'unit {units[u]}: 2-D velocity map')
    plt.colorbar(im, ax=ax, label='rate (Hz)', fraction=.046)

ax = fig.add_subplot(gs[2, 0])
ax.plot(L['lags']*1000, np.median(L['md'], 1), 'k-o', ms=3)
ax.axvline(LAG*1000, color='r', ls='--', label=f'peak {LAG*1000:.0f} ms')
ax.set(xlabel='neural lead time (ms)', ylabel='median modulation depth (Hz)',
       title='Direction tuning vs. neural lead'); ax.legend(fontsize=8)

ax = fig.add_subplot(gs[2, 1])
pub = L['per_unit_best'][L['strong']]
ax.hist(pub*1000, bins=np.arange(-410, 530, 40), color='0.4')
ax.axvline(np.median(pub)*1000, color='r', ls='--', label=f'median {np.median(pub)*1000:.0f} ms')
ax.set(xlabel='per-unit optimal lead (ms)', ylabel='units', title='Optimal lead (well-modulated units)')
ax.legend(fontsize=8)

ax = fig.add_subplot(gs[2, 2])
sc, st = V['sctr'], V['stun']
norm = st / np.nanmax(st, 1, keepdims=True)
for r in norm[np.argsort(-fit['r2'].values)[:40]]:
    ax.plot(sc, r, color='0.75', lw=.7)
ax.plot(sc, np.nanmean(norm, 0), 'r-o', lw=2.2, ms=4, label='population mean')
ax.set(xlabel='hand speed (mm/s)', ylabel='normalized rate',
       title='Speed tuning within each unit\'s PD ($\\pm45\\degree$)'); ax.legend(fontsize=8)

ax = fig.add_subplot(gs[2, 3], projection='polar')
sig = fit['p'].values < 0.01
ax.hist(fit['pd'].values[sig], bins=np.linspace(-np.pi, np.pi, 25), color='steelblue')
ax.set_title(f'Preferred velocity directions\n({sig.sum()}/{len(fit)} units, p<0.01)', pad=24,
             fontsize=10)
ax.set_rlabel_position(255); ax.tick_params(labelsize=8)
plt.savefig('fig03_velocity_tuning.png', dpi=135, bbox_inches='tight')
print('speed tuning pop mean:', np.round(np.nanmean(norm,0),3))
print('rate at slowest vs fastest speed bin (pop mean, Hz):',
      np.nanmean(st[:,0]), np.nanmean(st[:,-1]))
