import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from importlib.machinery import SourceFileLoader
exec(open('01_load_data.py').read().split("if __name__")[0])
pp = SourceFileLoader('pp','02_preprocess.py').load_module()
dt_ = SourceFileLoader('dt','04_direction_tuning.py').load_module()

d = load_session(); spikes = d['spikes']
tr = pd.read_pickle('./cache/trials.pkl')
keep = np.load('./cache/keep_units.npy')
fits = pd.read_pickle('./cache/dir_fits_straight.pkl').set_index('unit')
strt = tr[tr['is_straight']]

# most strongly tuned units, spread over preferred directions
top = fits.sort_values('r2', ascending=False)
sel, used = [], []
for u, row in top.iterrows():
    if all(abs(np.angle(np.exp(1j*(row['pd']-p)))) > np.pi/4 for p in used):
        sel.append(u); used.append(row['pd'])
    if len(sel) == 3: break
print('example units', sel, np.degrees(used))

cmap = plt.get_cmap('hsv'); centers = pp.dir_bin_centers()
bins_present = sorted(strt['dir_bin'].unique())
fig = plt.figure(figsize=(15, 4.4 * len(sel) + 0.8))
gs = fig.add_gridspec(len(sel), 3, width_ratios=[1.5, 1.2, 1.0], hspace=.5, wspace=.32,
                      top=0.90, bottom=0.06)
handles = []
for r, u in enumerate(sel):
    st = np.asarray(spikes[u].index)
    ax_r = fig.add_subplot(gs[r, 0]); ax_p = fig.add_subplot(gs[r, 1])
    ax_t = fig.add_subplot(gs[r, 2], projection='polar')
    y = 0
    for b in bins_present:
        ev = strt.loc[strt['dir_bin'] == b, 'move_onset_time'].values
        col = cmap((centers[b] % (2*np.pi))/(2*np.pi))
        tt, rate, per = dt_.psth(st, ev[:40], window=(-0.6, 0.8), sigma=0)
        for rel in per:
            ax_r.plot(rel, np.full_like(rel, y), '|', color=col, ms=3, mew=.7); y += 1
        tt, rate, _ = dt_.psth(st, ev, window=(-0.9, 1.1))   # wide, then trim edge artifacts
        m_ = (tt >= -0.6) & (tt <= 0.8)
        ln, = ax_p.plot(tt[m_], rate[m_], color=col, lw=1.8,
                        label=f'{np.degrees(centers[b]):.0f}$\\degree$')
        if r == 0: handles.append(ln)
        y += 4
    for a in (ax_r, ax_p):
        a.axvline(0, color='k', ls='--', lw=1)
        a.set_xlabel('time from move onset (s)'); a.set_xlim(-0.6, 0.8)
    ax_r.set(ylabel='trials (grouped by direction)', title=f'unit {u}: raster')
    ax_r.set_yticks([])
    ax_p.set(ylabel='firing rate (Hz)', title=f'unit {u}: direction-conditioned PSTH')

    # polar tuning curve, peri-movement window
    R = dt_.trial_rates(spikes[[u]], strt, 'move_onset_time', -0.05, 0.25)[:, 0]
    mu = np.array([R[strt['dir_bin'].values == b].mean() for b in bins_present])
    se = np.array([R[strt['dir_bin'].values == b].std()/np.sqrt((strt['dir_bin'].values==b).sum())
                   for b in bins_present])
    th = centers[bins_present]
    ax_t.errorbar(np.r_[th, th[0]], np.r_[mu, mu[0]], yerr=np.r_[se, se[0]],
                  color='k', marker='o', ms=4, lw=1.5, capsize=2)
    f = fits.loc[u]
    g = np.linspace(-np.pi, np.pi, 200)
    ax_t.plot(g, np.clip(f['b0'] + f['mod_depth']*np.cos(g - f['pd']), 0, None), 'r-', lw=2)
    ax_t.set_title(f"unit {u}: cosine fit\nPD={np.degrees(f['pd']):.0f}$\\degree$, "
                   f"MD={f['mod_depth']:.1f} Hz, $R^2$={f['r2']:.2f}", pad=26, fontsize=10)
    ax_t.set_rlabel_position(200); ax_t.tick_params(labelsize=8)
fig.legend(handles=handles, ncol=len(handles), loc='upper center', bbox_to_anchor=(0.5, 0.975),
           title='reach direction (target angle)', fontsize=9, title_fontsize=10)
fig.suptitle('Reach-direction tuning in macaque M1/PMd (MC_Maze, barrier-free trials)',
             y=0.995, fontsize=13)
plt.savefig('fig02_example_units_direction.png', dpi=140)
