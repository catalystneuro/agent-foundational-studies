"""Figures for orientation-selectivity demonstration (single prototype session)."""
import pickle, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pynapple as nap
from matplotlib.gridspec import GridSpec
from analysis_lib import load, ori_metrics

plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 130, 'savefig.dpi': 150, 'savefig.bbox': 'tight'})
REG_COLORS = {'VISp':'#1f77b4','VISl':'#2ca02c','VISrl':'#9467bd',
              'VISam':'#8c564b','VISpm':'#e377c2','LGd':'#d62728'}

d, tsg = load('cache_715093703.pkl')
res = pickle.load(open('results.pkl','rb'))
dg, sg = res['dg'], res['sg']
reg = np.array(d['meta']['region'])
tbl = d['stim']['drifting_gratings_presentations']
M = dg['metrics']
dirs = dg['dirs']

# ---------------- Fig 1: raw activity + stimulus ----------------
ori_all = tbl['orientation'].astype(float)
blk = tbl['stimulus_block'].astype(float)
b0 = blk == 2.0
t0 = tbl['start_time'][b0][0]; t1 = t0 + 40
sel_tr = (tbl['start_time'] >= t0) & (tbl['start_time'] < t1)

visp = np.where(reg == 'VISp')[0]
lgd = np.where(reg == 'LGd')[0]
uids = np.array(list(tsg.keys()))
order = np.concatenate([visp, lgd])

fig = plt.figure(figsize=(11, 6.5))
gs = GridSpec(3, 1, height_ratios=[1, 3.6, 1], hspace=0.32)
ax = fig.add_subplot(gs[0])
dir_list = np.arange(0, 360, 45)
dir_col = {dd: plt.cm.twilight(i/len(dir_list)) for i, dd in enumerate(dir_list)}
for k in np.where(sel_tr)[0]:
    o = ori_all[k]
    c = dir_col[o] if not np.isnan(o) else '0.85'
    ax.axvspan(tbl['start_time'][k]-t0, tbl['stop_time'][k]-t0, color=c, alpha=0.9, lw=0)
    if not np.isnan(o):
        ax.text((tbl['start_time'][k]+tbl['stop_time'][k])/2-t0, 0.5, f'{int(o)}',
                ha='center', va='center', fontsize=6.5, color='w', fontweight='bold')
ax.set_xlim(0, t1-t0); ax.set_yticks([]); ax.set_ylim(0, 1)
ax.set_title('Drifting-grating presentations (labels = direction of motion in deg; grey = blank sweep)')
ax.set_ylabel('stim')

ax = fig.add_subplot(gs[1])
ep = nap.IntervalSet(start=t0, end=t1)
for row, u in enumerate(order):
    s = tsg[uids[u]].restrict(ep).t - t0
    ax.plot(s, np.full_like(s, row), '|', ms=2.6,
            color=REG_COLORS[reg[u]], alpha=0.9, mew=0.5)
ax.axhline(len(visp)-0.5, color='k', lw=0.8, ls='--')
ax.set_xlim(0, t1-t0); ax.set_ylim(-1, len(order))
ax.set_ylabel('unit')
ax.text(1.005, 0.98, 'LGd', color=REG_COLORS['LGd'], ha='left', va='top',
        transform=ax.transAxes, fontweight='bold', rotation=90)
ax.text(1.005, 0.02, 'VISp', color=REG_COLORS['VISp'], ha='left', va='bottom',
        transform=ax.transAxes, fontweight='bold', rotation=90)

ax = fig.add_subplot(gs[2])
rt, rv = d['running']['t'], d['running']['v']
k = (rt >= t0) & (rt <= t1)
ax.plot(rt[k]-t0, rv[k], color='0.35', lw=0.8)
ax.set_xlim(0, t1-t0); ax.set_xlabel('time from block onset (s)')
ax.set_ylabel('running\n(cm/s)')
fig.suptitle(f"Session {d['session_id']} — raw spiking during drifting gratings", y=0.955)
fig.savefig('fig01_raw_activity.png'); plt.close(fig)
print('fig01 done')

# ---------------- Fig 2: example unit raster + PSTH by direction ----------------
sig = (M['anova_p'] < 0.01) & (M['perm_p'] < 0.05)
cand = np.where((reg == 'VISp') & sig & (M['evoked'] > 2))[0]
best = cand[np.argsort(-M['gOSI'][cand])[0]]
uid = uids[best]
ptf = M['pref_tf'][best]
tf_all = tbl['temporal_frequency'].astype(float)

fig, axes = plt.subplots(2, 8, figsize=(13.5, 4.6), sharex=True,
                         gridspec_kw={'height_ratios':[2,1], 'hspace':0.25, 'wspace':0.12})
win = (-0.3, 2.3)
for i, dd in enumerate(dirs):
    tr = np.where((ori_all == dd) & (tf_all == ptf))[0]
    starts = nap.Ts(t=tbl['start_time'][tr])
    pe = nap.compute_perievent(tsg[uid], starts, window=win)
    axr, axp = axes[0, i], axes[1, i]
    for j in range(len(pe)):
        tt = pe[j].t
        axr.plot(tt, np.full_like(tt, j), '|', ms=3.5, color='k', mew=0.7)
    axr.axvspan(0, 2, color='#ffd27f', alpha=0.35, lw=0, zorder=0)
    axr.set_title(f'{int(dd)}°', pad=4)
    axr.set_xlim(*win); axr.set_ylim(-0.5, len(pe)-0.5)
    if i: axr.set_yticks([])
    else: axr.set_ylabel('trial')
    edges = np.arange(win[0], win[1]+1e-9, 0.05)
    allsp = np.concatenate([pe[j].t for j in range(len(pe))])
    h, _ = np.histogram(allsp, bins=edges)
    axp.bar(edges[:-1], h/(len(pe)*0.05), width=0.05, align='edge', color=REG_COLORS['VISp'])
    axp.axvspan(0, 2, color='#ffd27f', alpha=0.35, lw=0, zorder=0)
    axp.set_xlim(*win)
    if i: axp.set_yticks([])
    else: axp.set_ylabel('rate (Hz)')
    axp.set_xlabel('t (s)')
ymax = max(a.get_ylim()[1] for a in axes[1])
for a in axes[1]: a.set_ylim(0, ymax)
fig.suptitle(f"VISp unit {uid}: direction-dependent responses "
             f"(TF = {ptf:g} Hz, gOSI = {M['gOSI'][best]:.2f}, DSI = {M['DSI'][best]:.2f})", y=1.0)
fig.savefig('fig02_example_raster_psth.png'); plt.close(fig)
print('fig02 done, unit', uid)

# ---------------- Fig 3: polar tuning curves, VISp examples vs LGd ----------------
def polar_panel(ax, u, color, title):
    m = dg['tc'][:, u]; e = dg['tc_sem'][:, u]
    th = np.deg2rad(np.append(dirs, dirs[0]))
    r = np.append(m, m[0]); ee = np.append(e, e[0])
    ax.plot(th, r, '-o', color=color, ms=3, lw=1.4)
    ax.fill_between(th, r-ee, r+ee, color=color, alpha=0.25)
    base = M['baseline'][u]
    ax.plot(np.linspace(0, 2*np.pi, 100), np.full(100, base), ':', color='0.4', lw=1)
    ax.set_title(title, pad=24, fontsize=8)
    ax.set_yticklabels([]); ax.set_xticks(np.deg2rad(dirs))
    ax.set_xticklabels([f'{int(x)}' for x in dirs], fontsize=6.5)
    ax.grid(alpha=0.35)

vp = np.where((reg == 'VISp') & sig)[0]; vp = vp[np.argsort(-M['gOSI'][vp])][:8]
lg = np.where(reg == 'LGd')[0]; lg = lg[np.argsort(-M['evoked'][lg])][:4]
fig, axes = plt.subplots(3, 4, figsize=(11, 9.4), subplot_kw={'projection':'polar'})
for a, u in zip(axes.ravel()[:8], vp):
    polar_panel(a, u, REG_COLORS['VISp'],
                f"VISp {uids[u]}\ngOSI={M['gOSI'][u]:.2f} DSI={M['DSI'][u]:.2f}")
for a, u in zip(axes.ravel()[8:], lg):
    polar_panel(a, u, REG_COLORS['LGd'],
                f"LGd {uids[u]}\ngOSI={M['gOSI'][u]:.2f} DSI={M['DSI'][u]:.2f}")
fig.suptitle('Direction tuning curves (mean ± SEM firing rate at preferred TF; dotted = blank-sweep baseline)',
             y=0.985)
fig.subplots_adjust(hspace=0.62, wspace=0.35, top=0.89, bottom=0.04)
fig.savefig('fig03_polar_tuning.png'); plt.close(fig)
print('fig03 done')
