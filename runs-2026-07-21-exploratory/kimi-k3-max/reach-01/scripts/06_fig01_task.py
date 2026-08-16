import numpy as np, pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

with open('data/session_cache.pkl','rb') as f:
    C = pickle.load(f)
tr, un, bh = C['trials'], C['units'], C['behav']

t = bh['hand_pos_t']; pos = bh['hand_pos']; vel = bh['hand_vel']
speed = np.hypot(vel[:,0], vel[:,1]) / 10.0  # native mm/s -> cm/s

fig = plt.figure(figsize=(13, 9))
gs = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.3)

# (a) target layout + example trajectories
ax = fig.add_subplot(gs[0, 0])
ang = tr['target_angle']
cmap = plt.cm.hsv
# plot 60 example reaches: from pos at move_onset-0.05 to move_onset+0.5
rng = np.random.default_rng(0)
idx = rng.choice(len(tr['start']), 60, replace=False)
for i in idx:
    t0, t1 = tr['move_onset'][i]-0.05, tr['move_onset'][i]+0.5
    m = (t >= t0) & (t <= t1)
    ax.plot(pos[m,0], pos[m,1], color=cmap(ang[i]/360), lw=0.7, alpha=0.6)
ax.scatter(tr['target_xy'][:,0], tr['target_xy'][:,1], c=ang/360, cmap='hsv', s=14, edgecolors='k', linewidths=0.3, zorder=5)
ax.scatter([0],[0], marker='+', c='k', s=80)
ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)')
ax.set_title('A  Reach trajectories & targets', loc='left', fontsize=11)
ax.set_aspect('equal')

# (b) example trial timeline: speed with events
ax = fig.add_subplot(gs[0, 1])
i = 100
t0, t1 = tr['start'][i]-0.2, tr['stop'][i]+0.3
m = (t >= t0) & (t <= t1)
ax.plot(t[m]-tr['go_cue'][i], speed[m], 'k', lw=1.2)
for ev, c, lbl in [(tr['target_on'][i],'tab:blue','target on'), (tr['go_cue'][i],'tab:green','go cue'), (tr['move_onset'][i],'tab:red','move onset')]:
    ax.axvline(ev-tr['go_cue'][i], color=c, ls='--', lw=1)
    ax.text(ev-tr['go_cue'][i]+0.02, 78, lbl, rotation=90, va='top', fontsize=8, color=c)
ax.set_xlabel('time from go cue (s)'); ax.set_ylabel('hand speed (cm/s)')
ax.set_title('B  Example trial speed profile', loc='left', fontsize=11)

# (c) RT and reach duration distributions
ax = fig.add_subplot(gs[0, 2])
dur = (tr['stop'] - tr['move_onset'])*1000
ax.hist(tr['rt'], bins=40, alpha=0.7, label='reaction time', color='tab:blue')
ax.hist(dur, bins=40, alpha=0.7, label='reach duration', color='tab:red')
ax.set_xlabel('duration (ms)'); ax.set_ylabel('trials')
ax.legend(fontsize=9)
ax.set_title('C  Trial timing', loc='left', fontsize=11)

# (d) mean speed aligned to move onset, by direction quartile... simple: all trials
ax = fig.add_subplot(gs[1, 0])
win = np.arange(-0.3, 0.8, 0.001)
prof = np.full((len(tr['start']), len(win)), np.nan)
for i in range(len(tr['start'])):
    m0 = np.searchsorted(t, tr['move_onset'][i]+win[0])
    m1 = m0 + len(win)
    if m1 <= len(t):
        prof[i] = speed[m0:m1]
ax.plot(win, np.nanmean(prof,0), 'k', lw=1.5)
ax.fill_between(win, np.nanmean(prof,0)-np.nanstd(prof,0)/np.sqrt(len(prof)),
                np.nanmean(prof,0)+np.nanstd(prof,0)/np.sqrt(len(prof)), alpha=0.3)
ax.axvline(0, color='tab:red', ls='--', lw=1)
ax.set_xlabel('time from move onset (s)'); ax.set_ylabel('hand speed (cm/s)')
ax.set_title('D  Speed aligned to move onset', loc='left', fontsize=11)

# (e) raw snippet: raster + velocity
ax = fig.add_subplot(gs[1, 1:])
snip_t0 = 600.0; snip_t1 = 608.0
# pick 25 units with most spikes
nsp = np.array([len(s) for s in un['spike_times']])
order = np.argsort(nsp)[::-1][:25]
ypos = 0
yt = []
for k, u in enumerate(order):
    st = un['spike_times'][u]
    st = st[(st >= snip_t0) & (st < snip_t1)]
    ax.vlines(st, ypos, ypos+0.8, color='k', lw=0.5)
    yt.append(ypos+0.4)
    ypos += 1
m = (t >= snip_t0) & (t < snip_t1)
scale = 0.06  # cm/s per raster-row
ax.plot(t[m], -3 + vel[m,0]/10*scale*2, color='tab:blue', lw=1, label='vx')
ax.plot(t[m], -9 + vel[m,1]/10*scale*2, color='tab:orange', lw=1, label='vy')
ax.plot(t[m], -16 + speed[m]*scale*2, color='tab:red', lw=1.2, label='speed')
for s in tr['start']:
    if snip_t0 <= s < snip_t1: ax.axvline(s, color='gray', ls=':', lw=0.7)
for mo in tr['move_onset']:
    if snip_t0 <= mo < snip_t1: ax.axvline(mo, color='tab:red', ls=':', lw=0.7)
ax.set_yticks([])
ax.set_xlabel('time (s)')
ax.legend(loc='lower right', fontsize=8, ncol=3, framealpha=0.9)
ax.set_title('E  Spike raster (25 units) with hand velocity — gray: trial start, red: move onset', loc='left', fontsize=11)
ax.set_xlim(snip_t0, snip_t1)

fig.suptitle('MC_Maze (DANDI 000128, sub-Jenkins): delayed reaching task overview', fontsize=13)
fig.savefig('figures/fig01_task_behavior.png', dpi=150, bbox_inches='tight')
print("saved fig01")
