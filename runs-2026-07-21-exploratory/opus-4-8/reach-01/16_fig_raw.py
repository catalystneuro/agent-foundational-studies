import numpy as np, pandas as pd, pickle, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, pynapple as nap
exec(open('01_load_data.py').read().split("if __name__")[0])
d = load_session(); spikes = d['spikes']
cz = np.load('./cache/cursor.npz'); cursor = nap.TsdFrame(t=cz['t'], d=cz['cp'], columns=['x','y'])
tr = pd.read_pickle('./cache/trials.pkl')
keep = np.load('./cache/keep_units.npy')
V = pickle.load(open('./cache/veltuning.pkl','rb'))
order = np.argsort(V['fit']['pd'].values)      # sort units by preferred velocity direction
keep = keep[order]; pds = V['fit']['pd'].values[order]

t0, t1 = tr['start_time'].iloc[8], tr['stop_time'].iloc[15]
ep = nap.IntervalSet(start=t0, end=t1)
fig, ax = plt.subplots(4, 1, figsize=(14, 9.4), sharex=True,
                       gridspec_kw=dict(height_ratios=[3, 1, 1, 1], hspace=.12, top=0.93))
for i, u in enumerate(keep):
    s = np.asarray(spikes[u].restrict(ep).index)
    ax[0].plot(s, np.full_like(s, i), '|', color=plt.get_cmap('hsv')((pds[i] % (2*np.pi))/(2*np.pi)),
               ms=2.8, mew=.7)
ax[0].set(ylabel='unit (sorted by preferred\nvelocity direction)')
c = cursor.restrict(ep)
ax[1].plot(c.index, c.values[:, 0], label='x'); ax[1].plot(c.index, c.values[:, 1], label='y')
ax[1].set(ylabel='cursor\nposition (mm)'); ax[1].legend(ncol=2, fontsize=8, loc='upper right')
v = d['hand_vel'].restrict(ep)
ax[2].plot(v.index, v.values[:, 0], label='$v_x$'); ax[2].plot(v.index, v.values[:, 1], label='$v_y$')
ax[2].set(ylabel='hand\nvelocity (mm/s)'); ax[2].legend(ncol=2, fontsize=8, loc='upper right')
sp = np.hypot(*v.values.T)
ax[3].plot(v.index, sp, 'k'); ax[3].set(ylabel='speed\n(mm/s)', xlabel='time (s)')
for a in ax:
    for _, r in tr.iloc[8:16].iterrows():
        a.axvline(r['target_on_time'], color='tab:green', lw=.8, alpha=.7)
        a.axvline(r['go_cue_time'], color='tab:orange', lw=.8, alpha=.7)
        a.axvline(r['move_onset_time'], color='tab:red', lw=.9, alpha=.8)
    a.set_xlim(t0, t1)
h = [plt.Line2D([], [], color=c_, label=l) for c_, l in
     [('tab:green', 'target on'), ('tab:orange', 'go cue'), ('tab:red', 'move onset')]]
fig.suptitle('DANDI:000128 MC_Maze, monkey Jenkins - M1/PMd population activity and hand '
             'kinematics (8 consecutive trials)', y=0.985)
fig.legend(handles=h, ncol=3, fontsize=9, loc='upper center', bbox_to_anchor=(0.5, 0.962),
           frameon=False)
plt.savefig('fig00_raw_session.png', dpi=140, bbox_inches='tight')
