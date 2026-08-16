import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, pynapple as nap
exec(open('01_load_data.py').read().split("if __name__")[0])
from importlib.machinery import SourceFileLoader
pp = SourceFileLoader('pp','02_preprocess.py').load_module()

d = load_session()
cz = np.load('./cache/cursor.npz')
cursor = nap.TsdFrame(t=cz['t'], d=cz['cp'], columns=['x','y'])
tr = pp.trial_geometry(d['trials'], cursor)
tr.to_pickle('./cache/trials.pkl')

cmap = plt.get_cmap('hsv')
centers = pp.dir_bin_centers()
colors = [cmap((c % (2*np.pi))/(2*np.pi)) for c in centers]

fig, axes = plt.subplots(1, 4, figsize=(19, 4.6))
for ax, straight, ttl in [(axes[0], True, 'Barrier-free reaches (straight)'),
                          (axes[1], False, 'Maze reaches (curved)')]:
    sub = tr[tr['is_straight'] == straight]
    for _, row in sub.sample(min(240, len(sub)), random_state=0).iterrows():
        seg = cursor.get(row['move_onset_time'], row['move_onset_time'] + 0.8).values
        ax.plot(seg[:,0], seg[:,1], lw=0.6, alpha=0.55, color=colors[int(row['dir_bin'])])
    ax.set(title=f'{ttl}  (n={len(sub)})', xlabel='x (mm)', ylabel='y (mm)', aspect='equal')
    ax.set_xlim(-160,160); ax.set_ylim(-160,160)

t0 = tr['move_onset_time'].values
speed = nap.Tsd(t=d['hand_vel'].index, d=np.hypot(*d['hand_vel'].values.T))
lags, pe = pp.perievent_continuous(speed, t0, window=(-0.4, 0.8))
pe = pe[:, :, 0]
axes[2].plot(lags, np.nanmedian(pe, 1), 'k', lw=2)
axes[2].fill_between(lags, *np.nanpercentile(pe, [25,75], axis=1), alpha=.25, color='k')
axes[2].axvline(0, color='r', ls='--', label='move onset')
axes[2].set(xlabel='time from move onset (s)', ylabel='hand speed (mm/s)',
            title='Speed profile (median $\\pm$ IQR)'); axes[2].legend()

ax = plt.subplot(1, 4, 4, projection='polar'); axes[3].remove()
cnt, _ = np.histogram(tr['reach_angle'], bins=np.linspace(-np.pi, np.pi, 33))
ax.bar(np.linspace(-np.pi, np.pi, 33)[:-1] + np.pi/32, cnt, width=2*np.pi/32,
       color=[cmap(((a+np.pi/32) % (2*np.pi))/(2*np.pi)) for a in np.linspace(-np.pi,np.pi,33)[:-1]])
ax.set_title('Reach direction distribution\n(n=%d trials)' % len(tr), pad=22)
ax.set_rlabel_position(255); ax.tick_params(labelsize=9)
plt.tight_layout(); plt.savefig('fig01_behavior_overview.png', dpi=140)
print(tr.groupby('dir_bin').size())
print('straight/curved', tr['is_straight'].value_counts().to_dict())
