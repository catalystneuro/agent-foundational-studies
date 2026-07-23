import numpy as np, pandas as pd, pynapple as nap
from importlib.machinery import SourceFileLoader
exec(open('01_load_data.py').read().split("if __name__")[0])
pp = SourceFileLoader('pp','02_preprocess.py').load_module()
dt_ = SourceFileLoader('dt','04_direction_tuning.py').load_module()

d = load_session(); spikes = d['spikes']
tr = pd.read_pickle('./cache/trials.pkl')
ang = tr['reach_angle'].values

# Held-in units with a reasonable firing rate
keep = np.array([u for u in spikes.index if spikes[u].rate > 1.0])
print('units kept', len(keep), 'of', len(spikes))
sp = spikes[list(keep)]

epochs_def = {'baseline': ('target_on_time', -0.25, -0.05),
              'preparatory': ('go_cue_time', -0.30, 0.0),
              'perimovement': ('move_onset_time', -0.05, 0.25),
              'late_movement': ('move_onset_time', 0.25, 0.55)}
res = {}
for name, (ref, t0, t1) in epochs_def.items():
    R = dt_.trial_rates(sp, tr, ref, t0, t1)
    fit = dt_.cosine_fit(R, ang); fit['unit'] = keep; fit['epoch'] = name
    res[name] = fit
    np.save(f'./cache/rates_{name}.npy', R)
    print(f"{name:14s} sig(p<0.01): {(fit['p']<0.01).sum():3d}/{len(fit)}  "
          f"median R2={fit['r2'].median():.3f}  median MD={fit['mod_depth'].median():.2f} Hz")
allfits = pd.concat(res.values()); allfits.to_pickle('./cache/dir_fits.pkl')
np.save('./cache/keep_units.npy', keep)

# Straight (barrier-free) trials only: target direction == actual reach direction
m = tr['is_straight'].values
R = dt_.trial_rates(sp, tr[m], 'move_onset_time', -0.05, 0.25)
fs = dt_.cosine_fit(R, ang[m])
print('STRAIGHT only  sig(p<0.01): %d/%d  median R2=%.3f  median MD=%.2f Hz'
      % ((fs['p']<0.01).sum(), len(fs), fs['r2'].median(), fs['mod_depth'].median()))
fs['unit']=keep; fs.to_pickle('./cache/dir_fits_straight.pkl')
Rc = dt_.trial_rates(sp, tr[~m], 'move_onset_time', -0.05, 0.25)
fc = dt_.cosine_fit(Rc, ang[~m])
print('CURVED   only  sig(p<0.01): %d/%d  median R2=%.3f  median MD=%.2f Hz'
      % ((fc['p']<0.01).sum(), len(fc), fc['r2'].median(), fc['mod_depth'].median()))
fc['unit']=keep; fc.to_pickle('./cache/dir_fits_curved.pkl')
dpd = np.degrees(np.angle(np.exp(1j*(fs['pd'].values - fc['pd'].values))))
print('PD straight vs curved: circ corr of |diff| median %.1f deg' % np.median(np.abs(dpd)))
