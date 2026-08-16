import numpy as np, pandas as pd, nemos as nmo, jax, time
import ov_data as ov, ov_analysis as oa
import pynapple as nap
jax.config.update("jax_enable_x64", True)

spikes, info = ov.load_spikes(ov.SESSIONS[0])
dg = ov.stimulus_table(info['nwbfile'],'drifting_gratings_presentations')
running = ov.running_speed(info['nwbfile'])
res = oa.dg_tuning(spikes, dg)
tbl = res['trial_table']

# mean running speed within each trial
ep = nap.IntervalSet(start=tbl['start_time'].values, end=tbl['stop_time'].values)
spd = np.array([np.nanmean(running.restrict(ep[i:i+1]).d) for i in range(len(ep))])
print('speed nan?', np.isnan(spd).sum(), spd.min(), spd.max())

dirs = res['dirs']; tfs = res['tfs']
counts = (res['rates'] * 2.0).round().astype(int)   # 2 s windows

dir_basis = nmo.basis.CyclicBSplineEval(n_basis_funcs=8, bounds=(0., 360.), label='direction')
tf_basis  = nmo.basis.BSplineEval(n_basis_funcs=4, label='tf')
spd_basis = nmo.basis.BSplineEval(n_basis_funcs=4, label='speed')
full = dir_basis + tf_basis + spd_basis
nuis = tf_basis + spd_basis
ltf = np.log2(tfs)
Xf = full.compute_features(dirs, ltf, spd)
Xn = nuis.compute_features(ltf, spd)
print('Xf', Xf.shape, 'Xn', Xn.shape, np.isnan(Xf).sum())

from sklearn.model_selection import KFold
def cv_ll(X, y, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=0); tot=[]
    for tr, te in kf.split(y):
        m = nmo.glm.GLM(solver_name='LBFGS', regularizer='Ridge', regularizer_strength=1e-3)
        if X is None:
            lam = np.full(len(te), max(y[tr].mean(), 1e-6))
        else:
            m.fit(X[tr], y[tr]); lam = np.clip(np.asarray(m.predict(X[te])), 1e-8, None)
        from scipy.special import gammaln
        tot.append(np.mean(y[te]*np.log(lam) - lam - gammaln(y[te]+1)))
    return float(np.mean(tot))

t0=time.time()
i = int(np.where(res['uids']==951877391)[0][0])
y = counts[i].astype(float)
print('full', cv_ll(Xf,y), 'nuis', cv_ll(Xn,y), 'null', cv_ll(None,y), '(%.1fs)'%(time.time()-t0))
# fitted tuning curve
m = nmo.glm.GLM(solver_name='LBFGS', regularizer='Ridge', regularizer_strength=1e-3).fit(Xf, y)
grid_dir = np.linspace(0,360,181)
Xg = full.compute_features(grid_dir, np.full_like(grid_dir, np.log2(res['pref_tf'][i])),
                           np.full_like(grid_dir, np.median(spd)))
pred = np.asarray(m.predict(Xg))/2.0
print('pred tuning peak at', grid_dir[np.argmax(pred)], pred.max(), 'min', pred.min())
