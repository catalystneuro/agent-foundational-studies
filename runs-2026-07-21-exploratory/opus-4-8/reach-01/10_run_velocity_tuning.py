import numpy as np, pandas as pd, pickle
from importlib.machinery import SourceFileLoader
vt = SourceFileLoader('vt','09_velocity_tuning.py').load_module()
bm = SourceFileLoader('bm','07_binned_matrix.py').load_module()
M = pickle.load(open('./cache/binned.pkl','rb'))

lags = np.arange(-0.40, 0.52, 0.02)
sweep = vt.lag_sweep(M, lags)
md = np.array([r['md'] for r in sweep]); r2 = np.array([r['r2'] for r in sweep])
best_i = int(np.argmax(np.median(md, 1)))
print('population-median MD peaks at lag = %d ms' % round(lags[best_i]*1000))
per_unit_best = lags[np.argmax(md, axis=0)]
strong = md.max(0) > np.median(md.max(0))          # well-modulated half of the population
print('per-unit optimal lag (well-modulated units): median %d ms, IQR %s' %
      (round(np.median(per_unit_best[strong])*1000),
       np.round(np.percentile(per_unit_best[strong],[25,75])*1000)))
pickle.dump(dict(lags=lags, md=md, r2=r2, pds=np.array([r['pd'] for r in sweep]),
                 best_lag=lags[best_i], per_unit_best=per_unit_best, strong=strong), open('./cache/lagsweep.pkl','wb'))

LAG = lags[best_i]
i_n, i_k = bm.shift_by_lag(M, int(round(LAG/M['bin_size'])))
mov = M['speed'][i_k] > 100
C, A, V, S = M['counts'][i_n][mov], M['vel_angle'][i_k][mov], M['vel'][i_k][mov], M['speed'][i_k][mov]
fit = vt.cosine_fit_counts(C, A, M['bin_size'])
print('velocity-direction tuning: sig(p<0.01) %d/%d, median R2 %.3f, median MD %.2f Hz'
      % ((fit['p']<0.01).sum(), len(fit), fit['r2'].median(), fit['mod_depth'].median()))
edges, occ, maps = vt.tuning_2d(M['counts'][i_n], M['vel'][i_k], M['bin_size'])
sctr, stun = vt.speed_tuning(M['counts'][i_n], M['speed'][i_k], M['vel_angle'][i_k],
                             fit['pd'].values, M['bin_size'])
# speed modulation: slope of rate vs speed within PD
slopes = np.array([np.polyfit(sctr[~np.isnan(r)], r[~np.isnan(r)], 1)[0] if (~np.isnan(r)).sum()>4
                   else np.nan for r in stun])
print('speed slope (Hz per 100 mm/s): median %.2f, %% positive %.0f'
      % (np.nanmedian(slopes)*100, 100*np.mean(slopes>0)))
pickle.dump(dict(fit=fit, edges=edges, occ=occ, maps=maps, sctr=sctr, stun=stun,
                 slopes=slopes, LAG=LAG), open('./cache/veltuning.pkl','wb'))
