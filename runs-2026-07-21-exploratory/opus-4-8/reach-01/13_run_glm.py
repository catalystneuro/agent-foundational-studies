import numpy as np, pandas as pd, pickle, time
from importlib.machinery import SourceFileLoader
gm = SourceFileLoader('gm','12_glm_models.py').load_module()
bm = SourceFileLoader('bm','07_binned_matrix.py').load_module()
M = pickle.load(open('./cache/binned.pkl','rb'))
V = pickle.load(open('./cache/veltuning.pkl','rb'))
LAG = V['LAG']; print('using neural lead', LAG)

i_n, i_k = bm.shift_by_lag(M, int(round(LAG/M['bin_size'])))
mov = M['speed'][i_k] > 50.0
Y = M['counts'][i_n][mov].astype(float)
feats = dict(angle=M['vel_angle'][i_k][mov], speed=M['speed'][i_k][mov],
             x=M['pos'][i_k][mov, 0], y=M['pos'][i_k][mov, 1])
groups = M['trial'][i_n][mov]
print('design bins', Y.shape, 'trials', len(np.unique(groups)))

res = {}
for name, (basis, names) in gm.make_bases().items():
    t0 = time.time()
    X = gm.design(basis, names, feats)
    de = gm.fit_cv(X, Y, groups)
    res[name] = de
    print(f'{name:26s} n_feat={X.shape[1]:3d}  median held-out dev. expl. = {np.median(de):.4f} '
          f'({time.time()-t0:.0f}s)')
pickle.dump(dict(res=res, units=M['units']), open('./cache/glm_results.pkl','wb'))
