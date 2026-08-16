import numpy as np, pandas as pd, pynapple as nap, pickle
from importlib.machinery import SourceFileLoader
exec(open('01_load_data.py').read().split("if __name__")[0])
bm = SourceFileLoader('bm','07_binned_matrix.py').load_module()

d = load_session()
cz = np.load('./cache/cursor.npz')
cursor = nap.TsdFrame(t=cz['t'], d=cz['cp'], columns=['x','y'])
tr = pd.read_pickle('./cache/trials.pkl')
keep = np.load('./cache/keep_units.npy')
sp = d['spikes'][list(keep)]
M = bm.build(sp, d['hand_vel'], cursor, tr)
M['units'] = keep
print({k: (np.shape(v) if hasattr(v,'__len__') else v) for k,v in M.items()})
print('total bins', len(M['t']), 'moving bins (>100mm/s)', (M['speed']>100).sum())
print('NaNs', np.isnan(M['vel']).sum(), np.isnan(M['pos']).sum())
pickle.dump(M, open('./cache/binned.pkl','wb'))
