import numpy as np, pandas as pd, pickle
from importlib.machinery import SourceFileLoader
dc = SourceFileLoader('dc','14_decoding.py').load_module()
M = pickle.load(open('./cache/binned.pkl','rb'))
tr = pd.read_pickle('./cache/trials.pkl')

# restrict to peri-movement bins so the decoder is trained on reaching, not inter-trial rest
mo = tr['move_onset_time'].values[M['trial']]
peri = (M['t'] - mo > -0.2) & (M['t'] - mo < 0.8)
print('peri-movement bins', peri.sum(), 'of', len(peri))
C = M['counts'][peri].astype(np.float32); T = M['trial'][peri]; Vd = M['vel'][peri]
lags = np.arange(0, 12, 2)                      # 0-220 ms of neural history
pred, Y, g, r2, good = dc.decode_velocity(C, T, Vd, lags)
print('velocity decoding R2: vx=%.3f vy=%.3f' % tuple(r2))
sp_true, sp_pred = np.hypot(*Y.T), np.hypot(*pred.T)
ang_err = np.degrees(np.abs(np.angle(np.exp(1j*(np.arctan2(*Y[:, ::-1].T)
                                               - np.arctan2(*pred[:, ::-1].T))))))
fast = sp_true > 100
print('speed R2 %.3f ; median direction error %.1f deg (moving bins)'
      % (1-((sp_true-sp_pred)**2).sum()/((sp_true-sp_true.mean())**2).sum(),
         np.median(ang_err[fast])))

# direction decoding from delay-period rates alone
lab = tr['dir_bin'].values
R_prep = np.load('./cache/rates_preparatory.npy')
R_move = np.load('./cache/rates_perimovement.npy')
R_base = np.load('./cache/rates_baseline.npy')
acc = {}
for nm, R in [('baseline', R_base), ('preparatory (delay)', R_prep), ('peri-movement', R_move)]:
    p = dc.decode_direction(R, lab, np.arange(len(lab)) % 5)
    acc[nm] = (p == lab).mean()
    print(f'{nm:22s} direction decoding accuracy {acc[nm]:.3f} (chance {1/len(np.unique(lab)):.3f})')
pickle.dump(dict(pred=pred, Y=Y, g=g, r2=r2, good=good, peri=peri, acc=acc,
                 lab=lab, ang_err=ang_err, fast=fast,
                 pred_move=dc.decode_direction(R_move, lab, np.arange(len(lab)) % 5)),
            open('./cache/decoding.pkl','wb'))
