"""Population decoding: recover instantaneous hand velocity from motor-cortical spiking.

Two complementary decoders, both cross-validated across reaches:
  * Bayesian decoding of movement direction from pynapple direction tuning curves
  * ridge regression of (vx, vy) from binned population rates
"""
import numpy as np, pandas as pd, os
np.seterr(all='ignore')
import pynapple as nap
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from s01_load import load_all, to_pynapple
from s04_velocity_tuning import movement_epochs, shifted_vel

CACHE = "cache"
BIN = 0.05
NFOLD = 5
SMOOTH = 0.1


def circ_err(a, b):
    return np.abs(np.angle(np.exp(1j * (a - b))))


if __name__ == "__main__":
    d = load_all(); spk, vel, pos = to_pynapple(d)
    K = pd.read_pickle(f"{CACHE}/kinematics.pkl")
    LAG = float(np.load(f"{CACHE}/best_lag.npy")[0])
    ep = movement_epochs(K)

    cnt = spk.count(BIN, ep=ep)
    rate = np.asarray(cnt.smooth(SMOOTH).values) / BIN
    tb = cnt.index.values
    V = np.asarray(shifted_vel(vel, LAG).interpolate(nap.Ts(tb)).values)
    spd = np.hypot(V[:, 0], V[:, 1]); ang = np.arctan2(V[:, 1], V[:, 0])
    keep = np.isfinite(spd) & (spd > 100)
    trial = np.searchsorted(np.asarray(ep.start), tb, side='right') - 1
    Xr, Yv, Ya, Ys, g = rate[keep], V[keep], ang[keep], spd[keep], trial[keep]
    print(f"decoding on {keep.sum()} bins from {len(np.unique(g))} reaches, {Xr.shape[1]} units")

    # ---- ridge decoding of the velocity vector ----
    gkf = GroupKFold(n_splits=NFOLD)
    pred = np.zeros_like(Yv)
    for tr, te in gkf.split(Xr, Yv, groups=g):
        m, s = Xr[tr].mean(0), Xr[tr].std(0) + 1e-9
        r = Ridge(alpha=100.0).fit((Xr[tr] - m) / s, Yv[tr])
        pred[te] = r.predict((Xr[te] - m) / s)
    r2 = 1 - ((Yv - pred) ** 2).sum(0) / ((Yv - Yv.mean(0)) ** 2).sum(0)
    cc = [np.corrcoef(Yv[:, i], pred[:, i])[0, 1] for i in range(2)]
    dec_ang = np.arctan2(pred[:, 1], pred[:, 0]); dec_spd = np.hypot(*pred.T)
    ang_err = circ_err(dec_ang, Ya)
    print(f"ridge decoding: R2 vx={r2[0]:.3f} vy={r2[1]:.3f}; r vx={cc[0]:.3f} vy={cc[1]:.3f}")
    print(f"  direction error: median {np.degrees(np.median(ang_err)):.1f} deg "
          f"(chance 90 deg);  speed r={np.corrcoef(dec_spd, Ys)[0,1]:.3f}")
    np.savez(f"{CACHE}/decoding.npz", true=Yv, pred=pred, ang=Ya, spd=Ys,
             tb=tb[keep], trial=g, r2=r2, ang_err=ang_err)

    # ---- Bayesian decoding of movement direction from pynapple tuning curves ----
    feat = nap.Tsd(t=tb, d=ang)
    folds = np.unique(g) % NFOLD
    fold_of_bin = folds[np.searchsorted(np.unique(g), g)]
    dec_bayes = np.full(len(Ya), np.nan)
    for f in range(NFOLD):
        tr_ep = nap.IntervalSet(start=np.asarray(ep.start)[folds != f],
                                end=np.asarray(ep.end)[folds != f])
        te_ep = nap.IntervalSet(start=np.asarray(ep.start)[folds == f],
                                end=np.asarray(ep.end)[folds == f])
        tc = nap.compute_1d_tuning_curves(group=spk, feature=feat, nb_bins=18,
                                          ep=tr_ep, minmax=(-np.pi, np.pi))
        dec, p = nap.decode_1d(tuning_curves=tc, group=spk, ep=te_ep, bin_size=BIN)
        di = dec.interpolate(nap.Ts(tb[keep][fold_of_bin == f]))
        dec_bayes[fold_of_bin == f] = np.asarray(di.values)
    m = np.isfinite(dec_bayes)
    berr = circ_err(dec_bayes[m], Ya[m])
    print(f"Bayesian direction decoding: median error {np.degrees(np.median(berr)):.1f} deg, "
          f"{np.mean(berr < np.pi/4):.1%} within 45 deg (chance 25%)")
    np.savez(f"{CACHE}/decoding_bayes.npz", dec=dec_bayes[m], true=Ya[m], err=berr)
