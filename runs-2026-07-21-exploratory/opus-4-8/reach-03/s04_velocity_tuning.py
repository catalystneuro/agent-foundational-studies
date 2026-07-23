"""Continuous 2D hand-velocity tuning during movement + neural lead-lag scan."""
import numpy as np, pandas as pd, os
np.seterr(all='ignore')
import pynapple as nap
from s01_load import load_all, to_pynapple

CACHE = "cache"


def movement_epochs(K, pad_pre=0.05, pad_post=0.05):
    return nap.IntervalSet(start=K.t_onset.values - pad_pre, end=K.t_end.values + pad_post)


def shifted_vel(vel, lag):
    """Velocity re-timestamped so that neural activity at t is paired with velocity at t+lag."""
    return nap.TsdFrame(t=vel.index.values - lag, d=vel.values, columns=['vx', 'vy'])


def vel_tuning_2d(spk, vel, ep, nb_bins=12, vmax=900.0):
    v = vel.restrict(ep)
    tc, binsxy = nap.compute_2d_tuning_curves(
        group=spk, features=v, nb_bins=nb_bins, ep=ep,
        minmax=(-vmax, vmax, -vmax, vmax))
    return tc, binsxy


def lag_scan(spk, vel, ep, lags, bin_size=0.02, smooth_std=0.05):
    """For each lag, fit rate ~ b0 + bx*vx + by*vy on smoothed binned rates; R2 per unit.

    Absolute R2 is small because single-trial spike counts are Poisson-noisy; the
    informative quantity is where the curve peaks as a function of lag.
    """
    cnt = spk.count(bin_size, ep=ep).smooth(smooth_std)
    tb = cnt.index.values
    out = []
    for lag in lags:
        vi = vel.interpolate(nap.Ts(tb + lag))
        V = np.asarray(vi.values)
        good = np.isfinite(V).all(1)
        X = np.column_stack([np.ones(good.sum()), V[good]])
        Y = np.asarray(cnt.values)[good] / bin_size
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        pred = X @ beta
        r2 = 1 - ((Y - pred) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)
        out.append(r2)
    return np.array(out)


if __name__ == "__main__":
    d = load_all()
    spk, vel, pos = to_pynapple(d)
    K = pd.read_pickle(f"{CACHE}/kinematics.pkl")
    ep = movement_epochs(K)
    print("movement epochs:", len(ep), "total", ep.tot_length(), "s")

    lags = np.arange(-0.30, 0.305, 0.01)
    R2 = lag_scan(spk, vel, ep, lags)
    np.save(f"{CACHE}/lag_r2.npy", R2); np.save(f"{CACHE}/lags.npy", lags)
    pop = np.nanmean(R2, 1)
    print("best population lag (s):", lags[np.argmax(pop)], "R2", pop.max())
    ok = np.nanmax(R2, 0) > 0.02          # units with any velocity signal
    best = lags[np.nanargmax(np.where(np.isnan(R2), -np.inf, R2)[:, ok], 0)]
    np.save(f"{CACHE}/best_lag_per_unit.npy", best)
    np.save(f"{CACHE}/lag_ok.npy", ok)
    print(f"units with R2>0.02: {ok.sum()}/{len(ok)}; per-unit best lag median "
          f"{np.median(best):.3f} s, IQR {np.percentile(best,[25,75])}")

    LAG = float(lags[np.argmax(pop)])
    vsh = shifted_vel(vel, LAG)
    tc, bins = vel_tuning_2d(spk, vsh, ep)
    Vm = np.asarray(vsh.restrict(ep).values)
    occ2d = np.histogram2d(Vm[:, 0], Vm[:, 1], bins=[bins[0].size, bins[1].size],
                           range=[[-900, 900], [-900, 900]])[0]
    np.save(f"{CACHE}/occ2d.npy", occ2d)
    np.save(f"{CACHE}/tc2d.npy", np.array([tc[k] for k in sorted(tc)]))
    np.save(f"{CACHE}/tc2d_bins.npy", np.array(bins))
    np.save(f"{CACHE}/best_lag.npy", np.array([LAG]))
    print("2D tuning maps:", np.array([tc[k] for k in sorted(tc)]).shape)
