"""Speed tuning from the continuous hand-velocity signal during movement.

At the trial level, peak reach speed is confounded with reach direction (targets sit
at different distances). Within a reach, however, direction is roughly constant while
speed sweeps from zero through peak and back, so the *continuous* signal covers the
(direction x speed) plane almost independently. All speed results therefore come from
the continuous data, using the neural lead measured in s04.
"""
import numpy as np, pandas as pd, os
np.seterr(all='ignore')
import pynapple as nap
from s01_load import load_all, to_pynapple
from s04_velocity_tuning import movement_epochs, shifted_vel

CACHE = "cache"
NDIR, NSPD = 12, 6
SPD_RANGE = (50.0, 1150.0)


def polar_features(vel, lag):
    v = shifted_vel(vel, lag)
    V = np.asarray(v.values)
    return nap.TsdFrame(t=v.index.values,
                        d=np.column_stack([np.arctan2(V[:, 1], V[:, 0]), np.hypot(V[:, 0], V[:, 1])]),
                        columns=['dir', 'speed'])


if __name__ == "__main__":
    d = load_all(); spk, vel, pos = to_pynapple(d)
    K = pd.read_pickle(f"{CACHE}/kinematics.pkl")
    df = pd.read_pickle(f"{CACHE}/dir_tuning.pkl")
    LAG = float(np.load(f"{CACHE}/best_lag.npy")[0])
    ep = movement_epochs(K)
    feat = polar_features(vel, LAG)

    F = np.asarray(feat.restrict(ep).values)
    m = F[:, 1] > SPD_RANGE[0]
    print(f"movement samples: {m.sum()};  corr(speed, cos dir)={np.corrcoef(F[m,1], np.cos(F[m,0]))[0,1]:.3f}, "
          f"corr(speed, sin dir)={np.corrcoef(F[m,1], np.sin(F[m,0]))[0,1]:.3f}")

    tc, bins = nap.compute_2d_tuning_curves(
        group=spk, features=feat, nb_bins=(NDIR, NSPD), ep=ep,
        minmax=(-np.pi, np.pi, SPD_RANGE[0], SPD_RANGE[1]))
    TC = np.array([tc[k] for k in sorted(tc)])          # (units, NDIR, NSPD)
    np.save(f"{CACHE}/tc_dirspeed.npy", TC)
    np.save(f"{CACHE}/tc_dirspeed_bins.npy", np.array(bins, dtype=object), allow_pickle=True)
    print("dir x speed tuning maps:", TC.shape)

    # occupancy, for masking under-sampled bins
    occ = np.histogram2d(F[:, 0], F[:, 1], bins=[bins[0].size, bins[1].size],
                         range=[[-np.pi, np.pi], SPD_RANGE])[0]
    np.save(f"{CACHE}/occ_dirspeed.npy", occ)

    # --- per-unit summary from the maps ---
    dctr, sctr = bins[0], bins[1]
    ok = occ > 200                                       # >=0.2 s of data per bin
    TCm = np.where(ok[None], TC, np.nan)
    # preferred direction from the highest speed bins
    hi = np.nanmean(TCm[:, :, -3:], axis=2)
    pd_cont = np.angle(np.nansum(np.exp(1j * dctr)[None] * np.nan_to_num(hi), axis=1))
    # speed slope along preferred vs anti-preferred direction
    def _at(unit, ang):
        j = np.argmin(np.abs(np.angle(np.exp(1j * (dctr - ang)))))
        return TCm[unit, j]
    slopes_pd, slopes_anti = [], []
    for u in range(TC.shape[0]):
        for ang, store in [(pd_cont[u], slopes_pd), (pd_cont[u] + np.pi, slopes_anti)]:
            y = _at(u, ang); g = np.isfinite(y)
            store.append(np.polyfit(sctr[g], y[g], 1)[0] * 100 if g.sum() > 2 else np.nan)
    df['pref_dir_cont'] = pd_cont
    df['speed_slope_pd'] = np.array(slopes_pd)           # Hz per 100 mm/s
    df['speed_slope_anti'] = np.array(slopes_anti)
    df.to_pickle(f"{CACHE}/dir_tuning.pkl")

    sig = df.p_str < 0.01
    print(f"speed slope at preferred direction (Hz per 100 mm/s): "
          f"median {df.speed_slope_pd[sig].median():.2f}, "
          f"at anti-preferred: {df.speed_slope_anti[sig].median():.2f}")
    from scipy.stats import wilcoxon
    a, b = df.speed_slope_pd[sig], df.speed_slope_anti[sig]
    g = np.isfinite(a) & np.isfinite(b)
    print("Wilcoxon PD vs anti-PD speed slope:", wilcoxon(a[g], b[g]))
    print(f"units with |speed slope at PD| larger than at anti-PD: {np.mean(np.abs(a[g])>np.abs(b[g])):.1%}")
