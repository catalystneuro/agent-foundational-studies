import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pynapple as nap
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import Ridge
from reach_lib import open_nwb, obs_intervals_set, load_units

BIN2 = 0.05          # s, bin size for the continuous analysis
SPD_TH = 100.0       # mm/s, "the hand is moving"

nwbfile = open_nwb("MC_Maze")
ep = obs_intervals_set(nwbfile)
units = load_units(nwbfile, ep)
t = np.load("_cache_t.npy"); vel = np.load("_cache_vel.npy")

counts = units.count(BIN2, ep)
Y = counts.values / BIN2
print("binned counts:", counts.shape, "| observed time %.0f s" % ep.tot_length())

# All trials are used here (including the curved maze reaches), which gives a far
# richer distribution of hand velocities than the straight centre-out reaches alone.
def lagged_velocity(lag):
    """Hand velocity at time (bin + lag), averaged into the same bins as `counts`."""
    v = nap.TsdFrame(t=t - lag, d=vel, columns=["vx", "vy"])
    out = v.restrict(ep).bin_average(BIN2, ep)
    assert np.allclose(out.t, counts.t)
    return out.values


def move_mask(V):
    return np.isfinite(V).all(1) & (np.hypot(V[:, 0], V[:, 1]) > SPD_TH)


# Contiguous blocks of trials define the cross-validation folds, so that train and
# test bins are never drawn from the same reach (spike counts are autocorrelated).
blocks = np.searchsorted(ep.start, counts.t, side="right") // 60


def decode_r2(Xr, Vt, blk, n_fold=5):
    """Cross-validated R^2 for ridge decoding of (vx, vy) from population rates."""
    fold = np.unique(blk) % n_fold
    fmap = dict(zip(np.unique(blk), fold))
    f = np.array([fmap[b] for b in blk])
    pred = np.empty_like(Vt)
    for k in range(n_fold):
        tr, te = f != k, f == k
        mdl = Ridge(alpha=1.0).fit(Xr[tr], Vt[tr])
        pred[te] = mdl.predict(Xr[te])
    return 1 - ((Vt - pred) ** 2).sum() / ((Vt - Vt.mean(0)) ** 2).sum()


V0 = lagged_velocity(0.0)
print("bins total %d | moving (speed>%.0f mm/s) %d (%.0f%%, %.0f s)"
      % (len(V0), SPD_TH, move_mask(V0).sum(), 100 * move_mask(V0).mean(), move_mask(V0).sum() * BIN2))


def lstsq_r2(X, Yv):
    with np.errstate(all="ignore"):
        B = np.linalg.lstsq(X, Yv, rcond=None)[0]
        sst = ((Yv - Yv.mean(0)) ** 2).sum(0)
        r2 = 1 - ((Yv - X @ B) ** 2).sum(0) / np.where(sst > 0, sst, np.nan)
    return B, r2


# ---------------- lag scan ----------------
# Run over ALL observed bins, not only movement bins: within a single reach the
# velocity direction is nearly constant, so the movement-only scan is almost flat.
# The rest-to-move transitions are what carry the timing information.
sig = np.load("_res_direction.npz")["sig"]
lags = np.round(np.arange(-0.30, 0.301, 0.02), 3)
r2_lag = np.full((len(lags), Y.shape[1]), np.nan)
r2_lag_pop = np.full(len(lags), np.nan)
Ysm = gaussian_filter1d(Y, 2.0, axis=0)            # ~100 ms, for the population decode
for i, lg in enumerate(tqdm(lags, desc="lag scan", mininterval=5)):
    V = lagged_velocity(lg); m = np.isfinite(V).all(1)
    r2_lag[i] = lstsq_r2(np.column_stack([np.ones(m.sum()), V[m]]), Y[m])[1]
    r2_lag_pop[i] = decode_r2(Ysm[m], V[m], blocks[m])

pop_r2 = np.nanmedian(r2_lag[:, sig], 1)
BEST_LAG = lags[np.argmax(r2_lag_pop)]
best_per_unit = lags[np.nanargmax(r2_lag, 0)]
print("single-unit velocity model: peak at %+.0f ms (median R2 %.4f over tuned units)"
      % (1000 * lags[np.argmax(pop_r2)], pop_r2.max()))
print("population decode of hand velocity: peak at %+.0f ms (cross-validated R2 %.3f; %.3f at lag 0)"
      % (BEST_LAG * 1000, r2_lag_pop.max(), r2_lag_pop[np.isclose(lags, 0)][0]))
print("per-unit best lag (tuned units): median %+.0f ms, IQR [%+.0f, %+.0f] ms"
      % tuple(1000 * np.percentile(best_per_unit[sig], [50, 25, 75])[[0, 1, 2]]))
print("(positive lag = neural activity LEADS the hand)")

# ---------------- velocity model at the optimal lag ----------------
V = lagged_velocity(BEST_LAG); m = move_mask(V)
Vv, Yv = V[m], Y[m]
speed = np.hypot(Vv[:, 0], Vv[:, 1]); vdir = np.arctan2(Vv[:, 1], Vv[:, 0])
print("moving-bin speed: median %.0f, p10 %.0f, p90 %.0f, max %.0f mm/s"
      % (np.median(speed), *np.percentile(speed, [10, 90]), speed.max()))

Bv, r2_vel = lstsq_r2(np.column_stack([np.ones(m.sum()), Vv]), Yv)
pd_vel = np.arctan2(Bv[2], Bv[1])
gain = np.hypot(Bv[1], Bv[2])
print("velocity gain: median %.4f Hz per mm/s -> %.1f Hz swing at 500 mm/s (tuned units)"
      % (np.median(gain[sig]), 500 * np.median(gain[sig])))

d2 = np.load("_res_direction.npz")
dpd = np.angle(np.exp(1j * (pd_vel - d2["move_pd"])))
print("PD agreement (continuous velocity model vs trial-based cosine fit), %d tuned units: "
      "median |diff| %.1f deg, %.0f%% within 45 deg, resultant length %.2f"
      % (sig.sum(), np.degrees(np.median(np.abs(dpd[sig]))),
         100 * (np.abs(dpd[sig]) < np.pi / 4).mean(), np.abs(np.mean(np.exp(1j * dpd[sig])))))

# ---------------- nested model comparison: direction only vs full velocity ----------------
Xc = np.ones((m.sum(), 1))
Xd = np.column_stack([np.ones(m.sum()), np.cos(vdir), np.sin(vdir)])          # direction only
Xs = np.column_stack([np.ones(m.sum()), speed])                              # speed only
Xv = np.column_stack([np.ones(m.sum()), Vv])                                 # velocity = speed x direction
Xvs = np.column_stack([np.ones(m.sum()), Vv, speed])                         # velocity + speed
mods = {"direction only": Xd, "speed only": Xs, "velocity": Xv, "velocity+speed": Xvs}
r2_mod = {k: lstsq_r2(X, Yv)[1] for k, X in mods.items()}
for k, v in r2_mod.items():
    print("  ols R2 %-16s median (tuned units) %.4f" % (k, np.median(v[sig])))

# ---------------- 2D velocity tuning maps (movement bins only) ----------------
LIM = 700
Vtsd = nap.TsdFrame(t=counts.t[m], d=Vv, columns=["vx", "vy"])
tc2d, edges2d = nap.compute_2d_tuning_curves(units, Vtsd, 13, minmax=(-LIM, LIM, -LIM, LIM))
occ, xe, ye = np.histogram2d(Vv[:, 0], Vv[:, 1], bins=13, range=[[-LIM, LIM]] * 2)
print("2D maps: %d/%d velocity bins with <25 samples (shown as blank)" % ((occ < 25).sum(), occ.size))

# ---------------- speed x (direction relative to PD) decomposition ----------------
rel = np.angle(np.exp(1j * (vdir[:, None] - pd_vel[None, :])))
sedges = np.array([100, 200, 300, 400, 500, 650, 900])
scent = (sedges[:-1] + sedges[1:]) / 2
aedges = np.linspace(-np.pi, np.pi, 7)
acent = (aedges[:-1] + aedges[1:]) / 2
sb = np.digitize(speed, sedges) - 1
grid = np.full((len(scent), len(acent), Y.shape[1]), np.nan)
for j in tqdm(range(Y.shape[1]), desc="speed x direction", mininterval=5):
    ab = np.digitize(rel[:, j], aedges) - 1
    for a in range(len(acent)):
        for s in range(len(scent)):
            k = (ab == a) & (sb == s)
            if k.sum() > 30:
                grid[s, a, j] = Yv[k, j].mean()
nrm = np.nanmean(grid, (0, 1))
pop_grid = np.nanmean(grid[..., sig] / nrm[sig], 2)
ipd, iap = int(np.argmin(np.abs(acent))), int(np.argmax(np.abs(acent)))
print("population grid (x mean rate): toward PD  %.2f (slow) -> %.2f (fast)" % (pop_grid[0, ipd], pop_grid[-1, ipd]))
print("                               against PD %.2f (slow) -> %.2f (fast)" % (pop_grid[0, iap], pop_grid[-1, iap]))

np.savez("_res_velocity.npz", lags=lags, r2_lag=r2_lag, pop_r2=pop_r2, r2_lag_pop=r2_lag_pop, best_lag=BEST_LAG,
         best_per_unit=best_per_unit, Bv=Bv, pd_vel=pd_vel, gain=gain, r2_vel=r2_vel,
         tc2d=np.stack([tc2d[k] for k in units.index]), xe=edges2d[0], ye=edges2d[1], occ=occ,
         grid=grid, pop_grid=pop_grid, scent=scent, acent=acent, dpd=dpd,
         bin2=BIN2, spd_th=SPD_TH, ipd=ipd, iap=iap,
         **{("r2mod_" + k.replace(" ", "_").replace("+", "_")): v for k, v in r2_mod.items()})
print("saved _res_velocity.npz")
