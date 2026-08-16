"""Trial-based reach direction tuning + neural lead (lag) estimation."""
import numpy as np
import pynapple as nap
import matplotlib.pyplot as plt
import mcmaze_io as mio

nap.nap_config.suppress_conversion_warnings = True
rng = np.random.default_rng(0)

nwbfile, nwb = mio.open_nwb()
spikes = nwb["units"]
trials_ep = nwb["trials"]
hand_pos, hand_vel = mio.load_kinematics(nwbfile)
tdf = mio.trial_table(nwbfile)
print("units:", len(spikes), "trials:", len(tdf))

# ---------------------------------------------------------------- lag scan
# M1/PMd activity leads hand kinematics. Estimate the lead by regressing each
# unit's binned firing rate on the hand velocity at a range of lags and taking
# the lag that maximises the population-average R^2.
BIN = 0.02
counts = spikes.count(BIN, ep=trials_ep)          # TsdFrame (time, unit)
rate = counts / BIN
lags = np.arange(-0.10, 0.31, 0.02)
mean_r2 = []
for lag in lags:
    # kinematics at time t+lag labelled at t  ->  positive lag = neural leads
    shifted = nap.TsdFrame(t=hand_vel.t - lag, d=hand_vel.values, columns=["vx", "vy"])
    V = shifted.interpolate(counts, ep=counts.time_support).values
    ok = np.isfinite(V).all(1)
    X = np.column_stack([np.ones(ok.sum()), V[ok]])
    Y = rate.values[ok]
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ beta
    r2 = 1 - resid.var(0) / Y.var(0)
    mean_r2.append(np.nanmean(r2))
    print(f"  lag {lag:+.2f}s  mean R2 = {mean_r2[-1]:.4f}")
mean_r2 = np.array(mean_r2)
BEST_LAG = float(lags[np.argmax(mean_r2)])
print("best lag:", BEST_LAG)
np.save("lag_scan.npy", np.column_stack([lags, mean_r2]))

# ------------------------------------------------- per-trial reach direction
MOVE_WIN = (0.0, 0.30)          # relative to movement onset (kinematics)
onset = tdf.move_onset_time.values
peak_dir = np.full(len(tdf), np.nan)
peak_spd = np.full(len(tdf), np.nan)
for i, t0 in enumerate(onset):
    seg = hand_vel.restrict(nap.IntervalSet(t0 + MOVE_WIN[0], t0 + MOVE_WIN[1])).values
    s = np.hypot(seg[:, 0], seg[:, 1])
    k = int(np.argmax(s))
    peak_dir[i] = np.arctan2(seg[k, 1], seg[k, 0]) % (2 * np.pi)
    peak_spd[i] = s[k]
tdf["reach_dir"] = peak_dir
tdf["peak_speed"] = peak_spd

# ------------------------------------ per-trial spike counts in movement window
# neural window is the movement window shifted earlier by the estimated lead
nwin = (MOVE_WIN[0] - BEST_LAG, MOVE_WIN[1] - BEST_LAG)
dur = nwin[1] - nwin[0]
unit_ids = list(spikes.keys())
trial_rate = np.zeros((len(tdf), len(unit_ids)))
for j, u in enumerate(unit_ids):
    st = spikes[u].t
    lo = np.searchsorted(st, onset + nwin[0])
    hi = np.searchsorted(st, onset + nwin[1])
    trial_rate[:, j] = (hi - lo) / dur

# baseline: 200 ms before target onset
btime = tdf.target_on_time.values
base_rate = np.zeros_like(trial_rate)
for j, u in enumerate(unit_ids):
    st = spikes[u].t
    base_rate[:, j] = (np.searchsorted(st, btime) - np.searchsorted(st, btime - 0.2)) / 0.2

np.savez("trial_data.npz", trial_rate=trial_rate, base_rate=base_rate,
         reach_dir=peak_dir, peak_speed=peak_spd, unit_ids=np.array(unit_ids),
         onset=onset, best_lag=BEST_LAG,
         target_angle=tdf.target_angle.values, num_barriers=tdf.num_barriers.values)

# ----------------------------------------------------- directional tuning curves
NDIR = 12
edges = np.linspace(0, 2 * np.pi, NDIR + 1)
centers = (edges[:-1] + edges[1:]) / 2
dbin = np.digitize(peak_dir, edges) - 1
tc = np.array([trial_rate[dbin == k].mean(0) for k in range(NDIR)])       # (dir, unit)
tc_sem = np.array([trial_rate[dbin == k].std(0) / np.sqrt((dbin == k).sum())
                   for k in range(NDIR)])
print("trials per direction bin:", np.bincount(dbin, minlength=NDIR))


def fit_cosine(theta, r):
    """Least-squares fit of r = b0 + b1*cos(theta - pd). Returns b0, b1, pd, R^2."""
    X = np.column_stack([np.ones_like(theta), np.cos(theta), np.sin(theta)])
    beta, *_ = np.linalg.lstsq(X, r, rcond=None)
    b0, bc, bs = beta
    b1 = np.hypot(bc, bs)
    pd = np.arctan2(bs, bc) % (2 * np.pi)
    pred = X @ beta
    ss = ((r - r.mean()) ** 2).sum()
    r2 = 1 - ((r - pred) ** 2).sum() / ss if ss > 0 else np.nan
    return b0, b1, pd, r2


# fit on single trials (not on bin means) so R^2 is honest
fits = np.array([fit_cosine(peak_dir, trial_rate[:, j]) for j in range(len(unit_ids))])
b0, b1, pdir, r2 = fits.T

# permutation test on directional modulation depth b1
NPERM = 500
perm_b1 = np.zeros((NPERM, len(unit_ids)))
for p in range(NPERM):
    sh = rng.permutation(len(peak_dir))
    perm_b1[p] = np.array([fit_cosine(peak_dir, trial_rate[sh, j])[1]
                           for j in range(len(unit_ids))])
pval = (perm_b1 >= b1[None, :]).mean(0)
sig = pval < 0.01
print(f"significantly direction-tuned units: {sig.sum()}/{len(unit_ids)} "
      f"({100*sig.mean():.0f}%)")

np.savez("tuning_fits.npz", tc=tc, tc_sem=tc_sem, centers=centers, b0=b0, b1=b1,
         pdir=pdir, r2=r2, pval=pval, unit_ids=np.array(unit_ids), edges=edges,
         dbin=dbin)

# ------------------------------------------------------------------ quick figure
order = np.argsort(-b1)
fig, axes = plt.subplots(2, 4, figsize=(14, 7), subplot_kw={"projection": "polar"})
for ax, j in zip(axes.ravel(), order[:8]):
    th = np.r_[centers, centers[0]]
    v = np.r_[tc[:, j], tc[0, j]]
    ax.plot(th, v, "o-", color="C0")
    fine = np.linspace(0, 2 * np.pi, 200)
    ax.plot(fine, np.maximum(b0[j] + b1[j] * np.cos(fine - pdir[j]), 0), "r-", lw=1)
    ax.set_title(f"unit {unit_ids[j]}  b1={b1[j]:.0f} Hz  R2={r2[j]:.2f}", pad=18, fontsize=9)
fig.suptitle("Reach direction tuning (movement window), 8 most modulated units")
fig.tight_layout()
fig.savefig("fig_check_polar.png", dpi=130)
print("saved fig_check_polar.png")
