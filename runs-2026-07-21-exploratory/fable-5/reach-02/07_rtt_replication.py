"""Cross-dataset replication: MC_RTT (DANDI:000129), monkey Indy, self-paced random-target reaching."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pynapple as nap
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import Ridge
from tqdm import tqdm
from reach_lib import open_nwb, obs_intervals_set, load_units

BIN2 = 0.05
nwbfile = open_nwb("MC_RTT")
ep = obs_intervals_set(nwbfile)
units = load_units(nwbfile, ep)
beh = nwbfile.processing["behavior"].data_interfaces
fv = beh["finger_vel"]
vv = np.asarray(fv.data[:])
tv = (np.asarray(fv.timestamps[:]) if fv.timestamps is not None
      else fv.starting_time + np.arange(len(vv)) / fv.rate)
print("MC_RTT: %d units, %d obs intervals totalling %.0f s" % (len(units), len(ep), ep.tot_length()))
print("finger_vel: shape %s, unit %r, conversion %g, dt %.4f s, t %.1f-%.1f s"
      % (vv.shape, fv.unit, fv.conversion, np.median(np.diff(tv)), tv[0], tv[-1]))
print("speed percentiles (mm/s): %s" % np.round(np.nanpercentile(np.hypot(*vv.T), [50, 90, 99]), 1))
print("NaN samples in finger_vel: %d of %d (%.3f%%); excluded by the isfinite masks below"
      % (np.isnan(vv).any(1).sum(), len(vv), 100 * np.isnan(vv).any(1).mean()))

counts = units.count(BIN2, ep); Y = counts.values / BIN2
blocks = (np.arange(len(counts.t)) // 200)          # ~10 s contiguous blocks for CV folds


def lagged(lag):
    return nap.TsdFrame(t=tv - lag, d=vv, columns=["vx", "vy"]).restrict(ep).bin_average(BIN2, ep).values


def decode_r2(Xr, Vt, blk, n_fold=5):
    f = blk % n_fold
    pred = np.empty_like(Vt)
    with np.errstate(all="ignore"):
        for k in range(n_fold):
            tr, te = f != k, f == k
            pred[te] = Ridge(alpha=1.0).fit(Xr[tr], Vt[tr]).predict(Xr[te])
    return 1 - ((Vt - pred) ** 2).sum() / ((Vt - Vt.mean(0)) ** 2).sum(), pred


V0 = lagged(0.0)
SPD_TH = float(np.nanpercentile(np.hypot(*V0.T), 60))
print("reaches here are slower and more continuous than in MC_Maze, so the movement threshold\n"
      "is set at the 60th speed percentile of this session: %.0f mm/s" % SPD_TH)

Ysm = gaussian_filter1d(Y, 2.0, axis=0)
lags = np.round(np.arange(-0.30, 0.301, 0.02), 3)
pop = np.full(len(lags), np.nan); su = np.full((len(lags), Y.shape[1]), np.nan)
for i, lg in enumerate(tqdm(lags, desc="RTT lag scan", mininterval=5)):
    V = lagged(lg); m = np.isfinite(V).all(1)
    X = np.column_stack([np.ones(m.sum()), V[m]])
    with np.errstate(all="ignore"):
        B = np.linalg.lstsq(X, Y[m], rcond=None)[0]
        sst = ((Y[m] - Y[m].mean(0)) ** 2).sum(0)
        su[i] = 1 - ((Y[m] - X @ B) ** 2).sum(0) / np.where(sst > 0, sst, np.nan)
    pop[i] = decode_r2(Ysm[m], V[m], blocks[m])[0]
BEST = lags[np.argmax(pop)]
print("MC_RTT population-decode peak at %+.0f ms (R2 %.3f); single-unit peak %+.0f ms"
      % (BEST * 1000, pop.max(), 1000 * lags[np.nanargmax(np.nanmedian(su, 1))]))

V = lagged(BEST); m = np.isfinite(V).all(1) & (np.hypot(*V.T) > SPD_TH)
Vm, Yv = V[m], Y[m]
sp = np.hypot(*Vm.T); th = np.arctan2(Vm[:, 1], Vm[:, 0])
with np.errstate(all="ignore"):
    B = np.linalg.lstsq(np.column_stack([np.ones(m.sum()), Vm]), Yv, rcond=None)[0]
pd_v = np.arctan2(B[2], B[1])

# permutation test on the strength of velocity tuning
rng = np.random.default_rng(0)
obs = np.hypot(B[1], B[2])
null = np.empty((300, Y.shape[1]))
for i in tqdm(range(300), desc="RTT permutations", mininterval=5):
    sh = rng.integers(500, len(Vm) - 500)                 # circular shift preserves autocorrelation
    Vs = np.roll(Vm, sh, axis=0)
    with np.errstate(all="ignore"):
        Bs = np.linalg.lstsq(np.column_stack([np.ones(len(Vs)), Vs]), Yv, rcond=None)[0]
    null[i] = np.hypot(Bs[1], Bs[2])
pval = (null >= obs).mean(0)
sigR = pval < 0.01
print("velocity-tuned units (circular-shift permutation p<0.01): %d/%d (%.0f%%)"
      % (sigR.sum(), len(sigR), 100 * sigR.mean()))

# speed x relative direction
rel = np.angle(np.exp(1j * (th[:, None] - pd_v[None, :])))
sedges = np.percentile(sp, [0, 20, 40, 60, 80, 95, 100])
scent = (sedges[:-1] + sedges[1:]) / 2
aedges = np.linspace(-np.pi, np.pi, 7); acent = (aedges[:-1] + aedges[1:]) / 2
sb = np.clip(np.digitize(sp, sedges) - 1, 0, len(scent) - 1)
grid = np.full((len(scent), len(acent), Y.shape[1]), np.nan)
for j in range(Y.shape[1]):
    ab = np.digitize(rel[:, j], aedges) - 1
    for a in range(len(acent)):
        for s in range(len(scent)):
            k = (ab == a) & (sb == s)
            if k.sum() > 30: grid[s, a, j] = Yv[k, j].mean()
g = grid[..., sigR] / np.nanmean(grid[..., sigR], (0, 1))
pop_grid = np.nanmean(g, 2)
ipd, iap = int(np.argmin(np.abs(acent))), int(np.argmax(np.abs(acent)))
print("toward PD  %.2f (slow) -> %.2f (fast);  away from PD %.2f -> %.2f"
      % (pop_grid[0, ipd], pop_grid[-1, ipd], pop_grid[0, iap], pop_grid[-1, iap]))

r2dec, pred = decode_r2(Ysm[m], Vm, blocks[m])
derr = np.degrees(np.angle(np.exp(1j * (np.arctan2(pred[:, 1], pred[:, 0]) - th))))
print("MC_RTT ridge decode on moving bins: R2 %.3f, median |direction error| %.1f deg" % (r2dec, np.median(np.abs(derr))))

np.savez("_res_rtt.npz", lags=lags, pop=pop, su=su, best=BEST, pd_v=pd_v, sigR=sigR,
         pop_grid=pop_grid, scent=scent, acent=acent, g=g, r2dec=r2dec, derr=derr,
         spd_th=SPD_TH, ipd=ipd, iap=iap)

# ---------------- figure 7 ----------------
fig, ax = plt.subplots(1, 4, figsize=(17, 4.3), gridspec_kw=dict(wspace=0.36))
ax[0].plot(lags * 1000, pop, "o-", color="crimson")
ax[0].axvline(BEST * 1000, color="crimson", ls="--"); ax[0].axvline(0, color="0.6", lw=1)
ax[0].set(xlabel="lag (ms): positive = neural leads hand", ylabel="cross-validated $R^2$",
          title="Motor cortex leads the hand\nby %+.0f ms" % (BEST * 1000))
ax[1].hist(np.degrees(pd_v[sigR]), bins=np.linspace(-180, 180, 25), color="seagreen", edgecolor="w")
ax[1].set(xlabel="preferred direction (deg)", ylabel="units", xticks=[-180, -90, 0, 90, 180],
          title="Preferred directions\n%d/%d units velocity-tuned" % (sigR.sum(), len(sigR)))
for i, (lab, c) in [(ipd, ("toward PD", "crimson")), (int(np.argmin(np.abs(np.abs(acent) - np.pi / 2))), ("orthogonal", "0.4")), (iap, ("away from PD", "steelblue"))]:
    mu = np.nanmean(g[:, i, :], 1); se = np.nanstd(g[:, i, :], 1) / np.sqrt(sigR.sum())
    ax[2].errorbar(scent, mu, yerr=se, marker="o", lw=2, color=c, label=lab, capsize=3)
ax[2].axhline(1, color="k", ls=":", lw=0.8); ax[2].legend(fontsize=8)
ax[2].set(xlabel="finger speed (mm/s)", ylabel="rate / mean rate",
          title="Speed scales the directional signal\n(replicates MC_Maze)")
ax[3].hist(derr, bins=np.linspace(-180, 180, 37), color="seagreen", edgecolor="w")
ax[3].set(xlabel="decoded minus actual direction (deg)", ylabel="bins", xticks=[-180, -90, 0, 90, 180],
          title="Velocity decoding\n$R^2$ = %.2f, median error %.0f$\\degree$" % (r2dec, np.median(np.abs(derr))))
fig.suptitle("Replication in an independent dataset: DANDI:000129 MC_RTT, monkey Indy, %d M1 units, self-paced random-target reaching" % len(units),
             y=1.06, fontsize=13)
fig.savefig("fig07_rtt_replication.png", dpi=140, bbox_inches="tight")
print("saved fig07")
