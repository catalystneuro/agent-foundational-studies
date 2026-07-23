"""Population decoding: Georgopoulos population vector and ridge decoding of hand velocity."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pynapple as nap
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import Ridge
from reach_lib import open_nwb, obs_intervals_set, load_units

BIN2, SPD_TH = 0.05, 100.0
Vr = np.load("_res_velocity.npz"); D = np.load("_res_direction.npz")
LAG = float(Vr["best_lag"]); sig = D["sig"]; pd_v = Vr["pd_vel"]

nwbfile = open_nwb("MC_Maze"); ep = obs_intervals_set(nwbfile); units = load_units(nwbfile, ep)
t = np.load("_cache_t.npy"); vel = np.load("_cache_vel.npy")
counts = units.count(BIN2, ep)
R = gaussian_filter1d(counts.values / BIN2, 2.0, axis=0)          # ~100 ms smoothing
V = nap.TsdFrame(t=t - LAG, d=vel, columns=["vx", "vy"]).restrict(ep).bin_average(BIN2, ep).values
spd = np.hypot(V[:, 0], V[:, 1])
mov = np.isfinite(V).all(1) & (spd > SPD_TH)
blocks = np.searchsorted(ep.start, counts.t, side="right") // 60
print("decoding on %d moving bins (%.0f s), lag %+.0f ms" % (mov.sum(), mov.sum() * BIN2, LAG * 1000))

# ---------------- 1. Georgopoulos population vector (no fitting to the test data) ----------------
z = (R - R.mean(0)) / np.maximum(R.std(0), 1e-9)
U = np.stack([np.cos(pd_v), np.sin(pd_v)], 1)
PV = z[:, sig] @ U[sig]                                            # bins x 2
pv_ang = np.arctan2(PV[:, 1], PV[:, 0])
true_ang = np.arctan2(V[:, 1], V[:, 0])
err = np.degrees(np.angle(np.exp(1j * (pv_ang[mov] - true_ang[mov]))))
print("population vector: median |direction error| %.1f deg; %.0f%% within 45 deg (chance 25%%)"
      % (np.median(np.abs(err)), 100 * (np.abs(err) < 45).mean()))
pv_len = np.hypot(*PV.T)
r_len = np.corrcoef(pv_len[mov], spd[mov])[0, 1]
print("population vector length vs hand speed: r = %.2f" % r_len)

# ---------------- 2. Ridge decoding of the full velocity vector ----------------
n_fold = 5
f = blocks % n_fold
pred = np.full_like(V, np.nan)
for k in range(n_fold):
    tr = (f != k) & np.isfinite(V).all(1)
    te = (f == k) & np.isfinite(V).all(1)
    with np.errstate(all="ignore"):
        mdl = Ridge(alpha=1.0).fit(R[tr], V[tr])
        pred[te] = mdl.predict(R[te])
ok = np.isfinite(pred).all(1)
def r2(a, b): return 1 - ((a - b) ** 2).sum() / ((a - a.mean(0)) ** 2).sum()
print("ridge decode, all observed bins:  R2 vx %.3f  vy %.3f  overall %.3f"
      % (r2(V[ok, 0], pred[ok, 0]), r2(V[ok, 1], pred[ok, 1]), r2(V[ok], pred[ok])))
print("ridge decode, moving bins only:   R2 %.3f" % r2(V[mov], pred[mov]))
dec_ang = np.arctan2(pred[:, 1], pred[:, 0])
derr = np.degrees(np.angle(np.exp(1j * (dec_ang[mov] - true_ang[mov]))))
print("ridge decode: median |direction error| %.1f deg; speed correlation r = %.2f"
      % (np.median(np.abs(derr)), np.corrcoef(np.hypot(*pred[mov].T), spd[mov])[0, 1]))

np.savez("_res_decode.npz", pv_err=err, dec_err=derr, pv_len=pv_len, pred=pred, V=V,
         mov=mov, spd=spd, tt=counts.t, r_len=r_len)

# ---------------- figure 6 ----------------
fig = plt.figure(figsize=(15.5, 9))
gs = fig.add_gridspec(3, 3, hspace=0.55, wspace=0.30, height_ratios=[1, 1, 1.15])

i0 = int(np.argmax(counts.t > 300))
sl = slice(i0, i0 + 320)                                            # 16 s
for r, (comp, lab) in enumerate([(0, "$v_x$"), (1, "$v_y$")]):
    ax = fig.add_subplot(gs[r, :])
    ax.plot(counts.t[sl], V[sl, comp], color="k", lw=1.6, label="actual hand velocity")
    ax.plot(counts.t[sl], pred[sl, comp], color="crimson", lw=1.4, label="decoded from 182 units")
    ax.set_ylabel("%s (mm/s)" % lab)
    if r == 0:
        ax.legend(ncol=2, fontsize=9, loc="upper right")
        ax.set_title("Hand velocity decoded from the population (5-fold cross-validated, %+.0f ms lag)" % (LAG * 1000))
    else:
        ax.set_xlabel("time (s)")

ax = fig.add_subplot(gs[2, 0])
bins = np.linspace(-180, 180, 49)
ax.hist(err, bins=bins, color="steelblue", alpha=0.85,
        label="population vector (median |err| %.0f$\\degree$)" % np.median(np.abs(err)))
ax.hist(derr, bins=bins, color="crimson", alpha=0.55,
        label="ridge decoder (median |err| %.0f$\\degree$)" % np.median(np.abs(derr)))
ax.axhline(mov.sum() / (len(bins) - 1), color="k", ls=":", lw=1.2, label="chance (uniform)")
ax.set(xlabel="decoded minus actual reach direction (deg)", ylabel="bins", xticks=[-180, -90, 0, 90, 180],
       title="Reach direction is decoded\nfrom single 50 ms bins")
ax.legend(fontsize=7.5, loc="upper left")

ax = fig.add_subplot(gs[2, 1])
ax.hexbin(spd[mov], np.hypot(*pred[mov].T), gridsize=45, cmap="Blues", bins="log", mincnt=1)
lim = [0, 1000]
ax.plot(lim, lim, "k--", lw=1); ax.set(xlim=lim, ylim=lim)
ax.set(xlabel="actual hand speed (mm/s)", ylabel="decoded speed (mm/s)",
       title="Speed is decoded too, though ridge\nshrinkage compresses it: r = %.2f" % np.corrcoef(np.hypot(*pred[mov].T), spd[mov])[0, 1])

ax = fig.add_subplot(gs[2, 2])
ax.hexbin(spd[mov], pv_len[mov], gridsize=45, cmap="Purples", bins="log", mincnt=1)
ax.set(xlabel="actual hand speed (mm/s)", ylabel="population vector length (a.u.)",
       title="Population vector length is only\nweakly related to speed: r = %.2f" % r_len)
fig.suptitle("Population coding of reach velocity (DANDI:000128 MC_Maze)", y=0.965, fontsize=13)
fig.savefig("fig06_population_decoding.png", dpi=140, bbox_inches="tight")
print("saved fig06")
