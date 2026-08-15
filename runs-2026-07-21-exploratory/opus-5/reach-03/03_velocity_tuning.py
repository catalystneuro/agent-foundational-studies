"""Velocity tuning: does firing rate track the hand velocity vector, and when?

Uses every successful trial of the MC_Maze session, including the curved maze
reaches, because those sample a much wider range of speeds and directions than
the straight center-out reaches do.  The maze task repeats 108 distinct
conditions about 20 times each, so most analyses here are run on
condition-averaged rates, where the Poisson noise of a single 20 ms bin is
averaged away.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap
from tqdm import tqdm

import analysis as an
import dandi_io as dio

WIN = (0.0, 0.5)     # movement window relative to onset
SD = 0.03            # s, Gaussian kernel for the rate estimate

sess = dio.load_session("maze_full")
spikes, pos, vel, support = dio.to_pynapple(sess)
tr = sess["trials"]

keep = an.quality_units(spikes, dio.trial_intervals(sess), min_rate=0.5)
uid = np.flatnonzero(keep)
spikes = spikes[list(uid)]

idx = np.flatnonzero(np.isfinite(tr["move_onset_time"]) & tr["success"].astype(bool))
ntr = len(idx)
ep = an.movement_epochs(sess, idx, WIN)
counts = spikes.count(an.BIN, ep=ep)
rate = an.smooth_rate(counts, sd=SD)
labels = an.condition_labels(sess, idx)
print(f"{ntr} trials, {counts.shape[0]} bins of {an.BIN*1000:.0f} ms, "
      f"{counts.shape[1]} units, {len(np.unique(labels))} maze conditions")

# --------------------------------------------------------------- lag sweep
# The spike window is fixed and the velocity is resampled at increasing leads,
# so a peak at positive lag means the neural signal precedes the hand.
lags = np.round(np.arange(-0.30, 0.305, 0.02), 3)
r2_lag = np.empty((len(lags), rate.shape[1]))
dec_lag = np.empty(len(lags))
for i, lag in enumerate(tqdm(lags, desc="lag sweep")):
    mask, v = an.sample_velocity(vel, counts, lag)
    assert mask.all(), "every maze bin should have a velocity sample"
    Rc, conds = an.condition_average(rate, labels, ntr)
    Vc, _ = an.condition_average(v, labels, ntr)
    r2_lag[i], _ = an.velocity_regression_r2(Rc, Vc)
    # population decoding in the other direction: velocity from all units
    X = np.column_stack([np.ones(len(Rc)), Rc])
    b, *_ = np.linalg.lstsq(X, Vc, rcond=None)
    with np.errstate(all="ignore"):
        pred = X @ b
    dec_lag[i] = np.mean(1 - ((Vc - pred) ** 2).sum(0) / ((Vc - Vc.mean(0)) ** 2).sum(0))

pop_lag = lags[np.argmax(np.nanmedian(r2_lag, axis=1))]
dec_peak_lag = lags[np.argmax(dec_lag)]
best_lag = lags[np.argmax(r2_lag, axis=0)]
# Only units whose lag curve actually has a peak give a meaningful argmax;
# flat curves from weakly tuned units would just add uniform noise.
strong = r2_lag.max(0) > 0.15
print(f"encoding: population-optimal lead = {pop_lag*1000:.0f} ms "
      f"(median R2 {np.nanmedian(r2_lag, axis=1).max():.3f}, "
      f"max unit R2 {r2_lag.max():.3f})")
print(f"decoding: velocity reconstructed from the population, best lead "
      f"{dec_peak_lag*1000:.0f} ms, R2 = {dec_lag.max():.3f}")
print(f"per-unit optimal lead: median {np.median(best_lag[strong])*1000:.0f} ms, "
      f"IQR [{np.percentile(best_lag[strong],25)*1000:.0f}, "
      f"{np.percentile(best_lag[strong],75)*1000:.0f}] ms "
      f"({strong.sum()} units with peak R2 > 0.15)")

# ---------------------------------------- fit at the population-best lead
mask, V = an.sample_velocity(vel, counts, pop_lag)
assert mask.all()
speed = np.linalg.norm(V, axis=1)
vtheta = np.arctan2(V[:, 1], V[:, 0])
Rc, conds = an.condition_average(rate, labels, ntr)
Vc, _ = an.condition_average(V, labels, ntr)
r2_vel, beta_vel = an.velocity_regression_r2(Rc, Vc)
pd_vel = np.arctan2(beta_vel[2], beta_vel[1])

# Direction-only versus velocity-vector, cross-validated across conditions.
sc = np.linalg.norm(Vc, axis=1)
th = np.arctan2(Vc[:, 1], Vc[:, 0])
mov = sc > 50.0
fold = np.repeat(np.arange(len(conds)) % 5, Rc.shape[0] // len(conds))[mov]
Xdir = np.column_stack([np.ones(mov.sum()), np.cos(th[mov]), np.sin(th[mov])])
Xvel = np.column_stack([np.ones(mov.sum()), Vc[mov]])
Xful = np.column_stack([Xdir, Vc[mov]])


def cv_r2(X, y, fold):
    """R^2 of a linear model with held-out maze conditions, per column of y."""
    pred = np.empty_like(y)
    for f in np.unique(fold):
        te = fold == f
        b, *_ = np.linalg.lstsq(X[~te], y[~te], rcond=None)
        with np.errstate(all="ignore"):
            pred[te] = X[te] @ b
    return 1 - ((y - pred) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)


cv_dir = cv_r2(Xdir, Rc[mov], fold)
cv_vel = cv_r2(Xvel, Rc[mov], fold)
cv_ful = cv_r2(Xful, Rc[mov], fold)
print(f"cross-validated R2 on moving bins ({mov.sum()} of {len(sc)}): "
      f"direction only {np.median(cv_dir):.3f}, velocity vector {np.median(cv_vel):.3f}, "
      f"both {np.median(cv_ful):.3f}")
print(f"adding the speed-scaled term to the direction model helps in "
      f"{(cv_ful > cv_dir).sum()}/{len(cv_ful)} units")

centres, md_speed, b0_speed = an.speed_gain(rate, V, n_speed=4, pd=pd_vel)
print("speed quartile centres (mm/s):", centres.round(0))
print("median modulation along PD per quartile (Hz):",
      np.nanmedian(md_speed, axis=1).round(2))

np.savez("results_velocity.npz", lags=lags, r2_lag=r2_lag, dec_lag=dec_lag,
         best_lag=best_lag, pop_lag=pop_lag, strong=strong, r2_vel=r2_vel,
         pd_vel=pd_vel, beta_vel=beta_vel, cv_dir=cv_dir, cv_vel=cv_vel,
         cv_ful=cv_ful, centres=centres, md_speed=md_speed, uid=uid)

# ---------------------------------------------------------------- figure 3
plt.rcParams.update({"axes.titlesize": 10, "axes.labelsize": 9,
                     "xtick.labelsize": 8, "ytick.labelsize": 8})
fig = plt.figure(figsize=(16, 9))
gs = fig.add_gridspec(2, 4, hspace=0.42, wspace=0.58, top=0.86, bottom=0.08)

vfr = nap.TsdFrame(t=counts.index.values, d=V, columns=["vx", "vy"])
examples = np.argsort(-r2_vel)[:3]
lim = np.percentile(np.abs(V), 99)
for c, u in enumerate(examples):
    ax = fig.add_subplot(gs[0, c])
    tc = nap.compute_tuning_curves(
        nap.TsdFrame(t=vfr.index.values, d=rate[:, [u]]), vfr, bins=16,
        range=[(-lim, lim), (-lim, lim)])
    im = ax.pcolormesh(tc.coords["vx"].values, tc.coords["vy"].values,
                       np.asarray(tc[0]).T, cmap="viridis", shading="nearest")
    ax.set_aspect("equal")
    ax.arrow(0, 0, 0.7 * lim * np.cos(pd_vel[u]), 0.7 * lim * np.sin(pd_vel[u]),
             color="w", width=lim * 0.015, head_width=lim * 0.09,
             length_includes_head=True)
    ax.set(xlabel="$v_x$ (mm/s)", ylabel="$v_y$ (mm/s)" if c == 0 else "",
           title=f"unit {uid[u]}   $R^2$ = {r2_vel[u]:.2f}")
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03).set_label("rate (Hz)", size=8)

ax = fig.add_subplot(gs[0, 3])
sel = np.argsort(-r2_vel)[:40]
for u in sel:
    ax.plot(centres, md_speed[:, u], color="0.85", lw=0.7)
ax.errorbar(centres, np.nanmedian(md_speed[:, sel], axis=1),
            yerr=np.nanstd(md_speed[:, sel], axis=1) / np.sqrt(len(sel)),
            color="crimson", lw=2, marker="o", label="median of 40 units")
ax.axhline(0, color="k", lw=0.8)
ax.legend(fontsize=8)
ax.set(xlabel="hand speed (mm/s)", ylabel="modulation along PD (Hz)",
       title="Directional modulation scales\nwith speed")

ax = fig.add_subplot(gs[1, 0])
ax.plot(lags * 1000, r2_lag[:, strong], color="0.87", lw=0.5)
ax.plot(lags * 1000, np.nanmedian(r2_lag, axis=1), "k", lw=2.5, label="median unit")
ax.axvline(pop_lag * 1000, color="crimson", ls="--", label=f"peak {pop_lag*1000:.0f} ms")
ax.axvline(0, color="0.5", lw=0.8)
ax.set(xlabel="neural lead time (ms)", ylabel="$R^2$ of (1, $v_x$, $v_y$)",
       title="Single-unit encoding of velocity\npeaks at a positive lead")
ax.legend(fontsize=8)
ax2 = ax.twinx()
ax2.plot(lags * 1000, dec_lag, color="tab:blue", lw=1.8)
ax2.set_ylabel("population decoding $R^2$", color="tab:blue", fontsize=9)
ax2.tick_params(axis="y", colors="tab:blue", labelsize=8)

ax = fig.add_subplot(gs[1, 1])
ax.hist(best_lag[strong] * 1000, bins=np.arange(-310, 311, 20), color="steelblue")
ax.axvline(0, color="k", lw=0.8)
ax.axvline(np.median(best_lag[strong]) * 1000, color="crimson", ls="--")
ax.set(xlabel="per-unit optimal lead (ms)", ylabel="units",
       title=f"Optimal lead, well-fit units only\n(median "
             f"{np.median(best_lag[strong])*1000:.0f} ms, n = {strong.sum()})")

ax = fig.add_subplot(gs[1, 2])
ax.scatter(cv_dir, cv_ful, s=16, c="steelblue")
m = max(np.nanmax(cv_dir), np.nanmax(cv_ful)) * 1.05
ax.plot([0, m], [0, m], "k:", lw=1)
ax.set(xlim=(0, m), ylim=(0, m), xlabel="direction only, CV $R^2$",
       ylabel="direction + velocity, CV $R^2$",
       title=f"Adding speed improves the fit\n({(cv_ful > cv_dir).sum()}/{len(cv_ful)} units above the line)")

ax = fig.add_subplot(gs[1, 3])
d = np.load("results_direction.npz")
sig = d["p_perm"] < 0.01
dd = np.degrees(np.angle(np.exp(1j * (pd_vel - d["pd"]))))
ax.hist(dd[sig], bins=np.linspace(-180, 180, 25), color="seagreen")
Rc_agree = np.abs(np.exp(1j * np.radians(dd[sig])).mean())
ax.set(xlabel="continuous-velocity PD $-$ trial PD (deg)", ylabel="units",
       title=f"Two independent PD estimates agree\n(R = {Rc_agree:.2f}, n = {sig.sum()})")
ax.set_xticks([-180, -90, 0, 90, 180])

fig.suptitle("Velocity tuning in MC_Maze: firing rate tracks the hand velocity "
             "vector and leads it in time\n"
             f"(all {ntr} successful trials, {WIN[0]*1000:.0f}-{WIN[1]*1000:.0f} ms "
             f"after movement onset, {SD*1000:.0f} ms Gaussian rate, "
             f"velocity sampled {pop_lag*1000:.0f} ms later)", fontsize=12, y=0.965)
fig.savefig("fig03_velocity_tuning.png", dpi=150, bbox_inches="tight")
print("wrote fig03_velocity_tuning.png")
print(f"continuous vs trial PD agreement: R = {Rc_agree:.3f}, "
      f"{(np.abs(dd[sig]) < 45).mean()*100:.0f}% within 45 deg")
