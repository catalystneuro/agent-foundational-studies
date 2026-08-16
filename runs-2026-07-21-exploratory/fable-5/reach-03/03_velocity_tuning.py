"""Stage 3: continuous velocity tuning - optimal lag, 2D velocity fields, speed gain."""

import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap
from tqdm import tqdm

import reachlib as rl

BIN = 0.02
SMOOTH_STD = 0.04   # s; symmetric Gaussian, so it cannot bias the lag estimate
SPEED_MIN = 50.0    # mm/s; direction is undefined when the hand is nearly still
MIN_SPIKES = 300    # a unit needs this many spikes during movement to be analysed

d = rl.load_mc_maze()
units, hand_vel, epochs, areas = d["units"], d["hand_vel"], d["epochs"], d["areas"]

counts = units.count(BIN, ep=epochs)
rate = counts.smooth(std=SMOOTH_STD) / BIN
tb = np.asarray(counts.t)
vx_raw, vy_raw = hand_vel.values[:, 0], hand_vel.values[:, 1]
tv = np.asarray(hand_vel.t)
print("bins: %d (%.0f s), units: %d" % (len(tb), len(tb) * BIN, counts.shape[1]))


def lagged_velocity(lag):
    """Velocity sampled ``lag`` seconds after each spike-count bin centre.

    Positive lag means the neural activity leads the movement. Bins whose
    shifted sample would fall outside the behavioural epoch are masked out.
    """
    tq = tb + lag
    i = np.searchsorted(epochs.start, tq, side="right") - 1
    inside = np.zeros(tq.size, dtype=bool)
    ok = i >= 0
    inside[ok] = tq[ok] <= epochs.end[i[ok]]
    return np.interp(tq, tv, vx_raw), np.interp(tq, tv, vy_raw), inside


def linear_velocity_r2(vx, vy, mask, Y):
    """R^2 of  rate ~ b0 + bx*vx + by*vy  fitted per unit."""
    X = np.column_stack([np.ones(mask.sum()), vx[mask], vy[mask]])
    Ym = Y[mask]
    beta, *_ = np.linalg.lstsq(X, Ym, rcond=None)
    res = Ym - X.dot(beta)
    ss_tot = ((Ym - Ym.mean(0)) ** 2).sum(0)
    return 1 - (res ** 2).sum(0) / np.where(ss_tot > 0, ss_tot, np.nan), beta


# Units that barely fire during movement give meaningless tuning estimates.
vx0, vy0, inside0 = lagged_velocity(0.0)
moving = inside0 & (np.hypot(vx0, vy0) > SPEED_MIN)
spikes_moving = counts.values[moving].sum(0)
keep_u = spikes_moving >= MIN_SPIKES
print("units with >=%d spikes during movement: %d/%d"
      % (MIN_SPIKES, keep_u.sum(), keep_u.size))
R = np.asarray(rate.values)[:, keep_u]
uidx = np.flatnonzero(keep_u)

# --- optimal lag between neural activity and hand velocity ----------------
lags = np.arange(-0.30, 0.401, 0.02)
r2_lag = np.empty((lags.size, keep_u.sum()))
for k, lag in enumerate(tqdm(lags, desc="lag sweep")):
    vx, vy, inside = lagged_velocity(lag)
    mask = inside & (np.hypot(vx, vy) > SPEED_MIN)
    r2_lag[k], _ = linear_velocity_r2(vx, vy, mask, R)

best_lag = lags[r2_lag.argmax(0)]
pop_curve = np.nanmedian(r2_lag, axis=1)
pop_lag = lags[pop_curve.argmax()]
print("population-optimal lag: %+.0f ms (positive = neural leads movement)" % (1000 * pop_lag))
print("per-unit best lag: median %+.0f ms, IQR %+.0f to %+.0f ms"
      % (1000 * np.median(best_lag), *(1000 * np.percentile(best_lag, [25, 75]))))

vx, vy, inside = lagged_velocity(pop_lag)
speed = np.hypot(vx, vy)
mask = inside & (speed > SPEED_MIN)
r2_lin, beta_lin = linear_velocity_r2(vx, vy, mask, R)
pd_cont = np.arctan2(beta_lin[2], beta_lin[1])
print("linear velocity model at the optimal lag: median R2 = %.3f (max %.3f)"
      % (np.nanmedian(r2_lin), np.nanmax(r2_lin)))

# --- 2D tuning curves in velocity space -----------------------------------
# Occupancy is strongly non-uniform over the velocity plane, so the maps are
# built explicitly from spike counts divided by dwell time per cell.
theta_c = np.arctan2(vy[mask], vx[mask])
Cm = counts.values[mask][:, keep_u].astype(float)

v_edges = np.linspace(-700, 700, 15)
tc_vel, occ_vel = rl.rate_map_2d(Cm, vx[mask], vy[mask], v_edges, v_edges, BIN,
                                 min_occupancy=30)
th_edges = np.linspace(-np.pi, np.pi, 17)
sp_edges = np.linspace(SPEED_MIN, 900, 8)
tc_ds, occ_ds = rl.rate_map_2d(Cm, theta_c, speed[mask], th_edges, sp_edges, BIN,
                               min_occupancy=30)
print("velocity-plane cells with enough data: %d/%d; direction x speed: %d/%d"
      % ((occ_vel >= 30).sum(), occ_vel.size, (occ_ds >= 30).sum(), occ_ds.size))

np.savez("cache/velocity_tuning.npz", lags=lags, r2_lag=r2_lag, best_lag=best_lag,
         pop_lag=pop_lag, r2_lin=r2_lin, beta_lin=beta_lin, pd_cont=pd_cont,
         keep_u=keep_u, uidx=uidx)

# --- Figure 5: lag and velocity fields ------------------------------------
dtun = np.load("cache/direction_tuning.npz", allow_pickle=True)
strong = np.argsort(-np.hypot(beta_lin[1], beta_lin[2]))[:4]

fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(1, 4, top=0.915, bottom=0.755, wspace=0.42)
gs_v = fig.add_gridspec(1, 4, top=0.715, bottom=0.435, wspace=0.42)
gs_p = fig.add_gridspec(1, 4, top=0.345, bottom=0.02, wspace=0.45)

ax = fig.add_subplot(gs[0])
ax.plot(1000 * lags, pop_curve, "k", lw=2)
q1, q3 = np.nanpercentile(r2_lag, [25, 75], axis=1)
ax.fill_between(1000 * lags, q1, q3, color="0.8")
ax.axvline(1000 * pop_lag, color="crimson", ls="--")
ax.axvline(0, color="0.5", lw=0.7)
ax.set_xlabel("lag (ms); >0 = neural leads")
ax.set_ylabel("$R^2$ (median, IQR)")
ax.set_title("velocity encoding peaks when\nactivity leads by %d ms" % (1000 * pop_lag),
             fontsize=9)

ax = fig.add_subplot(gs[1])
ax.hist(1000 * best_lag, bins=lags.size, color="tab:blue")
ax.axvline(1000 * np.median(best_lag), color="crimson", ls="--")
ax.set_xlabel("best lag per unit (ms)")
ax.set_ylabel("units")
ax.set_title("lag distribution\n(median %+d ms)" % (1000 * np.median(best_lag)), fontsize=9)

ax = fig.add_subplot(gs[2])
ax.hist(r2_lin, bins=25, color="0.4")
ax.axvline(np.nanmedian(r2_lin), color="crimson", ls="--")
ax.set_xlabel("$R^2$, linear velocity model")
ax.set_ylabel("units")
ax.set_title("variance of the smoothed rate\nexplained by velocity", fontsize=9)

ax = fig.add_subplot(gs[3])
ok = dtun["reliable_mov"][uidx]
dpd = rl.circ_diff(dtun["pd_mov"][uidx][ok], pd_cont[ok])
med = np.degrees(np.median(np.abs(dpd)))
frac = np.mean(np.abs(dpd) < np.pi / 4)
sc = ax.scatter(np.degrees(dtun["pd_mov"][uidx][ok]), np.degrees(pd_cont[ok]), s=18,
                c=r2_lin[ok], cmap="viridis")
ax.plot([-180, 180], [-180, 180], "k--", lw=0.8)
ax.set_xlabel("PD from trial reaches (deg)")
ax.set_ylabel("PD from continuous velocity (deg)")
ax.set_title("two independent PD estimates\nmedian |ΔPD| = %.0f° (chance 90°)" % med, fontsize=9)
plt.colorbar(sc, ax=ax, fraction=0.046, label="$R^2$")
print("trial PD vs continuous PD: median |dPD| %.0f deg (chance 90), %.0f%% within 45 deg, n=%d"
      % (med, 100 * frac, ok.sum()))

for col, u in enumerate(strong):
    ax = fig.add_subplot(gs_v[col])
    extent = [v_edges[0], v_edges[-1], v_edges[0], v_edges[-1]]
    im = ax.imshow(tc_vel[:, :, u].T, origin="lower", extent=extent, cmap="viridis",
                   aspect="equal")
    ax.set_xlabel("vx (mm/s)")
    if col == 0:
        ax.set_ylabel("vy (mm/s)")
    ax.set_title("unit %d ($R^2$=%.2f)" % (uidx[u], r2_lin[u]), fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, label="Hz" if col == 3 else "")

for col, u in enumerate(strong):
    ax = fig.add_subplot(gs_p[col], projection="polar")
    T, S = np.meshgrid(th_edges, sp_edges, indexing="ij")
    pc = ax.pcolormesh(T, S, np.ma.masked_invalid(tc_ds[:, :, u]), cmap="magma",
                       shading="flat")
    ax.set_thetagrids([0, 90, 180, 270], labels=["0°", "", "180°", ""])
    ax.set_rticks([300, 600, 900])
    ax.set_rlabel_position(45)
    ax.tick_params(labelsize=6)
    ax.set_title("unit %d" % uidx[u], fontsize=9, pad=16)
    plt.colorbar(pc, ax=ax, fraction=0.046, pad=0.14, label="Hz" if col == 3 else "")

fig.text(0.5, 0.728, "firing rate over the velocity plane (lag-corrected)",
         ha="center", fontsize=11.5)
fig.text(0.5, 0.385, "firing rate over direction (angle) × speed (radius, mm/s)",
         ha="center", fontsize=11.5)
fig.suptitle("Continuous velocity tuning in motor cortex", fontsize=13, y=0.965)
fig.savefig("fig05_velocity_fields.png", dpi=150)
plt.close(fig)

# --- Figure 6: speed gain at the preferred direction ----------------------
sp_edges = np.array([50, 100, 150, 200, 275, 350, 450, 600, 800])
sp_ctr = (sp_edges[:-1] + sp_edges[1:]) / 2
dth = rl.circ_diff(theta_c[None, :], pd_cont[:, None])
near = np.abs(dth) < np.pi / 4
away = np.abs(dth) > 3 * np.pi / 4
C = counts.values[mask][:, keep_u] / BIN
sp_bin = np.digitize(speed[mask], sp_edges) - 1

n_u = keep_u.sum()
gain_near = np.full((n_u, sp_ctr.size), np.nan)
gain_away = np.full_like(gain_near, np.nan)
for u in range(n_u):
    for b in range(sp_ctr.size):
        m1 = near[u] & (sp_bin == b)
        m2 = away[u] & (sp_bin == b)
        if m1.sum() > 50:
            gain_near[u, b] = C[m1, u].mean()
        if m2.sum() > 50:
            gain_away[u, b] = C[m2, u].mean()

top = np.argsort(-np.nan_to_num(r2_lin))[:25]
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
ax = axes[0]
for u in top:
    ax.plot(sp_ctr, gain_near[u], color="tab:red", alpha=0.22, lw=0.8)
    ax.plot(sp_ctr, gain_away[u], color="tab:blue", alpha=0.22, lw=0.8)
ax.plot(sp_ctr, np.nanmean(gain_near[top], 0), color="tab:red", lw=2.8,
        label="movement within 45° of PD")
ax.plot(sp_ctr, np.nanmean(gain_away[top], 0), color="tab:blue", lw=2.8,
        label="movement opposite to PD")
ax.set_xlabel("speed (mm/s)")
ax.set_ylabel("firing rate (Hz)")
ax.set_title("25 most velocity-tuned units:\nspeed scales the response", fontsize=10)
ax.legend(fontsize=8)

norm = np.nanmax(gain_near, axis=1, keepdims=True)
ax = axes[1]
gn = gain_near / norm
ga = gain_away / np.nanmax(gain_near, axis=1, keepdims=True)
for g, c, lab in [(gn, "tab:red", "at PD"), (ga, "tab:blue", "opposite to PD")]:
    med = np.nanmedian(g, 0)
    lo, hi = np.nanpercentile(g, [25, 75], axis=0)
    ax.plot(sp_ctr, med, color=c, lw=2.5, label=lab)
    ax.fill_between(sp_ctr, lo, hi, color=c, alpha=0.2)
ax.set_xlabel("speed (mm/s)")
ax.set_ylabel("rate / peak rate at PD")
ax.set_title("all %d analysed units,\nmedian and IQR" % n_u, fontsize=10)
ax.legend(fontsize=8)


def slope(g):
    ok = np.isfinite(g)
    return np.polyfit(sp_ctr[ok], g[ok], 1)[0] if ok.sum() > 2 else np.nan


slope_near = np.array([slope(g) for g in gain_near])
slope_away = np.array([slope(g) for g in gain_away])
ax = axes[2]
ax.scatter(1000 * slope_away, 1000 * slope_near, s=18, color="tab:blue")
lim = 1000 * np.nanmax(np.abs(np.r_[slope_near, slope_away])) * 1.05
ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8)
ax.axhline(0, color="0.6", lw=0.6)
ax.axvline(0, color="0.6", lw=0.6)
ax.set_xlabel("speed slope, anti-PD (Hz per m/s)")
ax.set_ylabel("speed slope, PD (Hz per m/s)")
ax.set_title("speed gain is direction-dependent", fontsize=10)
fig.suptitle("Speed modulation of motor-cortical firing", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig("fig06_speed_gain.png", dpi=150)
plt.close(fig)

n_pos = np.sum(slope_near > slope_away)
print("speed slope at PD: mean %+.1f Hz per m/s; anti-PD: %+.1f Hz per m/s"
      % (1000 * np.nanmean(slope_near), 1000 * np.nanmean(slope_away)))
print("PD slope exceeds anti-PD slope in %d/%d units" % (n_pos, n_u))
np.savez("cache/speed_gain.npz", sp_ctr=sp_ctr, gain_near=gain_near, gain_away=gain_away,
         slope_near=slope_near, slope_away=slope_away)
