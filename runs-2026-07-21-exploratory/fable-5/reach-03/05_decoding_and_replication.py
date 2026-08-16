"""Stage 5: population decoding of hand velocity, and replication in a second dataset.

Part A decodes hand velocity from the MC_Maze population, which tests whether
direction and speed tuning are strong enough to reconstruct the movement.
Part B repeats the continuous tuning analysis on MC_RTT (DANDI 000129), a
different monkey performing a self-paced random-target task, as an independent
check that the effects are not specific to one session or task design.
"""

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import reachlib as rl

BIN = 0.02
SPEED_MIN = 50.0
DECODE_LAGS = np.arange(0, 0.201, 0.02)  # neural bins preceding the velocity sample
N_FOLDS = 5

# =========================== Part A: decoding =============================
d = rl.load_mc_maze()
units, hand_vel, epochs = d["units"], d["hand_vel"], d["epochs"]
counts = units.count(BIN, ep=epochs)
tb = np.asarray(counts.t)
C = np.asarray(counts.values, dtype=float)
ep_id = rl.epoch_index(epochs, tb)

vx = np.interp(tb, hand_vel.t, hand_vel.values[:, 0])
vy = np.interp(tb, hand_vel.t, hand_vel.values[:, 1])

# Lagged design matrix: velocity at bin i is predicted from spike counts in the
# bins preceding it. Bins whose lagged neighbour sits in another trial are dropped.
n_lag = len(DECODE_LAGS)
shifts = np.round(DECODE_LAGS / BIN).astype(int)
X = np.zeros((len(tb), C.shape[1] * n_lag))
valid = np.ones(len(tb), dtype=bool)
for k, sh in enumerate(shifts):
    idx = np.arange(len(tb)) - sh
    ok = (idx >= 0) & (ep_id[np.clip(idx, 0, None)] == ep_id)
    X[ok, k * C.shape[1]:(k + 1) * C.shape[1]] = C[idx[ok]]
    valid &= ok
Yv = np.column_stack([vx, vy])
valid &= ep_id >= 0
print("decoding samples: %d bins, %d features" % (valid.sum(), X.shape[1]))

Xv, Yvv, ep_v = X[valid], Yv[valid], ep_id[valid]
fold = np.floor(N_FOLDS * ep_v / (ep_v.max() + 1)).astype(int)
pred = np.zeros_like(Yvv)
for f in tqdm(range(N_FOLDS), desc="ridge decoding"):
    tr, te = fold != f, fold == f
    mu, sd = Xv[tr].mean(0), Xv[tr].std(0) + 1e-9
    A = (Xv[tr] - mu) / sd
    b = Yvv[tr] - Yvv[tr].mean(0)
    W = np.linalg.solve(A.T @ A + 1e3 * np.eye(A.shape[1]), A.T @ b)
    pred[te] = ((Xv[te] - mu) / sd) @ W + Yvv[tr].mean(0)

ss_res = ((Yvv - pred) ** 2).sum(0)
ss_tot = ((Yvv - Yvv.mean(0)) ** 2).sum(0)
r2_dec = 1 - ss_res / ss_tot
sp_true, sp_pred = np.hypot(*Yvv.T), np.hypot(*pred.T)
mov = sp_true > 100
ang_err = rl.circ_diff(np.arctan2(pred[mov, 1], pred[mov, 0]),
                       np.arctan2(Yvv[mov, 1], Yvv[mov, 0]))
print("cross-validated decoding R2: vx %.3f, vy %.3f" % tuple(r2_dec))
print("speed correlation r = %.3f" % np.corrcoef(sp_true, sp_pred)[0, 1])
print("direction error: median |err| %.0f deg, %.0f%% within 45 deg (n=%d bins)"
      % (np.degrees(np.median(np.abs(ang_err))),
         100 * np.mean(np.abs(ang_err) < np.pi / 4), mov.sum()))

fig = plt.figure(figsize=(13.5, 7.5))
gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.32, width_ratios=[1.9, 1, 1])
seg = slice(2000, 2600)
for row, (lab, k) in enumerate([("vx", 0), ("vy", 1)]):
    ax = fig.add_subplot(gs[row, 0])
    ax.plot(np.arange(seg.stop - seg.start) * BIN, Yvv[seg, k], "k", lw=1.4, label="measured")
    ax.plot(np.arange(seg.stop - seg.start) * BIN, pred[seg, k], color="crimson", lw=1.4,
            label="decoded")
    ax.set_ylabel("%s (mm/s)" % lab)
    if row == 0:
        ax.legend(fontsize=8, ncol=2)
        ax.set_title("held-out segment: population decoding of hand velocity", fontsize=10)
    else:
        ax.set_xlabel("time within held-out segment (s)")

ax = fig.add_subplot(gs[0, 1])
sub = np.random.default_rng(0).choice(len(Yvv), 4000, replace=False)
ax.scatter(Yvv[sub, 0], pred[sub, 0], s=3, alpha=0.2, color="tab:blue")
lim = np.percentile(np.abs(Yvv[:, 0]), 99.5)
ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8)
ax.set_xlabel("measured vx (mm/s)")
ax.set_ylabel("decoded vx (mm/s)")
ax.set_title("$R^2$ = %.2f" % r2_dec[0], fontsize=10)

ax = fig.add_subplot(gs[0, 2])
ax.scatter(sp_true[sub], sp_pred[sub], s=3, alpha=0.2, color="tab:green")
ax.plot([0, sp_true.max()], [0, sp_true.max()], "k--", lw=0.8)
ax.set_xlabel("measured speed (mm/s)")
ax.set_ylabel("decoded speed (mm/s)")
ax.set_title("speed: r = %.2f" % np.corrcoef(sp_true, sp_pred)[0, 1], fontsize=10)

ax = fig.add_subplot(gs[1, 1], projection="polar")
h, e = np.histogram(ang_err, bins=np.linspace(-np.pi, np.pi, 37))
ax.bar(e[:-1], h, width=np.diff(e), align="edge", color="tab:purple", alpha=0.8)
ax.set_thetagrids([0, 90, 180, 270], labels=["0°", "", "180°", ""])
ax.set_yticklabels([])
ax.tick_params(labelsize=7)
ax.set_title("decoded direction error\n(bins with speed > 100 mm/s)", fontsize=10, pad=22)

ax = fig.add_subplot(gs[1, 2])
ax.bar(["vx", "vy", "speed"], [r2_dec[0], r2_dec[1],
                               np.corrcoef(sp_true, sp_pred)[0, 1] ** 2], color="0.4")
ax.set_ylabel("cross-validated $R^2$ (speed: $r^2$)")
ax.set_title("decoding accuracy\n(%d units, %d ms lags)" % (C.shape[1], 1000 * DECODE_LAGS[-1]),
             fontsize=10)
fig.suptitle("Hand velocity decoded from the motor-cortical population (MC_Maze)", fontsize=13)
fig.savefig("fig08_velocity_decoding.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ======================= Part B: MC_RTT replication =======================
r = rl.load_mc_rtt()
rtt_units, rtt_vel, rtt_ep = r["units"], r["finger_vel"], r["epochs"]
print("\nMC_RTT: %d units, %.0f s of self-paced reaching" % (len(rtt_units),
                                                             rtt_ep.tot_length()))
res = rl.continuous_velocity_tuning(rtt_units, rtt_vel, rtt_ep, bin_size=BIN,
                                    speed_min=SPEED_MIN, min_spikes=300)
print("MC_RTT units analysed: %d/%d" % (res["keep_u"].sum(), len(rtt_units)))
print("MC_RTT population-optimal lag: %+.0f ms" % (1000 * res["pop_lag"]))
print("MC_RTT velocity model: median R2 = %.3f (max %.3f)"
      % (np.nanmedian(res["r2_lin"]), np.nanmax(res["r2_lin"])))

Cm = np.asarray(res["counts"].values)[res["mask"]][:, res["keep_u"]].astype(float)
sp_moving = res["speed"][res["mask"]]
rtt_speed_edges = np.linspace(SPEED_MIN, np.percentile(sp_moving, 99), 9)
print("MC_RTT speed range analysed: %.0f-%.0f mm/s (median %.0f)"
      % (rtt_speed_edges[0], rtt_speed_edges[-1], np.median(sp_moving)))
centres, g_near, g_away = rl.speed_gain_curves(Cm, res["theta"], sp_moving,
                                               res["pd_cont"], BIN,
                                               speed_edges=rtt_speed_edges)


def slope(g):
    ok = np.isfinite(g)
    return np.polyfit(centres[ok], g[ok], 1)[0] if ok.sum() > 2 else np.nan


sl_near = np.array([slope(g) for g in g_near])
sl_away = np.array([slope(g) for g in g_away])
print("MC_RTT speed slope at PD %+.1f Hz per m/s, anti-PD %+.1f; PD steeper in %d/%d units"
      % (1000 * np.nanmean(sl_near), 1000 * np.nanmean(sl_away),
         np.sum(sl_near > sl_away), len(sl_near)))

th_edges = np.linspace(-np.pi, np.pi, 17)
sp_edges = np.linspace(SPEED_MIN, np.percentile(sp_moving, 99), 7)
tc_ds, occ = rl.rate_map_2d(Cm, res["theta"], sp_moving, th_edges, sp_edges,
                            BIN, min_occupancy=30)
best = np.argsort(-np.nan_to_num(res["r2_lin"]))[:3]

maze = np.load("cache/velocity_tuning.npz")
fig = plt.figure(figsize=(14, 8))
gs = fig.add_gridspec(2, 4, hspace=0.5, wspace=0.42)

ax = fig.add_subplot(gs[0, 0])
ax.plot(1000 * res["lags"], res["pop_curve"], "k", lw=2, label="MC_RTT (Indy)")
ax.plot(1000 * maze["lags"], np.nanmedian(maze["r2_lag"], axis=1), color="tab:orange", lw=2,
        label="MC_Maze (Jenkins)")
ax.axvline(0, color="0.5", lw=0.7)
ax.axvline(1000 * res["pop_lag"], color="k", ls="--", lw=0.9)
ax.set_xlabel("lag (ms); >0 = neural leads")
ax.set_ylabel("median $R^2$")
ax.set_title("both datasets peak with\nactivity leading movement", fontsize=9)
ax.legend(fontsize=7)

ax = fig.add_subplot(gs[0, 1], projection="polar")
h, e = np.histogram(res["pd_cont"], bins=np.linspace(-np.pi, np.pi, 17))
ax.bar(e[:-1], h, width=np.diff(e), align="edge", color="tab:blue", alpha=0.8)
ax.set_thetagrids([0, 90, 180, 270], labels=["0°", "", "180°", ""])
ax.tick_params(labelsize=7)
ax.set_title("MC_RTT preferred\ndirections (n=%d)" % res["keep_u"].sum(), fontsize=9, pad=20)

ax = fig.add_subplot(gs[0, 2])
gn = g_near / np.nanmax(g_near, axis=1, keepdims=True)
ga = g_away / np.nanmax(g_near, axis=1, keepdims=True)
for g, c, lab in [(gn, "tab:red", "at PD"), (ga, "tab:blue", "opposite to PD")]:
    ax.plot(centres, np.nanmedian(g, 0), color=c, lw=2.5, label=lab)
    lo, hi = np.nanpercentile(g, [25, 75], axis=0)
    ax.fill_between(centres, lo, hi, color=c, alpha=0.2)
ax.set_xlabel("speed (mm/s)")
ax.set_ylabel("rate / peak rate at PD")
ax.set_title("MC_RTT speed gain\n(median and IQR)", fontsize=9)
ax.legend(fontsize=7)

ax = fig.add_subplot(gs[0, 3])
ax.scatter(1000 * sl_away, 1000 * sl_near, s=18, color="tab:blue")
lim = 1000 * np.nanmax(np.abs(np.r_[sl_near, sl_away])) * 1.05
ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8)
ax.axhline(0, color="0.6", lw=0.6)
ax.axvline(0, color="0.6", lw=0.6)
ax.set_xlabel("speed slope, anti-PD (Hz per m/s)")
ax.set_ylabel("speed slope, PD (Hz per m/s)")
ax.set_title("MC_RTT: speed gain is\ndirection-dependent", fontsize=9)

for col, u in enumerate(best):
    ax = fig.add_subplot(gs[1, col], projection="polar")
    T, S = np.meshgrid(th_edges, sp_edges, indexing="ij")
    pc = ax.pcolormesh(T, S, np.ma.masked_invalid(tc_ds[:, :, u]), cmap="magma", shading="flat")
    ax.set_thetagrids([0, 90, 180, 270], labels=["0°", "", "180°", ""])
    ax.set_rticks(np.round(sp_edges[1::2]).astype(int))
    ax.set_rlabel_position(45)
    ax.tick_params(labelsize=6)
    ax.set_title("MC_RTT unit %d\n$R^2$ = %.2f" % (res["uidx"][u], res["r2_lin"][u]),
                 fontsize=9, pad=16)
    plt.colorbar(pc, ax=ax, fraction=0.046, pad=0.16, label="Hz" if col < 2 else "")

ax = fig.add_subplot(gs[1, 3])
ax.hist(res["r2_lin"], bins=20, color="0.4")
ax.axvline(np.nanmedian(res["r2_lin"]), color="crimson", ls="--")
ax.set_xlabel("$R^2$, linear velocity model")
ax.set_ylabel("units")
ax.set_title("MC_RTT velocity tuning\nstrength", fontsize=9)

fig.suptitle("Replication in a second dataset: MC_RTT, monkey Indy, self-paced random targets "
             "(DANDI 000129)", fontsize=12.5)
fig.savefig("fig09_mc_rtt_replication.png", dpi=150, bbox_inches="tight")
plt.close(fig)

np.savez("cache/rtt.npz", pop_lag=res["pop_lag"], r2_lin=res["r2_lin"],
         pd_cont=res["pd_cont"], slope_near=sl_near, slope_away=sl_away,
         r2_dec=r2_dec, ang_err=ang_err)
