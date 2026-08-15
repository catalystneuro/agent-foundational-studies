"""Does the same tuning hold across sessions, monkeys, tasks and areas?

Six sessions from three experiments are put through one identical pipeline:

    MC_Maze     x4   monkey Jenkins, motor cortex, delayed maze reaching
    MC_RTT      x1   monkey Indy, M1, self-paced random-target reaching
    Area2_Bump  x1   monkey Han, somatosensory area 2, center-out with bumps

Area 2 is included as a control with a directional prediction: it carries
proprioceptive feedback, so its activity should follow the hand rather than
lead it.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap
from sklearn.linear_model import Ridge
from tqdm import tqdm

import analysis as an
import dandi_io as dio

LAGS = np.round(np.arange(-0.30, 0.301, 0.025), 4)
SD = 0.03
KEYS = ["maze_full", "maze_large", "maze_medium", "maze_small", "rtt", "area2"]


def session_epochs(key, sess):
    """Equal-length analysis windows appropriate to each task."""
    tr = sess["trials"]
    if key == "rtt":
        # Self-paced task with no trial events: use the 600 ms segments the
        # benchmark release already carved the continuous recording into,
        # trimmed slightly so that consecutive segments stay separate.
        return nap.IntervalSet(tr["start_time"], tr["start_time"] + 0.595), None
    ok = np.isfinite(tr["move_onset_time"]) & tr["success"].astype(bool)
    if key == "area2":
        ok &= ~tr["ctr_hold_bump"].astype(bool)   # active reaches only
    idx = np.flatnonzero(ok)
    on = tr["move_onset_time"][idx]
    return nap.IntervalSet(on, on + 0.5), idx


def profile(key):
    sess = dio.load_session(key)
    spikes, pos, vel, support = dio.to_pynapple(sess)
    keep = an.quality_units(spikes, dio.trial_intervals(sess), min_rate=0.5)
    spikes = spikes[list(np.flatnonzero(keep))]
    ep, idx = session_epochs(key, sess)
    counts = spikes.count(an.BIN, ep=ep)
    nep = len(ep)
    rate = an.smooth_rate(counts, sd=SD)
    fold = an.time_block_folds(counts.index.values, n_folds=5, block=30.0)
    enc = np.full((len(LAGS), rate.shape[1]), np.nan)
    dec = np.full(len(LAGS), np.nan)
    dec_folds = np.full((len(LAGS), 5), np.nan)
    for i, lag in enumerate(LAGS):
        mask, V = an.sample_velocity(vel, counts, lag)
        Rt, fd = rate[mask], fold[mask]
        enc[i], _ = an.velocity_regression_r2(Rt, V)
        pred = np.empty_like(V)
        for f in range(5):
            te = fd == f
            pred[te] = Ridge(alpha=1.0).fit(Rt[~te], V[~te]).predict(Rt[te])
            dec_folds[i, f] = np.mean(
                1 - ((V[te] - pred[te]) ** 2).sum(0) / ((V[te] - V[te].mean(0)) ** 2).sum(0))
        dec[i] = np.mean(1 - ((V - pred) ** 2).sum(0) / ((V - V.mean(0)) ** 2).sum(0))

    lead = LAGS[np.nanargmax(dec)]
    mask, V = an.sample_velocity(vel, counts, lead)
    r2_best = enc[np.nanargmax(dec)]
    _, beta = an.velocity_regression_r2(rate[mask], V)
    pd_vel = np.arctan2(beta[2], beta[1])
    centres, md, _ = an.speed_gain(rate[mask], V, n_speed=4, pd=pd_vel)

    # discrete-direction tuning where the task defines one
    tr = sess["trials"]
    dirtuned = np.nan
    if key.startswith("maze"):
        sidx = an.straight_trials(sess)
        theta, _ = an.reach_direction(sess, sidx)
        on = tr["move_onset_time"][sidx]
    elif key == "area2":
        sel = np.flatnonzero(np.isfinite(tr["move_onset_time"]) &
                             tr["success"].astype(bool) &
                             ~tr["ctr_hold_bump"].astype(bool))
        theta = np.radians(tr["cond_dir"][sel])
        on = tr["move_onset_time"][sel]
    else:
        theta, on = None, None
    if theta is not None:
        sp2, _, _, _ = dio.to_pynapple(sess)
        sp2 = sp2[list(np.flatnonzero(keep))]
        rr, _ = an.epoch_rates(sp2, on, on + 0.35)
        fit = an.fit_cosine_tuning(rr, theta, n_perm=300)
        dirtuned = float((fit["p_perm"] < 0.01).mean())
        dir_r2 = float(np.nanmedian(fit["r2"][fit["p_perm"] < 0.01]))
        ntheta = len(theta)
    else:
        dir_r2, ntheta = np.nan, 0

    return dict(key=key, label=str(sess["session_label"]), monkey=str(sess["monkey"]),
                area=str(sess["area"]), n_units=rate.shape[1], n_epochs=nep,
                enc=enc, dec=dec, dec_folds=dec_folds, lead=lead,
                lead_folds=LAGS[np.nanargmax(dec_folds, axis=0)],
                dec_peak=float(np.nanmax(dec)),
                r2_best=r2_best, best_lag=LAGS[np.nanargmax(enc, axis=0)],
                strong=np.nanmax(enc, axis=0) > 0.02,
                centres=centres, md=md, dirtuned=dirtuned, dir_r2=dir_r2,
                ntheta=ntheta)


res = [profile(k) for k in tqdm(KEYS, desc="sessions")]

print(f"{'session':38s} {'units':>5s} {'epochs':>6s} {'lead':>6s} {'decR2':>6s} "
      f"{'medEnc':>7s} {'tuned':>6s} {'dirR2':>6s}")
for r in res:
    tuned = "   n/a" if np.isnan(r["dirtuned"]) else f"{r['dirtuned']*100:5.0f}%"
    print(f"{r['label']:38s} {r['n_units']:5d} {r['n_epochs']:6d} "
          f"{r['lead']*1000:5.0f}ms {r['dec_peak']:6.3f} "
          f"{np.nanmedian(r['r2_best']):7.3f} {tuned:>6s} {r['dir_r2']:6.3f}")

np.savez("results_cross_session.npz",
         **{f"{r['key']}_{k}": r[k] for r in res
            for k in ["enc", "dec", "lead", "dec_peak", "r2_best", "best_lag",
                      "strong", "centres", "md", "dirtuned", "dir_r2", "n_units",
                      "dec_folds", "lead_folds"]},
         lags=LAGS, keys=np.array(KEYS))

# ---------------------------------------------------------------- figure 5
plt.rcParams.update({"axes.titlesize": 10, "axes.labelsize": 9,
                     "xtick.labelsize": 8, "ytick.labelsize": 8})
fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))
fig.subplots_adjust(top=0.76, bottom=0.16, wspace=0.36)
colors = plt.get_cmap("tab10")(np.arange(len(res)))
short = [r["label"].split(" (")[0] + " / " + r["monkey"] for r in res]

ax = axes[0]
for r, c, s in zip(res, colors, short):
    ax.plot(LAGS * 1000, r["dec"], color=c, lw=1.8,
            ls="--" if r["area"] == "area2" else "-", label=s)
    ax.plot(r["lead"] * 1000, r["dec_peak"], "o", color=c, ms=5)
ax.axvline(0, color="0.5", lw=0.8)
ax.set(xlabel="neural lead time (ms)", ylabel="velocity decoding $R^2$",
       title="Motor cortex leads the hand;\narea 2 follows it")
ax.legend(fontsize=6.5, loc="lower center")

ax = axes[1]
y = np.arange(len(res))
ax.barh(y, [r["dec_peak"] for r in res], color=colors)
for i, r in enumerate(res):
    lo, hi = r["lead_folds"].min() * 1000, r["lead_folds"].max() * 1000
    ax.text(r["dec_peak"] + 0.01, i,
            f"{r['lead']*1000:+.0f} ms  [{lo:+.0f}, {hi:+.0f}]", va="center", fontsize=7.5)
ax.set_yticks(y)
ax.set_yticklabels(short, fontsize=7.5)
ax.set(xlabel="peak decoding $R^2$", xlim=(0, 1.05),
       title="Velocity is decodable in every\nsession (label = optimal lead)")
ax.invert_yaxis()

ax = axes[2]
# Discrete reach directions exist in every session except the self-paced task.
has_dir = [i for i, r in enumerate(res) if not np.isnan(r["dirtuned"])]
ax.bar(np.arange(len(has_dir)), [res[i]["dirtuned"] * 100 for i in has_dir],
       color=[colors[i] for i in has_dir])
for j, i in enumerate(has_dir):
    ax.text(j, res[i]["dirtuned"] * 100 + 1.5,
            f"{res[i]['ntheta']} trials\n$R^2$={res[i]['dir_r2']:.2f}",
            ha="center", fontsize=7)
ax.set_xticks(np.arange(len(has_dir)))
ax.set_xticklabels([short[i].split(" / ")[0].replace("MC_Maze ", "maze ")
                    for i in has_dir], rotation=35, ha="right", fontsize=7.5)
ax.set(ylabel="% of units direction tuned", ylim=(0, 105),
       title="Cosine tuning is detected wherever\nthere are enough trials")

ax = axes[3]
for r, c in zip(res, colors):
    ax.plot(r["centres"], np.nanmedian(r["md"], axis=1), marker="o", color=c,
            ls="--" if r["area"] == "area2" else "-")
ax.axhline(0, color="k", lw=0.8)
ax.set(xlabel="hand speed (mm/s)", ylabel="median modulation along PD (Hz)",
       title="Speed scaling of directional\nmodulation in every session")

fig.suptitle("The same direction and velocity tuning appears in six sessions across "
             "three experiments, two tasks, three monkeys and two cortical areas",
             fontsize=12, y=0.96)
fig.savefig("fig05_cross_session.png", dpi=150, bbox_inches="tight")
print("wrote fig05_cross_session.png")
