import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import pynapple as nap
from reach_lib import open_nwb, obs_intervals_set, load_units

BIN = 0.01
WIN_W = 0.30
DELAY_WIN = (0.10, 0.40)    # relative to target onset (planning epoch, before go cue)

nwbfile = open_nwb("MC_Maze")
ep = obs_intervals_set(nwbfile)
units = load_units(nwbfile, ep)
spk = [units[k].t for k in units.index]
trials = pd.read_pickle("_cache_trials.pkl")
t = np.load("_cache_t.npy"); vel = np.load("_cache_vel.npy")

# ---------- trial selection & reach direction measured from the hand itself ----------
sel = trials[(trials.num_barriers == 0) & trials.success].copy()
onset = sel.move_onset_time.values
i0, i1 = np.searchsorted(t, onset), np.searchsorted(t, onset + 0.2)
mv = np.stack([vel[a:b].mean(0) for a, b in zip(i0, i1)])
theta = np.arctan2(mv[:, 1], mv[:, 0])
peak_speed = np.array([np.hypot(*vel[a:b].T).max() for a, b in zip(i0, np.searchsorted(t, onset + 0.4))])
tgt = np.stack([r.target_pos[r.active_target] for r in sel.itertuples()]).astype(float)
circ_err = np.abs(np.angle(np.exp(1j * (theta - np.arctan2(tgt[:, 1], tgt[:, 0])))))
print("n straight trials: %d" % len(sel))
print("hand-vs-target direction disagreement: median %.1f deg, p95 %.1f deg"
      % (np.degrees(np.median(circ_err)), np.degrees(np.percentile(circ_err, 95))))
print("peak speed: median %.0f mm/s (range %.0f-%.0f)" % (np.median(peak_speed), peak_speed.min(), peak_speed.max()))
print("window containment: min(onset-start)=%.2f s, min(stop-onset)=%.2f s"
      % ((onset - sel.start_time.values).min(), (sel.stop_time.values - onset).min()))


def epoch_rates(events, win):
    a, b = events + win[0], events + win[1]
    cnt = np.stack([np.searchsorted(s, b) - np.searchsorted(s, a) for s in spk], 1)
    return cnt / (win[1] - win[0])


def cosine_fit(R, th, n_perm=1000, seed=0):
    """r = b0 + depth*cos(th - PD), fit by least squares; permutation test on depth."""
    X = np.stack([np.ones_like(th), np.cos(th), np.sin(th)], 1)
    with np.errstate(all="ignore"):
        B = np.linalg.lstsq(X, R, rcond=None)[0]
        pred = X @ B
        ss_tot = ((R - R.mean(0)) ** 2).sum(0)
        r2 = 1 - ((R - pred) ** 2).sum(0) / np.where(ss_tot > 0, ss_tot, np.nan)
    depth, pd_ = np.hypot(B[1], B[2]), np.arctan2(B[2], B[1])
    out = dict(b0=B[0], depth=depth, pd=pd_, r2=r2)
    if n_perm:
        rng = np.random.default_rng(seed)
        null = np.empty((n_perm, R.shape[1]))
        for i in range(n_perm):
            Bs = np.linalg.lstsq(X[rng.permutation(len(th))], R, rcond=None)[0]
            null[i] = np.hypot(Bs[1], Bs[2])
        out["p"] = (null >= depth).mean(0)
    return out


# ---------- scan the analysis window: when is direction signal strongest? ----------
offsets = np.round(np.arange(-0.30, 0.31, 0.05), 3)
scan = []
for o in offsets:
    f = cosine_fit(epoch_rates(onset, (o, o + WIN_W)), theta, n_perm=0)
    scan.append(np.nanmean(f["depth"]))
scan = np.array(scan)
BEST = offsets[np.argmax(scan)]
print("best 300 ms window starts %.2f s relative to movement onset "
      "(mean depth %.2f Hz vs %.2f Hz at onset)" % (BEST, scan.max(), scan[np.isclose(offsets, 0)][0]))
MOVE_WIN = (BEST, BEST + WIN_W)

R_move = epoch_rates(onset, MOVE_WIN)
R_delay = epoch_rates(sel.target_on_time.values, DELAY_WIN)
alive = R_move.sum(0) > 0
print("units with spikes in movement window: %d/%d" % (alive.sum(), len(alive)))

fit_move, fit_delay = cosine_fit(R_move, theta), cosine_fit(R_delay, theta)
sig = (fit_move["p"] < 0.01) & alive
print("directionally tuned during movement (perm p<0.01): %d/%d (%.0f%%)" % (sig.sum(), alive.sum(), 100 * sig.sum() / alive.sum()))
print("directionally tuned during delay: %d/%d" % (((fit_delay["p"] < 0.01) & alive).sum(), alive.sum()))
print("median modulation depth (tuned units): %.1f Hz" % np.median(fit_move["depth"][sig]))
print("median single-trial cosine R2 (tuned): %.3f" % np.median(fit_move["r2"][sig]))

# ---------- empirical tuning curves on 30-deg bins (target sampling is non-uniform) ----------
NB = 12
edges = np.linspace(-np.pi, np.pi, NB + 1)
dbin = np.clip(np.digitize(theta, edges) - 1, 0, NB - 1)
n_per = np.bincount(dbin, minlength=NB)
keep = n_per >= 15
centers = (edges[:-1] + edges[1:]) / 2
print("direction bins kept: %d/%d (trials/bin %s)" % (keep.sum(), NB, n_per[keep]))

tc = np.full((NB, R_move.shape[1]), np.nan); tc_sem = np.full_like(tc, np.nan)
for b in np.where(keep)[0]:
    tc[b] = R_move[dbin == b].mean(0)
    tc_sem[b] = R_move[dbin == b].std(0) / np.sqrt(n_per[b])

# R^2 of the cosine against the direction-binned means (the classic Georgopoulos measure)
kb = np.where(keep)[0]
cpred = fit_move["b0"] + fit_move["depth"] * np.cos(centers[kb][:, None] - fit_move["pd"])
with np.errstate(all="ignore"):
    sst = ((tc[kb] - tc[kb].mean(0)) ** 2).sum(0)
    r2_binned = 1 - ((tc[kb] - cpred) ** 2).sum(0) / np.where(sst > 0, sst, np.nan)
print("median cosine R2 on direction-binned means (tuned): %.2f" % np.median(r2_binned[sig]))

# ---------- PSTHs (all windows verified to lie inside observed trial epochs) ----------
W = (-0.5, 0.7)
grid = np.arange(0, t[-1] + BIN, BIN)
C = gaussian_filter1d(np.stack([np.histogram(s, bins=grid)[0] for s in spk], 1) / BIN, 2.5, axis=0)
off = np.arange(int(round(W[0] / BIN)), int(round(W[1] / BIN)))
psth = C[np.searchsorted(grid, onset)[:, None] + off[None, :]]
psth_dir = np.full((NB, len(off), C.shape[1]), np.nan)
for b in np.where(keep)[0]:
    psth_dir[b] = psth[dbin == b].mean(0)
tax = off * BIN

np.savez("_res_direction.npz", theta=theta, onset=onset, peak_speed=peak_speed, alive=alive,
         R_move=R_move, R_delay=R_delay, best_offset=BEST, offsets=offsets, scan=scan,
         tc=tc, tc_sem=tc_sem, centers=centers, keep=keep, n_per=n_per, dbin=dbin,
         psth_dir=psth_dir, tax=tax, r2_binned=r2_binned, sig=sig,
         **{f"move_{k}": v for k, v in fit_move.items()},
         **{f"delay_{k}": v for k, v in fit_delay.items()})

# ---------- figure 2: example units ----------
ex = np.argsort(np.where(sig, fit_move["depth"], -1))[::-1][:4]
cols = plt.cm.hsv((centers + np.pi) / (2 * np.pi))
fig = plt.figure(figsize=(16, 8.5))
gs = fig.add_gridspec(2, 4, hspace=0.62, wspace=0.42, height_ratios=[1, 1.1])
for j, ui in enumerate(ex):
    ax = fig.add_subplot(gs[0, j])
    for b in np.where(keep)[0]:
        ax.plot(tax, psth_dir[b, :, ui], color=cols[b], lw=1.4)
    ax.axvline(0, color="k", ls="--", lw=1)
    ax.axvspan(*MOVE_WIN, color="0.85", zorder=0)
    ax.set_title("unit %d" % units.index[ui], fontsize=11)
    ax.set_xlabel("time from movement onset (s)")
    if j == 0:
        ax.set_ylabel("firing rate (Hz)")
        ax.text(0.02, 0.98, "shading = rate window", transform=ax.transAxes, fontsize=7, va="top")
    ax = fig.add_subplot(gs[1, j], projection="polar")
    th_c = np.append(centers[kb], centers[kb][0] + 2 * np.pi)
    ax.errorbar(th_c, np.append(tc[kb, ui], tc[kb[0], ui]), yerr=np.append(tc_sem[kb, ui], tc_sem[kb[0], ui]),
                color="k", marker="o", ms=4, lw=1.4, capsize=2, label="observed")
    tt = np.linspace(-np.pi, np.pi, 200)
    ax.plot(tt, np.maximum(fit_move["b0"][ui] + fit_move["depth"][ui] * np.cos(tt - fit_move["pd"][ui]), 0),
            color="crimson", lw=2, label="cosine fit")
    ax.set_rlabel_position(285)
    ax.tick_params(labelsize=8)
    ax.set_title("PD %.0f$\\degree$  depth %.1f Hz  $R^2_{bin}$=%.2f"
                 % (np.degrees(fit_move["pd"][ui]), fit_move["depth"][ui], r2_binned[ui]), pad=20, fontsize=9)
    if j == 0:
        ax.legend(loc="lower left", bbox_to_anchor=(-0.35, -0.30), fontsize=8, frameon=False)
fig.suptitle("Reach-direction tuning in macaque motor cortex (DANDI:000128 MC_Maze, %d straight reaches)\n"
             "top: direction-conditioned PSTHs (line colour = reach direction);   bottom: mean rate in the %d-%d ms window with cosine fit"
             % (len(sel), MOVE_WIN[0] * 1000, MOVE_WIN[1] * 1000), y=0.99, fontsize=12)
fig.savefig("fig02_direction_tuning_examples.png", dpi=140, bbox_inches="tight")
print("saved fig02")
