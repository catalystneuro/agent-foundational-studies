"""Figures 1-3: raw data, neural lead, exemplar directionally tuned unit."""
import numpy as np
import pynapple as nap
import matplotlib.pyplot as plt
import matplotlib as mpl
import mcmaze_io as mio

nap.nap_config.suppress_conversion_warnings = True
mpl.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

nwbfile, nwb = mio.open_nwb()
spikes = nwb["units"]
trials_ep = nwb["trials"]
hand_pos, hand_vel = mio.load_kinematics(nwbfile)
tdf = mio.trial_table(nwbfile)
td = np.load("trial_data.npz")
fits = np.load("tuning_fits.npz")
reach_dir, peak_speed = td["reach_dir"], td["peak_speed"]
unit_ids = fits["unit_ids"]

# ============================================================== Figure 1: raw data
fig = plt.figure(figsize=(12, 8.5))
gs = fig.add_gridspec(3, 2, width_ratios=[2, 1], hspace=0.45, wspace=0.25,
                      height_ratios=[1, 1, 1.4])
t0, t1 = 100.0, 120.0
ep = nap.IntervalSet(t0, t1)
hp, hv = hand_pos.restrict(ep), hand_vel.restrict(ep)
sp = np.hypot(hv["vx"].values, hv["vy"].values)

ax = fig.add_subplot(gs[0, 0])
ax.plot(hp.t, hp["x"].values, label="x", lw=1)
ax.plot(hp.t, hp["y"].values, label="y", lw=1)
ax.set_ylabel("hand position\n(mm)")
ax.legend(loc="upper right", frameon=False, ncol=2)
ax.set_xlim(t0, t1)
ax.set_title("A   Hand kinematics and M1/PMd spiking, 20 s of the session", loc="left",
             fontweight="bold")

ax2 = fig.add_subplot(gs[1, 0], sharex=ax)
ax2.plot(hv.t, sp, color="k", lw=1)
ax2.set_ylabel("hand speed\n(mm/s)")

ax3 = fig.add_subplot(gs[2, 0], sharex=ax)
show = list(spikes.keys())[:60]
for i, u in enumerate(show):
    s = spikes[u].restrict(ep)
    ax3.plot(s.t, np.full(len(s), i), "|", color="k", ms=2.5, mew=0.6)
ax3.set_ylabel("unit")
ax3.set_xlabel("time (s)")
ax3.set_ylim(-1, len(show))
for a in (ax, ax2, ax3):
    for tr in tdf.itertuples():
        if t0 < tr.move_onset_time < t1:
            a.axvline(tr.move_onset_time, color="crimson", lw=0.8, alpha=0.7)
ax3.text(0.99, 1.02, "red = movement onset", color="crimson", ha="right",
         transform=ax3.transAxes)

# hand paths coloured by reach direction
axp = fig.add_subplot(gs[:2, 1])
cmap = plt.get_cmap("hsv")
rng = np.random.default_rng(1)
for i in rng.choice(len(tdf), 150, replace=False):
    t_on = tdf.move_onset_time.values[i]
    seg = hand_pos.restrict(nap.IntervalSet(t_on, t_on + 0.6)).values
    axp.plot(seg[:, 0], seg[:, 1], lw=0.7, alpha=0.75,
             color=cmap(reach_dir[i] / (2 * np.pi)))
axp.set_aspect("equal")
axp.set_xlabel("x (mm)")
axp.set_ylabel("y (mm)")
axp.set_title("B   Reach paths (colour = direction)", loc="left", fontweight="bold")

axh = fig.add_subplot(gs[2, 1], projection="polar")
h, e = np.histogram(reach_dir, bins=24, range=(0, 2 * np.pi))
axh.bar((e[:-1] + e[1:]) / 2, h, width=2 * np.pi / 24,
        color=[cmap(x / (2 * np.pi)) for x in (e[:-1] + e[1:]) / 2])
axh.set_title("C   Reach directions (n = %d trials)" % len(tdf), loc="left",
              fontweight="bold", pad=26)
axh.set_yticklabels([])
fig.savefig("fig01_raw_data.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("fig01 done")

# ============================================================ Figure 2: neural lead
lag = np.load("lag_scan.npy")
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
axes[0].plot(lag[:, 0] * 1000, lag[:, 1], "o-", color="C0")
best = lag[np.argmax(lag[:, 1]), 0]
axes[0].axvline(best * 1000, color="crimson", ls="--")
axes[0].set_xlabel("lag: kinematics minus spikes (ms)")
axes[0].set_ylabel("population mean $R^2$")
axes[0].set_title(f"A   Neural activity leads the hand by {best*1000:.0f} ms",
                  loc="left", fontweight="bold")
axes[0].annotate(f"{best*1000:.0f} ms", (best * 1000, lag[:, 1].max()),
                 xytext=(8, -12), textcoords="offset points", color="crimson")

# speed / direction distributions
axes[1].hist(peak_speed, bins=40, color="0.4")
axes[1].set_xlabel("peak hand speed (mm/s)")
axes[1].set_ylabel("trials")
axes[1].set_title("B   Reach speed varies over a 2-fold range", loc="left",
                  fontweight="bold")
fig.tight_layout()
fig.savefig("fig02_neural_lead.png", dpi=150)
plt.close(fig)
print("fig02 done")

# ================================================= Figure 3: exemplar unit, clock plot
b1, pdir, r2 = fits["b1"], fits["pdir"], fits["r2"]
ex = int(np.argmax(r2))
uid = unit_ids[ex]
print("exemplar unit", uid, "b1", b1[ex], "R2", r2[ex])

NDIR = 8
edges8 = np.linspace(0, 2 * np.pi, NDIR + 1)
centers8 = (edges8[:-1] + edges8[1:]) / 2
dbin8 = np.digitize(reach_dir, edges8) - 1
onset = td["onset"]
st = spikes[uid].t
PRE, POST, PBIN = 0.4, 0.6, 0.02
tb = np.arange(-PRE, POST, PBIN)

fig = plt.figure(figsize=(11, 9.5))
gs = fig.add_gridspec(3, 3, hspace=0.55, wspace=0.35)
pos = {0: (1, 2), 1: (0, 2), 2: (0, 1), 3: (0, 0), 4: (1, 0), 5: (2, 0), 6: (2, 1), 7: (2, 2)}
ymax = 0
axlist = []
for k in range(NDIR):
    r_, c_ = pos[k]
    ax = fig.add_subplot(gs[r_, c_])
    trs = np.where(dbin8 == k)[0]
    rng2 = np.random.default_rng(0)
    if len(trs) > 40:
        trs_plot = rng2.choice(trs, 40, replace=False)
    else:
        trs_plot = trs
    allc = np.zeros(len(tb) - 1)
    for i, tr in enumerate(trs):
        rel = st[(st > onset[tr] - PRE) & (st < onset[tr] + POST)] - onset[tr]
        allc += np.histogram(rel, bins=tb)[0]
        if tr in trs_plot:
            j = list(trs_plot).index(tr)
            ax.plot(rel, np.full(len(rel), j), "|", color="0.3", ms=2.5, mew=0.5)
    psth = allc / (len(trs) * PBIN)
    ax2 = ax.twinx()
    ax2.plot(tb[:-1] + PBIN / 2, psth, color="C3", lw=1.5)
    ax2.set_ylim(0, None)
    ymax = max(ymax, psth.max())
    axlist.append(ax2)
    ax.set_ylim(-1, 41)
    ax.set_yticks([])
    ax.axvline(0, color="k", lw=0.8, ls=":")
    ax.set_title(f"{np.degrees(centers8[k]):.0f}°  (n={len(trs)})", fontsize=9)
    if r_ == 2:
        ax.set_xlabel("time from movement onset (s)")
for a in axlist:
    a.set_ylim(0, ymax * 1.05)
    a.set_ylabel("rate (Hz)", color="C3", fontsize=8)
    a.tick_params(axis="y", colors="C3", labelsize=7)

axc = fig.add_subplot(gs[1, 1], projection="polar")
tc = fits["tc"]
th = np.r_[fits["centers"], fits["centers"][0]]
axc.plot(th, np.r_[tc[:, ex], tc[0, ex]], "o-", color="C0", ms=4)
fine = np.linspace(0, 2 * np.pi, 200)
axc.plot(fine, np.maximum(fits["b0"][ex] + b1[ex] * np.cos(fine - pdir[ex]), 0),
         color="crimson", lw=1.5)
axc.set_rlabel_position(135)
rmax = np.ceil(tc[:, ex].max())
axc.set_rticks([rmax / 2, rmax])
axc.tick_params(labelsize=7)
axc.set_title(f"unit {uid}\ncosine fit $R^2$={r2[ex]:.2f}", fontsize=9, pad=26)
fig.suptitle(f"Directional tuning of a single M1/PMd unit (rasters + PSTHs by reach direction)",
             fontweight="bold")
fig.savefig("fig03_exemplar_unit.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("fig03 done")
