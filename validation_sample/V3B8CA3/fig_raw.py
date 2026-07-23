"""Figures 1-2: behaviour overview and raw neural activity."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pynapple as nap

import pf_core as pf
from dandi_io import open_session

mpl.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 130, "savefig.dpi": 160})

SESSION = "Achilles_10252013"

spikes, position, pos2d, maze_ep, meta, (nwbfile, io) = pf.load_behavior_and_spikes(SESSION)
run_ep, directions = pf.find_run_epochs(position, pos_fs=meta["pos_fs_hz"])
speed = pf.compute_speed(position, run_ep)
trial_eps, dir_eps = pf.direction_epochs(run_ep, directions, speed)
all_run_ep = dir_eps["rightward"].union(dir_eps["leftward"])
pyr, pyr_keys = pf.select_pyramidal(spikes, all_run_ep)

print(f"{SESSION}: {meta['n_units_total']} units, {len(pyr)} putative pyramidal, "
      f"{len(run_ep)} traversals ({(directions==1).sum()} R / {(directions==-1).sum()} L)")

# ---------------------------------------------------------------- Figure 1 ---
fig = plt.figure(figsize=(11, 6.5))
gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.32,
                      left=0.07, right=0.98, top=0.90, bottom=0.09)

ax = fig.add_subplot(gs[0, 0])
p2 = pos2d.restrict(maze_ep)
ax.plot(p2["x"].values, p2["y"].values, lw=0.3, color="0.75", label="all tracking")
pr = pos2d.restrict(run_ep)
ax.plot(pr["x"].values, pr["y"].values, ".", ms=0.7, color="#1f77b4", label="scored traversals")
ax.set_xlabel("x (cm)"); ax.set_ylabel("y (cm)")
ax.legend(frameon=False, fontsize=7.5, loc="upper left", markerscale=6)
ax.set_title("A  2-D tracking, maze epoch", loc="left", fontweight="bold")
ax.set_aspect("equal")

ax = fig.add_subplot(gs[0, 1:])
t0 = run_ep.start[10]
win = nap.IntervalSet(start=t0 - 5, end=t0 + 115)
pw = position.restrict(win)
for lab, col in [("rightward", "#1f77b4"), ("leftward", "#d62728")]:
    ep = dir_eps[lab].intersect(win)
    for s, e in zip(ep.start, ep.end):
        ax.axvspan(s - win.start[0], e - win.start[0], color=col, alpha=0.16, lw=0)
ax.plot(pw.t - win.start[0], pw.values, ".", ms=1.6, color="k")
ax.set_xlabel("time from window start (s)"); ax.set_ylabel("linearized position (cm)")
ax.set_title("B  Track traversals (blue = rightward, red = leftward)",
             loc="left", fontweight="bold")
ax.set_xlim(0, 120)

ax = fig.add_subplot(gs[1, 0])
ax.hist(speed.values, bins=60, color="0.4")
ax.axvline(pf.SPEED_THRESHOLD_CMS, color="crimson", ls="--", lw=1.2)
ax.set_xlabel("running speed (cm/s)"); ax.set_ylabel("position samples")
ax.set_title("C  Speed during traversals", loc="left", fontweight="bold")

ax = fig.add_subplot(gs[1, 1])
eng_r = pf.RateMapEngine(pyr, position, dir_eps["rightward"], pos_fs=meta["pos_fs_hz"])
eng_l = pf.RateMapEngine(pyr, position, dir_eps["leftward"], pos_fs=meta["pos_fs_hz"])
w = eng_r.bin_centers[1] - eng_r.bin_centers[0]
ax.bar(eng_r.bin_centers - w / 4, eng_r.occupancy, width=w / 2, color="#1f77b4", label="rightward")
ax.bar(eng_l.bin_centers + w / 4, eng_l.occupancy, width=w / 2, color="#d62728", label="leftward")
ax.set_xlabel("position (cm)"); ax.set_ylabel("occupancy (s)")
ax.legend(frameon=False, fontsize=8)
ax.set_title("D  Spatial occupancy", loc="left", fontweight="bold")

ax = fig.add_subplot(gs[1, 2])
rates_on_track = np.array([len(spikes.restrict(all_run_ep)[k]) / all_run_ep.tot_length()
                           for k in spikes.keys()])
ctype = np.asarray(spikes.metadata["cell_type"])
bins = np.logspace(-2, 1.8, 30)
ax.hist(rates_on_track[ctype == "excitatory"], bins=bins, color="#4c72b0", alpha=0.85, label="excitatory")
ax.hist(rates_on_track[ctype == "inhibitory"], bins=bins, color="#dd8452", alpha=0.85, label="inhibitory")
ax.set_xscale("log")
ax.set_xlabel("firing rate on track (Hz)"); ax.set_ylabel("units")
ax.legend(frameon=False, fontsize=8)
ax.set_title("E  Unit firing rates", loc="left", fontweight="bold")

fig.suptitle(f"DANDI:000044  {SESSION}  (rat {meta['subject']}, 1.6 m linear track)",
             fontweight="bold")
fig.savefig("fig01_behavior_overview.png", bbox_inches="tight")
plt.close(fig)
print("wrote fig01_behavior_overview.png")

# ---------------------------------------------------------------- Figure 2 ---
# Raw activity: LFP + spike raster + position over one rightward and one
# leftward traversal, with units ordered by the location of their place field.
lfp_es = nwbfile.processing["ecephys"]["LFP"].electrical_series["LFP"]
ca1_ch, td_ratio = pf.pick_theta_channel(nwbfile, run_ep)
print(f"theta channel = {ca1_ch} (theta/delta = {td_ratio[ca1_ch]:.2f})")

t_start, t_stop = run_ep.start[10] - 2.0, run_ep.end[11] + 2.0
i0, i1 = int(t_start * lfp_es.rate), int(t_stop * lfp_es.rate)
lfp_raw = np.asarray(lfp_es.data[i0:i1, ca1_ch]) * lfp_es.conversion * 1e3  # mV
lfp = nap.Tsd(t=np.arange(i0, i1) / lfp_es.rate, d=lfp_raw)
theta = nap.apply_bandpass_filter(lfp, (6.0, 10.0), fs=lfp_es.rate)

win = nap.IntervalSet(start=t_start, end=t_stop)
maps_r = eng_r.rate_maps()
order = np.argsort(eng_r.bin_centers[np.nanargmax(maps_r, axis=1)])
ordered_keys = np.array(list(pyr.keys()))[order]

fig, axes = plt.subplots(3, 1, figsize=(10, 7.6), sharex=True,
                         gridspec_kw={"height_ratios": [1.0, 3.4, 1.0], "hspace": 0.30})

axes[0].plot(lfp.t - t_start, lfp.values, lw=0.5, color="0.6", label="raw LFP")
axes[0].plot(theta.t - t_start, theta.values, lw=1.1, color="#c44e52", label="6-10 Hz theta")
axes[0].set_ylabel("CA1 LFP (mV)")
axes[0].set_ylim(-np.percentile(np.abs(lfp.values), 99.8), np.percentile(np.abs(lfp.values), 99.8))
axes[0].legend(frameon=False, ncol=2, fontsize=8, loc="lower right")
axes[0].set_title("A  Hippocampal LFP (theta-dominated during running)",
                  loc="left", fontweight="bold", pad=6)

for row, k in enumerate(ordered_keys):
    st = pyr[k].restrict(win).t - t_start
    axes[1].plot(st, np.full_like(st, row), "|", ms=3.2, color="k", mew=0.7)
axes[1].set_ylabel("unit (ordered by place-field peak)")
axes[1].set_ylim(-1, len(ordered_keys))
axes[1].set_title("B  Spike raster of 105 putative pyramidal cells: the population "
                  "sweeps through the ordering once per traversal",
                  loc="left", fontweight="bold", pad=6)

pw = position.restrict(win)
axes[2].plot(pw.t - t_start, pw.values, ".", ms=2.5, color="k")
for lab, col in [("rightward", "#1f77b4"), ("leftward", "#d62728")]:
    ep = trial_eps[lab].intersect(win)
    for s, e in zip(ep.start, ep.end):
        for a in axes:
            a.axvspan(s - t_start, e - t_start, color=col, alpha=0.14, lw=0)
axes[2].set_ylabel("position (cm)"); axes[2].set_xlabel("time (s)")
axes[2].set_title("C  Linearized position (blue = rightward, red = leftward)",
                  loc="left", fontweight="bold", pad=6)
axes[2].set_xlim(0, t_stop - t_start)

fig.suptitle(f"Raw data, {SESSION}", fontweight="bold", y=0.955)
fig.savefig("fig02_raw_activity.png", bbox_inches="tight")
plt.close(fig)
print("wrote fig02_raw_activity.png")
