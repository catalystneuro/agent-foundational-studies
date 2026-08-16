"""Direction-specific place fields on the linear track (the replay template)."""

import numpy as np
import matplotlib.pyplot as plt
import pynapple as nap

from dandi_io import open_session
from swr_lib import linearize_position, running_speed

SESSION = "Achilles_10252013"
N_BINS = 50
TRACK = 1.6

nwbfile, nwb, h5 = open_session(SESSION)
units = nwb["units"]
pyr = units[units.cell_type == "excitatory"]
pos, TRACK = linearize_position(nwb)
speed = running_speed(pos)

# ------------------------------------------------------- running laps by direction
run_ep = (speed.threshold(0.05).time_support
          .merge_close_intervals(0.5).drop_short_intervals(0.5))
disp = np.array([pos.restrict(run_ep[i]).values[-1] - pos.restrict(run_ep[i]).values[0]
                 for i in range(len(run_ep))])
lap_len = np.abs(disp)
keep = lap_len > 0.5 * TRACK           # complete traversals only
print(f"{len(run_ep)} running bouts, {keep.sum()} full traversals "
      f"({(disp[keep] > 0).sum()} rightward, {(disp[keep] < 0).sum()} leftward)")

dir_ep = {
    "rightward": run_ep[np.where(keep & (disp > 0))[0]],
    "leftward": run_ep[np.where(keep & (disp < 0))[0]],
}

bins = np.linspace(0, TRACK, N_BINS + 1)
centers = (bins[:-1] + bins[1:]) / 2

tc, occ = {}, {}
for d, ep in dir_ep.items():
    tc[d] = nap.compute_1d_tuning_curves(pyr, pos, nb_bins=N_BINS,
                                         minmax=(0, TRACK), ep=ep)
    occ[d] = np.histogram(pos.restrict(ep).values, bins=bins)[0] * \
        float(np.median(np.diff(pos.times())))
    print(f"{d}: {ep.tot_length():.0f} s of running, "
          f"occupancy min {occ[d].min():.1f} s / bin")


def spatial_information(rate, occupancy):
    """Skaggs information in bits/spike."""
    p = occupancy / occupancy.sum()
    mean_rate = np.sum(p * rate)
    if mean_rate <= 0:
        return 0.0
    nz = rate > 0
    return float(np.sum(p[nz] * rate[nz] / mean_rate * np.log2(rate[nz] / mean_rate)))


rows = []
for d in dir_ep:
    for u in pyr.index:
        r = tc[d][u].values
        rows.append(dict(unit=u, direction=d, peak=np.nanmax(r),
                         si=spatial_information(np.nan_to_num(r), occ[d]),
                         com=centers[np.nanargmax(r)]))
import pandas as pd
stats = pd.DataFrame(rows)

# split-half stability
half = {}
for d, ep in dir_ep.items():
    n = len(ep)
    a = nap.compute_1d_tuning_curves(pyr, pos, N_BINS, minmax=(0, TRACK), ep=ep[: n // 2])
    b = nap.compute_1d_tuning_curves(pyr, pos, N_BINS, minmax=(0, TRACK), ep=ep[n // 2:])
    half[d] = {u: np.corrcoef(np.nan_to_num(a[u].values), np.nan_to_num(b[u].values))[0, 1]
               for u in pyr.index}
stats["stability"] = [half[r.direction][r.unit] for r in stats.itertuples()]

stats["place_cell"] = (stats.peak >= 1.0) & (stats.si >= 0.4) & (stats.stability >= 0.3)
print(stats.groupby("direction").place_cell.sum())
place_units = sorted(stats.query("place_cell").unit.unique())
print(f"{len(place_units)} / {len(pyr)} pyramidal cells have a place field "
      f"in at least one direction")

np.savez("place_fields_Achilles_10252013.npz",
         centers=centers,
         tc_right=np.stack([tc["rightward"][u].values for u in pyr.index], 1),
         tc_left=np.stack([tc["leftward"][u].values for u in pyr.index], 1),
         units=np.array(pyr.index),
         place_units=np.array(place_units),
         run_start=run_ep.start[keep], run_end=run_ep.end[keep], disp=disp[keep])
stats.to_csv("place_field_stats_Achilles_10252013.csv", index=False)

# --------------------------------------------------------------------- figures
fig = plt.figure(figsize=(13, 10))
gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35, height_ratios=[1.2, 1, 1])

for j, d in enumerate(dir_ep):
    ax = fig.add_subplot(gs[0, j])
    M = np.stack([np.nan_to_num(tc[d][u].values) for u in place_units])
    M = M / np.maximum(M.max(axis=1, keepdims=True), 1e-9)
    order = np.argsort(np.argmax(M, axis=1))
    im = ax.imshow(M[order], aspect="auto", origin="lower", cmap="viridis",
                   extent=[0, TRACK, 0, len(place_units)], vmin=0, vmax=1)
    ax.set_xlabel("position (m)")
    ax.set_ylabel("place cell (sorted)" if j == 0 else "")
    ax.set_title(f"{d} runs", fontsize=10)
    if j == 1:
        plt.colorbar(im, ax=ax, label="norm. rate")

ax = fig.add_subplot(gs[0, 2])
ax.plot(centers, occ["rightward"], label="rightward", color="#0077b6")
ax.plot(centers, occ["leftward"], label="leftward", color="#e63946")
ax.set_xlabel("position (m)")
ax.set_ylabel("occupancy (s)")
ax.set_title("running occupancy", fontsize=10)
ax.legend(fontsize=8)

# three example place cells with fields spread along the track
pc = stats.query("place_cell and peak > 3").sort_values("com")
ex = [pc.iloc[int(q * (len(pc) - 1))] for q in (0.25, 0.5, 0.75)]
for k, r in enumerate(ex):
    ax = fig.add_subplot(gs[1, k])
    for d, c in [("rightward", "#0077b6"), ("leftward", "#e63946")]:
        ax.plot(centers, tc[d][r.unit].values, color=c, lw=1.4, label=d)
    ax.set_title(f"unit {int(r.unit)}  (SI {r.si:.2f} bits/spike)", fontsize=9)
    ax.set_ylabel("rate (Hz)")
    ax.set_xlabel("position (m)")
    if k == 0:
        ax.legend(fontsize=7)

# spike raster over a few laps
ax = fig.add_subplot(gs[2, 0])
lap_ids = np.where(keep)[0]
lap0 = lap_ids[np.argmax(disp[keep] > 0)]      # first rightward traversal
t0, t1 = run_ep.start[lap0] - 1, run_ep.end[lap0] + 1
w = nap.IntervalSet(start=t0, end=t1)
M = np.stack([np.nan_to_num(tc["rightward"][u].values) for u in place_units])
order_units = np.array(place_units)[np.argsort(np.argmax(M, axis=1))]
for i, u in enumerate(order_units):
    sp_t = pyr[u].restrict(w).times()
    ax.plot(sp_t - t0, np.full(sp_t.size, i), "|", ms=4, color="k")
axp = ax.twinx()
axp.plot(pos.restrict(w).times() - t0, pos.restrict(w).values, color="#fb8500", lw=1)
axp.set_ylabel("position (m)", color="#fb8500")
ax.set_xlabel("time (s)")
ax.set_ylabel("cell (by field position)")
ax.set_title("sequential activation during one traversal", fontsize=9)

ax = fig.add_subplot(gs[2, 1])
ax.hist(stats.si, bins=30, color="#adb5bd", label="all")
ax.hist(stats.query("place_cell").si, bins=30, color="#023047", label="place cells")
ax.set_xlabel("spatial information (bits/spike)")
ax.set_ylabel("count")
ax.legend(fontsize=8)
ax.set_title("spatial information", fontsize=9)

ax = fig.add_subplot(gs[2, 2])
ax.scatter(stats.peak, stats.stability, s=8,
           c=np.where(stats.place_cell, "#023047", "#adb5bd"))
ax.axhline(0.3, color="0.5", ls="--", lw=0.8)
ax.axvline(1.0, color="0.5", ls="--", lw=0.8)
ax.set_xscale("log")
ax.set_xlabel("peak rate (Hz)")
ax.set_ylabel("split-half stability (r)")
ax.set_title("place-cell selection", fontsize=9)

fig.suptitle(f"{SESSION}: direction-specific place fields "
             f"({len(place_units)} place cells)", y=0.94)
fig.savefig("fig03_place_fields.png", dpi=150, bbox_inches="tight")
print("wrote fig03_place_fields.png")
