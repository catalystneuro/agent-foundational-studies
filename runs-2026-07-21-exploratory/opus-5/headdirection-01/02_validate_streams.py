"""Validate each data stream before analysis: behaviour, spikes, epoch structure."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap

import hd_lib

ASSET = "c72bd94f-b744-483e-a782-3a3c475a5276"  # sub-A3705 ses-200306

s = hd_lib.session_bundle(ASSET)
units, hd, pos, wake = s["units"], s["hd"], s["position"], s["wake"]

print("session:", s["name"], "subject:", s["subject"])
print("epochs:", list(zip(s["epoch_labels"], s["epochs"].start, s["epochs"].end)))
print("wake total (s):", round(wake.tot_length(), 1))
print("HD samples:", len(hd), "dropped NaN:", s["n_hd_nan"],
      "dt:", np.median(np.diff(hd.t)))
print("position range x:", pos["x"].values.min(), pos["x"].values.max(),
      " y:", pos["y"].values.min(), pos["y"].values.max())
print("units:", len(units))
print(units.metadata_columns if hasattr(units, "metadata_columns") else units)
print(units)

rates = units.restrict(wake).rates
print("wake firing rates: median %.2f Hz, range %.2f-%.2f" %
      (np.median(rates), rates.min(), rates.max()))
print("dataset HD labels:", int(np.sum(np.asarray(units.is_head_direction))), "of", len(units))

# --------------------------------------------------------------------------- #
fig = plt.figure(figsize=(14, 9))
gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.4], hspace=0.55, wspace=0.3)

# Session timeline
ax = fig.add_subplot(gs[0, :])
colors = {"home_cage": "0.85", "wake_square": "#4C72B0", "wake_triangle": "#DD8452"}
for lbl, st, en in zip(s["epoch_labels"], s["epochs"].start, s["epochs"].end):
    ax.axvspan(st / 60, en / 60, color=colors.get(lbl, "0.6"), alpha=0.6)
    ax.text((st + en) / 2 / 60, 1.03, lbl, ha="center", fontsize=9,
            transform=ax.get_xaxis_transform())
ax.plot(hd.t / 60, hd.d, ",", color="k", alpha=0.3)
ax.set_xlabel("time (min)")
ax.set_ylabel("head direction (rad)")
ax.set_title(f"{s['name']}: session structure and head-direction tracking", pad=18)
ax.set_ylim(0, 2 * np.pi)

# Zoomed HD trace
ax = fig.add_subplot(gs[1, 0])
t0 = wake.start[0] + 100
seg = nap.IntervalSet(start=t0, end=t0 + 60)
h = hd.restrict(seg)
ax.plot(h.t - t0, np.degrees(h.d), "k-", lw=0.8)
ax.set_xlabel("time (s)")
ax.set_ylabel("head direction (deg)")
ax.set_title("60 s of tracked head direction")

# Trajectory in the square arena
ax = fig.add_subplot(gs[1, 1])
sq = nap.IntervalSet(start=s["wake"].start[0], end=s["wake"].end[0])
p = pos.restrict(sq)
ax.plot(p["x"].values, p["y"].values, lw=0.3, color="#4C72B0")
ax.set_aspect("equal")
ax.set_xlabel("x (cm)")
ax.set_ylabel("y (cm)")
ax.set_title("trajectory, square arena")

# Occupancy over direction
ax = fig.add_subplot(gs[1, 2], projection="polar")
occ = hd_lib.angular_occupancy(hd, wake, 60)
edges = np.linspace(0, 2 * np.pi, 61)
ax.bar(edges[:-1], occ, width=np.diff(edges), align="edge", color="0.5")
ax.set_title("directional occupancy\n(wake)", pad=26)
ax.set_yticklabels([])

# Raster of a subset of units, aligned to HD
ax = fig.add_subplot(gs[2, :])
seg = nap.IntervalSet(start=t0, end=t0 + 60)
hd_lab = np.asarray(units.is_head_direction).astype(bool)
tc_probe, stats_probe = hd_lib.tuning_stats(units, hd, wake, nb_bins=120)
order = stats_probe.sort_values("pref_dir").index
tuned = [u for u in order if hd_lab[list(units.keys()).index(u)]]
sel = [tuned[i] for i in np.linspace(0, len(tuned) - 1, 25).astype(int)]
for i, u in enumerate(sel):
    st = units[u].restrict(seg).t
    ax.plot(st - t0, np.full_like(st, i), "|", color=plt.cm.hsv(
        stats_probe.loc[u, "pref_dir"] / (2 * np.pi)), ms=4)
sm = plt.cm.ScalarMappable(cmap="hsv", norm=plt.Normalize(0, 360))
cb = fig.colorbar(sm, ax=ax, pad=0.06, fraction=0.03)
cb.set_label("preferred direction (deg)")
ax.set_ylabel("unit (sorted by preferred direction)")
ax.set_xlabel("time (s)")
ax2 = ax.twinx()
ax.set_zorder(2)
ax.patch.set_visible(False)
ax2.set_zorder(1)
h = hd.restrict(seg)
ax2.plot(h.t - t0, np.degrees(h.d), "k-", lw=1.2, alpha=0.7)
ax2.set_ylabel("head direction (deg)")
ax.set_title("Spike raster of 25 tuned units, colour-coded by preferred direction, "
             "with the animal's head direction overlaid")

fig.savefig("fig01_data_validation.png", dpi=150, bbox_inches="tight")
print("saved fig01_data_validation.png")
