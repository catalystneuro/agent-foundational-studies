"""Prototype: stream Mouse28-140310, inspect structure, validate raw data streams."""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pynapple as nap
from common import SESSIONS, load_session, get_hd_angle, get_states, sorted_units

name = "Mouse28-140310"
nwb = load_session(SESSIONS[name])
print(nwb)

units = sorted_units(nwb)
print(f"\n{len(units)} units")
rates = np.array([len(units[k].t) / units[k].time_support.tot_length("s") for k in units.keys()])
print(f"mean rate: {rates.mean():.2f} Hz, range {rates.min():.2f}-{rates.max():.2f} Hz")

hd = get_hd_angle(nwb)
print(f"\nHD: {len(hd)} samples, {np.isnan(hd.values).mean()*100:.1f}% tracking failures, "
      f"rate ~{1/np.median(np.diff(hd.t)):.1f} Hz")

states = get_states(nwb)
for k, v in states.items():
    print(f"state {k}: {len(v)} epochs, total {v.tot_length('s'):.0f} s")

# ---- validation figure: raw streams ----
fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=False,
                         gridspec_kw=dict(height_ratios=[1, 1, 1, 1.4], hspace=0.55))

# 1) LED positions over a 60 s wake window
wake = states["Awake"]
ep = nap.IntervalSet(start=wake.start[0], end=wake.start[0] + 60)
red = nwb["SubjectPosition/RedLED"]
blue = nwb["SubjectPosition/BlueLED"]
red_w = red.restrict(ep)
blue_w = blue.restrict(ep)
axes[0].plot(red_w.t, np.asarray(red_w.values)[:, 0], ".", ms=1, label="RedLED x")
axes[0].plot(blue_w.t, np.asarray(blue_w.values)[:, 0], ".", ms=1, label="BlueLED x")
axes[0].set_ylabel("x position (a.u.)")
axes[0].legend(loc="upper right", fontsize=8)
axes[0].set_title("LED tracking (first 60 s of wake)")

# 2) HD angle over the same window
hd_w = hd.restrict(ep)
axes[1].plot(hd_w.t, hd_w.values, ".", ms=1, color="k")
axes[1].set_ylabel("HD (rad)")
axes[1].set_ylim(-0.2, 2 * np.pi + 0.2)
axes[1].set_title("Head-direction angle from LED difference")

# 3) HD occupancy over the whole wake period
hd_wake = hd.restrict(wake)
valid = ~np.isnan(hd_wake.values)
occ, edges = np.histogram(hd_wake.values[valid], bins=60, range=(0, 2 * np.pi))
centers = 0.5 * (edges[:-1] + edges[1:])
axes[2].plot(centers, occ / occ.sum(), color="k")
axes[2].set_ylabel("occupancy")
axes[2].set_xlabel("HD (rad)")
axes[2].set_title("Wake HD occupancy")

# 4) raster of all units over 30 s + state timeline
ep2 = nap.IntervalSet(start=wake.start[0], end=wake.start[0] + 30)
spk = units.restrict(ep2)
for i, k in enumerate(units.keys()):
    axes[3].plot(spk[k].t, np.full(len(spk[k].t), i), "|", ms=2, color="k")
axes[3].set_ylabel("unit #")
axes[3].set_xlabel("time (s)")
axes[3].set_title("Spike raster (30 s of wake)")

fig.suptitle(f"{name}: raw data validation", fontsize=13)
fig.savefig("figures/01_raw_data_validation.png", dpi=150, bbox_inches="tight")
print("\nsaved figures/01_raw_data_validation.png")
