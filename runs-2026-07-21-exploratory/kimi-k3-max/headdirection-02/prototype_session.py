"""Prototype HD-cell analysis on one session: Mouse17-130128 (DANDI 000056)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap
from tqdm import tqdm

from hd_utils import (
    SESSIONS, load_session, compute_head_direction, get_wake_epochs,
    tuning_curves_hd, classify_hd_cells,
)

rng = np.random.default_rng(42)
name = "Mouse17-130128"
nwb, io = load_session(SESSIONS[name])
print(nwb)

units = nwb["units"]
hd = compute_head_direction(nwb)
wake = get_wake_epochs(nwb, "Awake")
print(f"\n{len(units)} units; wake duration: {wake.tot_length():.0f} s in {len(wake)} epochs")
print(f"HD samples: {len(hd)}, NaN fraction: {np.mean(np.isnan(hd.values)):.3f}")

# ---------------------------------------------------------------- raw data fig
red = nwb["SubjectPosition/RedLED"]
blue = nwb["SubjectPosition/BlueLED"]
red_t, red_v = np.asarray(red.t), np.asarray(red.values, dtype=float)
blue_v = np.asarray(blue.values, dtype=float)

fig, axes = plt.subplots(3, 1, figsize=(11, 9), constrained_layout=True)
t0, t1 = 100.0, 160.0  # one minute of the first wake epoch
sl = (red_t >= t0) & (red_t <= t1)
axes[0].plot(red_t[sl], red_v[sl, 0], "r-", lw=0.6, label="Red LED x")
axes[0].plot(red_t[sl], blue_v[sl, 0], "b-", lw=0.6, label="Blue LED x")
axes[0].set_ylabel("x position (px)")
axes[0].legend(loc="upper right", fontsize=8)
axes[0].set_title("Dual-LED head tracking (60 s snippet)")

sl2 = (hd.t >= t0) & (hd.t <= t1)
axes[1].plot(hd.t[sl2], hd.values[sl2], "k.", ms=1.5)
axes[1].set_ylabel("Head direction (rad)")
axes[1].set_yticks([0, np.pi, 2 * np.pi], ["0", "$\\pi$", "$2\\pi$"])
axes[1].set_title("Head direction from LED difference vector")

hd_wake = hd.restrict(wake)
occ, edges = np.histogram(hd_wake.values[~np.isnan(hd_wake.values)], bins=60, range=(0, 2 * np.pi))
centers = 0.5 * (edges[:-1] + edges[1:])
axp = fig.add_subplot(3, 1, 3, projection="polar")
axes[2].remove()
axp.bar(centers, occ / 60.0, width=2 * np.pi / 60, color="gray")
axp.set_title("Wake HD occupancy (s per 6° bin)", pad=18)
fig.savefig("figures/01_raw_tracking.png", dpi=150)
plt.close(fig)
print("saved figures/01_raw_tracking.png")

# ------------------------------------------------------- tuning + classification
rates, centers, occupancy = tuning_curves_hd(units, hd, wake, bins=60)
print("tuning rates shape:", rates.shape)

result = classify_hd_cells(units, hd, wake, n_shuffles=1000, rng=rng)
keys = result["keys"]
is_hd = result["is_hd"]
hd_keys = [k for k, h in zip(keys, is_hd) if h]
print(f"\nHD cells: {is_hd.sum()}/{len(keys)}")
print("HD unit ids:", hd_keys)

# order HD cells by preferred angle
order = np.argsort(result["pref_angle"][is_hd])
hd_keys_sorted = [hd_keys[i] for i in order]

# ------------------------------------------------------- polar tuning grid
n_hd = len(hd_keys_sorted)
ncol = 4
nrow = int(np.ceil(n_hd / ncol))
fig = plt.figure(figsize=(3.0 * ncol, 3.0 * nrow))
for i, k in enumerate(hd_keys_sorted):
    ax = fig.add_subplot(nrow, ncol, i + 1, projection="polar")
    r = rates.sel(unit=k).values
    ax.plot(np.append(centers, centers[0]), np.append(r, r[0]), "k-", lw=1.5)
    ax.fill(np.append(centers, centers[0]), np.append(r, r[0]), "r", alpha=0.3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"unit {k}  MVL={result['mvl'][keys.index(k)]:.2f}", fontsize=9)
fig.suptitle(f"{name}: HD-cell tuning curves (n={n_hd}, sorted by preferred direction)")
fig.savefig("figures/02_tuning_curves.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved figures/02_tuning_curves.png")

# ------------------------------------------------------- MVL vs null scatter
fig, ax = plt.subplots(figsize=(6, 6))
mvl = result["mvl"]
null_med = result["null_median"]
ok = ~np.isnan(mvl)
ax.scatter(null_med[ok & ~is_hd], mvl[ok & ~is_hd], s=25, c="gray", alpha=0.7, label="not HD")
ax.scatter(null_med[ok & is_hd], mvl[ok & is_hd], s=35, c="crimson", label="HD cell")
lim = [0, max(1.0, np.nanmax(mvl) * 1.05)]
ax.plot(lim, lim, "k--", lw=0.8)
ax.axhline(0.3, color="crimson", ls=":", lw=1, label="MVL floor = 0.3")
ax.set_xlabel("Null MVL (median, random-time resampling)")
ax.set_ylabel("Observed MVL")
ax.set_title(f"{name}: directional selectivity vs occupancy-matched null")
ax.legend(fontsize=9)
fig.savefig("figures/03_mvl_vs_null.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved figures/03_mvl_vs_null.png")

# ------------------------------------------------------- population raster
# 60 s window: HD trace + spike raster of HD cells sorted by preferred angle
t0, t1 = 17897.0, 17957.0  # final long awake epoch
ep = nap.IntervalSet(start=t0, end=t1)
fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                         gridspec_kw={"height_ratios": [1, 2]}, constrained_layout=True)
hd_ep = hd.restrict(ep)
axes[0].plot(hd_ep.t, hd_ep.values, "k-", lw=0.8)
axes[0].set_ylabel("HD (rad)")
axes[0].set_yticks([0, np.pi, 2 * np.pi], ["0", "$\\pi$", "$2\\pi$"])
axes[0].set_title(f"{name}: HD cells sorted by preferred direction (60 s of wake)")
for i, k in enumerate(hd_keys_sorted):
    spk = units[k].restrict(ep)
    axes[1].plot(spk.t, np.full(len(spk), i), "|", color="crimson", ms=6)
axes[1].set_ylabel("HD cell (sorted)")
axes[1].set_xlabel("Time (s)")
axes[1].set_yticks(range(n_hd), [str(k) for k in hd_keys_sorted], fontsize=7)
fig.savefig("figures/04_raster.png", dpi=150)
plt.close(fig)
print("saved figures/04_raster.png")

np.savez("cache/prototype_results.npz",
         keys=np.array(keys), mvl=result["mvl"], p_value=result["p_value"],
         pref_angle=result["pref_angle"], null_median=result["null_median"],
         is_hd=is_hd)
io.close()
print("done")
