"""Load one session, sanity-check every data stream, and plot raw data."""
import matplotlib.pyplot as plt
import numpy as np

import gridlib as gl

PATH = "sub-11207/sub-11207_ses-03060501_behavior+ecephys.nwb"

s = gl.load_session(PATH)
pos, units, hd, speed = s["position"], s["units"], s["hd"], s["speed"]
print(s["meta"])
print(units)
print("position", pos.shape, "t", pos.index[0], pos.index[-1])
print("x range %.1f..%.1f  y %.1f..%.1f (cm)" % (
    pos["x"].values.min(), pos["x"].values.max(),
    pos["y"].values.min(), pos["y"].values.max()))
print("speed median %.1f cm/s, p99 %.1f" % (np.median(speed.values), np.percentile(speed.values, 99)))
print("run epochs: %d, total %.1f s of %.1f s" % (
    len(s["run_ep"]), s["run_ep"].tot_length(), s["meta"]["duration_s"]))

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
ax = axes[0, 0]
ax.plot(pos["x"].values, pos["y"].values, lw=0.3, color="0.5")
ax.set_aspect("equal"); ax.set_title("trajectory"); ax.set_xlabel("x (cm)"); ax.set_ylabel("y (cm)")

ax = axes[0, 1]
seg = pos.get(100, 160)
ax.plot(seg.index, seg["x"].values, label="x")
ax.plot(seg.index, seg["y"].values, label="y")
ax.legend(); ax.set_title("position, 100-160 s"); ax.set_xlabel("time (s)")

ax = axes[0, 2]
sp = speed.get(100, 160)
ax.plot(sp.index, sp.values, color="k", lw=0.8)
ax.axhline(gl.SPEED_MIN, color="r", ls="--")
ax.set_title("speed"); ax.set_ylabel("cm/s"); ax.set_xlabel("time (s)")

ax = axes[1, 0]
for i, u in enumerate(units.keys()):
    t = units[u].get(100, 160).times()
    ax.plot(t, np.full_like(t, i), "|", ms=5, color="k")
ax.set_yticks(range(len(units))); ax.set_yticklabels(units.get_info("unit_name"))
ax.set_title("spike raster, 100-160 s"); ax.set_xlabel("time (s)")

ax = axes[1, 1]
ax.hist(speed.values, bins=100, range=(0, 60), color="0.4")
ax.axvline(gl.SPEED_MIN, color="r", ls="--"); ax.set_title("speed distribution"); ax.set_xlabel("cm/s")

ax = axes[1, 2]
ax.hist(np.degrees(hd.values), bins=60, color="0.4")
ax.set_title("head direction distribution"); ax.set_xlabel("deg")

plt.tight_layout()
plt.savefig("fig_check_raw_streams.png", dpi=130)
print("saved fig_check_raw_streams.png")

# --- first rate maps ---------------------------------------------------------
edges = gl.map_edges(pos)
maps, occ, centers = gl.rate_maps(units, pos, s["run_ep"], edges)
print("map shape", maps.shape, "edges", edges[0], edges[-1], len(edges) - 1)

fig, axes = plt.subplots(2, len(units), figsize=(3 * len(units), 6.5))
ext = [edges[0], edges[-1], edges[0], edges[-1]]
for i, u in enumerate(units.keys()):
    ax = axes[0, i]
    spk_pos = units[u].restrict(s["run_ep"]).value_from(pos)
    ax.plot(pos["x"].values, pos["y"].values, lw=0.2, color="0.8")
    ax.plot(spk_pos["x"].values, spk_pos["y"].values, ".", ms=2.5, color="crimson")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(units.get_info("unit_name")[i], fontsize=10)
    ax = axes[1, i]
    ax.imshow(maps[i], origin="lower", extent=ext, cmap="jet")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("%.1f Hz peak" % np.nanmax(maps[i]), fontsize=9)
plt.tight_layout()
plt.savefig("fig_check_ratemaps.png", dpi=130)
print("saved fig_check_ratemaps.png")

for i, u in enumerate(units.keys()):
    ac = gl.spatial_autocorr(maps[i])
    g, r_out = gl.gridness(ac)
    sp_, ori, ell, pk = gl.autocorr_peaks(ac)
    print("%s gridness=%.2f spacing=%.1f cm orient=%.0f deg SI=%.2f bits/spk peaks=%d" % (
        units.get_info("unit_name")[i], g, sp_, ori,
        gl.spatial_information(maps[i], occ), len(pk)))
