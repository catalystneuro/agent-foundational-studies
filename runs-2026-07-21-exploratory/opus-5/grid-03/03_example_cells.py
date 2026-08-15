"""Single-session prototype: rate maps, autocorrelograms and gridness for every unit."""
import matplotlib.pyplot as plt
import numpy as np

import gridlib as gl

PATH = "sub-11265/sub-11265_ses-06020601_behavior+ecephys.nwb"

s = gl.load_session(PATH)
pos, units, ep = s["position"], s["units"], s["run_ep"]
edges = gl.map_edges(pos)
maps, occ, centers = gl.rate_maps(units, pos, ep, edges)
names = units.get_info("unit_name")

n = len(units)
fig, axes = plt.subplots(3, n, figsize=(1.9 * n, 6.4))
ext = [edges[0], edges[-1], edges[0], edges[-1]]
for i in range(n):
    sp = units[i].restrict(ep).value_from(pos)
    ax = axes[0, i]
    ax.plot(pos["x"].values, pos["y"].values, lw=0.2, color="0.8")
    ax.plot(sp["x"].values, sp["y"].values, ".", ms=1.8, color="crimson")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(names[i], fontsize=8)

    ax = axes[1, i]
    ax.imshow(maps[i], origin="lower", extent=ext, cmap="jet")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("%.1f Hz" % np.nanmax(maps[i]), fontsize=8)

    ac = gl.spatial_autocorr(maps[i])
    g, _ = gl.gridness(ac)
    spc, ori, ell, pk = gl.autocorr_peaks(ac)
    ax = axes[2, i]
    ax.imshow(ac, origin="lower", cmap="jet", vmin=-0.5, vmax=1)
    if len(pk):
        ax.plot(pk[:, 0], pk[:, 1], "wo", mfc="none", ms=4)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("g=%.2f\n%.0f cm" % (g, spc), fontsize=8)

for r, lab in enumerate(["traj + spikes", "rate map", "autocorr"]):
    axes[r, 0].set_ylabel(lab, fontsize=9)
plt.tight_layout()
plt.savefig("fig_check_session_all_units.png", dpi=130)
print("saved fig_check_session_all_units.png")
