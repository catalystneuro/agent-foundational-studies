"""Load one DANDI:000582 session and validate every data stream before analysis."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import gridlib as G

EXAMPLE = "sub-11265/sub-11265_ses-06020601_behavior+ecephys.nwb"

asset = [a for a in G.load_assets() if a["path"] == EXAMPLE][0]
s = G.load_session(asset, keep_lfp=True)
ep = G.run_epochs(s)
edges = G.map_edges(s["position"], ep)
occ = G.occupancy_map(s["position"], ep, edges, s["dt"])

print(f"session {s['path']}")
print(f"  subject {s['subject']}  duration {s['duration']:.1f} s  "
      f"tracking dt {s['dt'] * 1e3:.1f} ms")
print(f"  {len(s['units'])} units: {s['unit_names']}")
print(f"  histology: {set(s['histology'])}  depth(m): {set(s['depth'])}")
print(f"  arena extent: {edges[0][0]:.1f}..{edges[0][-1]:.1f} x "
      f"{edges[1][0]:.1f}..{edges[1][-1]:.1f} (cm)")
print(f"  running epochs: {len(ep)}, {ep.tot_length():.1f} s of {s['duration']:.1f} s")
print(f"  occupancy: {(occ >= G.MIN_OCC_S).mean() * 100:.1f}% of bins visited")

pos = s["position"]
fig = plt.figure(figsize=(14, 9))
gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.3)

ax = fig.add_subplot(gs[0, :])
ax.plot(pos.index.values, pos["x"].values, lw=0.6, label="x")
ax.plot(pos.index.values, pos["y"].values, lw=0.6, label="y")
ax.set(xlim=(0, 120), xlabel="time (s)", ylabel="position (cm)",
       title="Tracked position, first 120 s")
ax.legend(loc="upper right", fontsize=8)

ax = fig.add_subplot(gs[1, 0])
ax.plot(pos["x"].values, pos["y"].values, lw=0.3, color="0.4")
ax.set(aspect="equal", xlabel="x (cm)", ylabel="y (cm)", title="Full trajectory")

ax = fig.add_subplot(gs[1, 1])
im = ax.imshow(occ.T, origin="lower", cmap="viridis",
               extent=[edges[0][0], edges[0][-1], edges[1][0], edges[1][-1]])
ax.set(aspect="equal", xlabel="x (cm)", title="Occupancy (s per 3 cm bin)")
plt.colorbar(im, ax=ax, fraction=0.046)

ax = fig.add_subplot(gs[1, 2])
ax.hist(s["speed"].values, bins=80, range=(0, 100), color="0.4")
ax.axvline(G.SPEED_MIN_CMS, color="r", ls="--", label=f"{G.SPEED_MIN_CMS} cm/s cut")
ax.set(xlabel="running speed (cm/s)", ylabel="video frames", title="Speed distribution")
ax.legend(fontsize=8)

ax = fig.add_subplot(gs[2, 0])
seg = s["lfp"].restrict(G.nap.IntervalSet(100, 102))
ax.plot(seg.index.values, seg.values, lw=0.5)
ax.set(xlabel="time (s)", ylabel="LFP (a.u.)",
       title="MEC LFP, 2 s (theta rhythm visible)")

ax = fig.add_subplot(gs[2, 1:])
for i, k in enumerate(list(s["units"].keys())[:12]):
    t = s["units"][k].restrict(G.nap.IntervalSet(0, 60)).index.values
    ax.vlines(t, i, i + 0.8, lw=0.4, color="k")
ax.set(xlabel="time (s)", ylabel="unit", title="Spike raster, first 60 s",
       yticks=np.arange(min(12, len(s["units"]))) + 0.4,
       yticklabels=s["unit_names"][:12])
ax.tick_params(axis="y", labelsize=7)

fig.suptitle(f"Data validation: {s['path'].split('/')[-1]}  (DANDI:000582)", y=0.98)
fig.savefig("fig01_data_validation.png", dpi=130, bbox_inches="tight")
print("wrote fig01_data_validation.png")
