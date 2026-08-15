"""Prototype the grid-cell pipeline on a single DANDI:000582 session."""
import matplotlib.pyplot as plt
import numpy as np

import grid_lib as gl

scan = gl.list_assets()
cand = [s for s in scan if any(h == "MEC LII" for h in s["histology"])]
print("layer II sessions:", len(cand))
sess = sorted(cand, key=lambda s: -s["n_units"])[0]
print(sess["path"], sess["n_units"], sess["histology"])

d = gl.load_session(sess["asset_id"])
pos = d["position"]
half, center = gl.session_extent(pos)
edges = gl.map_edges(half)
occ = gl.occupancy_map(pos, d["dt"], edges, center)
print(f"box half-width {half:.1f} cm, map {occ.shape}, "
      f"visited bins {(occ >= gl.MIN_OCCUPANCY).sum()}/{occ.size}")
print("run epoch total (s):", round(float(d["run_ep"].tot_length()), 1),
      "of", round(float(d["position_all"].t[-1]), 1))

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
xy = np.asarray(pos.values) - center
axes[0, 0].plot(xy[:, 0], xy[:, 1], lw=0.3, color="0.5")
axes[0, 0].set_title("trajectory (speed-filtered)")
axes[0, 0].set_aspect("equal")
im = axes[1, 0].imshow(occ.T, origin="lower", extent=[edges[0], edges[-1]] * 2)
axes[1, 0].set_title("occupancy (s)")
plt.colorbar(im, ax=axes[1, 0])

res = []
for i, u in enumerate(d["units"]):
    st = np.asarray(d["units"][u].restrict(d["run_ep"]).t)
    r = gl.analyze_unit(st, pos, occ, edges, center)
    print(f"unit {u} {d['units'].histology[i]} n={len(st):5d} peak={r['peak_rate']:5.1f}Hz "
          f"g={r['gridness']:6.2f} spacing={r['spacing']:6.1f}cm orient={r['orientation']:5.1f} "
          f"SI={r['spatial_info']:.2f}")
    res.append((u, r))

best = sorted(res, key=lambda x: -np.nan_to_num(x[1]["gridness"], nan=-9))[:3]
for col, (u, r) in enumerate(best):
    a = axes[0, col + 1]
    a.plot(xy[:, 0], xy[:, 1], lw=0.2, color="0.75")
    a.plot(r["spike_xy"][:, 0], r["spike_xy"][:, 1], ".", ms=2, color="crimson")
    a.set_aspect("equal")
    a.set_title(f"unit {u}: spikes on path")
    b = axes[1, col + 1]
    b.imshow(r["autocorr"], origin="lower", cmap="jet", vmin=-0.5, vmax=1)
    b.set_title(f"autocorr  g={r['gridness']:.2f}  {r['spacing']:.0f} cm")

plt.tight_layout()
plt.savefig("fig_prototype_session.png", dpi=130)
print("saved fig_prototype_session.png")
