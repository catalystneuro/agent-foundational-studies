"""Cross-check the hand-rolled rate map against pynapple's compute_2d_tuning_curves.

The pipeline uses its own binning so that the identical code path can be reused
for tens of thousands of shuffles, so it is worth confirming that it agrees with
the reference implementation in Pynapple on the unsmoothed maps.
"""
import json

import numpy as np
import pynapple as nap

import grid_lib as gl

scan = {s["path"]: s for s in gl.list_assets()}
ext = json.load(open("session_extent.json"))
path = [e["path"] for e in ext if max(e["w"], e["h"]) < 120][0]
d = gl.load_session(scan[path]["asset_id"])

pos, ep = d["position"], d["run_ep"]
half, center = gl.session_extent(pos)
edges = gl.map_edges(half)
occ = gl.occupancy_map(pos, d["dt"], edges, center)

centred = nap.TsdFrame(t=pos.t, d=np.asarray(pos.values) - center, columns=["x", "y"],
                       time_support=pos.time_support)
nb = len(edges) - 1
tc, binsxy = nap.compute_2d_tuning_curves(
    d["units"], centred, nb_bins=nb, ep=ep,
    minmax=(edges[0], edges[-1], edges[0], edges[-1]),
)

print(f"session {path}\nmap {nb}x{nb} bins, {len(d['units'])} units")
for u in d["units"]:
    st = np.asarray(d["units"][u].restrict(ep).t)
    if len(st) < 100:
        continue
    mine = gl.rate_map(gl.spike_positions(st, pos, center), occ, edges, sigma=0)[0]
    ref = np.asarray(tc[u])  # pynapple returns (x, y) indexed maps
    ok = np.isfinite(mine) & np.isfinite(ref.T)
    r = np.corrcoef(mine[ok], ref.T[ok])[0, 1]
    print(f"  unit {u}: r = {r:.4f}  max diff = {np.nanmax(np.abs(mine[ok] - ref.T[ok])):.3f} Hz")
