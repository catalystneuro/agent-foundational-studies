"""Split-half stability control.

A hexagonal autocorrelogram could in principle arise from a lucky arrangement of
a few firing episodes. If the pattern is a genuine spatial code it must repeat
within the session, so each session is split into its first and second half and
the two rate maps are correlated bin by bin.
"""
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import pynapple as nap
from tqdm import tqdm

import grid_lib as gl


def run_session(sess):
    d = gl.load_session(sess["asset_id"])
    pos, ep = d["position"], d["run_ep"]
    half, center = gl.session_extent(pos)
    edges = gl.map_edges(half)
    t0, t1 = float(ep.start[0]), float(ep.end[-1])
    mid = 0.5 * (t0 + t1)
    halves = [
        ep.intersect(nap.IntervalSet(start=t0, end=mid)),
        ep.intersect(nap.IntervalSet(start=mid, end=t1)),
    ]
    occs, poss = [], []
    for h in halves:
        p = pos.restrict(h)
        poss.append(p)
        occs.append(gl.occupancy_map(p, d["dt"], edges, center))

    out = []
    for i, u in enumerate(d["units"]):
        st = np.asarray(d["units"][u].restrict(ep).t)
        if len(st) < 100:
            continue
        rms, gs = [], []
        for h, p, o in zip(halves, poss, occs):
            s = np.asarray(d["units"][u].restrict(h).t)
            r = gl.analyze_unit(s, p, o, edges, center)
            rms.append(r["rate_map"])
            gs.append(r["gridness"])
        ok = np.isfinite(rms[0]) & np.isfinite(rms[1])
        rho = float(np.corrcoef(rms[0][ok], rms[1][ok])[0, 1]) if ok.sum() > 50 else np.nan
        out.append(dict(path=sess["path"], unit=str(d["units"].unit_name[i]),
                        stability_r=rho, gridness_h1=gs[0], gridness_h2=gs[1]))
    return out


def main():
    scan = {s["path"]: s for s in gl.list_assets()}
    ext = json.load(open("session_extent.json"))
    sessions = [scan[e["path"]] for e in ext if max(e["w"], e["h"]) < 120]
    rows = []
    with ProcessPoolExecutor(max(1, min(8, (os.cpu_count() or 4) - 1))) as ex:
        for r in tqdm(ex.map(run_session, sessions), total=len(sessions), desc="stability"):
            rows += r
    df = pd.DataFrame(rows)
    df.to_csv("unit_stability.csv", index=False)
    print(df["stability_r"].describe())


if __name__ == "__main__":
    main()
