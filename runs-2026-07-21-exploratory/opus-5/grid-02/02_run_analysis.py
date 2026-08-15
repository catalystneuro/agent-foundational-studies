"""Run the grid-cell analysis over every standard-box session of DANDI:000582.

For each unit: rate map, spatial autocorrelogram, gridness score, grid spacing
and orientation, Skaggs spatial information, head-direction tuning, and a
field-standard shuffling control (random circular shifts of the spike train).

Outputs
-------
unit_metrics.csv      one row per unit
shuffle_gridness.npy  (n_units, N_SHUFFLE) shuffled gridness scores
maps.npz              rate map, autocorrelogram and spike positions per unit
"""
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from tqdm import tqdm

import grid_lib as gl

N_SHUFFLE = 200
MIN_SHIFT = 20.0  # s
MIN_SPIKES = 100
SEED = 20260731


def session_paths():
    scan = {s["path"]: s for s in gl.list_assets()}
    ext = json.load(open("session_extent.json"))
    std = [e["path"] for e in ext if max(e["w"], e["h"]) < 120]
    return [scan[p] for p in sorted(std)]


def run_session(sess):
    d = gl.load_session(sess["asset_id"])
    pos = d["position"]
    half, center = gl.session_extent(pos)
    edges = gl.map_edges(half)
    occ = gl.occupancy_map(pos, d["dt"], edges, center)
    ep = d["run_ep"]
    t0, t1 = float(ep.start[0]), float(ep.end[-1])
    rng = np.random.default_rng(abs(hash(sess["path"])) % (2 ** 31) + SEED)

    rows, shuffles, maps = [], [], {}
    for i, u in enumerate(d["units"]):
        st = np.asarray(d["units"][u].restrict(ep).t)
        key = f"{sess['path']}|{d['units'].unit_name[i]}"
        row = dict(
            path=sess["path"], subject=d["subject"], session=sess["path"].split("ses-")[1][:8],
            unit=str(d["units"].unit_name[i]), layer=str(d["units"].histology[i]),
            depth_mm=float(d["units"].depth[i]) * 1000.0, n_spikes=len(st),
            box_half_cm=half, run_time_s=float(ep.tot_length()),
        )
        if len(st) < MIN_SPIKES:
            row.update(gridness=np.nan, included=False)
            rows.append(row)
            shuffles.append(np.full(N_SHUFFLE, np.nan))
            continue

        r = gl.analyze_unit(st, pos, occ, edges, center)
        row.update(
            included=True, gridness=r["gridness"], spacing_cm=r["spacing"],
            orientation_deg=r["orientation"], peak_rate=r["peak_rate"],
            mean_rate=r["mean_rate"], spatial_info=r["spatial_info"],
            r_in=r["r_in"], r_out=r["r_out"],
        )

        # head-direction tuning (only sessions that tracked both LEDs)
        if d["hd"] is not None:
            centers, tc = gl.hd_tuning(st, d["hd"])
            mvl, pref = gl.mean_vector_length(centers, tc)
            row.update(hd_mvl=mvl, hd_pref=pref)
            maps[key + "|hdtc"] = tc
        else:
            row.update(hd_mvl=np.nan, hd_pref=np.nan)

        # shuffling control: random circular shift of the whole spike train
        sg = np.empty(N_SHUFFLE)
        hd_mvl_sh = np.empty(N_SHUFFLE)
        dur = t1 - t0
        for k in range(N_SHUFFLE):
            shift = rng.uniform(MIN_SHIFT, dur - MIN_SHIFT)
            sst = np.sort(gl.shift_spikes(st, t0, t1, shift))
            sxy = gl.spike_positions(sst, pos, center)
            rm = gl.rate_map(sxy, occ, edges)[0]
            sg[k] = gl.gridness_score(gl.autocorrelogram(rm))
            if d["hd"] is not None:
                c2, tc2 = gl.hd_tuning(sst, d["hd"])
                hd_mvl_sh[k] = gl.mean_vector_length(c2, tc2)[0]
            else:
                hd_mvl_sh[k] = np.nan
        shuffles.append(sg)
        row["gridness_p"] = float(np.mean(sg >= r["gridness"]))
        row["gridness_shuf_95"] = float(np.nanpercentile(sg, 95))
        row["hd_mvl_shuf_95"] = float(np.nanpercentile(hd_mvl_sh, 95)) if d["hd"] is not None else np.nan
        row["hd_p"] = (
            float(np.nanmean(hd_mvl_sh >= row["hd_mvl"])) if d["hd"] is not None else np.nan
        )
        rows.append(row)

        maps[key + "|rm"] = r["rate_map"].astype(np.float32)
        maps[key + "|ac"] = r["autocorr"].astype(np.float32)
        maps[key + "|sxy"] = r["spike_xy"].astype(np.float32)
    maps[sess["path"] + "|traj"] = (np.asarray(pos.values) - center).astype(np.float32)
    maps[sess["path"] + "|edges"] = edges.astype(np.float32)
    return rows, np.array(shuffles), maps


def main():
    sessions = session_paths()
    print(f"{len(sessions)} standard-box sessions, "
          f"{sum(s['n_units'] for s in sessions)} units")
    all_rows, all_shuf, all_maps = [], [], {}
    workers = max(1, min(8, (os.cpu_count() or 4) - 1))
    with ProcessPoolExecutor(workers) as ex:
        for rows, shuf, maps in tqdm(
            ex.map(run_session, sessions), total=len(sessions), desc="sessions"
        ):
            all_rows += rows
            all_shuf.append(shuf)
            all_maps.update(maps)

    df = pd.DataFrame(all_rows)
    df.to_csv("unit_metrics.csv", index=False)
    np.save("shuffle_gridness.npy", np.concatenate(all_shuf, axis=0))
    np.savez_compressed("maps.npz", **all_maps)
    print(df["included"].value_counts())
    print(df.loc[df.included, "gridness"].describe())


if __name__ == "__main__":
    main()
