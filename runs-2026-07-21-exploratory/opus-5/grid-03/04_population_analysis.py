"""Population analysis across all 1 x 1 m open-field sessions of DANDI:000582.

For every unit we compute gridness, grid spacing/orientation, spatial
information, split-half stability and head-direction tuning, plus a
circular-shift shuffle distribution used to set significance thresholds.

Writes: unit_metrics.csv, shuffle_gridness.npy, shuffle_mvl.npy
"""
import multiprocessing as mp
import os
import sys
import zlib

import numpy as np
import pandas as pd
import pynapple as nap
from tqdm import tqdm

import gridlib as gl

N_SHUFFLE = int(os.environ.get("N_SHUFFLE", 100))
MIN_SPIKES = 100          # units with fewer spikes during running are dropped
RNG_SEED = 20260731


def session_list():
    """One session per subject-day, square ~1 m arena, richest in units."""
    df = pd.read_csv("session_survey.csv")
    df = df[(df["shape"] == "square") & (df.ext_x < 110) & (df.ext_y < 110)].copy()
    df["date"] = df.session.astype(str).str.zfill(8).str[:6]
    df = df.sort_values(["n_units", "dur"], ascending=False)
    return df.groupby(["subject", "date"], as_index=False).first().sort_values("path")


def analyze_session(path):
    s = gl.load_session(path)
    pos, units, ep, hd = s["position"], s["units"], s["run_ep"], s["hd"]
    meta = s["meta"]
    edges = gl.map_edges(pos)
    maps, occ, _ = gl.rate_maps(units, pos, ep, edges)

    # split-half stability
    t_mid = 0.5 * (pos.index[0] + pos.index[-1])
    ep1 = ep.intersect(nap.IntervalSet(start=pos.index[0], end=t_mid))
    ep2 = ep.intersect(nap.IntervalSet(start=t_mid, end=pos.index[-1]))
    maps1, _, _ = gl.rate_maps(units, pos, ep1, edges)
    maps2, _, _ = gl.rate_maps(units, pos, ep2, edges)

    if meta["has_hd"]:
        hd_curves, hd_centers, mvl = gl.hd_tuning(units, hd, ep)
    else:
        hd_curves = np.full((gl.HD_BINS, len(units)), np.nan)
        hd_centers = np.linspace(0, 2 * np.pi, gl.HD_BINS, endpoint=False)
        mvl = np.full(len(units), np.nan)

    names = units.get_info("unit_name")
    hist = units.get_info("histology")
    rows, shuf_g, shuf_m = [], [], []
    for i, u in enumerate(units.keys()):
        n_spk = len(units[u].restrict(ep))
        ac = gl.spatial_autocorr(maps[i])
        g, r_out = gl.gridness(ac)
        spacing, ori, ell, pk = gl.autocorr_peaks(ac)
        rows.append(dict(
            path=meta["path"], subject=meta["subject_id"], session=meta["session_id"],
            unit=names[i], layer=str(hist[i]), n_spikes=n_spk,
            mean_rate=n_spk / ep.tot_length(), peak_rate=float(np.nanmax(maps[i])),
            gridness=g, spacing_cm=spacing, orientation_deg=ori, ellipticity=ell,
            spatial_info=gl.spatial_information(maps[i], occ),
            stability=gl.map_correlation(maps1[i], maps2[i]),
            hd_mvl=mvl[i],
            hd_pref_deg=(float(np.degrees(hd_centers[np.nanargmax(hd_curves[:, i])]))
                         if np.isfinite(hd_curves[:, i]).any() else np.nan),
            duration_s=meta["duration_s"], arena_cm=float(edges[-1] - edges[0]),
        ))

    # --- shuffle control: independent circular time shift per unit ------------
    rng = np.random.default_rng(zlib.crc32(path.encode()) ^ RNG_SEED)
    t0, t1 = float(pos.index[0]), float(pos.index[-1])
    keys = list(units.keys())
    for _ in range(N_SHUFFLE):
        shifts = rng.uniform(20.0, (t1 - t0) - 20.0, size=len(keys))
        sh = nap.TsGroup(
            {k: gl.shift_spikes(units[k], shifts[j], t0, t1) for j, k in enumerate(keys)},
            time_support=pos.time_support,
        )
        smaps, _, _ = gl.rate_maps(sh, pos, ep, edges)
        smvl = gl.hd_tuning(sh, hd, ep)[2] if meta["has_hd"] else np.full(len(keys), np.nan)
        for i in range(len(keys)):
            gs, _ = gl.gridness(gl.spatial_autocorr(smaps[i]))
            shuf_g.append((names[i], gs))
            shuf_m.append((names[i], smvl[i]))
    df = pd.DataFrame(rows)
    sg = pd.DataFrame(shuf_g, columns=["unit", "gridness"]).assign(path=meta["path"])
    sm = pd.DataFrame(shuf_m, columns=["unit", "hd_mvl"]).assign(path=meta["path"])
    return df, sg, sm


def _worker(path):
    return analyze_session(path)


if __name__ == "__main__":
    sessions = session_list()
    paths = list(sessions.path)
    print("%d sessions, %d units, %d shuffles/unit"
          % (len(paths), sessions.n_units.sum(), N_SHUFFLE))
    nproc = min(int(os.environ.get("NPROC", 6)), len(paths))
    out = []
    with mp.Pool(nproc) as pool:
        for res in tqdm(pool.imap_unordered(_worker, paths), total=len(paths), file=sys.stdout):
            out.append(res)

    units = pd.concat([o[0] for o in out], ignore_index=True)
    sg = pd.concat([o[1] for o in out], ignore_index=True)
    sm = pd.concat([o[2] for o in out], ignore_index=True)

    units["include"] = units.n_spikes >= MIN_SPIKES
    key = ["path", "unit"]
    inc = set(map(tuple, units.loc[units.include, key].values))
    sg["include"] = [tuple(x) in inc for x in sg[key].values]
    sm["include"] = [tuple(x) in inc for x in sm[key].values]

    units.to_csv("unit_metrics.csv", index=False)
    sg.to_csv("shuffle_gridness.csv", index=False)
    sm.to_csv("shuffle_mvl.csv", index=False)
    print(units.describe())
    print("units included: %d / %d" % (units.include.sum(), len(units)))
    thr = np.nanpercentile(sg.loc[sg.include, "gridness"], 95)
    thrm = np.nanpercentile(sm.loc[sm.include, "hd_mvl"], 95)
    print("shuffled gridness 95th pct = %.3f -> %d grid cells"
          % (thr, (units.loc[units.include, "gridness"] > thr).sum()))
    print("shuffled HD mvl 95th pct = %.3f" % thrm)
