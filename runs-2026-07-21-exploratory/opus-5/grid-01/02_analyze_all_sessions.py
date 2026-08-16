"""Grid analysis over every session of DANDI:000582.

For each well-isolated unit this computes a smoothed rate map, its spatial
autocorrelogram, a gridness score, grid spacing/orientation, Skaggs spatial
information, split-half stability and head-direction tuning, plus N_SHUFFLE
gridness scores from circularly time-shifted spike trains. The pooled shuffle
distribution gives the significance threshold for calling a cell a grid cell.
"""

import pickle
from multiprocessing import Pool

import numpy as np
import pynapple as nap
from tqdm import tqdm

import gridlib as G

N_SHUFFLE = 100
MIN_SHIFT_S = 20.0
SEED = 20260731


def analyze_session(asset):
    s = G.load_session(asset)
    ep = G.run_epochs(s)
    edges = G.map_edges(s["position"], ep)
    occ = G.occupancy_map(s["position"], ep, edges, s["dt"])

    t0, t1 = float(s["position"].index[0]), float(s["position"].index[-1])
    halves = [nap.IntervalSet(t0, (t0 + t1) / 2).intersect(ep),
              nap.IntervalSet((t0 + t1) / 2, t1).intersect(ep)]
    occ_half = [G.occupancy_map(s["position"], h, edges, s["dt"]) for h in halves]

    rng = np.random.default_rng(abs(hash(asset["path"])) % (2 ** 31) + SEED)
    rows, maps = [], {}

    for i, k in enumerate(s["units"].keys()):
        st = s["units"][k]
        spk = st.restrict(ep)
        n = len(spk)
        mean_rate = n / ep.tot_length()
        row = dict(session=s["path"], subject=s["subject"],
                   session_id=s["session_id"], unit=s["unit_names"][i],
                   histology=s["histology"][i], depth_m=s["depth"][i],
                   arena_cm=float(edges[0][-1] - edges[0][0]),
                   duration_s=s["duration"], n_spikes=int(n),
                   mean_rate=float(mean_rate), included=False)
        if n < G.MIN_SPIKES or mean_rate < G.MIN_MEAN_RATE_HZ:
            rows.append(row)
            continue

        rm, _ = G.rate_map(st, s["position"], ep, edges, occ)
        ac = G.spatial_autocorr(rm)
        g, info = G.grid_score(ac, return_curve=True)
        spacing, orientation, _ = G.grid_geometry(ac)

        # split-half stability of the rate map
        h1, _ = G.rate_map(st, s["position"], halves[0], edges, occ_half[0])
        h2, _ = G.rate_map(st, s["position"], halves[1], edges, occ_half[1])
        m = np.isfinite(h1) & np.isfinite(h2)
        stability = (float(np.corrcoef(h1[m], h2[m])[0, 1])
                     if m.sum() > 30 and np.std(h1[m]) > 0 and np.std(h2[m]) > 0
                     else np.nan)
        g1 = G.grid_score(G.spatial_autocorr(h1))
        g2 = G.grid_score(G.spatial_autocorr(h2))

        # head direction (only where two LEDs were tracked)
        if s["hd"] is not None:
            _, _, mvl, pref = G.hd_tuning(st, s["hd"], ep)
        else:
            mvl, pref = np.nan, np.nan

        # shuffle control
        T = t1 - t0
        offs = rng.uniform(MIN_SHIFT_S, T - MIN_SHIFT_S, N_SHUFFLE)
        sh = np.empty(N_SHUFFLE)
        for j, off in enumerate(offs):
            sr, _ = G.rate_map(G.shifted_spikes(st, t0, t1, off),
                               s["position"], ep, edges, occ)
            sh[j] = G.grid_score(G.spatial_autocorr(sr))

        row.update(included=True, grid_score=float(g), spacing_cm=spacing,
                   orientation_deg=orientation,
                   spatial_info=G.spatial_information(rm, occ),
                   peak_rate=float(np.nanmax(rm)), stability=stability,
                   grid_score_h1=float(g1), grid_score_h2=float(g2),
                   hd_mvl=mvl, hd_pref=pref,
                   shuffle=sh, shuffle_p=float(np.mean(sh >= g)),
                   r_in=info.get("r_in"), r_out=info.get("r_out"))
        rows.append(row)
        maps[(s["path"], s["unit_names"][i])] = dict(rate_map=rm.astype(np.float32),
                                                     autocorr=ac.astype(np.float32),
                                                     edges=[e.astype(np.float32)
                                                            for e in edges])
    s["io"].close()
    return rows, maps


if __name__ == "__main__":
    assets = G.load_assets()
    all_rows, all_maps = [], {}
    with Pool(8) as pool:
        for rows, maps in tqdm(pool.imap_unordered(analyze_session, assets),
                               total=len(assets), desc="sessions"):
            all_rows.extend(rows)
            all_maps.update(maps)

    with open("results.pkl", "wb") as fh:
        pickle.dump(dict(rows=all_rows, maps=all_maps,
                         params=dict(bin_cm=G.BIN_CM, sigma_cm=G.SMOOTH_SIGMA_CM,
                                     min_occ_s=G.MIN_OCC_S,
                                     speed_min=G.SPEED_MIN_CMS,
                                     n_shuffle=N_SHUFFLE)), fh)

    inc = [r for r in all_rows if r["included"]]
    gs = np.array([r["grid_score"] for r in inc])
    shuf = np.concatenate([r["shuffle"] for r in inc])
    thr = np.nanpercentile(shuf, 95)
    print(f"{len(all_rows)} units, {len(inc)} analysed, "
          f"{len(set(r['session'] for r in all_rows))} sessions")
    print(f"shuffle 95th percentile threshold = {thr:.3f}")
    print(f"grid cells: {(gs > thr).sum()} ({(gs > thr).mean() * 100:.1f}%)")
