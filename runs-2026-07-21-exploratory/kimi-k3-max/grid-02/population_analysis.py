"""Population analysis of grid cells across all 118 sessions of DANDI 000582.

For every unit with >= MIN_SPIKES spikes: rate map -> smoothed rate map ->
spatial autocorrelation -> grid score (opexebo). Units with grid score >
GS_CANDIDATE get N_SHUFFLES circular time-shift shuffles for a p-value.

Runs with a multiprocessing pool (fork); each worker gets its own remfile
DiskCache directory to avoid cache collisions.
"""
import os, json, time, pickle, zlib, warnings
import numpy as np
import multiprocessing as mp

BIN_WIDTH = 2.5          # cm
MIN_SPIKES = 200
GS_CANDIDATE = 0.3
N_SHUFFLES = 100
MIN_SHIFT_S = 30.0
N_WORKERS = 8
CACHE_ROOT = "/tmp/remfile_cache_grid02"

ASSETS = json.load(open("assets.json"))


def init_worker(counter):
    global WORKER_ID
    with counter.get_lock():
        WORKER_ID = counter.value
        counter.value += 1


def unit_grid_score(spk, t, xy, occ, arena_size, limits):
    """Rate map -> smooth -> autocorrelation -> grid score for one spike train."""
    import opexebo.analysis as opa
    import opexebo.general as opg
    sx = np.interp(spk, t, xy[:, 0])
    sy = np.interp(spk, t, xy[:, 1])
    st3 = np.vstack([spk, sx, sy])
    rmap = opa.rate_map(occ, st3, arena_size, bin_width=BIN_WIDTH, limits=limits)
    srmap = opg.smooth(rmap, sigma=2)
    ac = opa.autocorrelation(srmap)
    gs, stats = opa.grid_score(ac, bin_width=BIN_WIDTH)
    return gs, stats, srmap, ac


def process_session(args):
    """Process one session; returns (session_path, [unit dicts], error or None)."""
    idx, asset = args
    global WORKER_ID
    import remfile, h5py
    from pynwb import NWBHDF5IO
    import pynapple as nap
    import opexebo.analysis as opa

    warnings.filterwarnings("ignore", category=RuntimeWarning)  # benign scipy vecdot overflow inside opexebo
    path = asset["path"]
    try:
        url = f"https://api.dandiarchive.org/api/assets/{asset['asset_id']}/download/"
        cache = remfile.DiskCache(os.path.join(CACHE_ROOT, f"w{WORKER_ID}"))
        h5 = h5py.File(remfile.File(url, disk_cache=cache), "r")
        nwbfile = NWBHDF5IO(file=h5).read()
        nwb = nap.NWBFile(nwbfile)

        pos = nwb["SpatialSeriesLED1"]
        t = np.asarray(pos.index)
        xy = np.asarray(pos.values, dtype=float)
        good = ~np.isnan(xy).any(axis=1)
        t, xy = t[good], xy[good]
        T = t[-1] - t[0]

        xmin, xmax = xy[:, 0].min(), xy[:, 0].max()
        ymin, ymax = xy[:, 1].min(), xy[:, 1].max()
        limits = (xmin, xmax, ymin, ymax)
        arena_size = float(max(xmax - xmin, ymax - ymin))
        occ, coverage, _ = opa.spatial_occupancy(t, xy.T, arena_size,
                                                 bin_width=BIN_WIDTH, limits=limits)

        units = nwb["units"]
        hist = [str(nwbfile.units["histology"][u]) for u in range(len(units))]
        subject = path.split("/")[0]
        seed = zlib.crc32(path.encode()) % (2**32)

        out = []
        for u in range(len(units)):
            spk = np.asarray(units[u].index)
            spk = spk[(spk >= t[0]) & (spk <= t[-1])]
            rec = dict(session=path, subject=subject, unit=u,
                       histology=hist[u] if hist[u] else "unknown",
                       n_spikes=int(len(spk)), duration_s=float(T),
                       arena_cm=arena_size, grid_score=np.nan,
                       spacing=np.nan, orientation=np.nan,
                       p_value=np.nan, n_shuffles=0)
            if len(spk) < MIN_SPIKES:
                out.append(rec)
                continue
            gs, stats, srmap, ac = unit_grid_score(spk, t, xy, occ, arena_size, limits)
            rec["grid_score"] = float(gs) if gs is not None else np.nan
            rec["spacing"] = float(stats.get("grid_spacing", np.nan))
            rec["orientation"] = float(stats.get("grid_orientation", np.nan))
            if np.isfinite(gs) and gs > GS_CANDIDATE and T > 2 * MIN_SHIFT_S + 60:
                rng = np.random.default_rng(seed + u)
                sh = np.empty(N_SHUFFLES)
                for i in range(N_SHUFFLES):
                    off = rng.uniform(MIN_SHIFT_S, T - MIN_SHIFT_S)
                    spk_s = (spk - t[0] + off) % T + t[0]
                    gs_s, _, _, _ = unit_grid_score(spk_s, t, xy, occ, arena_size, limits)
                    sh[i] = gs_s if (gs_s is not None and np.isfinite(gs_s)) else -np.inf
                rec["p_value"] = float((np.sum(sh >= gs) + 1) / (N_SHUFFLES + 1))
                rec["n_shuffles"] = N_SHUFFLES
            out.append(rec)
        h5.close()
        return path, out, None
    except Exception:
        import traceback
        return path, [], traceback.format_exc()


def main():
    t0 = time.time()
    counter = mp.get_context("fork").Value("i", 0)
    ctx = mp.get_context("fork")
    results, failures = [], []
    with ctx.Pool(N_WORKERS, initializer=init_worker, initargs=(counter,)) as pool:
        for i, (path, recs, err) in enumerate(pool.imap_unordered(
                process_session, list(enumerate(ASSETS))), 1):
            if err:
                failures.append((path, err))
                print(f"[{i}/{len(ASSETS)}] FAILED {path}\n{err}", flush=True)
            else:
                results.extend(recs)
                n_sig = sum(1 for r in recs if r["p_value"] < 0.05)
                print(f"[{i}/{len(ASSETS)}] {path}: {len(recs)} units, "
                      f"{n_sig} sig grid cells ({time.time()-t0:.0f}s)", flush=True)
    with open("population_results.pkl", "wb") as f:
        pickle.dump(dict(results=results, failures=failures), f)
    print(f"\nDONE: {len(results)} units from {len(ASSETS)-len(failures)} sessions, "
          f"{len(failures)} failures, {time.time()-t0:.0f}s total")


if __name__ == "__main__":
    main()
