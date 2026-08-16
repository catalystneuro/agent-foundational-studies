"""03_run_all_sessions.py — grid-cell analysis across all 118 sessions of DANDI 000582.

For every session: occupancy map, smoothed rate maps, autocorrelations, grid
scores for all units with >= MIN_SPIKES spikes; units with grid score > 0.3 get
100 circular time-shift shuffles for a significance p-value. Per-session maps
are saved to results/maps/*.npz and the unit-level table to results/grid_scores.csv.

Runs with 8 fork-context workers; each worker gets its own remfile DiskCache
directory to avoid cache collisions.
"""

import json
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
import pandas as pd
from tqdm import tqdm

from grid_utils import analyze_session_units, load_session

RESULTS_DIR = "results"
MAPS_DIR = os.path.join(RESULTS_DIR, "maps")
N_WORKERS = 8


def process_session(args):
    """Analyze one session; save maps npz; return list of unit-level rows."""
    path, asset_id, worker_id = args
    cache_dir = f"/tmp/remfile_cache_grid_w{worker_id}"
    nwb, io = load_session(asset_id, cache_dir=cache_dir)
    results, meta = analyze_session_units(nwb, run_shuffles=True, seed=hash(path) % 1000)
    io.close()

    rows = []
    maps = {}
    for uid, res in results.items():
        row = {
            "session": path,
            "unit_id": uid,
            "unit_name": res["unit_name"],
            "layer": res["layer"],
            "hemisphere": res["hemisphere"],
            "n_spikes": res["n_spikes"],
            "grid_score": res["grid_score"],
            "grid_spacing": res["grid_spacing"],
            "grid_orientation": res["grid_orientation"],
            "shuffle_p": res.get("shuffle_p", np.nan),
            "shuffle_z": res.get("shuffle_z", np.nan),
            "arena_cm": meta["arena_size"][0],
            "duration_s": meta["duration"],
            "coverage": meta["coverage"],
        }
        rows.append(row)
        maps[f"rate_map_{uid}"] = res["rate_map"]
        maps[f"acorr_{uid}"] = res["acorr"]
        if "shuffle_scores" in res:
            maps[f"shuffle_{uid}"] = res["shuffle_scores"]

    os.makedirs(MAPS_DIR, exist_ok=True)
    safe = path.replace("/", "_").replace(".nwb", "")
    np.savez_compressed(os.path.join(MAPS_DIR, safe + ".npz"), **maps)
    return rows


def main():
    with open("asset_manifest.json") as f:
        manifest = json.load(f)
    items = sorted(manifest.items())

    os.makedirs(MAPS_DIR, exist_ok=True)
    all_rows = []
    failures = {}
    ctx = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=N_WORKERS, mp_context=ctx) as ex:
        futures = {
            ex.submit(process_session, (path, aid, i % N_WORKERS)): path
            for i, (path, aid) in enumerate(items)
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="sessions"):
            path = futures[fut]
            try:
                all_rows.extend(fut.result())
            except Exception:
                failures[path] = traceback.format_exc()
                print(f"\nFAILED: {path}\n{failures[path]}")

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(RESULTS_DIR, "grid_scores.csv"), index=False)
    print(f"\n{len(df)} units analyzed from {len(items) - len(failures)} sessions; "
          f"{len(failures)} sessions failed")
    if failures:
        raise RuntimeError(f"{len(failures)} sessions failed: {list(failures)}")


if __name__ == "__main__":
    main()
