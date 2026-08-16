"""Inspect NWB metadata for all sessions in DANDI 000017 without downloading whole files."""
import h5py
import remfile

TSV = "assets_000017.tsv"
cache = remfile.DiskCache("/tmp/remfile_cache_000017")

import time

rows = []
with open(TSV) as f:
    for line in f:
        asset_id, path = line.strip().split()
        rows.append((asset_id, path))

for asset_id, path in rows:
    url = f"https://api.dandiarchive.org/api/assets/{asset_id}/download/"
    t0 = time.time()
    try:
        rf = remfile.File(url, disk_cache=cache)
        h5 = h5py.File(rf, "r")
        gen = h5["general"]
        def g(name):
            try:
                v = gen[name][()] if name in gen else None
                if isinstance(v, bytes):
                    v = v.decode("utf-8", "replace")
                return v
            except Exception as e:
                return f"<err {e}>"
        session_desc = g("session_description")
        exp_desc = g("experiment_description")
        stim_desc = g("stimulus") if "stimulus" in gen else None
        n_trials = h5["intervals"].get("trials", {}).get("id", None)
        n_trials = len(n_trials) if n_trials is not None else "?"
        n_units = h5["units"].attrs.get("colnames", "?")
        try:
            n_units = len(h5["units"]["id"][()])
        except Exception:
            pass
        # list interval columns
        cols = list(h5["intervals"]["trials"].keys()) if "/intervals/trials" in h5 else []
        print(f"{path[:40]:42s} sess_desc={str(session_desc)[:60]:60s} exp={str(exp_desc)[:50]:50s} trials={n_trials} units={n_units} cols={cols}")
        h5.close()
    except Exception as e:
        print(f"{path[:40]:42s} ERROR: {e}")
    print(f"   (took {time.time()-t0:.1f}s)")