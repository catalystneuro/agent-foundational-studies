"""Scan all assets of DANDI 001262 to catalog per-fiber protocol composition.

Reads only small metadata datasets from each NWB file (stimuli table,
analysis table, units sizes) via remfile range requests.
Writes explore/catalog_001262.csv
"""
import json
import re
import sys
import time

import h5py
import numpy as np
import pandas as pd
import remfile
from concurrent.futures import ThreadPoolExecutor

ASSETS = "explore/assets_001262.json"
OUT = "explore/catalog_001262.csv"


def resolve_url(asset_id):
    import urllib.request
    u = ("https://api.dandiarchive.org/api/dandisets/001262/versions/"
         f"0.241205.0959/assets/{asset_id}/")
    with urllib.request.urlopen(u) as r:
        d = json.load(r)
    blob = [x for x in d.get("contentUrl", []) if "s3" in x]
    return blob[0] if blob else None


def probe(a):
    url = resolve_url(a["asset_id"])
    f = h5py.File(remfile.File(url), "r")
    st = "general/intracellular_ephys/intracellular_recordings/stimuli"
    stimtype = f[f"{st}/stimtype"][()].astype(str)
    freq = f[f"{st}/frequency"][()]
    level = f[f"{st}/level_dBSPL"][()]
    seq = f["general/intracellular_ephys/sequential_recordings/stimulus_type"][()].astype(str)
    at = "analysis/analysis_table"
    exp = f[f"{at}/experiment"][()].astype(str)
    bf = f[f"{at}/results_bf"][()]
    sr = f[f"{at}/results_sr"][()]
    nspk = int(f["units/spike_times"].shape[0])
    nswp = int(f["units/spike_times_index"].shape[0])

    # tone-scan frequencies from series names (BF_FREQ<f>), sentinel -99999 excluded
    tone_mask = stimtype == "TONE"
    tf = freq[tone_mask]
    tf = tf[tf > 0]
    tl = level[tone_mask]
    tl = tl[tl > -1000]

    seqkinds = sorted(set(s.split("_")[0] for s in seq))
    f.close()
    return {
        "path": a["path"], "url": url, "size": a["size"],
        "subject": a["path"].split("/")[0].replace("sub-", ""),
        "session": re.search(r"ses-(.+?)_icephys", a["path"]).group(1),
        "experiments": ",".join(sorted(set(exp))),
        "bf": float(bf[0]), "sr": float(sr[0]),
        "n_tone_freqs": len(np.unique(tf)) if len(tf) else 0,
        "freq_min": float(np.min(tf)) if len(tf) else np.nan,
        "freq_max": float(np.max(tf)) if len(tf) else np.nan,
        "n_tone_levels": len(np.unique(tl)) if len(tl) else 0,
        "level_max": float(np.max(tl)) if len(tl) else np.nan,
        "n_spikes": nspk, "n_sweeps": nswp,
        "seq_kinds": ",".join(seqkinds),
        "has_noise": any(s.startswith("NOISE") for s in seq),
        "has_click": any(s.startswith("CLICK") for s in seq),
        "has_ph": any(s.startswith("PH") for s in seq),
        "has_rlf": any(s.startswith("RLF") for s in seq),
    }


def main():
    assets = json.load(open(ASSETS))
    n0 = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    n1 = int(sys.argv[2]) if len(sys.argv) > 2 else len(assets)
    assets = assets[n0:n1]
    t0 = time.time()
    rows, errs = [], 0
    with ThreadPoolExecutor(16) as ex:
        for i, r in enumerate(ex.map(probe, assets)):
            rows.append(r)
            if (i + 1) % 50 == 0:
                dt = time.time() - t0
                print(f"{i+1}/{len(assets)}  {dt:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print("wrote", OUT, df.shape)


if __name__ == "__main__":
    main()
