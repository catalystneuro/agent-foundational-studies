# Harvest STRFs across many gerbil AN fibers
import requests
import h5py
import remfile
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import re
import sys

DANDI = "001262"
VER = "0.241205.0959"
CACHE = "/tmp/remfile_cache_strf"

T0, T1 = -0.010, 0.090
BIN_W = 0.001
EDGES = np.arange(T0, T1 + BIN_W, BIN_W)
CENTERS = 0.5 * (EDGES[:-1] + EDGES[1:])


def get_s3_url(asset_id):
    r = requests.get(
        f"https://api.dandiarchive.org/api/dandisets/{DANDI}/versions/{VER}/assets/{asset_id}/"
    )
    d = r.json()
    for u in d["contentUrl"]:
        if "s3" in u:
            return u
    return d["contentUrl"][0]


def analyze_fiber(asset_id, path):
    url = get_s3_url(asset_id)
    h5 = h5py.File(remfile.File(url, disk_cache=remfile.DiskCache(CACHE)), "r")

    # analysis table: does it have BF?
    at = h5["analysis/analysis_table"]
    experiments = [e.decode() for e in at["experiment"][:]]
    if "BF" not in experiments:
        return None
    bf_rows = at["results_bf"][:]
    table_bf = float(bf_rows[experiments.index("BF")])
    sr_table = float(at["results_sr"][:][experiments.index("BF")])

    IR = h5["general/intracellular_ephys/intracellular_recordings"]
    S = IR["stimuli"]
    stim = pd.DataFrame({
        "tag": [t.decode() for t in IR["tag"][:]],
        "frequency": S["frequency"][:].astype(float),
        "level": S["level_dBSPL"][:].astype(float),
        "delay": S["delay"][:].astype(float),
        "duration": S["duration"][:].astype(float),
    })
    bf = stim[stim["tag"].str.match(r"BF_FREQ\d+_rep\d+")].copy()
    bf = bf[bf["frequency"] > 0]
    if len(bf) == 0:
        return None

    lvl_counts = bf.groupby("level").size()
    level = float(lvl_counts.idxmax())
    bfl = bf[bf["level"] == level]
    freqs = np.sort(bfl["frequency"].unique())
    if len(freqs) < 12:
        return None
    reps = len(bfl) / len(freqs)
    if reps < 3:
        return None
    delay = float(bfl["delay"].iloc[0])
    dur = float(bfl["duration"].iloc[0])

    spike_times = h5["units/spike_times"][:]
    spike_index = h5["units/spike_times_index"][:].astype(np.int64)
    tags = [t.decode() for t in h5["units/tag"][:]]
    bounds = np.concatenate([[0], spike_index])
    tag2spk = {}
    for i, t in enumerate(tags):
        st = spike_times[bounds[i]:bounds[i + 1]]
        tag2spk[t] = st[np.isfinite(st)]

    psth = np.zeros((len(freqs), len(CENTERS)))
    counts = np.zeros(len(freqs))
    fidx = {f: i for i, f in enumerate(freqs)}
    for _, row in bfl.iterrows():
        st = tag2spk.get(row["tag"])
        if st is None:
            continue
        rel = st - delay
        sel = rel[(rel >= T0) & (rel < T1)]
        fi = fidx[row["frequency"]]
        psth[fi] += np.histogram(sel, EDGES)[0]
        counts[fi] += 1
    if (counts == 0).any():
        return None
    rate = psth / (counts[:, None] * BIN_W)
    baseline = rate[:, CENTERS < 0].mean(axis=1, keepdims=True)
    net = rate - baseline
    net_s = gaussian_filter(net, sigma=(1.5, 1.5))

    resp = (CENTERS >= 0) & (CENTERS < dur)
    tc = net_s[:, resp].mean(axis=1)
    cfi = int(np.argmax(tc))
    cf = float(freqs[cfi])
    peak_rate = float(tc[cfi])
    sr = float(baseline.mean())

    # latency at CF: first half-peak crossing after onset
    psth_cf = net_s[cfi]
    pk = float(psth_cf[(CENTERS > 0) & (CENTERS < dur)].max())
    half = pk / 2
    above = np.where((psth_cf > half) & (CENTERS > 0))[0]
    latency = float(CENTERS[above[0]]) if len(above) else np.nan

    # bandwidth at half max of tuning curve
    bw_oct = np.nan
    q_factor = np.nan
    if peak_rate > 0:
        over = tc >= peak_rate / 2
        # contiguous region around CF
        lo, hi = cfi, cfi
        while lo > 0 and over[lo - 1]:
            lo -= 1
        while hi < len(freqs) - 1 and over[hi + 1]:
            hi += 1
        if lo > 0 and hi < len(freqs) - 1:  # resolved on both sides
            f_lo, f_hi = float(freqs[lo]), float(freqs[hi])
            bw_oct = float(np.log2(f_hi / f_lo))
            q_factor = cf / (f_hi - f_lo) if f_hi > f_lo else np.nan

    h5.close()
    return {
        "asset_id": asset_id, "path": path,
        "subject": path.split("/")[0],
        "table_bf": table_bf, "table_sr": sr_table,
        "cf": cf, "latency_ms": latency * 1000, "peak_rate": peak_rate,
        "spont_rate": sr, "level": level, "n_freqs": len(freqs),
        "n_reps": float(reps), "bw_oct": bw_oct, "q_factor": q_factor,
        "freqs": freqs, "net_map": net_s, "centers": CENTERS, "dur": dur,
    }


def main(n_target=80, seed=0):
    cat = pd.read_csv("asset_catalog.csv")
    cat = cat[~cat["is_optical"]].reset_index(drop=True)
    # stratified sample across subjects for CF diversity
    subjects = cat["subject"].unique()
    rng = np.random.default_rng(seed)
    picks = []
    per_subj = max(1, int(np.ceil(n_target / len(subjects))))
    for subj in subjects:
        sub = cat[cat["subject"] == subj]
        k = min(per_subj, len(sub))
        idx = rng.choice(len(sub), size=k, replace=False)
        picks.append(sub.iloc[idx])
    sel = pd.concat(picks).head(n_target)
    print(f"selected {len(sel)} files across {sel['subject'].nunique()} subjects")

    results = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(analyze_fiber, r.asset_id, r.path): r.path
                for r in sel.itertuples()}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="fibers"):
            try:
                res = fut.result()
            except Exception as e:
                print(f"ERROR {futs[fut]}: {type(e).__name__}: {e}")
                continue
            if res is not None:
                results.append(res)

    print(f"got {len(results)} fibers with valid BF STRFs")
    # save
    metrics = pd.DataFrame([{k: v for k, v in r.items()
                             if k not in ("freqs", "net_map", "centers")}
                            for r in results])
    metrics.to_csv("population_metrics.csv", index=False)
    np.savez_compressed(
        "population_strf.npz",
        **{f"map_{i}": r["net_map"] for i in range(len(results))},
        **{f"freqs_{i}": r["freqs"] for i in range(len(results))},
        centers=CENTERS,
    )
    print("saved population_metrics.csv and population_strf.npz")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 80
    main(n)
