"""Compute frequency-tuning metrics for every unit in all 15 sessions of DANDI 000986.

Saves one CSV per session into results/ plus a combined all_units.csv.
"""
import json
import os
import time

import h5py
import remfile
import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO
import pynapple as nap
from scipy import stats
from tqdm import tqdm

RESP = (0.005, 0.055)   # response window (s) relative to tone onset
BASE = (-0.050, 0.0)    # baseline window (s)
MIN_RATE = 0.5          # Hz
OUT = "results"
os.makedirs(OUT, exist_ok=True)

with open("session_urls.json") as f:
    URLS = json.load(f)

disk_cache = remfile.DiskCache('/tmp/remfile_cache_auditory03')


def analyze_session(path, s3_url):
    rem_file = remfile.File(s3_url, disk_cache=disk_cache)
    h5py_file = h5py.File(rem_file, "r")
    io = NWBHDF5IO(file=h5py_file)
    nwbfile = io.read()
    nwb = nap.NWBFile(nwbfile)

    units = nwb["units"]
    trials = nwb["trials"]
    freqs = np.array(sorted(np.unique(trials["stim_frequency"])))
    tone_onsets = nap.Ts(t=np.array(trials["start"]))
    trial_freq = np.array(trials["stim_frequency"])

    t_start = float(units.time_support.start[0])
    t_end = float(units.time_support.end[0])
    duration = t_end - t_start

    sub = path.split("/")[0]
    ses = path.split("/")[1].split("_")[1]

    rows = []
    for u in range(len(units)):
        sp = units[u]
        rate = len(sp.t) / duration
        if rate < MIN_RATE:
            continue
        pe = nap.compute_perievent(sp, tone_onsets, window=(BASE[0], RESP[1]))
        n_trials = len(tone_onsets)
        resp = np.zeros(n_trials)
        base = np.zeros(n_trials)
        for j in range(n_trials):
            t = pe[j].t
            resp[j] = np.sum((t >= RESP[0]) & (t < RESP[1])) / (RESP[1] - RESP[0])
            base[j] = np.sum((t >= BASE[0]) & (t < BASE[1])) / (BASE[1] - BASE[0])
        net = resp - base
        groups = [net[trial_freq == f] for f in freqs]
        H, p = stats.kruskal(*groups)
        tuning = np.array([g.mean() for g in groups])
        sem = np.array([g.std() / np.sqrt(len(g)) for g in groups])
        bf_idx = int(np.argmax(tuning))
        bf = float(freqs[bf_idx])

        # tuning width at half max (octaves), excitatory peaks only
        peak = tuning[bf_idx]
        if peak > 0:
            above = tuning >= peak / 2
            width_oct = float(above.sum() - 1)  # adjacent freqs are 1 octave apart
        else:
            width_oct = np.nan

        # selectivity index on rectified responses: 0 = flat, 1 = single-freq
        rect = np.clip(tuning, 0, None)
        sel = float(1 - rect.sum() / (len(rect) * rect.max())) if rect.max() > 0 else np.nan

        # response latency at BF: first 1-ms bin where PSTH exceeds 20% of peak
        idx_bf = np.where(trial_freq == freqs[bf_idx])[0]
        edges = np.arange(-0.05, 0.15, 0.001)
        all_t = np.concatenate([pe[j].t for j in idx_bf]) if len(idx_bf) else np.array([np.nan])
        counts, _ = np.histogram(all_t, bins=edges)
        psth = counts / (len(idx_bf) * 0.001)
        base_rate = base.mean()
        pk = psth[(edges[:-1] >= 0) & (edges[:-1] < 0.1)].max()
        latency = np.nan
        if pk > base_rate:
            thr = base_rate + 0.2 * (pk - base_rate)
            post = np.where((edges[:-1] >= 0) & (psth > thr))[0]
            if len(post):
                latency = float(edges[post[0]] * 1000)  # ms

        row = dict(subject=sub, session=ses, unit=u, rate=rate, kw_H=H, kw_p=p,
                   bf=bf, peak_net=peak, width_oct=width_oct, selectivity=sel,
                   latency_ms=latency, n_trials=n_trials)
        for k, f in enumerate(freqs):
            row[f"tune_{int(f)}"] = tuning[k]
            row[f"sem_{int(f)}"] = sem[k]
        rows.append(row)

    h5py_file.close()
    return pd.DataFrame(rows)


all_dfs = []
for path, s3_url in URLS.items():
    tag = path.replace("/", "_").replace("_behavior.nwb", "")
    csv_path = f"{OUT}/{tag}.csv"
    if os.path.exists(csv_path):
        print(f"skip {tag} (cached csv)")
        all_dfs.append(pd.read_csv(csv_path))
        continue
    t0 = time.time()
    df = analyze_session(path, s3_url)
    df.to_csv(csv_path, index=False)
    all_dfs.append(df)
    n_sig = (df.kw_p < 0.01).sum()
    print(f"{tag}: {len(df)} units analyzed, {n_sig} tuned ({time.time()-t0:.0f} s)",
          flush=True)

all_units = pd.concat(all_dfs, ignore_index=True)
all_units.to_csv(f"{OUT}/all_units.csv", index=False)
print("TOTAL units:", len(all_units), " tuned:", (all_units.kw_p < 0.01).sum())
print("DONE")
