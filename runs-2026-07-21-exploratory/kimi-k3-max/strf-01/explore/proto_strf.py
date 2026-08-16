"""Prototype: spectrotemporal response field of one gerbil auditory nerve fiber.

File: sub-G160504_ses-G160504-519 (BF=2341 Hz, 42 tone freqs @ 50 Hz spacing,
5 reps, plus CLICK / RLF / PH / NOISE protocols).
"""
import json
import re

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import remfile

sample = json.load(open("explore/sample_urls.json"))
a = [x for x in sample if "G160504-519" in x["path"]][0]

disk_cache = remfile.DiskCache("/tmp/remfile_cache_strf")
f = h5py.File(remfile.File(a["url"], disk_cache=disk_cache), "r")

# ---- sweep spike trains -------------------------------------------------
spike_times = np.asarray(f["units/spike_times"][:], dtype=np.float64)
spike_index = np.asarray(f["units/spike_times_index"][:], dtype=np.int64)  # cumulative ends
tags = f["units/tag"][:].astype(str)
print("n sweeps:", len(tags), "n spikes:", len(spike_times))

starts = np.concatenate([[0], spike_index[:-1]])
sweep_spikes = [spike_times[s:e] for s, e in zip(starts, spike_index)]

# ---- stimulus metadata ---------------------------------------------------
st = "general/intracellular_ephys/intracellular_recordings/stimuli"
meta = {k: f[f"{st}/{k}"][:] for k in ("stimtype", "frequency", "level_dBSPL", "delay", "duration", "isi", "ramp")}
print("delay (s):", np.unique(meta["delay"][meta["stimtype"].astype(str) == "TONE"]))
print("duration (s):", np.unique(meta["duration"][meta["stimtype"].astype(str) == "TONE"]))

# sampling rate + sweep length from one series
sr = f["acquisition/BF_FREQ2000_rep1/starting_time"].attrs["rate"]
print("sampling rate:", sr)

# ---- parse BF tone sweeps ------------------------------------------------
bf_rows = []  # (freq, rep, sweep_idx)
for i, t in enumerate(tags):
    m = re.match(r"BF_FREQ(\d+)_rep(\d+)", t)
    if m:
        bf_rows.append((int(m.group(1)), int(m.group(2)), i))
freqs = sorted(set(r[0] for r in bf_rows))
reps = sorted(set(r[1] for r in bf_rows))
print("freqs:", len(freqs), freqs[:5], "...", freqs[-3:], "reps:", reps)

# level of BF tones
bf_level = []
for i, t in enumerate(tags):
    if re.match(r"BF_FREQ\d+_rep\d+", t):
        bf_level.append(meta["level_dBSPL"][i])
print("BF tone levels:", np.unique(np.round(bf_level, 1)))

# silent sweeps
sil_idx = [i for i, t in enumerate(tags) if t.startswith("BF_silent")]
print("silent sweeps:", len(sil_idx), "spike counts:", [len(sweep_spikes[i]) for i in sil_idx])

# sweep duration from a trace
ntr = f["acquisition/BF_FREQ2000_rep1/data"].shape[0]
print("sweep samples:", ntr, "=", ntr / sr, "s")

f.close()
