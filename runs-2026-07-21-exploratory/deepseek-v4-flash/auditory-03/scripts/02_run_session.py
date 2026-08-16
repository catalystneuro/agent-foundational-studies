"""Prototype: run tuning analysis on one session and make single-session figures."""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lindi
from pynwb import NWBHDF5IO
import pynapple as nap

sys.path.insert(0, os.path.dirname(__file__))
from tuning_analysis import (
    compute_session_tuning, RESP_WIN, BASE_WIN,
)

FIGDIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIGDIR, exist_ok=True)

ASSET_ID = "aacd1c8a-73f7-469e-bf08-0afd5c1052f9"  # sub-LA11_ses-1_behavior.nwb
URL = f"https://lindi.neurosift.org/dandi/dandisets/000986/assets/{ASSET_ID}/nwb.lindi.json"

local_cache = lindi.LocalCache()
f = lindi.LindiH5pyFile.from_lindi_file(URL, local_cache=local_cache)
io = NWBHDF5IO(file=f, mode="r")
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)

# ---- extract spike times per unit (bulk read to avoid ragged array) ----
st_all = f["units/spike_times"][()]
st_idx = f["units/spike_times_index"][()]
starts = np.concatenate([[0], st_idx])  # cumulative ends
spike_times_list = []
for k in range(len(st_idx)):
    a, b = starts[k], st_idx[k]
    spike_times_list.append(np.sort(st_all[a:b]))

# ---- trials ----
trials = nwb["trials"]
onsets = np.asarray(trials["start"])
freqs_per_trial = np.asarray(trials["stim_frequency"])
order = np.argsort(onsets)
onsets = onsets[order]
freqs_per_trial = freqs_per_trial[order]
frqs = np.sort(np.unique(freqs_per_trial))
print("frequencies (Hz):", frqs)
print("n units:", len(spike_times_list), "n trials:", len(onsets))

# ---- tuning analysis ----
res = compute_session_tuning(
    spike_times_list, onsets, freqs_per_trial, frqs,
    subject_id="LA11", session_id="LA11_ses-1",
)
metrics = res["metrics"]
print("\nMetrics summary:")
print(metrics[["unit", "spontaneous", "peak_rate", "bf_freq", "responsive", "tuned"]].head(10).to_string())
print(f"\nresponsive: {metrics.responsive.sum()}/{len(metrics)} = {metrics.responsive.mean():.2%}")
print(f"tuned:      {metrics.tuned.sum()}/{len(metrics)} = {metrics.tuned.mean():.2%}")
print("BF distribution (Hz):")
print(metrics.loc[metrics.tuned, "bf_freq"].value_counts().sort_index().to_string())

np.savez(
    os.path.join(FIGDIR, "..", "single_session_result.npz"),
    frqs=res["frqs"], rates_net=res["rates_net"],
    rates_resp=res["rates_resp"], rates_base=res["rates_base"],
    bf_idx=res["bf_idx"], bf_freq=res["bf_freq"],
    responsive=metrics.responsive.values, tuned=metrics.tuned.values,
)

print("\nSaved single_session_result.npz")