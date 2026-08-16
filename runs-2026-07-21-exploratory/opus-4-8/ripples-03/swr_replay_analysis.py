# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Sharp-Wave Ripples and Replay in the Hippocampus (DANDI 000044)
#
# This notebook demonstrates two hallmark phenomena of hippocampal area CA1 using a
# single freely-available recording from the DANDI Archive:
#
# 1. **Sharp-wave ripples (SWRs)** — transient 150-250 Hz oscillations in the CA1
#    local field potential that occur during non-REM sleep and quiet rest, during
#    which the local population fires a synchronous burst.
# 2. **Replay** — during those ripples the population re-expresses the ordered
#    place-cell firing sequences that encoded the animal's spatial trajectory on the
#    track, at a compressed time scale. We recover this with a Bayesian decoder:
#    the decoded position sweeps coherently across the track within single ripples.
#
# **Dataset.** DANDI:000044, Grosmark, Long & Buzsáki (2016), *"Diversity in neural
# firing dynamics supports both rigid and learned hippocampal sequences"*. We use the
# session `sub-Achilles_ses-Achilles-10252013`: 128-channel silicon-probe CA1
# recording (LFP at 1250 Hz) with 137 sorted units and linearized position on a 1.6 m
# linear track. The session has three epochs — PRE-sleep, Maze (track running),
# POST-sleep — plus REM / Non-REM / Awake state scoring.
#
# **Logic.** Place fields are estimated from track running (the *encoding* map). SWRs
# are detected in the POST-sleep LFP. Within each ripple we Bayesian-decode position
# from the run place fields and test whether the decoded trajectory is a spatially
# coherent line (replay) against a cell-identity shuffle.
#
# All data is streamed from S3 with `remfile` + disk caching; nothing is downloaded in
# full. Analysis uses Pynapple for spike/interval/tuning-curve handling.

# %% [markdown]
# ## Setup and streaming data load
#
# `load_data.py` streams one ripple LFP channel (the channel with the strongest
# ripple-band power, selected empirically across shanks), all unit spike times, the
# linearized position, the epoch table, and the sleep-state table, and caches them to
# `data_cache.npz` so the analysis is fast to re-run.

# %%
import warnings
import numpy as np
import matplotlib.pyplot as plt
import pynapple as nap
warnings.filterwarnings("ignore")

from load_data import load
from place_fields import build_pynapple, running_intervals
from ripples import detect_ripples, build_mask
from replay import build_template, score_ripple, shuffle_scores

d = load()
tsg, pos, epochs = build_pynapple(d)
lfp, lfp_t, fs = d["lfp"], d["lfp_t"], float(d["lfp_fs"])

post = (float(epochs["POSTEpoch"].start[0]), float(epochs["POSTEpoch"].end[0]))
print("epochs:", {k: (round(float(v.start[0]), 1), round(float(v.end[0]), 1)) for k, v in epochs.items()})
print("units:", len(d["spikes"]), "| excitatory:", int((d["cell_type"] == "excitatory").sum()))

# %% [markdown]
# ## 1. Sharp-wave ripple detection
#
# We restrict detection to POST-sleep Non-REM intervals (where SWRs occur), z-score
# the smoothed 150-250 Hz Hilbert envelope, and keep events whose envelope peaks
# above 5 SD with boundaries at 2 SD and a 15-250 ms duration.

# %%
stl, sts, ste = d["st_label"], d["st_start"], d["st_stop"]
nrem = [(max(s, post[0]), min(e, post[1])) for s, e, l in zip(sts, ste, stl)
        if l == "Non-REM" and e > post[0] and s < post[1]]
mask = build_mask(lfp_t, nrem)
rip = detect_ripples(lfp, lfp_t, fs, mask, high_thr=5.0, low_thr=2.0)
n_rip = len(rip["start"])
nrem_s = sum(e - s for s, e in nrem)
dur_ms = (rip["stop"] - rip["start"]) * 1000
print(f"{n_rip} ripples in {nrem_s:.0f}s POST Non-REM "
      f"(rate {n_rip/nrem_s:.2f} Hz); median duration {np.median(dur_ms):.0f} ms")

# %% [markdown]
# The ripple validation figure (`fig_ripples.png`) shows: an example event (wideband
# + ripple-band LFP), the envelope z-score with thresholds, the ripple-triggered
# average LFP (revealing the underlying sharp wave), the ripple-triggered
# time-frequency spectrogram (power concentrated at 150-200 Hz), the peri-ripple
# population firing rate (a synchronous burst), and duration / amplitude / rate
# distributions. It is produced by `viz_ripples.py`.

# %%
import subprocess, sys
subprocess.run([sys.executable, "viz_ripples.py"], check=True)

# %% [markdown]
# ## 2. Place fields on the linear track
#
# Place fields (1D tuning curves) are computed from excitatory (pyramidal) units
# during track running (speed-thresholded periods of the Maze epoch). Fields tile the
# track and, sorted by peak location, form the ordered sequence that replay reactivates.

# %%
posm = pos.restrict(epochs["MazeEpoch"])
run_ep, vel = running_intervals(posm)
rate, cpos, uids = build_template(tsg, posm, run_ep)
print(f"place-cell template: {rate.shape[1]} place cells x {rate.shape[0]} position bins "
      f"({cpos[0]:.2f}-{cpos[-1]:.2f} m); {run_ep.tot_length():.0f}s running")

# %%
# place-field map sorted by peak position
order = np.argsort(np.argmax(rate, 0))
fig, ax = plt.subplots(figsize=(7, 5))
im = ax.imshow((rate / rate.max(0, keepdims=True))[:, order].T, aspect="auto",
               origin="lower", extent=[cpos[0], cpos[-1], 0, rate.shape[1]], cmap="viridis")
ax.set(xlabel="position on track (m)", ylabel="place cell (sorted by peak)",
       title=f"CA1 place fields during running (n={rate.shape[1]})")
fig.colorbar(im, ax=ax, label="normalized firing rate")
fig.tight_layout()
fig.savefig("fig_place_fields.png", dpi=130, bbox_inches="tight")
print("saved fig_place_fields.png")

# %% [markdown]
# ## 3. Bayesian decoding of replay during ripples
#
# For each POST-sleep ripple we bin spikes at 20 ms and apply a memoryless Bayesian
# decoder using the running place-field template, giving a posterior over position at
# each time bin. A spatially coherent trajectory yields a high posterior-weighted
# correlation between time and decoded position. Significance is assessed per event
# against 250 cell-identity shuffles of the template. This is computed by
# `replay_analysis.py` (run below if results are not present) and visualized by
# `viz_replay.py`.

# %%
import os
if not os.path.exists("replay_results.npz"):
    subprocess.run([sys.executable, "replay_analysis.py"], check=True)

res = np.load("replay_results.npz", allow_pickle=True)
rr, pv = res["r"], res["pval"]
n_sig = int((pv < 0.05).sum())
print(f"{len(rr)} decoded ripples; significant replay (p<0.05): {n_sig} ({100*n_sig/len(rr):.0f}%)")
print(f"median |weighted corr|: observed {np.median(np.abs(rr)):.3f} "
      f"vs shuffle {np.median(np.abs(res['null_r'])):.3f}")

# %%
subprocess.run([sys.executable, "viz_replay.py"], check=True)

# %% [markdown]
# ## Summary
#
# `fig_replay_events.png` shows individual ripples in which the decoded position
# forms a clean line sweeping across the track — replayed trajectories compressed into
# tens of milliseconds. `fig_replay_stats.png` shows that, across all ripples, the
# distribution of trajectory-linearity scores is shifted well above the cell-identity
# shuffle, and a substantial fraction of ripples individually pass the shuffle test.
#
# Together the two analyses reproduce the canonical result: CA1 sharp-wave ripples in
# offline (POST-sleep) states carry temporally-compressed replay of the place-cell
# sequences expressed during earlier behavior.
