"""Assemble the staged analysis scripts into one jupytext notebook.

The staged scripts are the working pipeline; this glues them together verbatim so the
notebook and the pipeline can never drift apart.
"""
import re
import subprocess

HEADER = '''# ---
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
# # Spectrotemporal receptive fields in the auditory nerve
#
# **Dataset:** [DANDI:001262](https://dandiarchive.org/dandiset/001262) -
# *Single-unit auditory nerve fibre responses of young-adult and aging gerbils*
# (Heeringa & Koeppl, *Scientific Data* 2024, doi:10.1038/s41597-024-03259-3).
#
# A spectrotemporal receptive field (STRF) is the linear filter that maps a sound's
# time-frequency representation onto a neuron's firing rate. This notebook estimates
# STRFs by reverse correlation in single **auditory-nerve fibres** of the Mongolian
# gerbil, the first spiking stage of the auditory system, using the frozen broadband
# noise protocol in this dandiset.
#
# The dandiset is well suited to the demonstration for three reasons. Each NWB file
# holds one isolated fibre, the acoustic waveform of the noise is stored alongside the
# spikes (so the stimulus does not have to be reconstructed), and every fibre also has
# a pure-tone run whose best frequency was tabulated by the original authors. That
# last point gives an independent check: a STRF measured from broadband noise should
# peak at the frequency a tone sweep says the fibre prefers.
#
# **What the analysis does**
#
# 1. Streams NWB files from the DANDI S3 bucket with `remfile` + a local disk cache.
# 2. Turns the stored noise waveform into a cochleagram (gammatone filterbank ->
#    Hilbert envelope -> dB) and the spikes into a repeat-by-bin count matrix, using
#    `pynapple` objects on a shared time base.
# 3. Estimates the STRF by ridge-regularized reverse correlation, choosing the penalty
#    by cross-validation on held-out segments of the noise token.
# 4. Refits the same filter as a Poisson GLM in `NeMoS`, with and without a
#    spike-history term.
# 5. Repeats the estimate across every fibre in the dandiset that ran the noise
#    protocol, and compares the STRF best frequency with the tabulated tone BF.
#
# **Runtime** is roughly 25 minutes on a warm cache, dominated by the population loop.
# The first run also has to stream ~30 GB of NWB byte ranges.
#
# The helper module `anf_lib.py` must sit next to this notebook.

# %% [markdown]
# ## Setup

# %%
import json
import os
import time

import matplotlib
matplotlib.use("Agg")            # figures are written to disk, never shown
import matplotlib.pyplot as plt
import nemos as nmo
import numpy as np
import pandas as pd
from tqdm import tqdm

import anf_lib as al

print("analysis bin", round(al.BIN * 1e3, 4), "ms;",
      al.N_BANDS, "filterbank bands from", al.F_LO, "to", al.F_HI, "Hz")
'''

SECTIONS = [
    ("01_explore_fibre.py", '''# %% [markdown]
# ## 1. Load one fibre and look at every data stream
#
# Protocols are stored as rows of the `units` table, one row per stimulus repetition,
# tagged with the protocol name (`BF_FREQ*` for the tone run, `NOISE_NOISE<k>_rep<n>`
# for the noise run, `RLF_*` and `PH_*` for rate-level and Schroeder-phase runs).
# Spike times in each row are relative to that repetition's onset.
#
# The first figure shows the four streams that matter: the stored pressure waveform,
# its cochleagram, the spike raster over the 60 repeats, and the resulting PSTH. Note
# the structure of the stored token: two 1 s noise bursts, each followed by silence.
# The raster makes the key point on its own, that the fibre fires at the same moments
# in the noise on every repeat.'''),
    ("02_strf_single_fibre.py", '''# %% [markdown]
# ## 2. The STRF by regularized reverse correlation
#
# The STRF is estimated by regressing the repeat-averaged firing rate on a
# time-lagged cochleagram. Three choices are worth stating explicitly.
#
# *Where it is fitted.* Silence and the onset transient are trivially predictable from
# the stimulus envelope and inflate any prediction score, so the fit and the score use
# ongoing noise only (50 ms after burst onset to burst end). The printout also reports
# what the same procedure gives when silence is included, which is substantially
# higher.
#
# *How it is cross-validated.* The token is frozen, so repeats add no new stimulus.
# The only honest held-out data is a stretch of the token the filter has not seen, so
# cross-validation splits contiguous time blocks. Because the stored token contains
# the same noise twice, blocks are indexed by position within the repeating unit, and
# a segment and its twin always land in the same fold.
#
# *What "good" means.* The response is not deterministic, so prediction is compared
# against a noise ceiling: the split-half reliability of the PSTH itself, corrected by
# Spearman-Brown.'''),
    ("04_glm_nemos.py", '''# %% [markdown]
# ## 3. The same filter as a Poisson GLM (NeMoS)
#
# Reverse correlation estimates one free weight per frequency-by-lag cell and predicts
# a rate. A Poisson GLM expands the filter in a raised-cosine basis (roughly a quarter
# as many parameters) and models the spike train itself, which makes it natural to add
# a spike-history filter for refractoriness. Here the data are split into contiguous
# train (60%), validation (20%) and test (20%) blocks; the validation blocks pick the
# ridge strength and the test blocks are scored once.'''),
    ("05_population.py", '''# %% [markdown]
# ## 4. Across the population of fibres
#
# 143 of the 1160 files in the dandiset contain a noise run. For each fibre with a
# tabulated BF inside the noise passband, the STRF is fitted exactly as above, and the
# shape metrics are taken at one fixed ridge penalty so that fibres are comparable.
#
# The latency panel corrects for a real artifact of the analysis: a gammatone filter
# centred at 500 Hz rings for ~6 ms before its envelope peaks, while an 11 kHz filter
# peaks after 0.35 ms. Left uncorrected, that alone would produce a latency-versus-
# frequency trend with no physiology in it.
#
# This cell reads `all_tags.json`, the index of which files contain which protocols;
# if it is missing, the scan is run first (about 25 minutes over 1160 files).'''),
    ("06_revcor.py", '''# %% [markdown]
# ## 5. Reverse correlation on the waveform itself
#
# The cochleagram keeps only the envelope of each band. Averaging the raw pressure
# waveform preceding each spike keeps the carrier as well, which is the original
# revcor measurement of de Boer & de Jongh (1978). Low-BF fibres lock to the fine
# structure of the noise, so their revcor is a ringing filter whose spectrum peaks at
# the fibre's BF; high-BF fibres do not phase lock and their revcor is flat.'''),
]

SCAN_CELL = '''# %% [markdown]
# ### Index of which files ran which protocol
#
# The protocol vocabulary is only visible inside each file, so building the population
# requires opening all 1160 assets once and recording their tags. The result is cached
# in `all_tags.json`.

# %%
if not os.path.exists("all_tags.json"):
    import runpy
    runpy.run_path("scan_all_tags.py")
else:
    print("using cached all_tags.json:",
          sum(1 for r in json.load(open("all_tags.json")) if r.get("noise")),
          "files with a noise run")
'''

TAIL = '''# %% [markdown]
# ## Summary
#
# Reverse correlation on frozen broadband noise recovers a clean spectrotemporal
# receptive field in single auditory-nerve fibres: a narrow excitatory region at the
# fibre's best frequency, beginning within about a millisecond of the stimulus and
# followed by weaker suppression over the next 10 ms. The estimate is validated three
# ways. It predicts the response to held-out stretches of the noise well above chance
# and at a substantial fraction of the reliability ceiling; it collapses when the PSTH
# is shifted or shuffled; and its best frequency agrees with the best frequency
# measured independently from pure tones, which is the strongest check because the two
# come from different stimulus classes.
#
# The limits are worth stating as clearly as the results. The stimulus is a single
# frozen token, so cross-validation tests generalization to unseen segments of one
# noise process rather than to a new stimulus ensemble. Prediction is capped well
# below the ceiling because a linear filter on a 1 ms envelope representation cannot
# reproduce the fastest structure in the PSTH, which for low-BF fibres includes phase
# locking to the carrier (section 5). And the STRF's spectral width is measured at the
# resolution of the analysis filterbank, so it should be read as an upper bound on how
# sharply these fibres are tuned rather than as a tuning-curve measurement.
'''


def body(path):
    src = open(path).read()
    # drop the module docstring, the matplotlib backend line and the imports; the
    # setup cell above already covers them
    src = re.sub(r'^""".*?"""\n', "", src, flags=re.S)
    keep = []
    for line in src.split("\n"):
        if re.match(r"^(import |from |import matplotlib;)", line):
            continue
        keep.append(line)
    return "\n".join(keep).strip("\n")


out = [HEADER]
for path, md in SECTIONS:
    out.append(md)
    if path == "05_population.py":
        out.append(SCAN_CELL)
    out.append("\n# %%\n" + body(path))
out.append(TAIL)

text = "\n\n".join(out) + "\n"
open("strf_auditory_nerve.py", "w").write(text)
print("wrote strf_auditory_nerve.py", len(text.split("\n")), "lines")
subprocess.run(["jupytext", "--to", "notebook", "strf_auditory_nerve.py"], check=True)
print("wrote strf_auditory_nerve.ipynb")
