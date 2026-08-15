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
# # Theta phase entrainment of hippocampal neurons (DANDI:000059)
#
# Hippocampal neurons do not fire at arbitrary moments during locomotion. Their spikes
# are timed relative to the phase of the 5-11 Hz theta oscillation in the local field
# potential, a phenomenon usually called theta phase locking or theta entrainment. This
# notebook demonstrates that entrainment in real data, quantifies it against an explicit
# null model, and checks it against several alternative explanations.
#
# **Dataset.** DANDI:000059, "Cooling of Medial Septum Reveals Theta Phase Lag
# Coordination of Hippocampal Cell Assemblies" (Petersen & Buzsáki, *Neuron* 2020).
# Rats run a spatial alternation maze while silicon probes record the hippocampus, and
# on a subset of trials the medial septum is cooled through an implanted Peltier device.
# Cooling slows the theta rhythm without stopping it, which provides a causal
# manipulation of the oscillation inside the same recording.
#
# Each session is distributed as two NWB assets: a raw-ephys file that also carries a
# 1250 Hz LFP series, and a processed file with spike-sorted units, position, running
# speed, septal temperature and trial intervals. Both are streamed from S3 with
# `remfile` and a local disk cache; nothing is downloaded in full.
#
# **What is measured.** For each unit, the theta phase of the LFP at the time of every
# spike during running. The strength of entrainment is the mean resultant length (MRL)
# of that circular distribution, and its preferred phase is the circular mean. Phase 0
# is the peak of the band-passed LFP on the channel the original authors flagged as the
# theta reference; phase 180° is its trough.
#
# **Controls applied.** Chance level for each unit comes from circularly shifting that
# unit's own spike train within the running epochs, which preserves its spike count and
# inter-spike-interval structure while destroying its alignment to the LFP. Locking is
# further checked for split-half stability, for consistency with the spike-triggered LFP
# average, and (with a Poisson GLM) for whether theta phase still predicts spiking once
# running speed and the unit's own spike history are accounted for.

# %% [markdown]
# ## Setup
#
# The analysis is split across two helper modules that live next to this notebook:
# `theta_lib.py` (streaming, loading, filtering, circular statistics) and `analysis.py`
# (per-session phase-locking pipeline). The numbered scripts `01_…` to `06_…` are the
# same code organised as a runnable pipeline; this notebook calls into them so that
# figures and numbers cannot drift apart.

# %%
import os
import pickle
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynapple as nap

import analysis as an
import theta_lib as tl

pd.set_option("display.width", 140)
print("sessions used:")
for s in tl.SESSIONS:
    print("  ", s)
print(f"\ntheta band: {tl.THETA_BAND} Hz | running threshold: {tl.SPEED_THRESHOLD} cm/s "
      f"| minimum spikes per unit: {tl.MIN_SPIKES}")


# %%
def run_step(script):
    """Run one pipeline script, streaming its output."""
    print(f"--- {script} ---")
    subprocess.run(["python", script], check=True)


# %% [markdown]
# ## 1. Load one session and validate every data stream
#
# Before any analysis, each stream is inspected: the two NWB assets are checked for a
# shared clock (the last spike time must match the LFP duration), the LFP is plotted
# next to its band-passed version and its Hilbert phase, spikes are rastered against
# it, and running speed and septal temperature are plotted over the session.

# %%
run_step("01_load_data.py")

# %% [markdown]
# ![raw data validation](fig01_raw_data_validation.png)
#
# The wideband LFP shows a continuous ~8 Hz oscillation during running, the extracted
# phase advances monotonically through each cycle, and the septal temperature trace
# shows the cooling epoch (37 °C down to 20 °C and back) in the middle of the session.
#
# ![power spectrum](fig02_power_spectrum.png)
#
# The theta peak is present during maze running and absent during post-maze rest, and
# theta amplitude grows with running speed. Both are the standard checks that the
# 5-11 Hz band on this channel is hippocampal theta rather than filter ringing.

# %% [markdown]
# ## 2. Phase locking, session by session
#
# For each of the five sessions, every curated unit with at least 200 spikes in the
# running epochs gets a spike-phase distribution, an MRL, a preferred phase, a Rayleigh
# test, and a 200-sample circular-shift null. Units are also split into putative
# pyramidal cells and putative interneurons using firing rate and the 3-5 ms mass of
# the autocorrelogram, since this dandiset does not distribute spike waveforms.
#
# Cooled and normal running epochs are separated by septal temperature (>34 °C normal,
# <30 °C cooled) so that the main measurement is made on undisturbed theta.

# %%
run_step("02_run_all_sessions.py")

# %%
df = pd.read_csv("results/unit_stats.csv")
print(f"{len(df)} units from {df.session.nunique()} sessions")
print(df.groupby("cell_type").agg(n=("mrl", "size"), median_rate=("rate", "median"),
                                  median_burst=("burst", "median"),
                                  median_mrl=("mrl", "median"),
                                  median_null_mrl=("mrl_null_mean", "median")))

# %% [markdown]
# ## 3. Population result
#
# The observed MRLs sit far above the shuffled distribution, and about two thirds of
# units are individually significant. Sorting every locked unit by its preferred phase
# shows that preferred phases tile the theta cycle rather than piling up at one value,
# with a population-level bias.

# %%
run_step("03_population.py")

# %% [markdown]
# ![example units](fig03_example_units.png)
#
# Individual units, from the most strongly modulated to two that fail the test. The
# blue curve is the theta cycle for reference and the dashed line marks the preferred
# phase.
#
# ![population summary](fig04_population_summary.png)
#
# Top left: the observed MRL distribution against the circular-shift null. Top middle:
# each unit against its own 95th-percentile null, which is the test that generates the
# significance count. Bottom left: preferred phases of locked units, separated by cell
# type. Putative interneurons lock more strongly than putative pyramidal cells and
# prefer an earlier phase.
#
# ![population heatmap](fig05_population_heatmap.png)
#
# Every significantly locked unit, normalised to its own mean rate and sorted by
# preferred phase, over two theta cycles.

# %%
summary = pd.read_json("results/summary.json", typ="series")
print(summary)

# %% [markdown]
# ## 4. Is the effect an artefact?
#
# Three checks. The spike-triggered average of the raw LFP should oscillate at theta
# for locked units and stay inside its own circular-shift band for unlocked ones. The
# preferred phase should repeat across interleaved subsets of the running epochs rather
# than being a property of one stretch of the recording. And the phase preference
# should not be explained by speed or by the unit's own spike history.

# %%
run_step("04_examples.py")

# %% [markdown]
# ![STA and stability](fig06_sta_and_stability.png)
#
# Left: spike-triggered LFP averages with the 5-95% range of circular-shift shuffles
# shaded. The three locked units produce theta-rhythmic averages many times larger than
# chance; the two units with the weakest locking stay near their bands. Middle: the same
# unit's phase histogram computed on odd and on even running epochs. Right: preferred
# phase on odd versus even epochs for every unit with enough spikes, pooled over
# sessions.

# %% [markdown]
# ### A GLM control for speed and spike history
#
# A concentrated spike-phase distribution can in principle arise without entrainment:
# a unit that bursts at roughly theta frequency has autocorrelated spike times, and
# running speed modulates both firing rate and theta. The Poisson GLM below always
# contains a spline basis over running speed and a raised-cosine basis over the unit's
# own spike history; the question is how much held-out log-likelihood is gained by
# adding a cyclic spline basis over theta phase.

# %%
run_step("06_glm.py")

# %% [markdown]
# ![GLM](fig08_glm.png)

# %%
glm = pd.read_csv("results/glm_results.csv")
print(f"{(glm.d_ll_per_spike > 0).sum()}/{len(glm)} units improve with the phase term; "
      f"median Δ log-likelihood per spike = {glm.d_ll_per_spike.median():.4f}")

# %% [markdown]
# ## 5. A causal manipulation: cooling the medial septum
#
# Cooling the septum slows the theta rhythm. If the spike-phase distributions measured
# above reflect entrainment to that rhythm, they should track it when its frequency
# changes rather than disappearing.
#
# The comparison has to be matched. Cooled running time is shorter than normal running
# time, and the MRL is biased upward both by small spike counts and by short observation
# windows. Each unit's normal-condition MRL is therefore computed inside contiguous
# blocks of normal running epochs whose total duration equals the cooled duration, and
# then subsampled to the same spike count.

# %%
run_step("05_cooling.py")

# %% [markdown]
# ![cooling](fig07_cooling.png)

# %%
cool = pd.read_json("results/cooling_summary.json", typ="series")
print(cool)

# %% [markdown]
# ## 6. What the data show
#
# Hippocampal spiking during locomotion is locked to LFP theta phase. Across five
# sessions, about two thirds of curated units have spike-phase distributions that are
# individually significant against a circular-shift null that preserves each unit's
# spike count and burst structure. The effect is present in the spike-triggered LFP
# average, repeats across interleaved subsets of each session, and survives a GLM that
# already contains running speed and spike history.
#
# Putative interneurons are more strongly entrained than putative pyramidal cells and
# fire at an earlier phase. Cooling the medial septum slows theta by roughly 0.7 Hz and
# reduces its amplitude, and phase locking not only survives but strengthens, with
# preferred phases largely preserved. The measurement therefore tracks the oscillation
# itself rather than any fixed timing in the recording.
