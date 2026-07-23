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
# # Auditory frequency tuning in mouse auditory cortex
#
# **Dataset:** [DANDI:000986](https://dandiarchive.org/dandiset/000986), *Auditory cortex
# Neuropixels recordings and pupil diameter traces from mice during passive exposure to
# pure tones* (version `0.251031.1939`, associated preprint
# [doi:10.1101/2024.04.04.588209](https://doi.org/10.1101/2024.04.04.588209)).
#
# Neurons in the auditory cortex respond preferentially to a restricted range of sound
# frequencies. This notebook demonstrates that phenomenon directly from the archive: it
# streams all 15 sessions of DANDI:000986 (5 mice, 1426 units, 107,272 tone
# presentations), measures each unit's response to five pure tones spanning four octaves,
# and asks four increasingly demanding questions.
#
# 1. Does a unit's evoked firing rate depend on tone frequency at all?
# 2. Does the tuning curve reproduce on trials the estimate never saw?
# 3. Can tone identity be read out from the population on single trials?
# 4. Does an explicit encoding model, fit and scored on separate data, need
#    frequency-specific parameters to predict spiking?
#
# **Stimulus design.** Pure tones at 2, 4, 8, 16 and 32 kHz, each 25 ms long at 60 dB SPL,
# presented in random order roughly every 0.8 s while the mouse sits passively on a wheel.
# Blocks of tones alternate with 300 s spontaneous (no-tone) blocks. Pupil diameter and
# running speed are recorded throughout.
#
# **Access.** Every file is read straight from the DANDI S3 bucket with `remfile` plus a
# local disk cache. Nothing is downloaded in full.

# %% [markdown]
# ## Setup
#
# The pipeline lives in four small modules that this notebook drives:
#
# | module | contents |
# | --- | --- |
# | `dandi_io.py` | asset listing and streaming NWB access |
# | `tuning.py` | response matrices, per-unit statistics, decoder |
# | `glm_analysis.py` | NeMoS Poisson GLM |
# | `figures.py` | every figure |
#
# Long computations cache their results to `.pkl`, so re-running the notebook is fast.
# Delete the `.pkl` files to force a full recomputation (about 20 minutes).

# %%
import os
import pickle
import subprocess

import numpy as np

import dandi_io
import figures as F
import tuning as T

print(f"tone frequencies: {T.FREQS.astype(int)} Hz")
print(f"evoked window {T.EVOKED_WIN} s, baseline window {T.BASELINE_WIN} s, "
      f"PSTH bin {T.PSTH_BIN * 1000:.0f} ms")

# %% [markdown]
# ## 1. What is in the dandiset

# %%
assets = dandi_io.list_assets()
print(f"{len(assets)} sessions, {sum(a['size'] for a in assets) / 1e9:.1f} GB total\n")
for a in assets:
    print(f"  {a['path']:45s} {a['size'] / 1e6:7.1f} MB")

# %% [markdown]
# ## 2. Stream one session and inspect it
#
# `pynapple.NWBFile` gives a typed view of the file: spike trains as a `TsGroup`, trials
# and spontaneous blocks as `IntervalSet`s, pupil and running speed as `Tsd`s.

# %%
EXAMPLE = "sub-LA11/sub-LA11_ses-1_behavior.nwb"
example_asset = next(a for a in assets if a["path"] == EXAMPLE)
nwb, nwbfile = dandi_io.open_session(example_asset)
print(nwb)
print(f"\nsubject {nwbfile.subject.subject_id}, {nwbfile.subject.age}, "
      f"{nwbfile.subject.sex}, {nwbfile.subject.species}")
print(f"session: {nwbfile.session_description}")

# %% [markdown]
# The trials table carries the stimulus parameters. Amplitude and duration are fixed, so
# frequency is the only thing that varies across trials.

# %%
trials = nwbfile.trials.to_dataframe()
print(trials.head())
for col in ["stim_frequency", "stim_duration", "stim_amplitude"]:
    print(f"{col:16s} unique values: {np.unique(trials[col])}")
print(f"\ninter-tone interval: {np.median(np.diff(trials.start_time)):.3f} s")
print(f"trials per frequency:\n{trials.stim_frequency.value_counts().sort_index()}")

# %% [markdown]
# ## 3. Units and response measurement
#
# `load_session_arrays` drops units firing below 0.5 Hz across the session (the units
# table in this dandiset carries no quality metrics, so this is the only screen applied)
# and returns tone onsets sorted in time.

# %%
spikes, onsets, freqs, rates = T.load_session_arrays(nwb, nwbfile)
print(f"{len(spikes)} units kept, session firing rate "
      f"{rates.min():.2f}-{rates.max():.2f} Hz (median {np.median(rates):.2f})")
print(f"{len(onsets)} tone onsets spanning {onsets[-1] - onsets[0]:.0f} s")

# %% [markdown]
# Everything downstream comes from one object: for each unit, a trial-by-time-bin matrix
# of spike counts around tone onset. The evoked window (10-60 ms) and the baseline window
# (-100-0 ms) fall exactly on 5 ms bin edges, so the same matrix serves both the PSTHs and
# the per-trial rate measurements.

# %%
res = T.session_response_matrices(spikes, onsets, freqs)
print("psth     ", res["psth"].shape, " (units, frequencies, time bins)")
print("evoked   ", res["evoked"].shape, " (trials, units)")
print("baseline ", res["baseline"].shape)

# %% [markdown]
# ### Sanity check: are we aligned to the tone?
#
# The NWB trial `start_time` should be the tone onset. Averaging over every unit and every
# trial gives a flat pre-stimulus baseline and a sharp transient just after zero, which
# confirms it.

# %%
edges = T.psth_edges()
ctr_ms = (edges[:-1] + edges[1:]) / 2 * 1000
grand = res["psth"].mean(axis=(0, 1))
print(f"baseline {grand[:20].mean():.2f} Hz, peak {grand.max():.2f} Hz at "
      f"{ctr_ms[np.argmax(grand)]:.1f} ms")

# %% [markdown]
# ## 4. Raw data and stimulus design
#
# Figure 1 shows the raw material: spike rasters with tone onsets overlaid, the population
# rate (visibly locked to the tones), and pupil diameter across the whole two-hour session.
# Figure 2 shows the randomised stimulus sequence, the grand-average response with the
# measurement windows drawn on it, and the distribution of onset latencies across all
# sessions.

# %%
if not os.path.exists("example_raw.pkl"):
    subprocess.run(["python", "example_extras.py"], check=True)
with open("example_raw.pkl", "rb") as f:
    raw = pickle.load(f)

# %% [markdown]
# ## 5. Run the analysis on all 15 sessions
#
# For each session this computes the response matrices, the per-unit statistics, the
# cross-validated population decoder, the decoding-versus-population-size curve and the
# split-half reliability. Takes roughly 15 minutes from a cold cache.

# %%
if not os.path.exists("results.pkl"):
    subprocess.run(["python", "run_sessions.py"], check=True)
results = F.load()
ex = F.example(results)

n_units = sum(len(r["unit_ids"]) for r in results)
n_trials = sum(r["n_trials"] for r in results)
print(f"{len(results)} sessions from {len({r['subject'] for r in results})} mice, "
      f"{n_units} units, {n_trials} tone trials")

# %% [markdown]
# ## 6. Question 1: does firing rate depend on frequency?
#
# For each unit, a Kruskal-Wallis test compares the baseline-subtracted evoked rate across
# the five frequency groups, with Benjamini-Hochberg FDR correction across all units in a
# session. A paired Wilcoxon test of evoked against baseline defines "sound-responsive".

# %%
tuned = np.concatenate([r["reject_tuned"] for r in results])
responsive = np.concatenate([r["reject_responsive"] for r in results])
omega2 = np.concatenate([r["omega2"] for r in results])
selectivity = np.concatenate([r["selectivity"] for r in results])

print(f"sound-responsive : {responsive.sum():4d} / {responsive.size}  "
      f"({responsive.mean():.0%})")
print(f"frequency-tuned  : {tuned.sum():4d} / {tuned.size}  ({tuned.mean():.0%})")
print(f"among tuned units: median omega^2 = {np.median(omega2[tuned]):.3f}, "
      f"median selectivity = {np.median(selectivity[tuned]):.2f}")

# %% [markdown]
# Five example units from the example session, one per best frequency. Each column shows
# the raster grouped by tone frequency, the PSTH per frequency, and the tuning curve. The
# selectivity is obvious by eye: unit 61 responds to 4 kHz and essentially nothing else.

# %%
F.fig_example_units(ex, raw)
print("wrote fig03_example_units.png")

# %% [markdown]
# ## 7. Question 2: does tuning reproduce on held-out trials?
#
# A tuning curve estimated from a random half of the trials is correlated with the curve
# from the other half. Shuffling the frequency labels gives the null.

# %%
rel = np.concatenate([r["reliability"] for r in results])
rel_shuf = np.concatenate([r["reliability_shuffled"] for r in results])
print(f"split-half correlation: median r = {np.median(rel):.2f} "
      f"(shuffled {np.median(rel_shuf):.2f})")
print(f"units with r > 0.5: {(rel > 0.5).mean():.0%} observed vs "
      f"{(rel_shuf > 0.5).mean():.0%} shuffled")

# %% [markdown]
# ### Population view
#
# Sorting every tuned unit by its best frequency produces the block-diagonal structure
# expected if units tile the frequency axis, and the mean tuning curve of each
# best-frequency group peaks at its own frequency. Best frequencies span the whole tested
# range, with a mild over-representation of 8-16 kHz, which is where mouse hearing is most
# sensitive.

# %%
F.fig_population_tuning(results)
F.fig_statistics(results)
bf = np.concatenate([r["bf_idx"][r["reject_tuned"]] for r in results])
for k, f in enumerate(T.FREQS):
    print(f"BF {int(f / 1000):2d} kHz: {100 * np.mean(bf == k):4.1f}% of tuned units")

# %% [markdown]
# ## 8. Question 3: can tone identity be decoded on single trials?
#
# A naive-Bayes Poisson decoder whose templates are exactly the tuning curves measured on
# the training trials, cross-validated five ways. Because the templates *are* the tuning
# curves, decoding accuracy is a direct read-out of how much frequency information they
# carry.

# %%
acc = np.array([r["decode_acc"] for r in results])
acc_shuf = np.array([r["decode_acc_shuffled"] for r in results])
print(f"5-way decoding accuracy: {acc.mean():.3f} "
      f"(range {acc.min():.3f}-{acc.max():.3f}), chance = 0.20")
print(f"with shuffled frequency labels: {acc_shuf.mean():.3f}")

F.fig_decoding(results, ex)

# %% [markdown]
# Two features argue that this reflects genuine frequency tuning rather than a leak in the
# cross-validation. Accuracy climbs smoothly from just above chance with one unit to ~0.85
# with 160 units, exactly as expected if information accumulates across weakly informative
# neurons. And the errors that remain fall overwhelmingly on adjacent octaves, which is
# what overlapping tuning curves predict.

# %% [markdown]
# ## 9. Question 4: an explicit encoding model (NeMoS)
#
# The tests so far all summarise spiking inside a fixed window. A Poisson GLM instead
# models the whole spike train:
#
# $$\log \lambda(t) = b + \sum_k (h_k * s_k)(t)$$
#
# where $s_k(t)$ marks the onsets of tone $k$ and $h_k$ is a temporal kernel expanded in
# eight log-spaced raised cosines over 300 ms. The comparison model collapses the five
# streams into one, so it knows a tone occurred but not which one. Both are fit on
# alternating 60 s chunks and scored on the held-out chunks, so the comparison is honest
# and robust to slow drift in firing rate.

# %%
if not os.path.exists("glm_results.pkl"):
    subprocess.run(["python", "glm_analysis.py"], check=True)
with open("glm_results.pkl", "rb") as f:
    glm = pickle.load(f)

d_ll = (glm["ll_full"] - glm["ll_reduced"]) / glm["bin_size"]
print(f"{len(d_ll)} units fit")
print(f"frequency-specific model predicts held-out spiking better in "
      f"{np.mean(d_ll > 0):.0%} of units")
print(f"median improvement: {np.median(d_ll):.3f} nats/s")

F.fig_glm(ex)

# %% [markdown]
# The fitted kernels are frequency-specific in shape as well as amplitude, and the model
# and the window-based analysis agree on which units are tuned and on what their best
# frequency is, despite estimating them by completely different routes.

# %% [markdown]
# ## 10. Control: is this tuning, or is it brain state?
#
# Cortical responses in this dataset are modulated by arousal, and arousal drifts over a
# two-hour session. If frequency were confounded with state, the "tuning" could be an
# artefact. Because tones are randomly interleaved this is already unlikely, but it can be
# checked directly: split trials at the median pupil diameter at tone onset and re-measure
# every tuning curve within each half.

# %%
with open("arousal.pkl", "rb") as f:
    arousal = pickle.load(f)

t = arousal["tuned"]
r_arousal = np.corrcoef(arousal["tc_low"][t].ravel(), arousal["tc_high"][t].ravel())[0, 1]
bf_agree = np.mean(np.argmax(arousal["tc_low"][t], axis=1)
                   == np.argmax(arousal["tc_high"][t], axis=1))
print(f"tuning curves correlate r = {r_arousal:.2f} across arousal halves")
print(f"same best frequency in {bf_agree:.0%} of tuned units")
print(f"mean evoked rate: {arousal['mean_rate_low']:.2f} Hz at low pupil, "
      f"{arousal['mean_rate_high']:.2f} Hz at high pupil")

F.fig_arousal(arousal)

# %% [markdown]
# Tuning shape survives the split essentially unchanged. What does change is overall gain:
# responses are about 17% larger on high-pupil trials, consistent with the arousal-related
# gain modulation this dataset was collected to study. Frequency tuning and arousal
# modulation are separable effects here.

# %% [markdown]
# ## 11. Remaining figures and summary

# %%
F.fig_raw_data(ex, raw)
F.fig_design_and_alignment(results, ex)
print(sorted(f for f in os.listdir(".") if f.startswith("fig") and f.endswith(".png")))

# %% [markdown]
# ## Conclusions
#
# Frequency tuning in mouse auditory cortex is demonstrated here on four independent
# criteria, all pointing the same way.
#
# * **Prevalence.** 91% of the 1426 units are driven by tones, and 83% show a
#   statistically significant dependence of evoked rate on frequency (Kruskal-Wallis,
#   FDR *q* < 0.05). Median onset latency is 18 ms, the expected value for thalamocortical
#   input to auditory cortex.
# * **Reliability.** Tuning curves estimated on independent halves of the trials correlate
#   at median *r* = 0.98, against −0.03 for shuffled labels; 89% of units exceed *r* = 0.5
#   versus 18% under the null.
# * **Decodability.** A cross-validated Poisson naive-Bayes decoder reads tone identity
#   from single trials at 81% accuracy on average across sessions (chance 20%, shuffled
#   control 20%), rising smoothly with population size, with residual errors concentrated
#   on adjacent octaves.
# * **Encoding model.** A Poisson GLM that is allowed frequency-specific temporal kernels
#   predicts held-out spike trains better than an otherwise identical model that only knows
#   a tone occurred, in 84% of units.
#
# Best frequencies tile the tested range with a bias toward 8-16 kHz. Two caveats limit
# how far the tuning-curve shapes should be pushed: the stimulus set contains only five
# frequencies at octave spacing and a single intensity, so best frequency is quantised to
# octaves and cannot be localised more finely, and the units table in this dandiset carries
# no spike-sorting quality metrics or electrode positions, so no isolation-quality screen
# beyond a 0.5 Hz rate threshold was possible and tonotopic organisation across the probe
# could not be examined.

# %%
print(f"units {n_units} | tuned {tuned.mean():.0%} | decode {acc.mean():.2f} | "
      f"GLM improved {np.mean(d_ll > 0):.0%}")
