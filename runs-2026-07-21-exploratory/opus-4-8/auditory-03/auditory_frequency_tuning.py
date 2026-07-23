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
# # Frequency tuning in mouse auditory cortex
#
# **Dataset:** [DANDI:000986](https://dandiarchive.org/dandiset/000986), *Auditory cortex
# Neuropixels recordings and pupil diameter traces from mice during passive exposure to
# pure tones* (Lohse, Zhang, King and colleagues; preprint
# [doi:10.1101/2024.04.04.588209](https://doi.org/10.1101/2024.04.04.588209)).
#
# Neurons in the auditory system respond preferentially to a restricted range of sound
# frequencies. A neuron's *tuning curve* is its firing rate as a function of tone
# frequency, and the frequency at which that curve peaks is its *best frequency* (BF).
# This notebook demonstrates that phenomenon in real extracellular recordings, and then
# asks the questions that separate genuine tuning from an artefact of taking the maximum
# of a noisy curve:
#
# 1. Do individual units respond more to some tone frequencies than to others?
# 2. Is a unit's best frequency reproducible on trials that were not used to estimate it?
# 3. Do different units prefer different frequencies, so that the population as a whole
#    covers the audible range?
# 4. Does the tuning carry enough information to identify which tone was played on a
#    single trial?
# 5. Does an explicit encoding model (a Poisson GLM) predict spiking better when it is
#    allowed to know the tone frequency than when it is not?
#
# **The recordings.** Fifteen sessions from five head-fixed mice, Neuropixels 1.0 probes
# in auditory cortex. In each session roughly 7,400 pure tones of 25 ms duration were
# presented at 60 dB SPL, one every 0.8 s, drawn at random from five frequencies spaced
# one octave apart (2, 4, 8, 16 and 32 kHz). The mice were passively listening; there was
# no task. The files also contain pupil diameter and running speed, which are not used
# here.
#
# **Access.** Every file is read directly from the DANDI S3 bucket with `remfile` and a
# local disk cache, so nothing is downloaded in full. Spike times and trial tables are
# handled with `pynapple`, and the GLM at the end uses `nemos`.

# %%
import os

import matplotlib
matplotlib.use("Agg")           # this notebook is meant to run headless
import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap

import plotting as P
from dandi_io import list_assets, load_tone_session
from plotting import khz
from tuning_core import session_tuning

nap.nap_config.suppress_conversion_warnings = True

EVOKED = (0.010, 0.060)   # tone response window, in seconds from tone onset
BASE = (-0.100, -0.010)   # pre-tone baseline window

assets = list_assets("000986")
print(f"{len(assets)} sessions:")
for path, _ in assets:
    print("   ", path)

# %% [markdown]
# ## 1. One session: what is in the file
#
# Each NWB file holds a `units` table (spike times of sorted units) and a `trials` table
# with one row per tone, giving its onset time, frequency, level and duration. Pynapple
# turns the units table into a `TsGroup`, which is the container the rest of the analysis
# works with.

# %%
SESSION = "sub-LA11/sub-LA11_ses-1_behavior.nwb"
S = load_tone_session(dict(assets)[SESSION])
spikes, onsets, freq, level = S["spikes"], S["onsets"], S["freq"], S["level"]
ufreq = np.unique(freq)

print(S["nwbfile"].session_description)
print(f"subject {S['nwbfile'].subject.subject_id}, age {S['nwbfile'].subject.age}, "
      f"sex {S['nwbfile'].subject.sex}")
print(f"{len(spikes)} sorted units, {len(onsets)} tones")
print(f"frequencies: {ufreq/1000} kHz at {np.unique(level)} dB SPL")
print(f"tone duration: {np.unique(np.round(S['nwbfile'].trials.stim_duration[:], 4))} s")
print(f"median inter-tone interval: {np.median(np.diff(onsets)):.3f} s")
print(f"recording spans {(onsets[-1] - onsets[0]) / 60:.0f} min; "
      f"firing rates (5th/50th/95th pct): "
      f"{np.percentile(spikes.rate, [5, 50, 95]).round(2)} Hz")

# %% [markdown]
# ### Raw activity around the tones
#
# Before computing anything, look at the spikes themselves. The panel below shows six
# seconds of the recording for the 40 most active units, with the tones drawn as shaded
# bars. Tones arrive every 0.8 s, and since the response is over within about 150 ms
# (next section) the pre-tone window used as a baseline below is genuinely between
# responses. Tone frequency is randomized from tone to tone, so any residual carry-over
# would in any case be spread evenly across the five conditions.

# %%
import matplotlib.lines as mlines

COL = P.freq_colors(ufreq)
DUR = 6.0
t0 = onsets[1000]
win = nap.IntervalSet(start=t0, end=t0 + DUR)
fig, ax = plt.subplots(figsize=(12, 4.4), constrained_layout=True)
uids = np.array(list(spikes.keys()))[np.argsort(np.asarray(spikes.rate))[::-1][:40]]
for row, u in enumerate(uids):
    st = np.asarray(spikes[u].restrict(win).t)
    ax.plot(st, np.full_like(st, row), "|", color="k", ms=4, mew=0.7)
sel = (onsets > t0) & (onsets < t0 + DUR)
for o, f in zip(onsets[sel], freq[sel]):
    ax.axvspan(o, o + 0.025, color=COL[int(np.searchsorted(ufreq, f))], alpha=0.55, lw=0)
ax.legend(handles=[mlines.Line2D([], [], color=c, lw=4, label=f"{f/1000:g} kHz")
                   for c, f in zip(COL, ufreq)],
          fontsize=7.5, loc="center left", bbox_to_anchor=(1.005, 0.5), title="tone")
ax.set(xlabel="time (s)", ylabel="unit (40 most active)", xlim=(t0, t0 + DUR),
       title="Raw spiking during tone presentation; shaded bars are the 25 ms tones")
fig.savefig("fig0_raw_activity.png")
print("saved fig0_raw_activity.png")

# %% [markdown]
# ## 2. When does the tone response happen?
#
# The response window has to be chosen from the data rather than assumed. Averaging the
# firing rate of every unit over every tone gives the population response: activity rises
# about 8 ms after tone onset, peaks near 20 ms, and is back to baseline by roughly
# 100 ms. All later analysis counts spikes in a 10-60 ms window and subtracts a
# 100-10 ms pre-tone baseline.

# %%
if not os.path.exists("results_000986.npz"):
    import run_000986
    run_000986.main()                      # ~30 s: all 15 sessions
D = np.load("results_000986.npz", allow_pickle=True)

import make_figures_000986 as F
lat = F.fig_response_window()
print(f"median peak latency: {np.median(lat[D['p_driven'] < 0.01]):.0f} ms")

# %% [markdown]
# ![](fig1_response_window.png)
#
# The left panel is the grand average over all 1,564 units; the middle panel shows every
# unit separately, z-scored against its own pre-tone baseline and sorted by peak latency;
# the right panel is the distribution of peak latencies. This is the standard cortical
# tone response: short latency, transient, over within about 100 ms.

# %% [markdown]
# ## 3. Single units prefer particular frequencies
#
# For each unit and each of the five tones, the evoked rate is the firing rate in the
# response window minus the rate in the pre-tone baseline, averaged over the ~1,500
# repeats of that tone. A one-way ANOVA across the five tones, on the per-trial evoked
# rates, tests whether the unit responds differently to different frequencies.

# %%
res = session_tuning(spikes, onsets, freq, EVOKED, BASE)
print(f"{SESSION}: {(res['p_driven'] < 0.01).sum()}/{len(spikes)} units sound-driven, "
      f"{(res['p_anova'] < 0.01).sum()}/{len(spikes)} frequency-tuned (ANOVA p < 0.01)")

examples = F.pick_examples(SESSION)
F.fig_example_units(SESSION, examples)
for i in examples:
    print(f"unit {D['unit_ids'][i]:3d}  BF {khz(D['bf'][i]):>2s} kHz  "
          f"evoked rate at BF {D['evoked_rate'][i].max():6.1f} Hz  "
          f"ANOVA {F.pfmt(D['p_anova'][i])}")

# %% [markdown]
# ![](fig2_example_units.png)
#
# Each row is one unit. The raster (left) shows 60 trials per tone, grouped and coloured
# by frequency: the response is visibly confined to a band of frequencies. The PSTHs
# (middle) and the tuning curves (right) show the same thing quantitatively. These units
# respond several tens of spikes per second above baseline to their preferred tone and
# essentially not at all to a tone two octaves away, even though every tone was played at
# the same 60 dB level.

# %% [markdown]
# ## 4. The population covers the frequency range
#
# Pooling the 15 sessions gives 1,564 units, of which 1,312 are driven by tones
# (Wilcoxon, response vs baseline, p < 0.01) and 1,239 are frequency-tuned
# (ANOVA p < 0.01). Sorting their normalized tuning curves by best frequency produces the
# diagonal band below: different units peak at different frequencies, and all five tones
# are the preferred tone of a substantial group of units.

# %%
F.fig_population_tuning()
F.fig_bf_by_session()

tuned = D["p_anova"] < 0.01
for f in D["ufreq"]:
    n = int((D["bf"][tuned] == f).sum())
    print(f"BF {khz(f):>2s} kHz: {n:4d} units ({n/tuned.sum():.1%})")

# %% [markdown]
# ![](fig3_population_tuning.png)
#
# ![](fig4_bf_by_session.png)
#
# The heatmap is restricted to the 1,093 tuned units whose tuning curve has a positive
# peak, since a curve that is negative everywhere cannot be normalized to its peak. The
# remaining 146 tuned units are ones whose rate is *suppressed* by tones, more for some
# frequencies than others; they are frequency-selective too, but the sign of their
# response is inverted.
#
# The per-session panel on the left of the second figure shows that the mix of best
# frequencies differs between recordings. That is expected: auditory cortex is
# tonotopically organized, so where a probe lands along the tonotopic axis sets which
# frequencies dominate the sample. The dataset does not include probe coordinates or
# channel positions in the units table, so tonotopy cannot be mapped directly here; the
# session-to-session variability is the indirect signature of it.

# %% [markdown]
# ## 5. Is the tuning real? Split-half validation
#
# Best frequency is defined as the peak of a measured curve, so it is guaranteed to exist
# even for a unit with no tuning at all: noise has a maximum too. The test is whether the
# peak is in the same place on trials that were not used to find it. Trials are split at
# random into two halves, best frequency is estimated on the first half, and the tuning
# curve is read out on the second.
#
# The same section also asks how much information the population carries about the tone.
# A Poisson naive-Bayes decoder is trained on single-trial population spike counts in the
# response window and asked which of the five tones was played on held-out trials.

# %%
if not os.path.exists("results_crossval_000986.npz"):
    import run_crossval_decoding
    run_crossval_decoding.main()           # ~20 s
C = np.load("results_crossval_000986.npz", allow_pickle=True)

import make_figures_crossval as X
X.main()
print(f"best frequency identical across the two trial halves: "
      f"{np.mean(C['bf_a'][tuned] == C['bf_b'][tuned]):.1%} of tuned units "
      f"(chance {1/len(D['ufreq']):.0%})")
print(f"single-trial decoding accuracy: {C['acc'].mean():.1%} "
      f"± {C['acc'].std():.1%} s.d. across sessions (chance 20%)")

# %% [markdown]
# ![](fig5_crossvalidation_decoding.png)
#
# The split-half tuning curve (top left) still peaks sharply at the best frequency even
# though the peak was located on independent trials, and it falls to about a fifth of its
# peak one octave away. Best frequency is identical across the two halves for 86% of
# tuned units, against a 20% chance level. Single tones are decoded from one 50 ms window
# of population activity with 81% accuracy on average, and accuracy grows steadily with
# the number of units included, from just above chance for a single unit to over 90% for
# 200.

# %% [markdown]
# ## 6. An encoding model: frequency-specific stimulus filters
#
# The analysis so far counts spikes in a fixed window. A Poisson GLM makes the same claim
# in a form that can be scored on held-out data: each unit's binned spike train is
# modelled as a point process whose rate is set by five stimulus filters, one per tone
# frequency, each expanded in a raised-cosine basis (NeMoS). The comparison model is
# identical except that it has a single filter driven by every tone regardless of its
# frequency. If frequency tuning is real, the five-filter model should predict held-out
# spiking better.
#
# Two details matter for the fit. Cross-validation uses interleaved 30 s blocks rather
# than a held-out tail, because firing rates drift over the two hours of recording and a
# tail split scores that drift rather than the model. And JAX is put in float64 mode:
# in the default float32, LBFGS stops short of convergence on this problem.
#
# This section fits the 60 most active units of one session, which is what the run time
# of the population fit scales with; it takes a few minutes.

# %%
if not os.path.exists("results_glm_000986.npz"):
    import run_glm_nemos
    run_glm_nemos.main()                   # slow: fits a population GLM 10 times
G = np.load("results_glm_000986.npz", allow_pickle=True)

import make_figures_glm as GF
GF.main()

r2_full = 1 - G["ll_full"] / G["ll_null"]
r2_blind = 1 - G["ll_blind"] / G["ll_null"]
print(f"held-out pseudo-R2 (median over {len(G['unit_ids'])} units): "
      f"frequency-specific {np.median(r2_full):.4f}, "
      f"frequency-blind {np.median(r2_blind):.4f}")
print(f"frequency-specific model wins for {np.mean(G['ll_full'] > G['ll_blind']):.0%} "
      f"of units")

# %% [markdown]
# ![](fig6_glm_nemos.png)
#
# The fitted filters (left) separate cleanly by frequency: the example units have a large
# positive filter for one or two neighbouring tones and flat or negative filters for the
# rest, with the same 20-40 ms time course seen in the PSTHs. Evaluating each fitted model
# over the same 10-60 ms window used for the spike counts reproduces the measured tuning
# curves closely (r = 0.98 over all fitted units and tones), including the units whose
# response to non-preferred tones is suppression rather than excitation. The
# frequency-specific model predicts held-out spiking better than the frequency-blind one
# for 75% of units, though the absolute gain is small (median 0.002 bits per spike): a
# stimulus-only model of 10 ms bins of single-unit spiking explains a small fraction of
# the total variability whatever it knows about the stimulus, so the informative quantity
# here is the comparison between the two models rather than either pseudo-$R^2$ on its
# own.

# %% [markdown]
# ## 7. A dataset that was checked and rejected
#
# [DANDI:001419](https://dandiarchive.org/dandiset/001419) records mouse A1 and A2 with
# linear probes while presenting tones at 10 to 18 frequencies in half-octave steps and
# up to four sound levels. That stimulus set resolves tuning curve shape far better than
# the five octave-spaced tones used above, so it was the natural second dataset. Its
# sorted spike times, however, show no tone-locked response at all when aligned to the
# onsets in the file's own trials table, at any sound level, in any of the sessions
# checked, from three different experimenters. Since the alignment could not be
# established from the file contents, the dataset was excluded rather than analyzed with
# a guessed offset.

# %%
if not os.path.exists("figS1_001419_alignment_check.png"):
    import check_001419_alignment
    check_001419_alignment.main()

# %% [markdown]
# ![](figS1_001419_alignment_check.png)

# %% [markdown]
# ## Summary
#
# Across 1,564 units recorded in auditory cortex of five mice, 84% responded to 25 ms
# pure tones and 79% responded differently to different tone frequencies. Individual
# units have sharply peaked tuning curves whose peak location is reproducible on
# held-out trials for 86% of tuned units, best frequencies are spread over the whole
# 2-32 kHz range tested, and a Poisson naive-Bayes decoder reads the identity of a single
# tone out of one 50 ms window of population activity with 81% accuracy. A Poisson GLM
# with frequency-specific stimulus filters predicts held-out spiking better than an
# otherwise identical model that is blind to frequency.
#
# **What this analysis cannot say.** The five tones are spaced a full octave apart, so
# tuning bandwidth is barely resolved: for many units the tuning curve has not fallen to
# half its peak by the neighbouring tone, and a half-width in octaves can only be
# measured for a minority of them. All tones were presented at a single level (60 dB
# SPL), so this is a slice through the frequency response area rather than the area
# itself, and level-dependent effects such as broadening of tuning with increasing sound
# level cannot be seen. Finally, the units table carries no probe geometry, so the
# tonotopic map itself is not accessible in these files.

# %%
