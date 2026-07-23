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
# # Auditory frequency tuning, from the auditory nerve to the cortex
#
# Neurons in the auditory system are tuned to sound frequency. This notebook demonstrates
# that on real recordings from the DANDI Archive at two stages of the pathway:
#
# * **DANDI:001262** - single auditory-nerve fibres in Mongolian gerbils
#   ([Heeringa et al., Sci Data 2024](https://doi.org/10.1038/s41597-024-03259-3)). Each
#   fibre was probed with a frequency x level grid of 50 ms tone bursts, which yields the
#   classical frequency response area and V-shaped threshold tuning curve.
# * **DANDI:000986** - Neuropixels recordings from mouse auditory cortex during passive
#   presentation of 25 ms pure tones at 2, 4, 8, 16 and 32 kHz
#   ([preprint](https://doi.org/10.1101/2024.04.04.588209)). Roughly 1500 repeats of each
#   frequency per session give well-constrained single-unit tuning curves.
#
# Everything is streamed from the DANDI S3 bucket with `remfile` (local disk cache); no
# file is downloaded in full. Analysis uses `pynapple` for the time-series handling and
# `nemos` for the encoding model.
#
# ### A dataset we rejected
#
# The obvious first choice was **DANDI:001419** (mouse A1/A2 linear-probe recordings with
# 10 half-octave frequencies from 4-80 kHz). Its `units/spike_times` are integer-valued,
# i.e. rounded to whole seconds, so nothing can be aligned to a 300 ms tone. The same is
# true of its companion dandisets 001420 and 001421. That check is reproduced at the end
# of this notebook.

# %%
import pickle
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap
from tqdm.auto import tqdm

import anf_analysis as anf
import cortex_analysis as cx
import dandi_io as dio
import glm_analysis as glm
import make_figures as mf

nap.nap_config.suppress_conversion_warnings = True
os.makedirs("figures", exist_ok=True)
print("pynapple", nap.__version__)

# %% [markdown]
# ## Part 1 - The auditory nerve
#
# Each NWB file in DANDI:001262 holds one fibre. Its `units` table has one row per
# stimulus sweep, tagged with the stimulus name. Two tag families matter here:
# `CF_FREQ<f>_ABI<level>_rep<n>` sweeps make up a frequency x level grid (present in a
# minority of fibres) and `BF_FREQ<f>_rep<n>` sweeps are a fixed-level frequency sweep
# (present in most). Sweeps are 200 ms long and spike times are given relative to sweep
# onset, so we lay the sweeps end to end on a common pynapple timeline.

# %%
fibre_files = dio.list_assets("001262", "0.241205.0959")
print(f"{len(fibre_files)} fibre files in DANDI:001262")

fib = dio.load_fibre_001262(fibre_files[0][1])
print(f"fibre {fib['fibre']}  subject {fib['subject']}  age {fib['age_days']} days")
print(f"{len(fib['sweeps'])} tone sweeps, "
      f"{len(np.unique(fib['frequency']))} frequencies x {len(np.unique(fib['level']))} levels")
print("frequencies (kHz):", np.unique(fib["frequency"]) / 1e3)
print("levels (dB SPL):", np.unique(fib["level"]))

# %% [markdown]
# ### Where is the tone inside the sweep?
#
# The file does not document the tone timing, so we read it off the data. The PSTH shows
# spontaneous firing for the first 10 ms, a sharp onset response, adaptation to a lower
# sustained rate, tone offset near 60 ms, a period of post-stimulus suppression below the
# spontaneous rate, and then recovery. We therefore measure the driven rate over
# 10-60 ms and the spontaneous rate over the last 70 ms of each sweep.

# %%
print(mf.fig_anf_sweep(fib))

# %% [markdown]
# ### Frequency response area and threshold tuning curve
#
# For every cell of the frequency x level grid we take the mean driven rate. The
# threshold at each frequency is the lowest level whose response exceeds the spontaneous
# rate by 20 spikes/s (and stays above it at all higher levels). The tip of the resulting
# curve is the characteristic frequency (CF); its width 10 dB above the tip gives
# Q10 = CF / bandwidth, the standard measure of tuning sharpness.

# %%
ra = anf.response_area(fib)
tc = anf.threshold_curve(ra)
print(f"spontaneous rate {ra['spont']:.1f} spikes/s")
print("thresholds (dB SPL):", tc["thresholds"])
print(f"CF {tc['cf']:.0f} Hz, threshold {tc['cf_threshold']:.0f} dB SPL, Q10 {tc['q10']:.2f}")

# %% [markdown]
# ### Across fibres
#
# Repeating this for a sample of fibres spread across the archive gives a population of
# threshold curves. Re-running the batch takes roughly 15 minutes of streaming; the cached
# result is reused if present.

# %%
if os.path.exists("results_anf.pkl"):
    anf_res = pickle.load(open("results_anf.pkl", "rb"))
else:
    n_fibres = 400
    sel = fibre_files[:: max(1, len(fibre_files) // n_fibres)][:n_fibres]
    anf_res = []
    for path, url in tqdm(sel, desc="fibres"):
        r = anf.analyze_fibre(url)
        if r is not None:
            r["path"] = path
            anf_res.append(r)
    pickle.dump(anf_res, open("results_anf.pkl", "wb"))

anf_grid = [r for r in anf_res if np.isfinite(r.get("cf", np.nan))]
anf_q = [r["q10"] for r in anf_grid if np.isfinite(r["q10"])]
anf_iso = [r for r in anf_res if np.isfinite(r.get("iso_bf", np.nan))]
anf_bw = [r["iso_bw_octaves"] for r in anf_iso if np.isfinite(r["iso_bw_octaves"])]
print(f"{len(anf_res)} fibres analysed")
print(f"  {len(anf_grid)} with a frequency x level grid and a measurable threshold curve; "
      f"CF {min(r['cf'] for r in anf_grid)/1e3:.2f} - "
      f"{max(r['cf'] for r in anf_grid)/1e3:.1f} kHz, "
      f"Q10 median {np.median(anf_q):.2f} (n = {len(anf_q)})")
print(f"  {len(anf_iso)} with fixed-level frequency sweeps; "
      f"{sum(r['iso_p'] < 0.01 for r in anf_iso)} frequency-selective at p < 0.01")
print(f"  half-max tuning width: median {np.median(anf_bw):.2f} octaves "
      f"(n = {len(anf_bw)} measurable)")

# %%
print(mf.fig_anf_examples(anf_res))
print(mf.fig_anf_iso_examples(anf_res))
print(mf.fig_anf_population(anf_res))

# %% [markdown]
# Every fibre responds only to a restricted band of frequencies, and the band narrows
# (Q10 grows) as CF rises, which is the signature of cochlear filtering. The tips of the
# threshold curves trace out the animal's audiogram: fibres with CFs in the gerbil's most
# sensitive range have the lowest thresholds.
#
# Both the Q10 and the half-maximum width are computed from median-filtered curves. The
# frequency grid is much finer than a fibre's bandwidth, so on an unsmoothed curve a
# single noisy point next to the peak can truncate the width to an arbitrarily small
# value; smoothing removes that failure mode, and fibres whose curve never falls to the
# criterion inside the tested range are reported as not measurable rather than assigned a
# width.

# %% [markdown]
# ## Part 2 - Auditory cortex
#
# DANDI:000986 presents 25 ms pure tones at five octave-spaced frequencies, 60 dB SPL,
# about every 800 ms, with blocks of silence interleaved. We start with one session.

# %%
cortex_files = dio.list_assets("000986")
print(f"{len(cortex_files)} sessions in DANDI:000986")
sess = dio.load_tone_session_000986([u for p, u in cortex_files if "LA9_ses-1" in p][0])
spikes, evoked_ep, base_ep = cx.session_to_pynapple(sess)
print(f"subject {sess['subject']} session {sess['session']}: {len(spikes)} units, "
      f"{len(sess['tone_onset'])} tones")
print("frequencies (kHz):", np.unique(sess["frequency"]) / 1e3,
      " level (dB SPL):", np.unique(sess["intensity"]))

# %% [markdown]
# ### Raw data
#
# Spike raster, population rate, pupil diameter and running speed over eight seconds, with
# the tones shaded by frequency. This is a check that the spike times, the trial table and
# the behavioural traces all live on the same clock.

# %%
print(mf.fig_raw_cortex(sess, spikes))

# %% [markdown]
# ### Population PSTH
#
# Averaged over all units and all tones, the cortical response begins about 12 ms after
# tone onset and peaks near 20 ms. We measure the evoked rate over 10-60 ms and the
# baseline over the 100 ms before tone onset.

# %%
t_psth, rates = cx.population_psth(spikes, sess["tone_onset"])
fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(t_psth * 1e3, rates.mean(0), color="C3")
ax.axvspan(cx.EVOKED[0] * 1e3, cx.EVOKED[1] * 1e3, color="C1", alpha=0.15, lw=0,
           label="evoked window")
ax.axvspan(cx.BASELINE[0] * 1e3, cx.BASELINE[1] * 1e3, color="C0", alpha=0.15, lw=0,
           label="baseline")
ax.set_xlabel("time from tone onset (ms)")
ax.set_ylabel("mean rate (spikes/s)")
ax.set_title(f"population PSTH, {len(spikes)} units x {len(sess['tone_onset'])} tones")
ax.legend(fontsize=8)
fig.savefig("figures/fig_extra_population_psth.png", dpi=130, bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# ### Single-unit tuning
#
# For each unit we compare the evoked rate across the five frequencies. Two tests are run:
# a Wilcoxon signed-rank test of evoked against baseline (is the unit driven by tones at
# all?) and a permutation test on the between-frequency variance of the mean evoked rate
# (does the response depend on which frequency was played?).

# %%
units, freqs, spikes, _ = cx.analyze_session(sess, rng=np.random.default_rng(1))
n_resp = sum(u["p_responsive"] < 0.01 for u in units)
n_tuned = sum(u["p_tuned"] < 0.01 for u in units)
print(f"{len(units)} units: {n_resp} tone-responsive, {n_tuned} frequency-tuned (p < 0.01)")

best = max(units, key=lambda u: u["peak_driven"])
print(mf.fig_cortex_example_unit(sess, spikes, best["unit"]))

# %% [markdown]
# ### All sessions
#
# The same pipeline is run over all 15 sessions (5 mice), together with a cross-validated
# Bayesian decoder that reads the tone frequency out of 50 ms of population activity.

# %%
if os.path.exists("results_cortex.pkl"):
    cx_res = pickle.load(open("results_cortex.pkl", "rb"))
else:
    all_units, per_session = [], []
    for path, url in tqdm(cortex_files, desc="sessions"):
        s = dio.load_tone_session_000986(url)
        u_list, fqs, sp, _ = cx.analyze_session(s, rng=np.random.default_rng(1))
        conf, acc, _ = cx.decode_frequency(sp, s, rng=np.random.default_rng(2))
        _, acc_shuf, _ = cx.decode_frequency(sp, s, rng=np.random.default_rng(3), shuffle=True)
        for u in u_list:
            u["path"] = path
        all_units += u_list
        per_session.append(dict(path=path, session=s["session"], subject=s["subject"],
                                n_units=len(u_list), n_trials=len(s["tone_onset"]),
                                confusion=conf, accuracy=acc, accuracy_shuffled=acc_shuf,
                                freqs=fqs))
    cx_res = dict(units=all_units, sessions=per_session, freqs=per_session[0]["freqs"])
    pickle.dump(cx_res, open("results_cortex.pkl", "wb"))

n_tuned = sum(u["p_tuned"] < 0.01 for u in cx_res["units"])
print(f"{len(cx_res['units'])} units in {len(cx_res['sessions'])} sessions, "
      f"{n_tuned} ({100*n_tuned/len(cx_res['units']):.0f}%) frequency-tuned at p < 0.01")
print("decoding accuracy %.3f, shuffled control %.3f, chance %.3f" % (
    np.mean([s["accuracy"] for s in cx_res["sessions"]]),
    np.mean([s["accuracy_shuffled"] for s in cx_res["sessions"]]),
    1 / len(cx_res["freqs"])))

# %%
print(mf.fig_cortex_tuning_examples(cx_res["units"], cx_res["freqs"]))
print(mf.fig_cortex_population(cx_res))
print(mf.fig_decoding(cx_res))

# %% [markdown]
# Cortical units are tuned but broadly so: most respond to several of the five
# frequencies, with a clear preference. Sorting the normalised tuning curves by best
# frequency produces the diagonal band that is the population signature of frequency
# tuning. Because the tones were all presented at one level (60 dB SPL), the best
# frequency measured here is an iso-level preference and mixes the unit's tuning with the
# mouse's audiogram; it is not the same quantity as the CF of an auditory-nerve fibre,
# which is measured at threshold.
#
# One feature deserves scepticism: a substantial group of units responds strongly at
# 2 kHz, where mice are relatively insensitive. A 25 ms tone contains broadband onset
# energy, so part of that response is probably driven by the onset transient rather than
# by 2 kHz itself. It does not affect the main claim, since the frequency dependence is
# what is being demonstrated, but it does mean the low-frequency end of the best-frequency
# distribution should not be read as a population of genuine 2 kHz-tuned neurons.

# %% [markdown]
# ## Part 3 - A GLM encoding model
#
# Spike counts in a fixed window ignore both the time course of the response and the
# unit's own spiking dynamics. We fit a Poisson GLM (NeMoS) in which the spike train is
# driven by five frequency-specific tone kernels plus a spike-history filter, and compare
# the area under each tone kernel with the empirical tuning curve.

# %%
picks = sorted([u for u in cx_res["units"]
                if u["subject"] == sess["subject"] and u["session"] == sess["session"]],
               key=lambda u: -u["peak_driven"])[:3]
glm_fits, _ = glm.fit_session(sess, spikes, [u["unit"] for u in picks])
for f in glm_fits:
    print(f"unit {f['unit']}: held-out pseudo-R2 {f['r2_test']:.3f}, "
          f"kernel areas {np.round(f['gain'], 2)}")
print(mf.fig_glm(glm_fits, cx_res["freqs"], {u["unit"]: u["driven"] for u in picks}))

# %% [markdown]
# The kernels differ in amplitude and time course across frequencies, which is frequency
# tuning expressed as an encoding filter rather than as a spike count. The history filters
# are positive over the first tens of milliseconds, so these units fire in bursts; that is
# exactly the effect a raw spike count cannot separate from stimulus drive. The
# model-based and count-based tuning curves agree on the shape but not always on which
# arm is highest, because the GLM assigns part of the count to the history term.

# %% [markdown]
# ## Part 4 - Tuning width at the two stages, and why they cannot be compared
#
# Tuning width in octaves is the one measure both datasets nominally provide, and it is
# tempting to conclude from it that cortical tuning is broader than peripheral tuning.
# That conclusion is not supported here, because each estimate is biased by its own
# measurement limit and the two biases run in opposite directions:
#
# * The cortical tuning curve is sampled at five octave-spaced frequencies, so any width
#   below one octave is unresolvable; the estimate is quantised and floored at 0.5
#   octaves. It can only be biased downwards.
# * The nerve width is only measurable for fibres whose tuning curve fell to half maximum
#   on both sides inside the frequency range the experimenter chose to test. Broadly
#   tuned fibres are therefore systematically missing from the sample, which also biases
#   the distribution downwards.
#
# The two medians come out nearly identical, and the fraction of units wider than one
# octave is close as well, but neither number should be read as a comparison. The figure
# is shown as a description of each dataset, not as a contrast between them. Establishing
# the textbook result that tuning broadens from periphery to cortex would need the same
# stimulus set at both stages.

# %%
path, med_anf, frac_anf, frac_cx = mf.fig_bandwidth_comparison(anf_res, cx_res)
print(path)
print(f"auditory nerve: median width {med_anf:.2f} octaves, "
      f"{frac_anf:.0%} at least one octave wide")
print(f"auditory cortex: {frac_cx:.0%} at least one octave wide "
      f"(cannot resolve anything narrower)")

# %% [markdown]
# ## Appendix - the rejected dandiset
#
# DANDI:001419 looked ideal (10 half-octave frequencies, laminar probes in A1 and A2) but
# its spike times are rounded to whole seconds.

# %%
for ds, ver in [("001419", "0.250521.2343"), ("001420", "0.250521.2343"),
                ("001421", "0.250521.2343")]:
    path, url = dio.list_assets(ds, ver)[0]
    f = dio.open_stream(url)
    st = f["units/spike_times"][:20000]
    frac_int = float(np.mean(np.abs(st - np.round(st)) < 1e-9))
    print(f"DANDI:{ds}  {path.split('/')[-1]}: "
          f"{frac_int:.0%} of spike times are whole seconds, "
          f"smallest non-zero interval {np.diff(np.unique(st)).min():.3f} s")

# %% [markdown]
# ## Summary
#
# Frequency tuning is visible at both stages and in every analysis we ran. Single
# auditory-nerve fibres have sharp V-shaped threshold curves whose tips define a
# characteristic frequency, and sharpness grows with CF. Cortical units are broadly but
# reliably tuned: most are significantly frequency-selective, their best frequencies span
# the tested range, and a decoder recovers which of five frequencies was played from 50 ms
# of population activity far above chance. A GLM with frequency-specific kernels
# reproduces the same preferences while accounting for spike-history effects.
