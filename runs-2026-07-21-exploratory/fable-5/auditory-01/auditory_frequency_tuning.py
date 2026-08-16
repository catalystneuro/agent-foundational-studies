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
# **Dataset: DANDI 000986** — *Auditory cortex Neuropixels recordings and pupil diameter
# traces from mice during passive exposure to pure tones* (Jo & McCormick, University of
# Oregon; doi:10.1101/2024.04.04.588209).
#
# Neurons in the auditory system respond preferentially to a restricted range of sound
# frequencies. The frequency that drives a neuron most strongly is its *best frequency*
# (BF), and the orderly arrangement of best frequencies across auditory cortex is the
# basis of tonotopy. This notebook demonstrates frequency tuning directly from raw spike
# times in the archive.
#
# The dataset contains 15 sessions from 5 head-fixed mice. In each session a 25 ms pure
# tone at 60 dB SPL was played roughly every 0.8 s, drawn at random from
# 2, 4, 8, 16 and 32 kHz, for about 7,450 tones per session (~1,500 repeats per
# frequency). Neuropixels units in auditory cortex were spike-sorted by the original
# authors; pupil diameter and running speed were recorded simultaneously.
#
# The analysis proceeds in four steps:
#
# 1. Stream one session and verify every data stream visually.
# 2. Measure tone-evoked responses and build tuning curves for single units.
# 3. Pool all 15 sessions and characterize tuning at the population level, with a
#    split-half control for best-frequency selection bias.
# 4. Confirm the effect with two models: a NeMoS Poisson GLM that asks whether knowing
#    the tone frequency improves prediction of held-out spiking, and a NeMoS multinomial
#    classifier that decodes which tone was played from single-trial population activity.
#
# All data are streamed from the DANDI S3 bucket with `remfile` and a local disk cache;
# no file is downloaded in full. All time-series handling is done with `pynapple`.

# %%
import matplotlib
matplotlib.use("Agg")  # headless

import matplotlib.pyplot as plt
import nemos as nmo
import numpy as np
import pandas as pd
import pynapple as nap
from scipy import stats
from scipy.special import gammaln
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm

import dandi_auditory as da
from analysis_core import analyze_session, sparseness

rng = np.random.default_rng(0)
print("pynapple", nap.__version__, "| nemos", nmo.__version__)

# %% [markdown]
# ## 1. Dataset discovery
#
# The 15 assets of dandiset 000986, one NWB file per session.

# %%
assets = da.list_assets()
assets

# %% [markdown]
# ## 2. Load and inspect one session
#
# The prototype session is `sub-LA11_ses-1`, which has the largest number of sorted units.
# `dandi_auditory.load_session` streams the file and wraps the contents in pynapple
# objects: a `TsGroup` of spike trains, an `IntervalSet` of tone presentations carrying
# the frequency as metadata, and `Tsd` traces for pupil and running speed.

# %%
asset_id = assets.query("subject == 'LA11' and session == '1'")["asset_id"].iloc[0]
s = da.load_session(asset_id, with_behavior=True)
units, freqs, freq = s["units"], s["freqs"], s["frequency"]
onsets = s["trials"].start

print(f"subject {s['subject']}, session {s['session_id']}")
print(f"{len(units)} units, {len(onsets)} tones, frequencies {freqs/1000} kHz, "
      f"{s['tone_duration']*1000:.0f} ms duration")
print(units)

# %% [markdown]
# ### Verify the raw streams
#
# Before any analysis, plot the raw spike times, the tone onsets colour-coded by
# frequency, and the two behavioural traces over a 14 s window.

# %%
fig, axes = plt.subplots(
    4, 1, figsize=(12, 9), sharex=True,
    gridspec_kw={"height_ratios": [3, 1, 1, 0.6], "hspace": 0.15},
)
win = (float(onsets[0]) - 2, float(onsets[0]) + 12)

for i, u in enumerate(list(units.keys())[:60]):
    st = units[u].get(win[0], win[1]).t
    axes[0].plot(st, np.full_like(st, i), "|", color="k", ms=2.5, mew=0.5)
axes[0].set_ylabel("unit #")
axes[0].set_title(f"DANDI 000986 sub-{s['subject']} ses-{s['session_id']}: raw spiking, "
                  "pupil and locomotion (25 ms pure tones, 60 dB SPL)")

for f, c in zip(freqs, da.FREQ_COLORS):
    on = onsets[freq == f]
    on = on[(on > win[0]) & (on < win[1])]
    axes[1].vlines(on, 0, 1, color=c, lw=2, label=f"{f/1000:g} kHz")
axes[1].set_ylim(0, 1); axes[1].set_yticks([]); axes[1].set_ylabel("tone")
axes[1].legend(fontsize=8, loc="center left", bbox_to_anchor=(1.005, 0.5), frameon=False)

pup = s["pupil"].get(win[0], win[1])
axes[2].plot(pup.t, pup.d, color="#AA4499", lw=1)
axes[2].set_ylabel("pupil\n(frac. max)")

spd = s["speed"].get(win[0], win[1])
axes[3].plot(spd.t, spd.d, color="#44AA99", lw=1)
axes[3].set_ylabel("running\n(cm/s)")
axes[3].set_xlabel("time (s)")
axes[3].set_xlim(*win)
fig.savefig("fig01_raw_data_streams.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# Session-level sanity checks: firing-rate distribution, the inter-tone interval, and the
# number of trials per frequency (the five frequencies are balanced by design).

# %%
rates = units.get_info("rate")
isi = np.diff(onsets)
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
axes[0].hist(np.log10(rates + 1e-3), bins=40, color="#4477AA")
axes[0].set_xlabel("log10 firing rate (Hz)"); axes[0].set_ylabel("# units")
axes[0].set_title(f"{len(units)} units")
axes[1].hist(isi[isi < 2], bins=50, color="#4477AA")
axes[1].set_xlabel("inter-tone interval (s)"); axes[1].set_ylabel("# trials")
axes[1].set_title("stimulus timing (%d gaps >2 s excluded)" % np.sum(isi >= 2), fontsize=10)
axes[2].bar([f"{f/1000:g}" for f in freqs], [np.sum(freq == f) for f in freqs],
            color=da.FREQ_COLORS)
axes[2].set_xlabel("tone frequency (kHz)"); axes[2].set_ylabel("# trials")
axes[2].set_title("trials per frequency")
fig.tight_layout()
fig.savefig("fig02_session_quality.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# ## 3. Tone-evoked responses and single-unit tuning curves
#
# For every tone I count spikes in two windows relative to onset: a baseline window of
# -100 to 0 ms and an evoked window of 5 to 55 ms, which brackets the transient onset
# response visible in the PSTHs below. Counting is done with `TsGroup.count` over an
# `IntervalSet` built from the onsets, so each count is exact rather than the product of
# a binning grid that could straddle the onset.
#
# The tuning curve of a unit is its mean baseline-subtracted evoked rate at each of the
# five frequencies. Two tests are applied per unit:
#
# - *sound-responsive*: Wilcoxon signed-rank on evoked vs. baseline counts across all trials
# - *frequency-tuned*: Kruskal-Wallis on the baseline-subtracted evoked rate across the
#   five frequency groups
#
# Both use a threshold of p < 0.01. With ~1,500 repeats per frequency these tests are
# well powered, so the interesting question is not whether tuning is detectable but how
# large and how reliable it is; that is quantified further below.

# %%
base = da.trial_spike_counts(units, onsets, da.BASELINE_WINDOW)
evok = da.trial_spike_counts(units, onsets, da.EVOKED_WINDOW)
base_rate = base / (da.BASELINE_WINDOW[1] - da.BASELINE_WINDOW[0])
evok_rate = evok / (da.EVOKED_WINDOW[1] - da.EVOKED_WINDOW[0])
delta = evok_rate - base_rate

tuning = np.stack([delta[freq == f].mean(axis=0) for f in freqs])
tuning_sem = np.stack([stats.sem(delta[freq == f], axis=0) for f in freqs])

responsive = np.array([
    stats.wilcoxon(evok[:, i], base[:, i])[1] if (evok[:, i] - base[:, i]).any() else 1.0
    for i in range(len(units))]) < 0.01
tuned = np.array([
    stats.kruskal(*[delta[freq == f, i] for f in freqs])[1]
    for i in range(len(units))]) < 0.01
bf_idx = np.argmax(tuning, axis=0)

print(f"mean baseline {base_rate.mean():.2f} Hz, mean evoked {evok_rate.mean():.2f} Hz")
print(f"sound-responsive: {responsive.sum()}/{len(units)}")
print(f"frequency-tuned:  {tuned.sum()}/{len(units)}")

# %%
psth = da.psth_by_frequency(units, onsets, freq, freqs)   # (n_freq, n_bins, n_units)
bins = da.psth_bin_centers()
print("PSTH array", psth.shape)

# %% [markdown]
# ### Example units
#
# One example is shown for each of the five best frequencies: the strongest-modulated
# unit whose peak response is at that frequency. Top row, spike rasters (40 trials per
# frequency, grouped and colour-coded); middle row, PSTHs; bottom row, tuning curves with
# SEM error bars (often smaller than the marker, given ~1,500 repeats).

# %%
score = np.where(responsive & tuned, tuning.max(axis=0) - tuning.min(axis=0), -np.inf)
examples = [np.where((bf_idx == i) & np.isfinite(score))[0][
                np.argmax(score[(bf_idx == i) & np.isfinite(score)])]
            for i in range(len(freqs))
            if np.any((bf_idx == i) & np.isfinite(score))]
unit_ids = np.array(list(units.keys()))

fig, axes = plt.subplots(3, len(examples), figsize=(3.1 * len(examples), 8.5),
                         gridspec_kw={"height_ratios": [1.5, 1, 1], "hspace": 0.45})
for col, ui in enumerate(examples):
    uid = unit_ids[ui]
    ax, y = axes[0, col], 0
    for fi, f in enumerate(freqs):
        pe = nap.compute_perievent(units[uid], nap.Ts(onsets[freq == f][:40]),
                                   window=da.PSTH_WINDOW)
        for k in pe.keys():
            t = pe[k].t
            ax.plot(t, np.full_like(t, y), "|", color=da.FREQ_COLORS[fi], ms=3, mew=0.7)
            y += 1
    ax.axvspan(0, s["tone_duration"], color="0.85", zorder=0)
    ax.set_xlim(*da.PSTH_WINDOW); ax.set_ylim(0, y)
    ax.set_title(f"unit {uid}  (BF {freqs[bf_idx[ui]]/1000:g} kHz)", fontsize=10)
    if col == 0:
        ax.set_ylabel("trial (grouped by freq.)")

    ax = axes[1, col]
    for fi, f in enumerate(freqs):
        ax.plot(bins, psth[fi, :, ui], color=da.FREQ_COLORS[fi], lw=1.2,
                label=f"{f/1000:g} kHz")
    ax.axvspan(0, s["tone_duration"], color="0.85", zorder=0)
    ax.set_xlim(*da.PSTH_WINDOW); ax.set_xlabel("time from tone onset (s)")
    if col == 0:
        ax.set_ylabel("firing rate (Hz)")
    if col == len(examples) - 1:
        ax.legend(fontsize=7, frameon=False, loc="upper right")

    ax = axes[2, col]
    ax.errorbar(freqs / 1000, tuning[:, ui], yerr=tuning_sem[:, ui], marker="o",
                color="k", capsize=3, lw=1.5)
    ax.axhline(0, color="0.6", lw=0.8, ls="--")
    ax.set_xscale("log"); ax.set_xticks(freqs / 1000)
    ax.set_xticklabels([f"{f/1000:g}" for f in freqs])
    ax.set_xlabel("frequency (kHz)")
    if col == 0:
        ax.set_ylabel("evoked rate\n(Hz, baseline-subtracted)")
fig.suptitle(f"Tone-evoked responses, sub-{s['subject']} ses-{s['session_id']}", y=0.94)
fig.savefig("fig03_example_units.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# Different units in the same recording peak at different frequencies, and the tuning
# curves are smooth and monotone away from the peak. That is the phenomenon this notebook
# set out to demonstrate.
#
# ## 4. Population analysis across all 15 sessions
#
# The same pipeline is applied to every session and the units are pooled. This takes a
# few minutes on a cold cache. Each session also records a split-half version of the
# analysis: the best frequency is chosen from odd-numbered trials and the tuning curve is
# measured on even-numbered trials, which removes the selection bias that would otherwise
# manufacture a peak in the population average even from noise.

# %%
per_session = []
for _, row in tqdm(list(assets.iterrows()), desc="sessions"):
    r = analyze_session(row["asset_id"])
    r["path"] = row["path"]
    per_session.append(r)

def cat(key, axis=-1):
    return np.concatenate([r[key] for r in per_session], axis=axis)

pop = dict(
    freqs=per_session[0]["freqs"],
    tuning=cat("tuning"), tuning_even=cat("tuning_even"), psth=cat("psth"),
    responsive_p=cat("responsive_p", 0), tuned_p=cat("tuned_p", 0),
    bf=cat("bf", 0), bf_odd=cat("bf_odd", 0), bf_even=cat("bf_even", 0),
    baseline=cat("baseline", 0), evoked=cat("evoked", 0), sparse=cat("sparseness", 0),
    subject=np.concatenate([[r["subject"]] * r["n_units"] for r in per_session]),
)
summary = pd.DataFrame([
    {"path": r["path"], "subject": r["subject"], "n_units": r["n_units"],
     "n_responsive": int(np.sum(r["responsive_p"] < 0.01)),
     "n_tuned": int(np.sum(r["tuned_p"] < 0.01))}
    for r in per_session])
summary.to_csv("session_summary.csv", index=False)
summary

# %%
freqs = pop["freqs"]; n_freq = len(freqs)
pop_responsive = pop["responsive_p"] < 0.01
pop_tuned = pop["tuned_p"] < 0.01
sel = pop_responsive & pop_tuned
bf, bf_odd, bf_even = pop["bf"], pop["bf_odd"], pop["bf_even"]
n_units_total = pop["tuning"].shape[1]
print(f"pooled: {n_units_total} units from {len(set(pop['subject']))} mice; "
      f"{pop_responsive.sum()} sound-responsive, {pop_tuned.sum()} frequency-tuned")

# %%
fig = plt.figure(figsize=(14.5, 8))
gs = fig.add_gridspec(2, 3, hspace=0.5, wspace=0.55, width_ratios=[1.15, 1, 1])
oct_axis = np.log2(freqs / freqs[0])

ax = fig.add_subplot(gs[0, 0])
T = pop["tuning"][:, sel].T
T = T / (np.abs(T).max(axis=1, keepdims=True) + 1e-12)
order = np.lexsort((-T.max(axis=1), bf[sel]))
im = ax.imshow(T[order], aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1,
               interpolation="nearest", extent=[-0.5, n_freq - 0.5, len(order), 0])
ax.set_xticks(range(n_freq)); ax.set_xticklabels([f"{f/1000:g}" for f in freqs])
ax.set_xlabel("tone frequency (kHz)"); ax.set_ylabel("unit (sorted by best frequency)")
ax.set_title("(a) Peak-normalized tuning,\n%d tuned units" % len(order), fontsize=10)
fig.colorbar(im, ax=ax, fraction=0.046).set_label("rate / peak", fontsize=8)

ax = fig.add_subplot(gs[0, 1])
for i, f in enumerate(freqs):
    grp = sel & (bf_odd == i)
    if grp.sum() < 5:
        continue
    E = pop["tuning_even"][:, grp]
    E = E / (np.abs(E).max(axis=0, keepdims=True) + 1e-12)
    m, e = E.mean(axis=1), stats.sem(E, axis=1)
    ax.plot(oct_axis, m, color=da.FREQ_COLORS[i], marker="o", lw=1.8,
            label=f"BF {f/1000:g} kHz (n={grp.sum()})")
    ax.fill_between(oct_axis, m - e, m + e, color=da.FREQ_COLORS[i], alpha=0.25)
ax.set_xticks(oct_axis); ax.set_xticklabels([f"{f/1000:g}" for f in freqs])
ax.set_xlabel("tone frequency (kHz)"); ax.set_ylabel("normalized evoked rate")
ax.set_title("(b) Cross-validated tuning by BF group", fontsize=10)
ax.legend(fontsize=7, frameon=False)

ax = fig.add_subplot(gs[0, 2])
subs = sorted(set(pop["subject"][sel]))
width = 0.8 / len(subs)
for j, sb in enumerate(subs):
    grp = sel & (pop["subject"] == sb)
    ax.bar(np.arange(n_freq) + j * width - 0.4,
           [np.mean(bf[grp] == i) for i in range(n_freq)], width=width,
           label=f"{sb} (n={grp.sum()})")
ax.set_xticks(range(n_freq)); ax.set_xticklabels([f"{f/1000:g}" for f in freqs])
ax.set_xlabel("best frequency (kHz)"); ax.set_ylabel("fraction of tuned units")
ax.set_title("(c) Best-frequency distribution per mouse", fontsize=10)
ax.set_ylim(0, ax.get_ylim()[1] * 1.6)
ax.legend(fontsize=7, frameon=False, ncol=2, loc="upper center")

ax = fig.add_subplot(gs[1, 0])
idx = np.where(sel)[0]
at_bf = np.stack([pop["psth"][bf[i], :, i] for i in idx])
worst = np.argmin(pop["tuning"], axis=0)
at_worst = np.stack([pop["psth"][worst[i], :, i] for i in idx])
b0 = at_bf[:, bins < 0].mean(axis=1, keepdims=True)
for arr, c, lab in [(at_bf - b0, "#882255", "best frequency"),
                    (at_worst - b0, "#4477AA", "worst frequency")]:
    m, e = arr.mean(0), stats.sem(arr, axis=0)
    ax.plot(bins, m, color=c, lw=2, label=lab)
    ax.fill_between(bins, m - e, m + e, color=c, alpha=0.3)
ax.axvspan(0, 0.025, color="0.9", zorder=0); ax.axhline(0, color="k", lw=0.6)
ax.set_xlabel("time from tone onset (s)"); ax.set_ylabel("evoked rate (Hz)")
ax.set_title("(d) Population PSTH", fontsize=10)
ax.legend(fontsize=8, frameon=False)

ax = fig.add_subplot(gs[1, 1])
sp = pop["sparse"]
ax.hist(sp[sel & np.isfinite(sp)], bins=30, color="#117733", alpha=0.85, label="tuned")
ax.hist(sp[pop_responsive & ~pop_tuned & np.isfinite(sp)], bins=30, color="0.7",
        alpha=0.8, label="untuned")
ax.set_xlabel("lifetime sparseness of evoked rate\nacross the 5 tones")
ax.set_ylabel("# units")
ax.set_title("(e) Tuning selectivity\nmedian = %.2f (tuned units)"
             % np.nanmedian(sp[sel]), fontsize=10)
ax.legend(fontsize=8, frameon=False)

ax = fig.add_subplot(gs[1, 2])
x = np.arange(len(summary))
ax.bar(x, summary["n_units"], color="0.8", label="recorded")
ax.bar(x, summary["n_responsive"], color="#4477AA", label="sound-responsive")
ax.bar(x, summary["n_tuned"], color="#882255", label="frequency-tuned")
ax.set_xticks(x)
ax.set_xticklabels([p.split("/")[1].replace("_behavior.nwb", "").replace("sub-", "")
                    for p in summary["path"]], rotation=90, fontsize=6)
ax.set_ylabel("# units"); ax.set_title("(f) Yield per session", fontsize=10)
ax.legend(fontsize=7, frameon=False)

fig.suptitle("Frequency tuning in mouse auditory cortex, DANDI 000986 "
             f"({len(subs)} mice, {len(summary)} sessions, {n_units_total} units)", y=0.97)
fig.savefig("fig04_population_tuning.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# ### Is the tuning real, or a selection artifact?
#
# The strongest single check is whether a unit's best frequency computed from one half of
# its trials predicts the best frequency computed from the other half. If tuning were
# noise, the two halves would agree only at chance (20% for five frequencies).

# %%
fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))

M = np.zeros((n_freq, n_freq))
for a, b in zip(bf_odd[sel], bf_even[sel]):
    M[a, b] += 1
M = M / M.sum(axis=1, keepdims=True)
im = axes[0].imshow(M, cmap="magma", vmin=0, vmax=1)
axes[0].set_xticks(range(n_freq)); axes[0].set_xticklabels([f"{f/1000:g}" for f in freqs])
axes[0].set_yticks(range(n_freq)); axes[0].set_yticklabels([f"{f/1000:g}" for f in freqs])
axes[0].set_xlabel("BF from even trials (kHz)"); axes[0].set_ylabel("BF from odd trials (kHz)")
agree = np.mean(bf_odd[sel] == bf_even[sel])
axes[0].set_title("BF is stable across trial halves\n%.0f%% agreement (chance %.0f%%)"
                  % (100 * agree, 100 / n_freq), fontsize=10)
fig.colorbar(im, ax=axes[0], fraction=0.046)

axes[1].scatter(pop["baseline"][~sel], pop["evoked"][~sel], s=6, color="0.7", label="other")
axes[1].scatter(pop["baseline"][sel], pop["evoked"][sel], s=6, color="#882255", label="tuned")
lim = [0.05, max(pop["evoked"].max(), pop["baseline"].max()) * 1.2]
axes[1].plot(lim, lim, "k--", lw=1)
axes[1].set_xscale("log"); axes[1].set_yscale("log")
axes[1].set_xlim(*lim); axes[1].set_ylim(*lim)
axes[1].set_xlabel("baseline rate (Hz)"); axes[1].set_ylabel("tone-evoked rate (Hz)")
axes[1].set_title("Tones drive the population", fontsize=10)
axes[1].legend(fontsize=8, frameon=False)

depth = pop["tuning"].max(axis=0) - pop["tuning"].min(axis=0)
axes[2].hist(np.log10(depth[sel] + 1e-3), bins=40, color="#882255", alpha=0.85, label="tuned")
axes[2].hist(np.log10(depth[~pop_tuned] + 1e-3), bins=40, color="0.7", alpha=0.8,
             label="untuned")
axes[2].set_xlabel("log10 best-minus-worst evoked rate (Hz)"); axes[2].set_ylabel("# units")
axes[2].set_title("Depth of frequency modulation\nmedian %.1f Hz (tuned)"
                  % np.median(depth[sel]), fontsize=10)
axes[2].legend(fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig("fig05_tuning_validation.png", dpi=150, bbox_inches="tight")

print("BF split-half agreement %.3f (chance %.3f)" % (agree, 1 / n_freq))

# %% [markdown]
# ## 5. Model-based confirmation with NeMoS
#
# ### Encoding: does tone frequency improve prediction of held-out spiking?
#
# Spikes are binned at 5 ms over the 150 ms following each tone onset, for a random
# subset of 3,000 trials. A Poisson
# population GLM predicts each unit's counts from a design matrix that is the outer
# product of the tone identity (five one-hot columns) with a 10-element log-spaced
# raised-cosine basis over time since onset, giving 50 predictors. The reduced model
# keeps only the 10 time-basis columns, so it can capture the average onset response
# but not its
# frequency dependence. Both are fit on 80% of the trials and compared by
# log-likelihood on the held-out 20%.

# %%
BIN, N_BASIS = 0.005, 10
# The window starts at tone onset so the log-spaced basis puts its finest resolution
# on the first tens of milliseconds, where the transient response lives. The fit uses a
# random 3,000 trials; 600 repeats per frequency already pin down 50 parameters per unit.
GLM_WINDOW = (0.0, 0.150)
GLM_TRIALS = 3000
edges = np.arange(GLM_WINDOW[0], GLM_WINDOW[1] + BIN / 2, BIN)
n_bins, n_trials, n_units = len(edges) - 1, len(onsets), len(units)
t_since_onset = (edges[:-1] + edges[1:]) / 2

glm_trials = np.sort(rng.permutation(n_trials)[:GLM_TRIALS])
glm_onsets, glm_freq = onsets[glm_trials], freq[glm_trials]

counts = np.zeros((len(glm_trials), n_bins, n_units), dtype=np.float32)
for b in range(n_bins):
    ep = nap.IntervalSet(start=glm_onsets + edges[b], end=glm_onsets + edges[b + 1])
    counts[:, b, :] = np.asarray(units.count(ep=ep).values)

basis = nmo.basis.RaisedCosineLogEval(n_basis_funcs=N_BASIS)
B = basis.compute_features(t_since_onset)
onehot = (glm_freq[:, None] == freqs[None, :]).astype(np.float32)
X_full = np.einsum("tf,bk->tbfk", onehot, B).reshape(len(glm_trials) * n_bins, n_freq * N_BASIS)
X_red = np.tile(B, (len(glm_trials), 1))
y = counts.reshape(len(glm_trials) * n_bins, n_units)

is_train = np.zeros(len(glm_trials), bool)
is_train[rng.permutation(len(glm_trials))[: int(0.8 * len(glm_trials))]] = True
bin_train = np.repeat(is_train, n_bins)

def fit_score(X):
    model = nmo.glm.PopulationGLM(
        regularizer="Ridge", regularizer_strength=1e-3, solver_name="LBFGS",
        solver_kwargs={"maxiter": 150, "tol": 1e-7},
    ).fit(X[bin_train], y[bin_train])
    rate = np.asarray(model.predict(X[~bin_train]))
    yt = y[~bin_train]
    return model, (yt * np.log(rate + 1e-12) - rate - gammaln(yt + 1)).mean(axis=0)

model_full, ll_full = fit_score(X_full)
model_red, ll_red = fit_score(X_red)
mu = y[bin_train].mean(axis=0)
ll_null = (y[~bin_train] * np.log(mu + 1e-12) - mu - gammaln(y[~bin_train] + 1)).mean(axis=0)
pr2_full, pr2_red = 1 - ll_full / ll_null, 1 - ll_red / ll_null
d_ll = ll_full - ll_red

coef = np.asarray(model_full.coef_).reshape(n_freq, N_BASIS, n_units)
pred_rate = np.exp(np.einsum("fku,bk->fbu", coef, B)
                   + np.asarray(model_full.intercept_)[None, None, :]) / BIN
print("units where frequency improves held-out likelihood: %d/%d"
      % (np.sum(d_ll > 0), n_units))

# %% [markdown]
# ### Decoding: which tone was played?
#
# A NeMoS multinomial classifier GLM is trained on the single-trial population spike
# counts in the 5-55 ms evoked window, with 5-fold stratified cross-validation. Chance is
# 20%. The same classifier is also refit on random subsets of units to show how decoding
# accuracy grows with population size.

# %%
labels = np.searchsorted(freqs, freq)
Xd = (evok - evok.mean(0)) / (evok.std(0) + 1e-9)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

conf, acc = np.zeros((n_freq, n_freq)), []
for tr, te in cv.split(Xd, labels):
    clf = nmo.glm.ClassifierGLM(n_classes=n_freq, regularizer="Ridge",
                                regularizer_strength=0.01, solver_name="LBFGS")
    clf.fit(Xd[tr], labels[tr])
    pred = np.asarray(clf.predict(Xd[te]))
    acc.append(np.mean(pred == labels[te]))
    for a, b in zip(labels[te], pred):
        conf[a, b] += 1
conf = conf / conf.sum(axis=1, keepdims=True)
print("decoding accuracy %.3f +/- %.3f (chance %.2f)"
      % (np.mean(acc), np.std(acc), 1 / n_freq))

sizes = sorted({int(k) for k in [1, 2, 5, 10, 20, 50, 100, 200, n_units] if k <= n_units})
curve = np.zeros((len(sizes), 5))
tr, te = next(iter(cv.split(Xd, labels)))
for i, k in enumerate(tqdm(sizes, desc="decoder size")):
    for rep in range(5):
        subset = rng.choice(n_units, k, replace=False)
        clf = nmo.glm.ClassifierGLM(n_classes=n_freq, regularizer="Ridge",
                                    regularizer_strength=0.01, solver_name="LBFGS")
        clf.fit(Xd[tr][:, subset], labels[tr])
        curve[i, rep] = np.mean(np.asarray(clf.predict(Xd[te][:, subset])) == labels[te])

# %%
fig = plt.figure(figsize=(14, 7.8))
gs = fig.add_gridspec(2, 3, hspace=0.5, wspace=0.42)

best = int(np.argmax(d_ll))
ax = fig.add_subplot(gs[0, 0])
for fi, f in enumerate(freqs):
    ax.plot(t_since_onset, counts[glm_freq == f, :, best].mean(0) / BIN,
            color=da.FREQ_COLORS[fi], lw=1, alpha=0.45)
    ax.plot(t_since_onset, pred_rate[fi, :, best], color=da.FREQ_COLORS[fi], lw=2,
            label=f"{f/1000:g} kHz")
ax.axvspan(0, s["tone_duration"], color="0.9", zorder=0)
ax.set_xlabel("time from tone onset (s)"); ax.set_ylabel("firing rate (Hz)")
ax.set_title(f"GLM fit, unit {unit_ids[best]}\n(thin = data, thick = model)", fontsize=10)
ax.legend(fontsize=7, frameon=False)

ax = fig.add_subplot(gs[0, 1])
ax.hist(d_ll * 1000, bins=50, color="#4477AA")
ax.axvline(0, color="k", lw=1, ls="--")
ax.set_xlabel("held-out log-likelihood gain\nfrom tone frequency (millinats/bin)")
ax.set_ylabel("# units")
ax.set_title("Frequency improves prediction\nin %d/%d units" % (np.sum(d_ll > 0), n_units),
             fontsize=10)

ax = fig.add_subplot(gs[0, 2])
ax.scatter(pr2_red, pr2_full, s=8, color="#CC6677", alpha=0.7)
lim = [min(pr2_red.min(), pr2_full.min()), max(pr2_red.max(), pr2_full.max())]
ax.plot(lim, lim, "k--", lw=1)
ax.set_xlabel("pseudo-$R^2$, time-only model")
ax.set_ylabel("pseudo-$R^2$, time $\\times$ frequency")
ax.set_title("Held-out goodness of fit", fontsize=10)

ax = fig.add_subplot(gs[1, 0])
im = ax.imshow(conf, cmap="magma", vmin=0, vmax=conf.max())
ax.set_xticks(range(n_freq)); ax.set_xticklabels([f"{f/1000:g}" for f in freqs])
ax.set_yticks(range(n_freq)); ax.set_yticklabels([f"{f/1000:g}" for f in freqs])
ax.set_xlabel("decoded frequency (kHz)"); ax.set_ylabel("true frequency (kHz)")
ax.set_title("Single-trial decoding\naccuracy %.1f%% (chance %.0f%%)"
             % (100 * np.mean(acc), 100 / n_freq), fontsize=10)
fig.colorbar(im, ax=ax, fraction=0.046).set_label("P(decoded | true)", fontsize=8)

ax = fig.add_subplot(gs[1, 1])
ax.errorbar(sizes, curve.mean(1) * 100, yerr=curve.std(1) * 100, marker="o", color="k",
            capsize=3)
ax.axhline(100 / n_freq, color="0.5", ls="--", lw=1, label="chance")
ax.set_xscale("log"); ax.set_xlabel("# units in decoder"); ax.set_ylabel("accuracy (%)")
ax.set_title("Decoding scales with population size", fontsize=10)
ax.legend(fontsize=8, frameon=False)

ax = fig.add_subplot(gs[1, 2])
top = np.argsort(d_ll)[::-1][:30]
ax.barh(np.arange(len(top)), d_ll[top] * 1000, color="#117733")
ax.set_yticks([]); ax.invert_yaxis()
ax.set_xlabel("log-likelihood gain (millinats/bin)")
ax.set_title("30 most frequency-dependent units", fontsize=10)

fig.suptitle("GLM encoding and decoding of tone frequency, "
             f"sub-{s['subject']} ses-{s['session_id']}", y=0.98)
fig.savefig("fig06_glm_decoding.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# ## 6. Summary
#
# Every step points the same way. Individual units in mouse auditory cortex respond to a
# 25 ms pure tone with a sharp onset burst whose size depends on the tone's frequency,
# and different units in the same penetration prefer different frequencies, spanning the
# 2-32 kHz range tested. A unit's best frequency estimated from odd-numbered trials
# predicts its best frequency on even-numbered trials far above chance, so the tuning is
# a stable property of the neuron rather than a peak selected out of noise. A Poisson GLM
# that knows the tone frequency predicts held-out spiking better than one that knows only
# that a tone occurred, and a multinomial classifier reads the identity of the tone off
# the single-trial population response well above the 20% chance level, with accuracy
# that keeps climbing as more units are added to the decoder.
#
# The main limitation is the stimulus set: only five frequencies, spaced one octave
# apart, at a single 60 dB level. That is enough to establish that units are frequency
# selective and to locate a best frequency to within an octave, but it cannot resolve
# tuning bandwidth or the level dependence of a full frequency-response area. The
# dataset also does not include electrode depth or channel position for the sorted units,
# so the tonotopic gradient along the probe cannot be reconstructed here.

# %%
print("figures written:")
for f in ["fig01_raw_data_streams.png", "fig02_session_quality.png",
          "fig03_example_units.png", "fig04_population_tuning.png",
          "fig05_tuning_validation.png", "fig06_glm_decoding.png"]:
    print(" ", f)
