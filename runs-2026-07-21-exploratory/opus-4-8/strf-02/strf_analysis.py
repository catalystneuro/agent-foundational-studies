# %% [markdown]
# # Spectrotemporal Receptive Fields in Mouse Auditory Cortex
#
# This notebook demonstrates **spectrotemporal receptive fields (STRFs)** of
# single neurons in mouse primary auditory cortex, using Neuropixels recordings
# from the DANDI Archive.
#
# **Dataset:** [DANDI:000986](https://dandiarchive.org/dandiset/000986) —
# *Auditory cortex Neuropixels recordings and pupil diameter traces from mice
# during passive exposure to pure tones.* Head-fixed mice passively heard brief
# (25 ms, 60 dB) pure tones at five octave-spaced frequencies (2, 4, 8, 16,
# 32 kHz), randomly interleaved roughly every 800 ms, while sorted single-unit
# spike times were recorded from auditory cortex.
#
# **The STRF.** A neuron's spectrotemporal receptive field is the stimulus
# spectrogram, as a function of sound frequency and time, that on average
# precedes (drives) its spikes. Because the stimulus here is a random sequence
# of discrete tones, the STRF is obtained by **reverse correlation**: for each
# frequency we average the neuron's peri-onset firing-rate histogram across all
# presentations of that frequency. The resulting 2-D map (frequency × time lag)
# is the tone-triggered average, i.e. the STRF. A well-formed auditory STRF
# shows a frequency-tuned excitatory field appearing a short latency after tone
# onset, often flanked by inhibitory (suppressive) sidebands.
#
# **What we show.** (1) The raw stimulus and single-unit responses; (2) example
# single-neuron STRFs; (3) a detailed decomposition of one STRF into its
# frequency-tuning and temporal marginals; (4) population statistics of best
# frequency, onset latency and responsiveness across five mice; and (5) the
# best-frequency-aligned population-average STRF.

# %% [markdown]
# ## Setup

# %%
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pynapple as nap

from strf_utils import (
    load_session, get_units_tsgroup, get_trials,
    compute_strf, strf_metrics, FREQS,
)

# One representative session per mouse (five subjects), streamed from S3.
SESSIONS = {
    "LA11": "https://dandiarchive.s3.amazonaws.com/blobs/404/0c6/4040c62d-0c6e-4b94-9dda-19ebb36bcdf4",
    "LA12": "https://dandiarchive.s3.amazonaws.com/blobs/bfd/661/bfd66119-8cd8-4ae7-863a-da43bafdcc58",
    "LA3":  "https://dandiarchive.s3.amazonaws.com/blobs/cac/52e/cac52ee7-20d9-4f7d-a234-a04c22a94083",
    "LA8":  "https://dandiarchive.s3.amazonaws.com/blobs/ea8/b2d/ea8b2d92-31d0-45ab-a300-5e844b1f2a57",
    "LA9":  "https://dandiarchive.s3.amazonaws.com/blobs/cda/b38/cdab388d-11fd-4a5a-b1cf-1e686234c777",
}

# STRF sampling: lag window and bin edges (2.5 ms bins from -50 to +200 ms).
LAGS = np.arange(-0.050, 0.2001, 0.0025)
LAGC = 0.5 * (LAGS[:-1] + LAGS[1:])          # bin centers (s)
FREQ_KHZ = FREQS / 1000
RESPONSIVE_HZ = 2.0                            # min peak driven rate to count a unit
FREQ_LABELS = [f"{int(f)}" for f in FREQ_KHZ]

# %% [markdown]
# ## Load the prototype session and inspect the raw data
#
# We first load one session (mouse LA11) to verify the stimulus structure and
# the neural responses before computing STRFs.

# %%
nwbf, io = load_session(SESSIONS["LA11"])
units = get_units_tsgroup(nwbf)
onsets, freqs = get_trials(nwbf)

print(f"Session LA11: {len(units)} sorted units, {len(onsets)} tone presentations")
print(f"Frequencies (Hz): {np.unique(freqs).astype(int)}")
print(f"Recording span: {onsets.min():.0f}-{onsets.max():.0f} s")
print(f"Median unit firing rate: {np.median(units.rates):.2f} Hz")

# %% [markdown]
# ### Figure 1 — Raw data validation
#
# Left: the randomly interleaved tone sequence (a 30 s excerpt), colored by
# frequency. Right: a tone-onset raster for one strongly driven unit, with
# trials grouped by stimulus frequency; the peri-onset firing-rate histograms
# below show a clean, frequency-tuned, short-latency response — the raw
# ingredients of the STRF.

# %%
# pick a strongly responsive unit for the raster
peaks_proto = []
for k in units.keys():
    s = compute_strf(units[k].t, onsets, freqs, LAGS)
    m = strf_metrics(s, LAGC)
    peaks_proto.append((m["peak_driven"], k, m["best_freq"]))
peaks_proto.sort(reverse=True)
example_unit = peaks_proto[0][1]

fig = plt.figure(figsize=(14, 6))
gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.1], height_ratios=[3, 1],
                      hspace=0.35, wspace=0.28)

# --- stimulus sequence excerpt ---
ax0 = fig.add_subplot(gs[:, 0])
t0 = onsets[50]
sel = (onsets >= t0) & (onsets < t0 + 30)
fidx = np.array([np.where(FREQS == f)[0][0] for f in freqs[sel]])
ax0.scatter(onsets[sel] - t0, fidx, c=fidx, cmap="viridis", s=60, marker="|",
            linewidths=2.5)
ax0.set_yticks(range(5))
ax0.set_yticklabels(FREQ_LABELS)
ax0.set_xlabel("time (s)")
ax0.set_ylabel("tone frequency (kHz)")
ax0.set_title("Stimulus: randomly interleaved 25 ms pure tones (30 s excerpt)")
ax0.set_ylim(-0.5, 4.5)

# --- raster grouped by frequency for the example unit ---
ax1 = fig.add_subplot(gs[0, 1])
spk = np.sort(units[example_unit].t)
colors = plt.cm.viridis(np.linspace(0, 1, 5))
row = 0
band_edges = []
for fi, f in enumerate(FREQS):
    ons = onsets[freqs == f][:120]           # up to 120 trials/freq for clarity
    band_edges.append(row)
    for o in ons:
        j0 = np.searchsorted(spk, o - 0.05)
        j1 = np.searchsorted(spk, o + 0.2)
        rel = (spk[j0:j1] - o) * 1000
        ax1.plot(rel, np.full_like(rel, row), "|", color=colors[fi],
                 markersize=3, markeredgewidth=0.6)
        row += 1
band_edges.append(row)
ax1.axvline(0, color="k", lw=0.8, ls="--")
ax1.set_xlim(-50, 200)
ax1.set_ylim(0, row)
ax1.set_ylabel("trial (grouped by frequency)")
ax1.set_title(f"Tone-onset raster, unit {example_unit}")
# frequency band labels, placed just outside the right edge of the axes
for fi in range(5):
    mid = 0.5 * (band_edges[fi] + band_edges[fi + 1])
    ax1.text(1.01, mid / row, f"{FREQ_LABELS[fi]} kHz", va="center",
             fontsize=8, color=colors[fi], transform=ax1.transAxes,
             clip_on=False)

# --- peri-onset PSTH per frequency ---
ax2 = fig.add_subplot(gs[1, 1], sharex=ax1)
for fi, f in enumerate(FREQS):
    ons = onsets[freqs == f]
    counts = np.zeros(len(LAGS) - 1)
    for o in ons:
        j0 = np.searchsorted(spk, o + LAGS[0])
        j1 = np.searchsorted(spk, o + LAGS[-1])
        counts += np.histogram(spk[j0:j1] - o, bins=LAGS)[0]
    rate = counts / (len(ons) * np.diff(LAGS))
    ax2.plot(LAGC * 1000, rate, color=colors[fi], lw=1.3)
ax2.axvline(0, color="k", lw=0.8, ls="--")
ax2.set_xlabel("time from tone onset (ms)")
ax2.set_ylabel("rate (Hz)")
ax2.set_title("Peri-onset firing rate per frequency")

fig.savefig("fig1_raw_data.png", dpi=130, bbox_inches="tight")
print(f"saved fig1_raw_data.png (example unit {example_unit})")

# %% [markdown]
# ## Compute STRFs for every unit, across all five mice
#
# For each unit in each session we compute the reverse-correlation STRF and
# summarize it (best frequency, onset latency, peak driven rate). Units whose
# peak driven rate exceeds 2 Hz above baseline are counted as tone-responsive.

# %%
all_strfs = {}        # (subject, unit_id) -> STRF array
all_metrics = {}      # (subject, unit_id) -> metrics dict
session_summary = []

for subj, url in SESSIONS.items():
    nwbf_s, io_s = load_session(url)
    units_s = get_units_tsgroup(nwbf_s)
    onsets_s, freqs_s = get_trials(nwbf_s)
    n_resp = 0
    for k in tqdm(list(units_s.keys()), desc=f"{subj} STRFs"):
        s = compute_strf(units_s[k].t, onsets_s, freqs_s, LAGS)
        m = strf_metrics(s, LAGC)
        all_strfs[(subj, k)] = s
        all_metrics[(subj, k)] = m
        if m["peak_driven"] > RESPONSIVE_HZ:
            n_resp += 1
    session_summary.append((subj, len(units_s), n_resp))
    io_s.close()

print("\nSubject   units  responsive")
for subj, n, nr in session_summary:
    print(f"  {subj:6s}  {n:5d}  {nr:5d}  ({100*nr/n:.0f}%)")

# collect population arrays over responsive units
keys_resp = [k for k, m in all_metrics.items() if m["peak_driven"] > RESPONSIVE_HZ]
bf_all = np.array([all_metrics[k]["best_freq"] for k in keys_resp])
lat_all = np.array([all_metrics[k]["latency"] for k in keys_resp]) * 1000  # ms
peak_all = np.array([all_metrics[k]["peak_driven"] for k in keys_resp])
n_total = len(all_metrics)
n_resp_total = len(keys_resp)
print(f"\nTotal: {n_resp_total}/{n_total} units tone-responsive "
      f"({100*n_resp_total/n_total:.0f}%)")

# %% [markdown]
# ### Figure 2 — Example single-neuron STRFs
#
# Nine of the most strongly driven units, spanning the range of best
# frequencies. Each panel is a frequency × time-lag map of the driven firing
# rate (baseline subtracted): red is excitation, blue suppression. The
# excitatory field appears ~10-25 ms after tone onset (dashed line) at the
# neuron's preferred frequency, the defining signature of an auditory STRF.

# %%
# choose 9 strong units with diverse best frequencies
order = sorted(keys_resp, key=lambda k: all_metrics[k]["peak_driven"],
               reverse=True)
chosen, seen_bf = [], {}
for k in order:                       # spread across best frequencies
    bf = all_metrics[k]["best_freq"]
    if seen_bf.get(bf, 0) < 2:
        chosen.append(k)
        seen_bf[bf] = seen_bf.get(bf, 0) + 1
    if len(chosen) == 9:
        break
for k in order:                       # top up if needed
    if len(chosen) == 9:
        break
    if k not in chosen:
        chosen.append(k)

fig, axs = plt.subplots(3, 3, figsize=(13.5, 11))
for ax, k in zip(axs.flat, chosen):
    s = all_strfs[k]
    m = all_metrics[k]
    d = s - m["baseline"]
    vmax = np.nanmax(np.abs(d))
    im = ax.pcolormesh(LAGC * 1000, np.arange(5), d, cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax, shading="nearest")
    ax.axvline(0, color="k", lw=0.7, ls="--")
    ax.set_yticks(range(5))
    ax.set_yticklabels(FREQ_LABELS)
    lat = m["latency"] * 1000 if np.isfinite(m["latency"]) else np.nan
    ax.set_title(f"{k[0]} unit {k[1]}  |  BF={m['best_freq']/1000:g} kHz, "
                 f"lat={lat:.0f} ms", fontsize=10)
    ax.set_xlabel("time from onset (ms)")
    ax.set_ylabel("frequency (kHz)")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Δrate (Hz)", fontsize=8)
fig.suptitle("Example auditory-cortex spectrotemporal receptive fields",
             fontsize=14, y=1.002)
fig.tight_layout()
fig.savefig("fig2_example_strfs.png", dpi=130, bbox_inches="tight")
print("saved fig2_example_strfs.png")

# %% [markdown]
# ### Figure 3 — Anatomy of one STRF
#
# The STRF of the single most strongly driven unit, decomposed into its two
# marginals: the **frequency tuning curve** (mean driven rate in the response
# window vs. frequency) and the **temporal response profile** at the best
# frequency (driven rate vs. time lag, defining the onset latency).

# %%
kbest = order[0]
s = all_strfs[kbest]
m = all_metrics[kbest]
d = s - m["baseline"]
resp_mask = (LAGC >= 0.005) & (LAGC <= 0.060)

fig = plt.figure(figsize=(12, 5))
gs = fig.add_gridspec(2, 2, width_ratios=[2.4, 1], height_ratios=[1, 1],
                      hspace=0.45, wspace=0.3)

axS = fig.add_subplot(gs[:, 0])
vmax = np.nanmax(np.abs(d))
im = axS.pcolormesh(LAGC * 1000, np.arange(5), d, cmap="RdBu_r",
                    vmin=-vmax, vmax=vmax, shading="nearest")
axS.axvline(0, color="k", lw=0.8, ls="--")
axS.axhline(m["bf_idx"], color="k", lw=0.8, ls=":")
axS.set_yticks(range(5))
axS.set_yticklabels(FREQ_LABELS)
axS.set_xlabel("time from tone onset (ms)")
axS.set_ylabel("frequency (kHz)")
axS.set_title(f"STRF — {kbest[0]} unit {kbest[1]}")
plt.colorbar(im, ax=axS, label="Δrate (Hz)", fraction=0.046, pad=0.03)

# frequency tuning marginal
axF = fig.add_subplot(gs[0, 1])
tuning = np.nanmean(d[:, resp_mask], axis=1)
axF.plot(range(5), tuning, "o-", color="crimson")
axF.set_xticks(range(5))
axF.set_xticklabels(FREQ_LABELS)
axF.axhline(0, color="gray", lw=0.6)
axF.set_xlabel("frequency (kHz)")
axF.set_ylabel("driven rate (Hz)")
axF.set_title("Frequency tuning")

# temporal marginal at BF
axT = fig.add_subplot(gs[1, 1])
axT.plot(LAGC * 1000, d[m["bf_idx"]], color="navy")
axT.axvline(0, color="k", lw=0.7, ls="--")
if np.isfinite(m["latency"]):
    axT.axvline(m["latency"] * 1000, color="green", lw=1,
                label=f"latency {m['latency']*1000:.0f} ms")
    axT.legend(fontsize=8)
axT.set_xlabel("time from onset (ms)")
axT.set_ylabel("driven rate (Hz)")
axT.set_title(f"Temporal profile at BF ({m['best_freq']/1000:g} kHz)")

fig.savefig("fig3_strf_anatomy.png", dpi=130, bbox_inches="tight")
print(f"saved fig3_strf_anatomy.png (unit {kbest})")

# %% [markdown]
# ### Figure 4 — Population summary across five mice
#
# (a) Fraction of tone-responsive units per mouse. (b) Distribution of best
# frequencies (biased toward 2-8 kHz, the mouse hearing range). (c) Onset-
# latency distribution (median a few tens of ms, as expected for cortex).
# (d) Peak driven rate vs. onset latency for all responsive units.

# %%
fig, axs = plt.subplots(2, 2, figsize=(12, 9))

# (a) responsive fraction per subject
ax = axs[0, 0]
subj_names = [s[0] for s in session_summary]
frac = [100 * s[2] / s[1] for s in session_summary]
ax.bar(subj_names, frac, color="steelblue")
for i, (s) in enumerate(session_summary):
    ax.text(i, frac[i] + 1, f"{s[2]}/{s[1]}", ha="center", fontsize=9)
ax.set_ylabel("tone-responsive units (%)")
ax.set_xlabel("mouse")
ax.set_title("(a) Responsive fraction per mouse")
ax.set_ylim(0, 100)

# (b) best-frequency distribution
ax = axs[0, 1]
bf_counts = [np.sum(bf_all == f) for f in FREQS]
ax.bar(range(5), bf_counts, color=plt.cm.viridis(np.linspace(0, 1, 5)))
ax.set_xticks(range(5))
ax.set_xticklabels(FREQ_LABELS)
ax.set_xlabel("best frequency (kHz)")
ax.set_ylabel("number of units")
ax.set_title(f"(b) Best frequency  (n={n_resp_total} units)")

# (c) latency distribution
ax = axs[1, 0]
lat_valid = lat_all[np.isfinite(lat_all)]
ax.hist(lat_valid, bins=np.arange(0, 105, 5), color="darkorange",
        edgecolor="white")
med = np.median(lat_valid)
ax.axvline(med, color="k", ls="--", label=f"median {med:.0f} ms")
ax.legend()
ax.set_xlabel("onset latency (ms)")
ax.set_ylabel("number of units")
ax.set_title("(c) Onset latency")

# (d) peak rate vs latency
ax = axs[1, 1]
fin = np.isfinite(lat_all)
sc = ax.scatter(lat_all[fin], peak_all[fin],
                c=np.log2(bf_all[fin] / 1000), cmap="viridis", s=28,
                alpha=0.8, edgecolors="none")
ax.set_xlabel("onset latency (ms)")
ax.set_ylabel("peak driven rate (Hz)")
ax.set_yscale("log")
ax.set_title("(d) Response strength vs. latency")
cb = plt.colorbar(sc, ax=ax)
cb.set_label("log2 best freq (kHz)")
cb.set_ticks(np.log2(FREQ_KHZ))
cb.set_ticklabels(FREQ_LABELS)

fig.tight_layout()
fig.savefig("fig4_population_summary.png", dpi=130, bbox_inches="tight")
print("saved fig4_population_summary.png")

# %% [markdown]
# ### Figure 5 — Best-frequency-aligned population-average STRF
#
# Each responsive unit's STRF is normalized to its own peak and its frequency
# axis is shifted so that the best frequency sits at 0 octaves; averaging across
# all units yields the canonical population STRF. It shows a compact excitatory
# field, tuned in frequency and delayed in time, with weak suppressive flanks
# above and below the best frequency and after the excitatory transient.

# %%
n_oct = 4  # octaves either side of BF (5 freqs -> shifts span -4..+4)
oct_axis = np.arange(-n_oct, n_oct + 1)
stack = np.full((len(keys_resp), len(oct_axis), len(LAGC)), np.nan)
for i, k in enumerate(keys_resp):
    m = all_metrics[k]
    d = all_strfs[k] - m["baseline"]
    peak = np.nanmax(np.abs(d))
    if peak <= 0:
        continue
    dn = d / peak
    shift = n_oct - m["bf_idx"]              # place BF at center (index n_oct)
    for fi in range(5):
        stack[i, fi + shift, :] = dn[fi]
pop = np.nanmean(stack, axis=0)

fig, ax = plt.subplots(figsize=(8.5, 5.5))
vmax = np.nanmax(np.abs(pop))
im = ax.pcolormesh(LAGC * 1000, oct_axis, pop, cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, shading="nearest")
ax.axvline(0, color="k", lw=0.8, ls="--")
ax.axhline(0, color="k", lw=0.6, ls=":")
ax.set_xlabel("time from tone onset (ms)")
ax.set_ylabel("frequency relative to best (octaves)")
ax.set_title(f"Population-average STRF (n={n_resp_total} responsive units, "
             f"5 mice)")
plt.colorbar(im, ax=ax, label="normalized driven rate")
fig.tight_layout()
fig.savefig("fig5_population_strf.png", dpi=130, bbox_inches="tight")
print("saved fig5_population_strf.png")

# %% [markdown]
# ## Summary
#
# Reverse correlation of tone-triggered spikes recovers clear spectrotemporal
# receptive fields from mouse auditory-cortex Neuropixels recordings. Roughly
# half of the sorted units are tone-responsive; their STRFs are frequency-tuned
# excitatory fields with onset latencies of a few tens of milliseconds, best
# frequencies concentrated in the 2-8 kHz range that matches mouse hearing
# sensitivity, and, in many units, suppressive sidebands that sharpen tuning.
# The best-frequency-aligned population average condenses these features into
# the canonical auditory STRF: a compact, delayed, frequency-tuned excitatory
# lobe with weak inhibitory surround.

# %%
print("Done. Figures written:")
for fn in ["fig1_raw_data.png", "fig2_example_strfs.png",
           "fig3_strf_anatomy.png", "fig4_population_summary.png",
           "fig5_population_strf.png"]:
    print("  ", fn)
