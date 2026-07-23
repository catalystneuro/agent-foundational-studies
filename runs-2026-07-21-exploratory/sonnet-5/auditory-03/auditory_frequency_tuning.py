# %% [markdown]
# # Auditory Frequency Tuning in Mouse Auditory Cortex
#
# This notebook demonstrates **auditory frequency tuning**: the property of neurons in
# auditory cortex to respond preferentially to sounds of a particular frequency, with
# responses falling off for frequencies further away (in octaves) from that "best
# frequency" (BF).
#
# ## Dataset
#
# We use [DANDI:000986](https://dandiarchive.org/dandiset/000986), *"Auditory cortex
# Neuropixels recordings and pupil diameter traces from mice during passive exposure to
# pure tones"*. In each of 15 recording sessions (5 mice), a Neuropixels probe recorded
# spiking activity from auditory cortex of head-fixed mice while brief (25 ms) pure tones
# at five frequencies (2, 4, 8, 16, 32 kHz, one octave apart) were played at a fixed
# amplitude (60 dB). Running speed and pupil diameter were also recorded.
#
# ## Approach
#
# For each unit we compute the mean spike rate in a 100 ms window after tone onset
# ("evoked rate") separately for each of the five frequencies, compare it to a 100 ms
# pre-tone baseline window, and test for frequency selectivity with a one-way ANOVA
# across frequency groups. We first prototype this pipeline on a single session, then
# scale it to all 15 sessions to characterize frequency tuning at the population level.
#
# Data are streamed directly from the DANDI S3 bucket (no full download) using `remfile`
# with disk caching, and analyzed with `pynapple`.

# %%
import warnings

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap
import remfile
from dandi.dandiapi import DandiAPIClient
from pynwb import NWBHDF5IO
from scipy import stats
from tqdm import tqdm

matplotlib.use("Agg")

DANDISET_ID = "000986"
CACHE_DIR = "/tmp/remfile_cache"
RESP_WIN = 0.1  # s, post-tone-onset window used for evoked firing rate
BASE_WIN = 0.1  # s, pre-tone-onset window used for baseline firing rate
ALPHA = 0.01    # significance threshold for the frequency-tuning ANOVA

# %% [markdown]
# ## Resolve session files from the DANDI Archive
#
# We query the DANDI API for all NWB assets in the dandiset and resolve each to its
# public S3 URL (stripped of any presigned query parameters, since the bucket is public
# and a stable unsigned URL is preferable for reproducibility).

# %%
def get_session_urls(dandiset_id):
    client = DandiAPIClient()
    dandiset = client.get_dandiset(dandiset_id)
    urls = {}
    for asset in dandiset.get_assets():
        if asset.path.endswith(".nwb"):
            name = asset.path.split("/")[-1].replace("_behavior.nwb", "")
            # asset.get_content_url returns the direct (unsigned) S3 url when available
            base_url = asset.get_content_url(follow_redirects=1, strip_query=True)
            urls[name] = base_url
    return urls


SESSION_URLS = get_session_urls(DANDISET_ID)
print(f"Found {len(SESSION_URLS)} sessions in dandiset {DANDISET_ID}:")
for name, url in SESSION_URLS.items():
    print(f"  {name}: {url}")

PROTOTYPE_SESSION = "sub-LA3_ses-3"

# %% [markdown]
# ## Load one session and inspect its structure
#
# We start with `sub-LA3_ses-3`, a relatively small session (35 units), to prototype the
# analysis pipeline before scaling up.

# %%
def load_session(url, cache_dir=CACHE_DIR):
    disk_cache = remfile.DiskCache(cache_dir)
    rem_file = remfile.File(url, disk_cache=disk_cache)
    h5f = h5py.File(rem_file, "r")
    io = NWBHDF5IO(file=h5f, load_namespaces=True)
    nwbfile = io.read()
    nwb = nap.NWBFile(nwbfile)
    return nwb, io


nwb, io = load_session(SESSION_URLS[PROTOTYPE_SESSION])
print(nwb)

units = nwb["units"]
trials = nwb["trials"]
running = nwb["running_speed"]
pupil = nwb["pupil_diameter"]

print(f"\nn units: {len(units)}")
print(f"n trials: {len(trials)}")

frequencies = np.array(trials.stim_frequency)
starts = np.array(trials.start)
uniq_freqs = np.unique(frequencies)
n_freqs = len(uniq_freqs)
print(f"Tone frequencies (Hz): {uniq_freqs}")
print(f"Trials per frequency: {[int(np.sum(frequencies == f)) for f in uniq_freqs]}")
print(f"Stimulus amplitude(s) (dB): {np.unique(trials.stim_amplitude)}")
print(f"Stimulus duration(s) (s): {np.unique(trials.stim_duration)}")

# %% [markdown]
# ## Raw data overview
#
# Before any analysis, we visualize a snippet of raw spiking activity together with the
# tone presentations (colored by frequency), running speed, and pupil diameter, to confirm
# the data streams are aligned and sensible.

# %%
t0 = trials.start[100]
t1 = trials.start[130]
snippet_ep = nap.IntervalSet(start=t0 - 1, end=t1 + 1)
trials_snip = trials.intersect(snippet_ep)
running_snip = running.restrict(snippet_ep)
pupil_snip = pupil.restrict(snippet_ep)

cmap = plt.get_cmap("viridis")
freq_colors = {f: cmap(i / (n_freqs - 1)) for i, f in enumerate(uniq_freqs)}

fig, axes = plt.subplots(
    3, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1, 1]}
)

ax = axes[0]
for i, uid in enumerate(units.index[:20]):
    st = units[uid].restrict(snippet_ep).t
    ax.vlines(st, i, i + 0.8, color="k", lw=0.8)
for row in trials_snip.as_dataframe().itertuples():
    ax.axvspan(row.start, row.end, color=freq_colors[row.stim_frequency], alpha=0.3, lw=0)
ax.set_ylabel("Unit # (first 20)")
ax.set_title("Raw spike raster with tone presentations (color = frequency)")
handles = [plt.Line2D([0], [0], color=freq_colors[f], lw=6, alpha=0.5) for f in uniq_freqs]
labels = [f"{f/1000:.0f} kHz" for f in uniq_freqs]
ax.legend(handles, labels, loc="upper right", ncol=n_freqs, fontsize=8)

ax = axes[1]
ax.plot(running_snip.t, running_snip.d, color="tab:blue")
ax.set_ylabel("Running\nspeed (a.u.)")

ax = axes[2]
ax.plot(pupil_snip.t, pupil_snip.d, color="tab:orange")
ax.set_ylabel("Pupil\ndiameter (norm.)")
ax.set_xlabel("Time (s)")

plt.tight_layout()
plt.savefig("fig1_raw_data_overview.png", dpi=150)
plt.close(fig)
print("Saved fig1_raw_data_overview.png")

# %% [markdown]
# ## Compute evoked firing rates and frequency tuning curves
#
# For every unit and every trial we count spikes in a 100 ms window after tone onset
# (evoked rate) and in a 100 ms window before tone onset (baseline rate), using vectorized
# `searchsorted` lookups against each unit's sorted spike times. Trials are ~0.8 s apart in
# this paradigm, so a 100 ms window on either side of tone onset does not spill into
# neighboring trials. We then average the evoked rate within each frequency to build a
# tuning curve, and test for frequency selectivity per unit with a one-way ANOVA across
# the five frequency groups.


# %%
def compute_tuning(units, frequencies, starts, uniq_freqs, resp_win=RESP_WIN, base_win=BASE_WIN):
    n_units = len(units.index)
    n_freqs = len(uniq_freqs)
    tuning_means = np.zeros((n_units, n_freqs))
    tuning_sems = np.zeros((n_units, n_freqs))
    pvals = np.zeros(n_units)
    base_rate_mean = np.zeros(n_units)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for i, uid in enumerate(units.index):
            st = units[uid].t
            evoked_counts = np.searchsorted(st, starts + resp_win) - np.searchsorted(st, starts)
            base_counts = np.searchsorted(st, starts) - np.searchsorted(st, starts - base_win)
            evoked_rate = evoked_counts / resp_win
            base_rate = base_counts / base_win
            base_rate_mean[i] = base_rate.mean()

            groups = [evoked_rate[frequencies == f] for f in uniq_freqs]
            tuning_means[i] = [g.mean() for g in groups]
            tuning_sems[i] = [g.std() / np.sqrt(len(g)) for g in groups]
            _, p = stats.f_oneway(*groups)
            pvals[i] = p if not np.isnan(p) else 1.0

    return tuning_means, tuning_sems, pvals, base_rate_mean


tuning_means, tuning_sems, pvals, base_rate_mean = compute_tuning(
    units, frequencies, starts, uniq_freqs
)
sig_mask = pvals < ALPHA
bf_idx = np.argmax(tuning_means, axis=1)
bf_freq = uniq_freqs[bf_idx]

print(f"Significantly frequency-tuned units (ANOVA p<{ALPHA}): "
      f"{sig_mask.sum()}/{len(units.index)} ({100*sig_mask.mean():.0f}%)")
print("Best-frequency distribution among tuned units:",
      {int(f): int(np.sum(bf_freq[sig_mask] == f)) for f in uniq_freqs})


def format_pval(p):
    return "p<1e-300" if p == 0 else f"p={p:.1e}"


# %% [markdown]
# ## Example units: peri-stimulus rasters and PSTHs by frequency
#
# We pick the three most significantly tuned units and show, for each, a raster of spikes
# around tone onset (trials grouped and colored by frequency) and the corresponding
# peri-stimulus time histogram (PSTH) per frequency.

# %%
example_units = units.index[sig_mask][np.argsort(pvals[sig_mask])[:3]]
print("Example units:", list(example_units))

PERIEVENT_WIN = (-0.05, 0.15)
bin_size = 0.005
bins = np.arange(PERIEVENT_WIN[0], PERIEVENT_WIN[1] + bin_size, bin_size)
bin_centers = (bins[:-1] + bins[1:]) / 2

fig, axes = plt.subplots(2, len(example_units), figsize=(5 * len(example_units), 7), sharex="col")
for col, uid in enumerate(example_units):
    st = units[uid].t
    i = list(units.index).index(uid)
    ax_raster = axes[0, col]
    ax_psth = axes[1, col]

    row = 0
    for f in uniq_freqs:
        trial_idx = np.where(frequencies == f)[0]
        sub = trial_idx[:: max(1, len(trial_idx) // 40)]
        for ti in sub:
            t0_trial = starts[ti]
            rel = st[(st >= t0_trial + PERIEVENT_WIN[0]) & (st < t0_trial + PERIEVENT_WIN[1])] - t0_trial
            ax_raster.vlines(rel, row, row + 0.8, color=freq_colors[f], lw=0.6)
            row += 1
        row += 2
    ax_raster.axvline(0, color="k", lw=0.8, ls="--")
    ax_raster.set_title(f"Unit {uid} ({format_pval(pvals[i])})")
    ax_raster.set_ylabel("Trial (grouped by freq)")

    for f in uniq_freqs:
        trial_idx = np.where(frequencies == f)[0]
        counts = np.zeros(len(bins) - 1)
        for ti in trial_idx:
            t0_trial = starts[ti]
            rel = st[(st >= t0_trial + PERIEVENT_WIN[0]) & (st < t0_trial + PERIEVENT_WIN[1])] - t0_trial
            c, _ = np.histogram(rel, bins=bins)
            counts += c
        rate = counts / (len(trial_idx) * bin_size)
        ax_psth.plot(bin_centers, rate, color=freq_colors[f], label=f"{f/1000:.0f} kHz")
    ax_psth.axvline(0, color="k", lw=0.8, ls="--")
    ax_psth.set_xlabel("Time from tone onset (s)")
    ax_psth.set_ylabel("Firing rate (Hz)")
    if col == len(example_units) - 1:
        ax_psth.legend(fontsize=8, loc="upper right")

plt.tight_layout()
plt.savefig("fig2_example_psth.png", dpi=150)
plt.close(fig)
print("Saved fig2_example_psth.png")

# %% [markdown]
# ## Tuning curves for the example units
#
# Evoked firing rate (mean ± SEM across trials) as a function of tone frequency, with the
# pre-tone baseline rate shown as a dashed line for reference.

# %%
fig, axes = plt.subplots(1, len(example_units), figsize=(5 * len(example_units), 4))
for col, uid in enumerate(example_units):
    i = list(units.index).index(uid)
    ax = axes[col]
    ax.errorbar(uniq_freqs, tuning_means[i], yerr=tuning_sems[i], marker="o", color="tab:blue")
    ax.axhline(base_rate_mean[i], color="gray", ls="--", label="baseline")
    ax.set_xscale("log", base=2)
    ax.set_xticks(uniq_freqs)
    ax.set_xticklabels([f"{f/1000:.0f}" for f in uniq_freqs])
    ax.set_xlabel("Tone frequency (kHz)")
    ax.set_ylabel("Evoked firing rate (Hz)")
    ax.set_title(f"Unit {uid}")
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("fig3_tuning_curves_examples.png", dpi=150)
plt.close(fig)
print("Saved fig3_tuning_curves_examples.png")

# %% [markdown]
# ## Population heatmap for this session
#
# Normalized (0-1 per unit) tuning curves for all significantly tuned units in this
# session, sorted by best frequency, reveal a diversity of preferred frequencies across
# the simultaneously recorded population.

# %%
sig_idx = np.where(sig_mask)[0]
order = sig_idx[np.argsort(bf_idx[sig_idx])]
norm_curves = np.zeros((len(order), n_freqs))
for r, i in enumerate(order):
    tc = tuning_means[i]
    norm_curves[r] = (tc - tc.min()) / (tc.max() - tc.min())

fig, ax = plt.subplots(figsize=(6, 8))
im = ax.imshow(norm_curves, aspect="auto", cmap="viridis", interpolation="nearest")
ax.set_xticks(range(n_freqs))
ax.set_xticklabels([f"{f/1000:.0f}" for f in uniq_freqs])
ax.set_xlabel("Tone frequency (kHz)")
ax.set_ylabel("Unit (sorted by best frequency)")
ax.set_title(f"Normalized frequency tuning curves\n({PROTOTYPE_SESSION}, significantly tuned units)")
plt.colorbar(im, ax=ax, label="Normalized evoked rate")
plt.tight_layout()
plt.savefig("fig4_population_heatmap.png", dpi=150)
plt.close(fig)
print("Saved fig4_population_heatmap.png")

io.close()

# %% [markdown]
# ## Scaling up: all 15 sessions
#
# The pipeline above only needs each unit's spike times and the trial table, both of which
# are small relative to the raw ephys traces, so it streams and analyzes all 15 sessions in
# well under a minute. We repeat the tuning-curve computation for every session and pool
# results across all units and animals.

# %%
def analyze_session(url):
    nwb, io = load_session(url)
    units = nwb["units"]
    trials = nwb["trials"]
    frequencies = np.array(trials.stim_frequency)
    starts = np.array(trials.start)
    uniq_freqs = np.unique(frequencies)

    tuning_means, tuning_sems, pvals, base_rate_mean = compute_tuning(
        units, frequencies, starts, uniq_freqs
    )
    io.close()
    return dict(
        uniq_freqs=uniq_freqs,
        tuning_means=tuning_means,
        pvals=pvals,
        base_rate_mean=base_rate_mean,
        n_units=len(units.index),
    )


all_results = {}
for name, url in tqdm(SESSION_URLS.items(), desc="Analyzing sessions"):
    all_results[name] = analyze_session(url)

total_units = sum(r["n_units"] for r in all_results.values())
total_sig = sum(int(np.sum(r["pvals"] < ALPHA)) for r in all_results.values())
print(f"\nAcross all sessions: {total_sig}/{total_units} units "
      f"({100*total_sig/total_units:.1f}%) show significant frequency tuning (p<{ALPHA})")

# %% [markdown]
# ## Population summary
#
# Three complementary views of frequency tuning across the full dataset:
#
# - **(A)** the fraction of significantly tuned units in each of the 15 sessions,
# - **(B)** the pooled distribution of best frequencies among all tuned units,
# - **(C)** the average tuning curve after aligning every tuned unit's curve to its own
#   best frequency (expressed as octave distance from BF) and min-max normalizing it. By
#   construction the value at 0 octaves is 1.0 for every unit (that is where its own
#   maximum is), so the curve's peak is not itself informative; what demonstrates genuine
#   band-pass frequency tuning is the systematic, roughly symmetric fall-off in response
#   at +/-1 and +/-2 octaves from BF.

# %%
uniq_freqs_global = list(all_results.values())[0]["uniq_freqs"]
n_freqs_global = len(uniq_freqs_global)

all_bf_idx = []
all_aligned = []
frac_tuned = {}
n_units_list = {}

for name, res in all_results.items():
    sig = res["pvals"] < ALPHA
    frac_tuned[name] = np.mean(sig)
    n_units_list[name] = res["n_units"]
    tm = res["tuning_means"][sig]
    bf_i = np.argmax(tm, axis=1)
    all_bf_idx.extend(bf_i.tolist())
    for row, bi in zip(tm, bf_i):
        norm = (row - row.min()) / (row.max() - row.min())
        shift = n_freqs_global // 2 - bi
        aligned = np.roll(norm, shift)
        valid = np.ones(n_freqs_global, dtype=bool)
        if shift > 0:
            valid[:shift] = False
        elif shift < 0:
            valid[shift:] = False
        aligned = np.where(valid, aligned, np.nan)
        all_aligned.append(aligned)

all_bf_idx = np.array(all_bf_idx)
all_aligned = np.array(all_aligned)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

ax = axes[0]
names = list(frac_tuned.keys())
vals = [frac_tuned[n] * 100 for n in names]
order = np.argsort(vals)[::-1]
ax.barh(range(len(names)), [vals[i] for i in order], color="tab:blue")
ax.set_yticks(range(len(names)))
ax.set_yticklabels([names[i] for i in order], fontsize=7)
ax.set_xlabel("% significantly frequency-tuned units")
ax.set_title(f"(A) Tuned units per session\n(n={sum(n_units_list.values())} units, "
             f"{len(names)} sessions)")
ax.axvline(np.mean(vals), color="k", ls="--", lw=1, label=f"mean={np.mean(vals):.0f}%")
ax.legend(fontsize=8)

ax = axes[1]
counts = [int(np.sum(all_bf_idx == i)) for i in range(n_freqs_global)]
ax.bar(range(n_freqs_global), counts, color="tab:green")
ax.set_xticks(range(n_freqs_global))
ax.set_xticklabels([f"{f/1000:.0f}" for f in uniq_freqs_global])
ax.set_xlabel("Best frequency (kHz)")
ax.set_ylabel("Number of units")
ax.set_title(f"(B) Best-frequency distribution\n(n={len(all_bf_idx)} tuned units, pooled)")

ax = axes[2]
octave_offsets = np.arange(n_freqs_global) - n_freqs_global // 2
mean_curve = np.nanmean(all_aligned, axis=0)
sem_curve = np.nanstd(all_aligned, axis=0) / np.sqrt(np.sum(~np.isnan(all_aligned), axis=0))
ax.errorbar(octave_offsets, mean_curve, yerr=sem_curve, marker="o", color="tab:purple")
ax.set_xlabel("Octaves from best frequency")
ax.set_ylabel("Normalized evoked rate (mean +/- SEM)")
ax.set_title("(C) Population tuning curve\naligned to best frequency")

plt.tight_layout()
plt.savefig("fig5_population_summary.png", dpi=150)
plt.close(fig)
print("Saved fig5_population_summary.png")

# %% [markdown]
# ## Conclusion
#
# Individual auditory-cortex units show clear, statistically significant tuning to tone
# frequency: evoked responses rise sharply above baseline within ~10-20 ms of tone onset,
# and their magnitude depends systematically on tone frequency (Figures 2-3). Across the
# full dataset, 85% of the 1564 recorded units (15 sessions, 5 mice) pass a frequency ANOVA
# at p<0.01, best frequencies span the full 2-32 kHz range tested (Figure 5B), and the
# population-averaged tuning curve aligned to each unit's own best frequency falls off
# monotonically and roughly symmetrically over +/-2 octaves (Figure 5C). Together these
# results demonstrate robust auditory frequency tuning in mouse auditory cortex.
