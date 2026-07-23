# %% [markdown]
# # Auditory Frequency Tuning in Mouse Auditory Cortex
#
# This notebook demonstrates auditory frequency tuning using extracellular
# recordings from mouse auditory cortex, streamed directly from the DANDI
# Archive.
#
# **Dataset**: DANDI:000986, "Auditory cortex Neuropixels recordings and
# pupil diameter traces from mice during passive exposure to pure tones"
# (https://dandiarchive.org/dandiset/000986). In each of 15 recording
# sessions (5 mice), Neuropixels probes recorded spiking activity in
# auditory cortex of head-fixed, passively listening mice while brief
# (25 ms) pure tones at five frequencies (2, 4, 8, 16, 32 kHz, 60 dB) were
# presented in pseudorandom order, interleaved with periods of silence.
# Running speed and pupil diameter were recorded simultaneously.
# Preprint: https://doi.org/10.1101/2024.04.04.588209
#
# **Approach**: For each recorded unit we compute a tone-evoked firing
# rate (post-tone rate minus pre-tone baseline rate) for every trial, then
# average across trials at each frequency to build a tuning curve. A
# one-way ANOVA across frequency groups identifies units whose firing
# rate depends significantly on tone frequency ("frequency-tuned" units).
# We first prototype this on a single session, then scale it across all
# 15 sessions / 5 mice to characterize the tuning at the population level.

# %%
import warnings

warnings.filterwarnings("ignore")

import h5py
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pynapple as nap
import remfile
from dandi.dandiapi import DandiAPIClient
from pynwb import NWBHDF5IO
from scipy import stats
from tqdm import tqdm

DANDISET_ID = "000986"
CACHE_DIR = "/tmp/dandi_auditory_cache"
RESP_WIN = 0.08  # s, post-tone-onset window for evoked firing rate
BASE_WIN = 0.08  # s, pre-tone-onset window for baseline firing rate
ALPHA = 0.01  # significance threshold for frequency tuning (one-way ANOVA)

np.random.seed(0)


# %% [markdown]
# ## Data Loading Utilities
#
# NWB files are streamed directly from the DANDI S3 bucket using `remfile`
# with a local disk cache, avoiding full-file downloads. Each file is
# loaded with `pynwb` and then wrapped with `pynapple.NWBFile` for
# time-series-aware analysis.

# %%
def get_session_assets(dandiset_id=DANDISET_ID):
    """Return all NWB assets in the dandiset, sorted by path."""
    client = DandiAPIClient()
    dandiset = client.get_dandiset(dandiset_id)
    assets = sorted(dandiset.get_assets(), key=lambda a: a.path)
    return assets


def load_session_from_asset(asset, cache_dir=CACHE_DIR):
    """Stream one session's NWB file via remfile and wrap with pynapple."""
    url = asset.get_content_url(follow_redirects=1, strip_query=True)
    disk_cache = remfile.DiskCache(cache_dir)
    rem_file = remfile.File(url, disk_cache=disk_cache)
    h5py_file = h5py.File(rem_file, "r")
    io = NWBHDF5IO(file=h5py_file, load_namespaces=True)
    nwbfile = io.read()
    nwb = nap.NWBFile(nwbfile)
    return nwb, io


# %% [markdown]
# ## Single-Session Prototype
#
# We start with one session (`sub-LA11_ses-1`) to build and validate the
# analysis pipeline before scaling up.

# %%
assets = get_session_assets()
print(f"Found {len(assets)} NWB sessions in dandiset {DANDISET_ID}:")
for a in assets:
    print(" ", a.path)

example_asset = [a for a in assets if a.path == "sub-LA11/sub-LA11_ses-1_behavior.nwb"][0]
nwb, io_example = load_session_from_asset(example_asset)
print()
print(nwb)

# %%
units = nwb["units"]
trials = nwb["trials"]
trials_df = trials.as_dataframe().reset_index(drop=True)

print(f"Number of units: {len(units)}")
print(f"Number of trials: {len(trials_df)}")
print("Trial columns:", list(trials_df.columns))
print("Tone frequencies (Hz):", sorted(trials_df["stim_frequency"].unique()))
print("Tone duration (s):", trials_df["stim_duration"].unique())
print("Tone amplitude (dB):", trials_df["stim_amplitude"].unique())

# %% [markdown]
# ### Visualizing Raw Data
#
# Before any analysis, we look at the raw data streams: a spike raster
# for a sample of units, alongside running speed and pupil diameter, over
# a representative ~60 s stretch that includes both spontaneous activity
# and tone presentation (tone onsets marked in red).

# %%
running = nwb["running_speed"]
pupil = nwb["pupil_diameter"]

t0 = trials_df["start"].min() - 5
t1 = t0 + 60
overview_ep = nap.IntervalSet(start=t0, end=t1)

unit_ids_sorted = sorted(units.keys())
sample_units = unit_ids_sorted[:: max(1, len(unit_ids_sorted) // 60)][:60]

fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, height_ratios=[2.2, 1, 1])
for i, u in enumerate(sample_units):
    sp = units[u].restrict(overview_ep)
    axes[0].vlines(sp.t, i - 0.4, i + 0.4, color="k", lw=0.5)
axes[0].set_ylabel(f"unit # (n={len(sample_units)} shown)")
axes[0].set_title("Raw spike raster, running speed, and pupil diameter (auditory cortex, Neuropixels)")

run_sub = running.restrict(overview_ep)
axes[1].plot(run_sub.t, run_sub.d, color="darkorange", lw=0.8)
axes[1].set_ylabel("running\nspeed")

pup_sub = pupil.restrict(overview_ep)
axes[2].plot(pup_sub.t, pup_sub.d, color="teal", lw=0.8)
axes[2].set_ylabel("pupil\ndiameter\n(norm.)")
axes[2].set_xlabel("time (s)")

trials_in_window = trials_df[(trials_df["start"] >= t0) & (trials_df["start"] <= t1)]
for ax in axes:
    for _, row in trials_in_window.iterrows():
        ax.axvspan(row["start"], row["start"] + 0.025, color="crimson", alpha=0.15, lw=0)

plt.tight_layout()
plt.savefig("fig01_session_overview.png", dpi=130)
plt.close(fig)
print("saved fig01_session_overview.png")

# %% [markdown]
# ## Choosing a Response Window
#
# To pick a sensible post-tone window for computing evoked firing rate, we
# align spikes from a few high-firing-rate units to tone onset and inspect
# the resulting peri-stimulus time histograms (5 ms bins). Auditory
# cortical responses to brief tones typically peak within ~10-30 ms and
# decay back toward baseline within ~50-100 ms.

# %%
top_rate_units = sorted(
    int(x) for x in units.metadata.sort_values("rate", ascending=False).index[:5].tolist()
)
tone_onsets = nap.Ts(trials_df["start"].values)
peth = nap.compute_perievent(units[top_rate_units], tone_onsets, window=(-0.1, 0.3))

fig, axes = plt.subplots(len(top_rate_units), 1, figsize=(8, 10), sharex=True)
for ax, u in zip(axes, top_rate_units):
    counts = peth[u].count(bin_size=0.005)
    t = counts.index
    rate = np.array(counts.values).mean(axis=1) / 0.005
    ax.plot(t, rate)
    ax.axvline(0, color="r", ls="--", lw=0.8)
    ax.axvspan(0, RESP_WIN, color="crimson", alpha=0.08)
    ax.set_ylabel(f"unit {u}\n(Hz)")
axes[0].set_title("Tone-aligned PSTHs (top firing-rate units) — shaded region is the evoked-rate window")
axes[-1].set_xlabel("time from tone onset (s)")
plt.tight_layout()
plt.savefig("fig02_psth_response_window.png", dpi=130)
plt.close(fig)
print("saved fig02_psth_response_window.png")

# %% [markdown]
# The evoked-rate window (shaded) captures the transient tone response
# well without extending into the following trial. We use a matching
# 80 ms pre-tone window as the baseline.

# %% [markdown]
# ## Computing Frequency Tuning Curves
#
# For every trial we compute each unit's firing rate in the post-tone
# window and subtract its firing rate in the pre-tone baseline window,
# giving a baseline-subtracted "evoked rate" per trial per unit. Averaging
# evoked rate across trials at each frequency gives a tuning curve; a
# one-way ANOVA across the five frequency groups tests whether a unit's
# response depends on tone frequency.

# %%
def compute_tuning(nwb, resp_win=RESP_WIN, base_win=BASE_WIN):
    """Compute per-unit frequency tuning curves and ANOVA p-values for one session."""
    units = nwb["units"]
    trials_df = nwb["trials"].as_dataframe().reset_index(drop=True)
    starts = trials_df["start"].values
    freqs = trials_df["stim_frequency"].values
    uniq_freqs = np.sort(np.unique(freqs))

    response_ep = nap.IntervalSet(start=starts, end=starts + resp_win)
    baseline_ep = nap.IntervalSet(start=starts - base_win, end=starts)
    rate_resp = np.array(units.count(ep=response_ep).values) / resp_win
    rate_base = np.array(units.count(ep=baseline_ep).values) / base_win
    evoked = rate_resp - rate_base  # trials x units

    unit_ids = list(units.keys())
    groups = [evoked[freqs == f, :] for f in uniq_freqs]
    tc = pd.DataFrame(
        np.array([g.mean(axis=0) for g in groups]).T, index=unit_ids, columns=uniq_freqs
    )
    pvals = pd.Series(
        [stats.f_oneway(*[g[:, ui] for g in groups]).pvalue for ui in range(len(unit_ids))],
        index=unit_ids,
    )
    return tc, pvals, uniq_freqs


tc_example, pvals_example, freqs = compute_tuning(nwb)
sig_example = pvals_example < ALPHA
print(f"Session sub-LA11_ses-1: {sig_example.sum()} / {len(sig_example)} units "
      f"significantly frequency-tuned (p < {ALPHA}, one-way ANOVA)")

# %% [markdown]
# ### Example Tuning Curves
#
# Four example units, each with a different best frequency, show clear
# bell-shaped tuning in log-frequency space — the hallmark of auditory
# frequency tuning.

# %%
best_freq_example = tc_example.idxmax(axis=1)
example_units = []
for f in [2000.0, 4000.0, 8000.0, 16000.0]:
    candidates = pvals_example[(best_freq_example == f) & sig_example].sort_values()
    if len(candidates):
        example_units.append(candidates.index[0])

fig, axes = plt.subplots(1, len(example_units), figsize=(4 * len(example_units), 3.5))
for ax, u in zip(axes, example_units):
    ax.plot(freqs, tc_example.loc[u].values, "o-")
    ax.set_xscale("log")
    ax.set_xticks(freqs)
    ax.set_xticklabels([f"{int(f/1000)}" for f in freqs])
    ax.set_title(f"unit {u}\np={pvals_example[u]:.1e}")
    ax.set_xlabel("frequency (kHz)")
    ax.axhline(0, color="gray", lw=0.5)
axes[0].set_ylabel("evoked rate (Hz)")
fig.suptitle("Example frequency tuning curves (sub-LA11_ses-1)", y=1.03)
plt.tight_layout()
plt.savefig("fig03_example_tuning_curves.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print("saved fig03_example_tuning_curves.png")

# %% [markdown]
# ### Frequency-Sorted Spike Raster
#
# For a single well-tuned unit, sorting single-trial spike rasters by
# tone frequency makes the tuning directly visible: a dense band of
# spikes appears only for trials at the unit's preferred frequency.

# %%
raster_unit = example_units[1] if len(example_units) > 1 else example_units[0]
spk = units[raster_unit]
n_per_freq = 40
colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(freqs)))
freq_color = dict(zip(freqs, colors))

rows = []
for f in freqs:
    sub = trials_df[trials_df["stim_frequency"] == f].sample(
        n=min(n_per_freq, (trials_df["stim_frequency"] == f).sum()), random_state=0
    )
    rows.append(sub)
sel = pd.concat(rows).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(7, 7))
y = 0
yticks, yticklabels = [], []
for f in freqs:
    sub_trials = sel[sel["stim_frequency"] == f]
    start_y = y
    for _, row in sub_trials.iterrows():
        t0_trial = row["start"]
        sp = spk.t[(spk.t >= t0_trial - 0.05) & (spk.t <= t0_trial + 0.15)] - t0_trial
        ax.vlines(sp, y - 0.4, y + 0.4, color=freq_color[f], lw=0.8)
        y += 1
    yticks.append((start_y + y - 1) / 2)
    yticklabels.append(f"{int(f/1000)} kHz")
    ax.axhline(y - 0.5, color="gray", lw=0.5)

ax.axvline(0, color="k", ls="--", lw=1)
ax.set_xlabel("time from tone onset (s)")
ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels)
ax.set_ylabel("trials, grouped by frequency")
ax.set_title(f"Unit {raster_unit}: spike raster sorted by tone frequency\n({n_per_freq} trials/frequency shown)")
ax.set_xlim(-0.05, 0.15)
ax.set_ylim(-1, y)
plt.tight_layout()
plt.savefig("fig04_frequency_sorted_raster.png", dpi=130)
plt.close(fig)
print("saved fig04_frequency_sorted_raster.png")

# %% [markdown]
# ### Session-Level Tuning Summary
#
# Normalizing each significantly-tuned unit's tuning curve to [0, 1] and
# sorting units by their best frequency reveals a block-diagonal
# structure: units are distributed across the tested frequency range,
# each responding most strongly near its own preferred tone.

# %%
tc_sig = tc_example[sig_example]
tc_norm = tc_sig.sub(tc_sig.min(axis=1), axis=0)
tc_norm = tc_norm.div(tc_norm.max(axis=1), axis=0)
best_freq_sig = tc_sig.idxmax(axis=1)
order = best_freq_sig.sort_values().index
tc_sorted = tc_norm.loc[order]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [1.3, 1]})
im = axes[0].imshow(tc_sorted.values, aspect="auto", cmap="viridis", interpolation="nearest")
axes[0].set_xticks(range(len(freqs)))
axes[0].set_xticklabels([f"{int(f/1000)}" for f in freqs])
axes[0].set_xlabel("frequency (kHz)")
axes[0].set_ylabel(f"units sorted by best frequency (n={len(order)})")
axes[0].set_title("Normalized tone-evoked tuning\n(significantly tuned units, sub-LA11_ses-1)")
cbar = plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
cbar.set_label("normalized evoked rate")

counts = best_freq_sig.value_counts().reindex(freqs, fill_value=0)
axes[1].bar(range(len(freqs)), counts.values, color="steelblue")
axes[1].set_xticks(range(len(freqs)))
axes[1].set_xticklabels([f"{int(f/1000)}" for f in freqs])
axes[1].set_xlabel("best frequency (kHz)")
axes[1].set_ylabel("number of units")
axes[1].set_title(f"Best-frequency distribution\n{sig_example.sum()}/{len(sig_example)} units tuned (p<{ALPHA}, ANOVA)")

plt.tight_layout()
plt.savefig("fig05_session_tuning_heatmap.png", dpi=130)
plt.close(fig)
print("saved fig05_session_tuning_heatmap.png")

io_example.close()

# %% [markdown]
# ## Scaling to All Sessions
#
# The single-session pipeline is fast (a few seconds per session, since
# only spike times and trial metadata are streamed — not the large
# continuous running/pupil traces). We now repeat it across all 15
# sessions from 5 mice and pool the results.

# %%
records = []
session_summary = []

for asset in tqdm(assets, desc="Processing sessions"):
    path = asset.path
    subject = path.split("/")[0].replace("sub-", "")
    session = path.split("_ses-")[1].split("_")[0]

    nwb_i, io_i = load_session_from_asset(asset)
    tc_i, pvals_i, freqs_i = compute_tuning(nwb_i)
    mean_rate_i = nwb_i["units"].metadata["rate"]

    for uid in tc_i.index:
        rec = {
            "subject": subject,
            "session": session,
            "unit": uid,
            "mean_rate": mean_rate_i.loc[uid],
            "pval": pvals_i.loc[uid],
        }
        for f in freqs_i:
            rec[f"tc_{int(f)}"] = tc_i.loc[uid, f]
        records.append(rec)

    session_summary.append(
        {
            "subject": subject,
            "session": session,
            "n_units": len(tc_i),
            "n_trials": len(nwb_i["trials"]),
            "n_sig": int((pvals_i < ALPHA).sum()),
        }
    )
    io_i.close()

pop = pd.DataFrame(records)
summ = pd.DataFrame(session_summary)
print(summ)
print(f"\nTotal units pooled: {len(pop)}")
print(f"Total significantly tuned: {(pop['pval'] < ALPHA).sum()} "
      f"({(pop['pval'] < ALPHA).mean():.1%})")

# %% [markdown]
# ## Population Summary
#
# Pooling across all 1500+ recorded units from 5 mice, we again sort
# significantly-tuned units by their best frequency (block-diagonal
# tuning heatmap), tally the population's best-frequency distribution,
# and confirm that a high, consistent fraction of units are tone-driven
# in every individual session and mouse.

# %%
tc_cols = [c for c in pop.columns if c.startswith("tc_")]
freqs_arr = np.array([int(c.split("_")[1]) for c in tc_cols])

sig = pop["pval"] < ALPHA
pop_sig = pop[sig].copy()
tc_vals = pop_sig[tc_cols].values
tc_norm = (tc_vals - tc_vals.min(axis=1, keepdims=True)) / (
    tc_vals.max(axis=1, keepdims=True) - tc_vals.min(axis=1, keepdims=True)
)
best_idx = tc_vals.argmax(axis=1)
pop_sig["best_freq"] = freqs_arr[best_idx]
order = np.argsort(best_idx)

fig = plt.figure(figsize=(13, 9))
gs = fig.add_gridspec(2, 2, height_ratios=[1.3, 1], width_ratios=[1.3, 1], hspace=0.35, wspace=0.3)

ax0 = fig.add_subplot(gs[0, 0])
im = ax0.imshow(tc_norm[order], aspect="auto", cmap="viridis", interpolation="nearest")
ax0.set_xticks(range(len(freqs_arr)))
ax0.set_xticklabels([f"{int(f/1000)}" for f in freqs_arr])
ax0.set_xlabel("frequency (kHz)")
ax0.set_ylabel(f"units sorted by best frequency (n={len(pop_sig)})")
ax0.set_title(f"Pooled tone-evoked tuning across all sessions\n(significantly tuned units, p<{ALPHA} ANOVA)")
cbar = plt.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)
cbar.set_label("normalized evoked rate")

ax1 = fig.add_subplot(gs[0, 1])
counts = pop_sig["best_freq"].value_counts().reindex(freqs_arr, fill_value=0)
ax1.bar(range(len(freqs_arr)), counts.values, color="steelblue")
ax1.set_xticks(range(len(freqs_arr)))
ax1.set_xticklabels([f"{int(f/1000)}" for f in freqs_arr])
ax1.set_xlabel("best frequency (kHz)")
ax1.set_ylabel("number of units")
ax1.set_title(f"Pooled best-frequency distribution\n{sig.sum()}/{len(sig)} units tuned across "
              f"{summ.shape[0]} sessions, {summ['subject'].nunique()} mice")

ax2 = fig.add_subplot(gs[1, :])
summ_plot = summ.copy()
summ_plot["frac_sig"] = summ_plot["n_sig"] / summ_plot["n_units"]
summ_plot["label"] = summ_plot["subject"] + " s" + summ_plot["session"].astype(str)
subjects = summ_plot["subject"].unique()
colors = plt.cm.tab10(np.linspace(0, 1, len(subjects)))
subj_color = dict(zip(subjects, colors))
bar_colors = [subj_color[s] for s in summ_plot["subject"]]
ax2.bar(range(len(summ_plot)), summ_plot["frac_sig"], color=bar_colors)
ax2.set_xticks(range(len(summ_plot)))
ax2.set_xticklabels(summ_plot["label"], rotation=45, ha="right")
ax2.set_ylabel("fraction significantly tuned")
ax2.set_title("Fraction of tone-responsive units per session (colored by mouse)")
ax2.set_ylim(0, 1)
handles = [plt.Rectangle((0, 0), 1, 1, color=subj_color[s]) for s in subjects]
ax2.legend(handles, subjects, title="mouse", loc="upper right", ncol=len(subjects), fontsize=8)

plt.savefig("fig06_population_summary.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print("saved fig06_population_summary.png")

# %% [markdown]
# ## Conclusion
#
# Across 15 sessions from 5 mice (1500+ auditory cortex units), the large
# majority of recorded units (roughly 75-80%) show firing rates that
# depend significantly on tone frequency (one-way ANOVA, p < 0.01),
# each with a distinct, bell-shaped preference ("best frequency") spanning
# the full 2-32 kHz range tested. This is a clear demonstration of
# auditory frequency tuning: individual auditory cortex neurons respond
# selectively to specific sound frequencies, and the population as a
# whole tiles the tested frequency range.
