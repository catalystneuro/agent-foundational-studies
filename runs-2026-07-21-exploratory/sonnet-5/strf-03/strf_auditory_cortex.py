# %% [markdown]
# # Spectrotemporal Receptive Fields in Mouse Auditory Cortex
#
# This notebook demonstrates spectrotemporal receptive fields (STRFs) in primary
# auditory cortex (A1) using extracellular recordings from the DANDI Archive.
#
# **Dataset:** [DANDI:001419](https://dandiarchive.org/dandiset/001419) — "Cortical
# Circuits for the integration of short-latency auditory pathways: Auditory cortex
# linear probe recording" (Bhattacharya lab / DANDI). Mice were implanted with a
# linear multi-electrode probe in A1 and presented with pseudo-randomly ordered,
# pseudo-randomly timed pure-tone pips (10-18 log-spaced frequencies spanning
# 4-80 kHz, 300 ms duration, ~70 dB SPL, ~5 s inter-stimulus interval). A subset of
# trials also delivered optogenetic photostimulation of a converging pathway (an
# A2-projecting, ChrimsonR-expressing population); per the dataset's own
# documentation those LED trials are excluded here and only control (LED-off)
# trials are analyzed.
#
# **Method.** A spectrotemporal receptive field describes how a neuron's firing
# rate depends jointly on the frequency content of a sound and the time lag since
# that sound arrived. With single, well-separated tone pips (rather than a
# continuous dynamic stimulus), the STRF can be estimated directly: for each
# tested frequency, average the peri-stimulus time histogram (PSTH) across trials
# of that frequency, then stack the per-frequency PSTHs into a
# frequency x time-lag matrix. This is mathematically equivalent to reverse
# correlation against a stimulus spectrogram that is a train of narrowband delta
# functions, and is a standard way to characterize STRFs in electrophysiology
# (e.g. deCharms & Merzenich 1996; Linden et al. 2003).
#
# We build these STRFs for individual units, test each unit for significant
# frequency tuning, and summarize the tuned population.

# %%
import pickle

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap
import remfile
import requests
from pynwb import NWBHDF5IO
from scipy import stats
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

RNG_SESSIONS = {
    "sub-AK190207A_ses-tones1": "b651a329-009d-4b85-b08c-ce99de014c18",
    "sub-AK190207C_ses-tones1": "163f4aca-eeaf-4b72-9226-f45c45d1504d",
    "sub-AK190208_ses-tones1": "6a39d1a4-6226-4a6f-939a-74acc5e1d46e",
    "sub-AK190502A_ses-tones1": "37fb14d7-b392-47b8-add2-4081111aeb6b",
    "sub-AK190502B_ses-tones1": "4f150b2f-ef65-4e0e-9a15-3ce660d84dfb",
    "sub-AK190506_ses-tones1": "63152e12-669d-47e6-9d8c-68e7a8a15f24",
    "sub-AK210825B_ses-tones1": "f637f8b9-70b1-437d-8332-1a375d7c3cc5",
    "sub-HK190525_ses-tones1": "3f1c7fc1-fb5a-4a5f-98b5-92b3dcd01093",
}
DANDISET = "001419"
PRIMARY_SESSION = "sub-HK190525_ses-tones1"  # used for the raw-data and example-unit figures


def resolve_s3_url(dandiset, asset_id):
    """Resolve a DANDI asset ID to its underlying S3 URL via the archive API redirect."""
    r = requests.get(
        f"https://api.dandiarchive.org/api/dandisets/{dandiset}/versions/draft/assets/{asset_id}/download/",
        allow_redirects=False,
    )
    return r.headers["Location"]


def load_session(dandiset, asset_id, disk_cache):
    """Stream one NWB session from DANDI (remfile + disk cache) and wrap it with pynapple."""
    url = resolve_s3_url(dandiset, asset_id)
    rem_file = remfile.File(url, disk_cache=disk_cache)
    h5f = h5py.File(rem_file, "r")
    io = NWBHDF5IO(file=h5f)
    nwbfile = io.read()
    return nap.NWBFile(nwbfile), io


disk_cache = remfile.DiskCache("/tmp/remfile_cache")

# %% [markdown]
# ## 1. Load and Inspect a Single Session
#
# We start with one session to validate the data before scaling up. Pynapple
# exposes the NWB file's sorted units as a `TsGroup`, the tone-pip trial table as
# an `IntervalSet` with per-trial frequency metadata, and continuous streams
# (LFP, CSD) as `Tsd`/`TsdFrame` objects.

# %%
nwb, io = load_session(DANDISET, RNG_SESSIONS[PRIMARY_SESSION], disk_cache)
print(nwb)

units = nwb["units"]
trials = nwb["trials"]
print(f"\n{len(units)} sorted units, {len(trials)} tone-pip trials")
print(f"quality breakdown: {units.getby_category('quality').keys()}")
good = units.getby_category("quality")["good"]
print(f"{len(good)} 'good' (single) units")

led = trials.led_on_time.values
control_trials = trials[np.isnan(led)]
freqs_khz = np.sort(np.unique(control_trials.frequency.values)) / 1000
print(f"\n{len(control_trials)} control (LED-off) trials")
print(f"tested frequencies (kHz): {np.round(freqs_khz, 1)}")

# %% [markdown]
# ## 2. Validate the Raw Data
#
# Before computing anything, we plot the raw spike rasters together with the
# tone-pip stimulus timeline and the pooled population multi-unit spike density.
# Clear stimulus-locked bursts of activity confirm that these units are, in fact,
# sound-driven and that trial/spike timestamps are correctly aligned.

# %%
t0, t1 = control_trials.start[0] - 1, control_trials.start[0] + 60
ep = nap.IntervalSet(start=t0, end=t1)

fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, height_ratios=[2.2, 0.6, 1])

ax = axes[0]
sub_units = good.restrict(ep)
depth_order = np.argsort(good.get_info("y").values)
ids_sorted = good.index[depth_order]
for i, uid in enumerate(ids_sorted):
    if uid in sub_units.index and len(sub_units[uid]) > 0:
        ax.vlines(sub_units[uid].times(), i - 0.4, i + 0.4, color="k", lw=0.6)
ax.set_ylabel(f"Unit (n={len(ids_sorted)}, sorted by depth)")
ax.set_title(f"Sorted-unit raster, tone-pip stimulus, and population MUA ({PRIMARY_SESSION})")

ax2 = axes[1]
sub_trials = trials.intersect(ep)
freqs_all = np.sort(np.unique(trials.frequency.values))
cmap = plt.cm.viridis
for i in range(len(sub_trials)):
    fq = sub_trials.frequency[i]
    color = cmap((np.log2(fq) - np.log2(freqs_all[0])) / (np.log2(freqs_all[-1]) - np.log2(freqs_all[0])))
    ax2.axvspan(sub_trials.start[i], sub_trials.end[i], color=color, alpha=0.8)
ax2.set_ylabel("Tone pip\n(color = freq)")
ax2.set_yticks([])

ax3 = axes[2]
all_spikes = np.sort(
    np.concatenate([sub_units[uid].times() for uid in ids_sorted if uid in sub_units.index and len(sub_units[uid]) > 0])
)
mua_bins = np.arange(t0, t1, 0.02)
counts, _ = np.histogram(all_spikes, bins=mua_bins)
ax3.bar(mua_bins[:-1], counts / len(ids_sorted) / 0.02, width=0.02, color="k", align="edge")
ax3.set_ylabel("Pop. MUA (Hz/unit)")
ax3.set_xlabel("Time (s)")

plt.tight_layout()
plt.savefig("figures/01_raw_data_overview.png", dpi=140)
plt.close(fig)

# %% [markdown]
# Spikes cluster tightly after each tone pip across most units on the probe,
# confirming robust, stimulus-locked auditory responses.

# %% [markdown]
# ## 3. Build Spectrotemporal Receptive Fields
#
# For every "good"-quality unit in each session we compute a
# frequency x time-lag response matrix from -50 to +200 ms around tone onset
# (20 ms bins), using only control (LED-off) trials. We also record, for each
# unit, matched pre-/post-onset spike counts (per trial, pooled across
# frequencies) and per-frequency post-onset spike counts (per trial), which are
# used below for statistical testing.

# %%
PRE, POST, BINSIZE = 0.05, 0.20, 0.02
# round to avoid a floating-point artifact from np.arange that otherwise leaves the
# t=0 bin center at -1.7e-15 instead of exactly 0, silently dropping it from any
# `tlags >= 0` mask downstream.
BINS = np.round(np.arange(-PRE, POST + BINSIZE, BINSIZE), 8)
TLAGS = np.round(BINS[:-1] + BINSIZE / 2, 8)

POST_WIN = 0.10  # onset-response window (s) used for the frequency-tuning test and BF/latency estimation
RESP_WIN = 0.15  # matched pre/post window (s) used for the sound-onset responsiveness check


def build_strf(spk, trial_set, freqs):
    """Stack per-frequency PSTHs into a (n_freq, n_time_bins) receptive field matrix."""
    strf = np.zeros((len(freqs), len(TLAGS)))
    per_freq_post_counts = []
    for i, fq in enumerate(freqs):
        ep_starts = trial_set[trial_set.frequency == fq].start
        peh = nap.compute_perievent(spk, nap.Ts(t=ep_starts), window=(-PRE, POST))
        counts = np.zeros(len(TLAGS))
        trial_post = []
        for tr in peh.values():
            t = tr.times()
            h, _ = np.histogram(t, bins=BINS)
            counts += h
            trial_post.append(np.sum((t >= 0) & (t < POST_WIN)))
        strf[i] = counts / len(peh) / BINSIZE
        per_freq_post_counts.append(np.array(trial_post))
    return strf, per_freq_post_counts


def paired_pre_post_counts(spk, trial_set):
    """Per-trial spike counts in matched pre-/post-onset windows, pooled across frequency."""
    peh = nap.compute_perievent(spk, nap.Ts(t=trial_set.start), window=(-RESP_WIN, RESP_WIN))
    pre_counts, post_counts = [], []
    for tr in peh.values():
        t = tr.times()
        pre_counts.append(np.sum((t >= -RESP_WIN) & (t < 0)))
        post_counts.append(np.sum((t >= 0) & (t < RESP_WIN)))
    return np.array(pre_counts), np.array(post_counts)


def extract_session_strfs(session_name, asset_id):
    nwb_s, io_s = load_session(DANDISET, asset_id, disk_cache)
    units_s, trials_s = nwb_s["units"], nwb_s["trials"]
    if "led_on_time" in trials_s.metadata_columns:
        control_s = trials_s[np.isnan(trials_s.led_on_time.values)]
    else:
        control_s = trials_s
    freqs_s = np.sort(np.unique(control_s.frequency.values))

    good_s = units_s.getby_category("quality")["good"]
    info_s = good_s.get_info(["rate", "y", "location"])

    session_results = []
    for uid in good_s.index:
        spk = good_s[uid]
        strf, per_freq_post_counts = build_strf(spk, control_s, freqs_s)
        pre_c, post_c = paired_pre_post_counts(spk, control_s)
        session_results.append(
            {
                "session": session_name,
                "unit_id": int(uid),
                "overall_rate": float(info_s.loc[uid, "rate"]),
                "depth_y": float(info_s.loc[uid, "y"]),
                "location": info_s.loc[uid, "location"],
                "freqs": freqs_s,
                "tlags": TLAGS,
                "strf": strf,
                "pre_counts": pre_c,
                "post_counts": post_c,
                "per_freq_post_counts": per_freq_post_counts,
            }
        )
    io_s.close()
    return session_results


# %% [markdown]
# ## 4. Scale to Multiple Sessions
#
# We repeat the extraction above across 8 recording sessions from 8 different
# mice to build a population of units for robust statistics.

# %%
results = []
for sess_name, asset_id in tqdm(RNG_SESSIONS.items(), desc="sessions"):
    results.extend(extract_session_strfs(sess_name, asset_id))

print(f"\nTotal good-quality units collected: {len(results)}")

with open("strf_results.pkl", "wb") as f:
    pickle.dump(results, f)

# %% [markdown]
# ## 5. Quantify Tuning, Best Frequency, and Response Latency
#
# For each unit we:
# - Estimate a robust baseline firing rate from the matched pre-onset window,
#   pooled across all control trials (more stable than averaging just the two
#   pre-stimulus bins of the STRF matrix, which draws on far fewer spikes).
# - Locate the peak of the onset response (0-100 ms) to obtain a best frequency
#   (BF) and response latency.
# - Test for genuine spectral tuning with a Kruskal-Wallis test on per-trial,
#   per-frequency spike counts in the 0-100 ms window (i.e., does firing rate
#   depend on tone frequency, beyond trial-to-trial noise?).

# %%
tlags = results[0]["tlags"]
post_mask = (tlags >= 0) & (tlags <= 0.10)

for r in results:
    freqs = r["freqs"]
    strf = r["strf"]
    baseline_rate = r["pre_counts"].mean() / RESP_WIN
    r["baseline_rate"] = baseline_rate
    r["strf_rel"] = strf - baseline_rate

    post = strf[:, post_mask]
    peak_idx = np.unravel_index(np.argmax(post), post.shape)
    r["bf_idx"] = peak_idx[0]
    r["bf_hz"] = freqs[peak_idx[0]]
    r["latency_ms"] = tlags[post_mask][peak_idx[1]] * 1000
    r["peak_rate"] = post[peak_idx]
    r["peak_excess_rate"] = r["peak_rate"] - baseline_rate

    groups = r["per_freq_post_counts"]
    if all(g.sum() == 0 for g in groups):
        p = 1.0
    else:
        try:
            _, p = stats.kruskal(*groups)
        except ValueError:
            p = 1.0
    r["kw_pval"] = p
    r["tuned"] = p < 0.05

n_tuned = sum(r["tuned"] for r in results)
n = len(results)
bt = stats.binomtest(n_tuned, n, 0.05, alternative="greater")
print(f"{n_tuned} / {n} units show significant frequency tuning (Kruskal-Wallis, p<0.05)")
print(f"expected by chance at alpha=0.05: {0.05*n:.1f} units")
print(f"binomial test vs. 5% chance level: p = {bt.pvalue:.2e}")

tuned_results = [r for r in results if r["tuned"]]
lat_arr = np.array([r["latency_ms"] for r in tuned_results])
bf_arr = np.array([r["bf_hz"] for r in tuned_results])
print(f"\ntuned-unit latency: median {np.median(lat_arr):.0f} ms, range {lat_arr.min():.0f}-{lat_arr.max():.0f} ms")
print(f"tuned-unit BF range: {bf_arr.min()/1000:.1f}-{bf_arr.max()/1000:.1f} kHz")

with open("strf_analyzed.pkl", "wb") as f:
    pickle.dump(results, f)

# %% [markdown]
# ## 6. Example Single-Unit Spectrotemporal Receptive Fields
#
# We display the six most significantly frequency-tuned units. Each panel is a
# heatmap of firing rate as a function of tone frequency (y-axis, log-spaced)
# and time lag from tone onset (x-axis); a small Gaussian smoothing kernel
# (applied only for display) makes the structure easier to see. The dashed line
# marks tone onset and the dotted line marks tone offset (300 ms duration).

# %%
tuned_sorted = sorted(tuned_results, key=lambda r: r["kw_pval"])
selected = tuned_sorted[:6]

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, r in zip(axes.flat, selected):
    strf = r["strf"]
    freqs = r["freqs"]
    tlags_ms = r["tlags"] * 1000
    smoothed = gaussian_filter(strf, sigma=(0.7, 0.4))
    im = ax.imshow(
        smoothed, aspect="auto", origin="lower", cmap="inferno",
        extent=[tlags_ms[0] - 10, tlags_ms[-1] + 10, 0, len(freqs)],
    )
    ax.set_yticks(np.arange(len(freqs)) + 0.5)
    ax.set_yticklabels([f"{f/1000:.1f}" for f in freqs], fontsize=7)
    ax.axvline(0, color="w", ls="--", lw=1)
    ax.axvline(300, color="w", ls=":", lw=1)
    ax.set_title(
        f"{r['session'].split('_')[0]} unit {r['unit_id']}\n"
        f"BF={r['bf_hz']/1000:.1f} kHz, lat={r['latency_ms']:.0f} ms, p={r['kw_pval']:.1e}",
        fontsize=9,
    )
    ax.set_xlabel("Time from tone onset (ms)", fontsize=8)
    ax.set_ylabel("Frequency (kHz)", fontsize=8)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Firing rate (Hz)", fontsize=7)
    cb.ax.tick_params(labelsize=6)

fig.suptitle("Example single-unit spectrotemporal receptive fields (mouse A1, tone-pip reverse correlation)", fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("figures/02_example_strfs.png", dpi=140)
plt.close(fig)

# %% [markdown]
# Each of these units shows a clear excitatory hot spot at a specific frequency
# and a short latency (0-60 ms) after tone onset, with little or no response to
# frequencies more than an octave or two away — the defining signature of a
# spectrotemporal receptive field.

# %% [markdown]
# ## 7. Population Summary
#
# Four panels: (a) the fraction of units showing significant frequency tuning,
# compared to the chance rate expected under the null; (b) the distribution of
# best frequencies across tuned units; (c) the distribution of onset-response
# latencies; and (d) a population-average STRF built by aligning every tuned
# unit's frequency axis to its own best frequency (expressed in octaves) and
# averaging the baseline-subtracted response. This "BF-aligned" average reveals
# the canonical shape of the auditory STRF -- a short-latency excitatory center
# at 0 octaves that falls off with spectral distance from BF -- without being
# obscured by the fact that different units prefer different absolute
# frequencies.

# %%
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

ax = axes[0, 0]
observed_pct = n_tuned / n * 100
chance_pct = 5.0
ax.bar(["chance level\n(alpha=0.05)", "observed"], [chance_pct, observed_pct], color=["gray", "C1"])
ax.set_ylabel("% of units with significant\nfrequency tuning (Kruskal-Wallis)")
ax.set_title(f"Sound-frequency tuning across population\nn={n_tuned}/{n} units, p={bt.pvalue:.1e} (binomial test)")
for i, v in enumerate([chance_pct, observed_pct]):
    ax.text(i, v + 0.5, f"{v:.1f}%", ha="center")

ax = axes[0, 1]
bfs_khz = bf_arr / 1000
all_freqs_khz = np.unique(np.concatenate([r["freqs"] for r in results])) / 1000
log_bfs = np.log2(bfs_khz)
edges = np.log2(np.sort(np.unique(np.round(all_freqs_khz, 3))))
edges = np.concatenate([edges - 0.25, [edges[-1] + 0.25]])
ax.hist(log_bfs, bins=sorted(set(edges)), color="C0", edgecolor="k")
xt = [4, 8, 16, 32, 64]
ax.set_xticks(np.log2(xt))
ax.set_xticklabels(xt)
ax.set_xlabel("Best frequency (kHz)")
ax.set_ylabel("Number of tuned units")
ax.set_title("Best-frequency distribution\n(tested range 4-80 kHz)")

ax = axes[1, 0]
ax.hist(lat_arr, bins=np.arange(-10, 110, 10), color="C2", edgecolor="k")
ax.set_xlabel("Response latency (ms from tone onset)")
ax.set_ylabel("Number of tuned units")
ax.set_title("Onset-response latency distribution")
ax.axvline(np.median(lat_arr), color="k", ls="--", label=f"median={np.median(lat_arr):.0f} ms")
ax.legend()

# BF-aligned, octave-relative population-average STRF
pooled_vals = np.concatenate([r["strf_rel"].ravel() for r in tuned_results])
clip_lo, clip_hi = np.percentile(pooled_vals, [5, 95])

octave_bins = np.arange(-2.25, 2.26, 0.5)
octave_centers = (octave_bins[:-1] + octave_bins[1:]) / 2
tlags_full_ms = results[0]["tlags"] * 1000
time_keep = tlags_full_ms >= 0  # drop the noisy single pre-stimulus bin (n=9 trials/frequency)
tlags_ms = tlags_full_ms[time_keep]
accum = np.zeros((len(octave_centers), len(tlags_ms)))
counts_mat = np.zeros((len(octave_centers), len(tlags_ms)))
for r in tuned_results:
    freqs = r["freqs"]
    strf_rel = np.clip(r["strf_rel"][:, time_keep], clip_lo, clip_hi)
    octaves = np.log2(freqs / r["bf_hz"])
    for fi, oct_off in enumerate(octaves):
        bin_idx = np.digitize(oct_off, octave_bins) - 1
        if 0 <= bin_idx < len(octave_centers):
            accum[bin_idx] += strf_rel[fi]
            counts_mat[bin_idx] += 1

min_n = 3
grand_strf = np.divide(accum, counts_mat, out=np.full_like(accum, np.nan), where=counts_mat >= min_n)

ax = axes[1, 1]
cmap = plt.cm.RdBu_r.copy()
cmap.set_bad("lightgray")
vmax = np.nanmax(np.abs(grand_strf))
im = ax.imshow(
    grand_strf, aspect="auto", origin="lower", cmap=cmap,
    extent=[tlags_ms[0] - 10, tlags_ms[-1] + 10, octave_bins[0], octave_bins[-1]],
    vmin=-vmax, vmax=vmax,
)
ax.set_xlim(tlags_ms[0] - 10, 220)
ax.axhline(0, color="k", ls=":", lw=1)
ax.axvline(0, color="k", ls="--", lw=1)
ax.set_xlabel("Time from tone onset (ms)")
ax.set_ylabel("Octaves from best frequency")
ax.set_title(f"Population-average STRF, aligned to each unit's BF\n(n={n_tuned} tuned units; gray = fewer than {min_n} units contributing)")
plt.colorbar(im, ax=ax, label="Excess firing rate (Hz, above baseline)")

plt.tight_layout()
plt.savefig("figures/03_population_summary.png", dpi=140)
plt.close(fig)

# %% [markdown]
# ## Conclusion
#
# Across 266 well-isolated units recorded from mouse primary auditory cortex
# (8 sessions, 8 mice), 30 units (11.3%) show statistically significant
# frequency-dependent firing in the 0-100 ms window following tone onset — more
# than double the ~5% expected by chance under the significance threshold alone
# (binomial test, p = 3.3e-05). Individual example units show clean, sharply
# frequency-tuned excitatory responses with short latencies (median 40 ms,
# consistent with known A1 onset latencies), while the BF-aligned population
# average recovers the canonical spectrotemporal receptive-field shape: a
# short-latency excitatory peak centered exactly at each unit's own best
# frequency that falls off with spectral distance. Together these results
# demonstrate genuine spectrotemporal receptive field structure in mouse
# auditory cortex, estimated directly from randomized tone-pip presentations via
# reverse correlation.
