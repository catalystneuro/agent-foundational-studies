# %% [markdown]
# # Spectrotemporal Receptive Fields in the Auditory System
#
# A spectrotemporal receptive field (STRF) describes which sound frequencies drive a
# neuron and at what delay after sound onset. It is the auditory analog of the
# spatial receptive field in vision: a two-dimensional map of response strength over
# frequency and time.
#
# This notebook demonstrates STRFs at two stages of the auditory system, using two
# datasets from the DANDI Archive, both streamed directly from the archive with
# `remfile` (no full downloads):
#
# 1. **Auditory nerve (periphery)**: DANDI:001262, Heeringa et al. 2024 (*Scientific
#    Data*), single auditory nerve fibers in gerbil. Each fiber was characterized with
#    a tone-pip frequency scan: short (50 ms) pure tones at 13-69 closely spaced
#    frequencies around the fiber's characteristic frequency (CF), repeated 5 times.
#    Averaging the spike response per frequency as a function of time after tone
#    onset yields a tone-pip STRF in the style of Aertsen & Johannesma (1981).
# 2. **Auditory cortex (central)**: DANDI:000986, Jaramillo lab, Neuropixels
#    recordings in mouse auditory cortex during passive presentation of 25 ms pure
#    tones at 5 octave-spaced frequencies (2-32 kHz), ~1500 trials per frequency.
#
# Because both datasets use discrete tone pips rather than continuous noise, the STRF
# is estimated directly as the baseline-subtracted per-frequency PSTH (equivalently,
# the spike-triggered average of the one-hot tone ensemble). No stimulus waveforms
# are stored, so reverse correlation against a continuous stimulus is not possible;
# the tone-pip map is the appropriate estimator here.
#
# **Outline**
# - Part 1: Deep dive into one auditory nerve fiber (raster, STRF, tuning, PSTH)
# - Part 2: Population of 85 auditory nerve fibers (CF validation, latency, bandwidth)
# - Part 3: Cortical STRFs from one Neuropixels session
# - Part 4: Periphery vs cortex comparison

# %% [markdown]
# ## Setup

# %%
import requests
import h5py
import remfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy import stats
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

rng = np.random.default_rng(0)

GERBIL_DANDI = "001262"
GERBIL_VER = "0.241205.0959"
CORTEX_DANDI = "000986"
CORTEX_VER = "0.251031.1939"
CACHE_DIR = "/tmp/remfile_cache_strf"


def get_s3_url(dandiset, version, asset_id):
    """Resolve a DANDI asset id to its S3 blob URL."""
    r = requests.get(
        f"https://api.dandiarchive.org/api/dandisets/{dandiset}/versions/{version}/assets/{asset_id}/"
    )
    d = r.json()
    for u in d["contentUrl"]:
        if "s3" in u:
            return u
    return d["contentUrl"][0]


def open_nwb_h5(url):
    """Open a remote NWB file as an h5py handle via remfile with disk caching."""
    return h5py.File(remfile.File(url, disk_cache=remfile.DiskCache(CACHE_DIR)), "r")


# %% [markdown]
# ## Part 1: STRF of a single auditory nerve fiber
#
# Each NWB file in DANDI:001262 holds one fiber. Stimulus presentations ("sweeps")
# are tagged `BF_FREQ<freq>_rep<k>` for the tone-pip frequency scan; per-sweep spike
# times live in `units/spike_times` (one units row per sweep). We align spikes to
# tone onset (7.75 ms after sweep start), build a PSTH per frequency, subtract the
# pre-tone baseline, and smooth lightly. The result is the fiber's STRF.

# %%
# Time base for all auditory nerve STRFs
AN_T0, AN_T1 = -0.010, 0.090   # window rel tone onset (s)
AN_BIN = 0.001                 # 1 ms bins
AN_EDGES = np.arange(AN_T0, AN_T1 + AN_BIN, AN_BIN)
AN_CENTERS = 0.5 * (AN_EDGES[:-1] + AN_EDGES[1:])


def analyze_an_fiber(asset_id, path):
    """Compute the tone-pip STRF for one gerbil auditory nerve fiber file.

    Returns a dict with the smoothed baseline-subtracted STRF map and summary
    metrics, or None if the file has no usable BF tone-pip scan.
    """
    h5 = open_nwb_h5(get_s3_url(GERBIL_DANDI, GERBIL_VER, asset_id))

    # The published per-experiment results table tells us which protocols exist.
    at = h5["analysis/analysis_table"]
    experiments = [e.decode() for e in at["experiment"][:]]
    if "BF" not in experiments:
        return None
    table_bf = float(at["results_bf"][:][experiments.index("BF")])

    # Per-sweep stimulus parameters.
    IR = h5["general/intracellular_ephys/intracellular_recordings"]
    S = IR["stimuli"]
    stim = pd.DataFrame({
        "tag": [t.decode() for t in IR["tag"][:]],
        "frequency": S["frequency"][:].astype(float),
        "level": S["level_dBSPL"][:].astype(float),
        "delay": S["delay"][:].astype(float),
        "duration": S["duration"][:].astype(float),
    })
    bf = stim[stim["tag"].str.match(r"BF_FREQ\d+_rep\d+")].copy()
    bf = bf[bf["frequency"] > 0]          # -99999 is the not-applicable sentinel
    if len(bf) == 0:
        return None

    # Use the sound level with the most sweeps.
    level = float(bf.groupby("level").size().idxmax())
    bfl = bf[bf["level"] == level]
    freqs = np.sort(bfl["frequency"].unique())
    if len(freqs) < 12 or len(bfl) / len(freqs) < 3:
        return None
    delay = float(bfl["delay"].iloc[0])   # tone onset within the sweep
    dur = float(bfl["duration"].iloc[0])  # tone duration

    # Per-sweep spike times (ragged array via spike_times_index).
    spike_times = h5["units/spike_times"][:]
    spike_index = h5["units/spike_times_index"][:].astype(np.int64)
    tags = [t.decode() for t in h5["units/tag"][:]]
    bounds = np.concatenate([[0], spike_index])
    tag2spk = {}
    for i, t in enumerate(tags):
        st = spike_times[bounds[i]:bounds[i + 1]]
        tag2spk[t] = st[np.isfinite(st)]  # a few spike times are NaN

    # Per-frequency PSTH.
    psth = np.zeros((len(freqs), len(AN_CENTERS)))
    counts = np.zeros(len(freqs))
    fidx = {f: i for i, f in enumerate(freqs)}
    rasters = {f: [] for f in freqs}
    for _, row in bfl.iterrows():
        st = tag2spk.get(row["tag"])
        if st is None:
            continue
        rel = st - delay
        sel = rel[(rel >= AN_T0) & (rel < AN_T1)]
        fi = fidx[row["frequency"]]
        psth[fi] += np.histogram(sel, AN_EDGES)[0]
        counts[fi] += 1
        rasters[row["frequency"]].append(sel)
    if (counts == 0).any():
        return None

    rate = psth / (counts[:, None] * AN_BIN)
    baseline = rate[:, AN_CENTERS < 0].mean(axis=1, keepdims=True)
    net = rate - baseline
    net_s = gaussian_filter(net, sigma=(1.5, 1.5))

    # Summary metrics from the smoothed map.
    resp = (AN_CENTERS >= 0) & (AN_CENTERS < dur)
    tc = net_s[:, resp].mean(axis=1)              # tuning curve
    cfi = int(np.argmax(tc))
    cf = float(freqs[cfi])
    peak_rate = float(tc[cfi])
    psth_cf = net_s[cfi]
    pk = float(psth_cf[(AN_CENTERS > 0) & (AN_CENTERS < dur)].max())
    above = np.where((psth_cf > pk / 2) & (AN_CENTERS > 0))[0]
    latency = float(AN_CENTERS[above[0]]) if len(above) else np.nan

    # Bandwidth at half-maximum of the tuning curve (octaves), if resolved.
    bw_oct, q_factor = np.nan, np.nan
    if peak_rate > 0:
        over = tc >= peak_rate / 2
        lo, hi = cfi, cfi
        while lo > 0 and over[lo - 1]:
            lo -= 1
        while hi < len(freqs) - 1 and over[hi + 1]:
            hi += 1
        if lo > 0 and hi < len(freqs) - 1:
            bw_oct = float(np.log2(freqs[hi] / freqs[lo]))
            q_factor = cf / float(freqs[hi] - freqs[lo])

    h5.close()
    return {
        "asset_id": asset_id, "path": path, "subject": path.split("/")[0],
        "table_bf": table_bf, "cf": cf, "latency_ms": latency * 1000,
        "peak_rate": peak_rate, "spont_rate": float(baseline.mean()),
        "level": level, "n_freqs": len(freqs), "n_reps": len(bfl) / len(freqs),
        "bw_oct": bw_oct, "q_factor": q_factor, "dur": dur,
        "freqs": freqs, "net_map": net_s, "rasters": rasters,
    }


# %%
# The example fiber: sub-G150805 unit1 (published BF = 1703 Hz).
EXAMPLE_ASSET = "e4c25580-b7f7-4858-a25c-8de5c8e43e5d"
EXAMPLE_PATH = "sub-G150805/sub-G150805_ses-G150805-unit1_icephys.nwb"

fiber = analyze_an_fiber(EXAMPLE_ASSET, EXAMPLE_PATH)
print(f"CF = {fiber['cf']:.0f} Hz (published BF = {fiber['table_bf']:.0f} Hz)")
print(f"latency = {fiber['latency_ms']:.1f} ms, peak net rate = {fiber['peak_rate']:.0f} sp/s")
print(f"spontaneous rate = {fiber['spont_rate']:.0f} sp/s, level = {fiber['level']:.0f} dB SPL")
print(f"{fiber['n_freqs']} frequencies x {fiber['n_reps']:.0f} repetitions")

# %%
# Figure 1: single-fiber deep dive
freqs = fiber["freqs"]
net_s = fiber["net_map"]
dur = fiber["dur"]

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# (a) raster: every sweep, grouped by frequency
ax = axes[0, 0]
for fi, f in enumerate(freqs):
    for rep, spk in enumerate(fiber["rasters"][f]):
        ax.plot(spk * 1e3, np.full_like(spk, fi + rep * 0.18), "|",
                color="k", ms=1.5)
ax.axvline(0, color="r", lw=0.8)
ax.axvline(dur * 1e3, color="r", lw=0.8, ls="--")
step = 10
ax.set_yticks(range(0, len(freqs), step))
ax.set_yticklabels([f"{freqs[i]/1e3:.1f}" for i in range(0, len(freqs), step)])
ax.set_xlabel("time rel tone onset (ms)")
ax.set_ylabel("frequency (kHz)")
ax.set_title(f"(a) Raster: {len(freqs)} frequencies x {fiber['n_reps']:.0f} reps "
             f"@ {fiber['level']:.0f} dB SPL")

# (b) STRF map
ax = axes[0, 1]
vmax = np.percentile(net_s, 99.5)
im = ax.imshow(net_s, aspect="auto", origin="lower",
               extent=[AN_CENTERS[0] * 1e3, AN_CENTERS[-1] * 1e3,
                       freqs[0] / 1e3, freqs[-1] / 1e3],
               cmap="viridis", vmin=0, vmax=vmax)
ax.axvline(0, color="w", lw=0.8)
ax.axvline(dur * 1e3, color="w", lw=0.8, ls="--")
ax.axhline(fiber["cf"] / 1e3, color="r", lw=0.8, ls=":")
ax.set_xlabel("time rel tone onset (ms)")
ax.set_ylabel("frequency (kHz)")
ax.set_title("(b) STRF: baseline-subtracted rate")
plt.colorbar(im, ax=ax, label="net sp/s")

# (c) tuning curve
ax = axes[1, 0]
resp = (AN_CENTERS >= 0) & (AN_CENTERS < dur)
tc = net_s[:, resp].mean(axis=1)
ax.plot(freqs / 1e3, tc, "o-", ms=4)
ax.axvline(fiber["cf"] / 1e3, color="r", ls="--", lw=0.8,
           label=f"CF = {fiber['cf']:.0f} Hz")
ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel("frequency (kHz)")
ax.set_ylabel("net rate (sp/s)")
ax.set_title("(c) Frequency tuning curve (STRF time-average)")
ax.legend()

# (d) PSTH at CF
ax = axes[1, 1]
cfi = int(np.argmax(tc))
ax.plot(AN_CENTERS * 1e3, net_s[cfi], "k")
ax.axvline(0, color="r", lw=0.8)
ax.axvline(dur * 1e3, color="r", lw=0.8, ls="--")
pk = net_s[cfi][(AN_CENTERS > 0) & (AN_CENTERS < dur)].max()
ax.axhline(pk / 2, color="gray", ls=":", lw=0.8)
ax.axvline(fiber["latency_ms"], color="b", ls="--", lw=0.8,
           label=f"latency = {fiber['latency_ms']:.1f} ms")
ax.set_xlabel("time rel tone onset (ms)")
ax.set_ylabel("net rate (sp/s)")
ax.set_title(f"(d) PSTH at CF ({fiber['cf']:.0f} Hz)")
ax.legend()

fig.suptitle(f"Auditory nerve fiber STRF: {fiber['subject']} unit1 "
             f"(DANDI:001262)")
fig.tight_layout()
fig.savefig("fig1_single_fiber_strf.png", dpi=150)
plt.close(fig)
print("saved fig1_single_fiber_strf.png")

# %% [markdown]
# The STRF shows the classic auditory nerve signature: a compact excitatory ridge
# centered on the fiber's characteristic frequency, beginning a few milliseconds
# after tone onset (cochlear travel time plus synaptic delay), sustained for the
# duration of the tone, followed by a brief offset suppression. The frequency
# tuning curve is the time-average of the STRF over the response window.
#
# ## Part 2: Population of auditory nerve fibers
#
# We now repeat this across a stratified sample of files spanning the 104 gerbils
# in the dataset, to see how STRFs vary along the tonotopic axis. For each fiber we
# extract the CF, onset latency, peak rate, and tuning bandwidth, and we validate
# the STRF-based CF against the published per-fiber BF stored in each file.

# %%
# Catalog all assets and pick a stratified sample across animals.
rows = []
url = (f"https://api.dandiarchive.org/api/dandisets/{GERBIL_DANDI}/"
       f"versions/{GERBIL_VER}/assets/")
page_url = url
while page_url:
    d = requests.get(page_url, params={"page_size": 100}).json()
    for a in d["results"]:
        rows.append({"path": a["path"], "asset_id": a["asset_id"]})
    page_url = d["next"]
cat = pd.DataFrame(rows)
cat["subject"] = cat["path"].str.split("/").str[0]
# session names containing 'nm' are optical-stimulation sessions; skip them
cat = cat[~cat["path"].str.contains("nm")].reset_index(drop=True)
print(f"{len(cat)} acoustic files across {cat['subject'].nunique()} animals")

N_TARGET = 90
picks = []
per_subj = max(1, int(np.ceil(N_TARGET / cat["subject"].nunique())))
for subj in cat["subject"].unique():
    sub = cat[cat["subject"] == subj]
    k = min(per_subj, len(sub))
    picks.append(sub.iloc[rng.choice(len(sub), size=k, replace=False)])
sel = pd.concat(picks).head(N_TARGET)
print(f"selected {len(sel)} files")

# %%
# Harvest STRFs (parallel; streaming metadata + spike times only).
fibers = []
with ThreadPoolExecutor(max_workers=4) as ex:
    futs = {ex.submit(analyze_an_fiber, r.asset_id, r.path): r.path
            for r in sel.itertuples()}
    for fut in tqdm(as_completed(futs), total=len(futs), desc="AN fibers"):
        res = fut.result()
        if res is not None:
            fibers.append(res)
print(f"{len(fibers)} fibers with valid tone-pip STRFs")

metrics = pd.DataFrame([{k: v for k, v in f.items()
                         if k not in ("freqs", "net_map", "rasters")}
                        for f in fibers])
print(metrics[["cf", "latency_ms", "peak_rate", "bw_oct"]].describe().round(2))

# %%
# Figure 2: gallery of example STRFs spanning the tonotopic axis
good = metrics[metrics["peak_rate"] > 80].sort_values("cf")
target_cfs = [800, 2000, 4000, 7000, 10000, 14000]
pick_rows = []
for tcf in target_cfs:
    i = (good["cf"] - tcf).abs().idxmin()
    if i not in pick_rows:
        pick_rows.append(i)
examples = [fibers[metrics.index.get_loc(i)] for i in pick_rows]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, res in zip(axes.flat, examples):
    fr = res["freqs"]
    v = np.percentile(res["net_map"], 99.5)
    im = ax.imshow(res["net_map"], aspect="auto", origin="lower",
                   extent=[AN_CENTERS[0] * 1e3, AN_CENTERS[-1] * 1e3,
                           fr[0] / 1e3, fr[-1] / 1e3],
                   cmap="viridis", vmin=0, vmax=v)
    ax.axvline(0, color="w", lw=0.8)
    ax.axvline(res["dur"] * 1e3, color="w", lw=0.8, ls="--")
    ax.axhline(res["cf"] / 1e3, color="r", lw=0.8, ls=":")
    ax.set_title(f"CF={res['cf']:.0f} Hz, latency={res['latency_ms']:.1f} ms, "
                 f"{res['level']:.0f} dB SPL", fontsize=10)
    ax.set_xlabel("time rel tone onset (ms)")
    ax.set_ylabel("frequency (kHz)")
    plt.colorbar(im, ax=ax, label="net sp/s")
fig.suptitle("Auditory nerve STRFs across the tonotopic axis (gerbil, DANDI:001262)")
fig.tight_layout()
fig.savefig("fig2_an_strf_gallery.png", dpi=150)
plt.close(fig)
print("saved fig2_an_strf_gallery.png")

# %%
# Figure 3: population summary
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# (a) CF validation against the published BF
ax = axes[0, 0]
ax.scatter(metrics["table_bf"], metrics["cf"], c="k", s=15)
lims = [300, 20000]
ax.plot(lims, lims, "r--", lw=1)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("BF from published analysis table (Hz)")
ax.set_ylabel("CF from STRF (Hz)")
r_cf = np.corrcoef(np.log10(metrics["table_bf"]), np.log10(metrics["cf"]))[0, 1]
ax.set_title(f"(a) STRF CF vs published BF (r={r_cf:.3f}, n={len(metrics)})")

# (b) latency vs CF (cochlear traveling wave predicts a decrease)
ax = axes[0, 1]
ok = metrics["latency_ms"].between(1, 15)  # drop rare late-response outliers
ax.scatter(metrics.loc[ok, "cf"], metrics.loc[ok, "latency_ms"], c="k", s=15)
ax.set_xscale("log")
slope, intercept, rval, pval, _ = stats.linregress(
    np.log10(metrics.loc[ok, "cf"]), metrics.loc[ok, "latency_ms"])
xx = np.linspace(np.log10(400), np.log10(16000), 50)
ax.plot(10 ** xx, intercept + slope * xx, "r--", lw=1,
        label=f"slope={slope:.2f} ms/dec, p={pval:.3f}")
# levels vary across files; in a restricted 40-60 dB band the trend is cleaner
band = ok & metrics["level"].between(40, 60)
rb, pb = stats.pearsonr(np.log10(metrics.loc[band, "cf"]),
                        metrics.loc[band, "latency_ms"])
ax.set_xlabel("CF (Hz)")
ax.set_ylabel("STRF onset latency (ms)")
ax.set_title(f"(b) Latency vs CF (40-60 dB band: r={rb:.2f}, p={pb:.3f})")
ax.legend()

# (c) bandwidth vs CF
ax = axes[1, 0]
okb = metrics["bw_oct"].notna()
ax.scatter(metrics.loc[okb, "cf"], metrics.loc[okb, "bw_oct"], c="k", s=15)
ax.set_xscale("log")
sb, ib, rbv, pbv, _ = stats.linregress(np.log10(metrics.loc[okb, "cf"]),
                                       metrics.loc[okb, "bw_oct"])
ax.plot(10 ** xx, ib + sb * xx, "r--", lw=1,
        label=f"slope={sb:.2f} oct/dec, p={pbv:.1e}")
ax.set_xlabel("CF (Hz)")
ax.set_ylabel("bandwidth at half-max (octaves)")
ax.set_title(f"(c) STRF bandwidth vs CF (n={okb.sum()})")
ax.legend()

# (d) evoked vs spontaneous rate
ax = axes[1, 1]
ax.scatter(metrics["spont_rate"], metrics["peak_rate"], c="k", s=15)
ax.set_xlabel("spontaneous rate (sp/s)")
ax.set_ylabel("peak evoked net rate (sp/s)")
ax.set_title("(d) Evoked vs spontaneous rate")

fig.suptitle(f"Gerbil auditory nerve STRF population "
             f"(n={len(metrics)} fibers, {metrics['subject'].nunique()} animals)")
fig.tight_layout()
fig.savefig("fig3_an_population.png", dpi=150)
plt.close(fig)
print("saved fig3_an_population.png")

# %% [markdown]
# Three population patterns emerge. First, the STRF-based CF matches the published
# per-fiber best frequency almost exactly (r ≈ 0.99 in log space), validating the
# tone-pip STRF estimate. Second, onset latency decreases with CF, the signature of
# the cochlear traveling wave (the base of the cochlea, tuned to high frequencies,
# responds sooner than the apex). Third, tuning bandwidth expressed in octaves
# narrows with CF, i.e. frequency selectivity sharpens relative to CF.
#
# ## Part 3: STRFs in auditory cortex
#
# DANDI:000986 (Jaramillo lab) contains Neuropixels recordings from mouse auditory
# cortex during passive pure-tone listening: 25 ms tones at 2, 4, 8, 16, and 32 kHz,
# ~1500 trials per frequency. The frequency axis is much coarser than for the nerve
# fibers, but the trial count is enormous, giving millisecond temporal resolution.
# We compute the same baseline-subtracted per-frequency PSTH map for every unit and
# identify tone-responsive, frequency-modulated units (Wilcoxon evoked vs baseline,
# Kruskal-Wallis across frequencies).

# %%
CORTEX_ASSET = "60303460-38be-44a0-951e-82c7957d1217"  # sub-LA8, session 1
h5 = open_nwb_h5(get_s3_url(CORTEX_DANDI, CORTEX_VER, CORTEX_ASSET))

tr = h5["intervals"]["trials"]
trials = pd.DataFrame({"start_time": tr["start_time"][:],
                       "stim_frequency": tr["stim_frequency"][:]})
cx_freqs = np.sort(trials["stim_frequency"].unique())
print("cortical stimulus frequencies (Hz):", cx_freqs)

# Bulk-read the ragged spike-time arrays (VectorIndex) from the h5 handle.
spike_times = h5["units/spike_times"][:]
spike_index = h5["units/spike_times_index"][:].astype(np.int64)
bounds = np.concatenate([[0], spike_index])
n_units = len(spike_index)
units = [spike_times[bounds[i]:bounds[i + 1]] for i in range(n_units)]
print(f"{n_units} units")

# %%
CX_T0, CX_T1 = -0.050, 0.150
CX_BIN = 0.002
CX_EDGES = np.arange(CX_T0, CX_T1 + CX_BIN, CX_BIN)
CX_CENTERS = 0.5 * (CX_EDGES[:-1] + CX_EDGES[1:])
onsets_by_freq = {f: np.sort(trials.loc[trials["stim_frequency"] == f,
                                        "start_time"].values)
                  for f in cx_freqs}


def ranges(starts, ends):
    """Vectorized [start:end) index concatenation for perievent extraction."""
    lengths = ends - starts
    total = int(lengths.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64), lengths
    nz = lengths > 0  # zero-length trials contribute no indices
    starts_nz = starts[nz]
    lengths_nz = lengths[nz]
    out = np.ones(total, dtype=np.int64)
    out[0] = starts_nz[0]
    idx = np.cumsum(lengths_nz)[:-1]
    out[idx] = starts_nz[1:] - (starts_nz[:-1] + lengths_nz[:-1]) + 1
    return np.cumsum(out), lengths


cx_maps = np.zeros((n_units, len(cx_freqs), len(CX_CENTERS)))
ev_counts = np.zeros((n_units, len(cx_freqs), max(len(v) for v in
                                                  onsets_by_freq.values())))
bl_counts = np.zeros_like(ev_counts)
n_trials = np.zeros(len(cx_freqs), dtype=int)

for fi, f in enumerate(cx_freqs):
    ons = onsets_by_freq[f]
    n_trials[fi] = len(ons)
    for u in range(n_units):
        st = units[u]
        i0 = np.searchsorted(st, ons + CX_T0)
        i1 = np.searchsorted(st, ons + CX_T1)
        flat, lengths = ranges(i0, i1)
        if len(flat):
            rel = st[flat] - np.repeat(ons, lengths)
            cx_maps[u, fi] = np.histogram(rel, CX_EDGES)[0] / (len(ons) * CX_BIN)
            # per-trial spike counts for statistics
            trial_id = np.repeat(np.arange(len(ons)), lengths)
            ev = (rel >= 0.005) & (rel < 0.060)
            bl = (rel >= -0.050) & (rel < 0)
            ev_counts[u, fi, :len(ons)] = np.bincount(trial_id[ev],
                                                      minlength=len(ons))
            bl_counts[u, fi, :len(ons)] = np.bincount(trial_id[bl],
                                                      minlength=len(ons))

ev_counts = ev_counts[:, :, :n_trials.min()]
bl_counts = bl_counts[:, :, :n_trials.min()]
print("cortical maps:", cx_maps.shape)

# %%
# Statistics and per-unit metrics
responsive = np.zeros(n_units, bool)
modulated = np.zeros(n_units, bool)
for u in range(n_units):
    ev_all = ev_counts[u].ravel()
    if ev_all.sum() <= 10:
        continue
    try:
        responsive[u] = stats.wilcoxon(ev_all, bl_counts[u].ravel()).pvalue < 0.01
    except ValueError:
        pass
    modulated[u] = stats.kruskal(*[ev_counts[u, fi] for fi in
                                   range(len(cx_freqs))]).pvalue < 0.01
print(f"tone-responsive: {responsive.sum()}/{n_units}; "
      f"frequency-modulated: {modulated.sum()}/{n_units}")

cx_net = cx_maps - cx_maps[:, :, CX_CENTERS < 0].mean(axis=2, keepdims=True)
cx_net_s = gaussian_filter(cx_net, sigma=(0, 1.0, 1.5))
resp_win = (CX_CENTERS >= 0.005) & (CX_CENTERS < 0.060)
cx_tc = cx_net_s[:, :, resp_win].mean(axis=2)
cx_bf_idx = np.argmax(cx_tc, axis=1)
cx_bf = cx_freqs[cx_bf_idx]

cx_latency = np.full(n_units, np.nan)
for u in range(n_units):
    p = cx_net_s[u, cx_bf_idx[u]]
    win = (CX_CENTERS > 0) & (CX_CENTERS < 0.060)
    if p[win].max() <= 0:
        continue
    w = np.where((p > p[win].max() / 2) & win)[0]
    if len(w):
        cx_latency[u] = CX_CENTERS[w[0]] * 1000

# %%
# Figure 4: cortical STRF gallery: diverse examples (excitatory units with
# different dominant frequencies plus suppressive units), and the BF-aligned
# population average over excitatory units
both = modulated & responsive
mod_idx = np.where(both)[0]
dom_abs = np.argmax(np.abs(cx_tc[mod_idx]), axis=1)

picks = []
for fi in range(len(cx_freqs)):
    pool = mod_idx[dom_abs == fi]
    if len(pool) == 0:
        continue  # e.g. no 4 kHz-dominant unit in this session
    picks.append(int(pool[np.argmax(np.abs(cx_tc[pool, fi]))]))
# add the strongest suppressive unit not already included
minvals = cx_tc[mod_idx].min(axis=1)
for u in mod_idx[np.argsort(minvals)]:
    if int(u) not in picks:
        picks.append(int(u))
        break
picks = picks[:5]
print("gallery units:", picks)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, u in zip(axes.flat, picks):
    m = cx_net_s[u]
    v = np.abs(m).max()
    im = ax.imshow(m, aspect="auto", origin="lower",
                   extent=[CX_CENTERS[0] * 1e3, CX_CENTERS[-1] * 1e3,
                           0, len(cx_freqs)],
                   cmap="RdBu_r", vmin=-v, vmax=v)
    ax.axvline(0, color="k", lw=0.8)
    ax.axvline(25, color="k", lw=0.8, ls="--")
    ax.set_yticks(range(len(cx_freqs)))
    ax.set_yticklabels([f"{fr/1000:.0f}" for fr in cx_freqs])
    dom_f = cx_freqs[np.argmax(np.abs(cx_tc[u]))] / 1000
    excit = cx_tc[u].max() > abs(cx_tc[u].min())
    kind = "excitatory" if excit else "suppressive"
    title = f"unit {u} ({kind}, peak {dom_f:.0f} kHz"
    title += f", latency={cx_latency[u]:.0f} ms)" if excit else ")"
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("time rel tone onset (ms)")
    ax.set_ylabel("frequency (kHz)")
    plt.colorbar(im, ax=ax, label="net sp/s")

# BF-aligned population average over excitatory units, NaN-padded grid
ax = axes.flat[5]
exc = mod_idx[cx_tc[mod_idx].max(axis=1) > 0]
exc_bf = np.argmax(cx_tc[exc], axis=1)
pad = len(cx_freqs) - 1  # grid rows span -(n-1)..+(n-1) octaves
grid = np.full((len(exc), 2 * pad + 1, len(CX_CENTERS)), np.nan)
for k, (u, b) in enumerate(zip(exc, exc_bf)):
    grid[k, pad - b:pad - b + len(cx_freqs)] = cx_net_s[u]
avg = np.nanmean(grid, axis=0)[pad - 2:pad + 3]  # show -2..+2 oct
v = np.abs(avg).max()
im = ax.imshow(avg, aspect="auto", origin="lower",
               extent=[CX_CENTERS[0] * 1e3, CX_CENTERS[-1] * 1e3,
                       0, len(cx_freqs)],
               cmap="RdBu_r", vmin=-v, vmax=v)
ax.axvline(0, color="k", lw=0.8)
ax.axvline(25, color="k", lw=0.8, ls="--")
ax.set_yticks(range(len(cx_freqs)))
ax.set_yticklabels(["-2 oct", "-1 oct", "BF", "+1 oct", "+2 oct"])
ax.set_title(f"population average, BF-aligned (n={len(exc)} excitatory)",
             fontsize=10)
ax.set_xlabel("time rel tone onset (ms)")
ax.set_ylabel("frequency rel BF")
plt.colorbar(im, ax=ax, label="net sp/s")

fig.suptitle("Cortical STRFs (mouse auditory cortex, DANDI:000986, sub-LA8 ses-1)")
fig.tight_layout()
fig.savefig("fig4_cortical_strf_gallery.png", dpi=150)
plt.close(fig)
print("saved fig4_cortical_strf_gallery.png")

# %% [markdown]
# Cortical STRFs differ from nerve STRFs in two obvious ways: latencies are longer
# (the signal has passed the cochlear nucleus, inferior colliculus, and thalamus),
# and responses are more transient, often with suppression following the onset
# response (visible as blue in the maps). The BF-aligned population average shows a
# compact excitatory hotspot at BF followed by suppression.
#
# ## Part 4: Periphery vs cortex comparison

# %%
# Figure 5: side-by-side comparison
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

# (a) example AN STRF (CF ~4 kHz from the gallery)
res = examples[2]
fr = res["freqs"]
v = np.percentile(res["net_map"], 99.5)
ax = axes[0]
im = ax.imshow(res["net_map"], aspect="auto", origin="lower",
               extent=[AN_CENTERS[0] * 1e3, AN_CENTERS[-1] * 1e3,
                       fr[0] / 1e3, fr[-1] / 1e3],
               cmap="viridis", vmin=0, vmax=v)
ax.axvline(0, color="w", lw=0.8)
ax.axvline(res["dur"] * 1e3, color="w", lw=0.8, ls="--")
ax.set_title(f"(a) Auditory nerve fiber (CF={res['cf']:.0f} Hz)")
ax.set_xlabel("time rel tone onset (ms)")
ax.set_ylabel("frequency (kHz)")
plt.colorbar(im, ax=ax, label="net sp/s")

# (b) example cortical STRF (the 16 kHz BF unit from the gallery)
fi16 = int(np.where(cx_freqs == 16000)[0][0])
cand = np.where(both & (cx_bf_idx == fi16))[0]
u16 = cand[np.argmax(cx_tc[cand, fi16])]
m = cx_net_s[u16]
v = np.abs(m).max()
ax = axes[1]
im = ax.imshow(m, aspect="auto", origin="lower",
               extent=[CX_CENTERS[0] * 1e3, CX_CENTERS[-1] * 1e3,
                       0, len(cx_freqs)],
               cmap="RdBu_r", vmin=-v, vmax=v)
ax.axvline(0, color="k", lw=0.8)
ax.axvline(25, color="k", lw=0.8, ls="--")
ax.set_yticks(range(len(cx_freqs)))
ax.set_yticklabels([f"{fr/1000:.0f}" for fr in cx_freqs])
ax.set_title(f"(b) Cortical unit (BF=16 kHz)")
ax.set_xlabel("time rel tone onset (ms)")
ax.set_ylabel("frequency (kHz)")
plt.colorbar(im, ax=ax, label="net sp/s")

# (c) latency distributions
ax = axes[2]
an_lat = metrics.loc[metrics["latency_ms"].between(1, 15), "latency_ms"]
cx_lat = cx_latency[both & np.isfinite(cx_latency)]
bins = np.arange(0, 45, 2)
ax.hist(an_lat, bins=bins, alpha=0.7, density=True,
        label=f"auditory nerve (median {np.median(an_lat):.1f} ms)")
ax.hist(cx_lat, bins=bins, alpha=0.7, density=True,
        label=f"auditory cortex (median {np.median(cx_lat):.1f} ms)")
ax.axvline(np.median(an_lat), color="C0", ls="--", lw=1)
ax.axvline(np.median(cx_lat), color="C1", ls="--", lw=1)
ax.set_xlabel("STRF onset latency (ms)")
ax.set_ylabel("density")
ax.set_title("(c) Onset latency: periphery vs cortex")
ax.legend()
mw = stats.mannwhitneyu(an_lat, cx_lat)
print(f"latency: AN median {np.median(an_lat):.1f} ms (n={len(an_lat)}), "
      f"cortex median {np.median(cx_lat):.1f} ms (n={len(cx_lat)}), "
      f"Mann-Whitney p={mw.pvalue:.2e}")

fig.suptitle("STRFs at two stages of the auditory system")
fig.tight_layout()
fig.savefig("fig5_periphery_vs_cortex.png", dpi=150)
plt.close(fig)
print("saved fig5_periphery_vs_cortex.png")

# %% [markdown]
# ## Summary
#
# - **Auditory nerve STRFs** (DANDI:001262, gerbil) are compact excitatory ridges at
#   the fiber's CF with onset latencies of a few milliseconds, sustained through the
#   tone. Across 85 fibers from 47 animals, the STRF-based CF reproduces the
#   published BF (r ≈ 0.99), latency shortens with CF (cochlear traveling-wave
#   delay), and relative bandwidth narrows with CF (sharper tuning in octaves at
#   high CF).
# - **Cortical STRFs** (DANDI:000986, mouse auditory cortex) are coarser in
#   frequency (five octave-spaced tones) but reveal longer latencies (median ~17 ms
#   vs ~4.5 ms in the nerve) and more transient, often suppressive-followed
#   dynamics.
# - The tone-pip STRF (per-frequency PSTH, baseline-subtracted) is the appropriate
#   estimator for both datasets because stimuli are discrete tones and no stimulus
#   waveforms are stored for reverse correlation.
