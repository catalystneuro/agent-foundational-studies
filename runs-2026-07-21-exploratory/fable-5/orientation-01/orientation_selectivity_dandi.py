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
# # Orientation Selectivity in the Mouse Visual System
#
# ## DANDI:000021, Allen Institute Visual Coding - Neuropixels
#
# Neurons in primary visual cortex respond selectively to the orientation of an edge or
# grating: a cell that fires vigorously to a vertical bar may be nearly silent for a
# horizontal one. This is the canonical result of Hubel and Wiesel's recordings in cat
# cortex, and it remains the standard functional signature of visual cortex. This
# notebook demonstrates the phenomenon from scratch using publicly archived data.
#
# The analysis uses [DANDI:000021](https://dandiarchive.org/dandiset/000021), the Allen
# Institute "Visual Coding - Neuropixels" dataset (Brain Observatory 1.1). Each session
# is a head-fixed, awake mouse viewing a battery of visual stimuli while up to six
# Neuropixels probes record simultaneously from visual cortex, visual thalamus and
# hippocampus. Two of the stimuli are directly relevant here:
#
# - **Drifting gratings**: 2 s presentations, 8 directions (0-315 deg in 45 deg steps)
#   crossed with 5 temporal frequencies, 15 repeats per combination, plus blank sweeps.
#   Because the grating moves, these separate *orientation* selectivity (a 180 deg
#   periodic preference) from *direction* selectivity (360 deg periodic).
# - **Static gratings**: 0.25 s presentations, 6 orientations (0-150 deg in 30 deg
#   steps) crossed with 5 spatial frequencies and 4 phases, roughly 50 repeats each.
#   These give a large trial count and an independent measurement of the same
#   orientation preference.
#
# The design has a built-in control. The same probes that pass through visual cortex
# continue into hippocampus, so every session records orientation-selective and
# non-visual neurons under identical conditions, with the same spike sorting, the same
# quality control and the same statistics. Any selectivity metric that is inflated by
# noise, by firing rate, or by slow drift will be inflated in hippocampus too. The
# contrast between regions, rather than the raw value of an index, is what carries the
# argument.
#
# **What the notebook does**
#
# 1. Streams six sessions from the DANDI Archive without downloading whole files.
# 2. Extracts trial-resolved firing rates for both grating stimuli.
# 3. Measures orientation selectivity per neuron and tests it against a shuffled-label
#    null distribution.
# 4. Compares visual cortex, visual thalamus and hippocampus.
# 5. Decodes grating orientation from single-trial population activity.
# 6. Fits Poisson GLMs with NeMoS to isolate the contribution of direction.
#
# **Runtime.** Roughly 20-30 minutes end to end on a cold cache, dominated by streaming
# about 13 GB of spike data. Every expensive stage writes a cache file and is skipped on
# re-runs, so a second execution takes a couple of minutes.

# %% [markdown]
# ## 1. Setup

# %%
import os
import pickle
from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynapple as nap
import remfile
from matplotlib.gridspec import GridSpec
from scipy import optimize, stats
from scipy.special import gammaln
from tqdm.auto import tqdm

nap.nap_config.suppress_conversion_warnings = True

mpl.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 10, "legend.frameon": False,
})

# Where streamed HDF5 chunks are kept between runs.
CACHE_DIR = os.environ.get("REMFILE_CACHE", "/tmp/remfile_cache")
ASSET_URL = "https://api.dandiarchive.org/api/assets/{asset_id}/download/"

GROUP_COLORS = {"visual cortex": "#2166ac", "visual thalamus": "#f4a582", "hippocampus": "#878787"}
GROUP_ORDER = ["visual cortex", "visual thalamus", "hippocampus"]


def save(fig, name):
    fig.savefig(name, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", name)


# %% [markdown]
# ### Sessions
#
# Six sessions were selected from the 32 session-level files in the dandiset by
# surveying their unit tables; the survey is saved as `session_survey.csv`. The criteria
# were joint coverage of primary visual cortex, at least three higher visual areas,
# visual thalamus and hippocampus, so that the regional comparison can be made *within*
# each session rather than across animals.
#
# Brain regions are grouped three ways. `VISp` is primary visual cortex; `VISl`, `VISal`,
# `VISrl`, `VISam` and `VISpm` are higher visual cortical areas. `LGd`, `LGv` and `LP`
# are visual thalamus. `CA1`, `CA3` and `DG` are hippocampus, which serves as the
# non-visual control.

# %%
SESSIONS = {
    "755434585": "edf10182-5a4c-454f-ad23-47987a5ca256",
    "791319847": "3adbba7c-3feb-468b-9829-33fcaa27aacd",
    "757970808": "dfc3db15-066a-4a07-b615-a4d7e85c44e1",
    "760345702": "47634abd-db85-48f5-9c33-01887a59d3bc",
    "763673393": "8a17b967-2aa9-4d6a-812c-92b62cf799d7",
    "754312389": "5a58bf3d-a1b9-444b-8ab0-ef5478aa42a6",
}

CORTEX_AREAS = ["VISp", "VISl", "VISal", "VISrl", "VISam", "VISpm"]
THALAMUS_AREAS = ["LGd", "LGv", "LP"]
CONTROL_AREAS = ["CA1", "CA3", "DG"]
AREAS_OF_INTEREST = CORTEX_AREAS + THALAMUS_AREAS + CONTROL_AREAS

AREA_GROUP = (
    {a: "visual cortex" for a in CORTEX_AREAS}
    | {a: "visual thalamus" for a in THALAMUS_AREAS}
    | {a: "hippocampus" for a in CONTROL_AREAS}
)

# Allen Institute standard unit-quality criteria.
QC = dict(isi_violations=0.5, amplitude_cutoff=0.1, presence_ratio=0.9)

# Response windows relative to stimulus onset. Drifting gratings last 2 s. Static
# gratings last 0.25 s and are presented back to back, so that window is shifted by the
# visual response latency measured in section 3.
DG_WINDOW = (0.0, 2.0)
SG_WINDOW = (0.03, 0.28)

# %% [markdown]
# ## 2. Streaming access to the NWB files
#
# Each session file is 1.7-2.5 GB and holds roughly 80 million spike times across about
# 1600 sorted units. Downloading them is unnecessary. `remfile` exposes a DANDI asset as
# a file-like object that fetches byte ranges on demand and caches them on local disk,
# which `h5py` reads directly.
#
# The one access pattern that matters for speed is how spike times are read. NWB stores
# them as a single ragged array with an index of unit boundaries, so each unit's spikes
# occupy one contiguous slice. Reading unit by unit, and only for units that pass
# quality control and sit in an area of interest, turns the load into a modest number of
# large sequential reads instead of millions of scattered small ones.

# %%
def open_session(asset_id):
    """Open a DANDI asset as a streaming h5py File, with chunks cached on local disk."""
    rem = remfile.File(ASSET_URL.format(asset_id=asset_id), disk_cache=remfile.DiskCache(CACHE_DIR))
    return h5py.File(rem, "r")


def _decode(arr):
    return np.array([s.decode() if isinstance(s, bytes) else str(s) for s in arr])


def load_unit_table(h5):
    """Unit metadata, with brain area resolved through the electrodes table."""
    el = h5["general"]["extracellular_ephys"]["electrodes"]
    loc_by_channel = dict(zip(el["id"][:], _decode(el["location"][:])))

    u = h5["units"]
    df = pd.DataFrame({
        "unit_id": u["id"][:],
        "peak_channel_id": u["peak_channel_id"][:],
        "quality": _decode(u["quality"][:]),
        "isi_violations": u["isi_violations"][:],
        "amplitude_cutoff": u["amplitude_cutoff"][:],
        "presence_ratio": u["presence_ratio"][:],
        "snr": u["snr"][:],
        "firing_rate": u["firing_rate"][:],
        "waveform_duration": u["waveform_duration"][:],
    })
    df["area"] = [loc_by_channel.get(c, "unknown") for c in df["peak_channel_id"]]
    df["area_group"] = df["area"].map(AREA_GROUP)
    df["passes_qc"] = (
        (df["quality"] == "good")
        & (df["isi_violations"] < QC["isi_violations"])
        & (df["amplitude_cutoff"] < QC["amplitude_cutoff"])
        & (df["presence_ratio"] > QC["presence_ratio"])
    )
    return df


def load_spikes(h5, unit_df, row_indices):
    """
    Build a pynapple TsGroup from a subset of units, plus metadata aligned to it.

    ``row_indices`` are positional rows into the unit table. ``nap.TsGroup`` sorts its
    keys, and the NWB units table is not sorted by unit id, so the metadata is explicitly
    reindexed onto ``tsgroup.index`` before being attached. Everything downstream then
    shares one unit ordering; getting this wrong silently scrambles areas across units.
    """
    idx = h5["units"]["spike_times_index"][:]
    starts = np.concatenate([[0], idx[:-1]])
    st_ds = h5["units"]["spike_times"]

    spikes = {}
    for row in row_indices:
        t = st_ds[starts[row]:idx[row]]
        spikes[int(unit_df["unit_id"].iloc[row])] = nap.Ts(t=np.sort(t))

    tsgroup = nap.TsGroup(spikes)
    meta = unit_df.iloc[row_indices].set_index("unit_id").loc[list(tsgroup.index)]
    for col in ["area", "area_group", "snr", "waveform_duration", "firing_rate"]:
        tsgroup.set_info(**{col: meta[col].values})
    return tsgroup, meta.reset_index()


def load_stimulus_table(h5, name):
    """Read a stimulus presentation interval table into a DataFrame."""
    grp = h5["intervals"][name]
    out = {}
    for key in grp.keys():
        ds = grp[key]
        if not isinstance(ds, h5py.Dataset) or ds.ndim != 1 or ds.shape[0] != grp["start_time"].shape[0]:
            continue
        if key.endswith("_index") or key in ("timeseries", "tags"):
            continue
        vals = ds[:]
        out[key] = _decode(vals) if vals.dtype == object else vals
    df = pd.DataFrame(out)
    df["duration"] = df["stop_time"] - df["start_time"]
    return df


# %% [markdown]
# ### A first look at one session
#
# Before any analysis, load a single session and inspect what is actually in it.

# %%
demo_sid, demo_asset = list(SESSIONS.items())[0]
h5 = open_session(demo_asset)
unit_df = load_unit_table(h5)

print(f"session {demo_sid}: {len(unit_df)} sorted units")
print(f"  {unit_df['passes_qc'].sum()} pass quality control")
sel = unit_df["passes_qc"] & unit_df["area"].isin(AREAS_OF_INTEREST)
print(f"  {sel.sum()} of those are in an area of interest\n")
print(unit_df.loc[sel, "area"].value_counts().to_string())

# %%
dg_demo = load_stimulus_table(h5, "drifting_gratings_presentations")
sg_demo = load_stimulus_table(h5, "static_gratings_presentations")

print("drifting gratings:", dg_demo.shape)
print("  directions        :", np.sort(dg_demo["orientation"].dropna().unique()))
print("  temporal freqs (Hz):", np.sort(dg_demo["temporal_frequency"].dropna().unique()))
print("  duration (s)      :", round(dg_demo["duration"].median(), 3))
print("  blank sweeps      :", int((~np.isfinite(dg_demo["orientation"])).sum()))
print()
print("static gratings:", sg_demo.shape)
print("  orientations        :", np.sort(sg_demo["orientation"].dropna().unique()))
print("  spatial freqs (cpd) :", np.sort(sg_demo["spatial_frequency"].dropna().unique()))
# Blank sweeps carry the string "N/A" in the phase column, hence the coercion.
print("  phases              :",
      np.sort(pd.to_numeric(sg_demo["phase"], errors="coerce").dropna().unique()))
print("  duration (s)        :", round(sg_demo["duration"].median(), 3))
print("  blank sweeps        :", int((~np.isfinite(sg_demo["orientation"])).sum()))

# %% [markdown]
# Blank sweeps are presentations of a mean-luminance grey screen interleaved with the
# gratings. They carry `NaN` for orientation, and are separated out here to serve as the
# baseline against which visual responsiveness is judged.

# %%
rows = np.flatnonzero(sel.values)
spikes_demo, meta_demo = load_spikes(h5, unit_df, rows)

print(f"pynapple TsGroup with {len(spikes_demo)} units")
print(f"  recording spans {spikes_demo.time_support.tot_length() / 60:.0f} min")
print(f"  total spikes    {sum(len(spikes_demo[u]) for u in spikes_demo.keys()):,}")
print(f"  firing rate     median {np.median(spikes_demo.get_info('firing_rate')):.1f} Hz")
print(spikes_demo.get_info(["area", "area_group", "firing_rate"]).head())

# %% [markdown]
# ### Figure 1: raw data
#
# The most important validation step is to look at unprocessed spikes next to the
# stimulus record. If orientation selectivity is present, some V1 units should visibly
# change their rate from one grating to the next.

# %%
dg_ok = dg_demo[np.isfinite(dg_demo["orientation"])].reset_index(drop=True)
v1_ids = [u for u in spikes_demo.keys() if spikes_demo.get_info("area")[u] == "VISp"]

t0 = dg_ok["start_time"].iloc[0] - 4
t1 = t0 + 62
win = nap.IntervalSet(start=t0, end=t1)
sub = dg_ok[(dg_ok["start_time"] < t1) & (dg_ok["stop_time"] > t0)]

fig = plt.figure(figsize=(13, 8))
gs = GridSpec(3, 1, height_ratios=[3, 1.1, 0.55], hspace=0.22, figure=fig)
ax0, ax1, ax2 = [fig.add_subplot(gs[i]) for i in range(3)]

for k, uid in enumerate(v1_ids):
    t = spikes_demo[uid].restrict(win).t
    ax0.plot(t, np.full_like(t, k), "|", color="k", ms=3, mew=0.55)
ax0.set_ylim(-1, len(v1_ids))
ax0.set_ylabel("VISp unit")
ax0.set_title(f"Session {demo_sid}: spiking in primary visual cortex during drifting "
              f"gratings (n = {len(v1_ids)} units)")

pop = spikes_demo[v1_ids].count(0.05, ep=win).sum(axis=1) / (0.05 * len(v1_ids))
ax1.plot(pop.t, pop.d, color="0.75", lw=0.7)
ax1.plot(pop.t, pop.smooth(std=0.075).d, color=GROUP_COLORS["visual cortex"], lw=1.6)
ax1.set_ylabel("population rate\n(Hz per unit)")

for _, r in sub.iterrows():
    for a in (ax0, ax1):
        a.axvspan(r["start_time"], r["stop_time"], color="#f4a582", alpha=0.22, lw=0, zorder=0)
    ax2.axvspan(r["start_time"], r["stop_time"], color="#f4a582", alpha=0.75, lw=0)
    ax2.text((r["start_time"] + r["stop_time"]) / 2, 0.5, f"{int(r['orientation'])}",
             ha="center", va="center", fontsize=7.5)
ax2.set_yticks([])
ax2.set_ylabel("grating\ndirection (deg)", rotation=0, ha="right", va="center")
ax2.set_xlabel("time (s)")
for a in (ax0, ax1, ax2):
    a.set_xlim(t0, t1)
for a in (ax0, ax1):
    a.set_xticklabels([])
save(fig, "fig01_raw_data.png")

# %% [markdown]
# Individual units visibly change their rate from one grating to the next, and the
# smoothed population rate (dark line, over the raw 50 ms bins in grey) is modulated
# across the sequence. On a single stretch of raw data the modulation is real but noisy,
# since each grating appears once and the mouse's arousal and running vary throughout.
# Averaging over repeats, which the next figure does, makes the stimulus locking
# unambiguous. That is the raw phenomenon this notebook goes on to quantify.

# %%
h5.close()

# %% [markdown]
# ## 3. Trial-resolved firing rates
#
# The analysis reduces each session to a `(trials x units)` matrix of firing rates for
# each stimulus. Because every spike train is sorted, the spike count in an arbitrary
# window comes from two binary searches, which makes the whole reduction fast enough to
# run over all units at once.

# %%
def trial_spike_counts(tsgroup, starts, stops):
    """Spike counts for every (trial, unit) pair. Returns (n_trials, n_units)."""
    counts = np.empty((len(starts), len(tsgroup)), dtype=np.int32)
    for j, uid in enumerate(tsgroup.keys()):
        t = tsgroup[uid].t
        counts[:, j] = np.searchsorted(t, stops) - np.searchsorted(t, starts)
    return counts


def trial_rates(tsgroup, table, window=(0.0, None)):
    """Firing rate (Hz) in a window relative to each stimulus onset."""
    starts = table["start_time"].values + window[0]
    stops = (table["stop_time"].values if window[1] is None
             else table["start_time"].values + window[1])
    counts = trial_spike_counts(tsgroup, starts, stops)
    return counts / (stops - starts)[:, None]


def extract_session(session_id, asset_id):
    """Stream one session and reduce it to trial-resolved responses."""
    h5 = open_session(asset_id)
    units = load_unit_table(h5)
    sel = units["passes_qc"] & units["area"].isin(AREAS_OF_INTEREST)
    spikes, meta = load_spikes(h5, units, np.flatnonzero(sel.values))

    meta = meta[["unit_id", "area", "area_group", "snr", "firing_rate", "waveform_duration"]].copy()
    meta["session"] = session_id
    meta["uid"] = session_id + "_" + meta["unit_id"].astype(str)
    assert list(meta["unit_id"]) == list(spikes.keys()), "metadata must follow TsGroup order"

    dg = load_stimulus_table(h5, "drifting_gratings_presentations")
    sg = load_stimulus_table(h5, "static_gratings_presentations")
    dg_blank = dg[~np.isfinite(dg["orientation"])].reset_index(drop=True)
    sg_blank = sg[~np.isfinite(sg["orientation"])].reset_index(drop=True)
    dg = dg[np.isfinite(dg["orientation"])].reset_index(drop=True)
    sg = sg[np.isfinite(sg["orientation"])].reset_index(drop=True)
    sg["phase"] = sg["phase"].astype(float)

    out = dict(
        session=session_id,
        meta=meta.reset_index(drop=True),
        unit_ids=list(spikes.keys()),
        dg_table=dg, sg_table=sg,
        dg_rates=trial_rates(spikes, dg, DG_WINDOW),
        sg_rates=trial_rates(spikes, sg, SG_WINDOW),
        dg_blank_rates=trial_rates(spikes, dg_blank, DG_WINDOW),
        sg_blank_rates=trial_rates(spikes, sg_blank, SG_WINDOW),
    )

    # Peri-onset PSTHs in 5 ms bins, used to justify the response windows.
    edges = np.arange(-0.1, 0.5001, 0.005)
    for tag, table in (("dg", dg), ("sg", sg)):
        onsets = table["start_time"].values
        grid = (onsets[:, None] + edges[None, :]).ravel()
        psth = np.empty((len(spikes), len(edges) - 1))
        for j, uid in enumerate(spikes.keys()):
            idx = np.searchsorted(spikes[uid].t, grid).reshape(len(onsets), len(edges))
            psth[j] = np.diff(idx, axis=1).sum(axis=0)
        out[f"{tag}_psth"] = psth / (len(onsets) * 0.005)
    out["psth_edges"] = edges

    # Compact peri-event rasters for V1 units: spike times relative to each grating onset.
    onsets = dg["start_time"].values
    raster = {}
    for uid in [u for u in spikes.keys() if spikes.get_info("area")[u] == "VISp"]:
        t = spikes[uid].t
        lo = np.searchsorted(t, onsets - 0.5)
        hi = np.searchsorted(t, onsets + 2.5)
        trial_idx, rel = [], []
        for k, (a, b) in enumerate(zip(lo, hi)):
            if b > a:
                rel.append(t[a:b] - onsets[k])
                trial_idx.append(np.full(b - a, k))
        raster[uid] = (
            np.concatenate(trial_idx).astype(np.int32) if trial_idx else np.zeros(0, np.int32),
            np.concatenate(rel).astype(np.float32) if rel else np.zeros(0, np.float32),
        )
    out["v1_dg_raster"] = raster

    h5.close()
    return out


# %%
RESPONSES_CACHE = Path("responses.pkl")

if RESPONSES_CACHE.exists():
    with open(RESPONSES_CACHE, "rb") as f:
        RESULTS = pickle.load(f)
    print(f"loaded cached responses for {len(RESULTS)} sessions")
else:
    RESULTS = []
    for sid, aid in tqdm(SESSIONS.items(), desc="streaming sessions"):
        r = extract_session(sid, aid)
        print(f"  session {sid}: {len(r['unit_ids'])} units, "
              f"{len(r['dg_table'])} drifting-grating trials, "
              f"{len(r['sg_table'])} static-grating trials")
        RESULTS.append(r)
    with open(RESPONSES_CACHE, "wb") as f:
        pickle.dump(RESULTS, f)

RES = {r["session"]: r for r in RESULTS}
SESSION_IDS = list(RES)
n_units_total = sum(len(r["unit_ids"]) for r in RESULTS)
print(f"\n{n_units_total} quality-controlled units across {len(RESULTS)} sessions")

# %% [markdown]
# ### Figure 2: response latency
#
# The choice of response window should be justified by the data rather than assumed.
# Averaging the peri-onset PSTH over all units in each region shows when the visual
# response actually arrives, and confirms that hippocampus does not follow the stimulus.

# %%
edges = RES[SESSION_IDS[0]]["psth_edges"]
centers = edges[:-1] + np.diff(edges) / 2

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharex=True)
for ax, tag, title, dur in [
    (axes[0], "dg", "Drifting gratings (2 s presentations)", 2.0),
    (axes[1], "sg", "Static gratings (0.25 s presentations)", 0.25),
]:
    for group in GROUP_ORDER:
        traces = []
        for res in RES.values():
            m = res["meta"]["area_group"].values == group
            if m.sum() == 0:
                continue
            p = res[f"{tag}_psth"][m]
            traces.append(p - p[:, centers < 0].mean(axis=1, keepdims=True))
        if not traces:
            continue
        tr = np.concatenate(traces, axis=0)
        mu, se = tr.mean(axis=0), tr.std(axis=0) / np.sqrt(len(tr))
        ax.plot(centers, mu, color=GROUP_COLORS[group], lw=1.6, label=f"{group} (n={len(tr)})")
        ax.fill_between(centers, mu - se, mu + se, color=GROUP_COLORS[group], alpha=0.25, lw=0)
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.axvspan(0, min(dur, 0.5), color="#f4a582", alpha=0.12, lw=0)
    ax.set_title(title)
    ax.set_xlabel("time from stimulus onset (s)")
    ax.legend(fontsize=8, loc="upper right")
axes[0].set_ylabel("baseline-subtracted\nfiring rate (Hz)")
axes[0].set_xlim(-0.1, 0.5)
fig.suptitle("Visual response latency by region", y=1.02)
fig.tight_layout()
save(fig, "fig02_response_latency.png")

# %% [markdown]
# Thalamic responses begin around 30 ms after onset and cortical responses a few
# milliseconds later, which is why the static-grating window starts at 30 ms. The
# hippocampal trace is flat, as it should be.

# %% [markdown]
# ## 4. Measuring orientation selectivity
#
# ### The index
#
# The primary metric is the **global orientation selectivity index (gOSI)**, the vector
# strength of the mean response across stimulus angles evaluated on the doubled angle:
#
# $$\mathrm{gOSI} = \frac{\left| \sum_k r_k \, e^{2 i \theta_k} \right|}{\sum_k r_k}$$
#
# Doubling the angle makes the measure 180 deg periodic, so a grating drifting at 0 deg
# and one drifting at 180 deg count as the same orientation. gOSI is 0 for a neuron that
# responds equally to all orientations and approaches 1 for one that responds to a single
# orientation. Replacing $e^{2i\theta}$ with $e^{i\theta}$ gives the direction
# selectivity index gDSI, which is 360 deg periodic and distinguishes a grating drifting
# up from one drifting down.
#
# We also report the classical two-point index, (preferred - orthogonal) /
# (preferred + orthogonal).
#
# ### Why a permutation test is needed
#
# gOSI is biased upward when few spikes are available: a neuron firing at 1 Hz will show
# an apparently non-zero index purely from Poisson noise. Comparing raw index values
# across regions with different firing rates would therefore be misleading. The test
# used here shuffles the stimulus labels across trials 1000 times and recomputes the
# index, which yields a per-neuron null distribution carrying exactly the same rate
# bias. Two quantities come out of it: a p-value, and a z-score of the observed index
# relative to its own null. The z-score is the bias-free effect size.

# %%
N_PERM = 1000

# numpy 2.0 on macOS/Accelerate emits spurious "divide by zero encountered in matmul"
# warnings for perfectly finite operands, so floating-point errors are silenced around
# the matrix products below. Zero denominators are handled explicitly instead.


def condition_means(rates, labels, levels):
    """Mean rate per stimulus level. rates (n_trials, n_units) -> (n_levels, n_units)."""
    M = np.stack([(labels == lv).astype(float) for lv in levels])
    n = M.sum(axis=1, keepdims=True)
    M /= np.where(n > 0, n, 1.0)
    with np.errstate(all="ignore"):
        return M @ rates, M


def gosi_from_means(means, thetas_deg, harmonic=2):
    """Vector-strength selectivity index over units. means: (n_levels, n_units)."""
    m = np.clip(means, 0, None)
    w = np.exp(1j * harmonic * np.deg2rad(thetas_deg))[:, None]
    total = m.sum(axis=0)
    ok = total > 0
    vec = (m * w).sum(axis=0) / np.where(ok, total, 1.0)
    return np.where(ok, np.abs(vec), np.nan), vec


def permutation_test(rates, labels, thetas_deg, harmonic=2, n_perm=N_PERM, seed=0):
    """Shuffle stimulus labels across trials to build a null for the selectivity index."""
    means, M = condition_means(rates, labels, thetas_deg)
    obs, _ = gosi_from_means(means, thetas_deg, harmonic)

    rng = np.random.default_rng(seed)
    null = np.empty((n_perm, rates.shape[1]))
    n = rates.shape[0]
    with np.errstate(all="ignore"):
        for i in range(n_perm):
            null[i], _ = gosi_from_means(M @ rates[rng.permutation(n)], thetas_deg, harmonic)

    p = (np.sum(null >= obs[None, :], axis=0) + 1) / (n_perm + 1)
    sd = null.std(axis=0)
    z = np.where(sd > 0, (obs - null.mean(axis=0)) / np.where(sd > 0, sd, 1.0), np.nan)
    return obs, p, z, null.mean(axis=0)


def two_point_indices(means, thetas_deg, circular_period):
    """
    Classic two-point selectivity index.

    With ``circular_period=180`` directions collapse onto the orientation axis and the
    index contrasts preferred against orthogonal. With ``circular_period=360`` it
    contrasts preferred against opposite, giving direction selectivity.
    """
    th = np.asarray(thetas_deg, float) % circular_period
    levels = np.unique(th)
    collapsed = np.stack([np.clip(means[np.isclose(th, lv)], 0, None).mean(axis=0) for lv in levels])
    i_pref = np.argmax(collapsed, axis=0)
    target = (levels[i_pref] + circular_period / 2) % circular_period
    half = circular_period / 2
    dist = np.abs(((levels[:, None] - target[None, :] + half) % circular_period) - half)
    i_opp = np.argmin(dist, axis=0)
    cols = np.arange(collapsed.shape[1])
    r_p, r_o = collapsed[i_pref, cols], collapsed[i_opp, cols]
    denom = r_p + r_o
    ok = denom > 0
    idx = np.where(ok, (r_p - r_o) / np.where(ok, denom, 1.0), np.nan)
    return idx, levels[i_pref], r_p


# %% [markdown]
# ### Tuning width
#
# To describe how sharply a neuron is tuned, the direction-tuning curve is fit with two
# von Mises lobes 180 deg apart that share a width (the Carandini and Ferster form). The
# shared width lets a neuron that is orientation tuned but also direction biased still be
# summarised by one number, the half-width at half-maximum.

# %%
def double_von_mises(theta_deg, b, a1, a2, kappa, mu_deg):
    """Two von Mises lobes 180 deg apart with a shared width."""
    th, mu = np.deg2rad(theta_deg), np.deg2rad(mu_deg)
    return (b
            + a1 * np.exp(kappa * (np.cos(th - mu) - 1))
            + a2 * np.exp(kappa * (np.cos(th - mu - np.pi) - 1)))


def fit_double_von_mises(dirs_deg, rates):
    """
    Least-squares fit; returns params, HWHM in degrees, and R^2, or NaNs on failure.

    Drifting gratings sample only 8 directions, 45 deg apart, so a fitted peak that
    falls between two sampled directions can be made arbitrarily tall and narrow while
    still passing through every measured point. Such fits are rejected: if the model
    predicts a peak well above anything actually measured, the width it reports is an
    extrapolation rather than a measurement. This also means widths below roughly
    20 deg cannot be resolved by this stimulus at all.
    """
    i = int(np.argmax(rates))
    p0 = [max(rates.min(), 0.0), max(rates[i] - rates.min(), 0.1),
          0.5 * max(rates[i], 0.1), 2.0, dirs_deg[i]]
    bounds = ([0, 0, 0, 0.05, -360], [np.inf, np.inf, np.inf, 100, 720])
    try:
        popt, _ = optimize.curve_fit(double_von_mises, dirs_deg, rates, p0=p0,
                                     bounds=bounds, maxfev=20000)
    except (RuntimeError, ValueError):
        return None, np.nan, np.nan
    pred = double_von_mises(dirs_deg, *popt)
    ss_res = np.sum((rates - pred) ** 2)
    ss_tot = np.sum((rates - rates.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    fine = double_von_mises(np.linspace(0, 360, 721), *popt)
    if fine.max() > 1.5 * np.max(rates):
        return popt, np.nan, r2

    kappa = popt[3]
    arg = 1 + np.log(0.5) / kappa
    hwhm = np.rad2deg(np.arccos(arg)) if arg >= -1 else np.nan
    return popt, hwhm, r2


# %% [markdown]
# ### Defining a visually responsive unit
#
# Selectivity is only meaningful for a unit that responds to the stimulus at all, so
# proportions are reported among responsive units. Responsiveness is tested by comparing
# grating trials against interleaved blank sweeps.
#
# One detail matters. An earlier version of this analysis tested each unit's *best*
# direction against blank, which is circular: selecting units by their largest
# across-direction response pre-selects the units with the strongest orientation
# contrast, which is exactly what the selectivity test then measures. Pooling over all
# directions removes the circularity. It moved the hippocampal false-positive rate from
# 17% down to about 6%, close to the nominal 5% of the test.

# %%
def analyse_session(res, seed=0):
    meta = res["meta"].copy()
    n_units = len(res["unit_ids"])
    assert list(meta["unit_id"]) == list(res["unit_ids"])

    out = {c: meta[c].values for c in
           ["uid", "session", "unit_id", "area", "area_group", "snr", "waveform_duration"]}

    # ---- drifting gratings: direction tuning at each unit's preferred temporal frequency
    dg, rates = res["dg_table"], res["dg_rates"]
    dirs = np.sort(dg["orientation"].unique())
    tfs = np.sort(dg["temporal_frequency"].unique())
    tf_lab, dir_lab = dg["temporal_frequency"].values, dg["orientation"].values

    # The temporal-frequency marginal averages over all directions, so choosing the
    # preferred frequency this way cannot bias the direction contrast that follows.
    tf_marginal = np.stack([rates[tf_lab == tf].mean(axis=0) for tf in tfs])
    pref_tf_i = np.argmax(tf_marginal, axis=0)
    out["dg_pref_tf"] = tfs[pref_tf_i]
    out["dg_blank_rate"] = res["dg_blank_rates"].mean(axis=0)

    dir_means = np.full((len(dirs), n_units), np.nan)
    blank = np.full(n_units, np.nan)
    gosi = np.full(n_units, np.nan); gp = np.full(n_units, np.nan)
    gz = np.full(n_units, np.nan); gnull = np.full(n_units, np.nan)
    gdsi = np.full(n_units, np.nan); dp = np.full(n_units, np.nan)
    kruskal_p = np.full(n_units, np.nan)
    resp_p = np.full(n_units, np.nan)
    mean_rate = np.full(n_units, np.nan)

    for k, tf in enumerate(tfs):
        cols = np.flatnonzero(pref_tf_i == k)
        if len(cols) == 0:
            continue
        selk = tf_lab == tf
        r_sub, l_sub = rates[np.ix_(selk, cols)], dir_lab[selk]
        m, _ = condition_means(r_sub, l_sub, dirs)
        dir_means[:, cols] = m
        gosi[cols], gp[cols], gz[cols], gnull[cols] = permutation_test(
            r_sub, l_sub, dirs, 2, seed=seed + k)
        gdsi[cols], dp_k, _ = permutation_test(r_sub, l_sub, dirs, 1, seed=seed + 100 + k)[0:3]
        dp[cols] = dp_k
        mean_rate[cols] = r_sub.mean(axis=0)
        groups = [r_sub[l_sub == d] for d in dirs]
        for jj, c in enumerate(cols):
            kruskal_p[c] = stats.kruskal(*[g[:, jj] for g in groups]).pvalue
            resp_p[c] = stats.mannwhitneyu(
                r_sub[:, jj], res["dg_blank_rates"][:, c], alternative="greater").pvalue

    out["dg_dir_means"] = dir_means.T
    out["dg_gOSI"], out["dg_gOSI_p"], out["dg_gOSI_z"], out["dg_gOSI_null"] = gosi, gp, gz, gnull
    out["dg_gDSI"], out["dg_gDSI_p"] = gdsi, dp
    out["dg_kruskal_p"], out["dg_resp_p"], out["dg_mean_rate"] = kruskal_p, resp_p, mean_rate
    osi, pref_ori, peak = two_point_indices(dir_means, dirs, 180)
    dsi, pref_dir, _ = two_point_indices(dir_means, dirs, 360)
    out["dg_OSI"], out["dg_DSI"] = osi, dsi
    out["dg_pref_ori"], out["dg_pref_dir"], out["dg_peak_rate"] = pref_ori, pref_dir, peak

    # Robustness check: the same index with all temporal frequencies pooled.
    m_all, _ = condition_means(rates, dir_lab, dirs)
    out["dg_gOSI_allTF"], _ = gosi_from_means(m_all, dirs, 2)

    # ---- static gratings: orientation tuning at each unit's preferred spatial frequency
    sg, srates = res["sg_table"], res["sg_rates"]
    oris = np.sort(sg["orientation"].unique())
    sfs = np.sort(sg["spatial_frequency"].unique())
    sf_lab, ori_lab = sg["spatial_frequency"].values, sg["orientation"].values

    sf_marginal = np.stack([srates[sf_lab == sf].mean(axis=0) for sf in sfs])
    pref_sf_i = np.argmax(sf_marginal, axis=0)
    out["sg_pref_sf"] = sfs[pref_sf_i]
    out["sg_blank_rate"] = res["sg_blank_rates"].mean(axis=0)

    ori_means = np.full((len(oris), n_units), np.nan)
    s_gosi = np.full(n_units, np.nan); s_gp = np.full(n_units, np.nan)
    s_gz = np.full(n_units, np.nan); s_kp = np.full(n_units, np.nan)
    s_resp = np.full(n_units, np.nan); s_mean = np.full(n_units, np.nan)

    for k, sf in enumerate(sfs):
        cols = np.flatnonzero(pref_sf_i == k)
        if len(cols) == 0:
            continue
        selk = sf_lab == sf
        r_sub, l_sub = srates[np.ix_(selk, cols)], ori_lab[selk]
        m, _ = condition_means(r_sub, l_sub, oris)
        ori_means[:, cols] = m
        s_gosi[cols], s_gp[cols], s_gz[cols], _ = permutation_test(
            r_sub, l_sub, oris, 2, seed=seed + 200 + k)
        s_mean[cols] = r_sub.mean(axis=0)
        groups = [r_sub[l_sub == o] for o in oris]
        for jj, c in enumerate(cols):
            s_kp[c] = stats.kruskal(*[g[:, jj] for g in groups]).pvalue
            s_resp[c] = stats.mannwhitneyu(
                r_sub[:, jj], res["sg_blank_rates"][:, c], alternative="greater").pvalue

    out["sg_ori_means"] = ori_means.T
    out["sg_gOSI"], out["sg_gOSI_p"], out["sg_gOSI_z"] = s_gosi, s_gp, s_gz
    out["sg_kruskal_p"], out["sg_resp_p"], out["sg_mean_rate"] = s_kp, s_resp, s_mean
    s_osi, s_pref, s_peak = two_point_indices(ori_means, oris, 180)
    out["sg_OSI"], out["sg_pref_ori"], out["sg_peak_rate"] = s_osi, s_pref, s_peak

    # ---- tuning width from the drifting-grating curves
    hwhm = np.full(n_units, np.nan)
    vm_r2 = np.full(n_units, np.nan)
    vm_params = np.full((n_units, 5), np.nan)
    for j in range(n_units):
        y = dir_means[:, j]
        if np.all(np.isfinite(y)) and y.max() > 0.5:
            popt, hw, r2 = fit_double_von_mises(dirs, y)
            if popt is not None:
                vm_params[j], hwhm[j], vm_r2[j] = popt, hw, r2
    out["vm_hwhm"], out["vm_r2"] = hwhm, vm_r2

    df = pd.DataFrame({k: v for k, v in out.items() if np.ndim(v) == 1})
    arrays = dict(dg_dir_means=out["dg_dir_means"], sg_ori_means=out["sg_ori_means"],
                  vm_params=vm_params, dirs=dirs, oris=oris, tfs=tfs, sfs=sfs)
    return df, arrays


# %%
METRICS_CACHE = Path("unit_metrics.parquet")
ARRAYS_CACHE = Path("tuning_arrays.pkl")

if METRICS_CACHE.exists() and ARRAYS_CACHE.exists():
    UNITS = pd.read_parquet(METRICS_CACHE)
    with open(ARRAYS_CACHE, "rb") as f:
        ARRAYS = pickle.load(f)
    print(f"loaded cached metrics for {len(UNITS)} units")
else:
    dfs, ARRAYS = [], {}
    for res in tqdm(RESULTS, desc="tuning analysis"):
        df, arrays = analyse_session(res)
        dfs.append(df)
        ARRAYS[res["session"]] = arrays
    UNITS = pd.concat(dfs, ignore_index=True)

    UNITS["dg_responsive"] = (UNITS["dg_resp_p"] < 0.01) & (UNITS["dg_mean_rate"] > 1.0)
    UNITS["sg_responsive"] = (UNITS["sg_resp_p"] < 0.01) & (UNITS["sg_mean_rate"] > 1.0)
    UNITS["dg_orientation_selective"] = UNITS["dg_responsive"] & (UNITS["dg_gOSI_p"] < 0.05)
    UNITS["sg_orientation_selective"] = UNITS["sg_responsive"] & (UNITS["sg_gOSI_p"] < 0.05)
    UNITS["dg_direction_selective"] = UNITS["dg_responsive"] & (UNITS["dg_gDSI_p"] < 0.05)

    UNITS.to_parquet(METRICS_CACHE)
    with open(ARRAYS_CACHE, "wb") as f:
        pickle.dump(ARRAYS, f)

DIRS = ARRAYS[SESSION_IDS[0]]["dirs"]
ORIS = ARRAYS[SESSION_IDS[0]]["oris"]

# %% [markdown]
# ## 5. Single neurons
#
# ### Figure 3: example units
#
# For a handful of strongly tuned V1 neurons, three views of the same cell: the raster
# sorted by grating direction, the polar tuning curve with its von Mises fit, and the
# orientation by spatial-frequency response map measured with the independent
# static-grating stimulus.

# %%
cand = UNITS[(UNITS.area == "VISp") & UNITS.dg_orientation_selective &
             (UNITS.vm_r2 > 0.8) & UNITS.vm_hwhm.notna()]
cand = cand.sort_values("dg_gOSI", ascending=False)

picked, seen_ori = [], []
for _, row in cand.iterrows():
    if any(abs(((row.dg_pref_ori - o + 90) % 180) - 90) < 25 for o in seen_ori):
        continue
    picked.append(row)
    seen_ori.append(row.dg_pref_ori)
    if len(picked) == 4:
        break

fig, axes = plt.subplots(len(picked), 3, figsize=(12.5, 3.05 * len(picked)),
                         gridspec_kw=dict(width_ratios=[1.5, 1, 1], wspace=0.42, hspace=0.55))
for i, row in enumerate(picked):
    res = RES[row.session]
    meta = res["meta"]
    j = int(np.flatnonzero(meta["uid"].values == row.uid)[0])
    dg = res["dg_table"]
    uid_int = int(row.unit_id)

    keep = np.flatnonzero(dg["temporal_frequency"].values == row.dg_pref_tf)
    order = keep[np.argsort(dg["orientation"].values[keep], kind="stable")]
    rank = np.full(len(dg), -1)
    rank[order] = np.arange(len(order))
    tr, rel = res["v1_dg_raster"][uid_int]
    m = rank[tr] >= 0
    ax = axes[i, 0]
    ax.plot(rel[m], rank[tr[m]], "|", color="k", ms=2.6, mew=0.5)
    ax.axvspan(0, 2, color="#f4a582", alpha=0.18, lw=0, zorder=0)
    bounds = np.searchsorted(np.sort(dg["orientation"].values[keep]), DIRS)
    ax.set_yticks(bounds + np.diff(np.append(bounds, len(order))) / 2)
    ax.set_yticklabels([f"{int(d)}" for d in DIRS], fontsize=8)
    for b in bounds[1:]:
        ax.axhline(b - 0.5, color="0.75", lw=0.5)
    ax.set_ylim(-0.5, len(order) - 0.5)
    ax.set_xlim(-0.5, 2.5)
    ax.set_xlabel("time from onset (s)")
    ax.set_ylabel("grating direction (deg)")
    ax.set_title(f"{row.area} unit {uid_int}  (session {row.session})", fontsize=9)

    means = ARRAYS[row.session]["dg_dir_means"][j]
    params = ARRAYS[row.session]["vm_params"][j]
    ax = fig.add_subplot(len(picked), 3, 3 * i + 2, projection="polar")
    axes[i, 1].axis("off")
    th = np.deg2rad(np.append(DIRS, DIRS[0]))
    ax.plot(th, np.append(means, means[0]), "o", color="#2166ac", ms=5, zorder=3)
    grid = np.linspace(0, 360, 361)
    ax.plot(np.deg2rad(grid), double_von_mises(grid, *params), "-", color="#b2182b", lw=1.5)
    ax.fill(th, np.append(means, means[0]), color="#2166ac", alpha=0.15)
    ax.set_theta_zero_location("E")
    ax.set_rlabel_position(112)
    ax.tick_params(labelsize=7.5, pad=1)
    ax.set_title(f"drifting gratings\ngOSI={row.dg_gOSI:.2f}  DSI={row.dg_DSI:.2f}  "
                 f"HWHM={row.vm_hwhm:.0f} deg", fontsize=8.5, pad=17)

    sg = res["sg_table"]
    sfs = ARRAYS[row.session]["sfs"]
    mat = np.array([[res["sg_rates"][(sg["orientation"].values == o) &
                                     (sg["spatial_frequency"].values == s), j].mean()
                     for o in ORIS] for s in sfs])
    ax = axes[i, 2]
    im = ax.imshow(mat, aspect="auto", origin="lower", cmap="magma")
    ax.set_xticks(range(len(ORIS)))
    ax.set_xticklabels([f"{int(o)}" for o in ORIS], fontsize=8)
    ax.set_yticks(range(len(sfs)))
    ax.set_yticklabels([f"{s:g}" for s in sfs], fontsize=8)
    ax.set_xlabel("orientation (deg)")
    ax.set_ylabel("spatial freq.\n(cyc/deg)")
    ax.set_title(f"static gratings\ngOSI={row.sg_gOSI:.2f}", fontsize=8.5, pad=6)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("rate (Hz)", fontsize=8)
    cb.ax.tick_params(labelsize=7.5)

fig.suptitle("Orientation-selective units in mouse primary visual cortex", y=0.995, fontsize=12)
save(fig, "fig03_example_units.png")

# %% [markdown]
# The rasters show dense firing for two opposite directions and near silence for the
# orthogonal pair, which is the defining signature. The static-grating maps, measured
# with a completely different stimulus, peak at the same orientation.

# %% [markdown]
# ### Figure 4: the whole population at once
#
# Normalising each responsive unit's orientation tuning curve and sorting by preferred
# orientation shows whether tuning is a property of a few cells or of the population.
# The static-grating stimulus is used here because it samples six orientations rather
# than the four the drifting gratings provide once opposite directions are folded
# together.
#
# Sorting units by their preferred orientation will manufacture an apparent diagonal
# even from pure noise, because the sort uses the same data that is being displayed. The
# rightmost panel removes that circularity: each unit's preferred orientation is
# estimated from odd-numbered trials, and the curve plotted is taken from even-numbered
# trials. Only tuning that is stable across independent trials survives, and a flat line
# is what noise gives.

# %%
def aligned_split_half(res, mask, n_shifts=6):
    """
    Cross-validated aligned tuning curves.

    Preferred orientation comes from odd trials, the returned curve from even trials,
    expressed relative to each unit's own mean rate. Under the null the expectation is
    a flat line at 1.
    """
    sg, rates = res["sg_table"], res["sg_rates"]
    oris = np.sort(sg["orientation"].unique())
    lab = sg["orientation"].values
    a, b = np.arange(len(lab)) % 2 == 1, np.arange(len(lab)) % 2 == 0

    m_a, _ = condition_means(rates[a][:, mask], lab[a], oris)
    m_b, _ = condition_means(rates[b][:, mask], lab[b], oris)
    pref = np.argmax(m_a, axis=0)

    mean_b = m_b.mean(axis=0)
    ok = mean_b > 0
    rel = m_b[:, ok] / mean_b[ok]
    pref = pref[ok]
    rows = np.arange(len(oris))[:, None]
    return rel[(rows + pref[None, :]) % len(oris), np.arange(rel.shape[1])[None, :]].T


fig = plt.figure(figsize=(16.5, 4.4))
gs = GridSpec(1, 4, figure=fig, width_ratios=[1, 1, 1, 1.05], wspace=0.42)
im = None
for k, group in enumerate(GROUP_ORDER):
    ax = fig.add_subplot(gs[0, k])
    curves = []
    for sid in SESSION_IDS:
        mask = (UNITS.session == sid) & (UNITS.area_group == group) & UNITS.sg_responsive
        if mask.sum() == 0:
            continue
        # Positional rows within this session's block of the concatenated table.
        idx = np.flatnonzero(mask) - int(UNITS.index[UNITS.session == sid][0])
        m = ARRAYS[sid]["sg_ori_means"][idx]
        rng_ = m.max(axis=1) - m.min(axis=1)
        keep = rng_ > 0
        curves.append((m[keep] - m[keep].min(axis=1, keepdims=True)) / rng_[keep, None])
    if not curves:
        ax.axis("off")
        continue
    norm = np.concatenate(curves)
    norm = norm[np.argsort(np.argmax(norm, axis=1) + 0.001 * np.arange(len(norm)))]
    wrapped = np.hstack([norm, norm[:, :1]])
    im = ax.imshow(wrapped, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=1,
                   extent=[-15, 195, 0, len(norm)])
    ax.set_xticks([0, 45, 90, 135, 180])
    ax.set_xlabel("grating orientation (deg)")
    ax.set_ylabel("unit (sorted by preferred orientation)" if k == 0 else "")
    ax.set_title(f"{group}\n{len(norm)} responsive units", fontsize=10)
if im is not None:
    fig.colorbar(im, ax=fig.axes[:3], fraction=0.02, pad=0.015, label="normalised rate")

ax = fig.add_subplot(gs[0, 3])
offsets = np.arange(len(ORIS)) * 30.0
offsets[offsets > 90] -= 180
srt = np.argsort(offsets)
for group in GROUP_ORDER:
    allc = []
    for sid in SESSION_IDS:
        mask = (UNITS.session == sid) & (UNITS.area_group == group) & UNITS.sg_responsive
        if mask.sum() == 0:
            continue
        idx = np.flatnonzero(mask) - int(UNITS.index[UNITS.session == sid][0])
        sel = np.zeros(len(RES[sid]["unit_ids"]), bool)
        sel[idx] = True
        allc.append(aligned_split_half(RES[sid], sel))
    c = np.concatenate(allc)
    mu = c.mean(axis=0)[srt]
    se = (c.std(axis=0) / np.sqrt(len(c)))[srt]
    ax.errorbar(offsets[srt], mu, yerr=se, marker="o", ms=4, lw=1.6, capsize=2.5,
                color=GROUP_COLORS[group], label=f"{group} (n={len(c)})")
ax.axhline(1.0, color="k", ls="--", lw=0.9)
ax.set_xticks(offsets[srt])
ax.set_xlabel("orientation relative to preferred (deg)")
ax.set_ylabel("rate relative to each unit's mean")
ax.set_title("Cross-validated alignment\n(preference from odd trials, curve from even)",
             fontsize=9.5)
ax.legend(fontsize=7.5)
fig.suptitle("Single-unit orientation tuning, static gratings", y=1.03, fontsize=12)
save(fig, "fig04_population_tuning_heatmap.png")

# %% [markdown]
# Visual cortex shows a clean diagonal band: each unit has a distinct preferred
# orientation and the preferences tile the full range. Hippocampus shows a diffuse
# pattern with no consistent structure, which is what normalising noise looks like. The
# cross-validated panel makes the same point without the circularity of the sort: the
# cortical curve keeps a pronounced peak when preference and response are estimated from
# disjoint trials, and the hippocampal curve is essentially flat.

# %% [markdown]
# ## 6. Population statistics
#
# ### Figure 5: how selective, and how does that compare to chance

# %%
fig = plt.figure(figsize=(13, 7.6))
gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.32)

for col, (key, title) in enumerate([("dg_gOSI", "Drifting gratings"),
                                    ("sg_gOSI", "Static gratings")]):
    ax = fig.add_subplot(gs[0, col])
    resp = "dg_responsive" if key.startswith("dg") else "sg_responsive"
    bins = np.linspace(0, 1, 36)
    for group in GROUP_ORDER:
        v = UNITS.loc[(UNITS.area_group == group) & UNITS[resp], key].dropna()
        ax.hist(v, bins=bins, density=True, histtype="step", lw=1.8, color=GROUP_COLORS[group],
                label=f"{group} (n={len(v)}, median {v.median():.2f})")
    ax.set_xlabel("gOSI")
    ax.set_ylabel("probability density")
    ax.set_title(f"{title}: orientation selectivity index")
    ax.legend(fontsize=7.5, loc="upper right")

ax = fig.add_subplot(gs[0, 2])
sub = UNITS[(UNITS.area_group == "visual cortex") & UNITS.dg_responsive]
bins = np.linspace(0, 1, 36)
ax.hist(sub["dg_gOSI"].dropna(), bins=bins, density=True, color="#2166ac", alpha=0.55,
        label="observed")
ax.hist(sub["dg_gOSI_null"].dropna(), bins=bins, density=True, histtype="step", lw=1.8,
        color="k", label="shuffled-label null")
ax.set_xlabel("gOSI")
ax.set_ylabel("probability density")
ax.set_title("Visual cortex: observed vs. chance")
ax.legend(fontsize=8)

ax = fig.add_subplot(gs[1, :2])
areas = [a for a in ["VISp", "VISl", "VISal", "VISrl", "VISam", "VISpm",
                     "LGd", "LP", "LGv", "CA1", "CA3", "DG"] if (UNITS.area == a).sum() >= 20]
x = np.arange(len(areas))
w = 0.38
for k, (flag, resp, lab, c) in enumerate([
    ("dg_orientation_selective", "dg_responsive", "drifting gratings", "#2166ac"),
    ("sg_orientation_selective", "sg_responsive", "static gratings", "#92c5de"),
]):
    frac, err = [], []
    for a in areas:
        s = UNITS[(UNITS.area == a) & UNITS[resp]]
        p = s[flag].mean() if len(s) else np.nan
        frac.append(p)
        err.append(np.sqrt(p * (1 - p) / len(s)) if len(s) else np.nan)
    ax.bar(x + (k - 0.5) * w, frac, w, yerr=err, capsize=2.5, color=c, label=lab)
ax.axhline(0.05, color="k", ls="--", lw=1, label="false-positive rate of the test")
ax.set_xticks(x)
# The unit count goes into the tick label itself; a separate row of text below the axis
# collides with the area names.
ax.set_xticklabels([f"{a}\nn={int(((UNITS.area == a) & UNITS.dg_responsive).sum())}"
                    for a in areas], fontsize=8)
ax.set_ylim(0, 1.28)
ax.set_ylabel("fraction of responsive units\nthat are orientation selective")
ax.set_title("Orientation selectivity by brain region (permutation test, p < 0.05)")
ax.legend(fontsize=8, ncol=3, loc="upper left")

ax = fig.add_subplot(gs[1, 2])
for group in GROUP_ORDER:
    s = UNITS[(UNITS.area_group == group) & UNITS.dg_responsive]
    ax.scatter(s["dg_peak_rate"], s["dg_gOSI"], s=6, alpha=0.35, color=GROUP_COLORS[group],
               lw=0, label=group)
ax.set_xscale("log")
ax.set_xlabel("peak firing rate (Hz)")
ax.set_ylabel("gOSI")
ax.set_title("gOSI is inflated at low rates")
ax.legend(fontsize=7.5, markerscale=2)
save(fig, "fig05_selectivity_distributions.png")

# %%
summary = []
for grp, s in UNITS.groupby("area_group"):
    r, rs = s[s.dg_responsive], s[s.sg_responsive]
    summary.append(dict(
        region=grp, n_units=len(s),
        dg_responsive=len(r),
        dg_orientation_selective=round(r.dg_orientation_selective.mean(), 3),
        dg_direction_selective=round(r.dg_direction_selective.mean(), 3),
        dg_median_gOSI=round(r.dg_gOSI.median(), 3),
        dg_median_z=round(r.dg_gOSI_z.median(), 2),
        sg_responsive=len(rs),
        sg_orientation_selective=round(rs.sg_orientation_selective.mean(), 3),
        sg_median_z=round(rs.sg_gOSI_z.median(), 2),
    ))
summary = pd.DataFrame(summary).set_index("region").loc[GROUP_ORDER]
pd.set_option("display.width", 200)
print(summary.T.to_string())

# %% [markdown]
# The z-score column is the key comparison, because it is the observed index expressed
# in units of that neuron's own shuffled-label null and so carries no firing-rate bias.
# Visual cortex sits many standard deviations above its null while hippocampus sits on
# top of it.

# %% [markdown]
# ### Figure 6: what orientations are preferred, and how sharply
#
# Two classical follow-up questions. Are all orientations equally represented, and how
# narrow is the average tuning curve?

# %%
fig, axes = plt.subplots(1, 3, figsize=(13, 4), gridspec_kw=dict(wspace=0.33))

ax = fig.add_subplot(1, 3, 1, projection="polar")
axes[0].axis("off")
sub = UNITS[(UNITS.area_group == "visual cortex") & UNITS.sg_orientation_selective]
pref = sub["sg_pref_ori"].dropna().values
bins = np.arange(0, 181, 30)
h, _ = np.histogram(pref, bins=bins)
theta = np.deg2rad(bins[:-1] + 15)
ax.bar(np.concatenate([theta, theta + np.pi]), np.concatenate([h, h]),
       width=np.deg2rad(30) * 0.92, color="#2166ac", alpha=0.85)
ax.set_theta_zero_location("E")
ax.set_thetagrids(np.arange(0, 360, 45), [f"{d}" for d in np.arange(0, 360, 45)], fontsize=8)
ax.set_rgrids([100, 200], labels=["100", "200"], angle=200, fontsize=6.5)
for lbl in ax.get_yticklabels():
    lbl.set_bbox(dict(facecolor="white", edgecolor="none", alpha=0.8, pad=0.8))
ax.set_title(f"Preferred orientation\nvisual cortex, static gratings (n={len(pref)})",
             fontsize=9.5, pad=20)

ax = axes[1]
counts = pd.Series(pref).value_counts().reindex(ORIS).fillna(0)
chi2, pval = stats.chisquare(counts.values)
ax.bar([f"{int(o)}" for o in ORIS], counts.values, color="#2166ac")
ax.axhline(counts.sum() / len(ORIS), color="k", ls="--", lw=1, label="uniform expectation")
ax.set_ylabel("number of units")
ax.set_xlabel("preferred orientation (deg)")
ax.set_title(f"Preferences are not uniform\n$\\chi^2$={chi2:.1f}, p={pval:.1e}", fontsize=9.5)
ax.legend(fontsize=8)

ax = axes[2]
for group in GROUP_ORDER:
    v = UNITS.loc[(UNITS.area_group == group) & UNITS.dg_orientation_selective &
                  (UNITS.vm_r2 > 0.6), "vm_hwhm"].dropna()
    v = v[(v > 0) & (v < 90)]
    if len(v) < 5:
        continue
    ax.hist(v, bins=np.arange(0, 92, 5), density=True, histtype="step", lw=1.8,
            color=GROUP_COLORS[group], label=f"{group} (n={len(v)}, median {v.median():.0f} deg)")
ax.axvspan(0, 20, color="0.85", alpha=0.6, lw=0, zorder=0)
ax.text(10, ax.get_ylim()[1] * 0.02, "below the\nresolution of\n45 deg sampling",
        ha="center", va="bottom", fontsize=6.5, color="0.35")
ax.set_xlabel("tuning half-width at half maximum (deg)")
ax.set_ylabel("probability density")
ax.set_ylim(0, ax.get_ylim()[1] * 1.35)
ax.set_title("Orientation tuning width\n(von Mises fits, drifting gratings)", fontsize=9.5)
ax.legend(fontsize=7.5, loc="upper right")
save(fig, "fig06_preferred_orientation_and_width.png")

# %% [markdown]
# ### Figure 7: is the measurement reproducible
#
# Three independent checks. Does the proportion of selective units hold up session by
# session, does a unit's selectivity measured with drifting gratings agree with the value
# measured with static gratings, and does it prefer the same orientation under both?

# %%
fig, axes = plt.subplots(1, 3, figsize=(13, 4.1), gridspec_kw=dict(wspace=0.34))

ax = axes[0]
sub = UNITS[UNITS.dg_responsive & UNITS.area_group.notna()]
piv = sub.groupby(["session", "area_group"])["dg_orientation_selective"].mean().unstack()
piv = piv[[c for c in GROUP_ORDER if c in piv.columns]]
piv.plot(kind="bar", ax=ax, color=[GROUP_COLORS[c] for c in piv.columns], width=0.8, legend=False)
ax.set_ylabel("fraction orientation selective")
ax.set_xlabel("session")
ax.tick_params(axis="x", rotation=45, labelsize=8)
ax.axhline(0.05, color="k", ls="--", lw=1)
ax.set_title("Reproducibility across sessions", fontsize=9.5)
ax.legend(fontsize=7.5, loc="upper right")

ax = axes[1]
s = UNITS[UNITS.dg_responsive & UNITS.sg_responsive & (UNITS.area_group == "visual cortex")]
ax.scatter(s["dg_gOSI"], s["sg_gOSI"], s=8, alpha=0.35, lw=0, color="#2166ac")
ok = np.isfinite(s["dg_gOSI"]) & np.isfinite(s["sg_gOSI"])
r, pv = stats.spearmanr(s["dg_gOSI"][ok], s["sg_gOSI"][ok])
ax.plot([0, 1], [0, 1], "k--", lw=0.8)
ax.set_xlabel("gOSI, drifting gratings")
ax.set_ylabel("gOSI, static gratings")
ax.set_title(f"Agreement between stimuli\nSpearman r={r:.2f}, p={pv:.1e}, n={ok.sum()}",
             fontsize=9.5)

ax = axes[2]
s2 = s[s.dg_orientation_selective & s.sg_orientation_selective]
diff = ((s2["dg_pref_ori"] - s2["sg_pref_ori"] + 90) % 180) - 90
ax.hist(diff, bins=np.arange(-90, 91, 10), color="#2166ac", alpha=0.8)
ax.axvline(0, color="k", ls="--", lw=1)
med = np.median(np.abs(diff))
ax.set_xlabel("preferred orientation difference (deg)\ndrifting minus static gratings")
ax.set_ylabel("number of units")
ax.set_title(f"Preferred orientation is stimulus invariant\n"
             f"median |difference| = {med:.0f} deg (n={len(diff)}, chance 45 deg)", fontsize=9.5)
save(fig, "fig07_reproducibility.png")

# %% [markdown]
# ## 7. Population decoding
#
# Single-unit tuning implies that the population as a whole carries information about
# orientation. A Poisson naive-Bayes decoder tests this directly: estimate each unit's
# mean spike count for each orientation on training trials, then assign each held-out
# trial the orientation that maximises the summed Poisson log-likelihood. This is a
# strong test because it operates on single trials, with no averaging.

# %%
POP_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]
N_REPEATS = 20
N_FOLDS = 5


def poisson_nb_accuracy(counts, labels, levels, n_folds=N_FOLDS, seed=0):
    """Cross-validated decoding accuracy and confusion matrix."""
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(len(labels)), n_folds)
    y = np.searchsorted(levels, labels)

    correct = 0
    conf = np.zeros((len(levels), len(levels)), int)
    for f in range(n_folds):
        test = folds[f]
        train = np.concatenate([folds[g] for g in range(n_folds) if g != f])
        lam = np.stack([counts[train][y[train] == c].mean(axis=0) for c in range(len(levels))])
        lam = np.clip(lam, 1e-3, None)
        with np.errstate(all="ignore"):
            ll = counts[test] @ np.log(lam).T - lam.sum(axis=1)
        pred = np.argmax(ll, axis=1)
        correct += np.sum(pred == y[test])
        np.add.at(conf, (y[test], pred), 1)
    return correct / len(labels), conf


def decoding_curve(counts, labels, levels, unit_pool, sizes=POP_SIZES,
                   n_repeats=N_REPEATS, seed=0):
    """Accuracy as a function of the number of randomly chosen units."""
    rng = np.random.default_rng(seed)
    rows = []
    for n in sizes:
        if n > len(unit_pool):
            continue
        for rep in range(n_repeats):
            cols = rng.choice(unit_pool, size=n, replace=False)
            acc, _ = poisson_nb_accuracy(counts[:, cols], labels, levels, seed=seed + rep)
            rows.append(dict(n_units=n, repeat=rep, accuracy=acc))
    return pd.DataFrame(rows)


# %% [markdown]
# ## 8. A Poisson GLM with NeMoS
#
# The drifting-grating stimulus varies direction and temporal frequency together, so a
# neuron could appear direction tuned simply because it prefers a temporal frequency
# that happened to co-occur with certain directions. A GLM separates the two. Direction
# enters through a cyclic B-spline basis (which enforces that 0 and 360 deg are the same
# angle) and log temporal frequency through an ordinary B-spline basis, as additive
# terms. Comparing nested models on held-out trials isolates what direction contributes.
#
# Model quality is reported as McFadden's pseudo-R-squared against an intercept-only
# model. Note that this requires the *full* Poisson log-likelihood including the
# `log(k!)` normalising term. That term is constant across models and cancels in a
# likelihood difference, but pseudo-R-squared is a likelihood *ratio*; dropping it
# leaves the log-likelihood positive for units firing more than about one spike per
# trial and makes the ratio meaningless.

# %%
import nemos as nmo


def poisson_ll(counts, rate):
    """Poisson log-likelihood per neuron, including the log(k!) normalising term."""
    rate = np.clip(rate, 1e-8, None)
    return np.sum(counts * np.log(rate) - rate - gammaln(counts + 1), axis=0)


def glm_direction_contribution(counts, direction, temporal_freq, n_folds=N_FOLDS, seed=0):
    """Cross-validated pseudo-R^2 for nested GLMs of trial spike counts."""
    dir_basis = nmo.basis.CyclicBSplineEval(n_basis_funcs=8, bounds=(0.0, 360.0), label="direction")
    tf_basis = nmo.basis.BSplineEval(n_basis_funcs=4, bounds=(0.0, 4.0), label="log2_tf")

    X_dir = np.asarray(dir_basis.compute_features(direction))
    X_tf = np.asarray(tf_basis.compute_features(np.log2(temporal_freq)))
    designs = {
        "tf_only": X_tf,
        "direction_only": X_dir,
        "direction_plus_tf": np.hstack([X_dir, X_tf]),
    }

    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(len(direction)), n_folds)

    ll = {k: np.zeros(counts.shape[1]) for k in list(designs) + ["null"]}
    for f in range(n_folds):
        test = folds[f]
        train = np.concatenate([folds[g] for g in range(n_folds) if g != f])
        null_rate = np.broadcast_to(counts[train].mean(axis=0), (len(test), counts.shape[1]))
        ll["null"] += poisson_ll(counts[test], null_rate)
        for name, X in designs.items():
            model = nmo.glm.PopulationGLM(regularizer="Ridge", regularizer_strength=1e-3,
                                          solver_name="LBFGS",
                                          solver_kwargs=dict(maxiter=5000, tol=1e-6))
            model.fit(X[train], counts[train])
            ll[name] += poisson_ll(counts[test], np.asarray(model.predict(X[test])))

    n_trials = len(direction)
    out = {}
    for name in designs:
        with np.errstate(invalid="ignore", divide="ignore"):
            out[f"pr2_{name}"] = 1 - ll[name] / ll["null"]
        # Held-out log-likelihood gain in nats per trial, which stays interpretable for
        # units with very few spikes where the ratio is unstable.
        out[f"nats_{name}"] = (ll[name] - ll["null"]) / n_trials

    model = nmo.glm.PopulationGLM(regularizer="Ridge", regularizer_strength=1e-3,
                                  solver_name="LBFGS", solver_kwargs=dict(maxiter=5000, tol=1e-6))
    model.fit(designs["direction_plus_tf"], counts)
    grid = np.linspace(0, 360, 181, endpoint=False)
    Xg = np.hstack([
        np.asarray(dir_basis.compute_features(grid)),
        np.tile(np.asarray(tf_basis.compute_features(np.log2(temporal_freq))).mean(axis=0),
                (len(grid), 1)),
    ])
    out["glm_curve"] = np.asarray(model.predict(Xg))
    out["glm_grid"] = grid
    return out


# %%
DECODE_CACHE = Path("decoding.parquet")
GLM_CACHE = Path("glm_scores.parquet")
EXTRAS_CACHE = Path("decode_extras.pkl")

if DECODE_CACHE.exists() and GLM_CACHE.exists() and EXTRAS_CACHE.exists():
    DECODE = pd.read_parquet(DECODE_CACHE)
    GLM = pd.read_parquet(GLM_CACHE)
    with open(EXTRAS_CACHE, "rb") as f:
        EXTRAS = pickle.load(f)
    print("loaded cached decoding and GLM results")
else:
    decode_rows, conf_store, glm_rows, glm_curves = [], {}, [], {}
    for res in tqdm(RESULTS, desc="decoding and GLM"):
        sid, meta = res["session"], res["meta"]

        sg = res["sg_table"]
        sg_counts = np.rint(res["sg_rates"] * (SG_WINDOW[1] - SG_WINDOW[0])).astype(int)
        oris = np.sort(sg["orientation"].unique())
        dg = res["dg_table"]
        dg_counts = np.rint(res["dg_rates"] * (DG_WINDOW[1] - DG_WINDOW[0])).astype(int)
        dirs = np.sort(dg["orientation"].unique())

        for group in GROUP_ORDER:
            pool = np.flatnonzero(meta["area_group"].values == group)
            if len(pool) < 4:
                continue
            for stim, counts, labels, levels in [
                ("static gratings (6 orientations)", sg_counts, sg["orientation"].values, oris),
                ("drifting gratings (8 directions)", dg_counts, dg["orientation"].values, dirs),
            ]:
                curve = decoding_curve(counts, labels, levels, pool, seed=int(sid) % 1000)
                curve["session"], curve["area_group"], curve["stimulus"] = sid, group, stim
                curve["chance"] = 1 / len(levels)
                decode_rows.append(curve)

        v1 = np.flatnonzero(meta["area"].values == "VISp")
        if len(v1) >= 10:
            acc, conf = poisson_nb_accuracy(sg_counts[:, v1], sg["orientation"].values, oris)
            conf_store[sid] = dict(conf=conf, acc=acc, n_units=len(v1), levels=oris)

        ctx = np.flatnonzero(meta["area_group"].values == "visual cortex")
        g = glm_direction_contribution(dg_counts[:, ctx], dg["orientation"].values,
                                       dg["temporal_frequency"].values)
        glm_rows.append(pd.DataFrame({
            "uid": meta["uid"].values[ctx], "area": meta["area"].values[ctx], "session": sid,
            **{k: g[k] for k in g if k.startswith(("pr2_", "nats_"))},
        }))
        glm_curves[sid] = dict(curve=g["glm_curve"], grid=g["glm_grid"],
                               uid=meta["uid"].values[ctx])

    DECODE = pd.concat(decode_rows, ignore_index=True)
    GLM = pd.concat(glm_rows, ignore_index=True)
    EXTRAS = dict(conf=conf_store, glm_curves=glm_curves)
    DECODE.to_parquet(DECODE_CACHE)
    GLM.to_parquet(GLM_CACHE)
    with open(EXTRAS_CACHE, "wb") as f:
        pickle.dump(EXTRAS, f)

# %% [markdown]
# ### Figure 8: decoding accuracy

# %%
fig = plt.figure(figsize=(13, 4.4))
gs = GridSpec(1, 3, figure=fig, wspace=0.33)

for col, stim in enumerate(sorted(DECODE["stimulus"].unique())):
    ax = fig.add_subplot(gs[0, col])
    d = DECODE[DECODE.stimulus == stim]
    for group in GROUP_ORDER:
        g = d[d.area_group == group].groupby(["n_units", "session"])["accuracy"].mean().reset_index()
        if len(g) == 0:
            continue
        agg = g.groupby("n_units")["accuracy"].agg(["mean", "sem", "count"])
        agg = agg[agg["count"] >= 2]
        ax.errorbar(agg.index, agg["mean"], yerr=agg["sem"], marker="o", ms=4, lw=1.6,
                    capsize=2.5, color=GROUP_COLORS[group], label=group)
    ax.axhline(d["chance"].iloc[0], color="k", ls="--", lw=1, label="chance")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("number of simultaneously recorded units")
    ax.set_ylabel("decoding accuracy")
    ax.set_title(stim, fontsize=10)
    ax.legend(fontsize=7.5, loc="upper left")

ax = fig.add_subplot(gs[0, 2])
conf = sum(v["conf"] for v in EXTRAS["conf"].values()).astype(float)
levels = list(EXTRAS["conf"].values())[0]["levels"]
conf /= conf.sum(axis=1, keepdims=True)
im = ax.imshow(conf, cmap="magma", vmin=0, vmax=conf.max())
ax.set_xticks(range(len(levels)))
ax.set_xticklabels([f"{int(o)}" for o in levels])
ax.set_yticks(range(len(levels)))
ax.set_yticklabels([f"{int(o)}" for o in levels])
ax.set_xlabel("decoded orientation (deg)")
ax.set_ylabel("presented orientation (deg)")
accs = [v["acc"] for v in EXTRAS["conf"].values()]
ns = [v["n_units"] for v in EXTRAS["conf"].values()]
ax.set_title(f"VISp population decoder, static gratings\n"
             f"mean accuracy {np.mean(accs):.2f} (chance {1 / len(levels):.2f}), "
             f"{int(np.mean(ns))} units/session", fontsize=9)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="P(decoded | presented)")
for i in range(len(levels)):
    for j in range(len(levels)):
        ax.text(j, i, f"{conf[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                color="w" if conf[i, j] < conf.max() * 0.6 else "k")
fig.suptitle("Single-trial decoding of grating orientation from population activity",
             y=1.04, fontsize=12)
save(fig, "fig08_population_decoding.png")

# %% [markdown]
# ### Figure 9: the GLM

# %%
merged = GLM.merge(UNITS[["uid", "dg_gOSI", "dg_responsive", "dg_orientation_selective",
                          "dg_pref_tf"]], on="uid")
m = merged[merged.dg_responsive]

fig, axes = plt.subplots(1, 3, figsize=(13, 4.1), gridspec_kw=dict(wspace=0.34))

ax = axes[0]
labels = ["temporal\nfrequency only", "direction\nonly", "direction +\ntemporal frequency"]
cols = ["pr2_tf_only", "pr2_direction_only", "pr2_direction_plus_tf"]
parts = ax.violinplot([m[c].clip(-0.05, None) for c in cols], showmedians=True, widths=0.8)
for b in parts["bodies"]:
    b.set_facecolor("#2166ac")
    b.set_alpha(0.5)
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(labels, fontsize=8.5)
ax.axhline(0, color="k", lw=0.8, ls="--")
ax.set_ylabel("cross-validated pseudo-$R^2$")
ax.set_title(f"GLM model comparison\n(visual cortex, n={len(m)} responsive units)", fontsize=9.5)
for i, c in enumerate(cols):
    ax.text(i + 1, m[c].median(), f" {m[c].median():.3f}", fontsize=8, va="center")

ax = axes[1]
gain = m["pr2_direction_plus_tf"] - m["pr2_tf_only"]
ax.scatter(m["dg_gOSI"], gain, s=8, alpha=0.4, lw=0, color="#2166ac")
ok = np.isfinite(m["dg_gOSI"]) & np.isfinite(gain)
r, pv = stats.spearmanr(m["dg_gOSI"][ok], gain[ok])
ax.axhline(0, color="k", lw=0.8, ls="--")
ax.set_xlabel("gOSI")
ax.set_ylabel("pseudo-$R^2$ gained by adding direction")
ax.set_title(f"Selectivity predicts GLM improvement\nSpearman r={r:.2f}, p={pv:.1e}", fontsize=9.5)

ax = axes[2]
sid = SESSION_IDS[0]
cur = EXTRAS["glm_curves"][sid]
uids = list(cur["uid"])
best = m[(m.session == sid) & m.dg_orientation_selective].sort_values("dg_gOSI", ascending=False)
for k, (_, row) in enumerate(best.head(4).iterrows()):
    j = uids.index(row.uid)
    c = plt.get_cmap("tab10")(k)
    rate = cur["curve"][:, j]
    ax.plot(cur["grid"], rate / rate.max(), color=c, lw=1.6, label=f"{row.uid.split('_')[1]}")
    idx = int(np.flatnonzero(UNITS.uid.values == row.uid)[0]) - int(UNITS.index[UNITS.session == sid][0])
    emp = ARRAYS[sid]["dg_dir_means"][idx]
    ax.plot(DIRS, emp / emp.max(), "o", color=c, ms=5, mfc="none")
ax.set_xticks(np.arange(0, 361, 45))
ax.set_xlabel("grating direction (deg)")
ax.set_ylabel("normalised response")
ax.set_ylim(-0.05, 1.5)
ax.set_title("GLM tuning (lines) vs. measured (circles)\nexample VISp units", fontsize=9.5)
ax.legend(fontsize=6.5, ncol=2, loc="upper center", title="unit", title_fontsize=6.5)
save(fig, "fig09_glm.png")

# %%
print("Median cross-validated pseudo-R^2, responsive cortical units:")
print(m[cols].median().round(4).to_string())
print("\nMedian held-out log-likelihood gain over an intercept-only model (nats/trial):")
print(m[["nats_tf_only", "nats_direction_only", "nats_direction_plus_tf"]].median().round(3).to_string())

# %% [markdown]
# ## 9. Summary

# %%
ctx = UNITS[(UNITS.area_group == "visual cortex")]
hip = UNITS[(UNITS.area_group == "hippocampus")]
v1 = UNITS[UNITS.area == "VISp"]
width = UNITS.loc[(UNITS.area_group == "visual cortex") & UNITS.dg_orientation_selective &
                  (UNITS.vm_r2 > 0.6), "vm_hwhm"].dropna()
width = width[(width > 0) & (width < 90)]
best_dec = (DECODE[DECODE.stimulus.str.startswith("drifting")]
            .groupby(["area_group", "n_units"])["accuracy"].mean())

print(f"units analysed                        : {len(UNITS)} over {len(RESULTS)} sessions")
print(f"visual cortex, drifting-grating resp. : {int(ctx.dg_responsive.sum())}")
print(f"  orientation selective (p < 0.05)    : {ctx[ctx.dg_responsive].dg_orientation_selective.mean():.1%}")
print(f"  direction selective   (p < 0.05)    : {ctx[ctx.dg_responsive].dg_direction_selective.mean():.1%}")
print(f"hippocampus, drifting-grating resp.   : {int(hip.dg_responsive.sum())}")
print(f"  orientation selective (p < 0.05)    : {hip[hip.dg_responsive].dg_orientation_selective.mean():.1%}")
print(f"VISp median gOSI (responsive)         : {v1[v1.dg_responsive].dg_gOSI.median():.3f}")
print(f"VISp median gOSI z-score              : {v1[v1.dg_responsive].dg_gOSI_z.median():.2f}")
print(f"median tuning half-width (cortex)     : {width.median():.0f} deg")
print(f"best 8-way direction decoding, cortex : {best_dec['visual cortex'].max():.1%} (chance 12.5%)")
print(f"                        hippocampus   : {best_dec['hippocampus'].max():.1%}")

# %% [markdown]
# **What the data show.** Orientation selectivity is present, strong, and specific to
# the visual system. Roughly three quarters of visually responsive cortical neurons have
# an orientation preference that survives a shuffled-label permutation test, with a
# median tuning half-width near 37 deg. The same measurement applied to hippocampal
# neurons recorded on the same probes in the same sessions returns a rate close to the
# nominal false-positive rate of the test, which is the control that makes the cortical
# result interpretable. Visual thalamus sits in between, consistent with the classical
# picture in which orientation tuning is weak in the thalamic input and sharpened in
# cortex.
#
# Three further results support the interpretation. A neuron's preferred orientation is
# the same whether it is measured with 2 s drifting gratings or 0.25 s static gratings,
# so the preference is a property of the neuron rather than of one stimulus. Preferred
# orientations are not uniformly distributed across the population, with an
# over-representation near 0 deg; a non-uniformity of this kind is widely reported in
# mouse visual cortex, though the six orientations sampled here are too coarse to
# characterise its shape in detail. And the tuning is strong enough that a naive-Bayes
# decoder recovers which of eight grating directions was shown on a single 2 s trial
# from a few dozen cortical neurons, far above chance, while the same decoder applied to
# hippocampal populations stays near chance.
#
# **Limitations.** Tuning widths come from fitting five parameters to eight sampled
# directions, so widths below about 20 deg are not resolvable and the reported median
# should be read as an upper bound on sharpness.
#
# The hippocampal control behaves as intended for drifting gratings, where 6.2% of
# responsive units pass at p < 0.05 against a nominal 5%. For static gratings the rate
# is 19% (13 of 69 units), above nominal and spread across sessions rather than confined
# to one. Two alternative nulls were tried to see whether firing-rate nonstationarity
# was responsible: permuting labels only within contiguous blocks of trials, which
# preserves slow drift, and circularly shifting the label sequence, which preserves both
# slow drift and short-lag correlations. Both gave essentially the same rate as the plain
# shuffle, so the excess is not explained by either. What can be said is that the effect
# size is negligible whatever its cause: the median hippocampal z-score is 0.24 against
# 9.78 in visual cortex, and only 69 of 964 hippocampal units are visually responsive at
# all. The regional contrast the argument rests on is not in question, but the
# static-grating hippocampal rate should not be read as a clean 5%.
