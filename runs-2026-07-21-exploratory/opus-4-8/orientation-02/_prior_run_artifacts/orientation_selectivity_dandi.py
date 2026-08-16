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
# ## DANDI:000021 — Allen Institute Visual Coding, Neuropixels (Brain Observatory 1.1)
#
# Orientation selectivity is the property, first described by Hubel and Wiesel, that a
# neuron in visual cortex responds strongly to an edge or grating at one orientation and
# weakly or not at all to the orthogonal orientation. It is the canonical example of a
# receptive-field property that emerges in cortex: retinal and thalamic relay cells have
# largely circularly symmetric receptive fields, so a strong dependence of firing rate on
# stimulus orientation is expected to be much weaker upstream of V1 than within it.
#
# This notebook demonstrates orientation selectivity directly from Neuropixels recordings
# in the Allen Institute Visual Coding dataset. Head-fixed mice viewed full-field drifting
# gratings (8 directions of motion at 45 degree steps, 5 temporal frequencies, 2 s per
# presentation, 15 repeats per condition) and static gratings (6 orientations at 30 degree
# steps, 5 spatial frequencies, 4 phases, 0.25 s per presentation, roughly 50 repeats per
# orientation), interleaved with blank sweeps that give a spontaneous-rate baseline.
# Simultaneous recordings span primary visual cortex (VISp), four higher visual areas
# (VISl, VISrl, VISam, VISpm) and the dorsal lateral geniculate nucleus (LGd), which lets
# us compare cortex against its thalamic input in the same animal and the same stimulus
# presentations.
#
# ### What the analysis does
#
# 1. Streams the NWB session files from the DANDI Archive with `remfile` plus a disk cache,
#    never downloading a whole file.
# 2. Builds a Pynapple `TsGroup` of spike times and computes per-trial firing rates with
#    `TsGroup.count` over an `IntervalSet` of stimulus presentations.
# 3. Measures direction and orientation tuning curves, a global orientation selectivity
#    index (gOSI), a direction selectivity index (DSI), and tests tuning with a one-way
#    ANOVA across directions plus a trial-label permutation test on gOSI.
# 4. Fits Poisson GLMs with NeMoS using cyclic B-spline bases over stimulus direction
#    (360 degree period) and over stimulus orientation (180 degree period), and compares
#    their cross-validated pseudo-R squared.
# 5. Decodes stimulus orientation from population spike counts and compares VISp with LGd.
# 6. Repeats the whole pipeline on 8 sessions from 8 mice and pools the units.
#
# ### Headline result
#
# Individual VISp units show sharply bimodal direction tuning curves with peaks 180 degrees
# apart, that is, tuning to orientation rather than to direction of motion. Pooled across
# 8 sessions, 64 percent of VISp units are significantly orientation tuned with a median
# gOSI of 0.22, against 46 percent and 0.11 in LGd. The orientation preference measured
# with drifting gratings agrees with the preference measured independently with static
# gratings.

# %% [markdown]
# ## 1. Setup

# %%
import os
import pickle
import warnings

import h5py
import matplotlib
matplotlib.use("Agg")  # headless: figures are written to disk, never shown
import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap
import remfile
from scipy import stats
from tqdm.auto import tqdm

import nemos as nmo
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

plt.rcParams.update({
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 130,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})

REG_COLORS = {"VISp": "#1f77b4", "VISl": "#2ca02c", "VISrl": "#9467bd",
              "VISam": "#8c564b", "VISpm": "#e377c2", "LGd": "#d62728"}
REGIONS = ["VISp", "VISl", "VISrl", "VISam", "VISpm", "LGd"]

# Eight sessions from DANDI:000021, one per mouse. The first is the prototype session
# used for all single-session figures.
SESSIONS = [
    ("58703c97-c0a9-4736-b684-73c85c1a444a", "715093703"),
    ("02291b99-e583-498b-9929-b68bba2c50e2", "719161530"),
    ("b4aeeb19-cdc6-4895-ab7b-bc8a688cf6f5", "732592105"),
    ("96c200cf-29c2-457a-b2f3-99f11de5b039", "742951821"),
    ("286c7b06-3cde-4261-9090-e6fbe6c81945", "750332458"),
    ("be9f8fd8-8f16-4a66-acc6-9e04697650f3", "751348571"),
    ("4513d0c9-1e2b-4c8a-aa22-ae81822537c9", "760693773"),
    ("d6f3e82b-aca9-43fb-9810-048fc2124d50", "762120172"),
]
PROTOTYPE = SESSIONS[0][1]

STREAM_CACHE = "/tmp/remfile_cache"       # byte-range cache for the NWB files
N_PERM = 500                              # permutations for the gOSI null distribution
os.makedirs(STREAM_CACHE, exist_ok=True)

# %% [markdown]
# ## 2. Streaming the NWB files from DANDI
#
# Each session file is 2 to 3 GB. We open it over HTTP with `remfile`, which serves h5py
# byte-range requests and caches the fetched chunks on disk, and then pull out only what
# the analysis needs: spike times for well-isolated units in the six areas of interest,
# the electrode-to-brain-region mapping, and the grating stimulus tables. The extracted
# subset is written to a local pickle so re-running the notebook is fast.
#
# Unit selection uses the Allen quality metrics: `quality == 'good'`, ISI violations
# below 0.5, amplitude cutoff below 0.1, and presence ratio above 0.9.

# %%
def _decode(arr):
    return np.array([x.decode() if isinstance(x, (bytes, np.bytes_)) else x for x in arr])


def extract_session(asset_id, session_id, cache_dir="."):
    """Stream one NWB session from DANDI and cache the pieces we analyze."""
    out = os.path.join(cache_dir, f"cache_{session_id}.pkl")
    if os.path.exists(out):
        return out

    url = (f"https://api.dandiarchive.org/api/dandisets/000021/versions/draft/"
           f"assets/{asset_id}/download/")
    h = h5py.File(remfile.File(url, disk_cache=remfile.DiskCache(STREAM_CACHE)), "r")

    electrodes = h["general/extracellular_ephys/electrodes"]
    chan2loc = dict(zip(electrodes["id"][:], _decode(electrodes["location"][:])))

    units = h["units"]
    quality = _decode(units["quality"][:])
    region = np.array([chan2loc.get(p, "?") for p in units["peak_channel_id"][:]])
    keep = (
        (quality == "good")
        & np.isin(region, REGIONS)
        & (units["isi_violations"][:] < 0.5)
        & (units["amplitude_cutoff"][:] < 0.1)
        & (units["presence_ratio"][:] > 0.9)
    )

    unit_id = units["id"][:]
    idx_end = units["spike_times_index"][:]
    idx_start = np.concatenate([[0], idx_end[:-1]])
    spike_ds = units["spike_times"]
    spikes = {int(unit_id[i]): spike_ds[idx_start[i]:idx_end[i]]
              for i in tqdm(np.where(keep)[0], desc=f"{session_id} spikes", leave=False)}

    meta = dict(unit_id=unit_id[keep], region=region[keep],
                snr=units["snr"][:][keep],
                isolation=units["isolation_distance"][:][keep],
                isi_viol=units["isi_violations"][:][keep],
                waveform_duration=units["waveform_duration"][:][keep])

    stim = {}
    for name in ["drifting_gratings_presentations", "static_gratings_presentations"]:
        table = h["intervals"][name]
        cols = {}
        for c in table.keys():
            if c in ("tags", "tags_index", "timeseries", "timeseries_index", "id"):
                continue
            arr = table[c][:]
            cols[c] = _decode(arr) if arr.dtype.kind in "SO" else arr
        stim[name] = cols

    running = None
    node = h.get("processing/running/running_speed")
    if node is not None and "data" in node:
        running = dict(t=node["timestamps"][:], v=node["data"][:])

    pickle.dump(dict(meta=meta, spikes=spikes, stim=stim, running=running,
                     session_id=session_id), open(out, "wb"))
    return out


def load_session(session_id, cache_dir="."):
    """Load a cached session as (raw dict, pynapple TsGroup).

    `TsGroup` indexes units by unit id in ascending order, and every matrix computed
    from it (spike counts, tuning curves) has its columns in that same order. The unit
    ids stored in the NWB `units` table are *not* ascending, so the per-unit metadata
    read out of the file has to be permuted into TsGroup order before it can be used to
    label those columns. Getting this wrong silently scrambles the area labels, so the
    reordering is done once, here, and the reordered arrays are what the rest of the
    analysis uses.
    """
    d = pickle.load(open(os.path.join(cache_dir, f"cache_{session_id}.pkl"), "rb"))
    tsgroup = nap.TsGroup(
        {int(k): nap.Ts(t=v) for k, v in d["spikes"].items()},
        metadata={"region": d["meta"]["region"],
                  "snr": d["meta"]["snr"],
                  "waveform_duration": d["meta"]["waveform_duration"]},
    )
    order = np.argsort(np.asarray(d["meta"]["unit_id"]))
    d["meta"] = {k: np.asarray(v)[order] for k, v in d["meta"].items()}
    assert np.array_equal(d["meta"]["unit_id"], np.array(list(tsgroup.keys())))
    assert np.array_equal(d["meta"]["region"], np.asarray(tsgroup.region))
    return d, tsgroup


for asset_id, session_id in tqdm(SESSIONS, desc="sessions"):
    extract_session(asset_id, session_id)

data, tsg = load_session(PROTOTYPE)
region = np.array(data["meta"]["region"])
unit_ids = np.array(list(tsg.keys()))
print(tsg)
print("\nunits per area:", {r: int((region == r).sum()) for r in REGIONS})

# %% [markdown]
# ## 3. Inspecting the stimulus tables
#
# Note the Allen convention: for **drifting** gratings the `orientation` column holds the
# direction of motion (0 to 315 degrees), whereas for **static** gratings it holds the
# orientation of the grating itself (0 to 150 degrees). Rows with `orientation == NaN`
# are blank sweeps, which we use as the spontaneous baseline.

# %%
dg_table = data["stim"]["drifting_gratings_presentations"]
sg_table = data["stim"]["static_gratings_presentations"]
for name, table in [("drifting", dg_table), ("static", sg_table)]:
    ori = table["orientation"].astype(float)
    dur = table["stop_time"] - table["start_time"]
    print(f"{name:9s} n={len(ori):5d}  blanks={np.isnan(ori).sum():4d}  "
          f"levels={np.unique(ori[~np.isnan(ori)])}  "
          f"duration={dur.mean():.3f} +/- {dur.std():.4f} s")

# %% [markdown]
# ## 4. Figure 1: raw spiking during drifting gratings
#
# Before any analysis, look at the data. Each colored band is one 2 s grating
# presentation; every tick in the middle panel is a spike. Even by eye, several VISp
# units fire in bursts locked to particular gratings, whereas LGd units fire densely
# throughout.

# %%
block = dg_table["stimulus_block"].astype(float)
dg_ori = dg_table["orientation"].astype(float)
dg_tf = dg_table["temporal_frequency"].astype(float)

t0 = dg_table["start_time"][block == 2.0][0]
t1 = t0 + 40
shown = (dg_table["start_time"] >= t0) & (dg_table["start_time"] < t1)

visp_rows = np.where(region == "VISp")[0]
lgd_rows = np.where(region == "LGd")[0]
row_order = np.concatenate([visp_rows, lgd_rows])

fig = plt.figure(figsize=(11, 6.5))
gs = fig.add_gridspec(3, 1, height_ratios=[1, 3.6, 1], hspace=0.32)

ax = fig.add_subplot(gs[0])
dir_levels = np.arange(0, 360, 45)
dir_color = {d: plt.cm.twilight(i / len(dir_levels)) for i, d in enumerate(dir_levels)}
for k in np.where(shown)[0]:
    o = dg_ori[k]
    ax.axvspan(dg_table["start_time"][k] - t0, dg_table["stop_time"][k] - t0,
               color=dir_color[o] if not np.isnan(o) else "0.85", alpha=0.9, lw=0)
    if not np.isnan(o):
        ax.text((dg_table["start_time"][k] + dg_table["stop_time"][k]) / 2 - t0, 0.5,
                f"{int(o)}", ha="center", va="center", fontsize=6.5,
                color="w", fontweight="bold")
ax.set_xlim(0, t1 - t0)
ax.set_ylim(0, 1)
ax.set_yticks([])
ax.set_ylabel("stim")
ax.set_title("Drifting-grating presentations "
             "(labels = direction of motion in deg; grey = blank sweep)")

ax = fig.add_subplot(gs[1])
epoch = nap.IntervalSet(start=t0, end=t1)
for row, u in enumerate(row_order):
    s = tsg[unit_ids[u]].restrict(epoch).t - t0
    ax.plot(s, np.full_like(s, row), "|", ms=2.6, color=REG_COLORS[region[u]],
            alpha=0.9, mew=0.5)
ax.axhline(len(visp_rows) - 0.5, color="k", lw=0.8, ls="--")
ax.set_xlim(0, t1 - t0)
ax.set_ylim(-1, len(row_order))
ax.set_ylabel("unit")
ax.text(1.005, 0.98, "LGd", color=REG_COLORS["LGd"], ha="left", va="top",
        transform=ax.transAxes, fontweight="bold", rotation=90)
ax.text(1.005, 0.02, "VISp", color=REG_COLORS["VISp"], ha="left", va="bottom",
        transform=ax.transAxes, fontweight="bold", rotation=90)

ax = fig.add_subplot(gs[2])
rt, rv = data["running"]["t"], data["running"]["v"]
k = (rt >= t0) & (rt <= t1)
ax.plot(rt[k] - t0, rv[k], color="0.35", lw=0.8)
ax.set_xlim(0, t1 - t0)
ax.set_xlabel("time from block onset (s)")
ax.set_ylabel("running\n(cm/s)")
fig.suptitle(f"Session {PROTOTYPE} — raw spiking during drifting gratings", y=0.955)
fig.savefig("fig01_raw_activity.png")
plt.close(fig)

# %% [markdown]
# ## 5. Trial-resolved firing rates and tuning metrics
#
# Pynapple does the heavy lifting: an `IntervalSet` of stimulus presentations passed to
# `TsGroup.count` returns a trials-by-units count matrix in one call, which we divide by
# the presentation duration to get rates.
#
# Two selectivity indices are used, both computed on the mean rate per direction:
#
# * **global OSI** = |sum_k r_k exp(2 i theta_k)| / sum_k r_k. Doubling the angle folds
#   opposite directions onto the same orientation, so a neuron that fires equally at
#   45 and 225 degrees but not at 135 or 315 gets a high gOSI.
# * **DSI** = (r_pref - r_null) / (r_pref + r_null), where r_null is the response to the
#   direction opposite the preferred one.
#
# Tuning is called significant when a one-way ANOVA across the 8 directions gives
# p < 0.01 **and** a permutation test that shuffles direction labels within the unit's
# preferred temporal frequency gives p < 0.05 for the observed gOSI. The preferred
# temporal frequency is chosen from the direction-averaged response, so that choice
# carries no information about direction tuning and does not bias the test.

# %%
def trial_rates(tsgroup, table, mask):
    """Firing rate per trial (rows) and unit (columns) for the selected trials."""
    start, stop = table["start_time"][mask], table["stop_time"][mask]
    counts = tsgroup.count(ep=nap.IntervalSet(start=start, end=stop))
    return np.asarray(counts.values) / (stop - start)[:, None]


def ori_metrics(rates_by_dir, dirs_deg):
    """Selectivity indices from the mean rate at each direction."""
    r = np.clip(rates_by_dir, 0, None)
    theta = np.deg2rad(dirs_deg)
    total = r.sum()
    if total <= 0:
        return dict(gOSI=np.nan, gDSI=np.nan, OSI=np.nan, DSI=np.nan,
                    pref_dir=np.nan, pref_ori=np.nan)
    gOSI = np.abs((r * np.exp(2j * theta)).sum()) / total
    gDSI = np.abs((r * np.exp(1j * theta)).sum()) / total
    pref_ori = np.rad2deg(np.angle((r * np.exp(2j * theta)).sum()) / 2) % 180

    i = int(np.argmax(r))
    pref = dirs_deg[i]

    def rate_at(d):
        j = int(np.argmin(np.abs(((dirs_deg - d + 180) % 360) - 180)))
        return r[j]

    r_pref = r[i]
    r_orth = 0.5 * (rate_at(pref + 90) + rate_at(pref - 90))
    r_null = rate_at(pref + 180)
    OSI = (r_pref - r_orth) / (r_pref + r_orth) if (r_pref + r_orth) > 0 else np.nan
    DSI = (r_pref - r_null) / (r_pref + r_null) if (r_pref + r_null) > 0 else np.nan
    return dict(gOSI=gOSI, gDSI=gDSI, OSI=OSI, DSI=DSI,
                pref_dir=pref, pref_ori=pref_ori)


def drifting_analysis(tsgroup, table, n_perm=N_PERM, seed=0):
    """Direction tuning curves, selectivity indices and significance tests."""
    ori = table["orientation"].astype(float)
    tf = table["temporal_frequency"].astype(float)
    valid, blank = ~np.isnan(ori), np.isnan(ori)
    dirs, tfs = np.unique(ori[valid]), np.unique(tf[valid])

    R = trial_rates(tsgroup, table, valid)
    R_blank = trial_rates(tsgroup, table, blank)
    o, f = ori[valid], tf[valid]
    n_units = R.shape[1]

    grid = np.zeros((len(dirs), len(tfs), n_units))
    for i, d in enumerate(dirs):
        for j, t in enumerate(tfs):
            grid[i, j] = R[(o == d) & (f == t)].mean(0)
    pref_tf_idx = np.argmax(grid.mean(0), axis=0)   # direction-averaged: no circularity

    keys = ["gOSI", "gDSI", "OSI", "DSI", "pref_dir", "pref_ori",
            "anova_p", "perm_p", "evoked", "baseline", "pref_tf"]
    out = {k: np.full(n_units, np.nan) for k in keys}
    tc = np.zeros((len(dirs), n_units))
    tc_sem = np.zeros((len(dirs), n_units))
    # Two tuning curves from disjoint halves of the trials at each direction. Figure 10
    # uses one half to pick each unit's preferred direction and the other to measure the
    # curve, so that peak-alignment cannot manufacture tuning out of noise.
    tc_a = np.zeros((len(dirs), n_units))
    tc_b = np.zeros((len(dirs), n_units))
    rng = np.random.default_rng(seed)

    for u in range(n_units):
        at_pref_tf = f == tfs[pref_tf_idx[u]]
        groups = [R[at_pref_tf & (o == d), u] for d in dirs]
        means = np.array([g.mean() for g in groups])
        tc[:, u] = means
        tc_sem[:, u] = [g.std(ddof=1) / np.sqrt(len(g)) for g in groups]
        tc_a[:, u] = [g[0::2].mean() for g in groups]
        tc_b[:, u] = [g[1::2].mean() for g in groups]
        out["anova_p"][u] = stats.f_oneway(*groups).pvalue
        for k, v in ori_metrics(means, dirs).items():
            out[k][u] = v
        out["pref_tf"][u] = tfs[pref_tf_idx[u]]
        out["evoked"][u] = R[at_pref_tf, u].mean()
        out["baseline"][u] = R_blank[:, u].mean()

        y, labels = R[at_pref_tf, u], o[at_pref_tf]
        null = np.empty(n_perm)
        for p in range(n_perm):
            yp = rng.permutation(y)
            null[p] = ori_metrics(np.array([yp[labels == d].mean() for d in dirs]),
                                  dirs)["gOSI"]
        out["perm_p"][u] = (np.sum(null >= out["gOSI"][u]) + 1) / (n_perm + 1)

    return dict(dirs=dirs, tfs=tfs, grid=grid, tc=tc, tc_sem=tc_sem,
                tc_a=tc_a, tc_b=tc_b,
                R=R, ori=o, tf=f, metrics=out, blank_rates=R_blank)


def static_analysis(tsgroup, table):
    """Orientation tuning from static gratings, at each unit's preferred spatial freq."""
    ori = table["orientation"].astype(float)
    sf = table["spatial_frequency"].astype(float)
    valid = ~np.isnan(ori)
    oris, sfs = np.unique(ori[valid]), np.unique(sf[valid])

    R = trial_rates(tsgroup, table, valid)
    o, s = ori[valid], sf[valid]
    n_units = R.shape[1]
    pref_sf_idx = np.argmax(np.stack([R[s == v].mean(0) for v in sfs]), axis=0)

    tc = np.zeros((len(oris), n_units))
    tc_sem = np.zeros_like(tc)
    gosi = np.full(n_units, np.nan)
    pref = np.full(n_units, np.nan)
    pval = np.full(n_units, np.nan)
    for u in range(n_units):
        at_pref_sf = s == sfs[pref_sf_idx[u]]
        groups = [R[at_pref_sf & (o == v), u] for v in oris]
        means = np.array([g.mean() for g in groups])
        tc[:, u] = means
        tc_sem[:, u] = [g.std(ddof=1) / np.sqrt(len(g)) for g in groups]
        pval[u] = stats.f_oneway(*groups).pvalue
        r = np.clip(means, 0, None)
        if r.sum() > 0:
            theta = np.deg2rad(oris)
            gosi[u] = np.abs((r * np.exp(2j * theta)).sum()) / r.sum()
            pref[u] = np.rad2deg(np.angle((r * np.exp(2j * theta)).sum()) / 2) % 180

    return dict(oris=oris, sfs=sfs, tc=tc, tc_sem=tc_sem, gOSI=gosi,
                pref_ori=pref, anova_p=pval, R=R, ori=o, sf=s,
                pref_sf=sfs[pref_sf_idx])


dg = drifting_analysis(tsg, dg_table)
sg = static_analysis(tsg, sg_table)
M = dg["metrics"]
dirs = dg["dirs"]
tuned = (M["anova_p"] < 0.01) & (M["perm_p"] < 0.05)

for r in REGIONS:
    k = region == r
    print(f"{r:6s} n={k.sum():3d}  tuned={tuned[k].mean() * 100:5.1f}%  "
          f"median gOSI={np.nanmedian(M['gOSI'][k]):.3f}  "
          f"median DSI={np.nanmedian(M['DSI'][k]):.3f}")

# %% [markdown]
# ## 6. Figure 2: a single orientation-selective V1 unit
#
# The clearest single-cell evidence. Trials are grouped by direction of motion at the
# unit's preferred temporal frequency. This unit fires strongly at 45 and 225 degrees,
# which are the two directions of motion of the *same* grating orientation, and is nearly
# silent at 135 and 315 degrees, the orthogonal orientation. That two-lobed pattern is
# orientation selectivity, not direction selectivity.

# %%
candidates = np.where((region == "VISp") & tuned & (M["evoked"] > 2))[0]
best = candidates[np.argsort(-M["gOSI"][candidates])[0]]
best_uid = unit_ids[best]
best_tf = M["pref_tf"][best]

fig, axes = plt.subplots(2, 8, figsize=(13.5, 4.6), sharex=True,
                         gridspec_kw={"height_ratios": [2, 1], "hspace": 0.25,
                                      "wspace": 0.12})
window = (-0.3, 2.3)
edges = np.arange(window[0], window[1] + 1e-9, 0.05)
for i, d in enumerate(dirs):
    trials = np.where((dg_ori == d) & (dg_tf == best_tf))[0]
    aligned = nap.compute_perievent(tsg[best_uid],
                                    nap.Ts(t=dg_table["start_time"][trials]),
                                    window=window)
    ax_raster, ax_psth = axes[0, i], axes[1, i]
    for j in range(len(aligned)):
        tt = aligned[j].t
        ax_raster.plot(tt, np.full_like(tt, j), "|", ms=3.5, color="k", mew=0.7)
    ax_raster.axvspan(0, 2, color="#ffd27f", alpha=0.35, lw=0, zorder=0)
    ax_raster.set_title(f"{int(d)}$\\degree$", pad=4)
    ax_raster.set_xlim(*window)
    ax_raster.set_ylim(-0.5, len(aligned) - 0.5)
    if i:
        ax_raster.set_yticks([])
    else:
        ax_raster.set_ylabel("trial")

    all_spikes = np.concatenate([aligned[j].t for j in range(len(aligned))])
    hist, _ = np.histogram(all_spikes, bins=edges)
    ax_psth.bar(edges[:-1], hist / (len(aligned) * 0.05), width=0.05, align="edge",
                color=REG_COLORS["VISp"])
    ax_psth.axvspan(0, 2, color="#ffd27f", alpha=0.35, lw=0, zorder=0)
    ax_psth.set_xlim(*window)
    ax_psth.set_xlabel("t (s)")
    if i:
        ax_psth.set_yticks([])
    else:
        ax_psth.set_ylabel("rate (Hz)")
ymax = max(a.get_ylim()[1] for a in axes[1])
for a in axes[1]:
    a.set_ylim(0, ymax)
fig.suptitle(f"VISp unit {best_uid}: direction-dependent responses "
             f"(TF = {best_tf:g} Hz, gOSI = {M['gOSI'][best]:.2f}, "
             f"DSI = {M['DSI'][best]:.2f})", y=1.0)
fig.savefig("fig02_example_raster_psth.png")
plt.close(fig)
print("example unit:", best_uid)

# %% [markdown]
# ## 7. Figure 3: polar tuning curves, V1 against thalamus
#
# The eight most orientation-selective VISp units are bilobed, with the two lobes 180
# degrees apart. The four highest-firing LGd units are close to circular: they respond to
# a drifting grating regardless of its orientation. The dotted circle is the blank-sweep
# baseline rate.

# %%
def polar_panel(ax, u, color, title):
    m, e = dg["tc"][:, u], dg["tc_sem"][:, u]
    theta = np.deg2rad(np.append(dirs, dirs[0]))
    r, err = np.append(m, m[0]), np.append(e, e[0])
    ax.plot(theta, r, "-o", color=color, ms=3, lw=1.4)
    ax.fill_between(theta, r - err, r + err, color=color, alpha=0.25)
    ax.plot(np.linspace(0, 2 * np.pi, 100), np.full(100, M["baseline"][u]),
            ":", color="0.4", lw=1)
    ax.set_title(title, pad=24, fontsize=8)
    ax.set_yticklabels([])
    ax.set_xticks(np.deg2rad(dirs))
    ax.set_xticklabels([f"{int(x)}" for x in dirs], fontsize=6.5)
    ax.grid(alpha=0.35)


visp_best = np.where((region == "VISp") & tuned)[0]
visp_best = visp_best[np.argsort(-M["gOSI"][visp_best])][:8]
lgd_best = np.where(region == "LGd")[0]
lgd_best = lgd_best[np.argsort(-M["evoked"][lgd_best])][:4]

fig, axes = plt.subplots(3, 4, figsize=(11, 9.4), subplot_kw={"projection": "polar"})
for a, u in zip(axes.ravel()[:8], visp_best):
    polar_panel(a, u, REG_COLORS["VISp"],
                f"VISp {unit_ids[u]}\ngOSI={M['gOSI'][u]:.2f} DSI={M['DSI'][u]:.2f}")
for a, u in zip(axes.ravel()[8:], lgd_best):
    polar_panel(a, u, REG_COLORS["LGd"],
                f"LGd {unit_ids[u]}\ngOSI={M['gOSI'][u]:.2f} DSI={M['DSI'][u]:.2f}")
fig.suptitle("Direction tuning curves (mean +/- SEM firing rate at preferred TF; "
             "dotted = blank-sweep baseline)", y=0.985)
fig.subplots_adjust(hspace=0.62, wspace=0.35, top=0.89, bottom=0.04)
fig.savefig("fig03_polar_tuning.png")
plt.close(fig)

# %% [markdown]
# ## 8. Figure 4: population selectivity in the prototype session

# %%
fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.4))

ax = axes[0, 0]
by_region = [M["gOSI"][(region == r) & np.isfinite(M["gOSI"])] for r in REGIONS]
bp = ax.boxplot(by_region, tick_labels=REGIONS, widths=0.6, patch_artist=True,
                showfliers=False)
for patch, r in zip(bp["boxes"], REGIONS):
    patch.set_facecolor(REG_COLORS[r])
    patch.set_alpha(0.55)
for med in bp["medians"]:
    med.set_color("k")
for i, vals in enumerate(by_region):
    ax.plot(np.random.normal(i + 1, 0.07, len(vals)), vals, ".", ms=2.5,
            color="0.25", alpha=0.6)
ax.set_ylabel("global OSI")
ax.set_title("Orientation selectivity by area")
_, p = stats.mannwhitneyu(by_region[0], by_region[-1])
ax.text(0.02, 0.97, f"VISp vs LGd: Mann-Whitney p = {p:.1e}", transform=ax.transAxes,
        va="top", fontsize=8)

ax = axes[0, 1]
frac = [tuned[region == r].mean() * 100 for r in REGIONS]
counts = [(region == r).sum() for r in REGIONS]
ax.bar(REGIONS, frac, color=[REG_COLORS[r] for r in REGIONS], alpha=0.8)
for i, (f_, n_) in enumerate(zip(frac, counts)):
    ax.text(i, f_ + 1.5, f"{f_:.0f}%\n(n={n_})", ha="center", fontsize=7.5)
ax.set_ylabel("% units significantly tuned")
ax.set_ylim(0, max(frac) * 1.35)
ax.set_title("ANOVA p<0.01 across directions\nand permutation p<0.05 on gOSI", fontsize=9)

ax = axes[1, 0]
for r in ["VISp", "LGd"]:
    k = (region == r) & np.isfinite(M["gOSI"])
    x = np.sort(M["gOSI"][k])
    ax.plot(x, np.arange(1, len(x) + 1) / len(x), lw=2, color=REG_COLORS[r],
            label=f"{r} (n={k.sum()})")
ax.set_xlabel("global OSI")
ax.set_ylabel("cumulative fraction of units")
ax.legend(frameon=False)
ax.set_title("Cortex against its thalamic input")

ax = axes[1, 1]
for r in ["VISp", "LGd"]:
    k = region == r
    ax.plot(M["gOSI"][k], M["gDSI"][k], "o", ms=4, alpha=0.7, color=REG_COLORS[r],
            label=r, mec="none")
lim = np.nanmax([M["gOSI"], M["gDSI"]]) * 1.05
ax.plot([0, lim], [0, lim], "k--", lw=0.8)
ax.set_xlabel("global OSI")
ax.set_ylabel("global DSI")
ax.legend(frameon=False)
ax.set_title("Orientation against direction selectivity")
fig.tight_layout()
fig.savefig("fig04_population_selectivity.png")
plt.close(fig)

# %% [markdown]
# ## 9. Figure 5: static gratings, an independent measurement
#
# Drifting and static gratings are different stimulus sets presented in different blocks,
# so they give an internal replication. Static gratings sample orientation directly at
# 30 degree steps with roughly 50 repeats per orientation.

# %%
sg_tuned = sg["anova_p"] < 0.01
fig = plt.figure(figsize=(11.5, 7.2))
gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.32)

examples = np.where((region == "VISp") & sg_tuned)[0]
examples = examples[np.argsort(-sg["gOSI"][examples])][:3]
for i, u in enumerate(examples):
    ax = fig.add_subplot(gs[0, i])
    ax.errorbar(sg["oris"], sg["tc"][:, u], yerr=sg["tc_sem"][:, u], marker="o",
                color=REG_COLORS["VISp"], capsize=2, lw=1.5)
    ax.set_xticks(sg["oris"])
    ax.set_xlabel("orientation (deg)")
    if i == 0:
        ax.set_ylabel("firing rate (Hz)")
    ax.set_title(f"VISp {unit_ids[u]}\nstatic gratings, SF={sg['pref_sf'][u]:g} cpd, "
                 f"gOSI={sg['gOSI'][u]:.2f}", fontsize=8)

ax = fig.add_subplot(gs[1, 0])
for r in ["VISp", "LGd"]:
    k = (region == r) & np.isfinite(sg["gOSI"])
    x = np.sort(sg["gOSI"][k])
    ax.plot(x, np.arange(1, len(x) + 1) / len(x), lw=2, color=REG_COLORS[r], label=r)
ax.set_xlabel("global OSI (static gratings)")
ax.set_ylabel("cumulative fraction")
ax.legend(frameon=False)
ax.set_title("Static gratings replicate the\ncortex/thalamus difference", fontsize=9)

ax = fig.add_subplot(gs[1, 1])
k = ((region == "VISp") & tuned & sg_tuned
     & np.isfinite(sg["pref_ori"]) & np.isfinite(M["pref_ori"]))
a, b = M["pref_ori"][k], sg["pref_ori"][k]
ax.plot(a, b, "o", color=REG_COLORS["VISp"], ms=5, mec="none", alpha=0.8)
ax.plot([0, 180], [0, 180], "k--", lw=0.8)
delta = ((b - a + 90) % 180) - 90
ax.set_xlabel("preferred orientation, drifting (deg)")
ax.set_ylabel("preferred orientation, static (deg)")
ax.set_title(f"Cross-stimulus consistency (n = {k.sum()})", fontsize=9)

ax = fig.add_subplot(gs[1, 2])
ax.hist(delta, bins=np.arange(-90, 91, 15), color=REG_COLORS["VISp"], alpha=0.85)
ax.set_xlabel("preferred orientation difference (deg)\nstatic minus drifting")
ax.set_ylabel("units")
ax.set_title(f"median |delta| = {np.median(np.abs(delta)):.0f} deg", fontsize=9)
fig.suptitle("Orientation preference from static gratings agrees with drifting gratings",
             y=0.98)
fig.savefig("fig05_static_gratings.png")
plt.close(fig)

# %% [markdown]
# ## 10. Figure 6: Poisson GLMs with NeMoS
#
# A tuning curve built from binned means makes no commitment about the shape of the
# response. A GLM does, and it lets us ask a sharper question: is the response better
# described as a function of direction (period 360 degrees) or of orientation
# (period 180 degrees)?
#
# We fit two Poisson GLMs per unit on trial spike counts at the preferred temporal
# frequency, one with a cyclic B-spline basis over direction and one with a cyclic
# B-spline basis over `direction mod 180`. The orientation model has fewer parameters and
# cannot express any preference between opposite directions of motion. If the two models
# achieve the same cross-validated pseudo-R squared, then all of the reliable structure in
# the response is orientation structure.

# %%
b_dir = nmo.basis.CyclicBSplineEval(n_basis_funcs=8, order=4, bounds=(0.0, 360.0))
b_ori = nmo.basis.CyclicBSplineEval(n_basis_funcs=6, order=4, bounds=(0.0, 180.0))
fine_dirs = np.linspace(0, 359.9, 200)
TRIAL_DUR = 2.0


def cv_pseudo_r2(X, y, n_splits=5, seed=0):
    """Cross-validated McFadden-style pseudo-R^2 against an intercept-only model."""
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(len(y)), n_splits)
    ll_model = ll_null = 0.0
    for i in range(n_splits):
        test = folds[i]
        train = np.concatenate([folds[j] for j in range(n_splits) if j != i])
        glm = nmo.glm.GLM(regularizer="Ridge", regularizer_strength=1e-3,
                          observation_model=nmo.observation_models.PoissonObservations())
        glm.fit(X[train], y[train])
        mu = np.clip(np.asarray(glm.predict(X[test])), 1e-9, None)
        mu0 = np.clip(y[train].mean(), 1e-9, None)
        ll_model += np.sum(y[test] * np.log(mu) - mu)
        ll_null += np.sum(y[test] * np.log(mu0) - mu0)
    ll_sat = np.sum(np.where(y > 0, y * np.log(np.clip(y, 1e-9, None)) - y, 0.0))
    return 1 - (ll_sat - ll_model) / (ll_sat - ll_null)


glm_units = np.where(np.isin(region, ["VISp", "LGd"]))[0]
pr2_dir = np.full(len(unit_ids), np.nan)
pr2_ori = np.full(len(unit_ids), np.nan)
glm_curves = {}
show_glm = set(visp_best[:4].tolist())

for u in tqdm(glm_units, desc="GLM"):
    at_pref_tf = dg["tf"] == M["pref_tf"][u]
    x = dg["ori"][at_pref_tf]
    y = np.round(dg["R"][at_pref_tf, u] * TRIAL_DUR)
    if y.sum() < 20:
        continue
    Xd, Xo = b_dir.compute_features(x), b_ori.compute_features(x % 180)
    pr2_dir[u] = cv_pseudo_r2(Xd, y)
    pr2_ori[u] = cv_pseudo_r2(Xo, y)
    if u in show_glm:
        glm = nmo.glm.GLM(regularizer="Ridge", regularizer_strength=1e-3,
                          observation_model=nmo.observation_models.PoissonObservations())
        glm.fit(Xd, y)
        glm_curves[u] = np.asarray(
            glm.predict(b_dir.compute_features(fine_dirs))) / TRIAL_DUR

fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9))
for ax, u in zip(axes[:2], sorted(glm_curves, key=lambda u: -M["gOSI"][u])[:2]):
    ax.errorbar(dirs, dg["tc"][:, u], yerr=dg["tc_sem"][:, u], fmt="o", color="0.25",
                capsize=2, ms=4, label="measured")
    ax.plot(fine_dirs, glm_curves[u], color=REG_COLORS["VISp"], lw=2, label="Poisson GLM")
    ax.set_xticks(dirs)
    ax.set_xlabel("direction (deg)")
    ax.set_ylabel("firing rate (Hz)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(f"VISp {unit_ids[u]}  gOSI={M['gOSI'][u]:.2f}", fontsize=9)

ax = axes[2]
for r in ["VISp", "LGd"]:
    k = (region == r) & np.isfinite(pr2_dir)
    ax.plot(pr2_ori[k], pr2_dir[k], "o", ms=4, alpha=0.7, color=REG_COLORS[r],
            mec="none", label=r)
lo, hi = np.nanmin([pr2_ori, pr2_dir]), np.nanmax([pr2_ori, pr2_dir])
ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
ax.set_xlabel("cross-val. pseudo-$R^2$, orientation model (180$\\degree$ period)")
ax.set_ylabel("pseudo-$R^2$, direction model (360$\\degree$)")
ax.legend(frameon=False)
good = np.isfinite(pr2_dir) & (pr2_dir > 0.02)
ax.set_title("An orientation-only model captures\nmost of the explainable variance",
             fontsize=9)
fig.suptitle("NeMoS Poisson GLM with cyclic B-spline bases over grating direction", y=1.03)
fig.tight_layout()
fig.savefig("fig06_glm.png")
plt.close(fig)
print(f"median pseudo-R^2: orientation model {np.nanmedian(pr2_ori[good]):.3f}, "
      f"direction model {np.nanmedian(pr2_dir[good]):.3f} "
      f"(units with pseudo-R^2 > 0.02, n={good.sum()})")

# %% [markdown]
# ## 11. Figure 7: decoding orientation from population activity
#
# Single-trial spike-count vectors are fed to a multinomial logistic regression with
# 5-fold cross-validation. The static gratings are the harder and more informative test:
# 6 orientations, 0.25 s per trial, chance 16.7 percent.
#
# Both areas support well above-chance decoding, and V1 is consistently better, but the
# gap at the population level is much smaller than the gap in single-unit gOSI. Weak
# per-neuron orientation biases in the thalamus add up across tens of neurons. The
# cortical signature is therefore in the tuning of individual cells, not in whether
# orientation can be read out at all.

# %%
def decode(X, y, seed=0):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.05))
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    y_pred = cross_val_predict(clf, X, y, cv=cv)
    return (y_pred == y).mean(), confusion_matrix(y, y_pred, normalize="true")


stim_sets = {"static gratings (0.25 s)": (sg["R"], sg["ori"]),
             "drifting gratings (2 s)": (dg["R"], dg["ori"] % 180)}
n_match = min((region == "VISp").sum(), (region == "LGd").sum())
rng = np.random.default_rng(1)

fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.1))
R_static, y_static = stim_sets["static gratings (0.25 s)"]
static_oris = np.unique(y_static)
for ax, r in zip(axes[:2], ["VISp", "LGd"]):
    pop = rng.choice(np.where(region == r)[0], n_match, replace=False)
    acc, cm = decode(R_static[:, pop], y_static)
    im = ax.imshow(cm, vmin=0, vmax=max(0.5, cm.max()), cmap="magma")
    ax.set_xticks(range(len(static_oris)), [f"{int(x)}" for x in static_oris])
    ax.set_yticks(range(len(static_oris)), [f"{int(x)}" for x in static_oris])
    ax.set_xlabel("decoded orientation (deg)")
    ax.set_ylabel("true orientation (deg)")
    ax.set_title(f"{r} — static gratings, n={n_match} units\n"
                 f"accuracy {acc * 100:.1f}% (chance {100 / len(static_oris):.1f}%)",
                 fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, label="P(decoded | true)")

ax = axes[2]
sizes = [5, 10, 20, 40, n_match]
line_style = {"static gratings (0.25 s)": "-", "drifting gratings (2 s)": "--"}
for name, (R_, y_) in stim_sets.items():
    for r in ["VISp", "LGd"]:
        mean_acc, sd_acc = [], []
        for n in tqdm(sizes, desc=f"decode {name[:6]} {r}", leave=False):
            reps = [decode(R_[:, rng.choice(np.where(region == r)[0], n, replace=False)],
                           y_, seed=i)[0] for i in range(3)]
            mean_acc.append(np.mean(reps) * 100)
            sd_acc.append(np.std(reps) * 100)
        ax.errorbar(sizes, mean_acc, yerr=sd_acc, marker="o", ms=4, capsize=3,
                    ls=line_style[name], color=REG_COLORS[r], label=f"{r}, {name}")
    ax.axhline(100 / len(np.unique(y_)), color="0.5", ls=":", lw=1)
ax.text(n_match, 100 / 6 + 1.2, "chance (6 orientations)", ha="right", fontsize=7, color="0.45")
ax.text(n_match, 25 + 1.2, "chance (4 orientations)", ha="right", fontsize=7, color="0.45")
ax.set_xlabel("number of units in decoding population")
ax.set_ylabel("cross-validated accuracy (%)")
ax.legend(frameon=False, fontsize=7.5, loc="center left", bbox_to_anchor=(1.02, 0.5))
ax.set_title("Population orientation information is present in both\n"
             "areas; the V1 advantage is modest", fontsize=9)
fig.tight_layout()
fig.savefig("fig07_decoding.png")
plt.close(fig)

# %% [markdown]
# ## 12. Scaling to 8 sessions
#
# The prototype session gives 62 VISp units, which is enough to see the effect but not
# enough to be confident about the size of it. Running the same pipeline on all 8 sessions
# pools several hundred units per area, one session per mouse.

# %%
pooled = []
for _, session_id in tqdm(SESSIONS, desc="pooling sessions"):
    d_i, tsg_i = load_session(session_id)
    dg_i = drifting_analysis(tsg_i, d_i["stim"]["drifting_gratings_presentations"])
    sg_i = static_analysis(tsg_i, d_i["stim"]["static_gratings_presentations"])
    pooled.append(dict(session=session_id, region=np.array(d_i["meta"]["region"]),
                       unit_id=d_i["meta"]["unit_id"],
                       **dg_i["metrics"],
                       sg_gOSI=sg_i["gOSI"], sg_pref_ori=sg_i["pref_ori"],
                       sg_p=sg_i["anova_p"], tc=dg_i["tc"], dirs=dg_i["dirs"],
                       tc_a=dg_i["tc_a"], tc_b=dg_i["tc_b"]))

cat = lambda key: np.concatenate([r[key] for r in pooled])
p_region = cat("region")
p_gOSI, p_gDSI, p_DSI = cat("gOSI"), cat("gDSI"), cat("DSI")
p_pref_ori = cat("pref_ori")
p_sg_gOSI, p_sg_pref, p_sg_p = cat("sg_gOSI"), cat("sg_pref_ori"), cat("sg_p")
p_tuned = (cat("anova_p") < 0.01) & (cat("perm_p") < 0.05)

summary_lines = [f"sessions: {len(pooled)}"]
for r in REGIONS:
    k = p_region == r
    summary_lines.append(
        f"{r:6s} n={k.sum():4d} tuned={p_tuned[k].mean() * 100:5.1f}% "
        f"median gOSI={np.nanmedian(p_gOSI[k]):.3f} "
        f"median DSI={np.nanmedian(p_DSI[k]):.3f} "
        f"median gOSI(static)={np.nanmedian(p_sg_gOSI[k]):.3f}")
U, p_visp_lgd = stats.mannwhitneyu(p_gOSI[p_region == "VISp"], p_gOSI[p_region == "LGd"])
summary_lines.append(f"VISp vs LGd gOSI Mann-Whitney U={U:.0f} p={p_visp_lgd:.3e}")
open("pooled_stats.txt", "w").write("\n".join(summary_lines) + "\n")
print("\n".join(summary_lines))

# %% [markdown]
# ## 13. Figure 8: pooled population selectivity

# %%
fig, axes = plt.subplots(2, 2, figsize=(11, 7.6))

ax = axes[0, 0]
by_region = [p_gOSI[(p_region == r) & np.isfinite(p_gOSI)] for r in REGIONS]
bp = ax.boxplot(by_region, tick_labels=REGIONS, widths=0.62, patch_artist=True,
                showfliers=False)
for patch, r in zip(bp["boxes"], REGIONS):
    patch.set_facecolor(REG_COLORS[r])
    patch.set_alpha(0.55)
for med in bp["medians"]:
    med.set_color("k")
for i, r in enumerate(REGIONS):
    medians = [np.nanmedian(s["gOSI"][s["region"] == r]) for s in pooled
               if (s["region"] == r).sum() > 5]
    ax.plot(np.full(len(medians), i + 1), medians, "o", ms=4, mfc="w", mec="k",
            mew=0.9, zorder=5)
ax.set_ylabel("global OSI")
ax.set_title(f"Orientation selectivity across areas (N = {len(pooled)} sessions)\n"
             "white circles = per-session medians", fontsize=9)
ax.text(0.02, 0.97,
        f"VISp (n={len(by_region[0])}) vs LGd (n={len(by_region[-1])}): "
        f"p = {p_visp_lgd:.1e}", transform=ax.transAxes, va="top", fontsize=8)

ax = axes[0, 1]
frac = [p_tuned[p_region == r].mean() * 100 for r in REGIONS]
counts = [(p_region == r).sum() for r in REGIONS]
ax.bar(REGIONS, frac, color=[REG_COLORS[r] for r in REGIONS], alpha=0.85)
for i, (f_, n_) in enumerate(zip(frac, counts)):
    ax.text(i, f_ + 1.2, f"{f_:.0f}%\n(n={n_})", ha="center", fontsize=7.5)
ax.set_ylabel("% units significantly orientation tuned")
ax.set_ylim(0, max(frac) * 1.35)
ax.set_title("Direction ANOVA p<0.01 AND gOSI permutation p<0.05", fontsize=9)

ax = axes[1, 0]
for r in ["VISp", "LGd"]:
    k = (p_region == r) & np.isfinite(p_gOSI)
    x = np.sort(p_gOSI[k])
    ax.plot(x, np.arange(1, len(x) + 1) / len(x), lw=2.2, color=REG_COLORS[r],
            label=f"{r} (n={k.sum()}, median {np.median(x):.2f})")
ax.set_xlabel("global OSI")
ax.set_ylabel("cumulative fraction of units")
ax.legend(frameon=False, fontsize=8)
ax.set_title("Cortex against thalamus", fontsize=9)

ax = axes[1, 1]
k = np.isfinite(p_gOSI) & np.isfinite(p_sg_gOSI)
for r in ["VISp", "LGd"]:
    kk = k & (p_region == r)
    ax.plot(p_gOSI[kk], p_sg_gOSI[kk], "o", ms=3.5, alpha=0.55, mec="none",
            color=REG_COLORS[r], label=r)
rho = stats.spearmanr(p_gOSI[k], p_sg_gOSI[k])
ax.plot([0, 1], [0, 1], "k--", lw=0.8)
ax.set_xlabel("gOSI, drifting gratings")
ax.set_ylabel("gOSI, static gratings")
ax.legend(frameon=False, fontsize=8)
ax.set_title(f"Selectivity is consistent across stimulus types\n"
             f"Spearman r = {rho.statistic:.2f}", fontsize=9)
fig.tight_layout()
fig.savefig("fig08_pooled_population.png")
plt.close(fig)

# %% [markdown]
# ## 14. Figure 9: preferred orientation and direction selectivity
#
# Preferred orientations in VISp are not uniformly distributed: there is an excess near
# the cardinal axes (0 and 90 degrees), the well-documented cardinal bias of mouse visual
# cortex. Preferences measured with the two stimulus sets agree. Direction selectivity, by
# contrast, is weak in both areas, which is the point of the whole demonstration: what
# these neurons encode is the orientation of the grating, not which way it moves.

# %%
fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0))

ax = axes[0]
k = (p_region == "VISp") & p_tuned & np.isfinite(p_pref_ori)
bins = np.arange(0, 181, 15)
ax.hist(p_pref_ori[k], bins=bins, color=REG_COLORS["VISp"], alpha=0.85,
        edgecolor="w", lw=0.6)
ax.axhline(k.sum() / (len(bins) - 1), color="k", ls="--", lw=1, label="uniform")
doubled = np.deg2rad(2 * p_pref_ori[k])
z = np.abs(np.mean(np.exp(1j * doubled)))
rayleigh_p = np.exp(-k.sum() * z ** 2)
ax.set_xticks(np.arange(0, 181, 30))
ax.set_xlabel("preferred orientation (deg)")
ax.set_ylabel("VISp units")
ax.legend(frameon=False, fontsize=8)
ax.set_title(f"Cardinal bias in VISp (n={k.sum()})\nRayleigh p = {rayleigh_p:.1e}",
             fontsize=9)

ax = axes[1]
kk = ((p_region == "VISp") & p_tuned & (p_sg_p < 0.01)
      & np.isfinite(p_sg_pref) & np.isfinite(p_pref_ori))
delta = ((p_sg_pref[kk] - p_pref_ori[kk] + 90) % 180) - 90
ax.hist(delta, bins=np.arange(-90, 91, 15), color=REG_COLORS["VISp"], alpha=0.85,
        edgecolor="w", lw=0.6)
z2 = np.abs(np.mean(np.exp(1j * np.deg2rad(2 * delta))))
delta_p = np.exp(-len(delta) * z2 ** 2)
ax.set_xlabel("preferred orientation, static minus drifting (deg)")
ax.set_ylabel("VISp units")
ax.set_title(f"Cross-stimulus agreement (n={kk.sum()})\n"
             f"median |delta| = {np.median(np.abs(delta)):.0f} deg, "
             f"Rayleigh p = {delta_p:.1e}", fontsize=9)

ax = axes[2]
for r in ["VISp", "LGd"]:
    k2 = (p_region == r) & np.isfinite(p_DSI)
    ax.hist(p_DSI[k2], bins=np.linspace(0, 1, 21), histtype="step", lw=2, density=True,
            color=REG_COLORS[r], label=f"{r} (median {np.nanmedian(p_DSI[k2]):.2f})")
ax.set_xlabel("direction selectivity index")
ax.set_ylabel("density")
ax.legend(frameon=False, fontsize=8)
ax.set_title("Direction selectivity is weak in both areas", fontsize=9)
fig.tight_layout()
fig.savefig("fig09_preferred_orientation.png")
plt.close(fig)

# %% [markdown]
# ## 15. Figure 10: tuning curves aligned to each unit's preferred direction
#
# The most direct summary of the phenomenon, but it needs one methodological precaution.
# Rotating every unit's tuning curve so that its own peak sits at 0 and dividing by that
# peak puts a 1.0 at the centre *by construction*, even for a unit with no tuning at all,
# because the largest of eight noisy means is always above the rest. Aligning and
# normalizing on the same trials would therefore invent a peak for LGd and undercut the
# comparison the figure is meant to make.
#
# So the alignment is cross-validated. Each unit's preferred direction is taken from one
# half of its trials, and the curve that gets plotted is measured on the *other* half,
# expressed relative to that unit's own mean rate. A unit with no orientation tuning now
# gives a flat line at 1.0, because the direction picked on the first half carries no
# information about the second half. Any departure from flat is real tuning.
#
# If neurons were tuned to direction of motion, the population average would fall away
# from 0 and reach its minimum at 180 degrees. Instead the VISp average dips to a minimum
# near 90 degrees, the orthogonal orientation, and rises again at 180 degrees, the same
# orientation drifting the other way.

# %%
rel_dirs = np.arange(-180, 180, 45)
tc_a_all = np.concatenate([s["tc_a"].T for s in pooled])
tc_b_all = np.concatenate([s["tc_b"].T for s in pooled])


def cv_aligned(idx):
    """Align on half A, measure on half B, normalize by each unit's own mean rate."""
    a, b = tc_a_all[idx], tc_b_all[idx]
    keep = (b.mean(1) > 0) & np.isfinite(a).all(1) & np.isfinite(b).all(1)
    a, b = a[keep], b[keep]
    peak = np.argmax(a, axis=1)                        # preferred direction from half A
    rolled = np.array([np.roll(t, 4 - p) for t, p in zip(b, peak)])   # curve from half B
    return rolled / rolled.mean(1, keepdims=True), keep


fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2),
                         gridspec_kw={"width_ratios": [1, 1, 1.15]})

for ax, r in zip(axes[:2], ["VISp", "LGd"]):
    idx = np.where(p_region == r)[0]
    norm, keep = cv_aligned(idx)
    order = np.argsort(-p_gOSI[idx][keep])
    im = ax.imshow(norm[order], aspect="auto", cmap="viridis", vmin=0, vmax=2,
                   extent=[-202.5, 157.5, len(norm), 0])
    ax.set_xticks([-180, -90, 0, 90])
    ax.set_xlim(-202.5, 157.5)
    ax.set_xlabel("direction relative to preferred (deg)")
    ax.set_ylabel("unit (sorted by gOSI)")
    ax.set_title(f"{r} (n={len(norm)}), cross-validated", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, label="rate / unit mean rate")

ax = axes[2]
for r in ["VISp", "LGd"]:
    norm, _ = cv_aligned(np.where(p_region == r)[0])
    x = np.append(rel_dirs, 180)                      # wrap for a symmetric plot
    m = np.append(norm.mean(0), norm.mean(0)[0])
    s = np.append(norm.std(0), norm.std(0)[0]) / np.sqrt(len(norm))
    ax.plot(x, m, "-o", color=REG_COLORS[r], label=f"{r} (n={len(norm)})", ms=4)
    ax.fill_between(x, m - s, m + s, color=REG_COLORS[r], alpha=0.25)
ax.axhline(1.0, color="0.5", ls=":", lw=1)
ax.text(-175, 1.0, "no tuning", fontsize=7, color="0.45", va="bottom")
ax.set_xticks(np.arange(-180, 181, 90))
ax.set_xlabel("direction relative to preferred (deg)")
ax.set_ylabel("rate / unit mean rate")
ax.legend(frameon=False, fontsize=8)
ax.set_title("Population average: V1 dips at 90$\\degree$ and\nrecovers at 180$\\degree$",
             fontsize=9)
fig.tight_layout()
fig.savefig("fig10_aligned_tuning.png")
plt.close(fig)

visp_cv, _ = cv_aligned(np.where(p_region == "VISp")[0])
lgd_cv, _ = cv_aligned(np.where(p_region == "LGd")[0])
for name, arr in [("VISp", visp_cv), ("LGd", lgd_cv)]:
    prof = arr.mean(0)
    print(f"{name} cross-validated aligned profile: peak={prof[4]:.2f}, "
          f"orthogonal(90)={prof[6]:.2f}, opposite(180)={prof[0]:.2f} "
          f"(1.0 = no tuning)")

# %% [markdown]
# ## 16. Summary
#
# Working from 8 Neuropixels sessions in DANDI:000021, streamed rather than downloaded:
#
# * Individual VISp units respond selectively to grating orientation. The clearest
#   examples fire at 5 to 10 Hz for one orientation and at the blank-sweep baseline for
#   the orthogonal one, with matched responses to the two opposite directions of motion of
#   the preferred orientation (Figures 2 and 3).
# * Roughly two thirds of VISp units pass a conservative joint test for orientation tuning
#   (direction ANOVA p < 0.01 and a permutation test on gOSI p < 0.05), with a median gOSI
#   near 0.22. In LGd, the thalamic input recorded simultaneously in the same animals,
#   the corresponding numbers are about 46 percent and 0.11, and the difference in the gOSI
#   distributions is highly significant (Figure 8).
# * The effect is not an artifact of one stimulus set. Static gratings, a separate block
#   with a different presentation duration and a different set of spatial frequencies,
#   give the same ranking of areas and the same preferred orientation per unit, with a
#   median discrepancy of about 13 degrees (Figures 5 and 9).
# * A Poisson GLM whose only regressor is orientation, and which is therefore blind to
#   direction of motion, predicts held-out spike counts about as well as a GLM with the
#   full 360 degree direction basis (Figure 6). Orientation, not direction, is what these
#   responses depend on.
# * Preferred orientations in VISp cluster near the cardinal axes, reproducing the known
#   cardinal bias of mouse visual cortex (Figure 9).
# * Population decoding is a less discriminating assay than single-unit tuning: orientation
#   is decodable well above chance from LGd populations too, because weak per-neuron biases
#   accumulate. V1 is consistently but only modestly better (Figure 7).
#
# ### Caveats
#
# The temporal-frequency and spatial-frequency preferences of each unit are estimated from
# the same data used for the tuning curves, which is standard practice but does introduce
# a mild optimistic bias in the absolute gOSI values. Locomotion is known to gain-modulate
# mouse visual responses and is not regressed out here. Units were pooled across cortical
# layers, and layer identity is not resolved.
