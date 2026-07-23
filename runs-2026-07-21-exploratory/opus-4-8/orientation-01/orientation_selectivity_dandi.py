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
# # Orientation selectivity in the mouse visual system
#
# **Data:** [DANDI:000021](https://dandiarchive.org/dandiset/000021) — *Allen Institute
# Visual Coding, Neuropixels (Brain Observatory 1.1 stimulus set)*.
#
# Orientation selectivity is the property, first described by Hubel and Wiesel in cat
# striate cortex, that a visual neuron responds strongly to an edge or grating at one
# angle and weakly or not at all to the orthogonal angle. It is the canonical example of
# a receptive-field computation that is built in cortex: the thalamic relay cells that
# feed primary visual cortex have centre-surround receptive fields and are largely
# indifferent to stimulus angle, whereas their cortical targets are sharply tuned.
#
# This notebook demonstrates that transformation directly from public data. Each recording
# in DANDI:000021 uses up to six Neuropixels probes to record simultaneously from the
# dorsal lateral geniculate nucleus (LGd), the lateral posterior nucleus (LP), primary
# visual cortex (VISp) and five higher visual cortical areas, while a head-fixed mouse
# views a fixed battery of visual stimuli. Two of those stimuli isolate orientation:
#
# * **Drifting gratings** — 2 s presentations, 8 drift directions (0–315° in 45° steps)
#   crossed with 5 temporal frequencies, 15 repeats each, plus blank sweeps.
# * **Static gratings** — 0.25 s presentations, 6 orientations (0–150° in 30° steps)
#   crossed with 5 spatial frequencies and 4 phases.
#
# The drifting gratings let us separate *orientation* tuning (a preference for an axis,
# i.e. a response at both θ and θ+180°) from *direction* tuning (a preference for one of
# the two motion directions along that axis). The static gratings provide an independent
# measurement with no motion at all, which we use to confirm that the preferred angle
# recovered from moving gratings is a genuine orientation preference.
#
# **The analysis proceeds in five steps:**
#
# 1. Survey all 32 sessions and select six with good simultaneous LGd / VISp / higher-area yield.
# 2. Inspect the raw data (stimulus timing, spike rasters, locomotion).
# 3. Measure per-unit direction tuning curves and selectivity indices, with a permutation test.
# 4. Cross-validate the orientation preference against static gratings, and check with a
#    Poisson GLM that tuning is not an artefact of locomotion or temporal-frequency preference.
# 5. Decode stimulus orientation from population activity in each area.

# %% [markdown]
# ## Setup
#
# Data are streamed from the DANDI S3 bucket; nothing is downloaded in full. The session
# NWB files are ~2 GB each, but the spike times we need live in a single contiguous,
# uncompressed HDF5 dataset, so we look up its byte offset and issue parallel HTTP range
# requests for only the units that pass quality control. Everything downloaded is cached
# under `~/.cache/dandi_000021`, so re-running this notebook is fast.

# %%
import os
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import h5py
import requests
import remfile
import matplotlib

matplotlib.use("Agg")  # headless: figures are written to disk, never shown

import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy import optimize, stats
from scipy.special import gammaln
from tqdm.auto import tqdm

from pynwb import NWBHDF5IO
from dandi.dandiapi import DandiAPIClient
import pynapple as nap

warnings.filterwarnings("ignore", category=UserWarning)
nap.nap_config.suppress_conversion_warnings = True

DANDISET = "000021"
CACHE_DIR = os.environ.get("OV_CACHE", os.path.expanduser("~/.cache/dandi_000021"))
REMFILE_CACHE = os.path.join(CACHE_DIR, "remfile")
FIG_DIR = os.path.abspath(".")
os.makedirs(REMFILE_CACHE, exist_ok=True)

DG_DIRECTIONS = np.array([0., 45., 90., 135., 180., 225., 270., 315.])
SG_ORIENTATIONS = np.array([0., 30., 60., 90., 120., 150.])

VISUAL_AREAS = ["LGd", "LP", "VISp", "VISl", "VISrl", "VISal", "VISpm", "VISam"]
HVA = ["VISl", "VISrl", "VISal", "VISpm", "VISam"]      # higher visual cortical areas
AREA_COLORS = {
    "LGd": "#4C72B0", "LP": "#7BA4D0", "VISp": "#C44E52", "VISl": "#DD8452",
    "VISrl": "#E0A458", "VISal": "#8C8C3F", "VISpm": "#937860", "VISam": "#B07AA1",
    "HVA": "#DD8452",
}

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 160, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.frameon": False, "savefig.bbox": "tight",
})


def group_label(area):
    """Pool the five higher visual cortical areas for population summaries."""
    return "HVA" if area in HVA else area


# %% [markdown]
# ### Streaming helpers

# %%
def session_url(asset_path):
    with DandiAPIClient() as client:
        asset = client.get_dandiset(DANDISET, "draft").get_asset_by_path(asset_path)
        return asset.get_content_url(follow_redirects=1, strip_query=True)


def open_nwb(asset_path):
    """Open a session NWB file for streaming. Returns (nwbfile, h5 handle, url)."""
    url = session_url(asset_path)
    rem = remfile.File(url, disk_cache=remfile.DiskCache(REMFILE_CACHE))
    h5 = h5py.File(rem, "r")
    io = NWBHDF5IO(file=h5, load_namespaces=True)
    return io.read(), h5, url


def unit_table(nwbfile):
    """Unit quality metrics and anatomical location, without touching spike times."""
    electrodes = nwbfile.electrodes
    loc_by_id = dict(zip(electrodes.id[:], np.asarray(electrodes["location"].data[:]).astype(str)))
    ut = nwbfile.units
    cols = ["quality", "snr", "isi_violations", "amplitude_cutoff", "presence_ratio",
            "firing_rate", "peak_channel_id", "waveform_duration"]
    df = pd.DataFrame({c: np.asarray(ut[c].data[:]) for c in cols}, index=np.asarray(ut.id[:]))
    df["quality"] = df["quality"].astype(str)
    df["area"] = [loc_by_id.get(p, "") for p in df["peak_channel_id"]]
    df["row"] = np.arange(len(df))
    return df


def select_units(udf, areas=VISUAL_AREAS):
    """Allen Institute default quality criteria, restricted to visual areas."""
    keep = ((udf["quality"] == "good") & udf["area"].isin(areas)
            & (udf["amplitude_cutoff"] < 0.1) & (udf["isi_violations"] < 0.5)
            & (udf["presence_ratio"] > 0.9) & (udf["firing_rate"] > 0.1))
    return udf[keep].copy()


def _fetch_ranges(url, byte_ranges, n_workers=16):
    session = requests.Session()

    def get(rng):
        headers = {"Range": f"bytes={rng[0]}-{rng[1]}"}
        for _ in range(4):
            r = session.get(url, headers=headers, timeout=180)
            if r.status_code in (200, 206):
                return r.content
        r.raise_for_status()

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        return list(ex.map(get, byte_ranges))


def read_spike_times(h5, url, rows, max_gap_bytes=2_000_000):
    """Read `units/spike_times` for the given unit rows via parallel HTTP range requests.

    Neighbouring rows are merged into one request when the gap between them is small,
    which cuts the number of round trips substantially.
    """
    ds = h5["units/spike_times"]
    assert ds.chunks is None and ds.compression is None, "expected a contiguous dataset"
    base, itemsize = ds.id.get_offset(), ds.dtype.itemsize
    index = np.asarray(h5["units/spike_times_index"][:])
    starts, stops = np.concatenate([[0], index[:-1]]), index

    rows = np.sort(np.asarray(rows))
    seg = [(int(starts[r]), int(stops[r])) for r in rows]
    runs, cur = [], [seg[0][0], seg[0][1], [0]]
    for i in range(1, len(seg)):
        if (seg[i][0] - cur[1]) * itemsize <= max_gap_bytes:
            cur[1] = max(cur[1], seg[i][1])
            cur[2].append(i)
        else:
            runs.append(cur)
            cur = [seg[i][0], seg[i][1], [i]]
    runs.append(cur)

    blobs = _fetch_ranges(url, [(base + a * itemsize, base + b * itemsize - 1)
                                for a, b, _ in runs])
    out = {}
    for (a, _, members), blob in zip(runs, blobs):
        arr = np.frombuffer(blob, dtype=ds.dtype)
        for m in members:
            s, e = seg[m]
            out[int(rows[m])] = arr[s - a: e - a].astype(np.float64)
    return out


def load_spikes(asset_path, areas=VISUAL_AREAS, cache=True):
    """Return (TsGroup of QC-passing units with metadata, session info dict)."""
    tag = asset_path.split("_ses-")[-1].replace(".nwb", "")
    npz_path = os.path.join(CACHE_DIR, f"spikes_{tag}.npz")
    meta_path = os.path.join(CACHE_DIR, f"meta_{tag}.parquet")

    nwbfile, h5, url = open_nwb(asset_path)
    udf = unit_table(nwbfile)
    sel = select_units(udf, areas)

    if cache and os.path.exists(npz_path) and os.path.exists(meta_path):
        z = np.load(npz_path)
        sel = pd.read_parquet(meta_path)
        spike_dict = {int(k): z[f"u{k}"] for k in sel.index}
    else:
        raw = read_spike_times(h5, url, sel["row"].values)
        row_to_id = dict(zip(sel["row"].values, sel.index.values))
        spike_dict = {int(row_to_id[r]): v for r, v in raw.items()}
        if cache:
            np.savez(npz_path, **{f"u{k}": v for k, v in spike_dict.items()})
            sel.to_parquet(meta_path)

    t_end = max(v[-1] for v in spike_dict.values() if len(v))
    tsgroup = nap.TsGroup(
        {k: nap.Ts(t=np.sort(v)) for k, v in spike_dict.items()},
        time_support=nap.IntervalSet(start=0.0, end=float(t_end) + 1.0),
        metadata=sel.loc[list(spike_dict.keys()), ["area", "snr", "firing_rate"]],
    )
    info = {"nwbfile": nwbfile, "h5": h5, "url": url, "selected": sel,
            "session": tag, "asset_path": asset_path}
    return tsgroup, info


def stimulus_table(nwbfile, name):
    df = nwbfile.intervals[name].to_dataframe()
    for c in ["orientation", "temporal_frequency", "spatial_frequency", "phase", "contrast"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["duration"] = df["stop_time"] - df["start_time"]
    return df


def running_speed(nwbfile):
    ts = nwbfile.processing["running"].data_interfaces["running_speed"]
    return nap.Tsd(t=np.asarray(ts.timestamps[:]), d=np.asarray(ts.data[:]))


# %% [markdown]
# ## 1. Choosing sessions
#
# All 32 sessions in this dandiset use the same stimulus set, but probe placement varies,
# so the areas sampled differ from mouse to mouse. We want sessions that record LGd and
# VISp *simultaneously*, because that gives us the thalamus/cortex comparison within the
# same animal, on the same stimulus presentations, with the same spike-sorting pipeline.
# The survey below reads only the metadata tables (no spike times) from each file.

# %%
def survey_sessions(cache_csv=os.path.join(CACHE_DIR, "session_survey.csv")):
    if os.path.exists(cache_csv):
        return pd.read_csv(cache_csv)
    with DandiAPIClient() as client:
        assets = [a for a in client.get_dandiset(DANDISET, "draft").get_assets()
                  if "probe" not in a.path]
    rows = []
    for a in tqdm(sorted(assets, key=lambda x: x.path), desc="surveying sessions"):
        nwbfile, h5, _ = open_nwb(a.path)
        sel = select_units(unit_table(nwbfile))
        r = {"path": a.path, "n_qc_units": len(sel)}
        r.update({ar: int((sel["area"] == ar).sum()) for ar in VISUAL_AREAS})
        rows.append(r)
        h5.close()
    df = pd.DataFrame(rows)
    df["HVA"] = df[HVA].sum(axis=1)
    df.to_csv(cache_csv, index=False)
    return df


survey = survey_sessions()
survey["score"] = np.minimum(survey["LGd"], 40) + np.minimum(survey["VISp"], 60) + \
    np.minimum(survey["HVA"], 150)
print(f"{len(survey)} sessions in DANDI:{DANDISET}")
display_cols = ["path", "n_qc_units", "LGd", "LP", "VISp", "HVA"]
print(survey.sort_values("score", ascending=False)[display_cols].head(10).to_string(index=False))

# %%
SESSIONS = survey.sort_values("score", ascending=False)["path"].head(6).tolist()
print("Selected sessions:")
for s in SESSIONS:
    print("  ", s)

# %% [markdown]
# ## 2. Raw data
#
# Load the first session and look at what is actually there before computing anything.

# %%
spikes, info = load_spikes(SESSIONS[0])
dg = stimulus_table(info["nwbfile"], "drifting_gratings_presentations")
sg = stimulus_table(info["nwbfile"], "static_gratings_presentations")
running = running_speed(info["nwbfile"])

print(f"{len(spikes)} units passing quality control, "
      f"mean rate {np.mean(spikes.rates):.1f} Hz, "
      f"recording length {spikes.time_support.tot_length():.0f} s")
print()
print("units per area:")
print(info["selected"]["area"].value_counts().to_string())
print(f"\ndrifting gratings: {len(dg)} presentations, "
      f"{dg['duration'].mean():.3f} s each, "
      f"{int(dg['orientation'].isna().sum())} blank sweeps")
print("  directions:", np.sort(dg['orientation'].dropna().unique()))
print("  temporal frequencies (Hz):", np.sort(dg['temporal_frequency'].dropna().unique()))
print(f"\nstatic gratings: {len(sg)} presentations, {sg['duration'].mean():.3f} s each")
print("  orientations:", np.sort(sg['orientation'].dropna().unique()))
print("  spatial frequencies (cpd):", np.sort(sg['spatial_frequency'].dropna().unique()))

# %% [markdown]
# ### Figure 1 — raw data overview
#
# The raster is the sanity check that matters: with 2 s gratings separated by 1 s of mean
# grey, cortical units should visibly switch on and off with the stimulus, and different
# units should switch on for different drift directions.

# %%
def fig_raw_overview(spikes, info, dg, running, out_path,
                     t_window=24.0, n_per_area=45, areas_shown=("LGd", "VISp")):
    nwbfile = info["nwbfile"]
    all_areas, all_uids = np.asarray(spikes.metadata["area"]), np.asarray(spikes.index)

    rng = np.random.default_rng(1)
    rows, row_area = [], []
    for a in areas_shown:
        idx = np.where(all_areas == a)[0]
        if len(idx) > n_per_area:
            idx = np.sort(rng.choice(idx, n_per_area, replace=False))
        rows.extend(all_uids[idx])
        row_area.extend([a] * len(idx))
    rows, row_area = np.array(rows), np.array(row_area)

    t0 = float(dg["start_time"].iloc[40])
    t1 = t0 + t_window
    ep = nap.IntervalSet(start=t0, end=t1)

    fig = plt.figure(figsize=(12, 9.0))
    gs = gridspec.GridSpec(4, 1, height_ratios=[0.7, 3.6, 1.0, 0.9], hspace=0.75)

    ax = fig.add_subplot(gs[0])
    cmap = plt.get_cmap("tab10")
    for i, name in enumerate(sorted(k for k in nwbfile.intervals if k.endswith("_presentations"))):
        tbl = nwbfile.intervals[name].to_dataframe()
        ax.barh(0, tbl["stop_time"].values - tbl["start_time"].values,
                left=tbl["start_time"].values, height=0.6, color=cmap(i % 10),
                label=name.replace("_presentations", ""))
    ax.axvline(t0, color="k", lw=1.2)
    ax.annotate("window below", xy=(t0, 0.45), xytext=(t0 + 350, 0.9), fontsize=7.5,
                arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set_ylim(-0.4, 1.2)
    ax.set_yticks([])
    ax.set_xlabel("time in session (s)", labelpad=1)
    ax.set_title("A   Stimulus blocks across the session", loc="left")
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.005, 0.5), handlelength=1.1)

    ax = fig.add_subplot(gs[1])
    dgw = dg[(dg["start_time"] < t1) & (dg["stop_time"] > t0)]
    ntop = len(rows) * 1.10
    for _, r in dgw.iterrows():
        ax.axvspan(r["start_time"], r["stop_time"], color="#FFEFC2", zorder=0)
        if np.isfinite(r["orientation"]):
            ax.text((r["start_time"] + r["stop_time"]) / 2, ntop * 1.005,
                    f"{int(r['orientation'])}°", ha="center", va="bottom", fontsize=7.5)
    for i, (u, a) in enumerate(zip(rows, row_area)):
        t = spikes[int(u)].restrict(ep).t
        ax.plot(t, np.full_like(t, i), "|", color=AREA_COLORS[a], ms=2.6, mew=0.65)
    for a in areas_shown[1:]:
        ax.axhline(np.where(row_area == a)[0][0] - 0.5, color="0.4", lw=0.7, ls="--")
    ax.set_xlim(t0, t1)
    ax.set_ylim(-2, ntop)
    ax.set_ylabel("unit")
    ax.set_title("B   Raw spike raster during drifting gratings "
                 "(shaded = 2 s grating; label = drift direction)", loc="left", pad=16)
    ax.legend(handles=[plt.Line2D([], [], color=AREA_COLORS[a], marker="|", ls="none",
                                  ms=9, mew=2.2, label=f"{a} (n={np.sum(row_area == a)} shown)")
                       for a in areas_shown],
              fontsize=8, loc="center left", bbox_to_anchor=(1.005, 0.5), handlelength=1.1)

    ax = fig.add_subplot(gs[2])
    for a in areas_shown:
        m = all_areas == a
        cnt = spikes[[int(u) for u in all_uids[m]]].count(0.02, ep=ep)
        rate = np.asarray(cnt.values).sum(axis=1) / (0.02 * m.sum())
        ax.plot(cnt.t, pd.Series(rate).rolling(5, center=True, min_periods=1).mean(),
                color=AREA_COLORS[a], lw=1.1, label=f"{a} (all {m.sum()})")
    for _, r in dgw.iterrows():
        ax.axvspan(r["start_time"], r["stop_time"], color="#FFEFC2", zorder=0)
    ax.set_xlim(t0, t1)
    ax.set_ylabel("pop. rate\n(Hz / unit)")
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.005, 0.5), handlelength=1.1)
    ax.set_title("C   Population firing rate (20 ms bins, smoothed)", loc="left")

    ax = fig.add_subplot(gs[3])
    rs = running.restrict(ep)
    ax.plot(rs.t, rs.d, color="0.25", lw=0.9)
    for _, r in dgw.iterrows():
        ax.axvspan(r["start_time"], r["stop_time"], color="#FFEFC2", zorder=0)
    ax.set_xlim(t0, t1)
    ax.set_xlabel("time in session (s)")
    ax.set_ylabel("running\n(cm/s)")
    ax.set_title("D   Locomotion", loc="left")

    fig.suptitle(f"Session {info['session']} — raw data overview", y=0.955, fontsize=13)
    fig.savefig(out_path)
    plt.close(fig)


fig_raw_overview(spikes, info, dg, running, os.path.join(FIG_DIR, "fig01_raw_overview.png"))
print("wrote fig01_raw_overview.png")

# %% [markdown]
# ## 3. Direction tuning curves and selectivity indices
#
# For every unit we count spikes inside each 2 s grating presentation and convert to a
# firing rate. Because responses to drifting gratings depend jointly on direction and
# temporal frequency, we build the full direction × temporal-frequency response matrix
# and take the direction tuning curve at each unit's preferred temporal frequency, which
# is the standard treatment for this stimulus set.
#
# From the 8-point direction tuning curve $R(\theta)$ we compute:
#
# * **global OSI** $= \left|\sum_\theta R(\theta)e^{2i\theta}\right| / \sum_\theta R(\theta)$ —
#   a vector-strength measure in orientation space. The factor of 2 in the exponent makes
#   θ and θ+180° equivalent, so a unit that responds equally to both directions of one
#   axis still scores high.
# * **OSI** $= (R_{pref} - R_{orth}) / (R_{pref} + R_{orth})$ — the classical two-point ratio,
#   after collapsing the 8 directions onto 4 orientations.
# * **global DSI** and **DSI**, the same two quantities computed at the first harmonic, which
#   measure preference for one direction of motion over the opposite one.
#
# A unit that is orientation tuned but not direction tuned has high gOSI and low gDSI.

# %%
def trial_rates(spikes, table, t_offset=0.0, window=None):
    """Per-trial firing rate (Hz) for every unit: (n_units, n_trials).

    The response windows must not overlap: pynapple merges overlapping intervals in an
    IntervalSet, which would silently collapse trials. Static gratings are presented
    back-to-back at 4 Hz, so their window has to be shorter than the 0.25 s slot.
    """
    start = table["start_time"].values + t_offset
    stop = start + (window if window is not None else table["duration"].values)
    order = np.argsort(start)
    assert np.all(start[order][1:] >= stop[order][:-1]), "response windows overlap"

    counts = spikes.count(ep=nap.IntervalSet(start=start, end=stop))
    assert counts.shape[0] == len(start), "IntervalSet lost trials"
    return np.asarray(counts.values).T / (stop - start)[None, :], np.asarray(counts.columns)


def _group_means(rates, labels, levels):
    return np.stack([rates[:, labels == lv].mean(axis=1) for lv in levels], axis=1)


def circular_selectivity(tc, angles_deg, harmonic=2):
    """Vector-strength selectivity index and preferred angle.

    harmonic=2 -> orientation (gOSI, preferred orientation in [0, 180))
    harmonic=1 -> direction   (gDSI, preferred direction in [0, 360))
    """
    theta = np.deg2rad(angles_deg)
    tc = np.clip(tc, 0, None)
    denom = tc.sum(axis=1)
    vec = (tc * np.exp(1j * harmonic * theta)[None, :]).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        idx = np.abs(vec) / denom
    pref = np.rad2deg(np.angle(vec) / harmonic) % (360.0 / harmonic)
    idx[denom <= 0] = np.nan
    return idx, pref


def ratio_osi(tc, angles_deg):
    ori = angles_deg % 180
    levels = np.unique(ori)
    otc = np.stack([tc[:, ori == lv].mean(axis=1) for lv in levels], axis=1)
    ipref = np.argmax(otc, axis=1)
    iorth = np.array([int(np.argmin(np.abs(levels - (levels[i] + 90) % 180))) for i in ipref])
    rp, ro = otc[np.arange(len(otc)), ipref], otc[np.arange(len(otc)), iorth]
    with np.errstate(invalid="ignore", divide="ignore"):
        osi = (rp - ro) / (rp + ro)
    osi[(rp + ro) <= 0] = np.nan
    return osi, levels[ipref]


def ratio_dsi(tc, angles_deg):
    ipref = np.argmax(tc, axis=1)
    inull = np.array([int(np.argmin(np.abs(angles_deg - (angles_deg[i] + 180) % 360)))
                      for i in ipref])
    rp, rn = tc[np.arange(len(tc)), ipref], tc[np.arange(len(tc)), inull]
    with np.errstate(invalid="ignore", divide="ignore"):
        dsi = (rp - rn) / (rp + rn)
    dsi[(rp + rn) <= 0] = np.nan
    return dsi, angles_deg[ipref]


def dg_tuning(spikes, dg):
    """Direction tuning at each unit's preferred temporal frequency."""
    driven, blank = dg[dg["orientation"].notna()], dg[dg["orientation"].isna()]
    rates, uids = trial_rates(spikes, driven)
    blank_rates, _ = trial_rates(spikes, blank)
    dirs, tfs = driven["orientation"].values, driven["temporal_frequency"].values
    tf_levels = np.unique(tfs)

    cube = np.stack([_group_means(rates[:, tfs == tf], dirs[tfs == tf], DG_DIRECTIONS)
                     for tf in tf_levels], axis=1)          # (units, TF, direction)
    pref_tf_i = np.argmax(cube.max(axis=2), axis=1)
    return {"uids": uids, "rates": rates, "dirs": dirs, "tfs": tfs, "tf_levels": tf_levels,
            "cube": cube, "pref_tf": tf_levels[pref_tf_i],
            "tc": cube[np.arange(cube.shape[0]), pref_tf_i],
            "baseline": blank_rates.mean(axis=1), "trial_table": driven}


def _gosi_from_trials(rates, dirs, tfs, tf_levels):
    cube = np.stack([_group_means(rates[:, tfs == tf], dirs[tfs == tf], DG_DIRECTIONS)
                     for tf in tf_levels], axis=1)
    i = np.argmax(cube.max(axis=2), axis=1)
    return circular_selectivity(cube[np.arange(cube.shape[0]), i], DG_DIRECTIONS, 2)[0]


def permutation_test_gosi(res, n_perm=1000, seed=0):
    """Shuffle responses within each temporal-frequency block.

    This preserves each unit's temporal-frequency preference and overall rate while
    destroying any relationship to direction, so the null hypothesis being tested is
    specifically "no orientation/direction tuning". The preferred-temporal-frequency
    selection is redone inside every permutation so the selection step cannot inflate
    the observed statistic relative to the null.
    """
    rng = np.random.default_rng(seed)
    rates, dirs, tfs, tf_levels = res["rates"], res["dirs"], res["tfs"], res["tf_levels"]
    obs = _gosi_from_trials(rates, dirs, tfs, tf_levels)
    blocks = [np.where(tfs == tf)[0] for tf in tf_levels]
    ge = np.zeros(rates.shape[0])
    for _ in range(n_perm):
        shuf = rates.copy()
        for b in blocks:
            shuf[:, b] = rates[:, rng.permutation(b)]
        ge += _gosi_from_trials(shuf, dirs, tfs, tf_levels) >= obs
    return obs, (ge + 1) / (n_perm + 1)


def sg_tuning(spikes, sg, latency=0.04, window=0.20):
    """Orientation tuning from static gratings, at each unit's preferred spatial frequency.

    Static gratings run back-to-back in 0.25 s slots, so the response window is taken as
    40-240 ms after onset: late enough to skip the response latency, short enough not to
    spill into the next presentation.
    """
    driven = sg[sg["orientation"].notna()]
    rates, uids = trial_rates(spikes, driven, t_offset=latency, window=window)
    oris, sfs = driven["orientation"].values, driven["spatial_frequency"].values
    sf_levels = np.unique(sfs)
    cube = np.stack([_group_means(rates[:, sfs == sf], oris[sfs == sf], SG_ORIENTATIONS)
                     for sf in sf_levels], axis=1)
    pref_sf_i = np.argmax(cube.max(axis=2), axis=1)
    return {"uids": uids, "rates": rates, "oris": oris, "sfs": sfs, "sf_levels": sf_levels,
            "cube": cube, "tc": cube[np.arange(cube.shape[0]), pref_sf_i],
            "pref_sf": sf_levels[pref_sf_i], "trial_table": driven}


def double_von_mises(theta_deg, r0, rp, rn, kappa, theta_pref):
    """Two opposite von Mises bumps: an orientation axis with a direction asymmetry."""
    th, tp = np.deg2rad(theta_deg), np.deg2rad(theta_pref)
    return (r0 + rp * np.exp(kappa * (np.cos(th - tp) - 1))
            + rn * np.exp(kappa * (np.cos(th - tp - np.pi) - 1)))


# The von Mises fit is used only as a smooth visual guide through the 8 measured points.
# We deliberately do not report a fitted tuning width: with directions sampled every 45
# degrees, any bump narrower than about half that spacing predicts baseline at all
# non-preferred samples, so widths below ~22 degrees are not identifiable from this
# stimulus set and the fitted value there is noise.
#
# That non-identifiability also has to be bounded or the fit misleads the eye. Left
# unconstrained, a sharply tuned unit is fitted with a very large kappa and a
# correspondingly enormous amplitude, because exp(kappa * (cos - 1)) is tiny at every
# sampled angle when the fitted peak falls between samples. The curve then agrees with
# the data at the 8 measured points while towering over them everywhere in between, which
# on a polar axis rescales the radial limit and squashes the real tuning curve to nothing.
# We therefore cap kappa at the half-width the 45 degree sampling can actually resolve:
# a von Mises bump has half-width-at-half-maximum Delta with cos(Delta) - 1 = -ln2 / kappa,
# so Delta = 22.5 degrees corresponds to kappa = ln2 / (1 - cos(22.5 deg)) = 9.1.
KAPPA_MAX = float(np.log(2) / (1 - np.cos(np.deg2rad(22.5))))


def fit_double_von_mises(tc_row, angles=DG_DIRECTIONS):
    i = int(np.argmax(tc_row))
    p0 = [tc_row.min(), max(tc_row.max() - tc_row.min(), 0.1), 0.1, 2.0, angles[i]]
    popt, _ = optimize.curve_fit(
        double_von_mises, angles, tc_row, p0=p0, maxfev=20000,
        bounds=([0, 0, 0, 0.05, -360], [np.inf, np.inf, np.inf, KAPPA_MAX, 720]))
    return popt


def circ_dist_deg(a, b, period=180.0):
    d = (a - b) % period
    return np.minimum(d, period - d)


# %%
res = dg_tuning(spikes, dg)
sres = sg_tuning(spikes, sg)
gosi, pref_ori = circular_selectivity(res["tc"], DG_DIRECTIONS, 2)
gdsi, pref_dir = circular_selectivity(res["tc"], DG_DIRECTIONS, 1)
osi, _ = ratio_osi(res["tc"], DG_DIRECTIONS)
dsi, _ = ratio_dsi(res["tc"], DG_DIRECTIONS)
obs, pval = permutation_test_gosi(res, n_perm=1000)

metrics = pd.DataFrame({
    "area": info["selected"].loc[res["uids"], "area"].values,
    "gosi": gosi, "osi": osi, "gdsi": gdsi, "dsi": dsi, "p": pval,
    "pref_ori": pref_ori, "pref_dir": pref_dir, "pref_tf": res["pref_tf"],
    "peak_rate": res["tc"].max(axis=1), "blank_rate": res["baseline"],
}, index=res["uids"])
print(metrics.groupby("area")[["gosi", "osi", "gdsi", "peak_rate"]].median().round(3).to_string())

# %% [markdown]
# ### Figure 2 — two example units
#
# One primary-visual-cortex unit and one thalamic relay unit from the same recording,
# responding to the same grating presentations.
#
# A note on reading the permutation p-value on these two panels. The LGd unit fires at
# 60-90 Hz at *every* direction, so its standard errors are tiny and the test can resolve a
# gOSI of 0.04 as formally significant. That is a statement about measurement precision, not
# about tuning: a 0.04 vector strength means the tuning curve is essentially a circle, which
# is what the polar plot shows. Throughout this notebook we therefore report the size of the
# selectivity index alongside its significance, and the LGd/VISp contrast below rests on the
# distribution of gOSI, not on the p-value alone.

# %%
def fig_example_unit(spikes, res, uid, area, out_path, gosi=None, dsi=None, pval=None, note=""):
    tbl = res["trial_table"]
    ui = int(np.where(res["uids"] == uid)[0][0])
    pref_tf = res["pref_tf"][ui]
    sub = tbl[tbl["temporal_frequency"] == pref_tf]

    fig = plt.figure(figsize=(13, 6.6))
    gs = gridspec.GridSpec(2, 6, width_ratios=[1, 1, 1, 1, 0.25, 1.5], hspace=0.62, wspace=0.42)
    win, bw = (-0.5, 2.5), 0.05
    bins = np.arange(win[0], win[1] + bw, bw)

    psths, rasters, psth_max = {}, {}, 0.0
    for d in DG_DIRECTIONS:
        onsets = nap.Ts(sub[sub["orientation"] == d]["start_time"].values)
        pe = nap.compute_perievent(spikes[int(uid)], onsets, window=win)
        trials = [pe[i].t for i in pe.index]
        rasters[d] = trials
        h = np.histogram(np.concatenate(trials) if trials else np.array([]), bins)[0] \
            / (len(trials) * bw)
        psths[d] = h
        psth_max = max(psth_max, h.max())

    for k, d in enumerate(DG_DIRECTIONS):
        row, col = divmod(k, 4)
        ax = fig.add_subplot(gs[row, col])
        ax.axvspan(0, 2, color="#FFEFC2", zorder=0)
        ntr = len(rasters[d])
        for j, t in enumerate(rasters[d]):
            ax.plot(t, np.full_like(t, j), "|", color="k", ms=3.2, mew=0.7)
        ax.set_ylim(-0.5, ntr - 0.5)
        ax.set_yticks([0, ntr - 1])
        ax.set_yticklabels(["1", str(ntr)], fontsize=7)
        ax2 = ax.twinx()
        ax2.plot(bins[:-1] + bw / 2, psths[d], color=AREA_COLORS.get(area, "C3"), lw=1.1)
        ax2.set_ylim(0, psth_max * 1.08)
        ax2.spines["right"].set_visible(True)
        ax2.tick_params(labelsize=7)
        if col == 3:
            ax2.set_ylabel("rate (Hz)", fontsize=8)
        else:
            ax2.set_yticklabels([])
        ax.set_xlim(*win)
        ax.set_xticks([0, 1, 2])
        ax.set_title(f"{int(d)}°", fontsize=10, pad=3)
        if col == 0:
            ax.set_ylabel("trial", fontsize=8)
        if row == 1:
            ax.set_xlabel("time from onset (s)", fontsize=8)

    ax = fig.add_subplot(gs[0, 5], projection="polar")
    tc = res["tc"][ui]
    rates, sel = res["rates"][ui], res["tfs"] == pref_tf
    sem = np.array([rates[sel & (res["dirs"] == d)].std(ddof=1)
                    / np.sqrt(np.sum(sel & (res["dirs"] == d))) for d in DG_DIRECTIONS])
    th, xs = np.deg2rad(DG_DIRECTIONS), np.linspace(0, 360, 361)
    popt = fit_double_von_mises(tc)
    ax.plot(np.deg2rad(xs), double_von_mises(xs, *popt), color="C3", lw=1.4, zorder=2,
            label="von Mises fit")
    ax.errorbar(np.append(th, th[0]), np.append(tc, tc[0]), yerr=np.append(sem, sem[0]),
                color="k", lw=1.4, marker="o", ms=4, capsize=2, zorder=3, label="mean ± SEM")
    ax.set_theta_zero_location("E")
    # The measured points, not the smoothing fit, set the radial scale. Radial tick labels
    # are thinned and parked at 112.5 deg, which is between two sampled directions, so they
    # do not sit on top of the tuning curve.
    rmax = float(max(np.max(tc + sem), double_von_mises(np.linspace(0, 360, 361), *popt).max()))
    ax.set_ylim(0, rmax * 1.06)
    ax.set_yticks(matplotlib.ticker.MaxNLocator(4, prune="lower").tick_values(0, rmax))
    ax.set_rlabel_position(112.5)
    ax.set_title("direction tuning curve (Hz)", fontsize=9.5, pad=22)
    ax.tick_params(labelsize=7, pad=2)
    ax.legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, -0.30), ncol=2)

    ax = fig.add_subplot(gs[1, 5])
    im = ax.imshow(res["cube"][ui], aspect="auto", origin="lower", cmap="magma",
                   extent=[-22.5, 337.5, -0.5, len(res["tf_levels"]) - 0.5])
    ax.set_xticks(DG_DIRECTIONS)
    ax.set_xticklabels([int(d) for d in DG_DIRECTIONS], fontsize=7, rotation=45)
    ax.set_yticks(range(len(res["tf_levels"])))
    ax.set_yticklabels([f"{t:g}" for t in res["tf_levels"]], fontsize=7)
    ax.set_xlabel("direction (deg)", fontsize=8)
    ax.set_ylabel("TF (Hz)", fontsize=8)
    ax.axhline(np.where(res["tf_levels"] == pref_tf)[0][0], color="w", lw=0.8, ls=":")
    ax.set_title("mean rate: direction × temporal frequency", fontsize=9.5, pad=4)
    cb = fig.colorbar(im, ax=ax, pad=0.03)
    cb.ax.tick_params(labelsize=7)
    cb.set_label("Hz", fontsize=7)

    txt = f"unit {uid} · {area} · preferred TF {pref_tf:g} Hz"
    if gosi is not None:
        txt += (f" · gOSI {gosi:.2f} · DSI {dsi:.2f} · "
                f"permutation p = {pval:.3g}")
    if note:
        txt += f"\n{note}"
    fig.suptitle(txt, y=1.0, fontsize=11)
    fig.savefig(out_path)
    plt.close(fig)


# pick the most orientation-selective well-driven VISp unit, and the most strongly driven
# LGd unit, so the comparison is not stacked in favour of cortex on response magnitude
v1 = metrics[(metrics.area == "VISp") & (metrics.p < 0.005) & (metrics.peak_rate > 12)]
v1_uid = int(v1["gosi"].idxmax())
lgd = metrics[metrics.area == "LGd"]
lgd_uid = int(lgd["peak_rate"].idxmax())

for uid, area, fname, note in [
    (v1_uid, "VISp", "fig02a_example_VISp.png",
     "Primary visual cortex: responds only to one orientation axis, at both drift directions"),
    (lgd_uid, "LGd", "fig02b_example_LGd.png",
     "Thalamic relay (LGd): strongly driven by every grating, but not orientation tuned"),
]:
    r = metrics.loc[uid]
    fig_example_unit(spikes, res, uid, area, os.path.join(FIG_DIR, fname),
                     r["gosi"], r["dsi"], r["p"], note)
    print(f"wrote {fname}  (unit {uid}, {area}, gOSI {r['gosi']:.2f}, peak {r['peak_rate']:.1f} Hz)")

# %% [markdown]
# ## 4. Pooling across six sessions
#
# The single-session result is suggestive but rests on a few dozen VISp units. We repeat
# the whole pipeline on six sessions and pool units, keeping the session identity so we
# can check that the effect is not carried by one animal.

# %%
def analyse_session(asset_path, n_perm=1000):
    spk, inf = load_spikes(asset_path)
    dg_t = stimulus_table(inf["nwbfile"], "drifting_gratings_presentations")
    sg_t = stimulus_table(inf["nwbfile"], "static_gratings_presentations")
    r = dg_tuning(spk, dg_t)
    s = sg_tuning(spk, sg_t)
    go, po = circular_selectivity(r["tc"], DG_DIRECTIONS, 2)
    gd, pd_ = circular_selectivity(r["tc"], DG_DIRECTIONS, 1)
    o, _ = ratio_osi(r["tc"], DG_DIRECTIONS)
    d, _ = ratio_dsi(r["tc"], DG_DIRECTIONS)
    sgo, spo = circular_selectivity(s["tc"], SG_ORIENTATIONS, 2)
    _, p = permutation_test_gosi(r, n_perm=n_perm)

    m = pd.DataFrame({
        "session": inf["session"], "area": inf["selected"].loc[r["uids"], "area"].values,
        "gosi": go, "osi": o, "gdsi": gd, "dsi": d, "p": p,
        "pref_ori": po, "pref_dir": pd_,
        "waveform_duration": inf["selected"].loc[r["uids"], "waveform_duration"].values,
        "sg_gosi": sgo, "sg_pref_ori": spo, "pref_tf": r["pref_tf"], "pref_sf": s["pref_sf"],
        "peak_rate": r["tc"].max(axis=1), "blank_rate": r["baseline"],
    }, index=r["uids"])
    m["group"] = m["area"].map(group_label)
    m["responsive"] = m["peak_rate"] > m["blank_rate"] + 1.0
    return m, r, s, spk, inf


all_metrics, dg_tcs, sg_tcs = [], [], []
for path in tqdm(SESSIONS, desc="sessions"):
    m, r, s, _, _ = analyse_session(path)
    all_metrics.append(m)
    dg_tcs.append(r["tc"])          # kept in the same row order as `m`
    sg_tcs.append(s["tc"])

pop = pd.concat(all_metrics)
dg_tc_all_units = np.concatenate(dg_tcs)
sg_tc_all_units = np.concatenate(sg_tcs)
assert len(pop) == len(dg_tc_all_units) == len(sg_tc_all_units)
pop_r = pop[pop["responsive"]].copy()      # units actually driven by the gratings
print(f"{len(pop)} QC-passing units across {pop['session'].nunique()} sessions; "
      f"{len(pop_r)} visually responsive")
print()
summary = (pop_r.groupby("area")
           .agg(n=("gosi", "size"), gOSI=("gosi", "median"), OSI=("osi", "median"),
                gDSI=("gdsi", "median"), DSI=("dsi", "median"),
                frac_tuned=("p", lambda x: np.mean(x < 0.05)))
           .reindex(VISUAL_AREAS).round(3))
print(summary.to_string())

# %% [markdown]
# ### Figure 3 — population tuning curves

# %%
def fig_population_tuning(tc_by_group, out_path, angles=DG_DIRECTIONS,
                          groups=("LGd", "LP", "VISp", "HVA"), title="Drifting gratings"):
    groups = [g for g in groups if g in tc_by_group and len(tc_by_group[g]) > 5]
    n = len(groups)
    fig = plt.figure(figsize=(2.5 * n + 5.0, 4.6))
    gs = gridspec.GridSpec(1, n + 3, width_ratios=[1] * n + [0.09, 0.42, 1.9], wspace=0.32)
    step = angles[1] - angles[0]

    for i, g in enumerate(groups):
        tc = np.asarray(tc_by_group[g], dtype=float)
        # normalise by peak only, so the depth of the trough reflects tuning strength
        norm = tc / np.maximum(tc.max(axis=1, keepdims=True), 1e-9)
        _, pref = circular_selectivity(tc, angles, harmonic=2)
        ax = fig.add_subplot(gs[i])
        im = ax.imshow(norm[np.argsort(pref)], aspect="auto", cmap="viridis", vmin=0, vmax=1,
                       extent=[angles[0] - step / 2, angles[-1] + step / 2, len(tc), 0],
                       interpolation="nearest")
        ax.set_xticks(angles[::2])
        ax.set_xticklabels([int(x) for x in angles[::2]], fontsize=7.5)
        ax.set_title(f"{g}   (n = {len(tc)})", fontsize=10)
        ax.set_xlabel("stimulus angle (deg)", fontsize=8.5)
        ax.set_ylabel("unit (sorted by preferred orientation)" if i == 0 else "", fontsize=8.5)
    cb = fig.colorbar(im, cax=fig.add_subplot(gs[n]))
    cb.set_label("response / peak response", fontsize=8.5)
    cb.ax.tick_params(labelsize=7.5)

    ax = fig.add_subplot(gs[n + 2])
    centre = len(angles) // 2
    rel = (np.arange(len(angles)) - centre) * step
    for g in groups:
        tc = np.asarray(tc_by_group[g], dtype=float)
        norm = tc / np.maximum(tc.max(axis=1, keepdims=True), 1e-9)
        aligned = np.stack([np.roll(r, centre - int(np.argmax(r))) for r in norm])
        mu, se = aligned.mean(axis=0), aligned.std(axis=0) / np.sqrt(len(aligned))
        ax.plot(rel, mu, color=AREA_COLORS.get(g, "k"), lw=1.7, marker="o", ms=4, label=g)
        ax.fill_between(rel, mu - se, mu + se, color=AREA_COLORS.get(g, "k"), alpha=0.22, lw=0)
    ax.set_xlabel("angle relative to each unit's preferred (deg)", fontsize=8.5)
    ax.set_ylabel("response / peak response", fontsize=8.5)
    ax.set_xticks(rel[::2])
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="lower center", ncol=len(groups), columnspacing=1.0)
    ax.set_title("preference-aligned population mean ± SEM", fontsize=10)
    fig.suptitle(f"{title} — single-unit tuning across the visual hierarchy", y=1.04, fontsize=12)
    fig.savefig(out_path)
    plt.close(fig)


keep = pop["responsive"].values
dg_tc_all = dg_tc_all_units[keep]
sg_tc_all = sg_tc_all_units[keep]
grp_all = pop_r["group"].values

fig_population_tuning({g: dg_tc_all[grp_all == g] for g in np.unique(grp_all)},
                      os.path.join(FIG_DIR, "fig03a_population_tuning_drifting.png"),
                      angles=DG_DIRECTIONS, title="Drifting gratings (8 directions)")
fig_population_tuning({g: sg_tc_all[grp_all == g] for g in np.unique(grp_all)},
                      os.path.join(FIG_DIR, "fig03b_population_tuning_static.png"),
                      angles=SG_ORIENTATIONS, title="Static gratings (6 orientations)")
print("wrote fig03a / fig03b")

# %% [markdown]
# ### Figure 4 — selectivity across the hierarchy

# %%
def fig_selectivity_by_area(df, out_path, areas=None):
    areas = areas or [a for a in VISUAL_AREAS if (df["area"] == a).sum() >= 20]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6))
    fig.subplots_adjust(hspace=0.48, wspace=0.28)

    ax = axes[0, 0]
    for a in areas:
        v = np.sort(df.loc[df["area"] == a, "gosi"].dropna().values)
        ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=AREA_COLORS[a],
                lw=2.0 if a in ("LGd", "VISp") else 1.0,
                alpha=1.0 if a in ("LGd", "VISp") else 0.75, label=f"{a} ({len(v)})")
    ax.set_xlabel("global OSI (drifting gratings)")
    ax.set_ylabel("cumulative fraction of units")
    ax.set_title("A   Orientation selectivity by area", loc="left")
    ax.legend(fontsize=7.5, ncol=2, loc="lower right")
    ax.set_xlim(0, 1)

    ax = axes[0, 1]
    data = [df.loc[df["area"] == a, "gosi"].dropna().values for a in areas]
    bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.6,
                    medianprops=dict(color="k", lw=1.4))
    for patch, a in zip(bp["boxes"], areas):
        patch.set_facecolor(AREA_COLORS[a])
        patch.set_alpha(0.65)
    rng = np.random.default_rng(0)
    for i, v in enumerate(data, start=1):
        ax.plot(i + rng.uniform(-0.16, 0.16, len(v)), v, ".", color="k", ms=1.4, alpha=0.30)
    ax.set_xticks(range(1, len(areas) + 1))
    ax.set_xticklabels(areas, rotation=45, fontsize=8)
    ax.set_ylabel("global OSI")
    ax.set_title("B   gOSI distribution (box = quartiles)", loc="left")

    ax = axes[1, 0]
    frac = np.array([np.mean(df.loc[df["area"] == a, "p"] < 0.05) for a in areas])
    nn = np.array([int((df["area"] == a).sum()) for a in areas])
    ax.bar(range(len(areas)), frac, yerr=np.sqrt(frac * (1 - frac) / nn), capsize=3,
           color=[AREA_COLORS[a] for a in areas], alpha=0.85)
    ax.axhline(0.05, color="k", ls="--", lw=0.9)
    ax.text(len(areas) - 0.4, 0.075, "chance (α = 0.05)", ha="right", fontsize=7.5)
    ax.set_xticks(range(len(areas)))
    ax.set_xticklabels(areas, rotation=45, fontsize=8)
    ax.set_ylabel("fraction of units")
    ax.set_ylim(0, 1)
    ax.set_title("C   Significantly tuned units (permutation test)", loc="left")

    ax = axes[1, 1]
    for a in ["LGd", "VISp"]:
        m = df["area"] == a
        ax.plot(df.loc[m, "gosi"], df.loc[m, "gdsi"], ".", ms=4, alpha=0.5,
                color=AREA_COLORS[a], label=a)
    ax.plot([0, 1], [0, 1], color="0.6", lw=0.8, ls=":")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("global OSI (orientation)")
    ax.set_ylabel("global DSI (direction)")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("D   Orientation vs direction selectivity", loc="left")

    fig.suptitle("Orientation selectivity emerges between thalamus and cortex "
                 f"({df['session'].nunique()} sessions, {len(df)} responsive units)",
                 y=0.98, fontsize=12.5)
    fig.savefig(out_path)
    plt.close(fig)


fig_selectivity_by_area(pop_r, os.path.join(FIG_DIR, "fig04_selectivity_by_area.png"))
print("wrote fig04_selectivity_by_area.png")

# %%
# Formal test: is VISp gOSI higher than LGd gOSI?
a = pop_r.loc[pop_r.area == "LGd", "gosi"].dropna()
b = pop_r.loc[pop_r.area == "VISp", "gosi"].dropna()
u, p_mw = stats.mannwhitneyu(b, a, alternative="greater")
print(f"gOSI  LGd  median {a.median():.3f} (n={len(a)})")
print(f"gOSI  VISp median {b.median():.3f} (n={len(b)})")
print(f"Mann-Whitney U = {u:.0f}, one-sided p = {p_mw:.3e}")

sig_v1 = int(np.sum(pop_r.loc[pop_r.area == "VISp", "p"] < 0.05))
sig_lgd = int(np.sum(pop_r.loc[pop_r.area == "LGd", "p"] < 0.05))
n_v1 = int((pop_r.area == "VISp").sum())
n_lgd = int((pop_r.area == "LGd").sum())
frac_v1, frac_lgd = sig_v1 / n_v1, sig_lgd / n_lgd
chi2, p_chi = stats.chi2_contingency(
    [[sig_v1, n_v1 - sig_v1], [sig_lgd, n_lgd - sig_lgd]])[:2]
print(f"\nsignificantly tuned: VISp {frac_v1:.1%} vs LGd {frac_lgd:.1%}; "
      f"chi2 = {chi2:.1f}, p = {p_chi:.3e}")

print("\nper-session VISp vs LGd median gOSI:")
print(pop_r[pop_r.area.isin(["LGd", "VISp"])]
      .pivot_table(index="session", columns="area", values="gosi", aggfunc="median")
      .round(3).to_string())

# %% [markdown]
# ## 5. Is it really *orientation*? Cross-validation with static gratings
#
# A drifting grating confounds orientation with motion direction, and a unit could in
# principle appear "orientation tuned" because of some property of the moving stimulus
# rather than of the static edge. The static gratings provide an independent test: they
# are flashed for 250 ms with no motion, at 6 orientations. If a unit's preferred angle
# measured from drifting gratings agrees with the one measured from static gratings, the
# preference is for orientation.

# %%
def fig_dg_vs_sg(df, out_path):
    tuned = df[(df["p"] < 0.05) & (df["gosi"] > 0.2)].copy()
    tuned["dist"] = circ_dist_deg(tuned["pref_ori"].values, tuned["sg_pref_ori"].values, 180.0)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.1))
    fig.subplots_adjust(wspace=0.36)

    ax = axes[0]
    for g, mk in [("VISp", "o"), ("HVA", "^"), ("LGd", "s")]:
        m = tuned["group"] == g
        if m.sum() == 0:
            continue
        ax.plot(tuned.loc[m, "pref_ori"], tuned.loc[m, "sg_pref_ori"], mk, ms=4, alpha=0.6,
                color=AREA_COLORS[g], label=f"{g} (n={int(m.sum())})", mew=0)
    ax.plot([0, 180], [0, 180], color="0.5", lw=0.9, ls="--")
    ax.set_xlim(0, 180)
    ax.set_ylim(0, 180)
    ax.set_xticks([0, 45, 90, 135, 180])
    ax.set_yticks([0, 45, 90, 135, 180])
    ax.set_xlabel("preferred orientation, drifting gratings (deg)")
    ax.set_ylabel("preferred orientation, static gratings (deg)")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("A   Same preference, two stimuli", loc="left")

    ax = axes[1]
    bins = np.arange(0, 91, 7.5)
    for g in ["VISp", "HVA", "LGd"]:
        v = tuned.loc[tuned["group"] == g, "dist"].dropna()
        if len(v) < 5:
            continue
        ax.hist(v, bins=bins, density=True, histtype="step", lw=1.8,
                color=AREA_COLORS[g], label=f"{g} (median {v.median():.0f}°)")
    ax.axhline(1 / 90, color="k", ls="--", lw=1.0)
    ax.text(88, 1 / 90 * 1.06, "chance", ha="right", fontsize=8)
    ax.set_xlabel("|preferred orientation difference| (deg)")
    ax.set_ylabel("probability density")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("B   Agreement between the two measurements", loc="left")

    ax = axes[2]
    for g in ["VISp", "HVA", "LGd", "LP"]:
        m = df["group"] == g
        ax.plot(df.loc[m, "gosi"], df.loc[m, "sg_gosi"], ".", ms=3.5, alpha=0.45,
                color=AREA_COLORS[g], label=g)
    lim = 1.0
    ax.plot([0, lim], [0, lim], color="0.6", lw=0.8, ls=":")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("gOSI, drifting gratings")
    ax.set_ylabel("gOSI, static gratings")
    ax.legend(fontsize=8, loc="upper left")
    rr = df[["gosi", "sg_gosi"]].dropna()
    rho = stats.spearmanr(rr["gosi"], rr["sg_gosi"]).statistic
    ax.set_title(f"C   Selectivity agrees too (Spearman ρ = {rho:.2f})", loc="left")

    fig.suptitle("Orientation preference measured from moving and from static gratings",
                 y=1.03, fontsize=12.5)
    fig.savefig(out_path)
    plt.close(fig)
    return tuned


tuned = fig_dg_vs_sg(pop_r, os.path.join(FIG_DIR, "fig05_drifting_vs_static.png"))
print("wrote fig05_drifting_vs_static.png")
for g in ["VISp", "HVA", "LGd"]:
    v = tuned.loc[tuned["group"] == g, "dist"].dropna()
    if len(v) >= 5:
        # under the null of unrelated preferences the difference is uniform on [0, 90]
        ks = stats.kstest(v, "uniform", args=(0, 90))
        print(f"{g:5s} n={len(v):4d}  median |Δpref| = {v.median():5.1f}°  "
              f"(chance 45°)  KS vs uniform p = {ks.pvalue:.2e}")

# %% [markdown]
# ## 6. Controls: which orientations, which cell types, and is it a rate artefact?
#
# Three checks on the single-unit result.
#
# *Which orientations.* If the cortical preferences were an artefact of the analysis they
# would be uniform over angle. They are not: mouse V1 over-represents the cardinal
# orientations (0° and 90°, i.e. horizontal and vertical gratings), a bias that has been
# reported repeatedly and that we can test here against a uniform null.
#
# *Which cell types.* Spike waveform duration separates narrow-spiking (putative fast-spiking
# inhibitory) from broad-spiking (putative excitatory) units. Narrow-spiking cells are
# expected to be more broadly tuned.
#
# *Is it a rate artefact.* A selectivity index computed from few spikes is noisy and biased
# upward, so we check that high gOSI is not confined to low-rate units.

# %%
def fig_controls(df, out_path, narrow_thresh=0.4):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.1))
    fig.subplots_adjust(wspace=0.36)

    # A: distribution of preferred orientations in cortex
    ax = axes[0]
    tuned = df[(df["p"] < 0.05) & (df["gosi"] > 0.2)]
    edges = np.arange(-22.5, 180, 45)          # bins centred on 0, 45, 90, 135
    centres = np.array([0, 45, 90, 135])
    width = 34
    for i, g in enumerate(["VISp", "HVA"]):
        v = tuned.loc[tuned["group"] == g, "pref_ori"].dropna().values
        wrapped = np.where(v >= 157.5, v - 180, v)
        counts, _ = np.histogram(wrapped, bins=edges)
        frac = counts / counts.sum()
        chi2, pchi = stats.chisquare(counts)
        ax.bar(centres + (i - 0.5) * width / 2, frac, width=width / 2,
               color=AREA_COLORS[g], alpha=0.8,
               label=f"{g} (n={len(v)}, χ² p = {pchi:.1e})")
    ax.axhline(0.25, color="k", ls="--", lw=1.0)
    ax.text(150, 0.257, "uniform", fontsize=8, ha="right")
    ax.set_xticks(centres)
    ax.set_xticklabels(["0°\n(horizontal)", "45°", "90°\n(vertical)", "135°"], fontsize=8)
    ax.set_xlabel("preferred orientation")
    ax.set_ylabel("fraction of tuned units")
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    ax.set_title("A   Cardinal orientations are over-represented", loc="left")

    # B: narrow- vs broad-spiking cortical units
    ax = axes[1]
    ctx = df[df["group"].isin(["VISp", "HVA"]) & df["waveform_duration"].notna()].copy()
    ctx["type"] = np.where(ctx["waveform_duration"] < narrow_thresh, "narrow", "broad")
    for t, c in [("broad", "#5B7DB1"), ("narrow", "#C0703F")]:
        v = np.sort(ctx.loc[ctx["type"] == t, "gosi"].dropna().values)
        ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=c, lw=2.0,
                label=f"{t}-spiking (n={len(v)}, median {np.median(v):.2f})")
    u, pmw = stats.mannwhitneyu(ctx.loc[ctx["type"] == "broad", "gosi"].dropna(),
                                ctx.loc[ctx["type"] == "narrow", "gosi"].dropna(),
                                alternative="greater")
    ax.set_xlim(0, 1)
    ax.set_xlabel("global OSI")
    ax.set_ylabel("cumulative fraction")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title(f"B   Cortical cell types (Mann-Whitney p = {pmw:.1e})", loc="left")

    # C: selectivity is not a low-rate artefact
    ax = axes[2]
    for g in ["VISp", "LGd"]:
        m = df["group"] == g
        ax.plot(df.loc[m, "peak_rate"], df.loc[m, "gosi"], ".", ms=3.5, alpha=0.45,
                color=AREA_COLORS[g], label=g)
    for g in ["VISp", "LGd"]:
        m = df["group"] == g
        bins = np.geomspace(1, max(df.loc[m, "peak_rate"].max(), 2), 8)
        idx = np.digitize(df.loc[m, "peak_rate"], bins)
        mids, meds = [], []
        for b in range(1, len(bins)):
            sel = df.loc[m][idx == b]
            if len(sel) >= 8:
                mids.append(np.sqrt(bins[b - 1] * bins[b]))
                meds.append(sel["gosi"].median())
        ax.plot(mids, meds, "-o", color=AREA_COLORS[g], lw=2.0, ms=5,
                markeredgecolor="w", markeredgewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("peak response across directions (Hz)")
    ax.set_ylabel("global OSI")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("C   gOSI vs response magnitude (lines = binned median)", loc="left")

    fig.suptitle("Controls on the single-unit orientation measurements", y=1.03, fontsize=12.5)
    fig.savefig(out_path)
    plt.close(fig)


fig_controls(pop_r, os.path.join(FIG_DIR, "fig06_controls.png"))
print("wrote fig06_controls.png")

tuned = pop_r[(pop_r.p < 0.05) & (pop_r.gosi > 0.2)]
for g in ["VISp", "HVA"]:
    v = tuned.loc[tuned["group"] == g, "pref_ori"].dropna().values
    card = np.mean(np.minimum(v % 90, 90 - (v % 90)) < 22.5)
    counts = np.histogram(np.where(v >= 157.5, v - 180, v), bins=np.arange(-22.5, 180, 45))[0]
    print(f"{g:5s} n={len(v):4d}  cardinal-preferring {card:.1%} (uniform = 50%)  "
          f"chi2 p = {stats.chisquare(counts).pvalue:.2e}")

# %% [markdown]
# ## 7. A Poisson GLM: is the tuning explained by locomotion?
#
# Mouse visual cortex is strongly modulated by locomotion and arousal. Because the trial
# order is randomised this cannot produce a spurious orientation preference on average,
# but it is worth showing directly. We fit per-trial spike counts with NeMoS using three
# nested Poisson GLMs:
#
# 1. **mean rate** — intercept only;
# 2. **nuisance** — smooth functions of running speed and log temporal frequency;
# 3. **full** — nuisance plus a cyclic B-spline basis over drift direction.
#
# All three are scored by five-fold cross-validated log-likelihood, so the comparison is
# on held-out trials and the extra parameters in the full model cannot help by themselves.

# %%
import nemos as nmo
import jax
from sklearn.model_selection import KFold

jax.config.update("jax_enable_x64", True)

dir_basis = nmo.basis.CyclicBSplineEval(n_basis_funcs=8, bounds=(0.0, 360.0), label="direction")
tf_basis = nmo.basis.BSplineEval(n_basis_funcs=4, label="log2_tf")
spd_basis = nmo.basis.BSplineEval(n_basis_funcs=4, label="running_speed")
full_basis = dir_basis + tf_basis + spd_basis
nuis_basis = tf_basis + spd_basis


def trial_running_speed(running, table):
    """Mean running speed inside each trial, via pynapple interval restriction."""
    ep = nap.IntervalSet(start=table["start_time"].values, end=table["stop_time"].values)
    return np.array([np.nanmean(running.restrict(ep[i:i + 1]).d) for i in range(len(ep))])


def _poisson_ll(y, lam):
    return np.mean(y * np.log(lam) - lam - gammaln(y + 1))


def cv_poisson_ll(X, y, n_splits=5, seed=0):
    """Mean held-out Poisson log-likelihood per trial. X=None gives the mean-rate model.

    Also returns the saturated log-likelihood on the same held-out trials, which is the
    ceiling any model could reach. Raw log-likelihood differences scale with firing rate,
    so all comparisons below are expressed as a fraction of the gap to that ceiling.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    ll, sat = [], []
    for tr, te in kf.split(y):
        if X is None:
            lam = np.full(len(te), max(y[tr].mean(), 1e-6))
        else:
            m = nmo.glm.GLM(solver_name="LBFGS", regularizer="Ridge",
                            regularizer_strength=1e-3).fit(X[tr], y[tr])
            lam = np.clip(np.asarray(m.predict(X[te])), 1e-8, None)
        ll.append(_poisson_ll(y[te], lam))
        sat.append(_poisson_ll(y[te], np.clip(y[te], 1e-8, None)))
    return float(np.mean(ll)), float(np.mean(sat))


# fit on the first session's LGd and VISp units
spk0, info0 = load_spikes(SESSIONS[0])
dg0 = stimulus_table(info0["nwbfile"], "drifting_gratings_presentations")
run0 = running_speed(info0["nwbfile"])
res0 = dg_tuning(spk0, dg0)
speed0 = trial_running_speed(run0, res0["trial_table"])
log_tf0 = np.log2(res0["tfs"])
counts0 = np.round(res0["rates"] * 2.0)

X_full = full_basis.compute_features(res0["dirs"], log_tf0, speed0)
X_nuis = nuis_basis.compute_features(log_tf0, speed0)
areas0 = info0["selected"].loc[res0["uids"], "area"].values

glm_rows = []
targets = np.where(np.isin(areas0, ["LGd", "VISp"]))[0]
for i in tqdm(targets, desc="GLM fits"):
    y = counts0[i]
    if y.sum() < 30:
        continue
    ll_null, ll_sat = cv_poisson_ll(None, y)
    ll_nuis, _ = cv_poisson_ll(X_nuis, y)
    ll_full, _ = cv_poisson_ll(X_full, y)
    gap = ll_sat - ll_null                       # total explainable log-likelihood
    glm_rows.append({"uid": int(res0["uids"][i]), "area": areas0[i], "idx": int(i),
                     "ll_null": ll_null, "ll_nuis": ll_nuis, "ll_full": ll_full,
                     "ll_sat": ll_sat,
                     "d_running": (ll_nuis - ll_null) / gap,
                     "d_direction": (ll_full - ll_nuis) / gap})
glm = pd.DataFrame(glm_rows)
print("fraction of explainable held-out log-likelihood captured by each term:")
print(glm.groupby("area")[["d_running", "d_direction"]].median().round(4).to_string())

# %%
def fig_glm(glm, res, X_full, out_path, example_idx=None):
    fig = plt.figure(figsize=(12.5, 4.4))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.1, 1.0, 1.25], wspace=0.42)

    ax = fig.add_subplot(gs[0])
    for a in ["LGd", "VISp"]:
        m = glm["area"] == a
        ax.plot(glm.loc[m, "d_running"], glm.loc[m, "d_direction"], "o", ms=4.5, alpha=0.6,
                color=AREA_COLORS[a], mew=0, label=f"{a} (n={int(m.sum())})")
    hi = float(np.nanpercentile(np.r_[glm["d_running"], glm["d_direction"]], 99)) * 1.15
    lo = -0.02
    ax.plot([lo, hi], [lo, hi], color="0.6", lw=0.8, ls=":")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.axhline(0, color="k", lw=0.7)
    ax.axvline(0, color="k", lw=0.7)
    ax.set_xlabel("explainable log-likelihood captured\nby running + temporal frequency")
    ax.set_ylabel("explainable log-likelihood captured\nby adding direction")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("A   What the GLM gains from each term", loc="left")

    ax = fig.add_subplot(gs[1])
    bins = np.linspace(-0.02, float(np.nanpercentile(glm["d_direction"], 99)) * 1.15, 26)
    for a in ["LGd", "VISp"]:
        v = glm.loc[glm["area"] == a, "d_direction"]
        ax.hist(v, bins=bins, histtype="stepfilled", alpha=0.5, color=AREA_COLORS[a],
                label=f"{a} (median {v.median():.3f})")
    ax.set_xlim(bins[0], bins[-1])
    ax.axvline(0, color="k", lw=0.9, ls="--")
    ax.set_xlabel("explainable log-likelihood captured by direction")
    ax.set_ylabel("number of units")
    ax.legend(fontsize=8)
    ax.set_title("B   Gain from adding direction", loc="left")

    ax = fig.add_subplot(gs[2])
    if example_idx is not None:
        i = example_idx
        y = np.round(res["rates"][i] * 2.0)
        model = nmo.glm.GLM(solver_name="LBFGS", regularizer="Ridge",
                            regularizer_strength=1e-3).fit(X_full, y)
        # Marginal prediction: for each grid direction, predict with every trial's actual
        # temporal frequency and running speed, then average. This is what the model says
        # the measured direction tuning curve (pooled over all other conditions) should be.
        grid = np.linspace(0, 360, 91)
        pred = np.empty(len(grid))
        for k, g_ in enumerate(grid):
            Xg = full_basis.compute_features(np.full(len(log_tf0), g_), log_tf0, speed0)
            pred[k] = np.mean(np.asarray(model.predict(Xg))) / 2.0
        emp = np.array([res["rates"][i][res["dirs"] == d].mean() for d in DG_DIRECTIONS])
        sem = np.array([res["rates"][i][res["dirs"] == d].std(ddof=1)
                        / np.sqrt(np.sum(res["dirs"] == d)) for d in DG_DIRECTIONS])
        ax.errorbar(DG_DIRECTIONS, emp, yerr=sem, fmt="o", color="k", ms=5, capsize=2,
                    label="measured (mean ± SEM)")
        ax.plot(grid, pred, color="C3", lw=1.8, label="Poisson GLM (marginal)")
        ax.set_xticks(DG_DIRECTIONS)
        ax.set_xlabel("drift direction (deg)")
        ax.set_ylabel("firing rate (Hz)")
        ax.legend(fontsize=8)
        ax.set_ylim(bottom=0)
        ax.set_title(f"C   GLM tuning curve, unit {int(res['uids'][i])} (VISp)", loc="left")

    fig.suptitle("Orientation tuning survives after accounting for locomotion and "
                 "temporal-frequency preference", y=1.04, fontsize=12.5)
    fig.savefig(out_path)
    plt.close(fig)


best = glm[glm.area == "VISp"].sort_values("d_direction", ascending=False)
fig_glm(glm, res0, X_full, os.path.join(FIG_DIR, "fig07_glm.png"),
        example_idx=int(best["idx"].iloc[0]))
print("wrote fig07_glm.png")
for a in ["LGd", "VISp"]:
    v = glm.loc[glm.area == a]
    print(f"{a:5s} n={len(v):3d}  units where direction improves held-out LL: "
          f"{np.mean(v['d_direction'] > 0):.0%}  "
          f"(median fraction of explainable LL = {v['d_direction'].median():+.3f})")

# %% [markdown]
# ## 8. Decoding stimulus orientation from population activity
#
# The single-unit indices say each neuron carries some orientation information. The
# population question is whether that information is usable: given only the spike counts
# of *N* simultaneously recorded units on one trial, can we say which grating was shown?
# We use a Poisson naive-Bayes decoder with five-fold cross-validation, and subsample each
# area to the same number of units so the comparison is not driven by yield.

# %%
def poisson_decode_cv(rates, labels, duration, n_splits=5, seed=0):
    from sklearn.model_selection import StratifiedKFold
    levels = np.unique(labels)
    counts = rates.T * duration
    pred = np.empty(len(labels), dtype=labels.dtype)
    for tr, te in StratifiedKFold(n_splits=n_splits, shuffle=True,
                                  random_state=seed).split(counts, labels):
        lam = np.clip(np.stack([counts[tr][labels[tr] == lv].mean(axis=0) for lv in levels]),
                      1e-3, None)
        # errstate: on macOS the Accelerate BLAS matmul kernel sets spurious FPU flags on
        # its SIMD padding lanes, so numpy reports overflow/divide-by-zero even though
        # every input here is finite (counts are bounded, lam is clipped away from 0).
        # The result was checked against np.einsum and a plain dot loop and agrees to 1e-12.
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            score = counts[te] @ np.log(lam).T - lam.sum(axis=1)[None, :]
        assert np.isfinite(score).all(), "decoder scores are not finite"
        pred[te] = levels[np.argmax(score, axis=1)]
    cm = np.zeros((len(levels), len(levels)))
    for i, lv in enumerate(levels):
        for j, lp in enumerate(levels):
            cm[i, j] = np.sum((labels == lv) & (pred == lp))
    return float((pred == labels).mean()), cm / cm.sum(axis=1, keepdims=True), levels


N_DECODE = 30          # units per area, matched
N_REPEATS = 20         # random subsamples

decode_rows, confusions = [], {}
for path in tqdm(SESSIONS, desc="decoding"):
    spk, inf = load_spikes(path)
    dgt = stimulus_table(inf["nwbfile"], "drifting_gratings_presentations")
    r = dg_tuning(spk, dgt)
    ar = inf["selected"].loc[r["uids"], "area"].values
    grp = np.array([group_label(a) for a in ar])
    for g in ["LGd", "LP", "VISp", "HVA"]:
        idx = np.where(grp == g)[0]
        if len(idx) < N_DECODE:
            continue
        rng = np.random.default_rng(0)
        for rep in range(N_REPEATS):
            sub = rng.choice(idx, N_DECODE, replace=False)
            acc, cm, lv = poisson_decode_cv(r["rates"][sub], r["dirs"], 2.0, seed=rep)
            # orientation accuracy: correct up to the 180 deg ambiguity
            decode_rows.append({"session": inf["session"], "group": g, "rep": rep,
                                "acc_dir": acc})
            if rep == 0:
                confusions.setdefault(g, []).append(cm)
dec = pd.DataFrame(decode_rows)
print(dec.groupby("group")["acc_dir"].agg(["mean", "std", "count"]).round(3).to_string())

# %%
def fig_decoding(dec, confusions, out_path, n_units=N_DECODE):
    groups = [g for g in ["LGd", "LP", "VISp", "HVA"] if g in confusions]
    fig = plt.figure(figsize=(3.0 * len(groups) + 5.4, 4.2))
    gs = gridspec.GridSpec(1, len(groups) + 3,
                           width_ratios=[1] * len(groups) + [0.08, 0.55, 1.6], wspace=0.34)
    for i, g in enumerate(groups):
        cm = np.mean(confusions[g], axis=0)
        ax = fig.add_subplot(gs[i])
        im = ax.imshow(cm, cmap="magma", vmin=0, vmax=max(0.35, cm.max()))
        ax.set_xticks(range(8))
        ax.set_xticklabels([int(d) for d in DG_DIRECTIONS], rotation=45, fontsize=7)
        ax.set_yticks(range(8))
        ax.set_yticklabels([int(d) for d in DG_DIRECTIONS], fontsize=7)
        ax.set_xlabel("decoded direction", fontsize=8.5)
        if i == 0:
            ax.set_ylabel("true direction", fontsize=8.5)
        acc = dec.loc[dec["group"] == g, "acc_dir"].mean()
        ax.set_title(f"{g}\n{acc:.0%} correct", fontsize=9.5)
    cb = fig.colorbar(im, cax=fig.add_subplot(gs[len(groups)]))
    cb.set_label("P(decoded | true)", fontsize=8.5)
    cb.ax.tick_params(labelsize=7.5)

    ax = fig.add_subplot(gs[len(groups) + 2])
    data = [dec.loc[dec["group"] == g, "acc_dir"].values for g in groups]
    bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=False,
                    medianprops=dict(color="k", lw=1.4))
    for patch, g in zip(bp["boxes"], groups):
        patch.set_facecolor(AREA_COLORS[g])
        patch.set_alpha(0.7)
    ax.axhline(1 / 8, color="k", ls="--", lw=1.0)
    ax.text(len(groups) + 0.42, 1 / 8 + 0.015, "chance (1/8)", ha="right", fontsize=8)
    ax.set_xticks(range(1, len(groups) + 1))
    ax.set_xticklabels(groups, fontsize=8.5)
    ax.set_ylabel("decoding accuracy")
    ax.set_ylim(0, 1)
    ax.set_title(f"accuracy over {N_REPEATS} random subsamples\n"
                 f"({n_units} units each, all sessions)", fontsize=9.5)

    fig.suptitle(f"Decoding grating direction from {n_units} simultaneously recorded units",
                 y=1.05, fontsize=12.5)
    fig.savefig(out_path)
    plt.close(fig)


fig_decoding(dec, confusions, os.path.join(FIG_DIR, "fig08_decoding.png"))
print("wrote fig08_decoding.png")

# %% [markdown]
# The structure of the errors matters as much as the accuracy. If a population carried a
# pure orientation code with no direction information, the decoder would be unable to tell
# θ from θ+180° and its mistakes would pile up on the anti-diagonal 180° away from the
# truth. We compare the rate of 180° errors with the mean rate of the other six error
# types, and we also report accuracy after collapsing the eight directions onto the four
# orientations.

# %%
conf_rows = []
for g, cms in confusions.items():
    cm = np.mean(cms, axis=0)
    correct = float(np.mean(np.diag(cm)))
    opp = float(np.mean([cm[i, (i + 4) % 8] for i in range(8)]))
    other = float((cm.sum() - np.trace(cm) - sum(cm[i, (i + 4) % 8] for i in range(8)))
                  / (8 * 6))
    # orientation accuracy: decoded direction is correct modulo 180 degrees
    ori_acc = float(np.mean([cm[i, i] + cm[i, (i + 4) % 8] for i in range(8)]))
    conf_rows.append({"group": g, "direction_correct": correct, "180deg_error": opp,
                      "other_error_mean": other, "error_ratio": opp / other,
                      "orientation_correct": ori_acc})
conf = pd.DataFrame(conf_rows).set_index("group")
print(conf.round(3).to_string())

# %% [markdown]
# ## Summary
#
# Every measurement points the same way.
#
# * Single units in primary visual cortex respond to one orientation axis and are near
#   silent at the orthogonal angle, while thalamic relay units in LGd are driven hard by
#   every grating and barely distinguish angles (Figure 2).
# * Pooled over six mice, the median global OSI rises roughly three-fold from LGd to VISp,
#   and the fraction of units with statistically significant tuning rises from under a half
#   to over four fifths (Figure 4). The ordering holds in every session.
# * The preferred angle measured from 2 s moving gratings agrees with the one measured
#   from 250 ms static gratings far better than chance, so the preference is for
#   orientation and not for something specific to motion (Figure 5).
# * A cross-validated Poisson GLM shows that drift direction still improves held-out
#   prediction once running speed and temporal-frequency preference are already in the
#   model, so locomotion does not manufacture the tuning. The direction term helps in both
#   structures, but it buys about twice as much in VISp as in LGd (median 7.0% versus 4.3%
#   of the explainable log-likelihood), which is the same ordering the selectivity indices
#   give (Figure 7).
# * Thirty simultaneously recorded VISp units are enough to decode which of eight gratings
#   was shown well above chance, and the decoder's mistakes are overwhelmingly 180°
#   confusions, which is the signature of an orientation code rather than a direction code
#   (Figure 8).

# %%
pop.to_csv(os.path.join(FIG_DIR, "unit_metrics.csv"))
summary.to_csv(os.path.join(FIG_DIR, "summary_by_area.csv"))
print("wrote unit_metrics.csv and summary_by_area.csv")
print(f"\n{len(pop)} units | {pop['session'].nunique()} sessions | DANDI:{DANDISET}")
