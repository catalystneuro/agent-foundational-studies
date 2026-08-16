# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Orientation selectivity in the mouse visual system
#
# Neurons in visual cortex respond selectively to the *orientation* of an edge or
# grating in their receptive field. A cell that fires strongly to a vertical bar
# fires weakly to a horizontal one, and, because an orientation is defined modulo
# 180 degrees, it responds about equally well to a grating drifting left-to-right
# and to the same grating drifting right-to-left. Orientation selectivity is the
# canonical example of a computed feature: it is largely absent in the retinal
# input reaching the dorsal lateral geniculate nucleus and emerges in cortex.
#
# This notebook demonstrates the phenomenon in real extracellular recordings from
# the DANDI Archive and asks four questions, each with an explicit control:
#
# 1. **Do single units show orientation tuning?** Direction tuning curves and a
#    within-block label-shuffle permutation test.
# 2. **Is it specific to visual structures?** The same analysis applied to
#    simultaneously recorded hippocampal units, which serve as a negative control.
# 3. **Does the preference generalise across stimuli?** Preferred orientation
#    estimated from drifting gratings is compared with the value estimated from
#    *static* gratings, an independent stimulus class recorded in the same session.
# 4. **Does it survive a behavioural confound?** A Poisson GLM (NeMoS) asks whether
#    grating direction predicts held-out spike counts over and above running speed,
#    which strongly modulates firing rate in mouse visual cortex.
#
# ## Dataset
#
# **DANDI:000021**, *Allen Institute Visual Coding, Neuropixels (Brain Observatory
# 1.1 stimulus set)*. Head-fixed mice on a running wheel viewed a battery of visual
# stimuli while up to six Neuropixels probes recorded simultaneously from visual
# cortex, visual thalamus, hippocampus and midbrain. Two stimulus classes are used
# here:
#
# | stimulus | conditions | repeats | duration |
# |---|---|---|---|
# | drifting gratings | 8 directions x 5 temporal frequencies | 15 | 2 s |
# | static gratings | 6 orientations x 5 spatial frequencies x 4 phases | ~50 | 0.25 s |
#
# Files are read by streaming from the DANDI S3 bucket with `remfile` plus a local
# disk cache; nothing is downloaded in full. Eight sessions are analysed.

# %% [markdown]
# ## Setup
#
# `jax` is configured for 64-bit before NeMoS is imported, and matplotlib runs on
# the non-interactive Agg backend so the notebook executes headless.

# %%
import os
os.environ.setdefault("MPLBACKEND", "Agg")

import jax
jax.config.update("jax_enable_x64", True)

import glob
import json
import pickle
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import h5py
import remfile
from pynwb import NWBHDF5IO
import pynapple as nap
import nemos as nmo
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

N_SESSIONS = int(os.environ.get("N_SESSIONS", "8"))
OUT = "session_results"
os.makedirs(OUT, exist_ok=True)
print("pynapple", nap.__version__, "| nemos", nmo.__version__)

# %% [markdown]
# ## 1. Data access
#
# Session-level NWB assets are listed through the DANDI REST API. Loading is
# deliberately column-selective: `units.to_dataframe()` would pull `waveform_mean`
# and `spike_amplitudes`, several hundred megabytes per session that this analysis
# never touches.
#
# One subtlety matters for correctness. `nap.TsGroup` sorts its keys, so the
# per-unit metadata frame has to be reordered to match, or every unit annotation
# (brain area, quality metrics) is silently shuffled relative to its spike train.
# The assertion in `load_units` guards against that.

# %%
import json
import os

import numpy as np
import pandas as pd
import h5py
import remfile
from pynwb import NWBHDF5IO
import pynapple as nap

DANDISET = "000021"
VERSION = "0.251116.2246"
CACHE_DIR = os.environ.get("REMFILE_CACHE", "/tmp/remfile_cache_000021")
ASSET_JSON = os.environ.get("ASSET_JSON", "assets_000021.json")

# Cortical visual areas in this dataset, ordered roughly by hierarchy.
VISUAL_CORTEX = ["VISp", "VISl", "VISrl", "VISal", "VISpm", "VISam"]
THALAMUS = ["LGd", "LGv", "LP"]
CONTROL = ["CA1", "CA3", "DG"]

# Unit quality thresholds used throughout (Allen SDK defaults).
QC = dict(isi_violations=0.5, amplitude_cutoff=0.1, presence_ratio=0.9, snr=1.0)


# ----------------------------------------------------------------------------- loading


def get_assets():
    """Session-level (non-probe) NWB assets of dandiset 000021, smallest first."""
    if os.path.exists(ASSET_JSON):
        return json.load(open(ASSET_JSON))
    import urllib.request

    url = (
        f"https://api.dandiarchive.org/api/dandisets/{DANDISET}/versions/{VERSION}"
        "/assets/?page_size=250"
    )
    res = json.load(urllib.request.urlopen(url))
    main = [r for r in res["results"] if "probe" not in r["path"]]
    main.sort(key=lambda x: x["size"])
    out = [
        dict(
            path=r["path"],
            asset_id=r["asset_id"],
            size=r["size"],
            url=(
                f"https://api.dandiarchive.org/api/dandisets/{DANDISET}/versions/"
                f"{VERSION}/assets/{r['asset_id']}/download/"
            ),
        )
        for r in main
    ]
    json.dump(out, open(ASSET_JSON, "w"), indent=1)
    return out


def open_nwb(url):
    rf = remfile.File(url, disk_cache=remfile.DiskCache(CACHE_DIR))
    h5 = h5py.File(rf, "r")
    io = NWBHDF5IO(file=h5, load_namespaces=True)
    return io.read()


def load_units(nwbfile):
    """TsGroup of spike times plus a metadata frame (area, QC metrics) per unit.

    Only the columns needed downstream are read from disk.
    """
    ut = nwbfile.units
    spike_times = ut["spike_times"]  # VectorIndex: .data holds per-unit end offsets
    ends = np.asarray(spike_times.data[:], dtype=np.int64)
    starts = np.concatenate([[0], ends[:-1]])
    flat = np.asarray(spike_times.target.data[:], dtype=float)
    unit_ids = np.asarray(ut.id.data[:])

    meta_cols = [
        "peak_channel_id",
        "quality",
        "firing_rate",
        "snr",
        "isi_violations",
        "amplitude_cutoff",
        "presence_ratio",
        "waveform_duration",
    ]
    meta = {c: np.asarray(ut[c].data[:]) for c in meta_cols if c in ut.colnames}
    meta = pd.DataFrame(meta, index=unit_ids)
    meta.index.name = "unit_id"

    # unit -> brain area via the peak channel
    elec_id = np.asarray(nwbfile.electrodes.id.data[:])
    elec_loc = np.asarray(nwbfile.electrodes["location"].data[:]).astype(str)
    loc_map = pd.Series(elec_loc, index=elec_id)
    meta["area"] = loc_map.reindex(meta["peak_channel_id"].values).values
    meta["session_id"] = str(nwbfile.session_id)

    # TsGroup sorts its keys, so put the metadata in sorted-unit-id order too;
    # otherwise every per-unit annotation is silently shuffled.
    order = np.argsort(unit_ids)
    meta = meta.iloc[order]
    spikes = {
        int(unit_ids[i]): nap.Ts(t=flat[starts[i] : ends[i]]) for i in order
    }
    tsgroup = nap.TsGroup(spikes, metadata=meta)
    assert np.array_equal(np.asarray(tsgroup.index), meta.index.values)
    return tsgroup, meta


def load_stim_table(nwbfile, name):
    """Stimulus presentation table as a plain DataFrame (no `timeseries` column)."""
    tbl = nwbfile.intervals[name]
    cols = [c for c in tbl.colnames if c not in ("timeseries", "tags")]
    d = {c: np.asarray(tbl[c].data[:]) for c in cols}
    df = pd.DataFrame(d)
    for c in ("orientation", "temporal_frequency", "spatial_frequency", "contrast", "phase"):
        if c in df:
            df[c] = pd.to_numeric(
                pd.Series(df[c]).astype(str).str.replace("null", "nan"), errors="coerce"
            )
    return df


def load_running_speed(nwbfile):
    """Running speed as a pynapple Tsd, or None if absent."""
    proc = nwbfile.processing.get("running")
    if proc is None:
        return None
    for key in ("running_speed", "running_speed_end_times"):
        if key in proc.data_interfaces:
            ts = proc[key]
            t = np.asarray(ts.timestamps[:])
            v = np.asarray(ts.data[:])
            n = min(len(t), len(v))
            return nap.Tsd(t=t[:n], d=v[:n])
    return None


def passes_qc(meta):
    return (
        (meta["quality"].astype(str) == "good")
        & (meta["isi_violations"] < QC["isi_violations"])
        & (meta["amplitude_cutoff"] < QC["amplitude_cutoff"])
        & (meta["presence_ratio"] > QC["presence_ratio"])
        & (meta["snr"] > QC["snr"])
    )


# ----------------------------------------------------------- tuning / selectivity math


def trial_counts(tsgroup, starts, stops):
    """Spike counts per (trial, unit). Returns array (n_trials, n_units)."""
    ep = nap.IntervalSet(start=starts, end=stops)
    # count() with an IntervalSet of many short epochs: use restrict per epoch via
    # nap.compute_ ... simplest robust route is a vectorised searchsorted.
    out = np.zeros((len(starts), len(tsgroup)), dtype=float)
    for j, u in enumerate(tsgroup.index):
        t = tsgroup[u].t
        lo = np.searchsorted(t, starts, side="left")
        hi = np.searchsorted(t, stops, side="right")
        out[:, j] = hi - lo
    return out, ep


def osi_dsi(rates, directions_deg):
    """Global OSI / DSI from mean rates at each drift direction.

    OSI = |sum r * exp(2i*theta)| / sum r  (1 - circular variance at 2 theta)
    DSI = |sum r * exp(1i*theta)| / sum r
    Preferred orientation = 0.5 * angle(sum r exp(2i theta)) mapped to [0,180).
    """
    r = np.asarray(rates, dtype=float)
    th = np.deg2rad(np.asarray(directions_deg, dtype=float))
    tot = r.sum()
    if tot <= 0:
        return dict(osi=np.nan, dsi=np.nan, pref_ori=np.nan, pref_dir=np.nan)
    z2 = (r * np.exp(2j * th)).sum() / tot
    z1 = (r * np.exp(1j * th)).sum() / tot
    pref_ori = np.rad2deg(0.5 * np.angle(z2)) % 180.0
    pref_dir = np.rad2deg(np.angle(z1)) % 360.0
    return dict(osi=np.abs(z2), dsi=np.abs(z1), pref_ori=pref_ori, pref_dir=pref_dir)


def circ_dist_ori(a, b):
    """Smallest angular distance between two orientations, in degrees (0..90)."""
    d = np.abs(np.asarray(a) - np.asarray(b)) % 180.0
    return np.minimum(d, 180.0 - d)

# %% [markdown]
# ### Unit quality control
#
# Units are kept if they are labelled `good` by the Allen sorting pipeline and pass
# the standard metric thresholds (ISI violations < 0.5, amplitude cutoff < 0.1,
# presence ratio > 0.9, SNR > 1). This removes roughly two thirds of the raw
# clusters.

# %% [markdown]
# ## 2. Tuning curves and selectivity metrics
#
# For each trial the firing rate is the spike count inside the presentation window
# divided by its duration. Rates are then averaged within each stimulus condition.
# Drifting-grating trials are pooled across the five temporal frequencies, giving
# 75 repeats of each of the 8 drift directions; static-grating trials are pooled
# across spatial frequency and phase, giving roughly 200 repeats of each of the
# 6 orientations.
#
# Selectivity is quantified with resultant-vector (circular-variance) indices. With
# mean rate $r_k$ at stimulus angle $\theta_k$,
#
# $$\mathrm{OSI} = \frac{\left|\sum_k r_k e^{2i\theta_k}\right|}{\sum_k r_k},
# \qquad
# \mathrm{DSI} = \frac{\left|\sum_k r_k e^{i\theta_k}\right|}{\sum_k r_k},
# \qquad
# \theta_{\text{pref}} = \tfrac12 \arg\sum_k r_k e^{2i\theta_k}.$$
#
# Doubling the stimulus angle maps orientation (defined modulo 180 degrees) onto
# the full circle, so the same expression serves drift directions and static
# orientations. A ratio index $(R_{\text{pref}} - R_{\text{orth}}) /
# (R_{\text{pref}} + R_{\text{orth}})$ is reported alongside it.
#
# Significance comes from a permutation test in which stimulus labels are shuffled
# **within each stimulus block**. The drifting-grating trials are delivered in
# three blocks spread over a two-and-a-half-hour session, and firing rates drift
# slowly over that timescale; shuffling within block means slow drift cannot
# masquerade as tuning. Split-half reliability (the correlation between tuning
# curves computed from independent random halves of the trials) is reported as a
# model-free measure of how repeatable each curve is.

# %%
import numpy as np
import pandas as pd
import pynapple as nap
from scipy import stats


DIRECTIONS = np.array([0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0])
STATIC_ORIS = np.array([0.0, 30.0, 60.0, 90.0, 120.0, 150.0])
N_PERM = 1000


def _clean_trials(df, ori_values):
    """Drop blank-sweep trials and keep only the canonical stimulus values."""
    ok = df["orientation"].notna() & df["orientation"].isin(ori_values)
    return df[ok].reset_index(drop=True)


def trial_rate_matrix(tsgroup, df):
    """(n_trials, n_units) firing rates in the presentation window."""
    starts = df["start_time"].values
    stops = df["stop_time"].values
    dur = stops - starts
    counts = np.zeros((len(starts), len(tsgroup)), dtype=float)
    for j, u in enumerate(tsgroup.index):
        t = tsgroup[u].t
        counts[:, j] = np.searchsorted(t, stops, "right") - np.searchsorted(t, starts, "left")
    return counts / dur[:, None], counts


def _design(labels, values):
    """Trial-averaging matrix D with D[i, k] = 1/n_k if trial i had value k."""
    D = np.zeros((len(labels), len(values)))
    for k, v in enumerate(values):
        m = labels == v
        D[m, k] = 1.0 / m.sum()
    return D


def condition_means(rates, labels, values):
    """Mean and SEM rate per stimulus value -> (n_values, n_units) each."""
    m = np.stack([rates[labels == v].mean(0) for v in values])
    s = np.stack(
        [rates[labels == v].std(0, ddof=1) / np.sqrt((labels == v).sum()) for v in values]
    )
    return m, s


def selectivity_table(rates, labels, values, kind="direction", blocks=None, n_perm=N_PERM, seed=0):
    """Per-unit selectivity metrics with a label-shuffle null.

    kind="direction": labels are 0..315 drift directions; OSI uses 2*theta, DSI theta.
    kind="orientation": labels are 0..150 static orientations; angles are doubled
        internally so the same circular-variance formula applies.
    """
    rng = np.random.default_rng(seed)
    n_units = rates.shape[1]
    D = _design(labels, values)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        tc = D.T @ rates  # (n_val, n_units) mean rate per stimulus value

    # Doubling the stimulus angle maps orientation onto the full circle, so the
    # same resultant-vector formula serves drift directions and static orientations.
    v2 = np.exp(1j * 2.0 * np.deg2rad(values))
    tot = tc.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        z2 = (tc * v2[:, None]).sum(0) / tot
    osi = np.abs(z2)
    pref_ori = (0.5 * np.rad2deg(np.angle(z2))) % 180.0
    if kind == "direction":
        v1 = np.exp(1j * np.deg2rad(values))
        with np.errstate(invalid="ignore", divide="ignore"):
            z1 = (tc * v1[:, None]).sum(0) / tot
        dsi = np.abs(z1)
        pref_dir = np.rad2deg(np.angle(z1)) % 360.0
    else:
        dsi = np.full(n_units, np.nan)
        pref_dir = np.full(n_units, np.nan)

    # classic ratio index: (R_pref - R_orth) / (R_pref + R_orth) on the orientation curve
    if kind == "direction":
        ori_vals = np.array([0.0, 45.0, 90.0, 135.0])
        ori_tc = np.stack(
            [tc[np.isin(values, [o, o + 180.0])].mean(0) for o in ori_vals]
        )
    else:
        ori_vals = values
        ori_tc = tc
    ipref = np.argmax(ori_tc, axis=0)
    n_ori = len(ori_vals)
    iorth = (ipref + n_ori // 2) % n_ori
    rp = ori_tc[ipref, np.arange(n_units)]
    ro = ori_tc[iorth, np.arange(n_units)]
    with np.errstate(invalid="ignore", divide="ignore"):
        osi_ratio = (rp - ro) / (rp + ro)

    # one-way ANOVA across stimulus values
    groups = [rates[labels == v] for v in values]
    fstat, pval_anova = stats.f_oneway(*groups, axis=0)

    # Permutation null on OSI. Labels are shuffled *within* each stimulus block so
    # that slow drift in firing rate across the ~2.5 h session cannot masquerade as
    # tuning: the null preserves each block's rate level.
    if blocks is None:
        blocks = np.zeros(len(labels))
    block_idx = [np.where(blocks == b)[0] for b in np.unique(blocks)]
    null = np.zeros((n_perm, n_units))
    for k in range(n_perm):
        perm = np.arange(len(labels))
        for idx in block_idx:
            perm[idx] = rng.permutation(idx)
        with np.errstate(invalid="ignore", divide="ignore"):
            tcp = D[perm].T @ rates
            null[k] = np.abs((tcp * v2[:, None]).sum(0) / tcp.sum(0))
    pval_perm = (1.0 + (null >= osi[None, :]).sum(0)) / (n_perm + 1.0)
    osi_null_median = np.nanmedian(null, axis=0)

    out = pd.DataFrame(
        dict(
            osi=osi,
            dsi=dsi,
            osi_ratio=osi_ratio,
            pref_ori=pref_ori,
            pref_dir=pref_dir,
            mean_rate=rates.mean(0),
            max_rate=tc.max(0),
            p_anova=pval_anova,
            p_perm=pval_perm,
            osi_null_median=osi_null_median,
        ),
        index=list(tc_index(rates)),
    )
    return out, tc


def tc_index(rates):
    return range(rates.shape[1])


def split_half_reliability(rates, labels, values, kind, n_rep=50, seed=1):
    """Correlation between tuning curves from independent random trial halves."""
    rng = np.random.default_rng(seed)
    n_units = rates.shape[1]
    r = np.zeros((n_rep, n_units))
    idx_by_val = [np.where(labels == v)[0] for v in values]
    for k in range(n_rep):
        a, b = [], []
        for idx in idx_by_val:
            p = rng.permutation(idx)
            h = len(p) // 2
            a.append(rates[p[:h]].mean(0))
            b.append(rates[p[h : 2 * h]].mean(0))
        A, B = np.stack(a), np.stack(b)
        A = A - A.mean(0)
        B = B - B.mean(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            r[k] = (A * B).sum(0) / np.sqrt((A**2).sum(0) * (B**2).sum(0))
    return np.nanmean(r, axis=0)


def analyze_session(url, verbose=True):
    """Full per-session pipeline. Returns a dict of results."""
    nwbfile = open_nwb(url)
    tsg, meta = load_units(nwbfile)
    keep = passes_qc(meta)
    tsg = tsg[list(meta.index[keep])]
    meta = meta[keep].copy()

    dg = _clean_trials(load_stim_table(nwbfile, "drifting_gratings_presentations"), DIRECTIONS)
    sg = _clean_trials(load_stim_table(nwbfile, "static_gratings_presentations"), STATIC_ORIS)

    dg_rates, dg_counts = trial_rate_matrix(tsg, dg)
    sg_rates, sg_counts = trial_rate_matrix(tsg, sg)

    dg_labels = dg["orientation"].values
    sg_labels = sg["orientation"].values

    tab_dg, tc_dg = selectivity_table(
        dg_rates, dg_labels, DIRECTIONS, kind="direction", blocks=dg["stimulus_block"].values
    )
    tab_sg, tc_sg = selectivity_table(
        sg_rates, sg_labels, STATIC_ORIS, kind="orientation", blocks=sg["stimulus_block"].values
    )

    tab_dg.index = meta.index
    tab_sg.index = meta.index
    tab_dg["reliability"] = split_half_reliability(dg_rates, dg_labels, DIRECTIONS, "direction")
    tab_sg["reliability"] = split_half_reliability(sg_rates, sg_labels, STATIC_ORIS, "orientation")

    res = pd.concat(
        [
            meta[["area", "session_id", "firing_rate", "snr", "waveform_duration"]],
            tab_dg.add_prefix("dg_"),
            tab_sg.add_prefix("sg_"),
        ],
        axis=1,
    )
    res["ori_agreement_deg"] = circ_dist_ori(res["dg_pref_ori"], res["sg_pref_ori"])

    if verbose:
        print(
            f"session {meta['session_id'].iloc[0]}: {len(meta)} QC units, "
            f"{len(dg)} drifting-grating trials, {len(sg)} static-grating trials"
        )

    return dict(
        session_id=str(nwbfile.session_id),
        units=res,
        tc_dg=tc_dg,  # (8 directions, n_units) mean rate
        tc_sg=tc_sg,  # (6 orientations, n_units)
        dg_rates=dg_rates,
        dg_labels=dg_labels,
        dg_tf=dg["temporal_frequency"].values,
        sg_rates=sg_rates,
        sg_labels=sg_labels,
        dg_table=dg,
        tsgroup=tsg,
        nwbfile=nwbfile,
    )

# %% [markdown]
# ## 3. Prototype on a single session
#
# Before scaling up, run the whole pipeline on one session and look at the raw
# spikes.

# %%
assets = get_assets()
print(f"{len(assets)} session-level assets in dandiset {DANDISET}")
first = assets[0]
print(first["path"], round(first["size"] / 1e9, 2), "GB")

nwbfile = open_nwb(first["url"])
print("session", nwbfile.session_id, "|", nwbfile.session_description)
print("subject:", nwbfile.subject.subject_id, nwbfile.subject.genotype,
      nwbfile.subject.age, nwbfile.subject.sex)
for k, v in nwbfile.intervals.items():
    print(f"  {k:42s} {len(v):6d} presentations")

# %%
tsg_all, meta_all = load_units(nwbfile)
qc = passes_qc(meta_all)
print(f"{len(meta_all)} sorted units, {qc.sum()} pass quality control")
print(meta_all[qc].area.value_counts().head(12))

# %%
dg_preview = _clean_trials(
    load_stim_table(nwbfile, "drifting_gratings_presentations"), DIRECTIONS)
print(dg_preview.groupby("orientation").size())
print("presentation duration:",
      np.round(np.median(dg_preview.stop_time - dg_preview.start_time), 3), "s")
print("blocks:", sorted(dg_preview.stimulus_block.unique()))

# %% [markdown]
# ### Figures
#
# The plotting code for the whole notebook is defined in one place below.

# %%
import numpy as np
import pandas as pd
import matplotlib

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pynapple as nap


plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 160,
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 10,
        "legend.frameon": False,
    }
)

ORI_COLORS = plt.cm.hsv(np.linspace(0, 1, 9))[:8]


# --------------------------------------------------------------------------- helpers


def perievent_spikes(ts, starts, window=(-0.5, 2.5)):
    """List of relative spike-time arrays, one per trial."""
    t = ts.t
    out = []
    for s in starts:
        lo = np.searchsorted(t, s + window[0], "left")
        hi = np.searchsorted(t, s + window[1], "right")
        out.append(t[lo:hi] - s)
    return out


def psth(ts, starts, window=(-0.5, 2.5), bin_size=0.025):
    edges = np.arange(window[0], window[1] + bin_size, bin_size)
    t = ts.t
    counts = np.zeros(len(edges) - 1)
    for s in starts:
        lo = np.searchsorted(t, s + window[0], "left")
        hi = np.searchsorted(t, s + window[1], "right")
        counts += np.histogram(t[lo:hi] - s, edges)[0]
    return edges[:-1] + bin_size / 2, counts / (len(starts) * bin_size)


# ----------------------------------------------------------------------- figure 1


def fig_raw_data(res, running, fname="fig01_raw_data.png"):
    """Raw spiking during drifting gratings: population raster + example unit + behaviour."""
    u = res["units"]
    tsg = res["tsgroup"]
    dg = res["dg_table"]

    # a 30 s window at the start of the first drifting-gratings block
    t0 = dg["start_time"].iloc[0] - 2
    t1 = t0 + 32
    win = nap.IntervalSet(start=t0, end=t1)
    sub = dg[(dg.start_time >= t0) & (dg.stop_time <= t1)]

    # example unit: well driven, orientation-tuned, and with a legible preferred /
    # orthogonal rate contrast inside the specific window being plotted
    vis = u[u.area.isin(VISUAL_CORTEX) & (u.dg_max_rate > 15) & (u.dg_osi > 0.4)]
    best, best_ratio = None, -np.inf
    for uid in vis.index:
        d = circ_dist_ori(sub.orientation.values, u.loc[uid, "dg_pref_ori"])
        pref_ep = nap.IntervalSet(sub.start_time.values[d < 25], sub.stop_time.values[d < 25])
        orth_ep = nap.IntervalSet(sub.start_time.values[d > 65], sub.stop_time.values[d > 65])
        if len(pref_ep) == 0 or len(orth_ep) == 0:
            continue
        rp = len(tsg[uid].restrict(pref_ep)) / pref_ep.tot_length()
        ro = len(tsg[uid].restrict(orth_ep)) / orth_ep.tot_length()
        ratio = rp / (ro + 1.0)
        if rp > 5 and ratio > best_ratio:
            best, best_ratio = uid, ratio
    example = best

    order = u[u.area.isin(VISUAL_CORTEX + CONTROL)].sort_values("area")
    fig = plt.figure(figsize=(12, 8.5))
    gs = GridSpec(4, 1, height_ratios=[0.35, 2.0, 1.0, 0.8], hspace=0.45)

    # stimulus ribbon
    ax0 = fig.add_subplot(gs[0])
    for _, r in sub.iterrows():
        c = ORI_COLORS[int(np.where(DIRECTIONS == r.orientation)[0][0])]
        ax0.axvspan(r.start_time, r.stop_time, color=c, alpha=0.9)
        ax0.text(
            (r.start_time + r.stop_time) / 2,
            0.5,
            f"{int(r.orientation)}",
            ha="center",
            va="center",
            fontsize=6,
            rotation=90,
            color="w",
        )
    ax0.set_xlim(t0, t1)
    ax0.set_yticks([])
    ax0.set_title("Drifting-grating presentations (colour/number = drift direction, deg)")
    ax0.tick_params(labelbottom=False)

    # population raster
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ypos, ylabels, yticks = 0, [], []
    for area, grp in order.groupby("area", sort=False):
        start = ypos
        for uid in grp.index:
            t = tsg[uid].restrict(win).t
            ax1.plot(t, np.full_like(t, ypos), "|", ms=1.6, lw=0.3, alpha=0.85,
                     color="tab:blue" if area in VISUAL_CORTEX else "0.55")
            ypos += 1
        yticks.append((start + ypos) / 2)
        ylabels.append(area)
        ax1.axhline(ypos, color="k", lw=0.4, alpha=0.3)
    for _, r in sub.iterrows():
        ax1.axvspan(r.start_time, r.stop_time, color="k", alpha=0.045, lw=0)
    ax1.set_yticks(yticks)
    ax1.set_yticklabels(ylabels, fontsize=7)
    ax1.set_ylim(0, ypos)
    ax1.set_ylabel("unit (grouped by area)")
    ax1.set_title("Population raster: visual cortex (blue) vs hippocampus (grey)")
    ax1.tick_params(labelbottom=False)

    # example unit raster
    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    t = tsg[example].restrict(win).t
    ax2.plot(t, np.zeros_like(t), "|", ms=14, color="k")
    for _, r in sub.iterrows():
        c = ORI_COLORS[int(np.where(DIRECTIONS == r.orientation)[0][0])]
        ax2.axvspan(r.start_time, r.stop_time, color=c, alpha=0.25, lw=0)
    ax2.set_yticks([])
    ax2.set_ylim(-1, 1)
    ax2.set_title(
        f"Example unit {example} ({u.loc[example,'area']}), "
        f"OSI={u.loc[example,'dg_osi']:.2f}, preferred orientation "
        f"{u.loc[example,'dg_pref_ori']:.0f}°"
    )
    ax2.tick_params(labelbottom=False)

    # running speed
    ax3 = fig.add_subplot(gs[3], sharex=ax0)
    if running is not None:
        rs = running.restrict(win)
        ax3.plot(rs.t, rs.d, lw=0.7, color="tab:green")
        ax3.set_ylabel("running\n(cm/s)")
    ax3.set_xlabel("time in session (s)")
    ax3.set_xlim(t0, t1)
    ax3.tick_params(labelbottom=True)

    fig.suptitle(
        f"DANDI:000021 session {res['session_id']}: raw spiking during drifting gratings",
        y=0.985,
    )
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    return fname


# ----------------------------------------------------------------------- figure 2


def fig_example_units(res, n=6, fname="fig02_example_units.png"):
    """PSTH-by-direction and polar tuning curve for the most selective visual units."""
    u = res["units"]
    tsg = res["tsgroup"]
    dg = res["dg_table"]
    tc = res["tc_dg"]
    unit_pos = {uid: i for i, uid in enumerate(u.index)}

    vis = u[u.area.isin(VISUAL_CORTEX) & (u.dg_mean_rate > 1.0)]
    picks = vis.sort_values("dg_osi", ascending=False).index[:n]

    fig, axes = plt.subplots(2, n, figsize=(2.5 * n, 6.2),
                             subplot_kw=None, gridspec_kw=dict(height_ratios=[1.3, 1]))
    for k, uid in enumerate(picks):
        ax = axes[0, k]
        for i, d in enumerate(DIRECTIONS):
            starts = dg.loc[dg.orientation == d, "start_time"].values
            x, y = psth(tsg[uid], starts)
            ax.plot(x, y, color=ORI_COLORS[i], lw=1.0, label=f"{int(d)}°")
        ax.axvspan(0, 2, color="0.9", zorder=-5)
        ax.set_title(f"unit {uid}\n{u.loc[uid,'area']}  OSI={u.loc[uid,'dg_osi']:.2f}", pad=6)
        ax.set_xlabel("time from onset (s)")
        if k == 0:
            ax.set_ylabel("firing rate (Hz)")
        if k == 0:
            handles, labels = ax.get_legend_handles_labels()

    for k, uid in enumerate(picks):
        axes[1, k].remove()
        ax = fig.add_subplot(2, n, n + k + 1, projection="polar")
        r = tc[:, unit_pos[uid]]
        th = np.deg2rad(np.append(DIRECTIONS, DIRECTIONS[0]))
        ax.plot(th, np.append(r, r[0]), "o-", color="tab:red", ms=3, lw=1.2)
        ax.fill(th, np.append(r, r[0]), color="tab:red", alpha=0.15)
        po = np.deg2rad(u.loc[uid, "dg_pref_ori"])
        ax.plot([po, po + np.pi], [r.max()] * 2, "--", color="k", lw=1.0)
        ax.set_theta_zero_location("E")
        ax.set_rlabel_position(105)
        ax.set_yticklabels([])
        ax.tick_params(labelsize=6, pad=0)
        ax.set_title(f"pref ori {u.loc[uid,'dg_pref_ori']:.0f}°\nDSI={u.loc[uid,'dg_dsi']:.2f}",
                     fontsize=8, pad=14)

    fig.legend(handles, labels, ncol=8, fontsize=8, loc="upper center",
               bbox_to_anchor=(0.5, 0.955), title="drift direction", title_fontsize=8)
    fig.suptitle(
        "Direction tuning of the most orientation-selective visual-cortex units "
        f"(session {res['session_id']})",
        y=1.01,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90], h_pad=3.0, w_pad=1.6)
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    return fname


# ----------------------------------------------------------------------- figure 3


def fig_population_tuning(pool, fname="fig03_population_tuning.png"):
    """Normalised tuning curves for every unit, visual cortex vs hippocampal control."""
    u = pool["units"]
    tc = pool["tc_dg"]  # (8, n_units) aligned with u

    def block(mask, ax_h, ax_m, title):
        idx = np.where(mask)[0]
        M = tc[:, idx].T.copy()
        base = M.min(1, keepdims=True)
        rng_ = M.max(1, keepdims=True) - base
        rng_[rng_ == 0] = np.nan
        Mn = (M - base) / rng_
        peak = np.argmax(Mn, axis=1)
        # sort by peak direction, then by how sharply the response falls off, so the
        # preferred direction forms a diagonal and its 180 deg mirror is visible
        second = np.array([Mn[i, (peak[i] + 4) % 8] for i in range(len(idx))])
        order = np.lexsort((second, peak))
        im = ax_h.imshow(
            Mn[order], aspect="auto", cmap="magma", vmin=0, vmax=1,
            extent=[-22.5, 337.5, len(idx), 0], interpolation="nearest",
        )
        ax_h.set_xticks(DIRECTIONS)
        ax_h.set_xlabel("drift direction (deg)")
        ax_h.set_ylabel("unit (sorted by preferred direction)")
        ax_h.set_title(f"{title}\nn = {len(idx)} units")
        # tuning curves rotated so each unit's preferred direction sits at 0
        rolled = np.stack([np.roll(Mn[i], -peak[i]) for i in range(len(idx))])
        rolled = np.concatenate([rolled, rolled[:, :1]], axis=1)
        xx = np.arange(9) * 45
        m = np.nanmean(rolled, 0)
        se = np.nanstd(rolled, 0) / np.sqrt(np.isfinite(rolled).sum(0))
        ax_m.plot(xx, m, "o-", color="tab:red", ms=4)
        ax_m.fill_between(xx, m - se, m + se, color="tab:red", alpha=0.3)
        ax_m.axvline(180, color="0.5", ls="--", lw=1)
        ax_m.set_xticks(np.arange(0, 361, 90))
        ax_m.set_xlabel("direction relative to preferred (deg)")
        ax_m.set_ylabel("normalised rate")
        ax_m.set_ylim(0, 1.05)
        ax_m.set_title("mean tuning curve aligned to preferred direction\n"
                       "(dashed line: same orientation, opposite direction)", fontsize=9)
        return im

    fig = plt.figure(figsize=(12.5, 8.6))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.15, 1.0],
                  hspace=0.42, wspace=0.38)
    axes = [[fig.add_subplot(gs[r, 0]), fig.add_subplot(gs[r, 1])] for r in range(2)]

    vis = u["area"].isin(VISUAL_CORTEX).values & (u["dg_p_perm"].values < 0.05)
    ctl = u["area"].isin(CONTROL).values
    im = block(vis, axes[0][0], axes[0][1],
               "Visual cortex, orientation-selective units (p < 0.05)")
    im2 = block(ctl, axes[1][0], axes[1][1], "Hippocampus (CA1/CA3/DG), all units")
    # one colorbar per heatmap, taken out of that heatmap's own space, so it can
    # never collide with the y-axis of the line plot in the right-hand column
    for imk, ax_h in ((im, axes[0][0]), (im2, axes[1][0])):
        cb = fig.colorbar(imk, ax=ax_h, fraction=0.045, pad=0.03)
        cb.set_label("normalised rate", fontsize=8)
        cb.ax.tick_params(labelsize=7)
    fig.suptitle(
        "Population tuning: orientation structure is present in cortex, absent in hippocampus",
        y=0.955)
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    return fname


# ----------------------------------------------------------------------- figure 4


def fig_osi_by_area(pool, fname="fig04_osi_by_area.png"):
    u = pool["units"]
    areas = [a for a in VISUAL_CORTEX + THALAMUS + CONTROL
             if (u["area"] == a).sum() >= 20]
    data = [u.loc[u.area == a, "dg_osi"].dropna().values for a in areas]
    null = [u.loc[u.area == a, "dg_osi_null_median"].dropna().values for a in areas]
    frac = np.array([(u.loc[u.area == a, "dg_p_perm"] < 0.05).mean() for a in areas])
    ns = np.array([len(d) for d in data])
    colors = ["tab:blue" if a in VISUAL_CORTEX else
              "tab:orange" if a in THALAMUS else "0.5" for a in areas]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.0))

    ax = axes[0]
    parts = ax.violinplot(data, showextrema=False, widths=0.85)
    for pc, c in zip(parts["bodies"], colors):
        pc.set_facecolor(c)
        pc.set_alpha(0.55)
    ax.boxplot(data, widths=0.16, showfliers=False,
               medianprops=dict(color="k", lw=1.4), whiskerprops=dict(lw=0.8))
    ax.plot(np.arange(1, len(areas) + 1), [np.median(n) for n in null], "kv", ms=5,
            label="median shuffled null")
    ax.set_xticks(np.arange(1, len(areas) + 1))
    # two-line labels at a shallow rotation ran into each other; 55 deg with a
    # right anchor keeps eleven of them separated
    ax.set_xticklabels([f"{a} (n={n})" for a, n in zip(areas, ns)], fontsize=7.5,
                       rotation=55, ha="right", rotation_mode="anchor")
    ax.set_ylabel("orientation selectivity index (OSI)")
    ax.set_title("OSI by recorded structure")
    ax.legend(fontsize=8)
    ax.set_ylim(-0.02, 1.0)

    ax = axes[1]
    ax.bar(np.arange(len(areas)), frac, color=colors, alpha=0.8)
    ax.axhline(0.05, ls="--", color="k", lw=1, label="chance (α = 0.05)")
    ax.set_xticks(np.arange(len(areas)))
    ax.set_xticklabels(areas, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("fraction of units")
    ax.set_title("Units with significant orientation tuning\n(within-block label shuffle, p < 0.05)")
    ax.legend(fontsize=8)

    ax = axes[2]
    for a, c in [("VISp", "tab:blue"), ("VISl", "tab:cyan"), ("VISal", "tab:purple"),
                 ("LP", "tab:orange"), ("CA1", "0.4"), ("DG", "0.65")]:
        d = u.loc[u.area == a, "dg_osi"].dropna().values
        if len(d) < 20:
            continue
        ax.plot(np.sort(d), np.linspace(0, 1, len(d)), color=c, lw=1.8, label=f"{a} (n={len(d)})")
    nulld = u.loc[u.area.isin(VISUAL_CORTEX), "dg_osi_null_median"].dropna().values
    ax.plot(np.sort(nulld), np.linspace(0, 1, len(nulld)), color="k", ls=":", lw=1.5,
            label="shuffled null")
    ax.set_xlabel("OSI")
    ax.set_ylabel("cumulative fraction of units")
    ax.set_title("Cumulative OSI distributions")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.set_xlim(0, 0.9)

    fig.suptitle(
        f"Orientation selectivity across {pool['n_sessions']} sessions "
        f"({len(u)} quality-passing units)", y=1.01)
    fig.tight_layout(w_pad=2.5)
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    return fname


# ----------------------------------------------------------------------- figure 5


def fig_cross_stimulus(pool, fname="fig05_cross_stimulus.png", seed=0):
    """Preferred orientation measured with drifting gratings vs static gratings."""
    u = pool["units"]
    m = (u.area.isin(VISUAL_CORTEX) & (u.dg_p_perm < 0.05) & (u.sg_p_perm < 0.05))
    d = u[m]
    rng = np.random.default_rng(seed)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))

    ax = axes[0]
    ax.plot(d.dg_pref_ori, d.sg_pref_ori, "o", ms=4, alpha=0.45, color="tab:blue")
    for off in (-180, 0, 180):
        ax.plot([0, 180], [off, off + 180], "k--", lw=1)
    ax.set_xlabel("preferred orientation, drifting gratings (deg)")
    ax.set_ylabel("preferred orientation, static gratings (deg)")
    ax.set_title(f"Cross-stimulus agreement\nn = {len(d)} visual-cortex units")
    ax.set_xlim(0, 180)
    ax.set_ylim(0, 180)
    ax.set_xticks(np.arange(0, 181, 45))
    ax.set_yticks(np.arange(0, 181, 45))

    ax = axes[1]
    obs = circ_dist_ori(d.dg_pref_ori.values, d.sg_pref_ori.values)
    shuf = circ_dist_ori(d.dg_pref_ori.values,
                           rng.permutation(d.sg_pref_ori.values))
    bins = np.arange(0, 91, 7.5)
    ax.hist(obs, bins=bins, density=True, alpha=0.7, color="tab:blue", label="observed")
    ax.hist(shuf, bins=bins, density=True, histtype="step", lw=2, color="k",
            label="shuffled pairing")
    ax.axhline(1 / 90, ls=":", color="0.4", lw=1)
    ax.set_xlabel("|Δ preferred orientation| (deg)")
    ax.set_ylabel("density")
    ax.set_title(f"median |Δ| = {np.median(obs):.1f}° vs {np.median(shuf):.1f}° shuffled")
    ax.legend(fontsize=8)

    ax = axes[2]
    v = u[u.area.isin(VISUAL_CORTEX)]
    ax.plot(v.dg_osi, v.sg_osi, "o", ms=3, alpha=0.3, color="0.35")
    r = np.corrcoef(v.dg_osi.dropna(), v.sg_osi.dropna())[0, 1] if len(v) else np.nan
    ax.set_xlabel("OSI, drifting gratings")
    ax.set_ylabel("OSI, static gratings")
    ax.set_title(f"OSI consistency across stimulus classes\nPearson r = {r:.2f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    fig.suptitle("Orientation preference replicates across two independent stimulus classes", y=1.02)
    fig.tight_layout(w_pad=2.5)
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    return fname


# ----------------------------------------------------------------------- figure 6


def fig_decoding(dec, fname="fig06_decoding.png"):
    """Population decoding of grating direction / orientation, by brain structure."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    palette = {"VISp": "tab:blue", "VISl": "tab:cyan", "VISal": "tab:purple",
               "VISrl": "tab:green", "VISam": "tab:olive", "LP": "tab:orange",
               "CA1": "0.45"}

    for ax, task, chance in ((axes[0], "direction", 1 / 8), (axes[1], "orientation", 1 / 4)):
        d = dec[(dec.task == task) & (dec.kind == "observed")]
        for area, g in d.groupby("area"):
            m = g.groupby("n_units").acc.agg(["mean", "sem", "size"])
            ax.errorbar(m.index, m["mean"], yerr=m["sem"], marker="o", ms=4, lw=1.5,
                        capsize=2, color=palette.get(area, "k"),
                        label=f"{area} ({int(m['size'].max())} sess.)")
        sh = dec[(dec.task == task) & (dec.kind == "shuffled")].groupby("n_units").acc.mean()
        ax.axhline(chance, color="0.6", ls="--", lw=1)
        ax.set_xscale("log")
        ax.set_xticks(sorted(d.n_units.unique()))
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel("number of simultaneously recorded units")
        ax.set_ylabel("cross-validated accuracy")
        ax.plot(sh.index, sh.values, "k:", lw=1.5, label="label-shuffled")
        ax.set_title(f"{task.capitalize()} decoding "
                     f"({'8-way' if task=='direction' else '4-way'}; chance {chance:.2f})")
        ax.legend(fontsize=7, loc="upper left")

    ax = axes[2]
    # use the largest population size that most structures actually reach
    counts = dec[dec.task == "orientation"].groupby("n_units").area.nunique()
    nsel = int(counts[counts >= counts.max()].index.max())
    d = dec[(dec.task == "orientation") & (dec.n_units == nsel)]
    obs = d[d.kind == "observed"].groupby("area").acc.agg(["mean", "sem"])
    shf = d[d.kind == "shuffled"].groupby("area").acc.mean()
    areas = obs.index.tolist()
    x = np.arange(len(areas))
    ax.bar(x - 0.2, obs["mean"], 0.4, yerr=obs["sem"], capsize=3,
           color=[palette.get(a, "k") for a in areas], label="observed")
    ax.bar(x + 0.2, shf.reindex(areas), 0.4, color="0.8", label="label-shuffled")
    ax.axhline(0.25, color="k", ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(areas, rotation=45, ha="right")
    ax.set_ylabel("accuracy")
    ax.set_title(f"Orientation decoding at {nsel} units")
    ax.legend(fontsize=8)

    fig.suptitle("Grating orientation is linearly decodable from visual-cortex populations", y=1.02)
    fig.tight_layout(w_pad=2.5)
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    return fname


# ----------------------------------------------------------------------- figure 7


def fig_glm(glm_df, examples, fname="fig07_glm.png"):
    """NeMoS Poisson-GLM encoding results."""

    fig = plt.figure(figsize=(14, 7.6))
    gs = GridSpec(2, 4, figure=fig, hspace=0.55, wspace=0.35)

    for k, ex in enumerate(examples[:4]):
        ax = fig.add_subplot(gs[0, k])
        ax.errorbar(DIRECTIONS, ex["emp"], yerr=ex["sem"], fmt="o", ms=4, color="0.3",
                    capsize=2, label="observed")
        ax.plot(ex["grid"], ex["fit"], "-", color="tab:red", lw=1.8, label="GLM fit")
        ax.set_xticks(np.arange(0, 361, 90))
        ax.set_xlabel("drift direction (deg)")
        if k == 0:
            ax.set_ylabel("firing rate (Hz)")
            ax.legend(fontsize=7)
        ax.set_title(f"unit {ex['unit_id']} ({ex['area']})\n"
                     f"$\\Delta$LL$_{{dir|speed}}$ = {ex['d']:.2f} nats/trial", fontsize=9)

    areas = [a for a in VISUAL_CORTEX + THALAMUS + CONTROL
             if (glm_df.area == a).sum() >= 20]
    colors = ["tab:blue" if a in VISUAL_CORTEX else
              "tab:orange" if a in THALAMUS else "0.5" for a in areas]

    ax = fig.add_subplot(gs[1, :2])
    data = [glm_df.loc[glm_df.area == a, "d_dir_given_speed"].values for a in areas]
    parts = ax.violinplot(data, showextrema=False, widths=0.85)
    for pc, c in zip(parts["bodies"], colors):
        pc.set_facecolor(c)
        pc.set_alpha(0.55)
    ax.boxplot(data, widths=0.15, showfliers=False, medianprops=dict(color="k", lw=1.3))
    ax.axhline(0, color="k", lw=1, ls="--")
    ax.set_xticks(np.arange(1, len(areas) + 1))
    ax.set_xticklabels([f"{a}\n(n={len(d)})" for a, d in zip(areas, data)], fontsize=7.5)
    ax.set_ylabel("held-out $\\Delta$LL (nats/trial)")
    ax.set_yscale("symlog", linthresh=0.01)
    ax.set_title("Cross-validated gain from adding drift direction to a running-speed-only model")

    ax = fig.add_subplot(gs[1, 2:])
    frac_dir = np.array([(glm_df.loc[glm_df.area == a, "d_dir_given_speed"] > 0).mean()
                         for a in areas])
    frac_spd = np.array([(glm_df.loc[glm_df.area == a, "d_speed"] > 0).mean() for a in areas])
    x = np.arange(len(areas))
    ax.bar(x - 0.2, frac_dir, 0.4, color=colors, alpha=0.9, label="direction | speed")
    ax.bar(x + 0.2, frac_spd, 0.4, color="0.75", label="running speed alone")
    ax.axhline(0.5, color="k", ls=":", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(areas, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("fraction of units with $\\Delta$LL > 0")
    ax.set_title("Units whose held-out likelihood improves")
    ax.legend(fontsize=8)

    fig.suptitle("Poisson GLM (NeMoS): grating direction predicts spiking beyond locomotor state",
                 y=0.99)
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    return fname


# ----------------------------------------------------------------------- figure 8


def fig_summary(pool_, fname="fig08_summary.png"):
    """Preferred-orientation distribution, OSI vs DSI, and cell-type breakdown."""

    u = pool_["units"]
    v = u[u.area.isin(VISUAL_CORTEX) & (u.dg_p_perm < 0.05)]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))

    ax = axes[0]
    bins = np.arange(0, 181, 15)
    ax.hist(v.dg_pref_ori, bins=bins, color="tab:blue", alpha=0.8, edgecolor="w")
    ax.axhline(len(v) / (len(bins) - 1), color="k", ls="--", lw=1, label="uniform")
    for c in (0, 90, 180):
        ax.axvline(c, color="tab:red", ls=":", lw=1.2)
    ax.plot([], [], color="tab:red", ls=":", lw=1.2, label="cardinal (0/90 deg)")
    ax.set_xticks(np.arange(0, 181, 45))
    ax.set_xlabel("preferred orientation (deg)")
    ax.set_ylabel("number of units")
    ax.set_title(f"Preferred orientations, visual cortex\n(n = {len(v)} tuned units)")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(v.dg_osi, v.dg_dsi, "o", ms=3, alpha=0.35, color="tab:blue")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("OSI (orientation selectivity)")
    ax.set_ylabel("DSI (direction selectivity)")
    ax.set_title("Most tuned units are orientation-\nselective but not direction-selective")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax = axes[2]
    narrow = v.waveform_duration < 0.4
    for m, lab, c in ((narrow, "narrow-waveform", "tab:red"),
                      (~narrow, "broad-waveform", "tab:blue")):
        d = v.loc[m, "dg_osi"].dropna().values
        ax.plot(np.sort(d), np.linspace(0, 1, len(d)), color=c, lw=2,
                label=f"{lab} (n={len(d)})")
    ax.set_xlabel("OSI")
    ax.set_ylabel("cumulative fraction")
    ax.set_title("Waveform class")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(0, 0.9)

    fig.suptitle("Properties of the orientation-tuned population", y=1.02)
    fig.tight_layout(w_pad=2.5)
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    return fname

# %%
proto = analyze_session(first["url"])
running = load_running_speed(proto["nwbfile"])
print(fig_raw_data(proto, running))

# %% [markdown]
# ![raw data](fig01_raw_data.png)
#
# The example unit fires in bursts confined to a subset of grating presentations
# while hippocampal units below fire continuously and without regard to the
# stimulus. The animal runs almost throughout this window, so the modulation is not
# simply locomotion.

# %%
print(fig_example_units(proto))

# %% [markdown]
# ![example units](fig02_example_units.png)
#
# Each of these units responds to *two* drift directions 180 degrees apart, which is
# the signature of orientation rather than direction selectivity: the polar curves
# are bilobed and the DSI values are small. Responses are sustained for the full 2 s
# presentation.

# %% [markdown]
# ## 4. Scale to eight sessions
#
# The same pipeline is run over the eight smallest session files of the dandiset
# (file size is unrelated to the biology; it mostly tracks recording duration and
# probe count). Results are cached to disk so the notebook can be re-executed
# cheaply.

# %%
def run_all_sessions(n=N_SESSIONS):
    for a in tqdm(get_assets()[:n], desc="sessions"):
        sid = a["path"].split("ses-")[1].replace(".nwb", "")
        fp = os.path.join(OUT, f"{sid}.pkl")
        if not os.path.exists(fp):
            res = analyze_session(a["url"])
            keep = {k: res[k] for k in ("session_id", "units", "tc_dg", "tc_sg",
                                        "dg_rates", "dg_labels", "dg_tf",
                                        "sg_rates", "sg_labels")}
            keep["dg_blocks"] = res["dg_table"]["stimulus_block"].values
            pickle.dump(keep, open(fp, "wb"))
        # per-trial running speed, used by the GLM below
        rfp = os.path.join(OUT, f"{sid}_running.npz")
        if not os.path.exists(rfp):
            nwbf = open_nwb(a["url"])
            run = load_running_speed(nwbf)
            dg = _clean_trials(
                load_stim_table(nwbf, "drifting_gratings_presentations"), DIRECTIONS)
            np.savez(rfp, speed=trial_mean_speed(
                run, dg["start_time"].values, dg["stop_time"].values))


def trial_mean_speed(running, starts, stops):
    t, v = running.t, running.d
    csum = np.concatenate([[0.0], np.cumsum(v)])
    lo = np.searchsorted(t, starts, "left")
    hi = np.searchsorted(t, stops, "right")
    return (csum[hi] - csum[lo]) / np.maximum(hi - lo, 1)


run_all_sessions()

# %%
import glob
import os
import pickle

import numpy as np
import pandas as pd


def load_sessions(folder="session_results"):
    out = []
    for fp in sorted(glob.glob(os.path.join(folder, "*.pkl"))):
        with open(fp, "rb") as fh:
            res = pickle.load(fh)
        sp_fp = fp.replace(".pkl", "_running.npz")
        if os.path.exists(sp_fp):
            res["speed"] = np.load(sp_fp)["speed"]
        out.append(res)
    return out


def pool_sessions(sessions):
    units = pd.concat([s["units"] for s in sessions])
    tc_dg = np.concatenate([s["tc_dg"] for s in sessions], axis=1)
    tc_sg = np.concatenate([s["tc_sg"] for s in sessions], axis=1)
    assert tc_dg.shape[1] == len(units)
    return dict(units=units.reset_index(), tc_dg=tc_dg, tc_sg=tc_sg,
                n_sessions=len(sessions))

# %%
sessions = load_sessions(OUT)[:N_SESSIONS]
pooled = pool_sessions(sessions)
units = pooled["units"]
print(f"{len(sessions)} sessions, {len(units)} quality-passing units")
print(units.area.value_counts().head(15))

# %% [markdown]
# ## 5. Is the tuning orientation tuning, and is it specific to visual areas?

# %%
print(fig_population_tuning(pooled))

# %% [markdown]
# ![population tuning](fig03_population_tuning.png)
#
# Sorting every significantly tuned cortical unit by its preferred direction
# produces the expected diagonal, and a clear second band appears 180 degrees away:
# these cells respond to both directions of motion along their preferred axis. The
# mean curve aligned to each unit's preferred direction has a second peak at 180
# degrees reaching about 70% of the primary peak. In hippocampus the diagonal is
# present by construction (rows are sorted by their own maximum) but there is no
# second band and the aligned curve is flat away from the trivially selected peak,
# which is what a curve made of noise looks like.

# %%
print(fig_osi_by_area(pooled))

# %% [markdown]
# ![OSI by area](fig04_osi_by_area.png)
#
# OSI in every visual cortical area sits far above its own shuffled null, and about
# 80% of cortical units are individually significant. Visual thalamus (LGd, LP) is
# intermediate, and the hippocampal subfields are close to the null. The residual
# above-chance fraction in CA1 is discussed at the end.

# %% [markdown]
# ## 6. Does the preference generalise to a different stimulus?
#
# The strongest evidence that a tuning curve reflects a real receptive-field
# property rather than a fluctuation is that it replicates on data the estimate was
# not taken from. Static gratings were presented in the same sessions: brief (0.25 s)
# stationary gratings at 6 orientations, with no motion at all. Preferred
# orientation is estimated independently from each stimulus class and the two
# estimates compared.

# %%
print(fig_cross_stimulus(pooled))

# %% [markdown]
# ![cross-stimulus](fig05_cross_stimulus.png)
#
# Preferred orientations line up along the identity (and its wrap-around), with a
# median absolute discrepancy far below the ~45 degrees expected if the two
# estimates were unrelated.

# %% [markdown]
# ## 7. Population decoding
#
# Single-unit selectivity should translate into decodable population information.
# A multinomial logistic decoder is trained on single-trial spike-count vectors
# from a random subset of simultaneously recorded units in one structure, and
# tested on held-out trials (5-fold cross-validation). Accuracy is reported against
# a label-shuffled null run through the identical pipeline.

# %%
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


DIRECTIONS = np.array([0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0])


def _clf():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=0.1),
    )


def decode(X, y, n_splits=5, seed=0, shuffle_null=False, rng=None):
    """Cross-validated accuracy of a multinomial logistic decoder."""
    if shuffle_null:
        rng = rng or np.random.default_rng(seed)
        y = rng.permutation(y)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    acc = []
    for tr, te in skf.split(X, y):
        clf = _clf()
        clf.fit(X[tr], y[tr])
        acc.append((clf.predict(X[te]) == y[te]).mean())
    return float(np.mean(acc))


def decode_vs_population_size(
    X, y, sizes, n_boot=8, seed=0, shuffle_null=False
):
    """Accuracy as a function of the number of randomly drawn units."""
    rng = np.random.default_rng(seed)
    out = []
    n_units = X.shape[1]
    for s in sizes:
        if s > n_units:
            break
        accs = []
        for b in range(n_boot):
            cols = rng.choice(n_units, s, replace=False)
            accs.append(
                decode(X[:, cols], y, seed=seed + b, shuffle_null=shuffle_null, rng=rng)
            )
        out.append((s, float(np.mean(accs)), float(np.std(accs))))
    return out


def run_decoding(sessions, sizes=(5, 10, 20, 40, 80), areas=("VISp", "VISl", "VISal",
                                                             "VISrl", "VISam", "LP", "CA1")):
    """Decode drift direction (8-way) and orientation (4-way) per area per session."""
    rows = []
    for res in tqdm(sessions, desc="decoding sessions"):
        u = res["units"]
        X_all = res["dg_rates"]
        y_dir = res["dg_labels"].astype(int)
        y_ori = (y_dir % 180).astype(int)
        for area in areas:
            cols = np.where(u["area"].values == area)[0]
            if len(cols) < 5:
                continue
            X = X_all[:, cols]
            X = X[:, X.std(0) > 0]  # constant units break the z-scoring step
            if X.shape[1] < 5:
                continue
            for task, y, chance in (("direction", y_dir, 1 / 8), ("orientation", y_ori, 1 / 4)):
                for s, m, sd in decode_vs_population_size(X, y, sizes):
                    rows.append(
                        dict(session=res["session_id"], area=area, task=task,
                             n_units=s, acc=m, sd=sd, chance=chance, kind="observed")
                    )
                for s, m, sd in decode_vs_population_size(X, y, sizes, shuffle_null=True):
                    rows.append(
                        dict(session=res["session_id"], area=area, task=task,
                             n_units=s, acc=m, sd=sd, chance=chance, kind="shuffled")
                    )
    return pd.DataFrame(rows)

# %%
dec = run_decoding(sessions)
dec.to_csv("decoding_results.csv", index=False)
print(fig_decoding(dec))

# %% [markdown]
# ![decoding](fig06_decoding.png)
#
# Forty simultaneously recorded V1 units are enough to identify which of eight drift
# directions was shown on a held-out trial with better than 80% accuracy. The same
# analysis in CA1 stays near chance.

# %% [markdown]
# ## 8. Encoding model: does orientation survive the running-speed confound?
#
# Locomotion increases firing rates throughout mouse visual cortex, so a nuisance
# explanation has to be ruled out: if running happened to co-vary with the stimulus
# sequence, a rate difference between orientations could be behavioural rather than
# visual. Four nested Poisson GLMs are fit to single-trial spike counts with NeMoS
# and compared by 5-fold cross-validated held-out log-likelihood:
#
# | model | predictors |
# |---|---|
# | M0 | intercept only |
# | M1 | running speed (B-spline basis) |
# | M2 | drift direction (cyclic B-spline basis over 0-360 deg) |
# | M3 | drift direction + running speed |
#
# The quantity of interest is `M3 - M1`: the held-out likelihood gained by adding
# grating direction to a model that already knows how fast the animal was running.
# All units of a session share the same design matrix, so they are fit together with
# `nmo.glm.PopulationGLM`.

# %%
import numpy as np
import pandas as pd
import nemos as nmo
from sklearn.model_selection import KFold

DIRECTIONS = np.array([0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0])
SOLVER = dict(solver_name="LBFGS", regularizer="Ridge", regularizer_strength=1e-3,
              solver_kwargs=dict(maxiter=2000, tol=1e-8))


def build_design(directions_deg, speed, n_dir_basis=6, n_speed_basis=4):
    """Feature matrices for the direction and running-speed terms."""
    dir_basis = nmo.basis.CyclicBSplineEval(n_basis_funcs=n_dir_basis, bounds=(0.0, 360.0))
    X_dir = np.asarray(dir_basis.compute_features(np.asarray(directions_deg, dtype=float)))

    s = np.asarray(speed, dtype=float)
    s = np.clip(s, np.nanpercentile(s, 1), np.nanpercentile(s, 99))
    spd_basis = nmo.basis.BSplineEval(n_basis_funcs=n_speed_basis,
                                      bounds=(float(s.min()), float(s.max())))
    X_spd = np.asarray(spd_basis.compute_features(s))
    return X_dir, X_spd, dir_basis


def _poisson_ll(y, rate):
    """Mean Poisson log-likelihood per trial (dropping the constant log y! term)."""
    rate = np.clip(np.asarray(rate), 1e-8, None)
    return np.mean(y * np.log(rate) - rate, axis=0)


def _cv_loglik_population(X, Y, n_splits=5, seed=0):
    """Held-out per-unit Poisson log-likelihood. X=None gives the intercept-only model."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    lls = []
    for tr, te in kf.split(Y):
        if X is None:
            mu = np.maximum(Y[tr].mean(0), 1e-8)
            lls.append(_poisson_ll(Y[te], np.broadcast_to(mu, Y[te].shape)))
        else:
            model = nmo.glm.PopulationGLM(**SOLVER)
            model.fit(X[tr], Y[tr])
            lls.append(_poisson_ll(Y[te], np.asarray(model.predict(X[te]))))
    return np.mean(np.stack(lls), axis=0)


def fit_session(res, speed, seed=0):
    """Cross-validated log-likelihoods of the four nested models, all units at once."""
    Y = np.rint(res["dg_rates"] * 2.0)  # rates -> counts over the 2 s presentation
    X_dir, X_spd, dir_basis = build_design(res["dg_labels"], speed)

    ll_null = _cv_loglik_population(None, Y, seed=seed)
    ll_spd = _cv_loglik_population(X_spd, Y, seed=seed)
    ll_dir = _cv_loglik_population(X_dir, Y, seed=seed)
    ll_full = _cv_loglik_population(np.hstack([X_dir, X_spd]), Y, seed=seed)

    df = pd.DataFrame(
        dict(
            unit_id=res["units"].index.values,
            area=res["units"]["area"].values,
            session_id=res["session_id"],
            mean_count=Y.mean(0),
            ll_null=ll_null,
            ll_speed=ll_spd,
            ll_dir=ll_dir,
            ll_full=ll_full,
            d_dir=ll_dir - ll_null,
            d_speed=ll_spd - ll_null,
            d_dir_given_speed=ll_full - ll_spd,
        )
    )
    return df, dir_basis


def predicted_tuning(Y, X_dir, dir_basis, n_grid=181):
    """Fit the direction-only population GLM on all trials; return smooth tuning curves."""
    model = nmo.glm.PopulationGLM(**SOLVER)
    model.fit(X_dir, np.rint(np.asarray(Y, dtype=float)))
    grid = np.linspace(0, 360, n_grid)
    Xg = np.asarray(dir_basis.compute_features(grid))
    rate = np.asarray(model.predict(Xg)) / 2.0  # counts per 2 s -> Hz
    return grid, rate

# %%
glm_rows, glm_examples = [], []
for res in tqdm(sessions, desc="GLM"):
    df, dir_basis = fit_session(res, res["speed"])
    glm_rows.append(df)
    if not glm_examples:
        Y = np.rint(res["dg_rates"] * 2.0)
        X_dir, _, _ = build_design(res["dg_labels"], res["speed"])
        grid, rate = predicted_tuning(Y, X_dir, dir_basis)
        cand = df[df.area.isin(VISUAL_CORTEX)].sort_values(
            "d_dir_given_speed", ascending=False)
        for j in cand.index[:4]:
            col = int(np.where(res["units"].index.values == df.loc[j, "unit_id"])[0][0])
            sem = np.array([res["dg_rates"][res["dg_labels"] == d, col].std(ddof=1)
                            / np.sqrt((res["dg_labels"] == d).sum())
                            for d in DIRECTIONS])
            glm_examples.append(dict(
                unit_id=df.loc[j, "unit_id"], area=df.loc[j, "area"],
                emp=res["tc_dg"][:, col], sem=sem, grid=grid, fit=rate[:, col],
                d=df.loc[j, "d_dir_given_speed"]))
glm_df = pd.concat(glm_rows, ignore_index=True)
glm_df.to_csv("glm_results.csv", index=False)
print(fig_glm(glm_df, glm_examples))

# %% [markdown]
# ![GLM](fig07_glm.png)
#
# Running speed does carry information about firing rate almost everywhere,
# including hippocampus. Adding drift direction on top of it still improves
# held-out likelihood for the large majority of visual cortical units and for
# almost none of the hippocampal ones, so the orientation signal is not a
# locomotion artefact.

# %% [markdown]
# ## 9. Properties of the tuned population

# %%
print(fig_summary(pooled))
units.to_csv("unit_metrics.csv", index=False)

# %% [markdown]
# ![summary](fig08_summary.png)

# %% [markdown]
# ## 10. Summary statistics

# %%
vis = units[units.area.isin(VISUAL_CORTEX)]
ctl = units[units.area.isin(CONTROL)]
mw = stats.mannwhitneyu(vis.dg_osi.dropna(), ctl.dg_osi.dropna(), alternative="greater")
tuned = vis[vis.dg_p_perm < 0.05]
both = vis[(vis.dg_p_perm < 0.05) & (vis.sg_p_perm < 0.05)]
delta = circ_dist_ori(both.dg_pref_ori.values, both.sg_pref_ori.values)
rng = np.random.default_rng(0)
delta_shuf = circ_dist_ori(both.dg_pref_ori.values,
                           rng.permutation(both.sg_pref_ori.values))
d_card = np.minimum(circ_dist_ori(tuned.dg_pref_ori.values, 0.0),
                    circ_dist_ori(tuned.dg_pref_ori.values, 90.0))

counts = dec[dec.task == "orientation"].groupby("n_units").area.nunique()
nsel = int(counts[counts >= counts.max()].index.max())
dec_best = dec[(dec.task == "orientation") & (dec.n_units == nsel)]

summary = dict(
    n_sessions=len(sessions),
    n_units=int(len(units)),
    n_visual_cortex=int(len(vis)),
    frac_tuned_visual_cortex=float((vis.dg_p_perm < 0.05).mean()),
    frac_tuned_control=float((ctl.dg_p_perm < 0.05).mean()),
    median_osi_visual=float(vis.dg_osi.median()),
    median_osi_null_visual=float(vis.dg_osi_null_median.median()),
    median_osi_control=float(ctl.dg_osi.median()),
    mannwhitney_p=float(mw.pvalue),
    n_both_stimuli=int(len(both)),
    median_delta_pref_ori=float(np.median(delta)),
    median_delta_pref_ori_shuffled=float(np.median(delta_shuf)),
    frac_delta_under_30deg=float((delta < 30).mean()),
    frac_cardinal_preference=float((d_card < 22.5).mean()),
    frac_cardinal_chance=0.5,
    frac_cardinal_binomial_p=float(stats.binomtest(
        int((d_card < 22.5).sum()), len(d_card), 0.5, alternative="greater").pvalue),
    cardinal_bias_cos4=float(np.mean(np.cos(4 * np.deg2rad(tuned.dg_pref_ori.values)))),
    decoding_n_units=nsel,
    decoding_orientation=({a: float(g.acc.mean()) for a, g in
                           dec_best[dec_best.kind == "observed"].groupby("area")}),
    decoding_orientation_shuffled=({a: float(g.acc.mean()) for a, g in
                                    dec_best[dec_best.kind == "shuffled"].groupby("area")}),
    glm_frac_positive_dir_given_speed={
        a: float((glm_df.loc[glm_df.area == a, "d_dir_given_speed"] > 0).mean())
        for a in VISUAL_CORTEX + CONTROL if (glm_df.area == a).sum() >= 20},
)
json.dump(summary, open("summary_stats.json", "w"), indent=2)
print(json.dumps(summary, indent=2))

# %% [markdown]
# ## 11. Conclusions
#
# Orientation selectivity is present, strong and specific in this dataset. Units in
# every visual cortical area carry orientation-tuned responses with a median OSI
# several times the shuffled null, roughly four fifths of them individually
# significant against a within-block permutation test. The tuning is bilobed: the
# mean curve aligned to each unit's preferred direction shows a second peak at the
# opposite direction, which is the defining property of orientation rather than
# direction selectivity.
#
# Three controls make the result hard to explain away. Simultaneously recorded
# hippocampal units, analysed identically, sit at the shuffled null. Preferred
# orientation estimated from drifting gratings predicts preferred orientation
# estimated from static gratings, an independent stimulus class with no motion,
# with a median discrepancy far below chance. And a Poisson GLM shows that grating
# direction improves held-out likelihood after running speed is already in the
# model, so the effect is not a by-product of locomotion.
#
# Two observations deserve qualification. First, visual thalamus (LGd and LP) is
# not at the null: a substantial minority of thalamic units reach significance,
# with OSI values between cortex and hippocampus. This is consistent with the
# current literature, in which mouse dLGN contains a genuine orientation-biased
# subpopulation, and it is a reminder that "orientation selectivity is created in
# cortex" is an approximation. Second, CA1 shows a small but above-chance fraction
# of significant units and modestly above-chance orientation decoding. The GLM
# suggests why: hippocampal firing is strongly modulated by running speed, and any
# residual coupling between behavioural state and the stimulus sequence produces
# weak apparent tuning. The effect is an order of magnitude smaller than the
# cortical one and disappears in the direction-given-speed comparison.

