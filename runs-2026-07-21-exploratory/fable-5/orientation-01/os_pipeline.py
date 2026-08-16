"""
Shared loading and analysis routines for the orientation-selectivity demonstration
using DANDI:000021 (Allen Institute Visual Coding - Neuropixels, Brain Observatory 1.1).

Design notes
------------
Each session NWB file holds ~80 million spike times across ~1600 units.  Rather than
materialising all of them, we read the ragged ``spike_times`` dataset one unit at a
time, and only for units that pass quality control and sit in an area we care about.
Each unit's spikes are contiguous in the file, so this is a small number of large
sequential reads.
"""

import numpy as np
import pandas as pd
import h5py
import remfile
import pynapple as nap
from scipy import stats

nap.nap_config.suppress_conversion_warnings = True

DANDISET = "000021"
ASSET_URL = "https://api.dandiarchive.org/api/assets/{asset_id}/download/"
CACHE_DIR = "/tmp/remfile_cache"

# Visual cortical areas in the Allen CCF nomenclature, plus visual thalamus and a
# non-visual control region (hippocampus).
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


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------
def open_session(asset_id):
    """Open a DANDI asset as a streaming h5py File (chunks cached on local disk)."""
    rem = remfile.File(ASSET_URL.format(asset_id=asset_id), disk_cache=remfile.DiskCache(CACHE_DIR))
    return h5py.File(rem, "r")


def _decode(arr):
    return np.array([s.decode() if isinstance(s, bytes) else str(s) for s in arr])


def load_unit_table(h5):
    """Unit metadata with brain area resolved through the electrodes table."""
    el = h5["general"]["extracellular_ephys"]["electrodes"]
    loc_by_channel = dict(zip(el["id"][:], _decode(el["location"][:])))

    u = h5["units"]
    df = pd.DataFrame(
        {
            "unit_id": u["id"][:],
            "peak_channel_id": u["peak_channel_id"][:],
            "quality": _decode(u["quality"][:]),
            "isi_violations": u["isi_violations"][:],
            "amplitude_cutoff": u["amplitude_cutoff"][:],
            "presence_ratio": u["presence_ratio"][:],
            "snr": u["snr"][:],
            "firing_rate": u["firing_rate"][:],
            "waveform_duration": u["waveform_duration"][:],
        }
    )
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

    ``row_indices`` are positional rows into the units table.  Spike times for each unit
    occupy a contiguous slice of the ragged ``spike_times`` dataset delimited by
    ``spike_times_index``.

    The NWB units table is not sorted by unit id, but ``nap.TsGroup`` sorts its keys, so
    the metadata is explicitly reindexed onto ``tsgroup.index`` before being attached.
    Everything downstream is then in TsGroup order.
    """
    idx = h5["units"]["spike_times_index"][:]
    starts = np.concatenate([[0], idx[:-1]])
    st_ds = h5["units"]["spike_times"]

    spikes = {}
    for row in row_indices:
        t = st_ds[starts[row] : idx[row]]
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


def load_running_speed(h5):
    """Running speed as a pynapple Tsd, if present."""
    proc = h5.get("processing", {})
    if "running" not in proc:
        return None
    grp = proc["running"]["running_speed"]
    return nap.Tsd(t=grp["timestamps"][:], d=grp["data"][:])


# --------------------------------------------------------------------------------------
# Trial-resolved responses
# --------------------------------------------------------------------------------------
def trial_spike_counts(tsgroup, starts, stops):
    """
    Spike counts for every (trial, unit) pair.

    Each unit's spike train is sorted, so counts in an arbitrary window come from two
    binary searches.  Returns an array of shape (n_trials, n_units).
    """
    counts = np.empty((len(starts), len(tsgroup)), dtype=np.int32)
    for j, uid in enumerate(tsgroup.keys()):
        t = tsgroup[uid].t
        counts[:, j] = np.searchsorted(t, stops) - np.searchsorted(t, starts)
    return counts


def trial_rates(tsgroup, table, window=(0.0, None), blank_window=None):
    """
    Firing rate (Hz) in a response window relative to each stimulus onset.

    ``window`` is (offset_from_onset, offset_from_onset_end); if the second element is
    None the stimulus stop_time is used.
    """
    starts = table["start_time"].values + window[0]
    stops = table["stop_time"].values if window[1] is None else table["start_time"].values + window[1]
    counts = trial_spike_counts(tsgroup, starts, stops)
    return counts / (stops - starts)[:, None]


# --------------------------------------------------------------------------------------
# Orientation-tuning metrics
# --------------------------------------------------------------------------------------
def orientation_metrics(rates_by_ori, thetas_deg):
    """
    Standard orientation-tuning metrics from mean rates at each orientation.

    ``thetas_deg`` are orientations in degrees spanning 0-180 (static gratings) or
    directions spanning 0-360 (drifting gratings).  Metrics are computed on the
    orientation axis, i.e. on 2*theta, so opposite directions are equivalent.

    Returns gOSI (1 - circular variance at 2*theta), the classic two-point OSI, and the
    preferred orientation in degrees (0-180).
    """
    r = np.clip(np.asarray(rates_by_ori, dtype=float), 0, None)
    th = np.deg2rad(np.asarray(thetas_deg, dtype=float))
    total = r.sum()
    if total <= 0:
        return dict(gOSI=np.nan, OSI=np.nan, pref_ori=np.nan)

    vec = (r * np.exp(2j * th)).sum() / total
    gosi = np.abs(vec)
    pref = (np.angle(vec) / 2) % np.pi

    # Two-point OSI: response at the preferred orientation vs. the orthogonal one,
    # collapsing directions onto the 0-180 orientation axis first.
    ori = np.rad2deg(th) % 180
    uniq = np.unique(ori)
    collapsed = np.array([r[np.isclose(ori, o)].mean() for o in uniq])
    i_pref = int(np.argmax(collapsed))
    orth = (uniq[i_pref] + 90) % 180
    i_orth = int(np.argmin(np.abs(((uniq - orth + 90) % 180) - 90)))
    r_p, r_o = collapsed[i_pref], collapsed[i_orth]
    osi = (r_p - r_o) / (r_p + r_o) if (r_p + r_o) > 0 else np.nan

    return dict(gOSI=float(gosi), OSI=float(osi), pref_ori=float(np.rad2deg(pref)))


def direction_metrics(rates_by_dir, dirs_deg):
    """Direction selectivity index and preferred direction (drifting gratings only)."""
    r = np.clip(np.asarray(rates_by_dir, dtype=float), 0, None)
    th = np.deg2rad(np.asarray(dirs_deg, dtype=float))
    total = r.sum()
    if total <= 0:
        return dict(gDSI=np.nan, DSI=np.nan, pref_dir=np.nan)
    vec = (r * np.exp(1j * th)).sum() / total
    i_pref = int(np.argmax(r))
    null_deg = (dirs_deg[i_pref] + 180) % 360
    i_null = int(np.argmin(np.abs(((np.asarray(dirs_deg) - null_deg + 180) % 360) - 180)))
    r_p, r_n = r[i_pref], r[i_null]
    dsi = (r_p - r_n) / (r_p + r_n) if (r_p + r_n) > 0 else np.nan
    return dict(gDSI=float(np.abs(vec)), DSI=float(dsi), pref_dir=float(dirs_deg[i_pref]))


def permutation_test_gosi(trial_rates_vec, ori_labels, thetas, n_perm=1000, rng=None):
    """
    Null distribution for gOSI built by shuffling orientation labels across trials.

    Returns (observed gOSI, p-value).
    """
    rng = np.random.default_rng(0) if rng is None else rng
    obs_means = np.array([trial_rates_vec[ori_labels == t].mean() for t in thetas])
    obs = orientation_metrics(obs_means, thetas)["gOSI"]
    if not np.isfinite(obs):
        return np.nan, np.nan

    null = np.empty(n_perm)
    shuffled = ori_labels.copy()
    for i in range(n_perm):
        rng.shuffle(shuffled)
        means = np.array([trial_rates_vec[shuffled == t].mean() for t in thetas])
        null[i] = orientation_metrics(means, thetas)["gOSI"]
    p = (np.sum(null >= obs) + 1) / (n_perm + 1)
    return obs, float(p)


def von_mises_ori(theta_deg, baseline, amp, kappa, mu_deg):
    """Von Mises tuning on the orientation (pi-periodic) axis."""
    th = np.deg2rad(theta_deg)
    mu = np.deg2rad(mu_deg)
    return baseline + amp * np.exp(kappa * (np.cos(2 * (th - mu)) - 1))


def vm_hwhm_deg(kappa):
    """Half-width at half-maximum, in degrees of orientation, for a von Mises fit."""
    if kappa <= 0:
        return np.nan
    arg = 1 + np.log(0.5) / kappa
    if arg < -1:
        return np.nan
    return float(np.rad2deg(np.arccos(np.clip(arg, -1, 1)) / 2))
