"""Shared helpers for the orientation-selectivity analysis of DANDI:000021.

DANDI:000021 is the Allen Institute Visual Coding Neuropixels dataset,
"Brain Observatory 1.1" stimulus set. Each session NWB file holds sorted units
from up to six Neuropixels probes plus interval tables for every stimulus block.
The two blocks used here are:

  drifting_gratings_presentations : 2 s gratings, 8 directions x 5 temporal
      frequencies x 15 repeats, plus interleaved blank sweeps.
  static_gratings_presentations   : 0.25 s gratings, 6 orientations x 5 spatial
      frequencies x 4 phases, presented back to back.

The drifting-grating block varies *direction* (0-315 deg), so orientation is
direction modulo 180. The static-grating block varies *orientation* directly
(0-150 deg). The two blocks are separated in time and use different stimulus
parameters, which makes them a genuine cross-validation of each other.
"""

import json
import os

import h5py
import numpy as np
import pandas as pd
import pynapple as nap
import remfile
from pynwb import NWBHDF5IO

HERE = os.path.dirname(os.path.abspath(__file__))
SESSIONS = {s["session"]: s for s in json.load(open(os.path.join(HERE, "sessions_000021.json")))}
DISK_CACHE = remfile.DiskCache("/tmp/remfile_cache")

# Cortical visual areas in this dataset, ordered roughly by hierarchy, plus the
# thalamic relay nucleus LGd and the visual sector of the thalamic reticular
# nucleus, which serve as subcortical comparisons.
VISUAL_CORTEX = ["VISp", "VISl", "VISrl", "VISal", "VISpm", "VISam"]
THALAMUS = ["LGd", "LP"]

# Unit quality thresholds, following the criteria the Allen Institute recommends
# for this dataset (see the AllenSDK "unit quality metrics" documentation).
QC = dict(isi_violations=0.5, amplitude_cutoff=0.1, presence_ratio=0.9, snr=1.0)


def open_session(session_id):
    """Stream one session NWB file from S3 with an on-disk block cache."""
    rf = remfile.File(SESSIONS[str(session_id)]["url"], disk_cache=DISK_CACHE)
    io = NWBHDF5IO(file=h5py.File(rf, "r"), load_namespaces=True)
    return io.read()


def good_units(nwbfile):
    """Return (TsGroup of QC-passing units, metadata DataFrame).

    Units are labelled with the CCF structure of their peak channel, taken from
    the electrodes table, and filtered by the standard quality metrics.
    """
    units = nwbfile.units.to_dataframe()
    elec = nwbfile.electrodes.to_dataframe()
    units["area"] = elec.loc[units["peak_channel_id"].values, "location"].values
    units["probe_id"] = elec.loc[units["peak_channel_id"].values, "group_name"].values

    keep = (
        (units["isi_violations"] < QC["isi_violations"])
        & (units["amplitude_cutoff"] < QC["amplitude_cutoff"])
        & (units["presence_ratio"] > QC["presence_ratio"])
        & (units["snr"] > QC["snr"])
        & units["area"].isin(VISUAL_CORTEX + THALAMUS)
    )
    units = units[keep]

    spikes = {int(uid): np.asarray(row["spike_times"]) for uid, row in units.iterrows()}
    meta = units[["area", "probe_id", "snr", "firing_rate", "waveform_duration", "isi_violations"]].copy()
    meta.index = meta.index.astype(int)
    tsg = nap.TsGroup({k: nap.Ts(v) for k, v in spikes.items()}, metadata=meta)
    return tsg, meta


def stim_table(nwbfile, name):
    """Stimulus interval table as a DataFrame, blank sweeps dropped."""
    df = nwbfile.intervals[name].to_dataframe()
    # Blank sweeps store "null" in the parameter columns.
    for col in ("orientation", "spatial_frequency", "temporal_frequency", "phase"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["orientation"]).reset_index(drop=True)
    return df


def blank_table(nwbfile, name):
    """The interleaved blank sweeps of a stimulus block (used for baseline rate)."""
    df = nwbfile.intervals[name].to_dataframe()
    ori = pd.to_numeric(df["orientation"], errors="coerce")
    return df[ori.isna()].reset_index(drop=True)


def response_windows(starts, stops, offset, window, guard=0.005):
    """Non-overlapping per-trial counting windows.

    The window opens `offset` seconds after stimulus onset (skipping the visual
    response latency) and would last `window` seconds, but it is truncated at
    stimulus offset and at the next trial's onset. The truncation matters for the
    static gratings, which are presented back to back at ~4 Hz with occasional
    short frames: without it, adjacent windows overlap and pynapple silently
    merges them into a single interval.
    """
    starts, stops = np.asarray(starts, float), np.asarray(stops, float)
    assert np.all(np.diff(starts) > 0), "stimulus table must be sorted by onset"
    win_start = starts + offset
    win_end = np.minimum(starts + offset + window, stops)
    win_end = np.minimum(win_end, np.r_[starts[1:], np.inf] - guard)
    assert np.all(win_end > win_start), "counting window collapsed"
    return win_start, win_end


def trial_rates(tsg, starts, stops, offset, window):
    """Firing rate (Hz) of every unit on every trial. Returns (n_units, n_trials)."""
    a, b = response_windows(starts, stops, offset, window)
    ep = nap.IntervalSet(start=a, end=b)
    assert len(ep) == len(a), "intervals were merged; windows still overlap"
    counts = tsg.count(ep=ep)  # TsdFrame (n_trials, n_units): one count per interval
    assert counts.shape[0] == len(a)
    return np.asarray(counts).T / (b - a)


def invalid_trial_mask(nwbfile, probe_ids, starts, stops):
    """(n_units, n_trials) boolean: trial overlaps an invalid-data epoch for that unit's probe.

    The Allen files mark stretches where a probe dropped data in
    `invalid_times`; trials overlapping those stretches are excluded per probe.
    """
    bad = np.zeros((len(probe_ids), len(starts)), dtype=bool)
    if nwbfile.invalid_times is None:
        return bad
    inv = nwbfile.invalid_times.to_dataframe()
    for _, row in inv.iterrows():
        tag = str(row.get("tags", ""))
        overlap = (starts < row["stop_time"]) & (stops > row["start_time"])
        hit = np.array([str(p) in tag for p in probe_ids])
        bad[np.ix_(hit, overlap)] = True
    return bad


# ---------------------------------------------------------------------------
# Orientation-selectivity metrics
# ---------------------------------------------------------------------------

def osi_dsi(rates_by_dir, directions_deg):
    """Global OSI, DSI and preferred angles from mean rates at each direction.

    Uses the standard circular-variance formulation (Ringach et al. 2002):

        gOSI = |sum_k r_k exp(2 i theta_k)| / sum_k r_k
        gDSI = |sum_k r_k exp(1 i theta_k)| / sum_k r_k

    gOSI is 0 for a flat tuning curve and 1 for a unit that responds at a single
    orientation. Rates are clipped at 0 so a unit suppressed below baseline
    cannot produce a spurious vector.
    """
    r = np.clip(np.asarray(rates_by_dir, dtype=float), 0, None)
    th = np.deg2rad(np.asarray(directions_deg, dtype=float))
    denom = r.sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        vo = (r * np.exp(2j * th)).sum(axis=-1) / denom
        vd = (r * np.exp(1j * th)).sum(axis=-1) / denom
    pref_ori = np.rad2deg(np.angle(vo) / 2) % 180
    pref_dir = np.rad2deg(np.angle(vd)) % 360
    return np.abs(vo), np.abs(vd), pref_ori, pref_dir


def circ_dist_180(a, b):
    """Absolute difference between two orientations, wrapped into [0, 90] deg."""
    d = np.abs(np.asarray(a) - np.asarray(b)) % 180
    return np.minimum(d, 180 - d)


def tuning_by_group(rates, labels, levels=None):
    """NaN-aware mean rate per stimulus level.

    rates  : (n_units, n_trials), NaN where the trial was excluded for that unit
    labels : (n_trials,) stimulus value on each trial
    Returns (levels, means (n_units, n_levels), counts (n_units, n_levels)).
    """
    if levels is None:
        levels = np.array(sorted(np.unique(labels)))
    M = np.stack([(labels == v).astype(float) for v in levels])      # (n_levels, n_trials)
    valid = np.isfinite(rates).astype(float)
    filled = np.where(np.isfinite(rates), rates, 0.0)
    n = valid @ M.T
    with np.errstate(invalid="ignore", divide="ignore"):
        means = (filled @ M.T) / n
    return levels, means, n


def dir_tuning(rates, dirs_all, tf_all, baseline=None, dirs=None, tfs=None):
    """Direction tuning curve at each unit's preferred temporal frequency.

    The drifting-grating block crosses 8 directions with 5 temporal frequencies.
    Units in cortex are often tuned to low temporal frequencies and barely respond
    at 15 Hz, so pooling all temporal frequencies dilutes their orientation tuning
    while leaving high-pass thalamic units untouched. Following the standard
    analysis of this dataset, the tuning curve is therefore taken at the temporal
    frequency that drives each unit best (mean over directions).

    If `baseline` (per-unit blank-sweep rate) is given it is subtracted, so the
    metrics describe the stimulus-evoked response rather than the response plus
    an area-dependent amount of spontaneous activity.

    Returns (dirs, tuning (n_units, n_dirs), preferred TF index per unit).
    """
    if dirs is None:
        dirs = np.array(sorted(np.unique(dirs_all)))
    if tfs is None:
        tfs = np.array(sorted(np.unique(tf_all)))
    per_tf = np.stack([tuning_by_group(rates[:, tf_all == f], dirs_all[tf_all == f], dirs)[1]
                       for f in tfs])                        # (n_tf, n_units, n_dirs)
    if baseline is not None:
        per_tf = per_tf - baseline[None, :, None]
    best = np.nanargmax(np.nanmean(per_tf, axis=2), axis=0)  # (n_units,)
    tuning = per_tf[best, np.arange(per_tf.shape[1])]
    return dirs, tuning, best


def classic_osi(tuning, dirs):
    """(Rpref - Rorth) / (Rpref + Rorth), the definition used in most V1 papers.

    Rpref is the response at the best direction and Rorth the mean of the two
    directions 90 deg away from it. Reported alongside gOSI because published
    values for mouse V1 mostly use this form, which runs larger than gOSI.
    """
    t = np.clip(np.asarray(tuning, float), 0, None)
    k = np.nanargmax(t, axis=1)
    n = len(dirs)
    step = int(round(90.0 / (360.0 / n)))
    rp = t[np.arange(len(t)), k]
    ro = 0.5 * (t[np.arange(len(t)), (k + step) % n] + t[np.arange(len(t)), (k - step) % n])
    with np.errstate(invalid="ignore", divide="ignore"):
        return (rp - ro) / (rp + ro)


def perm_test_osi(rates, dirs_all, tf_all, baseline=None, n_perm=1000, seed=0):
    """Permutation test for orientation tuning, matched to the preferred-TF pipeline.

    Direction labels are shuffled *within* each temporal frequency, so the null
    preserves each unit's temporal-frequency tuning and its rate distribution and
    destroys only the relationship between direction and rate. The preferred-TF
    selection is redone inside every permutation, so the selection bias it
    introduces is present in the null as well.

    Returns (observed gOSI, p-value, mean null gOSI). The null mean is the noise
    floor of the statistic for that unit; gOSI minus the null mean is comparable
    across units and areas in a way the raw value is not.
    """
    dirs = np.array(sorted(np.unique(dirs_all)))
    tfs = np.array(sorted(np.unique(tf_all)))
    _, tuning, _ = dir_tuning(rates, dirs_all, tf_all, baseline, dirs, tfs)
    obs, _, _, _ = osi_dsi(tuning, dirs)

    rng = np.random.default_rng(seed)
    null = np.empty((n_perm, rates.shape[0]))
    shuffled = dirs_all.copy()
    tf_index = [np.where(tf_all == f)[0] for f in tfs]
    for i in range(n_perm):
        for idx in tf_index:
            shuffled[idx] = rng.permutation(dirs_all[idx])
        _, t, _ = dir_tuning(rates, shuffled, tf_all, baseline, dirs, tfs)
        null[i], _, _, _ = osi_dsi(t, dirs)
    p = (1 + (null >= obs[None, :]).sum(axis=0)) / (n_perm + 1)
    return obs, p, np.nanmean(null, axis=0)


def circ_corr_180(a, b):
    """Circular-circular correlation between two orientation variables (period 180 deg).

    Angles are doubled to map the 180 deg period onto the full circle, then the
    Jammalamadaka circular correlation coefficient is applied.
    """
    x = 2 * np.deg2rad(np.asarray(a, float))
    y = 2 * np.deg2rad(np.asarray(b, float))
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    mx = np.angle(np.mean(np.exp(1j * x)))
    my = np.angle(np.mean(np.exp(1j * y)))
    num = np.sum(np.sin(x - mx) * np.sin(y - my))
    den = np.sqrt(np.sum(np.sin(x - mx) ** 2) * np.sum(np.sin(y - my) ** 2))
    r = num / den
    # Permutation p-value; the asymptotic test is unreliable for concentrated data.
    rng = np.random.default_rng(1)
    null = np.empty(2000)
    for i in range(2000):
        yp = rng.permutation(y)
        myp = np.angle(np.mean(np.exp(1j * yp)))
        null[i] = np.sum(np.sin(x - mx) * np.sin(yp - myp)) / np.sqrt(
            np.sum(np.sin(x - mx) ** 2) * np.sum(np.sin(yp - myp) ** 2))
    return r, (1 + (np.abs(null) >= abs(r)).sum()) / 2001, len(x)
