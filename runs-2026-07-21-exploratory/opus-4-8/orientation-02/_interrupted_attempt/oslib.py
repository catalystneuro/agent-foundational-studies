"""Shared loading and tuning utilities for the orientation-selectivity analysis.

Data source: DANDI:000021, Allen Institute Visual Coding - Neuropixels
(Brain Observatory 1.1 stimulus set). Session-level NWB files are read over HTTP
with remfile + a disk cache; nothing is downloaded in full.
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
SESSION_JSON = os.path.join(HERE, "sessions_000021.json")
DISK_CACHE = remfile.DiskCache(os.environ.get("REMFILE_CACHE", "/tmp/remfile_cache"))

# Visual areas of interest plus the thalamic input nucleus used as a comparison.
CORTICAL_AREAS = ["VISp", "VISl", "VISrl", "VISal", "VISam", "VISpm"]
AREAS = CORTICAL_AREAS + ["LGd"]

# Allen Institute default unit-quality criteria.
QC = dict(isi_violations=0.5, amplitude_cutoff=0.1, presence_ratio=0.9)


def session_urls():
    if not os.path.exists(SESSION_JSON):
        from dandi.dandiapi import DandiAPIClient

        ds = DandiAPIClient().get_dandiset("000021", "draft")
        rows = []
        for a in ds.get_assets():
            if "probe-" in a.path:
                continue
            rows.append(
                dict(
                    session=a.path.split("ses-")[1].split(".")[0],
                    path=a.path,
                    size_gb=round(a.size / 1e9, 2),
                    url=a.get_content_url(follow_redirects=1, strip_query=True),
                )
            )
        rows.sort(key=lambda r: r["session"])
        json.dump(rows, open(SESSION_JSON, "w"), indent=1)
    return {s["session"]: s["url"] for s in json.load(open(SESSION_JSON))}


def open_session(session_id):
    """Open one session NWB file for streaming reads."""
    rf = remfile.File(session_urls()[str(session_id)], disk_cache=DISK_CACHE)
    io = NWBHDF5IO(file=h5py.File(rf, "r"), load_namespaces=True)
    return io.read()


def unit_table(nwbfile):
    """Units metadata with the recording area resolved through peak_channel_id."""
    u = nwbfile.units
    cols = [
        "quality",
        "snr",
        "isi_violations",
        "amplitude_cutoff",
        "presence_ratio",
        "firing_rate",
        "peak_channel_id",
        "waveform_duration",
    ]
    df = pd.DataFrame({c: np.asarray(u[c].data[:]) for c in cols}, index=np.asarray(u.id[:]))
    df["quality"] = [q.decode() if isinstance(q, bytes) else str(q) for q in df["quality"]]

    el = nwbfile.electrodes
    loc = np.asarray(el["location"].data[:])
    loc = np.array([l.decode() if isinstance(l, bytes) else str(l) for l in loc])
    eid = np.asarray(el.id[:])
    df["area"] = pd.Series(loc, index=eid).reindex(df["peak_channel_id"].values).values
    probe = np.asarray(el["probe_id"].data[:])
    df["probe_id"] = pd.Series(probe, index=eid).reindex(df["peak_channel_id"].values).values
    return df


def good_units(nwbfile, areas=AREAS):
    """Quality-filtered units in the areas of interest, as a pynapple TsGroup."""
    df = unit_table(nwbfile)
    keep = (
        (df["quality"] == "good")
        & (df["isi_violations"] < QC["isi_violations"])
        & (df["amplitude_cutoff"] < QC["amplitude_cutoff"])
        & (df["presence_ratio"] > QC["presence_ratio"])
        & df["area"].isin(areas)
    )
    df = df[keep]
    u = nwbfile.units
    ids = np.asarray(u.id[:])
    pos = {uid: i for i, uid in enumerate(ids)}
    spikes = {int(uid): np.asarray(u["spike_times"][pos[uid]]) for uid in df.index}
    tsg = nap.TsGroup({k: nap.Ts(t=v) for k, v in spikes.items()})
    for col in ("area", "snr", "firing_rate", "waveform_duration", "probe_id"):
        tsg.set_info(**{col: np.asarray(df[col].values)})
    return tsg, df


def stim_table(nwbfile, name):
    """Stimulus presentation table with numeric parameter columns."""
    iv = nwbfile.intervals[name]
    cols = [c for c in iv.colnames if c not in ("tags", "timeseries")]
    out = {}
    for c in cols:
        v = np.asarray(iv[c].data[:])
        if v.dtype.kind in "SO":
            v = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in v])
        out[c] = v
    df = pd.DataFrame(out)
    for c in ("orientation", "temporal_frequency", "spatial_frequency", "phase", "contrast"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["duration"] = df["stop_time"] - df["start_time"]
    return df


def trial_rates(tsg, df, offset=0.03, window=None):
    """Firing rate (Hz) of every unit on every trial.

    The response window runs from ``start_time + offset`` for ``window``
    seconds; ``offset`` skips the visual response latency and also keeps the
    windows of back-to-back presentations (the static gratings) from touching,
    which would otherwise let pynapple merge them into one interval.

    Returns an array of shape (n_units, n_trials).
    """
    start = df["start_time"].values + offset
    if window is None:
        window = float(np.median(df["duration"].values)) - offset
    stop = start + window
    ep = nap.IntervalSet(start=start, end=stop)
    assert len(ep) == len(df), f"IntervalSet merged trials: {len(ep)} != {len(df)}"
    counts = tsg.count(ep=ep)  # TsdFrame (n_trials, n_units)
    return np.asarray(counts.values).T / window


def invalid_trial_mask(nwbfile, probe_ids, starts, stops):
    """True where a trial overlaps an invalid-data interval on that unit's probe.

    DANDI:000021 sessions flag stretches where a probe's data are unusable
    (``nwbfile.intervals['invalid_times']``, tagged with the probe id). Units on
    that probe fall silent during those stretches, so the affected trials must
    be dropped rather than counted as zero-rate trials.
    """
    mask = np.zeros((len(probe_ids), len(starts)), dtype=bool)
    if "invalid_times" not in nwbfile.intervals:
        return mask
    iv = nwbfile.intervals["invalid_times"]
    tags = iv["tags"][:]
    for i in range(len(iv)):
        pid = None
        for t in tags[i]:
            t = t.decode() if isinstance(t, bytes) else str(t)
            if t.isdigit():
                pid = int(t)
        if pid is None:
            continue
        a, b = iv["start_time"][i], iv["stop_time"][i]
        overlap = (starts < b) & (stops > a)
        mask[np.asarray(probe_ids) == pid] |= overlap
    return mask


# ----------------------------------------------------------------------------
# Tuning metrics
# ----------------------------------------------------------------------------
def tuning_by_condition(rates, cond):
    """Mean and SEM of rates for each unique condition value."""
    levels = np.unique(cond[~np.isnan(cond)])
    mean = np.zeros((rates.shape[0], levels.size))
    sem = np.zeros_like(mean)
    for j, lv in enumerate(levels):
        r = rates[:, cond == lv]
        n = np.sum(~np.isnan(r), axis=1)
        with np.errstate(invalid="ignore"):
            mean[:, j] = np.nanmean(r, axis=1)
            sem[:, j] = np.nanstd(r, axis=1, ddof=1) / np.sqrt(np.maximum(n, 1))
    return levels, mean, sem


def gosi(levels_deg, tc):
    """Global orientation selectivity index: |sum r*exp(2i*theta)| / sum r.

    Works for both 8-direction (360 deg) and 6-orientation (180 deg) stimulus
    sets because doubling the angle folds opposite directions together.
    """
    th = np.deg2rad(levels_deg)
    r = np.clip(tc, 0, None)
    num = np.abs((r * np.exp(2j * th)).sum(axis=-1))
    den = r.sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def gdsi(levels_deg, tc):
    """Global direction selectivity index: |sum r*exp(i*theta)| / sum r."""
    th = np.deg2rad(levels_deg)
    r = np.clip(tc, 0, None)
    num = np.abs((r * np.exp(1j * th)).sum(axis=-1))
    den = r.sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def preferred_orientation(levels_deg, tc):
    """Vector-average preferred orientation in [0, 180) degrees."""
    th = np.deg2rad(levels_deg)
    r = np.clip(tc, 0, None)
    ang = np.angle((r * np.exp(2j * th)).sum(axis=-1))
    return np.rad2deg(ang / 2.0) % 180.0


def circ_dist_180(a, b):
    """Smallest angular distance between two orientations, in [0, 90]."""
    d = np.abs(a - b) % 180.0
    return np.minimum(d, 180.0 - d)


def permutation_pvalue(rates, cond, n_perm=1000, rng=None, stat="gosi"):
    """p-value for gOSI against a null that shuffles condition labels."""
    rng = np.random.default_rng(0 if rng is None else rng)
    levels, tc, _ = tuning_by_condition(rates, cond)
    fn = gosi if stat == "gosi" else gdsi
    obs = fn(levels, tc)
    null = np.zeros((n_perm, rates.shape[0]))
    for i in range(n_perm):
        _, tcp, _ = tuning_by_condition(rates, rng.permutation(cond))
        null[i] = fn(levels, tcp)
    return obs, (1 + (null >= obs[None, :]).sum(axis=0)) / (n_perm + 1)


def anova_pvalue(rates, cond):
    """One-way ANOVA across condition levels, per unit."""
    from scipy import stats

    levels = np.unique(cond[~np.isnan(cond)])
    groups = [rates[:, cond == lv] for lv in levels]
    return stats.f_oneway(*groups, axis=1).pvalue


def analyze_gratings(rates, cond, sub, n_perm=1000, seed=0):
    """Per-unit orientation/direction tuning at each unit's preferred sub-condition.

    ``cond`` is the stimulus orientation (or direction) of each trial and ``sub``
    is the nuisance parameter that is also varied (temporal frequency for
    drifting gratings, spatial frequency for static gratings). Trials with
    ``cond`` NaN are blank sweeps and are used as a baseline.

    Returns a dict of per-unit arrays plus the condition levels.
    """
    from scipy import stats

    blank = np.isnan(cond)
    levels = np.unique(cond[~blank])
    subs = np.unique(sub[~blank & ~np.isnan(sub)])
    n_units = rates.shape[0]
    rng = np.random.default_rng(seed)

    # Preferred sub-condition: the one with the largest mean response.
    sub_mean = np.stack([rates[:, (sub == s) & ~blank].mean(axis=1) for s in subs], axis=1)
    pref_sub_i = np.argmax(sub_mean, axis=1)

    out = dict(
        levels=levels,
        subs=subs,
        pref_sub=subs[pref_sub_i],
        sub_mean=sub_mean,
        baseline=rates[:, blank].mean(axis=1) if blank.any() else np.full(n_units, np.nan),
        tc=np.zeros((n_units, levels.size)),
        sem=np.zeros((n_units, levels.size)),
        n_trials=np.zeros(n_units, dtype=int),
        gosi=np.zeros(n_units),
        dsi=np.zeros(n_units),
        p_anova=np.ones(n_units),
        p_perm=np.ones(n_units),
        pref_ori=np.zeros(n_units),
    )

    for si, s in enumerate(subs):
        which = np.flatnonzero(pref_sub_i == si)
        if which.size == 0:
            continue
        m = (sub == s) & ~blank
        r = rates[np.ix_(which, np.flatnonzero(m))]
        c = cond[m]
        lv, tc, sem = tuning_by_condition(r, c)
        assert np.array_equal(lv, levels)
        out["tc"][which], out["sem"][which] = tc, sem
        out["n_trials"][which] = np.sum(~np.isnan(r), axis=1)
        out["gosi"][which] = gosi(levels, tc)
        out["dsi"][which] = gdsi(levels, tc)
        out["pref_ori"][which] = preferred_orientation(levels, tc)
        p = np.ones(which.size)
        for k in range(which.size):
            ok = ~np.isnan(r[k])
            groups = [r[k][ok & (c == lvl)] for lvl in levels]
            if min(len(g) for g in groups) >= 3:
                p[k] = stats.f_oneway(*groups).pvalue
        out["p_anova"][which] = p
        null = np.zeros((n_perm, which.size))
        for i in range(n_perm):
            _, tcp, _ = tuning_by_condition(r, rng.permutation(c))
            null[i] = gosi(levels, tcp)
        out["p_perm"][which] = (1 + (null >= out["gosi"][which][None, :]).sum(axis=0)) / (n_perm + 1)

    out["tuned"] = (out["p_anova"] < 0.01) & (out["p_perm"] < 0.05)
    return out


def align_to_preferred(tc, levels):
    """Rotate each unit's tuning curve so its peak sits at index 0, and normalize."""
    peak = np.argmax(tc, axis=1)
    n = levels.size
    idx = (np.arange(n)[None, :] + peak[:, None]) % n
    rolled = np.take_along_axis(tc, idx, axis=1)
    denom = rolled[:, [0]]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, rolled / denom, np.nan)
