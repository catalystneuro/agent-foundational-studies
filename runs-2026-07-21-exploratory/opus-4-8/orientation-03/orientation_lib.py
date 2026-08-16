"""Shared helpers for the orientation-selectivity analysis of DANDI:000021.

DANDI:000021 is the Allen Institute "Visual Coding - Neuropixels" dataset.  Each
session-level NWB file holds spike times for thousands of sorted units spanning
visual cortex, LGd and hippocampus, together with the timing and parameters of
every visual stimulus presentation.

Everything here streams from S3 with remfile + a local disk cache; no whole-file
downloads.
"""

import os

import h5py
import numpy as np
import pandas as pd
import pynapple as nap
import remfile
import requests
from pynwb import NWBHDF5IO

DANDISET = "000021"
CACHE_DIR = os.environ.get("REMFILE_CACHE", "/tmp/remfile_cache_000021")
ASSET_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "session_assets.csv")

# Cortical visual areas present in this dataset, ordered roughly V1 -> higher areas.
VISUAL_AREAS = ["VISp", "VISl", "VISrl", "VISal", "VISpm", "VISam"]
# Subcortical comparison region: the thalamic relay that drives V1.
CONTROL_AREAS = ["LGd"]

# Quality-metric thresholds.  These are the cutoffs recommended in the Allen
# Institute's Neuropixels cheat sheet for "clean" single units.
QC = dict(isi_violations=0.5, amplitude_cutoff=0.1, presence_ratio=0.9, snr=1.0)


# --------------------------------------------------------------------------- #
# Asset discovery
# --------------------------------------------------------------------------- #
def list_session_assets(refresh=False):
    """Return a DataFrame of session-level (non-probe) NWB assets in DANDI:000021."""
    if os.path.exists(ASSET_CACHE) and not refresh:
        return pd.read_csv(ASSET_CACHE)

    rows = []
    url = f"https://api.dandiarchive.org/api/dandisets/{DANDISET}/versions/draft/assets/?page_size=1000"
    while url:
        payload = requests.get(url, timeout=60).json()
        for asset in payload["results"]:
            if "probe" in asset["path"]:
                continue  # per-probe files hold LFP only
            rows.append(
                dict(
                    path=asset["path"],
                    asset_id=asset["asset_id"],
                    size_gb=round(asset["size"] / 1e9, 2),
                    session_id=asset["path"].split("ses-")[1].split(".")[0],
                )
            )
        url = payload.get("next")

    df = pd.DataFrame(rows).sort_values("session_id").reset_index(drop=True)
    # Resolve the direct S3 blob URL once so later runs skip the redirect.
    s3 = []
    for aid in df.asset_id:
        meta = requests.get(
            f"https://api.dandiarchive.org/api/dandisets/{DANDISET}/versions/draft/assets/{aid}/",
            timeout=60,
        ).json()
        s3.append([u for u in meta["contentUrl"] if "s3.amazonaws.com" in u][0])
    df["s3_url"] = s3
    df.to_csv(ASSET_CACHE, index=False)
    return df


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def open_session(s3_url):
    """Stream an NWB file from S3 and return the pynwb NWBFile (plus handles)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    rem = remfile.File(s3_url, disk_cache=remfile.DiskCache(CACHE_DIR))
    h5 = h5py.File(rem, "r")
    io = NWBHDF5IO(file=h5, load_namespaces=True)
    return io.read(), io


def unit_table(nwbfile):
    """Unit metadata + brain-region label, without pulling any spike times."""
    units = nwbfile.units
    cols = [
        "quality",
        "snr",
        "isi_violations",
        "amplitude_cutoff",
        "presence_ratio",
        "firing_rate",
        "peak_channel_id",
    ]
    df = pd.DataFrame({c: units[c][:] for c in cols}, index=np.asarray(units.id[:]))
    df.index.name = "unit_id"
    electrodes = nwbfile.electrodes.to_dataframe()
    df["location"] = electrodes.location.reindex(df.peak_channel_id).values
    df["row"] = np.arange(len(df))  # position in the ragged spike_times index
    return df


def passes_qc(df):
    return (
        (df.quality == "good")
        & (df.isi_violations < QC["isi_violations"])
        & (df.amplitude_cutoff < QC["amplitude_cutoff"])
        & (df.presence_ratio > QC["presence_ratio"])
        & (df.snr > QC["snr"])
    )


def load_spikes(nwbfile, unit_df):
    """Read spike times for the selected units only and wrap them in a TsGroup.

    ``nap.TsGroup`` stores its units in sorted key order, which is not the order
    of the NWB units table.  The unit table is therefore sorted here and returned
    alongside the TsGroup, so that column *j* of every trial-by-unit matrix
    corresponds to row *j* of the returned table.  Getting this wrong silently
    attaches each unit's tuning to another unit's brain area.
    """
    unit_df = unit_df.sort_index()
    spikes = {}
    for uid, row in zip(unit_df.index, unit_df.row):
        spikes[int(uid)] = nap.Ts(t=np.asarray(nwbfile.units["spike_times"][int(row)]))
    metadata = unit_df[["location", "firing_rate", "snr"]].copy()
    metadata.index = metadata.index.astype(int)
    group = nap.TsGroup(spikes, metadata=metadata)
    assert np.array_equal(np.asarray(list(group.keys())), unit_df.index.values.astype(int))
    return group, unit_df


def stimulus_table(nwbfile, name):
    """Stimulus presentation table as a DataFrame with numeric parameter columns."""
    df = nwbfile.intervals[name].to_dataframe()
    df = df.drop(columns=[c for c in ("timeseries", "tags") if c in df.columns])
    for col in ("orientation", "temporal_frequency", "spatial_frequency", "phase", "contrast"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# --------------------------------------------------------------------------- #
# Trial-resolved spike counts
# --------------------------------------------------------------------------- #
def trial_rates(spikes, stim, onset_lag=0.05, window=None):
    """Firing rate (Hz) of every unit on every stimulus trial.

    Parameters
    ----------
    spikes : nap.TsGroup
    stim : DataFrame with start_time / stop_time
    onset_lag : seconds skipped after stimulus onset to let the response arrive
    window : if given, response window length in seconds (else use full trial)

    Returns
    -------
    DataFrame, trials x units, in spikes/second.

    Notes
    -----
    Counting is done with ``np.searchsorted`` on each unit's spike train rather
    than with ``TsGroup.count(ep=...)``.  Pynapple's IntervalSet sorts and merges
    intervals, which silently collapses rows when consecutive stimulus windows
    touch (the static gratings run back-to-back at 4 Hz).  See
    :func:`check_counts_against_pynapple` for a correctness check of this
    implementation against pynapple on non-adjacent epochs.
    """
    start = stim.start_time.values + onset_lag
    stop = start + window if window is not None else stim.stop_time.values
    dur = stop - start
    unit_ids = np.asarray(list(spikes.keys()))
    out = np.empty((len(start), len(unit_ids)))
    for j, uid in enumerate(unit_ids):
        t = spikes[uid].t
        out[:, j] = np.searchsorted(t, stop) - np.searchsorted(t, start)
    return pd.DataFrame(out / dur[:, None], index=stim.index, columns=unit_ids)


def check_counts_against_pynapple(spikes, stim, onset_lag=0.05):
    """Verify trial_rates against pynapple's TsGroup.count on the same epochs."""
    start = stim.start_time.values + onset_lag
    stop = stim.stop_time.values
    ep = nap.IntervalSet(start=start, end=stop)
    assert len(ep) == len(stim), "epochs were merged; pynapple check not applicable"
    nap_counts = np.asarray(spikes.count(ep=ep).values)
    mine = trial_rates(spikes, stim, onset_lag=onset_lag).values * (stop - start)[:, None]
    return np.array_equal(nap_counts, np.rint(mine).astype(int))


# --------------------------------------------------------------------------- #
# Tuning metrics
# --------------------------------------------------------------------------- #
def tuning_curve(rates, angles):
    """Mean and SEM rate per unique angle.  Returns (angles, mean, sem, n)."""
    uniq = np.unique(angles)
    mean = np.stack([rates[angles == a].mean(axis=0) for a in uniq])
    sem = np.stack([rates[angles == a].std(axis=0, ddof=1) / np.sqrt((angles == a).sum()) for a in uniq])
    n = np.array([(angles == a).sum() for a in uniq])
    return uniq, mean, sem, n


def global_selectivity(angles_deg, mean_rates, harmonic=2):
    """Circular-variance selectivity index.

    harmonic=2 -> gOSI (180 deg periodic, orientation)
    harmonic=1 -> gDSI (360 deg periodic, direction)
    """
    theta = np.deg2rad(angles_deg)[:, None]
    r = np.clip(mean_rates, 0, None)
    denom = r.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        vec = (r * np.exp(1j * harmonic * theta)).sum(axis=0) / denom
    idx = np.abs(vec)
    pref = np.rad2deg(np.angle(vec)) / harmonic % (360.0 / harmonic)
    return np.where(denom > 0, idx, np.nan), pref


def classic_indices(angles_deg, mean_rates):
    """OSI and DSI computed from the preferred direction, Allen-style.

    OSI = (R_pref - R_orth) / (R_pref + R_orth),  orth = pref +/- 90 deg
    DSI = (R_pref - R_null) / (R_pref + R_null),  null = pref + 180 deg
    """
    angles_deg = np.asarray(angles_deg)
    r = np.clip(np.asarray(mean_rates), 0, None)
    ipref = np.argmax(r, axis=0)
    pref = angles_deg[ipref]

    def at(target):
        target = target % 360
        j = np.argmin(np.abs(((angles_deg[:, None] - target[None, :]) + 180) % 360 - 180), axis=0)
        return r[j, np.arange(r.shape[1])]

    r_pref = r[ipref, np.arange(r.shape[1])]
    r_orth = 0.5 * (at(pref + 90) + at(pref - 90))
    r_null = at(pref + 180)
    with np.errstate(invalid="ignore", divide="ignore"):
        osi = (r_pref - r_orth) / (r_pref + r_orth)
        dsi = (r_pref - r_null) / (r_pref + r_null)
    return pref, osi, dsi


def permutation_test(rates, angles, n_perm=1000, seed=0, harmonic=2):
    """P-value for tuning: is the observed selectivity index above chance?

    The angle labels are shuffled across trials, which destroys any relationship
    between stimulus and response while preserving each unit's overall rate and
    its trial-to-trial variability.
    """
    rng = np.random.default_rng(seed)
    rates = np.asarray(rates)
    angles = np.asarray(angles)
    uniq, mean, _, _ = tuning_curve(rates, angles)
    obs, _ = global_selectivity(uniq, mean, harmonic=harmonic)

    ge = np.zeros(rates.shape[1], dtype=int)
    for _ in range(n_perm):
        perm = rng.permutation(angles)
        _, m, _, _ = tuning_curve(rates, perm)
        null, _ = global_selectivity(uniq, m, harmonic=harmonic)
        ge += np.asarray(null >= obs, dtype=int)
    return obs, (ge + 1) / (n_perm + 1)


def shuffle_null(rates, angles, n_perm=500, seed=0, harmonic=2):
    """Distribution of the selectivity index under shuffled stimulus labels.

    Returns (observed, p_value, null_median).  The null median is the amount of
    apparent selectivity produced by trial-to-trial noise alone, and subtracting
    it gives a bias-corrected index that is comparable across firing rates.
    """
    rng = np.random.default_rng(seed)
    rates = np.asarray(rates)
    angles = np.asarray(angles)
    uniq, mean, _, _ = tuning_curve(rates, angles)
    obs, _ = global_selectivity(uniq, mean, harmonic=harmonic)

    null = np.empty((n_perm, rates.shape[1]))
    for i in range(n_perm):
        _, m, _, _ = tuning_curve(rates, rng.permutation(angles))
        null[i], _ = global_selectivity(uniq, m, harmonic=harmonic)
    p = (np.sum(null >= obs[None, :], axis=0) + 1) / (n_perm + 1)
    return obs, p, np.nanmedian(null, axis=0)


def split_half_preference(rates, angles, seed=0, harmonic=2):
    """Preferred angle estimated independently from two random halves of trials.

    If tuning is a real stimulus-driven property the two estimates agree; if the
    apparent tuning is noise they are unrelated.
    """
    rng = np.random.default_rng(seed)
    rates = np.asarray(rates)
    angles = np.asarray(angles)
    first = np.zeros(len(angles), dtype=bool)
    for a in np.unique(angles):
        idx = np.where(angles == a)[0]
        first[rng.permutation(idx)[: len(idx) // 2]] = True

    prefs = []
    for mask in (first, ~first):
        uniq, mean, _, _ = tuning_curve(rates[mask], angles[mask])
        _, pref = global_selectivity(uniq, mean, harmonic=harmonic)
        prefs.append(pref)
    return prefs[0], prefs[1]


def circular_distance(a, b, period=180.0):
    """Smallest absolute difference between two angles on a circle of given period."""
    d = (np.asarray(a) - np.asarray(b)) % period
    return np.minimum(d, period - d)


# --------------------------------------------------------------------------- #
# One-session pipeline
# --------------------------------------------------------------------------- #
ONSET_LAG = 0.05  # s, skipped after stimulus onset to let the visual response arrive
SG_WINDOW = 0.22  # s, response window for the 250 ms static gratings


def area_group(location):
    location = np.asarray(location)
    return np.where(
        location == "VISp", "VISp", np.where(np.isin(location, VISUAL_AREAS), "higher visual", "LGd")
    )


def analyze_session(s3_url, session_id, n_perm=500, seed=0, with_glm=True, keep_raw=False):
    """Full per-unit orientation analysis for one session.

    Returns a per-unit DataFrame, and (optionally) the raw trial matrices so the
    caller can plot example cells.
    """
    import glm_lib as gl

    nwbfile, io = open_session(s3_url)
    units = unit_table(nwbfile)
    sel = units[passes_qc(units) & units.location.isin(VISUAL_AREAS + CONTROL_AREAS)]
    spikes, sel = load_spikes(nwbfile, sel)

    dg = stimulus_table(nwbfile, "drifting_gratings_presentations")
    drive, blank = dg[dg.orientation.notna()], dg[dg.orientation.isna()]
    rates_df = trial_rates(spikes, drive, onset_lag=ONSET_LAG)
    assert np.array_equal(np.asarray(rates_df.columns), sel.index.values.astype(int))
    rates = rates_df.values
    blank_rates = trial_rates(spikes, blank, onset_lag=ONSET_LAG).values
    ang = drive.orientation.values
    tf = drive.temporal_frequency.values

    uniq, mean, sem, _ = tuning_curve(rates, ang)
    gosi, p_osi, null_osi = shuffle_null(rates, ang, n_perm=n_perm, seed=seed, harmonic=2)
    gdsi, p_dsi, null_dsi = shuffle_null(rates, ang, n_perm=n_perm, seed=seed + 1, harmonic=1)
    _, pref_ori = global_selectivity(uniq, mean, harmonic=2)
    pref_dir, osi, dsi = classic_indices(uniq, mean)
    h1, h2 = split_half_preference(rates, ang, seed=seed, harmonic=2)

    # preferred temporal frequency and the direction tuning curve measured there
    tfs = np.unique(tf)
    tf_mean = np.stack([rates[tf == t].mean(axis=0) for t in tfs])
    pref_tf = tfs[np.argmax(tf_mean, axis=0)]

    # static gratings: independent, motionless test of the same preference
    sg = stimulus_table(nwbfile, "static_gratings_presentations")
    sgd = sg[sg.orientation.notna()]
    sg_rates = trial_rates(spikes, sgd, onset_lag=0.03, window=SG_WINDOW).values
    sg_ang = sgd.orientation.values
    us, ms, _, _ = tuning_curve(sg_rates, sg_ang)
    gosi_sg, pref_sg = global_selectivity(us, ms, harmonic=2)
    sg_peak = ms.max(axis=0)

    peak = mean.max(axis=0)
    responsive = (peak > 1.0) & (peak > 1.5 * blank_rates.mean(axis=0))

    df = pd.DataFrame(
        dict(
            session=session_id,
            unit_id=sel.index.values,
            area=sel.location.values,
            firing_rate=sel.firing_rate.values,
            peak_rate=peak,
            blank_rate=blank_rates.mean(axis=0),
            responsive=responsive,
            gOSI=gosi,
            gOSI_null=null_osi,
            gOSI_corrected=gosi - null_osi,
            p_OSI=p_osi,
            gDSI=gdsi,
            gDSI_null=null_dsi,
            gDSI_corrected=gdsi - null_dsi,
            p_DSI=p_dsi,
            OSI=osi,
            DSI=dsi,
            pref_ori=pref_ori,
            pref_dir=pref_dir,
            pref_tf=pref_tf,
            half1_ori=h1,
            half2_ori=h2,
            sg_pref_ori=pref_sg,
            sg_gOSI=gosi_sg,
            sg_peak_rate=sg_peak,
        )
    )
    df["group"] = area_group(df.area)

    if with_glm:
        dur = drive.stop_time.values - drive.start_time.values - ONSET_LAG
        counts = np.rint(rates * dur[:, None])
        ll = gl.cross_validated_loglik(counts, ang, seed=seed)
        for k, v in ll.items():
            df[f"ll_{k}"] = v
        df["glm_label"] = gl.classify(ll)
        # false-positive control: the same procedure on shuffled stimulus labels
        rng = np.random.default_rng(seed + 99)
        ll_shuf = gl.cross_validated_loglik(counts, rng.permutation(ang), seed=seed)
        df["glm_label_shuffled"] = gl.classify(ll_shuf)

    io.close()
    if keep_raw:
        raw = dict(rates=rates, ang=ang, tf=tf, uniq=uniq, mean=mean, sem=sem, spikes=spikes,
                   drive=drive, blank=blank, sg_rates=sg_rates, sg_ang=sg_ang, sg_mean=ms,
                   unit_ids=sel.index.values, blank_rates=blank_rates)
        return df, raw
    return df, None
