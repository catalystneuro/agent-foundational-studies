"""Shared loading and analysis utilities for orientation-selectivity analysis
of DANDI:000021 (Allen Institute Visual Coding - Neuropixels, Brain Observatory 1.1).

All data access is streamed from the DANDI S3 mirror with remfile + a local disk
cache; nothing is downloaded in full.
"""

import os
import numpy as np
import pandas as pd
import requests
import h5py
import remfile
import pynapple as nap

DANDISET = "000021"
CACHE_DIR = os.environ.get("REMFILE_CACHE", "/tmp/remfile_cache_o21")
ASSET_API = "https://api.dandiarchive.org/api/dandisets/{d}/versions/draft/assets/"

VISUAL_CORTEX = ["VISp", "VISl", "VISrl", "VISal", "VISam", "VISpm"]
THALAMUS = ["LGd", "LP"]
CONTROL = ["CA1", "DG", "CA3"]


# --------------------------------------------------------------------------- #
# Asset discovery
# --------------------------------------------------------------------------- #
def list_session_assets():
    """Return a DataFrame of session-level NWB files (one row per session)."""
    import re

    rows, url, params = [], ASSET_API.format(d=DANDISET), {"page_size": 100}
    while url:
        r = requests.get(url, params=params).json()
        rows.extend(r["results"])
        url, params = r.get("next"), None
    df = pd.DataFrame(rows)
    is_session = df["path"].str.match(r"sub-\d+/sub-\d+_ses-\d+\.nwb$")
    df = df[is_session].copy()
    df["session_id"] = df["path"].str.extract(r"ses-(\d+)")
    df["subject_id"] = df["path"].str.extract(r"sub-(\d+)")
    df["url"] = "https://api.dandiarchive.org/api/assets/" + df["asset_id"] + "/download/"
    return df.sort_values("session_id").reset_index(drop=True)


def open_session(url):
    """Open a streamed NWB file and return the raw h5py handle."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    disk_cache = remfile.DiskCache(CACHE_DIR)
    rem_file = remfile.File(url, disk_cache=disk_cache)
    return h5py.File(rem_file, "r")


# --------------------------------------------------------------------------- #
# Units
# --------------------------------------------------------------------------- #
# Allen Institute default quality-metric thresholds for "good" sorted units.
QC = dict(isi_violations=0.5, amplitude_cutoff=0.1, presence_ratio=0.9)


def unit_table(f):
    """Units metadata table with the brain region resolved via peak channel."""
    u = f["units"]
    el = f["general"]["extracellular_ephys"]["electrodes"]
    chan2loc = dict(zip(el["id"][:], el["location"][:].astype(str)))

    df = pd.DataFrame(
        {
            "unit_id": u["id"][:],
            "peak_channel_id": u["peak_channel_id"][:],
            "quality": u["quality"][:].astype(str),
            "isi_violations": u["isi_violations"][:],
            "amplitude_cutoff": u["amplitude_cutoff"][:],
            "presence_ratio": u["presence_ratio"][:],
            "snr": u["snr"][:],
            "firing_rate": u["firing_rate"][:],
            "waveform_duration": u["waveform_duration"][:],
        }
    )
    df["region"] = [chan2loc.get(c, "unknown") for c in df["peak_channel_id"]]
    df["pass_qc"] = (
        (df["quality"] == "good")
        & (df["isi_violations"] < QC["isi_violations"])
        & (df["amplitude_cutoff"] < QC["amplitude_cutoff"])
        & (df["presence_ratio"] > QC["presence_ratio"])
    )
    df["row"] = np.arange(len(df))
    return df


def load_spikes(f, units_df, time_support=None, progress=True):
    """Load spike times for the rows of `units_df` into a pynapple TsGroup.

    `spike_times` is stored contiguously with a ragged index, so each unit is a
    single byte-range read from S3.

    NWB unit ids in this dandiset are NOT monotonic with row order, and TsGroup
    sorts by key, so the metadata table is returned re-ordered to match the
    TsGroup key order. Always use the returned table, never the input one.
    """
    from tqdm.auto import tqdm

    u = f["units"]
    idx = u["spike_times_index"][:]
    st = u["spike_times"]
    uid = u["id"][:]
    data = {}
    it = tqdm(units_df["row"].values, desc="loading spike trains", disable=not progress)
    for r in it:
        lo = 0 if r == 0 else idx[r - 1]
        hi = idx[r]
        data[int(uid[r])] = nap.Ts(t=st[lo:hi])
    tsg = nap.TsGroup(data, time_support=time_support)
    order = list(tsg.keys())
    df = units_df.set_index("unit_id").loc[order].reset_index()
    assert list(df["unit_id"]) == order
    return tsg, df


# --------------------------------------------------------------------------- #
# Stimulus tables
# --------------------------------------------------------------------------- #
def stim_table(f, name):
    """Read a stimulus presentation interval table into a DataFrame."""
    g = f["intervals"][name]
    out = {"start_time": g["start_time"][:], "stop_time": g["stop_time"][:]}
    for k in ["orientation", "temporal_frequency", "spatial_frequency", "contrast",
              "phase", "stimulus_block"]:
        if k in g:
            v = g[k][:]
            if v.dtype.kind in "SO":
                v = pd.to_numeric(pd.Series(v.astype(str)), errors="coerce").values
            out[k] = v
    df = pd.DataFrame(out)
    df["duration"] = df["stop_time"] - df["start_time"]
    return df


def drifting_gratings(f):
    """Valid drifting-grating trials (blank sweeps dropped) + the blank-sweep table."""
    df = stim_table(f, "drifting_gratings_presentations")
    blank = df[df["orientation"].isna()].reset_index(drop=True)
    trials = df[df["orientation"].notna()].reset_index(drop=True)
    trials["direction"] = trials["orientation"] % 360.0
    trials["ori"] = trials["direction"] % 180.0
    return trials, blank


def static_gratings(f):
    df = stim_table(f, "static_gratings_presentations")
    blank = df[df["orientation"].isna()].reset_index(drop=True)
    trials = df[df["orientation"].notna()].reset_index(drop=True)
    trials["ori"] = trials["orientation"] % 180.0
    return trials, blank


def to_intervalset(df):
    return nap.IntervalSet(start=df["start_time"].values, end=df["stop_time"].values)


# --------------------------------------------------------------------------- #
# Trial spike counts
# --------------------------------------------------------------------------- #
def trial_rates(tsgroup, trials):
    """Firing rate (Hz) of every unit on every trial.

    Returns an (n_trials, n_units) array. Uses a single merged binning pass:
    spike times are searchsorted into trial windows, which is far faster than
    restricting the TsGroup once per trial.
    """
    starts = trials["start_time"].values
    stops = trials["stop_time"].values
    order = np.argsort(starts)
    assert np.all(order == np.arange(len(starts))), "trials must be time-sorted"
    dur = stops - starts

    counts = np.zeros((len(starts), len(tsgroup)), dtype=float)
    for j, k in enumerate(tsgroup.keys()):
        t = tsgroup[k].t
        lo = np.searchsorted(t, starts, side="left")
        hi = np.searchsorted(t, stops, side="right")
        counts[:, j] = hi - lo
    return counts / dur[:, None], counts


def condition_means(rates, labels):
    """Mean rate per condition. Returns (unique_labels, mean[n_cond, n_units],
    sem[n_cond, n_units], n_per_cond)."""
    uniq = np.unique(labels)
    mu = np.zeros((len(uniq), rates.shape[1]))
    sem = np.zeros_like(mu)
    n = np.zeros(len(uniq), dtype=int)
    for i, c in enumerate(uniq):
        m = labels == c
        n[i] = m.sum()
        mu[i] = rates[m].mean(0)
        sem[i] = rates[m].std(0, ddof=1) / np.sqrt(n[i]) if n[i] > 1 else np.nan
    return uniq, mu, sem, n


# --------------------------------------------------------------------------- #
# Selectivity metrics
# --------------------------------------------------------------------------- #
def global_osi(directions_deg, rates):
    """Global OSI (1 - circular variance at 2*theta) and preferred orientation.

    rates: (n_dir,) or (n_dir, n_units). Negative rates are clipped to 0.
    """
    r = np.clip(np.atleast_2d(np.asarray(rates, float).T).T, 0, None)
    th = np.deg2rad(np.asarray(directions_deg, float))
    w = r / np.where(r.sum(0) == 0, np.nan, r.sum(0))
    vec = (w * np.exp(2j * th[:, None])).sum(0)
    osi = np.abs(vec)
    pref = (np.rad2deg(np.angle(vec)) / 2.0) % 180.0
    return osi, pref


def global_dsi(directions_deg, rates):
    """Global DSI (1 - circular variance at theta) and preferred direction."""
    r = np.clip(np.atleast_2d(np.asarray(rates, float).T).T, 0, None)
    th = np.deg2rad(np.asarray(directions_deg, float))
    w = r / np.where(r.sum(0) == 0, np.nan, r.sum(0))
    vec = (w * np.exp(1j * th[:, None])).sum(0)
    return np.abs(vec), np.rad2deg(np.angle(vec)) % 360.0


def classic_osi_dsi(directions_deg, rates):
    """Classic two-point OSI = (Rpref-Rorth)/(Rpref+Rorth) and
    DSI = (Rpref-Rnull)/(Rpref+Rnull), evaluated at the peak direction."""
    d = np.asarray(directions_deg, float)
    r = np.asarray(rates, float)
    if r.ndim == 1:
        r = r[:, None]
    ipref = np.argmax(r, axis=0)
    rpref = r[ipref, np.arange(r.shape[1])]

    def at(target):
        j = np.abs((d[:, None] - target[None, :] + 180) % 360 - 180).argmin(0)
        return r[j, np.arange(r.shape[1])]

    dpref = d[ipref]
    rorth = 0.5 * (at((dpref + 90) % 360) + at((dpref - 90) % 360))
    rnull = at((dpref + 180) % 360)
    with np.errstate(invalid="ignore", divide="ignore"):
        osi = (rpref - rorth) / (rpref + rorth)
        dsi = (rpref - rnull) / (rpref + rnull)
    return osi, dsi, dpref


def permutation_p(rates_trials, labels, n_perm=1000, seed=0, metric="gosi"):
    """Permutation test: is the observed selectivity larger than chance?

    rates_trials: (n_trials, n_units); labels: (n_trials,) direction in degrees.
    Shuffles the direction label across trials and recomputes the metric.
    """
    rng = np.random.default_rng(seed)
    uniq, mu, _, _ = condition_means(rates_trials, labels)
    fn = global_osi if metric == "gosi" else global_dsi
    obs = fn(uniq, mu)[0]
    ge = np.zeros_like(obs)
    lab = labels.copy()
    for _ in range(n_perm):
        rng.shuffle(lab)
        _, mus, _, _ = condition_means(rates_trials, lab)
        ge += (fn(uniq, mus)[0] >= obs).astype(float)
    return (ge + 1) / (n_perm + 1), obs


def circ_dist_180(a, b):
    """Absolute difference between two orientations in degrees, wrapped to [0, 90]."""
    d = np.abs((np.asarray(a) - np.asarray(b)) % 180.0)
    return np.minimum(d, 180.0 - d)


# --------------------------------------------------------------------------- #
# Von Mises tuning-curve fit
# --------------------------------------------------------------------------- #
def double_von_mises(theta_deg, base, amp1, amp2, kappa, mu_deg):
    """Two opposite lobes 180 deg apart: captures orientation tuning with a
    possible direction bias. kappa sets the width."""
    th = np.deg2rad(theta_deg)
    mu = np.deg2rad(mu_deg)
    return (
        base
        + amp1 * np.exp(kappa * (np.cos(th - mu) - 1))
        + amp2 * np.exp(kappa * (np.cos(th - mu - np.pi) - 1))
    )


def fit_von_mises(directions_deg, rates):
    """Least-squares fit of the double von Mises; returns (params, hwhm_deg, r2)."""
    from scipy.optimize import curve_fit

    d = np.asarray(directions_deg, float)
    r = np.asarray(rates, float)
    i = int(np.argmax(r))
    p0 = [r.min(), max(r.max() - r.min(), 1e-3), max(r.max() - r.min(), 1e-3) / 2, 2.0, d[i]]
    # kappa is capped at 9 (HWHM >= 22.5 deg): the stimulus samples direction every
    # 45 deg, so anything narrower than half that spacing is not identifiable and
    # the fit would otherwise place a spike between two measured points.
    amax = max(r.max(), 1e-3) * 3 + 1
    bounds = ([0, 0, 0, 0.05, -360], [max(r.max(), 1e-3) * 2 + 1, amax, amax, 9.0, 720])
    popt, _ = curve_fit(double_von_mises, d, r, p0=p0, bounds=bounds, maxfev=20000)
    pred = double_von_mises(d, *popt)
    ss = ((r - r.mean()) ** 2).sum()
    r2 = 1 - ((r - pred) ** 2).sum() / ss if ss > 0 else np.nan
    kappa = popt[3]
    # half width at half maximum of exp(kappa*(cos x - 1))
    arg = 1 + np.log(0.5) / kappa
    hwhm = np.rad2deg(np.arccos(np.clip(arg, -1, 1)))
    return popt, hwhm, r2


# --------------------------------------------------------------------------- #
# Full per-session analysis (used for both the prototype and the session sweep)
# --------------------------------------------------------------------------- #
def analyze_session(dg_rates, dgb_rates, direction, tf, sg_rates, sg_ori,
                    unit_id, region, n_perm=1000, seed=1, progress=True):
    """Compute per-unit orientation-selectivity metrics for one session.

    Returns (results DataFrame, tuning dict).
    """
    from scipy import stats
    from tqdm.auto import tqdm

    dirs = np.unique(direction)
    tfs = np.unique(tf)
    n_units = dg_rates.shape[1]

    # --- visual responsiveness: best direction x TF condition vs blank sweeps
    cond = np.array([f"{a}_{b}" for a, b in zip(direction, tf)])
    ucond, cmu, _, _ = condition_means(dg_rates, cond)
    best = np.argmax(cmu, axis=0)
    p_resp = np.ones(n_units)
    for j in range(n_units):
        m = cond == ucond[best[j]]
        p_resp[j] = stats.mannwhitneyu(dg_rates[m, j], dgb_rates[:, j],
                                       alternative="greater").pvalue
    # the tested condition is the one with the largest mean rate, chosen on the
    # same data, so the p-value is Bonferroni-corrected for the number of
    # direction x temporal-frequency conditions that selection ranged over
    responsive = (p_resp < 0.01 / len(ucond)) & (cmu.max(0) > 0.5)

    # --- direction tuning at each unit's preferred temporal frequency
    pref_tf = np.zeros(n_units)
    tc = np.zeros((len(dirs), n_units))
    tc_sem = np.zeros_like(tc)
    mask_pref = np.zeros((len(dg_rates), n_units), dtype=bool)
    for j in range(n_units):
        _, mu_tf, _, _ = condition_means(dg_rates[:, [j]], tf)
        pref_tf[j] = tfs[np.argmax(mu_tf[:, 0])]
        m = tf == pref_tf[j]
        mask_pref[:, j] = m
        # evaluate explicitly over all 8 directions: a direction x TF cell can be
        # empty in some sessions, and condition_means would then return a short curve
        for i, dd in enumerate(dirs):
            r = dg_rates[m & (direction == dd), j]
            tc[i, j] = r.mean() if len(r) else np.nan
            tc_sem[i, j] = r.std(ddof=1) / np.sqrt(len(r)) if len(r) > 1 else np.nan
        if np.any(np.isnan(tc[:, j])):
            tc[np.isnan(tc[:, j]), j] = np.nanmean(tc[:, j])

    gosi, pref_ori = global_osi(dirs, tc)
    gdsi, pref_dir = global_dsi(dirs, tc)
    osi, dsi, peak_dir = classic_osi_dsi(dirs, tc)

    # --- Kruskal-Wallis across directions, and a permutation test on gOSI
    p_kw = np.ones(n_units)
    for j in range(n_units):
        m = mask_pref[:, j]
        groups = [dg_rates[m & (direction == dd), j] for dd in dirs]
        groups = [g for g in groups if len(g) > 0]
        p_kw[j] = stats.kruskal(*groups).pvalue

    rng = np.random.default_rng(seed)
    p_perm = np.ones(n_units)
    for j in tqdm(np.where(responsive)[0], desc="gOSI permutation", disable=not progress):
        m = mask_pref[:, j]
        r = dg_rates[m, j]
        lab = direction[m].copy()
        null = np.empty(n_perm)
        for i in range(n_perm):
            rng.shuffle(lab)
            mu = np.array([r[lab == dd].mean() if np.any(lab == dd) else np.nan
                           for dd in dirs])
            mu = np.nan_to_num(mu, nan=np.nanmean(mu))
            null[i] = global_osi(dirs, mu)[0][0]
        p_perm[j] = (np.sum(null >= gosi[j]) + 1) / (n_perm + 1)
    sig_ori = responsive & (p_perm < 0.01)

    # --- static gratings (independent stimulus, 6 orientations, 0.25 s each)
    sg_dirs = np.unique(sg_ori)
    _, sg_tc, sg_sem, _ = condition_means(sg_rates, sg_ori)
    sg_gosi, sg_pref = global_osi(sg_dirs, sg_tc)

    # --- split-half: preferred orientation and cross-validated tuning curve
    # The split is stratified by direction: a purely random split can leave a
    # direction with no trials in one half, which makes the tuning curve ragged.
    rng2 = np.random.default_rng(seed + 100)
    pref_a = np.zeros(n_units)
    pref_b = np.zeros(n_units)
    tc_b_aligned = np.zeros((len(dirs), n_units))
    for j in range(n_units):
        muA = np.zeros(len(dirs))
        muB = np.zeros(len(dirs))
        for i, dd in enumerate(dirs):
            m = np.where(mask_pref[:, j] & (direction == dd))[0]
            perm = rng2.permutation(len(m))
            A, B = m[perm[: len(m) // 2]], m[perm[len(m) // 2:]]
            muA[i] = dg_rates[A, j].mean() if len(A) else np.nan
            muB[i] = dg_rates[B, j].mean() if len(B) else np.nan
        muA = np.nan_to_num(muA, nan=np.nanmean(muA) if np.any(np.isfinite(muA)) else 0.0)
        muB = np.nan_to_num(muB, nan=np.nanmean(muB) if np.any(np.isfinite(muB)) else 0.0)
        pref_a[j] = global_osi(dirs, muA)[1][0]
        pref_b[j] = global_osi(dirs, muB)[1][0]
        # align half B to the peak direction estimated from half A (no circularity)
        tc_b_aligned[:, j] = np.roll(muB, -int(np.argmax(muA)))

    res = pd.DataFrame(dict(
        unit_id=unit_id, region=region, responsive=responsive,
        gosi=gosi, osi=osi, gdsi=gdsi, dsi=dsi,
        pref_ori=pref_ori, pref_dir=pref_dir, peak_dir=peak_dir, pref_tf=pref_tf,
        p_resp=p_resp, p_kw=p_kw, p_perm=p_perm, sig_ori=sig_ori,
        peak_rate=tc.max(0), mean_rate=tc.mean(0), blank_rate=dgb_rates.mean(0),
        sg_gosi=sg_gosi, sg_pref_ori=sg_pref,
        pref_a=pref_a, pref_b=pref_b,
    ))
    res["ori_diff_dg_sg"] = circ_dist_180(res.pref_ori, res.sg_pref_ori)
    res["split_half_diff"] = circ_dist_180(pref_a, pref_b)

    tuning = dict(tc=tc, tc_sem=tc_sem, dirs=dirs, tc_b_aligned=tc_b_aligned,
                  sg_tc=sg_tc, sg_sem=sg_sem, sg_dirs=sg_dirs)
    return res, tuning


def load_and_prepare(url, keep_regions=None, progress=True):
    """Stream one session and return everything the analysis needs."""
    if keep_regions is None:
        keep_regions = VISUAL_CORTEX + THALAMUS + CONTROL
    f = open_session(url)
    units = unit_table(f)
    sel = units[units.pass_qc & units.region.isin(keep_regions)].reset_index(drop=True)
    dg, dg_blank = drifting_gratings(f)
    sg, sg_blank = static_gratings(f)
    t_end = max(dg.stop_time.max(), sg.stop_time.max()) + 60
    ep = nap.IntervalSet(start=0.0, end=t_end)
    spikes, sel = load_spikes(f, sel, time_support=ep, progress=progress)
    dg_rates, _ = trial_rates(spikes, dg)
    sg_rates, _ = trial_rates(spikes, sg)
    dgb_rates, _ = trial_rates(spikes, dg_blank)
    return dict(f=f, spikes=spikes, units=sel, dg=dg, sg=sg, dg_blank=dg_blank,
                dg_rates=dg_rates, sg_rates=sg_rates, dgb_rates=dgb_rates)
