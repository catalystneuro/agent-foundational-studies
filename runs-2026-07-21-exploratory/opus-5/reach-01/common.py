"""Shared loading utilities for the MC_Maze reach-tuning analysis (DANDI 000128).

Streams the NWB file from the DANDI S3 bucket with remfile + a local disk cache,
then caches the extracted arrays (kinematics, spike times, trial table) to a local
.npz/.pkl so that downstream scripts start instantly.
"""

import os
import pickle

import numpy as np
import pandas as pd

DANDISET_ID = "000128"
ASSET_PATH = "sub-Jenkins/sub-Jenkins_ses-full_desc-train_behavior+ecephys.nwb"
REMFILE_CACHE = "/tmp/remfile_cache"
LOCAL_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mc_maze_cache.pkl")

# Analysis windows (seconds relative to movement onset)
MOVE_WIN = (-0.05, 0.35)      # generous peri-movement window (trial bookkeeping)
KIN_WIN = (0.00, 0.25)        # kinematics window defining the reach direction/speed
NEURAL_WIN = (-0.10, 0.15)    # spike-count window, shifted 100 ms earlier (M1 leads)


def get_asset_url():
    from dandi.dandiapi import DandiAPIClient

    with DandiAPIClient() as client:
        ds = client.get_dandiset(DANDISET_ID, "draft")
        asset = ds.get_asset_by_path(ASSET_PATH)
        return asset.get_content_url(follow_redirects=1, strip_query=True)


def _download():
    import h5py
    import remfile
    from pynwb import NWBHDF5IO

    os.makedirs(REMFILE_CACHE, exist_ok=True)
    url = get_asset_url()
    print(f"streaming {url}")
    rf = remfile.File(url, disk_cache=remfile.DiskCache(REMFILE_CACHE))
    h5 = h5py.File(rf, "r")
    io = NWBHDF5IO(file=h5, load_namespaces=True)
    nwbfile = io.read()

    beh = nwbfile.processing["behavior"]
    hv = beh["hand_vel"]
    hp = beh["hand_pos"]

    print("  reading kinematics ...")
    t_kin = np.asarray(hv.timestamps[:], dtype=np.float64)
    # conversion 0.001: stored in mm and mm/s -> SI metres, m/s
    hand_vel = np.asarray(hv.data[:], dtype=np.float64) * hv.conversion
    hand_pos = np.asarray(hp.data[:], dtype=np.float64) * hp.conversion

    print("  reading units ...")
    units = nwbfile.units
    spike_times = [np.asarray(s, dtype=np.float64) for s in units["spike_times"][:]]
    obs = np.asarray(units["obs_intervals"][0])  # identical across units in this file
    elec_tbl = nwbfile.electrodes.to_dataframe()
    # NOTE: pynwb mis-resolves this DynamicTableRegion (it hands back the ragged
    # index rather than the data), so read the row indices from HDF5 directly.
    # Second quirk of this file: the stored indices are *within-array* channel
    # numbers (0-95) rather than global electrode rows, so the sequence restarts
    # when the unit list moves from the first array to the second. We recover the
    # global row by detecting that reset. Electrode rows 0-95 are PMd, 96-191 M1.
    chan = np.asarray(h5["units/electrodes"][:], dtype=int)
    assert len(chan) == len(units)
    resets = np.flatnonzero(np.diff(chan) < 0)
    assert len(resets) == 1, f"expected one array boundary, found {resets}"
    n_per_array = len(elec_tbl) // 2
    assert chan.max() < n_per_array
    array_id = np.zeros(len(chan), dtype=int)
    array_id[resets[0] + 1:] = 1
    elec_rows = array_id * n_per_array + chan
    area = elec_tbl["location"].values[elec_rows].astype(str)
    heldout = np.asarray(units["heldout"][:], dtype=bool)

    print("  reading trials ...")
    trials = nwbfile.trials.to_dataframe()
    # resolve the reached target from the target list + active_target index
    tgt = np.full((len(trials), 2), np.nan)
    for i, (tp, ai) in enumerate(zip(trials["target_pos"].values, trials["active_target"].values)):
        tp = np.atleast_2d(np.asarray(tp))
        if 0 <= ai < len(tp):
            tgt[i] = tp[ai]
    trials = trials.drop(columns=["target_pos", "barrier_pos"])
    trials["target_x"] = tgt[:, 0] * 1e-3
    trials["target_y"] = tgt[:, 1] * 1e-3

    payload = dict(
        t_kin=t_kin,
        hand_vel=hand_vel,
        hand_pos=hand_pos,
        spike_times=spike_times,
        obs_intervals=obs,
        area=area,
        heldout=heldout,
        trials=trials,
        n_elec_m1=int((elec_tbl["location"] == "M1").sum()),
        n_elec_pmd=int((elec_tbl["location"] == "PMd").sum()),
        session_description=nwbfile.session_description,
    )
    return payload


def load_raw(force=False):
    """Return the raw payload dict, using the on-disk pickle cache when available."""
    if os.path.exists(LOCAL_CACHE) and not force:
        with open(LOCAL_CACHE, "rb") as fh:
            return pickle.load(fh)
    payload = _download()
    with open(LOCAL_CACHE, "wb") as fh:
        pickle.dump(payload, fh, protocol=4)
    return payload


def load_pynapple(force=False):
    """Load the session as pynapple objects.

    Returns
    -------
    dict with keys:
        units    : nap.TsGroup   (182 sorted units, metadata: area, heldout)
        vel      : nap.TsdFrame  (hand velocity, columns vx/vy, m/s)
        pos      : nap.TsdFrame  (hand position, columns x/y, m)
        speed    : nap.Tsd       (hand speed, m/s)
        obs_ep   : nap.IntervalSet (intervals in which spikes were observed)
        trials   : pd.DataFrame
    """
    import pynapple as nap

    p = load_raw(force=force)

    obs_ep = nap.IntervalSet(start=p["obs_intervals"][:, 0], end=p["obs_intervals"][:, 1])

    units = nap.TsGroup(
        {i: nap.Ts(t=st) for i, st in enumerate(p["spike_times"])},
        time_support=obs_ep,
        area=p["area"],
        heldout=p["heldout"],
    )

    vel = nap.TsdFrame(t=p["t_kin"], d=p["hand_vel"], columns=["vx", "vy"], time_support=obs_ep)
    pos = nap.TsdFrame(t=p["t_kin"], d=p["hand_pos"], columns=["x", "y"], time_support=obs_ep)
    speed = nap.Tsd(t=vel.t, d=np.linalg.norm(vel.values, axis=1), time_support=obs_ep)

    return dict(
        units=units,
        vel=vel,
        pos=pos,
        speed=speed,
        obs_ep=obs_ep,
        trials=p["trials"],
        meta=p,
    )


def trial_table(data, min_speed=0.05):
    """Build the per-trial analysis table.

    Adds the empirically measured reach direction (angle of hand displacement over
    the peri-movement window), the peak speed, and the target direction.
    """
    tr = data["trials"].copy()
    pos, speed = data["pos"], data["speed"]

    onset = tr["move_onset_time"].values
    t0, t1 = onset + MOVE_WIN[0], onset + MOVE_WIN[1]

    px = np.asarray(pos["x"].values)
    py = np.asarray(pos["y"].values)
    tk = pos.t
    i0 = np.searchsorted(tk, t0)
    i1 = np.searchsorted(tk, t1)
    i0 = np.clip(i0, 0, len(tk) - 1)
    i1 = np.clip(i1, 0, len(tk) - 1)

    dx = px[i1] - px[i0]
    dy = py[i1] - py[i0]
    tr["reach_dir"] = np.arctan2(dy, dx)
    tr["reach_dist"] = np.hypot(dx, dy)
    tr["target_dir"] = np.arctan2(tr["target_y"] - py[i0], tr["target_x"] - px[i0])

    sp = np.asarray(speed.values)
    tr["peak_speed"] = np.array([sp[a:b].max() if b > a else np.nan for a, b in zip(i0, i1)])
    tr["mean_speed"] = np.array([sp[a:b].mean() if b > a else np.nan for a, b in zip(i0, i1)])

    tr["move_start"] = t0
    tr["move_stop"] = t1
    tr["straight"] = tr["num_barriers"].values == 0

    good = (
        tr["success"].values
        & np.isfinite(tr["reach_dir"].values)
        & (tr["peak_speed"].values > min_speed)
        & (tr["move_stop"].values < data["obs_ep"].end[-1])
    )
    tr["use"] = good
    return tr


def move_epochs(tr):
    import pynapple as nap

    t = tr[tr["use"]]
    return nap.IntervalSet(start=t["move_start"].values, end=t["move_stop"].values)


def counts_in_epochs(units, starts, stops):
    """Spike counts for every unit in each [start, stop) window -> (n_win, n_units)."""
    n_win = len(starts)
    out = np.zeros((n_win, len(units)), dtype=np.float64)
    for j, k in enumerate(units.keys()):
        st = units[k].t
        out[:, j] = np.searchsorted(st, stops) - np.searchsorted(st, starts)
    return out


def rates_in_epochs(units, starts, stops):
    return counts_in_epochs(units, starts, stops) / (stops - starts)[:, None]


def add_movement_kinematics(data, tr):
    """Per-trial reach direction and speed, from the mean hand-velocity vector.

    Direction is the angle of the *mean velocity vector* over KIN_WIN after
    movement onset, and speed is the mean of the instantaneous speed over the
    same window. Using measured hand velocity rather than the nominal target
    keeps the analysis honest on the curved maze trials, where the hand does not
    travel straight to the target.
    """
    vel, speed = data["vel"], data["speed"]
    onset = tr["move_onset_time"].values
    tk = vel.t
    i0 = np.clip(np.searchsorted(tk, onset + KIN_WIN[0]), 0, len(tk) - 1)
    i1 = np.clip(np.searchsorted(tk, onset + KIN_WIN[1]), 0, len(tk) - 1)

    vx, vy = np.asarray(vel["vx"].values), np.asarray(vel["vy"].values)
    sp = np.asarray(speed.values)
    mvx = np.array([vx[a:b].mean() if b > a else np.nan for a, b in zip(i0, i1)])
    mvy = np.array([vy[a:b].mean() if b > a else np.nan for a, b in zip(i0, i1)])
    tr = tr.copy()
    tr["mv_dir"] = np.arctan2(mvy, mvx)
    tr["mv_speed"] = np.array([sp[a:b].mean() if b > a else np.nan for a, b in zip(i0, i1)])
    tr["mv_x"], tr["mv_y"] = mvx, mvy
    return tr


def fit_cosine(rates, theta):
    """Least-squares cosine tuning fit  r = b0 + b1*cos(th) + b2*sin(th).

    rates : (n_trials, n_units).  Returns dict of per-unit parameters, with the
    preferred direction PD = atan2(b2, b1) and modulation depth hypot(b1, b2).
    """
    X = np.column_stack([np.ones_like(theta), np.cos(theta), np.sin(theta)])
    beta, *_ = np.linalg.lstsq(X, rates, rcond=None)
    with np.errstate(all="ignore"):        # BLAS emits spurious matmul warnings here
        pred = X @ beta
    ss_res = ((rates - pred) ** 2).sum(0)
    ss_tot = ((rates - rates.mean(0)) ** 2).sum(0)
    return dict(
        b0=beta[0], b1=beta[1], b2=beta[2],
        pd=np.arctan2(beta[2], beta[1]),
        mod_depth=np.hypot(beta[1], beta[2]),
        r2=1 - ss_res / np.where(ss_tot == 0, np.nan, ss_tot),
    )


BIN = 0.02          # bin size (s) for the continuous velocity analysis
BEST_LAG = 0.10     # neural activity leads the hand by this much (s); measured in step 3


def bin_spikes(data, bin_size=BIN):
    """Spike counts in fixed bins inside the observed epochs.

    Returns (counts float32 (n_bins, n_units), bin centres t_c).
    """
    cnt = data["units"].count(bin_size, ep=data["obs_ep"])
    return np.asarray(cnt.values, dtype=np.float32), np.asarray(cnt.t)


def velocity_at(data, t_c, bin_size=BIN, lag=BEST_LAG):
    """Mean hand velocity in each bin, shifted forward in time by `lag`.

    A positive `lag` pairs the spikes in a bin with the hand velocity `lag`
    seconds *later*, i.e. it assumes the neurons lead the movement.
    Returns (vxy (n_bins, 2), valid mask).
    """
    vel = data["vel"]
    tk = vel.t
    cs = np.concatenate([np.zeros((1, 2)), np.cumsum(np.asarray(vel.values), axis=0)])

    lo = np.searchsorted(tk, t_c - bin_size / 2 + lag)
    hi = np.searchsorted(tk, t_c + bin_size / 2 + lag)
    n = hi - lo
    valid = n > 0
    vxy = (cs[hi] - cs[lo]) / np.where(valid, n, 1)[:, None]
    vxy[~valid] = np.nan
    # reject bins whose velocity window straddles a gap in the kinematics record
    ii = np.clip(hi - 1, 0, len(tk) - 1)
    jj = np.clip(lo, 0, len(tk) - 1)
    valid &= np.where(valid, tk[ii] - tk[jj], 0) < 2 * bin_size
    return vxy, valid


def binned_dataset(data, bin_size=BIN, lag=BEST_LAG):
    counts, t_c = bin_spikes(data, bin_size)
    vxy, valid = velocity_at(data, t_c, bin_size, lag)
    return counts, vxy, t_c, valid


def cosine_speed_fit(counts, vxy, bin_size=BIN):
    """Linear velocity model  rate = b0 + bx*vx + by*vy  fit per unit.

    Under this model the rate is a cosine of direction whose amplitude grows in
    proportion to speed, which is exactly the classical velocity-tuning claim.
    """
    X = np.column_stack([np.ones(len(vxy)), vxy])
    y = counts / bin_size
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    with np.errstate(all="ignore"):
        pred = X @ beta
    ss_res = ((y - pred) ** 2).sum(0)
    ss_tot = ((y - y.mean(0)) ** 2).sum(0)
    return dict(b0=beta[0], bx=beta[1], by=beta[2],
                pd=np.arctan2(beta[2], beta[1]),
                gain=np.hypot(beta[1], beta[2]),          # Hz per (m/s)
                r2=1 - ss_res / ss_tot)


def run_ids(t_c, bin_size=BIN):
    """Label contiguous runs of bins (bins are not contiguous across trial gaps)."""
    return np.concatenate([[0], np.cumsum(np.diff(t_c) > 1.5 * bin_size)])


def lag_matrix(counts, t_c, lag_bins, bin_size=BIN):
    """Stack spike counts at several bin lags; column block j is counts shifted by
    lag_bins[j] bins into the past. Rows that would reach across a gap are invalid."""
    rid = run_ids(t_c, bin_size)
    n, k = counts.shape
    X = np.zeros((n, k * len(lag_bins)), dtype=np.float32)
    valid = np.ones(n, bool)
    idx = np.arange(n)
    for j, L in enumerate(lag_bins):
        src = np.clip(idx - L, 0, n - 1)
        ok = (idx - L >= 0) & (rid[src] == rid)
        X[:, j * k:(j + 1) * k] = counts[src]
        X[~ok, j * k:(j + 1) * k] = 0.0
        valid &= ok
    return X, valid


def poisson_pseudo_r2(y, rate, rate_null):
    """Per-column McFadden pseudo-R^2 for Poisson counts (rates in counts/bin)."""
    eps = 1e-9

    def ll(lam):
        lam = np.maximum(lam, eps)
        return (y * np.log(lam) - lam).sum(0)

    ll_model = ll(rate)
    ll_null = ll(np.broadcast_to(rate_null, y.shape))
    ll_sat = np.where(y > 0, y * np.log(np.maximum(y, eps)) - y, 0).sum(0)
    denom = ll_sat - ll_null
    return (ll_model - ll_null) / np.where(denom == 0, np.nan, denom)


def contiguous_folds(t_c, n_folds=5, bin_size=BIN):
    """Contiguous time-block CV folds, so train and test are not neighbouring bins."""
    blocks = np.array_split(np.arange(len(t_c)), n_folds)
    for i in range(n_folds):
        test = blocks[i]
        train = np.concatenate([blocks[j] for j in range(n_folds) if j != i])
        yield train, test


AREA_COLORS = {"M1": "#1b6ca8", "PMd": "#d1495b"}
