"""Shared loading and analysis helpers for the reach direction / velocity tuning analysis.

Data source: DANDI Archive, streamed with remfile + a local disk cache.
  - Dandiset 000128 (MC_Maze): monkey Jenkins, delayed center-out / maze reaching,
    182 sorted units across PMd and M1 Utah arrays, 1 kHz hand kinematics.
  - Dandiset 000129 (MC_RTT): monkey Indy, self-paced random-target reaching,
    130 sorted units in M1, 1 kHz fingertip kinematics.
"""

import os

import numpy as np
import pynapple as nap
import remfile
import requests
import h5py
from pynwb import NWBHDF5IO

REMFILE_CACHE = os.environ.get("REMFILE_CACHE", "/tmp/remfile_cache")
NPZ_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")

MC_MAZE = ("000128", "26e85f09-39b7-480f-b337-278a8f034007")
MC_RTT = ("000129", "2ae6bf3c-788b-4ece-8c01-4b4a5680b25b")


def dandi_s3_url(dandiset_id, asset_id):
    """Resolve a DANDI asset to its S3 URL without downloading it."""
    url = (
        f"https://api.dandiarchive.org/api/dandisets/{dandiset_id}"
        f"/versions/draft/assets/{asset_id}/download/"
    )
    return requests.get(url, allow_redirects=False).headers["Location"]


def open_nwb(dandiset_id, asset_id):
    """Stream an NWB file from DANDI. Returns (nwbfile, h5py_file)."""
    os.makedirs(REMFILE_CACHE, exist_ok=True)
    rem = remfile.File(
        dandi_s3_url(dandiset_id, asset_id), disk_cache=remfile.DiskCache(REMFILE_CACHE)
    )
    h5 = h5py.File(rem, "r")
    nwbfile = NWBHDF5IO(file=h5, load_namespaces=True).read()
    return nwbfile, h5


def unit_areas(h5):
    """Per-unit recording area, dereferenced from the raw HDF5 electrode indices.

    Returns ``(labels, resolved)``. In MC_Maze the electrode table lists 192
    channels (96 PMd + 96 M1) but every one of the 182 units points into the
    first 96 rows, so the array-of-origin cannot be recovered from the file. In
    that case ``resolved`` is False and every unit is labelled "M1/PMd".
    """
    elec_idx = h5["units/electrodes"][:]
    locations = h5["general/extracellular_ephys/electrodes/location"][:]
    locations = np.array([s.decode() if isinstance(s, bytes) else s for s in locations])
    labels = locations[elec_idx]
    n_areas_in_table = len(np.unique(locations))
    resolved = len(np.unique(labels)) == n_areas_in_table
    if not resolved:
        labels = np.array(["/".join(sorted(np.unique(locations)))] * len(elec_idx))
    return labels, resolved


def contiguous_epochs(timestamps, max_gap=0.002):
    """IntervalSet covering stretches of regularly sampled behaviour.

    The kinematics in MC_Maze are only recorded inside trials, so the timestamp
    vector contains gaps that must not be interpolated across.
    """
    breaks = np.flatnonzero(np.diff(timestamps) > max_gap)
    starts = np.concatenate([[timestamps[0]], timestamps[breaks + 1]])
    ends = np.concatenate([timestamps[breaks], [timestamps[-1]]])
    return nap.IntervalSet(start=starts, end=ends)


def load_mc_maze(verbose=True):
    """Load MC_Maze into pynapple objects, caching the heavy kinematic arrays."""
    os.makedirs(NPZ_CACHE, exist_ok=True)
    npz = os.path.join(NPZ_CACHE, "mc_maze.npz")

    nwbfile, h5 = open_nwb(*MC_MAZE)
    trials = nwbfile.trials.to_dataframe()
    areas, areas_resolved = unit_areas(h5)
    if verbose and not areas_resolved:
        print("note: unit-to-array assignment is not recoverable from this file; "
              "units are treated as one motor-cortical population.")
    spike_times = [np.asarray(st) for st in nwbfile.units["spike_times"][:]]

    if os.path.exists(npz):
        z = np.load(npz)
        t, pos, vel = z["t"], z["pos"], z["vel"]
    else:
        beh = nwbfile.processing["behavior"]
        if verbose:
            print("streaming 1 kHz kinematics (~270 MB, cached after first run)...")
        t = np.asarray(beh["hand_pos"].timestamps[:])
        pos = np.asarray(beh["hand_pos"].data[:])
        vel = np.asarray(beh["hand_vel"].data[:])
        np.savez_compressed(npz, t=t, pos=pos, vel=vel)

    epochs = contiguous_epochs(t)
    units = nap.TsGroup(
        {i: nap.Ts(st) for i, st in enumerate(spike_times)},
        time_support=epochs,
        area=areas,
    )
    # Stored in mm and mm/s (NWB conversion factor 0.001 against meters).
    hand_pos = nap.TsdFrame(t=t, d=pos, columns=["x", "y"], time_support=epochs)
    hand_vel = nap.TsdFrame(t=t, d=vel, columns=["vx", "vy"], time_support=epochs)
    return dict(
        nwbfile=nwbfile,
        units=units,
        hand_pos=hand_pos,
        hand_vel=hand_vel,
        trials=trials,
        epochs=epochs,
        areas=areas,
    )


def load_mc_rtt(verbose=True):
    """Load MC_RTT (self-paced random-target reaching) into pynapple objects."""
    os.makedirs(NPZ_CACHE, exist_ok=True)
    npz = os.path.join(NPZ_CACHE, "mc_rtt.npz")

    nwbfile, h5 = open_nwb(*MC_RTT)
    spike_times = [np.asarray(st) for st in nwbfile.units["spike_times"][:]]
    beh = nwbfile.processing["behavior"]

    if os.path.exists(npz):
        z = np.load(npz)
        t, pos, vel = z["t"], z["pos"], z["vel"]
    else:
        if verbose:
            print("streaming MC_RTT kinematics...")
        fv = beh["finger_vel"]
        n = fv.data.shape[0]
        t = fv.starting_time + np.arange(n) / fv.rate
        vel = np.asarray(fv.data[:])
        pos = np.asarray(beh["cursor_pos"].data[:])
        np.savez_compressed(npz, t=t, pos=pos, vel=vel)

    epochs = contiguous_epochs(t)
    units = nap.TsGroup({i: nap.Ts(st) for i, st in enumerate(spike_times)}, time_support=epochs)
    cursor_pos = nap.TsdFrame(t=t, d=pos, columns=["x", "y"], time_support=epochs)
    finger_vel = nap.TsdFrame(t=t, d=vel, columns=["vx", "vy"], time_support=epochs)
    return dict(
        nwbfile=nwbfile,
        units=units,
        cursor_pos=cursor_pos,
        finger_vel=finger_vel,
        epochs=epochs,
    )


# --------------------------------------------------------------------------
# Fast windowed spike counting
# --------------------------------------------------------------------------

def window_rates(spike_lists, starts, stops):
    """Firing rate of every unit inside each [start, stop) window.

    Returns an (n_windows, n_units) array in spikes/s. Uses searchsorted rather
    than repeated ``restrict`` calls, which matters when there are thousands of
    windows and millions of behavioural samples.
    """
    dur = np.asarray(stops) - np.asarray(starts)
    out = np.empty((len(starts), len(spike_lists)))
    for j, st in enumerate(spike_lists):
        out[:, j] = np.searchsorted(st, stops) - np.searchsorted(st, starts)
    return out / dur[:, None]


def peri_event_matrix(t, values, events, offsets):
    """Sample a 1-D signal at ``events[:, None] + offsets[None, :]``."""
    return np.interp(np.asarray(events)[:, None] + offsets[None, :], t, values)


# --------------------------------------------------------------------------
# Tuning-curve fitting
# --------------------------------------------------------------------------

def fit_cosine(theta, rate):
    """Least-squares cosine fit  r = b0 + b1*cos(theta) + b2*sin(theta).

    Parameters
    ----------
    theta : (n_obs,) reach directions in radians
    rate : (n_obs, n_units) firing rates

    Returns
    -------
    dict with preferred direction, modulation depth, baseline and R^2 per unit.
    """
    rate = np.atleast_2d(rate.T).T if rate.ndim == 1 else rate
    X = np.column_stack([np.ones_like(theta), np.cos(theta), np.sin(theta)])
    beta, *_ = np.linalg.lstsq(X, rate, rcond=None)
    pred = X.dot(beta)
    ss_res = ((rate - pred) ** 2).sum(0)
    ss_tot = ((rate - rate.mean(0)) ** 2).sum(0)
    return dict(
        baseline=beta[0],
        pd=np.arctan2(beta[2], beta[1]),
        depth=np.hypot(beta[1], beta[2]),
        r2=1 - ss_res / np.where(ss_tot == 0, np.nan, ss_tot),
        pred=pred,
    )


def cosine_permutation_test(theta, rate, n_perm=1000, seed=0):
    """Permutation p-value for the depth of cosine directional tuning."""
    rng = np.random.default_rng(seed)
    obs = fit_cosine(theta, rate)["depth"]
    null = np.empty((n_perm, rate.shape[1]))
    for i in range(n_perm):
        null[i] = fit_cosine(theta[rng.permutation(theta.size)], rate)["depth"]
    return obs, (1 + (null >= obs).sum(0)) / (1 + n_perm)


def circ_diff(a, b):
    """Signed angular difference a - b wrapped to (-pi, pi]."""
    return (a - b + np.pi) % (2 * np.pi) - np.pi


def circ_mean(a):
    """Circular mean of angles."""
    return np.angle(np.exp(1j * np.asarray(a)).mean())


def circ_corr(a, b):
    """Jammalamadaka circular-circular correlation coefficient."""
    da, db = np.sin(a - circ_mean(a)), np.sin(b - circ_mean(b))
    return (da * db).sum() / np.sqrt((da ** 2).sum() * (db ** 2).sum())


def rate_map_2d(counts, f1, f2, edges1, edges2, bin_size, min_occupancy=20):
    """Occupancy-normalised 2-D firing-rate maps.

    Parameters
    ----------
    counts : (n_samples, n_units) spike counts per time bin
    f1, f2 : (n_samples,) behavioural features sampled at the same times
    edges1, edges2 : bin edges for the two features
    bin_size : duration of one time bin, in seconds
    min_occupancy : cells visited fewer times than this are set to NaN

    Returns
    -------
    (n_bins1, n_bins2, n_units) array of firing rates in Hz.
    """
    n1, n2 = len(edges1) - 1, len(edges2) - 1
    i1 = np.digitize(f1, edges1) - 1
    i2 = np.digitize(f2, edges2) - 1
    ok = (i1 >= 0) & (i1 < n1) & (i2 >= 0) & (i2 < n2)
    flat = i1[ok] * n2 + i2[ok]
    occ = np.bincount(flat, minlength=n1 * n2).astype(float)
    maps = np.empty((n1 * n2, counts.shape[1]))
    for u in range(counts.shape[1]):
        maps[:, u] = np.bincount(flat, weights=counts[ok, u], minlength=n1 * n2)
    with np.errstate(invalid="ignore", divide="ignore"):
        maps = maps / (occ[:, None] * bin_size)
    maps[occ < min_occupancy] = np.nan
    return maps.reshape(n1, n2, counts.shape[1]), occ.reshape(n1, n2)


def epoch_index(epochs, t):
    """Index of the epoch containing each time point, or -1 if outside."""
    i = np.searchsorted(epochs.start, t, side="right") - 1
    out = np.where((i >= 0) & (t <= epochs.end[np.clip(i, 0, None)]), i, -1)
    return out


def continuous_velocity_tuning(units, vel, epochs, bin_size=0.02, smooth_std=0.04,
                               speed_min=50.0, min_spikes=300,
                               lags=np.arange(-0.30, 0.401, 0.02)):
    """Lag-resolved linear velocity tuning from continuously sampled kinematics.

    Bins spikes, smooths them into a rate estimate, and for each candidate lag
    regresses each unit's rate on the hand velocity sampled ``lag`` seconds
    later (positive lag = neural activity leads the movement). Returns the
    per-unit optimal lag, the population optimum, and the velocity fit at that
    optimum.
    """
    counts = units.count(bin_size, ep=epochs)
    rate = np.asarray((counts.smooth(std=smooth_std) / bin_size).values)
    tb = np.asarray(counts.t)
    tv, vx_raw, vy_raw = np.asarray(vel.t), vel.values[:, 0], vel.values[:, 1]

    def sample(lag):
        tq = tb + lag
        inside = epoch_index(epochs, tq) >= 0
        return np.interp(tq, tv, vx_raw), np.interp(tq, tv, vy_raw), inside

    def fit(vx, vy, mask, Y):
        X = np.column_stack([np.ones(mask.sum()), vx[mask], vy[mask]])
        Ym = Y[mask]
        beta, *_ = np.linalg.lstsq(X, Ym, rcond=None)
        ss_tot = ((Ym - Ym.mean(0)) ** 2).sum(0)
        ss_res = ((Ym - X.dot(beta)) ** 2).sum(0)
        return 1 - ss_res / np.where(ss_tot > 0, ss_tot, np.nan), beta

    vx0, vy0, inside0 = sample(0.0)
    moving = inside0 & (np.hypot(vx0, vy0) > speed_min)
    keep_u = np.asarray(counts.values)[moving].sum(0) >= min_spikes
    R = rate[:, keep_u]

    r2_lag = np.empty((len(lags), keep_u.sum()))
    for k, lag in enumerate(lags):
        vx, vy, inside = sample(lag)
        r2_lag[k], _ = fit(vx, vy, inside & (np.hypot(vx, vy) > speed_min), R)

    pop_curve = np.nanmedian(r2_lag, axis=1)
    pop_lag = lags[pop_curve.argmax()]
    vx, vy, inside = sample(pop_lag)
    speed = np.hypot(vx, vy)
    mask = inside & (speed > speed_min)
    r2_lin, beta = fit(vx, vy, mask, R)
    return dict(counts=counts, tb=tb, keep_u=keep_u, uidx=np.flatnonzero(keep_u),
                lags=lags, r2_lag=r2_lag, best_lag=lags[r2_lag.argmax(0)],
                pop_curve=pop_curve, pop_lag=pop_lag, r2_lin=r2_lin, beta_lin=beta,
                pd_cont=np.arctan2(beta[2], beta[1]), mask=mask, vx=vx, vy=vy,
                speed=speed, theta=np.arctan2(vy[mask], vx[mask]), bin_size=bin_size)


def speed_gain_curves(counts_masked, theta, speed, pd, bin_size,
                      speed_edges=np.array([50, 100, 150, 200, 275, 350, 450, 600, 800]),
                      min_samples=50):
    """Firing rate versus speed, separately for movements near and opposite the PD."""
    centres = (speed_edges[:-1] + speed_edges[1:]) / 2
    dth = circ_diff(theta[None, :], np.asarray(pd)[:, None])
    near, away = np.abs(dth) < np.pi / 4, np.abs(dth) > 3 * np.pi / 4
    sp_bin = np.digitize(speed, speed_edges) - 1
    C = counts_masked / bin_size
    n_u = C.shape[1]
    g_near = np.full((n_u, centres.size), np.nan)
    g_away = np.full_like(g_near, np.nan)
    for u in range(n_u):
        for b in range(centres.size):
            m1, m2 = near[u] & (sp_bin == b), away[u] & (sp_bin == b)
            if m1.sum() > min_samples:
                g_near[u, b] = C[m1, u].mean()
            if m2.sum() > min_samples:
                g_away[u, b] = C[m2, u].mean()
    return centres, g_near, g_away
