"""grid_utils.py — shared loading and grid-cell analysis helpers for DANDI 000582.

Sargolini et al. 2006 (Science) MEC dataset. Position values are in cm despite
the NWB unit metadata saying "meters" (1 m box spans +/-50, 1.5 m box +/-75).
"""

import h5py
import numpy as np
import pynapple as nap
import remfile
from pynwb import NWBHDF5IO

from opexebo.analysis import spatial_occupancy, rate_map, autocorrelation, grid_score
from opexebo.general import smooth

DANDI_API_URL = "https://api.dandiarchive.org/api/assets/{asset_id}/download/"
BIN_WIDTH = 2.5          # cm, opexebo default
SMOOTH_SIGMA = 2.0       # bins (5 cm), standard for Moser-lab rate maps
SPEED_THRESHOLD = 2.5    # cm/s, removes LED-jitter stationary epochs
MIN_SPIKES = 200         # minimum spike count for a unit to be analyzed
N_SHUFFLES = 100         # circular time-shift shuffles for candidate grid cells
SHUFFLE_GS_CANDIDATE = 0.3   # only shuffle units with observed gs above this
SHUFFLE_MIN_SHIFT = 20.0     # seconds; minimum circular shift


def load_session(asset_id, cache_dir="/tmp/remfile_cache_grid"):
    """Stream one NWB session from DANDI with a remfile disk cache."""
    disk_cache = remfile.DiskCache(cache_dir)
    rem_file = remfile.File(DANDI_API_URL.format(asset_id=asset_id), disk_cache=disk_cache)
    h5py_file = h5py.File(rem_file, "r")
    io = NWBHDF5IO(file=h5py_file)
    nwbfile = io.read()
    nwb = nap.NWBFile(nwbfile)
    return nwb, io


def get_position(nwb):
    """Return (t, xy) with NaN frames dropped. xy in cm, shape (N, 2)."""
    pos = nwb["SpatialSeriesLED1"]
    t = pos.t
    xy = pos.values
    valid = ~np.isnan(xy[:, 0]) & ~np.isnan(xy[:, 1])
    return t[valid], xy[valid]


def compute_speed(t, xy):
    """Instantaneous speed (cm/s), one value per position frame (first = NaN)."""
    dt = np.diff(t)
    d = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    speed = np.concatenate([[np.nan], d / dt])
    return speed


def session_arena(xy):
    """Arena limits and size from the observed position range."""
    x_min, x_max = np.nanmin(xy[:, 0]), np.nanmax(xy[:, 0])
    y_min, y_max = np.nanmin(xy[:, 1]), np.nanmax(xy[:, 1])
    limits = (x_min, x_max, y_min, y_max)
    arena_size = (x_max - x_min, y_max - y_min)
    return limits, arena_size


def running_position(t, xy, speed):
    """Restrict position to running epochs (speed >= threshold)."""
    keep = ~np.isnan(speed) & (speed >= SPEED_THRESHOLD)
    return t[keep], xy[keep]


def filter_spikes(t, xy, speed, spike_times, t_run):
    """Spikes during running epochs, restricted to the running time range."""
    valid = ~np.isnan(speed)
    spk_speed = np.interp(spike_times, t[valid], speed[valid])
    keep = (spk_speed >= SPEED_THRESHOLD) & (spike_times >= t_run[0]) & (spike_times <= t_run[-1])
    return spike_times[keep]


def rate_map_acorr(occ_map, t_run, xy_run, spk_times, limits, arena_size):
    """Smoothed rate map, autocorrelation, and grid score for one spike train."""
    spk_xy = np.column_stack(
        [np.interp(spk_times, t_run, xy_run[:, 0]),
         np.interp(spk_times, t_run, xy_run[:, 1])]
    )
    spikes_tracking = np.vstack([spk_times, spk_xy.T])  # (3, N): [t, x, y], row major
    rmap = rate_map(occ_map, spikes_tracking, arena_size, bin_width=BIN_WIDTH, limits=limits)
    rmap_smooth = smooth(rmap, SMOOTH_SIGMA)
    acorr = autocorrelation(rmap_smooth)
    score, stats = grid_score(acorr, bin_width=BIN_WIDTH)
    return rmap_smooth, acorr, score, stats, spk_xy


def shuffle_grid_scores(t, speed, t_run, xy_run, occ_map, limits, arena_size,
                        spk_times, n_shuffles=N_SHUFFLES, seed=0):
    """Circular time-shift shuffle null distribution of the grid score.

    Spike times are shifted by a random offset (>= SHUFFLE_MIN_SHIFT s, wrapping
    around the session), which destroys the spike-position relationship while
    preserving the spike train's temporal structure and the position statistics.
    """
    rng = np.random.default_rng(seed)
    t0, t1 = t[0], t[-1]
    duration = t1 - t0
    scores = np.full(n_shuffles, np.nan)
    for i in range(n_shuffles):
        shift = rng.uniform(SHUFFLE_MIN_SHIFT, duration - SHUFFLE_MIN_SHIFT)
        shifted = (spk_times - t0 + shift) % duration + t0
        # filter shifted spikes by speed and range, same as the observed train
        valid = ~np.isnan(speed)
        spk_speed = np.interp(shifted, t[valid], speed[valid])
        keep = (spk_speed >= SPEED_THRESHOLD) & (shifted >= t_run[0]) & (shifted <= t_run[-1])
        spk = shifted[keep]
        if len(spk) < MIN_SPIKES:
            continue
        _, _, score, _, _ = rate_map_acorr(occ_map, t_run, xy_run, spk, limits, arena_size)
        scores[i] = score
    return scores


def analyze_session_units(nwb, run_shuffles=True, seed=0):
    """Analyze all units of one loaded session. Returns (results, meta)."""
    t, xy = get_position(nwb)
    speed = compute_speed(t, xy)
    limits, arena_size = session_arena(xy)
    t_run, xy_run = running_position(t, xy, speed)
    occ_map, coverage, bin_edges = spatial_occupancy(
        t_run, xy_run.T, arena_size, bin_width=BIN_WIDTH, limits=limits
    )

    units = nwb["units"]
    results = {}
    for j, uid in enumerate(units.keys()):
        spk_times = filter_spikes(t, xy, speed, units[uid].t, t_run)
        if len(spk_times) < MIN_SPIKES:
            continue
        rmap, acorr, score, stats, spk_xy = rate_map_acorr(
            occ_map, t_run, xy_run, spk_times, limits, arena_size
        )
        res = {
            "n_spikes": len(spk_times),
            "rate_map": rmap,
            "acorr": acorr,
            "grid_score": score,
            "grid_spacing": stats.get("grid_spacing", np.nan),
            "grid_orientation": stats.get("grid_orientation", np.nan),
            "unit_name": units["unit_name"][uid],
            "layer": units["histology"][uid] if "histology" in units.metadata_columns else "",
            "hemisphere": units["hemisphere"][uid] if "hemisphere" in units.metadata_columns else "",
        }
        if run_shuffles and score > SHUFFLE_GS_CANDIDATE:
            null = shuffle_grid_scores(t, speed, t_run, xy_run, occ_map,
                                       limits, arena_size, spk_times,
                                       seed=seed * 1000 + j)
            res["shuffle_scores"] = null
            res["shuffle_p"] = (np.nansum(null >= score) + 1) / (np.sum(~np.isnan(null)) + 1)
            res["shuffle_z"] = (score - np.nanmean(null)) / np.nanstd(null)
        results[uid] = res

    meta = {"arena_size": arena_size, "coverage": coverage,
            "duration": t[-1] - t[0], "n_frames_run": len(t_run)}
    return results, meta
