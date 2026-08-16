"""Shared pipeline for reach direction + velocity tuning (MC_Maze, DANDI 000128)."""
import numpy as np
import h5py, remfile, pynapple as nap
from pynwb import NWBHDF5IO

ASSET_ID = "26e85f09-39b7-480f-b337-278a8f034007"
S3_URL = f"https://api.dandiarchive.org/api/assets/{ASSET_ID}/download/"
CACHE = "/tmp/remfile_cache_mcmaze"

# MC_Maze kinematics are labeled m / m/s in NWB but are actually mm / mm/s
# (reach amplitude ~130 "m" but true scale is ~130 mm).  We rescale to mm.
MM_FACTOR = 1000.0

def load_nwb():
    disk_cache = remfile.DiskCache(CACHE)
    h5f = h5py.File(remfile.File(S3_URL, disk_cache=disk_cache), "r")
    nwb = nap.NWBFile(NWBHDF5IO(file=h5f).read())
    return nwb, h5f

def bin_epoch(fr, ep, bin_size):
    """Mean of a (already-restricted) TsdFrame/Tsd over `bin_size` bins within ep.

    Returns (bin_centers, arr (n_bins, n_cols)). Bins are aligned to ep.start.
    """
    t0, t1 = float(ep.start[0]), float(ep.end[0])
    edges = np.arange(t0, t1 + bin_size * 0.99, bin_size)
    ts = np.asarray(fr.t)
    vals = np.asarray(fr)
    idx = np.searchsorted(ts, edges, side="left")
    n_cols = vals.shape[1]
    X = np.full((len(edges) - 1, n_cols), np.nan)
    for k in range(len(edges) - 1):
        lo, hi = idx[k], idx[k + 1]
        if hi > lo:
            X[k] = vals[lo:hi].mean(axis=0)
    return (edges[:-1] + edges[1:]) / 2.0, X

def circular_mean(angles_rad, weights=None, axis=0):
    """Weighted circular mean in radians."""
    ang = np.asarray(angles_rad)
    if weights is None:
        weights = np.ones(ang.shape[0])
    vec = np.exp(1j * ang) * np.asarray(weights)
    R = np.sum(vec, axis=axis)
    return np.angle(R), np.abs(R) / np.sum(weights)
