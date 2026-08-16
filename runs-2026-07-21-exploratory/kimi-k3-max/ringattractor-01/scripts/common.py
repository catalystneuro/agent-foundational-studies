"""Shared utilities for the ring-attractor HD analysis (DANDI 000056, Peyrache et al. 2015)."""
import numpy as np
import h5py
import remfile
import pynapple as nap
from pynwb import NWBHDF5IO

DANDI_VERSION = "0.250624.0430"

# One session per mouse, chosen for small file size and good HD-cell yield
# (yield = HD cells / total units, established in prior exploration).
SESSIONS = {
    "Mouse28-140310": "656704ea-a4cd-40f4-8158-a6533ebf2eee",  # 20/47 HD
    "Mouse25-140123": "bdb30f7d-ba69-4d2e-8504-2efab69cd8d7",  # 14/34 HD
    "Mouse17-130128": "4cc64fe0-7b1e-404c-8b86-fb5659292830",  # 7/39 HD
    "Mouse20-130514": "748aa5de-c0de-4aa7-a7ef-2aad2f87a7eb",  # 6/17 HD
    "Mouse24-131213": "ada02790-6eb6-48ee-902d-9ba017303586",  # 6/22 HD
}

DISK_CACHE = remfile.DiskCache("/tmp/remfile_cache_hd")


def asset_url(asset_id):
    return f"https://api.dandiarchive.org/api/assets/{asset_id}/download/"


def load_session(asset_id):
    """Stream an NWB session from DANDI via remfile and return the pynapple NWBFile."""
    url = asset_url(asset_id)
    rem_file = remfile.File(url, disk_cache=DISK_CACHE)
    h5py_file = h5py.File(rem_file, "r")
    io = NWBHDF5IO(file=h5py_file)
    nwbfile = io.read()
    nwb = nap.NWBFile(nwbfile)
    return nwb


def get_hd_angle(nwb):
    """Head direction from the two LEDs. Tracking failures are sentinel -1.

    Returns a Tsd of HD angle in radians [0, 2pi), NaN where tracking failed.
    """
    red = nwb["SubjectPosition/RedLED"]
    blue = nwb["SubjectPosition/BlueLED"]
    red_xy = np.asarray(red.values)
    blue_xy = np.asarray(blue.values)
    good = (red_xy[:, 0] > 0) & (red_xy[:, 1] > 0) & (blue_xy[:, 0] > 0) & (blue_xy[:, 1] > 0)
    dx = red_xy[:, 0] - blue_xy[:, 0]
    dy = red_xy[:, 1] - blue_xy[:, 1]
    ang = np.arctan2(dy, dx) % (2 * np.pi)
    ang[~good] = np.nan
    return nap.Tsd(t=red.t, d=ang)


def get_states(nwb):
    """Return dict of IntervalSets keyed by state label."""
    states = nwb["states"]
    out = {}
    for label in np.unique(states["label"]):
        out[label] = states[states["label"] == label]
    return out


def sorted_units(nwb):
    """Units as a TsGroup with per-unit spike times sorted (one Mouse17 unit is unsorted)."""
    units = nwb["units"]
    data = {k: nap.Ts(np.sort(units[k].t)) for k in units.keys()}
    return nap.TsGroup(data)


def circ_mean(angles, w=None, axis=0):
    """Weighted circular mean of angles in radians."""
    s = np.nansum(w * np.sin(angles), axis=axis) if w is not None else np.nansum(np.sin(angles), axis=axis)
    c = np.nansum(w * np.cos(angles), axis=axis) if w is not None else np.nansum(np.cos(angles), axis=axis)
    return np.arctan2(s, c) % (2 * np.pi)


def mean_vector_length(angles, w=None):
    s = np.sum(w * np.sin(angles)) if w is not None else np.sum(np.sin(angles))
    c = np.sum(w * np.cos(angles)) if w is not None else np.sum(np.cos(angles))
    n = np.sum(w) if w is not None else len(angles)
    return np.sqrt(s**2 + c**2) / n


def flat_tuning(rates):
    """Scale each unit's tuning curve (NNLS, w>=0) so the population rate
    landscape sum_i w_i * rate_i(x) is ~constant across angles.

    The Bayesian decoder's Poisson normalization term exp(-dt * sum_i rate_i(x))
    otherwise acts as a static bias toward under-represented directions, which
    both real and control decodes share. Flattening removes that bias.
    Returns (scaled_rates, scales).
    """
    from scipy.optimize import nnls
    tc = np.nan_to_num(rates, nan=0.0)
    A = tc.T  # bins x cells
    target = np.full(A.shape[0], tc.sum(axis=0).mean())
    w, _ = nnls(A, target)
    return tc * w[:, None], w
