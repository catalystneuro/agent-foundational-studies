"""Shared helpers: load session, define run bouts, map spikes to positions."""
import numpy as np
import h5py
import remfile
import pynapple as nap
from pynwb import NWBHDF5IO

S3_URL = "https://dandiarchive.s3.amazonaws.com/blobs/763/2d8/7632d81b-2819-473d-8946-34dc939e6028"
CACHE = "/tmp/remfile_cache_placefields02"
MIN_BOUT_S = 1.0
MERGE_GAP_S = 0.3
MIN_SPAN_M = 0.3
MIN_SPEED = 0.15


def load_session():
    disk_cache = remfile.DiskCache(CACHE)
    h5py_file = h5py.File(remfile.File(S3_URL, disk_cache=disk_cache), "r")
    nwb = nap.NWBFile(NWBHDF5IO(file=h5py_file).read())
    return nwb, h5py_file


def compute_bouts(t, x):
    """Run bouts from the linearized series.

    Returns list of (i0, i1, direction) index ranges into t/x, and dt.
    """
    dt = np.median(np.diff(t))
    valid = ~np.isnan(x)
    d = np.diff(valid.astype(int))
    starts = list(np.where(d == 1)[0] + 1)
    ends = list(np.where(d == -1)[0] + 1)
    if valid[0]:
        starts = [0] + starts
    if valid[-1]:
        ends = ends + [len(valid)]
    segs = list(zip(starts, ends))
    merged = []
    for s, e in segs:
        if merged and t[s] - t[merged[-1][1] - 1] < MERGE_GAP_S:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    bouts = []
    for s, e in merged:
        m = ~np.isnan(x[s:e])
        xb, tb = x[s:e][m], t[s:e][m]
        dur = t[e - 1] - t[s]
        if dur < MIN_BOUT_S or len(xb) < 5:
            continue
        if xb.max() - xb.min() < MIN_SPAN_M:
            continue
        if np.median(np.abs(np.gradient(xb, tb))) < MIN_SPEED:
            continue
        bouts.append((s, e, 1 if xb[-1] > xb[0] else -1))
    return bouts, dt


def bout_axis(t, x, bouts):
    """Concatenated run-bout time axis. Returns tau (per-sample), offsets, total."""
    dt = np.median(np.diff(t))
    tau = np.full(len(t), np.nan)
    offsets = []
    acc = 0.0
    for s, e, _ in bouts:
        offsets.append(acc)
        tau[s:e] = acc + (t[s:e] - t[s])
        acc += tau[e - 1] - tau[s] + dt
    return tau, np.array(offsets), acc


def spike_maps(st, t, x, bouts, offsets):
    """Map spike times to (tau, position, bout direction); drops out-of-bout spikes."""
    b_starts = np.array([t[s] for s, e, _ in bouts])
    b_ends = np.array([t[e - 1] for s, e, _ in bouts])
    dirs = np.array([b[2] for b in bouts])
    idx = np.clip(np.searchsorted(b_starts, st, side="right") - 1, 0, len(bouts) - 1)
    inb = (st >= b_starts[idx]) & (st <= b_ends[idx])
    st_in = st[inb]
    idx_in = idx[inb]
    tau_s = offsets[idx_in] + (st_in - b_starts[idx_in])
    # position at spike time: interpolate within the bout's valid samples
    x_s = np.empty(len(st_in))
    for i, (ii, sti) in enumerate(zip(idx_in, st_in)):
        s, e, _ = bouts[ii]
        m = ~np.isnan(x[s:e])
        x_s[i] = np.interp(sti, t[s:e][m], x[s:e][m])
    return tau_s, x_s, dirs[idx_in]
