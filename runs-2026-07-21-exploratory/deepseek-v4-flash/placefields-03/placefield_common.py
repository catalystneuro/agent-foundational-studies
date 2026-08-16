"""Shared loading + rate-map helpers for the place-field figure scripts."""
import json
import numpy as np
import h5py, remfile
from pynwb import NWBHDF5IO
import pynapple as nap
from scipy.signal.windows import gaussian

FS = 39.0625
T0 = 18079.5
MAZE_END = 20147.0
NBINS = 50
LFP_FS = 1250.0
LFP_CH = 117
LFP_UV = 3.815e-7 * 1e6


def load_all():
    with open("_asset.json") as f:
        info = json.load(f)
    rem_file = remfile.File(info["s3_url"], disk_cache=remfile.DiskCache("/tmp/remfile_cache_000044"))
    h5f = h5py.File(rem_file, "r")
    io = NWBHDF5IO(file=h5f)
    nwb = nap.NWBFile(io.read())
    return nwb, h5f


def build_bouts(nwb):
    """Return times, linearized series, and the run-bout structure."""
    units = nwb["units"]
    lin_raw = np.asarray(nwb["1.6mLinearMazeLinearizedTimeSeries"].values).ravel()
    t = T0 + np.arange(len(lin_raw)) / FS
    good = np.flatnonzero(np.isfinite(lin_raw))
    gaps = np.diff(good)
    be = np.where(gaps > 0.3 * FS)[0]
    starts = np.concatenate([[good[0]], good[be + 1]])
    ends = np.concatenate([good[be], [good[-1]]])
    bouts = []
    for s, e in zip(starts, ends):
        ix = good[(good >= s) & (good <= e)]
        seg = lin_raw[ix]
        if (t[e] - t[s]) >= 1.0 and abs(seg[-1] - seg[0]) > 0.3 \
                and np.median(np.abs(np.diff(seg))) * FS > 0.15:
            bouts.append((s, e, 1 if seg[-1] > seg[0] else 0))
    b_start_i = np.array([b[0] for b in bouts])
    b_end_i = np.array([b[1] for b in bouts])
    b_dir = np.array([b[2] for b in bouts])
    sizes = b_end_i - b_start_i + 1
    bout_offsets = np.concatenate([[0], np.cumsum(sizes)])[:-1]
    tau_run = np.concatenate([np.arange(s) / FS + bout_offsets[i] / FS for i, s in enumerate(sizes)])
    run_lin = np.concatenate([lin_raw[s:e + 1] for s, e in zip(b_start_i, b_end_i)])
    run_dir = np.repeat(b_dir, sizes)
    edges = np.linspace(0, 1.6, NBINS + 1)
    return dict(units=units, lin_raw=lin_raw, t=t, b_start_i=b_start_i, b_end_i=b_end_i,
                b_dir=b_dir, b_start=t[b_start_i], b_end=t[b_end_i], nb=len(bouts),
                bout_offsets=bout_offsets, tau_run=tau_run, run_lin=run_lin,
                run_dir=run_dir, TOTAL=len(run_lin), edges=edges)


def make_spike_lookup(B):
    """Return spike_at(u) -> (lin_pos, dir, tau, wall_t) (cached)."""
    _spk = {}
    def spike_at(u):
        if u in _spk:
            return _spk[u]
        units = B["units"]
        st_ = np.asarray(units[u].t)
        if len(st_) == 0:
            _spk[u] = (np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0))
            return _spk[u]
        bi = np.clip(np.searchsorted(B["b_end"], st_, side="right"), 0, B["nb"] - 1)
        inside = (st_ >= B["b_start"][bi]) & (st_ <= B["b_end"][bi])
        tw = st_[inside]
        tau = B["bout_offsets"][bi[inside]] / FS + (tw - B["b_start"][bi[inside]])
        idx = np.clip(np.searchsorted(B["tau_run"], tau, side="left"), 0, B["TOTAL"] - 1)
        _spk[u] = (B["run_lin"][idx], B["run_dir"][idx], tau, tw)
        return _spk[u]
    return spike_at


def occ_maps(B):
    """Occupancy histograms pooled and per direction (seconds per bin)."""
    dt = 1 / FS
    edges = B["edges"]
    occ_bin = np.histogram(B["run_lin"], bins=edges)[0] * dt
    occ_pos = np.histogram(B["run_lin"][B["run_dir"] == 1], bins=edges)[0] * dt
    occ_neg = np.histogram(B["run_lin"][B["run_dir"] == 0], bins=edges)[0] * dt
    smooth_w = gaussian(NBINS, 1.5)
    smooth_w /= smooth_w.sum()
    return occ_bin, occ_pos, occ_neg, smooth_w


def rate1d(spike_lin, occ, edges, smooth_w):
    cnt = np.histogram(spike_lin, bins=edges)[0]
    r = cnt / np.maximum(occ, 1e-12)
    r = np.convolve(r, smooth_w, mode="same")
    r[occ <= 0] = 0.0
    r[r < 0] = 0.0
    return r


def skaggs_si(rate, occ):
    mask = (occ > 0) & (rate > 0)
    if mask.sum() == 0:
        return 0.0
    p = occ / occ.sum()
    lbda = (p * rate).sum()
    if lbda <= 0:
        return 0.0
    return float(np.sum(p[mask] * (rate[mask] / lbda) * np.log2(rate[mask] / lbda)))