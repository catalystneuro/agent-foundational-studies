"""Shared helpers for the hippocampal place-cell analysis of DANDI:000044.

Grosmark & Buzsaki (2016), "Diversity in neural firing dynamics supports both
rigid and learned hippocampal sequences".  Eight sessions of bilateral CA1
silicon-probe recordings in four rats running on linear or circular mazes,
flanked by pre- and post-task sleep epochs.
"""

import json
import os

import h5py
import numpy as np
import pandas as pd
import pynapple as nap
import remfile
from pynwb import NWBHDF5IO

CACHE_DIR = os.environ.get("REMFILE_CACHE", "/tmp/remfile_cache")
URL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "session_urls.json")

# Analysis parameters, fixed once here so every figure uses the same numbers.
BIN_WIDTH = 0.02  # m, spatial bin
SPEED_THRESH = 0.05  # m/s, minimum running speed
SPEED_SMOOTH_S = 0.25  # s, boxcar applied to the speed trace
MIN_RUN_DUR = 0.4  # s, discard very short run fragments
MIN_TRAVERSAL_SPAN = 0.5  # m, a traversal must cover at least this much track
MIN_TRAVERSALS = 10  # per direction, else that direction is not analysed
TC_SMOOTH_BINS = 1.5  # gaussian sigma (in bins) for the displayed rate maps
MIN_MEAN_RATE = 0.1  # Hz over the maze epoch, to keep a unit at all
MAX_MEAN_RATE = 5.0  # Hz, exclude fast-firing units from the pyramidal pool
N_SHUFFLE = 200
MIN_PEAK_RATE = 1.0  # Hz, in-field peak of the smoothed rate map
MIN_RELIABILITY = 0.4  # odd/even traversal map correlation
MIN_SPIKES = 30  # spikes emitted while running in that direction
DIRS = ("right", "left")


def session_urls():
    return json.load(open(URL_FILE))


def load_session(url):
    """Stream one NWB file from the DANDI S3 bucket and wrap it for pynapple."""
    rem = remfile.File(url, disk_cache=remfile.DiskCache(CACHE_DIR))
    h5 = h5py.File(rem, "r")
    io = NWBHDF5IO(file=h5, load_namespaces=True)
    nwbfile = io.read()
    return nap.NWBFile(nwbfile), nwbfile, io


def maze_name(nwbfile):
    """e.g. '1.6mLinearMaze', 'CircularMaze', '2mLinearMaze'."""
    b = nwbfile.processing["behavior"]
    key = [k for k in b.data_interfaces if k.endswith("LinearizedPosition")][0]
    return key.replace("LinearizedPosition", "")


def get_position(nwb, nwbfile):
    """Linearized track position as a 1D Tsd, with untracked samples dropped.

    The NWB SpatialSeries is only defined while the animal is on the track
    proper; samples in the reward areas at either end are NaN. Returns
    (Tsd, track_length_in_metres).
    """
    key = maze_name(nwbfile) + "LinearizedTimeSeries"
    tsdframe = nwb[key]
    v = np.asarray(tsdframe.values).ravel()
    t = np.asarray(tsdframe.t)
    ok = np.isfinite(v)
    pos = nap.Tsd(t=t[ok], d=v[ok])
    return pos, float(np.ceil(np.nanmax(v) / BIN_WIDTH) * BIN_WIDTH)


def get_position_2d(nwb, nwbfile):
    """Raw 2D tracking (x, y) in meters, NaN samples dropped."""
    tsdframe = nwb[maze_name(nwbfile) + "SpatialSeries"]
    v = np.asarray(tsdframe.values)
    t = np.asarray(tsdframe.t)
    ok = np.isfinite(v).all(axis=1)
    return nap.TsdFrame(t=t[ok], d=v[ok], columns=["x", "y"])


def n_bins(track_length):
    return int(round(track_length / BIN_WIDTH))


def _boxcar(x, n):
    if n < 2:
        return x
    return np.convolve(x, np.ones(n) / n, mode="same")


def _mask_to_intervalset(t, mask, med_dt):
    """Contiguous True runs of `mask` -> IntervalSet, breaking on sampling gaps."""
    m = mask.astype(int) * np.concatenate([[True], np.diff(t) < 5 * med_dt])
    edges = np.diff(np.concatenate([[0], m, [0]]))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0] - 1
    keep = ends > starts
    starts, ends = starts[keep], ends[keep]
    if len(starts) == 0:
        return nap.IntervalSet(start=np.array([]), end=np.array([]))
    return nap.IntervalSet(start=t[starts], end=t[ends])


def running_epochs(pos, speed_thresh=SPEED_THRESH, min_dur=MIN_RUN_DUR):
    """Split the track traversals into rightward and leftward running epochs.

    Returns (velocity Tsd, dict of IntervalSets keyed 'right'/'left'/'run').
    """
    t = pos.t
    # Gaps between traversals are long (the animal sits in the reward area), so
    # differentiate within contiguous stretches only.
    dt = np.diff(t)
    med_dt = np.median(dt)
    seg_bounds = np.concatenate([[0], np.where(dt > 5 * med_dt)[0] + 1, [len(t)]])

    vel = np.full(len(t), np.nan)
    for a, b in zip(seg_bounds[:-1], seg_bounds[1:]):
        if b - a < 3:
            continue
        vel[a:b] = np.gradient(pos.values[a:b], t[a:b])
    nsmooth = max(1, int(round(SPEED_SMOOTH_S / med_dt)))
    for a, b in zip(seg_bounds[:-1], seg_bounds[1:]):
        if b - a > nsmooth:
            vel[a:b] = _boxcar(vel[a:b], nsmooth)

    ok = np.isfinite(vel)
    vel_tsd = nap.Tsd(t=t[ok], d=vel[ok])

    eps = {}
    for name, mask in [
        ("right", vel_tsd.values > speed_thresh),
        ("left", vel_tsd.values < -speed_thresh),
    ]:
        ep = _mask_to_intervalset(vel_tsd.t, mask, med_dt).drop_short_intervals(min_dur)
        # A traversal must cover a decent stretch of track; this rejects the
        # brief in-and-out jitter at the reward areas.
        keep = []
        for k in range(len(ep)):
            p = pos.restrict(ep[k : k + 1])
            if len(p) > 2 and (p.values.max() - p.values.min()) >= MIN_TRAVERSAL_SPAN:
                keep.append(k)
        eps[name] = nap.IntervalSet(start=ep.start[keep], end=ep.end[keep])
    eps["run"] = eps["right"].union(eps["left"])
    return vel_tsd, eps


def select_pyramidal(units, maze_ep, min_rate=MIN_MEAN_RATE, max_rate=MAX_MEAN_RATE):
    """Putative CA1 pyramidal cells: labelled excitatory, with moderate rates."""
    rates = units.restrict(maze_ep).rate
    ctype = units.get_info("cell_type")
    keep = [
        i for i in units.index if ctype[i] == "excitatory" and min_rate <= rates[i] <= max_rate
    ]
    return units[keep]


def tuning_curves(units, pos, ep, track_length):
    """Occupancy-normalized 1D rate maps via pynapple.

    Returns (DataFrame [bin x unit] in Hz, occupancy in seconds, bin edges).
    """
    fs = 1.0 / np.median(np.diff(pos.t))
    xr = nap.compute_tuning_curves(
        units, pos, bins=n_bins(track_length), range=[(0.0, track_length)], epochs=ep, fs=fs
    )
    bins = np.asarray(xr.attrs["bin_edges"]).ravel()
    occ = np.asarray(xr.attrs["occupancy"]).ravel().astype(float) / fs
    tc = pd.DataFrame(
        np.asarray(xr.values).T, index=0.5 * (bins[:-1] + bins[1:]), columns=list(units.index)
    )
    return tc, occ, bins


def smooth_tc(tc, sigma=TC_SMOOTH_BINS):
    """Gaussian-smooth each column of a tuning-curve DataFrame (no wrap-around)."""
    if sigma <= 0:
        return tc
    half = int(np.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma) ** 2)
    k /= k.sum()
    out = tc.copy()
    for c in tc.columns:
        y = np.nan_to_num(tc[c].values.astype(float), nan=0.0)
        pad = np.concatenate([np.full(half, y[0]), y, np.full(half, y[-1])])
        out[c] = np.convolve(pad, k, mode="same")[half:-half]
    return out


def spatial_info_bits_per_spike(tc, occ):
    """Skaggs spatial information (bits/spike) for each column of `tc`."""
    p = occ / occ.sum()
    out = {}
    for c in tc.columns:
        lam = np.nan_to_num(tc[c].values.astype(float), nan=0.0)
        lam_bar = np.sum(p * lam)
        if lam_bar <= 0:
            out[c] = 0.0
            continue
        nz = lam > 0
        ratio = lam[nz] / lam_bar
        out[c] = float(np.sum(p[nz] * ratio * np.log2(ratio)))
    return pd.Series(out)


def sparsity(tc, occ):
    """Spatial sparsity: <lam>^2 / <lam^2>; low values mean a compact field."""
    p = occ / occ.sum()
    out = {}
    for c in tc.columns:
        lam = np.nan_to_num(tc[c].values.astype(float), nan=0.0)
        den = np.sum(p * lam**2)
        out[c] = float(np.sum(p * lam) ** 2 / den) if den > 0 else np.nan
    return pd.Series(out)


def field_stats(tc, bins):
    """Peak rate, peak location, and field width (contiguous >=50% of peak)."""
    centers = 0.5 * (bins[:-1] + bins[1:])
    bw = centers[1] - centers[0]
    rows = []
    for c in tc.columns:
        lam = np.nan_to_num(tc[c].values.astype(float), nan=0.0)
        i = int(np.argmax(lam))
        peak = lam[i]
        if peak <= 0:
            rows.append((c, 0.0, np.nan, np.nan))
            continue
        thr = 0.5 * peak
        lo, hi = i, i
        while lo > 0 and lam[lo - 1] >= thr:
            lo -= 1
        while hi < len(lam) - 1 and lam[hi + 1] >= thr:
            hi += 1
        rows.append((c, peak, centers[i], (hi - lo + 1) * bw))
    return pd.DataFrame(
        rows, columns=["unit", "peak_rate", "peak_pos", "width"]
    ).set_index("unit")


def map_corr(a, b):
    """Pearson r between two rate maps; NaN when either map is flat."""
    a, b = np.nan_to_num(np.asarray(a, float)), np.nan_to_num(np.asarray(b, float))
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def shuffled_si_threshold(
    units, pos, ep, occ, track_length, n_shuffle=N_SHUFFLE, rng=None, pct=95
):
    """Null distribution of spatial information from circularly shifted spikes.

    Each unit's spike train is rolled by a random offset along a timeline formed
    by concatenating the running epochs. That preserves the spike count and the
    fine-scale ISI structure while destroying the alignment to position.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    ep_dur = ep.tot_length()
    starts, ends = np.asarray(ep.start), np.asarray(ep.end)
    cum = np.concatenate([[0], np.cumsum(ends - starts)])

    def to_flat(ts):
        idx = np.clip(np.searchsorted(starts, ts, side="right") - 1, 0, len(starts) - 1)
        return cum[idx] + (ts - starts[idx])

    def from_flat(x):
        idx = np.clip(np.searchsorted(cum[1:], x, side="right"), 0, len(starts) - 1)
        return starts[idx] + (x - cum[idx])

    flat = {i: to_flat(np.asarray(units[i].restrict(ep).t)) for i in units.index}
    null = {i: np.zeros(n_shuffle) for i in units.index}
    for s in range(n_shuffle):
        shift = rng.uniform(0.1 * ep_dur, 0.9 * ep_dur)
        shifted = {}
        for i, fl in flat.items():
            if len(fl) == 0:
                shifted[i] = nap.Ts(t=np.array([]), time_support=ep)
                continue
            shifted[i] = nap.Ts(t=np.sort(from_flat(np.mod(fl + shift, ep_dur))), time_support=ep)
        grp = nap.TsGroup(shifted, time_support=ep)
        tc, _, _ = tuning_curves(grp, pos, ep, track_length)
        # Smooth exactly as the observed maps are smoothed, otherwise the null is
        # inflated (smoothing lowers spatial information) and the test loses power.
        si = spatial_info_bits_per_spike(smooth_tc(tc), occ)
        for i in units.index:
            null[i][s] = si[i]
    thr = pd.Series({i: np.percentile(null[i], pct) for i in units.index})
    return thr, null


def maze_epoch(nwbfile):
    df = nwbfile.epochs.to_dataframe()
    row = df[df.label == "MazeEpoch"]
    return nap.IntervalSet(start=row.start_time.values, end=row.stop_time.values)


def direction_metrics(pyr, pos, ep, track_length, n_shuffle=N_SHUFFLE, rng=None):
    """All place-field metrics for one running direction.

    A unit is called a place cell when it satisfies four criteria at once:
    spatial information above its own circular-shift null, a peak rate of at
    least MIN_PEAK_RATE, split-half (odd vs even traversal) map correlation
    above MIN_RELIABILITY, and at least MIN_SPIKES spikes while running.  The
    spike-count and peak-rate criteria matter because spatial information per
    spike is strongly inflated for units that fire only a handful of times.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    tc, occ, bins = tuning_curves(pyr, pos, ep, track_length)
    tcs = smooth_tc(tc)
    si = spatial_info_bits_per_spike(tcs, occ)
    spars = sparsity(tcs, occ)
    thr, _ = shuffled_si_threshold(
        pyr, pos, ep, occ, track_length, n_shuffle=n_shuffle, rng=rng
    )
    fstats = field_stats(tcs, bins)

    odd = nap.IntervalSet(start=ep.start[::2], end=ep.end[::2])
    even = nap.IntervalSet(start=ep.start[1::2], end=ep.end[1::2])
    tc_o = smooth_tc(tuning_curves(pyr, pos, odd, track_length)[0])
    tc_e = smooth_tc(tuning_curves(pyr, pos, even, track_length)[0])
    rel = pd.Series({i: map_corr(tc_o[i].values, tc_e[i].values) for i in tc.columns})
    nspk = pd.Series({i: len(pyr[i].restrict(ep)) for i in pyr.index})

    df = pd.DataFrame(
        {
            "si": si,
            "si_thresh": thr,
            "sparsity": spars,
            "reliability": rel,
            "n_spikes": nspk,
            "peak_rate": fstats["peak_rate"],
            "peak_pos": fstats["peak_pos"],
            "width": fstats["width"],
        }
    )
    # Selection without the reliability criterion. Comparing a within-direction
    # split-half correlation against a between-direction correlation is only fair
    # if the selection did not itself put a floor on the former, so the
    # directionality analysis uses this mask instead.
    df["is_place_cell_norel"] = (
        (df.si > df.si_thresh)
        & (df.peak_rate >= MIN_PEAK_RATE)
        & (df.n_spikes >= MIN_SPIKES)
    )
    df["is_place_cell"] = df.is_place_cell_norel & (df.reliability >= MIN_RELIABILITY)
    return {"tc": tc, "tc_smooth": tcs, "occ": occ, "bins": bins, "metrics": df}


def prepare(name, url):
    """Load a session and derive everything the analysis needs."""
    nwb, nwbfile, io = load_session(url)
    maze = maze_epoch(nwbfile)
    pos, track_length = get_position(nwb, nwbfile)
    vel, eps = running_epochs(pos)
    units = nwb["units"]
    pyr = select_pyramidal(units, maze)
    return dict(
        name=name,
        nwb=nwb,
        nwbfile=nwbfile,
        io=io,
        subject=nwbfile.subject.subject_id,
        maze=maze,
        maze_type=maze_name(nwbfile),
        pos=pos,
        track_length=track_length,
        vel=vel,
        eps=eps,
        units=units,
        pyr=pyr,
    )


def analyze_session(name, url, n_shuffle=N_SHUFFLE, seed=0, verbose=True):
    """Full single-session place-cell analysis. Returns a dict of results."""
    s = prepare(name, url)
    eps, pos, pyr, L = s["eps"], s["pos"], s["pyr"], s["track_length"]
    res = {
        "session": name,
        "subject": s["subject"],
        "maze_type": s["maze_type"],
        "track_length": L,
        "maze_duration": float(s["maze"].tot_length()),
        "n_units": len(s["units"]),
        "n_pyramidal": len(pyr),
        "n_traversals": {d: len(eps[d]) for d in DIRS},
        "run_duration": {d: float(eps[d].tot_length()) for d in DIRS + ("run",)},
        "unit_index": np.asarray(pyr.index),
        "location": np.asarray([pyr.get_info("location")[i] for i in pyr.index]),
        "per_direction": {},
    }
    if verbose:
        print(
            f"{name} ({s['subject']}, {s['maze_type']}, {L:.2f} m): "
            f"{len(s['units'])} units -> {len(pyr)} pyramidal; "
            f"{len(eps['right'])} rightward / {len(eps['left'])} leftward traversals"
        )

    rng = np.random.default_rng(seed)
    for d in DIRS:
        if len(eps[d]) < MIN_TRAVERSALS:
            if verbose:
                print(f"  {d}: only {len(eps[d])} traversals, skipped")
            continue
        out = direction_metrics(pyr, pos, eps[d], L, n_shuffle=n_shuffle, rng=rng)
        res["per_direction"][d] = {
            "tc_smooth": out["tc_smooth"].values,
            "bins": out["bins"],
            "occupancy": out["occ"],
            "metrics": out["metrics"],
        }
        if verbose:
            m = out["metrics"]
            print(
                f"  {d}: {int(m.is_place_cell.sum())}/{len(pyr)} place cells, "
                f"median SI {np.median(m.si[m.is_place_cell]):.2f} bits/spike, "
                f"median width {100 * np.median(m.width[m.is_place_cell]):.0f} cm"
            )

    analysed = list(res["per_direction"])
    pc = np.zeros(len(pyr), bool)
    for d in analysed:
        pc |= res["per_direction"][d]["metrics"].is_place_cell.values
    res["is_place_cell_either"] = pc
    res["directions_analysed"] = analysed

    if len(analysed) == 2:
        a = res["per_direction"]["right"]["tc_smooth"]
        b = res["per_direction"]["left"]["tc_smooth"]
        res["dir_corr"] = np.array([map_corr(a[:, j], b[:, j]) for j in range(a.shape[1])])
    else:
        res["dir_corr"] = np.full(len(pyr), np.nan)
    s["io"].close()
    return res
