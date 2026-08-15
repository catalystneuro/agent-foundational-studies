"""Shared helpers for the DANDI:000044 hippocampal place-field analysis.

Dandiset 000044 (Grosmark & Buzsaki 2016, "Diversity in neural firing dynamics
supports both rigid and learned hippocampal sequences"): bilateral silicon-probe
recordings from dorsal CA1 of freely moving rats running back and forth on a
linear track, flanked by pre- and post-run sleep sessions.
"""

import json
import os
import re

import h5py
import numpy as np
import pandas as pd
import pynapple as nap
import remfile
from pynwb import NWBHDF5IO
from scipy.ndimage import gaussian_filter1d

CACHE_DIR = os.environ.get("REMFILE_CACHE", "/tmp/remfile_cache_000044")
ASSET_FILE = "assets_000044.json"
DANDISET = "000044"

NBINS = 40           # spatial bins across the track
SPEED_THRESH = 0.05  # m/s, excludes pauses on the track
POS_FS = 39.0626     # position sampling rate of this dandiset (Hz)
SMOOTH_BINS = 1.0    # s.d. of the Gaussian smoothing applied to rate maps, in bins

# ---------------------------------------------------------------- asset lookup


def get_assets():
    """Return {path: download_url} for every NWB asset in the dandiset."""
    if os.path.exists(ASSET_FILE):
        return json.load(open(ASSET_FILE))
    import urllib.request

    url = (f"https://api.dandiarchive.org/api/dandisets/{DANDISET}"
           "/versions/draft/assets/?page_size=100")
    with urllib.request.urlopen(url) as r:
        payload = json.load(r)
    assets = {
        a["path"]: (f"https://api.dandiarchive.org/api/dandisets/{DANDISET}"
                    f"/versions/draft/assets/{a['asset_id']}/download/")
        for a in sorted(payload["results"], key=lambda a: a["path"])
    }
    json.dump(assets, open(ASSET_FILE, "w"), indent=1)
    return assets


# ------------------------------------------------------------------- loading


def open_session(url):
    """Stream one NWB file from S3 with an on-disk chunk cache."""
    f = remfile.File(url, disk_cache=remfile.DiskCache(CACHE_DIR))
    io = NWBHDF5IO(file=h5py.File(f, "r"), load_namespaces=True)
    return io.read(), io


def maze_type(url):
    """Return the name of the maze used in a session without loading the data."""
    nwbfile, io = open_session(url)
    key = [k for k in nwbfile.processing["behavior"].data_interfaces
           if k.endswith("LinearizedPosition")][0]
    io.close()
    return key.replace("LinearizedPosition", "")


def load_session(url):
    """Load the pieces needed for place-field analysis into pynapple objects.

    Returns a dict with:
      units      TsGroup of spike trains, with cell_type / location metadata
      lin        Tsd of linearized position along the track (m), NaNs dropped
      pos2d      TsdFrame of raw 2D tracking (m)
      speed      Tsd of running speed along the track (m/s)
      laps       IntervalSet, one interval per uninterrupted track traversal
      lap_dir    +1 / -1 direction of travel for each lap
      maze       IntervalSet of the maze (running) epoch
      track_len  length of the track in m
    """
    nwbfile, io = open_session(url)

    epochs = nwbfile.intervals["epochs"].to_dataframe()
    maze_row = epochs[epochs["label"].str.contains("Maze")].iloc[0]
    maze = nap.IntervalSet(start=maze_row["start_time"], end=maze_row["stop_time"])

    beh = nwbfile.processing["behavior"]
    lin_key = [k for k in beh.data_interfaces if k.endswith("LinearizedPosition")][0]
    maze_name = lin_key.replace("LinearizedPosition", "")
    lin_ss = list(beh[lin_key].spatial_series.values())[0]
    pos_ss = list(beh[maze_name + "Position"].spatial_series.values())[0]

    m = re.match(r"([\d.]+)m", maze_name)
    lin_raw = np.asarray(lin_ss.data[:]).ravel()
    track_len = float(m.group(1)) if m else float(np.nanmax(lin_raw))

    t = lin_ss.starting_time + np.arange(lin_raw.size) / lin_ss.rate
    dt = 1.0 / lin_ss.rate
    pos2d = nap.TsdFrame(t=t, d=np.asarray(pos_ss.data[:]), columns=["x", "y"])

    # The linearized trace is defined only while the animal is actually on the
    # track; time spent in the end zones is NaN. Each contiguous stretch of
    # valid samples is therefore one traversal ("lap").
    valid = np.isfinite(lin_raw)
    idx = np.flatnonzero(valid)
    segments = np.split(idx, np.flatnonzero(np.diff(idx) != 1) + 1)
    lin = nap.Tsd(t=t[valid], d=lin_raw[valid], time_support=maze)

    starts, ends, dirs = [], [], []
    for seg in segments:
        if seg.size < 10:  # drop tracking fragments shorter than ~0.25 s
            continue
        travel = lin_raw[seg[-1]] - lin_raw[seg[0]]
        if abs(travel) < 0.5 * track_len:  # require crossing half the track
            continue
        starts.append(t[seg[0]])
        ends.append(t[seg[-1]] + dt)
        dirs.append(1 if travel > 0 else -1)
    laps = nap.IntervalSet(start=np.array(starts), end=np.array(ends))
    lap_dir = np.array(dirs)

    # Running speed from the linearized trace, computed lap by lap so that the
    # off-track gaps do not create spurious jumps.
    sp_t, sp_v = [], []
    for i in range(len(laps)):
        seg = lin.restrict(laps[i])
        if len(seg) < 3:
            continue
        sp_t.append(seg.times())
        sp_v.append(np.abs(np.gradient(seg.values, seg.times())))
    speed = nap.Tsd(t=np.concatenate(sp_t), d=np.concatenate(sp_v), time_support=maze)

    udf = nwbfile.units.to_dataframe()
    spikes = {int(i): nap.Ts(np.asarray(row["spike_times"]))
              for i, row in udf.iterrows()}
    meta = pd.DataFrame(
        {"cell_type": udf["cell_type"].values,
         "location": udf["location"].values,
         "shank_id": udf["shank_id"].values},
        index=[int(i) for i in udf.index],
    )
    units = nap.TsGroup(spikes, metadata=meta)

    return dict(units=units, lin=lin, pos2d=pos2d, speed=speed, laps=laps,
                lap_dir=lap_dir, maze=maze, track_len=track_len,
                maze_name=maze_name, session=nwbfile.session_id or "",
                subject=nwbfile.subject.subject_id, io=io)


# ------------------------------------------------------------------ analysis


def direction_epochs(laps, lap_dir):
    """Split laps into two IntervalSets by running direction."""
    return {
        name: nap.IntervalSet(start=laps.start[lap_dir == d], end=laps.end[lap_dir == d])
        for d, name in ((1, "rightward"), (-1, "leftward"))
    }


def run_epochs(speed, ep, thresh=SPEED_THRESH):
    """Portions of `ep` during which the animal is running faster than `thresh`."""
    return speed.restrict(ep).threshold(thresh, "above").time_support


def tuning_curves(units, lin, ep, track_len, nbins=NBINS, as_xarray=False,
                  smooth=SMOOTH_BINS):
    """Occupancy-normalized 1D firing-rate maps (Hz) over track position.

    `smooth` is the s.d., in bins, of a Gaussian applied along position; it is
    the usual mild smoothing applied to rate maps and is set to 0 for decoding.
    """
    tc = nap.compute_tuning_curves(
        units, lin, bins=nbins, range=[(0.0, track_len)], epochs=ep, fs=POS_FS,
        return_pandas=not as_xarray,
    )
    if smooth:
        axis = 0 if not as_xarray else -1
        vals = gaussian_filter1d(np.nan_to_num(np.asarray(tc)), smooth,
                                 axis=axis, mode="nearest")
        tc = tc.copy(data=vals) if as_xarray else pd.DataFrame(
            vals, index=tc.index, columns=tc.columns)
    return tc


def occupancy(lin, ep, track_len, nbins=NBINS):
    """Seconds spent in each spatial bin."""
    x = lin.restrict(ep)
    edges = np.linspace(0, track_len, nbins + 1)
    counts, _ = np.histogram(x.values, bins=edges)
    return counts / POS_FS, edges


def spatial_information(tc, lin, ep, track_len, nbins=NBINS):
    """Skaggs spatial information in bits/spike for each column of `tc`.

    SI = sum_i p_i * (r_i / rbar) * log2(r_i / rbar), with p_i the fraction of
    time spent in bin i and rbar the occupancy-weighted mean rate.
    """
    occ, _ = occupancy(lin, ep, track_len, nbins)
    p = occ / occ.sum()
    r = tc.values
    ok = np.isfinite(r) & (p[:, None] > 0)
    rbar = np.nansum(np.where(ok, r * p[:, None], 0.0), axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = r / rbar
        term = np.where(ok & (r > 0), p[:, None] * ratio * np.log2(ratio), 0.0)
    si = np.nansum(term, axis=0)
    si[rbar <= 0] = np.nan
    return pd.Series(si, index=tc.columns)


def field_stats(tc):
    """Peak rate, peak position, mean rate and sparsity for every unit."""
    r = np.nan_to_num(tc.values)
    mean_r = r.mean(axis=0)
    mean_r2 = (r ** 2).mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        sparsity = mean_r ** 2 / mean_r2
    return pd.DataFrame(
        {"peak_rate": r.max(axis=0),
         "peak_pos": tc.index.values[r.argmax(axis=0)],
         "mean_rate": mean_r,
         "sparsity": sparsity},
        index=tc.columns,
    )


def field_width(tc, frac=0.5):
    """Width (m) of the contiguous region around the peak above `frac` * peak."""
    pos = tc.index.values
    binw = pos[1] - pos[0]
    widths = {}
    for c in tc.columns:
        r = np.nan_to_num(tc[c].values)
        pk = int(r.argmax())
        thr = frac * r[pk]
        lo, hi = pk, pk
        while lo > 0 and r[lo - 1] >= thr:
            lo -= 1
        while hi < len(r) - 1 and r[hi + 1] >= thr:
            hi += 1
        widths[c] = (hi - lo + 1) * binw
    return pd.Series(widths)


def half_tuning_curves(s, units, k, parity, nbins=NBINS):
    """Tuning curves built from every other lap in one running direction."""
    sel = np.flatnonzero(s["lap_dir"] == (1 if k == "rightward" else -1))[parity::2]
    ep = run_epochs(s["speed"], nap.IntervalSet(s["laps"].start[sel], s["laps"].end[sel]))
    return tuning_curves(units, s["lin"], ep, s["track_len"], nbins)


def split_half_stability(s, units, k, nbins=NBINS):
    """Correlation between rate maps built from odd and even laps."""
    a = half_tuning_curves(s, units, k, 0, nbins)
    b = half_tuning_curves(s, units, k, 1, nbins)
    return pd.Series({u: np.corrcoef(np.nan_to_num(a[u]), np.nan_to_num(b[u]))[0, 1]
                      for u in a.columns})


def shuffle_si(units, lin, ep, track_len, n_shuffles=500, nbins=NBINS, seed=0):
    """Null distribution of spatial information from circularly shifted spikes.

    Each unit's spike train is shifted by a random offset within the running
    epochs (wrapping around), which destroys the relationship to position while
    preserving spike count and fine-scale temporal structure.
    """
    rng = np.random.default_rng(seed)
    starts, ends = ep.start, ep.end
    durs = ends - starts
    cum = np.concatenate([[0], np.cumsum(durs)])
    total = cum[-1]

    def to_flat(times):
        i = np.clip(np.searchsorted(starts, times, side="right") - 1, 0, len(durs) - 1)
        return cum[i] + (times - starts[i])

    def from_flat(u):
        i = np.clip(np.searchsorted(cum, u, side="right") - 1, 0, len(durs) - 1)
        return starts[i] + (u - cum[i])

    flat = {k: to_flat(units[k].restrict(ep).times()) for k in units.keys()}

    null = np.full((n_shuffles, len(units)), np.nan)
    for sh in range(n_shuffles):
        shifted = {}
        for k in units.keys():
            if flat[k].size == 0:
                shifted[k] = nap.Ts(np.array([starts[0]]))
                continue
            off = rng.uniform(20.0, max(total - 20.0, 21.0))
            shifted[k] = nap.Ts(np.sort(from_flat(np.mod(flat[k] + off, total))))
        grp = nap.TsGroup(shifted, time_support=ep)
        tc = tuning_curves(grp, lin, ep, track_len, nbins)
        null[sh] = spatial_information(tc, lin, ep, track_len, nbins).values
    return null


def classify_place_cells(s, units, n_shuffles=500, nbins=NBINS, seed=0):
    """Per-direction rate maps, spatial information, shuffle test and field stats.

    A unit counts as a place cell in a given direction when its spatial
    information exceeds the 99th percentile of its own circular-shift null and
    its peak rate is at least 1 Hz.
    """
    deps = direction_epochs(s["laps"], s["lap_dir"])
    run = {k: run_epochs(s["speed"], v) for k, v in deps.items()}
    tc, si, null, stats = {}, {}, {}, {}
    for i, k in enumerate(run):
        tc[k] = tuning_curves(units, s["lin"], run[k], s["track_len"], nbins)
        si[k] = spatial_information(tc[k], s["lin"], run[k], s["track_len"], nbins)
        null[k] = shuffle_si(units, s["lin"], run[k], s["track_len"],
                             n_shuffles, nbins, seed + i)
        fs = field_stats(tc[k])
        fs["si"] = si[k].values
        fs["si_thresh"] = np.nanpercentile(null[k], 99, axis=0)
        fs["si_p"] = (null[k] >= si[k].values[None, :]).mean(axis=0)
        fs["width"] = field_width(tc[k]).values
        fs["stability"] = split_half_stability(s, units, k, nbins).reindex(fs.index).values
        fs["is_place"] = (fs["si"] > fs["si_thresh"]) & (fs["peak_rate"] >= 1.0)
        fs["is_place_strict"] = fs["is_place"] & (fs["stability"] > 0.3)
        stats[k] = fs
    return dict(tc=tc, si=si, null=null, stats=stats, run=run)


def track_units(s, min_rate=0.1):
    """Putative pyramidal cells that fire at all while the animal is on the track."""
    exc = s["units"].getby_category("cell_type")["excitatory"]
    deps = direction_epochs(s["laps"], s["lap_dir"])
    on_track = run_epochs(s["speed"], deps["rightward"].union(deps["leftward"]))
    return exc[np.asarray(exc.restrict(on_track).rates > min_rate)]
