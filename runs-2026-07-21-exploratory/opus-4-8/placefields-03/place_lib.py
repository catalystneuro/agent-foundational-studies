"""Loading and preprocessing helpers for DANDI:000044 (Grosmark & Buzsaki 2016, hc-11)."""
import numpy as np
import lindi
import pynapple as nap
from pynwb import NWBHDF5IO

LINDI_BASE = "https://lindi.neurosift.org/dandi/dandisets/000044/assets/{}/nwb.lindi.json"
CACHE_DIR = "/tmp/lindi_cache_000044"

SESSIONS = {
    "Achilles_10252013": "5349c68b-c0a7-46c0-9900-cda050722fa4",
    "Achilles_11012013": "8855c8cc-9d8b-4d5b-8ef0-fe87916f839a",
    "Cicero_09012014":   "3cc5b7b3-02e2-490a-9f19-d20670355084",
    "Cicero_09102014":   "f61dfe09-3db2-464a-b386-2e828b2e7276",
    "Cicero_09172014":   "e381ebb3-128e-4f3f-9517-11277d7aed9b",
    "Gatsby_08022013":   "31ea0aab-4777-424e-9a93-9605b2bdcc29",
    "Gatsby_08282013":   "f7687af7-3bc9-4d20-8d88-ef293d2a3381",
    "Buddy_06272013":    "82714afb-724f-4e2b-b102-c9c47b5cba73",
}

# Three of the eight sessions used a circular maze, where the direction-flip that puts both
# headings in a common frame does not apply. The analysis is limited to the linear-track
# sessions (one 2 m track, the rest 1.6 m).
LINEAR_SESSIONS = ["Achilles_10252013", "Cicero_09012014", "Cicero_09172014",
                   "Gatsby_08022013", "Buddy_06272013"]


def open_nwb(session):
    """Stream one session's NWB file through LINDI with a local cache."""
    url = LINDI_BASE.format(SESSIONS[session])
    f = lindi.LindiH5pyFile.from_lindi_file(url, local_cache=lindi.LocalCache(cache_dir=CACHE_DIR))
    return NWBHDF5IO(file=f, mode="r").read()


def _spatial_series(nwbfile, kind):
    """Return the linear-maze Position container matching `kind` ('Linearized' or 'Spatial')."""
    beh = nwbfile.processing["behavior"]
    name = [k for k in beh.data_interfaces if "Maze" in k and
            ("Linearized" in k) == (kind == "Linearized")][0]
    container = beh[name]
    return list(container.spatial_series.values())[0]


def load_behavior(nwbfile):
    """Position on the linear maze, in a single spatial frame, with run direction.

    The NWB SpatialSeries in this dandiset store the sampling *period* in the `rate`
    field, so timestamps are rebuilt as start + i * period rather than i / rate.

    The archived "linearized" variable runs 0 -> track length on every traversal,
    irrespective of heading, so it is direction-collapsed. Leftward runs are flipped
    (L - x) to put both headings in one spatial frame anchored at the left reward well.
    """
    lin_ss = _spatial_series(nwbfile, "Linearized")
    xy_ss = _spatial_series(nwbfile, "Spatial")
    period = float(lin_ss.rate)          # seconds per sample (mislabeled as rate)
    t0 = float(lin_ss.starting_time)
    lin = np.asarray(lin_ss.data[:]).squeeze()
    xy = np.asarray(xy_ss.data[:])
    t = t0 + np.arange(lin.size) * period

    track_len = float(np.nanmax(lin))
    ok = np.isfinite(lin)
    # contiguous blocks of defined linearized position = individual track traversals
    edges = np.flatnonzero(np.diff(ok.astype(int)))
    starts = np.r_[0 if ok[0] else [], edges[ok[edges + 1]] + 1]
    stops = np.r_[edges[~ok[edges + 1]] + 1, lin.size if ok[-1] else []]
    starts, stops = np.atleast_1d(starts).astype(int), np.atleast_1d(stops).astype(int)

    pos = np.full(lin.shape, np.nan)
    direction = np.full(lin.shape, np.nan)
    runs = []
    for a, b in zip(starts, stops):
        if b - a < 10:
            continue
        rightward = xy[b - 1, 0] > xy[a, 0]      # heading from raw camera x
        pos[a:b] = lin[a:b] if rightward else track_len - lin[a:b]
        direction[a:b] = 1.0 if rightward else -1.0
        runs.append((t[a], t[b - 1], 1.0 if rightward else -1.0))

    good = np.isfinite(pos)
    position = nap.Tsd(t=t[good], d=pos[good])
    heading = nap.Tsd(t=t[good], d=direction[good])
    runs = np.array(runs)
    run_ep = nap.IntervalSet(start=runs[:, 0], end=runs[:, 1])
    right_ep = nap.IntervalSet(start=runs[runs[:, 2] > 0, 0], end=runs[runs[:, 2] > 0, 1])
    left_ep = nap.IntervalSet(start=runs[runs[:, 2] < 0, 0], end=runs[runs[:, 2] < 0, 1])

    speed = np.abs(np.gradient(position.d, position.t))
    speed = nap.Tsd(t=position.t, d=speed).smooth(0.25)

    raw_xy = nap.TsdFrame(t=t, d=xy, columns=["x", "y"])
    return dict(position=position, heading=heading, speed=speed, run_ep=run_ep,
                right_ep=right_ep, left_ep=left_ep, track_len=track_len,
                raw_xy=raw_xy, period=period)


def load_units(nwbfile):
    """TsGroup of sorted units with cell_type / location / shank metadata."""
    ut = nwbfile.units
    st = ut["spike_times"]
    spikes = {i: np.asarray(st[i]) for i in range(len(ut.id))}
    meta = {c: np.asarray(ut[c][:]) for c in ("cell_type", "location", "shank_id")}
    tsg = nap.TsGroup(spikes)
    for k, v in meta.items():
        tsg.set_info(**{k: v})
    return tsg


def epochs_of(nwbfile):
    df = nwbfile.epochs.to_dataframe()
    return {row.label: nap.IntervalSet(start=row.start_time, end=row.stop_time)
            for row in df.itertuples()}


def load_lfp(nwbfile, window, channels=None):
    """Read a time window of LFP (volts) for selected channels as a pynapple TsdFrame."""
    import pynapple as nap
    es = list(nwbfile.processing["ecephys"]["LFP"].electrical_series.values())[0]
    fs = float(es.rate)
    i0 = max(int(round((window[0] - es.starting_time) * fs)), 0)
    i1 = min(int(round((window[1] - es.starting_time) * fs)), es.data.shape[0])
    if channels is None:
        channels = np.arange(es.data.shape[1])
    channels = np.atleast_1d(channels)
    d = np.stack([es.data[i0:i1, c] for c in channels], axis=1).astype(float) * es.conversion
    t = es.starting_time + np.arange(i0, i1) / fs
    return nap.TsdFrame(t=t, d=d, columns=[f"ch{c}" for c in channels])


def pick_theta_channel(nwbfile, window, candidates):
    """Channel with the largest 6-10 Hz to 1-4 Hz power ratio in `window`."""
    from scipy.signal import welch
    lfp = load_lfp(nwbfile, window, candidates)
    fs = 1.0 / np.median(np.diff(lfp.t))
    f, p = welch(np.asarray(lfp.values).T, fs=fs, nperseg=int(4 * fs))
    theta = p[:, (f >= 6) & (f <= 10)].mean(1)
    delta = p[:, (f >= 1) & (f <= 4)].mean(1)
    return int(candidates[np.argmax(theta / delta)]), theta / delta
