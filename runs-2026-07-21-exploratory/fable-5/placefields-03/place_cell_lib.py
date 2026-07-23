"""Shared loading / preprocessing helpers for the DANDI:000044 place-cell analysis.

Dandiset 000044 (Grosmark & Buzsaki 2016, "Diversity in neural firing dynamics
supports both rigid and learned hippocampal sequences") contains bilateral
silicon-probe recordings from dorsal CA1 of rats running back and forth on a
1.6 m linear track for water reward, flanked by pre- and post-run sleep sessions.

Two quirks of the NWB files are handled here:

1. The behavioral SpatialSeries store the sampling *period* (0.0256 s) in the
   ``rate`` field instead of the sampling rate. PyNWB therefore reconstructs
   timestamps 1525x too sparse. We rebuild them as
   ``starting_time + arange(n) * rate``, which lands the last sample exactly one
   frame before the end of the MazeEpoch.
2. The file's own linearized position series is only defined on about 13% of
   frames. We recover a full-coverage linearization by projecting the 2D
   tracking onto the principal axis of the track and fitting the affine map to
   the file's own linearized values (r = 1.0000 on the overlapping samples), so
   the resulting coordinate is in the dataset's native 0-1.6 m frame.
"""

import numpy as np
import lindi
import pynapple as nap
from pynwb import NWBHDF5IO

nap.nap_config.suppress_conversion_warnings = True

DANDISET = "000044"
LINDI_TEMPLATE = "https://lindi.neurosift.org/dandi/dandisets/{d}/assets/{a}/nwb.lindi.json"
CACHE_DIR = "./lindi_cache"

# All eight sessions of the dandiset (asset id, subject, session label, maze type).
# Three sessions used a circular maze; the linearization below assumes a straight
# track, so the analysis is scoped to the five linear-track sessions.
ALL_SESSIONS = [
    ("5349c68b-c0a7-46c0-9900-cda050722fa4", "Achilles", "Achilles_10252013", "1.6m linear"),
    ("8855c8cc-9d8b-4d5b-8ef0-fe87916f839a", "Achilles", "Achilles_11012013", "circular"),
    ("82714afb-724f-4e2b-b102-c9c47b5cba73", "Buddy", "Buddy_06272013", "1.6m linear"),
    ("3cc5b7b3-02e2-490a-9f19-d20670355084", "Cicero", "Cicero_09012014", "1.6m linear"),
    ("f61dfe09-3db2-464a-b386-2e828b2e7276", "Cicero", "Cicero_09102014", "circular"),
    ("e381ebb3-128e-4f3f-9517-11277d7aed9b", "Cicero", "Cicero_09172014", "2m linear"),
    ("31ea0aab-4777-424e-9a93-9605b2bdcc29", "Gatsby", "Gatsby_08022013", "1.6m linear"),
    ("f7687af7-3bc9-4d20-8d88-ef293d2a3381", "Gatsby", "Gatsby_08282013", "circular"),
]
SESSIONS = [s for s in ALL_SESSIONS if "linear" in s[3]]
SPEED_THRESHOLD = 0.10  # m/s, standard "running" cut for this dataset
POS_SMOOTH_STD = 0.10   # s, Gaussian std used before differentiating position
MIN_RUN_DURATION = 0.5  # s
N_POSITION_BINS = 40    # 4 cm bins over the 1.6 m track


def open_nwb(asset_id):
    """Stream one asset of dandiset 000044 through LINDI with a local chunk cache."""
    url = LINDI_TEMPLATE.format(d=DANDISET, a=asset_id)
    f = lindi.LindiH5pyFile.from_lindi_file(url, local_cache=lindi.LocalCache(cache_dir=CACHE_DIR))
    io = NWBHDF5IO(file=f, mode="r")
    return io.read(), io


def _corrected_timestamps(spatial_series):
    """Rebuild the SpatialSeries timestamps (see module docstring, quirk 1)."""
    n = spatial_series.data.shape[0]
    period = spatial_series.rate  # this field actually holds the sampling period
    return spatial_series.starting_time + np.arange(n) * period, 1.0 / period


PERP_TOLERANCE = 0.15   # m, half-width of the accepted corridor around the track axis


def linearize(xy, file_linear):
    """Project 2D tracking onto the track axis and express it in metres.

    The scale is inherited from the file's own linearized series by fitting an
    affine map on the frames where that series is defined, so one unit of the
    returned coordinate is one metre along the track. The origin is placed at
    the far end of the left reward platform, so the coordinate runs from 0 to
    somewhat more than 1.6 m: the extra length is the two reward platforms,
    which the file's linearization excludes but where the rat spends much of
    the session.

    Returns (linear_position, on_track_mask, info).
    """
    finite2d = np.isfinite(xy).all(axis=1)
    known = finite2d & np.isfinite(file_linear)

    # Track axis from the frames the file itself considered "on track"
    ref = xy[known]
    mu = ref.mean(axis=0)
    _, _, vt = np.linalg.svd(ref - mu, full_matrices=False)
    along, across = vt[0], vt[1]

    with np.errstate(all="ignore"):
        proj = (xy - mu) @ along
        perp = (xy - mu) @ across

    # Affine map from the projection onto the dataset's metric linearization
    slope, intercept = np.polyfit(proj[known], file_linear[known], 1)
    linear = slope * proj + intercept

    # Reject frames far off the track axis: for part of the MazeEpoch the rat is
    # tracked in a small holding area well away from the maze.
    perp0 = np.median(perp[known])
    on_axis = finite2d & (np.abs(perp - perp0) < PERP_TOLERANCE)

    # Anchor the origin at the robust low end of the travelled extent
    lo, hi = np.percentile(linear[on_axis], [0.2, 99.8])
    linear = linear - lo
    extent = hi - lo
    on_track = on_axis & (linear > -0.02) & (linear < extent + 0.02)
    linear = np.clip(linear, 0.0, extent)

    return linear, on_track, dict(slope=slope, intercept=intercept, extent=extent,
                                  perp=perp, perp0=perp0, along=along, mu=mu)


def fill_short_gaps(values, max_gap_samples):
    """Linearly interpolate over NaN runs no longer than `max_gap_samples`.

    The tracking drops out for a frame or two very often; leaving those as gaps
    fragments every downstream epoch. Longer dropouts are left as NaN.
    """
    out = values.copy()
    bad = ~np.isfinite(out)
    if not bad.any():
        return out
    idx = np.arange(len(out))
    good = ~bad
    filled = np.interp(idx, idx[good], out[good])

    # Only accept the interpolation inside short gaps
    edges = np.diff(np.concatenate([[0], bad.view(np.int8), [0]]))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    for s, e in zip(starts, ends):
        if (e - s) <= max_gap_samples and s > 0 and e < len(out):
            out[s:e] = filled[s:e]
    return out


def valid_blocks(t, dt, tol=1.5):
    """Index bounds (start, stop) of stretches of `t` sampled at ~dt without gaps."""
    breaks = np.where(np.diff(t) > tol * dt)[0]
    starts = np.concatenate([[0], breaks + 1])
    stops = np.concatenate([breaks + 1, [len(t)]])
    return list(zip(starts, stops))


def _smooth_per_block(values, blocks, std_samples):
    """Gaussian-smooth each contiguous block independently with edge replication."""
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import medfilt

    out = np.empty_like(values)
    for s, e in blocks:
        seg = values[s:e]
        if len(seg) >= 5:
            seg = medfilt(seg, 5)
        out[s:e] = gaussian_filter1d(seg, std_samples, mode="nearest")
    return out


def contiguous_epochs(t, mask, dt, max_gap_samples=2, min_duration=0.3):
    """IntervalSet covering runs of consecutive True samples in `mask`."""
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return nap.IntervalSet(start=[], end=[])
    breaks = np.where(np.diff(idx) > max_gap_samples)[0]
    starts = np.concatenate([[idx[0]], idx[breaks + 1]])
    ends = np.concatenate([idx[breaks], [idx[-1]]])
    s, e = t[starts], t[ends] + dt
    keep = (e - s) >= min_duration
    return nap.IntervalSet(start=s[keep], end=e[keep])


def load_session(asset_id, keep_lfp=False):
    """Load one session and return the pieces needed for the place-field analysis."""
    nwbfile, io = open_nwb(asset_id)

    epochs_df = nwbfile.intervals["epochs"].to_dataframe()
    maze_row = epochs_df[epochs_df.label == "MazeEpoch"].iloc[0]
    maze_ep = nap.IntervalSet(start=maze_row.start_time, end=maze_row.stop_time)

    # Interface names carry the maze type ("1.6mLinearMazePosition",
    # "2mLinearMazePosition", ...), so look them up by suffix.
    beh = nwbfile.processing["behavior"]
    lin_name = next(k for k in beh.data_interfaces if k.endswith("LinearizedPosition"))
    pos_name = next(k for k in beh.data_interfaces
                    if k.endswith("Position") and k != lin_name)
    ss2d = next(iter(beh[pos_name].spatial_series.values()))
    sslin = next(iter(beh[lin_name].spatial_series.values()))

    t, fs = _corrected_timestamps(ss2d)
    dt = 1.0 / fs
    xy = np.asarray(ss2d.data[:])
    file_linear = np.asarray(sslin.data[:]).ravel()

    linear, on_track, lin_info = linearize(xy, file_linear)

    # Fill dropouts up to ~0.75 s so that single traversals are not fragmented
    masked = np.where(on_track, linear, np.nan)
    filled = fill_short_gaps(masked, max_gap_samples=int(0.75 * fs))
    valid = np.isfinite(filled)
    pos_ep = contiguous_epochs(t, valid, dt, max_gap_samples=1, min_duration=0.5)

    position = nap.Tsd(t=t[valid], d=filled[valid], time_support=pos_ep).restrict(pos_ep)
    xy_filled = np.column_stack([fill_short_gaps(np.where(on_track, xy[:, i], np.nan),
                                                 int(0.75 * fs)) for i in range(2)])
    xy_tsd = nap.TsdFrame(t=t[valid], d=xy_filled[valid], columns=["x", "y"],
                          time_support=pos_ep).restrict(pos_ep)

    # Speed and direction from a median-filtered, lightly smoothed position trace.
    # The median filter removes isolated tracking jumps, and the smoothing is
    # done per contiguous block with edge replication (pynapple's Tsd.smooth
    # zero-pads, which manufactures huge velocities at block boundaries).
    smoothed_vals = _smooth_per_block(position.values, valid_blocks(t[valid], dt),
                                      std_samples=POS_SMOOTH_STD * fs)
    smoothed = nap.Tsd(t=position.t, d=smoothed_vals, time_support=pos_ep)
    velocity = smoothed.derivative()
    speed = nap.Tsd(t=velocity.t, d=np.abs(velocity.values), time_support=pos_ep)

    # Spike times are not sorted within every unit of every file, so sort them
    # explicitly rather than relying on pynapple's implicit reordering.
    units = nwbfile.units
    spikes = nap.TsGroup(
        {i: nap.Ts(t=np.sort(np.asarray(units["spike_times"][i]))) for i in range(len(units))},
        time_support=nap.IntervalSet(start=epochs_df.start_time.min(),
                                     end=epochs_df.stop_time.max()),
    )
    spikes.set_info(
        cell_type=np.asarray(units["cell_type"][:]).astype(str),
        location=np.asarray(units["location"][:]).astype(str),
        shank_id=np.asarray(units["shank_id"][:]),
    )

    out = dict(
        session_id=nwbfile.session_id,
        subject=nwbfile.subject.subject_id,
        nwbfile=nwbfile,
        io=io,
        epochs=epochs_df,
        maze_ep=maze_ep,
        states=nwbfile.processing["behavior"]["states"].to_dataframe(),
        position=position,
        xy=xy_tsd,
        velocity=velocity,
        speed=speed,
        spikes=spikes,
        fs=fs,
        lin_info=lin_info,
        pos_ep=pos_ep,
        extent=lin_info["extent"],
    )
    if keep_lfp:
        out["lfp"] = nwbfile.processing["ecephys"]["LFP"].electrical_series["LFP"]
    return out


def lap_epochs(session, end_frac=0.15, max_lap_duration=90.0, max_gap=10.0, tol=0.03):
    """Detect complete end-to-end traversals of the track.

    The trace is split at midline crossings, giving alternating left-side and
    right-side excursions. A lap runs from the rat's final departure from the
    extreme of one excursion to its first arrival at the extreme of the next,
    and is kept only if both extremes actually reach the ends of the track.
    This is far more robust than thresholding instantaneous speed, which
    fragments single traversals whenever the tracking momentarily stalls.

    Returns a dict with "rightward", "leftward" and "both" IntervalSets.
    """
    pos = session["position"]
    t, x = pos.t, pos.values
    extent = session["extent"]
    lo_end, hi_end = end_frac * extent, (1 - end_frac) * extent
    mid = 0.5 * extent

    above = (x > mid).astype(np.int8)
    bounds = np.concatenate([[0], np.where(np.diff(above) != 0)[0] + 1, [len(x)]])
    segments = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]

    # Per excursion: which side, its extreme value, last-departure and
    # first-arrival sample indices at that extreme
    marks = []
    for s, e in segments:
        seg = x[s:e]
        side = 1 if above[s] else -1
        if side < 0:
            extreme = seg.min()
            at = np.where(seg <= extreme + tol)[0]
        else:
            extreme = seg.max()
            at = np.where(seg >= extreme - tol)[0]
        marks.append(dict(side=side, extreme=extreme,
                          depart=s + at[-1], arrive=s + at[0]))

    laps = {"rightward": [], "leftward": []}
    for a, b in zip(marks[:-1], marks[1:]):
        if a["side"] == b["side"]:
            continue
        rightward = a["side"] < 0
        reaches_ends = (
            (a["extreme"] < lo_end and b["extreme"] > hi_end) if rightward
            else (a["extreme"] > hi_end and b["extreme"] < lo_end)
        )
        if not reaches_ends:
            continue
        i0, i1 = a["depart"], b["arrive"]
        if i1 <= i0:
            continue
        duration = t[i1] - t[i0]
        if not (MIN_RUN_DURATION <= duration <= max_lap_duration):
            continue
        if np.max(np.diff(t[i0:i1 + 1])) > max_gap:  # tracking dropout inside the lap
            continue
        laps["rightward" if rightward else "leftward"].append((t[i0], t[i1]))

    eps = {}
    for name, iv in laps.items():
        iv = np.asarray(iv, dtype=float).reshape(-1, 2)
        # Intersecting with pos_ep drops the untracked stretches inside a lap,
        # so a lap interrupted by a dropout contributes only its tracked parts.
        eps[name] = nap.IntervalSet(start=iv[:, 0], end=iv[:, 1]).intersect(session["pos_ep"])
    eps["both"] = eps["rightward"].union(eps["leftward"])
    eps["n_laps"] = {k: len(v) for k, v in laps.items()}
    return eps


def moving_epochs(session, laps, speed_threshold=SPEED_THRESHOLD):
    """Within-lap epochs where the rat is moving in that lap's direction.

    The lap windows are deliberately generous (they run from one end extreme to
    the other and so include pauses and small back-steps). Intersecting them
    with the locally consistent direction of travel keeps only the parts of the
    traversal that actually contribute to a directional rate map.
    """
    vel = session["velocity"]
    dt = 1.0 / session["fs"]
    out = {}
    for name, sign in (("rightward", 1), ("leftward", -1)):
        mask = (np.abs(vel.values) > speed_threshold) & (np.sign(vel.values) == sign)
        consistent = contiguous_epochs(vel.t, mask, dt, max_gap_samples=8, min_duration=0.1)
        out[name] = laps[name].intersect(consistent)
    out["both"] = out["rightward"].union(out["leftward"])
    return out


def place_fields(spikes, position, ep, extent, bins=N_POSITION_BINS, smooth_bins=1.0, fs=None):
    """1D firing-rate maps (Hz) as a function of linearized position."""
    from scipy.ndimage import gaussian_filter1d

    tc = nap.compute_tuning_curves(
        spikes,
        position,
        bins=bins,
        range=[(0.0, extent)],
        epochs=ep,
        fs=fs,
        feature_names=["position"],
    )
    if smooth_bins:
        tc.data = gaussian_filter1d(np.nan_to_num(tc.data), smooth_bins, axis=-1, mode="nearest")
    return tc


def _binned(spikes, position, ep, extent, bins):
    """Align spikes to the position samples inside `ep`.

    Returns (bin_index_per_sample, spike_counts_per_sample [n_units, n_samples],
    bin_edges). Working on the concatenated stream of position samples makes the
    circular-shift shuffle below both exact and fast.
    """
    pos = position.restrict(ep)
    edges = np.linspace(0.0, extent, bins + 1)
    bin_idx = np.clip(np.digitize(pos.values, edges) - 1, 0, bins - 1)

    t = pos.t
    counts = np.zeros((len(spikes), len(t)), dtype=np.int32)
    for row, uid in enumerate(spikes.keys()):
        st = spikes[uid].restrict(ep).t
        if len(st) == 0:
            continue
        j = np.clip(np.searchsorted(t, st), 0, len(t) - 1)
        np.add.at(counts[row], j, 1)
    return bin_idx, counts, edges


def _rate_maps(bin_idx, counts, occupancy, smooth_bins=1.0):
    from scipy.ndimage import gaussian_filter1d

    nbins = len(occupancy)
    maps = np.stack([
        np.bincount(bin_idx, weights=c, minlength=nbins) for c in counts
    ])
    with np.errstate(invalid="ignore", divide="ignore"):
        maps = maps / occupancy[None, :]
    maps = np.nan_to_num(maps)
    if smooth_bins:
        maps = gaussian_filter1d(maps, smooth_bins, axis=-1, mode="nearest")
    return maps


def skaggs_information(maps, occupancy):
    """Skaggs spatial information in bits/spike for each row of `maps`."""
    p = occupancy / occupancy.sum()
    mean_rate = (p[None, :] * maps).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = maps / mean_rate[:, None]
        term = p[None, :] * ratio * np.log2(ratio)
    return np.nansum(np.where(maps > 0, term, 0.0), axis=1)


def spatial_info_with_shuffle(spikes, position, ep, extent, fs, bins=N_POSITION_BINS,
                              n_shuffles=500, smooth_bins=1.0, seed=0, progress=True):
    """Observed rate maps, Skaggs information, and a circular-shift null.

    Each unit's spike train is circularly shifted along the concatenated stream
    of run-epoch samples, which destroys the spike-position relationship while
    preserving the spike train's own temporal statistics.
    """
    from tqdm import tqdm

    bin_idx, counts, edges = _binned(spikes, position, ep, extent, bins)
    occupancy = np.bincount(bin_idx, minlength=bins) / fs
    occupancy[occupancy == 0] = np.nan

    maps = _rate_maps(bin_idx, counts, occupancy, smooth_bins)
    si = skaggs_information(maps, np.nan_to_num(occupancy))

    rng = np.random.default_rng(seed)
    n_samples = counts.shape[1]
    # Shift by at least 20 s (longer than a lap), or a quarter of the record if
    # the session has less running data than that allows.
    min_shift = max(1, min(int(20 * fs), n_samples // 4))
    null = np.empty((n_shuffles, len(spikes)))
    it = range(n_shuffles)
    if progress:
        it = tqdm(it, desc="shuffles", leave=False)
    for i in it:
        shifts = rng.integers(min_shift, n_samples - min_shift, size=len(spikes))
        shifted = np.stack([np.roll(c, s) for c, s in zip(counts, shifts)])
        null[i] = skaggs_information(
            _rate_maps(bin_idx, shifted, occupancy, smooth_bins), np.nan_to_num(occupancy)
        )
    pvals = (null >= si[None, :]).sum(axis=0) / n_shuffles
    return dict(maps=maps, si=si, null=null, pvals=pvals, occupancy=occupancy,
                edges=edges, centers=0.5 * (edges[:-1] + edges[1:]),
                n_spikes=counts.sum(axis=1))


def split_half_stability(spikes, position, ep, extent, fs, bins=N_POSITION_BINS,
                         smooth_bins=1.0):
    """Correlation between rate maps built from alternate (odd / even) laps."""
    idx = np.arange(len(ep))
    halves = []
    for sel in (idx[0::2], idx[1::2]):
        sub = ep[sel]
        bin_idx, counts, _ = _binned(spikes, position, sub, extent, bins)
        occ = np.bincount(bin_idx, minlength=bins) / fs
        occ[occ == 0] = np.nan
        halves.append(_rate_maps(bin_idx, counts, occ, smooth_bins))
    a, b = halves
    ok = np.isfinite(a) & np.isfinite(b)
    r = np.array([
        np.corrcoef(a[i][ok[i]], b[i][ok[i]])[0, 1] if ok[i].sum() > 3 and
        a[i][ok[i]].std() > 0 and b[i][ok[i]].std() > 0 else np.nan
        for i in range(a.shape[0])
    ])
    return r, a, b


PLACE_CELL_MIN_PEAK = 1.0     # Hz
PLACE_CELL_MIN_SPIKES = 50
PLACE_CELL_MIN_STABILITY = 0.5
PLACE_CELL_ALPHA = 0.05


def classify_place_cells(res):
    """Boolean mask of units meeting the place-cell criteria in one direction."""
    return (
        (res["pvals"] < PLACE_CELL_ALPHA)
        & (res["maps"].max(axis=1) >= PLACE_CELL_MIN_PEAK)
        & (res["n_spikes"] >= PLACE_CELL_MIN_SPIKES)
        & (np.nan_to_num(res["stability"], nan=-1) > PLACE_CELL_MIN_STABILITY)
    )


def analyze_session(asset_id, n_shuffles=500, seed=0, progress=True):
    """Run the whole single-session place-field pipeline."""
    sess = load_session(asset_id)
    laps = lap_epochs(sess)
    mov = moving_epochs(sess, laps)

    spikes = sess["spikes"]
    ctype = spikes.get_info("cell_type")
    exc = spikes[np.where(ctype == "excitatory")[0]]
    inh = spikes[np.where(ctype == "inhibitory")[0]]

    out = dict(session=sess, laps=laps, moving=mov, exc=exc, inh=inh, directions={})
    for d in ("rightward", "leftward"):
        res = spatial_info_with_shuffle(exc, sess["position"], mov[d], sess["extent"],
                                        sess["fs"], n_shuffles=n_shuffles, seed=seed,
                                        progress=progress)
        res["stability"], res["half_a"], res["half_b"] = split_half_stability(
            exc, sess["position"], mov[d], sess["extent"], sess["fs"]
        )
        res["is_place_cell"] = classify_place_cells(res)
        out["directions"][d] = res

    res_inh = spatial_info_with_shuffle(inh, sess["position"], mov["rightward"],
                                        sess["extent"], sess["fs"],
                                        n_shuffles=n_shuffles, seed=seed, progress=progress)
    res_inh["stability"], _, _ = split_half_stability(inh, sess["position"], mov["rightward"],
                                                      sess["extent"], sess["fs"])
    out["inhibitory"] = res_inh
    out["is_place_cell_any"] = (
        out["directions"]["rightward"]["is_place_cell"]
        | out["directions"]["leftward"]["is_place_cell"]
    )
    return out


def decoding_error(sess, spikes, ep, bin_size=0.25, bins=N_POSITION_BINS, seed=0):
    """Two-fold cross-validated Bayesian decoding of position from the population."""
    idx = np.arange(len(ep))
    folds = [(idx[0::2], idx[1::2]), (idx[1::2], idx[0::2])]
    decoded_all, truth_all, prob_all = [], [], []
    for train_i, test_i in folds:
        tc = place_fields(spikes, sess["position"], ep[train_i], sess["extent"],
                          bins=bins, fs=sess["fs"])
        decoded, prob = nap.decode_bayes(tc, spikes, ep[test_i], bin_size=bin_size)
        decoded_all.append(decoded)
        prob_all.append(prob)
        truth_all.append(sess["position"].interpolate(decoded))
    d = np.concatenate([x.values for x in decoded_all])
    y = np.concatenate([x.values for x in truth_all])
    rng = np.random.default_rng(seed)
    return dict(decoded=d, truth=y, error=np.abs(d - y),
                chance=np.abs(d - rng.permutation(y)),
                pieces=decoded_all, probs=prob_all)


def field_metrics(tc):
    """Peak rate, field width and centre of mass for each unit's rate map."""
    rates = np.asarray(tc.data)
    centers = np.asarray(tc.coords["position"].values)
    peak_rate = rates.max(axis=1)
    peak_pos = centers[rates.argmax(axis=1)]
    with np.errstate(invalid="ignore", divide="ignore"):
        above = rates >= 0.5 * peak_rate[:, None]
        bin_w = centers[1] - centers[0]
        width = above.sum(axis=1) * bin_w
        total = rates.sum(axis=1)
        com = (rates * centers).sum(axis=1) / np.where(total > 0, total, np.nan)
        # Skaggs sparsity: (sum p*r)^2 / sum(p*r^2), with p = uniform over sampled bins
        p = np.asarray(tc.attrs["occupancy"], dtype=float)
        p = p / p.sum()
        mean_r = (p * rates).sum(axis=1)
        sparsity = mean_r**2 / np.where((p * rates**2).sum(axis=1) > 0,
                                        (p * rates**2).sum(axis=1), np.nan)
    return dict(peak_rate=peak_rate, peak_pos=peak_pos, width=width,
                com=com, sparsity=sparsity, mean_rate=mean_r)
