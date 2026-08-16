"""Core routines for the hippocampal place-cell analysis of DANDI:000044.

Everything downstream of loading lives here so the prototype driver and the
final jupytext notebook share one implementation.
"""

import numpy as np
import pandas as pd
import pynapple as nap

from dandi_io import open_session, SESSIONS

nap.nap_config.suppress_conversion_warnings = True

# ----------------------------------------------------------------------------
# Analysis parameters
# ----------------------------------------------------------------------------
TRACK_LENGTH_CM = 160.0     # default; the actual value is read per session
POS_BIN_CM = 4.0            # spatial bin width
N_POS_BINS = 40             # default; recomputed per session as round(L / POS_BIN_CM)
SMOOTH_BINS = 1.0           # gaussian sigma, in bins, for rate-map smoothing
MIN_RUN_DURATION = 0.5      # s
MIN_RUN_COVERAGE_FRAC = 0.5  # a run must span at least half the track
SPEED_THRESHOLD_CMS = 5.0   # exclude near-stationary samples
MIN_MEAN_RATE_HZ = 0.1      # exclude near-silent units
MAX_MEAN_RATE_HZ = 10.0     # exclude fast-spiking units from the pyramidal set
PEAK_RATE_CRITERION_HZ = 1.0
N_SHUFFLES = 500
SHUFFLE_ALPHA = 0.01


# ----------------------------------------------------------------------------
# Loading and behavioural preprocessing
# ----------------------------------------------------------------------------
def load_behavior_and_spikes(session_name):
    """Load one session and return (spikes, position_cm, maze_ep, meta, handles).

    Note on timestamps: the SpatialSeries objects in this dandiset store the
    sampling *period* (0.0256 s) in the NWB ``rate`` field rather than the
    sampling frequency. Reconstructing t as ``starting_time + i * rate`` gives a
    total duration that matches the MazeEpoch exactly, which is the check used
    below, so we rebuild the time base by hand instead of trusting the default.
    """
    nwb, nwbfile, io = open_session(session_name)

    behavior = nwbfile.processing["behavior"]
    lin_name = [k for k in behavior.data_interfaces if "Linearized" in k][0]
    maze_type = "circular" if "Circular" in lin_name else "linear"
    pos_name = [k for k in behavior.data_interfaces if k.endswith("Position") and "Linearized" not in k][0]
    lin_ss = list(behavior[lin_name].spatial_series.values())[0]
    pos_ss = list(behavior[pos_name].spatial_series.values())[0]

    dt = float(lin_ss.rate)            # actually the sampling period, see docstring
    t0 = float(lin_ss.starting_time)
    lin = np.asarray(lin_ss.data[:]).ravel()
    xy = np.asarray(pos_ss.data[:])
    t = t0 + np.arange(lin.size) * dt

    epochs = nwb["epochs"]
    maze_ep = epochs[np.asarray(epochs.label) == "MazeEpoch"]
    dur_mismatch = abs(lin.size * dt - (maze_ep.end[0] - maze_ep.start[0]))
    assert dur_mismatch < 1.0, f"time base does not match MazeEpoch ({dur_mismatch:.2f} s off)"

    good = np.isfinite(lin)
    position = nap.Tsd(t=t[good], d=lin[good] * 100.0, time_support=maze_ep)  # m -> cm
    xy_good = np.isfinite(xy).all(axis=1)
    position_2d = nap.TsdFrame(t=t[xy_good], d=xy[xy_good] * 100.0,
                               columns=["x", "y"], time_support=maze_ep)

    # Data-quality check on the units table. In three of the eight sessions one
    # row (unit id 2) holds two full-session spike trains concatenated end to
    # end, so its spike times jump backwards to zero part way through. Pynapple
    # silently sorts such a train, which would turn it into a plausible-looking
    # but meaningless unit, so the check is done on the raw ragged array and the
    # offending rows are dropped.
    unit_ids = np.asarray(nwbfile.units.id[:])
    corrupt = []
    for row, uid in enumerate(unit_ids):
        st = np.asarray(nwbfile.units["spike_times"][row])
        if st.size and np.any(np.diff(st) < 0):
            corrupt.append(int(uid))

    spikes = nwb["units"]
    if corrupt:
        spikes = spikes[[k for k in spikes.keys() if k not in corrupt]]

    track_length = float(np.ceil(np.nanmax(lin) * 100.0 / 10.0) * 10.0)
    meta = dict(
        session=session_name,
        subject=nwbfile.subject.subject_id,
        maze=lin_name.replace("LinearizedPosition", ""),
        maze_type=maze_type,
        track_length_cm=track_length,
        n_units_total=len(spikes),
        n_units_excluded_corrupt=len(corrupt),
        corrupt_unit_ids=tuple(corrupt),
        maze_duration_s=float(maze_ep.end[0] - maze_ep.start[0]),
        pos_fs_hz=1.0 / dt,
    )
    return spikes, position, position_2d, maze_ep, meta, (nwbfile, io)


MAX_TRACKING_GAP = 1.0      # s; brief LED dropouts inside a single traversal
MIN_MONOTONICITY = 0.8      # net displacement / total range, to reject back-and-forth


def find_run_epochs(position, track_length=TRACK_LENGTH_CM,
                    min_duration=MIN_RUN_DURATION,
                    min_coverage_frac=MIN_RUN_COVERAGE_FRAC, pos_fs=39.0625,
                    max_gap=MAX_TRACKING_GAP, min_monotonicity=MIN_MONOTONICITY):
    """Split the linearized position into individual track traversals.

    The linearized signal is only defined while the animal is on the track, so
    contiguous blocks of valid samples are candidate traversals. Brief tracking
    dropouts fragment single traversals, so blocks separated by less than
    ``max_gap`` are merged first. A merged block is kept as a run if it lasts
    long enough, spans enough of the track, and is close to monotonic (so that
    back-and-forth wandering is excluded); it is then labelled by the sign of
    its net displacement.
    """
    min_coverage = min_coverage_frac * track_length
    t, d = position.t, position.values
    gap = np.diff(t) > max_gap
    seg_start_idx = np.concatenate([[0], np.flatnonzero(gap) + 1])
    seg_end_idx = np.concatenate([np.flatnonzero(gap), [t.size - 1]])

    starts, ends, directions = [], [], []
    for i0, i1 in zip(seg_start_idx, seg_end_idx):
        seg = d[i0:i1 + 1]
        if t[i1] - t[i0] < min_duration:
            continue
        span = seg.max() - seg.min()
        if span < min_coverage:
            continue
        net = seg[-1] - seg[0]
        if abs(net) < min_monotonicity * span:
            continue
        starts.append(t[i0])
        ends.append(t[i1])
        directions.append(1 if net > 0 else -1)

    run_ep = nap.IntervalSet(start=np.array(starts), end=np.array(ends))
    return run_ep, np.array(directions)


def compute_speed(position, run_ep):
    """Absolute running speed (cm/s) on the linearized track, per traversal."""
    t, d = position.restrict(run_ep).t, position.restrict(run_ep).values
    v = np.gradient(d, t)
    speed = nap.Tsd(t=t, d=np.abs(v), time_support=run_ep)
    return speed


def direction_epochs(run_ep, directions, speed, speed_threshold=SPEED_THRESHOLD_CMS):
    """Split traversals by travel direction.

    Returns two dicts keyed by 'rightward'/'leftward': whole traversals (useful
    as trials) and the same traversals masked by the running-speed threshold
    (used for rate maps, so that pauses at the reward ends do not contribute).
    """
    moving = speed.threshold(speed_threshold, "above").time_support
    trials, masked = {}, {}
    for label, sign in [("rightward", 1), ("leftward", -1)]:
        sel = np.flatnonzero(directions == sign)
        ep = nap.IntervalSet(start=run_ep.start[sel], end=run_ep.end[sel])
        trials[label] = ep
        masked[label] = ep.intersect(moving).drop_short_intervals(0.1)
    return trials, masked


def pick_theta_channel(nwbfile, run_ep, n_seconds=60.0):
    """Choose an LFP channel by theta/delta power ratio during running.

    The electrode table in this dandiset has no anatomical location, so the CA1
    channel is identified functionally: on the track the hippocampal LFP is
    dominated by 6-10 Hz theta, so the channel with the largest theta-to-delta
    ratio is the most hippocampal one available.
    """
    from scipy.signal import welch
    es = nwbfile.processing["ecephys"]["LFP"].electrical_series["LFP"]
    fs = float(es.rate)
    i0 = int(run_ep.start[0] * fs)
    i1 = i0 + int(n_seconds * fs)
    block = np.asarray(es.data[i0:i1, :]) * es.conversion
    freqs, pxx = welch(block, fs=fs, nperseg=int(4 * fs), axis=0)
    theta = pxx[(freqs >= 6) & (freqs <= 10)].mean(axis=0)
    delta = pxx[(freqs >= 2) & (freqs <= 4)].mean(axis=0)
    return int(np.argmax(theta / delta)), theta / delta


# ----------------------------------------------------------------------------
# Unit selection
# ----------------------------------------------------------------------------
def select_pyramidal(spikes, run_epochs_all):
    """Putative pyramidal cells with a usable firing rate during track running."""
    sub = spikes.restrict(run_epochs_all)
    rate_on_track = np.array([len(sub[k]) / run_epochs_all.tot_length() for k in sub.keys()])
    cell_type = np.asarray(spikes.metadata["cell_type"])
    keep = (cell_type == "excitatory") & (rate_on_track >= MIN_MEAN_RATE_HZ) \
        & (rate_on_track <= MAX_MEAN_RATE_HZ)
    keys = np.array(list(spikes.keys()))[keep]
    return spikes[list(keys)], keys


# ----------------------------------------------------------------------------
# Rate maps, spatial information, shuffling
# ----------------------------------------------------------------------------
def _virtual_time(x, ep):
    """Map real times inside `ep` onto a concatenated ('virtual') time axis."""
    starts, ends = ep.start, ep.end
    cum = np.concatenate([[0.0], np.cumsum(ends - starts)])
    idx = np.clip(np.searchsorted(starts, x, side="right") - 1, 0, len(starts) - 1)
    return cum[idx] + (x - starts[idx]), cum[-1]


def _smooth(rate_maps, sigma=SMOOTH_BINS):
    from scipy.ndimage import gaussian_filter1d
    if sigma <= 0:
        return rate_maps
    return gaussian_filter1d(rate_maps, sigma, axis=-1, mode="nearest")


class RateMapEngine:
    """Fast occupancy-normalised rate maps plus a circular-shift shuffle test.

    Spike times and position samples are projected onto a virtual time axis in
    which the selected traversals are concatenated. Shuffling is then a circular
    shift of each spike train on that axis, which preserves the spike train's
    temporal structure while destroying its alignment to position.
    """

    def __init__(self, spikes, position, ep, n_bins=N_POS_BINS,
                 track_length=TRACK_LENGTH_CM, pos_fs=39.0625):
        self.ep = ep
        self.n_bins = n_bins
        self.bin_edges = np.linspace(0.0, track_length, n_bins + 1)
        self.bin_centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])
        self.unit_ids = np.array(list(spikes.keys()))

        pos = position.restrict(ep)
        self.pos_vt, self.total_time = _virtual_time(pos.t, ep)
        self.pos_val = pos.values
        order = np.argsort(self.pos_vt)
        self.pos_vt, self.pos_val = self.pos_vt[order], self.pos_val[order]

        self.pos_dt = 1.0 / pos_fs
        self.occupancy = np.histogram(self.pos_val, bins=self.bin_edges)[0] * self.pos_dt
        self.occupancy_valid = self.occupancy > 0

        self.spike_vt = []
        for k in self.unit_ids:
            st = spikes[k].restrict(ep).t
            vt, _ = _virtual_time(st, ep)
            self.spike_vt.append(np.sort(vt))
        self.mean_rates = np.array([len(v) / self.total_time for v in self.spike_vt])

    def _maps_from_vt(self, spike_vt_list):
        maps = np.zeros((len(spike_vt_list), self.n_bins))
        for i, vt in enumerate(spike_vt_list):
            p = np.interp(vt, self.pos_vt, self.pos_val)
            counts = np.histogram(p, bins=self.bin_edges)[0]
            with np.errstate(divide="ignore", invalid="ignore"):
                maps[i] = np.where(self.occupancy_valid, counts / self.occupancy, np.nan)
        return maps

    def rate_maps(self, smooth=True):
        maps = self._maps_from_vt(self.spike_vt)
        maps = np.where(np.isnan(maps), 0.0, maps)
        return _smooth(maps) if smooth else maps

    def spatial_information(self, maps):
        """Skaggs information in bits/spike (and bits/s) from a set of rate maps."""
        p_x = self.occupancy / self.occupancy.sum()
        mean_rate = (maps * p_x[None, :]).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = maps / mean_rate[:, None]
            term = np.where(maps > 0, maps * np.log2(np.where(ratio > 0, ratio, 1.0)), 0.0)
        bits_per_sec = (term * p_x[None, :]).sum(axis=1)
        bits_per_spike = np.where(mean_rate > 0,
                                  bits_per_sec / np.where(mean_rate > 0, mean_rate, 1.0), 0.0)
        return bits_per_spike, bits_per_sec

    def shuffle_information(self, n_shuffles=N_SHUFFLES, seed=0, progress=True):
        """Null distribution of spatial information under circular shifts."""
        rng = np.random.default_rng(seed)
        lo, hi = 0.05 * self.total_time, 0.95 * self.total_time
        out = np.zeros((n_shuffles, len(self.unit_ids)))
        it = range(n_shuffles)
        if progress:
            from tqdm import tqdm
            it = tqdm(it, desc="shuffles", leave=False)
        for s in it:
            offsets = rng.uniform(lo, hi, size=len(self.unit_ids))
            shifted = [np.sort((vt + o) % self.total_time)
                       for vt, o in zip(self.spike_vt, offsets)]
            maps = _smooth(np.nan_to_num(self._maps_from_vt(shifted)))
            out[s] = self.spatial_information(maps)[0]
        return out

    def split_half_maps(self, smooth=True):
        """Rate maps from odd and even traversals, for a reliability measure."""
        starts, ends = self.ep.start, self.ep.end
        cum = np.concatenate([[0.0], np.cumsum(ends - starts)])
        halves = []
        for parity in (0, 1):
            sel = np.arange(len(starts)) % 2 == parity
            keep_lo, keep_hi = cum[:-1][sel], cum[1:][sel]
            occ_mask = np.zeros(self.pos_vt.size, bool)
            spk_masks = [np.zeros(v.size, bool) for v in self.spike_vt]
            for a, b in zip(keep_lo, keep_hi):
                occ_mask |= (self.pos_vt >= a) & (self.pos_vt < b)
                for j, v in enumerate(self.spike_vt):
                    spk_masks[j] |= (v >= a) & (v < b)
            occ = np.histogram(self.pos_val[occ_mask], bins=self.bin_edges)[0] * self.pos_dt
            maps = np.zeros((len(self.spike_vt), self.n_bins))
            for j, v in enumerate(self.spike_vt):
                p = np.interp(v[spk_masks[j]], self.pos_vt, self.pos_val)
                counts = np.histogram(p, bins=self.bin_edges)[0]
                with np.errstate(divide="ignore", invalid="ignore"):
                    maps[j] = np.where(occ > 0, counts / np.where(occ > 0, occ, 1), 0.0)
            halves.append(_smooth(maps) if smooth else maps)
        return halves


def split_half_correlation(map_a, map_b):
    out = np.full(map_a.shape[0], np.nan)
    for i in range(map_a.shape[0]):
        a, b = map_a[i], map_b[i]
        if a.std() > 0 and b.std() > 0:
            out[i] = np.corrcoef(a, b)[0, 1]
    return out


def field_properties(maps, bin_centers):
    """Peak rate, peak location and field width (contiguous >50 % of peak)."""
    peak_rate = maps.max(axis=1)
    peak_bin = maps.argmax(axis=1)
    peak_pos = bin_centers[peak_bin]
    bin_w = bin_centers[1] - bin_centers[0]
    widths = np.zeros(maps.shape[0])
    for i in range(maps.shape[0]):
        if peak_rate[i] <= 0:
            continue
        above = maps[i] >= 0.5 * peak_rate[i]
        j = peak_bin[i]
        lo = j
        while lo - 1 >= 0 and above[lo - 1]:
            lo -= 1
        hi = j
        while hi + 1 < maps.shape[1] and above[hi + 1]:
            hi += 1
        widths[i] = (hi - lo + 1) * bin_w
    return peak_rate, peak_pos, widths


def classify_place_cells(engine, n_shuffles=N_SHUFFLES, seed=0, progress=True):
    """Run the full per-direction place-cell battery and return a DataFrame."""
    maps = engine.rate_maps()
    info_bits, info_rate = engine.spatial_information(maps)
    null = engine.shuffle_information(n_shuffles=n_shuffles, seed=seed, progress=progress)
    p_val = (null >= info_bits[None, :]).sum(axis=0) / n_shuffles
    z = (info_bits - null.mean(axis=0)) / np.where(null.std(axis=0) > 0, null.std(axis=0), np.nan)

    half_a, half_b = engine.split_half_maps()
    stability = split_half_correlation(half_a, half_b)
    peak_rate, peak_pos, width = field_properties(maps, engine.bin_centers)

    df = pd.DataFrame({
        "unit": engine.unit_ids,
        "mean_rate": engine.mean_rates,
        "peak_rate": peak_rate,
        "peak_pos_cm": peak_pos,
        "field_width_cm": width,
        "info_bits_per_spike": info_bits,
        "info_bits_per_sec": info_rate,
        "shuffle_p": p_val,
        "info_z": z,
        "stability": stability,
    })
    df["is_place_cell"] = (df.shuffle_p < SHUFFLE_ALPHA) \
        & (df.peak_rate >= PEAK_RATE_CRITERION_HZ) \
        & (df.stability > 0.5)
    return df, maps, null


# ----------------------------------------------------------------------------
# Session-level driver
# ----------------------------------------------------------------------------
DIRECTIONS = ("rightward", "leftward")
DIR_COLORS = {"rightward": "#1f77b4", "leftward": "#d62728"}


def spike_positions_by_trial(unit_ts, position, trial_ep):
    """Position of every spike of one unit, grouped by traversal."""
    out = []
    for i in range(len(trial_ep)):
        ep = trial_ep[i]
        st = unit_ts.restrict(ep).t
        p = position.restrict(ep)
        out.append(np.interp(st, p.t, p.values) if p.t.size > 1 else np.array([]))
    return out


DECODE_BIN_SIZE = 0.2  # s


def decode_fold(res, direction, parity, units=None, shuffle_tc=False, rng=None,
                bin_size=DECODE_BIN_SIZE):
    """Two-fold cross-validated Bayesian decoding of position for one direction.

    Tuning curves are fitted on traversals of one parity and the posterior is
    evaluated on the held-out traversals, so nothing in the decoded epochs
    contributed to the encoding model.
    """
    from scipy.ndimage import gaussian_filter1d

    units = res["pyr"] if units is None else units
    position = res["position"]
    trials = res["trial_eps"][direction]
    idx = np.arange(len(trials))
    train_ep = res["dir_eps"][direction].intersect(trials[idx[idx % 2 == parity]])
    test_trials = trials[idx[idx % 2 != parity]]

    tc = nap.compute_tuning_curves(units, position, bins=res["meta"]["n_pos_bins"],
                                   range=[(0, res["meta"]["track_length_cm"])],
                                   epochs=train_ep, feature_names=["position"])
    tc.data = gaussian_filter1d(np.nan_to_num(tc.data), SMOOTH_BINS, axis=-1, mode="nearest")
    if shuffle_tc:
        rng = np.random.default_rng() if rng is None else rng
        tc.data = np.stack([np.roll(row, rng.integers(tc.data.shape[-1])) for row in tc.data])

    decoded, prob = nap.decode_bayes(tc, units, test_trials, bin_size=bin_size)
    true = np.interp(decoded.t, position.t, position.values)
    return decoded, prob, true, test_trials, tc


def decoding_error(res, units=None, shuffle_tc=False, rng=None, bin_size=DECODE_BIN_SIZE):
    """Pool absolute decoding error over both directions and both folds."""
    errs = []
    for direction in DIRECTIONS:
        for parity in (0, 1):
            dec, _, true, _, _ = decode_fold(res, direction, parity, units=units,
                                             shuffle_tc=shuffle_tc, rng=rng,
                                             bin_size=bin_size)
            ok = np.isfinite(true)
            errs.append(np.abs(dec.values[ok] - true[ok]))
    return np.concatenate(errs)


def analyze_session(session_name, n_shuffles=N_SHUFFLES, seed=0, progress=True,
                    keep_handles=False):
    """Full single-session place-cell pipeline."""
    spikes, position, pos2d, maze_ep, meta, handles = load_behavior_and_spikes(session_name)
    if meta["maze_type"] != "linear":
        raise ValueError(f"{session_name} uses a {meta['maze_type']} maze; this pipeline "
                         "handles linear tracks only")
    n_bins = int(round(meta["track_length_cm"] / POS_BIN_CM))
    meta["n_pos_bins"] = n_bins
    run_ep, directions = find_run_epochs(position, track_length=meta["track_length_cm"],
                                         pos_fs=meta["pos_fs_hz"])
    speed = compute_speed(position, run_ep)
    trial_eps, dir_eps = direction_epochs(run_ep, directions, speed)
    all_run_ep = dir_eps["rightward"].union(dir_eps["leftward"])
    pyr, pyr_keys = select_pyramidal(spikes, all_run_ep)

    per_dir = {}
    for label in DIRECTIONS:
        engine = RateMapEngine(pyr, position, dir_eps[label], n_bins=n_bins,
                               track_length=meta["track_length_cm"],
                               pos_fs=meta["pos_fs_hz"])
        df, maps, null = classify_place_cells(engine, n_shuffles=n_shuffles,
                                              seed=seed, progress=progress)
        df.insert(0, "direction", label)
        df.insert(0, "session", session_name)
        per_dir[label] = dict(engine=engine, table=df, maps=maps, null=null)

    meta.update(
        n_runs=len(run_ep),
        n_runs_right=int((directions == 1).sum()),
        n_runs_left=int((directions == -1).sum()),
        run_time_s=float(run_ep.tot_length()),
        n_pyramidal=len(pyr),
        median_speed_cms=float(np.median(speed.values)),
    )
    out = dict(meta=meta, spikes=spikes, pyr=pyr, position=position, position_2d=pos2d,
               maze_ep=maze_ep, run_ep=run_ep, directions=directions, speed=speed,
               trial_eps=trial_eps, dir_eps=dir_eps, per_dir=per_dir)
    if keep_handles:
        out["handles"] = handles
    else:
        handles[1].close()
    return out
