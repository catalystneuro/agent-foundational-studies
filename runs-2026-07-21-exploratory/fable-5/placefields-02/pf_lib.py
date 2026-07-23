"""Loading, preprocessing and place-field utilities for the DANDI:000044 analysis.

DANDI:000044 is the Grosmark & Buzsaki (2016) dataset: bilateral silicon-probe
recordings from dorsal hippocampus (CA1) in rats running back and forth on a
linear track for water reward, flanked by pre- and post-run sleep sessions.
"""
import numpy as np
import pynapple as nap
import lindi
from pynwb import NWBHDF5IO
from scipy.ndimage import uniform_filter1d, gaussian_filter1d

DANDISET = "000044"
LINDI_TMPL = "https://lindi.neurosift.org/dandi/dandisets/{ds}/assets/{asset}/nwb.lindi.json"

# asset_id -> session label. All eight sessions of Grosmark & Buzsaki (2016).
SESSIONS = {
    "5349c68b-c0a7-46c0-9900-cda050722fa4": "Achilles_10252013",
    "8855c8cc-9d8b-4d5b-8ef0-fe87916f839a": "Achilles_11012013",
    "3cc5b7b3-02e2-490a-9f19-d20670355084": "Cicero_09012014",
    "f61dfe09-3db2-464a-b386-2e828b2e7276": "Cicero_09102014",
    "e381ebb3-128e-4f3f-9517-11277d7aed9b": "Cicero_09172014",
    "31ea0aab-4777-424e-9a93-9605b2bdcc29": "Gatsby_08022013",
    "f7687af7-3bc9-4d20-8d88-ef293d2a3381": "Gatsby_08282013",
    "82714afb-724f-4e2b-b102-c9c47b5cba73": "Buddy_06272013",
}
PROTOTYPE = "5349c68b-c0a7-46c0-9900-cda050722fa4"  # Achilles_10252013

CACHE = "/tmp/lindi_cache"


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------

def open_session(asset_id):
    """Stream one NWB file from DANDI through LINDI (chunks cached on disk)."""
    url = LINDI_TMPL.format(ds=DANDISET, asset=asset_id)
    f = lindi.LindiH5pyFile.from_lindi_file(url, local_cache=lindi.LocalCache(cache_dir=CACHE))
    io = NWBHDF5IO(file=f, mode="r")
    return io.read()


def _behavior_series(nwbfile):
    """Return (2-D SpatialSeries, linearized SpatialSeries, maze name)."""
    beh = nwbfile.processing["behavior"]
    pos_key = [k for k in beh.data_interfaces
               if k.endswith("Position") and "Linearized" not in k][0]
    lin_key = [k for k in beh.data_interfaces if "LinearizedPosition" in k][0]
    pos, lin = beh[pos_key], beh[lin_key]
    return (pos[list(pos.spatial_series)[0]],
            lin[list(lin.spatial_series)[0]],
            pos_key)


def load_behavior(nwbfile):
    """Position as pynapple TsdFrame/Tsd on a manually reconstructed time base.

    NOTE: in this dandiset the SpatialSeries `rate` field actually holds the
    sampling PERIOD (0.0256 s -> 39.06 Hz), not a frequency. Building the time
    base as `starting_time + arange(n) * rate` reproduces the MazeEpoch bounds,
    whereas the nominal `arange(n) / rate` would stretch a 34-minute session to
    36 days. We therefore build the time base by hand rather than relying on
    pynapple's automatic NWB conversion. `check_timebase` asserts this.
    """
    pos_ss, lin_ss, maze_name = _behavior_series(nwbfile)
    n = pos_ss.data.shape[0]
    period = pos_ss.rate  # mislabelled: this is seconds/sample, not Hz
    t = pos_ss.starting_time + np.arange(n) * period

    xy = np.asarray(pos_ss.data[:], dtype=float)
    lin = np.asarray(lin_ss.data[:], dtype=float).ravel()

    position2d = nap.TsdFrame(t=t, d=xy, columns=["x", "y"])
    linear = nap.Tsd(t=t, d=lin)
    return position2d, linear, 1.0 / period, maze_name


def check_timebase(nwbfile, tol=1.0):
    """Verify the reconstructed time base lands inside the maze epoch.

    Returns (maze_duration, track_duration, ok). Raises if the mismatch is
    larger than `tol` seconds at either end.
    """
    pos2d, lin, fs, _ = load_behavior(nwbfile)
    maze = load_epochs(nwbfile)["MazeEpoch"]
    t0, t1 = float(pos2d.index[0]), float(pos2d.index[-1])
    m0, m1 = float(maze.start[0]), float(maze.end[0])
    if not (m0 - tol <= t0 and t1 <= m1 + tol):
        raise ValueError(
            f"reconstructed position time base [{t0:.1f}, {t1:.1f}] does not lie "
            f"inside MazeEpoch [{m0:.1f}, {m1:.1f}]")
    return m1 - m0, t1 - t0, True


def load_epochs(nwbfile):
    """The pre-sleep / maze / post-sleep epochs, keyed by label."""
    df = nwbfile.intervals["epochs"].to_dataframe()
    return {r["label"]: nap.IntervalSet(start=r["start_time"], end=r["stop_time"])
            for _, r in df.iterrows()}


def load_units(nwbfile):
    """TsGroup of spike times carrying cell_type / location / shank metadata."""
    u = nwbfile.units
    spikes = {i: nap.Ts(np.asarray(u["spike_times"][i])) for i in range(len(u))}
    meta = {
        "cell_type": np.asarray(u["cell_type"][:]),
        "location": np.asarray(u["location"][:]),
        "shank_id": np.asarray(u["shank_id"][:]),
    }
    return nap.TsGroup(spikes, metadata=meta)


# ----------------------------------------------------------------------------
# Preprocessing: running epochs split by direction of travel
# ----------------------------------------------------------------------------

def maze_kind(maze_name):
    """'circular' for the closed-loop mazes, 'linear' for the straight tracks."""
    return "circular" if "Circular" in maze_name else "linear"


def make_run_epochs(linear, fs, speed_thresh=0.10, smooth_s=0.25,
                    min_dur=0.5, merge_gap=0.2, period=None):
    """Split the linearized position into leftward / rightward running epochs.

    The linearized SpatialSeries is NaN whenever the animal is off the track, so
    the valid samples already isolate track traversals. Within each contiguous
    block of valid tracking we smooth the position, differentiate it, and keep
    samples where the animal moves faster than `speed_thresh` m/s. The sign of
    the velocity gives the direction of travel.

    `period` must be given for a closed-loop (circular) maze: position then
    wraps at `period` metres, so increments are unwrapped onto the shortest arc
    before differentiating. Without this the wrap point produces a spurious
    full-track-length jump in velocity on every lap.

    Returns (lin_valid, velocity, epochs_dict) where epochs_dict maps
    'rightward' / 'leftward' / 'run' to IntervalSets.
    """
    t = linear.index.values
    d = linear.values.astype(float)
    valid = np.isfinite(d)

    idx = np.where(valid)[0]
    breaks = np.where(np.diff(idx) > 1)[0]
    blocks = np.split(idx, breaks + 1)

    win = max(3, int(round(smooth_s * fs)))
    vel = np.full(len(d), np.nan)
    for b in blocks:
        if len(b) < win:
            continue
        x = d[b]
        if period is not None:
            step = np.diff(x)
            step = (step + period / 2.0) % period - period / 2.0
            x = np.r_[x[0], x[0] + np.cumsum(step)]
        sm = uniform_filter1d(x, size=win, mode="nearest")
        vel[b] = np.gradient(sm, 1.0 / fs)

    good = np.isfinite(vel)
    lin_valid = nap.Tsd(t=t[good], d=d[good])
    velocity = nap.Tsd(t=t[good], d=vel[good])

    def _epochs_from_mask(mask):
        """Contiguous True runs of `mask` (defined on the `good` samples)."""
        tt = t[good]
        m = np.asarray(mask)
        if not m.any():
            return nap.IntervalSet(start=[], end=[])
        edges = np.diff(m.astype(int))
        starts = np.r_[0] if m[0] else np.array([], int)
        starts = np.r_[starts, np.where(edges == 1)[0] + 1]
        ends = np.where(edges == -1)[0]
        if m[-1]:
            ends = np.r_[ends, len(m) - 1]
        # only keep sample-contiguous stretches (never bridge a tracking gap)
        ivs, dt_max = [], 5.0 / fs
        for s, e in zip(starts, ends):
            seg = np.arange(s, e + 1)
            cut = np.where(np.diff(tt[seg]) > dt_max)[0]
            for sub in np.split(seg, cut + 1):
                if len(sub) > 1:
                    ivs.append((tt[sub[0]], tt[sub[-1]]))
        if not ivs:
            return nap.IntervalSet(start=[], end=[])
        ivs = np.array(ivs)
        ep = nap.IntervalSet(start=ivs[:, 0], end=ivs[:, 1])
        return ep.merge_close_intervals(merge_gap).drop_short_intervals(min_dur)

    v = velocity.values
    eps = {"rightward": _epochs_from_mask(v > speed_thresh),
           "leftward": _epochs_from_mask(v < -speed_thresh)}
    eps["run"] = eps["rightward"].union(eps["leftward"])
    return lin_valid, velocity, eps


def track_range(linear, pad=0.0):
    """Outer bin edges spanning the sampled extent of the linearized track."""
    d = linear.values[np.isfinite(linear.values)]
    return (float(np.min(d)) - pad, float(np.max(d)) + pad)


# ----------------------------------------------------------------------------
# Compressed time axis: concatenates a set of epochs end to end
# ----------------------------------------------------------------------------

def _compressed_time(ep, t):
    """Map real times `t` onto a 'within-epoch' axis that concatenates `ep`.

    Times outside `ep` map to NaN. Returns (tau, total_duration).
    """
    starts, ends = np.asarray(ep.start), np.asarray(ep.end)
    durs = ends - starts
    offs = np.r_[0.0, np.cumsum(durs)[:-1]]
    T = durs.sum()
    j = np.searchsorted(starts, t, side="right") - 1
    tau = np.full(len(t), np.nan)
    ok = (j >= 0) & (j < len(starts))
    jj = j[ok]
    inside = t[ok] <= ends[jj]
    sel = np.where(ok)[0][inside]
    tau[sel] = offs[j[sel]] + (t[sel] - starts[j[sel]])
    return tau, T


# ----------------------------------------------------------------------------
# Place fields: rate maps, Skaggs spatial information, circular-shift shuffles
# ----------------------------------------------------------------------------

def spatial_information(rate_map, occupancy):
    """Skaggs spatial information (bits/spike), sparsity and mean rate per unit.

    rate_map: (n_units, n_bins) in Hz; occupancy: (n_bins,) in seconds.
    Bins are assumed already restricted to those with valid occupancy.
    """
    p = occupancy / occupancy.sum()
    lam = np.asarray(rate_map, dtype=float)
    mean_rate = (p[None, :] * lam).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = lam / mean_rate[:, None]
        term = p[None, :] * ratio * np.log2(ratio)
        sparsity = mean_rate ** 2 / (p[None, :] * lam ** 2).sum(axis=1)
    term[~np.isfinite(term)] = 0.0
    return term.sum(axis=1), sparsity, mean_rate


class DirectionalRateMaps:
    """Rate maps for one running direction, plus a matched circular-shift null.

    Occupancy and the spike->position-bin lookup are built once on a compressed
    time axis that concatenates the running epochs, so generating a shuffle is
    just a modular shift of spike times plus a searchsorted and a bincount.
    Smoothing is applied to the spike counts and the occupancy alike, so the
    ratio remains a properly normalised rate.
    """

    def __init__(self, linear, ep, fs, n_bins=40, track_range=(0.0, 1.6),
                 smooth_bins=1.0, min_occupancy=0.5, circular=False):
        self.ep, self.fs, self.n_bins = ep, fs, n_bins
        self.linear, self.min_occupancy = linear, min_occupancy
        self.circular = circular
        self.edges = np.linspace(track_range[0], track_range[1], n_bins + 1)
        self.centers = 0.5 * (self.edges[1:] + self.edges[:-1])
        self.bin_size = self.edges[1] - self.edges[0]
        self.smooth_bins = smooth_bins

        p = linear.restrict(ep)
        tau_pos, self.T = _compressed_time(ep, p.index.values)
        keep = np.isfinite(tau_pos)
        tau_pos, pos = tau_pos[keep], p.values[keep]
        order = np.argsort(tau_pos)
        self.tau_pos, pos = tau_pos[order], pos[order]
        self.pos_bin = np.clip(np.digitize(pos, self.edges) - 1, 0, n_bins - 1)

        dt = 1.0 / fs
        raw_occ = np.bincount(self.pos_bin, minlength=n_bins) * dt
        self.raw_occupancy = raw_occ
        self.valid_bins = raw_occ > min_occupancy
        self.occupancy = self._smooth(raw_occ)

    def _smooth(self, m):
        if self.smooth_bins <= 0:
            return m
        return gaussian_filter1d(m, self.smooth_bins, axis=-1,
                                 mode="wrap" if self.circular else "nearest")

    def _nearest_sample(self, ts):
        """Index of the position sample closest in time to each spike.

        This matches pynapple's `value_from` convention, so the rate maps agree
        with `nap.compute_tuning_curves` bin for bin.
        """
        j = np.clip(np.searchsorted(self.tau_pos, ts), 1, len(self.tau_pos) - 1)
        closer_left = np.abs(ts - self.tau_pos[j - 1]) <= np.abs(self.tau_pos[j] - ts)
        return np.where(closer_left, j - 1, j)

    def _counts_from_tau(self, tau_spk):
        counts = np.zeros((len(tau_spk), self.n_bins))
        for i, ts in enumerate(tau_spk):
            if len(ts) == 0:
                continue
            counts[i] = np.bincount(self.pos_bin[self._nearest_sample(ts)],
                                    minlength=self.n_bins)
        return counts

    def _map_from_tau(self, tau_spk):
        counts = self._smooth(self._counts_from_tau(tau_spk))
        with np.errstate(divide="ignore", invalid="ignore"):
            rm = counts / np.maximum(self.occupancy, 1e-12)
        rm[:, ~self.valid_bins] = np.nan
        return rm

    def spike_taus(self, units):
        out = []
        for u in units.keys():
            tau, _ = _compressed_time(self.ep, units[u].index.values)
            out.append(np.sort(tau[np.isfinite(tau)]))
        return out

    def rate_maps(self, units):
        return self._map_from_tau(self.spike_taus(units))

    def si(self, rate_map):
        """Spatial information / sparsity / mean rate over the valid bins."""
        return spatial_information(rate_map[:, self.valid_bins],
                                   self.occupancy[self.valid_bins])

    def shuffle_si(self, units, n_shuffles=1000, seed=0, progress=True):
        """Null distribution of spatial information from circular time shifts.

        Each unit's spike train is shifted by an independent random offset on
        the compressed (running-epochs-only) time axis and wrapped around. This
        destroys the spike/position relationship while preserving each unit's
        spike count and its fine-scale temporal structure (burstiness), which a
        naive rate-matched Poisson null would not.
        """
        rng = np.random.default_rng(seed)
        taus = self.spike_taus(units)
        occ_ok = self.occupancy[self.valid_bins]
        null = np.empty((n_shuffles, len(taus)))
        it = range(n_shuffles)
        if progress:
            from tqdm import tqdm
            it = tqdm(it, desc="shuffles", leave=False)
        for k in it:
            shifts = rng.uniform(1.0, self.T - 1.0, size=len(taus))
            shifted = [np.sort((t + s) % self.T) for t, s in zip(taus, shifts)]
            rm = self._map_from_tau(shifted)
            null[k] = spatial_information(rm[:, self.valid_bins], occ_ok)[0]
        return null

    def lap_split_maps(self, units):
        """Rate maps computed separately from odd- and even-numbered laps.

        Used for a cross-validated check that fields are stable within a
        session rather than driven by a single traversal.
        """
        starts, ends = np.asarray(self.ep.start), np.asarray(self.ep.end)
        out = []
        for sel in (np.arange(0, len(starts), 2), np.arange(1, len(starts), 2)):
            sub = nap.IntervalSet(start=starts[sel], end=ends[sel])
            half = DirectionalRateMaps(
                self.linear, sub, self.fs, self.n_bins,
                (self.edges[0], self.edges[-1]), self.smooth_bins,
                self.min_occupancy, self.circular)
            out.append(half.rate_maps(units))
        return out


def field_metrics(rate_map, centers, bin_size, frac=0.5, circular=False):
    """Peak rate, peak position and field width for each unit's rate map.

    The field is the contiguous run of bins around the peak that stay above
    `frac` of the peak rate; its width is that run's extent in metres. On a
    closed-loop track the run is allowed to wrap around the ends.
    """
    n_units, n_bins = rate_map.shape
    peak_rate = np.full(n_units, np.nan)
    peak_pos = np.full(n_units, np.nan)
    width = np.full(n_units, np.nan)
    for i in range(n_units):
        r = rate_map[i]
        ok = np.isfinite(r)
        if not ok.any() or np.nanmax(r) <= 0:
            continue
        pk = int(np.nanargmax(r))
        peak_rate[i] = r[pk]
        peak_pos[i] = centers[pk]
        above = np.where(ok, r >= frac * r[pk], False)
        n_field = 1
        for step in (-1, +1):
            j = pk
            while n_field < n_bins:
                j += step
                if circular:
                    j %= n_bins
                elif not (0 <= j < n_bins):
                    break
                if not above[j]:
                    break
                n_field += 1
        width[i] = n_field * bin_size
    return peak_rate, peak_pos, width
